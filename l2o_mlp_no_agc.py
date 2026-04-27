import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    CH_TAPS, SNR_RANGE, DFE_TAPS, CTLE_TAPS, CTLE_HP_ALPHA, FFE_TAPS, FFE_MAIN_CURSOR, FFE_INIT,
    L2O_STATE_DIM, NO_AGC_STATE_DIM, L2O_MLP_HIDDEN_DIM, L2O_OVERDRIVE_MAX,
    EMA_BETA,
    BATCH_SIZE, EPOCHS, UNROLL_LEN,
    ABLATE_CTLE, OVERSAMPLE_FACTOR, OVERSAMPLE_MODE,
    PHASE_SEARCH_MAX_DELAY, PHASE_SEARCH_SYNC_LEN,
    MU_FFE_MAX, MU_DFE_MAX, MU_CTLE_MAX, ERR_DIR_TAU,
)
from wireline_channel import WirelineChannelGenerator
from utils import add_channel_args, get_channel_generator
from ctle_frequency_utils import apply_frequency_domain_ctle
from oversampling_utils import (
    choose_best_symbol_phase,
    choose_best_symbol_phase_per_example,
    upsample_symbols,
)
from feature_normalization import (
    StreamingFeatureNormalizer,
    build_no_agc_state,
    build_agc_state,
)


class DifferentiableCTLE(nn.Module):
    def __init__(self, num_taps=15):
        super().__init__()
        self.num_taps = num_taps
        self.register_buffer('base_lp', self._generate_base_lp())
        self.register_buffer('base_hp', self._generate_base_hp())

    def _generate_base_lp(self):
        lp = torch.zeros(self.num_taps)
        lp[0] = 1.0
        return lp

    def _generate_base_hp(self):
        hp = torch.zeros(self.num_taps)
        hp[0] = 1.0
        hp[1] = -CTLE_HP_ALPHA
        return hp

    def forward(self, rx_signal, peaking_gain):
        filter_taps = self.base_lp.unsqueeze(0) + peaking_gain.unsqueeze(-1) * self.base_hp.unsqueeze(0)
        rx_padded = F.pad(rx_signal, (self.num_taps - 1, 0))
        batch_size = rx_signal.shape[0]
        rx_reshaped = rx_padded.view(1, batch_size, -1)
        filter_taps_flipped = torch.flip(filter_taps, dims=[-1])
        filtered_rx = F.conv1d(rx_reshaped, filter_taps_flipped, groups=batch_size)
        return filtered_rx.view(batch_size, -1)


class DifferentiableDFE(nn.Module):
    def __init__(self, num_taps=10):
        super().__init__()
        self.num_taps = num_taps

    def forward(self, rx_eq, decision_buffer, dfe_weights):
        feedback = torch.sum(decision_buffer * dfe_weights, dim=1, keepdim=True)
        y_out = rx_eq - feedback
        return y_out


class MultiRateLearnedMLPNoAGC(nn.Module):
    """
    MLP-based learned optimizer for no-AGC mode with scale-aware feature normalization.

    Key improvements over the broken version:
    1. StreamingFeatureNormalizer applied per feature across history
       (NOT incorrect flattened slice indexing)
    2. Scale-aware state construction (build_no_agc_state) normalizes by FFE RMS
    3. Separate mu_ffe and mu_dfe heads (FFE input is raw amplitude, DFE input is decisions)
    4. Smooth e_dir = tanh(e_t / tau_err) during training (not hard sign)
    """

    def __init__(self, state_dim=NO_AGC_STATE_DIM, history_len=10, hidden_dim=64, use_two_head=False):
        super().__init__()
        self.state_dim = state_dim
        self.history_len = history_len
        self.use_two_head = use_two_head
        self.input_dim = state_dim * history_len

        self.normalizer = StreamingFeatureNormalizer(feature_dim=state_dim, momentum=0.99, eps=1e-5)

        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.head_ffe_base = nn.Linear(hidden_dim, 1)
        self.head_dfe_base = nn.Linear(hidden_dim, 1)

        if self.use_two_head:
            self.head_ffe_overdrive = nn.Linear(hidden_dim, 1)
            self.head_dfe_overdrive = nn.Linear(hidden_dim, 1)

        self.head_ctle = nn.Linear(hidden_dim, 1)

    def forward(self, history_buffer, update_ctle=False):
        """
        Args:
            history_buffer: [batch_size, history_len, state_dim] - NOT flattened
            update_ctle: whether to update CTLE at this timestep

        Returns:
            mu_ffe: [batch_size, 1] - FFE step size
            mu_dfe: [batch_size, 1] - DFE step size
            mu_ctle: [batch_size, 1] - CTLE step size
            mu_overdrive: [batch_size, 1] - overdrive component
        """
        if self.training:
            self.normalizer.update(history_buffer.detach())

        normalized = self.normalizer.normalize(history_buffer)
        flat_features = normalized.reshape(history_buffer.shape[0], -1)

        features = self.mlp(flat_features)

        mu_ffe_base = torch.sigmoid(self.head_ffe_base(features)) * MU_FFE_MAX
        mu_dfe_base = torch.sigmoid(self.head_dfe_base(features)) * MU_DFE_MAX

        mu_ffe = mu_ffe_base
        mu_dfe = mu_dfe_base
        mu_overdrive = torch.zeros_like(mu_ffe)

        if self.use_two_head:
            raw_ffe_over = F.softplus(self.head_ffe_overdrive(features))
            raw_dfe_over = F.softplus(self.head_dfe_overdrive(features))
            mu_ffe = mu_ffe + torch.clamp(raw_ffe_over, max=L2O_OVERDRIVE_MAX)
            mu_dfe = mu_dfe + torch.clamp(raw_dfe_over, max=L2O_OVERDRIVE_MAX)
            mu_overdrive = raw_ffe_over + raw_dfe_over

        if update_ctle:
            mu_ctle = torch.tanh(self.head_ctle(features)) * MU_CTLE_MAX
        else:
            mu_ctle = torch.zeros_like(mu_dfe)

        return mu_ffe, mu_dfe, mu_ctle, mu_overdrive


def train_learned_optimizer(channel_gen, dfe, ctle, learned_opt, epochs=100, batch_size=64,
                              unroll_len=50, history_len=10, ablate_ctle=False):
    """
    Trains the learned MLP optimizer using TBPTT with sliding history buffer.

    Key fixes:
    1. No warmup adaptation: during target_idx < 0, only update buffers, not equalizer params
    2. Correct history reshape: [batch, history_len, state_dim] -> normalize -> flatten
    3. Scale-aware state via build_no_agc_state
    4. Smooth e_dir during training, hard sign at eval
    """
    if ablate_ctle:
        print("!!! RUNNING IN ABLATION MODE: CTLE CONTROL DISABLED !!!")

    meta_optimizer = torch.optim.Adam(learned_opt.parameters(), lr=1e-3)
    total_seq_len = 500
    ctle_update_rate = 10
    loss_history = []
    ss_history = []

    for epoch in range(epochs):
        tx_symbols = torch.sign(torch.randn(batch_size, total_seq_len))
        tx_frontend = upsample_symbols(tx_symbols, OVERSAMPLE_FACTOR, OVERSAMPLE_MODE)
        rx_base, h_batch = channel_gen.generate_received_signal(tx_frontend, batch_size)

        with torch.no_grad():
            if ablate_ctle:
                rx_frontend = apply_frequency_domain_ctle(
                    rx_base,
                    peaking_gain=0.5,
                    samples_per_symbol=OVERSAMPLE_FACTOR,
                    fc=0.25,
                )
            else:
                if OVERSAMPLE_FACTOR != 1:
                    raise ValueError(
                        "OVERSAMPLE_FACTOR > 1 currently supported only in the "
                        "frequency-domain CTLE path."
                    )
                rx_frontend = ctle(rx_base, torch.ones(batch_size, 1) * 0.5)

            rx_init, best_phase, common_delay = choose_best_symbol_phase_per_example(
                tx_symbols,
                rx_frontend,
                OVERSAMPLE_FACTOR,
                max_delay=PHASE_SEARCH_MAX_DELAY,
                sync_len=PHASE_SEARCH_SYNC_LEN,
            )

        dfe_weights = torch.zeros(batch_size, dfe.num_taps)

        w_ffe = torch.zeros(batch_size, FFE_TAPS)
        w_ffe[:, FFE_MAIN_CURSOR] = FFE_INIT
        ffe_buffer = torch.zeros(batch_size, FFE_TAPS)

        latent_peaking = torch.zeros(batch_size, 1)

        rx_buffer = torch.zeros(batch_size, ctle.num_taps)

        decision_buffer = torch.zeros(batch_size, dfe.num_taps)
        ema_error = torch.ones(batch_size, 1)

        history_buffer = torch.zeros(batch_size, history_len, NO_AGC_STATE_DIM)

        effective_seq_len = total_seq_len - common_delay
        epoch_total_mse = 0
        epoch_ss_mse = 0
        num_steps = 0
        ss_steps = 0

        for t_start in range(0, effective_seq_len, unroll_len):
            meta_optimizer.zero_grad()
            loss = 0

            dfe_weights = dfe_weights.detach()
            w_ffe = w_ffe.detach()
            ffe_buffer = ffe_buffer.detach()
            latent_peaking = latent_peaking.detach()
            decision_buffer = decision_buffer.detach()
            ema_error = ema_error.detach()
            rx_buffer = rx_buffer.detach()
            history_buffer = history_buffer.detach()

            current_block_len = min(unroll_len, effective_seq_len - t_start)

            for t in range(t_start, t_start + current_block_len):
                if ablate_ctle:
                    rx_eq = rx_init[:, (t + common_delay):(t + common_delay + 1)]
                    ctle_peaking = torch.full((batch_size, 1), 0.5, device=latent_peaking.device)
                else:
                    rx_t = rx_base[:, (t + common_delay):(t + common_delay + 1)]
                    ctle_peaking = torch.sigmoid(latent_peaking)

                    rx_buffer = torch.roll(rx_buffer, shifts=1, dims=1)
                    rx_buffer[:, 0] = rx_t.squeeze(-1)

                    current_taps = ctle.base_lp.unsqueeze(0) + ctle_peaking * ctle.base_hp.unsqueeze(0)
                    rx_eq = torch.sum(rx_buffer * current_taps, dim=1, keepdim=True)

                ffe_buffer = torch.roll(ffe_buffer, shifts=1, dims=1)
                ffe_buffer[:, 0] = rx_eq.squeeze(-1)

                dfe_feedback = torch.sum(decision_buffer * dfe_weights, dim=1, keepdim=True)

                target_idx = t - FFE_MAIN_CURSOR
                can_compute_error = (target_idx >= 0)

                if can_compute_error:
                    ffe_out = torch.sum(ffe_buffer * w_ffe, dim=1, keepdim=True)
                    y_out = ffe_out - dfe_feedback

                    target_symbol = tx_symbols[:, target_idx:target_idx + 1]
                    e_t = target_symbol - y_out

                    grad_proxy_ctle = e_t * ffe_buffer[:, FFE_MAIN_CURSOR:FFE_MAIN_CURSOR+1] / (
                        ffe_buffer.pow(2).sum(dim=1, keepdim=True) + 1e-6
                    )

                    z_raw = build_no_agc_state(
                        e_t, ema_error, ffe_buffer, rx_buffer,
                        dfe_weights, ctle_peaking, grad_proxy_ctle,
                    )

                    history_buffer = torch.roll(history_buffer, shifts=1, dims=1)
                    history_buffer[:, 0, :] = z_raw

                    update_ctle_flag = (t % ctle_update_rate == 0)
                    mu_ffe, mu_dfe, mu_ctle, mu_overdrive = learned_opt(history_buffer, update_ctle_flag)

                    if ablate_ctle:
                        mu_ctle = torch.zeros_like(mu_ctle)

                    ema_error = EMA_BETA * ema_error + (1 - EMA_BETA) * (e_t.detach() ** 2)

                    if learned_opt.training:
                        e_dir = torch.tanh(e_t / ERR_DIR_TAU)
                    else:
                        e_dir = torch.sign(e_t)

                    ffe_rms = torch.sqrt(ffe_buffer.pow(2).mean(dim=1, keepdim=True) + 1e-6)
                    ffe_dir = ffe_buffer / ffe_rms.clamp_min(1e-6)

                    dfe_weights = dfe_weights - mu_dfe * e_dir * decision_buffer
                    w_ffe = w_ffe + mu_ffe * e_dir * ffe_dir

                    if update_ctle_flag:
                        latent_peaking = latent_peaking + mu_ctle

                    tau_soft = 0.1
                    soft_decision = torch.tanh(y_out / tau_soft)

                    decision_buffer = torch.roll(decision_buffer, shifts=1, dims=1)
                    decision_buffer[:, 0] = soft_decision.squeeze(-1)

                    step_mse = torch.mean(e_t ** 2)
                    loss += step_mse

                    if learned_opt.use_two_head:
                        loss += 0.01 * torch.mean(mu_overdrive ** 2)

                    epoch_total_mse += step_mse.item()
                    num_steps += 1

                    if t >= 200:
                        epoch_ss_mse += step_mse.item()
                        ss_steps += 1
                else:
                    grad_proxy_ctle = torch.zeros_like(ffe_buffer[:, FFE_MAIN_CURSOR:FFE_MAIN_CURSOR+1])

                    z_raw = build_no_agc_state(
                        torch.zeros_like(ema_error),
                        ema_error,
                        ffe_buffer,
                        rx_buffer,
                        dfe_weights,
                        ctle_peaking,
                        grad_proxy_ctle,
                    )

                    history_buffer = torch.roll(history_buffer, shifts=1, dims=1)
                    history_buffer[:, 0, :] = z_raw

                    update_ctle_flag = (t % ctle_update_rate == 0)
                    mu_ffe, mu_dfe, mu_ctle, mu_overdrive = learned_opt(history_buffer, update_ctle_flag)

                    if ablate_ctle:
                        mu_ctle = torch.zeros_like(mu_ctle)

                    if update_ctle_flag:
                        latent_peaking = latent_peaking + mu_ctle

                    soft_decision = torch.zeros(batch_size, 1)
                    decision_buffer = torch.roll(decision_buffer, shifts=1, dims=1)
                    decision_buffer[:, 0] = soft_decision.squeeze(-1)

            if loss.requires_grad:
                (loss / current_block_len).backward()
                torch.nn.utils.clip_grad_norm_(learned_opt.parameters(), 1.0)
                meta_optimizer.step()

        avg_epoch_mse = epoch_total_mse / num_steps
        avg_ss_mse = epoch_ss_mse / ss_steps if ss_steps > 0 else avg_epoch_mse
        loss_history.append(avg_epoch_mse)
        ss_history.append(avg_ss_mse)
        print(f"Epoch {epoch + 1}/{epochs} | Avg MSE: {avg_epoch_mse:.6f} | SS MSE: {avg_ss_mse:.6f}")

    return learned_opt, loss_history, ss_history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Learned Optimizer (MLP) - No AGC Variant")
    parser = add_channel_args(parser)
    parser.add_argument("--two_head", action="store_true",
                        help="Enable two-head overdrive architecture for FFE/DFE step size")
    args = parser.parse_args()

    args.disable_agc = True
    print("!!! WARNING: Running in NO-AGC mode - channel IR normalization disabled !!!")
    print("    Channel impulse responses will preserve true insertion loss physics.")

    print("Initializing modules...")
    channel_gen = get_channel_generator(args, samples_per_symbol=OVERSAMPLE_FACTOR)
    ctle = DifferentiableCTLE(num_taps=CTLE_TAPS)
    dfe = DifferentiableDFE(num_taps=DFE_TAPS)

    learned_opt = MultiRateLearnedMLPNoAGC(
        state_dim=NO_AGC_STATE_DIM,
        history_len=L2O_MLP_HISTORY_LEN,
        hidden_dim=L2O_MLP_HIDDEN_DIM,
        use_two_head=args.two_head
    )

    print(f"Channel type: {args.channel_type}")
    print(f"State dim: {NO_AGC_STATE_DIM}")
    print(f"Normalization: StreamingFeatureNormalizer with scale-aware build_no_agc_state")
    print(f"FFE/DFE heads: separate (mu_ffe, mu_dfe)")
    print(f"Error direction: smooth tanh(e_t/ERR_DIR_TAU) during training")
    print(f"Two-head overdrive enabled: {args.two_head}")
    print("-" * 50)

    trained_model, loss_history, ss_history = train_learned_optimizer(
        channel_gen=channel_gen,
        dfe=dfe,
        ctle=ctle,
        learned_opt=learned_opt,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        unroll_len=UNROLL_LEN,
        history_len=L2O_MLP_HISTORY_LEN,
        ablate_ctle=ABLATE_CTLE
    )

    print("-" * 50)
    print("Meta-training completed!")

    final_stage_start = int(len(loss_history) * 0.8)
    final_stage_avg_mse = sum(loss_history[final_stage_start:]) / len(loss_history[final_stage_start:])
    final_stage_ss_mse = sum(ss_history[final_stage_start:]) / len(ss_history[final_stage_start:])

    print(f"Final epoch Average MSE: {loss_history[-1]:.6f}")
    print(f"Final epoch Steady-State MSE: {ss_history[-1]:.6f}")
    print(f"Final stage (last 20%) Avg MSE: {final_stage_avg_mse:.6f}")
    print(f"Final stage (last 20%) Steady-State MSE: {final_stage_ss_mse:.6f}")

    suffix = "_ablate_ctle" if ABLATE_CTLE else ""
    two_head_suffix = "_two_head" if args.two_head else ""
    model_path = f"./models/l2o_mlp_noagc_model_{args.channel_type}{suffix}{two_head_suffix}_dfe={DFE_TAPS}.pth"
    torch.save(trained_model.state_dict(), model_path)
    print(f"Trained model saved to {model_path}")
    print("-" * 50)