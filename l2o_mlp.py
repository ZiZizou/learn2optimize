import torch
import torch.nn as nn
import torch.nn.functional as F

# Import common configuration
from config import *

# ==========================================
# MLP-Based Learned Optimizer
# This version uses a feedforward MLP with a sliding history buffer
# instead of an RNN to provide temporal context.
# ==========================================

from wireline_channel import WirelineChannelGenerator
from utils import add_channel_args, get_channel_generator

# ==========================================
# 2. Differentiable Parametric CTLE
# ==========================================
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
        """
        rx_signal: [batch, seq_len]
        peaking_gain: [batch, 1] - The parameter adapted by the learned optimizer
        """
        filter_taps = self.base_lp.unsqueeze(0) + peaking_gain.unsqueeze(-1) * self.base_hp.unsqueeze(0)

        rx_padded = F.pad(rx_signal, (self.num_taps - 1, 0))
        batch_size = rx_signal.shape[0]
        rx_reshaped = rx_padded.view(1, batch_size, -1)

        filter_taps_flipped = torch.flip(filter_taps, dims=[-1])
        filtered_rx = F.conv1d(rx_reshaped, filter_taps_flipped, groups=batch_size)
        return filtered_rx.view(batch_size, -1)


# ==========================================
# 3. Differentiable Decision Feedback Equalizer (DFE)
# ==========================================
class DifferentiableDFE(nn.Module):
    def __init__(self, num_taps=10):
        super().__init__()
        self.num_taps = num_taps

    def forward(self, rx_eq, decision_buffer, dfe_weights):
        """
        rx_eq: [batch, 1] - CTLE output
        decision_buffer: [batch, num_taps] - Past symbols (decisions)
        dfe_weights: [batch, num_taps] - Current learned DFE taps
        """
        feedback = torch.sum(decision_buffer * dfe_weights, dim=1, keepdim=True)
        y_out = rx_eq - feedback
        return y_out


# ==========================================
# 4. MLP-Based Learned Optimizer (replaces RNN)
# ==========================================
class MultiRateLearnedMLP(nn.Module):
    """
    Feedforward MLP with sliding history buffer.

    Instead of using an RNN's hidden state to accumulate temporal context,
    this MLP takes a concatenated vector of the last N states as input.

    Scientific Rationale:
    - The MLP is stateless between forward passes but receives explicit
      temporal context via the history buffer.
    - This allows direct gradient flow without BPTT through the RNN recurrence.
    - The history buffer acts as an "explicit memory" that the MLP can
      learn to interpret.

    Two-Head Overdrive Architecture:
    - Base head (sigmoid bounded): For steady-state tracking, mu in [0, 0.05]
    - Overdrive head (softplus + clamp): For acquisition, additional mu in [0, 0.45]
    - Total max mu = 0.5 (physical stability limit for NLMS)
    """
    def __init__(self, state_dim=6, history_len=10, hidden_dim=64, use_two_head=False):
        super().__init__()
        self.state_dim = state_dim
        self.history_len = history_len
        self.use_two_head = use_two_head
        input_dim = state_dim * history_len

        # Deep feedforward network to handle increased input dimensionality
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Base head: always used, tightly bounded [0, 0.05]
        self.head_dfe_base = nn.Linear(hidden_dim, 1)

        # Overdrive head: only instantiated if two_head is True
        if self.use_two_head:
            self.head_dfe_overdrive = nn.Linear(hidden_dim, 1)

        self.head_ctle = nn.Linear(hidden_dim, 1)

    def forward(self, history_buffer, update_ctle=False):
        """
        Args:
            history_buffer: [batch_size, state_dim * history_len] - flattened history
            update_ctle: whether to update CTLE at this timestep

        Returns:
            mu_dfe: [batch_size, 1] - DFE step size
            mu_ctle: [batch_size, 1] - CTLE step size
            mu_overdrive: [batch_size, 1] - Overdrive component (for regularization)
        """
        features = self.mlp(history_buffer)

        # 1. Base Step Size: tightly bounded between (0, 0.05) using sigmoid
        mu_base = torch.sigmoid(self.head_dfe_base(features)) * L2O_DFE_HEAD_SCALE

        # 2. Overdrive Step Size
        if self.use_two_head:
            # Softplus allows smooth gradients near zero.
            # Clamp to 0.45 so max theoretical combined speed is 0.5 (safe physical limit)
            raw_overdrive = F.softplus(self.head_dfe_overdrive(features))
            mu_overdrive = torch.clamp(raw_overdrive, max=L2O_OVERDRIVE_MAX)
        else:
            # Return zero tensor if overdrive is disabled
            mu_overdrive = torch.zeros_like(mu_base)

        mu_dfe = mu_base + mu_overdrive

        if update_ctle:
            mu_ctle = torch.tanh(self.head_ctle(features)) * L2O_CTLE_HEAD_SCALE
        else:
            mu_ctle = torch.zeros_like(mu_dfe)

        # RETURN mu_overdrive so we can regularize it in the training loop
        return mu_dfe, mu_ctle, mu_overdrive


# ==========================================
# 5. TBPTT Meta-Training Loop with History Buffer
# ==========================================
def cross_correlate_sync_batch(tx, rx, max_delay=50, sync_len=None):
    """
    Computes integer sample delay for each element in the batch using
    cross-correlation.
    """
    batch_size = tx.shape[0]
    seq_len = tx.shape[1]
    delays = []

    # Use up to 200 symbols for sync, but not more than available
    if sync_len is None:
        sync_len = min(200, seq_len - max_delay)

    if sync_len <= 0:
        raise ValueError(f"Sequence length {seq_len} too short for max_delay {max_delay}")

    tx_sync = tx[:, :sync_len]
    rx_sync = rx[:, :sync_len + max_delay]

    for i in range(batch_size):
        corrs = []
        for d in range(max_delay):
            c = torch.dot(tx_sync[i], rx_sync[i, d:d+sync_len])
            corrs.append(c.item())
        delays.append(torch.argmax(torch.abs(torch.tensor(corrs))).item())
    return delays


def train_learned_optimizer(channel_gen, dfe, ctle, learned_opt, epochs=100, batch_size=64,
                              unroll_len=50, history_len=10, ablate_ctle=False):
    """
    Trains the learned MLP optimizer using TBPTT with a sliding history buffer.

    Args:
        history_len: Number of past states to store in the buffer (default: 10)
    """
    if ablate_ctle:
        print("!!! RUNNING IN ABLATION MODE: CTLE CONTROL DISABLED !!!")

    meta_optimizer = torch.optim.Adam(learned_opt.parameters(), lr=1e-3)
    total_seq_len = 500
    ctle_update_rate = 10
    loss_history = []
    ss_history = []  # Steady-state MSE history
    unroll_history = []  # Track unroll length per epoch
    state_dim = 6  # [e_t, ema_error, norm_sq, dfe_weight[0], ctle_peaking, grad_proxy_ctle]

    for epoch in range(epochs):
        tx_symbols = torch.sign(torch.randn(batch_size, total_seq_len))
        rx_base, h_batch = channel_gen.generate_received_signal(tx_symbols, batch_size)

        with torch.no_grad():
            rx_init = ctle(rx_base, torch.ones(batch_size, 1) * 0.5)
            batch_delays = cross_correlate_sync_batch(tx_symbols, rx_init)

        common_delay = int(torch.median(torch.tensor(batch_delays, dtype=torch.float)).item())

        # Initialize DFE weights
        dfe_weights = torch.zeros(batch_size, dfe.num_taps)

        # Initialize Multi-Tap FFE
        # Initialize w_ffe with center tap = FFE_INIT, all others = 0
        w_ffe = torch.zeros(batch_size, FFE_TAPS)
        w_ffe[:, FFE_MAIN_CURSOR] = FFE_INIT
        ffe_buffer = torch.zeros(batch_size, FFE_TAPS)

        # Initialize CTLE latent parameter
        latent_peaking = torch.zeros(batch_size, 1)

        # Initialize RX buffer for CTLE
        rx_buffer = torch.zeros(batch_size, ctle.num_taps)

        # Initialize decision buffer and EMA
        decision_buffer = torch.zeros(batch_size, dfe.num_taps)
        ema_error = torch.ones(batch_size, 1)
        ema_beta = 0.95

        # ============================================================
        # History Buffer Initialization (replaces hidden_state)
        # ============================================================
        # Shape: [batch_size, history_len, state_dim]
        # Initially filled with zeros (neutral state)
        history_buffer = torch.zeros(batch_size, history_len, state_dim)

        effective_seq_len = total_seq_len - common_delay
        epoch_total_mse = 0
        epoch_ss_mse = 0  # Steady-state MSE (after burn-in)
        num_steps = 0
        ss_steps = 0

        for t_start in range(0, effective_seq_len, unroll_len):
            meta_optimizer.zero_grad()
            loss = 0

            # Detach all state variables at block boundaries
            dfe_weights = dfe_weights.detach()
            w_ffe = w_ffe.detach()
            ffe_buffer = ffe_buffer.detach()
            latent_peaking = latent_peaking.detach()
            decision_buffer = decision_buffer.detach()
            ema_error = ema_error.detach()
            rx_buffer = rx_buffer.detach()
            history_buffer = history_buffer.detach()  # Detach history for TBPTT

            current_block_len = min(unroll_len, effective_seq_len - t_start)

            for t in range(t_start, t_start + current_block_len):
                rx_t = rx_base[:, (t + common_delay):(t + common_delay + 1)]

                # Compute physical CTLE gain from latent parameter
                ctle_peaking = torch.sigmoid(latent_peaking)

                # Shift buffer and insert newest sample
                rx_buffer = torch.roll(rx_buffer, shifts=1, dims=1)
                rx_buffer[:, 0] = rx_t.squeeze(-1)

                # Compute dynamic CTLE taps
                current_taps = ctle.base_lp.unsqueeze(0) + ctle_peaking * ctle.base_hp.unsqueeze(0)
                rx_eq = torch.sum(rx_buffer * current_taps, dim=1, keepdim=True)

                # ============================================================
                # Multi-Tap FFE with Causality Shift
                # ============================================================
                # Shift FFE buffer and insert newest rx_eq at index 0
                ffe_buffer = torch.roll(ffe_buffer, shifts=1, dims=1)
                ffe_buffer[:, 0] = rx_eq.squeeze(-1)

                # DFE feedback computation
                dfe_feedback = torch.sum(decision_buffer * dfe_weights, dim=1, keepdim=True)

                # Causality shift: We make decisions for symbol at (t - FFE_MAIN_CURSOR)
                # Only compute error when we have enough history (target_idx >= 0)
                target_idx = t - FFE_MAIN_CURSOR
                can_compute_error = (target_idx >= 0)

                if can_compute_error:
                    # Compute FFE output (dot product of weights and buffer)
                    ffe_out = torch.sum(ffe_buffer * w_ffe, dim=1, keepdim=True)
                    # Total equalizer output
                    y_out = ffe_out - dfe_feedback

                    target_symbol = tx_symbols[:, target_idx:target_idx + 1]
                    e_t = target_symbol - y_out

                    # NLMS Normalization (includes full FFE buffer energy)
                    norm_sq = torch.sum(ffe_buffer ** 2, dim=1, keepdim=True) + \
                              torch.sum(decision_buffer ** 2, dim=1, keepdim=True) + 1e-6

                    # Gradient proxy for CTLE (use center tap aligned with main cursor)
                    grad_proxy_ctle = e_t * ffe_buffer[:, FFE_MAIN_CURSOR:FFE_MAIN_CURSOR+1] / norm_sq

                    # Build state features vector
                    state_features = torch.cat([
                        e_t, ema_error, norm_sq,
                        dfe_weights[:, 0:1],
                        ctle_peaking,
                        grad_proxy_ctle
                    ], dim=1)

                    # ============================================================
                    # Step 2: History Buffer Update
                    # Shift buffer and insert newest state
                    # ============================================================
                    history_buffer = torch.roll(history_buffer, shifts=1, dims=1)
                    history_buffer[:, 0, :] = state_features

                    # Flatten for MLP input: [batch, history_len * state_dim]
                    flat_history = history_buffer.view(batch_size, -1)

                    # Forward pass through MLP
                    update_ctle_flag = (t % ctle_update_rate == 0)
                    mu_dfe, mu_ctle, mu_overdrive = learned_opt(flat_history, update_ctle_flag)

                    if ablate_ctle:
                        mu_ctle = torch.zeros_like(mu_ctle)

                    # Update EMA
                    ema_error = ema_beta * ema_error + (1 - ema_beta) * (e_t.detach() ** 2)

                    # Update DFE weights
                    dfe_weights = dfe_weights - (mu_dfe * e_t * decision_buffer / norm_sq)

                    # Update Multi-Tap FFE weights
                    w_ffe = w_ffe + (mu_dfe * e_t * ffe_buffer / norm_sq)

                    # Update CTLE if flag is set
                    if update_ctle_flag:
                        latent_peaking = latent_peaking + mu_ctle

                    # Soft decisions for differentiable meta-training
                    tau = 0.1
                    soft_decision = torch.tanh(y_out / tau)

                    decision_buffer = torch.roll(decision_buffer, shifts=1, dims=1)
                    decision_buffer[:, 0] = soft_decision.squeeze(-1)

                    # Accumulate loss
                    step_mse = torch.mean(e_t ** 2)
                    loss += step_mse

                    # OVERDRIVE PENALTY: Heavily penalize the network for using the overdrive head
                    # This forces mu_overdrive to collapse to 0 during steady-state tracking
                    if learned_opt.use_two_head:
                        loss += 0.01 * torch.mean(mu_overdrive ** 2)

                    epoch_total_mse += step_mse.item()
                    num_steps += 1

                    # Steady-state tracking (after 200 symbols)
                    if t >= 200:
                        epoch_ss_mse += step_mse.item()
                        ss_steps += 1
                else:
                    # Warmup phase: not enough history for valid error computation
                    # Still update ffe_buffer to build up history for future steps
                    # Use zero error placeholder for state (will be ignored)
                    norm_sq = torch.sum(ffe_buffer ** 2, dim=1, keepdim=True) + \
                              torch.sum(decision_buffer ** 2, dim=1, keepdim=True) + 1e-6
                    grad_proxy_ctle = torch.zeros_like(ffe_buffer[:, FFE_MAIN_CURSOR:FFE_MAIN_CURSOR+1])

                    state_features = torch.cat([
                        torch.zeros_like(ema_error),  # placeholder e_t = 0
                        ema_error,
                        norm_sq,
                        dfe_weights[:, 0:1],
                        ctle_peaking,
                        grad_proxy_ctle
                    ], dim=1)

                    # ============================================================
                    # History Buffer Update (for warmup too)
                    # ============================================================
                    history_buffer = torch.roll(history_buffer, shifts=1, dims=1)
                    history_buffer[:, 0, :] = state_features

                    # Flatten for MLP input
                    flat_history = history_buffer.view(batch_size, -1)

                    # Forward pass through MLP
                    update_ctle_flag = (t % ctle_update_rate == 0)
                    mu_dfe, mu_ctle, mu_overdrive = learned_opt(flat_history, update_ctle_flag)

                    if ablate_ctle:
                        mu_ctle = torch.zeros_like(mu_ctle)

                    # Still update CTLE but with zero error (no learning)
                    if update_ctle_flag:
                        latent_peaking = latent_peaking + mu_ctle

                    # Use zero soft decision during warmup to avoid noise buildup
                    soft_decision = torch.zeros(batch_size, 1)
                    decision_buffer = torch.roll(decision_buffer, shifts=1, dims=1)
                    decision_buffer[:, 0] = soft_decision.squeeze(-1)

            # Backward pass and optimizer step
            if loss.requires_grad:
                (loss / current_block_len).backward()
                torch.nn.utils.clip_grad_norm_(learned_opt.parameters(), 1.0)
                meta_optimizer.step()

        avg_epoch_mse = epoch_total_mse / num_steps
        avg_ss_mse = epoch_ss_mse / ss_steps if ss_steps > 0 else avg_epoch_mse
        loss_history.append(avg_epoch_mse)
        ss_history.append(avg_ss_mse)
        unroll_history.append(unroll_len)  # Track unroll length for this epoch
        print(f"Epoch {epoch + 1}/{epochs} | Avg MSE: {avg_epoch_mse:.6f} | SS MSE: {avg_ss_mse:.6f}")

    return learned_opt, loss_history, ss_history


# ==========================================
# 6. Execution Block
# ==========================================
if __name__ == "__main__":
    import argparse

    # --- CONFIGURATION (from config.py) ---
    # See config.py for all common settings
    # --------------------------------------------------

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train Learned Optimizer (MLP)")
    parser = add_channel_args(parser)
    parser.add_argument("--two_head", action="store_true",
                        help="Enable the two-head overdrive architecture for DFE step size")
    args = parser.parse_args()

    # Instantiate modules
    print("Initializing modules...")
    channel_gen = get_channel_generator(args)
    ctle = DifferentiableCTLE(num_taps=CTLE_TAPS)
    dfe = DifferentiableDFE(num_taps=DFE_TAPS)

    # Create MLP-based optimizer instead of RNN
    learned_opt = MultiRateLearnedMLP(
        state_dim=L2O_STATE_DIM,
        history_len=L2O_MLP_HISTORY_LEN,
        hidden_dim=L2O_MLP_HIDDEN_DIM,
        use_two_head=args.two_head
    )

    print(f"Channel type: {args.channel_type}")
    print(f"Channel taps: {CH_TAPS}")
    print(f"DFE taps: {DFE_TAPS}")
    print(f"CTLE taps: {CTLE_TAPS}")
    print(f"SNR range: {SNR_RANGE} dB")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Unroll length (TBPTT): {UNROLL_LEN}")
    print(f"MLP history length: {L2O_MLP_HISTORY_LEN}")
    print(f"MLP hidden dimension: {L2O_MLP_HIDDEN_DIM}")
    print(f"Two-head overdrive enabled: {args.two_head}")
    print("-" * 50)

    # Train the learned optimizer
    print("Starting meta-training with MLP optimizer...")
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
    
    # Calculate final stage averages:
    # 1. Overall Average (last 20% of epochs)
    # 2. Steady-State Average (last 20% of epochs, after symbol 200)
    final_stage_start = int(len(loss_history) * 0.8)
    final_stage_avg_mse = sum(loss_history[final_stage_start:]) / len(loss_history[final_stage_start:])
    final_stage_ss_mse = sum(ss_history[final_stage_start:]) / len(ss_history[final_stage_start:])
    
    print(f"Final epoch Average MSE: {loss_history[-1]:.6f}")
    print(f"Final epoch Steady-State MSE: {ss_history[-1]:.6f}")
    print(f"Final stage (last 20%) Avg MSE: {final_stage_avg_mse:.6f}")
    print(f"Final stage (last 20%) Steady-State MSE: {final_stage_ss_mse:.6f}")
    
    # Save the trained model
    suffix = "_ablate_ctle" if ABLATE_CTLE else ""
    two_head_suffix = "_two_head" if args.two_head else ""
    model_path = f"./models/l2o_mlp_model_{args.channel_type}{suffix}{two_head_suffix}_dfe={DFE_TAPS}.pth"
    torch.save(trained_model.state_dict(), model_path)
    print(f"Trained model saved to {model_path}")
    print("-" * 50)


# Notes on MLP vs RNN:
# 1. The MLP receives explicit history via a sliding buffer
# 2. Gradient flow is more direct (no BPTT through recurrence needed)
# 3. The MLP must learn to interpret the history pattern
# 4. Memory is bounded by history_len * state_dim (fixed)
# 5. TBPTT still applies to the unrolled DFE/FFE updates
