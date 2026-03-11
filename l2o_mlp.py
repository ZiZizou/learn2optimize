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

# ===========================================
# 1. Channel Generation (from l2o_basic.py)
# ==========================================
class WirelineChannelGenerator:
    def __init__(self, num_taps=50, snr_range=(15, 25)):
        """
        Generates channels with lengths of 20-50 taps
        and AWGN with SNR of 15-25 dB[cite: 55].
        """
        self.num_taps = num_taps
        self.snr_range = snr_range

    def generate_batch(self, batch_size):
        t = torch.linspace(0, 5, self.num_taps)

        channels = []
        for _ in range(batch_size):
            tau = torch.empty(1).uniform_(0.5, 1.5).item()
            h = torch.exp(-t / tau)

            num_reflections = torch.randint(1, 4, (1,)).item()
            for _ in range(num_reflections):
                idx = torch.randint(5, self.num_taps, (1,)).item()
                h[idx] += torch.empty(1).uniform_(-0.2, 0.2).item()

            h = h / torch.norm(h)
            channels.append(h)

        return torch.stack(channels)

    def add_noise(self, signal, snr_db):
        """Adds AWGN based on specified SNR[cite: 55]."""
        snr_linear = 10 ** (snr_db / 10)
        signal_power = torch.mean(signal ** 2, dim=-1, keepdim=True)
        noise_power = signal_power / snr_linear
        noise = torch.randn_like(signal) * torch.sqrt(noise_power)
        return signal + noise

    def generate_received_signal(self, tx_symbols, batch_size):
        """
        Generates the full received signal by convolving symbols with channel
        and adding AWGN noise. Returns (rx_noisy, h_batch).
        """
        channel_len = self.num_taps
        tx_padded = F.pad(tx_symbols, (channel_len - 1, 0))
        tx_reshaped = tx_padded.view(1, batch_size, -1)
        h_batch = self.generate_batch(batch_size)

        # Main Cursor Normalization / Ideal AGC
        peak_vals, _ = torch.max(torch.abs(h_batch), dim=1, keepdim=True)
        h_batch = h_batch / (peak_vals + 1e-8)

        h_reshaped = h_batch.view(batch_size, 1, channel_len)
        h_reshaped_flipped = torch.flip(h_reshaped, dims=[-1])

        # Apply the channel filters using grouped convolution
        rx_clean = F.conv1d(tx_reshaped, h_reshaped_flipped, groups=batch_size)
        rx_clean = rx_clean.view(batch_size, -1)

        # Add AWGN
        snr_db = torch.empty(batch_size).uniform_(self.snr_range[0], self.snr_range[1])
        snr_linear = 10 ** (snr_db / 10)
        signal_power = torch.mean(rx_clean ** 2, dim=-1, keepdim=True)
        noise_power = signal_power / snr_linear.unsqueeze(-1)
        noise_std = torch.sqrt(noise_power)

        rx_noisy = rx_clean + noise_std * torch.randn_like(rx_clean)

        return rx_noisy, h_batch


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
    """
    def __init__(self, state_dim=6, history_len=10, hidden_dim=64):
        super().__init__()
        self.state_dim = state_dim
        self.history_len = history_len
        input_dim = state_dim * history_len

        # Deep feedforward network to handle increased input dimensionality
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Separate heads for DFE step size and CTLE step size
        self.head_dfe = nn.Linear(hidden_dim, 1)
        self.head_ctle = nn.Linear(hidden_dim, 1)

    def forward(self, history_buffer, update_ctle=False):
        """
        Args:
            history_buffer: [batch_size, state_dim * history_len] - flattened history
            update_ctle: whether to update CTLE at this timestep

        Returns:
            mu_dfe: [batch_size, 1] - DFE step size
            mu_ctle: [batch_size, 1] - CTLE step size
        """
        features = self.mlp(history_buffer)

        # Bound step sizes exactly as in l2o_basic.py
        mu_dfe = torch.sigmoid(self.head_dfe(features)) * 0.05

        if update_ctle:
            mu_ctle = torch.tanh(self.head_ctle(features)) * 0.05
        else:
            mu_ctle = torch.zeros_like(mu_dfe)

        return mu_dfe, mu_ctle


# ==========================================
# 5. TBPTT Meta-Training Loop with History Buffer
# ==========================================
def cross_correlate_sync_batch(tx, rx, max_delay=50):
    """
    Computes integer sample delay for each element in the batch using
    cross-correlation.
    """
    batch_size = tx.shape[0]
    delays = []
    sync_len = 200
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

        # Initialize 1-tap FFE (VGA)
        w_main = torch.ones(batch_size, 1) * FFE_INIT

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
            w_main = w_main.detach()
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

                # DFE and FFE computation
                dfe_feedback = torch.sum(decision_buffer * dfe_weights, dim=1, keepdim=True)
                y_out = (w_main * rx_eq) - dfe_feedback
                e_t = tx_symbols[:, t:t+1] - y_out

                # NLMS Normalization
                norm_sq = (rx_eq ** 2) + torch.sum(decision_buffer ** 2, dim=1, keepdim=True) + 1e-6

                # Gradient proxy for CTLE
                grad_proxy_ctle = e_t * rx_eq / norm_sq

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
                mu_dfe, mu_ctle = learned_opt(flat_history, update_ctle_flag)

                if ablate_ctle:
                    mu_ctle = torch.zeros_like(mu_ctle)

                # Update EMA
                ema_error = ema_beta * ema_error + (1 - ema_beta) * (e_t.detach() ** 2)

                # Update DFE weights
                dfe_weights = dfe_weights - (mu_dfe * e_t * decision_buffer / norm_sq)

                # Update FFE (w_main)
                w_main = w_main + (mu_dfe * e_t * rx_eq / norm_sq)

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
                epoch_total_mse += step_mse.item()
                num_steps += 1
                
                # Steady-state tracking (after 200 symbols)
                if t >= 200:
                    epoch_ss_mse += step_mse.item()
                    ss_steps += 1

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
    # --- CONFIGURATION (from config.py) ---
    # See config.py for all common settings
    # --------------------------------------------------

    # Instantiate modules
    print("Initializing modules...")
    channel_gen = WirelineChannelGenerator(num_taps=CH_TAPS, snr_range=SNR_RANGE)
    ctle = DifferentiableCTLE(num_taps=CTLE_TAPS)
    dfe = DifferentiableDFE(num_taps=DFE_TAPS)

    # Create MLP-based optimizer instead of RNN
    learned_opt = MultiRateLearnedMLP(
        state_dim=L2O_STATE_DIM,
        history_len=L2O_MLP_HISTORY_LEN,
        hidden_dim=L2O_MLP_HIDDEN_DIM
    )

    print(f"Channel taps: {CH_TAPS}")
    print(f"DFE taps: {DFE_TAPS}")
    print(f"CTLE taps: {CTLE_TAPS}")
    print(f"SNR range: {SNR_RANGE} dB")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Unroll length (TBPTT): {UNROLL_LEN}")
    print(f"MLP history length: {L2O_MLP_HISTORY_LEN}")
    print(f"MLP hidden dimension: {L2O_MLP_HIDDEN_DIM}")
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
    print("-" * 50)


# Notes on MLP vs RNN:
# 1. The MLP receives explicit history via a sliding buffer
# 2. Gradient flow is more direct (no BPTT through recurrence needed)
# 3. The MLP must learn to interpret the history pattern
# 4. Memory is bounded by history_len * state_dim (fixed)
# 5. TBPTT still applies to the unrolled DFE/FFE updates
