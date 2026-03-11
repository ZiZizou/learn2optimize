import torch
import torch.nn as nn
import torch.nn.functional as F

# Import common configuration
from config import *

# CONCRETE OUTCOMES OF THIS CODE
# 1. Baseline Verification
# Your initial milestone requires implementing the simulation environment alongside classical baseline algorithms.
# Before introducing the neural network, you must have a verified, fixed-step NLMS and RLS implementation converging on
# the synthetic proxy channel. This establishes the numerical upper and lower bounds for your convergence speed and steady-state MSE.

# 2. Verifying Gradient Flow (The Overfit Test)
# The primary objective of this phase is to set up the unrolled computational graph and test meta-training on a single channel
# to ensure gradients flow.

# Because you are unrolling the DFE updates over time, the chain rule operations ∏∇w​f(wt​) expose you to exploding gradients
# in the unrolled graph. You must verify that the meta-loss derivative with respect to the optimizer network weights does not vanish to zero or explode to NaN over your chosen TBPTT horizon. If the network cannot memorize and drastically accelerate convergence on one static channel, it will fail on 5,000.

# 3. Teacher Forcing Sanity Check
# Decision-feedback instability, where error propagation creates biased gradients, is a primary risk. You must
#implement teacher forcing by using ground-truth past symbols for the feedback filter during the unrolled meta-training. The prototype phase must prove that this mechanism mathematically stabilizes the loss landscape compared to using hard decisions during the backward pass.

# 4. Parameter Projection Mechanics
# You proposed mapping the CTLE gain G∈[0,1] via a sigmoid function to project parameters to a feasible set. You must verify that the
# gradients pass cleanly through this projection. If the pre-activation values become too large, the sigmoid will saturate, killing the gradient and halting CTLE adaptation entirely.


# ===========================================
# 1. Channel Generation (Alternative Approach) [UNUSED]
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
        # Alternative to Rayleigh fading[cite: 54]: Simulate wireline low-pass RC/RLC step responses
        # Here we use a simplified proxy: exponential decay + random reflections
        t = torch.linspace(0, 5, self.num_taps)

        channels = []
        for _ in range(batch_size):
            tau = torch.empty(1).uniform_(0.5, 1.5).item()
            # Main cursor + post-cursor tail (skin-effect proxy)
            # Standard RC low-pass proxy: instantaneous rise, exponential decay
            # Peak (main cursor) is strictly at index 0
            h = torch.exp(-t / tau)

            # Add discrete reflections (via/connector proxy)
            num_reflections = torch.randint(1, 4, (1,)).item()
            for _ in range(num_reflections):
                idx = torch.randint(5, self.num_taps, (1,)).item()
                h[idx] += torch.empty(1).uniform_(-0.2, 0.2).item()

            h = h / torch.norm(h) # Normalize energy
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

        # 1. Pad tx_symbols for causal convolution (no look-ahead)
        tx_padded = F.pad(tx_symbols, (channel_len - 1, 0))

        # 2. Reshape for grouped 1D convolution
        tx_reshaped = tx_padded.view(1, batch_size, -1)
        h_batch = self.generate_batch(batch_size)

        # Prompt 2: Main Cursor Normalization / Ideal AGC
        # Normalize so the main cursor (peak amplitude) is exactly 1.0
        peak_vals, _ = torch.max(torch.abs(h_batch), dim=1, keepdim=True)
        h_batch = h_batch / (peak_vals + 1e-8)

        h_reshaped = h_batch.view(batch_size, 1, channel_len)

        # Prompt 1: Fix Time-Reversal
        # F.conv1d computes cross-correlation, not convolution.
        # Flip the channel weights to physically model causal convolution.
        h_reshaped_flipped = torch.flip(h_reshaped, dims=[-1])

        # 3. Apply the channel filters using grouped convolution
        rx_clean = F.conv1d(tx_reshaped, h_reshaped_flipped, groups=batch_size)
        rx_clean = rx_clean.view(batch_size, -1)

        # 4. Add AWGN
        snr_db = torch.empty(batch_size).uniform_(self.snr_range[0], self.snr_range[1])
        snr_linear = 10 ** (snr_db / 10)
        signal_power = torch.mean(rx_clean ** 2, dim=-1, keepdim=True)
        noise_power = signal_power / snr_linear.unsqueeze(-1)
        noise_std = torch.sqrt(noise_power)

        rx_noisy = rx_clean + noise_std * torch.randn_like(rx_clean)

        return rx_noisy, h_batch

# =============================================================================


# either the CTLE parameters nor the DFE parameters are encoded as standard trainable weights.
# It is easy to assume the CTLE has trainable weights because it is defined as a PyTorch nn.Module
# (class DifferentiableCTLE(nn.Module)), whereas the DFE is just handled mathematically in the loop.
# However, if you look at how they are actually implemented and updated, both are treated as dynamic
# state variables rather than static neural network weights.

# ==========================================
# 1. Differentiable Parametric CTLE
# ==========================================
class DifferentiableCTLE(nn.Module):
    def __init__(self, num_taps=15):
        super().__init__()
        self.num_taps = num_taps
        self.register_buffer('base_lp', self._generate_base_lp())
        self.register_buffer('base_hp', self._generate_base_hp())

    def _generate_base_lp(self):
        # Flat response (pass-through / all-pass proxy)
        lp = torch.zeros(self.num_taps)
        lp[0] = 1.0
        return lp

    def _generate_base_hp(self):
        # High-pass filter: amplifies transitions, subtracts immediate post-cursor
        hp = torch.zeros(self.num_taps)
        hp[0] = 1.0
        hp[1] = -CTLE_HP_ALPHA
        return hp

    def forward(self, rx_signal, peaking_gain):
        """
        rx_signal: [batch, seq_len]
        peaking_gain: [batch, 1] - The parameter adapted by the learned optimizer
        """
        # filter_taps shape: [batch, 1, num_taps] for grouped conv1d
        filter_taps = self.base_lp.unsqueeze(0) + peaking_gain.unsqueeze(-1) * self.base_hp.unsqueeze(0)

        # Apply filter using 1D convolution
        rx_padded = F.pad(rx_signal, (self.num_taps - 1, 0))
        batch_size = rx_signal.shape[0]
        rx_reshaped = rx_padded.view(1, batch_size, -1)

        # Prompt 1: Fix Time-Reversal
        # F.conv1d computes cross-correlation, not convolution.
        # Flip the filter taps to physically model causal FIR filtering.
        filter_taps_flipped = torch.flip(filter_taps, dims=[-1])

        filtered_rx = F.conv1d(rx_reshaped, filter_taps_flipped, groups=batch_size)
        return filtered_rx.view(batch_size, -1)

# ==========================================
# 2. Differentiable Decision Feedback Equalizer (DFE)
# ==========================================
class DifferentiableDFE(nn.Module):
    def __init__(self, num_taps=10):
        """
        Implements a Decision Feedback Equalizer (DFE) that uses past symbol
        decisions to cancel post-cursor Inter-Symbol Interference (ISI).

        Scientific Definition: The feedback filter computes an interference penalty
        F_t = sum(w_k * d_hat_{t-k}), where w_k are the filter taps and d_hat are
        past decisions.

        The output of the DFE is y_t = x_t - F_t.
        """
        super().__init__()
        self.num_taps = num_taps

    def forward(self, rx_eq, decision_buffer, dfe_weights):
        """
        rx_eq: [batch, 1] - CTLE output
        decision_buffer: [batch, num_taps] - Past symbols (decisions)
        dfe_weights: [batch, num_taps] - Current learned DFE taps

        Returns:
            y_out: [batch, 1] - Equalized output after DFE subtraction
        """
        # Compute inner product of weights and buffer (feedback interference)
        # This computes F_t = sum(w_k * d_hat_{t-k})
        feedback = torch.sum(decision_buffer * dfe_weights, dim=1, keepdim=True)

        # Subtract interference from CTLE output
        y_out = rx_eq - feedback

        return y_out

# ==========================================
# 2. Multi-Rate Learned Optimizer
# ==========================================
class MultiRateLearnedNLMS(nn.Module):
    def __init__(self, state_dim=5, hidden_dim=32):
        super().__init__()
        # State: [e_t, EMA_t, norm(X), current_mu_dfe, current_peaking]
        self.rnn = nn.GRUCell(state_dim, hidden_dim)

        # Separate heads for DFE step size and CTLE step size
        self.head_dfe = nn.Linear(hidden_dim, 1)
        self.head_ctle = nn.Linear(hidden_dim, 1)

    def forward(self, state_features, hidden_state, update_ctle=False):
        hidden_state = self.rnn(state_features, hidden_state)

        # Prompt 3: Bound Initial L2O Step Sizes
        # Reduce upper bound from 0.5 to 0.05 to prevent early-epoch instability
        mu_dfe = torch.sigmoid(self.head_dfe(hidden_state)) * 0.05

        if update_ctle:
            # CTLE step size is typically smaller to prevent instability
            mu_ctle = torch.tanh(self.head_ctle(hidden_state)) * 0.05
        else:
            mu_ctle = torch.zeros_like(mu_dfe)

        return mu_dfe, mu_ctle, hidden_state


# ==========================================
# Notes on why a RNN was used compared to a feedforward MLP:

# A simple Multi-Layer Perceptron (MLP) could map a static list of current features to a step size, but it would function as a purely reactive,
# memoryless controller. Optimization is fundamentally a sequential, history-dependent process.

# Memory of the "Optimization Path" (Statefulness)
# Optimization is an iterative process where the current state ($w_t$) is a direct result of all previous decisions.
#   * MLP Approach: An MLP is stateless. It would see the current error $e_t$ and the current gradient, but it wouldn't know if the error has been steadily decreasing for 50 cycles or if it's oscillating wildly.
#   * RNN Advantage: The hidden_state in your MultiRateLearnedNLMS acts as a "memory bank." It captures the momentum and curvature of the loss landscape. It can distinguish between "I'm in a flat plateau and need to speed up" vs. "I'm near the minimum and
#     need to decelerate to prevent overshoot.

# * Learning the "Dynamics" via BPTT
# In the code, you unroll the loop for unroll_len=50 and then call .backward(). This is the "BPTT" part.
#   * Unique Feature: BPTT allows the RNN to understand the long-term consequences of its current step size.
#   * The Logic: If the RNN picks a massive $\mu$ at $t=5$, it might lower the error at $t=6$, but cause the DFE to explode at $t=20$. Because the loss is calculated over the entire 50-step window, the RNN is penalized for "short-sighted" gains. An MLP,
#     typically trained on instantaneous samples, lacks this "look-ahead" capability during its own training phase.

# * Handling Channel Drift
# An MLP trained on static features will completely fail to adapt to out-of-distribution channel drift. If the channel slowly shifts due to temperature
# changes, an MLP has no temporal mechanism to realize the steady-state weights are slowly decaying in effectiveness. The RNN's hidden state continuously
# monitors the error sequence, allowing it to "wake up" and temporarily increase $\mu$ to track the drift.

# * Why not just use a "Feature List" with an MLP?
# You could pass "history" features to an MLP (e.g., a window of the last 10 errors). However:
#   1. Fixed Window: You'd have to pre-define exactly how much history matters (e.g., exactly 10 steps). The RNN learns to keep what is useful and discard what isn't dynamically.
#   2. Vanishing Gradients: Simple MLPs struggle to learn dependencies across long sequences of updates. The GRU architecture is specifically designed to manage gradient flow through time, ensuring that the "meta-loss" from step 50 can effectively update
#      the optimizer's weights to improve its decision at step 1.
# ==========================================

# ==========================================
# 3. TBPTT Meta-Training Loop with EMA Safeguard
# ==========================================
def cross_correlate_sync_batch(tx, rx, max_delay=50):
    """
    Computes integer sample delay for each element in the batch using
    cross-correlation. This accounts for channel + CTLE group delay.
    """
    batch_size = tx.shape[0]
    delays = []
    # Sync using a subset of the sequence for speed
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
                            initial_unroll=10, max_unroll=100, unroll_step_epoch=20, ablate_ctle=False):
    """
    Trains the learned optimizer using TBPTT with progressive curriculum learning.

    Curriculum Parameters:
        initial_unroll: Starting TBPTT truncation length (default: 10)
        max_unroll: Maximum TBPTT truncation length (default: 50)
        unroll_step_epoch: Epoch interval between unroll length increases (default: 20)
    """
    if ablate_ctle:
        print("!!! RUNNING IN ABLATION MODE: CTLE CONTROL DISABLED !!!")

    meta_optimizer = torch.optim.Adam(learned_opt.parameters(), lr=1e-3)
    total_seq_len = 500
    ctle_update_rate = 10
    loss_history = []
    ss_history = []  # Steady-state MSE history
    unroll_history = []

    for epoch in range(epochs):
        # ============================================================
        # Prompt 2: Mathematical Formulation of the Progressive Schedule
        # Calculate the active unroll length for the current epoch
        # Lactive = min(Lmax, Linit + (floor(epoch/step_epoch) * ΔL))
        # ============================================================
        current_unroll_len = min(
            max_unroll,
            initial_unroll + (epoch // unroll_step_epoch) * UNROLL_DELTA
        )
        print(f"Epoch {epoch + 1}/{epochs} | Active TBPTT Horizon: {current_unroll_len}")

        tx_symbols = torch.sign(torch.randn(batch_size, total_seq_len))
        rx_base, h_batch = channel_gen.generate_received_signal(tx_symbols, batch_size)

        with torch.no_grad():
            rx_init = ctle(rx_base, torch.ones(batch_size, 1) * 0.5)
            batch_delays = cross_correlate_sync_batch(tx_symbols, rx_init)

        common_delay = int(torch.median(torch.tensor(batch_delays, dtype=torch.float)).item())

        hidden_state = torch.zeros(batch_size, 32)
        dfe_weights = torch.zeros(batch_size, dfe.num_taps)

        # Prompt 2: Integrate 1-Tap FFE (VGA)
        # Initialize w_main to FFE_INIT to scale the main cursor amplitude
        w_main = torch.ones(batch_size, 1) * FFE_INIT

        # Prompt 4: Unbounded Latent Parameter Projection
        # Use unbounded latent variable, sigmoid to get physical [0,1] gain
        latent_peaking = torch.zeros(batch_size, 1)  # sigmoid(0) = 0.5

        # Prompt 3: Stateful CTLE in the Unrolled Loop
        # Initialize RX buffer for scalar sample-by-sample CTLE processing
        rx_buffer = torch.zeros(batch_size, ctle.num_taps)

        decision_buffer = torch.zeros(batch_size, dfe.num_taps)
        ema_error = torch.ones(batch_size, 1)
        ema_beta = 0.95

        effective_seq_len = total_seq_len - common_delay
        epoch_total_mse = 0
        epoch_ss_mse = 0  # Steady-state MSE (after burn-in)
        num_steps = 0
        ss_steps = 0

        # ============================================================
        # Prompt 3: Dynamic Graph Truncation Application
        # Use current_unroll_len instead of static unroll_len
        # ============================================================
        for t_start in range(0, effective_seq_len, current_unroll_len):
            meta_optimizer.zero_grad()
            loss = 0

            hidden_state = hidden_state.detach()
            dfe_weights = dfe_weights.detach()
            w_main = w_main.detach()
            latent_peaking = latent_peaking.detach()
            decision_buffer = decision_buffer.detach()
            ema_error = ema_error.detach()
            rx_buffer = rx_buffer.detach()

            # Use current_unroll_len for block length calculation
            current_block_len = min(current_unroll_len, effective_seq_len - t_start)

            for t in range(t_start, t_start + current_block_len):
                rx_t = rx_base[:, (t + common_delay):(t + common_delay + 1)]

                # Prompt 4: Unbounded Latent Parameter Projection
                # Compute physical CTLE gain from unbounded latent parameter
                ctle_peaking = torch.sigmoid(latent_peaking)

                # Prompt 3: Stateful CTLE in the Unrolled Loop
                # Shift buffer and insert newest sample
                rx_buffer = torch.roll(rx_buffer, shifts=1, dims=1)
                rx_buffer[:, 0] = rx_t.squeeze(-1)

                # Compute dynamic taps [batch, CTLE_TAPS]
                current_taps = ctle.base_lp.unsqueeze(0) + ctle_peaking * ctle.base_hp.unsqueeze(0)

                # Direct dot product (FIR filtering) - bypass F.conv1d
                rx_eq = torch.sum(rx_buffer * current_taps, dim=1, keepdim=True)

                # Prompt 2: Integrate 1-Tap FFE (VGA)
                # y_out = w_main * rx_eq - dfe_feedback
                dfe_feedback = torch.sum(decision_buffer * dfe_weights, dim=1, keepdim=True)
                y_out = (w_main * rx_eq) - dfe_feedback
                e_t = tx_symbols[:, t:t+1] - y_out

                # Prompt 2: Corrected NLMS Normalization
                # Use squared L2 norm including forward signal power
                norm_sq = (rx_eq ** 2) + torch.sum(decision_buffer ** 2, dim=1, keepdim=True) + 1e-6
                # Prompt 2: Provide Gradient Context to the State
                # Calculate differentiable gradient proxy for CTLE
                grad_proxy_ctle = e_t * rx_eq / norm_sq

                state_features = torch.cat([
                    e_t, ema_error, norm_sq,
                    dfe_weights[:, 0:1],
                    ctle_peaking,
                    grad_proxy_ctle  # Prompt 2: Added CTLE gradient context
                ], dim=1)

                update_ctle_flag = (t % ctle_update_rate == 0)
                mu_dfe, mu_ctle, hidden_state = learned_opt(state_features, hidden_state, update_ctle_flag)

                if ablate_ctle:
                    mu_ctle = torch.zeros_like(mu_ctle)

                ema_error = ema_beta * ema_error + (1 - ema_beta) * (e_t.detach() ** 2)

                # Gradient flow through mu_dfe and mu_ctle (using corrected norm_sq)
                dfe_weights = dfe_weights - (mu_dfe * e_t * decision_buffer / norm_sq)

                # Prompt 2: Update w_main (1-tap FFE) using same mu_dfe step size
                # w_main = w_main + (mu_dfe * e_t * rx_eq / norm_sq)
                w_main = w_main + (mu_dfe * e_t * rx_eq / norm_sq)

                if update_ctle_flag:
                    # Prompt 1: Differentiable Direct Policy for CTLE
                    # Remove torch.sign() to enable gradient flow; use direct signed delta
                    latent_peaking = latent_peaking + mu_ctle

                # Prompt 3: Soft Decisions for Differentiable Meta-Training
                # Replace hard teacher forcing with differentiable soft-sign
                tau = 0.1  # Temperature parameter
                soft_decision = torch.tanh(y_out / tau)

                decision_buffer = torch.roll(decision_buffer, shifts=1, dims=1)
                decision_buffer[:, 0] = soft_decision.squeeze(-1)

                step_mse = torch.mean(e_t ** 2)
                loss += step_mse
                epoch_total_mse += step_mse.item()
                num_steps += 1
                
                # Steady-state tracking (after 200 symbols)
                if t >= 200:
                    epoch_ss_mse += step_mse.item()
                    ss_steps += 1

            if loss.requires_grad:
                (loss / current_block_len).backward()
                torch.nn.utils.clip_grad_norm_(learned_opt.parameters(), 1.0)
                meta_optimizer.step()

        avg_epoch_mse = epoch_total_mse / num_steps
        avg_ss_mse = epoch_ss_mse / ss_steps if ss_steps > 0 else avg_epoch_mse
        loss_history.append(avg_epoch_mse)
        ss_history.append(avg_ss_mse)
        unroll_history.append(current_unroll_len)
        print(f"Epoch {epoch + 1}/{epochs} | Horizon: {current_unroll_len} | Avg MSE: {avg_epoch_mse:.6f} | SS MSE: {avg_ss_mse:.6f}")

    return learned_opt, loss_history, ss_history, unroll_history


# ==========================================
# 4. Execution Block
# ==========================================
if __name__ == "__main__":
    # --- CONFIGURATION (from config.py) ---
    # See config.py for all common settings
    # Note: Override EPOCHS for longer progressive training
    EPOCHS = 620  # Override default for progressive training
    # --------------------------------------------------

    # Instantiate modules
    print("Initializing modules...")
    channel_gen = WirelineChannelGenerator(num_taps=CH_TAPS, snr_range=SNR_RANGE)
    ctle = DifferentiableCTLE(num_taps=CTLE_TAPS)
    dfe = DifferentiableDFE(num_taps=DFE_TAPS)
    learned_opt = MultiRateLearnedNLMS(state_dim=6, hidden_dim=32)

    print(f"Channel taps: {CH_TAPS}")
    print(f"DFE taps: {DFE_TAPS}")
    print(f"CTLE taps: {CTLE_TAPS}")
    print(f"SNR range: {SNR_RANGE} dB")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Curriculum Schedule:")
    print(f"  - Initial unroll: {INITIAL_UNROLL}")
    print(f"  - Max unroll: {MAX_UNROLL}")
    print(f"  - Step epoch: {UNROLL_STEP_EPOCH}")
    print("-" * 50)

    # Train the learned optimizer
    print("Starting meta-training with progressive curriculum...")
    trained_model, loss_history, ss_history, unroll_history = train_learned_optimizer(
        channel_gen=channel_gen,
        dfe=dfe,
        ctle=ctle,
        learned_opt=learned_opt,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        initial_unroll=INITIAL_UNROLL,
        max_unroll=MAX_UNROLL,
        unroll_step_epoch=UNROLL_STEP_EPOCH,
        ablate_ctle=ABLATE_CTLE
    )

    print("-" * 50)
    print("Meta-training completed!")
    
    # Calculate final stage averages:
    # 1. Overall Average (epochs where unroll == max_unroll)
    # 2. Steady-State Average (epochs where unroll == max_unroll, after symbol 200)
    final_stage_mses = [loss_history[i] for i, u in enumerate(unroll_history) if u == MAX_UNROLL]
    final_stage_ss_mses = [ss_history[i] for i, u in enumerate(unroll_history) if u == MAX_UNROLL]
    
    final_avg_mse = sum(final_stage_mses) / len(final_stage_mses) if final_stage_mses else loss_history[-1]
    final_ss_mse = sum(final_stage_ss_mses) / len(final_stage_ss_mses) if final_stage_ss_mses else ss_history[-1]

    print(f"Final epoch Average MSE: {loss_history[-1]:.6f}")
    print(f"Final epoch Steady-State MSE: {ss_history[-1]:.6f}")
    print(f"Final stage (unroll={MAX_UNROLL}) Avg MSE: {final_avg_mse:.6f}")
    print(f"Final stage (unroll={MAX_UNROLL}) Steady-State MSE: {final_ss_mse:.6f}")
    print("-" * 50)

# A note on why ablation of CTLE does not cause that much of a difference in performance:
# Final average MSE: 0.484707 [ctle ablation] vs Final average MSE: 0.464771[no ablation]
# Convolving a 2-tap difference filter against a 50-tap heavy exponential tail alters the
# immediate post-cursor but mathematically cannot shorten a 50-tap tail enough to fit inside a 10-tap Decision Feedback Equalizer (DFE).
# The optimizer extracts a tiny ~0.02 MSE improvement by slightly reshaping the main cursor, but it hits the physical limitation of the
# 2-tap proxy. To demonstrate a massive performance gap between ablate_ctle = True and False, the CTLE must be given a physical impulse
# response capable of acting as an inverse filter to the channel's dominant pole.

# A note on the difference in memory for the different implementations
# 1. Benchmark NLMS: Memoryless (First-Order Instantaneous)
# The Normalized Least Mean Squares (NLMS) algorithm effectively has zero temporal memory of the optimization landscape.
#     State Variable: The current filter weights (wt​).
#     Mechanism: NLMS is a stochastic gradient descent algorithm. Its update step at time t relies exclusively on the instantaneous error (et​) and the current contents of the decision buffer (xt​).
#     Mathematical Implication: It does not remember past errors, past gradients, or the trajectory of the weights. Because it evaluates the loss surface strictly locally at the present symbol, it is highly susceptible to the eigenvalue spread of the channel, resulting in the slow convergence noted in classical baselines.

# 2. Benchmark RLS: Deterministic Covariance (Second-Order Long-Term)
# Recursive Least Squares (RLS) utilizes explicit, mathematically rigid long-term memory to solve the exact least-squares problem at every step.
#     State Variable: The inverse correlation matrix (Pt​).
#     Mechanism: RLS explicitly remembers the geometry of the entire history of input signals. It accumulates the cross-correlation of all past inputs into a matrix, exponentially weighted by a forgetting factor (λ).
#     Mathematical Implication: By remembering the deterministic curvature of the input space, RLS can essentially jump straight toward the optimal MMSE solution regardless of channel eigenvalue spread. This makes it incredibly fast but computationally prohibitive (O(N2)) for high-speed hardware.

# 3. Learned Optimizer: Non-Linear Recurrent Memory (Heuristic/Trajectory)
# Your proposed learned optimizer (L2O) replaces rigid DSP mathematics with a parameterized, non-linear memory cell .
#     State Variables: The neural network's hidden state (ht​) via a GRU, supplemented by engineered momentum features like the Exponential Moving Average (EMA) of the error.
#     Mechanism: Instead of memorizing input covariance like RLS, the GRU learns what temporal features are useful to remember. It tracks the trajectory of the error surface over time. If the sequence of past gradients indicates the weights are far from convergence, the hidden state retains this momentum and outputs a large step size (μt​).
#     Mathematical Implication: This allows the optimizer to exhibit "gear shifting" —taking large steps early when uncertainty is high, and small steps late when the EMA error stabilizes. It approximates the convergence benefits of second-order memory without the O(N2) matrix inversion penalty.
