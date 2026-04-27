import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    CH_TAPS, SNR_RANGE, DFE_TAPS, CTLE_TAPS, CTLE_HP_ALPHA, FFE_TAPS, FFE_MAIN_CURSOR, FFE_INIT,
    L2O_STATE_DIM, NO_AGC_STATE_DIM, L2O_HIDDEN_DIM, L2O_DFE_HEAD_SCALE, L2O_CTLE_HEAD_SCALE, L2O_OVERDRIVE_MAX,
    EMA_BETA, L2O_OVERDRIVE_PENALTY,
    BATCH_SIZE, EPOCHS, UNROLL_LEN,
    ABLATE_CTLE, OVERSAMPLE_FACTOR, OVERSAMPLE_MODE,
    PHASE_SEARCH_MAX_DELAY, PHASE_SEARCH_SYNC_LEN,
    MU_FFE_MAX, MU_DFE_MAX, MU_CTLE_MAX, ERR_DIR_TAU,
)
from oversampling_utils import choose_best_symbol_phase, choose_best_symbol_phase_per_example, upsample_symbols

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


from wireline_channel import WirelineChannelGenerator
from utils import add_channel_args, get_channel_generator
from ctle_frequency_utils import apply_frequency_domain_ctle
from feature_normalization import (
    StreamingFeatureNormalizer,
    build_no_agc_state,
    build_agc_state,
) 
# It is easy to assume the CTLE has trainable weights because it is defined as a PyTorch nn.Module 
# (class DifferentiableCTLE(nn.Module)), whereas the DFE is just handled mathematically in the loop. 
# However, if you look at how they are actually implemented and updated, both are treated as dynamic 
#state variables rather than static neural network weights.

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
        hp[1] = -1 * CTLE_HP_ALPHA
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
# 2. Multi-Rate Learned Optimizer (Normalized/AGC mode)
# ==========================================
class MultiRateLearnedNLMS(nn.Module):
    def __init__(self, state_dim=L2O_STATE_DIM, hidden_dim=L2O_HIDDEN_DIM, use_two_head=False):
        super().__init__()
        self.rnn = nn.GRUCell(state_dim, hidden_dim)
        self.use_two_head = use_two_head

        self.head_dfe_base = nn.Linear(hidden_dim, 1)

        if self.use_two_head:
            self.head_dfe_overdrive = nn.Linear(hidden_dim, 1)

        self.head_ctle = nn.Linear(hidden_dim, 1)

    def forward(self, state_features, hidden_state, update_ctle=False):
        hidden_state = self.rnn(state_features, hidden_state)

        mu_base = torch.sigmoid(self.head_dfe_base(hidden_state)) * L2O_DFE_HEAD_SCALE

        if self.use_two_head:
            raw_overdrive = F.softplus(self.head_dfe_overdrive(hidden_state))
            mu_overdrive = torch.clamp(raw_overdrive, max=L2O_OVERDRIVE_MAX)
        else:
            mu_overdrive = torch.zeros_like(mu_base)

        mu_dfe = mu_base + mu_overdrive

        if update_ctle:
            mu_ctle = torch.tanh(self.head_ctle(hidden_state)) * L2O_CTLE_HEAD_SCALE
        else:
            mu_ctle = torch.zeros_like(mu_dfe)

        return mu_dfe, mu_ctle, mu_overdrive, hidden_state


# ==========================================
# 2b. Multi-Rate Learned Optimizer (No-AGC mode)
# ==========================================
class MultiRateLearnedNLMSNoAGC(nn.Module):
    """
    GRU-based learned optimizer for no-AGC mode with scale-aware feature normalization.

    Key design:
    1. StreamingFeatureNormalizer for per-feature standardization
    2. Scale-aware state via build_no_agc_state (normalizes by FFE RMS)
    3. Separate mu_ffe and mu_dfe heads (different input statistics)
    4. Smooth e_dir = tanh(e_t / tau_err) during training
    5. Optional LayerNorm on hidden state for stabilization
    """

    def __init__(self, state_dim=NO_AGC_STATE_DIM, hidden_dim=L2O_HIDDEN_DIM, use_two_head=False):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.use_two_head = use_two_head

        self.normalizer = StreamingFeatureNormalizer(feature_dim=state_dim, momentum=0.99, eps=1e-5)
        self.rnn = nn.GRUCell(state_dim, hidden_dim)

        self.head_ffe_base = nn.Linear(hidden_dim, 1)
        self.head_dfe_base = nn.Linear(hidden_dim, 1)

        if self.use_two_head:
            self.head_ffe_overdrive = nn.Linear(hidden_dim, 1)
            self.head_dfe_overdrive = nn.Linear(hidden_dim, 1)

        self.head_ctle = nn.Linear(hidden_dim, 1)

    def forward(self, z_raw, h, update_ctle=False, update_stats=False):
        """
        Args:
            z_raw: [batch, state_dim] raw unnormalized state features
            h: [batch, hidden_dim] RNN hidden state
            update_ctle: whether to compute CTLE step size
            update_stats: whether to update normalization stats (training only)

        Returns:
            mu_ffe: [batch, 1] FFE step size
            mu_dfe: [batch, 1] DFE step size
            mu_ctle: [batch, 1] CTLE step size
            h_new: [batch, hidden_dim] updated hidden state
        """
        z_norm = self.normalizer.normalize(z_raw)
        h_new = self.rnn(z_norm, h)

        if self.training and update_stats:
            self.normalizer.update(z_raw.detach())

        mu_ffe_base = torch.sigmoid(self.head_ffe_base(h_new)) * MU_FFE_MAX
        mu_dfe_base = torch.sigmoid(self.head_dfe_base(h_new)) * MU_DFE_MAX

        mu_ffe = mu_ffe_base
        mu_dfe = mu_dfe_base

        if self.use_two_head:
            mu_ffe = mu_ffe + torch.clamp(F.softplus(self.head_ffe_overdrive(h_new)), max=L2O_OVERDRIVE_MAX)
            mu_dfe = mu_dfe + torch.clamp(F.softplus(self.head_dfe_overdrive(h_new)), max=L2O_OVERDRIVE_MAX)

        mu_ctle = torch.tanh(self.head_ctle(h_new)) * MU_CTLE_MAX if update_ctle else torch.zeros_like(mu_dfe)

        return mu_ffe, mu_dfe, mu_ctle, h_new


# ==========================================
# Notes on why a RNN was used compared to a feedforward MLP:

# A simple Multi-Layer Perceptron (MLP) could map a static list of current features to a step size, but it would function as a purely reactive,
# memoryless controller. Optimization is fundamentally a sequential, history-dependent process.

# Memory of the "Optimization Path" (Statefulness)
# Optimization is an iterative process where the current state ($w_t$) is a direct result of all previous decisions.
#   * MLP Approach: An MLP is stateless. It would see the current error $e_t$ and the current gradient, but it wouldn't know if the error has been steadily decreasing for 50 cycles or if it’s oscillating wildly.
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
# monitors the error sequence, allowing it to "wake up" and temporarily increase μ to track the drift.

# * Why not just use a "Feature List" with an MLP?
# You could pass "history" features to an MLP (e.g., a window of the last 10 errors). However:
#   1. Fixed Window: You'd have to pre-define exactly how much history matters (e.g., exactly 10 steps). The RNN learns to keep what is useful and discard what isn't dynamically.
#   2. Vanishing Gradients: Simple MLPs struggle to learn dependencies across long sequences of updates. The GRU architecture is specifically designed to manage gradient flow through time, ensuring that the "meta-loss" from step 50 can effectively update   
#      the optimizer's weights to improve its decision at step 1.
# ==========================================

# ==========================================
# 3. TBPTT Meta-Training Loop with EMA Safeguard
# ==========================================
def cross_correlate_sync_batch(tx, rx, max_delay=50, sync_len=None):
    """
    Computes integer sample delay for each element in the batch using
    cross-correlation. This accounts for channel + CTLE group delay.
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

def train_learned_optimizer(channel_gen, dfe, ctle, learned_opt, epochs=100, batch_size=64, unroll_len=50, ablate_ctle=False):
    """
    Trains the learned optimizer using TBPTT with robust synchronization.
    """
    if ablate_ctle:
        print("!!! RUNNING IN ABLATION MODE: CTLE CONTROL DISABLED !!!")

    meta_optimizer = torch.optim.Adam(learned_opt.parameters(), lr=1e-3)
    total_seq_len = 500
    ctle_update_rate = 10
    loss_history = []
    ss_history = []  # Steady-state MSE history
    unroll_history = []  # Track unroll length per epoch

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

            rx_init, best_phase, common_delay = choose_best_symbol_phase(
                tx_symbols,
                rx_frontend,
                OVERSAMPLE_FACTOR,
                max_delay=PHASE_SEARCH_MAX_DELAY,
                sync_len=PHASE_SEARCH_SYNC_LEN,
            )

        hidden_state = torch.zeros(batch_size, L2O_HIDDEN_DIM)
        dfe_weights = torch.zeros(batch_size, dfe.num_taps)

        # Prompt 2: Integrate Multi-Tap FFE
        # Initialize w_ffe with center tap = FFE_INIT, all others = 0
        w_ffe = torch.zeros(batch_size, FFE_TAPS)
        w_ffe[:, FFE_MAIN_CURSOR] = FFE_INIT
        ffe_buffer = torch.zeros(batch_size, FFE_TAPS)

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

        for t_start in range(0, effective_seq_len, unroll_len):
            meta_optimizer.zero_grad()
            loss = 0
            
            hidden_state = hidden_state.detach()
            dfe_weights = dfe_weights.detach()
            w_ffe = w_ffe.detach()
            ffe_buffer = ffe_buffer.detach()
            latent_peaking = latent_peaking.detach()
            decision_buffer = decision_buffer.detach()
            ema_error = ema_error.detach()
            rx_buffer = rx_buffer.detach()
            
            current_block_len = min(unroll_len, effective_seq_len - t_start)

            for t in range(t_start, t_start + current_block_len):
                if ablate_ctle:
                    # O(1) fetch from pre-computed continuous-time waveform
                    # The CTLE was already applied as a static LTI pre-filter
                    rx_eq = rx_init[:, (t + common_delay):(t + common_delay + 1)]

                    # CTLE is static; use fixed peaking gain
                    ctle_peaking = torch.full((batch_size, 1), 0.5, device=latent_peaking.device)
                else:
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

                    # ============================================================
                    # NLMS Normalization (includes full FFE buffer energy)
                    # ============================================================
                    norm_sq = torch.sum(ffe_buffer ** 2, dim=1, keepdim=True) + \
                              torch.sum(decision_buffer ** 2, dim=1, keepdim=True) + 1e-6

                    # Gradient proxy for CTLE (use center tap aligned with main cursor)
                    grad_proxy_ctle = e_t * ffe_buffer[:, FFE_MAIN_CURSOR:FFE_MAIN_CURSOR+1] / norm_sq

                    state_features = torch.cat([
                        e_t, ema_error, norm_sq,
                        dfe_weights[:, 0:1],
                        ctle_peaking,
                        grad_proxy_ctle
                    ], dim=1)

                    update_ctle_flag = (t % ctle_update_rate == 0)
                    mu_dfe, mu_ctle, mu_overdrive, hidden_state = learned_opt(state_features, hidden_state, update_ctle_flag)

                    if ablate_ctle:
                        mu_ctle = torch.zeros_like(mu_ctle)

                    ema_error = ema_beta * ema_error + (1 - ema_beta) * (e_t.detach() ** 2)

                    # Gradient flow through mu_dfe and mu_ctle (using corrected norm_sq)
                    dfe_weights = dfe_weights - (mu_dfe * e_t * decision_buffer / norm_sq)

                    # ============================================================
                    # Update Multi-Tap FFE weights
                    # ============================================================
                    w_ffe = w_ffe + (mu_dfe * e_t * ffe_buffer / norm_sq)

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

                    # OVERDRIVE PENALTY: Penalize the network for using the overdrive head
                    # This encourages overdrive to collapse to 0 during steady-state tracking
                    if learned_opt.use_two_head:
                        loss += L2O_OVERDRIVE_PENALTY * torch.mean(mu_overdrive ** 2)

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

                    update_ctle_flag = (t % ctle_update_rate == 0)
                    mu_dfe, mu_ctle, mu_overdrive, hidden_state = learned_opt(state_features, hidden_state, update_ctle_flag)

                    if ablate_ctle:
                        mu_ctle = torch.zeros_like(mu_ctle)

                    # Still update buffers but with zero error (no learning)
                    if update_ctle_flag:
                        latent_peaking = latent_peaking + mu_ctle

                    # Use zero soft decision during warmup to avoid noise buildup
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
        unroll_history.append(unroll_len)
        print(f"Epoch {epoch + 1}/{epochs} | Avg MSE: {avg_epoch_mse:.6f} | SS MSE: {avg_ss_mse:.6f}")

    return learned_opt, loss_history, ss_history

# Currently, the code assumes the "hard work" of sampling the analog channel into discrete taps is already done, and it operates entirely in the normalized digital domain.

# ==========================================
# 4. Execution Block
# ==========================================
def train_learned_optimizer_rnn_noagc(channel_gen, dfe, ctle, learned_opt, epochs=100, batch_size=64,
                                        unroll_len=50, ablate_ctle=False):
    """
    TBPTT training loop for no-AGC RNN optimizer.

    Key fixes vs the broken MLP version:
    1. No warmup adaptation: during target_idx < 0, only update buffers
    2. Scale-aware state via build_no_agc_state (normalizes by FFE RMS)
    3. Separate mu_ffe and mu_dfe heads
    4. Smooth e_dir = tanh(e_t / tau_err) during training
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
                rx_frontend = ctle(rx_base, torch.ones(batch_size, 1) * 0.5)

            rx_init, best_phase, common_delay = choose_best_symbol_phase_per_example(
                tx_symbols,
                rx_frontend,
                OVERSAMPLE_FACTOR,
                max_delay=PHASE_SEARCH_MAX_DELAY,
                sync_len=PHASE_SEARCH_SYNC_LEN,
            )

        h_state = torch.zeros(batch_size, L2O_HIDDEN_DIM)
        dfe_weights = torch.zeros(batch_size, dfe.num_taps)

        w_ffe = torch.zeros(batch_size, FFE_TAPS)
        w_ffe[:, FFE_MAIN_CURSOR] = FFE_INIT
        ffe_buffer = torch.zeros(batch_size, FFE_TAPS)

        latent_peaking = torch.zeros(batch_size, 1)
        rx_buffer = torch.zeros(batch_size, ctle.num_taps)

        decision_buffer = torch.zeros(batch_size, dfe.num_taps)
        ema_error = torch.ones(batch_size, 1)

        effective_seq_len = total_seq_len - common_delay
        epoch_total_mse = 0
        epoch_ss_mse = 0
        num_steps = 0
        ss_steps = 0

        for t_start in range(0, effective_seq_len, unroll_len):
            meta_optimizer.zero_grad()
            loss = 0

            h_state = h_state.detach()
            dfe_weights = dfe_weights.detach()
            w_ffe = w_ffe.detach()
            ffe_buffer = ffe_buffer.detach()
            latent_peaking = latent_peaking.detach()
            decision_buffer = decision_buffer.detach()
            ema_error = ema_error.detach()
            rx_buffer = rx_buffer.detach()

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

                    update_ctle_flag = (t % ctle_update_rate == 0)
                    mu_ffe, mu_dfe, mu_ctle, h_state = learned_opt(
                        z_raw, h_state, update_ctle_flag, update_stats=True
                    )

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
                        loss += 0.01 * torch.mean(mu_ffe ** 2 + mu_dfe ** 2)

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

                    update_ctle_flag = (t % ctle_update_rate == 0)
                    mu_ffe, mu_dfe, mu_ctle, h_state = learned_opt(
                        z_raw, h_state, update_ctle_flag, update_stats=False
                    )

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

    parser = argparse.ArgumentParser(description="Train Learned Optimizer (Basic)")
    parser = add_channel_args(parser)
    parser.add_argument("--two_head", action="store_true",
                        help="Enable the two-head overdrive architecture for DFE step size")
    parser.add_argument("--no_agc", action="store_true",
                        help="Use no-AGC mode with scale-aware state normalization")
    args = parser.parse_args()

    print("Initializing modules...")
    channel_gen = get_channel_generator(args, samples_per_symbol=OVERSAMPLE_FACTOR)
    ctle = DifferentiableCTLE(num_taps=CTLE_TAPS)
    dfe = DifferentiableDFE(num_taps=DFE_TAPS)

    if args.no_agc or (args.channel_ir_norm_mode == "none" and not args.no_agc):
        if args.no_agc:
            print("!!! Running in NO-AGC mode (explicit --no_agc) !!!")
        else:
            print("!!! Running in NO-AGC mode (auto-selected: channel_ir_norm_mode=none) !!!")
        learned_opt = MultiRateLearnedNLMSNoAGC(
            state_dim=NO_AGC_STATE_DIM,
            hidden_dim=L2O_HIDDEN_DIM,
            use_two_head=args.two_head
        )
        print(f"Using no-AGC RNN optimizer with state_dim={NO_AGC_STATE_DIM}")
        print(f"Normalization: StreamingFeatureNormalizer + build_no_agc_state")
        print(f"FFE/DFE heads: separate (mu_ffe, mu_dfe)")
        print(f"Error direction: smooth tanh(e_t/ERR_DIR_TAU) during training")

        trained_model, loss_history, ss_history = train_learned_optimizer_rnn_noagc(
            channel_gen=channel_gen,
            dfe=dfe,
            ctle=ctle,
            learned_opt=learned_opt,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            unroll_len=UNROLL_LEN,
            ablate_ctle=ABLATE_CTLE
        )

        suffix = "_ablate_ctle" if ABLATE_CTLE else ""
        two_head_suffix = "_two_head" if args.two_head else ""
        model_path = f"./models/l2o_rnn_noagc_model_{args.channel_type}{suffix}{two_head_suffix}_dfe={DFE_TAPS}.pth"
    else:
        learned_opt = MultiRateLearnedNLMS(
            state_dim=L2O_STATE_DIM,
            hidden_dim=L2O_HIDDEN_DIM,
            use_two_head=args.two_head
        )
        print("Using normalized (AGC) RNN optimizer")

        trained_model, loss_history, ss_history = train_learned_optimizer(
            channel_gen=channel_gen,
            dfe=dfe,
            ctle=ctle,
            learned_opt=learned_opt,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            unroll_len=UNROLL_LEN,
            ablate_ctle=ABLATE_CTLE
        )

        suffix = "_ablate_ctle" if ABLATE_CTLE else ""
        two_head_suffix = "_two_head" if args.two_head else ""
        model_path = f"./models/l2o_basic_model_{args.channel_type}{suffix}{two_head_suffix}_dfe={DFE_TAPS}.pth"

    print(f"Channel type: {args.channel_type}")
    print(f"Channel taps: {CH_TAPS}")
    print(f"DFE taps: {DFE_TAPS}")
    print(f"CTLE taps: {CTLE_TAPS}")
    print(f"SNR range: {SNR_RANGE} dB")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Unroll length (TBPTT): {UNROLL_LEN}")
    print(f"Two-head overdrive enabled: {args.two_head}")
    print("-" * 50)

    print("Meta-training completed!")

    final_stage_start = int(len(loss_history) * 0.8)
    final_stage_avg_mse = sum(loss_history[final_stage_start:]) / len(loss_history[final_stage_start:])
    final_stage_ss_mse = sum(ss_history[final_stage_start:]) / len(ss_history[final_stage_start:])

    print(f"Final epoch Average MSE: {loss_history[-1]:.6f}")
    print(f"Final epoch Steady-State MSE: {ss_history[-1]:.6f}")
    print(f"Final stage (last 20%) Avg MSE: {final_stage_avg_mse:.6f}")
    print(f"Final stage (last 20%) Steady-State MSE: {final_stage_ss_mse:.6f}")

    torch.save(trained_model.state_dict(), model_path)
    print(f"Trained model saved to {model_path}")
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
#     Mathematical Implication: This allows the optimizer to exhibit adaptive step-sizing behavior — taking larger steps early when uncertainty is high, and smaller steps as the error stabilizes. It approximates the convergence benefits of second-order memory without the O(N2) matrix inversion penalty.


# A note about Truncated BPTT :
# 1. Chunking the Sequence
# The outer loop divides the total transmission sequence into smaller blocks defined by the unroll_len variable (which is set to 50).
#     for t_start in range(0, effective_seq_len, unroll_len):

# 2. Severing the Graph (Detaching)
# Before processing a new chunk, the code purposefully cuts the computational graph's connection to the previous chunk's history. If it didn't do this, PyTorch would try to calculate the chain rule all the way back to t=0, crashing your memory. It does this using the .detach() method on all state variables:
#     hidden_state = hidden_state.detach()
#     dfe_weights = dfe_weights.detach()
#     decision_buffer = decision_buffer.detach()

# 3. The Unrolled Forward Pass
# Inside the inner loop, the model processes the signal step-by-step for the duration of the chunk. It calculates the instantaneous error et​, squares it to find the step's Mean Squared Error (MSE), and accumulates it into a running total.
#     step_mse = torch.mean(e_t ** 2)
#     loss += step_mse

# 4. Flowing Backwards
# Once the inner loop finishes its 50-step sequence, the code takes the accumulated loss, averages it, and fires the backward pass. Because PyTorch built a computational graph during the inner loop, this command sends the gradients flowing backward through those 50 steps to figure out how the learned optimizer's weights should change.
#     (loss / current_block_len).backward()

# 5. Safeguards and Weight Updates

# Before actually updating the optimizer's neural network weights, the code applies gradient clipping. This acts as a safety valve to ensure the unrolled chain rule operations ∏∇w​f(wt​) don't result in numbers so large that they break the model. Finally, the optimizer takes a step to update its own parameters.

#     torch.nn.utils.clip_grad_norm_(learned_opt.parameters(), 1.0)
#     meta_optimizer.step()