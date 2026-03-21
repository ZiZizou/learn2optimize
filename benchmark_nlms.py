import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd

# Import common configuration
from config import *

# ==========================================

from wireline_channel import WirelineChannelGenerator

# ==========================================
# 2. Differentiable Parametric CTLE (Identical to l2o_basic.py)
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
        peaking_gain: [batch, 1]
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
# 3. NLMS Algorithm with Multi-Tap FFE
# ==========================================
def run_nlms_dfe(rx_signal, tx_symbols, num_taps, mu=0.1, eps=1e-6, teacher_forcing=True,
                 use_gear_shift=False, mu_fast=0.2, mu_slow=0.01, gear_threshold=0.5, ema_alpha=0.05):
    """
    NLMS DFE with integrated Multi-Tap FFE (Feed-Forward Equalizer).

    The multi-tap FFE spans multiple symbol periods to cancel precursor ISI
    before the causal DFE handles post-cursor energy.

    Forward: y_t = sum(w_ffe[i] * rx_eq[t-i]) - w_fb^T * d_fb

    Parameters:
        mu: Static step size (used when use_gear_shift=False)
        use_gear_shift: If True, enable variable step-size (gear-shifting)
        mu_fast: Acquisition gear step size (default: 0.2)
        mu_slow: Tracking gear step size (default: 0.01)
        gear_threshold: MSE threshold to trigger gear shift (default: 0.5)
        ema_alpha: Smoothing factor for error variance (default: 0.05)
    """
    seq_len = rx_signal.shape[0]
    weights = torch.zeros(num_taps, dtype=torch.float32)

    # Initialize Multi-Tap FFE with center tap = FFE_INIT
    w_ffe = torch.zeros(FFE_TAPS, dtype=torch.float32)
    w_ffe[FFE_MAIN_CURSOR] = FFE_INIT
    ffe_buffer = torch.zeros(FFE_TAPS, dtype=torch.float32)

    decision_buffer = torch.zeros(num_taps, dtype=torch.float32)
    mse_log = []

    # Step size selection: static or gear-shifting
    if use_gear_shift:
        current_mu = mu_fast  # Start with fast acquisition
        ema_error_sq = 1.0    # Start high to prevent premature shifting
    else:
        current_mu = mu        # Use static mu

    for t in range(seq_len):
        # Shift FFE buffer and insert newest sample at index 0
        ffe_buffer = torch.roll(ffe_buffer, shifts=1, dims=0)
        ffe_buffer[0] = rx_signal[t]

        # DFE feedback computation
        feedback = torch.dot(decision_buffer, weights)

        # Causality shift: target symbol is at t - FFE_MAIN_CURSOR
        target_idx = t - FFE_MAIN_CURSOR
        if target_idx >= 0:
            # Compute FFE output (dot product)
            ffe_out = torch.dot(w_ffe, ffe_buffer)
            # Total equalizer output
            eq_out = ffe_out - feedback
            decision = torch.sign(eq_out)
            e_t = tx_symbols[target_idx] - eq_out
            mse_log.append((e_t.item())**2)

            # Gear-shifting: Update EMA of squared error
            if use_gear_shift:
                ema_error_sq = (1 - ema_alpha) * ema_error_sq + ema_alpha * (e_t.item() ** 2)

                # Gear-shifting logic
                if ema_error_sq < gear_threshold:
                    current_mu = mu_slow
                else:
                    current_mu = mu_fast

            # Normalization includes full FFE buffer energy and feedback buffer
            norm_sq = torch.dot(ffe_buffer, ffe_buffer) + torch.dot(decision_buffer, decision_buffer) + eps

            # Update feedback weights using current_mu
            update_step = (current_mu * e_t * decision_buffer) / norm_sq
            weights = weights - update_step

            # Update Multi-Tap FFE weights
            update_ffe = (current_mu * e_t * ffe_buffer) / norm_sq
            w_ffe = w_ffe + update_ffe

            decision_buffer = torch.roll(decision_buffer, shifts=1)
            if teacher_forcing:
                decision_buffer[0] = tx_symbols[target_idx]
            else:
                decision_buffer[0] = decision
        else:
            # Warmup phase: not enough history for valid error computation
            # Log zero error but still build up ffe_buffer
            mse_log.append(0.0)

            # Update feedback weights with zero error (no learning)
            decision_buffer = torch.roll(decision_buffer, shifts=1)
            if teacher_forcing:
                decision_buffer[0] = tx_symbols[t]
            else:
                decision_buffer[0] = 0.0

    return torch.tensor(mse_log), weights, w_ffe


# ==========================================
# RLS Implementation Explanation
#   1. Initialization (Setting the Stage)
#   Code: Lines 153–155


#    1 # Initialize weights w_0 = 0, but force first element (main tap) to 1.0
#    2 weights = torch.zeros(system_order, dtype=torch.float32)
#    3 weights[0] = 1.0  # Main tap initialized to 1.0
#    4
#    5 # Initialize inverse correlation matrix P_0 = delta^(-1) * I
#    6 P = torch.eye(system_order, dtype=torch.float32) / delta
#    * Intuition: We start with a "best guess" for weights.
#    * The Matrix $P$: This is the most important part of RLS. It represents our uncertainty. We initialize it with large values (via 1/delta), telling the    
#      algorithm: "I don't know anything yet, so be ready to make big changes."

#   ---


#   2. Constructing the Input ($u_t$)
#   Code: Line 166
#    1 u_t = torch.cat([rx_signal[t:t+1], -decision_buffer]).unsqueeze(1)
#    * Intuition: We bundle everything the equalizer needs to look at into one vector: the current noisy signal ($x_t$) and the previous "decisions" (the      
#      feedback).
#    * Note: The feedback is negated here so that the math naturally performs subtraction (Equalizer = Gain × Signal - Feedback).

#   ---

#   3. Filtering & Error Calculation
#   Code: Lines 170–173


#    1 eq_out = torch.dot(weights, u_t.squeeze())
#    2 e_t = tx_symbols[t] - eq_out
#    * Intuition: We multiply our current weights by the input to get an estimate of the symbol. The error ($e_t$) is simply the difference between what we    
#      wanted to see (the true symbol) and what the filter actually produced.

#   ---

#   4. Calculating the Gain ($k_t$)
#   Code: Lines 176–181


#    1 P_u = torch.matmul(P, u_t)
#    2 u_P_u = torch.matmul(u_t.t(), P_u)
#    3 denom = lam + u_P_u.squeeze()
#    4 k_t = P_u / denom
#    * Intuition: This is the "Kalman Gain." Think of it as a sophisticated step-size. Unlike LMS (which uses a fixed $\mu$), RLS looks at the matrix $P$ to
#      decide exactly how much to trust this specific piece of new data. If the input is in a direction we are already sure about, the gain is small. If it's
#      something new, the gain is large.

#   ---


#   5. Updating the Weights
#   Code: Line 184
#    1 weights = weights + (k_t.squeeze() * e_t)
#    * Intuition: We nudge the weights in the direction that reduces the error. Because we use the Gain ($k_t$) instead of a simple constant, the weights jump 
#      to the optimal values much faster than they would in standard NLMS.

#   ---


#   6. Updating the Inverse Matrix ($P$)
#   Code: Lines 189–190
#    1 k_u_t = torch.matmul(k_t, u_t.t())
#    2 P = (P - torch.matmul(k_u_t, P)) / lam
#    * Intuition: This is the "learning" step for the algorithm's internal model. We update $P$ to reflect that we now have more information about the signal.
#    * The Forgetting Factor ($\lambda$): We divide by lam (usually 0.99). This tells the algorithm to "slowly forget" very old data, allowing it to adapt if
#      the cable properties change over time.
# ==========================================

# ==========================================
# 3b. RLS Algorithm with Multi-Tap FFE
# ==========================================
def run_rls_dfe(rx_signal, tx_symbols, num_taps, lam=0.99, delta=0.01, teacher_forcing=True):
    """
    RLS DFE with integrated Multi-Tap FFE (Feed-Forward Equalizer).

    The RLS system is expanded to order (FFE_TAPS + num_taps):
    - First FFE_TAPS elements: w_ffe (forward FFE taps)
    - Remaining num_taps elements: w_fb (feedback taps)

    Forward: y_t = sum(w_ffe[i] * rx_eq[t-i]) - w_fb^T * d_fb

    The augmented input vector is: u_t = [ffe_buffer, -decision_buffer]^T
    """
    seq_len = rx_signal.shape[0]
    system_order = FFE_TAPS + num_taps  # FFE_TAPS forward + num_taps feedback

    # Initialize weights with center tap = 1.0
    weights = torch.zeros(system_order, dtype=torch.float32)
    weights[FFE_MAIN_CURSOR] = 1.0  # Center tap initialized to 1.0

    # Initialize inverse correlation matrix P_0 = delta^(-1) * I
    P = torch.eye(system_order, dtype=torch.float32) / delta

    # Initialize FFE buffer and decision buffer
    ffe_buffer = torch.zeros(FFE_TAPS, dtype=torch.float32)
    decision_buffer = torch.zeros(num_taps, dtype=torch.float32)

    mse_log = []

    for t in range(seq_len):
        # Shift FFE buffer and insert newest sample at index 0
        ffe_buffer = torch.roll(ffe_buffer, shifts=1, dims=0)
        ffe_buffer[0] = rx_signal[t]

        # Causality shift: target symbol is at t - FFE_MAIN_CURSOR
        target_idx = t - FFE_MAIN_CURSOR

        # 1. Construct augmented input: u_t = [ffe_buffer, -decision_buffer]
        # Negate decision buffer so RLS naturally subtracts feedback (aligns with DFE)
        u_t = torch.cat([ffe_buffer, -decision_buffer]).unsqueeze(1)  # [system_order, 1]

        # 2. Filter output and error
        all_feedback = torch.dot(weights, u_t.squeeze())
        eq_out = all_feedback

        if target_idx >= 0:
            e_t = tx_symbols[target_idx] - eq_out
            mse_log.append((e_t.item()) ** 2)

            # 3. Compute Kalman Gain vector: k_t = P * u_t / (lambda + u_t^T * P * u_t)
            P_u = torch.matmul(P, u_t)  # [system_order, 1]
            u_P_u = torch.matmul(u_t.t(), P_u)  # [1, 1] scalar

            # k_t = P * u_t / (lambda + u_t^T * P * u_t)
            denom = lam + u_P_u.squeeze()
            k_t = P_u / denom  # [system_order, 1]

            # 4. Weight Update: w_t = w_{t-1} + k_t * e_t (addition for RLS)
            weights = weights + (k_t.squeeze() * e_t)

            # 5. Update Inverse Correlation Matrix: P_t = (P_{t-1} - k_t * u_t^T * P_{t-1}) / lambda
            k_u_t = torch.matmul(k_t, u_t.t())  # [system_order, system_order]
            P = (P - torch.matmul(k_u_t, P)) / lam

            # 6. Make decision and update shift register
            decision = torch.sign(eq_out)
            decision_buffer = torch.roll(decision_buffer, shifts=1)
            if teacher_forcing:
                decision_buffer[0] = tx_symbols[target_idx]
            else:
                decision_buffer[0] = decision
        else:
            # Warmup phase: not enough history for valid error computation
            mse_log.append(0.0)

            # Update shift register with current symbol but no weight update
            decision_buffer = torch.roll(decision_buffer, shifts=1)
            if teacher_forcing:
                decision_buffer[0] = tx_symbols[t]
            else:
                decision_buffer[0] = 0.0

    return torch.tensor(mse_log), weights

# ==========================================
# 3c. Batch NLMS Wrapper
# ==========================================
def run_batch_nlms_dfe(rx_batch, tx_batch, num_taps, mu=0.1, eps=1e-6, teacher_forcing=False,
                       use_gear_shift=False, mu_fast=0.2, mu_slow=0.01, gear_threshold=0.5, ema_alpha=0.05):
    """
    Wrapper to run NLMS DFE over a batch of channels.

    Inputs:
        rx_batch: [batch_size, seq_len]
        tx_batch: [batch_size, seq_len]

    Output:
        avg_mse_history: Averaged MSE across batch dimension [seq_len]
        final_weights: Final DFE weights from last channel [num_taps]
    """
    batch_size = rx_batch.shape[0]
    mse_histories = []

    for i in range(batch_size):
        rx_i = rx_batch[i]  # [seq_len]
        tx_i = tx_batch[i]  # [seq_len]

        mse_history, weights, w_main = run_nlms_dfe(
            rx_i, tx_i, num_taps=num_taps, mu=mu, eps=eps, teacher_forcing=teacher_forcing,
            use_gear_shift=use_gear_shift, mu_fast=mu_fast, mu_slow=mu_slow,
            gear_threshold=gear_threshold, ema_alpha=ema_alpha
        )
        mse_histories.append(mse_history)

    # Stack and average across batch
    mse_stacked = torch.stack(mse_histories, dim=0)  # [batch_size, seq_len]
    avg_mse_history = torch.mean(mse_stacked, dim=0)   # [seq_len]

    return avg_mse_history, weights


# ==========================================
# 3d. Batch RLS Wrapper
# ==========================================
def run_batch_rls_dfe(rx_batch, tx_batch, num_taps, lam=0.99, delta=0.01, teacher_forcing=False):
    """
    Wrapper to run RLS DFE over a batch of channels.

    Inputs:
        rx_batch: [batch_size, seq_len]
        tx_batch: [batch_size, seq_len]

    Output:
        avg_mse_history: Averaged MSE across batch dimension [seq_len]
        final_weights: Final DFE weights from last channel [num_taps + 1]
    """
    batch_size = rx_batch.shape[0]
    mse_histories = []

    for i in range(batch_size):
        rx_i = rx_batch[i]  # [seq_len]
        tx_i = tx_batch[i]  # [seq_len]

        mse_history, weights = run_rls_dfe(
            rx_i, tx_i, num_taps=num_taps, lam=lam, delta=delta, teacher_forcing=teacher_forcing
        )
        mse_histories.append(mse_history)

    # Stack and average across batch
    mse_stacked = torch.stack(mse_histories, dim=0)  # [batch_size, seq_len]
    avg_mse_history = torch.mean(mse_stacked, dim=0)  # [seq_len]

    return avg_mse_history, weights


# ==========================================
# 4. Synchronization Helper
# ==========================================
def cross_correlate_sync(tx, rx, max_delay=100):
    """
    Finds the integer sample delay between tx and rx using cross-correlation.
    Synchronizes to the post-CTLE signal to account for group delay.
    """
    # Use the first 500 symbols for robust synchronization
    tx_sync = tx[:500]
    rx_sync = rx[:1000]

    corrs = []
    for d in range(max_delay):
        # Compute correlation at delay d
        c = torch.dot(tx_sync, rx_sync[d:d+500])
        corrs.append(c.item())

    return torch.argmax(torch.abs(torch.tensor(corrs))).item()


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

# ==========================================
# 5. Benchmark Execution
# ==========================================
if __name__ == "__main__":
    # --- CONFIGURATION (from config.py) ---
    # See config.py for all common settings
    # --------------------------------------------------

    # Use a batch size > 1 for statistically valid evaluation
    BENCH_BATCH_SIZE = 100
    EVAL_LENGTH = EVAL_SEQ_LENGTH  # Evaluation sequence length (from config.py)
    # Use a smaller BURN_IN for benchmark (200 is used for L2O eval with 500 symbols)
    BURN_IN = 500  # Steady-state starts after symbol 500

    print(f"Initializing Benchmark...")
    print(f"Channel Taps: {CH_TAPS}")
    print(f"DFE Taps: {DFE_TAPS}")
    print(f"CTLE Taps: {CTLE_TAPS} (Fixed Peaking: {FIXED_PEAKING})")
    print(f"SNR Range: {SNR_RANGE} dB")
    print(f"Batch Size: {BENCH_BATCH_SIZE}")
    print("-" * 30)

    # 1. Generate test data (batch of channels)
    gen = WirelineChannelGenerator(num_taps=CH_TAPS, snr_range=SNR_RANGE)
    tx = torch.sign(torch.randn(BENCH_BATCH_SIZE, EVAL_LENGTH))
    rx_raw, h_true = gen.generate_received_signal(tx, batch_size=BENCH_BATCH_SIZE)

    # 2. Apply Fixed CTLE
    ctle = DifferentiableCTLE(num_taps=CTLE_TAPS)
    peaking_gain = torch.ones(BENCH_BATCH_SIZE, 1) * FIXED_PEAKING
    rx_ctle = ctle(rx_raw, peaking_gain)

    # 3. Delay Alignment (Synchronization via Cross-Correlation)
    # Use common delay (median) for batch alignment
    batch_delays = cross_correlate_sync_batch(tx, rx_ctle)
    common_delay = int(torch.median(torch.tensor(batch_delays, dtype=torch.float)).item())
    print(f"Synchronized main cursor at delay (median): {common_delay}")

    # Align rx and tx using the common delay
    rx_aligned = rx_ctle[:, common_delay:]
    tx_aligned = tx[:, common_delay:]

    print(f"Aligned sequence length: {tx_aligned.shape[1]}")
    print("-" * 30)

    # 4. Parameter Sweep (NLMS DFE) - using batch wrapper
    mu_values = NLMS_MU_VALUES
    plt.figure(figsize=(10, 6))

    # Define burn-in period to ignore initial convergence outliers
    print(f"Acquisition vs Steady-State Performance (Batch-Averaged):")
    print(f"{'Method':<25} | {'MSE @ 500':<12} | {'Steady-State':<15}")
    print("-" * 55)

    for mu_val in mu_values:
        avg_mse_history, final_w = run_batch_nlms_dfe(
            rx_aligned, tx_aligned, num_taps=DFE_TAPS, mu=mu_val, teacher_forcing=False
        )

        # 1. Acquisition MSE (Average of symbols 0-500)
        acq_mse = torch.mean(avg_mse_history[:500]).item()
        acq_mse_db = 10 * torch.log10(torch.tensor(acq_mse)).item()

        # 2. Steady-State MSE (Average of symbols BURN_IN to end)
        ss_mse = torch.mean(avg_mse_history[BURN_IN:]).item()
        ss_mse_db = 10 * torch.log10(torch.tensor(ss_mse)).item()

        method_name = f"NLMS (mu={mu_val})"
        print(f"{method_name:<25} | {acq_mse_db:>7.2f} dB | {ss_mse_db:>10.2f} dB")

        smoothed_mse = pd.Series(avg_mse_history.numpy()).ewm(span=200).mean()
        plt.plot(10 * torch.log10(torch.tensor(smoothed_mse)), label=f'NLMS (mu={mu_val})')

    # 4b. Run Gear-Shifting NLMS (Variable Step-Size) - using batch wrapper
    avg_mse_history_gs, final_w_gs = run_batch_nlms_dfe(
        rx_aligned, tx_aligned, num_taps=DFE_TAPS,
        mu=0.1,  # Base mu (used when use_gear_shift=False)
        teacher_forcing=False,
        use_gear_shift=True,  # Enable gear-shifting
        mu_fast=GEAR_SHIFT_MU_FAST,
        mu_slow=GEAR_SHIFT_MU_SLOW,
        gear_threshold=GEAR_SHIFT_THRESHOLD,
        ema_alpha=GEAR_SHIFT_EMA_ALPHA
    )

    # Acquisition vs Steady-State for Gear-Shift
    acq_mse_gs = torch.mean(avg_mse_history_gs[:500]).item()
    acq_mse_gs_db = 10 * torch.log10(torch.tensor(acq_mse_gs)).item()
    ss_mse_gs = torch.mean(avg_mse_history_gs[BURN_IN:]).item()
    ss_mse_gs_db = 10 * torch.log10(torch.tensor(ss_mse_gs)).item()

    print(f"{'NLMS Gear-Shift':<25} | {acq_mse_gs_db:>7.2f} dB | {ss_mse_gs_db:>10.2f} dB")

    smoothed_mse_gs = pd.Series(avg_mse_history_gs.numpy()).ewm(span=200).mean()
    plt.plot(
        10 * torch.log10(torch.tensor(smoothed_mse_gs)),
        color='purple', linewidth=2,
        label=f'NLMS Gear-Shift (mu_fast={GEAR_SHIFT_MU_FAST}, mu_slow={GEAR_SHIFT_MU_SLOW})'
    )

    # 5. Run RLS DFE (optimal linear upper bound) - using batch wrapper
    avg_mse_history_rls, final_w_rls = run_batch_rls_dfe(
        rx_aligned, tx_aligned, num_taps=DFE_TAPS,
        lam=RLS_LAMBDA, delta=RLS_DELTA, teacher_forcing=False
    )

    # Acquisition vs Steady-State for RLS
    acq_mse_rls = torch.mean(avg_mse_history_rls[:500]).item()
    acq_mse_rls_db = 10 * torch.log10(torch.tensor(acq_mse_rls)).item()
    ss_mse_rls = torch.mean(avg_mse_history_rls[BURN_IN:]).item()
    ss_mse_rls_db = 10 * torch.log10(torch.tensor(ss_mse_rls)).item()

    print(f"{'RLS (lam='+str(RLS_LAMBDA)+')':<25} | {acq_mse_rls_db:>7.2f} dB | {ss_mse_rls_db:>10.2f} dB")
    print("-" * 75)

    smoothed_mse_rls = pd.Series(avg_mse_history_rls.numpy()).ewm(span=200).mean()
    plt.plot(
        10 * torch.log10(torch.tensor(smoothed_mse_rls)),
        color='black', linestyle='dashed', linewidth=2,
        label='RLS (lambda=0.99, delta=0.01)'
    )

    plt.axhline(y=-20, color='r', linestyle='--', label='Target MSE (-20 dB)')
    plt.title(f'NLMS vs Gear-Shift vs RLS Baseline (CH={CH_TAPS}, DFE={DFE_TAPS})')
    plt.xlabel('Symbols')
    plt.ylabel('MSE (dB)')
    plt.legend()
    plt.grid(True)
    plt.show()


# Note about the meaning of the inverse P matrix in RLS:
# The inverse matrix $P$ is strictly about the input signal’s statistics, not the equalizer weights themselves.

#   To understand $P$ intuitively, you can think of it as the "Inverse Energy Map" of your input signal. Here is the breakdown:


#   1. Is it about the weights or the signal?
#   It is about the input signal.
#   Specifically, $P$ is the inverse of the Autocorrelation Matrix of your input vector $u_t$.
#    * The weights ($w$): Represent what the filter is doing (the "solution").
#    * The matrix ($P$): Represents how much we know about the signal environment (the "context").


#   2. What information does it hold?
#   $P$ tracks how every tap in your equalizer correlates with every other tap.
#    * On the Diagonal: It tracks the inverse of the signal power at each tap. If a tap has seen very little signal energy, the value in $P$ stays high. This  
#      tells the algorithm: "We haven't seen much data here yet, so if an error occurs, make a large change to this weight."
#    * Off the Diagonal: It tracks the "redundancy" between taps. If Tap 1 and Tap 2 always move together (high correlation), $P$ recognizes this. It ensures  
#      that the weight update doesn't "over-correct" both taps for the same error.

#   3. Clarifying the "Memory" (The 50 vs. 100 vs. 10 confusion)
#   There are three different "lengths" to keep track of in your code:


#    * Filter Length (Size of $P$): In your code, DFE_TAPS = 10. Therefore, $P$ is an $11 \times 11$ matrix (10 feedback taps + 1 main tap). It only tracks the
#      correlations between the 11 pieces of data currently sitting in your equalizer's memory. It does not care that the channel has 50 taps; it only cares   
#      about the 11 it is trying to optimize.
#    * Channel Memory: CH_TAPS = 50. This is the physical reality of the wire—how long the echoes last.
#    * RLS Temporal Memory (Forgetting Factor): This is determined by lam = 0.99. The "memory" of RLS (how many previous symbols influence the current $P$) is 
#      roughly:
#       $$\text{Memory} \approx \frac{1}{1 - \lambda} = \frac{1}{1 - 0.99} = \mathbf{100 \text{ symbols}}$$
#       So, $P$ is an $11 \times 11$ "map" built using the statistical history of the last 100 symbols.


#   The "Simple" Intuition
#   Imagine you are trying to tune a guitar with 11 strings ($11$ taps).
#    * The Weights are the current tightness of the strings.
#    * The Matrix $P$ is your knowledge of how sensitive each string is.
#        * If you haven't plucked string #5 yet, your "uncertainty" ($P$) for that string is very high.
#        * The first time you pluck it and hear a wrong note (an error), RLS will use that high $P$ value to make a huge adjustment to that string's weight.   
#        * As you keep plucking it, $P$ gets smaller, and your adjustments become finer and more precise.


#   In short: $P$ is the algorithm's confidence map of the input signal space. It tells the equalizer which directions are "well-explored" and which directions
#   still need "bold" updates.