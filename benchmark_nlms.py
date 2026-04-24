import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd

from config import (
    CH_TAPS, SNR_RANGE, DFE_TAPS, CTLE_TAPS, FFE_TAPS, FFE_MAIN_CURSOR, FFE_INIT,
    RLS_LAMBDA, RLS_DELTA, NLMS_MU_VALUES,
    VSS_MU_MAX, VSS_MU_MIN, VSS_ALPHA, VSS_GAMMA, VSS_MOMENTUM,
    FIXED_PEAKING, EVAL_SEQ_LENGTH,
    VSS_MU_MAX, VSS_MU_MIN, VSS_ALPHA, VSS_GAMMA, VSS_MOMENTUM,
    OVERSAMPLE_FACTOR, OVERSAMPLE_MODE, PHASE_SEARCH_MAX_DELAY, PHASE_SEARCH_SYNC_LEN,
)
from oversampling_utils import choose_best_symbol_phase, upsample_symbols
from wireline_channel import WirelineChannelGenerator
from s4p_channel import S4pChannelGenerator
from ctle_frequency_utils import apply_frequency_domain_ctle

# DifferentiableCTLE not needed for benchmark (uses frequency-domain CTLE only)

# ==========================================
# 3. NLMS Algorithm with Multi-Tap FFE
# ==========================================
def run_nlms_dfe(rx_signal, tx_symbols, num_taps, mu=0.1, eps=1e-6, teacher_forcing=True,
                 use_vss=False, vss_mu_max=0.1, vss_mu_min=0.005, vss_alpha=0.99, vss_gamma=1e-3,
                 vss_momentum=0.0, use_soft_decision=True, tau=0.1):
    """
    NLMS DFE with integrated Multi-Tap FFE (Feed-Forward Equalizer).

    The multi-tap FFE spans multiple symbol periods to cancel precursor ISI
    before the causal DFE handles post-cursor energy.

    Forward: y_t = sum(w_ffe[i] * rx_eq[t-i]) - w_fb^T * d_fb

    Parameters:
        mu: Static step size (used when use_vss=False)
        use_vss: If True, enable continuous variable step-size (VSS)
        vss_mu_max: Upper bound for VSS (fast acquisition, default: 0.1)
        vss_mu_min: Lower bound for VSS (fine tracking, default: 0.005)
        vss_alpha: Memory factor for VSS (default: 0.99, close to 1 for smooth decay)
        vss_gamma: Error scaling factor for VSS (default: 1e-3)
        vss_momentum: Momentum coefficient for heavy-ball optimization (default: 0.0, no momentum)
                      Typical values: 0.9 or 0.95 for momentum-enabled updates
        use_soft_decision: If True, use tanh(eq_out/tau) soft decisions; if False, use hard sign decisions
        tau: Soft decision temperature parameter (default: 0.1)
    """
    seq_len = rx_signal.shape[0]
    weights = torch.zeros(num_taps, dtype=torch.float32)

    # Initialize Multi-Tap FFE with center tap = FFE_INIT
    w_ffe = torch.zeros(FFE_TAPS, dtype=torch.float32)
    w_ffe[FFE_MAIN_CURSOR] = FFE_INIT
    ffe_buffer = torch.zeros(FFE_TAPS, dtype=torch.float32)

    decision_buffer = torch.zeros(num_taps, dtype=torch.float32)
    mse_log = []
    mu_log = []  # Track step sizes
    hard_decision_log = []
    target_log = []

    # Initialize momentum buffers for heavy-ball optimization
    dfe_momentum_buffer = torch.zeros(num_taps, dtype=torch.float32)
    ffe_momentum_buffer = torch.zeros(FFE_TAPS, dtype=torch.float32)

    # Step size selection: static or continuous VSS
    if use_vss:
        current_mu = vss_mu_max  # Start fast for rapid acquisition
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
            decision = torch.tanh(eq_out / tau) if use_soft_decision else (torch.sign(eq_out) if eq_out != 0 else 1.0)
            e_t = tx_symbols[target_idx] - eq_out
            mse_log.append((e_t.item())**2)
            mu_log.append(current_mu)

            # BER: use hard decision (sign) for detection, compare to tx
            hard_dec = torch.sign(eq_out) if eq_out != 0 else 1.0
            hard_decision_log.append(hard_dec.item())
            target_log.append(tx_symbols[target_idx].item())

            # Continuous Variable Step-Size (VSS) Update
            if use_vss:
                current_mu = (vss_alpha * current_mu) + (vss_gamma * (e_t.item() ** 2))
                current_mu = max(vss_mu_min, min(vss_mu_max, current_mu))

            # Normalization includes full FFE buffer energy and feedback buffer
            norm_sq = torch.dot(ffe_buffer, ffe_buffer) + torch.dot(decision_buffer, decision_buffer) + eps

            # Calculate instantaneous gradients scaled by current_mu
            inst_grad_dfe = (current_mu * e_t * decision_buffer) / norm_sq
            inst_grad_ffe = (current_mu * e_t * ffe_buffer) / norm_sq

            # Update momentum buffers (Heavy-ball momentum)
            dfe_momentum_buffer = (vss_momentum * dfe_momentum_buffer) + inst_grad_dfe
            ffe_momentum_buffer = (vss_momentum * ffe_momentum_buffer) + inst_grad_ffe

            # Apply updates to weights using momentum
            # DFE subtracts feedback, FFE adds it (standard DFE math)
            weights = weights - dfe_momentum_buffer
            w_ffe = w_ffe + ffe_momentum_buffer

            decision_buffer = torch.roll(decision_buffer, shifts=1)
            if teacher_forcing:
                decision_buffer[0] = tx_symbols[target_idx]
            else:
                decision_buffer[0] = decision
        else:
            # Warmup phase: not enough history for valid error computation
            # Log zero error but still build up ffe_buffer
            mse_log.append(0.0)
            mu_log.append(current_mu)

            # Update feedback weights with zero error (no learning)
            decision_buffer = torch.roll(decision_buffer, shifts=1)
            if teacher_forcing:
                decision_buffer[0] = tx_symbols[t]
            else:
                decision_buffer[0] = 0.0

    # Compute BER (all valid symbols)
    hard_decisions = torch.tensor(hard_decision_log)
    targets = torch.tensor(target_log)
    ber_all = torch.mean((hard_decisions != targets).float()).item()

    return torch.tensor(mse_log), weights, w_ffe, torch.tensor(mu_log), ber_all


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
def run_rls_dfe(rx_signal, tx_symbols, num_taps, lam=0.99, delta=0.01, teacher_forcing=True, use_soft_decision=True, tau=0.1):
    """
    RLS DFE with integrated Multi-Tap FFE (Feed-Forward Equalizer).

    The RLS system is expanded to order (FFE_TAPS + num_taps):
    - First FFE_TAPS elements: w_ffe (forward FFE taps)
    - Remaining num_taps elements: w_fb (feedback taps)

    Forward: y_t = sum(w_ffe[i] * rx_eq[t-i]) - w_fb^T * d_fb

    The augmented input vector is: u_t = [ffe_buffer, -decision_buffer]^T

    Parameters:
        use_soft_decision: If True, use tanh(eq_out/tau) soft decisions; if False, use hard sign decisions
        tau: Soft decision temperature parameter (default: 0.1)
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
    hard_decision_log = []
    target_log = []

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

            # BER: use hard decision (sign) for detection, compare to tx
            hard_dec = torch.sign(eq_out) if eq_out != 0 else 1.0
            hard_decision_log.append(hard_dec.item())
            target_log.append(tx_symbols[target_idx].item())

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
            decision = torch.tanh(eq_out / tau) if use_soft_decision else (torch.sign(eq_out) if eq_out != 0 else 1.0)
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

    # Compute BER (all valid symbols)
    hard_decisions = torch.tensor(hard_decision_log)
    targets = torch.tensor(target_log)
    ber_all = torch.mean((hard_decisions != targets).float()).item()

    return torch.tensor(mse_log), weights, ber_all

# ==========================================
# 3c. Batch NLMS Wrapper
# ==========================================
def run_batch_nlms_dfe(rx_batch, tx_batch, num_taps, mu=0.1, eps=1e-6, teacher_forcing=False,
                       use_vss=False, vss_mu_max=0.1, vss_mu_min=0.005, vss_alpha=0.99, vss_gamma=1e-3,
                       vss_momentum=0.0, use_soft_decision=True, tau=0.1):
    """
    Wrapper to run NLMS DFE over a batch of channels.

    Inputs:
        rx_batch: [batch_size, seq_len]
        tx_batch: [batch_size, seq_len]

    Output:
        avg_mse_history: Averaged MSE across batch dimension [seq_len]
        all_dfe_weights: List of final DFE weights per channel [batch_size, num_taps]
        all_ffe_weights: List of final FFE weights per channel [batch_size, FFE_TAPS]
        avg_mu_history: Averaged step size across batch dimension [seq_len] (for VSS)

    Parameters:
        use_soft_decision: If True, use tanh soft decisions; if False, use hard sign decisions
        tau: Soft decision temperature parameter (default: 0.1)
    """
    batch_size = rx_batch.shape[0]
    mse_histories = []
    mu_histories = []
    all_dfe_weights = []
    all_ffe_weights = []
    ber_list = []

    for i in range(batch_size):
        rx_i = rx_batch[i]  # [seq_len]
        tx_i = tx_batch[i]  # [seq_len]

        mse_history, weights, w_main, mu_history, ber = run_nlms_dfe(
            rx_i, tx_i, num_taps=num_taps, mu=mu, eps=eps, teacher_forcing=teacher_forcing,
            use_vss=use_vss, vss_mu_max=vss_mu_max, vss_mu_min=vss_mu_min,
            vss_alpha=vss_alpha, vss_gamma=vss_gamma, vss_momentum=vss_momentum,
            use_soft_decision=use_soft_decision, tau=tau
        )
        mse_histories.append(mse_history)
        mu_histories.append(mu_history)
        all_dfe_weights.append(weights.cpu().numpy())
        all_ffe_weights.append(w_main.cpu().numpy())
        ber_list.append(ber)

    # Stack and average across batch
    mse_stacked = torch.stack(mse_histories, dim=0)  # [batch_size, seq_len]
    avg_mse_history = torch.mean(mse_stacked, dim=0)   # [seq_len]

    mu_stacked = torch.stack(mu_histories, dim=0)  # [batch_size, seq_len]
    avg_mu_history = torch.mean(mu_stacked, dim=0)   # [seq_len]
    avg_ber = torch.mean(torch.tensor(ber_list)).item()

    return avg_mse_history, all_dfe_weights, all_ffe_weights, avg_mu_history, avg_ber


# ==========================================
# 3d. Batch RLS Wrapper
# ==========================================
def run_batch_rls_dfe(rx_batch, tx_batch, num_taps, lam=0.99, delta=0.01, teacher_forcing=False, use_soft_decision=True, tau=0.1):
    """
    Wrapper to run RLS DFE over a batch of channels.

    Inputs:
        rx_batch: [batch_size, seq_len]
        tx_batch: [batch_size, seq_len]

    Output:
        avg_mse_history: Averaged MSE across batch dimension [seq_len]
        final_weights: Final combined weights (FFE + DFE) from last channel [FFE_TAPS + num_taps]

    Parameters:
        use_soft_decision: If True, use tanh soft decisions; if False, use hard sign decisions
        tau: Soft decision temperature parameter (default: 0.1)
    """
    batch_size = rx_batch.shape[0]
    mse_histories = []
    all_combined_weights = []
    ber_list = []

    for i in range(batch_size):
        rx_i = rx_batch[i]  # [seq_len]
        tx_i = tx_batch[i]  # [seq_len]

        mse_history, weights, ber = run_rls_dfe(
            rx_i, tx_i, num_taps=num_taps, lam=lam, delta=delta, teacher_forcing=teacher_forcing,
            use_soft_decision=use_soft_decision, tau=tau
        )
        mse_histories.append(mse_history)
        all_combined_weights.append(weights.cpu().numpy())
        ber_list.append(ber)

    # Stack and average across batch
    mse_stacked = torch.stack(mse_histories, dim=0)  # [batch_size, seq_len]
    avg_mse_history = torch.mean(mse_stacked, dim=0)  # [seq_len]
    avg_ber = torch.mean(torch.tensor(ber_list)).item()

    return avg_mse_history, all_combined_weights, avg_ber


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
    import argparse
    parser = argparse.ArgumentParser(description="NLMS/RLS Baseline Benchmark")
    parser.add_argument("--decision_type", type=str, default="soft", choices=["soft", "hard", "both"],
                        help="Decision type: 'soft' (tanh), 'hard' (sign), or 'both' (plot side-by-side)")
    parser.add_argument("--synthetic_channel", type=str, default=None,
                        help="Path to .pt file with synthetic channel data (e.g. synthetic_channels.pt). "
                             "If not provided, generates random channels using WirelineChannelGenerator.")
    args = parser.parse_args()

    use_soft_list = []
    if args.decision_type in ["soft", "both"]:
        use_soft_list.append(True)
    if args.decision_type in ["hard", "both"]:
        use_soft_list.append(False)

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
    print(f"Decision Type: {args.decision_type}")
    print("-" * 30)

    # 1. Generate or load test data (batch of channels)
    tx = torch.sign(torch.randn(BENCH_BATCH_SIZE, EVAL_LENGTH))
    tx_frontend = upsample_symbols(tx, OVERSAMPLE_FACTOR, OVERSAMPLE_MODE)

    if args.synthetic_channel is not None:
        print(f"Loading synthetic channels from: {args.synthetic_channel}")
        gen = S4pChannelGenerator(
            touchstone_file_path=args.synthetic_channel,
            snr_range=SNR_RANGE,
            disable_agc=False,
            samples_per_symbol=OVERSAMPLE_FACTOR,
        )
        rx_raw, h_true = gen.generate_received_signal(tx_frontend, batch_size=BENCH_BATCH_SIZE)
    else:
        gen = WirelineChannelGenerator(num_taps=CH_TAPS, snr_range=SNR_RANGE, samples_per_symbol=OVERSAMPLE_FACTOR)
        rx_raw, h_true = gen.generate_received_signal(tx_frontend, batch_size=BENCH_BATCH_SIZE)

    # 2. Apply Fixed CTLE (frequency-domain path only for oversampling)
    rx_ctle = apply_frequency_domain_ctle(
        rx_raw,
        peaking_gain=FIXED_PEAKING,
        samples_per_symbol=OVERSAMPLE_FACTOR,
        fc=0.25,
    )

    # 3. Phase selection and decimation to symbol rate
    rx_aligned, best_phase, common_delay = choose_best_symbol_phase(
        tx,
        rx_ctle,
        OVERSAMPLE_FACTOR,
        max_delay=PHASE_SEARCH_MAX_DELAY,
        sync_len=PHASE_SEARCH_SYNC_LEN,
    )
    tx_aligned = tx

    print(f"Oversample factor: {OVERSAMPLE_FACTOR}, Best phase: {best_phase}")
    print(f"Synchronized main cursor - Median delay: {common_delay}")
    print(f"Aligned sequence length: {tx_aligned.shape[1]}")
    print("-" * 30)

    # 4. Parameter Sweep (NLMS DFE) - using batch wrapper
    mu_values = NLMS_MU_VALUES

    if args.decision_type == "both":
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('NLMS/RLS Baseline Comparison', fontsize=14, fontweight='bold')
    else:
        fig, axes = plt.subplots(1, 1, figsize=(10, 6))
        if not isinstance(axes, np.ndarray):
            axes = [axes]

    results_by_decision = {}

    for use_soft in use_soft_list:
        decision_label = "Soft (tanh)" if use_soft else "Hard (sign)"
        ax = axes[use_soft_list.index(use_soft)] if args.decision_type == "both" else axes[0]
        ax.set_title(f"{decision_label} Decisions" if args.decision_type == "both" else "NLMS vs VSS vs RLS Baseline")

        print(f"\n{'='*60}")
        print(f"Running with {decision_label} decisions")
        print(f"{'='*60}")

        # Define burn-in period to ignore initial convergence outliers
        print(f"Acquisition vs Steady-State Performance (Batch-Averaged):")
        print(f"{'Method':<35} | {'MSE @ 500':<12} | {'Steady-State':<15} | {'BER':<12}")
        print("-" * 75)

        for mu_val in mu_values:
            avg_mse_history, nlms_all_dfe, nlms_all_ffe, mu_nlms, ber_nlms = run_batch_nlms_dfe(
                rx_aligned, tx_aligned, num_taps=DFE_TAPS, mu=mu_val, teacher_forcing=False,
                use_soft_decision=use_soft
            )

            # 1. Acquisition MSE (Average of symbols 0-500)
            acq_mse = torch.mean(avg_mse_history[:500]).item()
            acq_mse_db = 10 * torch.log10(torch.tensor(acq_mse)).item()

            # 2. Steady-State MSE (Average of symbols BURN_IN to end)
            ss_mse = torch.mean(avg_mse_history[BURN_IN:]).item()
            ss_mse_db = 10 * torch.log10(torch.tensor(ss_mse)).item()

            method_name = f"NLMS (mu={mu_val}) [{decision_label}]"
            print(f"{method_name:<35} | {acq_mse_db:>7.2f} dB | {ss_mse_db:>10.2f} dB | {ber_nlms:.4e}")

            smoothed_mse = pd.Series(avg_mse_history.numpy()).ewm(span=200).mean()
            ax.plot(10 * torch.log10(torch.tensor(smoothed_mse)),
                    label=f'NLMS (mu={mu_val})')

        # Run VSS NLMS
        avg_mse_vss, vss_all_dfe, vss_all_ffe, mu_vss, ber_vss = run_batch_nlms_dfe(
            rx_aligned, tx_aligned, num_taps=DFE_TAPS, mu=0.1, teacher_forcing=False,
            use_vss=True, vss_mu_max=VSS_MU_MAX, vss_mu_min=VSS_MU_MIN,
            vss_alpha=VSS_ALPHA, vss_gamma=VSS_GAMMA, use_soft_decision=use_soft
        )
        acq_mse_vss = torch.mean(avg_mse_vss[:500]).item()
        acq_mse_vss_db = 10 * torch.log10(torch.tensor(acq_mse_vss)).item()
        ss_mse_vss = torch.mean(avg_mse_vss[BURN_IN:]).item()
        ss_mse_vss_db = 10 * torch.log10(torch.tensor(ss_mse_vss)).item()
        print(f"{'NLMS VSS (μ_max='+str(VSS_MU_MAX)+')':<35} | {acq_mse_vss_db:>7.2f} dB | {ss_mse_vss_db:>10.2f} dB | {ber_vss:.4e}")
        smoothed_vss = pd.Series(avg_mse_vss.numpy()).ewm(span=200).mean()
        ax.plot(10 * torch.log10(torch.tensor(smoothed_vss)), '--',
                label=f'VSS NLMS (μ_max={VSS_MU_MAX})', alpha=0.7)

        # Run RLS DFE
        signal_variance = torch.var(rx_aligned).item()
        dynamic_delta = RLS_DELTA * signal_variance
        print(f"Signal Variance: {signal_variance:.6f}, Dynamic RLS Delta: {dynamic_delta:.6e}")
        avg_mse_history_rls, final_w_rls, ber_rls = run_batch_rls_dfe(
            rx_aligned, tx_aligned, num_taps=DFE_TAPS,
            lam=RLS_LAMBDA, delta=dynamic_delta, teacher_forcing=False,
            use_soft_decision=use_soft
        )

        acq_mse_rls = torch.mean(avg_mse_history_rls[:500]).item()
        acq_mse_rls_db = 10 * torch.log10(torch.tensor(acq_mse_rls)).item()
        ss_mse_rls = torch.mean(avg_mse_history_rls[BURN_IN:]).item()
        ss_mse_rls_db = 10 * torch.log10(torch.tensor(ss_mse_rls)).item()
        print(f"{'RLS (lam='+str(RLS_LAMBDA)+')':<35} | {acq_mse_rls_db:>7.2f} dB | {ss_mse_rls_db:>10.2f} dB | {ber_rls:.4e}")

        smoothed_mse_rls = pd.Series(avg_mse_history_rls.numpy()).ewm(span=200).mean()
        ax.plot(10 * torch.log10(torch.tensor(smoothed_mse_rls)),
                color='black', linestyle='dashed', linewidth=2,
                label=f'RLS (λ={RLS_LAMBDA})')

        print("-" * 60)

        ax.axhline(y=-20, color='r', linestyle='--', label='Target MSE (-20 dB)')
        ax.set_xlabel('Symbols')
        ax.set_ylabel('MSE (dB)')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
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