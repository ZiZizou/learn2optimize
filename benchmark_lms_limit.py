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
# 3. NLMS Algorithm with 1-Tap FFE
# ==========================================
def run_nlms_dfe(rx_signal, tx_symbols, num_taps, mu=0.1, eps=1e-6, teacher_forcing=True):
    """
    NLMS DFE with integrated 1-tap FFE (Variable Gain Amplifier).

    The 1-tap FFE (w_main) scales the main cursor amplitude to match the
    target +/-1 symbol alphabet before DFE subtraction.

    Forward: y_t = w_main * rx_signal[t] - w_fb^T * d_fb
    """
    seq_len = rx_signal.shape[0]
    weights = torch.zeros(num_taps, dtype=torch.float32)
    w_main = torch.tensor(FFE_INIT, dtype=torch.float32)  # 1-tap FFE/VGA
    decision_buffer = torch.zeros(num_taps, dtype=torch.float32)
    mse_log = []

    for t in range(seq_len):
        # Forward pass: w_main * x_t - w_fb^T * d_fb
        feedback = torch.dot(decision_buffer, weights)
        eq_out = w_main * rx_signal[t] - feedback
        decision = torch.sign(eq_out)
        e_t = tx_symbols[t] - eq_out
        mse_log.append((e_t.item())**2)

        # Normalization includes both the main tap input and feedback buffer
        norm_sq = (rx_signal[t] ** 2) + torch.dot(decision_buffer, decision_buffer) + eps

        # Update feedback weights
        update_step = (mu * e_t * decision_buffer) / norm_sq
        weights = weights - update_step

        # Update main tap (1-tap FFE)
        update_main = (mu * e_t * rx_signal[t]) / norm_sq
        w_main = w_main + update_main

        decision_buffer = torch.roll(decision_buffer, shifts=1)
        if teacher_forcing:
            decision_buffer[0] = tx_symbols[t]
        else:
            decision_buffer[0] = decision

    return torch.tensor(mse_log), weights, w_main


# ==========================================
# 3b. RLS Algorithm with 1-Tap FFE
# ==========================================
def run_rls_dfe(rx_signal, tx_symbols, num_taps, lam=0.99, delta=0.01, teacher_forcing=True):
    """
    RLS DFE with integrated 1-tap FFE (Variable Gain Amplifier).

    The RLS system is expanded to order (num_taps + 1):
    - First element (index 0): w_main (forward gain)
    - Remaining elements (1 to num_taps): w_fb (feedback taps)

    Forward: y_t = w_main * x_t - w_fb^T * d_fb

    The augmented input vector is: u_t = [x_t, d_fb^T]^T
    """
    seq_len = rx_signal.shape[0]
    system_order = num_taps + 1  # 1 main tap + num_taps feedback taps

    # Initialize weights w_0 = 0, but force first element (main tap) to 1.0
    weights = torch.zeros(system_order, dtype=torch.float32)
    weights[0] = 1.0  # Main tap initialized to 1.0

    # Initialize inverse correlation matrix P_0 = delta^(-1) * I
    P = torch.eye(system_order, dtype=torch.float32) / delta

    # Initialize decision buffer (shift register for feedback taps only)
    decision_buffer = torch.zeros(num_taps, dtype=torch.float32)

    mse_log = []

    for t in range(seq_len):
        # 1. Construct augmented input: u_t = [x_t, -decision_buffer]
        # Negate decision buffer so RLS naturally subtracts feedback (aligns with DFE)
        u_t = torch.cat([rx_signal[t:t+1], -decision_buffer]).unsqueeze(1)  # [num_taps+1, 1]

        # 2. Filter output and error
        # y_t = w^T * u_t = w_main * x_t + w_fb^T * (-d_fb) = w_main * x_t - w_fb^T * d_fb
        all_feedback = torch.dot(weights, u_t.squeeze())
        eq_out = all_feedback
        e_t = tx_symbols[t] - eq_out
        mse_log.append((e_t.item()) ** 2)

        # 3. Compute Kalman Gain vector: k_t = P * u_t / (lambda + u_t^T * P * u_t)
        P_u = torch.matmul(P, u_t)  # [num_taps+1, 1]
        u_P_u = torch.matmul(u_t.t(), P_u)  # [1, 1] scalar

        # k_t = P * u_t / (lambda + u_t^T * P * u_t)
        denom = lam + u_P_u.squeeze()
        k_t = P_u / denom  # [num_taps+1, 1]

        # 4. Weight Update: w_t = w_{t-1} + k_t * e_t (addition for RLS)
        weights = weights + (k_t.squeeze() * e_t)

        # Force main tap to stay near 1.0 (optional: let it vary, or re-insert constraint)
        # weights[0] = weights[0]  # Allow w_main to adapt freely

        # 5. Update Inverse Correlation Matrix: P_t = (P_{t-1} - k_t * u_t^T * P_{t-1}) / lambda
        k_u_t = torch.matmul(k_t, u_t.t())  # [num_taps+1, num_taps+1]
        P = (P - torch.matmul(k_u_t, P)) / lam

        # 6. Make decision and update shift register
        decision = torch.sign(eq_out)
        decision_buffer = torch.roll(decision_buffer, shifts=1)
        if teacher_forcing:
            decision_buffer[0] = tx_symbols[t]
        else:
            decision_buffer[0] = decision

    return torch.tensor(mse_log), weights

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

# ==========================================
# 5. Benchmark Execution
# ==========================================
if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)

    # --- CONFIGURATION (DIAGNOSTIC MODE) ---
    # Match DFE taps to channel length (50 taps = 1 main + 49 post-cursors)
    CH_TAPS = 50
    DFE_TAPS = 49  # Match channel length for full ISI cancellation
    CTLE_TAPS = 15

    SNR_RANGE = (15, 25)  # dB
    SEQ_LENGTH = 10000
    FIXED_PEAKING = 0.5  # Initial/Fixed CTLE gain
    # --------------------------------------------------

    print(f"Initializing Diagnostic Benchmark...")
    print(f"Channel Taps: {CH_TAPS}")
    print(f"DFE Taps: {DFE_TAPS} (matching channel for full ISI cancellation)")
    print(f"CTLE Taps: {CTLE_TAPS} (Fixed Peaking: {FIXED_PEAKING})")
    print(f"SNR Range: {SNR_RANGE} dB")
    print(f"Reflections: DISABLED (diagnostic mode)")
    print("-" * 30)

    # 1. Generate test data (Retain ground-truth channel h)
    gen = WirelineChannelGenerator(num_taps=CH_TAPS, snr_range=SNR_RANGE)
    tx = torch.sign(torch.randn(1, SEQ_LENGTH))
    rx_raw, h_true = gen.generate_received_signal(tx, batch_size=1)

    # 2. Apply Fixed CTLE
    ctle = DifferentiableCTLE(num_taps=CTLE_TAPS)
    peaking_gain = torch.ones(1, 1) * FIXED_PEAKING
    rx_ctle = ctle(rx_raw, peaking_gain).squeeze()
    tx = tx.squeeze()

    # 3. Delay Alignment (Synchronization via Cross-Correlation)
    # This accounts for both channel delay AND CTLE group delay
    cursor_delay = cross_correlate_sync(tx, rx_ctle)
    print(f"Synchronized main cursor at delay: {cursor_delay}")

    # Align rx and tx (use [cursor_delay:] for both to avoid -0 bug)
    rx_aligned = rx_ctle[cursor_delay:]
    tx_aligned = tx[cursor_delay:]

    print(f"Aligned sequence length: {len(tx_aligned)}")
    print("-" * 30)

    # 4. Parameter Sweep (NLMS DFE)
    mu_values = NLMS_MU_VALUES
    plt.figure(figsize=(10, 6))

    # Define burn-in period to ignore initial convergence outliers
    print(f"Steady-State Performance (Averaged from symbol {BURN_IN} to end):")
    print("-" * 60)

    for mu_val in mu_values:
        mse_history, final_w, w_main = run_nlms_dfe(rx_aligned, tx_aligned, num_taps=DFE_TAPS, mu=mu_val, teacher_forcing=True)
        
        # Calculate steady-state MSE
        ss_mse = torch.mean(mse_history[BURN_IN:]).item()
        ss_mse_db = 10 * torch.log10(torch.tensor(ss_mse)).item()
        print(f"NLMS (mu={mu_val:<4}): Avg MSE = {ss_mse:.6f} ({ss_mse_db:.2f} dB) | Final w_main={w_main.item():.3f}")

        smoothed_mse = pd.Series(mse_history.numpy()).ewm(span=200).mean()
        plt.plot(10 * torch.log10(torch.tensor(smoothed_mse)), label=f'NLMS (mu={mu_val}, w_main={w_main.item():.3f})')

    # 5. Run RLS DFE (optimal linear upper bound)
    mse_history_rls, final_w_rls = run_rls_dfe(
        rx_aligned, tx_aligned, num_taps=DFE_TAPS,
        lam=0.99, delta=0.01, teacher_forcing=True
    )
    
    # Calculate steady-state MSE for RLS
    ss_mse_rls = torch.mean(mse_history_rls[BURN_IN:]).item()
    ss_mse_rls_db = 10 * torch.log10(torch.tensor(ss_mse_rls)).item()
    print(f"RLS (lam=0.99): Avg MSE = {ss_mse_rls:.6f} ({ss_mse_rls_db:.2f} dB) | Final w_main={final_w_rls[0].item():.3f}")
    print("-" * 60)

    smoothed_mse_rls = pd.Series(mse_history_rls.numpy()).ewm(span=200).mean()
    plt.plot(
        10 * torch.log10(torch.tensor(smoothed_mse_rls)),
        color='black', linestyle='dashed', linewidth=2,
        label='RLS (lambda=0.99, delta=0.01)'
    )

    plt.axhline(y=-20, color='r', linestyle='--', label='Target MSE (-20 dB)')
    plt.title(f'NLMS vs RLS Baseline - Diagnostic (CH={CH_TAPS}, DFE={DFE_TAPS}, No Reflections)')
    plt.xlabel('Symbols')
    plt.ylabel('MSE (dB)')
    plt.legend()
    plt.grid(True)
    plt.show()
