import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd

# ==========================================
# 1. Channel Generation (Identical to l2o_basic.py)
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
            h = torch.exp(-t / tau) * torch.sin(t + 1e-3) 
            
            # Add discrete reflections (via/connector proxy)
            num_reflections = torch.randint(1, 4, (1,)).item()
            for _ in range(num_reflections):
                idx = torch.randint(5, self.num_taps, (1,)).item()
                h[idx] += torch.empty(1).uniform_(-0.2, 0.2).item()
                
            h = h / torch.norm(h) # Normalize energy
            channels.append(h)
            
        return torch.stack(channels)

    def generate_received_signal(self, tx_symbols, batch_size):
        """
        Generates the full received signal by convolving symbols with channel
        and adding AWGN noise based on the SNR range.
        """
        channel_len = self.num_taps

        # 1. Pad tx_symbols for causal convolution (no look-ahead)
        tx_padded = F.pad(tx_symbols, (channel_len - 1, 0))

        # 2. Reshape for grouped 1D convolution
        tx_reshaped = tx_padded.view(1, batch_size, -1)
        h_batch = self.generate_batch(batch_size)
        h_reshaped = h_batch.view(batch_size, 1, channel_len)

        # 3. Apply the channel filters using grouped convolution
        rx_clean = F.conv1d(tx_reshaped, h_reshaped, groups=batch_size)
        rx_clean = rx_clean.view(batch_size, -1)

        # 4. Add AWGN with SNR in the specified range
        snr_db = torch.empty(batch_size).uniform_(self.snr_range[0], self.snr_range[1])
        snr_linear = 10 ** (snr_db / 10)

        # Compute noise std based on signal power and target SNR
        signal_power = torch.mean(rx_clean ** 2, dim=-1, keepdim=True)
        noise_power = signal_power / snr_linear.unsqueeze(-1)
        noise_std = torch.sqrt(noise_power)

        # Add noise to received signal
        rx_noisy = rx_clean + noise_std * torch.randn_like(rx_clean)

        return rx_noisy

# ==========================================
# 2. Differentiable Parametric CTLE (Identical to l2o_basic.py)
# ==========================================
class DifferentiableCTLE(nn.Module):
    def __init__(self, num_taps=15):
        """
        Approximates a CTLE using a parameterized FIR filter for differentiable 
        time-domain simulation. In a real system, this is an analog IIR filter.
        """
        super().__init__()
        self.num_taps = num_taps
        # Base low-pass channel characteristics (fixed)
        self.register_buffer('base_lp', self._generate_base_lp())
        # High-pass peaking characteristics (scaled by learned parameter)
        self.register_buffer('base_hp', self._generate_base_hp())

    def _generate_base_lp(self):
        t = torch.linspace(0, 1, self.num_taps)
        return torch.exp(-t * 5) # Simple low-pass proxy

    def _generate_base_hp(self):
        t = torch.linspace(0, 1, self.num_taps)
        # Simple high-pass proxy (derivative of low-pass)
        hp = -5 * torch.exp(-t * 5)
        hp[0] += 1.0 # Add impulse
        return hp

    def forward(self, rx_signal, peaking_gain):
        """
        rx_signal: [batch, seq_len]
        peaking_gain: [batch, 1]
        """
        # Construct dynamic FIR filter based on peaking gain
        filter_taps = self.base_lp.unsqueeze(0) + peaking_gain.unsqueeze(-1) * self.base_hp.unsqueeze(0)
        filter_taps = filter_taps.unsqueeze(1) # [batch, 1, 1, num_taps]

        # Apply filter using 1D convolution
        rx_padded = F.pad(rx_signal, (self.num_taps - 1, 0))
        rx_padded = rx_padded.unsqueeze(1) # [batch, 1, seq_len + taps - 1]
        
        # We must use grouped convolution to apply a different filter per batch element
        batch_size = rx_signal.shape[0]
        rx_reshaped = rx_padded.view(1, batch_size, -1)
        
        filtered_rx = F.conv1d(rx_reshaped, filter_taps, groups=batch_size)
        return filtered_rx.view(batch_size, -1)

# ==========================================
# 3. NLMS Algorithm
# ==========================================
def run_nlms_dfe(rx_signal, tx_symbols, num_taps, mu=0.1, eps=1e-6, teacher_forcing=True):
    seq_len = rx_signal.shape[0]
    weights = torch.zeros(num_taps, dtype=torch.float32)
    decision_buffer = torch.zeros(num_taps, dtype=torch.float32)
    mse_log = []
    
    for t in range(seq_len):
        feedback = torch.dot(decision_buffer, weights)
        eq_out = rx_signal[t] - feedback
        decision = torch.sign(eq_out)
        e_t = tx_symbols[t] - eq_out
        mse_log.append((e_t.item())**2)
        
        norm_sq = torch.dot(decision_buffer, decision_buffer) + eps
        update_step = (mu * e_t * decision_buffer) / norm_sq
        weights = weights + update_step
        
        decision_buffer = torch.roll(decision_buffer, shifts=1)
        if teacher_forcing:
            decision_buffer[0] = tx_symbols[t]
        else:
            decision_buffer[0] = decision
            
    return torch.tensor(mse_log), weights


# ==========================================
# 3b. RLS Algorithm
# ==========================================
def run_rls_dfe(rx_signal, tx_symbols, num_taps, lam=0.99, delta=0.01, teacher_forcing=True):
    """
    Implements RLS (Recursive Least Squares) DFE.

    Scientific Definition: RLS minimizes the exponentially weighted least squares
    cost function. Unlike NLMS, which only tracks a weight vector, RLS tracks an
    inverse correlation matrix P_t of size N x N.

    Args:
        rx_signal: Received signal after CTLE [seq_len]
        tx_symbols: Transmitted symbols [seq_len]
        num_taps: Number of DFE taps
        lam: Forgetting factor (0 < lam < 1), typically 0.99
        delta: Regularization constant for P_0 initialization
        teacher_forcing: If True, use ground truth symbols in decision buffer

    Returns:
        mse_log: MSE history tensor
        final_weights: Learned DFE weights
    """
    seq_len = rx_signal.shape[0]

    # Initialize weights w_0 = 0
    weights = torch.zeros(num_taps, dtype=torch.float32)

    # Initialize inverse correlation matrix P_0 = delta^(-1) * I
    P = torch.eye(num_taps, dtype=torch.float32) / delta

    # Initialize decision buffer (shift register)
    decision_buffer = torch.zeros(num_taps, dtype=torch.float32)

    mse_log = []

    for t in range(seq_len):
        # 1. Filter output and error
        feedback = torch.dot(decision_buffer, weights)
        eq_out = rx_signal[t] - feedback
        e_t = tx_symbols[t] - eq_out
        mse_log.append((e_t.item()) ** 2)

        # 2. Compute Kalman Gain vector: k_t = P * u_t / (lambda + u_t^T * P * u_t)
        u_t = decision_buffer.unsqueeze(1)  # [num_taps, 1]
        P_u = torch.matmul(P, u_t)  # [num_taps, 1]
        u_P_u = torch.matmul(u_t.t(), P_u)  # [1, 1] scalar

        # k_t = P * u_t / (lambda + u_t^T * P * u_t)
        denom = lam + u_P_u.squeeze()
        k_t = P_u / denom  # [num_taps, 1]

        # 3. Weight Update: w_t = w_{t-1} + k_t * e_t
        weights = weights + (k_t.squeeze() * e_t)

        # 4. Update Inverse Correlation Matrix: P_t = (P_{t-1} - k_t * u_t^T * P_{t-1}) / lambda
        k_u_t = torch.matmul(k_t, u_t.t())  # [num_taps, num_taps]
        P = (P - torch.matmul(k_u_t, P)) / lam

        # 5. Make decision and update shift register
        decision = torch.sign(eq_out)
        decision_buffer = torch.roll(decision_buffer, shifts=1)
        if teacher_forcing:
            decision_buffer[0] = tx_symbols[t]
        else:
            decision_buffer[0] = decision

    return torch.tensor(mse_log), weights

# ==========================================
# 4. Benchmark Execution
# ==========================================
if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)

    # --- CONFIGURATION (Modify these to change taps) ---
    CH_TAPS = 50      # Number of taps in the wireline channel
    DFE_TAPS = 10     # Number of taps in the Decision Feedback Equalizer
    CTLE_TAPS = 15    # Number of taps in the CTLE FIR approximation
    
    SNR_RANGE = (15, 25)  # dB
    SEQ_LENGTH = 1000
    FIXED_PEAKING = 0.5  # Initial/Fixed CTLE gain
    # --------------------------------------------------
    
    print(f"Initializing Benchmark...")
    print(f"Channel Taps: {CH_TAPS}")
    print(f"DFE Taps: {DFE_TAPS}")
    print(f"CTLE Taps: {CTLE_TAPS} (Fixed Peaking: {FIXED_PEAKING})")
    print(f"SNR Range: {SNR_RANGE} dB")
    print("-" * 30)

    # 1. Generate test data
    gen = WirelineChannelGenerator(num_taps=CH_TAPS, snr_range=SNR_RANGE)
    tx = torch.sign(torch.randn(1, SEQ_LENGTH))
    rx_raw = gen.generate_received_signal(tx, batch_size=1)

    # 2. Apply Fixed CTLE
    ctle = DifferentiableCTLE(num_taps=CTLE_TAPS)
    peaking_gain = torch.ones(1, 1) * FIXED_PEAKING
    rx_ctle = ctle(rx_raw, peaking_gain).squeeze()
    tx = tx.squeeze()

    # 3. Parameter Sweep (NLMS DFE)
    mu_values = [0.01, 0.05, 0.1, 0.2]
    plt.figure(figsize=(10, 6))

    for mu_val in mu_values:
        mse_history, final_w = run_nlms_dfe(rx_ctle, tx, num_taps=DFE_TAPS, mu=mu_val, teacher_forcing=True)
        smoothed_mse = pd.Series(mse_history.numpy()).ewm(span=100).mean()
        plt.plot(10 * torch.log10(torch.tensor(smoothed_mse)), label=f'NLMS (mu={mu_val})')

    # 4. Run RLS DFE (optimal linear upper bound)
    mse_history_rls, final_w_rls = run_rls_dfe(
        rx_ctle, tx, num_taps=DFE_TAPS,
        lam=0.99, delta=0.01, teacher_forcing=True
    )
    smoothed_mse_rls = pd.Series(mse_history_rls.numpy()).ewm(span=100).mean()
    plt.plot(
        10 * torch.log10(torch.tensor(smoothed_mse_rls)),
        color='black', linestyle='dashed', linewidth=2,
        label='RLS (lambda=0.99, delta=0.01)'
    )

    plt.axhline(y=-20, color='r', linestyle='--', label='Target MSE (-20 dB)')
    plt.title(f'NLMS vs RLS Baseline with Fixed CTLE (CH={CH_TAPS}, DFE={DFE_TAPS})')
    plt.xlabel('Symbols')
    plt.ylabel('MSE (dB)')
    plt.legend()
    plt.grid(True)
    plt.show()
