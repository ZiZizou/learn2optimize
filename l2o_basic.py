import torch
import torch.nn as nn
import torch.nn.functional as F

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
            h = torch.exp(-t / tau) * torch.sin(t + 1e-3) 
            
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
        # Sample SNR uniformly from the range for each batch
        snr_db = torch.empty(batch_size).uniform_(self.snr_range[0], self.snr_range[1])
        snr_linear = 10 ** (snr_db / 10)

        # Compute noise std based on signal power and target SNR
        signal_power = torch.mean(rx_clean ** 2, dim=-1, keepdim=True)  # [batch, 1]
        noise_power = signal_power / snr_linear.unsqueeze(-1)
        noise_std = torch.sqrt(noise_power)

        # Add noise to received signal
        rx_noisy = rx_clean + noise_std * torch.randn_like(rx_clean)

        return rx_noisy

# =============================================================================


# either the CTLE parameters nor the DFE parameters are encoded as standard trainable weights. 
# It is easy to assume the CTLE has trainable weights because it is defined as a PyTorch nn.Module 
# (class DifferentiableCTLE(nn.Module)), whereas the DFE is just handled mathematically in the loop. 
# However, if you look at how they are actually implemented and updated, both are treated as dynamic 
#state variables rather than static neural network weights.

# ==========================================
# 1. Differentiable Parametric CTLE
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
        peaking_gain: [batch, 1] - The parameter adapted by the learned optimizer
        """
        # Construct dynamic FIR filter based on peaking gain
        # filter_taps shape: [batch, 1, num_taps]
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
        
        # Project and bound the step sizes
        mu_dfe = torch.sigmoid(self.head_dfe(hidden_state)) * 0.5 
        
        if update_ctle:
            # CTLE step size is typically smaller to prevent instability
            mu_ctle = torch.tanh(self.head_ctle(hidden_state)) * 0.05 
        else:
            mu_ctle = torch.zeros_like(mu_dfe)
            
        return mu_dfe, mu_ctle, hidden_state

# ==========================================
# 3. TBPTT Meta-Training Loop with EMA Safeguard
# ==========================================
def train_learned_optimizer(channel_gen, dfe, ctle, learned_opt, epochs=100, batch_size=64, unroll_len=50, ablate_ctle=False):
    """
    Trains the learned optimizer using TBPTT.

    Args:
        ... (previous args)
        ablate_ctle: If True, the optimizer will NOT update the CTLE peaking gain.
                     This allows for comparison with DFE-only benchmarks.
    """
    if ablate_ctle:
        print("!!! RUNNING IN ABLATION MODE: CTLE CONTROL DISABLED !!!")

    meta_optimizer = torch.optim.Adam(learned_opt.parameters(), lr=1e-3)
    # ... (rest of setup)
    total_seq_len = 500
    ctle_update_rate = 10 
    mu_safe = 0.01 
    ema_beta = 0.95
    loss_history = []

    for epoch in range(epochs):
        tx_symbols = torch.sign(torch.randn(batch_size, total_seq_len))
        rx_base = channel_gen.generate_received_signal(tx_symbols, batch_size)
        
        hidden_state = torch.zeros(batch_size, 32)
        dfe_weights = torch.zeros(batch_size, dfe.num_taps)
        ctle_peaking = torch.ones(batch_size, 1) * 0.5 
        
        ema_error = torch.ones(batch_size, 1) * 1.0
        decision_buffer = torch.zeros(batch_size, dfe.num_taps)
        
        for t_start in range(0, total_seq_len, unroll_len):
            meta_optimizer.zero_grad()ls ~
            loss = 0
            
            hidden_state = hidden_state.detach()
            dfe_weights = dfe_weights.detach()
            ctle_peaking = ctle_peaking.detach()
            decision_buffer = decision_buffer.detach()
            ema_error = ema_error.detach()
            
            for t in range(t_start, min(t_start + unroll_len, total_seq_len)):
                rx_t = rx_base[:, t:t+1]
                rx_eq = ctle(rx_t, ctle_peaking)

                y_out = dfe(rx_eq, decision_buffer, dfe_weights)
                e_t = tx_symbols[:, t:t+1] - y_out

                x_norm = torch.norm(decision_buffer, dim=1, keepdim=True) + 1e-6
                state_features = torch.cat([
                    e_t, ema_error, x_norm,
                    dfe_weights[:, 0:1],
                    ctle_peaking
                ], dim=1)

                update_ctle_flag = (t % ctle_update_rate == 0)
                mu_dfe, mu_ctle, hidden_state = learned_opt(state_features, hidden_state, update_ctle_flag)

                # --- ABLATION LOGIC ---
                if ablate_ctle:
                    mu_ctle = torch.zeros_like(mu_ctle)
                # ----------------------

                ema_error_new = ema_beta * ema_error + (1 - ema_beta) * (e_t ** 2)
                stable_mask = (ema_error_new <= ema_error * 1.5).float()
                mu_dfe_applied = stable_mask * mu_dfe + (1 - stable_mask) * mu_safe
                ema_error = ema_error_new

                update_step = mu_dfe_applied * e_t * decision_buffer / x_norm
                dfe_weights = dfe_weights + update_step

                if update_ctle_flag:
                    ctle_peaking = ctle_peaking + mu_ctle * torch.sign(e_t) * torch.sign(rx_eq)
                    ctle_peaking = torch.clamp(ctle_peaking, 0.0, 1.0)

                decision_buffer = torch.roll(decision_buffer, shifts=1, dims=1)
                decision_buffer[:, 0] = tx_symbols[:, t]
                loss += torch.mean(e_t ** 2)

            avg_loss = loss / unroll_len
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(learned_opt.parameters(), 1.0)
            meta_optimizer.step()

        epoch_loss = loss.item() / (total_seq_len // unroll_len * unroll_len)
        loss_history.append(epoch_loss)
        print(f"Epoch {epoch + 1}/{epochs} | Avg MSE: {epoch_loss:.6f}")

    return learned_opt, loss_history


# ==========================================
# 4. Execution Block
# ==========================================
if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)

    # --- CONFIGURATION (Modify these to change taps) ---
    CH_TAPS = 50      # Number of taps in the wireline channel
    DFE_TAPS = 10     # Number of taps in the Decision Feedback Equalizer
    CTLE_TAPS = 15    # Number of taps in the CTLE FIR approximation
    
    SNR_RANGE = (15, 25)  # dB
    BATCH_SIZE = 64
    EPOCHS = 100
    UNROLL_LEN = 50   # TBPTT truncation length (20-50 as per config)
    ABLATE_CTLE = False # Set to True to disable CTLE control (Ablation Mode)
    # --------------------------------------------------

    # Instantiate modules
    print("Initializing modules...")
    channel_gen = WirelineChannelGenerator(num_taps=CH_TAPS, snr_range=SNR_RANGE)
    ctle = DifferentiableCTLE(num_taps=CTLE_TAPS)
    dfe = DifferentiableDFE(num_taps=DFE_TAPS)
    learned_opt = MultiRateLearnedNLMS(state_dim=5, hidden_dim=32)

    print(f"Channel taps: {CH_TAPS}")
    print(f"DFE taps: {DFE_TAPS}")
    print(f"CTLE taps: {CTLE_TAPS}")
    print(f"SNR range: {SNR_RANGE} dB")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Unroll length (TBPTT): {UNROLL_LEN}")
    print("-" * 50)

    # Train the learned optimizer
    print("Starting meta-training...")
    trained_model, loss_history = train_learned_optimizer(
        channel_gen=channel_gen,
        dfe=dfe,
        ctle=ctle,
        learned_opt=learned_opt,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        unroll_len=UNROLL_LEN,
        ablate_ctle=ABLATE_CTLE
    )

    print("-" * 50)
    print("Meta-training completed!")
    print(f"Final average MSE: {loss_history[-1]:.6f}")