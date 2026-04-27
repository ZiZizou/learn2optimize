"""
Wireline Channel Generator for learned optimizer experiments.

Generates synthetic wireline channels using a simplified RC/RLC low-pass model
with discrete reflections to mimic real-world channel characteristics.

Scientific Notes on Normalization:
- Channel IR normalization (pre-convolution scaling of impulse response):
  This is NOT the same as receiver AGC.
- Modes: "none" (preserve raw amplitude), "peak" (normalize peak to 1), "l2"
- When normalization is "none", raw channel impulse responses preserve true
  insertion loss physics with absolute amplitude variation.
- Receiver AGC is a separate causal gain control block applied after the channel.
"""

import torch
import torch.nn.functional as F


class WirelineChannelGenerator:
    def __init__(self, num_taps=50, snr_range=(15, 25), disable_agc=False, samples_per_symbol=1):
        """
        Generates channels with lengths of 20-50 taps
        and AWGN with SNR of 15-25 dB[cite: 55].

        Args:
            disable_agc: Deprecated alias for channel_ir_norm_mode="none".
                         If True, bypasses L2/peak normalization to preserve
                         true insertion loss physics (raw attenuated voltages).
                         Use channel_ir_norm_mode="none" for new code.
            samples_per_symbol: Number of samples per symbol for channel time grid.
        """
        self.num_taps = num_taps
        self.snr_range = snr_range
        self.disable_agc = disable_agc
        self.samples_per_symbol = samples_per_symbol

    def generate_batch(self, batch_size):
        num_samples = self.num_taps * self.samples_per_symbol
        t = torch.linspace(0, 5, num_samples) / float(self.samples_per_symbol)

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
                idx = torch.randint(5 * self.samples_per_symbol, num_samples, (1,)).item()
                h[idx] += torch.empty(1).uniform_(-0.2, 0.2).item()

            if not self.disable_agc:
                h = h / torch.norm(h)  # L2 normalize channel IR (not receiver AGC)
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
        channel_len_samples = self.num_taps * self.samples_per_symbol

        # 1. Pad tx_symbols for causal convolution (no look-ahead)
        tx_padded = F.pad(tx_symbols, (channel_len_samples - 1, 0))

        # 2. Reshape for grouped 1D convolution
        tx_reshaped = tx_padded.view(1, batch_size, -1)
        h_batch = self.generate_batch(batch_size)

        # Channel IR normalization (pre-convolution scaling)
        # This is NOT receiver AGC - it's just scaling the impulse response
        # before convolution to bound absolute amplitude.
        if not self.disable_agc:
            peak_vals, _ = torch.max(torch.abs(h_batch), dim=1, keepdim=True)
            h_batch = h_batch / (peak_vals + 1e-8)

        h_reshaped = h_batch.view(batch_size, 1, channel_len_samples)

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
