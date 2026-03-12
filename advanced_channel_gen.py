"""
Advanced Wireline Channel Generator with Severe Long-Term Dependencies,
Complex Nonlinearities, and Dynamic Channel Drift.

This module provides a PyTorch-based channel model that captures realistic
wireline impairments including impedance mismatches, resonant reflections,
power amplifier saturation, and time-varying channel characteristics.
"""

import torch
import torch.nn.functional as F


class AdvancedWirelineChannelGenerator:
    """
    Advanced wireline channel generator implementing severe long-term dependencies,
    complex nonlinearities, and dynamic channel drift.

    Attributes:
        num_taps (int): Number of taps in the impulse response (default: 150).
        snr_db (float): Signal-to-noise ratio in dB.
        device (torch.device): Device to run computations on.
    """

    def __init__(self, num_taps=150, snr_db=20.0, device=None):
        """
        Initialize the advanced wireline channel generator.

        Args:
            num_taps (int): Number of taps in the impulse response. Default is 150.
            snr_db (float): Signal-to-noise ratio in dB. Default is 20.0.
            device (torch.device, optional): Device for computations. If None, uses CUDA if available.
        """
        self.num_taps = num_taps
        self.snr_db = snr_db
        self.device = device if device is not None else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

    def _sample_channel_parameters(self, batch_size):
        """
        Sample channel parameters for each batch element.

        Args:
            batch_size (int): Number of samples in the batch.

        Returns:
            dict: Dictionary containing tau, beta, gamma, and omega0 parameters.
        """
        # Sample decay time constant tau from [1.0, 3.0]
        tau = torch.empty(batch_size, device=self.device).uniform_(1.0, 3.0)

        # Sample reflection amplitude beta from [0.1, 0.3]
        beta = torch.empty(batch_size, device=self.device).uniform_(0.1, 0.3)

        # Sample decay rate gamma from [0.05, 0.1]
        gamma = torch.empty(batch_size, device=self.device).uniform_(0.05, 0.1)

        # Sample resonant frequency omega0 from [pi/8, pi/4]
        omega0 = torch.empty(batch_size, device=self.device).uniform_(
            torch.pi / 8, torch.pi / 4
        )

        return {
            'tau': tau,
            'beta': beta,
            'gamma': gamma,
            'omega0': omega0
        }

    def _compute_base_impulse_response(self, params):
        """
        Compute the base impulse response h[n] for each batch element.

        h[n] = exp(-n/tau) + beta * exp(-gamma * n) * cos(omega0 * n)

        Args:
            params (dict): Dictionary with tau, beta, gamma, omega0 parameters.

        Returns:
            torch.Tensor: Impulse response of shape [batch_size, num_taps].
        """
        batch_size = params['tau'].shape[0]
        n = torch.arange(self.num_taps, device=self.device, dtype=torch.float32)

        # Expand dimensions for broadcasting: n becomes [1, num_taps]
        n_expanded = n.unsqueeze(0)

        # Compute the base low-pass decay: exp(-n/tau)
        # tau is [batch_size, 1]
        tau = params['tau'].unsqueeze(1)
        base_decay = torch.exp(-n_expanded / tau)

        # Compute the resonant ringing component: beta * exp(-gamma * n) * cos(omega0 * n)
        beta = params['beta'].unsqueeze(1)
        gamma = params['gamma'].unsqueeze(1)
        omega0 = params['omega0'].unsqueeze(1)

        ringing = beta * torch.exp(-gamma * n_expanded) * torch.cos(omega0 * n_expanded)

        # Combine the two components
        h = base_decay + ringing

        # Normalize so that the peak amplitude is 1.0 (main cursor at index 0)
        # The peak should be at index 0, but we normalize by the maximum value
        max_vals = h.max(dim=1, keepdim=True)[0]
        h = h / max_vals

        return h

    def _apply_nonlinearity(self, linear_signal, kappa):
        """
        Apply power amplifier soft-clipping nonlinearity.

        y = tanh(kappa * x) / tanh(kappa)

        Args:
            linear_signal (torch.Tensor): Input signal before nonlinearity.
            kappa (torch.Tensor or float): Severity parameter. Higher kappa means more compression.

        Returns:
            torch.Tensor: Signal after applying soft-clipping nonlinearity.
        """
        # Ensure kappa is a tensor
        if not isinstance(kappa, torch.Tensor):
            kappa = torch.tensor(kappa, device=self.device)

        # Handle broadcasting: if kappa is scalar, expand to match signal dimensions
        if kappa.dim() == 0:
            kappa = kappa.unsqueeze(0).unsqueeze(-1)  # [1, 1]
        elif kappa.dim() == 1:
            kappa = kappa.unsqueeze(-1)  # [batch, 1]

        # Apply the scaled hyperbolic tangent
        # y = tanh(kappa * x) / tanh(kappa)
        numerator = torch.tanh(kappa * linear_signal)
        denominator = torch.tanh(kappa)

        # Avoid division by zero
        denominator = torch.where(
            denominator > 1e-8,
            denominator,
            torch.ones_like(denominator)
        )

        nonlinear_signal = numerator / denominator

        return nonlinear_signal

    def _sample_kappa(self, batch_size):
        """
        Sample the nonlinearity severity parameter kappa for each batch element.

        Args:
            batch_size (int): Number of samples in the batch.

        Returns:
            torch.Tensor: Kappa values sampled uniformly from [0.5, 2.0].
        """
        return torch.empty(batch_size, device=self.device).uniform_(0.5, 2.0)

    def _generate_time_varying_channel(self, h0, seq_len, drift_amplitude=0.1, drift_frequency=0.01):
        """
        Generate time-varying channel impulse responses using sinusoidal drift.

        ht[n] = h0[n] * (1 + A * sin(2 * pi * f_drift * t))

        Args:
            h0 (torch.Tensor): Base impulse response of shape [batch_size, num_taps].
            seq_len (int): Sequence length (number of time steps).
            drift_amplitude (float): Amplitude A of the sinusoidal drift.
            drift_frequency (float): Normalized frequency f_drift of the drift.

        Returns:
            torch.Tensor: Time-varying channel of shape [batch_size, seq_len, num_taps].
        """
        batch_size = h0.shape[0]

        # Create time indices: [0, 1, 2, ..., seq_len-1]
        t = torch.arange(seq_len, device=self.device, dtype=torch.float32)

        # Compute drift modulation: 1 + A * sin(2 * pi * f_drift * t)
        # Shape: [seq_len]
        drift_modulation = 1.0 + drift_amplitude * torch.sin(2 * torch.pi * drift_frequency * t)

        # Expand for broadcasting: [1, seq_len, 1]
        drift_modulation = drift_modulation.unsqueeze(0).unsqueeze(-1)

        # Expand h0 for each time step: [batch_size, 1, num_taps]
        h0_expanded = h0.unsqueeze(1)

        # Apply drift to create time-varying channel
        h_time_varying = h0_expanded * drift_modulation

        # Re-normalize at each time step to maintain peak amplitude of 1.0
        max_vals = h_time_varying.max(dim=2, keepdim=True)[0]
        h_time_varying = h_time_varying / max_vals

        return h_time_varying

    def _time_varying_convolution(self, tx_symbols, h_time_varying):
        """
        Perform time-varying convolution using unfold method.

        Args:
            tx_symbols (torch.Tensor): Transmitted symbols of shape [batch_size, seq_len]
                                       or [batch_size, 1, seq_len].
            h_time_varying (torch.Tensor): Time-varying channel of shape [batch_size, seq_len, num_taps].

        Returns:
            torch.Tensor: Convolution result of shape [batch_size, seq_len].
        """
        # Ensure tx_symbols has shape [batch_size, 1, seq_len] for conv operations
        if tx_symbols.dim() == 2:
            tx_symbols = tx_symbols.unsqueeze(1)
        elif tx_symbols.dim() != 3:
            raise ValueError(f"Expected tx_symbols to be 2D or 3D, got {tx_symbols.dim()}D")

        batch_size, _, seq_len = tx_symbols.shape
        num_taps = h_time_varying.shape[2]

        # Pad the input sequence with num_taps - 1 zeros for convolution
        padding = num_taps - 1
        tx_padded = F.pad(tx_symbols, (padding, 0), value=0)

        # Use unfold to create sliding windows
        # Output shape: [batch_size, 1, seq_len, num_taps]
        tx_unfolded = F.unfold(
            tx_padded.unsqueeze(2),  # [batch_size, 1, 1, seq_len + padding]
            kernel_size=(1, num_taps),
            stride=1
        )

        # Reshape to [batch_size, seq_len, num_taps]
        tx_unfolded = tx_unfolded.transpose(1, 2).reshape(batch_size, seq_len, num_taps)

        # Flip the channel taps for physical convolution (correlation vs convolution)
        h_flipped = h_time_varying.flip(-1)

        # Perform element-wise multiplication and sum along the taps dimension
        # rx_clean = sum(tx_unfolded * h_flipped, dim=-1)
        rx_clean = torch.sum(tx_unfolded * h_flipped, dim=-1)

        return rx_clean

    def _add_awgn(self, clean_signal, snr_db):
        """
        Add additive white Gaussian noise to the signal.

        Args:
            clean_signal (torch.Tensor): Clean received signal.
            snr_db (float): Signal-to-noise ratio in dB.

        Returns:
            torch.Tensor: Noisy received signal.
        """
        # Calculate signal power
        signal_power = torch.mean(clean_signal ** 2)

        # Calculate noise power from SNR
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear

        # Generate Gaussian noise
        noise = torch.randn_like(clean_signal) * torch.sqrt(noise_power)

        # Add noise to signal
        noisy_signal = clean_signal + noise

        return noisy_signal

    def generate_received_signal(self, tx_symbols, batch_size):
        """
        Generate the received signal after passing through the advanced wireline channel.

        Args:
            tx_symbols (torch.Tensor): Transmitted symbols of shape [batch_size, seq_len]
                                       or [seq_len].
            batch_size (int): Number of samples in the batch.

        Returns:
            tuple: (rx_noisy, h_time_varying)
                - rx_noisy: Noisy received signal of shape [batch_size, seq_len].
                - h_time_varying: Time-varying channel impulse responses of shape
                                  [batch_size, seq_len, num_taps].
        """
        # Handle input shape - if 1D, assume single batch
        if tx_symbols.dim() == 1:
            tx_symbols = tx_symbols.unsqueeze(0)
            batch_size = 1

        # Get sequence length
        if tx_symbols.dim() == 2:
            seq_len = tx_symbols.shape[1]
        else:
            seq_len = tx_symbols.shape[-1]

        # Ensure batch_size matches input
        batch_size = tx_symbols.shape[0]

        # Step 1: Sample channel parameters for each batch element
        params = self._sample_channel_parameters(batch_size)

        # Step 2: Compute base impulse response h0
        h0 = self._compute_base_impulse_response(params)

        # Step 3: Generate time-varying channel
        # Use fixed drift parameters, or could sample these as well
        h_time_varying = self._generate_time_varying_channel(h0, seq_len)

        # Step 4: Perform time-varying convolution
        rx_clean = self._time_varying_convolution(tx_symbols, h_time_varying)

        # Step 5: Apply nonlinearity (soft-clipping)
        # Sample kappa for each batch
        kappa = self._sample_kappa(batch_size)

        # Expand kappa to match signal shape for broadcasting
        # kappa: [batch_size] -> [batch_size, 1]
        kappa_expanded = kappa.unsqueeze(1)

        rx_nonlinear = self._apply_nonlinearity(rx_clean, kappa_expanded)

        # Step 6: Add AWGN
        rx_noisy = self._add_awgn(rx_nonlinear, self.snr_db)

        # Ensure output shapes are correct
        # rx_noisy: [batch_size, seq_len]
        # h_time_varying: [batch_size, seq_len, num_taps]

        if batch_size == 1:
            rx_noisy = rx_noisy.squeeze(0)

        return rx_noisy, h_time_varying
