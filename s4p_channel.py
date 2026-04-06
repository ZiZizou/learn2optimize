"""
S4P Channel Generator for learned optimizer experiments.

Loads pre-processed Touchstone (S-parameter) channel data and simulates
realistic SerDes channel behavior with crosstalk from FEXT/NEXT aggressors.

The physics model: The receiver voltage is the linear superposition of the
main signal (victim) and all crosstalk interference. Crosstalk aggressors
are driven by independent, uncorrelated random bit sequences.
"""

import torch
import torch.nn.functional as F
import random


class S4pChannelGenerator:
    """
    Channel generator that consumes pre-processed S4P channel data.

    Each channel geometry in the dataset contains:
        - 'thru': torch.Tensor [num_taps] - Victim channel impulse response
        - 'fext': List[torch.Tensor] or None - Far-end crosstalk impulse responses
        - 'next': List[torch.Tensor] or None - Near-end crosstalk impulse responses

    The generator randomly samples channel geometries and superimposes
    independent crosstalk from each aggressor.
    """

    def __init__(self, touchstone_file_path, snr_range=(15, 25), disable_agc=False):
        """
        Initialize the S4P channel generator.

        Args:
            touchstone_file_path: Path to the .pt file containing the list
                                  of channel dictionaries.
            snr_range: Tuple of (min_snr_db, max_snr_db) for AWGN.
            disable_agc: If True, bypasses peak normalization to preserve
                        true insertion loss physics.
        """
        self.touchstone_file_path = touchstone_file_path
        self.snr_range = snr_range
        self.disable_agc = disable_agc

        # Load the pre-processed channel dataset
        self.dataset = torch.load(touchstone_file_path)
        if not isinstance(self.dataset, list):
            raise ValueError(
                f"Expected dataset to be a list of channel dictionaries, "
                f"got {type(self.dataset)}"
            )

        # Validate first entry to ensure expected structure
        if len(self.dataset) > 0:
            sample = self.dataset[0]
            required_keys = {'thru', 'fext', 'next'}
            if not required_keys.issubset(sample.keys()):
                raise ValueError(
                    f"Channel dictionary missing required keys. "
                    f"Expected {required_keys}, got {sample.keys()}"
                )

    def _convolve_1d(self, tx_symbols, h_filter):
        """
        Perform causal 1D convolution using F.conv1d with filter flip.

        This matches the exact approach used in WirelineChannelGenerator:
        - Pad input for causal convolution
        - Flip filter (F.conv1d computes cross-correlation)
        - Use grouped convolution for batch processing

        Args:
            tx_symbols: [batch_size, seq_len] transmit symbols
            h_filter: [batch_size, 1, num_taps] channel filter (per batch)

        Returns:
            [batch_size, seq_len + num_taps - 1] convolved output
        """
        batch_size, seq_len = tx_symbols.shape
        num_taps = h_filter.shape[-1]

        # Pad for causal convolution (no look-ahead)
        tx_padded = F.pad(tx_symbols, (num_taps - 1, 0))

        # Reshape for grouped 1D convolution
        tx_reshaped = tx_padded.view(1, batch_size, -1)

        # Flip filter for true convolution (F.conv1d does cross-correlation)
        h_flipped = torch.flip(h_filter, dims=[-1])

        # Apply convolution with grouped filters
        rx = F.conv1d(tx_reshaped, h_flipped, groups=batch_size)
        return rx.view(batch_size, -1)

    def _generate_aggressor_bits(self, seq_len):
        """
        Generate independent random binary symbols for crosstalk aggressors.

        Returns:
            torch.Tensor of shape [seq_len] with +1 or -1 values
        """
        return torch.sign(torch.randn(seq_len))

    def generate_received_signal(self, tx_symbols, batch_size):
        """
        Generate the full received signal by convolving symbols with
        channel impulse responses and adding crosstalk + AWGN.

        Args:
            tx_symbols: [batch_size, seq_len] victim transmit symbols
            batch_size: Number of channels to simulate

        Returns:
            tuple: (rx_noisy, h_thru_batch)
                - rx_noisy: [batch_size, seq_len + max_taps - 1] received signal
                - h_thru_batch: [batch_size, num_taps] thru impulse responses
        """
        batch_size_actual, seq_len = tx_symbols.shape

        # Randomly sample channel geometries (with replacement for batch_size > dataset size)
        sampled_indices = [
            random.randint(0, len(self.dataset) - 1)
            for _ in range(batch_size_actual)
        ]
        sampled_dicts = [self.dataset[i] for i in sampled_indices]

        # Extract and stack THRU impulse responses
        h_thru_list = []
        max_taps = 0
        for d in sampled_dicts:
            # Ensure thru is 1D
            h_thru_1d = d['thru'].flatten()
            h_thru_list.append(h_thru_1d)
            max_taps = max(max_taps, h_thru_1d.shape[-1])

        # Pad thru tensors to same length if needed
        h_thru_padded = []
        for h in h_thru_list:
            if h.shape[-1] < max_taps:
                h = F.pad(h, (0, max_taps - h.shape[-1]))
            h_thru_padded.append(h)

        h_thru_batch = torch.stack(h_thru_padded).squeeze(1)  # [batch_size, max_taps]

        # Normalize thru channel (peak normalization) unless disabled
        if not self.disable_agc:
            peak_vals, _ = torch.max(torch.abs(h_thru_batch), dim=1, keepdim=True)
            h_thru_batch = h_thru_batch / (peak_vals + 1e-8)

        # Reshape thru for convolution
        h_thru_reshaped = h_thru_batch.unsqueeze(1)  # [batch_size, 1, max_taps]

        # 1. Convolve victim symbols with thru channel
        rx_main = self._convolve_1d(tx_symbols, h_thru_reshaped)

        # 2. Accumulate crosstalk from FEXT and NEXT aggressors
        rx_total = rx_main.clone()

        for i in range(batch_size_actual):
            channel_dict = sampled_dicts[i]

            # Handle FEXT (Far-End Crosstalk)
            if channel_dict.get('fext') is not None:
                for h_fext in channel_dict['fext']:
                    # Generate INDEPENDENT random bits for this aggressor
                    tx_aggressor = self._generate_aggressor_bits(seq_len)

                    # Convolve with FEXT impulse response
                    # Ensure h_fext is at least 1D, then reshape to [1, 1, num_taps]
                    h_fext_1d = h_fext.flatten()
                    h_fext_reshaped = h_fext_1d.unsqueeze(0).unsqueeze(1)  # [1, 1, num_taps]

                    rx_fext = self._convolve_1d(
                        tx_aggressor.unsqueeze(0),
                        h_fext_reshaped
                    )
                    # rx_fext has shape [1, output_len], index with [0] not [i]
                    rx_total[i] += rx_fext[0]

            # Handle NEXT (Near-End Crosstalk)
            if channel_dict.get('next') is not None:
                for h_next in channel_dict['next']:
                    # Generate INDEPENDENT random bits for this aggressor
                    tx_aggressor = self._generate_aggressor_bits(seq_len)

                    # Convolve with NEXT impulse response
                    # Ensure h_next is at least 1D, then reshape to [1, 1, num_taps]
                    h_next_1d = h_next.flatten()
                    h_next_reshaped = h_next_1d.unsqueeze(0).unsqueeze(1)  # [1, 1, num_taps]

                    rx_next = self._convolve_1d(
                        tx_aggressor.unsqueeze(0),
                        h_next_reshaped
                    )
                    # rx_next has shape [1, output_len], index with [0] not [i]
                    rx_total[i] += rx_next[0]

        # 3. Add AWGN
        snr_db = torch.empty(batch_size_actual).uniform_(
            self.snr_range[0], self.snr_range[1]
        )
        snr_linear = 10 ** (snr_db / 10)
        signal_power = torch.mean(rx_total ** 2, dim=-1, keepdim=True)
        noise_power = signal_power / snr_linear.unsqueeze(-1)
        noise_std = torch.sqrt(noise_power)

        rx_noisy = rx_total + noise_std * torch.randn_like(rx_total)

        return rx_noisy, h_thru_batch
