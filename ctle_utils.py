"""
DEPRECATED: Continuous-Time CTLE Wrapper using SerDesPy/scipy

This module is DEPRECATED in favor of the frequency-domain approach
implemented in ctle_frequency_utils.py.

The frequency-domain approach evaluates the exact analog transfer function
H(s) directly at FFT frequency bins, bypassing the bilinear transform
entirely and enabling parallelized GPU processing.

Please use ctle_frequency_utils.apply_frequency_domain_ctle instead.

---

Legacy Documentation:
This module provided a physically-accurate continuous-time CTLE implementation
that replaced the discrete FIR-based DifferentiableCTLE when ablate_ctle=True.

The CTLE (Continuous-Time Linear Equalizer) is modeled as an analog pole-zero
filter: H(s) = (s + ω_z) / (s + ω_p)

When ablate_ctle=True, the CTLE is static (LTI), allowing us to pre-filter
the entire batch at once for computational efficiency while maintaining
mathematical equivalence to sample-by-sample filtering.
"""

import torch
import numpy as np

# Try to import serdespy for framework integration
try:
    import serdespy
    HAS_SERDESPY = True
except ImportError:
    HAS_SERDESPY = False

from scipy import signal as sp_signal


def _create_continuous_time_ctle(peaking_gain_db=3.0, fs=1.0, fc=0.25):
    """
    Creates a continuous-time CTLE filter using pole-zero placement.

    A CTLE is typically modeled as:
        H(s) = (s + ω_z) / (s + ω_p)

    Where:
        - ω_z (zero frequency) determines the high-frequency boost start
        - ω_p (pole frequency) determines the -3dB bandwidth
        - peaking_gain_db controls the amount of peaking/boost

    Parameters:
        peaking_gain_db: Amount of peaking in dB (typical range: 0-12 dB)
        fs: Sampling frequency (normalized)
        fc: Normalized corner frequency (relative to fs)

    Returns:
        b, a: Numerator and denominator coefficients for the analog filter
    """
    # Convert peaking gain from dB to linear scale
    gain_linear = 10 ** (peaking_gain_db / 20.0)

    # Pole frequency (where response is -3dB from DC)
    # Zero is placed at a slightly lower frequency to create the boost
    wp = 2 * np.pi * fc
    wz = wp / gain_linear  # Zero at lower frequency creates boost

    # CTLE transfer function: H(s) = (s + ω_z) / (s + ω_p)
    # In standard form: (s/a + 1) / (s/b + 1) where a=ω_z, b=ω_p
    b = [1.0 / wz, 1.0]  # Numerator: s + ω_z
    a = [1.0 / wp, 1.0]  # Denominator: s + ω_p

    return b, a


def apply_serdespy_ctle(rx_base, peaking_gain=0.5):
    """
    Applies a continuous-time SerDes CTLE to a batch of received signals.

    When ablate_ctle=True, the CTLE is static (LTI), so we can pre-filter
    the entire batch at once for efficiency. This is mathematically equivalent
    to sample-by-sample filtering due to the linearity property of LTI systems.

    Parameters:
        rx_base: [batch_size, seq_len] PyTorch tensor - the received signal before CTLE
        peaking_gain: float in [0, 1], controls the CTLE peaking amount

    Returns:
        rx_filtered: [batch_size, seq_len] PyTorch tensor after CTLE filtering
    """
    if not HAS_SERDESPY:
        raise ImportError(
            "serdespy is required when ablate_ctle=True. "
            "Install with: uv pip install serdespy"
        )

    device = rx_base.device
    rx_np = rx_base.detach().cpu().numpy()

    # Map peaking_gain [0, 1] to CTLE peaking in dB [0, 12]
    # Typical SerDes CTLE provides 0-12 dB of peaking
    peaking_gain_db = peaking_gain * 12.0

    # Create the continuous-time CTLE filter
    b, a = _create_continuous_time_ctle(peaking_gain_db=peaking_gain_db)

    # Discretize the analog filter using bilinear transform (Tustin)
    # This preserves the frequency response characteristics
    b_z, a_z = sp_signal.bilinear(b, a, fs=1.0)

    batch_size, seq_len = rx_np.shape
    rx_filtered_np = np.zeros_like(rx_np)

    # Apply filter to each batch element
    for i in range(batch_size):
        # Apply the discrete-time equivalent of the continuous-time CTLE
        rx_filtered_np[i] = sp_signal.lfilter(b_z, a_z, rx_np[i])

    return torch.tensor(rx_filtered_np, dtype=torch.float32, device=device)
