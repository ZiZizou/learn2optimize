"""
Frequency-Domain CTLE Evaluation Module

Evaluates the analog CTLE transfer function H(s) = (s + ω_z)/(s + ω_p)
on the discrete-time FFT frequency grid.

Unit convention:
- All frequencies are in cycles / symbol.
- samples_per_symbol controls the FFT bin spacing and Nyquist limit.
- fc is the pole frequency in cycles / symbol.
"""

import torch


def _compute_analog_response(peaking_gain_db, fc, samples_per_symbol, seq_len, device):
    """
    Parameters:
        peaking_gain_db: Peaking amount in dB.
        fc: Pole frequency in cycles / symbol.
        samples_per_symbol: Sample rate in samples / symbol.
        seq_len: Number of time-domain samples.
        device: PyTorch device.

    Returns:
        H_s: Complex frequency response of shape [seq_len//2 + 1]
    """
    gain_linear = 10 ** (peaking_gain_db / 20.0)

    # Angular frequencies in radians / symbol.
    wp = 2 * torch.pi * fc
    wz = wp / gain_linear

    # FFT bins in cycles / symbol.
    f_bins = torch.fft.rfftfreq(
        seq_len,
        d=1.0 / float(samples_per_symbol),
    ).to(device)

    s = 1j * 2 * torch.pi * f_bins
    H_s = (s + wz) / (s + wp)

    return H_s


def apply_frequency_domain_ctle(rx_base, peaking_gain=0.5, fs=None, samples_per_symbol=1, fc=0.25):
    """
    Parameters:
        rx_base: [batch, seq_len] received signal.
        peaking_gain: float in [0, 1], mapped to [0, 12] dB.
        fs: Deprecated alias for samples_per_symbol. Do not use.
        samples_per_symbol: Samples per symbol (e.g. 1, 2, 4).
        fc: CTLE pole frequency in cycles / symbol.

    Returns:
        rx_filtered: [batch, seq_len] float32.
    """
    seq_len = rx_base.shape[1]
    device = rx_base.device

    if fs is not None and fs != samples_per_symbol:
        raise ValueError(
            f"Conflicting rates: fs={fs}, samples_per_symbol={samples_per_symbol}"
        )

    peaking_gain_db = peaking_gain * 12.0

    H_s = _compute_analog_response(
        peaking_gain_db,
        fc,
        samples_per_symbol,
        seq_len,
        device,
    )

    rx_spectrum = torch.fft.rfft(rx_base, dim=1)
    filtered_spectrum = rx_spectrum * H_s
    rx_filtered = torch.fft.irfft(filtered_spectrum, n=seq_len, dim=1)

    return rx_filtered.to(torch.float32)
