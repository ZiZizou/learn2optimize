"""
Frequency-Domain CTLE Evaluation Module

This module provides a physically-accurate continuous-time CTLE implementation
by evaluating the exact analog transfer function H(s) along the imaginary axis
s = jω, bypassing the bilinear transform entirely.

The CTLE (Continuous-Time Linear Equalizer) is modeled as an analog pole-zero
filter: H(s) = (s + ω_z) / (s + ω_p)

The frequency response is computed directly using PyTorch's FFT frequency bins,
enabling parallelized batch processing on GPU accelerators.
"""

import torch
import numpy as np


def _compute_analog_response(peaking_gain_db, fc, fs, seq_len, device):
    """
    Evaluates the exact continuous-time CTLE transfer function H(s) at FFT frequency bins.

    Parameters:
        peaking_gain_db: Amount of peaking in dB (typical range: 0-12 dB)
        fc: Normalized corner frequency (relative to fs)
        fs: Sampling frequency (normalized)
        seq_len: Sequence length (number of time-domain samples)
        device: PyTorch device for tensor allocation

    Returns:
        H_s: Complex frequency response tensor of shape [seq_len//2 + 1]
    """
    # Convert peaking gain from dB to linear scale
    gain_linear = 10 ** (peaking_gain_db / 20.0)

    # Pole frequency (where response is -3dB from DC)
    # Zero is placed at a slightly lower frequency to create the boost
    # ω_p = 2π * fc * fs (angular frequency)
    wp = 2 * np.pi * fc * fs
    # ω_z = ω_p / gain_linear (zero at lower frequency creates boost)
    wz = wp / gain_linear

    # Generate digital frequency bins for real FFT (PyTorch convention)
    # rfftfreq gives frequencies in cycles/unit, multiply by 2π for angular frequency
    f_bins = torch.fft.rfftfreq(seq_len, d=1.0/fs).to(device)

    # Evaluate H(s) = (s + ω_z) / (s + ω_p) at s = jω = j * 2π * f_bins
    s = 1j * 2 * np.pi * f_bins

    # Complex transfer function evaluation
    H_s = (s + wz) / (s + wp)

    # Handle DC gain explicitly to ensure no baseline drift
    # At s=0: H(0) = ω_z / ω_p = 1 / gain_linear
    H_s[0] = wz / wp

    return H_s


def apply_frequency_domain_ctle(rx_base, peaking_gain=0.5, fs=1.0, fc=0.25):
    """
    Applies a continuous-time CTLE filter to a batch of received signals
    using parallelized frequency-domain processing.

    This method evaluates the exact analog transfer function H(s) directly
    at the FFT frequency bins, avoiding bilinear transform discretization
    and eliminating sequential CPU-bound filtering.

    Parameters:
        rx_base: [batch_size, seq_len] PyTorch tensor - received signal before CTLE
        peaking_gain: float in [0, 1], controls the CTLE peaking amount
        fs: Sampling frequency (normalized, default: 1.0)
        fc: Normalized corner frequency (relative to fs, default: 0.25)

    Returns:
        rx_filtered: [batch_size, seq_len] PyTorch float32 tensor after CTLE filtering
    """
    seq_len = rx_base.shape[1]
    device = rx_base.device

    # Map peaking_gain [0, 1] to CTLE peaking in dB [0, 12]
    # Typical SerDes CTLE provides 0-12 dB of peaking
    peaking_gain_db = peaking_gain * 12.0

    # Retrieve complex transfer function evaluated at FFT bins
    H_s = _compute_analog_response(peaking_gain_db, fc, fs, seq_len, device)

    # Transform entire batch to frequency domain (parallel across batch)
    rx_spectrum = torch.fft.rfft(rx_base, dim=1)

    # Apply analog response via element-wise complex multiplication
    # This mathematically guarantees zero discretization phase distortion
    filtered_spectrum = rx_spectrum * H_s

    # Convert modified spectrum back to time domain
    rx_filtered = torch.fft.irfft(filtered_spectrum, n=seq_len, dim=1)

    return rx_filtered.to(torch.float32)
