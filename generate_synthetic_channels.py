"""
Synthetic SerDes Channel Generator with Physics-Based Data Augmentation.

Reads industry-standard .s4p (Touchstone) frequency-domain files, converts them to
time-domain Impulse Responses (IRs), and applies physics-based data augmentations
to synthetically expand the dataset for Learned Optimizer (L2O) training.

Augmentations:
- Cascade: Simulates plugging two cables together (frequency-domain transfer multiplication)
- Insertion Loss: Simulates variations in dielectric loss/cable length (frequency-domain model)
- Random Reflections: Simulates via/connector impedance mismatches (delayed echo superposition)
- Mixup: Mathematical regularization, NOT physics (Linear interpolation between two channels)

Physics Note:
The THRU (main victim channel), FEXT (Far-End Crosstalk), and NEXT (Near-End Crosstalk)
are kept separate. Augmentations are applied ONLY to the THRU channel because:
- Aggressor transmitters send independent, asynchronous random bit sequences
- Superposition applies to voltages at the receiver, not to impulse responses
- FEXT/NEXT act as bounded colored noise signatures specific to the physical layout

Scientific Notes on S-Parameter Handling:
- A 4-port .s4p file stores a 4x4 scattering matrix S(f) vs frequency in single-ended basis
- For a differential pair represented in single-ended ports, the physically relevant transfer
  is often Sdd21 (differential-to-differential), not raw S21
- Sdd21 = 0.5 * (S21 - S23 - S41 + S43) for port pairing (1,3) input and (2,4) output
- Mixup is a regularization technique, not a physical model - it should not be marketed
  as physics-based augmentation
"""

import argparse
import os
import pathlib
import random
from typing import List, Optional, Dict, Union, Tuple

import numpy as np
import torch

try:
    import skrf
    HAS_SKRF = True
except ImportError:
    HAS_SKRF = False
    print("Warning: scikit-rf not installed. Use --dummy_data flag for synthetic generation.")

from s4p_inspector import (
    inspect_touchstone_file,
    extract_transfer_function,
    sdd21_from_se_s4p,
    PortMode,
    TransferMode,
    print_inspection_report,
)


# ==========================================
# Configuration
# ==========================================
from config import BAUD_RATE_HZ, CH_TAPS, OVERSAMPLE_FACTOR

def next_power_of_two(n: int) -> int:
    """Return the smallest power of two >= n."""
    return 1 << (n - 1).bit_length()


def interp_complex_to_grid_phase_aware(
    f_src: np.ndarray,
    h_src: np.ndarray,
    f_tgt: np.ndarray,
) -> np.ndarray:
    """
    Phase-aware interpolation of complex transfer function.

    Uses log-magnitude and unwrapped phase interpolation, which is superior
    to real/imag linear interpolation for dispersive channels where phase
    varies smoothly but real/imag components can change rapidly.

    Args:
        f_src: Source frequency grid (Hz)
        h_src: Complex transfer function on source grid
        f_tgt: Target frequency grid (Hz)

    Returns:
        Complex transfer function interpolated to target grid
    """
    eps = 1e-12
    mag = np.maximum(np.abs(h_src), eps)
    log_mag = np.log(mag)
    phase = np.unwrap(np.angle(h_src))

    log_mag_tgt = np.interp(f_tgt, f_src, log_mag, left=log_mag[0], right=log_mag[-1])
    phase_tgt = np.interp(f_tgt, f_src, phase, left=phase[0], right=phase[-1])

    return np.exp(log_mag_tgt + 1j * phase_tgt)


def resample_transfer_to_ir(
    freq_hz: np.ndarray,
    H: np.ndarray,
    fs_target: float,
    num_taps: int,
    span_ui: int,
    samples_per_symbol: int,
) -> Tuple[np.ndarray, Dict]:
    """
    Resample a transfer function to target grid and compute impulse response.

    Uses measured frequency spacing to determine FFT size and phase-aware
    interpolation. Bandwidth adequacy is checked and group delay is estimated.

    Args:
        freq_hz: Measured frequency grid (Hz)
        H: Complex transfer function on measured grid
        fs_target: Target sample rate (Hz)
        num_taps: Number of output taps (span_ui * samples_per_symbol)
        span_ui: Channel IR span in unit intervals
        samples_per_symbol: Oversampling factor

    Returns:
        Tuple of (ir_cropped, metadata) where:
            - ir_cropped: Cropped impulse response of length num_taps
            - metadata: Dict with alignment and quality info including
                        group_delay_samples, precursor_ratio, bandwidth_ratio
    """
    df_meas = np.median(np.diff(freq_hz))
    n_fft_min = int(np.ceil(fs_target / df_meas))
    safety_factor = 1.5
    n_fft = next_power_of_two(max(n_fft_min, int(safety_factor * num_taps), 1024))

    f_target = np.fft.rfftfreq(n_fft, d=1.0 / fs_target)

    target_nyquist = fs_target / 2
    measured_fmax = freq_hz[-1]
    bandwidth_ratio = measured_fmax / target_nyquist if target_nyquist > 0 else 0.0

    if bandwidth_ratio < 1.0:
        print(
            f"  [warn] target Nyquist {target_nyquist:.3e} Hz exceeds measured "
            f"bandwidth {measured_fmax:.3e} Hz (ratio={bandwidth_ratio:.2f}); "
            f"high-frequency bins will be zero-filled"
        )

    H_tgt = interp_complex_to_grid_phase_aware(freq_hz, H, f_target)
    ir_full = np.fft.irfft(H_tgt, n=n_fft)

    peak_idx = int(np.argmax(np.abs(ir_full)))

    group_delay_samples = estimate_group_delay(f_target, H_tgt)
    aligned_idx = peak_idx - int(round(group_delay_samples))
    aligned_idx = max(0, min(aligned_idx, len(ir_full) - num_taps))

    pre_cursor_energy = np.sum(ir_full[:peak_idx]**2)
    total_energy = np.sum(ir_full**2) + 1e-12
    precursor_ratio = pre_cursor_energy / total_energy

    start = aligned_idx
    stop = start + num_taps

    ir_cropped = ir_full[start:stop]
    if ir_cropped.shape[0] < num_taps:
        ir_cropped = np.pad(ir_cropped, (0, num_taps - ir_cropped.shape[0]))

    metadata = {
        "peak_index": peak_idx,
        "crop_start": start,
        "crop_stop": stop,
        "n_fft": n_fft,
        "bandwidth_ratio": bandwidth_ratio,
        "group_delay_samples": group_delay_samples,
        "precursor_ratio": precursor_ratio,
    }

    return ir_cropped, metadata


def estimate_group_delay(f: np.ndarray, H: np.ndarray) -> float:
    """
    Estimate bulk group delay from phase slope in the passband.

    Fits phase(f) ≈ -2π f τ + c over low-loss band (where |H| is near maximum).
    Returns delay in samples at the given frequency grid.

    Args:
        f: Frequency grid (Hz)
        H: Complex transfer function on that grid

    Returns:
        Group delay in samples (can be negative if peak is early)
    """
    mag = np.abs(H)
    max_mag = np.max(mag)

    low_loss_mask = mag > 0.5 * max_mag
    if np.sum(low_loss_mask) < 5:
        return 0.0

    f_low = f[low_loss_mask]
    phase = np.unwrap(np.angle(H[low_loss_mask]))

    f_mean = np.mean(f_low)
    numer = np.sum((f_low - f_mean) * (phase + 2 * np.pi * f_low))
    denom = np.sum((f_low - f_mean) ** 2)

    if denom < 1e-10:
        return 0.0

    tau_hz = -numer / denom / (2 * np.pi)

    if len(f) > 1:
        df = (f[-1] - f[0]) / (len(f) - 1)
        fs = 1.0 / df if df > 0 else 1.0
        tau_samples = tau_hz * fs
    else:
        tau_samples = 0.0

    return tau_samples


def extract_and_resample_ir(
    filepath: pathlib.Path,
    span_ui: int = CH_TAPS,
    samples_per_symbol: int = OVERSAMPLE_FACTOR,
    baud_rate_hz: float = BAUD_RATE_HZ,
    port_pairing: str = "auto",
    transfer_mode: str = "auto",
    normalization: str = "l2",
):
    """
    Extract impulse response from a Touchstone file at a target sample rate.

    Uses the s4p_inspector module to properly handle:
    - 4-port single-ended files representing differential pairs
    - Mixed-mode vs single-ended mode detection
    - Correct transfer function selection (S21 vs Sdd21)

    Args:
        filepath: Path to the .s4p file
        span_ui: Channel impulse response span in unit intervals (UI)
        samples_per_symbol: Number of samples per symbol (oversampling factor)
        baud_rate_hz: Baud rate in Hz (defines the frequency grid)
        port_pairing: Port pairing for 4-port files - "auto", "13-24", "12-34", or "14-23"
        transfer_mode: Transfer extraction mode - "auto", "s21", or "sdd21"
        normalization: Normalization mode - "none", "peak", or "l2"

    Returns:
        torch.Tensor of shape [span_ui * samples_per_symbol]
    """
    if not HAS_SKRF:
        raise RuntimeError("scikit-rf is required to parse .s4p files")

    # Inspect the file
    report = inspect_touchstone_file(filepath, baud_rate_hz)

    # Resolve pairing
    if port_pairing == "auto":
        resolved_pairing = report.port_pairing
    else:
        resolved_pairing = port_pairing

    # Resolve transfer mode
    if transfer_mode == "auto":
        resolved_mode = report.transfer_mode
    elif transfer_mode == "s21":
        resolved_mode = TransferMode.S21
    elif transfer_mode == "sdd21":
        resolved_mode = TransferMode.SDD21
    else:
        resolved_mode = TransferMode.S21

    net = skrf.Network(str(filepath))

    if net.f[0] > 0:
        try:
            net = net.extrapolate_to_dc()
        except Exception:
            pass

    # Extract the appropriate transfer function
    f_meas, H, transfer_name = extract_transfer_function(
        net,
        role="thru",
        transfer_mode=resolved_mode,
        port_pairing=resolved_pairing,
        inspection_report=report,
    )

    num_taps = int(span_ui * samples_per_symbol)
    fs_target = float(baud_rate_hz) * float(samples_per_symbol)

    # Resample to target grid and get IR
    ir, ir_meta = resample_transfer_to_ir(
        f_meas, H, fs_target, num_taps, span_ui, samples_per_symbol
    )

    ir = ir.astype(np.float32)

    # Apply normalization if requested
    if normalization == "none":
        pass  # preserve raw amplitude
    elif normalization == "peak":
        peak = np.max(np.abs(ir))
        if peak > 0:
            ir = ir / peak
    elif normalization == "l2":
        ir = ir / (np.linalg.norm(ir) + 1e-12)

    return torch.from_numpy(ir)


def resample_ir(ir: np.ndarray, num_taps: int) -> np.ndarray:
    """
    Resample/interpolate an impulse response to a fixed number of taps.

    Args:
        ir: Original impulse response array
        num_taps: Target number of taps

    Returns:
        Resampled impulse response of length num_taps
    """
    x_orig = np.linspace(0, 1, len(ir))
    x_new = np.linspace(0, 1, num_taps)
    return np.interp(x_new, x_orig, ir)


# ==========================================
# Step 1: Recursive Directory Traversal & File Classification
# ==========================================

def parse_s4p_directory_tree(
    base_dir: pathlib.Path,
    span_ui: int = CH_TAPS,
    samples_per_symbol: int = OVERSAMPLE_FACTOR,
    baud_rate_hz: float = BAUD_RATE_HZ,
    port_pairing: str = "auto",
    transfer_mode: str = "auto",
    normalization: str = "l2",
) -> List[Dict[str, Union[torch.Tensor, List[torch.Tensor], None]]]:
    """
    Recursively walk a directory tree and parse S4P files into channel groups.

    Each leaf directory containing .s4p files is treated as a channel group with:
    - Exactly 1 THRU (main victim channel)
    - 0 or more FEXT (Far-End Crosstalk) files
    - 0 or more NEXT (Near-End Crosstalk) files

    Args:
        base_dir: Root directory to walk
        span_ui: Channel IR span in unit intervals
        samples_per_symbol: Number of samples per symbol (oversampling factor)
        baud_rate_hz: Baud rate in Hz for frequency grid generation
        port_pairing: Port pairing for 4-port files - "auto", "13-24", "12-34", "14-23"
        transfer_mode: Transfer extraction mode - "auto", "s21", or "sdd21"
        normalization: Normalization mode - "none", "peak", or "l2"

    Returns:
        List of channel group dictionaries:
        [{
            'thru': torch.Tensor of shape [span_ui * sps],  # Main victim channel
            'fext': List[torch.Tensor] or None,             # List of FEXT IRs
            'next': List[torch.Tensor] or None,             # List of NEXT IRs
            'source_dir': str                              # Source directory name for logging
        }, ...]
    """
    if not HAS_SKRF:
        raise RuntimeError("scikit-rf is required to parse .s4p files. Install with: pip install scikit-rf")

    if not base_dir.exists():
        raise FileNotFoundError(f"Directory not found: {base_dir}")

    parsed_groups = []

    print(f"Walking directory tree: {base_dir}")

    for root, dirs, files in os.walk(base_dir):
        # Filter for .s4p files only
        s4p_files = [f for f in files if f.lower().endswith('.s4p')]

        if not s4p_files:
            continue

        # Initialize group
        group = {
            'thru': None,
            'fext': [],
            'next': [],
            'source_dir': os.path.basename(root)
        }

        # Process each file
        for filename in s4p_files:
            filepath = pathlib.Path(root) / filename

            try:
                ir = extract_and_resample_ir(
                    filepath,
                    span_ui=span_ui,
                    samples_per_symbol=samples_per_symbol,
                    baud_rate_hz=baud_rate_hz,
                    port_pairing=port_pairing,
                    transfer_mode=transfer_mode,
                    normalization=normalization,
                )
                ir_tensor = ir

                filename_lower = filename.lower()

                if 'thru' in filename_lower:
                    if group['thru'] is not None:
                        print(f"  Warning: Multiple thru files in {root}, using first")
                    group['thru'] = ir_tensor

                elif 'fext' in filename_lower:
                    group['fext'].append(ir_tensor)

                elif 'next' in filename_lower:
                    group['next'].append(ir_tensor)

                else:
                    # If no classification, try to use as thru anyway
                    if group['thru'] is None:
                        print(f"  Warning: Could not classify {filename}, treating as thru")
                        group['thru'] = ir_tensor
                    else:
                        print(f"  Warning: Could not classify {filename}, skipping")

            except Exception as e:
                print(f"  Warning: Failed to parse {filepath}: {e}")
                continue

        # Only add groups that have a valid thru channel
        if group['thru'] is not None:
            parsed_groups.append(group)
            fext_count = len(group['fext'])
            next_count = len(group['next'])
            print(f"  Found group: {group['source_dir']} - 1 thru, {fext_count} fext, {next_count} next")

    if not parsed_groups:
        raise RuntimeError("No valid channel groups found (no directories with thru files)")

    print(f"\nTotal: Found {len(parsed_groups)} channel groups")
    return parsed_groups


def inspect_s4p_directory(
    base_dir: pathlib.Path,
    baud_rate_hz: float = BAUD_RATE_HZ,
    port_pairing: str = "auto",
    transfer_mode: str = "auto",
    output_file=None,
) -> None:
    """
    Inspect all .s4p files in a directory tree and print human-readable reports.

    Args:
        base_dir: Root directory to walk
        baud_rate_hz: Baud rate in Hz for Nyquist frequency calculation
        port_pairing: Port pairing for 4-port files - "auto", "13-24", "12-34", or "14-23"
        transfer_mode: Transfer extraction mode - "auto", "s21", or "sdd21"
        output_file: File object to write reports to (default: None = stdout)
    """
    if not HAS_SKRF:
        raise RuntimeError("scikit-rf is required to parse .s4p files. Install with: pip install scikit-rf")

    if not base_dir.exists():
        raise FileNotFoundError(f"Directory not found: {base_dir}")

    def write(msg):
        print(msg)
        if output_file:
            output_file.write(msg + "\n")

    write(f"Inspecting .s4p files from {base_dir} (recursive)...")
    write("")

    inspected_count = 0
    for root, dirs, files in os.walk(base_dir):
        s4p_files = [f for f in files if f.lower().endswith('.s4p')]

        if not s4p_files:
            continue

        for filename in s4p_files:
            filepath = pathlib.Path(root) / filename
            try:
                report = inspect_touchstone_file(filepath, baud_rate_hz)
                print_inspection_report_to_file(report, output_file=output_file, write_fn=write)
                write("")
                inspected_count += 1
            except Exception as e:
                write(f"Warning: Failed to inspect {filepath}: {e}")
                continue

    write(f"Inspected {inspected_count} file(s)")


# ==========================================
# Dummy Data Generation
# ==========================================

def generate_dummy_base_channels(
    num_channels: int = 10,
    span_ui: int = CH_TAPS,
    samples_per_symbol: int = OVERSAMPLE_FACTOR,
) -> List[Dict[str, Union[torch.Tensor, List[torch.Tensor], None]]]:
    """
    Generate synthetic base channels when no .s4p files are available.
    Uses a simplified RC low-pass model with reflections (same as wireline_channel.py).

    Args:
        num_channels: Number of dummy channel groups to generate
        span_ui: Channel IR span in unit intervals
        samples_per_symbol: Number of samples per symbol

    Returns:
        List of channel group dictionaries (same structure as parse_s4p_directory_tree)
    """
    num_taps = span_ui * samples_per_symbol
    t = np.linspace(0, 5, num_taps) / float(samples_per_symbol)

    groups = []

    for i in range(num_channels):
        tau = np.random.uniform(0.5, 1.5)
        h = np.exp(-t / tau)

        num_reflections = np.random.randint(1, 4)
        for _ in range(num_reflections):
            idx = np.random.randint(5 * samples_per_symbol, num_taps)
            h[idx] += np.random.uniform(-0.2, 0.2)

        h = h / (np.linalg.norm(h) + 1e-8)
        thru_tensor = torch.tensor(h, dtype=torch.float32)

        # Generate random FEXT/NEXT (sometimes)
        fext_list = []
        next_list = []

        if random.random() > 0.3:  # 70% chance of having FEXT
            h_fext = np.exp(-t / np.random.uniform(0.5, 1.5))
            h_fext = h_fext / (np.linalg.norm(h_fext) + 1e-8)
            fext_list.append(torch.tensor(h_fext, dtype=torch.float32))

        if random.random() > 0.3:  # 70% chance of having NEXT
            h_next = np.exp(-t / np.random.uniform(0.5, 1.5))
            h_next = h_next / (np.linalg.norm(h_next) + 1e-8)
            next_list.append(torch.tensor(h_next, dtype=torch.float32))

        groups.append({
            'thru': thru_tensor,
            'fext': fext_list if fext_list else None,
            'next': next_list if next_list else None,
            'source_dir': f'dummy_channel_{i}'
        })

    return groups


# ==========================================
# Step 3: Augmentation Functions (apply to THRU only)
# ==========================================

def build_frequency_grid(
    num_taps: int,
    fs_hz: float,
) -> np.ndarray:
    """Build frequency grid for real FFT of length num_taps."""
    n_fft = num_taps * 2 - 1
    return np.fft.rfftfreq(n_fft, d=1.0 / fs_hz)


def augment_il_freq(
    H_batch: np.ndarray,
    f_grid: np.ndarray,
    alpha_c_range: tuple = (0.0, 0.5),
    alpha_d_range: tuple = (0.0, 0.1),
    delta_l_range: tuple = (0.0, 1.0),
    delay_per_length: float = 1e-9,
) -> np.ndarray:
    """
    Insertion Loss Augmentation: Simulates extra propagation loss in frequency domain.

    Uses an effective model:
        H_aug(f) = H(f) * exp(-(α_c·√f + α_d·f)·Δℓ) * exp(-j·2π·f·τ'·Δℓ)

    Where:
    - α_c: conductor/skin-effect loss coefficient (∝ √f)
    - α_d: dielectric loss coefficient (∝ f)
    - Δℓ: effective additional length (normalised)
    - τ': delay per unit length

    This is NOT a full RLGC telegrapher equation model - it is an effective
    extra-propagation-loss model that captures frequency-dependent loss behavior
    better than simple RC filtering.

    Args:
        H_batch: Batch of complex transfer functions [batch, n_freq]
        f_grid: Frequency grid in Hz [n_freq]
        alpha_c_range: Range for conductor loss coefficient α_c
        alpha_d_range: Range for dielectric loss coefficient α_d
        delta_l_range: Range for effective length multiplier Δℓ
        delay_per_length: Base delay per unit length in seconds (default 1ns)

    Returns:
        Augmented transfer functions [batch, n_freq]
    """
    batch_size = H_batch.shape[0]

    alpha_c = np.random.uniform(alpha_c_range[0], alpha_c_range[1], batch_size)
    alpha_d = np.random.uniform(alpha_d_range[0], alpha_d_range[1], batch_size)
    delta_l = np.random.uniform(delta_l_range[0], delta_l_range[1], batch_size)

    f_ghz = f_grid / 1e9

    loss_factor = np.exp(-(
        alpha_c[:, None] * np.sqrt(f_ghz) +
        alpha_d[:, None] * f_ghz
    ) * delta_l[:, None])

    phase_factor = np.exp(-1j * 2 * np.pi * f_grid * delay_per_length * delta_l[:, None])

    A_loss = loss_factor * phase_factor

    return H_batch * A_loss


def augment_reflections_freq(
    H_batch: np.ndarray,
    f_grid: np.ndarray,
    num_refs_range: tuple = (1, 4),
    delay_ui_min: float = 0.5,
    delay_ui_max: float = 10.0,
    amp_max: float = 0.15,
    ui_period_s: float = 1e-9,
) -> np.ndarray:
    """
    Reflection Augmentation: Simulates impedance mismatches via delayed echoes in frequency domain.

    Uses:
        R(f) = 1 + Σ a_k · exp(-j·2π·f·τ_k)
        H_aug(f) = H(f) · R(f)

    Where:
    - τ_k are physical delays in UI (not sample index)
    - a_k are bounded echo amplitudes (|a_k| ≤ amp_max)
    - The exponential phase factor respects oversampling naturally

    This is superior to integer-index spike injection because it works correctly
    in frequency domain and respects physical delay units.

    Args:
        H_batch: Batch of complex transfer functions [batch, n_freq]
        f_grid: Frequency grid in Hz [n_freq]
        num_refs_range: Range for number of reflections to add
        delay_ui_min: Minimum delay in UI
        delay_ui_max: Maximum delay in UI
        amp_max: Maximum reflection amplitude (as fraction of peak)
        ui_period_s: Unit interval period in seconds

    Returns:
        Augmented transfer functions [batch, n_freq]
    """
    batch_size = H_batch.shape[0]
    n_freq = H_batch.shape[1]

    num_refs = np.random.randint(num_refs_range[0], num_refs_range[1] + 1, batch_size)

    delays_ui = np.random.uniform(delay_ui_min, delay_ui_max, (batch_size, num_refs_range[1]))
    amps = np.random.uniform(-amp_max, amp_max, (batch_size, num_refs_range[1]))

    mask = np.arange(num_refs_range[1]) < num_refs[:, None]
    delays_ui = delays_ui * mask
    amps = amps * mask

    delays_s = delays_ui * ui_period_s

    phase_terms = np.exp(-1j * 2 * np.pi * f_grid[np.newaxis, :] * delays_s[:, :, np.newaxis])
    R = 1.0 + np.sum(amps[:, :, np.newaxis] * phase_terms, axis=1)

    return H_batch * R


def augment_cascade_freq(
    H_batch: np.ndarray,
    f_grid: np.ndarray,
    thru_pool: np.ndarray,
    pool_f_grid: np.ndarray,
    mode: str = "scalar",
) -> np.ndarray:
    """
    Cascade Augmentation: Simulates plugging two cables together via frequency-domain transfer.

    Scalar mode (approximation):
        H_total(f) = H_1(f) · H_2(f)
    Network mode (more accurate):
        Cascades full S-parameter networks before extracting the transfer function.
        This correctly handles interface mismatches but is more computationally expensive.

    Scalar mode note:
        Assumes matched interfaces between segments. Negligible mismatch interaction.
        Scalar multiplication ignores these effects - approximation, not full multi-reflection cascade.

    Args:
        H_batch: Batch of complex transfer functions [batch, n_freq]
        f_grid: Frequency grid for H_batch [n_freq]
        thru_pool: Pool of THRU transfer functions for sampling [N, n_freq_pool]
        pool_f_grid: Frequency grid for pool [n_freq_pool]
        mode: 'scalar' for transfer multiplication, 'network' for full S-parameter cascade

    Returns:
        Augmented transfer functions [batch, n_freq]
    """
    batch_size = H_batch.shape[0]

    indices = np.random.randint(0, len(thru_pool), batch_size)
    H_cascade = thru_pool[indices]

    if mode == "network":
        H_batch = cascade_network_mode(H_batch, H_cascade, f_grid)
    else:
        H_cascade_interp = interpolate_transfer(H_cascade, pool_f_grid, f_grid)
        H_batch = H_batch * H_cascade_interp

    return H_batch


def cascade_network_mode(
    H_batch: np.ndarray,
    H_cascade: np.ndarray,
    f_grid: np.ndarray,
) -> np.ndarray:
    """
    Network-mode cascade: Cascades S-parameter networks before extracting transfer function.

    Builds minimal 2-port S-parameter networks from the transfer functions and cascades
    them using the skrf network cascade operator. This correctly handles reference
    impedance transitions and interface mismatches, unlike scalar multiplication which
    assumes perfectly matched interfaces.

    For a 2-port network representing a channel with S21 (forward transfer) and S12
    (reverse transfer), we approximate the channel as a reciprocal, symmetric network
    with S11≈0 (matched input) and S22≈0 (matched output), so:

        S = [[0, H], [H, 0]]  (approximately, for a well-matched thru)

    Then cascades using skrf's network multiplication: S_total = S1 @ S2.

    Args:
        H_batch: Batch of complex transfer functions [batch, n_freq]
        H_cascade: Batch of cascade-channel transfer functions [batch, n_freq]
        f_grid: Frequency grid [n_freq]

    Returns:
        Cascaded transfer functions [batch, n_freq]
    """
    if not HAS_SKRF:
        import warnings
        warnings.warn(
            "scikit-rf not available; falling back to scalar cascade approximation. "
            "Install scikit-rf for proper network-mode cascade."
        )
        H_interp = interpolate_transfer(H_cascade, f_grid, f_grid)
        return H_batch * H_interp

    batch_size = H_batch.shape[0]
    n_freq = H_batch.shape[1]

    H_interp = interpolate_transfer(H_cascade, f_grid, f_grid)

    S_total_batch = np.zeros((batch_size, n_freq, 2, 2), dtype=np.complex128)

    for i in range(batch_size):
        S1 = np.zeros((n_freq, 2, 2), dtype=np.complex128)
        S2 = np.zeros((n_freq, 2, 2), dtype=np.complex128)

        S1[:, 0, 1] = H_batch[i]
        S1[:, 1, 0] = H_batch[i]
        S1[:, 0, 0] = 0.0
        S1[:, 1, 1] = 0.0

        S2[:, 0, 1] = H_interp[i]
        S2[:, 1, 0] = H_interp[i]
        S2[:, 0, 0] = 0.0
        S2[:, 1, 1] = 0.0

        try:
            net1 = skrf.Network(frequency=f_grid, s=S1)
            net2 = skrf.Network(frequency=f_grid, s=S2)
            net_cascade = net1 ** net2
            S_total_batch[i] = net_cascade.s
        except Exception:
            S_total_batch[i] = S1

    H_result = S_total_batch[:, :, 1, 0]
    return H_result


def interpolate_transfer(
    H_src: np.ndarray,
    f_src: np.ndarray,
    f_tgt: np.ndarray,
) -> np.ndarray:
    """
    Interpolate transfer functions to target frequency grid using phase-aware method.

    Args:
        H_src: Source transfer functions [batch, n_src]
        f_src: Source frequency grid [n_src]
        f_tgt: Target frequency grid [n_tgt]

    Returns:
        Interpolated transfer functions [batch, n_tgt]
    """
    eps = 1e-12
    batch_size = H_src.shape[0]
    n_tgt = len(f_tgt)

    mag = np.maximum(np.abs(H_src), eps)
    log_mag = np.log(mag)
    phase = np.unwrap(np.angle(H_src), axis=1)

    log_mag_tgt = np.zeros((batch_size, n_tgt))
    phase_tgt = np.zeros((batch_size, n_tgt))

    for i in range(batch_size):
        log_mag_tgt[i] = np.interp(f_tgt, f_src, log_mag[i])
        phase_tgt[i] = np.interp(f_tgt, f_src, phase[i])

    H_tgt = np.exp(log_mag_tgt + 1j * phase_tgt)

    return H_tgt


def transfer_to_ir_batch(
    H_batch: np.ndarray,
    f_grid: np.ndarray,
    fs_hz: float,
    num_taps: int,
) -> np.ndarray:
    """
    Convert batch of transfer functions to impulse responses.

    Args:
        H_batch: Batch of complex transfer functions [batch, n_freq]
        f_grid: Frequency grid [n_freq]
        fs_hz: Target sample rate in Hz
        num_taps: Number of output taps

    Returns:
        Batch of impulse responses [batch, num_taps]
    """
    n_fft = num_taps * 2 - 1

    ir_full = np.fft.irfft(H_batch, n=n_fft, axis=1)

    ir_cropped = ir_full[:, :num_taps]
    if ir_cropped.shape[1] < num_taps:
        pad_width = [(0, 0), (0, num_taps - ir_cropped.shape[1])]
        ir_cropped = np.pad(ir_cropped, pad_width)

    return ir_cropped


def ir_to_transfer_batch(
    h_batch: np.ndarray,
    fs_hz: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert batch of impulse responses to frequency domain.

    Args:
        h_batch: Batch of impulse responses [batch, num_taps]
        fs_hz: Sample rate in Hz

    Returns:
        Tuple of (H_batch, f_grid) where:
            - H_batch: Complex transfer functions [batch, n_freq]
            - f_grid: Frequency grid [n_freq]
    """
    n_fft = h_batch.shape[1] * 2 - 1

    H_batch = np.fft.rfft(h_batch, n=n_fft, axis=1)

    f_grid = np.fft.rfftfreq(n_fft, d=1.0 / fs_hz)

    return H_batch, f_grid


def normalize_l2(x: np.ndarray) -> np.ndarray:
    """L2 normalize along the last axis."""
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (norm + 1e-12)


def apply_augmentations_to_ir(
    h_batch: torch.Tensor,
    fs_hz: float,
    augmentations: Dict[str, bool],
    cascade_params: Dict,
    il_params: Dict,
    reflection_params: Dict,
    pool: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Apply frequency-domain augmentations to a batch of impulse responses.

    Augmentations are applied in the frequency domain where physically meaningful,
    then converted back to time domain.

    Args:
        h_batch: Batch of IRs [batch, num_taps]
        fs_hz: Sample rate in Hz
        augmentations: Dict of augmentation flags
        cascade_params: Params for cascade augmentation (includes 'mode': 'scalar'|'network')
        il_params: Params for insertion loss augmentation
        reflection_params: Params for reflection augmentation
        pool: Optional pool of THRU IRs for cascade/mixup [N, num_taps]

    Returns:
        Augmented batch [batch, num_taps]
    """
    h_np = h_batch.cpu().numpy()

    H_batch, f_grid = ir_to_transfer_batch(h_np, fs_hz)

    if augmentations.get('cascade', False) and pool is not None:
        pool_np = pool.cpu().numpy()
        pool_H, pool_f = ir_to_transfer_batch(pool_np, fs_hz)
        cascade_mode = cascade_params.get('mode', 'scalar')
        H_batch = augment_cascade_freq(H_batch, f_grid, pool_H, pool_f, mode=cascade_mode)

    if augmentations.get('il_tilt', False):
        H_batch = augment_il_freq(
            H_batch, f_grid,
            alpha_c_range=il_params.get('alpha_c_range', (0.0, 0.5)),
            alpha_d_range=il_params.get('alpha_d_range', (0.0, 0.1)),
            delta_l_range=il_params.get('delta_l_range', (0.0, 1.0)),
        )

    if augmentations.get('reflections', False):
        H_batch = augment_reflections_freq(
            H_batch, f_grid,
            num_refs_range=reflection_params.get('num_refs_range', (1, 4)),
            delay_ui_min=reflection_params.get('delay_ui_min', 0.5),
            delay_ui_max=reflection_params.get('delay_ui_max', 10.0),
            amp_max=reflection_params.get('amp_max', 0.15),
        )

    h_aug_np = transfer_to_ir_batch(H_batch, f_grid, fs_hz, h_np.shape[1])

    return torch.from_numpy(h_aug_np).to(h_batch.device)


def augment_cascade(
    h_batch: torch.Tensor,
    thru_pool: torch.Tensor
) -> torch.Tensor:
    """
    Channel Cascade (Convolution): Simulates plugging two cables together.

    DEPRECATED: Use augment_cascade_freq via apply_augmentations_to_ir for
    frequency-domain cascade before cropping.

    This version works on already-cropped IRs and does NOT preserve physical scaling.

    Args:
        h_batch: Batch of THRU channels [batch_size, num_taps]
        thru_pool: Pool of base THRU channels to sample from [N, num_taps]

    Returns:
        Augmented batch of same shape [batch_size, num_taps]
    """
    batch_size, num_taps = h_batch.shape

    indices = torch.randint(0, len(thru_pool), (batch_size,))
    h_cascade = thru_pool[indices]

    conv_length = num_taps * 2 - 1

    h_batch_fft = torch.fft.rfft(h_batch, n=conv_length)
    h_cascade_fft = torch.fft.rfft(h_cascade, n=conv_length)

    product_fft = h_batch_fft * h_cascade_fft

    result = torch.fft.irfft(product_fft, n=conv_length)

    result = result[:, :num_taps]

    return result / (torch.norm(result, dim=1, keepdim=True) + 1e-8)


def augment_il_tilt(
    h_batch: torch.Tensor,
    alpha_range: tuple = (0.0, 0.4)
) -> torch.Tensor:
    """
    Insertion Loss Tilting (RC Filtering): Simulates extra high-frequency loss.

    DEPRECATED: Use augment_il_freq via apply_augmentations_to_ir for
    frequency-domain skin-effect and dielectric model.

    Uses a randomized 1st-order low-pass filter:
    h_new[n] = (1 - alpha)*h[n] + alpha*h[n-1]

    Args:
        h_batch: Batch of channels [batch_size, num_taps]
        alpha_range: Tuple of (min, max) alpha values for the filter

    Returns:
        Augmented batch of same shape [batch_size, num_taps]
    """
    batch_size = h_batch.shape[0]

    alpha = torch.empty(batch_size, 1).uniform_(alpha_range[0], alpha_range[1])

    h_shifted = torch.roll(h_batch, shifts=1, dims=1)
    h_shifted[:, 0] = 0.0

    h_aug = (1.0 - alpha) * h_batch + alpha * h_shifted

    return h_aug / (torch.norm(h_aug, dim=1, keepdim=True) + 1e-8)


def augment_reflections(
    h_batch: torch.Tensor,
    max_refs: int = 3,
    max_ratio: float = 0.15,
    min_idx: int = 10
) -> torch.Tensor:
    """
    Random Reflection (Impedance Mismatch): Injects discrete spikes into post-cursor tail.

    DEPRECATED: Use augment_reflections_freq via apply_augmentations_to_ir for
    frequency-domain delayed-echo model with physical delay units.

    Vectorized implementation using scatter_add_ for GPU-efficient batch processing.
    Reflection amplitude is scaled by peak amplitude to bound reflection coefficient.

    Args:
        h_batch: Batch of channels [batch_size, num_taps]
        max_refs: Maximum number of reflections to inject per channel
        max_ratio: Maximum reflection coefficient (e.g., 0.15 = max 15% of peak)
        min_idx: Minimum index for reflection placement (post-cursor region)

    Returns:
        Augmented batch of same shape [batch_size, num_taps]
    """
    h_aug = h_batch.clone()
    batch_size, num_taps = h_aug.shape
    device = h_aug.device

    peak_amps = torch.max(torch.abs(h_aug), dim=1, keepdim=True)[0]

    num_refs = torch.randint(1, max_refs + 1, (batch_size, 1), device=device)

    indices = torch.randint(min_idx, num_taps, (batch_size, max_refs), device=device)
    amp_ratios = torch.empty((batch_size, max_refs), device=device).uniform_(-max_ratio, max_ratio)

    mask = torch.arange(max_refs, device=device).expand(batch_size, max_refs) < num_refs
    amp_ratios = amp_ratios * mask

    scaled_amps = amp_ratios * peak_amps
    h_aug.scatter_add_(1, indices, scaled_amps)

    return h_aug / (torch.norm(h_aug, dim=1, keepdim=True) + 1e-8)


def augment_mixup(
    h_batch: torch.Tensor,
    thru_pool: torch.Tensor,
    lam_range: tuple = (0.2, 0.8)
) -> torch.Tensor:
    """
    Mixup: Vectorized phase-aligned linear interpolation between two channels.

    Vectorized implementation using torch.gather with modulo indexing for
    GPU-efficient batch processing. Physics-based correction: Standard mixup
    can create invalid multi-drop topologies when channels have different
    "flight times" (main cursor at different indices). This version phase-aligns
    channels before interpolation so the result represents a valid point-to-point
    link, not a multi-path channel.

    h_new = lambda * h_batch + (1 - lambda) * h_random (aligned)

    Args:
        h_batch: Batch of THRU channels [batch_size, num_taps]
        thru_pool: Pool of base THRU channels to sample from [N, num_taps]
        lam_range: Tuple of (min, max) lambda values for interpolation

    Returns:
        Augmented batch of same shape [batch_size, num_taps]
    """
    batch_size, num_taps = h_batch.shape
    device = h_batch.device

    # Sample random channels from thru_pool
    indices = torch.randint(0, len(thru_pool), (batch_size,), device=device)
    h_random = thru_pool[indices].clone()

    # 1. Batch argmax for main cursors: shape [batch_size]
    idx_batch = torch.argmax(torch.abs(h_batch), dim=1)
    idx_random = torch.argmax(torch.abs(h_random), dim=1)

    # Calculate shifts: shape [batch_size, 1]
    shifts = (idx_batch - idx_random).unsqueeze(1)

    # 2. Create base index grid: shape [batch_size, num_taps]
    base_idx = torch.arange(num_taps, device=device).unsqueeze(0).expand(batch_size, num_taps)

    # 3. Calculate shifted indices with modulo wrap-around
    unmod_idx = base_idx - shifts
    gather_idx = unmod_idx % num_taps

    # Apply shift using gather (vectorized roll)
    h_random_aligned = torch.gather(h_random, 1, gather_idx)

    # 4. Mask out the wrapped artifacts to maintain causality
    # Any position where unmod_idx is out of bounds had wrapped values
    artifact_mask = (unmod_idx < 0) | (unmod_idx >= num_taps)
    h_random_aligned = h_random_aligned.masked_fill(artifact_mask, 0.0)

    # 5. Linear interpolation on phase-aligned tensors
    lam = torch.empty((batch_size, 1), device=device).uniform_(lam_range[0], lam_range[1])
    h_aug = lam * h_batch + (1.0 - lam) * h_random_aligned

    return h_aug / (torch.norm(h_aug, dim=1, keepdim=True) + 1e-8)


# ==========================================
# Step 4: Composable Pipeline
# ==========================================

def sample_random_groups(
    groups: List[Dict],
    batch_size: int
) -> tuple:
    """
    Sample random channel groups and extract their THRU tensors.

    Returns:
        Tuple of (thru_batch, associated_groups)
        - thru_batch: [batch_size, num_taps] tensor
        - associated_groups: list of original group dicts
    """
    indices = torch.randint(0, len(groups), (batch_size,))
    thru_batch = torch.stack([groups[i]['thru'] for i in indices])
    associated_groups = [groups[i] for i in indices]
    return thru_batch, associated_groups


def generate_synthetic_channels(
    parsed_groups: List[Dict],
    num_generated: int,
    batch_size: int = 64,
    channel_cascade: bool = False,
    cascade_mode: str = "scalar",
    insertion_loss_tilting: bool = False,
    random_reflection: bool = False,
    mixup: bool = False,
    use_freq_domain_aug: bool = True,
    fs_hz: float = BAUD_RATE_HZ * OVERSAMPLE_FACTOR,
) -> List[Dict]:
    """
    Generate synthetic channels by applying augmentations to THRU channels only.

    Pipeline order (as specified):
    1. Mixup/Cascade (Creates the new base geometry)
    2. Insertion Loss Tilting (Applies global frequency-domain shift)
    3. Random Reflections (Injects local time-domain spikes)

    For each generated sample:
    - A base channel group is randomly selected
    - Augmentations are applied ONLY to the THRU channel
    - Associated FEXT/NEXT are carried forward (one randomly sampled if multiple exist)

    Args:
        parsed_groups: List of channel group dicts from parse_s4p_directory_tree
        num_generated: Total number of synthetic channels to generate
        batch_size: Batch size for generation loop
        channel_cascade: Whether to apply cascade augmentation
        insertion_loss_tilting: Whether to apply IL tilt augmentation
        random_reflection: Whether to apply reflection augmentation
        mixup: Whether to apply mixup augmentation
        use_freq_domain_aug: Use frequency-domain augmentations (preserves physical scaling)
        fs_hz: Sample rate for frequency-domain operations

    Returns:
        List of synthetic channel group dicts with metadata:
        [{
            'thru': torch.Tensor [num_taps],
            'fext': torch.Tensor or None,
            'next': torch.Tensor or None,
            'physics_valid': True,  # False if mixup was applied
            'augmentation_class': 'physical' or 'regularization',
        }, ...]
    """
    thru_pool = torch.stack([g['thru'] for g in parsed_groups])
    pool_fs = float(fs_hz)

    synthetic_dataset = []
    num_complete_batches = num_generated // batch_size
    remainder = num_generated % batch_size

    augmentations = {
        'cascade': channel_cascade,
        'il_tilt': insertion_loss_tilting,
        'reflections': random_reflection,
    }
    cascade_params = {'mode': cascade_mode}
    il_params = {
        'alpha_c_range': (0.0, 0.5),
        'alpha_d_range': (0.0, 0.1),
        'delta_l_range': (0.0, 1.0),
    }
    reflection_params = {
        'num_refs_range': (1, 4),
        'delay_ui_min': 0.5,
        'delay_ui_max': 10.0,
        'amp_max': 0.15,
    }

    for batch_idx in range(num_complete_batches):
        h_current, associated_groups = sample_random_groups(parsed_groups, batch_size)

        if use_freq_domain_aug and any(augmentations.values()):
            h_current = apply_augmentations_to_ir(
                h_current, pool_fs, augmentations, cascade_params,
                il_params, reflection_params, thru_pool
            )
        else:
            if mixup:
                h_current = augment_mixup(h_current, thru_pool)
            if channel_cascade:
                h_current = augment_cascade(h_current, thru_pool)
            if insertion_loss_tilting:
                h_current = augment_il_tilt(h_current)
            if random_reflection:
                h_current = augment_reflections(h_current)

        for i in range(batch_size):
            physics_valid = not mixup
            aug_class = "regularization" if mixup else "physical"

            group = {
                'thru': h_current[i],
                'fext': random.choice(associated_groups[i]['fext']) if associated_groups[i]['fext'] else None,
                'next': random.choice(associated_groups[i]['next']) if associated_groups[i]['next'] else None,
                'physics_valid': physics_valid,
                'augmentation_class': aug_class,
            }
            synthetic_dataset.append(group)

    if remainder > 0:
        h_current, associated_groups = sample_random_groups(parsed_groups, remainder)

        if use_freq_domain_aug and any(augmentations.values()):
            h_current = apply_augmentations_to_ir(
                h_current, pool_fs, augmentations, cascade_params,
                il_params, reflection_params, thru_pool
            )
        else:
            if mixup:
                h_current = augment_mixup(h_current, thru_pool)
            if channel_cascade:
                h_current = augment_cascade(h_current, thru_pool)
            if insertion_loss_tilting:
                h_current = augment_il_tilt(h_current)
            if random_reflection:
                h_current = augment_reflections(h_current)

        for i in range(remainder):
            physics_valid = not mixup
            aug_class = "regularization" if mixup else "physical"

            group = {
                'thru': h_current[i],
                'fext': random.choice(associated_groups[i]['fext']) if associated_groups[i]['fext'] else None,
                'next': random.choice(associated_groups[i]['next']) if associated_groups[i]['next'] else None,
                'physics_valid': physics_valid,
                'augmentation_class': aug_class,
            }
            synthetic_dataset.append(group)

    return synthetic_dataset


# ==========================================
# Step 5: CLI and Main
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic SerDes channels with physics-based data augmentation"
    )
    parser.add_argument(
        "--s4p_dir",
        type=str,
        default=None,
        help="Root directory containing .s4p files (recursive). If None, uses dummy data."
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default="synthetic_channels.pt",
        help="Output file path for the generated channels (default: synthetic_channels.pt). 'synth' is auto-added if not present."
    )
    parser.add_argument(
        "--num_generated",
        type=int,
        default=50000,
        help="Total number of synthetic channel groups to generate (default: 50000)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for generation loop (default: 64)"
    )
    parser.add_argument(
        "--num_taps",
        type=int,
        default=CH_TAPS,
        help=f"Channel IR span in unit intervals (default: {CH_TAPS})"
    )
    parser.add_argument(
        "--samples_per_symbol",
        type=int,
        default=OVERSAMPLE_FACTOR,
        help=f"Number of samples per symbol for oversampling (default: {OVERSAMPLE_FACTOR})"
    )
    parser.add_argument(
        "--channel_cascade",
        action="store_true",
        help="Apply cascade augmentation (frequency-domain transfer multiplication). "
             "Note: This is an approximation assuming matched interfaces."
    )
    parser.add_argument(
        "--cascade_mode",
        type=str,
        default="scalar",
        choices=["scalar", "network"],
        help="Cascade implementation mode: 'scalar' multiplies transfer functions "
             "(ignores interface mismatch), 'network' cascades full S-parameter "
             "networks before extraction (more accurate but requires more computation)."
    )
    parser.add_argument(
        "--insertion_loss_tilting",
        action="store_true",
        help="Apply insertion loss augmentation (frequency-domain skin-effect and dielectric loss model)."
    )
    parser.add_argument(
        "--random_reflection",
        action="store_true",
        help="Apply reflection augmentation (frequency-domain delayed-echo superposition)."
    )
    parser.add_argument(
        "--mixup",
        action="store_true",
        help="Apply mixup augmentation (linear interpolation between channels)"
    )
    parser.add_argument(
        "--dummy_data",
        action="store_true",
        help="Use dummy synthetic data instead of parsing .s4p files"
    )
    parser.add_argument(
        "--num_base_channels",
        type=int,
        default=10,
        help="Number of base channel groups to generate when using dummy data (default: 10)"
    )
    parser.add_argument(
        "--port_pairing",
        type=str,
        default="auto",
        choices=["auto", "13-24", "12-34", "14-23"],
        help="Port pairing for 4-port files: 'auto' to infer, or explicit pairing "
             "(13-24, 12-34, 14-23). For differential pairs in single-ended format, "
             "'13-24' is often correct (ports 1,3 input; ports 2,4 output)."
    )
    parser.add_argument(
        "--transfer_mode",
        type=str,
        default="auto",
        choices=["auto", "s21", "sdd21"],
        help="Transfer extraction mode: 'auto' to use inspector inference, 's21' for "
             "raw single-ended S21, 'sdd21' for mixed-mode differential transfer. "
             "For differential pairs, 'sdd21' is usually physically more relevant than 's21'."
    )
    parser.add_argument(
        "--normalization",
        type=str,
        default="none",
        choices=["none", "peak", "l2"],
        help="Normalization mode for extracted channels: 'none' preserves raw amplitude "
             "(recommended for physics-based generation), 'peak' normalizes to peak=1, "
             "'l2' normalizes L2 norm to 1."
    )
    parser.add_argument(
        "--inspect_only",
        action="store_true",
        help="Only inspect Touchstone files and print reports without generating channels."
    )
    parser.add_argument(
        "--inspect_output",
        type=str,
        default=None,
        help="Output file path for --inspect_only reports. If not specified, prints to stdout."
    )

    args = parser.parse_args()

    span_ui = args.num_taps
    sps = args.samples_per_symbol

    num_taps = span_ui * sps

    if args.inspect_only:
        if args.s4p_dir is None:
            raise ValueError("--inspect_only requires --s4p_dir to be specified")
        out_file = None
        if args.inspect_output:
            out_file = open(args.inspect_output, 'w', encoding='utf-8')
            print(f"Writing inspection reports to {args.inspect_output}")
        inspect_s4p_directory(
            pathlib.Path(args.s4p_dir),
            baud_rate_hz=BAUD_RATE_HZ,
            port_pairing=args.port_pairing,
            transfer_mode=args.transfer_mode,
            output_file=out_file,
        )
        if out_file:
            out_file.close()
            print(f"Inspection complete. Output written to {args.inspect_output}")
        return

    if args.dummy_data or args.s4p_dir is None:
        print(f"Generating {args.num_base_channels} dummy base channel groups...")
        parsed_groups = generate_dummy_base_channels(
            num_channels=args.num_base_channels,
            span_ui=span_ui,
            samples_per_symbol=sps,
        )
        print(f"  Generated {len(parsed_groups)} dummy base channel groups")
    else:
        s4p_dir = pathlib.Path(args.s4p_dir)
        if not s4p_dir.exists():
            raise FileNotFoundError(f"Directory not found: {s4p_dir}")
        print(f"Parsing .s4p files from {s4p_dir} (recursive)...")
        parsed_groups = parse_s4p_directory_tree(
            s4p_dir,
            span_ui=span_ui,
            samples_per_symbol=sps,
            baud_rate_hz=BAUD_RATE_HZ,
            port_pairing=args.port_pairing,
            transfer_mode=args.transfer_mode,
            normalization=args.normalization,
        )
        print(f"  Parsed {len(parsed_groups)} channel groups from .s4p files")

    # Check if any augmentations are enabled
    any_aug = (args.channel_cascade or args.insertion_loss_tilting or
               args.random_reflection or args.mixup)

    if any_aug:
        print(f"\nGenerating {args.num_generated} synthetic channel groups with augmentations:")
        print(f"  - Cascade: {args.channel_cascade}")
        print(f"  - IL Tilt: {args.insertion_loss_tilting}")
        print(f"  - Reflections: {args.random_reflection}")
        print(f"  - Mixup: {args.mixup}")
    else:
        print(f"\nNo augmentations enabled. Saving {len(parsed_groups)} base channel groups as-is.")
        num_taps_out = span_ui * sps
        payload = {
            "meta": {
                "samples_per_symbol": sps,
                "baud_rate_hz": BAUD_RATE_HZ,
                "num_taps": num_taps_out,
                "span_ui": span_ui,
            },
            "channels": parsed_groups,
        }
        torch.save(payload, args.out_file)
        print(f"Saved to {args.out_file}")
        return

    # Generate synthetic channels
    synthetic_dataset = generate_synthetic_channels(
        parsed_groups=parsed_groups,
        num_generated=args.num_generated,
        batch_size=args.batch_size,
        channel_cascade=args.channel_cascade,
        cascade_mode=args.cascade_mode,
        insertion_loss_tilting=args.insertion_loss_tilting,
        random_reflection=args.random_reflection,
        mixup=args.mixup,
        use_freq_domain_aug=True,
        fs_hz=BAUD_RATE_HZ * sps,
    )

    # Auto-add 'synth' to output filename if not present
    if 'synth' not in args.out_file.lower():
        base, ext = os.path.splitext(args.out_file)
        args.out_file = f"{base}_synth{ext}"

    # Make filename rate-aware so different oversampling factors don't collide
    if 'sps' not in args.out_file.lower():
        base, ext = os.path.splitext(args.out_file)
        args.out_file = f"{base}_sps{sps}_baud{int(BAUD_RATE_HZ)}{ext}"

    # Summary statistics
    print(f"\n--- Summary ---")
    print(f"Base channel groups: {len(parsed_groups)}")
    print(f"Generated {len(synthetic_dataset)} synthetic channel groups")
    print(f"IR length: {num_taps} taps ({span_ui} UI x {sps} sps)")
    print(f"Baud rate: {BAUD_RATE_HZ:.3e} Hz, Target FS: {BAUD_RATE_HZ * sps:.3e} Hz")

    # Count FEXT/NEXT availability
    fext_count = sum(1 for g in synthetic_dataset if g['fext'] is not None)
    next_count = sum(1 for g in synthetic_dataset if g['next'] is not None)
    print(f"Groups with FEXT: {fext_count} ({100*fext_count/len(synthetic_dataset):.1f}%)")
    print(f"Groups with NEXT: {next_count} ({100*next_count/len(synthetic_dataset):.1f}%)")

    # THRU statistics
    thru_stack = torch.stack([g['thru'] for g in synthetic_dataset])
    print(f"\nTHRU statistics:")
    print(f"  Shape: {thru_stack.shape}")
    print(f"  Mean: {thru_stack.mean():.6f}")
    print(f"  Std: {thru_stack.std():.6f}")
    print(f"  Min: {thru_stack.min():.6f}")
    print(f"  Max: {thru_stack.max():.6f}")

    # Save output with metadata
    augmentation_summary = []
    if args.channel_cascade:
        augmentation_summary.append("cascade")
    if args.insertion_loss_tilting:
        augmentation_summary.append("il_tilt")
    if args.random_reflection:
        augmentation_summary.append("reflections")
    if args.mixup:
        augmentation_summary.append("mixup")

    payload = {
        "meta": {
            "samples_per_symbol": sps,
            "baud_rate_hz": BAUD_RATE_HZ,
            "num_taps": num_taps,
            "span_ui": span_ui,
            "normalization": args.normalization,
            "port_pairing": args.port_pairing,
            "transfer_mode": args.transfer_mode,
            "cascade_mode": args.cascade_mode,
            "augmentations": augmentation_summary if augmentation_summary else "none",
        },
        "channels": synthetic_dataset,
    }
    torch.save(payload, args.out_file)
    print(f"\nSaved to {args.out_file}")


if __name__ == "__main__":
    main()
