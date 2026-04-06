"""
Synthetic SerDes Channel Generator with Physics-Based Data Augmentation.

Reads industry-standard .s4p (Touchstone) frequency-domain files, converts them to
time-domain Impulse Responses (IRs), and applies physics-based data augmentations
to synthetically expand the dataset for Learned Optimizer (L2O) training.

Augmentations:
- Cascade: Simulates plugging two cables together (Convolution)
- Insertion Loss (IL) Tilting: Simulates variations in dielectric loss/cable length (RC filtering)
- Random Reflections: Simulates via/connector impedance mismatches (Time-domain spike injection)
- Mixup: Mathematical regularization (Linear interpolation between two channels)

Physics Note:
The THRU (main victim channel), FEXT (Far-End Crosstalk), and NEXT (Near-End Crosstalk)
are kept separate. Augmentations are applied ONLY to the THRU channel because:
- Aggressor transmitters send independent, asynchronous random bit sequences
- Superposition applies to voltages at the receiver, not to impulse responses
- FEXT/NEXT act as bounded colored noise signatures specific to the physical layout
"""

import argparse
import os
import pathlib
import random
from typing import List, Optional, Dict, Union

import numpy as np
import torch

try:
    import skrf
    HAS_SKRF = True
except ImportError:
    HAS_SKRF = False
    print("Warning: scikit-rf not installed. Use --dummy_data flag for synthetic generation.")


# ==========================================
# Configuration
# ==========================================
CH_TAPS = 50  # Number of taps in the wireline channel (from config.py)


# ==========================================
# Step 2: S-Parameter Extraction Helper
# ==========================================

def extract_and_resample_ir(filepath: pathlib.Path, num_taps: int = CH_TAPS) -> np.ndarray:
    """
    Extract impulse response from an S4P file and resample to fixed number of taps.

    Args:
        filepath: Path to the .s4p file
        num_taps: Fixed number of taps to resample to

    Returns:
        1D numpy array of shape [num_taps]
    """
    if not HAS_SKRF:
        raise RuntimeError("scikit-rf is required to parse .s4p files")

    # Read the network
    network = skrf.Network(str(filepath))

    # Extract S21 (forward transmission from port 1 to port 2)
    # For differential THRU channels, S21 represents the differential insertion loss
    if network.number_of_ports == 4:
        # 4-port file: extract S21
        s21 = network.s[:, 1, 0]  # row=1, col=0 -> S21
    elif network.number_of_ports == 2:
        # 2-port file: extract S21 directly
        s21 = network.s[:, 1, 0]
    else:
        raise ValueError(f"Unsupported number of ports: {network.number_of_ports}")

    # Get frequency info
    freq = network.frequency
    f = freq.f
    n = len(s21)

    # Check if frequency starts at 0
    if f[0] != 0:
        print(f"  Warning: Frequency doesn't start at 0 Hz for {filepath.name}, results may be distorted")

    # Compute impulse response using IFFT
    # IFFT of S21 gives the time-domain impulse response
    ir = np.fft.ifftshift(np.fft.ifft(s21, n=n))

    # Take magnitude (the physical impulse response is the envelope)
    # For a causal system, we use the real part or magnitude
    ir = np.abs(ir)

    # Resample/interpolate to fixed num_taps
    ir_resampled = resample_ir(ir, num_taps)

    # Normalize
    ir_resampled = ir_resampled / (np.linalg.norm(ir_resampled) + 1e-8)

    return ir_resampled


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
    num_taps: int = CH_TAPS
) -> List[Dict[str, Union[torch.Tensor, List[torch.Tensor], None]]]:
    """
    Recursively walk a directory tree and parse S4P files into channel groups.

    Each leaf directory containing .s4p files is treated as a channel group with:
    - Exactly 1 THRU (main victim channel)
    - 0 or more FEXT (Far-End Crosstalk) files
    - 0 or more NEXT (Near-End Crosstalk) files

    Args:
        base_dir: Root directory to walk
        num_taps: Number of taps to resample IRs to

    Returns:
        List of channel group dictionaries:
        [{
            'thru': torch.Tensor of shape [num_taps],  # Main victim channel
            'fext': List[torch.Tensor] or None,       # List of FEXT IRs
            'next': List[torch.Tensor] or None,       # List of NEXT IRs
            'source_dir': str                           # Source directory name for logging
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
                ir = extract_and_resample_ir(filepath, num_taps)
                ir_tensor = torch.tensor(ir, dtype=torch.float32)

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


# ==========================================
# Dummy Data Generation
# ==========================================

def generate_dummy_base_channels(
    num_channels: int = 10,
    num_taps: int = CH_TAPS
) -> List[Dict[str, Union[torch.Tensor, List[torch.Tensor], None]]]:
    """
    Generate synthetic base channels when no .s4p files are available.
    Uses a simplified RC low-pass model with reflections (same as wireline_channel.py).

    Args:
        num_channels: Number of dummy channel groups to generate
        num_taps: Number of taps per channel

    Returns:
        List of channel group dictionaries (same structure as parse_s4p_directory_tree)
    """
    groups = []

    for i in range(num_channels):
        t = np.linspace(0, 5, num_taps)
        tau = np.random.uniform(0.5, 1.5)
        h = np.exp(-t / tau)

        # Add discrete reflections
        num_reflections = np.random.randint(1, 4)
        for _ in range(num_reflections):
            idx = np.random.randint(5, num_taps)
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

def augment_cascade(
    h_batch: torch.Tensor,
    thru_pool: torch.Tensor
) -> torch.Tensor:
    """
    Channel Cascade (Convolution): Simulates plugging two cables together.

    Randomly selects channels from the thru_pool to convolve with h_batch using
    frequency-domain convolution for efficiency.

    Args:
        h_batch: Batch of THRU channels [batch_size, num_taps]
        thru_pool: Pool of base THRU channels to sample from [N, num_taps]

    Returns:
        Augmented batch of same shape [batch_size, num_taps]
    """
    batch_size, num_taps = h_batch.shape

    # Sample random channels from thru_pool for each batch element
    indices = torch.randint(0, len(thru_pool), (batch_size,))
    h_cascade = thru_pool[indices]  # [batch_size, num_taps]

    # Use FFT-based convolution for efficiency
    conv_length = num_taps * 2 - 1

    # Convert to frequency domain
    h_batch_fft = torch.fft.rfft(h_batch, n=conv_length)
    h_cascade_fft = torch.fft.rfft(h_cascade, n=conv_length)

    # Multiply in frequency domain (convolution theorem)
    product_fft = h_batch_fft * h_cascade_fft

    # Back to time domain
    result = torch.fft.irfft(product_fft, n=conv_length)

    # Truncate back to num_taps and normalize
    result = result[:, :num_taps]

    return result / (torch.norm(result, dim=1, keepdim=True) + 1e-8)


def augment_il_tilt(
    h_batch: torch.Tensor,
    alpha_range: tuple = (0.0, 0.4)
) -> torch.Tensor:
    """
    Insertion Loss Tilting (RC Filtering): Simulates extra high-frequency loss.

    Uses a randomized 1st-order low-pass filter:
    h_new[n] = (1 - alpha)*h[n] + alpha*h[n-1]

    Args:
        h_batch: Batch of channels [batch_size, num_taps]
        alpha_range: Tuple of (min, max) alpha values for the filter

    Returns:
        Augmented batch of same shape [batch_size, num_taps]
    """
    batch_size = h_batch.shape[0]

    # Random alpha per channel: 0 = no loss, 0.4 = heavy loss
    alpha = torch.empty(batch_size, 1).uniform_(alpha_range[0], alpha_range[1])

    # Shift right by 1 (causal filter: output depends on current and previous input)
    h_shifted = torch.roll(h_batch, shifts=1, dims=1)
    h_shifted[:, 0] = 0.0  # Zero initial condition

    # Apply first-order low-pass filter
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

    Physics-based correction: Reflection amplitude is now scaled by the main cursor
    (peak amplitude) of each channel, ensuring the reflection coefficient Gamma
    remains bounded. A reflection cannot exceed the incident signal amplitude.

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

    for i in range(batch_size):
        # Find the main cursor amplitude for this specific channel
        # This ensures reflections are bounded by the actual signal energy
        peak_amp = torch.max(torch.abs(h_aug[i]))

        num_refs = torch.randint(1, max_refs + 1, (1,)).item()
        for _ in range(num_refs):
            idx = torch.randint(min_idx, num_taps, (1,)).item()
            # Scale reflection by peak amplitude (physically bounded Gamma)
            amp_ratio = torch.empty(1).uniform_(-max_ratio, max_ratio).item()
            h_aug[i, idx] += amp_ratio * peak_amp

    return h_aug / (torch.norm(h_aug, dim=1, keepdim=True) + 1e-8)


def augment_mixup(
    h_batch: torch.Tensor,
    thru_pool: torch.Tensor,
    lam_range: tuple = (0.2, 0.8)
) -> torch.Tensor:
    """
    Mixup: Phase-aligned linear interpolation between two channels for regularization.

    Physics-based correction: Standard mixup can create invalid multi-drop topologies
    when channels have different "flight times" (main cursor at different indices).
    This version phase-aligns channels before interpolation so the result represents
    a valid point-to-point link, not a multi-path channel.

    h_new = lambda * h_batch + (1 - lambda) * h_random (aligned)

    Args:
        h_batch: Batch of THRU channels [batch_size, num_taps]
        thru_pool: Pool of base THRU channels to sample from [N, num_taps]
        lam_range: Tuple of (min, max) lambda values for interpolation

    Returns:
        Augmented batch of same shape [batch_size, num_taps]
    """
    batch_size = h_batch.shape[0]

    # Sample random channels from thru_pool
    indices = torch.randint(0, len(thru_pool), (batch_size,))
    h_random = thru_pool[indices]

    # Phase-align h_random to h_batch before mixing
    for i in range(batch_size):
        # Find main cursors (peak locations) in both channels
        idx_batch = torch.argmax(torch.abs(h_batch[i]))
        idx_random = torch.argmax(torch.abs(h_random[i]))

        # Calculate shift to align h_random's peak to h_batch's peak
        shift = (idx_batch - idx_random).item()

        # Roll to shift (positive shift = delay, negative = advance)
        h_random_aligned = torch.roll(h_random[i], shifts=shift, dims=0)

        # Zero out wrapped-around artifacts to maintain causality
        if shift > 0:
            # Delayed: zeros at the beginning (causal)
            h_random_aligned[:shift] = 0.0
        elif shift < 0:
            # Advanced: zeros at the end (causal)
            h_random_aligned[shift:] = 0.0

        h_random[i] = h_random_aligned

    # Random lambda uniformly sampled
    lam = torch.empty(batch_size, 1).uniform_(lam_range[0], lam_range[1])

    # Linear interpolation on phase-aligned tensors
    h_aug = lam * h_batch + (1.0 - lam) * h_random

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
    insertion_loss_tilting: bool = False,
    random_reflection: bool = False,
    mixup: bool = False,
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

    Returns:
        List of synthetic channel group dicts:
        [{
            'thru': torch.Tensor [num_taps],
            'fext': torch.Tensor or None,
            'next': torch.Tensor or None,
        }, ...]
    """
    # Build a pool of just THRU channels for cascade/mixup operations
    thru_pool = torch.stack([g['thru'] for g in parsed_groups])

    synthetic_dataset = []
    num_complete_batches = num_generated // batch_size
    remainder = num_generated % batch_size

    for batch_idx in range(num_complete_batches):
        # Sample random groups
        h_current, associated_groups = sample_random_groups(parsed_groups, batch_size)

        # Apply augmentations in order (only to THRU)
        if mixup:
            h_current = augment_mixup(h_current, thru_pool)
        if channel_cascade:
            h_current = augment_cascade(h_current, thru_pool)
        if insertion_loss_tilting:
            h_current = augment_il_tilt(h_current)
        if random_reflection:
            h_current = augment_reflections(h_current)

        # Build output groups
        for i in range(batch_size):
            group = {
                'thru': h_current[i],  # [num_taps]
            }

            # Randomly sample one FEXT if available
            if associated_groups[i]['fext']:
                group['fext'] = random.choice(associated_groups[i]['fext'])
            else:
                group['fext'] = None

            # Randomly sample one NEXT if available
            if associated_groups[i]['next']:
                group['next'] = random.choice(associated_groups[i]['next'])
            else:
                group['next'] = None

            synthetic_dataset.append(group)

    # Handle remainder
    if remainder > 0:
        h_current, associated_groups = sample_random_groups(parsed_groups, remainder)

        if mixup:
            h_current = augment_mixup(h_current, thru_pool)
        if channel_cascade:
            h_current = augment_cascade(h_current, thru_pool)
        if insertion_loss_tilting:
            h_current = augment_il_tilt(h_current)
        if random_reflection:
            h_current = augment_reflections(h_current)

        for i in range(remainder):
            group = {
                'thru': h_current[i],
            }

            if associated_groups[i]['fext']:
                group['fext'] = random.choice(associated_groups[i]['fext'])
            else:
                group['fext'] = None

            if associated_groups[i]['next']:
                group['next'] = random.choice(associated_groups[i]['next'])
            else:
                group['next'] = None

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
        help=f"Number of taps per channel (default: {CH_TAPS})"
    )
    parser.add_argument(
        "--channel_cascade",
        action="store_true",
        help="Apply cascade augmentation (convolution with random channels)"
    )
    parser.add_argument(
        "--insertion_loss_tilting",
        action="store_true",
        help="Apply insertion loss tilting (RC low-pass filtering)"
    )
    parser.add_argument(
        "--random_reflection",
        action="store_true",
        help="Apply random reflection augmentation (impedance mismatch spikes)"
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

    args = parser.parse_args()

    # Parse or generate base channel groups
    if args.dummy_data or args.s4p_dir is None:
        print(f"Generating {args.num_base_channels} dummy base channel groups...")
        parsed_groups = generate_dummy_base_channels(
            num_channels=args.num_base_channels,
            num_taps=args.num_taps
        )
        print(f"  Generated {len(parsed_groups)} dummy base channel groups")
    else:
        s4p_dir = pathlib.Path(args.s4p_dir)
        if not s4p_dir.exists():
            raise FileNotFoundError(f"Directory not found: {s4p_dir}")
        print(f"Parsing .s4p files from {s4p_dir} (recursive)...")
        parsed_groups = parse_s4p_directory_tree(s4p_dir, num_taps=args.num_taps)
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
        torch.save(parsed_groups, args.out_file)
        print(f"Saved to {args.out_file}")
        return

    # Generate synthetic channels
    synthetic_dataset = generate_synthetic_channels(
        parsed_groups=parsed_groups,
        num_generated=args.num_generated,
        batch_size=args.batch_size,
        channel_cascade=args.channel_cascade,
        insertion_loss_tilting=args.insertion_loss_tilting,
        random_reflection=args.random_reflection,
        mixup=args.mixup,
    )

    # Mark each group as synthetic
    for group in synthetic_dataset:
        group['synth'] = True

    # Auto-add 'synth' to output filename if not present
    if 'synth' not in args.out_file.lower():
        base, ext = os.path.splitext(args.out_file)
        args.out_file = f"{base}_synth{ext}"

    # Summary statistics
    print(f"\n--- Summary ---")
    print(f"Base channel groups: {len(parsed_groups)}")
    print(f"Generated {len(synthetic_dataset)} synthetic channel groups")

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

    # Save output
    torch.save(synthetic_dataset, args.out_file)
    print(f"\nSaved to {args.out_file}")


if __name__ == "__main__":
    main()
