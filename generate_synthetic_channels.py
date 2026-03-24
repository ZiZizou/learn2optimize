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
"""

import argparse
import pathlib
from typing import List, Optional

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
# Step 2: Extract S-Parameters to Impulse Responses
# ==========================================

def parse_s4p_files(s4p_dir: pathlib.Path, num_taps: int = CH_TAPS) -> torch.Tensor:
    """
    Parse a directory of .s4p files and convert them to time-domain impulse responses.

    Args:
        s4p_dir: Directory containing .s4p Touchstone files
        num_taps: Fixed number of taps to resample/interpolate IRs to

    Returns:
        base_tensor: Tensor of shape [N, num_taps] containing all parsed base channels
    """
    if not HAS_SKRF:
        raise RuntimeError("scikit-rf is required to parse .s4p files. Install with: pip install scikit-rf")

    s4p_files = list(s4p_dir.glob("*.s4p"))
    if not s4p_files:
        raise FileNotFoundError(f"No .s4p files found in {s4p_dir}")

    base_channels = []
    for s4p_file in s4p_files:
        try:
            # Read the 4-port single-ended network
            network = skrf.Network(str(s4p_file))

            # Convert to mixed-mode (differential) S-parameters
            # p=2 indicates we want differential mode for ports 1,2 (originally single-ended)
            mm_network = skrf.se2gmm(network, p=2)

            # Extract Differential Insertion Loss Sdd21 = mm_network.s[p, q] where:
            # Sdd21 is at index [1, 0] in the mixed-mode S-matrix (output port 2, input port 1)
            sdd21 = mm_network.s[1, 0]

            # Get frequency array and compute time-domain impulse response
            # Using the built-in impulse_response method
            ir = mm_network.impulse_response(port=("mixed", 2, 1), window=("kaiser", 6.0))

            # ir is typically [N, 2, 2] or similar; extract the differential channel
            # The impulse response from se2gmm with port=("mixed", 2, 1) gives Sdd21 response
            if ir.ndim > 1:
                # Take the appropriate component (usually [0, 0] or just flatten)
                ir = ir.flatten()

            # Resample/interpolate to fixed num_taps
            ir_resampled = resample_ir(ir, num_taps)

            # Normalize
            ir_resampled = ir_resampled / (np.linalg.norm(ir_resampled) + 1e-8)

            base_channels.append(ir_resampled)

            print(f"  Parsed: {s4p_file.name} -> {len(ir)} taps -> resampled to {num_taps}")

        except Exception as e:
            print(f"  Warning: Failed to parse {s4p_file.name}: {e}")
            continue

    if not base_channels:
        raise RuntimeError("No valid channels could be parsed from the .s4p files")

    return torch.tensor(np.array(base_channels), dtype=torch.float32)


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


def generate_dummy_base_channels(num_channels: int = 10, num_taps: int = CH_TAPS) -> torch.Tensor:
    """
    Generate synthetic base channels when no .s4p files are available.
    Uses a simplified RC low-pass model with reflections (same as wireline_channel.py).

    Args:
        num_channels: Number of dummy channels to generate
        num_taps: Number of taps per channel

    Returns:
        Tensor of shape [num_channels, num_taps]
    """
    t = np.linspace(0, 5, num_taps)
    channels = []

    for _ in range(num_channels):
        tau = np.random.uniform(0.5, 1.5)
        h = np.exp(-t / tau)

        # Add discrete reflections
        num_reflections = np.random.randint(1, 4)
        for _ in range(num_reflections):
            idx = np.random.randint(5, num_taps)
            h[idx] += np.random.uniform(-0.2, 0.2)

        h = h / (np.linalg.norm(h) + 1e-8)
        channels.append(h)

    return torch.tensor(np.array(channels), dtype=torch.float32)


# ==========================================
# Step 3: Implement the 4 Augmentation Functions
# ==========================================

def augment_cascade(h_batch: torch.Tensor, base_pool: torch.Tensor) -> torch.Tensor:
    """
    Channel Cascade (Convolution): Simulates plugging two cables together.

    Randomly selects channels from the base_pool to convolve with h_batch using
    frequency-domain convolution for efficiency.

    Args:
        h_batch: Batch of channels [batch_size, num_taps]
        base_pool: Pool of base channels to sample from [N, num_taps]

    Returns:
        Augmented batch of same shape [batch_size, num_taps]
    """
    batch_size, num_taps = h_batch.shape

    # Sample random channels from base_pool for each batch element
    indices = torch.randint(0, len(base_pool), (batch_size,))
    h_cascade = base_pool[indices]  # [batch_size, num_taps]

    # Use FFT-based convolution for efficiency
    # Pad both to same length for proper convolution
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


def augment_il_tilt(h_batch: torch.Tensor, alpha_range: tuple = (0.0, 0.4)) -> torch.Tensor:
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
    max_amp: float = 0.15,
    min_idx: int = 10
) -> torch.Tensor:
    """
    Random Reflection (Impedance Mismatch): Injects discrete spikes into post-cursor tail.

    Args:
        h_batch: Batch of channels [batch_size, num_taps]
        max_refs: Maximum number of reflections to inject per channel
        max_amp: Maximum amplitude of reflection spikes
        min_idx: Minimum index for reflection placement (post-cursor region)

    Returns:
        Augmented batch of same shape [batch_size, num_taps]
    """
    h_aug = h_batch.clone()
    batch_size, num_taps = h_aug.shape

    for i in range(batch_size):
        num_refs = torch.randint(1, max_refs + 1, (1,)).item()
        for _ in range(num_refs):
            idx = torch.randint(min_idx, num_taps, (1,)).item()
            amp = torch.empty(1).uniform_(-max_amp, max_amp).item()
            h_aug[i, idx] += amp

    return h_aug / (torch.norm(h_aug, dim=1, keepdim=True) + 1e-8)


def augment_mixup(h_batch: torch.Tensor, base_pool: torch.Tensor, lam_range: tuple = (0.2, 0.8)) -> torch.Tensor:
    """
    Mixup: Linear interpolation between two channels for regularization.

    h_new = lambda * h_batch + (1 - lambda) * h_random

    Args:
        h_batch: Batch of channels [batch_size, num_taps]
        base_pool: Pool of base channels to sample from [N, num_taps]
        lam_range: Tuple of (min, max) lambda values for interpolation

    Returns:
        Augmented batch of same shape [batch_size, num_taps]
    """
    batch_size = h_batch.shape[0]

    # Sample random channels from base_pool
    indices = torch.randint(0, len(base_pool), (batch_size,))
    h_random = base_pool[indices]

    # Random lambda uniformly sampled
    lam = torch.empty(batch_size, 1).uniform_(lam_range[0], lam_range[1])

    # Linear interpolation
    h_aug = lam * h_batch + (1.0 - lam) * h_random

    return h_aug / (torch.norm(h_aug, dim=1, keepdim=True) + 1e-8)


# ==========================================
# Step 4: CLI and Composable Pipeline
# ==========================================

def sample_random_channels(pool: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Sample a random batch of channels from a pool."""
    indices = torch.randint(0, len(pool), (batch_size,))
    return pool[indices].clone()


def generate_synthetic_channels(
    base_pool: torch.Tensor,
    num_generated: int,
    batch_size: int = 64,
    channel_cascade: bool = False,
    insertion_loss_tilting: bool = False,
    random_reflection: bool = False,
    mixup: bool = False,
) -> torch.Tensor:
    """
    Generate synthetic channels by applying augmentations to base channels.

    Pipeline order (as specified):
    1. Mixup/Cascade (Creates the new base geometry)
    2. Insertion Loss Tilting (Applies global frequency-domain shift)
    3. Random Reflections (Injects local time-domain spikes)

    Args:
        base_pool: Pool of parsed base channels [N, num_taps]
        num_generated: Total number of synthetic channels to generate
        batch_size: Batch size for generation loop
        channel_cascade: Whether to apply cascade augmentation
        insertion_loss_tilting: Whether to apply IL tilt augmentation
        random_reflection: Whether to apply reflection augmentation
        mixup: Whether to apply mixup augmentation

    Returns:
        Tensor of shape [num_generated, num_taps]
    """
    all_channels = []
    num_complete_batches = num_generated // batch_size
    remainder = num_generated % batch_size

    for batch_idx in range(num_complete_batches):
        # Sample a starting batch from the parsed S4P base pool
        h_current = sample_random_channels(base_pool, batch_size)

        # Apply augmentations in order
        if mixup:
            h_current = augment_mixup(h_current, base_pool)
        if channel_cascade:
            h_current = augment_cascade(h_current, base_pool)
        if insertion_loss_tilting:
            h_current = augment_il_tilt(h_current)
        if random_reflection:
            h_current = augment_reflections(h_current)

        all_channels.append(h_current)

    # Handle remainder
    if remainder > 0:
        h_current = sample_random_channels(base_pool, remainder)
        if mixup:
            h_current = augment_mixup(h_current, base_pool)
        if channel_cascade:
            h_current = augment_cascade(h_current, base_pool)
        if insertion_loss_tilting:
            h_current = augment_il_tilt(h_current)
        if random_reflection:
            h_current = augment_reflections(h_current)
        all_channels.append(h_current)

    return torch.cat(all_channels, dim=0)


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic SerDes channels with physics-based data augmentation"
    )
    parser.add_argument(
        "--s4p_dir",
        type=str,
        default=None,
        help="Directory containing .s4p Touchstone files. If None, uses dummy data."
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default="synthetic_channels.pt",
        help="Output file path for the generated channels (default: synthetic_channels.pt)"
    )
    parser.add_argument(
        "--num_generated",
        type=int,
        default=50000,
        help="Total number of synthetic channels to generate (default: 50000)"
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
        help="Number of base channels to generate when using dummy data (default: 10)"
    )

    args = parser.parse_args()

    # Parse or generate base channels
    if args.dummy_data or args.s4p_dir is None:
        print(f"Generating {args.num_base_channels} dummy base channels...")
        base_pool = generate_dummy_base_channels(
            num_channels=args.num_base_channels,
            num_taps=args.num_taps
        )
        print(f"  Generated {len(base_pool)} dummy base channels")
    else:
        s4p_dir = pathlib.Path(args.s4p_dir)
        if not s4p_dir.exists():
            raise FileNotFoundError(f"Directory not found: {s4p_dir}")
        print(f"Parsing .s4p files from {s4p_dir}...")
        base_pool = parse_s4p_files(s4p_dir, num_taps=args.num_taps)
        print(f"  Parsed {len(base_pool)} base channels from .s4p files")

    # Check if any augmentations are enabled
    any_aug = (args.channel_cascade or args.insertion_loss_tilting or
               args.random_reflection or args.mixup)

    if any_aug:
        print(f"\nGenerating {args.num_generated} synthetic channels with augmentations:")
        print(f"  - Cascade: {args.channel_cascade}")
        print(f"  - IL Tilt: {args.insertion_loss_tilting}")
        print(f"  - Reflections: {args.random_reflection}")
        print(f"  - Mixup: {args.mixup}")
    else:
        print(f"\nNo augmentations enabled. Saving {len(base_pool)} base channels as-is.")
        # Just save the base pool
        torch.save(base_pool, args.out_file)
        print(f"Saved to {args.out_file}")
        return

    # Generate synthetic channels
    synthetic_channels = generate_synthetic_channels(
        base_pool=base_pool,
        num_generated=args.num_generated,
        batch_size=args.batch_size,
        channel_cascade=args.channel_cascade,
        insertion_loss_tilting=args.insertion_loss_tilting,
        random_reflection=args.random_reflection,
        mixup=args.mixup,
    )

    # Summary statistics
    print(f"\n--- Summary ---")
    print(f"Base pool size: {len(base_pool)} channels")
    print(f"Generated {len(synthetic_channels)} synthetic channels")
    print(f"Shape: {synthetic_channels.shape}")
    print(f"dtype: {synthetic_channels.dtype}")
    print(f"Mean: {synthetic_channels.mean():.6f}")
    print(f"Std: {synthetic_channels.std():.6f}")
    print(f"Min: {synthetic_channels.min():.6f}")
    print(f"Max: {synthetic_channels.max():.6f}")

    # Save output
    torch.save(synthetic_channels, args.out_file)
    print(f"\nSaved to {args.out_file}")


if __name__ == "__main__":
    main()
