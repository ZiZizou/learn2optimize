"""
IEEE 802.3ck SerDes Channel S-Parameter Preprocessor.

Crawls ZIP files containing .s4p Touchstone files, extracts and groups them by channel,
converts to time-domain Impulse Responses (IRs), and combines thru + crosstalk (FEXT/NEXT)
into a single noisy impulse response per channel.

Output: extracted_base_pool.pt - PyTorch tensor of shape [N, 50] for use as base_pool
        in generate_synthetic_channels.py
"""

import argparse
import pathlib
import re
import shutil
import tempfile
import zipfile
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import torch

try:
    import skrf
    HAS_SKRF = True
except ImportError:
    HAS_SKRF = False


# ==========================================
# Configuration
# ==========================================
NUM_TAPS = 50  # Fixed number of taps for all IRs


# ==========================================
# S-Parameter to Impulse Response
# ==========================================

def load_s4p_as_ir(s4p_path: pathlib.Path, num_taps: int = NUM_TAPS) -> Optional[np.ndarray]:
    """
    Load a .s4p file and convert to a time-domain impulse response.

    Args:
        s4p_path: Path to the .s4p file
        num_taps: Number of taps to resample the IR to

    Returns:
        Normalized impulse response array of shape [num_taps], or None if failed
    """
    if not HAS_SKRF:
        raise RuntimeError("scikit-rf is required. Install with: uv pip install scikit-rf")

    try:
        # Read the 4-port single-ended network
        network = skrf.Network(str(s4p_path))

        # Check if network has valid frequency data
        if network.frequency is None or len(network.frequency) == 0:
            print(f"    Warning: {s4p_path.name} has no frequency points, skipping")
            return None

        # Check number of ports - need 4 ports for differential conversion
        if network.number_of_ports != 4:
            print(f"    Warning: {s4p_path.name} has {network.number_of_ports} ports, expected 4, skipping")
            return None

        # Convert to mixed-mode (differential) S-parameters in-place
        # p=2 indicates we want 2 differential ports (4 single-ended -> 2 differential)
        network.se2gmm(p=2)

        # After conversion, the S-matrix is now in mixed-mode
        # Sdd21 is at index [1, 0] in the mixed-mode S-matrix (differential output port 2, input port 1)
        # Note: ports are 0-indexed in skrf
        sdd21 = network.s[1, 0]  # This is a 1D array of complex S21 values

        # Compute time-domain impulse response manually using FFT
        # The S-parameter is frequency domain data; we need to IFFT to get time domain
        n_freq = len(sdd21)
        freq = network.frequency

        # Extrapolate to DC if needed (first frequency should be 0 for accurate impulse response)
        # Use simple DC extrapolation: set DC value to match first point trend
        sdd21_extended = np.concatenate([[sdd21[0]], sdd21])
        n_extended = len(sdd21_extended)

        # Apply window and IFFT
        window = np.kaiser(n_extended, 6.0)
        sdd21_windowed = sdd21_extended * window

        # IFFT to get time-domain impulse response
        ir = np.fft.ifft(sdd21_windowed)
        ir = np.real(ir)  # Take real part (causal response)

        # Handle different output shapes
        if ir.ndim > 1:
            ir = ir.flatten()

        if len(ir) == 0:
            print(f"    Warning: {s4p_path.name} produced empty IR")
            return None

        # Resample to fixed num_taps
        ir_resampled = resample_ir(ir, num_taps)

        # Normalize
        norm = np.linalg.norm(ir_resampled)
        if norm < 1e-10:
            print(f"    Warning: {s4p_path.name} has near-zero norm after IR extraction")
            return None

        return ir_resampled / norm

    except Exception as e:
        print(f"    Warning: Failed to process {s4p_path.name}: {e}")
        return None


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
# ZIP Extraction and File Grouping
# ==========================================

def extract_zips_and_find_s4p(
    zip_dir: pathlib.Path,
    temp_dir: pathlib.Path
) -> List[pathlib.Path]:
    """
    Extract all ZIP files in zip_dir to temp_dir, recursively handling nested ZIPs.

    Args:
        zip_dir: Directory containing ZIP files
        temp_dir: Temporary directory for extraction

    Returns:
        List of all .s4p file paths found
    """
    s4p_files = []

    zip_files = list(zip_dir.glob("*.zip"))
    print(f"Found {len(zip_files)} ZIP files to process")

    for zip_file in zip_files:
        print(f"  Processing {zip_file.name}...")
        try:
            with zipfile.ZipFile(zip_file, 'r') as zf:
                # Check if this ZIP contains nested ZIPs or direct .s4p files
                names = zf.namelist()

                # Separate nested ZIPs from direct s4p files
                nested_zips = [n for n in names if n.lower().endswith('.zip')]
                direct_s4ps = [n for n in names if n.lower().endswith('.s4p')]

                if nested_zips:
                    # Extract nested ZIPs and search them
                    for nested_zip_name in nested_zips:
                        # Extract nested ZIP to a subdirectory
                        nested_base = pathlib.Path(nested_zip_name).stem
                        extract_subdir = temp_dir / f"{zip_file.stem}_{nested_base}"
                        extract_subdir.mkdir(exist_ok=True)

                        # Extract the nested ZIP content
                        with zf.open(nested_zip_name) as nested_zf_data:
                            nested_zip_path = extract_subdir / nested_zip_name
                            with open(nested_zip_path, 'wb') as out_f:
                                out_f.write(nested_zf_data.read())

                        # Now open and extract the nested ZIP
                        with zipfile.ZipFile(nested_zip_path, 'r') as nested_zf:
                            nested_zf.extractall(extract_subdir)

                            # Find all .s4p files in the nested ZIP
                            for name in nested_zf.namelist():
                                if name.lower().endswith('.s4p'):
                                    s4p_files.append(extract_subdir / name)

                if direct_s4ps:
                    # Direct .s4p files - extract them
                    extract_subdir = temp_dir / zip_file.stem
                    extract_subdir.mkdir(exist_ok=True)
                    zf.extractall(extract_subdir)

                    for name in direct_s4ps:
                        s4p_files.append(extract_subdir / name)

        except Exception as e:
            print(f"    Warning: Error processing {zip_file.name}: {e}")
            continue

    return s4p_files


def group_s4p_files(s4p_files: List[pathlib.Path]) -> List[Dict[str, List[pathlib.Path]]]:
    """
    Group .s4p files by their channel prefix.

    Files for the same channel share a common prefix before the _thru, _NEXT, or _FEXT suffix.

    Args:
        s4p_files: List of paths to .s4p files

    Returns:
        List of groups, each containing dict with 'thru_path', 'fext_paths', 'next_paths'
    """

    # Regex patterns to identify file types and extract group prefix
    # Pattern 1: <prefix>_THRU.s4p (THRU as suffix, no number)
    thru_pattern = re.compile(
        r'^(.+?)_THRU\.s4p$',
        re.IGNORECASE
    )

    # Pattern 2: <prefix>_FEXT<n>.s4p (FEXT followed by number at end)
    fext_suffix_pattern = re.compile(
        r'^(.+?)_FEXT\d+\.s4p$',
        re.IGNORECASE
    )

    # Pattern 3: <prefix>_NEXT<n>.s4p (NEXT followed by number at end)
    next_suffix_pattern = re.compile(
        r'^(.+?)_NEXT\d+\.s4p$',
        re.IGNORECASE
    )

    # Pattern 4: Thru_<prefix>.s4p (Thru as prefix, like Thru_Link_25_...)
    thru_prefix_pattern = re.compile(
        r'^Thru_(.+)\.s4p$',
        re.IGNORECASE
    )

    # Pattern 5: FEXT_<prefix>.s4p (FEXT as prefix)
    fext_prefix_pattern = re.compile(
        r'^FEXT_(.+)\.s4p$',
        re.IGNORECASE
    )

    # Pattern 6: NEXT_<prefix>.s4p (NEXT as prefix)
    next_prefix_pattern = re.compile(
        r'^NEXT_(.+)\.s4p$',
        re.IGNORECASE
    )

    def extract_prefix_heuristic(filename):
        """Extract prefix by removing known suffixes/prefixes"""
        # Remove _THRU, _FEXT<n>, _NEXT<n> suffixes
        name = re.sub(r'_THRU(\d*)\.s4p$', '', filename, flags=re.IGNORECASE)
        name = re.sub(r'_FEXT\d+\.s4p$', '', name, flags=re.IGNORECASE)
        name = re.sub(r'_NEXT\d+\.s4p$', '', name, flags=re.IGNORECASE)
        # Remove Thru_, FEXT_, NEXT_ prefixes
        name = re.sub(r'^Thru_', '', name, flags=re.IGNORECASE)
        name = re.sub(r'^FEXT_', '', name, flags=re.IGNORECASE)
        name = re.sub(r'^NEXT_', '', name, flags=re.IGNORECASE)
        return name

    # Dictionary to group files by prefix
    groups = defaultdict(lambda: {
        'thru': None,
        'fext': [],
        'next': []
    })

    for s4p_path in s4p_files:
        filename = s4p_path.name
        categorized = False

        # Try thru suffix pattern: xxx_THU.s4p
        match = thru_pattern.match(filename)
        if match:
            prefix = match.group(1)
            groups[prefix]['thru'] = s4p_path
            categorized = True

        # Try fext suffix pattern: xxx_FEXT<n>.s4p
        if not categorized:
            match = fext_suffix_pattern.match(filename)
            if match:
                prefix = match.group(1)
                groups[prefix]['fext'].append(s4p_path)
                categorized = True

        # Try next suffix pattern: xxx_NEXT<n>.s4p
        if not categorized:
            match = next_suffix_pattern.match(filename)
            if match:
                prefix = match.group(1)
                groups[prefix]['next'].append(s4p_path)
                categorized = True

        # Try thru prefix pattern: Thru_xxx.s4p
        if not categorized:
            match = thru_prefix_pattern.match(filename)
            if match:
                prefix = match.group(1)
                groups[prefix]['thru'] = s4p_path
                categorized = True

        # Try fext prefix pattern: FEXT_xxx.s4p
        if not categorized:
            match = fext_prefix_pattern.match(filename)
            if match:
                prefix = match.group(1)
                groups[prefix]['fext'].append(s4p_path)
                categorized = True

        # Try next prefix pattern: NEXT_xxx.s4p
        if not categorized:
            match = next_prefix_pattern.match(filename)
            if match:
                prefix = match.group(1)
                groups[prefix]['next'].append(s4p_path)
                categorized = True

        if not categorized:
            # Fallback: use heuristic extraction
            prefix = extract_prefix_heuristic(filename)
            if prefix != filename:  # Only add if we actually extracted something
                # Try to determine type from original filename
                if '_THRU' in filename.upper():
                    groups[prefix]['thru'] = s4p_path
                elif '_FEXT' in filename.upper():
                    groups[prefix]['fext'].append(s4p_path)
                elif '_NEXT' in filename.upper():
                    groups[prefix]['next'].append(s4p_path)
                else:
                    print(f"    Warning: Could not categorize {filename}")
            else:
                print(f"    Warning: Could not categorize {filename}")
        print(f"    Warning: Could not categorize {filename}, attempting heuristic grouping")

    # Filter to only include groups with a valid thru file
    valid_groups = []
    for prefix, group in groups.items():
        if group['thru'] is not None:
            valid_groups.append(group)
        else:
            print(f"    Warning: Group '{prefix}' has no thru file, skipping")

    return valid_groups


def find_and_group_s4p_files(temp_dir: pathlib.Path) -> List[Dict[str, List[pathlib.Path]]]:
    """
    Recursively find all .s4p files in temp_dir and group them by channel.

    Args:
        temp_dir: Directory to search

    Returns:
        List of grouped channels
    """
    s4p_files = []

    # Recursively find all .s4p files
    for p in temp_dir.rglob("*.s4p"):
        s4p_files.append(p)

    print(f"Found {len(s4p_files)} .s4p files total")

    if not s4p_files:
        return []

    return group_s4p_files(s4p_files)


# ==========================================
# Main Processing
# ==========================================

def process_channel_group(
    group: Dict[str, List[pathlib.Path]],
    num_taps: int = NUM_TAPS
) -> Optional[np.ndarray]:
    """
    Process a single channel group and return the combined IR.

    Args:
        group: Dict with 'thru', 'fext' (list), 'next' (list)
        num_taps: Number of taps

    Returns:
        Combined impulse response array, or None if processing failed
    """
    # Load thru IR
    h_thru = load_s4p_as_ir(group['thru'], num_taps)
    if h_thru is None:
        return None

    h_total = h_thru.copy()

    # Process FEXT crosstalk
    for fext_path in group['fext']:
        h_fext = load_s4p_as_ir(fext_path, num_taps)
        if h_fext is not None:
            # Random polarity to simulate random data on aggressor lanes
            sign = np.random.choice([-1, 1])
            h_total += sign * h_fext

    # Process NEXT crosstalk
    for next_path in group['next']:
        h_next = load_s4p_as_ir(next_path, num_taps)
        if h_next is not None:
            # Random polarity to simulate random data on aggressor lanes
            sign = np.random.choice([-1, 1])
            h_total += sign * h_next

    # Normalize
    norm = np.linalg.norm(h_total)
    if norm < 1e-10:
        return None

    return h_total / norm


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess IEEE 802.3ck SerDes channel S-parameter files"
    )
    parser.add_argument(
        "--zip_dir",
        type=str,
        default="wireline_s4p_models/802_3ck/original",
        help="Directory containing ZIP files with .s4p files"
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default="extracted_base_pool.pt",
        help="Output file for combined channels tensor"
    )
    parser.add_argument(
        "--num_taps",
        type=int,
        default=NUM_TAPS,
        help=f"Number of taps per channel (default: {NUM_TAPS})"
    )
    parser.add_argument(
        "--temp_dir",
        type=str,
        default=None,
        help="Temporary directory for extraction (default: system temp)"
    )

    args = parser.parse_args()

    zip_dir = pathlib.Path(args.zip_dir)
    if not zip_dir.exists():
        raise FileNotFoundError(f"ZIP directory not found: {zip_dir}")

    # Create temporary directory for extraction
    if args.temp_dir:
        temp_dir = pathlib.Path(args.temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_cleanup = False
    else:
        temp_dir = pathlib.Path(tempfile.mkdtemp())
        temp_cleanup = True

    print(f"Using temp directory: {temp_dir}")

    try:
        # Extract ZIPs and find all .s4p files
        print("\n=== Step 1: Extracting ZIP files ===")
        s4p_files = extract_zips_and_find_s4p(zip_dir, temp_dir)

        if not s4p_files:
            print("No .s4p files found!")
            return

        # Group files by channel
        print("\n=== Step 2: Grouping .s4p files by channel ===")
        groups = find_and_group_s4p_files(temp_dir)
        print(f"Found {len(groups)} valid channel groups")

        if not groups:
            print("No valid channel groups found!")
            return

        # Process each group
        print("\n=== Step 3: Processing channel groups ===")
        combined_channels = []
        failed_count = 0

        for i, group in enumerate(groups):
            if (i + 1) % 10 == 0:
                print(f"  Processing group {i + 1}/{len(groups)}...")

            try:
                h_combined = process_channel_group(group, args.num_taps)
                if h_combined is not None:
                    combined_channels.append(h_combined)
                else:
                    failed_count += 1
            except Exception as e:
                print(f"    Warning: Failed to process group {i}: {e}")
                failed_count += 1

        print(f"\nSuccessfully processed {len(combined_channels)} channels")
        if failed_count > 0:
            print(f"Failed: {failed_count} channels")

        if not combined_channels:
            print("No valid channels to save!")
            return

        # Convert to PyTorch tensor
        channels_tensor = torch.tensor(
            np.array(combined_channels),
            dtype=torch.float32
        )

        # Summary statistics
        print(f"\n=== Summary ===")
        print(f"Shape: {channels_tensor.shape}")
        print(f"dtype: {channels_tensor.dtype}")
        print(f"Mean: {channels_tensor.mean():.6f}")
        print(f"Std: {channels_tensor.std():.6f}")
        print(f"Min: {channels_tensor.min():.6f}")
        print(f"Max: {channels_tensor.max():.6f}")

        # Save
        torch.save(channels_tensor, args.out_file)
        print(f"\nSaved to {args.out_file}")

    finally:
        # Cleanup temp directory if we created it
        if temp_cleanup and temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
                print(f"\nCleaned up temp directory: {temp_dir}")
            except Exception as e:
                print(f"\nWarning: Could not cleanup temp directory: {e}")


if __name__ == "__main__":
    main()
