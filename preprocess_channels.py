#!/usr/bin/env python3
"""
Preprocess channel s4p files from zip archives.

This script:
1. Walks through all zip files in the source directory
2. Extracts s4p files (recursively handling nested zips)
3. Groups files by unique channel prefix
4. Organizes each channel's thru/fext/next files into its own directory

A unique channel is identified by a common prefix shared across thru, fext, and next files.
Directory names include the source zip name to ensure uniqueness.
"""

import os
import re
import zipfile
import shutil
import hashlib
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set

# Windows MAX_PATH limitation
MAX_PATH = 260


# Patterns to identify file types and extract channel prefix
# Order matters - more specific patterns first

# Suffix patterns: ChannelName_thru/fext/next variants
SUFFIX_PATTERNS = [
    (re.compile(r'_xtalk\d+_FEXT$', re.IGNORECASE), 'fext'),
    (re.compile(r'_xtalk\d+_NEXT$', re.IGNORECASE), 'next'),
    (re.compile(r'_thru\d*$', re.IGNORECASE), 'thru'),
    (re.compile(r'_fext\d*$', re.IGNORECASE), 'fext'),
    (re.compile(r'_next\d*$', re.IGNORECASE), 'next'),
    (re.compile(r'_FEXT\d*$', re.IGNORECASE), 'fext'),
    (re.compile(r'_NEXT\d*$', re.IGNORECASE), 'next'),
    (re.compile(r'_THRU$', re.IGNORECASE), 'thru'),
    (re.compile(r'_f\d+$', re.IGNORECASE), 'fext'),
    (re.compile(r'_n\d+$', re.IGNORECASE), 'next'),
    (re.compile(r'_t$', re.IGNORECASE), 'thru'),
    (re.compile(r'_FEXT\d*_', re.IGNORECASE), 'fext'),
    (re.compile(r'_NEXT\d*_', re.IGNORECASE), 'next'),
    (re.compile(r'_THRU_', re.IGNORECASE), 'thru'),
]

# Prefix patterns: Type_ChannelName
PREFIX_PATTERNS = [
    (re.compile(r'^Thru_', re.IGNORECASE), 'thru'),
    (re.compile(r'^FEXT_', re.IGNORECASE), 'fext'),
    (re.compile(r'^NEXT_', re.IGNORECASE), 'next'),
    (re.compile(r'^fext', re.IGNORECASE), 'fext'),
    (re.compile(r'^next', re.IGNORECASE), 'next'),
    (re.compile(r'^thru', re.IGNORECASE), 'thru'),
]


def extract_channel_prefix(filepath: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract channel prefix and file type from a file path."""
    filename = os.path.basename(filepath)

    if not filename.lower().endswith('.s4p'):
        return None, None

    # Check if prefix-type file (Thru_, FEXT_, NEXT_, etc.)
    is_prefix_type = False
    detected_ftype = None
    for pattern, ftype in PREFIX_PATTERNS:
        match = pattern.search(filename)
        if match:
            is_prefix_type = True
            detected_ftype = ftype
            break

    if is_prefix_type:
        return None, detected_ftype

    # Try suffix patterns
    name_without_ext = re.sub(r'\.s4p$', '', filename, flags=re.IGNORECASE)

    for pattern, ftype in SUFFIX_PATTERNS:
        match = pattern.search(name_without_ext)
        if match:
            channel = name_without_ext[:match.start()]
            dir_channel, num_meaningful = get_channel_from_directory(filepath)
            if dir_channel and num_meaningful > 1:
                return dir_channel, ftype
            return channel, ftype

    return None, None


def get_channel_from_directory(filepath: str) -> Tuple[Optional[str], int]:
    """Extract channel name from the directory structure within a zip."""
    dirname = os.path.dirname(filepath)
    if not dirname:
        return None, 0

    # Handle both Unix and Windows path separators (zips use /)
    parts = dirname.replace('\\', '/').split('/')
    parts = [p for p in parts if p]

    meaningful_parts = []
    skip_list = ['src', 'data', 'files', 'channel', 'ch']
    for part in parts:
        if len(part) > 3 and not part.lower() in skip_list:
            meaningful_parts.append(part)

    if not meaningful_parts:
        return None, 0

    if len(meaningful_parts) > 1:
        return '_'.join(meaningful_parts), len(meaningful_parts)

    return meaningful_parts[0], 1


def shorten_channel_name(channel_name: str, max_len: int = 100) -> str:
    """
    Shorten channel name if it would cause path to exceed MAX_PATH.
    Uses truncation with hash suffix to ensure uniqueness.
    """
    if len(channel_name) <= max_len:
        return channel_name

    # Truncate and add hash suffix to ensure uniqueness
    hash_suffix = hashlib.md5(channel_name.encode()).hexdigest()[:8]
    shortened = channel_name[:max_len - 9] + "_" + hash_suffix
    return shortened


def get_safe_channel_name(channel_name: str, filename: str, output_base: str) -> str:
    """
    Get a safe channel name that won't cause path length issues.
    """
    # Estimate the full path length
    estimated_path = os.path.join(output_base, channel_name, filename)

    if len(estimated_path) < MAX_PATH - 10:
        return channel_name

    # Shorten the channel name
    # Reserve space for filename and some buffer
    max_channel_len = MAX_PATH - len(filename) - len(output_base) - 50
    if max_channel_len < 20:
        max_channel_len = 20  # Minimum sensible channel name length

    return shorten_channel_name(channel_name, max_channel_len)


def process_zip_file_simple(zip_path: str, output_path: Path) -> Tuple[int, int]:
    """Process a single zip file, reading directly from zip without temp extraction."""
    zip_name = os.path.splitext(os.path.basename(zip_path))[0]
    channels: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    with zipfile.ZipFile(zip_path, 'r') as zf:
        for member in zf.namelist():
            if not member.lower().endswith('.s4p'):
                continue

            channel, ftype = extract_channel_prefix(member)
            if channel is None:
                channel = get_channel_from_directory(member)[0]

            if channel is not None and ftype is not None:
                full_channel = f"{zip_name}_{channel}"
                channels[full_channel].append((member, os.path.basename(member)))

    # Create directories and write files
    files_copied = 0
    for full_channel, files in channels.items():
        # Get a safe channel name that won't exceed MAX_PATH
        sample_filename = files[0][1] if files else "file.s4p"
        safe_channel = get_safe_channel_name(full_channel, sample_filename, str(output_path))

        channel_dir = output_path / safe_channel
        channel_dir.mkdir(parents=True, exist_ok=True)

        for member, filename in files:
            dst_file = channel_dir / filename

            # Handle collisions
            if dst_file.exists():
                base, ext = os.path.splitext(filename)
                counter = 1
                while dst_file.exists():
                    dst_file = channel_dir / f"{base}_{counter}{ext}"
                    counter += 1

            # Read from zip and write directly
            try:
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    data = zf.read(member)
                with open(dst_file, 'wb') as f:
                    f.write(data)
                files_copied += 1
            except Exception as e:
                print(f"Warning: Failed to copy {member}: {e}")

    return len(channels), files_copied


def process_nested_zip_file(zip_path: str, output_path: Path) -> Tuple[int, int]:
    """Process a zip file that contains nested zip files."""
    zip_name = os.path.splitext(os.path.basename(zip_path))[0]
    all_channels: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    total_files_copied = 0

    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for member in zf.namelist():
                if member.lower().endswith('.zip'):
                    # Nested zip - extract and process
                    nested_zip_path = os.path.join(temp_dir, os.path.basename(member))
                    zf.extract(member, temp_dir)

                    nested_channels, nested_files = process_zip_file_simple(nested_zip_path, output_path)
                    total_files_copied += nested_files

                elif member.lower().endswith('.s4p'):
                    channel, ftype = extract_channel_prefix(member)
                    if channel is None:
                        channel = get_channel_from_directory(member)[0]

                    if channel is not None and ftype is not None:
                        full_channel = f"{zip_name}_{channel}"
                        all_channels[full_channel].append((member, os.path.basename(member)))

    # Copy direct s4p files
    for full_channel, files in all_channels.items():
        sample_filename = files[0][1] if files else "file.s4p"
        safe_channel = get_safe_channel_name(full_channel, sample_filename, str(output_path))

        channel_dir = output_path / safe_channel
        channel_dir.mkdir(parents=True, exist_ok=True)

        for member, filename in files:
            dst_file = channel_dir / filename

            if dst_file.exists():
                base, ext = os.path.splitext(filename)
                counter = 1
                while dst_file.exists():
                    dst_file = channel_dir / f"{base}_{counter}{ext}"
                    counter += 1

            try:
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    data = zf.read(member)
                with open(dst_file, 'wb') as f:
                    f.write(data)
                total_files_copied += 1
            except Exception as e:
                print(f"Warning: Failed to copy {member}: {e}")

    return len(all_channels), total_files_copied


def preprocess_channels(source_dir: str, output_dir: str):
    """Main function to preprocess all channel files."""
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if source_path.is_file() and source_path.suffix.lower() == '.zip':
        zip_files = [source_path]
    else:
        zip_files = sorted(source_path.glob('*.zip'))

    print(f"Found {len(zip_files)} zip files to process")

    total_channels = 0
    total_files = 0

    for zip_file in zip_files:
        print(f"\nProcessing: {zip_file.name}")

        with zipfile.ZipFile(zip_file, 'r') as zf:
            namelist = zf.namelist()
            has_nested_zips = any(n.lower().endswith('.zip') for n in namelist)

        if has_nested_zips:
            print(f"  Found nested zip files, processing recursively...")
            channels, files = process_nested_zip_file(str(zip_file), output_path)
        else:
            channels, files = process_zip_file_simple(str(zip_file), output_path)

        print(f"  Found {channels} unique channels, copied {files} files")
        total_channels += channels
        total_files += files

    print(f"\n{'='*60}")
    print(f"Total: {total_channels} unique channels, {total_files} files copied")
    print(f"Output written to: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Preprocess channel s4p files from zip archives'
    )
    parser.add_argument(
        '--source',
        type=str,
        default='C:/Users/Atharva/Documents/school_stuff/winter_2026/ECE_675D/learned_optimizer/wireline_s4p_models/802_3ck',
        help='Source directory containing zip files'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='C:/Users/Atharva/Documents/school_stuff/winter_2026/ECE_675D/learned_optimizer/wireline_s4p_models/preprocessed_802_3ck',
        help='Output directory for preprocessed channels'
    )

    args = parser.parse_args()
    preprocess_channels(args.source, args.output)


if __name__ == '__main__':
    main()
