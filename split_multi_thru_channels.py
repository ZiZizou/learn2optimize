#!/usr/bin/env python3
"""
Split directories that contain multiple 'thru' files into separate subdirectories.

For each directory with multiple thru files:
1. Identify unique channels based on the 'victim' identifier in the thru filename
2. Create subdirectories for each unique channel
3. Move the thru file and its associated fext/next files to the appropriate subdirectory

The association is based on pattern matching - e.g., Thru_Host_Tx3_Mod_Tx3 is associated
with FEXT_Host_Tx3_Mod_Tx5 (crosstalk from Tx5 into Tx3's receive).
"""

import os
import re
import shutil
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


def parse_thru_filename(filename: str) -> Optional[Tuple[str, str]]:
    """
    Parse a thru filename to extract the victim identifier.
    Examples:
        Thru_Host_Tx3_Mod_Tx3.s4p -> ('Host_Tx3', 'Thru_Host_Tx3_Mod_Tx3')
        C2M_0p0in_85Ohms_thru1.s4p -> ('C2M_0p0in_85Ohms', None)

    Returns: (victim_id, full_thru_prefix) or None if not a thru file
    """
    if not filename.lower().endswith('.s4p'):
        return None

    name = filename[:-4]  # Remove .s4p extension

    # Pattern 1: Thru_VictimTx_Mod_AggressorTx (e.g., Thru_Host_Tx3_Mod_Tx3)
    match = re.match(r'(Thru_.*?_Tx\d+)_Mod_Tx\d+', name, re.IGNORECASE)
    if match:
        return match.group(1), name

    # Pattern 2: Thru_VictimTxAggressorTx (e.g., Thru_Host_Tx3_Tx3) - less common
    match = re.match(r'(Thru_.*?_Tx\d+)[_-]?Tx\d+', name, re.IGNORECASE)
    if match:
        return match.group(1), name

    # Pattern 3: ChannelName_thru suffix (most common for non-Thru_ prefix files)
    match = re.match(r'(.*?)_thru\d*$', name, re.IGNORECASE)
    if match:
        return match.group(1), None  # No specific thru prefix needed

    # Pattern 4: Just ChannelName_thru (no number)
    match = re.match(r'(.*?)_thru$', name, re.IGNORECASE)
    if match:
        return match.group(1), None

    return None


def parse_fext_next_filename(filename: str) -> Optional[Tuple[str, str]]:
    """
    Parse a fext/next filename to extract the victim identifier.
    Examples:
        FEXT_Host_Tx3_Mod_Tx5.s4p -> ('Host_Tx3', 'FEXT_Host_Tx3_Mod_Tx5')
        NEXT_Host_Tx3_Mod_Tx5.s4p -> ('Host_Tx3', 'NEXT_Host_Tx3_Mod_Tx5')
        C2M_0p0in_85Ohms_xtalk1_Fext.s4p -> ('C2M_0p0in_85Ohms', 'xtalk1_Fext')

    Returns: (victim_id, suffix_identifier) or None if not a fext/next file
    """
    if not filename.lower().endswith('.s4p'):
        return None

    name = filename[:-4]  # Remove .s4p extension

    # Pattern 1: FEXT_VictimTx_Mod_AggressorTx or NEXT_VictimTx_Mod_AggressorTx
    match = re.match(r'(FEXT|NEXT)_(.*?_Tx\d+)_Mod_Tx\d+', name, re.IGNORECASE)
    if match:
        ftype = match.group(1).lower()
        victim_id = match.group(2)
        return victim_id, f"ftype_{victim_id}"  # Will be used for grouping

    # Pattern 2: ChannelName_xtalkN_Fext or ChannelName_xtalkN_Next
    match = re.match(r'(.*?)_xtalk\d+_(Fext|Next)$', name, re.IGNORECASE)
    if match:
        return match.group(1), match.group(2).lower()

    # Pattern 3: ChannelName_FEXTN or ChannelName_NEXTN
    match = re.match(r'(.*?)_(FEXT|NEXT)\d*$', name, re.IGNORECASE)
    if match:
        return match.group(1), match.group(2).lower()

    return None


def get_file_type(filename: str) -> str:
    """Determine file type from filename."""
    name = filename.lower()

    # Check for Thru
    if name.startswith('thru_') or 'thru' in name.split('_')[0].lower():
        return 'thru'

    # Check for FEXT
    if name.startswith('fext_') or '_fext' in name or '_f' in name.split('_')[-1]:
        return 'fext'

    # Check for NEXT
    if name.startswith('next_') or '_next' in name or '_n' in name.split('_')[-1]:
        return 'next'

    return 'unknown'


def find_thru_victim_pattern(filename: str) -> Optional[str]:
    """
    Find the victim pattern in a thru filename for associating with fext/next files.
    E.g., Thru_Host_Tx3_Mod_Tx3 -> Host_Tx3
          C2M_0p0in_85Ohms_thru1 -> C2M_0p0in_85Ohms
    """
    name = filename[:-4]  # Remove extension

    # Thru_VictimTx_Mod_AggressorTx pattern
    match = re.match(r'Thru_(.*?_Tx\d+)_Mod_Tx\d+', name, re.IGNORECASE)
    if match:
        return match.group(1)

    # ChannelName_thru pattern
    match = re.match(r'(.*?)_thru\d*$', name, re.IGNORECASE)
    if match:
        return match.group(1)

    # ChannelName_thru (no number)
    match = re.match(r'(.*?)_thru$', name, re.IGNORECASE)
    if match:
        return match.group(1)

    return None


def find_associated_victim(filename: str) -> Optional[str]:
    """
    Find the victim identifier in a fext/next filename.
    E.g., FEXT_Host_Tx3_Mod_Tx5 -> Host_Tx3
          C2M_0p0in_85Ohms_xtalk1_Fext -> C2M_0p0in_85Ohms
    """
    name = filename[:-4]  # Remove extension

    # FEXT_VictimTx_Mod_AggressorTx or NEXT_VictimTx_Mod_AggressorTx
    match = re.match(r'(FEXT|NEXT)_(.*?_Tx\d+)_Mod_Tx\d+', name, re.IGNORECASE)
    if match:
        return match.group(2)

    # ChannelName_xtalkN_Fext or ChannelName_xtalkN_Next
    match = re.match(r'(.*?)_xtalk\d+_(Fext|Next)$', name, re.IGNORECASE)
    if match:
        return match.group(1)

    # ChannelName_FEXTN or ChannelName_NEXTN
    match = re.match(r'(.*?)_(FEXT|NEXT)\d*$', name, re.IGNORECASE)
    if match:
        return match.group(1)

    return None


def process_directory(dir_path: Path) -> Tuple[int, int]:
    """
    Process a directory with potentially multiple thru files.
    Returns: (num_subdirs_created, num_files_moved)
    """
    files = list(dir_path.iterdir())
    s4p_files = [f for f in files if f.suffix.lower() == '.s4p']

    if not s4p_files:
        return 0, 0

    # Find all thru files
    thru_files = []
    other_files = []  # fext/next files

    for f in s4p_files:
        fname = f.name
        ftype = get_file_type(fname)

        if ftype == 'thru':
            thru_files.append(f)
        elif ftype in ['fext', 'next']:
            other_files.append(f)

    # If only one thru file or no thru files, no splitting needed
    if len(thru_files) <= 1:
        return 0, 0

    # Group files by victim identifier
    groups: Dict[str, List[Path]] = defaultdict(list)

    for thru in thru_files:
        victim = find_thru_victim_pattern(thru.name)
        if victim:
            groups[victim].append(thru)

            # Find associated fext/next files
            for other in other_files:
                other_victim = find_associated_victim(other.name)
                if other_victim == victim:
                    groups[victim].append(other)

    # If no valid groupings found, return
    if not groups:
        print(f"  Warning: Could not determine groupings for {dir_path.name}")
        return 0, 0

    # Create subdirectories and move files
    subdirs_created = 0
    files_moved = 0

    for victim, files_in_group in groups.items():
        # Create subdirectory name
        subdir_name = f"{dir_path.name}_{victim}"
        subdir_path = dir_path / subdir_name

        if subdir_path.exists():
            print(f"  Warning: Subdirectory already exists: {subdir_name}")
            # Move files anyway, handling collisions
            for f in files_in_group:
                dst = subdir_path / f.name
                counter = 1
                while dst.exists():
                    dst = subdir_path / f"{f.stem}_{counter}{f.suffix}"
                    counter += 1
                shutil.move(str(f), str(dst))
                files_moved += 1
        else:
            subdir_path.mkdir(parents=True, exist_ok=True)
            subdirs_created += 1

            for f in files_in_group:
                dst = subdir_path / f.name
                shutil.move(str(f), str(dst))
                files_moved += 1

    return subdirs_created, files_moved


def process_all_directories(base_path: Path):
    """Process all directories in the base path."""
    dirs = [d for d in base_path.iterdir() if d.is_dir()]

    total_subdirs = 0
    total_files_moved = 0
    dirs_processed = 0
    dirs_modified = 0

    for dir_path in sorted(dirs):
        subdirs, files_moved = process_directory(dir_path)

        if subdirs > 0 or files_moved > 0:
            dirs_modified += 1
            total_subdirs += subdirs
            total_files_moved += files_moved
            print(f"  Modified: created {subdirs} subdirs, moved {files_moved} files")

        dirs_processed += 1

    print(f"\n{'='*60}")
    print(f"Processed {dirs_processed} directories")
    print(f"Modified {dirs_modified} directories")
    print(f"Created {total_subdirs} new subdirectories")
    print(f"Moved {total_files_moved} files")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Split directories with multiple thru files into subdirectories'
    )
    parser.add_argument(
        '--path',
        type=str,
        default='C:/Users/Atharva/Documents/school_stuff/winter_2026/ECE_675D/learned_optimizer/wireline_s4p_models/preprocessed_802_3ck',
        help='Path to preprocessed channels directory (file or directory)'
    )

    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"Error: Path does not exist: {path}")
        return

    if path.is_file():
        print(f"Error: Path must be a directory: {path}")
        return

    # Check if it's a channel directory (contains s4p files) or a parent directory
    s4p_files = list(path.glob('*.s4p'))

    if s4p_files:
        # It's a channel directory - process it directly
        print(f"Processing single directory: {path}")
        subdirs, files_moved = process_directory(path)
        print(f"Created {subdirs} subdirectories, moved {files_moved} files")
    else:
        # It's a parent directory - process all subdirectories
        print(f"Processing directories in: {path}")
        process_all_directories(path)


if __name__ == '__main__':
    main()
