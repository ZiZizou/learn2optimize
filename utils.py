"""
Centralized Channel Factory for learned optimizer experiments.

Provides a unified interface for selecting between standard and advanced
channel generators via command-line arguments.
"""

import argparse
import os
import torch

from wireline_channel import WirelineChannelGenerator
from advanced_channel_gen import AdvancedWirelineChannelGenerator
from s4p_channel import S4pChannelGenerator
from config import BAUD_RATE_HZ, CH_TAPS, OVERSAMPLE_FACTOR, S4P_DATASET_DIR, S4P_DATASET_TEMPLATE, SNR_RANGE


def resolve_s4p_dataset_path():
    filename = S4P_DATASET_TEMPLATE.format(
        sps=OVERSAMPLE_FACTOR,
        baud=int(BAUD_RATE_HZ),
    )
    return os.path.join(S4P_DATASET_DIR, filename)


def add_channel_args(parser: argparse.ArgumentParser):
    """
    Add channel type selection arguments to an argument parser.

    Args:
        parser: An existing argparse.ArgumentParser to extend.

    Returns:
        The extended parser.
    """
    parser.add_argument(
        "--channel_type",
        type=str,
        choices=["standard", "advanced"],
        default="standard",
        help="Select the channel generation model: 'standard' for WirelineChannelGenerator, "
             "'advanced' for AdvancedWirelineChannelGenerator with resonant ringing, "
             "PA saturation, and dynamic drift."
    )
    parser.add_argument(
        "--disable_agc",
        action="store_true",
        help="DEPRECATED: Use --channel_ir_norm_mode=none instead. "
             "Disables channel IR normalization to preserve true insertion loss "
             "physics with raw, attenuated voltages rather than being normalized "
             "to unit peak/L2 norm. This is NOT receiver AGC - it is pre-convolution "
             "IR scaling."
    )
    parser.add_argument(
        "--channel_ir_norm_mode",
        type=str,
        choices=["none", "peak", "l2"],
        default=None,
        help="Channel impulse response normalization mode: 'none' preserves raw "
             "amplitude (for no-AGC training), 'peak' normalizes peak to 1, "
             "'l2' normalizes L2 norm to 1. Default: 'peak' for backward compat."
    )
    parser.add_argument(
        "--touchstone_channel",
        type=str,
        default=None,
        help="Path to the .pt file containing processed S4P channel dictionaries. "
             "If provided, overrides standard synthetic channels with realistic S-parameter "
             "derived channel data including FEXT/NEXT crosstalk."
    )
    return parser


def get_channel_generator(args, device=None, samples_per_symbol=1):
    """
    Factory function that returns the appropriate channel generator based on args.

    Args:
        args: Parsed command-line arguments.
        device: Optional torch device to pass to the advanced generator.
        samples_per_symbol: Number of samples per symbol for channel time grid.

    Returns:
        WirelineChannelGenerator, AdvancedWirelineChannelGenerator, or
        S4pChannelGenerator instance.
    """
    disable_agc = getattr(args, 'disable_agc', False)
    norm_mode = getattr(args, 'channel_ir_norm_mode', None)

    if disable_agc and norm_mode is not None and norm_mode != "none":
        raise ValueError(
            "Conflicting arguments: --disable_agc and --channel_ir_norm_mode "
            "cannot both be set. Use --channel_ir_norm_mode=none instead of --disable_agc."
        )

    if disable_agc and norm_mode is None:
        import warnings
        warnings.warn(
            "DEPRECATED: --disable_agc is deprecated. "
            "Use --channel_ir_norm_mode=none instead. "
            "This is NOT receiver AGC - it is pre-convolution IR scaling.",
            DeprecationWarning,
            stacklevel=2
        )
        norm_mode = "none"

    if norm_mode is None:
        norm_mode = "peak"  # Legacy default

    is_raw_mode = (norm_mode == "none")

    # S4P Touchstone channel overrides other channel types
    if getattr(args, 'touchstone_channel', None) is not None:
        dataset_path = args.touchstone_channel
        print(f"[s4p] loading dataset: {dataset_path}")
        return S4pChannelGenerator(
            touchstone_file_path=dataset_path,
            snr_range=SNR_RANGE,
            disable_agc=is_raw_mode,
            samples_per_symbol=samples_per_symbol,
        )

    if args.channel_type == "advanced":
        return AdvancedWirelineChannelGenerator(
            num_taps=CH_TAPS,
            snr_range=SNR_RANGE,
            device=device,
            disable_agc=is_raw_mode,
            samples_per_symbol=samples_per_symbol
        )
    else:
        return WirelineChannelGenerator(
            num_taps=CH_TAPS,
            snr_range=SNR_RANGE,
            disable_agc=is_raw_mode,
            samples_per_symbol=samples_per_symbol
        )
