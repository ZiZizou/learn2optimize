"""
Centralized Channel Factory for learned optimizer experiments.

Provides a unified interface for selecting between standard and advanced
channel generators via command-line arguments.
"""

import argparse
import torch

from wireline_channel import WirelineChannelGenerator
from advanced_channel_gen import AdvancedWirelineChannelGenerator
from config import CH_TAPS, SNR_RANGE


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
        help="Disable automatic gain control (AGC) normalization. When enabled, channel "
             "impulse responses preserve true insertion loss physics with raw, attenuated "
             "voltages rather than being normalized to unit peak/L2 norm."
    )
    return parser


def get_channel_generator(args, device=None):
    """
    Factory function that returns the appropriate channel generator based on args.

    Args:
        args: Parsed command-line arguments (must contain channel_type attribute).
        device: Optional torch device to pass to the advanced generator.

    Returns:
        WirelineChannelGenerator or AdvancedWirelineChannelGenerator instance.
    """
    disable_agc = getattr(args, 'disable_agc', False)
    if args.channel_type == "advanced":
        return AdvancedWirelineChannelGenerator(
            num_taps=CH_TAPS,
            snr_range=SNR_RANGE,
            device=device,
            disable_agc=disable_agc
        )
    else:
        return WirelineChannelGenerator(
            num_taps=CH_TAPS,
            snr_range=SNR_RANGE,
            disable_agc=disable_agc
        )
