"""
Visualize per-example sync behavior on S4P-derived channels.

Loads a .pt file of channel data and shows, for 4 random channels:
  1. Input TX bitstream
  2. RX after channel convolution (pre-sync)
  3. RX aligned via choose_best_symbol_phase_per_example (post-sync)

Run: python visualize_sync.py --channel_pt <path_to_channel.pt>
"""

import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

from config import OVERSAMPLE_FACTOR, PHASE_SEARCH_MAX_DELAY, PHASE_SEARCH_SYNC_LEN
from oversampling_utils import choose_best_symbol_phase_per_example, upsample_symbols


def load_channels_auto(path: str):
    """Load channel dicts from generate_synthetic_channels.py output.

    File structure (from generate_synthetic_channels.py):
        {
            'meta': {...},
            'channels': [
                {'thru': tensor, 'fext': list|None, 'next': list|None,
                 'physics_valid': bool, 'augmentation_class': str},
                ...
            ]
        }

    Returns list of dicts with keys 'ir' (thru tensor), 'snr' (float).
    """
    raw = torch.load(path, map_location='cpu', weights_only=False)

    if isinstance(raw, dict) and 'channels' in raw:
        channel_list = list(raw['channels'])
        meta = raw.get('meta', {})
    elif isinstance(raw, (list, tuple)):
        channel_list = list(raw)
        meta = {}
    elif isinstance(raw, dict):
        channel_list = [raw]
        meta = {}
    else:
        raise ValueError(f"Unknown channel file format: {type(raw)}")

    snr_default = meta.get('snr_db', 30.0) if isinstance(meta, dict) else 30.0

    out = []
    for i, ch in enumerate(channel_list):
        if 'thru' not in ch:
            raise KeyError(
                f"Channel dict at index {i} missing 'thru' key. "
                f"Available keys: {list(ch.keys())}"
            )
        thru = ch['thru']
        snr = ch.get('snr_db', snr_default)
        out.append({'ir': thru, 'snr': snr})
    return out


def convolve_channel(tx_upsampled: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Convolve tx [seq] with channel IR h [K] -> [seq + K - 1]."""
    tx = tx_upsampled.flatten().to(dtype=torch.float32)
    h_flat = h.flatten().to(dtype=torch.float32)
    n_out = tx.shape[0] + h_flat.shape[0] - 1
    return F.conv1d(
        tx.view(1, 1, -1),
        h_flat.view(1, 1, -1),
        padding=0
    ).flatten()[:n_out]


def main():
    parser = argparse.ArgumentParser(description="Visualize per-example sync on S4P channels")
    parser.add_argument("--channel_pt", type=str, required=True,
                        help="Path to .pt file with channel dictionaries (keys: 'channel_ir', 'snr')")
    parser.add_argument("--n_channels", type=int, default=4,
                        help="Number of distinct channels to visualize (default: 4)")
    parser.add_argument("--seq_len", type=int, default=48,
                        help="Bitstream length in symbols (default: 48)")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed (default: 7)")
    parser.add_argument("--channel_norm_mode", type=str, default="peak",
                        choices=["none", "peak"],
                        help="Channel IR normalization mode: 'none' for raw amplitude (use with normalization=none channels), 'peak' to peak-normalize IRs for display (default: peak)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- Load ---
    print(f"Loading: {args.channel_pt}")
    all_channels = load_channels_auto(args.channel_pt)
    print(f"  -> {len(all_channels)} channel(s) available")

    n_ch = min(args.n_channels, len(all_channels))
    ch_indices = np.random.choice(len(all_channels), n_ch, replace=False).tolist()
    print(f"  -> Showing channels: {ch_indices}")

    # Get oversample factor from the channel file metadata, not config
    raw_meta = torch.load(args.channel_pt, map_location='cpu', weights_only=False)
    channel_meta = raw_meta.get('meta', {}) if isinstance(raw_meta, dict) else {}
    sps = channel_meta.get('samples_per_symbol', OVERSAMPLE_FACTOR)
    norm_mode = args.channel_norm_mode
    print(f"  -> Using samples_per_symbol={sps} (from channel file, config default={OVERSAMPLE_FACTOR})")
    print(f"  -> Channel normalization mode: '{norm_mode}'")

    # Batch size equals number of distinct channels
    B = n_ch
    seq_len = args.seq_len

    # --- TX bitstream (same for all channels, just for visualization) ---
    # Use a repeating pattern so alignment is visually obvious
    base_pattern = torch.tensor([1, -1, 1, 1, -1, -1, 1, -1], dtype=torch.float32)
    tx_symbols = base_pattern.repeat((B, seq_len // 8 + 1))[:, :seq_len]

    print(f"  TX symbols shape={tx_symbols.shape}, values: {tx_symbols[0,:20].tolist()}")

    tx_up = upsample_symbols(tx_symbols, sps)  # [B, T*P]
    T_up = tx_up.shape[1]

    print(f"  tx_up shape={tx_up.shape}, first 20 values: {tx_up[0,:20].tolist()}")

    # --- Build RX: each batch element uses a DIFFERENT channel from the file ---
    # rx_batch[b] uses channel ch_indices[b]
    rx_batch = torch.zeros(B, T_up)  # placeholder, real length varies

    # Auto-gain: If normalization=none, channel IRs have raw small amplitudes.
    # Peak-normalize them for visualization so signals are visible.
    auto_gain = (norm_mode == 'none')

    for b in range(B):
        ch = all_channels[ch_indices[b]]
        h = ch['ir']  # [K]

        if auto_gain:
            peak = h.abs().max()
            if peak > 0:
                h = h / peak

        snr_db = ch['snr']

        h_abs_max = h.abs().max().item()
        h_norm = h.norm().item()
        print(f"  ch{ch_indices[b]}: IR shape={h.shape}, IR abs_max={h_abs_max:.8f}, IR L2={h_norm:.8f}, IR dtype={h.dtype}")
        print(f"  ch{ch_indices[b]}: IR first 10 values: {h[:10].tolist()}")

        raw = convolve_channel(tx_up[b], h)  # [T_up + K - 1]
        if raw.ndim == 0:
            raw = raw.unsqueeze(0)
        raw = raw.to(dtype=torch.float32)

        raw_abs_max = raw.abs().max().item()
        print(f"  ch{ch_indices[b]}: raw conv shape={raw.shape}, abs_max={raw_abs_max:.8f}")

        # Add channel-specific noise
        sig_pow = raw.pow(2).mean()
        noise_std = np.sqrt(sig_pow.item() / (10 ** (snr_db / 10)))
        rx_b = raw + noise_std * torch.randn_like(raw)

        print(f"  ch{ch_indices[b]}: sig_pow={sig_pow.item():.8f}, noise_std={noise_std:.6f}")

        # Truncate/pad to T_up (all same length for batch)
        if rx_b.shape[0] >= T_up:
            rx_batch[b] = rx_b[:T_up]
        else:
            rx_batch[b, :rx_b.shape[0]] = rx_b

        print(f"  ch{ch_indices[b]}: rx_batch[{b}] max={rx_batch[b].abs().max().item():.6f}, rx_batch[{b}, ::sps] max={rx_batch[b, ::sps].abs().max().item():.6f}")

    print(f"  -> RX batch shape: {rx_batch.shape}, min={rx_batch.min().item():.6f}, max={rx_batch.max().item():.6f}")

    # --- Run per-example sync ---
    sync_len = min(PHASE_SEARCH_SYNC_LEN, seq_len)
    best_rx, phase, delay = choose_best_symbol_phase_per_example(
        tx_symbols,
        rx_batch,
        sps,
        max_delay=PHASE_SEARCH_MAX_DELAY,
        sync_len=sync_len,
        use_normalized_corr=True,
    )

    # --- Stats ---
    print(f"\nSync results:")
    print(f"  channels: {ch_indices}")
    print(f"  phase:    {phase.tolist()}")
    print(f"  delay:    {delay.tolist()}")
    print(f"  delay:    min={delay.min().item()}, median={delay.float().median().item():.1f}, max={delay.max().item()}")

    # --- Plot ---
    t_sym = np.arange(seq_len)

    fig, axes = plt.subplots(B, 3, figsize=(15, B * 2.8))
    if B == 1:
        axes = axes.reshape(1, -1)

    for b in range(B):
        ch_label = f"ch={ch_indices[b]}"

        # TX
        axes[b, 0].step(t_sym, tx_symbols[b].numpy(), where='mid', lw=1.5)
        axes[b, 0].set_ylabel(ch_label, fontsize=9)
        axes[b, 0].set_ylim(-1.8, 1.8)
        axes[b, 0].set_xlim(0, seq_len - 1)
        axes[b, 0].grid(True, alpha=0.3)
        axes[b, 0].set_title("TX bitstream", fontsize=10)

        # RX pre-sync: downsample oversampled to symbol rate
        rx_pre = rx_batch[b, ::sps].numpy()
        axes[b, 1].step(t_sym, rx_pre[:seq_len], where='mid', lw=1.0, alpha=0.8)
        axes[b, 1].set_ylim(-1.8, 1.8)
        axes[b, 1].set_xlim(0, seq_len - 1)
        axes[b, 1].grid(True, alpha=0.3)
        axes[b, 1].set_title(
            f"RX pre-sync  (delay={delay[b].item()}, phase={phase[b].item()})",
            fontsize=9
        )

        # RX post-sync
        axes[b, 2].step(t_sym, best_rx[b].numpy(), where='mid', lw=1.0, color='tab:green')
        axes[b, 2].set_ylim(-1.8, 1.8)
        axes[b, 2].set_xlim(0, seq_len - 1)
        axes[b, 2].grid(True, alpha=0.3)
        axes[b, 2].set_title("RX post-sync (aligned)", fontsize=10)

        if b < B - 1:
            for col in range(3):
                axes[b, col].set_xticklabels([])
        else:
            for col in range(3):
                axes[b, col].set_xlabel("Symbol index", fontsize=8)

    fig.suptitle(
        f"Per-Example Sync on {n_ch} S4P Channels\n"
        f"'{args.channel_pt.split('/')[-1]}' | seq={seq_len}, OS={sps}",
        fontsize=11
    )
    plt.tight_layout()
    plt.show()

    # --- MSE summary ---
    print(f"\nAlignment MSE (TX vs RX post-sync, first {seq_len} symbols):")
    for b in range(B):
        mse = ((tx_symbols[b] - best_rx[b, :seq_len]).pow(2).mean().item())
        pre_rms = rx_batch[b, ::sps][:seq_len].pow(2).mean().item()
        print(f"  B{b} ({ch_label}): MSE={mse:.4f}  (pre-sync RMS={pre_rms:.4f})")


if __name__ == "__main__":
    main()
