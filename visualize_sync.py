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
    """Convolve tx [seq] with channel IR h [K] -> [seq + K - 1].

    Uses the same approach as wireline_channel.py and s4p_channel.py:
    - Pad input for causal convolution
    - Flip filter (F.conv1d computes cross-correlation)
    - Use grouped convolution
    """
    tx = tx_upsampled.flatten().float()
    h_flat = h.flatten().float()

    seq_len = tx.shape[0]
    num_taps = h_flat.shape[0]

    # For causal convolution: output_len = seq_len + num_taps - 1
    # Approach from wireline_channel.py: pre-pad input, then conv1d with padding=0
    tx_padded = F.pad(tx, (num_taps - 1, 0))  # Add num_taps-1 zeros to left

    tx_reshaped = tx_padded.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len + num_taps - 1]
    h_reshaped = h_flat.unsqueeze(0).unsqueeze(0)  # [1, 1, num_taps]
    h_flipped = torch.flip(h_reshaped, dims=[-1])

    result = F.conv1d(tx_reshaped, h_flipped, padding=0)

    return result.squeeze()


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

    # Get oversample factor and num_taps from the channel file metadata, not config
    raw_meta = torch.load(args.channel_pt, map_location='cpu', weights_only=False)
    channel_meta = raw_meta.get('meta', {}) if isinstance(raw_meta, dict) else {}

    sps = channel_meta.get('samples_per_symbol', OVERSAMPLE_FACTOR)
    num_taps_meta = channel_meta.get('num_taps', None)
    norm_mode = args.channel_norm_mode

    # Warn if config oversampling disagrees with dataset metadata
    if sps != OVERSAMPLE_FACTOR:
        print(f"  WARNING: config OVERSAMPLE_FACTOR={OVERSAMPLE_FACTOR} but dataset samples_per_symbol={sps}. Using dataset metadata.")

    # Fail loudly if metadata is missing (no silent fallback for num_taps)
    if num_taps_meta is None:
        raise KeyError(
            f"Channel file metadata missing 'num_taps'. "
            f"Available meta keys: {list(channel_meta.keys())}"
        )

    print(f"  Loaded channel metadata: samples_per_symbol={sps}, num_taps={num_taps_meta}, normalization={norm_mode}")

    # Batch size equals number of distinct channels
    B = n_ch
    seq_len = args.seq_len

    # --- TX bitstream (same for all channels, just for visualization) ---
    # Use a repeating pattern so alignment is visually obvious
    base_pattern = torch.tensor([1, -1, 1, 1, -1, -1, 1, -1], dtype=torch.float32)
    tx_symbols = base_pattern.repeat((B, seq_len // 8 + 1))[:, :seq_len]

    print(f"  TX symbols shape={tx_symbols.shape}, values: {tx_symbols[0,:20].tolist()}")

    tx_up = upsample_symbols(tx_symbols, sps)  # [B, T_up]
    T_up = tx_up.shape[1]
    rx_full_len = T_up + num_taps_meta - 1

    print(f"  TX upsampled length: {T_up}")
    print(f"  Expected full RX length: {rx_full_len} (T_up + num_taps - 1 = {T_up} + {num_taps_meta} - 1)")

    # rx_batch must be rx_full_len, NOT T_up
    rx_batch = torch.zeros(B, rx_full_len)  # Full convolution output length

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
        print(f"  ch{ch_indices[b]}: raw conv BEFORE squeeze shape={raw.shape if raw.ndim > 0 else 'scalar'}")
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

        # rx_b length is T_up + num_taps - 1 = rx_full_len
        # Store in rx_batch without truncation (keep full signal for sync)
        rx_batch[b] = rx_b[:rx_full_len]

        print(f"  ch{ch_indices[b]}: rx_batch[{b}] max={rx_batch[b].abs().max().item():.6f}, rx_batch[{b}, ::sps] max={rx_batch[b, ::sps].abs().max().item():.6f}")

    print(f"  -> RX batch shape: {rx_batch.shape}, min={rx_batch.min().item():.6f}, max={rx_batch.max().item():.6f}")

    # Hard assertion: verify RX has full convolution length
    assert rx_batch.shape[1] == rx_full_len, (
        f"ERROR: RX length is {rx_batch.shape[1]} but expected {rx_full_len}. "
        f"Full convolution tail is missing."
    )

    # --- Build phase-aligned pre-sync stream for interpretable comparison ---
    # rx_phase[b, p, :] = rx_batch[b, p::sps] gives symbol-rate stream at phase p
    rx_phase = rx_batch.view(B, sps, -1)  # [B, sps, rx_full_len // sps]
    rx_phase = rx_phase[:, :, :seq_len]    # [B, sps, seq_len]

    # --- Run per-example sync ---
    sync_len = min(PHASE_SEARCH_SYNC_LEN, seq_len)
    best_rx, best_phase, best_delay = choose_best_symbol_phase_per_example(
        tx_symbols,
        rx_batch,
        sps,
        max_delay=PHASE_SEARCH_MAX_DELAY,
        sync_len=sync_len,
        use_normalized_corr=True,
    )

    # --- Stats ---
    # Warn if all examples returned phase=0 and delay=0 (suspicious sync collapse)
    all_zero = (best_phase == 0).all() and (best_delay == 0).all()
    if all_zero:
        print(f"\n  WARNING: all examples returned phase=0 and delay=0. "
              f"This often indicates truncated RX or wrong oversample factor.")

    print(f"\nSync results:")
    for b in range(B):
        ch_id = ch_indices[b]
        mse_post = ((best_rx[b, :seq_len] - tx_symbols[b, :seq_len]) ** 2).mean().item()
        print(f"  ch={ch_id}  phase={best_phase[b].item()}  delay={best_delay[b].item()}  mse={mse_post:.4f}")

    # --- Plot ---
    t_sym = np.arange(seq_len)

    fig, axes = plt.subplots(B, 4, figsize=(20, B * 2.8))
    if B == 1:
        axes = axes.reshape(1, -1)

    for b in range(B):
        ch_label = f"ch={ch_indices[b]}"
        bp = best_phase[b].item()
        bd = best_delay[b].item()

        # TX
        axes[b, 0].step(t_sym, tx_symbols[b].numpy(), where='mid', lw=1.5)
        axes[b, 0].set_ylabel(ch_label, fontsize=9)
        axes[b, 0].set_ylim(-1.8, 1.8)
        axes[b, 0].set_xlim(0, seq_len - 1)
        axes[b, 0].grid(True, alpha=0.3)
        axes[b, 0].set_title("TX bitstream", fontsize=10)

        # RX pre-sync: use best_phase[b] for phase, but start at delay=0
        # This shows the same phase as post-sync but WITHOUT delay correction applied
        # so the viewer can see what delay correction removes
        rx_pre_sync = rx_phase[b, bp, :].numpy()  # [seq_len], phase-selected, no delay shift
        axes[b, 1].step(t_sym, rx_pre_sync, where='mid', lw=1.0, alpha=0.8)
        axes[b, 1].set_ylim(-1.8, 1.8)
        axes[b, 1].set_xlim(0, seq_len - 1)
        axes[b, 1].grid(True, alpha=0.3)
        axes[b, 1].set_title(f"RX pre-sync  (phase={bp}, delay={bd} to fix)", fontsize=9)

        # RX post-sync (already aligned by choose_best_symbol_phase_per_example)
        axes[b, 2].step(t_sym, best_rx[b].numpy(), where='mid', lw=1.0, color='tab:green')
        axes[b, 2].set_ylim(-1.8, 1.8)
        axes[b, 2].set_xlim(0, seq_len - 1)
        axes[b, 2].grid(True, alpha=0.3)
        axes[b, 2].set_title("RX post-sync (aligned)", fontsize=10)

        # Raw oversampled RX (shows channel ISI shape)
        raw_idx = np.arange(rx_full_len)
        axes[b, 3].plot(raw_idx, rx_batch[b].numpy(), lw=0.5, alpha=0.7)
        axes[b, 3].set_xlim(0, min(rx_full_len, 300))
        axes[b, 3].grid(True, alpha=0.3)
        axes[b, 3].set_title("RX oversampled (first 300 samples)", fontsize=9)

        if b < B - 1:
            for col in range(4):
                axes[b, col].set_xticklabels([])
        else:
            for col in range(4):
                axes[b, col].set_xlabel("Symbol index / Sample index", fontsize=8)

    fig.suptitle(
        f"Per-Example Sync on {n_ch} S4P Channels\n"
        f"'{args.channel_pt.split('/')[-1]}' | seq={seq_len}, sps={sps}, num_taps={num_taps_meta}",
        fontsize=11
    )
    plt.tight_layout()
    plt.show()

    # --- MSE summary ---
    K = min(seq_len, 250)
    print(f"\nAlignment quality (first {K} symbols):")
    for b in range(B):
        ch_id = ch_indices[b]
        mse_post = ((best_rx[b, :K] - tx_symbols[b, :K]) ** 2).mean().item()
        rx_pre_dec = rx_phase[b, best_phase[b].item(), :K]
        mse_pre = ((rx_pre_dec - tx_symbols[b, :K]) ** 2).mean().item()
        sign_agree = ((best_rx[b, :K].sign() == tx_symbols[b, :K].sign()).float().mean().item())
        print(f"  ch={ch_id}: pre_mse={mse_pre:.4f}  post_mse={mse_post:.4f}  sign_agree={sign_agree:.3f}")


if __name__ == "__main__":
    main()
