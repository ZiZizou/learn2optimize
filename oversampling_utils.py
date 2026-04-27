import torch
import torch.nn.functional as F


def upsample_symbols(
    tx_symbols: torch.Tensor,
    oversample_factor: int,
    mode: str = "zoh",
) -> torch.Tensor:
    if oversample_factor == 1:
        return tx_symbols
    if mode != "zoh":
        raise NotImplementedError(
            f"Unsupported oversampling mode: {mode}"
        )
    return tx_symbols.repeat_interleave(oversample_factor, dim=1)


@torch.no_grad()
def choose_best_symbol_phase(
    tx_symbols: torch.Tensor,
    rx_oversampled: torch.Tensor,
    oversample_factor: int,
    max_delay: int = 64,
    sync_len: int = 128,
    use_normalized_corr: bool = False,
):
    """
    Legacy batch-global sync. Do not use for S4P-derived heterogeneous channels.
    Kept only for backward compatibility / ablation reproduction.

    Find best phase and integer delay for symbol synchronization.

    Args:
        tx_symbols: [batch, seq_len] transmitted symbols
        rx_oversampled: [batch, seq_len * oversample_factor] received signal
        oversample_factor: samples per symbol
        max_delay: maximum search delay in symbols
        sync_len: number of symbols to use for sync correlation
        use_normalized_corr: if True, use rho(d) = |⟨x,y_d⟩|/(|x||y_d|+eps)
                           for amplitude-invariant sync (recommended for no-AGC)

    Returns:
        best_rx: [batch, seq_len] phase-aligned received signal
        best_phase: integer phase offset
        best_delay: integer delay in symbols
    """
    if oversample_factor == 1:
        return rx_oversampled, 0, 0

    sync_len = min(
        sync_len,
        tx_symbols.shape[1],
        rx_oversampled.shape[1] // oversample_factor,
    )
    tx_ref = tx_symbols[:, :sync_len]

    best_phase = 0
    best_delay = 0
    best_score = None
    best_rx = rx_oversampled[:, ::oversample_factor]

    for phase in range(oversample_factor):
        rx_phase = rx_oversampled[:, phase::oversample_factor]
        if rx_phase.shape[1] <= sync_len:
            continue

        phase_max_delay = min(max_delay, rx_phase.shape[1] - sync_len)
        if phase_max_delay <= 0:
            continue

        windows = rx_phase.unfold(1, sync_len, 1)[:, :phase_max_delay, :]

        if use_normalized_corr:
            tx_norm_sq = tx_ref.pow(2).sum(dim=-1, keepdim=True)
            win_norm_sq = windows.pow(2).sum(dim=-1)
            corr_denom = torch.sqrt(tx_norm_sq * win_norm_sq + 1e-6)
            corr = (windows * tx_ref[:, None, :]).sum(dim=-1).abs() / corr_denom
        else:
            corr = (windows * tx_ref[:, None, :]).sum(dim=-1).abs()

        delays = corr.argmax(dim=1)
        scores = corr.gather(1, delays[:, None]).squeeze(1)

        if use_normalized_corr:
            score = scores.mean().item()
        else:
            score = scores.mean().item()

        if best_score is None or score > best_score:
            best_score = score
            best_phase = phase
            best_delay = int(delays.float().median().item())
            best_rx = rx_phase

    return best_rx, best_phase, best_delay


@torch.no_grad()
def choose_best_symbol_phase_per_example(
    tx_symbols: torch.Tensor,
    rx_oversampled: torch.Tensor,
    oversample_factor: int,
    max_delay: int = 64,
    sync_len: int = 128,
    use_normalized_corr: bool = True,
):
    """
    Per-example normalized-correlation sync for amplitude-invariant delay selection.

    Uses rho(d) = |⟨x,y_d⟩|/(|x||y_d||+eps) for each example in the batch.
    This removes amplitude bias so strong channels don't dominate sync selection.

    This is a wrapper that calls choose_best_symbol_phase_per_example_vectorized.

    Args:
        tx_symbols: [batch, seq_len] transmitted symbols
        rx_oversampled: [batch, seq_len * oversample_factor] received signal
        oversample_factor: samples per symbol
        max_delay: maximum search delay in symbols
        sync_len: number of symbols to use for sync correlation
        use_normalized_corr: if True, use normalized correlation (default, recommended)

    Returns:
        best_rx: [batch, seq_len] phase-aligned received signal per example
        best_phase_per_example: [batch] integer phase per example
        best_delay_per_example: [batch] integer delay per example

    IMPORTANT: The returned best_rx is already synchronized. Downstream code must
    index it as best_rx[:, t:t+1], NOT best_rx[:, t + best_delay].
    """
    return choose_best_symbol_phase_per_example_vectorized(
        tx_symbols, rx_oversampled, oversample_factor, max_delay, sync_len,
        use_normalized_corr=use_normalized_corr,
    )


@torch.no_grad()
def choose_best_symbol_phase_per_example_vectorized(
    tx_symbols: torch.Tensor,
    rx_oversampled: torch.Tensor,
    oversample_factor: int,
    max_delay: int = 64,
    sync_len: int = 128,
    use_normalized_corr: bool = True,
):
    """
    Vectorized per-example normalized-correlation sync for amplitude-invariant delay selection.

    Uses rho(d) = |<x,y_d>| / (||x|| ||y_d|| + eps) for each example in the batch.
    Uses joint (phase, delay) argmax: (p*, d*) = argmax_{p,d} rho_{b,p,d}

    Args:
        tx_symbols: [batch, seq_len] transmitted symbols
        rx_oversampled: [batch, seq_len * oversample_factor] received signal
        oversample_factor: samples per symbol
        max_delay: maximum search delay in symbols
        sync_len: number of symbols to use for sync correlation
        use_normalized_corr: if True, use normalized correlation (default, recommended)

    Returns:
        best_rx: [batch, seq_len] phase-aligned received signal per example
        best_phase: [batch] integer phase per example
        best_delay: [batch] integer delay per example

    IMPORTANT: The returned best_rx is already synchronized. Downstream code must
    index it as best_rx[:, t:t+1], NOT best_rx[:, t + best_delay].
    """
    if oversample_factor == 1:
        return rx_oversampled, torch.zeros(tx_symbols.shape[0], dtype=torch.long, device=tx_symbols.device), \
               torch.zeros(tx_symbols.shape[0], dtype=torch.long, device=tx_symbols.device)

    B, T_rx = rx_oversampled.shape
    P = oversample_factor

    sync_len_eff = min(sync_len, tx_symbols.shape[1], T_rx // P)
    tx_ref = tx_symbols[:, :sync_len_eff]

    t_pad = ((T_rx + P - 1) // P) * P
    rx_pad = F.pad(rx_oversampled, (0, t_pad - T_rx))
    rx_phase = rx_pad.view(B, -1, P).transpose(1, 2).contiguous()

    phase_ids = torch.arange(P, device=rx_oversampled.device)
    phase_lens = ((T_rx - phase_ids + P - 1) // P).clamp_min(0)

    search_len_p = phase_lens - sync_len_eff + 1
    valid_sync_counts = search_len_p.clamp_min(0)
    D_search = min(max_delay, int(valid_sync_counts.max().item()))

    if D_search <= 0:
        raise ValueError(f"No valid sync windows: max_delay={max_delay}, phase_lens={phase_lens.tolist()}")

    delay_ids = torch.arange(D_search, device=rx_oversampled.device)
    valid_mask = delay_ids.view(1, 1, D_search) < valid_sync_counts.view(1, P, 1)

    windows = rx_phase.unfold(dimension=-1, size=sync_len_eff, step=1)[:, :, :D_search, :]

    tx_norm = tx_ref.pow(2).sum(dim=-1).sqrt().view(B, 1, 1)
    corr = (windows * tx_ref[:, None, None, :]).sum(dim=-1).abs()

    if use_normalized_corr:
        win_norm = windows.pow(2).sum(dim=-1).sqrt()
        rho = corr / (tx_norm * win_norm + 1e-6)
    else:
        rho = corr

    neg_inf = torch.finfo(rho.dtype).min
    rho = rho.masked_fill(~valid_mask, neg_inf)

    rho_flat = rho.view(B, -1)
    best_flat = rho_flat.argmax(dim=1)
    best_phase = best_flat // D_search
    best_delay = best_flat % D_search

    seq_len = tx_symbols.shape[1]
    valid_out_counts = (phase_lens - seq_len + 1).clamp_min(1)
    max_out_start = valid_out_counts - 1
    best_delay_clamped = torch.minimum(best_delay, max_out_start[best_phase])

    b_idx = torch.arange(B, device=rx_oversampled.device)
    best_phase_stream = rx_phase[b_idx, best_phase, :]

    sample_ids = torch.arange(seq_len, device=rx_oversampled.device)
    gather_idx = best_delay_clamped[:, None] + sample_ids[None, :]
    best_rx = best_phase_stream.gather(1, gather_idx)

    assert best_phase.dtype == torch.long, f"best_phase dtype {best_phase.dtype} expected torch.long"
    assert best_delay.dtype == torch.long, f"best_delay dtype {best_delay.dtype} expected torch.long"
    assert best_rx.ndim == 2, f"best_rx ndim {best_rx.ndim} expected 2"
    assert best_rx.shape[0] == tx_symbols.shape[0], f"best_rx batch {best_rx.shape[0]} expected {tx_symbols.shape[0]}"

    return best_rx, best_phase.long(), best_delay_clamped.long()
