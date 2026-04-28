"""
Unified synchronization backend for learned optimizer experiments.

This module provides a single, well-instrumented sync API that replaces
all fragmented sync implementations across the codebase.

Design principles:
- One algorithm for both symbol-rate and oversampled cases
- Always per-example, never batch-global
- Normalized correlation by default (amplitude invariant)
- Joint phase-delay search for oversampled waveforms
- Explicit polarity tracking
- Confidence diagnostics and boundary-hit detection
"""

from dataclasses import dataclass, field
from typing import Literal, Optional
import torch
import torch.nn.functional as F
import warnings


@dataclass
class SyncConfig:
    oversample_factor: int
    max_delay_symbols: int
    sync_len_symbols: int
    metric: Literal["normalized"] = "normalized"
    polarity_mode: Literal["report", "flip", "error"] = "report"
    return_debug_tensors: bool = False
    eps: float = 1e-6
    ambiguous_score_thresh: float = 0.2
    ambiguous_margin_thresh: float = 0.02


@dataclass
class SyncResult:
    rx_aligned: torch.Tensor
    best_phase: torch.Tensor
    best_delay_symbols: torch.Tensor
    best_total_sample_offset: torch.Tensor
    best_score: torch.Tensor
    best_signed_corr: torch.Tensor
    second_best_score: torch.Tensor
    margin: torch.Tensor
    polarity: torch.Tensor
    hit_delay_boundary: torch.Tensor
    hit_phase_boundary: torch.Tensor
    ambiguous: torch.Tensor
    effective_sync_len: int

    score_grid: Optional[torch.Tensor] = None
    valid_mask: Optional[torch.Tensor] = None


def summarize_sync(result: SyncResult) -> dict:
    return {
        "score_mean": result.best_score.mean().item(),
        "score_min": result.best_score.min().item(),
        "score_max": result.best_score.max().item(),
        "margin_mean": result.margin.mean().item(),
        "margin_min": result.margin.min().item(),
        "delay_boundary_frac": result.hit_delay_boundary.float().mean().item(),
        "phase_boundary_frac": result.hit_phase_boundary.float().mean().item(),
        "negative_polarity_frac": (result.polarity < 0).float().mean().item(),
        "ambiguous_frac": result.ambiguous.float().mean().item(),
        "effective_sync_len": result.effective_sync_len,
    }


def align_rx_to_tx(
    tx_symbols: torch.Tensor,
    rx_signal: torch.Tensor,
    seq_len: int,
    cfg: SyncConfig,
) -> SyncResult:
    """
    Per-example synchronized alignment of RX waveform to TX symbols.

    Uses joint (phase, delay) argmax with normalized correlation:
        (p*, d*) = argmax_{p,d} rho_{b,p,d}
    where rho = |⟨x, y_{b,p,d}⟩| / (||x|| ||y_{b,p,d}|| + eps)

    Args:
        tx_symbols: [B, T] transmitted symbols
        rx_signal: [B, T_rx] received signal (oversampled if P > 1)
        seq_len: number of output symbols after alignment
        cfg: SyncConfig with oversample_factor, max_delay_symbols, etc.

    Returns:
        SyncResult with aligned waveform and diagnostics
    """
    B, T_tx = tx_symbols.shape
    P = cfg.oversample_factor
    T_rx = rx_signal.shape[1]

    if P == 1:
        return _align_symbol_rate(tx_symbols, rx_signal, seq_len, cfg)

    return _align_oversampled(tx_symbols, rx_signal, seq_len, cfg)


def _align_symbol_rate(
    tx_symbols: torch.Tensor,
    rx_signal: torch.Tensor,
    seq_len: int,
    cfg: SyncConfig,
) -> SyncResult:
    B, T_tx = tx_symbols.shape
    T_rx = rx_signal.shape[1]

    sync_len_eff = min(cfg.sync_len_symbols, T_tx, T_rx)
    tx_ref = tx_symbols[:, :sync_len_eff]

    D_search = min(cfg.max_delay_symbols, T_rx - sync_len_eff)
    if D_search <= 0:
        raise ValueError(
            f"No valid sync windows: max_delay={cfg.max_delay_symbols}, "
            f"rx_len={T_rx}, sync_len={sync_len_eff}"
        )

    rx_windows = rx_signal.unfold(dimension=-1, size=sync_len_eff, step=1)[:, :D_search, :]

    tx_norm = tx_ref.pow(2).sum(dim=-1).sqrt().view(B, 1)
    corr_signed = (rx_windows * tx_ref[:, None, :]).sum(dim=-1)
    corr_abs = corr_signed.abs()

    if cfg.metric == "normalized":
        win_norm = rx_windows.pow(2).sum(dim=-1).sqrt()
        rho = corr_abs / (tx_norm * win_norm + cfg.eps)
    else:
        rho = corr_abs

    valid_mask = torch.ones(B, D_search, dtype=torch.bool, device=rx_signal.device)
    rho = rho.masked_fill(~valid_mask, float("-inf"))

    best_flat = rho.argmax(dim=1)
    best_delay = best_flat
    best_phase = torch.zeros(B, dtype=torch.long, device=rx_signal.device)

    second_best, _ = rho.topk(k=2, dim=1)
    second_best_score = second_best[:, 1]
    margin = rho.max(dim=1).values - second_best_score

    best_score = rho.max(dim=1).values
    best_signed_corr = corr_signed.gather(1, best_delay.unsqueeze(1)).squeeze(1)
    polarity = torch.sign(best_signed_corr)
    polarity = torch.where(polarity == 0, torch.ones_like(polarity), polarity)

    hit_delay_boundary = (best_delay == 0) | (best_delay == D_search - 1)

    rx_aligned = rx_signal[:, best_delay:best_delay + seq_len]

    ambiguous = (best_score < cfg.ambiguous_score_thresh) | (
        margin < cfg.ambiguous_margin_thresh
    )

    best_total_sample_offset = best_delay

    return SyncResult(
        rx_aligned=rx_aligned,
        best_phase=best_phase,
        best_delay_symbols=best_delay,
        best_total_sample_offset=best_total_sample_offset,
        best_score=best_score,
        best_signed_corr=best_signed_corr,
        second_best_score=second_best_score,
        margin=margin,
        polarity=polarity,
        hit_delay_boundary=hit_delay_boundary,
        hit_phase_boundary=torch.zeros(B, dtype=torch.bool, device=rx_signal.device),
        ambiguous=ambiguous,
        effective_sync_len=sync_len_eff,
    )


def _align_oversampled(
    tx_symbols: torch.Tensor,
    rx_signal: torch.Tensor,
    seq_len: int,
    cfg: SyncConfig,
) -> SyncResult:
    B, T_rx = rx_signal.shape
    P = cfg.oversample_factor

    sync_len_eff = min(cfg.sync_len_symbols, tx_symbols.shape[1], T_rx // P)
    tx_ref = tx_symbols[:, :sync_len_eff]

    t_pad = ((T_rx + P - 1) // P) * P
    rx_pad = F.pad(rx_signal, (0, t_pad - T_rx))
    rx_phase = rx_pad.view(B, -1, P).transpose(1, 2).contiguous()

    phase_ids = torch.arange(P, device=rx_signal.device)
    phase_lens = ((T_rx - phase_ids + P - 1) // P).clamp_min(0)

    search_len_p = phase_lens - sync_len_eff + 1
    valid_sync_counts = search_len_p.clamp_min(0)
    D_search = min(cfg.max_delay_symbols, int(valid_sync_counts.max().item()))

    if D_search <= 0:
        raise ValueError(
            f"No valid sync windows: max_delay={cfg.max_delay_symbols}, phase_lens={phase_lens.tolist()}"
        )

    delay_ids = torch.arange(D_search, device=rx_signal.device)
    valid_mask = delay_ids.view(1, 1, D_search) < valid_sync_counts.view(1, P, 1)

    windows = rx_phase.unfold(dimension=-1, size=sync_len_eff, step=1)[:, :, :D_search, :]

    tx_norm = tx_ref.pow(2).sum(dim=-1).sqrt().view(B, 1, 1)
    corr_signed = (windows * tx_ref[:, None, None, :]).sum(dim=-1)
    corr_abs = corr_signed.abs()

    if cfg.metric == "normalized":
        win_norm = windows.pow(2).sum(dim=-1).sqrt()
        rho = corr_abs / (tx_norm * win_norm + cfg.eps)
    else:
        rho = corr_abs

    neg_inf = torch.finfo(rho.dtype).min
    rho = rho.masked_fill(~valid_mask, neg_inf)

    rho_flat = rho.view(B, -1)
    best_flat = rho_flat.argmax(dim=1)
    best_phase = best_flat // D_search
    best_delay = best_flat % D_search

    valid_out_counts = (phase_lens - seq_len + 1).clamp_min(1)
    max_out_start = valid_out_counts - 1
    best_delay_clamped = torch.minimum(best_delay, max_out_start[best_phase])

    second_best_scores, _ = rho_flat.topk(k=2, dim=1)
    second_best_score = second_best_scores[:, 1]
    best_score = rho_flat.max(dim=1).values
    margin = best_score - second_best_score

    best_signed_corr = corr_signed.gather(
        2, best_delay.unsqueeze(-1).unsqueeze(-1).expand(-1, P, -1)
    ).gather(1, best_phase.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, D_search)).squeeze(-1).squeeze(-1)

    polarity = torch.sign(best_signed_corr)
    polarity = torch.where(polarity == 0, torch.ones_like(polarity), polarity)

    hit_phase_boundary = (best_phase == 0) | (best_phase == P - 1)
    hit_delay_boundary = (best_delay == 0) | (best_delay == D_search - 1)

    b_idx = torch.arange(B, device=rx_signal.device)
    best_phase_stream = rx_phase[b_idx, best_phase, :]

    sample_ids = torch.arange(seq_len, device=rx_signal.device)
    gather_idx = best_delay_clamped[:, None] + sample_ids[None, :]
    rx_aligned = best_phase_stream.gather(1, gather_idx)

    ambiguous = (best_score < cfg.ambiguous_score_thresh) | (
        margin < cfg.ambiguous_margin_thresh
    )

    best_total_sample_offset = best_phase + P * best_delay_clamped

    return SyncResult(
        rx_aligned=rx_aligned,
        best_phase=best_phase.long(),
        best_delay_symbols=best_delay_clamped.long(),
        best_total_sample_offset=best_total_sample_offset.long(),
        best_score=best_score,
        best_signed_corr=best_signed_corr,
        second_best_score=second_best_score,
        margin=margin,
        polarity=polarity,
        hit_delay_boundary=hit_delay_boundary,
        hit_phase_boundary=hit_phase_boundary,
        ambiguous=ambiguous,
        effective_sync_len=sync_len_eff,
    )


def cross_correlate_sync_batch_deprecated(
    tx, rx, max_delay=50, sync_len=None
):
    """
    DEPRECATED: Compatibility wrapper for legacy cross_correlate_sync_batch.

    This function exists only to provide a deprecation warning for any
    code still calling the old API. All callers should migrate to
    align_rx_to_tx().

    Args:
        tx: [B, T] transmitted symbols
        rx: [B, T_rx] received signal
        max_delay: maximum delay search in symbols
        sync_len: sync correlation length

    Returns:
        List of integer delays per example (legacy behavior)
    """
    warnings.warn(
        "cross_correlate_sync_batch is deprecated. "
        "Use align_rx_to_tx from sync_utils instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    B, T_tx = tx.shape
    seq_len = T_tx
    if sync_len is None:
        sync_len = min(200, seq_len - max_delay)

    if sync_len <= 0:
        raise ValueError(f"Sequence length {seq_len} too short for max_delay {max_delay}")

    cfg = SyncConfig(
        oversample_factor=1,
        max_delay_symbols=max_delay,
        sync_len_symbols=sync_len,
        polarity_mode="report",
    )

    result = align_rx_to_tx(tx, rx, seq_len=seq_len, cfg=cfg)

    return result.best_delay_symbols.tolist()
