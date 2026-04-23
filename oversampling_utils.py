import torch


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
):
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
        corr = (windows * tx_ref[:, None, :]).sum(dim=-1).abs()
        delays = corr.argmax(dim=1)
        scores = corr.gather(1, delays[:, None]).squeeze(1)
        score = scores.mean().item()

        if best_score is None or score > best_score:
            best_score = score
            best_phase = phase
            best_delay = int(torch.median(delays.float()).item())
            best_rx = rx_phase

    return best_rx, best_phase, best_delay
