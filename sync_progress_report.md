# Sync Progress Report

## 1. Overview

This document catalogs the current state of symbol synchronization across all training and evaluation scripts in the learned optimizer codebase. Two synchronization functions exist in `oversampling_utils.py`:

- **`choose_best_symbol_phase`** (lines 20-93): Batch-global sync returning scalar delay
- **`choose_best_symbol_phase_per_example`** (lines 97-126): Per-example sync returning `[batch]` delay tensor

The choice of function is governed by `channel_ir_norm_mode`:
- `peak` / `l2` (AGC modes): Uses `choose_best_symbol_phase` with batch-common delay
- `none` (no-AGC mode): Uses `choose_best_symbol_phase_per_example` with per-example delays

---

## 2. Synchronization Functions

### 2.1 `choose_best_symbol_phase` (oversampling_utils.py:20)

**Signature:**
```python
def choose_best_symbol_phase(
    tx_symbols: torch.Tensor,
    rx_oversampled: torch.Tensor,
    oversample_factor: int,
    max_delay: int = 64,
    sync_len: int = 128,
    use_normalized_corr: bool = False,
) -> (best_rx, best_phase, best_delay)
```

**Behavior:**
- Loops over `phase` in `range(oversample_factor)` (Python loop)
- For each phase, computes raw or normalized correlation per delay
- Returns **batch-common** `best_delay` as Python int (median across batch)
- `best_rx` is a single phase-aligned waveform shared across batch

**Returns:**
- `best_rx`: `[batch, seq_len]` - same waveform for all examples
- `best_phase`: integer
- `best_delay`: integer (median across batch)

### 2.2 `choose_best_symbol_phase_per_example` (oversampling_utils.py:97)

**Signature:**
```python
def choose_best_symbol_phase_per_example(
    tx_symbols: torch.Tensor,
    rx_oversampled: torch.Tensor,
    oversample_factor: int,
    max_delay: int = 64,
    sync_len: int = 128,
) -> (best_rx, best_phase, best_delay)
```

**Behavior:**
- Wrapper that calls `choose_best_symbol_phase_per_example_vectorized`
- Returns per-example `best_phase` and `best_delay` as tensors

**Returns:**
- `best_rx`: `[batch, seq_len]`
- `best_phase`: `[batch]` tensor of `torch.long`
- `best_delay`: `[batch]` tensor of `torch.long`

### 2.3 `choose_best_symbol_phase_per_example_vectorized` (oversampling_utils.py:130)

**Scientific Definition Followed:**
- Joint optimization: `(p_b*, d_b*) = argmax_{p,d} rho_{b,p,d}`
- Normalized correlation: `rho = |<x,y>| / (||x|| ||y|| + eps)`
- No Python loop over phase; uses tensor reshape + unfold

**Key Implementation Steps:**
```python
# 1. Pad and reshape to [B, P, S_max]
t_pad = ((T_rx + P - 1) // P) * P
rx_pad = F.pad(rx_oversampled, (0, t_pad - T_rx))
rx_phase = rx_pad.view(B, -1, P).transpose(1, 2).contiguous()

# 2. Compute valid per-phase lengths
phase_lens = ((T_rx - phase_ids + P - 1) // P).clamp_min(0)

# 3. Build sync windows [B, P, D, S]
windows = rx_phase.unfold(dimension=-1, size=sync_len_eff, step=1)[:, :, :D_search, :]

# 4. Normalized correlation
rho = corr / (tx_norm * win_norm + 1e-6)

# 5. Joint argmax over flattened (phase, delay) grid
rho_flat = rho.view(B, -1)
best_flat = rho_flat.argmax(dim=1)
best_phase = best_flat // D_search
best_delay = best_flat % D_search
```

---

## 3. File-by-File Sync State

### 3.1 l2o_basic.py

| Section | Mode | Function Used | Delay Handling |
|---------|------|-------------|----------------|
| Lines 325-360 (peak path) | `peak` | `choose_best_symbol_phase` | Scalar common_delay; `effective_seq_len = total_seq_len - common_delay` (line 360) |
| Lines 570-605 (none path) | `none` | `choose_best_symbol_phase_per_example` | `effective_seq_len = total_seq_len - common_delay.max().item()` (line 599); per-example indexing |

**Peak mode (line 331):**
```python
rx_init, best_phase, common_delay = choose_best_symbol_phase(
    tx_symbols,
    rx_frontend,
    OVERSAMPLE_FACTOR,
    max_delay=PHASE_SEARCH_MAX_DELAY,
    sync_len=PHASE_SEARCH_SYNC_LEN,
)
```

**None mode (line 578):**
```python
rx_init, best_phase, common_delay = choose_best_symbol_phase_per_example(
    tx_symbols,
    rx_frontend,
    OVERSAMPLE_FACTOR,
    max_delay=PHASE_SEARCH_MAX_DELAY,
    sync_len=PHASE_SEARCH_SYNC_LEN,
)
```

**Outer loop (none mode, line 599-605):**
```python
effective_seq_len = total_seq_len - common_delay.max().item()
# ...
for t_start in range(0, effective_seq_len, unroll_len):
    current_block_len = min(unroll_len, effective_seq_len - t_start)
```

**Per-example indexing (line 622):**
```python
rx_eq = rx_init[:, (t + common_delay):(t + common_delay + 1)]
# common_delay is [batch] tensor; PyTorch broadcasts with t (Python int)
```

**Limitation:** Outer loop uses `.max().item()` to get scalar for `range()`. Per-example delays used for indexing, but sequence length per example not individually enforced.

---

### 3.2 l2o_progressive.py

| Section | Mode | Function Used | Delay Handling |
|---------|------|-------------|----------------|
| Lines 352-397 (peak path) | `peak` | `choose_best_symbol_phase` | Scalar common_delay; `effective_seq_len = total_seq_len - common_delay` (line 387) |
| Lines 605-655 (none path) | `none` | `choose_best_symbol_phase_per_example` | `effective_seq_len = total_seq_len - common_delay.max().item()` (line 634) |

**Peak mode (line 358):**
```python
rx_init, best_phase, common_delay = choose_best_symbol_phase(
    tx_symbols,
    rx_frontend,
    OVERSAMPLE_FACTOR,
    max_delay=PHASE_SEARCH_MAX_DELAY,
    sync_len=PHASE_SEARCH_SYNC_LEN,
)
```

**None mode (line 613):**
```python
rx_init, best_phase, common_delay = choose_best_symbol_phase_per_example(
    tx_symbols,
    rx_frontend,
    OVERSAMPLE_FACTOR,
    max_delay=PHASE_SEARCH_MAX_DELAY,
    sync_len=PHASE_SEARCH_SYNC_LEN,
)
```

**Outer loop distinction:** Uses `current_unroll_len` (progressive curriculum) instead of static `unroll_len`:
```python
current_unroll_len = min(
    max_unroll,
    initial_unroll + (epoch // unroll_step_epoch) * UNROLL_DELTA
)
for t_start in range(0, effective_seq_len, current_unroll_len):
```

Same `.max().item()` limitation as l2o_basic.

---

### 3.3 l2o_mlp.py

| Mode | Function Used | Delay Handling |
|------|-------------|----------------|
| Both modes | `choose_best_symbol_phase` (line 237) | Scalar common_delay (only function used) |

**l2o_mlp.py does not have a no-AGC path.** It delegates no-AGC training to `l2o_mlp_no_agc.py` via `train_learned_optimizer_no_agc`.

```python
no_agc_mode = (args.channel_ir_norm_mode == "none")
if no_agc_mode:
    # Uses MultiRateLearnedMLPNoAGC from l2o_mlp_no_agc.py
    trained_model, loss_history, ss_history = train_learned_optimizer_no_agc(...)
else:
    # Uses MultiRateLearnedMLP (standard AGC)
    trained_model, loss_history, ss_history = train_learned_optimizer(...)
```

---

### 3.4 l2o_mlp_no_agc.py

| Mode | Function Used | Delay Handling |
|------|-------------|----------------|
| `none` | `choose_best_symbol_phase_per_example` (line 188) | `effective_seq_len = total_seq_len - common_delay.max().item()` (line 211) |

**Self-contained training function** (`train_learned_optimizer`, line 82) with its own sync call at line 188:
```python
rx_init, best_phase, common_delay = choose_best_symbol_phase_per_example(
    tx_symbols,
    rx_frontend,
    OVERSAMPLE_FACTOR,
    max_delay=PHASE_SEARCH_MAX_DELAY,
    sync_len=PHASE_SEARCH_SYNC_LEN,
)
```

**Per-example indexing (line 234):**
```python
rx_eq = rx_init[:, (t + common_delay):(t + common_delay + 1)]
```

---

### 3.5 evaluate_l2o.py

| Location | Function Used | Delay Handling |
|----------|-------------|----------------|
| Line 57 (inference) | `choose_best_symbol_phase` | Scalar `common_delay` |
| Line 353 (pre-alignment) | `choose_best_symbol_phase` | Scalar `common_delay` |

**Line 57:**
```python
rx_init, best_phase, common_delay = choose_best_symbol_phase(
    tx_symbols,
    rx_frontend,
    OVERSAMPLE_FACTOR,
    max_delay=PHASE_SEARCH_MAX_DELAY,
    sync_len=PHASE_SEARCH_SYNC_LEN,
)
# ...
effective_len = seq_len - common_delay
for t in range(effective_len):
    rx_eq = rx_init[:, (t + common_delay):(t + common_delay + 1)]
```

**Line 353:**
```python
rx_aligned, best_phase, common_delay = choose_best_symbol_phase(
    tx_symbols,
    rx_frontend,
    OVERSAMPLE_FACTOR,
    max_delay=PHASE_SEARCH_MAX_DELAY,
    sync_len=PHASE_SEARCH_SYNC_LEN,
    use_normalized_corr=use_normalized_corr,  # Optional normalized corr
)
```

**Note:** `evaluate_l2o.py` always uses batch-global sync, regardless of channel normalization mode. This is appropriate for evaluation but inconsistent with no-AGC training behavior.

---

## 4. Channel IR Normalization Modes

Defined in `utils.py` (lines 55-62):
```python
parser.add_argument(
    "--channel_ir_norm_mode",
    type=str,
    choices=["none", "peak", "l2"],
    default=None,
    help="Channel impulse response normalization mode: 'none' preserves raw "
         "amplitude (for no-AGC training), 'peak' normalizes peak to 1, "
         "'l2' normalizes L2 norm to 1. Default: 'peak' for backward compat."
)
```

| Mode | Meaning | Sync Function Used |
|------|---------|-------------------|
| `peak` | IR normalized to peak=1 | `choose_best_symbol_phase` (batch-global) |
| `l2` | IR normalized to L2 norm=1 | `choose_best_symbol_phase` (batch-global) |
| `none` | Raw amplitude preserved | `choose_best_symbol_phase_per_example` (per-example) |

---

## 5. Summary Table

| File | Peak/L2 Mode Sync | None Mode Sync | Outer Loop |
|------|-------------------|----------------|------------|
| l2o_basic.py | `choose_best_symbol_phase` (scalar delay) | `choose_best_symbol_phase_per_example` (per-example delay, `.max().item()` for loop) | Fixed `unroll_len` |
| l2o_progressive.py | `choose_best_symbol_phase` (scalar delay) | `choose_best_symbol_phase_per_example` (per-example delay, `.max().item()` for loop) | Progressive `current_unroll_len` |
| l2o_mlp.py | `choose_best_symbol_phase` (scalar delay) | N/A (delegates to l2o_mlp_no_agc.py) | Fixed `unroll_len` |
| l2o_mlp_no_agc.py | N/A | `choose_best_symbol_phase_per_example` (per-example delay, `.max().item()` for loop) | Fixed `unroll_len` |
| evaluate_l2o.py | `choose_best_symbol_phase` (scalar delay) | `choose_best_symbol_phase` (scalar delay - always batch-global) | Single-step loop |

---

## 6. Known Limitations

1. **Outer loop scalar conversion:** In no-AGC paths, `effective_seq_len = total_seq_len - common_delay.max().item()` uses `.max()` arbitrarily. Could use `.min()` or median. This means some examples may have shorter valid sequences than `effective_seq_len` allows.

2. **Per-example sequence length not enforced:** The outer TBPTT loop uses a single `effective_seq_len` for all examples. Examples with shorter delays could theoretically index beyond their valid range, though `.max()` provides a conservative bound.

3. **evaluate_l2o.py uses batch-global sync always:** Even in no-AGC evaluation scenarios, batch-global sync is used. This is likely intentional for fair comparison but is inconsistent with training behavior.

4. **l2o_mlp.py peak mode only:** The MLP standard path uses `choose_best_symbol_phase` exclusively. No per-example sync exists for the standard MLP optimizer.
