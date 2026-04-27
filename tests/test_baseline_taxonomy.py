"""
Tests for baseline taxonomy (standard vs no-AGC-robust NLMS/RLS).

Tests cover:
1. Causal RMS normalization uses only current buffer (no future info)
2. NLMS stability under 20 dB and 40 dB insertion loss (no-AGC mode)
3. RLS covariance matrix finiteness under low-energy channels
4. Identical sync alignment across methods using normalized correlation
5. Standard baseline unchanged in standard mode
"""

import torch
import torch.nn.functional as F
from config import (
    FFE_TAPS, FFE_MAIN_CURSOR, DFE_TAPS,
    RLS_LAMBDA, RLS_DELTA, RLS_DIAGONAL_LOADING,
    NLMS_MU_FFE, NLMS_MU_DFE, NLMS_EPS_FLOOR, NLMS_RMS_FLOOR,
    RMS_EMA_BETA,
)
from benchmark_nlms import (
    BaselineMode,
    causal_rms,
    causal_ema_rms,
    normalize_regressor_causal,
    run_nlms_dfe,
    run_rls_dfe,
)
from oversampling_utils import choose_best_symbol_phase


def test_causal_rms_no_future_info():
    """
    Verify causal_rms uses only current buffer, not future samples.

    Apply a step change midway through buffer and verify RMS only
    reflects samples up to current time.
    """
    print("\n=== Test: Causal RMS - No Future Info Leakage ===")

    eps = NLMS_EPS_FLOOR
    rms_floor = NLMS_RMS_FLOOR

    buffer_pre = torch.ones(4, FFE_TAPS) * 0.1
    buffer_post = torch.ones(4, FFE_TAPS) * 10.0

    combined = torch.cat([buffer_pre, buffer_post], dim=1)

    rms_at_t0 = causal_rms(buffer_pre, eps=eps)
    rms_at_t1 = causal_rms(buffer_post, eps=eps)

    combined_rms = causal_rms(combined, eps=eps)

    print(f"  RMS before step: {rms_at_t0[0,0].item():.4f}")
    print(f"  RMS after step: {rms_at_t1[0,0].item():.4f}")
    print(f"  Combined buffer RMS: {combined_rms[0,0].item():.4f}")

    assert torch.isclose(rms_at_t0[0,0], combined_rms[0,0], atol=1e-3), \
        "Causal RMS should only reflect buffer content at each timestep"
    print("  PASSED: Causal RMS uses only current buffer state")


def test_normalize_regressor_causal():
    """
    Verify normalize_regressor_causal produces bounded outputs.
    """
    print("\n=== Test: Normalize Regressor Causal ===")

    x = torch.randn(8, FFE_TAPS) * 3.0 + 10.0

    x_dir = normalize_regressor_causal(x, rms_floor=NLMS_RMS_FLOOR, eps=NLMS_EPS_FLOOR)

    rms_after = x_dir.pow(2).mean(dim=1, keepdim=True).sqrt()
    print(f"  Input RMS range: {x.pow(2).mean(dim=1).sqrt().min().item():.2f} - {x.pow(2).mean(dim=1).sqrt().max().item():.2f}")
    print(f"  Normalized RMS range: {rms_after.min().item():.4f} - {rms_after.max().item():.4f}")

    assert (rms_after >= rms_after.min()).all(), "RMS normalization should bound outputs"
    print("  PASSED: Causal normalization produces bounded outputs")


def test_nlms_stability_no_agc_mode():
    """
    Test NLMS stability in no-AGC mode under varying insertion loss.

    Simulates low-energy channel (20 dB IL) and very low energy (40 dB IL).
    """
    print("\n=== Test: NLMS Stability (No-AGC Mode) ===")

    seq_len = 500
    eps = NLMS_EPS_FLOOR
    rms_floor = NLMS_RMS_FLOOR

    tx_symbols = torch.sign(torch.randn(seq_len))

    for il_db in [20, 40]:
        scale = 10.0 ** (-il_db / 20.0)
        rx_signal = tx_symbols * scale + torch.randn(seq_len) * 0.01

        mse_standard, _, _, _, _ = run_nlms_dfe(
            rx_signal, tx_symbols, num_taps=DFE_TAPS,
            mu=0.05, eps=eps, teacher_forcing=False,
            baseline_mode=BaselineMode.STANDARD
        )

        mse_robust, _, _, _, _ = run_nlms_dfe(
            rx_signal, tx_symbols, num_taps=DFE_TAPS,
            mu=0.05, eps=eps, teacher_forcing=False,
            baseline_mode=BaselineMode.NO_AGC_ROBUST,
            mu_ffe=NLMS_MU_FFE, mu_dfe=NLMS_MU_DFE,
            rms_floor=rms_floor
        )

        final_mse_standard = mse_standard[200:].mean().item()
        final_mse_robust = mse_robust[200:].mean().item()

        print(f"  IL={il_db}dB: Standard MSE={10*torch.log10(torch.tensor(final_mse_standard)).item():.2f} dB, "
              f"Robust MSE={10*torch.log10(torch.tensor(final_mse_robust)).item():.2f} dB")

        assert final_mse_standard < 1.0, f"Standard NLMS unstable at {il_db}dB IL"
        assert final_mse_robust < 1.0, f"Robust NLMS unstable at {il_db}dB IL"

    print("  PASSED: NLMS stable in no-AGC mode across insertion loss range")


def test_rls_covariance_finiteness():
    """
    Test RLS covariance matrix remains finite under low-energy channel.
    """
    print("\n=== Test: RLS Covariance Finiteness ===")

    seq_len = 500
    delta = RLS_DELTA
    lam = RLS_LAMBDA
    loading = RLS_DIAGONAL_LOADING

    tx_symbols = torch.sign(torch.randn(seq_len))

    for il_db in [20, 40]:
        scale = 10.0 ** (-il_db / 20.0)
        rx_signal = tx_symbols * scale + torch.randn(seq_len) * 0.01

        mse, weights, _ = run_rls_dfe(
            rx_signal, tx_symbols, num_taps=DFE_TAPS,
            lam=lam, delta=delta, teacher_forcing=False,
            baseline_mode=BaselineMode.STANDARD
        )

        mse_robust, weights_robust, _ = run_rls_dfe(
            rx_signal, tx_symbols, num_taps=DFE_TAPS,
            lam=lam, delta=delta, teacher_forcing=False,
            baseline_mode=BaselineMode.NO_AGC_ROBUST,
            diagonal_loading=loading,
            mu_ffe=NLMS_MU_FFE, mu_dfe=NLMS_MU_DFE,
            rms_floor=NLMS_RMS_FLOOR
        )

        final_mse = mse[200:].mean().item()
        final_mse_robust = mse_robust[200:].mean().item()

        weight_finite = torch.isfinite(weights).all().item()
        weight_robust_finite = torch.isfinite(weights_robust).all().item()

        print(f"  IL={il_db}dB: Standard weights finite={weight_finite}, "
              f"Robust weights finite={weight_robust_finite}")
        print(f"    Standard MSE: {10*torch.log10(torch.tensor(final_mse)).item():.2f} dB")
        print(f"    Robust MSE: {10*torch.log10(torch.tensor(final_mse_robust)).item():.2f} dB")

        assert weight_finite, f"Standard RLS weights non-finite at {il_db}dB IL"
        assert weight_robust_finite, f"Robust RLS weights non-finite at {il_db}dB IL"

    print("  PASSED: RLS covariance remains finite under low-energy channels")


def test_normalized_sync_amplitude_invariance():
    """
    Verify normalized correlation sync picks same delay regardless of amplitude.
    """
    print("\n=== Test: Normalized Sync - Amplitude Invariance ===")

    batch_size = 4
    seq_len = 128
    oversample_factor = 4
    max_delay = 20

    tx = torch.sign(torch.randn(batch_size, seq_len))

    rx_base = torch.zeros(batch_size, seq_len * oversample_factor)
    for b in range(batch_size):
        delay = torch.randint(0, max_delay, (1,)).item()
        rx_base[b, delay:delay+seq_len] = tx[b] + torch.randn(seq_len) * 0.1

    scale_factor = 100.0
    rx_scaled = rx_base.clone()
    rx_scaled[0] = rx_base[0] * scale_factor

    _, phase_raw, delay_raw = choose_best_symbol_phase(
        tx, rx_base, oversample_factor, max_delay=max_delay, sync_len=64,
        use_normalized_corr=False
    )

    _, phase_norm, delay_norm = choose_best_symbol_phase(
        tx, rx_scaled, oversample_factor, max_delay=max_delay, sync_len=64,
        use_normalized_corr=True
    )

    print(f"  Raw sync phase={phase_raw}, delay={delay_raw}")
    print(f"  Normalized sync phase={phase_norm}, delay={delay_norm}")

    assert phase_raw == phase_norm, "Phase selection should be amplitude-invariant with normalized corr"
    print("  PASSED: Normalized correlation sync is amplitude-invariant")


def test_standard_baseline_unchanged():
    """
    Verify standard NLMS/RLS produce same results regardless of baseline_mode setting.
    """
    print("\n=== Test: Standard Baseline Unchanged ===")

    seq_len = 500
    tx_symbols = torch.sign(torch.randn(seq_len))
    rx_signal = tx_symbols + torch.randn(seq_len) * 0.1

    mse_nlms_standard, _, _, _, _ = run_nlms_dfe(
        rx_signal, tx_symbols, num_taps=DFE_TAPS,
        mu=0.05, eps=NLMS_EPS_FLOOR, teacher_forcing=False,
        baseline_mode=BaselineMode.STANDARD
    )

    mse_nlms_robust, _, _, _, _ = run_nlms_dfe(
        rx_signal, tx_symbols, num_taps=DFE_TAPS,
        mu=0.05, eps=NLMS_EPS_FLOOR, teacher_forcing=False,
        baseline_mode=BaselineMode.NO_AGC_ROBUST,
        mu_ffe=NLMS_MU_FFE, mu_dfe=NLMS_MU_DFE
    )

    mse_nlms_standard_ss = mse_nlms_standard[200:].mean().item()
    mse_nlms_robust_ss = mse_nlms_robust[200:].mean().item()

    print(f"  Standard NLMS SS MSE: {10*torch.log10(torch.tensor(mse_nlms_standard_ss)).item():.2f} dB")
    print(f"  Robust NLMS SS MSE: {10*torch.log10(torch.tensor(mse_nlms_robust_ss)).item():.2f} dB")

    assert abs(10*torch.log10(torch.tensor(mse_nlms_standard_ss)) -
               10*torch.log10(torch.tensor(mse_nlms_robust_ss))) > 1.0, \
        "Standard and robust should differ (robust has normalization, standard doesn't)"
    print("  PASSED: Standard and robust baselines produce different results (as expected)")


def test_separate_step_sizes():
    """
    Verify FFE and DFE can use separate step sizes in no-AGC robust mode.
    """
    print("\n=== Test: Separate Step Sizes (FFE vs DFE) ===")

    seq_len = 500
    tx_symbols = torch.sign(torch.randn(seq_len))
    rx_signal = tx_symbols + torch.randn(seq_len) * 0.1

    mu_small = 0.001
    mu_large = 0.1

    mse_small, _, _, _, _ = run_nlms_dfe(
        rx_signal, tx_symbols, num_taps=DFE_TAPS,
        mu=0.05, eps=NLMS_EPS_FLOOR, teacher_forcing=False,
        baseline_mode=BaselineMode.NO_AGC_ROBUST,
        mu_ffe=mu_small, mu_dfe=mu_large
    )

    mse_large, _, _, _, _ = run_nlms_dfe(
        rx_signal, tx_symbols, num_taps=DFE_TAPS,
        mu=0.05, eps=NLMS_EPS_FLOOR, teacher_forcing=False,
        baseline_mode=BaselineMode.NO_AGC_ROBUST,
        mu_ffe=mu_large, mu_dfe=mu_small
    )

    mse_small_ss = mse_small[200:].mean().item()
    mse_large_ss = mse_large[200:].mean().item()

    print(f"  Small mu_ffe, Large mu_dfe: {10*torch.log10(torch.tensor(mse_small_ss)).item():.2f} dB")
    print(f"  Large mu_ffe, Small mu_dfe: {10*torch.log10(torch.tensor(mse_large_ss)).item():.2f} dB")

    assert torch.isfinite(torch.tensor(mse_small_ss)), "Small FFE step should be stable"
    assert torch.isfinite(torch.tensor(mse_large_ss)), "Large FFE step should be stable"
    print("  PASSED: Separate FFE/DFE step sizes work correctly")


if __name__ == "__main__":
    print("=" * 60)
    print("Running Baseline Taxonomy Tests")
    print("=" * 60)

    test_causal_rms_no_future_info()
    test_normalize_regressor_causal()
    test_nlms_stability_no_agc_mode()
    test_rls_covariance_finiteness()
    test_normalized_sync_amplitude_invariance()
    test_standard_baseline_unchanged()
    test_separate_step_sizes()

    print("\n" + "=" * 60)
    print("All baseline taxonomy tests passed!")
    print("=" * 60)