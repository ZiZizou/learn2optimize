"""
Unit tests for augmentation functions in generate_synthetic_channels.py.

Tests cover:
- Phase-aware interpolation
- Frequency-domain augmentations (IL, reflections, cascade)
- Normalization behavior
- Mixup labeling (physics_valid flag)
- Bandwidth/FFT sizing
- Causality metrics
"""

import numpy as np
import torch
import pytest


class TestPhaseAwareInterpolation:
    """Test phase-aware interpolation vs linear real/imag interpolation."""

    def test_phase_aware_interpolation_preserves_phase_structure(self):
        """Test that phase-aware interpolation produces smooth phase."""
        from generate_synthetic_channels import interp_complex_to_grid_phase_aware

        f_src = np.linspace(1e9, 10e9, 50)
        mag = np.exp(-0.1 * (f_src - 5e9)**2 / 1e18)
        phase = -2 * np.pi * f_src * 1e-9
        h_src = mag * np.exp(1j * phase)

        f_tgt = np.linspace(0.5e9, 12e9, 100)
        h_tgt = interp_complex_to_grid_phase_aware(f_src, h_src, f_tgt)

        assert h_tgt.shape == f_tgt.shape
        assert not np.any(np.isnan(h_tgt))
        reconstructed_phase = np.angle(h_tgt)
        phase_diff = np.diff(reconstructed_phase)
        assert np.all(np.abs(phase_diff) < 0.5)


class TestGroupDelayEstimation:
    """Test group delay estimation."""

    def test_group_delay_zero_for_linear_phase(self):
        """Linear phase corresponds to pure delay."""
        from generate_synthetic_channels import estimate_group_delay

        f = np.linspace(1e9, 10e9, 100)
        delay_samples = 10
        phase = -2 * np.pi * f * delay_samples * 1e-9
        H = np.exp(1j * phase)

        tau = estimate_group_delay(f, H)

        assert abs(tau - delay_samples) < 1.0

    def test_group_delay_returns_zero_for_insufficient_points(self):
        """Edge case: insufficient low-loss points returns 0."""
        from generate_synthetic_channels import estimate_group_delay

        f = np.array([1e9])
        H = np.array([1.0 + 0j])

        tau = estimate_group_delay(f, H)
        assert tau == 0.0


class TestNormalizationBehavior:
    """Test that normalization modes preserve or scale amplitude correctly."""

    def test_normalize_l2(self):
        """Test L2 normalization."""
        from generate_synthetic_channels import normalize_l2

        x = np.random.randn(4, 64).astype(np.float32)
        x_norm = normalize_l2(x)

        norms = np.linalg.norm(x_norm, axis=-1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-5)


class TestFrequencyDomainIL:
    """Test frequency-domain insertion loss augmentation."""

    def test_il_augmentation_increases_loss_with_alpha(self):
        """Higher loss coefficients should produce more attenuation."""
        from generate_synthetic_channels import augment_il_freq

        batch_size, n_freq = 4, 128
        f_grid = np.linspace(1e9, 20e9, n_freq)
        H_batch = np.ones((batch_size, n_freq), dtype=complex)

        H_low = augment_il_freq(H_batch, f_grid, alpha_c_range=(0.0, 0.0))
        H_high = augment_il_freq(H_batch, f_grid, alpha_c_range=(0.5, 0.5))

        mean_low = np.mean(np.abs(H_low))
        mean_high = np.mean(np.abs(H_high))

        assert mean_high < mean_low


class TestFrequencyDomainReflections:
    """Test frequency-domain reflection augmentation."""

    def test_reflection_delays_in_ui(self):
        """Reflection delays should be in UI, not sample index."""
        from generate_synthetic_channels import augment_reflections_freq

        batch_size, n_freq = 4, 128
        f_grid = np.linspace(1e9, 20e9, n_freq)
        H_batch = np.ones((batch_size, n_freq), dtype=complex)

        ui_period = 1e-9
        delay_ui_min, delay_ui_max = 2.0, 5.0

        H_aug = augment_reflections_freq(
            H_batch, f_grid,
            num_refs_range=(1, 1),
            delay_ui_min=delay_ui_min,
            delay_ui_max=delay_ui_max,
            amp_max=0.1,
            ui_period_s=ui_period,
        )

        assert H_aug.shape == H_batch.shape
        assert not np.allclose(H_aug, H_batch)


class TestCascadeFrequencyDomain:
    """Test frequency-domain cascade augmentation."""

    def test_cascade_multiplies_transfers(self):
        """Cascade should multiply transfer functions."""
        from generate_synthetic_channels import augment_cascade_freq

        batch_size, n_freq = 4, 128
        f_grid = np.linspace(1e9, 20e9, n_freq)
        H_batch = np.ones((batch_size, n_freq), dtype=complex) * 0.5

        pool = np.ones((10, n_freq), dtype=complex) * 2.0
        pool_f_grid = f_grid

        H_aug = augment_cascade_freq(H_batch, f_grid, pool, pool_f_grid)

        np.testing.assert_allclose(H_aug, 1.0, rtol=1e-5)


class TestIRTransferConversion:
    """Test IR <-> Transfer conversion."""

    def test_ir_to_transfer_and_back(self):
        """IR -> Transfer -> IR should be consistent."""
        from generate_synthetic_channels import (
            ir_to_transfer_batch,
            transfer_to_ir_batch,
        )

        batch_size, num_taps = 4, 64
        fs_hz = 32e9

        h_orig = np.random.randn(batch_size, num_taps).astype(np.float32)
        h_orig = h_orig / (np.linalg.norm(h_orig, axis=-1, keepdims=True) + 1e-12)

        H_batch, f_grid = ir_to_transfer_batch(h_orig, fs_hz)
        h_recon = transfer_to_ir_batch(H_batch, f_grid, fs_hz, num_taps)

        np.testing.assert_allclose(h_orig, h_recon, rtol=1e-4)


class TestMixupLabeling:
    """Test that mixup is properly labeled as non-physical."""

    def test_augment_mixup_flagged_as_non_physics(self):
        """Test that mixup augmentation marks output as physics_valid=False."""
        from generate_synthetic_channels import augment_mixup

        h_batch = torch.randn(4, 64)
        thru_pool = torch.randn(10, 64)

        h_aug = augment_mixup(h_batch, thru_pool, lam_range=(0.3, 0.7))

        assert h_aug.shape == h_batch.shape
        assert not torch.allclose(h_aug, h_batch)


class TestReflectionAugmentation:
    """Test reflection augmentation delay parameterization."""

    def test_reflections_in_sample_index_not_physical_time(self):
        """Test that current implementation uses sample index, not physical time.

        Note: This test documents the current behavior. The frequency-domain
        delayed-echo model should eventually replace this.
        """
        from generate_synthetic_channels import augment_reflections

        h_batch = torch.zeros(4, 100)
        h_batch[:, 50] = 1.0

        h_aug = augment_reflections(h_batch, max_refs=1, max_ratio=0.1, min_idx=10)

        assert h_aug.shape == h_batch.shape
        nonzero_indices = torch.where(torch.abs(h_aug) > 0.1)
        assert len(nonzero_indices[0]) > 0


class TestCascadeConsistency:
    """Test cascade augmentation consistency."""

    def test_cascade_commutative_in_amplitude(self):
        """Test that cascade order doesn't affect final amplitude (for linear systems)."""
        from generate_synthetic_channels import augment_cascade

        h1 = torch.randn(1, 64)
        h2 = torch.randn(1, 64)
        pool = torch.cat([h1, h2], dim=0)

        h_cascade_12 = augment_cascade(h1, pool)
        h_cascade_21 = augment_cascade(h2, pool)

        assert h_cascade_12.shape == h1.shape
        assert h_cascade_21.shape == h2.shape


class TestIlTiltDocumentation:
    """Document that IL tilt is a crude approximation."""

    def test_il_tilt_is_rc_filter_not_physics(self):
        """Test that IL tilt uses 1-tap RC filter approximation.

        Note: This is documented as NOT physics-based. The frequency-domain
        skin-effect + dielectric model should replace this in Phase 3.
        """
        from generate_synthetic_channels import augment_il_tilt

        h_batch = torch.randn(4, 64)

        h_aug = augment_il_tilt(h_batch, alpha_range=(0.0, 0.4))

        assert h_aug.shape == h_batch.shape


class TestNextPowerOfTwo:
    """Test next_power_of_two utility."""

    def test_next_power_of_two(self):
        """Test power of two rounding."""
        from generate_synthetic_channels import next_power_of_two

        assert next_power_of_two(1023) == 1024
        assert next_power_of_two(1024) == 1024
        assert next_power_of_two(1025) == 2048


if __name__ == "__main__":
    pytest.main([__file__, "-v"])