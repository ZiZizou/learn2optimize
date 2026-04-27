"""
Unit tests for s4p_inspector module.

Tests cover:
- Mixed-mode Sdd21 formula correctness
- Port-pairing scoring
- Quality metrics computation
- Inspection report generation
"""

import numpy as np
import pytest


class TestSdd21Formula:
    """Test the mixed-mode Sdd21 formula against analytical expressions."""

    def test_sdd21_identity_pairing_13_24(self):
        """Test Sdd21 = 0.5*(S21 - S23 - S41 + S43) for 13-24 pairing."""
        from s4p_inspector import sdd21_from_se_s4p

        nfreq = 10
        S = np.zeros((nfreq, 4, 4), dtype=complex)

        S[:, 1, 0] = 1.0 + 0.0j  # S21 = 1
        S[:, 3, 2] = 1.0 + 0.0j  # S43 = 1
        S[:, 3, 0] = 0.1 + 0.0j  # S41 = 0.1 (small cross-coupling)
        S[:, 1, 2] = 0.1 + 0.0j  # S23 = 0.1 (small cross-coupling)

        sdd21 = sdd21_from_se_s4p(S, pairing="13-24")

        expected = 0.5 * (1.0 - 0.1 - 0.1 + 1.0)
        np.testing.assert_allclose(np.abs(sdd21), np.abs(expected), rtol=1e-10)
        np.testing.assert_allclose(sdd21.real, expected.real, rtol=1e-10)

    def test_sdd21_with_differential_signal(self):
        """Test Sdd21 when input is purely differential (p+ = +1, p- = -1)."""
        from s4p_inspector import sdd21_from_se_s4p

        nfreq = 5
        S = np.zeros((nfreq, 4, 4), dtype=complex)

        amplitude = 0.5
        S[:, 1, 0] = amplitude  # S21
        S[:, 1, 2] = -amplitude  # S23 (opposite sign due to differential)
        S[:, 3, 0] = -amplitude  # S41 (opposite sign due to differential)
        S[:, 3, 2] = amplitude  # S43

        sdd21 = sdd21_from_se_s4p(S, pairing="13-24")

        expected = amplitude
        np.testing.assert_allclose(sdd21, expected, rtol=1e-10)

    def test_sdd21_zero_mode_conversion(self):
        """Test Sdd21 when there's no mode conversion (pure differential)."""
        from s4p_inspector import sdd21_from_se_s4p

        nfreq = 1
        S = np.zeros((nfreq, 4, 4), dtype=complex)

        S[0, 1, 0] = 0.8
        S[0, 1, 2] = -0.8
        S[0, 3, 0] = -0.8
        S[0, 3, 2] = 0.8

        sdd21 = sdd21_from_se_s4p(S, pairing="13-24")

        assert np.abs(sdd21[0] - 0.8) < 1e-10

    def test_sdd21_unknown_pairing_raises(self):
        """Test that unknown pairing raises ValueError."""
        from s4p_inspector import sdd21_from_se_s4p

        S = np.zeros((10, 4, 4), dtype=complex)

        with pytest.raises(ValueError, match="Unknown pairing"):
            sdd21_from_se_s4p(S, pairing="invalid")


class TestModeConversionTerms:
    """Test common-to-differential and differential-to-common terms."""

    def test_sdc21_formula(self):
        """Test Sdc21 (common-to-differential) formula."""
        from s4p_inspector import sdc21_from_se_s4p

        nfreq = 5
        S = np.zeros((nfreq, 4, 4), dtype=complex)

        S[:, 1, 0] = 0.5 + 0.1j
        S[:, 1, 2] = 0.5 - 0.1j
        S[:, 3, 0] = -0.5 - 0.1j
        S[:, 3, 2] = -0.5 + 0.1j

        sdc21 = sdc21_from_se_s4p(S, pairing="13-24")

        expected = 0.5 * (
            (0.5 + 0.1j) + (0.5 - 0.1j) - (-0.5 - 0.1j) - (-0.5 + 0.1j)
        )
        np.testing.assert_allclose(sdc21, expected, rtol=1e-10)

    def test_scd21_formula(self):
        """Test Scd21 (differential-to-common) formula."""
        from s4p_inspector import scd21_from_se_s4p

        nfreq = 5
        S = np.zeros((nfreq, 4, 4), dtype=complex)

        S[:, 1, 0] = 0.5 + 0.1j
        S[:, 1, 2] = 0.5 - 0.1j
        S[:, 3, 0] = -0.5 - 0.1j
        S[:, 3, 2] = -0.5 + 0.1j

        scd21 = scd21_from_se_s4p(S, pairing="13-24")

        expected = 0.5 * (
            (0.5 + 0.1j) - (0.5 - 0.1j) + (-0.5 - 0.1j) - (-0.5 + 0.1j)
        )
        np.testing.assert_allclose(scd21, expected, rtol=1e-10)


class TestQualityMetrics:
    """Test quality metric computations."""

    def test_bandwidth_ratio_calculation(self):
        """Test bandwidth ratio computation."""
        from s4p_inspector import compute_quality_metrics, TouchstoneMetadata

        class MockNetwork:
            number_of_ports = 2
            f = np.array([1e9, 2e9, 3e9, 4e9, 5e9])
            s = np.zeros((5, 2, 2), dtype=complex)
            s[:, 1, 0] = 0.5 + 0.1j
            s[:, 0, 1] = 0.5 + 0.1j

        net = MockNetwork()
        metrics = compute_quality_metrics(net, baud_rate_hz=2e9, fs_target=4e9)

        assert metrics["measured_fmax_hz"] == 5e9
        assert metrics["target_nyquist_hz"] == 2e9
        assert metrics["bandwidth_ratio"] == 2.5

    def test_passivity_check(self):
        """Test passivity violation detection."""
        from s4p_inspector import compute_quality_metrics

        class MockNetwork:
            number_of_ports = 2
            f = np.array([1e9, 2e9])
            s = np.zeros((2, 2, 2), dtype=complex)
            s[0, 1, 0] = 0.5
            s[0, 0, 1] = 0.5
            s[1, 1, 0] = 0.5
            s[1, 0, 1] = 0.5

        net = MockNetwork()
        metrics = compute_quality_metrics(net, baud_rate_hz=2e9)

        assert metrics["passivity_ok"] is True
        assert metrics["passivity_violations"] == 0


class TestPortPairingScores:
    """Test port pairing inference and scoring."""

    def test_prefers_strong_thru_path(self):
        """Test that pairing with strong thru path is preferred."""
        from s4p_inspector import infer_port_topology, PortMode

        class MockNetwork:
            number_of_ports = 4
            f = np.array([1e9, 2e9, 3e9, 4e9, 5e9, 6e9, 7e9, 8e9, 9e9, 10e9])
            s = np.zeros((10, 4, 4), dtype=complex)

            s[:, 1, 0] = 0.9 + 0.05j  # S21: strong (for 13-24)
            s[:, 3, 2] = 0.9 + 0.05j  # S43: strong (for 13-24)
            s[:, 1, 2] = 0.05 + 0.01j  # S23: weak cross-coupling
            s[:, 3, 0] = 0.05 + 0.01j  # S41: weak cross-coupling

            s[:, 2, 0] = 0.3 + 0.05j  # S31: medium (for 12-34)
            s[:, 3, 1] = 0.3 + 0.05j  # S42: medium (for 12-34)
            s[:, 2, 1] = 0.05 + 0.01j
            s[:, 3, 0] = 0.05 + 0.01j

        result = infer_port_topology(MockNetwork())

        assert result["port_mode"] == PortMode.SINGLE_ENDED
        assert result["best_pairing"] == "13-24"


class TestTransferExtraction:
    """Test transfer function extraction."""

    def test_s21_extraction_2port(self):
        """Test S21 extraction from 2-port network."""
        from s4p_inspector import extract_transfer_function, TransferMode

        class MockNetwork:
            number_of_ports = 2
            f = np.array([1e9, 2e9, 3e9])
            s = np.zeros((3, 2, 2), dtype=complex)
            s[:, 1, 0] = 0.5 + 0.1j
            s[:, 0, 1] = 0.5 - 0.1j

        net = MockNetwork()
        freq, H, name = extract_transfer_function(net, role="thru", transfer_mode=TransferMode.S21)

        np.testing.assert_allclose(freq, [1e9, 2e9, 3e9])
        np.testing.assert_allclose(H, [0.5+0.1j, 0.5+0.1j, 0.5+0.1j])
        assert name == "S21"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
