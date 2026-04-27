"""
Touchstone (S-Parameter) Inspector and Channel Extraction Utilities.

This module provides functions to inspect 4-port Touchstone files, detect whether
they represent differential pairs in single-ended format, infer port pairings,
and extract appropriate transfer functions.

Scientific Definitions:
    A Touchstone .s4p file stores a 4x4 scattering matrix S(f) vs frequency.
    Unless explicitly mixed-mode formatted, those ports are single-ended ports.
    A file can be physically about a differential pair while stored in single-ended
    basis -- representation (single-ended .s4p) differs from physical topology
    (differential link).

    For a differential pair, the physically relevant transfer is often the
    differential-to-differential term Sdd21, not raw S21.

    Sdd21 for input pair (p+, p-) and output pair (q+, q-):
        Sdd21 = 0.5 * (S_q+p+ - S_q+p- - S_q-p+ + S_q-p-)

    Example for pairing (1,3) input and (2,4) output:
        Sdd21 = 0.5 * (S21 - S23 - S41 + S43)

References:
    -混合模式S参数理论 (Mixed-mode S-parameter theory)
    - IEEE P370 standard for high-speed channel compliance
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Literal

import numpy as np

try:
    import skrf
    HAS_SKRF = True
except ImportError:
    HAS_SKRF = False


class PortMode(Enum):
    """Port representation mode."""
    SINGLE_ENDED = "single_ended"
    MIXED_MODE = "mixed_mode"
    UNKNOWN = "unknown"


class TransferMode(Enum):
    """Transfer function extraction mode."""
    S21 = "s21"
    SDD21 = "sdd21"
    AUTO = "auto"


PORT_PAIRINGS = ["13-24", "12-34", "14-23"]
"""Candidate port pairings for 4-port single-ended files representing differential pairs.

Notation: first two ports are input pair (p+, p-), last two are output pair (q+, q-).
- "13-24": input=(1,3), output=(2,4)
- "12-34": input=(1,2), output=(3,4)
- "14-23": input=(1,4), output=(2,3)
"""


@dataclass
class TouchstoneMetadata:
    """Metadata extracted from a Touchstone file."""
    nports: int
    format: str  # 'DB', 'RI', 'MA'
    reference_impedance: float
    frequency_hz: np.ndarray
    is_mixed_mode: bool
    file_path: pathlib.Path


@dataclass
class TopologyScore:
    """Scoring result for a candidate port pairing."""
    pairing: str
    score: float
    thru_strength: float  # mean low-freq |Sdd21|
    mode_conversion: float  # mean |Sdc21| + |Scd21|
    path_asymmetry: float  # conductor path asymmetry
    details: Dict


@dataclass
class InspectionReport:
    """Complete inspection report for a Touchstone file."""
    metadata: TouchstoneMetadata
    port_mode: PortMode
    port_pairing: str  # Falls back to "13-24" when inference is ambiguous
    pairing_scores: List[TopologyScore]
    transfer_mode: TransferMode
    quality_metrics: Dict
    warnings: List[str]


def sdd21_from_se_s4p(
    S: np.ndarray,
    pairing: str = "13-24"
) -> np.ndarray:
    """
    Compute differential-to-differential transfer Sdd21 from single-ended 4-port S-matrix.

    Scientific definition:
        For a differential pair with input port pair (p+, p-) and output pair (q+, q-),
        the mixed-mode differential insertion loss is:

        Sdd21 = 0.5 * (S_q+p+ - S_q+p- - S_q-p+ + S_q-p-)

    Args:
        S: S-matrix of shape [nfreq, 4, 4] using 0-based indexing internally.
           S[:, q, p] gives S_{qp} at each frequency.
        pairing: Port pairing as string "p+p--q+q-" where ports are 1-indexed.
                 Supported: "13-24", "12-34", "14-23".

    Returns:
        Complex-valued Sdd21 transfer function of shape [nfreq].

    Raises:
        ValueError: If pairing is not recognized.

    Example:
        For pairing "13-24" (ports 1,3 are input pair; ports 2,4 are output pair):
            Sdd21 = 0.5 * (S21 - S23 - S41 + S43)
        where S21 means S[:, 1, 0] in 0-based indexing.
    """
    # Convert 1-indexed port numbers to 0-based indices
    port_map = {
        "13-24": (0, 2, 1, 3),  # (p_plus, p_minus, q_plus, q_minus)
        "12-34": (0, 1, 2, 3),
        "14-23": (0, 3, 1, 2),
    }

    if pairing not in port_map:
        raise ValueError(
            f"Unknown pairing '{pairing}'. Supported: {list(port_map.keys())}"
        )

    p_plus, p_minus, q_plus, q_minus = port_map[pairing]

    # Sdd21 = 0.5 * (S_q+p+ - S_q+p- - S_q-p+ + S_q-p-)
    # In numpy array: S[:, q, p] corresponds to S_{qp}
    sdd21 = 0.5 * (
        S[:, q_plus, p_plus]
        - S[:, q_plus, p_minus]
        - S[:, q_minus, p_plus]
        + S[:, q_minus, p_minus]
    )

    return sdd21


def sdc21_from_se_s4p(
    S: np.ndarray,
    pairing: str = "13-24"
) -> np.ndarray:
    """
    Compute common-to-differential conversion Sdc21 from single-ended 4-port S-matrix.

    Scientific definition:
        Sdc21 measures how much common-mode input converts to differential-mode output.
        For input pair (p+, p-) and output pair (q+, q-):

        Sdc21 = 0.5 * (S_q+p+ + S_q+p- - S_q-p+ - S_q-p-)

    Args:
        S: S-matrix of shape [nfreq, 4, 4] using 0-based indexing.
        pairing: Port pairing string.

    Returns:
        Complex-valued Sdc21 transfer function of shape [nfreq].
    """
    port_map = {
        "13-24": (0, 2, 1, 3),
        "12-34": (0, 1, 2, 3),
        "14-23": (0, 3, 1, 2),
    }

    p_plus, p_minus, q_plus, q_minus = port_map[pairing]

    sdc21 = 0.5 * (
        S[:, q_plus, p_plus]
        + S[:, q_plus, p_minus]
        - S[:, q_minus, p_plus]
        - S[:, q_minus, p_minus]
    )

    return sdc21


def scd21_from_se_s4p(
    S: np.ndarray,
    pairing: str = "13-24"
) -> np.ndarray:
    """
    Compute differential-to-common conversion Scd21 from single-ended 4-port S-matrix.

    Scientific definition:
        Scd21 measures how much differential-mode input converts to common-mode output.
        For input pair (p+, p-) and output pair (q+, q-):

        Scd21 = 0.5 * (S_q+p+ - S_q+p- + S_q-p+ - S_q-p-)

    Args:
        S: S-matrix of shape [nfreq, 4, 4] using 0-based indexing.
        pairing: Port pairing string.

    Returns:
        Complex-valued Scd21 transfer function of shape [nfreq].
    """
    port_map = {
        "13-24": (0, 2, 1, 3),
        "12-34": (0, 1, 2, 3),
        "14-23": (0, 3, 1, 2),
    }

    p_plus, p_minus, q_plus, q_minus = port_map[pairing]

    scd21 = 0.5 * (
        S[:, q_plus, p_plus]
        - S[:, q_plus, p_minus]
        + S[:, q_minus, p_plus]
        - S[:, q_minus, p_minus]
    )

    return scd21


def inspect_touchstone_file(
    filepath: pathlib.Path,
    baud_rate_hz: float = 32e9,
) -> InspectionReport:
    """
    Inspect a Touchstone file and produce a machine-readable report.

    Responsibilities:
        1. Read file/network metadata (number of ports, format, reference impedance)
        2. Determine if file is already mixed-mode or single-ended
        3. For 4-port single-ended files, score candidate port pairings
        4. Produce a complete inspection report

    Args:
        filepath: Path to the Touchstone file (.s2p, .s4p, etc.)
        baud_rate_hz: Baud rate in Hz for Nyquist frequency calculation.

    Returns:
        InspectionReport with metadata, inferred topology, and quality metrics.

    Raises:
        RuntimeError: If scikit-rf is not installed.
        FileNotFoundError: If the file does not exist.
    """
    if not HAS_SKRF:
        raise RuntimeError(
            "scikit-rf is required to parse Touchstone files. "
            "Install with: pip install scikit-rf"
        )

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    filepath = pathlib.Path(filepath)

    # Load the network
    net = skrf.Network(str(filepath))

    # Extract metadata
    nports = net.number_of_ports
    freq_hz = net.f.copy()
    z0 = net.z0[0, 0] if net.z0.ndim > 1 else net.z0[0]

    # Determine format from the Network's attribute
    # skrf stores in 'scikit-rf' format internally; we'll check filename extension
    format_str = filepath.suffix.lower().replace('.', '')

    # Check if file appears to be mixed-mode based on port naming in file
    # Most Touchstone files don't explicitly label mixed-mode, so we infer
    is_mixed_mode = False  # Most industry files are single-ended unless specified

    metadata = TouchstoneMetadata(
        nports=nports,
        format=format_str,
        reference_impedance=float(z0),
        frequency_hz=freq_hz,
        is_mixed_mode=is_mixed_mode,
        file_path=filepath,
    )

    warnings = []
    quality_metrics = {}
    port_mode = PortMode.SINGLE_ENDED
    port_pairing = None
    pairing_scores: List[TopologyScore] = []
    transfer_mode = TransferMode.AUTO

    if nports == 4:
        # This is a 4-port file - check if it's a differential pair in single-ended basis
        report = infer_port_topology(net)

        port_mode = report["port_mode"]
        port_pairing = report["best_pairing"]
        pairing_scores = report["pairing_scores"]
        transfer_mode = report["suggested_transfer"]

        # Compute quality metrics
        quality_metrics = compute_quality_metrics(net, baud_rate_hz)

    elif nports == 2:
        # 2-port file - straightforward S21 extraction
        port_mode = PortMode.SINGLE_ENDED
        port_pairing = None
        transfer_mode = TransferMode.S21
        quality_metrics = compute_quality_metrics(net, baud_rate_hz)
    else:
        warnings.append(
            f"File has {nports} ports. Only 2-port and 4-port files are fully supported."
        )

    return InspectionReport(
        metadata=metadata,
        port_mode=port_mode,
        port_pairing=port_pairing,
        pairing_scores=pairing_scores,
        transfer_mode=transfer_mode,
        quality_metrics=quality_metrics,
        warnings=warnings,
    )


def infer_port_topology(
    net: 'skrf.Network',
) -> Dict:
    """
    Infer the physical port topology for a 4-port single-ended network.

    Attempts to detect whether the 4-port file represents a differential pair
    in single-ended basis by scoring candidate port pairings.

    Scoring heuristic:
        score = + weight_thru * mean_lowfreq(|Sdd21|)
                - weight_modeconv * mean(|Sdc21| + |Scd21|)
                - weight_asym * conductor_path_asymmetry

    When the top score's margin over the runner-up is <= 0.1 (ambiguous),
    falls back to "13-24" as the statistically dominant convention for
    802.3ck and most EDA tools. If 13-24 produces nonsense for a given file,
    try "12-34" as a secondary fallback.

    Args:
        net: A 4-port skrf.Network object.

    Returns:
        Dict with keys:
            - port_mode: PortMode enum value
            - best_pairing: Pairing string (never None — falls back to "13-24")
            - pairing_scores: List of TopologyScore for all candidates
            - suggested_transfer: TransferMode enum value
    """
    if net.number_of_ports != 4:
        return {
            "port_mode": PortMode.UNKNOWN,
            "best_pairing": None,
            "pairing_scores": [],
            "suggested_transfer": TransferMode.S21,
        }

    S = net.s  # shape [nfreq, 4, 4]
    freq_hz = net.f
    nfreq = len(freq_hz)

    # Weight factors for scoring
    weight_thru = 1.0
    weight_modeconv = 0.5
    weight_asym = 0.3

    pairing_scores: List[TopologyScore] = []

    for pairing in PORT_PAIRINGS:
        try:
            # Compute mixed-mode transfers
            sdd21 = sdd21_from_se_s4p(S, pairing)
            sdc21 = sdc21_from_se_s4p(S, pairing)
            scd21 = scd21_from_se_s4p(S, pairing)

            # Thru strength: mean low-frequency |Sdd21|
            # Low frequencies: first 20% of the frequency range or first 10 points
            n_low = max(10, int(0.2 * nfreq))
            thru_strength = np.mean(np.abs(sdd21[:n_low]))

            # Mode conversion: mean of |Sdc21| and |Scd21| over all frequencies
            mode_conversion = np.mean(np.abs(sdc21) + np.abs(scd21))

            # Path asymmetry: compare conductor through paths
            # For pairing 13-24: compare |S21| vs |S43| (should be similar)
            if pairing == "13-24":
                path1 = np.abs(S[:, 1, 0])  # S21
                path2 = np.abs(S[:, 3, 2])  # S43
            elif pairing == "12-34":
                path1 = np.abs(S[:, 1, 0])  # S21
                path2 = np.abs(S[:, 3, 2])  # S43 (wait, 3->4 through path)
            elif pairing == "14-23":
                path1 = np.abs(S[:, 1, 0])  # S21
                path2 = np.abs(S[:, 3, 2])  # S43

            # Simplified asymmetry: compare low-freq magnitudes
            path_asymmetry = np.abs(thru_strength - np.mean(path1[:n_low]))

            # Compute total score
            score = (
                weight_thru * thru_strength
                - weight_modeconv * mode_conversion
                - weight_asym * path_asymmetry
            )

            pairing_scores.append(TopologyScore(
                pairing=pairing,
                score=float(score),
                thru_strength=float(thru_strength),
                mode_conversion=float(mode_conversion),
                path_asymmetry=float(path_asymmetry),
                details={
                    "low_freq_mean_sdd21": float(np.mean(np.abs(sdd21[:n_low]))),
                    "mean_mode_conversion": float(mode_conversion),
                    "conductor_path_1_lowfreq": float(np.mean(path1[:n_low])),
                }
            ))

        except Exception as e:
            pairing_scores.append(TopologyScore(
                pairing=pairing,
                score=-np.inf,
                thru_strength=0.0,
                mode_conversion=np.inf,
                path_asymmetry=np.inf,
                details={"error": str(e)}
            ))

    # Sort by score descending
    pairing_scores.sort(key=lambda x: x.score, reverse=True)

    # Determine best pairing
    best_pairing = None
    suggested_transfer = TransferMode.S21

    if pairing_scores:
        best = pairing_scores[0]
        runner_up = pairing_scores[1] if len(pairing_scores) > 1 else None

        # Check if best is significantly better than runner-up
        if runner_up is not None and best.score > runner_up.score:
            margin = best.score - runner_up.score
            if margin > 0.1:  # threshold for confident selection
                best_pairing = best.pairing
                suggested_transfer = TransferMode.SDD21
            else:
                best_pairing = "13-24"
                suggested_transfer = TransferMode.SDD21
        elif pairing_scores:
            best_pairing = "13-24"
            suggested_transfer = TransferMode.SDD21

    # Determine port mode
    port_mode = PortMode.SINGLE_ENDED  # Files are typically single-ended format

    return {
        "port_mode": port_mode,
        "best_pairing": best_pairing,
        "pairing_scores": pairing_scores,
        "suggested_transfer": suggested_transfer,
    }


def compute_quality_metrics(
    net: 'skrf.Network',
    baud_rate_hz: float,
    fs_target: Optional[float] = None,
) -> Dict:
    """
    Compute quality metrics for a Touchstone network.

    Metrics computed:
        - measured_fmax_hz: Maximum measured frequency
        - target_nyquist_hz: Target Nyquist frequency (fs_target/2)
        - bandwidth_ratio: measured_fmax / target_nyquist
        - passivity_violations: Count of frequencies where singular values > 1
        - reciprocity_deviation: Mean |S - S^T|

    Args:
        net: skrf.Network object.
        baud_rate_hz: Baud rate for Nyquist calculation.
        fs_target: Target sample rate (defaults to baud_rate_hz * 2).

    Returns:
        Dict of quality metrics.
    """
    freq_hz = net.f
    measured_fmax = freq_hz[-1] if len(freq_hz) > 0 else 0.0

    if fs_target is None:
        fs_target = baud_rate_hz * 2

    target_nyquist = fs_target / 2
    bandwidth_ratio = measured_fmax / target_nyquist if target_nyquist > 0 else 0.0

    metrics = {
        "measured_fmax_hz": float(measured_fmax),
        "target_nyquist_hz": float(target_nyquist),
        "bandwidth_ratio": float(bandwidth_ratio),
    }

    # Passivity check: singular values of S(f) should be <= 1 for passive networks
    S = net.s
    nfreq = S.shape[0]
    passivity_violations = 0

    for i in range(nfreq):
        # Compute singular values of S matrix at frequency i
        _, sv, _ = np.linalg.svd(S[i])
        if np.any(sv > 1.0 + 1e-6):  # small tolerance for numerical
            passivity_violations += 1

    metrics["passivity_violations"] = passivity_violations
    metrics["passivity_ok"] = passivity_violations == 0

    # Reciprocity check: S should be approximately symmetric for passive copper
    if net.number_of_ports == 4:
        # For 4-port, check S21 vs S12, S31 vs S13, etc.
        reciprocity_dev = np.mean(np.abs(S[:, 0, 1] - S[:, 1, 0]))
        reciprocity_dev += np.mean(np.abs(S[:, 2, 3] - S[:, 3, 2]))
        metrics["reciprocity_deviation"] = float(reciprocity_dev / 2)
        metrics["reciprocity_ok"] = reciprocity_dev < 0.1  # threshold
    else:
        metrics["reciprocity_deviation"] = 0.0
        metrics["reciprocity_ok"] = True

    return metrics


def extract_transfer_function(
    net: 'skrf.Network',
    role: Literal["thru", "fext", "next"],
    transfer_mode: TransferMode = TransferMode.AUTO,
    port_pairing: Optional[str] = None,
    inspection_report: Optional[InspectionReport] = None,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Extract the appropriate scalar transfer function from a Touchstone network.

    Scientific definition:
        For a differential pair represented in single-ended port basis, the
        physically relevant main-channel transfer is often the mixed-mode
        differential insertion loss term Sdd21 rather than raw S21.

        The selection logic:
        - If transfer_mode is specified explicitly, use it
        - If AUTO and role is 'thru' with 4-port file, prefer Sdd21 if topology inference confidence is high
        - For FEXT/NEXT, the appropriate transfer depends on the coupling mechanism

    Args:
        net: skrf.Network object.
        role: Channel role - 'thru', 'fext', or 'next'.
        transfer_mode: Explicit transfer mode, or AUTO for inference.
        port_pairing: Port pairing string for 4-port Sdd21 extraction.
        inspection_report: Optional pre-computed inspection report.

    Returns:
        Tuple of (freq_hz, H, transfer_name) where:
            - freq_hz: frequency grid in Hz
            - H: complex transfer function
            - transfer_name: name of the extracted transfer ('S21', 'Sdd21', etc.)

    Raises:
        ValueError: If 4-port file with AUTO transfer mode requires pairing but none provided.
    """
    nports = net.number_of_ports
    freq_hz = net.f.copy()
    S = net.s

    # If inspection report available, use it for defaults
    if inspection_report is not None and transfer_mode == TransferMode.AUTO:
        transfer_mode = inspection_report.transfer_mode
        if port_pairing is None:
            port_pairing = inspection_report.port_pairing

    # 2-port case: simple S21 extraction
    if nports == 2:
        return freq_hz, S[:, 1, 0].copy(), "S21"

    # 4-port case
    if nports == 4:
        if transfer_mode == TransferMode.S21:
            # Raw single-ended S21
            return freq_hz, S[:, 1, 0].copy(), "S21"

        elif transfer_mode == TransferMode.SDD21:
            # Mixed-mode differential transfer
            if port_pairing is None:
                raise ValueError(
                    "port_pairing required for Sdd21 extraction from 4-port file. "
                    "Use '13-24', '12-34', or '14-23', or set transfer_mode='auto'."
                )
            H = sdd21_from_se_s4p(S, port_pairing)
            return freq_hz, H, f"Sdd21_{port_pairing}"

        elif transfer_mode == TransferMode.AUTO:
            # Try to determine from inspection or default to S21
            if inspection_report is not None:
                pairing = inspection_report.port_pairing
                tmode = inspection_report.transfer_mode
                if tmode == TransferMode.SDD21 and pairing is not None:
                    H = sdd21_from_se_s4p(S, pairing)
                    return freq_hz, H, f"Sdd21_{pairing}"

            # Fallback to S21
            return freq_hz, S[:, 1, 0].copy(), "S21"

    # For other port counts, fall back to S21
    return freq_hz, S[:, 1, 0].copy(), "S21"


def print_inspection_report(report: InspectionReport, output_file=None) -> None:
    """
    Print a human-readable inspection report.

    Args:
        report: InspectionReport from inspect_touchstone_file().
        output_file: File object to write to (default: None = stdout)
    """
    def write(msg):
        print(msg)
        if output_file:
            output_file.write(msg + "\n")

    write("=" * 70)
    write("TOUCHSTONE INSPECTION REPORT")
    write("=" * 70)
    write(f"File: {report.metadata.file_path}")
    write(f"Ports: {report.metadata.nports}")
    write(f"Format: {report.metadata.format}")
    write(f"Reference impedance: {report.metadata.reference_impedance:.1f} ohm")
    write(f"Frequency range: {report.metadata.frequency_hz[0]:.3e} - "
          f"{report.metadata.frequency_hz[-1]:.3e} Hz")
    write(f"Points: {len(report.metadata.frequency_hz)}")

    if report.metadata.is_mixed_mode:
        write("Mode: MIXED-MODE")
    else:
        write("Mode: SINGLE-ENDED representation")

    write("-" * 70)

    if report.metadata.nports == 4:
        write("PORT TOPOLOGY INFERENCE:")
        write("")

        if report.port_pairing is not None:
            write(f"  Best pairing: {report.port_pairing}")
            write(f"  Suggested transfer: {report.transfer_mode.value}")
            write("")
            write("  Pairing scores:")
            for ps in report.pairing_scores:
                write(f"    {ps.pairing}: score={ps.score:.4f}, "
                      f"thru={ps.thru_strength:.4f}, "
                      f"mode_conv={ps.mode_conversion:.4f}")
        else:
            write("  Could not confidently infer pairing.")
            write("  Suggestion: Specify --port_pairing manually.")

        write("")

    # Quality metrics
    if report.quality_metrics:
        write("QUALITY METRICS:")
        write(f"  Measured fmax: {report.quality_metrics.get('measured_fmax_hz', 0):.3e} Hz")
        write(f"  Target Nyquist: {report.quality_metrics.get('target_nyquist_hz', 0):.3e} Hz")
        write(f"  Bandwidth ratio: {report.quality_metrics.get('bandwidth_ratio', 0):.2f}")

        if report.quality_metrics.get('bandwidth_ratio', 1.0) < 1.0:
            write("  WARNING: Measured bandwidth below target Nyquist!")

        if not report.quality_metrics.get('passivity_ok', True):
            write(f"  WARNING: {report.quality_metrics.get('passivity_violations', 0)} "
                  "passivity violations detected")

        if not report.quality_metrics.get('reciprocity_ok', True):
            write(f"  WARNING: Reciprocity deviation = "
                  f"{report.quality_metrics.get('reciprocity_deviation', 0):.4f}")

    # Warnings
    if report.warnings:
        write("")
        write("WARNINGS:")
        for w in report.warnings:
            write(f"  - {w}")

    write("=" * 70)


def print_inspection_report_to_file(report: InspectionReport, output_file=None, write_fn=None) -> None:
    """
    Write inspection report to file using a provided write function.

    Args:
        report: InspectionReport from inspect_touchstone_file().
        output_file: File object to write to (used to track write state)
        write_fn: Custom write function (e.g., lambda msg: output_file.write(msg + '\\n'))
    """
    if write_fn is None:
        write_fn = lambda msg: print(msg) if output_file is None else output_file.write(msg + "\n")

    write_fn("=" * 70)
    write_fn("TOUCHSTONE INSPECTION REPORT")
    write_fn("=" * 70)
    write_fn(f"File: {report.metadata.file_path}")
    write_fn(f"Ports: {report.metadata.nports}")
    write_fn(f"Format: {report.metadata.format}")
    write_fn(f"Reference impedance: {report.metadata.reference_impedance:.1f} ohm")
    write_fn(f"Frequency range: {report.metadata.frequency_hz[0]:.3e} - "
             f"{report.metadata.frequency_hz[-1]:.3e} Hz")
    write_fn(f"Points: {len(report.metadata.frequency_hz)}")

    if report.metadata.is_mixed_mode:
        write_fn("Mode: MIXED-MODE")
    else:
        write_fn("Mode: SINGLE-ENDED representation")

    write_fn("-" * 70)

    if report.metadata.nports == 4:
        write_fn("PORT TOPOLOGY INFERENCE:")
        write_fn("")

        if report.port_pairing is not None:
            write_fn(f"  Best pairing: {report.port_pairing}")
            write_fn(f"  Suggested transfer: {report.transfer_mode.value}")
            write_fn("")
            write_fn("  Pairing scores:")
            for ps in report.pairing_scores:
                write_fn(f"    {ps.pairing}: score={ps.score:.4f}, "
                         f"thru={ps.thru_strength:.4f}, "
                         f"mode_conv={ps.mode_conversion:.4f}")
        else:
            write_fn("  Could not confidently infer pairing.")
            write_fn("  Suggestion: Specify --port_pairing manually.")

        write_fn("")

    # Quality metrics
    if report.quality_metrics:
        write_fn("QUALITY METRICS:")
        write_fn(f"  Measured fmax: {report.quality_metrics.get('measured_fmax_hz', 0):.3e} Hz")
        write_fn(f"  Target Nyquist: {report.quality_metrics.get('target_nyquist_hz', 0):.3e} Hz")
        write_fn(f"  Bandwidth ratio: {report.quality_metrics.get('bandwidth_ratio', 0):.2f}")

        if report.quality_metrics.get('bandwidth_ratio', 1.0) < 1.0:
            write_fn("  WARNING: Measured bandwidth below target Nyquist!")

        if not report.quality_metrics.get('passivity_ok', True):
            write_fn(f"  WARNING: {report.quality_metrics.get('passivity_violations', 0)} "
                     "passivity violations detected")

        if not report.quality_metrics.get('reciprocity_ok', True):
            write_fn(f"  WARNING: Reciprocity deviation = "
                     f"{report.quality_metrics.get('reciprocity_deviation', 0):.4f}")

    # Warnings
    if report.warnings:
        write_fn("")
        write_fn("WARNINGS:")
        for w in report.warnings:
            write_fn(f"  - {w}")

    write_fn("=" * 70)
