"""
Script to compare input features between AGC and No-AGC approaches.

This demonstrates how channel normalization affects the scale of state features
that the learned optimizer (MLP) receives during training/inference.
"""

import torch
import numpy as np
from wireline_channel import WirelineChannelGenerator
from advanced_channel_gen import AdvancedWirelineChannelGenerator

# Same config as used in training
BATCH_SIZE = 4
SEQ_LEN = 500
DFE_TAPS = 10
FFE_TAPS = 15
FFE_MAIN_CURSOR = 7
DFE_INIT = 0.01

def run_pipeline(channel_gen, batch_size=BATCH_SIZE, seq_len=SEQ_LEN, num_steps=50):
    """
    Run a few steps of the equalizer pipeline and collect state features.
    Returns statistics about the feature magnitudes.
    """
    tx_symbols = torch.sign(torch.randn(batch_size, seq_len))
    rx_base, h_batch = channel_gen.generate_received_signal(tx_symbols, batch_size)

    print(f"\n{'='*60}")
    print(f"Channel Generator: {channel_gen.__class__.__name__}")
    print(f"  disable_agc: {getattr(channel_gen, 'disable_agc', False)}")
    print(f"{'='*60}")

    # Print channel statistics
    print(f"\nChannel IR statistics:")
    print(f"  h_batch shape: {h_batch.shape}")
    print(f"  h_batch norm per sample: {torch.norm(h_batch, dim=1)}")
    print(f"  h_batch max |peak| per sample: {torch.max(torch.abs(h_batch), dim=1)[0]}")
    print(f"  h_batch mean magnitude: {torch.mean(torch.abs(h_batch)):.6f}")

    # Simple NLMS initialization
    dfe_weights = torch.zeros(batch_size, DFE_TAPS)
    ffe_buffer = torch.zeros(batch_size, FFE_TAPS)
    ffe_buffer[:, FFE_MAIN_CURSOR] = 1.0  # FFE_INIT equivalent
    decision_buffer = torch.zeros(batch_size, DFE_TAPS)
    ema_error = torch.ones(batch_size, 1) * 0.1

    # Collect feature statistics
    e_t_magnitudes = []
    norm_sq_values = []
    ffe_out_magnitudes = []
    dfe_feedback_magnitudes = []

    # Run a few steps starting from a reasonable point (after warmup)
    start_idx = 50
    for t in range(start_idx, start_idx + num_steps):
        # Simulate received signal at this step (simplified)
        rx_t = rx_base[:, t:t+1]

        # Update FFE buffer
        ffe_buffer = torch.roll(ffe_buffer, shifts=1, dims=1)
        ffe_buffer[:, 0] = rx_t.squeeze(-1)

        # FFE output
        ffe_out = torch.sum(ffe_buffer * torch.zeros_like(ffe_buffer), dim=1, keepdim=True)
        ffe_out = torch.sum(ffe_buffer * ffe_buffer, dim=1, keepdim=True) * 0.1  # Simplified

        # DFE feedback
        dfe_feedback = torch.sum(decision_buffer * dfe_weights, dim=1, keepdim=True)

        # Simulated error (using rx_t as proxy for received signal)
        # In real pipeline, error = tx_symbol - (ffe_out - dfe_feedback)
        target_symbol = tx_symbols[:, t:t+1]
        # Simulated y_out for demonstration
        y_out = rx_t[:, :1] * 0.5  # Very rough approximation
        e_t = target_symbol[:, :1] - y_out

        # NLMS normalization
        norm_sq = torch.sum(ffe_buffer ** 2, dim=1, keepdim=True) + \
                  torch.sum(decision_buffer ** 2, dim=1, keepdim=True) + 1e-6

        # Collect statistics
        e_t_magnitudes.append(torch.abs(e_t).mean().item())
        norm_sq_values.append(norm_sq.mean().item())
        ffe_out_magnitudes.append(torch.abs(ffe_out).mean().item())
        dfe_feedback_magnitudes.append(torch.abs(dfe_feedback).mean().item())

        # Gradient proxy for CTLE
        grad_proxy_ctle = e_t * ffe_buffer[:, FFE_MAIN_CURSOR:FFE_MAIN_CURSOR+1] / norm_sq

        # Build full state feature vector (same as in training)
        state_features = torch.cat([
            e_t,
            ema_error,
            norm_sq,
            dfe_weights[:, 0:1],
            torch.tensor([[0.5]]).expand(batch_size, 1),  # ctle_peaking
            grad_proxy_ctle
        ], dim=1)

        if t == start_idx:
            print(f"\nFirst state_features tensor (batch_size=4, features=6):")
            print(f"  Shape: {state_features.shape}")
            for i, name in enumerate(['e_t', 'ema_error', 'norm_sq', 'dfe_w[0]', 'ctle', 'grad_proxy']):
                feat_vals = state_features[0, i].item()
                print(f"    [{name}]: {feat_vals:.6f}")

    # Print aggregate statistics
    print(f"\nFeature Statistics over {num_steps} steps (mean across batch):")
    print(f"  |e_t| mean: {np.mean(e_t_magnitudes):.6f} (varies with channel energy)")
    print(f"  norm_sq mean: {np.mean(norm_sq_values):.6f} (scales with channel loss!)")
    print(f"  |ffe_out| mean: {np.mean(ffe_out_magnitudes):.6f}")
    print(f"  |dfe_feedback| mean: {np.mean(dfe_feedback_magnitudes):.6f}")

    # Compute effective gradient scaling (what NLMS would actually do)
    print(f"\nEffective gradient scaling (e_t / norm_sq):")
    effective_scaling = [e/n for e, n in zip(e_t_magnitudes, norm_sq_values)]
    print(f"  mean: {np.mean(effective_scaling):.8f}")
    print(f"  std:  {np.std(effective_scaling):.8f}")

    return {
        'e_t': e_t_magnitudes,
        'norm_sq': norm_sq_values,
        'effective_scaling': effective_scaling
    }


def compare_channels():
    """Compare how channel loss affects features."""

    print("\n" + "="*70)
    print("DEMONSTRATION: How AGC Normalization Affects Feature Scales")
    print("="*70)

    # Test with different channel strengths by modifying SNR
    print("\n" + "-"*70)
    print("Test 1: Standard channel with AGC (default)")
    print("-"*70)

    gen_agc = WirelineChannelGenerator(num_taps=50, snr_range=(20, 20), disable_agc=False)
    stats_agc = run_pipeline(gen_agc)

    print("\n" + "-"*70)
    print("Test 2: Standard channel WITHOUT AGC (raw physics)")
    print("-"*70)

    gen_no_agc = WirelineChannelGenerator(num_taps=50, snr_range=(20, 20), disable_agc=True)
    stats_no_agc = run_pipeline(gen_no_agc)

    print("\n" + "="*70)
    print("KEY OBSERVATION")
    print("="*70)

    mean_e_agc = np.mean(stats_agc['e_t'])
    mean_e_no_agc = np.mean(stats_no_agc['e_t'])
    mean_norm_agc = np.mean(stats_agc['norm_sq'])
    mean_norm_no_agc = np.mean(stats_no_agc['norm_sq'])

    print(f"\nWith AGC:    |e_t| = {mean_e_agc:.6f},  norm_sq = {mean_norm_agc:.6f}")
    print(f"Without AGC: |e_t| = {mean_e_no_agc:.6f},  norm_sq = {mean_norm_no_agc:.6f}")

    if mean_norm_no_agc > 0:
        ratio = mean_norm_agc / mean_norm_no_agc
        print(f"\nnorm_sq ratio (AGC/no-AGC): {ratio:.4f}")
        print(f"This means the NLMS effective step size is ~{1/ratio:.4f}x larger WITHOUT AGC")
        print(f"(assuming the MLP outputs the same mu_dfe in both cases)")

    print("\n" + "="*70)
    print("WHY THIS MATTERS FOR THE LEARNED OPTIMIZER")
    print("="*70)
    print("""
The MLP learns mu_dfe (step size) based on the feature patterns it sees in training.

With AGC:
  - All channels have ~same energy (norm_sq is predictable)
  - MLP learns step sizes that work well for this stable regime

Without AGC:
  - Channel energy varies (some channels lose 20dB = 100x less energy)
  - norm_sq varies proportionally
  - Same mu_dfe produces different effective updates per channel
  - MLP can't learn a single step size that works across all channels

The LayerNorm only normalizes the MLP's internal activations - it cannot
correct for the fact that the effective gradient magnitude varies with
channel strength in the actual NLMS update equation.
""")


if __name__ == "__main__":
    torch.manual_seed(42)
    compare_channels()
