import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
from config import *

# Import architectures from training scripts
from l2o_basic import MultiRateLearnedNLMS, DifferentiableCTLE, DifferentiableDFE, cross_correlate_sync_batch
from wireline_channel import WirelineChannelGenerator
from l2o_mlp import MultiRateLearnedMLP
from benchmark_nlms import run_batch_nlms_dfe, run_batch_rls_dfe

def run_l2o_inference(model, model_type, channel_gen, ctle, dfe, batch_size=100, seq_len=500, ctle_peaking=0.5):
    """
    Evaluates a trained L2O model on a fresh batch of channels without weight updates.
    """
    model.eval()
    tx_symbols = torch.sign(torch.randn(batch_size, seq_len))
    rx_base, _ = channel_gen.generate_received_signal(tx_symbols, batch_size)

    with torch.no_grad():
        rx_init = ctle(rx_base, torch.ones(batch_size, 1) * ctle_peaking)
        batch_delays = cross_correlate_sync_batch(tx_symbols, rx_init)
    
    common_delay = int(torch.median(torch.tensor(batch_delays, dtype=torch.float)).item())
    
    # Initialize state
    hidden_state = torch.zeros(batch_size, 32) if model_type != "mlp" else None
    history_buffer = torch.zeros(batch_size, L2O_MLP_HISTORY_LEN, L2O_STATE_DIM) if model_type == "mlp" else None
    
    dfe_weights = torch.zeros(batch_size, dfe.num_taps)
    w_main = torch.ones(batch_size, 1) * FFE_INIT
    latent_peaking = torch.zeros(batch_size, 1)
    rx_buffer = torch.zeros(batch_size, ctle.num_taps)
    decision_buffer = torch.zeros(batch_size, dfe.num_taps)
    ema_error = torch.ones(batch_size, 1)
    
    mse_history = []
    
    with torch.no_grad():
        effective_len = seq_len - common_delay
        for t in range(effective_len):
            rx_t = rx_base[:, (t + common_delay):(t + common_delay + 1)]
            ctle_peaking = torch.sigmoid(latent_peaking)
            
            rx_buffer = torch.roll(rx_buffer, shifts=1, dims=1)
            rx_buffer[:, 0] = rx_t.squeeze(-1)
            
            current_taps = ctle.base_lp.unsqueeze(0) + ctle_peaking * ctle.base_hp.unsqueeze(0)
            rx_eq = torch.sum(rx_buffer * current_taps, dim=1, keepdim=True)
            
            dfe_feedback = torch.sum(decision_buffer * dfe_weights, dim=1, keepdim=True)
            y_out = (w_main * rx_eq) - dfe_feedback
            e_t = tx_symbols[:, t:t+1] - y_out
            
            norm_sq = (rx_eq ** 2) + torch.sum(decision_buffer ** 2, dim=1, keepdim=True) + 1e-6
            grad_proxy_ctle = e_t * rx_eq / norm_sq
            
            state_features = torch.cat([
                e_t, ema_error, norm_sq,
                dfe_weights[:, 0:1],
                ctle_peaking,
                grad_proxy_ctle
            ], dim=1)
            
            if model_type == "mlp":
                history_buffer = torch.roll(history_buffer, shifts=1, dims=1)
                history_buffer[:, 0, :] = state_features
                mu_dfe, mu_ctle = model(history_buffer.view(batch_size, -1), update_ctle=(t % 10 == 0))
            else:
                mu_dfe, mu_ctle, hidden_state = model(state_features, hidden_state, update_ctle=(t % 10 == 0))
            
            ema_error = 0.95 * ema_error + 0.05 * (e_t ** 2)
            dfe_weights = dfe_weights - (mu_dfe * e_t * decision_buffer / norm_sq)
            w_main = w_main + (mu_dfe * e_t * rx_eq / norm_sq)
            
            if t % 10 == 0:
                latent_peaking = latent_peaking + mu_ctle
                
            tau = 0.1
            decision = torch.tanh(y_out / tau) # Match meta-training distribution
            decision_buffer = torch.roll(decision_buffer, shifts=1, dims=1)
            decision_buffer[:, 0] = decision.squeeze(-1)
            
            mse_history.append(torch.mean(e_t**2).item())
            
    return torch.tensor(mse_history)

if __name__ == "__main__":
    # Parse command-line arguments (fall back to config defaults)
    parser = argparse.ArgumentParser(description="Evaluate L2O models against baselines")
    parser.add_argument("--batch_size", type=int, default=EVAL_BATCH_SIZE,
                        help=f"Batch size for evaluation (default: {EVAL_BATCH_SIZE})")
    parser.add_argument("--seq_len", type=int, default=EVAL_SEQ_LENGTH,
                        help=f"Sequence length for evaluation (default: {EVAL_SEQ_LENGTH})")
    parser.add_argument("--burn_in", type=int, default=EVAL_BURN_IN,
                        help=f"Steady-state start index (default: {EVAL_BURN_IN})")
    parser.add_argument("--ctle_peaking", type=float, default=FIXED_CTLE_PEAKING,
                        help=f"CTLE peaking gain (default: {FIXED_CTLE_PEAKING})")
    args = parser.parse_args()

    batch_size = args.batch_size
    seq_len = args.seq_len
    burn_in = args.burn_in
    ctle_peaking = args.ctle_peaking

    # Use config values only if they meet minimum thresholds; otherwise use defaults
    DEFAULT_SEQ_LEN = 500
    DEFAULT_BURN_IN = 200
    if seq_len < DEFAULT_SEQ_LEN or burn_in < DEFAULT_BURN_IN:
        print(f"Warning: Config values (seq_len={seq_len}, burn_in={burn_in}) below minimum thresholds.")
        print(f"         Using default values (seq_len={DEFAULT_SEQ_LEN}, burn_in={DEFAULT_BURN_IN})")
        seq_len = DEFAULT_SEQ_LEN
        burn_in = DEFAULT_BURN_IN

    torch.manual_seed(42)
    print("Starting Comparative Evaluation (Inference Mode)...")
    print(f"Settings: batch_size={batch_size}, seq_len={seq_len}, burn_in={burn_in}")
    print("-" * 60)

    # Setup environment
    gen = WirelineChannelGenerator(num_taps=CH_TAPS, snr_range=SNR_RANGE)
    ctle = DifferentiableCTLE(num_taps=CTLE_TAPS)
    dfe = DifferentiableDFE(num_taps=DFE_TAPS)

    # Generate batch data once - will be reused for both L2O and baseline evaluation
    tx_symbols = torch.sign(torch.randn(batch_size, seq_len))
    rx_base, _ = gen.generate_received_signal(tx_symbols, batch_size)
    rx_init = ctle(rx_base, torch.ones(batch_size, 1) * 0.5)

    # Synchronize using cross-correlation to find common delay
    batch_delays = cross_correlate_sync_batch(tx_symbols, rx_init)
    common_delay = int(torch.median(torch.tensor(batch_delays, dtype=torch.float)).item())

    # Align data using common delay (same as L2O evaluation)
    tx_aligned = tx_symbols[:, common_delay:]
    rx_aligned = rx_init[:, common_delay:]

    print(f"Batch size: {batch_size}, Common delay: {common_delay}")
    print(f"Aligned sequence length: {tx_aligned.shape[1]}")
    print("-" * 60)

    results = {}

    # 1. Evaluate L2O Models
    models_to_test = [
        ("RNN Basic", "l2o_basic_model.pth", "rnn", MultiRateLearnedNLMS(6, 32)),
        ("MLP", "l2o_mlp_model.pth", "mlp", MultiRateLearnedMLP(L2O_STATE_DIM, L2O_MLP_HISTORY_LEN, L2O_MLP_HIDDEN_DIM)),
        ("RNN Progressive", "l2o_progressive_model.pth", "rnn", MultiRateLearnedNLMS(6, 32))
    ]

    plt.figure(figsize=(12, 7))

    for name, path, mtype, model in models_to_test:
        try:
            model.load_state_dict(torch.load(path))
            print(f"Loaded {name} from {path}")
            mse_trace = run_l2o_inference(model, mtype, gen, ctle, dfe, batch_size=batch_size, seq_len=seq_len, ctle_peaking=ctle_peaking)
            avg_mse = torch.mean(mse_trace).item()
            ss_mse = torch.mean(mse_trace[burn_in:]).item()
            results[name] = (avg_mse, ss_mse)

            smoothed = pd.Series(mse_trace).ewm(span=20).mean()
            plt.plot(10 * torch.log10(torch.tensor(smoothed)), label=f"{name} (SS: {10*np.log10(ss_mse):.2f} dB)")
        except FileNotFoundError:
            print(f"Skipping {name}: weight file not found.")

    # 2. Run Benchmarks on the SAME batch data (fair comparison)
    # NLMS Bench (using batch wrapper)
    avg_mse_nlms, _ = run_batch_nlms_dfe(
        rx_aligned, tx_aligned, num_taps=DFE_TAPS, mu=0.05, teacher_forcing=False  # Matched to L2O head scale
    )
    results["NLMS (0.05)"] = (torch.mean(avg_mse_nlms).item(), torch.mean(avg_mse_nlms[burn_in:]).item())
    smoothed_nlms = pd.Series(avg_mse_nlms.numpy()).ewm(span=20).mean()
    plt.plot(10 * torch.log10(torch.tensor(smoothed_nlms)), '--', label="NLMS mu=0.05", alpha=0.6)

    # Gear-Shift NLMS Bench
    avg_mse_gs, _ = run_batch_nlms_dfe(
        rx_aligned, tx_aligned, num_taps=DFE_TAPS, mu=0.05, teacher_forcing=False,
        use_gear_shift=True, mu_fast=GEAR_SHIFT_MU_FAST, mu_slow=GEAR_SHIFT_MU_SLOW,
        gear_threshold=GEAR_SHIFT_THRESHOLD, ema_alpha=GEAR_SHIFT_EMA_ALPHA
    )
    results["NLMS (Gear-Shift)"] = (torch.mean(avg_mse_gs).item(), torch.mean(avg_mse_gs[burn_in:]).item())
    smoothed_gs = pd.Series(avg_mse_gs.numpy()).ewm(span=20).mean()
    plt.plot(10 * torch.log10(torch.tensor(smoothed_gs)), '--', label="NLMS Gear-Shift", alpha=0.6)

    # RLS Bench (using batch wrapper)
    avg_mse_rls, _ = run_batch_rls_dfe(
        rx_aligned, tx_aligned, num_taps=DFE_TAPS, lam=RLS_LAMBDA, delta=RLS_DELTA, teacher_forcing=False
    )
    results["RLS"] = (torch.mean(avg_mse_rls).item(), torch.mean(avg_mse_rls[burn_in:]).item())
    smoothed_rls = pd.Series(avg_mse_rls.numpy()).ewm(span=20).mean()
    plt.plot(10 * torch.log10(torch.tensor(smoothed_rls)), 'k:', label="RLS Baseline", linewidth=2)

    # Final Reporting
    print("\nComparison Table (Evaluation on {} symbols, batch-avg):".format(seq_len))
    print(f"{'Method':<20} | {'Avg MSE (dB)':<15} | {'SS MSE (dB)':<15}")
    print("-" * 55)
    for name, (avg, ss) in results.items():
        print(f"{name:<20} | {10*np.log10(avg):>12.2f} dB | {10*np.log10(ss):>12.2f} dB")

    plt.axhline(y=-20, color='r', linestyle='--', label='Target -20dB')
    plt.title(f"Inference Comparison: Learned Optimizers vs Benchmarks\n(Evaluated on {seq_len} symbols, CH={CH_TAPS}, DFE={DFE_TAPS})")
    plt.xlabel("Symbols")
    plt.ylabel("MSE (dB)")
    plt.legend()
    plt.grid(True)
    plt.show()
