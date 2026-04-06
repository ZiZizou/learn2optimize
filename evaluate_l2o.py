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
from l2o_mlp import MultiRateLearnedMLP
from l2o_mlp_no_agc import MultiRateLearnedMLPNoAGC
from benchmark_nlms import run_batch_nlms_dfe, run_batch_rls_dfe
from utils import add_channel_args, get_channel_generator
from ctle_frequency_utils import apply_frequency_domain_ctle

def run_l2o_inference(model, model_type, rx_base, tx_symbols, ctle, dfe, ctle_peaking=0.5, ablate_ctle=False):
    """
    Evaluates a trained L2O model on a pre-generated batch of channels without weight updates.

    Args:
        rx_base: Pre-generated received signal tensor [batch_size, seq_len]
        tx_symbols: Pre-generated TX symbols tensor [batch_size, seq_len]
        ablate_ctle: If True, skip CTLE updates (use fixed CTLE) to match training with ABLATE_CTLE=True
    """
    model.eval()
    batch_size, seq_len = tx_symbols.shape

    with torch.no_grad():
        if ablate_ctle:
            # Use continuous-time serdespy CTLE (Static LTI pre-filtering)
            rx_init = apply_frequency_domain_ctle(rx_base, peaking_gain=ctle_peaking)
        else:
            rx_init = ctle(rx_base, torch.ones(batch_size, 1) * ctle_peaking)
        batch_delays = cross_correlate_sync_batch(tx_symbols, rx_init)

    common_delay = int(torch.median(torch.tensor(batch_delays, dtype=torch.float)).item())

    # Initialize state
    hidden_state = torch.zeros(batch_size, 32) if model_type != "mlp" else None
    history_buffer = torch.zeros(batch_size, L2O_MLP_HISTORY_LEN, L2O_STATE_DIM) if model_type == "mlp" else None

    dfe_weights = torch.zeros(batch_size, dfe.num_taps)
    # Initialize Multi-Tap FFE
    w_ffe = torch.zeros(batch_size, FFE_TAPS)
    w_ffe[:, FFE_MAIN_CURSOR] = FFE_INIT
    ffe_buffer = torch.zeros(batch_size, FFE_TAPS)
    latent_peaking = torch.zeros(batch_size, 1)
    rx_buffer = torch.zeros(batch_size, ctle.num_taps)
    decision_buffer = torch.zeros(batch_size, dfe.num_taps)
    ema_error = torch.ones(batch_size, 1)

    mse_history = []
    mu_history = []  # Track step sizes for analysis

    with torch.no_grad():
        effective_len = seq_len - common_delay
        for t in range(effective_len):
            if ablate_ctle:
                # O(1) fetch from pre-computed continuous-time waveform
                # The CTLE was already applied as a static LTI pre-filter
                rx_eq = rx_init[:, (t + common_delay):(t + common_delay + 1)]

                # CTLE is static; use fixed peaking gain
                # Must be [batch_size, 1] to match tensor shapes in torch.cat
                ctle_peaking = torch.full((batch_size, 1), 0.5, device=latent_peaking.device)
            else:
                rx_t = rx_base[:, (t + common_delay):(t + common_delay + 1)]
                ctle_peaking = torch.sigmoid(latent_peaking)

                rx_buffer = torch.roll(rx_buffer, shifts=1, dims=1)
                rx_buffer[:, 0] = rx_t.squeeze(-1)

                current_taps = ctle.base_lp.unsqueeze(0) + ctle_peaking * ctle.base_hp.unsqueeze(0)
                rx_eq = torch.sum(rx_buffer * current_taps, dim=1, keepdim=True)

            # ============================================================
            # Multi-Tap FFE with Causality Shift
            # ============================================================
            # Shift FFE buffer and insert newest rx_eq at index 0
            ffe_buffer = torch.roll(ffe_buffer, shifts=1, dims=1)
            ffe_buffer[:, 0] = rx_eq.squeeze(-1)

            # DFE feedback computation
            dfe_feedback = torch.sum(decision_buffer * dfe_weights, dim=1, keepdim=True)

            # Causality shift: We make decisions for symbol at (t - FFE_MAIN_CURSOR)
            target_idx = t - FFE_MAIN_CURSOR

            if target_idx >= 0:
                # Compute FFE output (dot product of weights and buffer)
                ffe_out = torch.sum(ffe_buffer * w_ffe, dim=1, keepdim=True)
                # Total equalizer output
                y_out = ffe_out - dfe_feedback

                target_symbol = tx_symbols[:, target_idx:target_idx + 1]
                e_t = target_symbol - y_out

                norm_sq = torch.sum(ffe_buffer ** 2, dim=1, keepdim=True) + \
                          torch.sum(decision_buffer ** 2, dim=1, keepdim=True) + 1e-6
                grad_proxy_ctle = e_t * ffe_buffer[:, FFE_MAIN_CURSOR:FFE_MAIN_CURSOR+1] / norm_sq

                state_features = torch.cat([
                    e_t, ema_error, norm_sq,
                    dfe_weights[:, 0:1],
                    ctle_peaking,
                    grad_proxy_ctle
                ], dim=1)

                if model_type == "mlp":
                    history_buffer = torch.roll(history_buffer, shifts=1, dims=1)
                    history_buffer[:, 0, :] = state_features
                    mu_dfe, mu_ctle, _ = model(history_buffer.view(batch_size, -1), update_ctle=(t % 10 == 0))
                else:
                    mu_dfe, mu_ctle, _, hidden_state = model(state_features, hidden_state, update_ctle=(t % 10 == 0))

                # Ablation: Zero out CTLE updates to match training with ABLATE_CTLE=True
                if ablate_ctle:
                    mu_ctle = torch.zeros_like(mu_ctle)

                ema_error = 0.95 * ema_error + 0.05 * (e_t ** 2)
                dfe_weights = dfe_weights - (mu_dfe * e_t * decision_buffer / norm_sq)
                w_ffe = w_ffe + (mu_dfe * e_t * ffe_buffer / norm_sq)

                if t % 10 == 0:
                    latent_peaking = latent_peaking + mu_ctle

                tau = 0.1
                decision = torch.tanh(y_out / tau) # Match meta-training distribution
                decision_buffer = torch.roll(decision_buffer, shifts=1, dims=1)
                decision_buffer[:, 0] = decision.squeeze(-1)

                mse_history.append(torch.mean(e_t**2).item())
                mu_history.append(torch.mean(mu_dfe).item())
            else:
                # Warmup phase: not enough history for valid error computation
                # Build state features with zero error placeholder
                norm_sq = torch.sum(ffe_buffer ** 2, dim=1, keepdim=True) + \
                          torch.sum(decision_buffer ** 2, dim=1, keepdim=True) + 1e-6
                grad_proxy_ctle = torch.zeros_like(ffe_buffer[:, FFE_MAIN_CURSOR:FFE_MAIN_CURSOR+1])

                state_features = torch.cat([
                    torch.zeros_like(ema_error),  # placeholder e_t = 0
                    ema_error,
                    norm_sq,
                    dfe_weights[:, 0:1],
                    ctle_peaking,
                    grad_proxy_ctle
                ], dim=1)

                if model_type == "mlp":
                    history_buffer = torch.roll(history_buffer, shifts=1, dims=1)
                    history_buffer[:, 0, :] = state_features
                    mu_dfe, mu_ctle, _ = model(history_buffer.view(batch_size, -1), update_ctle=(t % 10 == 0))
                else:
                    mu_dfe, mu_ctle, _, hidden_state = model(state_features, hidden_state, update_ctle=(t % 10 == 0))

                if ablate_ctle:
                    mu_ctle = torch.zeros_like(mu_ctle)

                if t % 10 == 0:
                    latent_peaking = latent_peaking + mu_ctle

                # Use zero decision during warmup
                decision_buffer = torch.roll(decision_buffer, shifts=1, dims=1)
                decision_buffer[:, 0] = 0

    return torch.tensor(mse_history), w_ffe, dfe_weights, torch.tensor(mu_history)

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
    parser.add_argument("--ablate_ctle", action="store_true",
                        help="Enable ablation mode for models trained with ABLATE_CTLE=True")
    parser.add_argument("--two_head", action="store_true",
                        help="Evaluate models trained with two-head overdrive architecture")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to a specific trained model file. If not provided, "
                             "auto-generates paths based on channel_type and ablation settings.")
    # Strategy selection flags (default all to True for backward compatibility)
    parser.add_argument("--include_l2o", action="store_true", default=True,
                        help="Include L2O models in evaluation (default: True)")
    parser.add_argument("--no_l2o", action="store_true",
                        help="Exclude all L2O models from evaluation")
    parser.add_argument("--no_rnn_basic", action="store_true",
                        help="Exclude RNN Basic model")
    parser.add_argument("--no_mlp", action="store_true",
                        help="Exclude MLP model")
    parser.add_argument("--no_rnn_progressive", action="store_true",
                        help="Exclude RNN Progressive model")
    parser.add_argument("--include_nlms", action="store_true", default=True,
                        help="Include static NLMS benchmark (default: True)")
    parser.add_argument("--no_nlms", action="store_true",
                        help="Exclude static NLMS benchmark")
    parser.add_argument("--include_vss", action="store_true", default=True,
                        help="Include VSS NLMS benchmark (default: True)")
    parser.add_argument("--no_vss", action="store_true",
                        help="Exclude VSS NLMS benchmark")
    parser.add_argument("--include_rls", action="store_true", default=True,
                        help="Include RLS benchmark (default: True)")
    parser.add_argument("--no_rls", action="store_true",
                        help="Exclude RLS benchmark")
    parser.add_argument("--no_baseline", action="store_true",
                        help="Exclude all baseline algorithms (NLMS, VSS, RLS)")
    parser = add_channel_args(parser)
    args = parser.parse_args()

    batch_size = args.batch_size
    seq_len = args.seq_len
    burn_in = args.burn_in
    ctle_peaking = args.ctle_peaking
    ablate_ctle = args.ablate_ctle
    model_path = args.model_path

    # Process strategy selection flags
    include_l2o = args.include_l2o and not args.no_l2o
    include_nlms = args.include_nlms and not args.no_nlms and not args.no_baseline
    include_vss = args.include_vss and not args.no_vss and not args.no_baseline
    include_rls = args.include_rls and not args.no_rls and not args.no_baseline

    # L2O model exclusion flags
    exclude_rnn_basic = args.no_rnn_basic
    exclude_mlp = args.no_mlp
    exclude_rnn_progressive = args.no_rnn_progressive

    print(f"Strategy selection: L2O={include_l2o}, NLMS={include_nlms}, VSS={include_vss}, RLS={include_rls}")
    print(f"L2O exclusions: RNN_Basic={exclude_rnn_basic}, MLP={exclude_mlp}, RNN_Progressive={exclude_rnn_progressive}")

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
    print(f"Channel type: {args.channel_type}")
    if args.disable_agc:
        print("*** NO-AGC MODE: Channel normalization disabled - evaluating with raw physics-preserved voltages ***")
    if ablate_ctle:
        print("*** ABLATION MODE: CTLE updates disabled (matching ABLATE_CTLE=True training) ***")
    print("-" * 60)

    # Setup environment
    gen = get_channel_generator(args)
    ctle = DifferentiableCTLE(num_taps=CTLE_TAPS)
    dfe = DifferentiableDFE(num_taps=DFE_TAPS)

    # Generate batch data once - will be reused for both L2O and baseline evaluation
    tx_symbols = torch.sign(torch.randn(batch_size, seq_len))
    rx_base, _ = gen.generate_received_signal(tx_symbols, batch_size)

    if ablate_ctle:
        print("Applying continuous-time SerDesPy CTLE for evaluation...")
        rx_init = apply_frequency_domain_ctle(rx_base, peaking_gain=ctle_peaking)
    else:
        rx_init = ctle(rx_base, torch.ones(batch_size, 1) * ctle_peaking)

    # Synchronize using cross-correlation to find common delay
    batch_delays = cross_correlate_sync_batch(tx_symbols, rx_init)
    common_delay = int(torch.median(torch.tensor(batch_delays, dtype=torch.float)).item())

    # rx_aligned drops the startup delay so index 0 corresponds to tx_symbols[0]
    rx_aligned = rx_init[:, common_delay:]

    # tx_symbols starts at index 0, but we must truncate the end so lengths match
    tx_aligned = tx_symbols[:, :rx_aligned.shape[1]]

    print(f"Batch size: {batch_size}, Common delay: {common_delay}")
    print(f"Aligned sequence length: {tx_aligned.shape[1]}")
    print("-" * 60)

    results = {}
    # Store weights per channel per method: weights[method][channel_idx] = (ffe_taps, dfe_taps)
    all_ffe_weights = {}  # method -> list of FFE arrays
    all_dfe_weights = {}  # method -> list of DFE arrays
    step_size_history = {}  # method -> tensor of step sizes over time

    # 1. Evaluate L2O Models
    # If model_path is provided, use it for a single model; otherwise auto-generate paths
    suffix = "_ablate_ctle" if ablate_ctle else ""
    ablate_label = " (Ablated)" if ablate_ctle else ""

    if model_path:
        # Use the explicitly provided model path
        model_type = "rnn"  # Default, user should specify if MLP
        if "mlp" in model_path.lower():
            model_type = "mlp"
        # Check if this is a no-AGC model
        is_no_agc_model = "_no_agc" in model_path.lower()
        # Auto-detect two-head from model filename
        is_two_head_model = "_two_head" in model_path.lower()
        if is_no_agc_model:
            mlp_class = MultiRateLearnedMLPNoAGC
        else:
            mlp_class = MultiRateLearnedMLP
        models_to_test = [
            (f"Custom Model{ablate_label}", model_path, model_type,
             mlp_class(L2O_STATE_DIM, L2O_MLP_HISTORY_LEN, L2O_MLP_HIDDEN_DIM, use_two_head=is_two_head_model) if model_type == "mlp"
             else MultiRateLearnedNLMS(6, 32, use_two_head=is_two_head_model))
        ]
    else:
        # Auto-generate paths based on channel_type and ablation settings
        two_head_suffix = "_two_head" if args.two_head else ""
        two_head_label = " (Two-Head)" if args.two_head else ""
        no_agc_suffix = "_no_agc" if args.disable_agc else ""
        no_agc_label = " (No AGC)" if args.disable_agc else ""

        # For MLP models, select the appropriate class based on disable_agc flag
        if args.disable_agc:
            mlp_class = MultiRateLearnedMLPNoAGC
        else:
            mlp_class = MultiRateLearnedMLP

        all_models = [
            (f"RNN Basic{ablate_label}{two_head_label}", f"./models/l2o_basic_model_{args.channel_type}{suffix}{two_head_suffix}_dfe={DFE_TAPS}.pth", "rnn", MultiRateLearnedNLMS(6, 32, use_two_head=args.two_head)),
            (f"MLP{ablate_label}{two_head_label}{no_agc_label}", f"./models/l2o_mlp_model_{args.channel_type}{suffix}{two_head_suffix}{no_agc_suffix}_dfe={DFE_TAPS}.pth", "mlp", mlp_class(L2O_STATE_DIM, L2O_MLP_HISTORY_LEN, L2O_MLP_HIDDEN_DIM, use_two_head=args.two_head)),
            (f"RNN Progressive{ablate_label}{two_head_label}", f"./models/l2o_progressive_model_{args.channel_type}{suffix}{two_head_suffix}_dfe={DFE_TAPS}.pth", "rnn", MultiRateLearnedNLMS(6, 32, use_two_head=args.two_head))
        ]
        # Filter based on exclusion flags
        models_to_test = []
        for name, path, mtype, model in all_models:
            excluded = False
            if "RNN Basic" in name and exclude_rnn_basic:
                excluded = True
            if "MLP" in name and exclude_mlp:
                excluded = True
            if "RNN Progressive" in name and exclude_rnn_progressive:
                excluded = True
            if not excluded:
                models_to_test.append((name, path, mtype, model))

    plt.figure(figsize=(12, 7))

    # 1. Evaluate L2O Models (if enabled)
    if include_l2o:
        for name, path, mtype, model in models_to_test:
            try:
                model.load_state_dict(torch.load(path))
                print(f"Loaded {name} from {path}")
                mse_trace, w_ffe_final, dfe_final, mu_trace = run_l2o_inference(model, mtype, rx_base, tx_symbols, ctle, dfe, ctle_peaking=ctle_peaking, ablate_ctle=ablate_ctle)
                avg_mse = torch.mean(mse_trace).item()
                ss_mse = torch.mean(mse_trace[burn_in:]).item()
                ss_mu = torch.mean(mu_trace[burn_in:]).item() if len(mu_trace) > burn_in else torch.mean(mu_trace).item()
                results[name] = (avg_mse, ss_mse)
                step_size_history[name] = mu_trace
                # Store all channel weights (not averaged)
                all_ffe_weights[name] = [w_ffe_final[i].cpu().numpy() for i in range(batch_size)]
                all_dfe_weights[name] = [dfe_final[i].cpu().numpy() for i in range(batch_size)]

                smoothed = pd.Series(mse_trace).ewm(span=20).mean()
                plt.plot(10 * torch.log10(torch.tensor(smoothed)), label=f"{name} (SS: {10*np.log10(ss_mse):.2f} dB)")
            except FileNotFoundError:
                print(f"Skipping {name}: weight file not found.")

    # 2. Run Benchmarks on the SAME batch data (fair comparison)

    # Static NLMS Bench (if enabled)
    if include_nlms:
        avg_mse_nlms, nlms_all_dfe, nlms_all_ffe, mu_nlms = run_batch_nlms_dfe(
            rx_aligned, tx_aligned, num_taps=DFE_TAPS, mu=0.05, teacher_forcing=False
        )
        results["NLMS (0.05)"] = (torch.mean(avg_mse_nlms).item(), torch.mean(avg_mse_nlms[burn_in:]).item())
        all_ffe_weights["NLMS (0.05)"] = nlms_all_ffe
        all_dfe_weights["NLMS (0.05)"] = nlms_all_dfe
        step_size_history["NLMS (0.05)"] = mu_nlms  # Static mu=0.05
        smoothed_nlms = pd.Series(avg_mse_nlms.numpy()).ewm(span=20).mean()
        plt.plot(10 * torch.log10(torch.tensor(smoothed_nlms)), '--', label="NLMS mu=0.05", alpha=0.6)

    # VSS NLMS Bench (if enabled)
    if include_vss:
        avg_mse_vss, vss_all_dfe, vss_all_ffe, mu_vss = run_batch_nlms_dfe(
            rx_aligned, tx_aligned, num_taps=DFE_TAPS, mu=0.1, teacher_forcing=False,
            use_vss=True, vss_mu_max=VSS_MU_MAX, vss_mu_min=VSS_MU_MIN,
            vss_alpha=VSS_ALPHA, vss_gamma=VSS_GAMMA
        )
        results["NLMS (VSS)"] = (torch.mean(avg_mse_vss).item(), torch.mean(avg_mse_vss[burn_in:]).item())
        all_ffe_weights["NLMS (VSS)"] = vss_all_ffe
        all_dfe_weights["NLMS (VSS)"] = vss_all_dfe
        step_size_history["NLMS (VSS)"] = mu_vss
        smoothed_vss = pd.Series(avg_mse_vss.numpy()).ewm(span=20).mean()
        plt.plot(10 * torch.log10(torch.tensor(smoothed_vss)), 'b--', label="NLMS Continuous VSS", alpha=0.6)

        # VSS NLMS with Momentum Bench (if enabled)
        avg_mse_vss_mom, vss_mom_all_dfe, vss_mom_all_ffe, mu_vss_mom = run_batch_nlms_dfe(
            rx_aligned, tx_aligned, num_taps=DFE_TAPS, mu=0.1, teacher_forcing=False,
            use_vss=True, vss_mu_max=VSS_MU_MAX, vss_mu_min=VSS_MU_MIN,
            vss_alpha=VSS_ALPHA, vss_gamma=VSS_GAMMA, vss_momentum=VSS_MOMENTUM
        )
        results["NLMS (VSS+Mom)"] = (torch.mean(avg_mse_vss_mom).item(), torch.mean(avg_mse_vss_mom[burn_in:]).item())
        all_ffe_weights["NLMS (VSS+Mom)"] = vss_mom_all_ffe
        all_dfe_weights["NLMS (VSS+Mom)"] = vss_mom_all_dfe
        step_size_history["NLMS (VSS+Mom)"] = mu_vss_mom
        smoothed_vss_mom = pd.Series(avg_mse_vss_mom.numpy()).ewm(span=20).mean()
        plt.plot(10 * torch.log10(torch.tensor(smoothed_vss_mom)), 'm:', label=f"NLMS Continuous VSS (Mom={VSS_MOMENTUM})", alpha=0.6)

    # RLS Bench (if enabled)
    if include_rls:
        avg_mse_rls, rls_all_combined = run_batch_rls_dfe(
            rx_aligned, tx_aligned, num_taps=DFE_TAPS, lam=RLS_LAMBDA, delta=RLS_DELTA, teacher_forcing=False
        )
        results["RLS"] = (torch.mean(avg_mse_rls).item(), torch.mean(avg_mse_rls[burn_in:]).item())
        # RLS combines FFE and DFE into one weight vector [FFE_TAPS + DFE_TAPS]
        all_ffe_weights["RLS"] = [w[:FFE_TAPS] for w in rls_all_combined]
        all_dfe_weights["RLS"] = [w[FFE_TAPS:] for w in rls_all_combined]
        smoothed_rls = pd.Series(avg_mse_rls.numpy()).ewm(span=20).mean()
        plt.plot(10 * torch.log10(torch.tensor(smoothed_rls)), 'k:', label="RLS Baseline", linewidth=2)

    # Final Reporting
    print("\nComparison Table (Evaluation on {} symbols, batch-avg):".format(seq_len))
    print(f"{'Method':<20} | {'Avg MSE (dB)':<15} | {'SS MSE (dB)':<15}")
    print("-" * 55)
    for name, (avg, ss) in results.items():
        print(f"{name:<20} | {10*np.log10(avg):>12.2f} dB | {10*np.log10(ss):>12.2f} dB")

    # Step Size Summary
    print("\n" + "="*60)
    print("Step Size Summary (averaged over steady-state, symbols {} onwards):".format(burn_in))
    print("="*60)
    print(f"{'Method':<20} | {'Avg SS Step Size':<15} | {'Final SS Step Size':<18}")
    print("-" * 60)
    for name in results.keys():
        if name in step_size_history:
            mu_trace = step_size_history[name]
            avg_ss_mu = torch.mean(mu_trace[burn_in:]).item() if len(mu_trace) > burn_in else torch.mean(mu_trace).item()
            final_ss_mu = mu_trace[-1].item() if len(mu_trace) > 0 else 0.0
            print(f"{name:<20} | {avg_ss_mu:<15.6f} | {final_ss_mu:<18.6f}")
        elif name == "RLS":
            print(f"{name:<20} | {'N/A (RLS)':<15} | {'N/A (forgetting λ)':<18}")
        else:
            print(f"{name:<20} | {'N/A':<15} | {'N/A':<18}")

    # Print final FFE and DFE taps for each method (show 2-3 example channels)
    print("\nFinal Equalizer Taps (2-3 example channels):")
    print("=" * 80)
    num_example_channels = min(3, batch_size)

    for ch_idx in range(num_example_channels):
        print(f"\n--- Channel {ch_idx} ---")
        for name in results.keys():
            # Check if this method has weights for this channel
            if ch_idx < len(all_ffe_weights[name]):
                ffe_taps = all_ffe_weights[name][ch_idx]
                dfe_taps = all_dfe_weights[name][ch_idx]
                print(f"  {name}:")
                print(f"    FFE: {ffe_taps}")
                print(f"    DFE: {dfe_taps}")
            else:
                print(f"  {name}: N/A (benchmark only has 1 channel)")

    plt.title(f"Inference Comparison: Learned Optimizers vs Benchmarks\n(Evaluated on {seq_len} symbols, CH={CH_TAPS}, DFE={DFE_TAPS})")
    plt.xlabel("Symbols")
    plt.ylabel("MSE (dB)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Bar plots - separate plots for each channel
    if all_ffe_weights and all_dfe_weights:
        methods = list(all_ffe_weights.keys())
        num_methods = len(methods)
        colors = [plt.cm.tab10.colors[i] for i in range(num_methods)]

        # Create separate figure for each example channel
        for ch_idx in range(num_example_channels):
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle(f'Channel {ch_idx} - Final Equalizer Tap Weights', fontsize=14, fontweight='bold')

            # Determine which methods have data for this channel
            available_methods = [(i, m) for i, m in enumerate(methods) if ch_idx < len(all_ffe_weights[m])]

            if not available_methods:
                plt.close(fig)
                continue

            # FFE taps bar plot for this channel
            ax1 = axes[0]
            x = np.arange(FFE_TAPS)
            width = 0.8 / float(len(available_methods))
            for i, (orig_idx, method) in enumerate(available_methods):
                ffe_taps = all_ffe_weights[method][ch_idx]
                ax1.bar(x + i * width, ffe_taps, width, label=method, color=colors[orig_idx], alpha=0.8)
            ax1.set_xlabel('Tap Index')
            ax1.set_ylabel('Weight Value')
            ax1.set_title(f'FFE Taps ({FFE_TAPS} taps)')
            ax1.set_xticks(x + width * (len(available_methods) - 1) / 2)
            ax1.set_xticklabels(x)
            ax1.legend()
            ax1.grid(True, axis='y', alpha=0.3)
            ax1.axhline(y=0, color='black', linewidth=0.5)

            # DFE taps bar plot for this channel
            ax2 = axes[1]
            x = np.arange(DFE_TAPS)
            width = 0.8 / float(len(available_methods))
            for i, (orig_idx, method) in enumerate(available_methods):
                dfe_taps = all_dfe_weights[method][ch_idx]
                ax2.bar(x + i * width, dfe_taps, width, label=method, color=colors[orig_idx], alpha=0.8)
            ax2.set_xlabel('Tap Index')
            ax2.set_ylabel('Weight Value')
            ax2.set_title(f'DFE Taps ({DFE_TAPS} taps)')
            ax2.set_xticks(x + width * (len(available_methods) - 1) / 2)
            ax2.set_xticklabels(x)
            ax2.legend()
            ax2.grid(True, axis='y', alpha=0.3)
            ax2.axhline(y=0, color='black', linewidth=0.5)

            plt.tight_layout()
            plt.show()
