"""
Centralized configuration file for learned optimizer experiments.

All common settings (channel, DFE, CTLE, training parameters) are defined here.
Import these values in any solver file:
    from config import *
"""

import torch

# ==========================================
# Random Seed
# ==========================================
SEED = 42
torch.manual_seed(SEED)

# ==========================================
# Channel Configuration
# ==========================================
CH_TAPS = 50           # Number of taps in symbol intervals; sample-domain tap count = CH_TAPS * OVERSAMPLE_FACTOR
SNR_RANGE = (15, 25)   # dB (signal-to-noise ratio range)

# ==========================================
# Oversampling Configuration
# ==========================================
OVERSAMPLE_FACTOR = 1  # Front-end oversampling factor (1 = symbol rate, no oversampling)
OVERSAMPLE_MODE = "zoh"  # Zero-order hold interpolation only for now
PHASE_SEARCH_MAX_DELAY = 64  # Max delay for phase search in symbols
PHASE_SEARCH_SYNC_LEN = 128  # Sync length for phase search

# ==========================================
# Equalizer Configuration
# ==========================================
DFE_TAPS = 10          # Number of taps in the Decision Feedback Equalizer
CTLE_TAPS = 15         # Number of taps in the CTLE FIR approximation
FIXED_PEAKING = 0.5   # Initial/Fixed CTLE gain for benchmark

# CTLE Filter Taps (defines the CTLE response characteristics)
# Low-pass: base_lp[0] = 1.0 (pass-through / all-pass proxy)
# High-pass: base_hp[0] = 1.0, base_hp[1] = -0.8 (amplifies transitions, subtracts post-cursor)
CTLE_HP_ALPHA = 0.8   # High-pass filter coefficient (0.8 = -6dB at 1/2 Nyquist)

# ==========================================
# FFE (Multi-Tap) Configuration
# ==========================================
FFE_TAPS = 10          # Number of taps in the Feed-Forward Equalizer
FFE_MAIN_CURSOR = FFE_TAPS // 2  # Index of main cursor (center tap for odd number of taps)
FFE_INIT = 1.0        # Initial value for FFE/VGA main cursor

# ==========================================
# RLS Configuration
# ==========================================
RLS_LAMBDA = 0.99     # Forgetting factor (memory ~ 1/(1-lambda) = 100 symbols)
RLS_DELTA = 0.01      # Initialization factor for inverse correlation matrix

# ==========================================
# Training Configuration
# ==========================================
BATCH_SIZE = 64        # Batch size for meta-training
EPOCHS = 200           # Number of training epochs
SEQ_LENGTH = 10000     # Sequence length for benchmark testing

# ==========================================
# TBPTT Configuration (for learned optimizers)
# ==========================================
UNROLL_LEN = 50        # Static unroll length (used in l2o_basic.py)

# Progressive curriculum parameters (used in l2o_progressive.py)
INITIAL_UNROLL = 10    # Starting TBPTT horizon
MAX_UNROLL = 300       # Target TBPTT horizon
UNROLL_STEP_EPOCH = 20  # Epochs between each unroll increase
UNROLL_DELTA = 10      # Increment step size for unroll length

# ==========================================
# NLMS Benchmark Configuration
# ==========================================
NLMS_MU_VALUES = [0.01, 0.05, 0.1, 0.2]  # Static mu sweep values

# DEPRECATED: Gear-shifting (Variable Step-Size) NLMS parameters
# These parameters are no longer used - we use continuous VSS NLMS instead
# GEAR_SHIFT_MU_FAST = 0.2       # Acquisition gear step size
# GEAR_SHIFT_MU_SLOW = 0.01     # Tracking gear step size
# GEAR_SHIFT_THRESHOLD = 0.5    # MSE threshold to trigger gear shift
# GEAR_SHIFT_EMA_ALPHA = 0.05   # Smoothing factor for error variance

# Continuous Variable Step-Size (VSS) NLMS parameters
VSS_MU_MAX = 0.5       # Upper bound (fast acquisition)
VSS_MU_MIN = 0.005     # Lower bound (fine tracking/steady-state)
VSS_ALPHA = 0.99       # Memory factor (close to 1 for smooth decay)
VSS_GAMMA = 1e-3      # Error scaling factor (controls reaction to error spikes)
VSS_MOMENTUM = 0.0     # Momentum coefficient for heavy-ball optimization (0.0 = no momentum)

# Benchmark settings
BURN_IN = 2000          # Symbols to skip for steady-state calculation
TARGET_MSE_DB = -20     # Target MSE in dB

# ==========================================
# Learned Optimizer Configuration
# ==========================================
L2O_HIDDEN_DIM = 32     # Hidden dimension for GRU/RNN cell
L2O_STATE_DIM = 6       # State features dimension
L2O_DFE_HEAD_SCALE = 0.05    # DFE step size bound
L2O_OVERDRIVE_MAX = 1.2
L2O_CTLE_HEAD_SCALE = 0.1   # CTLE step size bound
L2O_META_LR = 1e-3     # Meta-optimizer learning rate
CTLE_UPDATE_RATE = 10   # Update CTLE every N symbols
EMA_BETA = 0.95        # EMA decay for error tracking

# L1 Penalty on step size to force lazy convergence at steady-state
# Higher values = faster decay to zero step size, but may hurt acquisition
L2O_MU_PENALTY = 0.0   # L1 penalty weight on mu_dfe
L2O_OVERDRIVE_PENALTY = 0.00001  # L2 penalty weight on mu_overdrive (quadratic)

# MLP-specific configuration (for l2o_mlp.py)
L2O_MLP_HISTORY_LEN = 10   # Number of past states to concatenate for MLP
L2O_MLP_HIDDEN_DIM = 64    # Hidden dimension for MLP (larger than RNN due to increased input)

# ==========================================
# Ablation Settings
# ==========================================
ABLATE_CTLE = True     # Set to True to disable CTLE control

# ==========================================
# Evaluation Settings (for evaluate_l2o.py and benchmark_nlms.py)
# ==========================================
EVAL_BATCH_SIZE = 100   # Number of channels to evaluate over
EVAL_SEQ_LENGTH = 1000  # Sequence length for evaluation
EVAL_BURN_IN = 200     # Symbol index to start steady-state calculation
FIXED_CTLE_PEAKING = 0.5  # CTLE peaking gain for baseline evaluation
