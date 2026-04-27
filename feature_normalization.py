"""
Feature normalization utilities for learned optimizer inputs.

Provides:
- StreamingFeatureNormalizer: online EMA-based feature standardization
- calibrate_feature_stats: offline calibration utility
- signed_log1p: sign-preserving log transform for heavy-tailed features

Scientific Notes:
- Streaming normalization is standard and compatible with TBPTT, on-the-fly
  bit generation, randomized channels, and online learning.
- Stats are detached from autograd, accumulated over many batches,
  frozen at eval time.
- Offline calibration (optional) runs a few batches without weight updates
  to initialize stats before training.
"""

import torch
import torch.nn as nn

from oversampling_utils import upsample_symbols


class StreamingFeatureNormalizer(nn.Module):
    """
    Online feature standardization using exponential moving average statistics.

    Works with:
    - Single timestep state: [batch, feature_dim]
    - History buffer: [batch, history, feature_dim]

    Stats update order:
    1. Normalize using old stored stats (no current-batch info leakage)
    2. Update stats using detached current observations

    Scientific rationale:
    - Per-feature normalization prevents blending of unrelated distributions
    - EMA smoothing avoids single-batch outlier hijacking
    - Detached updates prevent normalization from depending on itself

    Usage:
        normalizer = StreamingFeatureNormalizer(feature_dim=7)
        z_raw = build_no_agc_state(...)
        z_norm = normalizer.normalize(z_raw)        # Uses old stats
        if training:
            normalizer.update(z_raw.detach())      # Update after normalize
    """

    def __init__(self, feature_dim, momentum=0.99, eps=1e-5):
        super().__init__()
        self.feature_dim = feature_dim
        self.momentum = momentum
        self.eps = eps

        self.register_buffer('mean', torch.zeros(feature_dim))
        self.register_buffer('var', torch.ones(feature_dim))
        self.register_buffer('initialized', torch.tensor(False))
        self.register_buffer('count', torch.tensor(0.0))

    def normalize(self, x):
        """
        Standardize features using stored EMA stats.

        Args:
            x: torch.Tensor of shape [..., feature_dim]

        Returns:
            torch.Tensor of same shape, standardized
        """
        mean = self.mean.view(*([1] * (x.dim() - 1)), -1)
        std = torch.sqrt(self.var.clamp_min(self.eps)).view(*([1] * (x.dim() - 1)), -1)
        return (x - mean) / std

    @torch.no_grad()
    def update(self, x):
        """
        Update EMA statistics with detached observations.

        Args:
            x: torch.Tensor of shape [..., feature_dim]
                   For history: [batch, history, feature_dim]
                   For single step: [batch, feature_dim]
        """
        xf = x.detach().reshape(-1, self.feature_dim)
        batch_mean = xf.mean(dim=0)
        batch_var = xf.var(dim=0, unbiased=False).clamp_min(self.eps)
        batch_count = xf.shape[0]

        if self.initialized.item():
            self.mean.mul_(self.momentum).add_((1 - self.momentum) * batch_mean)
            self.var.mul_(self.momentum).add_((1 - self.momentum) * batch_var)
        else:
            self.mean.copy_(batch_mean)
            self.var.copy_(batch_var)
            self.initialized.fill_(True)

        self.count.add_(batch_count)

    def eval_mode(self):
        """Freeze stats at evaluation - syntactic sugar."""
        self.requires_grad_(False)

    def train_mode(self, training=True):
        """Resume stats updates in training - syntactic sugar."""
        self.requires_grad_(training)

    def state_dict(self):
        """Override to ensure normalizer buffers are saved in checkpoint."""
        return {k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in super().state_dict().items()}

    def load_state_dict(self, state_dict):
        """Override to handle buffer restoration."""
        super().load_state_dict(state_dict)


def calibrate_feature_stats(model, channel_gen, num_batches=100, device=None):
    """
    Run offline calibration pass to initialize normalizer stats.

    This runs a few batches through the model WITHOUT weight updates to
    accumulate initial statistics. Useful for:
    - Initializing normalizer before training
    - Recalibrating after loading a checkpoint
    - Fair comparison between eval modes

    Args:
        model: The learned optimizer model (MLP or RNN no-AGC variant)
        channel_gen: Channel generator (must support generate_received_signal)
        num_batches: Number of batches to run for calibration
        device: Optional device to run on

    Returns:
        model: The same model (stats are updated in-place)
    """
    model.train()
    with torch.no_grad():
        for _ in range(num_batches):
            batch_size = 32
            seq_len = 500
            tx_symbols = torch.sign(torch.randn(batch_size, seq_len, device=device or torch.device('cpu')))
            tx_frontend = upsample_symbols(tx_symbols, 1, "zoh")
            rx_base, _ = channel_gen.generate_received_signal(tx_frontend, batch_size)

            if device:
                rx_base = rx_base.to(device)

            rx_init = rx_base[:, :seq_len]

            w_ffe = torch.zeros(batch_size, 10)
            w_ffe[:, 5] = 1.0
            ffe_buffer = torch.zeros(batch_size, 10)
            dfe_weights = torch.zeros(batch_size, 10)
            latent_peaking = torch.zeros(batch_size, 1)
            rx_buffer = torch.zeros(batch_size, 15)
            decision_buffer = torch.zeros(batch_size, 10)
            ema_error = torch.ones(batch_size, 1)
            h_state = torch.zeros(batch_size, 32)

            for t in range(100):
                ffe_buffer = torch.roll(ffe_buffer, shifts=1, dims=1)
                ffe_buffer[:, 0] = rx_init[:, t:t+1].squeeze(-1)

                if t >= 5:
                    target_idx = t - 5
                    e_t = tx_symbols[:, target_idx:target_idx+1] - torch.zeros(batch_size, 1)

                    grad_proxy_ctle = torch.zeros(batch_size, 1)
                    z_raw = build_no_agc_state(
                        e_t, ema_error, ffe_buffer, rx_buffer,
                        dfe_weights, torch.sigmoid(latent_peaking), grad_proxy_ctle,
                    )

                    if hasattr(model, 'normalizer'):
                        model.normalizer.update(z_raw.detach())

                    ema_error = 0.95 * ema_error + 0.05 * (e_t ** 2)

                decision_buffer = torch.roll(decision_buffer, shifts=1, dims=1)
                decision_buffer[:, 0] = 0.0

    model.eval()
    return model


def signed_log1p(x):
    """
    Sign-preserving log transform for heavy-tailed features.

    Transforms large positive values while preserving sign and
    handling zeros gracefully.

    Args:
        x: torch.Tensor

    Returns:
        torch.Tensor with same shape, log-transformed
    """
    return torch.sign(x) * torch.log1p(torch.abs(x))


def build_no_agc_state(
    e_t,
    ema_error,
    ffe_buffer,
    rx_buffer,
    dfe_weights,
    ctle_peaking,
    grad_proxy_ctle,
    eps=1e-6,
):
    """
    Build scale-aware state features for no-AGC mode.

    All features are normalized by FFE RMS to make the optimizer
    robust to absolute amplitude variations from insertion loss.

    State dimensions (7 total):
        e_scaled: error normalized by FFE RMS
        ema_error_scaled: EMA error normalized by FFE RMS^2
        log_ffe_rms: log of FFE buffer RMS
        dfe_weight0: first DFE tap (bounded, no scaling needed)
        ctle_peaking: CTLE gain (bounded, no scaling needed)
        grad_scaled: CTLE gradient proxy normalized by FFE RMS
        log_rx_rms: log of RX buffer RMS

    Args:
        e_t: [batch, 1] error signal
        ema_error: [batch, 1] EMA of error squared
        ffe_buffer: [batch, FFE_TAPS] FFE sample buffer
        rx_buffer: [batch, CTLE_TAPS] CTLE input buffer
        dfe_weights: [batch, DFE_TAPS] DFE tap weights
        ctle_peaking: [batch, 1] CTLE gain
        grad_proxy_ctle: [batch, 1] gradient proxy for CTLE
        eps: small constant for numerical stability

    Returns:
        torch.Tensor of shape [batch, 7]
    """
    ffe_rms = torch.sqrt(ffe_buffer.pow(2).mean(dim=1, keepdim=True) + eps)
    rx_rms = torch.sqrt(rx_buffer.pow(2).mean(dim=1, keepdim=True) + eps)

    e_scaled = e_t / (ffe_rms + eps)
    ema_error_scaled = ema_error / (ffe_rms.pow(2) + eps)
    grad_scaled = grad_proxy_ctle / (ffe_rms + eps)

    return torch.cat(
        [
            signed_log1p(e_scaled),
            torch.log(ema_error_scaled + eps),
            torch.log(ffe_rms + eps),
            dfe_weights[:, 0:1],
            ctle_peaking,
            signed_log1p(grad_scaled),
            torch.log(rx_rms + eps),
        ],
        dim=1,
    )


def build_agc_state(e_t, ema_error, norm_sq, dfe_weight0, ctle_peaking, grad_proxy_ctle):
    """
    Build state features for normalized (AGC) mode.

    In normalized mode, raw features are acceptable since
    channel IR is peak/l2 normalized and amplitude is bounded.

    Args:
        e_t: [batch, 1] error signal
        ema_error: [batch, 1] EMA of error squared
        norm_sq: [batch, 1] normalization factor (FFE + DFE buffer energy)
        dfe_weight0: [batch, 1] first DFE tap
        ctle_peaking: [batch, 1] CTLE gain
        grad_proxy_ctle: [batch, 1] gradient proxy for CTLE

    Returns:
        torch.Tensor of shape [batch, 6]
    """
    return torch.cat(
        [
            e_t,
            ema_error,
            norm_sq,
            dfe_weight0,
            ctle_peaking,
            grad_proxy_ctle,
        ],
        dim=1,
    )