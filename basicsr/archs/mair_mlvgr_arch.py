"""
MaIR with Multi-Layer Variance-Gated Refinement (ML-VGR).

Core innovation: Inter-Stage Content-Adaptive Adaptation with
Per-Channel Variance Gating + Full Backbone Fine-Tuning.

v2.1 changes (addressing v2's Urban100 collapse + slow convergence):
┌─────────────────────────────────────────────────────────────────────┐
│ v2 diagnosis:                                                        │
│   channel_mod (SE with global pooling) applies UNCONDITIONALLY —     │
│   even when gate=0, output is x*ch_scale, not x. This constant      │
│   perturbation destroys Urban100's periodic high-freq patterns.      │
│   Also: only 50% backbone unfrozen → insufficient learning capacity. │
│                                                                      │
│ v2.1 fixes:                                                          │
│   1. Remove channel_mod → pure variance-gated residual               │
│   2. Unfreeze ALL backbone layers at conservative lr (0.01x)         │
│      → 4x more trainable backbone params, 2-3x total gradient       │
│   3. Adapters at stages [1,2] only (remove stage 0 — too early)      │
│   4. gate_init_bias=-3.0 → sigmoid≈0.047 (faster activation)        │
│   5. Adapters act as content-adaptive gradient modulators,            │
│      guiding backbone fine-tuning to avoid 偏科                       │
└─────────────────────────────────────────────────────────────────────┘

v2.3 changes (addressing seesaw effect + 3000-iter HF forgetting):
┌─────────────────────────────────────────────────────────────────────┐
│ v2/v2.2 diagnosis:                                                    │
│   Variance-only gate cannot distinguish "complex texture" (Urban100   │
│   periodic patterns: high var + high gradient) from "natural edge"    │
│   (Set14 edges: high gradient, moderate var). Single signal → gate    │
│   makes identical decisions for fundamentally different content →     │
│   seesaw. Also: L1 loss gradient dominated by low-freq → backbone    │
│   gradually forgets HF textures after ~3K iters ("3K curse").        │
│                                                                      │
│ v2.3 fixes:                                                          │
│   1. Gradient-Augmented Gate: concat [variance, gradient_magnitude]   │
│      → 2C-channel input to gate network. Variance captures texture   │
│      complexity; gradient captures edge strength. Two orthogonal     │
│      signals let gate make finer decisions per content type.          │
│   2. High-Frequency Residual Shortcut: hf = x - local_mean, added   │
│      with per-channel learnable scale (init=0). Direct HF pathway    │
│      bypasses the gate entirely → prevents catastrophic HF forgetting │
│      regardless of L1 loss bias. Adapters: ~5.6K params each.        │
│   3. Training: +FFTFreqLoss (weight=0.02) for explicit HF signal.    │
│   Backward compatible: use_gradient_gate=False, use_hf_shortcut=     │
│   False reverts to v2.1 behavior exactly.                            │
└─────────────────────────────────────────────────────────────────────┘

Architecture flow:
  conv_first → [Layer 0] → [Layer 1] → Adapter₁
             → [Layer 2] → Adapter₂ → [Layer 3]
             → norm → patch_unembed → conv_after_body → +skip → upsample

v3.0 — SSGR (Spectral-Spatial Gated Refinement):
┌─────────────────────────────────────────────────────────────────────┐
│ Target: Avg PSNR +0.03-0.05, requiring deeper representation gains. │
│                                                                      │
│ v2.3 adapter fine-tuning ceiling is ~+0.005-0.015. To break through, │
│ add a frequency-domain branch that directly enhances HF features:    │
│                                                                      │
│ SSGRAdapter = Spatial Branch (v2.3 VGR) + Spectral Branch (NEW):     │
│   Spatial: gradient-augmented variance gate → content-aware refine   │
│   Spectral: rfft2 → learnable magnitude modulation → irfft2         │
│   Both branches zero-init, residual-added in parallel.               │
│                                                                      │
│ Why dual-domain works:                                               │
│   - Spatial branch handles local details (edges, textures)           │
│   - Spectral branch handles global frequency balance (sharpness,     │
│     periodic patterns, HF energy distribution)                       │
│   - Orthogonal optimization landscapes → compound rather than cancel │
│                                                                      │
│ Training: from-scratch 500K iters (backbone + adapters co-evolve).   │
│ Params: ~7.5K per adapter × 3 positions = ~22.5K total (~3% of      │
│ backbone). Cross-task: no SR/denoising-specific assumptions.          │
└─────────────────────────────────────────────────────────────────────┘

Architecture flow (SSGR, adapter_stages=[0,1,2]):
  conv_first → [Layer 0] → Adapter₀ → [Layer 1] → Adapter₁
             → [Layer 2] → Adapter₂ → [Layer 3]
             → norm → patch_unembed → conv_after_body → +skip → upsample

Cross-task: purely spatial/frequency domain, no SR/denoising-specific components.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.archs.mair_arch import (
    MaIR,
    mair_ids_generate,
    mair_shift_ids_generate,
)
from basicsr.utils import get_root_logger
from basicsr.utils.registry import ARCH_REGISTRY


# ──────────────────────────────────────────────────────────────────────────────
# Core: Per-Channel Variance-Gated Adapter (v2.3 — gradient gate + HF shortcut)
# ──────────────────────────────────────────────────────────────────────────────

class VGRAdapter(nn.Module):
    """
    Per-channel variance-gated adapter for inter-layer placement.

    v2.3 enhancements over v2.1:
    - Gradient-augmented gate: gate sees [variance, gradient_magnitude] instead
      of variance only. Gradient magnitude (local Sobel-like finite differences,
      smoothed to var_kernel scale) is orthogonal to variance — together they
      distinguish periodic textures (high var + high grad → Urban100) from
      natural edges (high grad + moderate var → Set14), enabling content-type-
      specific gating that breaks the seesaw effect.
    - High-frequency residual shortcut: hf = x - local_mean is added back with
      a per-channel learnable scale (initialized to 0, bounded by tanh * bound).
      This direct HF pathway bypasses the gate entirely, preventing the
      "3000-iter catastrophic HF forgetting" caused by L1-dominated gradients.

    Backward compatible: use_gradient_gate=False + use_hf_shortcut=False
    reproduces v2.1 behavior exactly.

    Each adapter has ~5.6K trainable params (for 60-channel features with v2.3).
    """

    def __init__(
        self,
        channels,
        bottleneck_ratio=2,
        var_kernel=5,
        gate_init_bias=-3.0,
        # v2.3 additions
        use_gradient_gate=False,
        use_hf_shortcut=False,
        hf_shortcut_bound=0.15,
    ):
        super().__init__()
        self.var_kernel = max(var_kernel, 3)
        if self.var_kernel % 2 == 0:
            self.var_kernel += 1
        self.use_gradient_gate = use_gradient_gate
        self.use_hf_shortcut = use_hf_shortcut
        self.hf_shortcut_bound = hf_shortcut_bound

        mid_refine = max(channels // bottleneck_ratio, 4)
        mid_gate = max(channels // 8, 4)

        # ── Spatial refinement: depthwise conv + wider bottleneck ──
        self.refine = nn.Sequential(
            nn.Conv2d(
                channels, channels, 3, 1, 1,
                groups=channels, bias=False,
            ),
            nn.GELU(),
            nn.Conv2d(channels, mid_refine, 1),
            nn.GELU(),
            nn.Conv2d(mid_refine, channels, 1),
        )

        # ── Per-channel gate ──
        # v2.3: dual-signal input [variance, gradient] → 2C channels
        # v2.1: variance-only → C channels
        gate_in_channels = channels * 2 if use_gradient_gate else channels
        self.var_gate = nn.Sequential(
            nn.Conv2d(gate_in_channels, mid_gate, 1),
            nn.GELU(),
            nn.Conv2d(mid_gate, channels, 1),
        )

        # ── HF shortcut: per-channel learnable scale (v2.3) ──
        if use_hf_shortcut:
            self.hf_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

        # ── Zero-init refine output → exact identity at start ──
        nn.init.zeros_(self.refine[-1].weight)
        if self.refine[-1].bias is not None:
            nn.init.zeros_(self.refine[-1].bias)

        # Gate init: sigmoid(gate_init_bias) controls initial gate opening
        nn.init.zeros_(self.var_gate[-1].weight)
        nn.init.constant_(self.var_gate[-1].bias, gate_init_bias)

    def _local_gradient(self, x):
        """Per-channel local gradient magnitude, smoothed to var_kernel scale.

        Uses finite differences (Sobel-like) in horizontal and vertical
        directions, summed as L1 magnitude, then avg-pooled to match the
        spatial smoothing of the variance map.

        Cost: 2 pad + 2 subtract + abs + add + 1 avg_pool. No learnable params.
        """
        gx = F.pad(
            x[:, :, :, 1:] - x[:, :, :, :-1],
            (0, 1, 0, 0), mode='replicate',
        )
        gy = F.pad(
            x[:, :, 1:, :] - x[:, :, :-1, :],
            (0, 0, 0, 1), mode='replicate',
        )
        grad = gx.abs() + gy.abs()
        pad = self.var_kernel // 2
        return F.avg_pool2d(grad, self.var_kernel, stride=1, padding=pad)

    def forward(self, tokens, x_size):
        """
        Args:
            tokens: [B, H*W, C] backbone intermediate tokens
            x_size: (H, W) spatial dimensions
        Returns:
            refined tokens: [B, H*W, C]
        """
        b, hw, c = tokens.shape
        h, w = x_size

        # Reshape to 2D for conv operations
        x = tokens.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()

        # ── Step 1: Local variance (content complexity signal) ──
        pad = self.var_kernel // 2
        mu = F.avg_pool2d(x, self.var_kernel, stride=1, padding=pad)
        var = F.avg_pool2d(
            (x - mu).square(), self.var_kernel, stride=1, padding=pad,
        )

        # ── Step 2: Gate input ──
        if self.use_gradient_gate:
            grad = self._local_gradient(x)
            gate_input = torch.cat([var, grad], dim=1)  # [B, 2C, H, W]
        else:
            gate_input = var  # [B, C, H, W]
        gate = torch.sigmoid(self.var_gate(gate_input))

        # ── Step 3: Spatial refinement ──
        delta = self.refine(x)

        # ── Step 4: Gated residual (true identity when gate=0) ──
        x_out = x + gate * delta

        # ── Step 5: High-frequency residual shortcut (v2.3) ──
        if self.use_hf_shortcut:
            hf = x - mu  # high-freq component = input minus local mean
            hf_weight = self.hf_scale.tanh() * self.hf_shortcut_bound
            x_out = x_out + hf_weight * hf

        # Reshape back to tokens
        return x_out.permute(0, 2, 3, 1).reshape(b, hw, c)


# ──────────────────────────────────────────────────────────────────────────────
# Core: Spectral-Spatial Gated Refinement Adapter (v3.0 — SSGR)
# ──────────────────────────────────────────────────────────────────────────────

class SSGRAdapter(nn.Module):
    """
    Spectral-Spatial Gated Refinement adapter (v3.0 → v3.3 unified).

    Dual-domain processing:
      1. Spatial Branch: gradient-augmented variance-gated refinement.
      2. Spectral Branch: learnable frequency-domain modulation.
      3. HF Shortcut (configurable via hf_shortcut_bound).

    Task adaptation via config only — one architecture, no code branching:
      SR:   adapter_mode='additive',    spectral_mode='additive',  stats from features
      DN:   adapter_mode='subtractive', spectral_mode='suppressive', stats from noisy input

    v3.3 — External Statistics Source (stats_source):
      Gate statistics (variance, gradient) can be computed from an external
      tensor (e.g. the noisy input image) instead of from the feature tokens.
      This is critical for denoising: noise characteristics are visible in the
      input image but are abstracted away in deep features. The gate still
      controls the same refine/spectral branches — only the statistical
      evidence it uses changes.

      forward(tokens, x_size, stats_source=None):
        - stats_source=None (SR default): gate stats from feature tokens
        - stats_source=inp_tokens (DN):   gate stats from noisy input
    """

    def __init__(
        self,
        channels,
        bottleneck_ratio=2,
        var_kernel=5,
        gate_init_bias=-3.0,
        spectral_hidden_ratio=0.25,
        hf_shortcut_bound=0.15,
        # v3.1 denoising-adaptive extensions
        noise_adaptive_gate=False,
        spectral_mode='additive',
        spectral_modulation_bound=0.2,
        # v3.2: adapter mode
        adapter_mode='additive',
        # Ablation toggles
        use_spatial=True,
        use_spectral=True,
        use_gate=True,
    ):
        super().__init__()
        self.var_kernel = max(var_kernel, 3)
        if self.var_kernel % 2 == 0:
            self.var_kernel += 1
        self.hf_shortcut_bound = hf_shortcut_bound
        self.noise_adaptive_gate = noise_adaptive_gate
        self.spectral_mode = spectral_mode
        self.spectral_modulation_bound = spectral_modulation_bound
        self.adapter_mode = adapter_mode
        self.use_spatial = use_spatial
        self.use_spectral = use_spectral
        self.use_gate = use_gate

        mid_refine = max(channels // bottleneck_ratio, 4)
        mid_gate = max(channels // 8, 4)

        # ── Spatial branch: gradient-augmented variance-gated refinement ──
        self.refine = nn.Sequential(
            nn.Conv2d(
                channels, channels, 3, 1, 1,
                groups=channels, bias=False,
            ),
            nn.GELU(),
            nn.Conv2d(channels, mid_refine, 1),
            nn.GELU(),
            nn.Conv2d(mid_refine, channels, 1),
        )
        self.var_gate = nn.Sequential(
            nn.Conv2d(channels * 2, mid_gate, 1),   # 2C: [var/signal_var, grad]
            nn.GELU(),
            nn.Conv2d(mid_gate, channels, 1),
        )

        # ── Spectral branch: FFT magnitude modulation ──
        mid_spec = max(int(channels * spectral_hidden_ratio), 4)
        self.spectral_enhance = nn.Sequential(
            nn.Conv2d(channels, mid_spec, 1),
            nn.GELU(),
            nn.Conv2d(mid_spec, channels, 1),
        )
        self.spectral_scale = nn.Parameter(torch.zeros(1))

        # ── HF shortcut: per-channel learnable scale ──
        self.hf_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

        # ── Zero-init all output layers → exact identity at start ──
        nn.init.zeros_(self.refine[-1].weight)
        if self.refine[-1].bias is not None:
            nn.init.zeros_(self.refine[-1].bias)
        nn.init.zeros_(self.var_gate[-1].weight)
        # Subtractive mode: gate is inverted (1 - sigmoid), so negate bias
        # to keep initial effective gate ≈ sigmoid(|bias|) ≈ 0.047 (conservative)
        if adapter_mode == 'subtractive':
            nn.init.constant_(self.var_gate[-1].bias, -gate_init_bias)
        else:
            nn.init.constant_(self.var_gate[-1].bias, gate_init_bias)
        nn.init.zeros_(self.spectral_enhance[-1].weight)
        if self.spectral_enhance[-1].bias is not None:
            nn.init.zeros_(self.spectral_enhance[-1].bias)

    def _local_gradient(self, x):
        """Per-channel local gradient magnitude, smoothed to var_kernel scale."""
        gx = F.pad(
            x[:, :, :, 1:] - x[:, :, :, :-1],
            (0, 1, 0, 0), mode='replicate',
        )
        gy = F.pad(
            x[:, :, 1:, :] - x[:, :, :-1, :],
            (0, 0, 0, 1), mode='replicate',
        )
        grad = gx.abs() + gy.abs()
        pad = self.var_kernel // 2
        return F.avg_pool2d(grad, self.var_kernel, stride=1, padding=pad)

    def forward(self, tokens, x_size, stats_source=None):
        """
        Args:
            tokens: [B, H*W, C] feature tokens (inter-layer) or [B, C, H, W] pixel tensor
            x_size: (H, W) spatial dimensions
            stats_source: optional [B, C_src, H, W] tensor for gate statistics.
                          If None, stats are computed from tokens (SR default).
                          If provided, stats are computed from this tensor (DN: noisy input).
        Returns:
            refined tokens: [B, H*W, C] or [B, C, H, W] matching input format
        """
        # Handle both token [B, HW, C] and pixel [B, C, H, W] input formats
        if tokens.dim() == 3:
            b, hw, c = tokens.shape
            h, w = x_size
            x = tokens.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
            input_is_tokens = True
        else:
            b, c, h, w = tokens.shape
            x = tokens
            input_is_tokens = False

        # ── Compute statistics ──
        if stats_source is not None:
            s = stats_source
            if s.shape[2] != h or s.shape[3] != w:
                s = F.interpolate(s, size=(h, w), mode='bilinear', align_corners=False)
        else:
            s = x

        pad = self.var_kernel // 2
        mu = F.avg_pool2d(s, self.var_kernel, stride=1, padding=pad)
        var = F.avg_pool2d(
            (s - mu).square(), self.var_kernel, stride=1, padding=pad,
        )
        grad = self._local_gradient(s)

        # Project stats to C channels if stats_source has different channel count
        if var.shape[1] != c:
            var = var.mean(dim=1, keepdim=True).expand(-1, c, -1, -1)
            grad = grad.mean(dim=1, keepdim=True).expand(-1, c, -1, -1)

        # mu for HF shortcut: always from x itself (high-freq of features, not stats source)
        mu_x = F.avg_pool2d(x, self.var_kernel, stride=1, padding=pad)

        # ── Spatial branch ──
        if self.use_spatial:
            if self.noise_adaptive_gate:
                # v3.1: subtract per-channel noise floor → gate sees structure only
                noise_floor = var.mean(dim=[2, 3], keepdim=True)
                signal_var = F.relu(var - noise_floor)
                gate_input = torch.cat([grad, signal_var], dim=1)
            else:
                # v3.0: raw variance + gradient
                gate_input = torch.cat([var, grad], dim=1)

            if self.use_gate:
                raw_gate = torch.sigmoid(self.var_gate(gate_input))
            else:
                # Ablation: gate forced to 1 (fully open, no content adaptation)
                raw_gate = torch.ones_like(x)

            if self.adapter_mode == 'subtractive':
                # v3.2: invert gate → high in flat/noisy regions → subtract noise
                gate = 1.0 - raw_gate
                noise_est = gate * self.refine(x)
                spatial_delta = -noise_est
            else:
                gate = raw_gate
                spatial_delta = gate * self.refine(x)
        else:
            spatial_delta = 0.0

        # ── Spectral branch: FFT magnitude modulation ──
        if self.use_spectral:
            x_fft = torch.fft.rfft2(x, norm='ortho')
            mag = x_fft.abs()
            phase = x_fft.angle()

            if self.spectral_mode == 'suppressive':
                # v3.2: learned frequency suppression mask [0, 1]
                raw_mask = self.spectral_enhance(mag)
                spec_w = self.spectral_scale.tanh().abs()
                mag_new = mag * (1.0 - spec_w * torch.sigmoid(raw_mask))
            elif self.spectral_mode == 'multiplicative':
                # v3.1 Wiener-inspired: proportional to current magnitude
                raw_mask = self.spectral_enhance(mag)
                spec_w = self.spectral_scale.tanh() * self.spectral_modulation_bound
                mag_new = mag * (1.0 + spec_w * torch.tanh(raw_mask))
            else:
                # v3.0 additive: direct magnitude injection
                mag_delta = self.spectral_enhance(mag)
                spec_w = self.spectral_scale.tanh() * 0.2
                mag_new = mag + spec_w * mag_delta

            x_spec = torch.fft.irfft2(
                torch.polar(mag_new, phase), s=(h, w), norm='ortho',
            )
            spectral_delta = x_spec - x
        else:
            spectral_delta = 0.0

        # ── HF shortcut ──
        hf = x - mu_x
        hf_weight = self.hf_scale.tanh() * self.hf_shortcut_bound

        # ── Combine: all paths are residual-added ──
        x_out = x + spatial_delta + spectral_delta + hf_weight * hf

        if input_is_tokens:
            return x_out.permute(0, 2, 3, 1).reshape(b, h * w, c)
        else:
            return x_out


# ──────────────────────────────────────────────────────────────────────────────
# Registered Architecture: MaIR_MLVGR
# ──────────────────────────────────────────────────────────────────────────────

@ARCH_REGISTRY.register()
class MaIR_MLVGR(nn.Module):
    """
    MaIR with Multi-Layer Variance-Gated Refinement.

    Architecture:
      conv_first → [Layer0] → [Layer1] → Adapter₁
                 → [Layer2] → Adapter₂ → [Layer3]
                 → norm → unembed → conv_after_body → +skip → upsample

    v2.3 strategy: Gradient-Augmented Gate + HF Shortcut
      - Dual-signal gate (variance + gradient) breaks seesaw effect
      - HF residual shortcut prevents 3K-iter catastrophic forgetting
      - +FFTFreqLoss provides explicit HF supervision signal
      - Backward compatible with v2.1/v2.2 via flags
    """

    def __init__(
        self,
        # Adapter parameters
        freeze_backbone_after_load=True,
        adapter_type='vgr',           # 'vgr' (v2.x) or 'ssgr' (v3.0/v3.1)
        adapter_stages=None,
        adapter_bottleneck_ratio=2,
        adapter_var_kernel=5,
        adapter_gate_init_bias=-3.0,
        # v2.3 VGR adapter enhancements
        adapter_use_gradient_gate=False,
        adapter_use_hf_shortcut=False,
        adapter_hf_shortcut_bound=0.15,
        # v3.0 SSGR adapter parameters
        adapter_spectral_hidden_ratio=0.25,
        # v3.1 SSGR denoising-adaptive extensions
        adapter_noise_adaptive_gate=False,
        adapter_spectral_mode='additive',
        adapter_spectral_modulation_bound=0.2,
        # Ablation toggles
        adapter_use_spatial=True,
        adapter_use_spectral=True,
        adapter_use_gate=True,
        # Backbone trainable control
        backbone_trainable_prefixes=None,
        backbone_trainable_keywords=None,
        backbone_forbidden_keywords=None,
        # MaIR backbone kwargs
        **mair_kwargs,
    ):
        super().__init__()
        self.backbone = MaIR(**mair_kwargs)
        self.freeze_backbone_after_load = freeze_backbone_after_load

        self.backbone_trainable_prefixes = tuple(
            prefix.replace('backbone.', '')
            for prefix in (backbone_trainable_prefixes or [])
        )
        self.backbone_trainable_keywords = tuple(
            keyword.replace('backbone.', '')
            for keyword in (backbone_trainable_keywords or [])
        )
        self.backbone_forbidden_keywords = tuple(
            keyword.replace('backbone.', '')
            for keyword in (backbone_forbidden_keywords or [])
        )

        embed_dim = self.backbone.embed_dim
        n_layers = len(self.backbone.layers)

        # Default: place adapters after all layers except the last
        if adapter_stages is None:
            adapter_stages = list(range(n_layers - 1))
        self.adapter_stages = sorted(set(adapter_stages))

        # Create inter-stage adapters
        self.adapters = nn.ModuleDict()
        for stage_idx in self.adapter_stages:
            if stage_idx < n_layers:
                if adapter_type == 'ssgr':
                    self.adapters[str(stage_idx)] = SSGRAdapter(
                        embed_dim,
                        bottleneck_ratio=adapter_bottleneck_ratio,
                        var_kernel=adapter_var_kernel,
                        gate_init_bias=adapter_gate_init_bias,
                        spectral_hidden_ratio=adapter_spectral_hidden_ratio,
                        hf_shortcut_bound=adapter_hf_shortcut_bound,
                        noise_adaptive_gate=adapter_noise_adaptive_gate,
                        spectral_mode=adapter_spectral_mode,
                        spectral_modulation_bound=adapter_spectral_modulation_bound,
                        use_spatial=adapter_use_spatial,
                        use_spectral=adapter_use_spectral,
                        use_gate=adapter_use_gate,
                    )
                else:
                    self.adapters[str(stage_idx)] = VGRAdapter(
                        embed_dim,
                        bottleneck_ratio=adapter_bottleneck_ratio,
                        var_kernel=adapter_var_kernel,
                        gate_init_bias=adapter_gate_init_bias,
                        use_gradient_gate=adapter_use_gradient_gate,
                        use_hf_shortcut=adapter_use_hf_shortcut,
                        hf_shortcut_bound=adapter_hf_shortcut_bound,
                    )

        self._backbone_frozen = False

    # ── Scan ID helpers ──

    def _get_scan_ids(self, h, w):
        if self.backbone.dynamic_ids or (self.backbone.image_size != (h, w)):
            xs_scan_ids, xs_inverse_ids = mair_ids_generate(
                inp_shape=(1, 1, h, w),
                scan_len=self.backbone.scan_len,
            )
            xs_shift_scan_ids, xs_shift_inverse_ids = mair_shift_ids_generate(
                inp_shape=(1, 1, h, w),
                scan_len=self.backbone.scan_len,
                shift_len=self.backbone.scan_len // 2,
            )
            device = next(self.parameters()).device
            xs_scan_ids = xs_scan_ids.to(device)
            xs_inverse_ids = xs_inverse_ids.to(device)
            xs_shift_scan_ids = xs_shift_scan_ids.to(device)
            xs_shift_inverse_ids = xs_shift_inverse_ids.to(device)
            return (
                xs_scan_ids, xs_inverse_ids,
                xs_shift_scan_ids, xs_shift_inverse_ids,
            )
        return (
            self.backbone.xs_scan_ids,
            self.backbone.xs_inverse_ids,
            self.backbone.xs_shift_scan_ids,
            self.backbone.xs_shift_inverse_ids,
        )

    # ── Forward pass ──

    def forward_features(self, x):
        _, _, h, w = x.shape
        x_size = (h, w)
        scan_ids = self._get_scan_ids(h, w)

        tokens = self.backbone.patch_embed(x)
        tokens = self.backbone.pos_drop(tokens)

        for i, layer in enumerate(self.backbone.layers):
            tokens = layer(tokens, scan_ids, x_size)
            # Apply inter-stage adapter if present
            stage_key = str(i)
            if stage_key in self.adapters:
                tokens = self.adapters[stage_key](tokens, x_size)

        tokens = self.backbone.norm(tokens)
        feat = self.backbone.patch_unembed(tokens, x_size)
        return feat

    def forward(self, x):
        self.backbone.mean = self.backbone.mean.type_as(x)
        x = (x - self.backbone.mean) * self.backbone.img_range

        if self.backbone.upsampler == 'pixelshuffle':
            x = self.backbone.conv_first(x)
            x = self.backbone.conv_after_body(self.forward_features(x)) + x
            x = self.backbone.conv_before_upsample(x)
            x = self.backbone.conv_last(self.backbone.upsample(x))
        elif self.backbone.upsampler == 'pixelshuffledirect':
            x = self.backbone.conv_first(x)
            x = self.backbone.conv_after_body(self.forward_features(x)) + x
            x = self.backbone.upsample(x)
        else:
            x_first = self.backbone.conv_first(x)
            res = self.backbone.conv_after_body(
                self.forward_features(x_first)
            ) + x_first
            x = x + self.backbone.conv_last(res)

        x = x / self.backbone.img_range + self.backbone.mean
        return x

    # ── Weight loading ──

    def load_pretrained_mair(self, path, strict=False, param_key='params'):
        """Load pretrained MaIR backbone weights."""
        if not os.path.exists(path):
            raise FileNotFoundError(f'Pretrained path not found: {path}')

        load_net = torch.load(path, map_location='cpu')
        if param_key:
            if isinstance(load_net, dict) and param_key in load_net:
                load_net = load_net[param_key]
            elif isinstance(load_net, dict):
                for key in ['params_ema', 'params']:
                    if key in load_net:
                        load_net = load_net[key]
                        break
        load_net = {k.replace('module.', ''): v for k, v in load_net.items()}
        has_backbone_prefix = any(
            k.startswith('backbone.') for k in load_net
        )

        try:
            logger = get_root_logger()
        except Exception:
            logger = None

        if has_backbone_prefix:
            backbone_state = {
                k.replace('backbone.', ''): v
                for k, v in load_net.items()
                if k.startswith('backbone.')
            }
            n_loaded, n_skipped = self._load_matching(
                self.backbone, backbone_state, strict=strict,
            )
            # Load adapter weights if present (for resume)
            adapter_state = {
                k.replace('adapters.', ''): v
                for k, v in load_net.items()
                if k.startswith('adapters.')
            }
            n_adapter = 0
            if adapter_state:
                n_adapter, _ = self._load_matching(
                    self.adapters, adapter_state, strict=False,
                )
            if logger:
                logger.info(
                    f'[MaIR_MLVGR] Loaded full ckpt: '
                    f'backbone={n_loaded} (skip={n_skipped}), '
                    f'adapters={n_adapter}.'
                )
        else:
            n_loaded, n_skipped = self._load_matching(
                self.backbone, load_net, strict=strict,
            )
            if logger:
                logger.info(
                    f'[MaIR_MLVGR] Loaded backbone: '
                    f'matched={n_loaded}, skip={n_skipped}.'
                )

        if self.freeze_backbone_after_load:
            self.freeze_backbone()

    def _load_matching(self, module, state_dict, strict=False):
        if strict:
            module.load_state_dict(state_dict, strict=True)
            return len(state_dict), 0
        module_state = module.state_dict()
        matched = {}
        skipped = 0
        for name, value in state_dict.items():
            if name in module_state and module_state[name].shape == value.shape:
                matched[name] = value
            else:
                skipped += 1
        module.load_state_dict(matched, strict=False)
        return len(matched), skipped

    def freeze_backbone(self, exclude=None):
        exclude = set(exclude or [])
        trainable_prefixes = tuple(exclude) + self.backbone_trainable_prefixes
        trainable_names = []
        for name, p in self.backbone.named_parameters():
            keep = (
                name in exclude
                or any(name.startswith(pf) for pf in trainable_prefixes)
                or any(kw in name for kw in self.backbone_trainable_keywords)
            )
            if any(kw in name for kw in self.backbone_forbidden_keywords):
                keep = False
            p.requires_grad = keep
            if keep:
                trainable_names.append(name)
        self._backbone_frozen = True
        try:
            logger = get_root_logger()
        except Exception:
            logger = None
        if logger:
            n_frozen = sum(
                1 for _, p in self.backbone.named_parameters()
                if not p.requires_grad
            )
            preview = ', '.join(trainable_names[:8]) or 'none'
            more = (
                '' if len(trainable_names) <= 8
                else f' (+{len(trainable_names) - 8} more)'
            )
            logger.info(
                f'[MaIR_MLVGR] Backbone: {len(trainable_names)} trainable, '
                f'{n_frozen} frozen. Preview: {preview}{more}'
            )

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
        self._backbone_frozen = False
