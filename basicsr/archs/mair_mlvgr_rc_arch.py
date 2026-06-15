"""
MaIR-MLVGR with RC-SSGR: Residual-Calibrated Spectral-Spatial Gated Refinement.

Literature anchors for thesis writing:
  [Mamba23] Selective state-space models for efficient long-sequence modeling.
  [MaIR25] Locality/continuity-preserving Mamba backbone for image restoration.
  [EDSR17] Residual scaling/stabilized residual learning in super-resolution.
  [RCAN18] Channel attention for adaptive channel-wise feature recalibration.
  [FFL21] Frequency-domain gaps and frequency losses for reconstruction.
  [FFC20] Spatial-spectral feature processing with Fourier-domain operators.

KEY INSIGHT (diagnosed from A0/A1/A2/A5 experiments):
  Original SSGR follows an "enhancement" paradigm (x + deltas), which:
    - Can modify low-frequency content (hurts Set5/Set14/B100)
    - Applies corrections uniformly (over-processes smooth regions)
    - Duplicates FFTFreqLoss's frequency supervision
    - Has no final calibration of aggregated deltas

RC-SSGR PARADIGM SHIFT: "enhancement" → "calibration"
  x_out = x + α_stage · confidence · high_freq_only(fuse(spatial_delta, spectral_delta, hf))

Five safety mechanisms:
  1. Cross-domain fusion:   Merge spatial/spectral/hf deltas into one (no triple sum)
  2. Low-frequency protection: Apply high-pass to fused delta (never touch LF)
  3. Feature-scale bound:   tanh(delta/feat_std)*feat_std*bound (relative scaling)
  4. Confidence gating:     Only modify in reliable HF regions
  5. Stage-aware scale:     Early stages more conservative (0.015/0.030/0.050)

Registered architectures:
  MaIR_MLVGR with adapter_type='rcssgr'  (new adapter type in original arch)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.mair_mlvgr_arch import MaIR_MLVGR


class RCSSGRAdapter(nn.Module):
    """
    Residual-Calibrated Spectral-Spatial Gated Refinement.

    Key difference from SSGRAdapter:
      SSGR: x_out = x + spatial_delta + spectral_delta + hf_weight * hf
      RC-SSGR:
        fused_delta = delta_fuse(x, spatial_delta, spectral_delta, hf)
        fused_delta_hf = high_pass(fused_delta)              # low-freq protection
        bounded = tanh(fused_delta_hf / feat_std) * feat_std * delta_bound
        confidence = sigmoid(MLP(var, grad, hf_energy))
        x_out = x + stage_scale * confidence * bounded

    This replaces the "uniform enhancement" paradigm with
    "region-confident, bounded, high-frequency residual calibration".
    """

    def __init__(
        self,
        channels,
        stage_idx=1,
        bottleneck_ratio=2,
        var_kernel=5,
        gate_init_bias=-2.0,
        spectral_hidden_ratio=0.125,
        spectral_mode='multiplicative',
        spectral_modulation_bound=0.05,
        max_res_scale=None,
        delta_bound=0.30,
        # Ablation toggles
        use_spatial=True,
        use_spectral=True,
        use_confidence=True,
        use_highpass=True,
        use_stage_scale=True,
        use_delta_bound=True,
        use_channel_calib=True,
    ):
        super().__init__()
        self.channels = channels
        self.stage_idx = stage_idx
        self.var_kernel = max(var_kernel, 3)
        if self.var_kernel % 2 == 0:
            self.var_kernel += 1

        self.spectral_mode = spectral_mode
        self.spectral_modulation_bound = spectral_modulation_bound
        self.delta_bound = delta_bound

        # Stage-wise max residual scale: earlier stages more conservative
        # (Raised from [0.015/0.030/0.050] — those were too restrictive and
        # made effective gradient ~1e-10, preventing learning in 30k iter.)
        if max_res_scale is None:
            if stage_idx == 0:
                max_res_scale = 0.05
            elif stage_idx == 1:
                max_res_scale = 0.10
            else:
                max_res_scale = 0.15
        self.max_res_scale = float(max_res_scale)

        # Ablation toggles (store on self for forward)
        self.use_spatial = use_spatial
        self.use_spectral = use_spectral
        self.use_confidence = use_confidence
        self.use_highpass = use_highpass
        self.use_stage_scale = use_stage_scale
        self.use_delta_bound = use_delta_bound
        self.use_channel_calib = use_channel_calib

        mid_refine = max(channels // bottleneck_ratio, 4)
        mid_gate = max(channels // 8, 4)
        mid_spec = max(int(channels * spectral_hidden_ratio), 4)

        # ── Spatial branch: local structure residual ──
        self.spatial_refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False),
            nn.GELU(),
            nn.Conv2d(channels, mid_refine, 1),
            nn.GELU(),
            nn.Conv2d(mid_refine, channels, 1),
        )

        # Spatial gate: variance + gradient
        self.var_gate = nn.Sequential(
            nn.Conv2d(channels * 2, mid_gate, 1),
            nn.GELU(),
            nn.Conv2d(mid_gate, channels, 1),
        )

        # ── Spectral branch: magnitude modulation ──
        self.spectral_enhance = nn.Sequential(
            nn.Conv2d(channels, mid_spec, 1),
            nn.GELU(),
            nn.Conv2d(mid_spec, channels, 1),
        )
        # Small non-zero init so spectral_enhance receives gradient from t=0.
        # tanh(0.02) * 0.05 ≈ 0.001 — negligible perturbation, but non-zero grad flow.
        self.spectral_scale = nn.Parameter(torch.tensor([0.02]))

        # ── Cross-domain residual fusion ──
        # Input: x, spatial_delta, spectral_delta, hf  → delta
        self.delta_fuse = nn.Sequential(
            nn.Conv2d(channels * 4, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )

        # ── Confidence map: where RC-SSGR is allowed to modify ──
        # Input: local variance, gradient, high-frequency energy
        self.confidence = nn.Sequential(
            nn.Conv2d(channels * 3, mid_gate, 1),
            nn.GELU(),
            nn.Conv2d(mid_gate, channels, 1),
        )

        # Global stage residual scale — initialize to 0.5 so gradients are meaningful.
        # tanh(0.5) ≈ 0.46. With max_res_scale=0.10 (stage 1),
        # initial effective scale = 0.46 × 0.10 = 0.046.
        # Combined with confidence (~0.12) and delta_bound (0.30):
        # max effective perturbation = 0.046 × 0.12 × 0.30 ≈ 1.7e-3
        # (100× larger than previous 3e-5, still safe, allows real learning.)
        # Residual scaling is inspired by EDSR [EDSR17]. Here it is stage-aware
        # and bounded, so RC-SSGR behaves as a calibrator rather than a new trunk.
        # Channel-wise residual calibration inspired by RCAN [RCAN18].
        # The initial multiplier is exactly 1, so the adapter starts unchanged:
        #   delta <- delta * (1 + 0.1 * tanh(channel_calib(delta))).
        # It can then suppress harmful correction channels and emphasize useful
        # high-frequency channels without changing the backbone architecture.
        self.channel_calib = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid_gate, 1),
            nn.GELU(),
            nn.Conv2d(mid_gate, channels, 1),
        )

        self.res_scale = nn.Parameter(torch.tensor([0.5]))

        # ── Near-identity initialization ──
        # Rationale: zero-init everywhere would block gradients from flowing
        # into spatial/spectral branches (delta_fuse[-1]=0 multiplicatively
        # kills grad for its inputs). We use tiny-normal init on branch outputs
        # and fusion, so grads flow, but initial perturbation is negligible
        # (res_scale + confidence + delta_bound still bound the output).

        # Spatial branch: small init (raised from 1e-5 to 1e-3 so branch output
        # is observable; 5 safety layers (gate, confidence, scale, bound) still bound final effect)
        nn.init.normal_(self.spatial_refine[-1].weight, std=1e-3)
        if self.spatial_refine[-1].bias is not None:
            nn.init.zeros_(self.spatial_refine[-1].bias)

        # Gate: zero-init + negative bias → initial gate ≈ sigmoid(-2) ≈ 0.12
        nn.init.zeros_(self.var_gate[-1].weight)
        nn.init.constant_(self.var_gate[-1].bias, gate_init_bias)

        # Spectral branch: small init (same rationale as spatial)
        nn.init.normal_(self.spectral_enhance[-1].weight, std=1e-3)
        if self.spectral_enhance[-1].bias is not None:
            nn.init.zeros_(self.spectral_enhance[-1].bias)

        # Fusion: raised to 1e-2 — this is the gate for gradient flow into all branches.
        # With input channels = 4C, each branch contributes roughly equally. Output
        # magnitude ≈ 1e-2 * |input| ≈ 1e-2. Combined with high-pass, delta_bound(0.30),
        # confidence(0.12), scale(0.046): final perturbation ≈ 1.6e-5 per pixel — safe.
        nn.init.normal_(self.delta_fuse[-1].weight, std=1e-2)
        if self.delta_fuse[-1].bias is not None:
            nn.init.zeros_(self.delta_fuse[-1].bias)

        # Confidence: zero-init + negative bias → initial confidence ≈ 0.12
        nn.init.zeros_(self.confidence[-1].weight)
        nn.init.constant_(self.confidence[-1].bias, gate_init_bias)

        nn.init.zeros_(self.channel_calib[-1].weight)
        if self.channel_calib[-1].bias is not None:
            nn.init.zeros_(self.channel_calib[-1].bias)

    def _local_gradient(self, x):
        gx = F.pad(
            x[:, :, :, 1:] - x[:, :, :, :-1],
            (0, 1, 0, 0), mode='replicate',
        )
        gy = F.pad(
            x[:, :, 1:, :] - x[:, :, :-1, :],
            (0, 0, 0, 1), mode='replicate',
        )
        grad = gx.abs() + gy.abs()
        return self._avg_pool_same(grad)

    def _avg_pool_same(self, x):
        pad = self.var_kernel // 2
        x = F.pad(x, (pad, pad, pad, pad), mode='replicate')
        return F.avg_pool2d(x, self.var_kernel, stride=1, padding=0)

    def _high_pass(self, x):
        low = self._avg_pool_same(x)
        return x - low

    def forward(self, tokens, x_size, stats_source=None):
        # Accept optional stats_source for API compatibility (ignored for RC-SSGR)
        input_is_tokens = tokens.dim() == 3

        if input_is_tokens:
            b, hw, c = tokens.shape
            h, w = x_size
            x = tokens.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        else:
            b, c, h, w = tokens.shape
            x = tokens

        # Local statistics
        mu = self._avg_pool_same(x)
        var = self._avg_pool_same((x - mu).square())
        grad = self._local_gradient(x)
        hf = x - mu
        hf_energy = self._avg_pool_same(hf.abs())

        # ── Spatial residual ──
        if self.use_spatial:
            raw_gate = torch.sigmoid(self.var_gate(torch.cat([var, grad], dim=1)))
            spatial_delta = raw_gate * self.spatial_refine(x)
        else:
            spatial_delta = torch.zeros_like(x)

        # ── Spectral residual ──
        # Frequency-domain losses help expose reconstruction gaps not captured
        # by pixel losses alone [FFL21]. We use this branch as a lightweight
        # feature-level magnitude correction rather than a dominant objective.
        if self.use_spectral:
            x_fft = torch.fft.rfft2(x, norm='ortho')
            mag = x_fft.abs()
            # log1p compresses the heavy-tailed FFT magnitude distribution and
            # prevents the DC/low-frequency bins from dominating the tiny
            # spectral adapter. This is more stable for PSNR-oriented SR.
            raw_mask = self.spectral_enhance(torch.log1p(mag))
            if self.spectral_mode == 'multiplicative':
                spec_w = self.spectral_scale.tanh() * self.spectral_modulation_bound
                modulation = (1.0 + spec_w * torch.tanh(raw_mask)).clamp_min(1e-4)
                # Preserve the original complex phase and only calibrate the
                # magnitude. This follows the spatial-spectral motivation of
                # Fourier-domain operators such as FFC [FFC20], while avoiding
                # angle/polar reconstruction noise for tiny residual updates.
                x_spec = torch.fft.irfft2(
                    x_fft * modulation.to(dtype=x_fft.dtype),
                    s=(h, w),
                    norm='ortho',
                )
            else:
                spec_w = self.spectral_scale.tanh() * self.spectral_modulation_bound
                mag_new = mag + spec_w * raw_mask
                phase = x_fft.angle()
                mag_new = mag_new.clamp_min(1e-8)                   # numerical safety
                x_spec = torch.fft.irfft2(
                    torch.polar(mag_new, phase), s=(h, w), norm='ortho',
                )
            spectral_delta = x_spec - x
        else:
            spectral_delta = torch.zeros_like(x)

        # ── Cross-domain fusion ──
        delta = self.delta_fuse(
            torch.cat([x, spatial_delta, spectral_delta, hf], dim=1)
        )

        # ── Low-frequency protection ──
        if self.use_channel_calib:
            delta = delta * (1.0 + 0.1 * torch.tanh(self.channel_calib(delta)))

        if self.use_highpass:
            delta = self._high_pass(delta)
            # Remove residual DC leakage after high-pass filtering. This keeps
            # RC-SSGR focused on texture/detail correction and protects PSNR on
            # smooth or low-frequency-dominant images.
            delta = delta - delta.mean(dim=(-2, -1), keepdim=True)

        # ── Feature-scale bound ──
        if self.use_delta_bound:
            feat_scale = x.detach().flatten(2).std(
                dim=-1, unbiased=False,
            ).view(b, c, 1, 1).clamp_min(1e-6)
            delta = torch.tanh(delta / feat_scale) * feat_scale * self.delta_bound

        # ── Confidence map ──
        if self.use_confidence:
            confidence = torch.sigmoid(
                self.confidence(torch.cat([var, grad, hf_energy], dim=1))
            )
        else:
            confidence = 1.0

        # ── Stage-aware residual scale ──
        if self.use_stage_scale:
            scale = self.res_scale.tanh() * self.max_res_scale
        else:
            scale = self.res_scale.tanh() * 0.05

        # ── Final: bounded, confident, HF-only calibration ──
        x_out = x + scale * confidence * delta

        if input_is_tokens:
            return x_out.permute(0, 2, 3, 1).reshape(b, h * w, c)
        return x_out


@ARCH_REGISTRY.register()
class MaIR_MLVGR_RC(MaIR_MLVGR):
    """
    MaIR_MLVGR with RC-SSGR adapters.

    Usage in yml:
      network_g:
        type: MaIR_MLVGR_RC
        adapter_type: 'ssgr'    # lets parent class create adapter slots;
                                # all will be replaced by RCSSGRAdapter below
        adapter_stages: [0, 1, 2]

        # RC-SSGR hyperparameters
        adapter_gate_init_bias: -3.0
        adapter_spectral_hidden_ratio: 0.125
        adapter_spectral_mode: 'multiplicative'
        adapter_spectral_modulation_bound: 0.05
        adapter_delta_bound: 0.20

        # Optional ablation toggles (all default True)
        adapter_use_spatial: true
        adapter_use_spectral: true
        adapter_use_confidence: true
        adapter_use_highpass: true
        adapter_use_stage_scale: true
        adapter_use_delta_bound: true
        adapter_use_channel_calib: true
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        adapter_bottleneck_ratio = kwargs.get('adapter_bottleneck_ratio', 2)
        adapter_var_kernel = kwargs.get('adapter_var_kernel', 5)
        adapter_gate_init_bias = kwargs.get('adapter_gate_init_bias', -3.0)
        adapter_spectral_hidden_ratio = kwargs.get(
            'adapter_spectral_hidden_ratio', 0.125
        )
        adapter_spectral_mode = kwargs.get('adapter_spectral_mode', 'multiplicative')
        adapter_spectral_modulation_bound = kwargs.get(
            'adapter_spectral_modulation_bound', 0.05
        )
        adapter_delta_bound = kwargs.get('adapter_delta_bound', 0.20)

        # Ablation toggles
        adapter_use_spatial = kwargs.get('adapter_use_spatial', True)
        adapter_use_spectral = kwargs.get('adapter_use_spectral', True)
        adapter_use_confidence = kwargs.get('adapter_use_confidence', True)
        adapter_use_highpass = kwargs.get('adapter_use_highpass', True)
        adapter_use_stage_scale = kwargs.get('adapter_use_stage_scale', True)
        adapter_use_delta_bound = kwargs.get('adapter_use_delta_bound', True)
        adapter_use_channel_calib = kwargs.get('adapter_use_channel_calib', True)

        embed_dim = self.backbone.embed_dim

        # Replace all adapters with RCSSGRAdapter
        new_adapters = nn.ModuleDict()
        for stage_idx_str in self.adapters.keys():
            stage_idx = int(stage_idx_str)
            new_adapters[stage_idx_str] = RCSSGRAdapter(
                embed_dim,
                stage_idx=stage_idx,
                bottleneck_ratio=adapter_bottleneck_ratio,
                var_kernel=adapter_var_kernel,
                gate_init_bias=adapter_gate_init_bias,
                spectral_hidden_ratio=adapter_spectral_hidden_ratio,
                spectral_mode=adapter_spectral_mode,
                spectral_modulation_bound=adapter_spectral_modulation_bound,
                delta_bound=adapter_delta_bound,
                use_spatial=adapter_use_spatial,
                use_spectral=adapter_use_spectral,
                use_confidence=adapter_use_confidence,
                use_highpass=adapter_use_highpass,
                use_stage_scale=adapter_use_stage_scale,
                use_delta_bound=adapter_use_delta_bound,
                use_channel_calib=adapter_use_channel_calib,
            )
        self.adapters = new_adapters

        # Ensure backbone freezing takes effect even when BasicSR's standard
        # load_network() is used (instead of the custom load_pretrained_mair).
        # Without this, Stage 1 may silently train backbone params too.
        if self.freeze_backbone_after_load:
            self.freeze_backbone()
