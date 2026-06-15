"""
MaIRUNet with SSGR (Spectral-Spatial Gated Refinement) adapters.

Cross-task universal design: same SSGRAdapter module used in MaIR (SR/CDN)
is applied here to MaIRUNet (Real Denoising / Motion Deblurring / Dehazing).

SSGRAdapter interface: forward(tokens [B, H*W, C], x_size (H, W)) → tokens
This matches MaIRUNet's inter-block token format exactly.

Adapter insertion points (configurable via adapter_positions):
  - 'enc1': after encoder_level1  (dim=48,   H×W)
  - 'enc2': after encoder_level2  (dim=96,   H/2×W/2)
  - 'enc3': after encoder_level3  (dim=192,  H/4×W/4)
  - 'lat':  after latent          (dim=384,  H/8×W/8)
  - 'dec3': after decoder_level3  (dim=192,  H/4×W/4)
  - 'dec2': after decoder_level2  (dim=96,   H/2×W/2)
  - 'dec1': after decoder_level1  (dim=96,   H×W)

Default: ['enc2', 'enc3', 'lat'] — 3 adapters at deepest encoder levels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

from basicsr.models.archs.mairunet_arch import MaIRUNet
from basicsr.models.archs.shift_scanf_util import (
    mair_ids_generate,
    mair_shift_ids_generate,
)


# ──────────────────────────────────────────────────────────────────────────────
# SSGRAdapter — inlined from basicsr/archs/mair_mlvgr_arch.py to avoid
# cross-package import issues between realDenoising/basicsr and root basicsr.
# Kept identical to the canonical version; update both if you change one.
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
        noise_adaptive_gate=False,
        spectral_mode='additive',
        spectral_modulation_bound=0.2,
        adapter_mode='additive',
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

        mid_refine = max(channels // bottleneck_ratio, 4)
        mid_gate = max(channels // 8, 4)

        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False),
            nn.GELU(),
            nn.Conv2d(channels, mid_refine, 1),
            nn.GELU(),
            nn.Conv2d(mid_refine, channels, 1),
        )
        self.var_gate = nn.Sequential(
            nn.Conv2d(channels * 2, mid_gate, 1),
            nn.GELU(),
            nn.Conv2d(mid_gate, channels, 1),
        )

        mid_spec = max(int(channels * spectral_hidden_ratio), 4)
        self.spectral_enhance = nn.Sequential(
            nn.Conv2d(channels, mid_spec, 1),
            nn.GELU(),
            nn.Conv2d(mid_spec, channels, 1),
        )
        self.spectral_scale = nn.Parameter(torch.zeros(1))
        self.hf_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

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
        gx = F.pad(x[:, :, :, 1:] - x[:, :, :, :-1], (0, 1, 0, 0), mode='replicate')
        gy = F.pad(x[:, :, 1:, :] - x[:, :, :-1, :], (0, 0, 0, 1), mode='replicate')
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
                          C_src may differ from C — stats are projected to match C channels.
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
        # stats_src: the tensor from which var/grad are computed
        if stats_source is not None:
            # Resize stats_source to match spatial dims if needed
            s = stats_source
            if s.shape[2] != h or s.shape[3] != w:
                s = F.interpolate(s, size=(h, w), mode='bilinear', align_corners=False)
        else:
            s = x

        pad = self.var_kernel // 2
        mu = F.avg_pool2d(s, self.var_kernel, stride=1, padding=pad)
        var = F.avg_pool2d((s - mu).square(), self.var_kernel, stride=1, padding=pad)
        grad = self._local_gradient(s)

        # Project stats to C channels if stats_source has different channel count
        # var: [B, C_src, H, W], grad: [B, C_src, H, W] → need [B, C, H, W]
        if var.shape[1] != c:
            # Expand: repeat stats across channels (e.g. 3→48 or 3→96)
            # Mean across source channels → [B,1,H,W] → expand to [B,C,H,W]
            var = var.mean(dim=1, keepdim=True).expand(-1, c, -1, -1)
            grad = grad.mean(dim=1, keepdim=True).expand(-1, c, -1, -1)

        # ── Spatial branch ──
        if self.noise_adaptive_gate:
            noise_floor = var.mean(dim=[2, 3], keepdim=True)
            signal_var = F.relu(var - noise_floor)
            gate_input = torch.cat([grad, signal_var], dim=1)
        else:
            gate_input = torch.cat([var, grad], dim=1)

        raw_gate = torch.sigmoid(self.var_gate(gate_input))

        if self.adapter_mode == 'subtractive':
            # Invert: high gate in flat/noisy regions → strong noise removal
            gate = 1.0 - raw_gate
            noise_est = gate * self.refine(x)
            spatial_delta = -noise_est
        else:
            gate = raw_gate
            spatial_delta = gate * self.refine(x)

        # ── Spectral branch ──
        x_fft = torch.fft.rfft2(x, norm='ortho')
        mag = x_fft.abs()
        phase = x_fft.angle()

        if self.spectral_mode == 'suppressive':
            # v3.2: learned frequency suppression mask [0, 1]
            # Acts as content-adaptive band-stop/low-pass filter
            raw_mask = self.spectral_enhance(mag)
            spec_w = self.spectral_scale.tanh().abs()  # [0, 1] strength
            mag_new = mag * (1.0 - spec_w * torch.sigmoid(raw_mask))
        elif self.spectral_mode == 'multiplicative':
            raw_mask = self.spectral_enhance(mag)
            spec_w = self.spectral_scale.tanh() * self.spectral_modulation_bound
            mag_new = mag * (1.0 + spec_w * torch.tanh(raw_mask))
        else:
            mag_delta = self.spectral_enhance(mag)
            spec_w = self.spectral_scale.tanh() * 0.2
            mag_new = mag + spec_w * mag_delta

        x_spec = torch.fft.irfft2(torch.polar(mag_new, phase), s=(h, w), norm='ortho')
        spectral_delta = x_spec - x

        # ── HF shortcut ──
        # Always from x itself (high-freq of features, not stats source)
        mu_x = F.avg_pool2d(x, self.var_kernel, stride=1, padding=pad) if stats_source is not None else mu
        hf = x - mu_x
        hf_weight = self.hf_scale.tanh() * self.hf_shortcut_bound

        # ── Combine ──
        x_out = x + spatial_delta + spectral_delta + hf_weight * hf
        if input_is_tokens:
            return x_out.permute(0, 2, 3, 1).reshape(b, h * w, c)
        else:
            return x_out


# Channel dims at each MaIRUNet position
_POSITION_CHANNELS = {
    'enc1': lambda dim: dim,             # 48
    'enc2': lambda dim: dim * 2,         # 96
    'enc3': lambda dim: dim * 4,         # 192
    'lat':  lambda dim: dim * 8,         # 384
    'dec3': lambda dim: dim * 4,         # 192
    'dec2': lambda dim: dim * 2,         # 96
    'dec1': lambda dim: dim * 2,         # 96 (skip concat keeps 2×dim)
}


class MaIRUNet_SSGR(nn.Module):
    """
    MaIRUNet with SSGR adapters for cross-task image restoration.

    Wraps the original MaIRUNet backbone and inserts SSGRAdapter modules
    at configurable positions within the U-Net hierarchy. Each adapter
    automatically uses the correct channel dimension for its position.
    """

    def __init__(
        self,
        adapter_positions=None,
        adapter_bottleneck_ratio=2,
        adapter_var_kernel=5,
        adapter_gate_init_bias=-3.0,
        adapter_spectral_hidden_ratio=0.25,
        adapter_hf_shortcut_bound=0.15,
        # v3.1 denoising-adaptive extensions
        adapter_noise_adaptive_gate=False,
        adapter_spectral_mode='additive',
        adapter_spectral_modulation_bound=0.2,
        # v3.2: adapter mode
        adapter_mode='additive',
        # v3.3: positions that receive stats_source=inp_img
        adapter_stats_from_input=None,
        # Backbone freezing: True = freeze all, list = freeze all EXCEPT these prefixes
        freeze_backbone=False,
        backbone_trainable_prefixes=None,
        **unet_kwargs,
    ):
        super().__init__()
        self.backbone = MaIRUNet(**unet_kwargs)

        # Freeze backbone
        if freeze_backbone:
            trainable_prefixes = tuple(backbone_trainable_prefixes or [])
            for name, p in self.backbone.named_parameters():
                if trainable_prefixes and any(name.startswith(pf) for pf in trainable_prefixes):
                    p.requires_grad = True  # keep trainable
                else:
                    p.requires_grad = False

        base_dim = unet_kwargs.get('dim', 48)

        if adapter_positions is None:
            adapter_positions = ['enc2', 'enc3', 'lat']
        self.adapter_positions = adapter_positions

        # Positions that use noisy input image for gate statistics
        # (instead of computing stats from feature tokens)
        self.stats_from_input = set(adapter_stats_from_input or [])

        # Inter-layer adapters (token-space)
        self.adapters = nn.ModuleDict()
        for pos in self.adapter_positions:
            if pos not in _POSITION_CHANNELS:
                raise ValueError(
                    f"Unknown adapter position '{pos}'. "
                    f"Valid: {list(_POSITION_CHANNELS.keys())}"
                )
            ch = _POSITION_CHANNELS[pos](base_dim)
            self.adapters[pos] = SSGRAdapter(
                ch,
                bottleneck_ratio=adapter_bottleneck_ratio,
                var_kernel=adapter_var_kernel,
                gate_init_bias=adapter_gate_init_bias,
                spectral_hidden_ratio=adapter_spectral_hidden_ratio,
                hf_shortcut_bound=adapter_hf_shortcut_bound,
                noise_adaptive_gate=adapter_noise_adaptive_gate,
                spectral_mode=adapter_spectral_mode,
                spectral_modulation_bound=adapter_spectral_modulation_bound,
                adapter_mode=adapter_mode,
            )

    def load_state_dict(self, state_dict, strict=True):
        """Override to handle MaIRUNet checkpoints without 'backbone.' prefix.

        If the checkpoint has keys like 'encoder_level1.0.xxx' (raw MaIRUNet),
        remap them to 'backbone.encoder_level1.0.xxx'. Adapter keys (not in
        the checkpoint) will be left at their zero-init values.
        """
        has_backbone_prefix = any(
            k.startswith('backbone.') for k in state_dict
        )
        if not has_backbone_prefix:
            # Remap raw MaIRUNet keys → backbone.* keys
            remapped = {}
            for k, v in state_dict.items():
                remapped[f'backbone.{k}'] = v
            state_dict = remapped
            strict = False  # adapter keys won't be in checkpoint

        return super().load_state_dict(state_dict, strict=strict)

    def _apply_adapter(self, tokens, pos, x_size, stats_source=None):
        """Apply adapter if registered at this position."""
        if pos in self.adapters:
            ss = stats_source if pos in self.stats_from_input else None
            tokens = self.adapters[pos](tokens, x_size, stats_source=ss)
        return tokens

    def forward(self, inp_img):
        B, C, H, W = inp_img.shape
        bb = self.backbone

        # Generate scan IDs — always regenerate to avoid stale IDs after
        # validation (eval-mode forward overwrites IDs with val image size,
        # but trainig_img_size is unchanged → training skips regeneration).
        bb._generate_ids((B, C, H, W))
        if bb.training:
            bb.trainig_img_size = H

        ids_l1 = (bb.xs_scan_ids_l1, bb.xs_inverse_ids_l1,
                  bb.xs_shift_scan_ids_l1, bb.xs_shift_inverse_ids_l1)
        ids_l2 = (bb.xs_scan_ids_l2, bb.xs_inverse_ids_l2,
                  bb.xs_shift_scan_ids_l2, bb.xs_shift_inverse_ids_l2)
        ids_l3 = (bb.xs_scan_ids_l3, bb.xs_inverse_ids_l3,
                  bb.xs_shift_scan_ids_l3, bb.xs_shift_inverse_ids_l3)
        ids_lat = (bb.xs_scan_ids_lat, bb.xs_inverse_ids_lat,
                   bb.xs_shift_scan_ids_lat, bb.xs_shift_inverse_ids_lat)

        # ── Encoder ──
        inp_enc_level1 = bb.patch_embed(inp_img)
        out_enc_level1 = inp_enc_level1
        for layer in bb.encoder_level1:
            out_enc_level1 = layer(out_enc_level1, ids_l1, [H, W])
        out_enc_level1 = self._apply_adapter(out_enc_level1, 'enc1', (H, W), inp_img)

        inp_enc_level2 = bb.down1_2(out_enc_level1, H, W)
        out_enc_level2 = inp_enc_level2
        for layer in bb.encoder_level2:
            out_enc_level2 = layer(out_enc_level2, ids_l2, [H // 2, W // 2])
        out_enc_level2 = self._apply_adapter(
            out_enc_level2, 'enc2', (H // 2, W // 2), inp_img,
        )

        inp_enc_level3 = bb.down2_3(out_enc_level2, H // 2, W // 2)
        out_enc_level3 = inp_enc_level3
        for layer in bb.encoder_level3:
            out_enc_level3 = layer(out_enc_level3, ids_l3, [H // 4, W // 4])
        out_enc_level3 = self._apply_adapter(
            out_enc_level3, 'enc3', (H // 4, W // 4), inp_img,
        )

        inp_enc_level4 = bb.down3_4(out_enc_level3, H // 4, W // 4)
        latent = inp_enc_level4
        for layer in bb.latent:
            latent = layer(latent, ids_lat, [H // 8, W // 8])
        latent = self._apply_adapter(latent, 'lat', (H // 8, W // 8), inp_img)

        # ── Decoder ──
        from einops import rearrange

        inp_dec_level3 = bb.up4_3(latent, H // 8, W // 8)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 2)
        inp_dec_level3 = rearrange(
            inp_dec_level3, "b (h w) c -> b c h w", h=H // 4, w=W // 4,
        ).contiguous()
        inp_dec_level3 = bb.reduce_chan_level3(inp_dec_level3)
        inp_dec_level3 = rearrange(
            inp_dec_level3, "b c h w -> b (h w) c",
        ).contiguous()
        out_dec_level3 = inp_dec_level3
        for layer in bb.decoder_level3:
            out_dec_level3 = layer(out_dec_level3, ids_l3, [H // 4, W // 4])
        out_dec_level3 = self._apply_adapter(
            out_dec_level3, 'dec3', (H // 4, W // 4), inp_img,
        )

        inp_dec_level2 = bb.up3_2(out_dec_level3, H // 4, W // 4)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 2)
        inp_dec_level2 = rearrange(
            inp_dec_level2, "b (h w) c -> b c h w", h=H // 2, w=W // 2,
        ).contiguous()
        inp_dec_level2 = bb.reduce_chan_level2(inp_dec_level2)
        inp_dec_level2 = rearrange(
            inp_dec_level2, "b c h w -> b (h w) c",
        ).contiguous()
        out_dec_level2 = inp_dec_level2
        for layer in bb.decoder_level2:
            out_dec_level2 = layer(out_dec_level2, ids_l2, [H // 2, W // 2])
        out_dec_level2 = self._apply_adapter(
            out_dec_level2, 'dec2', (H // 2, W // 2), inp_img,
        )

        inp_dec_level1 = bb.up2_1(out_dec_level2, H // 2, W // 2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 2)
        out_dec_level1 = inp_dec_level1
        for layer in bb.decoder_level1:
            out_dec_level1 = layer(out_dec_level1, ids_l1, [H, W])

        for layer in bb.refinement:
            out_dec_level1 = layer(out_dec_level1, ids_l1, [H, W])
        out_dec_level1 = self._apply_adapter(
            out_dec_level1, 'dec1', (H, W), inp_img,
        )

        out_dec_level1 = rearrange(
            out_dec_level1, "b (h w) c -> b c h w", h=H, w=W,
        ).contiguous()

        if bb.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + bb.skip_conv(inp_enc_level1)
            return bb.output(out_dec_level1)
        else:
            return bb.output(out_dec_level1) + inp_img
