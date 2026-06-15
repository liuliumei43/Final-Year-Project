"""
Test-time Local Converter (TLC).

Reference: "Improving Image Restoration by Revisiting Global Information
Aggregation" (Chu et al., ECCV 2022).

Training uses small patches (e.g. 128x128) so global pooling sees a limited
receptive field. At test time the full image is much larger, so global pooling
sees a different statistical distribution -> train/test mismatch, drops PSNR.

TLC fixes this by replacing nn.AdaptiveAvgPool2d(1) with a local avg pooling
window of the training patch size, applied densely (stride 1, reflective pad).
Only touched at inference; training is unchanged.

Usage:
    from basicsr.models.archs.tlc_util import convert_to_tlc
    net_g = define_network(opt['network_g'])
    net_g = convert_to_tlc(net_g, train_size=(1, 3, 128, 128))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalAvgPool2d(nn.Module):
    """Drop-in replacement for nn.AdaptiveAvgPool2d(1) at test time.

    Behaves exactly like global avg pool when H,W <= kernel; otherwise applies
    a sliding-window local avg that matches training-time statistics.
    """

    def __init__(self, base_size):
        super().__init__()
        # base_size: (H_train, W_train) — the spatial size seen during training
        # at this layer's input. For the top-level ShuffleAttn, that's 128x128.
        self.base_size = base_size
        self.auto_pad = True

    def forward(self, x):
        if self.training:
            # Training: fall back to global avg pool (original behavior).
            return F.adaptive_avg_pool2d(x, 1)

        h, w = x.shape[-2:]
        kh = min(self.base_size[0], h)
        kw = min(self.base_size[1], w)

        if kh >= h and kw >= w:
            return F.adaptive_avg_pool2d(x, 1)

        # Integral-image trick: O(HW) regardless of kernel size.
        # out[i,j] = mean of x[i:i+kh, j:j+kw]
        s = x.cumsum(-1).cumsum(-2)
        s = F.pad(s, (1, 0, 1, 0))
        s = (
            s[..., :-kh, :-kw]
            + s[..., kh:, kw:]
            - s[..., :-kh, kw:]
            - s[..., kh:, :-kw]
        ) / (kh * kw)

        # Pad back to input spatial size (so downstream conv still matches).
        if self.auto_pad:
            _, _, ph, pw = s.shape
            pad_h = h - ph
            pad_w = w - pw
            s = F.pad(
                s,
                (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
                mode='replicate',
            )
        return s


def _replace_module(root, name_path, new_module):
    parts = name_path.split('.')
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def convert_to_tlc(model, train_size=(1, 3, 128, 128)):
    """Replace every nn.AdaptiveAvgPool2d(1) in `model` with LocalAvgPool2d.

    Args:
        model: nn.Module (already loaded with weights).
        train_size: (N, C, H, W) — training input size. Only H, W are used.

    Returns the same model (modified in place) for chaining.
    """
    base_h, base_w = train_size[-2], train_size[-1]
    replaced = []
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.AdaptiveAvgPool2d):
            # Only replace when the target is global pool (output_size=1 or (1,1)).
            out = module.output_size
            is_global = out == 1 or out == (1, 1)
            if is_global:
                _replace_module(model, name, LocalAvgPool2d((base_h, base_w)))
                replaced.append(name)

    if len(replaced) == 0:
        print('[TLC] Warning: no AdaptiveAvgPool2d(1) found — nothing replaced.')
    else:
        print(f'[TLC] Replaced {len(replaced)} global pools with local '
              f'{base_h}x{base_w} pools: {replaced}')
    return model
