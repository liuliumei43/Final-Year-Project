import math
import torch
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F

# from basicsr.archs.vgg_arch import VGGFeatureExtractor
from basicsr.utils.registry import LOSS_REGISTRY
from .loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


@weighted_loss
def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target)**2 + eps)


@LOSS_REGISTRY.register()
class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * mse_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class YChannelMSELoss(nn.Module):
    """Metric-aligned luminance MSE loss for SR benchmarks.

    SR benchmarks in this repo report PSNR/SSIM on the Y channel after
    cropping borders. This loss mirrors that evaluation target in training so
    fine-tuning is pushed toward the actual reported metric instead of only RGB
    reconstruction quality.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', crop_border=0):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')
        self.loss_weight = float(loss_weight)
        self.reduction = reduction
        self.crop_border = max(int(crop_border), 0)

    @staticmethod
    def _rgb_to_y(x):
        # Tensors are RGB in [0, 1]. The evaluation path converts RGB tensor ->
        # BGR image -> Matlab-style BT.601 Y, which is equivalent to:
        # Y = 16 + 65.481 R + 128.553 G + 24.966 B, then divided by 255.
        r = x[:, 0:1, :, :]
        g = x[:, 1:2, :, :]
        b = x[:, 2:3, :, :]
        y = 16.0 + 65.481 * r + 128.553 * g + 24.966 * b
        return y / 255.0

    def forward(self, pred, target, weight=None, **kwargs):
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1e4, neginf=-1e4)
        target = torch.nan_to_num(target, nan=0.0, posinf=1e4, neginf=-1e4)
        pred_y = self._rgb_to_y(pred)
        target_y = self._rgb_to_y(target)

        if self.crop_border > 0:
            pred_y = pred_y[:, :, self.crop_border:-self.crop_border, self.crop_border:-self.crop_border]
            target_y = target_y[:, :, self.crop_border:-self.crop_border, self.crop_border:-self.crop_border]

        return self.loss_weight * mse_loss(pred_y, target_y, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps (float): A value used to control the curvature near zero. Default: 1e-12.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-12):
        super(CharbonnierLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * charbonnier_loss(pred, target, weight, eps=self.eps, reduction=self.reduction)


@LOSS_REGISTRY.register()
class FFTFreqLoss(nn.Module):
    """频域损失：在 FFT 幅度（及可选的高频加权）上约束 pred 与 target 一致。
    用于配合 FFT 分支训练，避免频域分支学到噪声、促进恢复正确高频成分。
    Args:
        loss_weight: 损失权重，默认 0.1。
        use_high_freq_weight: 是否对高频分量加权（超分中高频更重要），默认 True。
        high_freq_gamma: 高频权重指数，越大越强调高频。默认 2.0。
    """
    def __init__(self, loss_weight=0.1, use_high_freq_weight=True, high_freq_gamma=2.0, reduction='mean'):
        super().__init__()
        self.loss_weight = loss_weight
        self.use_high_freq_weight = use_high_freq_weight
        self.high_freq_gamma = high_freq_gamma
        self.reduction = reduction

    def _get_high_freq_weight(self, H, W2, device):
        """ 频域权重 (H, W2)，W2 为 rfft2 后频域宽度 (= W//2+1)。中心低频、外围高频。 """
        h = torch.linspace(-1, 1, H, device=device)
        w = torch.linspace(-1, 1, W2, device=device)
        grid_h, grid_w = torch.meshgrid(h, w, indexing='ij')
        r = (grid_h ** 2 + grid_w ** 2).sqrt().clamp(1e-6, 1)
        weight = (r ** self.high_freq_gamma)
        return weight

    def forward(self, pred, target, weight=None, **kwargs):
        """
        pred/target: [B, C, H, W]，通常为 RGB 或 Y 通道。
        """
        # 2D 实 FFT，与 FFTBlock 一致
        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')
        B, C, H, W2 = pred_fft.shape
        # 幅度 L1 更稳
        pred_mag = pred_fft.abs()
        target_mag = target_fft.abs()
        diff = (pred_mag - target_mag).abs()
        if self.use_high_freq_weight:
            w = self._get_high_freq_weight(H, W2, pred.device)
            w = w.unsqueeze(0).unsqueeze(0)
            diff = diff * w
        if self.reduction == 'mean':
            loss = diff.mean()
        elif self.reduction == 'sum':
            loss = diff.sum()
        else:
            loss = diff
        return self.loss_weight * loss


@LOSS_REGISTRY.register()
class DualDomainFreqLoss(nn.Module):
    """Amplitude-phase-wavelet-edge consistency loss for dual-domain SR."""

    def __init__(self, loss_weight=1.0, amplitude_weight=1.0, phase_weight=0.5,
                 wavelet_weight=0.5, edge_weight=0.25, use_high_freq_weight=True,
                 high_freq_gamma=1.4, phase_safe_mag=1e-3, detail_boost=1.5,
                 reduction='mean'):
        super().__init__()
        self.loss_weight = loss_weight
        self.amplitude_weight = float(amplitude_weight)
        self.phase_weight = float(phase_weight)
        self.wavelet_weight = float(wavelet_weight)
        self.edge_weight = float(edge_weight)
        self.use_high_freq_weight = use_high_freq_weight
        self.high_freq_gamma = float(high_freq_gamma)
        self.phase_safe_mag = float(phase_safe_mag)
        self.detail_boost = float(detail_boost)
        self.reduction = reduction

    def _reduce(self, tensor):
        if self.reduction == 'mean':
            return tensor.mean()
        if self.reduction == 'sum':
            return tensor.sum()
        return tensor

    def _get_high_freq_weight(self, h, w2, device, dtype):
        hh = torch.linspace(-1, 1, h, device=device, dtype=dtype)
        ww = torch.linspace(-1, 1, w2, device=device, dtype=dtype)
        grid_h, grid_w = torch.meshgrid(hh, ww, indexing='ij')
        radius = (grid_h ** 2 + grid_w ** 2).sqrt().clamp(1e-6, 1)
        return radius ** self.high_freq_gamma

    def _haar_dwt(self, x):
        h, w = x.shape[-2:]
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')

        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]

        ll = (x00 + x01 + x10 + x11) * 0.5
        lh = (x00 - x01 + x10 - x11) * 0.5
        hl = (x00 + x01 - x10 - x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5
        return ll, lh, hl, hh

    def _sobel(self, x):
        dtype = x.dtype
        device = x.device
        kernel_x = x.new_tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=dtype, device=device)
        kernel_y = x.new_tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=dtype, device=device)
        kernel_x = kernel_x.view(1, 1, 3, 3).repeat(x.shape[1], 1, 1, 1)
        kernel_y = kernel_y.view(1, 1, 3, 3).repeat(x.shape[1], 1, 1, 1)
        grad_x = F.conv2d(x, kernel_x, padding=1, groups=x.shape[1])
        grad_y = F.conv2d(x, kernel_y, padding=1, groups=x.shape[1])
        return grad_x, grad_y

    def forward(self, pred, target, weight=None, **kwargs):
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1e4, neginf=-1e4)
        target = torch.nan_to_num(target, nan=0.0, posinf=1e4, neginf=-1e4)

        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')

        pred_abs = pred_fft.abs().clamp_min(self.phase_safe_mag)
        target_abs = target_fft.abs().clamp_min(self.phase_safe_mag)

        pred_mag = torch.log1p(pred_abs)
        target_mag = torch.log1p(target_abs)
        amp_diff = torch.nan_to_num((pred_mag - target_mag).abs(), nan=0.0, posinf=10.0, neginf=10.0)

        pred_phase_vec = pred_fft / pred_abs
        target_phase_vec = target_fft / target_abs
        phase_alignment = torch.real(pred_phase_vec * torch.conj(target_phase_vec)).clamp(-1.0, 1.0)
        phase_diff = torch.nan_to_num(1.0 - phase_alignment, nan=0.0, posinf=2.0, neginf=2.0)

        if self.use_high_freq_weight:
            freq_weight = self._get_high_freq_weight(
                pred_mag.shape[-2], pred_mag.shape[-1], pred_mag.device, pred_mag.dtype)
            freq_weight = freq_weight.view(1, 1, pred_mag.shape[-2], pred_mag.shape[-1])
            amp_diff = amp_diff * freq_weight
            phase_diff = phase_diff * freq_weight

        phase_mask = ((target_fft.abs() > self.phase_safe_mag) & (pred_fft.abs() > self.phase_safe_mag)).to(phase_diff.dtype)
        phase_loss = self._reduce(phase_diff * phase_mask)
        amplitude_loss = self._reduce(amp_diff)

        pred_ll, pred_lh, pred_hl, pred_hh = self._haar_dwt(pred)
        target_ll, target_lh, target_hl, target_hh = self._haar_dwt(target)
        wavelet_loss = self._reduce(torch.nan_to_num((pred_ll - target_ll).abs(), nan=0.0, posinf=10.0, neginf=10.0))
        wavelet_loss = wavelet_loss + self.detail_boost * (
            self._reduce(torch.nan_to_num((pred_lh - target_lh).abs(), nan=0.0, posinf=10.0, neginf=10.0)) +
            self._reduce(torch.nan_to_num((pred_hl - target_hl).abs(), nan=0.0, posinf=10.0, neginf=10.0)) +
            self._reduce(torch.nan_to_num((pred_hh - target_hh).abs(), nan=0.0, posinf=10.0, neginf=10.0))
        )

        pred_gx, pred_gy = self._sobel(pred)
        target_gx, target_gy = self._sobel(target)
        edge_loss = self._reduce(torch.nan_to_num((pred_gx - target_gx).abs(), nan=0.0, posinf=10.0, neginf=10.0))
        edge_loss = edge_loss + self._reduce(torch.nan_to_num((pred_gy - target_gy).abs(), nan=0.0, posinf=10.0, neginf=10.0))

        total = (
            self.amplitude_weight * amplitude_loss +
            self.phase_weight * phase_loss +
            self.wavelet_weight * wavelet_loss +
            self.edge_weight * edge_loss
        )
        return self.loss_weight * total


@LOSS_REGISTRY.register()
class WeightedTVLoss(L1Loss):
    """Weighted TV loss.

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        if reduction not in ['mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: mean | sum')
        super(WeightedTVLoss, self).__init__(loss_weight=loss_weight, reduction=reduction)

    def forward(self, pred, weight=None):
        if weight is None:
            y_weight = None
            x_weight = None
        else:
            y_weight = weight[:, :, :-1, :]
            x_weight = weight[:, :, :, :-1]

        y_diff = super().forward(pred[:, :, :-1, :], pred[:, :, 1:, :], weight=y_weight)
        x_diff = super().forward(pred[:, :, :, :-1], pred[:, :, :, 1:], weight=x_weight)

        loss = x_diff + y_diff

        return loss


@LOSS_REGISTRY.register()
class EdgeConsistencyLoss(nn.Module):
    """Sobel edge consistency loss for structure-aware restoration."""

    def __init__(self, loss_weight=0.02, reduction='mean'):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')
        self.loss_weight = loss_weight
        self.reduction = reduction

    def _sobel(self, x):
        kernel_x = x.new_tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        kernel_y = x.new_tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        kernel_x = kernel_x.view(1, 1, 3, 3).repeat(x.shape[1], 1, 1, 1)
        kernel_y = kernel_y.view(1, 1, 3, 3).repeat(x.shape[1], 1, 1, 1)
        grad_x = F.conv2d(x, kernel_x, padding=1, groups=x.shape[1])
        grad_y = F.conv2d(x, kernel_y, padding=1, groups=x.shape[1])
        return grad_x, grad_y

    def _reduce(self, tensor):
        if self.reduction == 'mean':
            return tensor.mean()
        if self.reduction == 'sum':
            return tensor.sum()
        return tensor

    def forward(self, pred, target, weight=None, **kwargs):
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1e4, neginf=-1e4)
        target = torch.nan_to_num(target, nan=0.0, posinf=1e4, neginf=-1e4)
        pred_gx, pred_gy = self._sobel(pred)
        target_gx, target_gy = self._sobel(target)
        loss = (pred_gx - target_gx).abs() + (pred_gy - target_gy).abs()
        return self.loss_weight * self._reduce(loss)


@LOSS_REGISTRY.register()
class BalancedDetailFreqLoss(nn.Module):
    """
    Frequency loss with explicit smooth/detail balance.

    stage1_plus benefits from strong frequency supervision on Urban100-like
    images, but that same pressure can slightly over-sharpen smoother samples.
    This loss keeps the FFT magnitude constraint while adding:
    1. detail-aware Sobel consistency for real edges;
    2. smooth-region low-pass consistency to protect flat areas.
    """

    def __init__(
        self,
        loss_weight=0.03,
        fft_weight=1.0,
        edge_weight=0.3,
        smooth_weight=0.2,
        use_high_freq_weight=True,
        high_freq_gamma=1.5,
        detail_kernel_size=5,
        smooth_kernel_size=5,
        reduction='mean',
    ):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = float(loss_weight)
        self.fft_weight = float(fft_weight)
        self.edge_weight = float(edge_weight)
        self.smooth_weight = float(smooth_weight)
        self.use_high_freq_weight = bool(use_high_freq_weight)
        self.high_freq_gamma = float(high_freq_gamma)
        self.detail_kernel_size = max(int(detail_kernel_size), 3)
        if self.detail_kernel_size % 2 == 0:
            self.detail_kernel_size += 1
        self.smooth_kernel_size = max(int(smooth_kernel_size), 3)
        if self.smooth_kernel_size % 2 == 0:
            self.smooth_kernel_size += 1
        self.reduction = reduction

    def _reduce(self, tensor):
        if self.reduction == 'mean':
            return tensor.mean()
        if self.reduction == 'sum':
            return tensor.sum()
        return tensor

    def _get_high_freq_weight(self, h, w2, device, dtype):
        hh = torch.linspace(-1, 1, h, device=device, dtype=dtype)
        ww = torch.linspace(-1, 1, w2, device=device, dtype=dtype)
        grid_h, grid_w = torch.meshgrid(hh, ww, indexing='ij')
        radius = (grid_h ** 2 + grid_w ** 2).sqrt().clamp(1e-6, 1)
        return radius ** self.high_freq_gamma

    def _detail_prior(self, x):
        smooth = F.avg_pool2d(
            x,
            kernel_size=self.detail_kernel_size,
            stride=1,
            padding=self.detail_kernel_size // 2,
        )
        return (x - smooth).abs()

    def _lowpass(self, x):
        return F.avg_pool2d(
            x,
            kernel_size=self.smooth_kernel_size,
            stride=1,
            padding=self.smooth_kernel_size // 2,
        )

    def _sobel(self, x):
        kernel_x = x.new_tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        kernel_y = x.new_tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        kernel_x = kernel_x.view(1, 1, 3, 3).repeat(x.shape[1], 1, 1, 1)
        kernel_y = kernel_y.view(1, 1, 3, 3).repeat(x.shape[1], 1, 1, 1)
        grad_x = F.conv2d(x, kernel_x, padding=1, groups=x.shape[1])
        grad_y = F.conv2d(x, kernel_y, padding=1, groups=x.shape[1])
        return grad_x, grad_y

    def forward(self, pred, target, weight=None, **kwargs):
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1e4, neginf=-1e4)
        target = torch.nan_to_num(target, nan=0.0, posinf=1e4, neginf=-1e4)

        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')
        fft_diff = (pred_fft.abs() - target_fft.abs()).abs()
        if self.use_high_freq_weight:
            freq_weight = self._get_high_freq_weight(
                fft_diff.shape[-2], fft_diff.shape[-1], fft_diff.device, fft_diff.dtype
            )
            fft_diff = fft_diff * freq_weight.view(1, 1, fft_diff.shape[-2], fft_diff.shape[-1])
        fft_loss = self._reduce(fft_diff)

        detail = self._detail_prior(target)
        detail_norm = detail / (detail.mean(dim=(2, 3), keepdim=True) + 1e-6)
        detail_mask = detail_norm / (detail_norm + 1.0)
        smooth_mask = 1.0 - detail_mask

        pred_gx, pred_gy = self._sobel(pred)
        target_gx, target_gy = self._sobel(target)
        edge_diff = (pred_gx - target_gx).abs() + (pred_gy - target_gy).abs()
        edge_loss = self._reduce(edge_diff * (0.5 + detail_mask))

        smooth_pred = self._lowpass(pred)
        smooth_target = self._lowpass(target)
        smooth_diff = (smooth_pred - smooth_target).abs()
        smooth_loss = self._reduce(smooth_diff * (0.5 + smooth_mask))

        total = self.fft_weight * fft_loss
        total = total + self.edge_weight * edge_loss
        total = total + self.smooth_weight * smooth_loss
        return self.loss_weight * total


@LOSS_REGISTRY.register()
class TeacherSelectiveConsistencyLoss(nn.Module):
    """
    Selective teacher distillation for SR.

    The teacher only supervises regions where it is closer to GT than the
    current student, with extra emphasis on line/tone-rich areas. This is meant
    to preserve a strong student carrier (e.g. Urban100 gains) while borrowing
    teacher strengths on Manga-like structures.
    """

    def __init__(
        self,
        loss_weight=0.03,
        tone_weight=1.0,
        line_weight=0.5,
        reliability_temperature=0.02,
        detail_kernel_size=5,
        variance_kernel_size=5,
        reduction='mean',
    ):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')
        self.loss_weight = float(loss_weight)
        self.tone_weight = float(tone_weight)
        self.line_weight = float(line_weight)
        self.reliability_temperature = max(float(reliability_temperature), 1e-6)
        self.detail_kernel_size = max(int(detail_kernel_size), 3)
        if self.detail_kernel_size % 2 == 0:
            self.detail_kernel_size += 1
        self.variance_kernel_size = max(int(variance_kernel_size), 3)
        if self.variance_kernel_size % 2 == 0:
            self.variance_kernel_size += 1
        self.reduction = reduction

    def _reduce(self, tensor):
        if self.reduction == 'mean':
            return tensor.mean()
        if self.reduction == 'sum':
            return tensor.sum()
        return tensor

    @staticmethod
    def _rgb_to_y(x):
        r = x[:, 0:1, :, :]
        g = x[:, 1:2, :, :]
        b = x[:, 2:3, :, :]
        y = 16.0 + 65.481 * r + 128.553 * g + 24.966 * b
        return y / 255.0

    def _detail_prior(self, x):
        smooth = F.avg_pool2d(
            x,
            kernel_size=self.detail_kernel_size,
            stride=1,
            padding=self.detail_kernel_size // 2,
        )
        return x - smooth

    def _local_variance(self, x):
        mean = F.avg_pool2d(
            x,
            kernel_size=self.variance_kernel_size,
            stride=1,
            padding=self.variance_kernel_size // 2,
        )
        mean_sq = F.avg_pool2d(
            x * x,
            kernel_size=self.variance_kernel_size,
            stride=1,
            padding=self.variance_kernel_size // 2,
        )
        return (mean_sq - mean * mean).clamp_min(0.0)

    def _sobel(self, x):
        kernel_x = x.new_tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        kernel_y = x.new_tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        kernel_x = kernel_x.view(1, 1, 3, 3).repeat(x.shape[1], 1, 1, 1)
        kernel_y = kernel_y.view(1, 1, 3, 3).repeat(x.shape[1], 1, 1, 1)
        grad_x = F.conv2d(x, kernel_x, padding=1, groups=x.shape[1])
        grad_y = F.conv2d(x, kernel_y, padding=1, groups=x.shape[1])
        return grad_x, grad_y

    @staticmethod
    def _normalize_mask(x):
        x = x / (x.mean(dim=(2, 3), keepdim=True) + 1e-6)
        return x / (x + 1.0)

    def forward(self, pred, teacher, weight=None, gt=None, **kwargs):
        if gt is None:
            raise ValueError('TeacherSelectiveConsistencyLoss requires gt=... in forward().')

        pred = torch.nan_to_num(pred, nan=0.0, posinf=1e4, neginf=-1e4)
        teacher = torch.nan_to_num(teacher, nan=0.0, posinf=1e4, neginf=-1e4)
        gt = torch.nan_to_num(gt, nan=0.0, posinf=1e4, neginf=-1e4)

        pred_y = self._rgb_to_y(pred)
        teacher_y = self._rgb_to_y(teacher)
        gt_y = self._rgb_to_y(gt)

        tone_mask = self._normalize_mask(self._detail_prior(gt_y).abs() + self._local_variance(gt_y))
        grad_x, grad_y = self._sobel(gt_y)
        line_mask = self._normalize_mask(grad_x.abs() + grad_y.abs())

        pred_err = (pred_y.detach() - gt_y).abs()
        teacher_err = (teacher_y.detach() - gt_y).abs()
        reliability = torch.sigmoid((pred_err - teacher_err) / self.reliability_temperature)

        guide = (self.tone_weight * tone_mask + self.line_weight * line_mask)
        guide = guide / max(self.tone_weight + self.line_weight, 1e-6)
        guide = reliability * guide

        distill = torch.sqrt((pred - teacher) ** 2 + 1e-12).mean(dim=1, keepdim=True)
        return self.loss_weight * self._reduce(distill * guide)


# @LOSS_REGISTRY.register()
# class PerceptualLoss(nn.Module):
#     """Perceptual loss with commonly used style loss.
#
#     Args:
#         layer_weights (dict): The weight for each layer of vgg feature.
#             Here is an example: {'conv5_4': 1.}, which means the conv5_4
#             feature layer (before relu5_4) will be extracted with weight
#             1.0 in calculating losses.
#         vgg_type (str): The type of vgg network used as feature extractor.
#             Default: 'vgg19'.
#         use_input_norm (bool):  If True, normalize the input image in vgg.
#             Default: True.
#         range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
#             Default: False.
#         perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
#             loss will be calculated and the loss will multiplied by the
#             weight. Default: 1.0.
#         style_weight (float): If `style_weight > 0`, the style loss will be
#             calculated and the loss will multiplied by the weight.
#             Default: 0.
#         criterion (str): Criterion used for perceptual loss. Default: 'l1'.
#     """
#
#     def __init__(self,
#                  layer_weights,
#                  vgg_type='vgg19',
#                  use_input_norm=True,
#                  range_norm=False,
#                  perceptual_weight=1.0,
#                  style_weight=0.,
#                  criterion='l1'):
#         super(PerceptualLoss, self).__init__()
#         self.perceptual_weight = perceptual_weight
#         self.style_weight = style_weight
#         self.layer_weights = layer_weights
#         self.vgg = VGGFeatureExtractor(
#             layer_name_list=list(layer_weights.keys()),
#             vgg_type=vgg_type,
#             use_input_norm=use_input_norm,
#             range_norm=range_norm)
#
#         self.criterion_type = criterion
#         if self.criterion_type == 'l1':
#             self.criterion = torch.nn.L1Loss()
#         elif self.criterion_type == 'l2':
#             self.criterion = torch.nn.L2loss()
#         elif self.criterion_type == 'fro':
#             self.criterion = None
#         else:
#             raise NotImplementedError(f'{criterion} criterion has not been supported.')
#
#     def forward(self, x, gt):
#         """Forward function.
#
#         Args:
#             x (Tensor): Input tensor with shape (n, c, h, w).
#             gt (Tensor): Ground-truth tensor with shape (n, c, h, w).
#
#         Returns:
#             Tensor: Forward results.
#         """
#         # extract vgg features
#         x_features = self.vgg(x)
#         gt_features = self.vgg(gt.detach())
#
#         # calculate perceptual loss
#         if self.perceptual_weight > 0:
#             percep_loss = 0
#             for k in x_features.keys():
#                 if self.criterion_type == 'fro':
#                     percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
#                 else:
#                     percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
#             percep_loss *= self.perceptual_weight
#         else:
#             percep_loss = None
#
#         # calculate style loss
#         if self.style_weight > 0:
#             style_loss = 0
#             for k in x_features.keys():
#                 if self.criterion_type == 'fro':
#                     style_loss += torch.norm(
#                         self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
#                 else:
#                     style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(
#                         gt_features[k])) * self.layer_weights[k]
#             style_loss *= self.style_weight
#         else:
#             style_loss = None
#
#         return percep_loss, style_loss
#
#     def _gram_mat(self, x):
#         """Calculate Gram matrix.
#
#         Args:
#             x (torch.Tensor): Tensor with shape of (n, c, h, w).
#
#         Returns:
#             torch.Tensor: Gram matrix.
#         """
#         n, c, h, w = x.size()
#         features = x.view(n, c, w * h)
#         features_t = features.transpose(1, 2)
#         gram = features.bmm(features_t) / (c * h * w)
#         return gram


@LOSS_REGISTRY.register()
class GANLoss(nn.Module):
    """Define GAN loss.

    Args:
        gan_type (str): Support 'vanilla', 'lsgan', 'wgan', 'hinge'.
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
        loss_weight (float): Loss weight. Default: 1.0.
            Note that loss_weight is only for generators; and it is always 1.0
            for discriminators.
    """

    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0, loss_weight=1.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan':
            self.loss = self._wgan_loss
        elif self.gan_type == 'wgan_softplus':
            self.loss = self._wgan_softplus_loss
        elif self.gan_type == 'hinge':
            self.loss = nn.ReLU()
        else:
            raise NotImplementedError(f'GAN type {self.gan_type} is not implemented.')

    def _wgan_loss(self, input, target):
        """wgan loss.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return -input.mean() if target else input.mean()

    def _wgan_softplus_loss(self, input, target):
        """wgan loss with soft plus. softplus is a smooth approximation to the
        ReLU function.

        In StyleGAN2, it is called:
            Logistic loss for discriminator;
            Non-saturating loss for generator.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return F.softplus(-input).mean() if target else F.softplus(input).mean()

    def get_target_label(self, input, target_is_real):
        """Get target label.

        Args:
            input (Tensor): Input tensor.
            target_is_real (bool): Whether the target is real or fake.

        Returns:
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise,
                return Tensor.
        """

        if self.gan_type in ['wgan', 'wgan_softplus']:
            return target_is_real
        target_val = (self.real_label_val if target_is_real else self.fake_label_val)
        return input.new_ones(input.size()) * target_val

    def forward(self, input, target_is_real, is_disc=False):
        """
        Args:
            input (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the targe is real or fake.
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.

        Returns:
            Tensor: GAN loss value.
        """
        target_label = self.get_target_label(input, target_is_real)
        if self.gan_type == 'hinge':
            if is_disc:  # for discriminators in hinge-gan
                input = -input if target_is_real else input
                loss = self.loss(1 + input).mean()
            else:  # for generators in hinge-gan
                loss = -input.mean()
        else:  # other gan types
            loss = self.loss(input, target_label)

        # loss_weight is always 1.0 for discriminators
        return loss if is_disc else loss * self.loss_weight


@LOSS_REGISTRY.register()
class MultiScaleGANLoss(GANLoss):
    """
    MultiScaleGANLoss accepts a list of predictions
    """

    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0, loss_weight=1.0):
        super(MultiScaleGANLoss, self).__init__(gan_type, real_label_val, fake_label_val, loss_weight)

    def forward(self, input, target_is_real, is_disc=False):
        """
        The input is a list of tensors, or a list of (a list of tensors)
        """
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    # Only compute GAN loss for the last layer
                    # in case of multiscale feature matching
                    pred_i = pred_i[-1]
                # Safe operation: 0-dim tensor calling self.mean() does nothing
                loss_tensor = super().forward(pred_i, target_is_real, is_disc).mean()
                loss += loss_tensor
            return loss / len(input)
        else:
            return super().forward(input, target_is_real, is_disc)


def r1_penalty(real_pred, real_img):
    """R1 regularization for discriminator. The core idea is to
        penalize the gradient on real data alone: when the
        generator distribution produces the true data distribution
        and the discriminator is equal to 0 on the data manifold, the
        gradient penalty ensures that the discriminator cannot create
        a non-zero gradient orthogonal to the data manifold without
        suffering a loss in the GAN game.

        Ref:
        Eq. 9 in Which training methods for GANs do actually converge.
        """
    grad_real = autograd.grad(outputs=real_pred.sum(), inputs=real_img, create_graph=True)[0]
    grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(fake_img.shape[2] * fake_img.shape[3])
    grad = autograd.grad(outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True)[0]
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_lengths.detach().mean(), path_mean.detach()


def gradient_penalty_loss(discriminator, real_data, fake_data, weight=None):
    """Calculate gradient penalty for wgan-gp.

    Args:
        discriminator (nn.Module): Network for the discriminator.
        real_data (Tensor): Real input data.
        fake_data (Tensor): Fake input data.
        weight (Tensor): Weight tensor. Default: None.

    Returns:
        Tensor: A tensor for gradient penalty.
    """

    batch_size = real_data.size(0)
    alpha = real_data.new_tensor(torch.rand(batch_size, 1, 1, 1))

    # interpolate between real_data and fake_data
    interpolates = alpha * real_data + (1. - alpha) * fake_data
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = discriminator(interpolates)
    gradients = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]

    if weight is not None:
        gradients = gradients * weight

    gradients_penalty = ((gradients.norm(2, dim=1) - 1)**2).mean()
    if weight is not None:
        gradients_penalty /= torch.mean(weight)

    return gradients_penalty


@LOSS_REGISTRY.register()
class GANFeatLoss(nn.Module):
    """Define feature matching loss for gans

    Args:
        criterion (str): Support 'l1', 'l2', 'charbonnier'.
        loss_weight (float): Loss weight. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, criterion='l1', loss_weight=1.0, reduction='mean'):
        super(GANFeatLoss, self).__init__()
        if criterion == 'l1':
            self.loss_op = L1Loss(loss_weight, reduction)
        elif criterion == 'l2':
            self.loss_op = MSELoss(loss_weight, reduction)
        elif criterion == 'charbonnier':
            self.loss_op = CharbonnierLoss(loss_weight, reduction)
        else:
            raise ValueError(f'Unsupported loss mode: {criterion}. Supported ones are: l1|l2|charbonnier')

        self.loss_weight = loss_weight

    def forward(self, pred_fake, pred_real):
        num_d = len(pred_fake)
        loss = 0
        for i in range(num_d):  # for each discriminator
            # last output is the final prediction, exclude it
            num_intermediate_outputs = len(pred_fake[i]) - 1
            for j in range(num_intermediate_outputs):  # for each layer output
                unweighted_loss = self.loss_op(pred_fake[i][j], pred_real[i][j].detach())
                loss += unweighted_loss / num_d
        return loss * self.loss_weight
