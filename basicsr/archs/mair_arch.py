# Code Implementation of the MaIR Model
# MaIR: A Locality- and Continuity-Preserving Mamba for Image Restoration

import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint  
import torch.nn.functional as F
from functools import partial
from typing import Optional, Callable
try:
    from timm.layers import DropPath, to_2tuple, trunc_normal_
except ImportError:
    from timm.models.layers import DropPath, to_2tuple, trunc_normal_  
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref  
from einops import rearrange, repeat  
import time
import sys
import os

# 兼容不同运行目录：优先从项目根导入，避免写死 /autodl-tmp/ 等路径
_arch_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_arch_dir, '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
try:
    from basicsr.archs.shift_scanf_util import mair_ids_generate, mair_ids_scan, mair_ids_inverse, mair_shift_ids_generate
    from basicsr.utils.registry import ARCH_REGISTRY
    from basicsr.utils import get_root_logger
except Exception:
    from shift_scanf_util import mair_ids_generate, mair_ids_scan, mair_ids_inverse, mair_shift_ids_generate

# 定义负无穷常量，用于某些计算中
NEG_INF = -1000000


class ShuffleAttn(nn.Module):
    """
    通道混洗注意力模块 (Shuffle Attention)
    通过通道混洗和门控机制实现轻量级的注意力计算
    
    Args:
        in_features: 输入特征通道数
        out_features: 输出特征通道数
        hidden_features: 隐藏层特征数（未使用）
        group: 分组数量，用于通道混洗
        act_layer: 激活函数层
        input_resolution: 输入分辨率 (H, W)
    """
    def __init__(self, in_features, out_features, hidden_features=None, group=4, act_layer=nn.GELU, input_resolution=(64,64)):
        super().__init__()
        self.group = group  # 分组数量
        self.input_resolution = input_resolution  # 输入分辨率
        self.in_features = in_features  # 输入通道数
        self.out_features = out_features  # 输出通道数
        
        # 门控机制：全局平均池化 + 分组卷积 + Sigmoid激活
        self.gating = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化，将空间维度压缩为1x1
            nn.Conv2d(in_features, out_features, groups=self.group, kernel_size=1, stride=1, padding=0),  # 分组1x1卷积
            nn.Sigmoid()  # Sigmoid激活，生成0-1之间的门控权重
        )
    
    def channel_shuffle(self, x):
        """
        通道混洗操作：将通道重新排列以增强不同通道组之间的信息交互
        
        Args:
            x: 输入张量 [B, C, H, W]
            
        Returns:
            混洗后的张量 [B, C, H, W]
        """
        batchsize, num_channels, height, width = x.data.size()
        assert num_channels % self.group == 0, "通道数必须能被分组数整除"
        group_channels = num_channels // self.group  # 每组通道数
        
        # 将通道分为group组，每组group_channels个通道
        x = x.reshape(batchsize, group_channels, self.group, height, width)
        # 交换维度，实现混洗
        x = x.permute(0, 2, 1, 3, 4)
        # 恢复原始形状
        x = x.reshape(batchsize, num_channels, height, width)

        return x
    
    def channel_rearrange(self, x):
        """
        通道重排操作：将混洗后的通道恢复原状
        
        Args:
            x: 输入张量 [B, C, H, W]
            
        Returns:
            重排后的张量 [B, C, H, W]
        """
        batchsize, num_channels, height, width = x.data.size()
        assert num_channels % self.group == 0
        group_channels = num_channels // self.group
        
        # 重新组织通道维度
        x = x.reshape(batchsize, self.group, group_channels, height, width)
        # 交换维度以恢复原始顺序
        x = x.permute(0, 2, 1, 3, 4)
        # 恢复原始形状
        x = x.reshape(batchsize, num_channels, height, width)

        return x

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 [B, C, H, W]
            
        Returns:
            经过混洗注意力处理的特征 [B, C, H, W]
        """
        x = self.channel_shuffle(x)  # 通道混洗
        x = self.gating(x)  # 门控机制生成注意力权重
        x = self.channel_rearrange(x)  # 通道重排恢复

        return x
    
    def flops(self):
        """
        计算该模块的浮点运算次数 (FLOPs)
        
        Returns:
            flops: 浮点运算次数
        """
        flops = 0
        H, W = self.input_resolution
        
        # 全局平均池化的FLOPs：对每个通道的每个像素进行累加
        flops += H * W * self.in_features

        # 分组1x1卷积的FLOPs：分组卷积的计算量是普通卷积的1/group
        flops += H * W * self.in_features * self.out_features // self.group

        # Sigmoid激活的FLOPs：每个元素需要exp和除法运算，约4次浮点运算
        flops += H * W * self.out_features * 4
        return flops


class SequenceExemplarGate(nn.Module):

    def __init__(self, channels, num_sequences=4, hidden_ratio=0.25, pool_size=8, scale_init=0.02):
        super().__init__()
        if channels % num_sequences != 0:
            raise ValueError(f'SequenceExemplarGate requires channels divisible by num_sequences, got {channels}.')
        self.num_sequences = int(num_sequences)
        self.seq_channels = channels // self.num_sequences
        self.pool_size = max(int(pool_size), 4)
        hidden = max(int(self.seq_channels * hidden_ratio), 8)
        self.attn_scale = hidden ** -0.5

        self.query_proj = nn.Sequential(
            nn.Linear(self.seq_channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.kv_proj = nn.Conv2d(self.seq_channels, hidden * 2, 1)
        self.out_proj = nn.Linear(hidden, self.seq_channels)
        self.res_scale = nn.Parameter(torch.tensor(float(scale_init)))

        nn.init.normal_(self.query_proj[0].weight, std=1e-3)
        nn.init.constant_(self.query_proj[0].bias, 0.0)
        nn.init.normal_(self.query_proj[2].weight, std=1e-3)
        nn.init.constant_(self.query_proj[2].bias, 0.0)
        nn.init.normal_(self.kv_proj.weight, std=1e-3)
        if self.kv_proj.bias is not None:
            nn.init.constant_(self.kv_proj.bias, 0.0)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x):
        b, c, h, w = x.shape
        # 先把通道维中拼接的 4 个扫描方向显式拆出来，
        # 这样就能在原始 MaIR 的“四路求和”聚合前分别调节每一路响应。
        seq = x.view(b, self.num_sequences, self.seq_channels, h, w)
        seq_desc = seq.mean(dim=(-2, -1))

        # 先对 4 个方向取平均，再做低分辨率池化，构造一份共享的上下文记忆。
        # 每个扫描方向都基于这份共享上下文决定自己应该保留或抑制多少信息。
        pooled = F.adaptive_avg_pool2d(seq.mean(dim=1), self.pool_size)
        k, v = self.kv_proj(pooled).chunk(2, dim=1)
        k = k.flatten(2)
        v = v.flatten(2).transpose(1, 2)

        q = self.query_proj(seq_desc.reshape(b * self.num_sequences, self.seq_channels))
        q = q.view(b, self.num_sequences, -1)
        attn = torch.softmax(torch.bmm(q, k) * self.attn_scale, dim=-1)
        ctx = torch.bmm(attn, v)
        gate = self.out_proj(ctx.reshape(b * self.num_sequences, -1))
        gate = gate.view(b, self.num_sequences, self.seq_channels, 1, 1)

        # 残差式门控保证初始状态接近恒等映射：gate 初始接近 1，
        # 只有当共享上下文明确提示某个方向更重要或应被抑制时，才学习小幅修正。
        gate = 1.0 + torch.tanh(self.res_scale) * torch.tanh(gate)
        return (seq * gate).view(b, c, h, w)

    def flops(self, H, W):
        hidden = self.query_proj[0].out_features
        pooled_tokens = self.pool_size * self.pool_size
        seq = self.num_sequences
        c = self.seq_channels
        flops = seq * c * hidden * 2
        flops += pooled_tokens * c * hidden * 2
        flops += seq * pooled_tokens * hidden
        flops += seq * hidden * c
        return flops

class GroupedFusion(nn.Module):
    """
    基于分组卷积和通道混洗的轻量级融合层
    用于将空域和频域特征(embed_dim * 2)高效降维融合回 embed_dim。
    卷积小随机初始化(std=1e-3)，保留空域残差，便于融合层有梯度可学。
    """
    def __init__(self, in_channels, out_channels, groups=4):
        super().__init__()
        self.groups = groups
        assert in_channels % groups == 0 and out_channels % groups == 0, "通道数必须能被 groups 整除"

        # 分组 1x1 卷积
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1, stride=1, padding=0,
            groups=groups
        )
        # 小随机初始化：便于融合层有梯度可学，避免全 0 导致学不动
        nn.init.normal_(self.conv.weight, std=1e-3)
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0.0)

    def channel_shuffle(self, x):
        """通道混洗，打乱分组卷积后的通道，促进信息交互"""
        batchsize, num_channels, height, width = x.shape
        channels_per_group = num_channels // self.groups

        # 维度变换: [B, groups, C//groups, H, W]
        x = x.view(batchsize, self.groups, channels_per_group, height, width)
        # 交换组和通道内索引: [B, C//groups, groups, H, W]
        x = torch.transpose(x, 1, 2).contiguous()
        # 展平回 [B, C, H, W]
        x = x.view(batchsize, -1, height, width)
        return x

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        """
        x_cat: [B, 2C, H, W] = [spatial_feat, freq_feat] 在通道维拼接后的特征。
        假设前 C 个通道是空域主干特征，作为残差保留。
        """
        out_channels = self.conv.out_channels
        # 提取空间特征作为残差（对应 backbone 的输出通道）
        spatial_residual = x_cat[:, :out_channels, :, :]

        x = self.conv(x_cat)
        x = self.channel_shuffle(x)
        # 融合结果 + 空域残差
        return x + spatial_residual

    def flops(self, H, W):
        """计算融合层的 FLOPs（分组卷积 + 残差相加）"""
        in_c = self.conv.in_channels
        out_c = self.conv.out_channels
        conv_flops = H * W * in_c * out_c // self.groups
        add_flops = H * W * out_c  # 残差相加
        return conv_flops + add_flops


class GatedFusion(nn.Module):
    """
    门控融合：用空域特征生成门控，加权频域分支，避免频域在平坦区域引入噪声导致 PSNR 下降。
    out = gate * freq_feat + (1 - gate) * spatial_feat，gate 由 spatial 分支学习。
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.gate = nn.Sequential(
            nn.Conv2d(embed_dim, max(embed_dim // 4, 1), 1),
            nn.GELU(),
            nn.Conv2d(max(embed_dim // 4, 1), 1, 1),
            nn.Sigmoid(),
        )
        if self.gate[2].bias is not None:
            nn.init.constant_(self.gate[2].bias, 0.0)

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        """
        x_cat: [B, 2C, H, W] = [spatial_feat, freq_feat] 通道维拼接，前 C 通道为空域。
        """
        C = x_cat.shape[1] // 2
        spatial = x_cat[:, :C]
        freq = x_cat[:, C:]
        gate = self.gate(spatial)
        return spatial + gate * (freq - spatial)

    def flops(self, H, W):
        C = self.embed_dim
        c4 = max(C // 4, 1)
        return H * W * (C * c4 + c4 * 1)


class CrossGatedFusion(nn.Module):
    """
    双域交互融合：spatial->freq 门控 + freq->spatial 门控，再融合。
    """
    def __init__(self, embed_dim):
        super().__init__()
        c4 = max(embed_dim // 4, 1)
        self.s2f = nn.Sequential(
            nn.Conv2d(embed_dim, c4, 1),
            nn.GELU(),
            nn.Conv2d(c4, embed_dim, 1),
            nn.Sigmoid(),
        )
        self.f2s = nn.Sequential(
            nn.Conv2d(embed_dim, c4, 1),
            nn.GELU(),
            nn.Conv2d(c4, embed_dim, 1),
            nn.Sigmoid(),
        )
        self.proj = nn.Conv2d(embed_dim * 2, embed_dim, 1)

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        C = x_cat.shape[1] // 2
        spatial = x_cat[:, :C]
        freq = x_cat[:, C:]

        gate_f = self.s2f(spatial)
        gate_s = self.f2s(freq)

        freq = freq * gate_f
        spatial = spatial * gate_s

        out = self.proj(torch.cat([spatial, freq], dim=1))
        return out + spatial

    def flops(self, H, W):
        C = self.proj.out_channels
        c4 = max(C // 4, 1)
        # s2f + f2s + proj
        return H * W * (C * c4 + c4 * C) * 2 + H * W * (C * 2) * C


class FFTBlock(nn.Module):
    """
    频域处理：幅度 + 相位双分支
    """
    def __init__(self, embed_dim, fft_ratio=1, high_freq_emphasis=False, alpha_init=0.1,
                 phase_scale_init=1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.high_freq_emphasis = high_freq_emphasis
        hidden = max(embed_dim // 2, 8)

        # 幅度分支
        self.mag_proj = nn.Sequential(
            nn.Conv2d(embed_dim, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.GELU(),
            nn.Conv2d(hidden, embed_dim, 1),
        )

        # 相位分支
        self.phase_proj = nn.Sequential(
            nn.Conv2d(embed_dim, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.GELU(),
            nn.Conv2d(hidden, embed_dim, 1),
        )

        # 融合回 2C（real/imag）
        self.fuse = nn.Conv2d(embed_dim * 2, embed_dim * 2, 1)

        self.norm = nn.LayerNorm(embed_dim)
        self.alpha_logit = nn.Parameter(torch.tensor(math.log(alpha_init / (1 - alpha_init + 1e-8))))
        self.phase_scale = nn.Parameter(torch.tensor(phase_scale_init))

    def _get_high_freq_weight(self, H, W2, device):
        h = torch.linspace(-1, 1, H, device=device)
        w = torch.linspace(-1, 1, W2, device=device)
        grid_h, grid_w = torch.meshgrid(h, w, indexing='ij')
        r = (grid_h ** 2 + grid_w ** 2).sqrt().clamp(1e-6, 1)
        return (r ** 1.5).unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        B, C, H, W = x.shape
        fft = torch.fft.rfft2(x, norm='ortho')
        real, imag = fft.real, fft.imag

        if self.high_freq_emphasis:
            w = self._get_high_freq_weight(H, W // 2 + 1, x.device)
            real, imag = real * w, imag * w

        # 幅度 + 相位
        mag = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
        phase = torch.atan2(imag, real)

        mag_feat = self.mag_proj(mag)
        phase_feat = self.phase_proj(phase) * self.phase_scale

        freq_feat = self.fuse(torch.cat([mag_feat, phase_feat], dim=1))
        r, i = freq_feat.chunk(2, dim=1)
        fft_out = torch.complex(r, i)

        out = torch.fft.irfft2(fft_out, s=(H, W), norm='ortho')
        out = self.norm(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        alpha = torch.sigmoid(self.alpha_logit)
        return x + alpha * out

    def flops(self, H, W):
        flops = 0
        C = self.embed_dim
        hidden = max(C // 2, 8)

        N = H * W
        if N > 0:
            flops += C * (5 * N * math.log2(N))  # FFT + iFFT

        freq_W = W // 2 + 1

        # mag 分支
        flops += H * freq_W * C * hidden
        flops += H * freq_W * hidden
        flops += H * freq_W * hidden * 9
        flops += H * freq_W * hidden
        flops += H * freq_W * hidden * C

        # phase 分支
        flops += H * freq_W * C * hidden
        flops += H * freq_W * hidden
        flops += H * freq_W * hidden * 9
        flops += H * freq_W * hidden
        flops += H * freq_W * hidden * C

        # fuse 1x1
        flops += H * freq_W * (C * 2) * (C * 2)

        # layernorm
        flops += H * W * C
        return flops


    
class Mlp(nn.Module):
    """
    多层感知机 (MLP) 模块
    标准的全连接前馈网络，用于特征变换
    
    Args:
        in_features: 输入特征维度
        hidden_features: 隐藏层特征维度，默认为in_features
        out_features: 输出特征维度，默认为in_features
        act_layer: 激活函数层，默认为GELU
        drop: Dropout比率
        input_resolution: 输入分辨率 (H, W)
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., input_resolution=(64,64)):
        super().__init__()
        out_features = out_features or in_features  # 如果未指定，使用输入维度
        hidden_features = hidden_features or in_features  # 如果未指定，使用输入维度
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.input_resolution = input_resolution
        
        # 两层全连接网络
        self.fc1 = nn.Linear(in_features, hidden_features)  # 第一层：扩展维度
        self.act = act_layer()  # 激活函数
        self.fc2 = nn.Linear(hidden_features, out_features)  # 第二层：恢复维度

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 [B, L, C] 或 [B, H, W, C]
            
        Returns:
            变换后的特征 [B, L, C] 或 [B, H, W, C]
        """
        x = self.fc1(x)  # 第一层全连接
        x = self.act(x)  # 激活
        x = self.fc2(x)  # 第二层全连接
        return x

    def flops(self):
        """
        计算该模块的浮点运算次数 (FLOPs)
        
        Returns:
            flops: 浮点运算次数
        """
        flops = 0
        H, W = self.input_resolution

        # 两层全连接的FLOPs：每层包括乘法和加法
        flops += 2 * H * W * self.in_features * self.hidden_features
        # 激活函数的FLOPs（简化计算）
        flops += H * W * self.hidden_features

        return flops


class VMM(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            input_resolution=(64, 64),
            use_seq_exemplar_gate=False,
            seq_exemplar_hidden_ratio=0.25,
            seq_exemplar_pool_size=8,
            seq_exemplar_scale_init=0.02,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.input_resolution = input_resolution

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

        self.gating = ShuffleAttn(in_features=self.d_inner*4, out_features=self.d_inner*4, group=self.d_inner)
        # 可选的第二阶段序列路由：保留原始 ShuffleAttn，
        # 再利用跨方向池化上下文进一步细调 4 个扫描方向，
        # 最后再按原始 MaIR 的方式折叠回单一路径特征。
        self.sequence_gate = (
            SequenceExemplarGate(
                self.d_inner * 4,
                num_sequences=4,
                hidden_ratio=seq_exemplar_hidden_ratio,
                pool_size=seq_exemplar_pool_size,
                scale_init=seq_exemplar_scale_init,
            )
            if use_seq_exemplar_gate else None
        )

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor, 
                     mair_ids,
                     x_proj_bias: torch.Tensor=None,
                     ):
        # print(x.shape) C=360
        B, C, H, W = x.shape
        L = H * W
        D, N = self.A_logs.shape
        K, D, R = self.dt_projs_weight.shape
        K=4
        # print("hello")
        xs = mair_ids_scan(x, mair_ids[0])

        x_dbl = F.conv1d(xs.reshape(B, -1, L), self.x_proj_weight.reshape(-1, D, 1), bias=(x_proj_bias.reshape(-1) if x_proj_bias is not None else None), groups=K)
        dts, Bs, Cs = torch.split(x_dbl.reshape(B, K, -1, L), [R, N, N], dim=2)
        dts = F.conv1d(dts.reshape(B, -1, L), self.dt_projs_weight.reshape(K * D, -1, 1), groups=K)
        
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        out_y = self.selective_scan(
            xs, dts,
            -torch.exp(self.A_logs.float()).view(-1, self.d_state), Bs, Cs, self.Ds.float().view(-1), z=None,
            delta_bias=self.dt_projs_bias.float().view(-1),
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        return mair_ids_inverse(out_y, mair_ids[1], shape=(B, -1, H, W)) #B, C, L

    def forward(self, x: torch.Tensor, mair_ids, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y = self.forward_core(x, mair_ids)
        assert y.dtype == torch.float32
        y = y * self.gating(y)
        if self.sequence_gate is not None:
            y = self.sequence_gate(y)
        # MaIR 最终仍然按原始设计对 4 个扫描方向直接求和，
        # 新增 gate 只负责在这一步之前重新分配各方向的贡献强度。
        y1, y2, y3, y4 = torch.chunk(y, 4, dim=1)
        y = y1 + y2 + y3 + y4
        y = y.permute(0, 2, 3, 1).contiguous()
        
        y = self.out_norm(y)
        y = y * F.silu(z)
        y = self.out_proj(y)
        if self.dropout is not None:
            y = self.dropout()
        return y

    def flops_forward_core(self, H, W):
        flops = 0
        # flops of x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight) in Core
        flops += 4 * (H * W) * self.d_inner * (self.dt_rank + self.d_state * 2)
        # flops of dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dt_rank=12, d_inner=360
        flops += 4 * (H * W) * self.dt_rank * self.d_inner
        # print(flops/1e6, (4 * H * W) * (self.d_state * self.d_state * 2)/1e6)
        # 610.46784 M 8.388608 M

        # Flops of discretization
        flops += (4 * H * W) * (self.d_state * self.d_state * 2)

        # Flops of Vmamba selective_scan
        # # h' = Ah(t) + Bx(t)
        # flops += (4 * H * W) * (self.d_state * self.d_state + self.d_inner * self.d_state)
        # # y = Ch(t) + DBx(t)
        # flops += (4 * H * W) * (self.d_inner * self.d_inner + self.d_inner * self.d_state)
        # 640*360*36*90*16/1e9=11.94G 
        flops += 4 * 9 * H * W * self.d_inner * self.d_state
        # print(4 * 9 * H * W * self.d_inner * self.d_state/1e9)
        return flops
    
    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # flop of in_proj
        flops += H * W * self.d_model * self.d_inner * 2
        # flops of x = self.act(self.conv2d(x))
        flops += H * W * self.d_inner * 3 * 3 + H * W * self.d_inner
        # print(H, W, self.d_state, self.d_inner)
        flops += self.flops_forward_core(H, W)
        # 64 64 16 360
        flops += self.gating.flops()
        if self.sequence_gate is not None:
            flops += self.sequence_gate.flops(H, W)
        # y = y1 + y2 + y3 + y4
        flops += 4 * H * W * self.d_inner
        # flops of y = self.out_norm(y)
        flops += H * W * self.d_inner
        # flops of y = y * F.silu(z)
        flops += 2 * H * W * self.d_inner

        # flops of out = self.out_proj(y)
        flops += H * W * self.d_inner * self.d_model

        return flops


class RMB(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            ssm_ratio: float = 2.,
            input_resolution= (64, 64),
            is_light_sr: bool = False,
            shift_size=0,
            mlp_ratio=1.5,
            use_seq_exemplar_gate=False,
            seq_exemplar_hidden_ratio=0.25,
            seq_exemplar_pool_size=8,
            seq_exemplar_scale_init=0.02,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = VMM(
            d_model=hidden_dim,
            d_state=d_state,
            expand=ssm_ratio,
            dropout=attn_drop_rate,
            input_resolution=input_resolution,
            use_seq_exemplar_gate=use_seq_exemplar_gate,
            seq_exemplar_hidden_ratio=seq_exemplar_hidden_ratio,
            seq_exemplar_pool_size=seq_exemplar_pool_size,
            seq_exemplar_scale_init=seq_exemplar_scale_init,
            **kwargs
        )
        self.drop_path = DropPath(drop_path)
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.conv_blk = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim,input_resolution=input_resolution)
        
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))
        self.hidden_dim = hidden_dim
        self.input_resolution = input_resolution

        self.shift_size = shift_size

    def forward(self, input, mair_ids, x_size):
        # x [B,HW,C]
        B, L, C = input.shape
        input = input.view(B, *x_size, C).contiguous()  # [B,H,W,C]

        x = self.ln_1(input)
        if self.shift_size > 0:
            x = input*self.skip_scale + self.drop_path(self.self_attention(x, (mair_ids[2], mair_ids[3])))
        else:
            x = input*self.skip_scale + self.drop_path(self.self_attention(x, (mair_ids[0], mair_ids[1])))
        
        x = x*self.skip_scale2 + self.conv_blk(self.ln_2(x))

        x = x.reshape(B, -1, C)
        return x
    
    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # flops of norm1 self.ln_1 -> layer_norm1
        flops += self.hidden_dim * H * W
        # flops of SS2D
        flops += self.self_attention.flops()
        # flops of input * self.skip_scale and residual
        flops += self.hidden_dim * H * W * 2 
        # flops of norm2 self.ln_2 -> layer_norm2
        flops += self.hidden_dim * H * W 
        # flops of MLP
        flops += self.conv_blk.flops()
        # flops of input * self.skip_scale2 and residual
        flops += self.hidden_dim * H * W * 2 
        
        return flops
    


class BasicLayer(nn.Module):
    """ The Basic MaIR Layer in one Residual Mamba Group
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 drop_path=0.,
                 d_state=16,
                 ssm_ratio=2.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 is_light_sr=False,
                 scan_len=4,
                 mlp_ratio=2,
                 use_seq_exemplar_gate=False,
                 seq_exemplar_hidden_ratio=0.25,
                 seq_exemplar_pool_size=8,
                 seq_exemplar_scale_init=0.02
                 ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.ssm_ratio=ssm_ratio
        self.mlp_ratio=mlp_ratio
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(RMB(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                d_state=d_state,
                ssm_ratio=self.ssm_ratio,
                input_resolution=input_resolution,
                is_light_sr=is_light_sr,
                shift_size=0 if (i % 2 == 0) else scan_len // 2,
                mlp_ratio=self.mlp_ratio,
                use_seq_exemplar_gate=use_seq_exemplar_gate,
                seq_exemplar_hidden_ratio=seq_exemplar_hidden_ratio,
                seq_exemplar_pool_size=seq_exemplar_pool_size,
                seq_exemplar_scale_init=seq_exemplar_scale_init)
                )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, mair_ids, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, mair_ids, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


@ARCH_REGISTRY.register()
class MaIR(nn.Module):
    r""" Mamba-based Image Restoration Network (MaIR)
           A PyTorch implementation of : `MaIR: A Locality- and Continuity-Preserving Mamba for Image Restoration`.
           
       Args:
           img_size (int | tuple(int)): Input image size. Default 64
           patch_size (int | tuple(int)): Patch size. Default: 1
           in_chans (int): Number of input image channels. Default: 3
           embed_dim (int): Patch embedding dimension. Default: 96
           d_state (int): num of hidden state in the state space model. Default: 16
           ssm_ratio (int): enlarge ratio in MaIR Module
           mlp_ratio (int): enlarge ratio in the hidden space of MLP
           depths (tuple(int)): Depth of each RSSG
           drop_rate (float): Dropout rate. Default: 0
           drop_path_rate (float): Stochastic depth rate. Default: 0.1
           norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
           patch_norm (bool): If True, add normalization after patch embedding. Default: True
           use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
           upscale: Upscale factor. 2/3/4 for image SR, 1 for denoising
           img_range: Image range. 1. or 255.
           upsampler: The reconstruction reconstruction module. 'pixelshuffle'/None
           resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
           scan_len: Stripe width of the NSS
       """
    def __init__(self,
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=60,
                 depths=(6, 6, 6, 6),
                 drop_rate=0.,
                 d_state=16,
                 ssm_ratio=1.5,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=2,
                 img_range=1.,
                 upsampler='pixelshuffledirect',
                 resi_connection='1conv',
                 dynamic_ids=False,
                 scan_len=8,
                 mlp_ratio=2,
                 use_seq_exemplar_gate=False,
                 seq_exemplar_hidden_ratio=0.25,
                 seq_exemplar_pool_size=8,
                 seq_exemplar_scale_init=0.02,
                 **kwargs):

        super(MaIR, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.ssm_ratio=ssm_ratio
        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.num_out_ch = num_out_ch

        self.dynamic_ids = dynamic_ids
        self.scan_len = scan_len
        img_size_ids = to_2tuple(img_size)
        self.image_size = img_size_ids

        if not self.dynamic_ids:
            self._generate_ids((1, 1, img_size_ids[0], img_size_ids[1]))

        # transfer 2D feature map into 1D token sequence, pay attention to whether using normalization
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # return 2D feature map from 1D token sequence
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.is_light_sr = True if self.upsampler=='pixelshuffledirect' else False
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Residual State Space Group (RSSG)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers): # 6-layer
            layer = RMG(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                d_state = d_state,
                ssm_ratio=self.ssm_ratio,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
                is_light_sr = self.is_light_sr,
                scan_len=scan_len,
                mlp_ratio=mlp_ratio,
                use_seq_exemplar_gate=use_seq_exemplar_gate,
                seq_exemplar_hidden_ratio=seq_exemplar_hidden_ratio,
                seq_exemplar_pool_size=seq_exemplar_pool_size,
                seq_exemplar_scale_init=seq_exemplar_scale_init
            )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in the end of all residual groups
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        # -------------------------3. high-quality image reconstruction ------------------------ #
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch)

        else:
            # for image denoising
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}
    
    def _generate_ids(self, inp_shape):
        B,C,H,W = inp_shape

        xs_scan_ids, xs_inverse_ids = mair_ids_generate(inp_shape=(1, 1, H, W), scan_len=self.scan_len)# [B,H,W,C]
        if torch.cuda.is_available():
            self.xs_scan_ids = xs_scan_ids.cuda()
            self.xs_inverse_ids = xs_inverse_ids.cuda()
        else:
            self.xs_scan_ids = xs_scan_ids
            self.xs_inverse_ids = xs_inverse_ids

        xs_shift_scan_ids, xs_shift_inverse_ids = mair_shift_ids_generate(inp_shape=(1, 1, H, W), scan_len=self.scan_len, shift_len=self.scan_len//2)# [B,H,W,C]
        if torch.cuda.is_available():
            self.xs_shift_scan_ids = xs_shift_scan_ids.cuda()
            self.xs_shift_inverse_ids = xs_shift_inverse_ids.cuda()
        else:
            self.xs_shift_scan_ids = xs_shift_scan_ids
            self.xs_shift_inverse_ids = xs_shift_inverse_ids

        del xs_scan_ids, xs_inverse_ids, xs_shift_scan_ids, xs_shift_inverse_ids

    def forward_features(self, x):
        B,C,H,W = x.shape
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x) # N,L,C
        x = self.pos_drop(x)

        if self.dynamic_ids or (self.image_size != (H, W)):
            xs_scan_ids, xs_inverse_ids = mair_ids_generate(inp_shape=(1, 1, H, W), scan_len=self.scan_len)# [B,H,W,C]
            xs_shift_scan_ids, xs_shift_inverse_ids = mair_shift_ids_generate(inp_shape=(1, 1, H, W), scan_len=self.scan_len, shift_len=self.scan_len//2)# [B,H,W,C]
            if torch.cuda.is_available():
                xs_scan_ids, xs_inverse_ids = xs_scan_ids.cuda(), xs_inverse_ids.cuda()
                xs_shift_scan_ids, xs_shift_inverse_ids = xs_shift_scan_ids.cuda(), xs_shift_inverse_ids.cuda()
            for layer in self.layers:
                x = layer(x, (xs_scan_ids, xs_inverse_ids, xs_shift_scan_ids, xs_shift_inverse_ids), x_size)
        else:
            for layer in self.layers:
                x = layer(x, (self.xs_scan_ids, self.xs_inverse_ids, self.xs_shift_scan_ids, self.xs_shift_inverse_ids), x_size)
        
        x = self.norm(x)  # b seq_len c
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffle':
            # for classical SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))

        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)

        else:
            # for image denoising
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first)) + x_first
            x = x + self.conv_last(res)

        x = x / self.img_range + self.mean

        return x

    def flops_layers(self):
        flops = 0
        h, w = self.patches_resolution

        # flops of forward_features
        flops += self.patch_embed.flops()
        print("self.patches_resolution:", self.patches_resolution)

        for layer in self.layers:
            flops += layer.flops()

        # flops of self.norm
        flops += h * w * self.embed_dim 

        # flops of self.patch_unembed
        flops += h * w * 9 * self.embed_dim * self.embed_dim

        # flops of self.conv_after_body
        flops += h * w * 9 * self.embed_dim * self.embed_dim

        # flops of Residual
        flops += h * w * self.embed_dim

        return flops

    def flops(self):
        flops = 0
        h, w = self.patches_resolution
        # x = self.conv_first(x)
        flops += h * w * 3 * self.embed_dim * 9

        if self.upsampler == 'pixelshuffle':
            # for classical SR

            # x = self.conv_after_body(self.forward_features(x)) + x
            flops += self.flops_layers()

            # x = self.conv_before_upsample(x)
            # nn.Conv2d(embed_dim, num_feat (=64), 3, 1, 1), nn.LeakyReLU(inplace=True))
            flops += h * w * 9 * self.embed_dim * 64
            flops += h * w * 64

            # self.upsample(x)
            if self.upscale == 2:
                flops += h * w * 9 * 64 * 4*64
            elif self.upscale == 3:
                flops += h * w * 9 * 64 * 9*64
            # x = self.conv_last()
            flops += h * w * 9 * 64 * 3

        elif self.upsampler == 'pixelshuffledirect':
            # x = self.conv_after_body(self.forward_features(x)) + x
            flops += self.flops_layers()

            # flops of UpsampleOneStep
            # self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch)
            flops += h * w * 9 * self.embed_dim * (self.upscale**2) * self.num_out_ch

        return flops


class MaIR_FFT(nn.Module):
    """
    在 MaIR 预训练基础上增加 FFT 频域分支，实现空域+频域双信息融合。
    可加载 MaIR 预训练权重并冻结主干参数，仅训练新增模块。
    可选 use_gated_fusion：门控融合，让网络学习何时信任频域分支，减轻对 PSNR 的拖累。
    """
    def __init__(self, use_fft=True, freeze_backbone_after_load=True, bypass_fft_fusion=False,
                 use_gated_fusion=False, use_cross_gated_fusion=False, high_freq_emphasis=False, alpha_init=0.1,
                 fft_hidden_ratio=0.5, fft_depth=1, fft_band_gating=False,
                 fft_band_low=0.33, fft_band_high=0.66, **mair_kwargs):
        super().__init__()
        # 初始化原始 MaIR 模型（主干网络）
        self.backbone = MaIR(**mair_kwargs)
        self.use_fft = use_fft
        self.freeze_backbone_after_load = freeze_backbone_after_load
        # 为 True 时前向只走 backbone 空域，不走 FFT 与融合，用于对比「纯预训练 backbone」的 PSNR
        self.bypass_fft_fusion = bypass_fft_fusion

        # 获取主干网络的嵌入维度
        embed_dim = self.backbone.embed_dim

        # 频域分支：幅度+相位双分支 FFTBlock
        self.fft_branch = FFTBlock(
            embed_dim,
            high_freq_emphasis=high_freq_emphasis,
            alpha_init=alpha_init,
            # phase_scale_init 可在需要时暴露为配置
        ) if use_fft else None

        # 空域+频域融合模块：可选门控融合，减少频域分支在不利区域的干扰
        if use_fft:
            if use_cross_gated_fusion:
                self.fusion = CrossGatedFusion(embed_dim)
            elif use_gated_fusion:
                self.fusion = GatedFusion(embed_dim)
            else:
                self.fusion = GroupedFusion(in_channels=embed_dim * 2, out_channels=embed_dim, groups=4)
        else:
            self.fusion = None

        # 标记主干是否已被冻结
        self._backbone_frozen = False

    def load_pretrained_mair(self, path, strict=False, param_key='params'):
        """
        加载权重：支持两种 checkpoint 格式
        1) 纯 backbone（MaIR）：只含 conv_first.weight 等，无 backbone. 前缀 → 只加载 backbone，FFT/fusion 随机初始化
        2) 完整 MaIR_FFT（续训）：含 backbone.xxx, fft_branch.xxx, fusion.xxx → 全部加载，用于 resume 不断点
        """
        import os
        if not os.path.exists(path):
            raise FileNotFoundError(f"Pretrained MaIR path not found: {path}")

        load_net = torch.load(path, map_location='cpu')
        if param_key and param_key in load_net:
            load_net = load_net[param_key]
        load_net = {k.replace('module.', ''): v for k, v in load_net.items()}

        # 判断是否为完整 MaIR_FFT 检查点（续训时 check_resume 会把 pretrain_network_g 设为 net_g_8000.pth）
        has_backbone_prefix = any(k.startswith('backbone.') for k in load_net)

        try:
            logger = get_root_logger()
        except Exception:
            logger = None

        if has_backbone_prefix:
            # 完整 MaIR_FFT checkpoint：按前缀分别加载 backbone / fft_branch / fusion
            backbone_state = {k.replace('backbone.', ''): v for k, v in load_net.items() if k.startswith('backbone.')}
            self.backbone.load_state_dict(backbone_state, strict=strict)
            n_backbone = len(backbone_state)
            n_fft, n_fusion = 0, 0
            if self.fft_branch is not None:
                fft_state = {k: v for k, v in load_net.items() if k.startswith('fft_branch.')}
                if fft_state:
                    self.fft_branch.load_state_dict({k.replace('fft_branch.', ''): v for k, v in fft_state.items()}, strict=strict)
                    n_fft = len(fft_state)
            if self.fusion is not None:
                fusion_state = {k: v for k, v in load_net.items() if k.startswith('fusion.')}
                if fusion_state:
                    self.fusion.load_state_dict({k.replace('fusion.', ''): v for k, v in fusion_state.items()}, strict=strict)
                    n_fusion = len(fusion_state)
            if logger is not None:
                logger.info(
                    f'[MaIR_FFT] Loaded full checkpoint (resume) from {path}: backbone={n_backbone}, fft_branch={n_fft}, fusion={n_fusion}, '
                    f'freeze_backbone={self.freeze_backbone_after_load}.'
                )
        else:
            # 纯 backbone 预训练（如 MaIR_T_lightSR_x2.pth）
            backbone_keys = set(self.backbone.state_dict().keys())
            backbone_state = {k: v for k, v in load_net.items() if k in backbone_keys}
            self.backbone.load_state_dict(backbone_state, strict=strict)
            if logger is not None:
                n_matched, n_total = len(backbone_state), len(backbone_keys)
                logger.info(
                    f'[MaIR_FFT] Loaded backbone from {path}: {n_matched}/{n_total} keys matched, '
                    f'freeze_backbone={self.freeze_backbone_after_load}. '
                    f'If {n_matched} << {n_total}, pretrain may not match this backbone.'
                )

        if self.freeze_backbone_after_load:
            self.freeze_backbone()

        return backbone_state

    def freeze_backbone(self, exclude=None):
        """
        冻结 backbone 全部参数，仅保留 FFT 与 fusion 可训练。
        Args:
            exclude: 不冻结的参数名称列表（例如需要微调的部分层）
        """
        exclude = set(exclude or [])
        for name, p in self.backbone.named_parameters():
            if name in exclude:
                continue
            p.requires_grad = False
        self._backbone_frozen = True

    def unfreeze_backbone(self):
        """解冻 backbone（例如在微调阶段需要更新 backbone）。"""
        for p in self.backbone.parameters():
            p.requires_grad = True
        self._backbone_frozen = False

    def flops(self):
        """
        计算整个网络（包括 FFT 分支和融合模块）的浮点运算次数（FLOPs）。
        """
        flops = 0
        H, W = self.backbone.patches_resolution  # 当前特征图的分辨率

        # 主干网络的 FLOPs
        flops += self.backbone.flops()
        if self.use_fft and self.fft_branch is not None:
            flops += self.fft_branch.flops(H, W)     
            if self.fusion is not None:
                flops += self.fusion.flops(H, W)      # GroupedFusion 
        return flops

    def forward(self, x):
        """
        前向传播：输入带噪声的图像，输出恢复后的图像。
        """
        # 将图像减去均值并乘以缩放因子（归一化操作）
        self.backbone.mean = self.backbone.mean.type_as(x)
        x = (x - self.backbone.mean) * self.backbone.img_range

        # 根据 upsampler 类型分情况处理 
        if self.backbone.upsampler == 'pixelshuffle':
            # 适用于经典超分辨率（SR）
            x = self.backbone.conv_first(x)                     # 浅层特征提取
            if self.use_fft and self.fft_branch is not None and not self.bypass_fft_fusion:
                spatial = self.backbone.forward_features(x)     # 主干深度特征（空域）
                freq = self.fft_branch(spatial)                # 频域分支：接在深层特征上，与 spatial 语义对齐
                combined = self.fusion(torch.cat([spatial, freq], dim=1))  # 拼接并融合
            else:
                combined = self.backbone.forward_features(x)
            x = self.backbone.conv_after_body(combined) + x     # 残差连接
            x = self.backbone.conv_before_upsample(x)           # 上采样前处理
            x = self.backbone.conv_last(self.backbone.upsample(x))  # 上采样 + 输出

        elif self.backbone.upsampler == 'pixelshuffledirect':
            # 适用于轻量级超分辨率（直接上采样）
            x = self.backbone.conv_first(x)
            if self.use_fft and self.fft_branch is not None and not self.bypass_fft_fusion:
                spatial = self.backbone.forward_features(x)
                freq = self.fft_branch(spatial)                # 频域接深层特征，语义对齐
                combined = self.fusion(torch.cat([spatial, freq], dim=1))
            else:
                combined = self.backbone.forward_features(x)
            x = self.backbone.conv_after_body(combined) + x
            x = self.backbone.upsample(x)                       # 一步上采样

        else:
            # 适用于图像去噪（upsampler 为 None）
            x_first = self.backbone.conv_first(x)               # 浅层特征
            if self.use_fft and self.fft_branch is not None and not self.bypass_fft_fusion:
                spatial = self.backbone.forward_features(x_first)
                freq = self.fft_branch(spatial)                 # 频域接深层特征，语义对齐
                combined = self.fusion(torch.cat([spatial, freq], dim=1))
            else:
                combined = self.backbone.forward_features(x_first)
            res = self.backbone.conv_after_body(combined) + x_first  # 残差
            x = x + self.backbone.conv_last(res)                # 输出与输入相加（去噪）

        # 恢复图像范围并加回均值
        x = x / self.backbone.img_range + self.backbone.mean
        return x


class RMG(nn.Module):
    """Residual Mamba Group (RMG).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 d_state=16,
                 ssm_ratio=4.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=None,
                 patch_size=None,
                 resi_connection='1conv',
                 is_light_sr = False,
                 scan_len=4,
                 mlp_ratio=2,
                 use_seq_exemplar_gate=False,
                 seq_exemplar_hidden_ratio=0.25,
                 seq_exemplar_pool_size=8,
                 seq_exemplar_scale_init=0.02
                ):
        super(RMG, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution # [64, 64]

        self.residual_group = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            d_state = d_state,
            ssm_ratio=ssm_ratio,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
            is_light_sr = is_light_sr,
            scan_len=scan_len,
            mlp_ratio = mlp_ratio,
            use_seq_exemplar_gate=use_seq_exemplar_gate,
            seq_exemplar_hidden_ratio=seq_exemplar_hidden_ratio,
            seq_exemplar_pool_size=seq_exemplar_pool_size,
            seq_exemplar_scale_init=seq_exemplar_scale_init
            )

        # build the last conv layer in each residual state space group
        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1))

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

    def forward(self, x, mair_ids, x_size):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, mair_ids, x_size), x_size))) + x

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        h, w = self.input_resolution
        flops += h * w * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops


class PatchEmbed(nn.Module):
    r""" transfer 2D feature map into 1D token sequence

    Args:
        img_size (int): Image size.  Default: None.
        patch_size (int): Patch token size. Default: None.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        h, w = self.img_size
        if self.norm is not None:
            flops += h * w * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r""" return 2D feature map from 1D token sequence

    Args:
        img_size (int): Image size.  Default: None.
        patch_size (int): Patch token size. Default: None.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])  # b Ph*Pw c
        return x

    def flops(self):
        flops = 0
        return flops



class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch):
        self.num_feat = num_feat
        m = []
        m.append(nn.Conv2d(num_feat, (scale**2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


# 若在 basicsr 环境中则注册 MaIR_FFT，便于配置中 type: MaIR_FFT
try:
    ARCH_REGISTRY.register()(MaIR_FFT)
except NameError:
    pass

if __name__ == '__main__':
    torch.cuda.set_device(7)
    # net = MaIR(img_size=(640, 360), embed_dim=60, d_state=1, ssm_ratio=1.1, dynamic_ids=False, mlp_ratio=1.6,upscale=2).cuda()
    # net = MaIR(img_size=(320, 180), embed_dim=60, d_state=1, ssm_ratio=1.1, dynamic_ids=False, mlp_ratio=1.6,upscale=4).cuda() ##原始代码。
    
    # 你现在的代码
   # 修改建议
    net = MaIR_FFT(
    use_fft=True, 
    img_size=(320, 180), 
    embed_dim=60, 
    d_state=1, 
    ssm_ratio=1.1, 
    dynamic_ids=False, 
    mlp_ratio=1.6,
    upscale=4).cuda()
    

    # net = MaIR(img_size=(64, 64), embed_dim=60, d_state=16, ssm_ratio=1.5, dynamic_ids=False, mlp_ratio=1.4,upscale=2).cuda()
    # net = MaIR(img_size=(320, 180), depths=(6, 6, 6, 6, 6, 6), embed_dim=180, d_state=16, ssm_ratio=2.0, dynamic_ids=False,
    #             upscale=4, mlp_ratio=2.5, upsampler='pixelshuffle').cuda()
    print(get_parameter_number(net))
    # FLOPS calculated here just for test, we use fvcore to report the final FLOPS in lightweight SR.
    print('FLOPS calculated by Ours: %.2f G'%(net.flops()/1e9))
