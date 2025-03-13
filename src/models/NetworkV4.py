# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

# This file was modified by Robbie Watt (2024) for the purpose of downscaling
# climate data

"""Enhanced Network architectures integrating cross-attention, channel attention and spatial attention.
基于气象条件的交叉注意力基础上，增加了通道注意力（CAM）与空间注意力（SAM）来进一步强化特征提取能力。
"""

import numpy as np
import torch
from torch.nn.functional import silu
import torch.nn.functional as F

#----------------------------------------------------------------------------
# Unified routine for initializing weights and biases.

def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform':
        return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':
        return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform':
        return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':
        return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')

#----------------------------------------------------------------------------
# Fully-connected layer.

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode='kaiming_normal', init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x

#----------------------------------------------------------------------------
# Convolutional layer with optional up/downsampling.

class Conv2d(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels, kernel, bias=True, up=False, down=False,
        resample_filter=[1,1], fused_resample=False, init_mode='kaiming_normal', init_weight=1, init_bias=0,
    ):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        init_kwargs = dict(mode=init_mode, fan_in=in_channels*kernel*kernel, fan_out=out_channels*kernel*kernel)
        self.weight = torch.nn.Parameter(weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs) * init_weight) if kernel else None
        self.bias = torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias) if kernel and bias else None
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer('resample_filter', f if up or down else None)

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = self.resample_filter.to(x.dtype) if self.resample_filter is not None else None
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.fused_resample and self.up and w is not None:
            x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=max(f_pad - w_pad, 0))
            x = torch.nn.functional.conv2d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            x = torch.nn.functional.conv2d(x, w, padding=w_pad+f_pad)
            x = torch.nn.functional.conv2d(x, f.tile([self.out_channels, 1, 1, 1]), groups=self.out_channels, stride=2)
        else:
            if self.up:
                x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if self.down:
                x = torch.nn.functional.conv2d(x, f.tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if w is not None:
                x = torch.nn.functional.conv2d(x, w, padding=w_pad)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x

#----------------------------------------------------------------------------
# Group normalization.

class GroupNorm(torch.nn.Module):
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        return torch.nn.functional.group_norm(x, num_groups=self.num_groups,
                                                weight=self.weight.to(x.dtype), bias=self.bias.to(x.dtype), eps=self.eps)

#----------------------------------------------------------------------------
# Channel Attention Module (CAM)

class CAM(torch.nn.Module):
    def __init__(self, in_ch, reduction=2, relu_a=0.01):
        super().__init__()
        hidden_dim = in_ch // reduction
        self.mlp = torch.nn.Sequential(
            Linear(in_ch, hidden_dim, init_mode='kaiming_normal'),
            torch.nn.LeakyReLU(negative_slope=relu_a),
            Linear(hidden_dim, in_ch, init_mode='kaiming_normal')
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        # Global max pooling and average pooling along spatial dimensions
        max_pool = torch.max(torch.max(x, dim=2)[0], dim=2)[0]  # [B, C]
        avg_pool = torch.mean(torch.mean(x, dim=2), dim=2)         # [B, C]
        attn = self.sigmoid(self.mlp(max_pool) + self.mlp(avg_pool))  # [B, C]
        attn = attn.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        return x * attn

#----------------------------------------------------------------------------
# Spatial Attention Module (SAM)

class SAM(torch.nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        # Max and average pooling along channel dimension
        max_pool, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
        avg_pool = torch.mean(x, dim=1, keepdim=True)      # [B, 1, H, W]
        pooled = torch.cat([max_pool, avg_pool], dim=1)      # [B, 2, H, W]
        attn = self.sigmoid(self.conv(pooled))             # [B, 1, H, W]
        return x * attn

#----------------------------------------------------------------------------
# Attention weight computation (for self-attention).
# 使用FP32计算注意力权重以节省内存。

class AttentionOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k):
        w = torch.einsum('ncq,nck->nqk', q.to(torch.float32), (k / np.sqrt(k.shape[1])).to(torch.float32)).softmax(dim=2).to(q.dtype)
        ctx.save_for_backward(q, k, w)
        return w

    @staticmethod
    def backward(ctx, dw):
        q, k, w = ctx.saved_tensors
        db = torch._softmax_backward_data(grad_output=dw.to(torch.float32), output=w.to(torch.float32),
                                            dim=2, input_dtype=torch.float32)
        dq = torch.einsum('nck,nqk->ncq', k.to(torch.float32), db).to(q.dtype) / np.sqrt(k.shape[1])
        dk = torch.einsum('ncq,nqk->nck', q.to(torch.float32), db).to(k.dtype) / np.sqrt(k.shape[1])
        return dq, dk

# 新增气象注意力模块
class MeteoAttentionModule(torch.nn.Module):
    def __init__(self, precip_channels=1, meteo_channels=6, out_channels=64, downsample_factor=4):
        super().__init__()
        # 卷积层提取特征
        self.precip_conv = torch.nn.Conv2d(precip_channels, out_channels, kernel_size=3, padding=1)
        self.meteo_conv = torch.nn.Conv2d(meteo_channels, out_channels, kernel_size=3, padding=1)
        # 交叉注意力
        self.cross_attn = torch.nn.MultiheadAttention(embed_dim=out_channels, num_heads=4)
        # 归一化
        self.norm = torch.nn.GroupNorm(8, out_channels)
        self.downsample_factor = downsample_factor
        self.upsample = torch.nn.Upsample(scale_factor=downsample_factor, mode='bilinear')

    def forward(self, coarse_precip, meteo_conditions):
        # 输入形状：coarse_precip [B, 1, H, W], meteo_conditions [B, 6, H, W]
        B, _, H, W = coarse_precip.shape

        # 下采样（可选，根据计算资源调整）
        coarse_precip_down = F.interpolate(coarse_precip, scale_factor=1 / self.downsample_factor, mode='bilinear')
        meteo_conditions_down = F.interpolate(meteo_conditions, scale_factor=1 / self.downsample_factor,
                                              mode='bilinear')

        # 特征提取
        query = self.precip_conv(coarse_precip_down)  # [B, 64, H/4, W/4]
        key_value = self.meteo_conv(meteo_conditions_down)  # [B, 64, H/4, W/4]

        # 展平为序列以适配 MultiheadAttention
        query = query.view(B, 64, -1).permute(2, 0, 1)  # [H/4*W/4, B, 64]
        key = key_value.view(B, 64, -1).permute(2, 0, 1)  # [H/4*W/4, B, 64]
        value = key_value.view(B, 64, -1).permute(2, 0, 1)  # [H/4*W/4, B, 64]

        # 交叉注意力计算
        attn_output, _ = self.cross_attn(query, key, value)

        # 恢复空间形状并上采样
        attn_output = attn_output.permute(1, 2, 0).view(B, 64, H // self.downsample_factor, W // self.downsample_factor)
        attn_output = self.upsample(attn_output)  # [B, 64, H, W]

        return self.norm(attn_output)

# 新增残差注意力模块
class ResidualAttentionBlock(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = Conv2d(channels, channels, kernel=3)
        self.norm = GroupNorm(channels)
        self.attn = torch.nn.Sequential(
            Conv2d(channels, channels, kernel=1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        residual = x
        x = self.conv(silu(self.norm(x)))
        attn_weights = self.attn(x)
        return residual * attn_weights + x

#----------------------------------------------------------------------------
# Unified U-Net Block integrating:
# 1. 标准卷积残差模块
# 2. 残差注意力模块
# 3. 通道注意力模块 (CAM) 和空间注意力模块 (SAM)
# 4. 气象注意力模块
# 5. 自注意力模块（可选）
class UNetBlock(torch.nn.Module):
    def __init__(self,
                 in_channels, out_channels, emb_channels, up=False, down=False, attention=False,
                 num_heads=None, channels_per_head=64, dropout=0, skip_scale=1, eps=1e-5,
                 resample_filter=[1,1], resample_proj=False, adaptive_scale=True,
                 init=dict(), init_zero=dict(init_weight=0), init_attn=None,
                 use_cam=True, use_sam=True
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.use_cam = use_cam
        self.use_sam = use_sam
        # 自注意力头数
        self.num_heads = 0 if not attention else (num_heads if num_heads is not None else out_channels // channels_per_head)
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        self.norm0 = GroupNorm(in_channels, eps=eps)
        self.conv0 = Conv2d(in_channels, out_channels, kernel=3, up=up, down=down, resample_filter=resample_filter, **init)
        self.affine = Linear(emb_channels, out_channels * (2 if adaptive_scale else 1), **init)
        self.norm1 = GroupNorm(out_channels, eps=eps)
        self.conv1 = Conv2d(out_channels, out_channels, kernel=3, **init_zero)
        self.residual_attention = ResidualAttentionBlock(out_channels)

        # 通道注意力 (CAM) 和空间注意力 (SAM)
        if self.use_cam:
            self.cam = CAM(out_channels)
        if self.use_sam:
            self.sam = SAM()

        # 自注意力模块（如果启用）
        if self.num_heads:
            self.norm2 = GroupNorm(out_channels, eps=eps)
            self.qkv = Conv2d(out_channels, out_channels * 3, kernel=1, **(init_attn if init_attn is not None else init))
            self.proj = Conv2d(out_channels, out_channels, kernel=1, **init_zero)

    def forward(self, x, emb):
        x = self.conv0(silu(self.norm0(x)))

        params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
        if self.adaptive_scale:
            scale, shift = params.chunk(2, dim=1)
            x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = silu(self.norm1(x + params))

        x = self.conv1(torch.nn.functional.dropout(x, p=self.dropout, training=self.training))

        # 添加残差注意力
        x = self.residual_attention(x)

        # 再应用通道注意力和空间注意力（顺序可根据实验调优）
        if self.use_cam:
            x = self.cam(x)
        if self.use_sam:
            x = self.sam(x)

        # self-attention
        if self.num_heads > 0:
            q, k, v = self.qkv(self.norm2(x)).reshape(x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1).unbind(2)
            w = AttentionOp.apply(q, k)
            a = torch.einsum('nqk,nck->ncq', w, v)
            x = self.proj(a.reshape(*x.shape)).add_(x)

        x = x * self.skip_scale

        return x

#----------------------------------------------------------------------------
# Timestep embedding (PositionalEmbedding)

class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(0, self.num_channels // 2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        return torch.cat([x.cos(), x.sin()], dim=1)

#----------------------------------------------------------------------------
# Reimplementation of the ADM architecture with the enhanced attention modules.
class UNet(torch.nn.Module):
    def __init__(self,
                 img_resolution,
                 in_channels=2,  # [noised_residual_precip, coarse_precip]
                 out_channels=1,
                 label_dim=0,
                 augment_dim=0,
                 model_channels=128,
                 channel_mult=[1,2,3,4],
                 channel_mult_emb=4,
                 num_blocks=2,
                 attn_resolutions=[32,16,8],
                 dropout=0.10,
                 label_dropout=0,
                 use_diffuse=True,
                 use_cam=True,
                 use_sam=True,
                 meteo_channels=64):
        super().__init__()

        self.label_dropout = label_dropout
        emb_channels = model_channels * channel_mult_emb
        init = dict(init_mode='kaiming_uniform', init_weight=np.sqrt(1/3), init_bias=np.sqrt(1/3))
        init_zero = dict(init_mode='kaiming_uniform', init_weight=0, init_bias=0)
        block_kwargs = dict(emb_channels=emb_channels, channels_per_head=64, dropout=dropout,
                            init=init, init_zero=init_zero, use_cam=use_cam, use_sam=use_sam)

        self.map_noise = PositionalEmbedding(num_channels=model_channels) if use_diffuse else None
        self.map_augment = Linear(augment_dim, model_channels, bias=False, **init_zero) if augment_dim else None
        self.map_layer0 = Linear(model_channels, emb_channels, **init)
        self.map_layer1 = Linear(emb_channels, emb_channels, **init)
        self.map_label = Linear(label_dim, emb_channels, bias=False, init_mode='kaiming_normal', init_weight=np.sqrt(label_dim)) if label_dim else None

        self.meteo_attention = MeteoAttentionModule(precip_channels=1, meteo_channels=6, out_channels=meteo_channels)
        # 输入融合层：将 2 + meteo_channels 调整为 model_channels
        self.input_fusion = Conv2d(in_channels + meteo_channels, model_channels, kernel=3)

        # Encoder
        self.enc = torch.nn.ModuleDict()
        cout = model_channels  # 输入已融合为 model_channels
        for level, mult in enumerate(channel_mult):
            resx = img_resolution[0] >> level
            resy = img_resolution[1] >> level
            if level == 0:
                cin = cout  # 输入通道数为 model_channels
                cout = model_channels * mult
                self.enc[f'{resx}x{resy}_conv'] = Conv2d(cin, cout, kernel=3, **init)
            else:
                self.enc[f'{resx}x{resy}_down'] = UNetBlock(cout, cout, down=True, **block_kwargs)
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                attn_flag = resx in attn_resolutions
                self.enc[f'{resx}x{resy}_block{idx}'] = UNetBlock(cin, cout, attention=attn_flag, **block_kwargs)
        skips = [block.out_channels for block in self.enc.values() if hasattr(block, 'out_channels')]

        # Decoder
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            resx = img_resolution[0] >> level
            resy = img_resolution[1] >> level
            if level == len(channel_mult) - 1:
                self.dec[f'{resx}x{resy}_in0'] = UNetBlock(cout, cout, attention=True, **block_kwargs)
                self.dec[f'{resx}x{resy}_in1'] = UNetBlock(cout, cout, **block_kwargs)
            else:
                self.dec[f'{resx}x{resy}_up'] = UNetBlock(cout, cout, up=True, **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                attn_flag = resx in attn_resolutions
                self.dec[f'{resx}x{resy}_block{idx}'] = UNetBlock(cin, cout, attention=attn_flag, **block_kwargs)
        self.out_norm = GroupNorm(cout)
        self.out_conv = Conv2d(cout, out_channels, kernel=3, **init_zero)

        self.multiscale_fusion = Conv2d(2, 1, kernel=3)

    def forward(self, x, noise_labels=None, class_labels=None,
                augment_labels=None, meteo_conditions=None, multi_scale_residuals=None):
        # x: [B, 2, H, W]，即 [noised_residual_precip, coarse_precip]
        coarse_precip = x[:, 1:2, :, :]  # [B, 1, H, W]
        meteo_features = self.meteo_attention(coarse_precip, meteo_conditions)  # [B, 64, H, W]

        # 在输入处融合 meteo_features
        x = torch.cat([x, meteo_features], dim=1)  # [B, 66, H, W]
        x = self.input_fusion(x)  # [B, model_channels, H, W]

        # Embedding
        emb = torch.zeros(1, self.map_layer1.in_features, device=x.device)
        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (torch.rand(x.shape[0], 1, device=x.device) >= self.label_dropout).to(tmp.dtype)
            emb = self.map_label(tmp)
        if self.map_noise is not None:
            emb_n = self.map_noise(noise_labels)
            emb_n = silu(self.map_layer0(emb_n))
            emb_n = self.map_layer1(emb_n)
            emb = emb + emb_n
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        emb = silu(emb)

        # Encoder
        skips = []
        for key, block in self.enc.items():
            x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
            skips.append(x)

        # Decoder
        for key, block in self.dec.items():
            if x.shape[1] != block.in_channels:
                x = torch.cat([x, skips.pop()], dim=1)
            x = block(x, emb)

        if multi_scale_residuals is not None:
            fused_residual = self.multiscale_fusion(torch.cat(multi_scale_residuals, dim=1))
            x = x + fused_residual

        x = self.out_conv(silu(self.out_norm(x)))
        return x

#----------------------------------------------------------------------------
# Improved preconditioning (EDM)

class EDMPrecond(torch.nn.Module):
    def __init__(self,
                 img_resolution,
                 in_channels,
                 out_channels,
                 label_dim=0,
                 use_fp16=False,
                 sigma_min=0,
                 sigma_max=float('inf'),
                 sigma_data=1.0,
                 model_type='UNet',
                 **model_kwargs):
        super().__init__()
        self.img_resolution = img_resolution
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

        # 保证气象注意力相关参数正确传递
        model_kwargs['use_cam'] = model_kwargs.get('use_cam', True)
        model_kwargs['use_sam'] = model_kwargs.get('use_sam', True)

        self.model = globals()[model_type](
            img_resolution=img_resolution, in_channels=in_channels,
            out_channels=out_channels, label_dim=label_dim, **model_kwargs)

    def forward(self, x, sigma, condition_img=None, class_labels=None,
                force_fp32=True, meteo_conditions=None, multi_scale_residuals=None, **model_kwargs):
        if condition_img is not None:
            in_img = torch.cat([x, condition_img], dim=1)
        else:
            in_img = x
        sigma = sigma.reshape(-1, 1, 1, 1)
        class_labels = (None if self.label_dim == 0 else
                        torch.zeros(1, self.label_dim, device=in_img.device) if class_labels is None
                        else class_labels.to(torch.float32).reshape(-1, self.label_dim))
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and in_img.device.type == 'cuda') else torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        F_x = self.model(x=(c_in * in_img).to(dtype), noise_labels=c_noise.flatten(),
                         class_labels=class_labels, meteo_conditions=meteo_conditions,
                         multi_scale_residuals=multi_scale_residuals)
        D_x = c_skip * x + c_out * F_x

        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)