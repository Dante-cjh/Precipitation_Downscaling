# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

# This file was modified by Robbie Watt (2024) for the purpose of downscaling
# climate data

"""Model architectures and preconditioning schemes used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import numpy as np
import torch
from torch.nn.functional import silu

#----------------------------------------------------------------------------
# Unified routine for initializing weights and biases.

def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform': return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':  return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform': return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':  return np.sqrt(1 / fan_in) * torch.randn(*shape)
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
# Convolutional layer with optional up/down sampling.

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
        x = torch.nn.functional.group_norm(x, num_groups=self.num_groups, weight=self.weight.to(x.dtype), bias=self.bias.to(x.dtype), eps=self.eps)
        return x

#----------------------------------------------------------------------------
# Attention weight computation, i.e., softmax(Q^T * K).
# Performs all computation using FP32, but uses the original datatype for
# inputs/outputs/gradients to conserve memory.

class AttentionOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k):
        w = torch.einsum('ncq,nck->nqk', q.to(torch.float32), (k / np.sqrt(k.shape[1])).to(torch.float32)).softmax(dim=2).to(q.dtype)
        ctx.save_for_backward(q, k, w)
        return w

    @staticmethod
    def backward(ctx, dw):
        q, k, w = ctx.saved_tensors
        db = torch._softmax_backward_data(grad_output=dw.to(torch.float32), output=w.to(torch.float32), dim=2, input_dtype=torch.float32)
        dq = torch.einsum('nck,nqk->ncq', k.to(torch.float32), db).to(q.dtype) / np.sqrt(k.shape[1])
        dk = torch.einsum('ncq,nqk->nck', q.to(torch.float32), db).to(k.dtype) / np.sqrt(k.shape[1])
        return dq, dk


class MeteoEncoder(torch.nn.Module):
    """Meteorological Condition Encoder (NEW)"""
    def __init__(self, in_channels=6, emb_dim=256):  # 输入6个气象变量
        super().__init__()
        self.net = torch.nn.Sequential(
            Conv2d(in_channels, 128, kernel=3),
            GroupNorm(128),
            torch.nn.SiLU(),  # 使用 SiLU 激活函数，注意要用类名而不是直接调用
            Conv2d(128, emb_dim, kernel=3),
            GroupNorm(emb_dim),
            torch.nn.SiLU()  # 使用 SiLU 激活函数
        )

    def forward(self, meteo_conditions):
        """Input: [batch, 6, 224, 224] (r2, t, u10, v10, lsm, z)
           Output: [batch, emb_dim, 56, 56]"""
        return self.net(meteo_conditions)


class CrossAttention(torch.nn.Module):
    """Meteorological Cross-Attention Layer (NEW)"""
    def __init__(self, query_dim, context_dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.scale = dim_head ** -0.5

        self.to_q = Conv2d(query_dim, hidden_dim, kernel=1)
        self.to_kv = Conv2d(context_dim, 2 * hidden_dim, kernel=1)
        self.to_out = Conv2d(hidden_dim, query_dim, kernel=1)

    def forward(self, x, context):
        # 如果 context 的空间尺寸与 x 不一致，调整它
        if context.shape[2:] != x.shape[2:]:
            context = torch.nn.functional.interpolate(context, size=x.shape[2:], mode='bilinear', align_corners=False)

        b, c, h, w = x.shape
        # Projections
        q = self.to_q(x)  # [b, h_dim, h, w]
        k, v = self.to_kv(context).chunk(2, dim=1)  # [b, h_dim, h, w] each

        # Rearrange for multi-head attention
        q = q.view(b, self.heads, -1, h * w).permute(0, 1, 3, 2)  # [b, heads, h*w, d_head]
        k = k.view(b, self.heads, -1, h * w).permute(0, 1, 2, 3)  # [b, heads, d_head, h*w]
        v = v.view(b, self.heads, -1, h * w).permute(0, 1, 3, 2)  # [b, heads, h*w, d_head]

        # Attention matrix
        sim = torch.matmul(q, k) * self.scale  # [b, heads, h*w, h*w]
        attn = sim.softmax(dim=-1)

        # Aggregate values
        out = torch.matmul(attn, v)  # [b, heads, h*w, d_head]
        out = out.permute(0, 1, 3, 2).reshape(b, -1, h, w)
        return self.to_out(out) + x  # Residual connection


#----------------------------------------------------------------------------
# Unified U-Net block with optional up/downsampling and self-attention.
# Represents the union of all features employed by the DDPM++, NCSN++, and
# ADM architectures.

class UNetBlock(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels, emb_channels, up=False, down=False,
        attention=False, meteo_attention=False,
        num_heads=None, channels_per_head=64, dropout=0, skip_scale=1, eps=1e-5,
        resample_filter=[1,1], resample_proj=False, adaptive_scale=True,
        init=dict(), init_zero=dict(init_weight=0), init_attn=None,
        meteo_context_dim=None
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.meteo_attention = meteo_attention  # new
        self.num_heads = 0 if not attention else num_heads if num_heads is not None else out_channels // channels_per_head
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=3, up=up, down=down, resample_filter=resample_filter, **init)
        self.affine = Linear(in_features=emb_channels, out_features=out_channels*(2 if adaptive_scale else 1), **init)
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=3, **init_zero)

        # Meteorological cross-attention
        if self.meteo_attention:
            # 使用 meteo_context_dim 指定气象条件通道数，若未提供，则默认使用 emb_channels
            context_dim = meteo_context_dim if meteo_context_dim is not None else emb_channels
            self.meteo_attn = CrossAttention(query_dim=out_channels, context_dim=context_dim, heads=4, dim_head=32)

        if in_channels != out_channels or up or down:
            kernel = 1 if resample_proj or (in_channels != out_channels) else 0
            self.skip = Conv2d(in_channels, out_channels, kernel=kernel, up=up, down=down, resample_filter=resample_filter, **init)
        else:
            self.skip = None

        # Original self-attention
        self.num_heads = 0 if not attention else num_heads if num_heads is not None else out_channels // channels_per_head
        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(in_channels=out_channels, out_channels=out_channels*3, kernel=1, **(init_attn if init_attn is not None else init))
            self.proj = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=1, **init_zero)


    def forward(self, x, emb, meteo_context=None):
        orig = x
        x = self.conv0(silu(self.norm0(x)))

        # Adaptive scaling
        params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
        if self.adaptive_scale:
            scale, shift = params.chunk(chunks=2, dim=1)
            x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = silu(self.norm1(x.add_(params)))

        x = self.conv1(torch.nn.functional.dropout(x, p=self.dropout, training=self.training))
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        # Meteorological cross-attention
        if self.meteo_attention and meteo_context is not None:
            x = self.meteo_attn(x, context=meteo_context) + x

        # self-attention
        if self.num_heads > 0:
            q, k, v = self.qkv(self.norm2(x)).reshape(x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1).unbind(2)
            w = AttentionOp.apply(q, k)
            a = torch.einsum('nqk,nck->ncq', w, v)
            x = self.proj(a.reshape(*x.shape)).add_(x)
            x = x * self.skip_scale

        return x

#----------------------------------------------------------------------------
# Timestep embedding used in the DDPM++ and ADM architectures.
class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

#----------------------------------------------------------------------------
# Timestep embedding used in the NCSN++ architecture.

class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer('freqs', torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

#----------------------------------------------------------------------------
# Reimplementation of the ADM architecture from the paper
# "Diffusion Models Beat GANS on Image Synthesis". Equivalent to the
# original implementation by Dhariwal and Nichol, available at
# https://github.com/openai/guided-diffusion

class UNet(torch.nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution at input/output.
        in_channels,                        # Number of color channels at input.
        out_channels,                       # Number of color channels at output.
        label_dim           = 0,            # Number of class labels, 0 = unconditional.
        augment_dim         = 0,            # Augmentation label dimensionality, 0 = no augmentation.

        model_channels      = 128,          # Base multiplier for the number of channels.
        channel_mult        = [1,2,3,4],    # Per-resolution multipliers for the number of channels.
        channel_mult_emb    = 4,            # Multiplier for the dimensionality of the embedding vector.
        num_blocks          = 2,            # Number of residual blocks per resolution.
        attn_resolutions    = [32,16,8],    # List of resolutions with self-attention.
        dropout             = 0.10,         # List of resolutions with self-attention.
        label_dropout       = 0,            # Dropout probability of class labels for classifier-free guidance.
        use_diffuse = True,                 # Use Unet for diffusion
        use_meteo_attn = True,  # ! NEW: 气象注意力开关
        meteo_emb_dim = 256  # ! NEW: 气象编码维度
    ):
        super().__init__()
        self.meteo_encoder = MeteoEncoder(emb_dim=meteo_emb_dim)

        self.label_dropout = label_dropout
        emb_channels = model_channels * channel_mult_emb
        init = dict(init_mode='kaiming_uniform', init_weight=np.sqrt(1/3), init_bias=np.sqrt(1/3))
        init_zero = dict(init_mode='kaiming_uniform', init_weight=0, init_bias=0)

        # 根据分辨率决定是否使用交叉注意力：在中低分辨率层 (例如分辨率 <= 112) 启用
        def use_meteo_atten(res):
            return use_meteo_attn and (res <= 56)

        block_kwargs = dict(emb_channels=emb_channels, channels_per_head=64, dropout=dropout, init=init, init_zero=init_zero, meteo_context_dim=meteo_emb_dim)

        # Mapping.
        self.map_noise = PositionalEmbedding(num_channels=model_channels) if use_diffuse else None
        self.map_augment = Linear(in_features=augment_dim, out_features=model_channels, bias=False, **init_zero) if augment_dim else None
        self.map_layer0 = Linear(in_features=model_channels, out_features=emb_channels, **init)
        self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)
        self.map_label = Linear(in_features=label_dim, out_features=emb_channels, bias=False, init_mode='kaiming_normal', init_weight=np.sqrt(label_dim)) if label_dim else None

        assert len(img_resolution) == 2

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        for level, mult in enumerate(channel_mult):
            resx = img_resolution[0] >> level
            resy = img_resolution[1] >> level
            if level == 0:
                cin = cout
                cout = model_channels * mult
                self.enc[f'{resx}x{resy}_conv'] = Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)
            else:
                self.enc[f'{resx}x{resy}_down'] = UNetBlock(in_channels=cout, out_channels=cout,down=True,
                                                            **block_kwargs, meteo_attention=use_meteo_atten(resx))
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                self.enc[f'{resx}x{resy}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=(resx in
                                                                                                                 attn_resolutions),
                                                                  **block_kwargs, meteo_attention=use_meteo_atten(resx))
        skips = [block.out_channels for block in self.enc.values() if hasattr(block, 'out_channels')]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            resx = img_resolution[0] >> level
            resy = img_resolution[1] >> level
            if level == len(channel_mult) - 1:
                self.dec[f'{resx}x{resy}_in0'] = UNetBlock(in_channels=cout, out_channels=cout, attention=True,
                                                           **block_kwargs, meteo_attention=use_meteo_atten(resx))
                self.dec[f'{resx}x{resy}_in1'] = UNetBlock(in_channels=cout, out_channels=cout, **block_kwargs,
                                                           meteo_attention=use_meteo_atten(resx))
            else:
                self.dec[f'{resx}x{resy}_up'] = UNetBlock(in_channels=cout, out_channels=cout, up=True, **block_kwargs,
                                                          meteo_attention=use_meteo_atten(resx))
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                self.dec[f'{resx}x{resy}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=(resx in
                                                                                                                 attn_resolutions),
                                                                  **block_kwargs, meteo_attention=use_meteo_atten(resx))
        self.out_norm = GroupNorm(num_channels=cout)
        self.out_conv = Conv2d(in_channels=cout, out_channels=out_channels, kernel=3, **init_zero)

    def forward(self, x, noise_labels=None, class_labels=None,
                augment_labels=None, meteo_conditions=None):
        # 气象编码
        meteo_emb = self.meteo_encoder(meteo_conditions) if meteo_conditions is not None else None

        # Mapping.
        emb = torch.zeros([1, self.map_layer1.in_features], device=x.device)
        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (torch.rand([x.shape[0], 1],
                                        device=x.device) >= self.label_dropout).to(
                    tmp.dtype)
            emb = self.map_label(tmp)
        if self.map_noise is not None:
            emb_n = self.map_noise(noise_labels)
            emb_n = silu(self.map_layer0(emb_n))
            emb_n = self.map_layer1(emb_n)
            emb = emb + emb_n
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)

        emb = silu(emb)

        # Encoder.
        skips = []
        for block in self.enc.values():
            if isinstance(block, UNetBlock):
                x = block(x, emb, meteo_context=meteo_emb)
            else:
                x = block(x)
            skips.append(x)

        # Decoder处理
        for block in self.dec.values():
            if x.shape[1] != block.in_channels:
                x = torch.cat([x, skips.pop()], dim=1)
            if isinstance(block, UNetBlock):
                x = block(x, emb, meteo_context=meteo_emb)  # ! NEW: 传入气象编码
            else:
                x = block(x)

        x = self.out_conv(silu(self.out_norm(x)))
        return x

#----------------------------------------------------------------------------
# Improved preconditioning proposed in the paper "Elucidating the Design
# Space of Diffusion-Based Generative Models" (EDM).

class EDMPrecond(torch.nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution.
        in_channels,                       # Number of color channels.
        out_channels,                       # Number of color channels.
        label_dim       = 0,                # Number of class labels, 0 = unconditional.
        use_fp16        = False,            # Execute the underlying model at FP16 precision?
        sigma_min       = 0,                # Minimum supported noise level.
        sigma_max       = float('inf'),     # Maximum supported noise level.
        sigma_data      = 1.0,              # Expected standard deviation of
                 # the training data.
        model_type      = 'UNet',   # Class name of the underlying model.
        **model_kwargs,                     # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

        # 新增气象相关参数 (NEW)
        model_kwargs['use_meteo_attn'] = model_kwargs.get('use_meteo_attn', True)
        model_kwargs['meteo_emb_dim'] = model_kwargs.get('meteo_emb_dim', 256)

        self.model = globals()[model_type](
            img_resolution=img_resolution, in_channels=in_channels,
            out_channels=out_channels, label_dim=label_dim, **model_kwargs)

    def forward(self, x, sigma, condition_img=None, class_labels=None,
                force_fp32=True, meteo_conditions=None, **model_kwargs):

        # 条件拼接
        if condition_img is not None:
            in_img = torch.cat([x, condition_img], dim=1)
        else:
            in_img = x

        # 扩散系数计算（保持原样）
        sigma = sigma.reshape(-1, 1, 1, 1)
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=in_img.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and in_img.device.type == 'cuda') else torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        F_x = self.model((c_in * in_img).to(dtype),
                         noise_labels=c_noise.flatten(),
                         class_labels=class_labels,
                         meteo_conditions=meteo_conditions,
                         **model_kwargs).to(dtype)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

#----------------------------------------------------------------------------