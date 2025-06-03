import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from timm.models.layers import trunc_normal_, DropPath
from timm.layers import trunc_normal_, DropPath


def wav_norm(wav_input):
    """
    对批量语音信号进行归一化处理。
    Args:
        wav_input: Tensor, shape [batch_size, signal_length] 或 [batch_size, ..., signal_length]
    Returns:
        wav_input_norm: 归一化后的语音信号
        batch_mean: 每个信号的均值
        batch_var: 每个信号的方差
    """
    device = wav_input.device
    batch_mean = wav_input.mean(dim=-1, keepdim=True)
    batch_var = wav_input.var(dim=-1, keepdim=True, unbiased=False)
    wav_input_norm = (wav_input - batch_mean) / torch.sqrt(batch_var + 1e-8)
    return wav_input_norm, batch_mean, batch_var

def wav_denorm(wav_output, batch_mean, batch_var):
    """
    将归一化后的语音信号恢复成原始分布。
    Args:
        wav_output: 已归一化的语音信号
        batch_mean: 均值
        batch_var: 方差
    Returns:
        wav_output_denorm: 去归一化后的信号
    """
    device = wav_output.device
    if batch_mean.device != device or batch_var.device != device:
        print(f"Warning: Device mismatch: wav_output={device}, batch_mean={batch_mean.device}, batch_var={batch_var.device}")
    wav_output_denorm = wav_output * torch.sqrt(batch_var + 1e-8) + batch_mean
    return wav_output_denorm

def enframe(wav_input, num_frame, frame_length, stride_length):
    """
    将一维语音信号按帧分段处理。
    Args:
        wav_input: Tensor, [batch_size, signal_length]
        num_frame: 帧数
        frame_length: 每帧长度
        stride_length: 帧之间的步长（重叠控制）
    Returns:
        frame_input: Tensor, [batch_size, num_frame, frame_length]
    """
    batch_size, signal_length = wav_input.shape
    device = wav_input.device
    frame_start = torch.arange(0, num_frame * stride_length, stride_length, device=device)
    frame_idx = frame_start.unsqueeze(1) + torch.arange(frame_length, device=device)
    frame_idx = frame_idx.long()
    frame_idx = frame_idx.unsqueeze(0).expand(batch_size, -1, -1)
    wav_input_expand = wav_input.unsqueeze(1).expand(-1, num_frame, -1)
    frame_input = torch.gather(wav_input_expand, 2, frame_idx)
    return frame_input

def deframe(frame_output, num_frame, frame_length, stride_length):
    """
    将分帧的信号还原为完整语音信号（非重叠重建）
    Args:
        frame_output: Tensor, [batch_size, num_frame, frame_length]
        num_frame: 帧数
        frame_length: 每帧长度
        stride_length: 帧之间的步长
    Returns:
        wav_output: Tensor, [batch_size, reconstructed_length]
    """
    batch_size = frame_output.size(0)
    device = frame_output.device
    wav1 = frame_output[:, :-1, :stride_length].reshape(batch_size, -1)
    wav2 = frame_output[:, -1, :frame_length]
    wav_output = torch.cat([wav1, wav2], dim=1)
    if wav_output.device != device:
        print(f"Warning: deframe output device mismatch: {wav_output.device}")
    return wav_output

def to_2tuple(x):
    return (x, x) if isinstance(x, int) else x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        self.dim = dim
        self.window_size = window_size if isinstance(window_size, tuple) else (window_size, window_size)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class PatchEmbed(nn.Module):
    def __init__(self, img_size=(128, 128), patch_size=2, in_chans=1, embed_dim=96, norm_layer=nn.LayerNorm):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]  # [64, 64]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]  # 4096

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=patch_size, padding=1)
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # (B, 96, 64, 64)
        x = x.flatten(2).transpose(1, 2)  # (B, 4096, 96)
        if self.norm:
            x = self.norm(x)
        # print(f"PatchEmbed output range: [{x.min().item():.4f}, {x.max().item():.4f}]")
        return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        # print(f"SwinBlock input: B={B}, L={L}, C={C}, H={H}, W={W}")
        assert L == H * W, f"input feature has wrong size: L={L}, H*W={H*W}"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x

class ResidualTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=8, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        # 6 个 SwinTransformerBlock
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                norm_layer=norm_layer
            ) for i in range(6)
        ])

    def forward(self, x):
        # 残差连接：output = input + Sequential(blocks)(input)
        shortcut = x
        for blk in self.blocks:
            x = blk(x)
        x = shortcut + x
        return x

class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=8,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size

        # 4 个 ResidualTransformerBlock
        self.blocks = nn.ModuleList([
            ResidualTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                norm_layer=norm_layer
            ) for _ in range(4)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x