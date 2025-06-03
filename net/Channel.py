
import torch
import torch.nn as nn
import numpy as np

class Channel(nn.Module):
    def __init__(self, channel_type='awgn', snr_db=10.0, freq_offset=0.0050, use_csi=False, use_std=False, std=0.0, device='cuda'):
        """
        初始化信道模型，适配 Encoder 输出 (B, 1, 128, 128) 和 Decoder 输入 (B, 1, 128, 128)。

        参数:
            channel_type: 信道类型 ('none', 'awgn', 'rayleigh', 'frequency_offset')
            snr_db: 信噪比（dB），默认 10.0
            freq_offset: 频率偏移，默认 0.0050
            use_csi: 是否使用完美 CSI，默认 False
            use_std: 是否使用固定标准差，默认 False
            std: 噪声标准差（若 use_std=True），默认 0.0
            device: 设备，默认 'cuda'
        """
        super().__init__()
        self.device = torch.device(device)
        self.chan_type = channel_type
        self.use_std = use_std
        self.use_csi = use_csi
        self.freq_offset = freq_offset
        self.std = std
        self.snr_db = snr_db

        valid_types = {'none', 'awgn', 'rayleigh', 'frequency_offset'}
        if self.chan_type not in valid_types:
            raise ValueError(f"不支持的信道类型: {self.chan_type}, 应为 {valid_types}")

        if abs(self.freq_offset) > 0.5:
            raise ValueError(f"频率偏移 {self.freq_offset} 超出合理范围 [-0.5, 0.5]")

        # 归一化层
        self.norm = nn.Tanh()  # 确保输出范围接近 [-1, 1]
        self.to(self.device)

    def generate_complex_noise(self, shape, std):
        """生成复高斯噪声，实部和虚部独立，标准差为 std。"""
        noise_real = torch.normal(mean=0.0, std=std, size=shape, device=self.device)
        noise_imag = torch.normal(mean=0.0, std=std, size=shape, device=self.device)
        return noise_real + 1j * noise_imag

    def gaussian_noise_layer(self, input_layer, std):
        """应用 AWGN 信道：y = x + n。"""
        noise = self.generate_complex_noise(input_layer.shape, std)
        return input_layer + noise

    def apply_frequency_offset(self, input_layer, nt_half):
        """应用频率偏移：y = x * exp(1j*2π*f*t)。"""
        if self.freq_offset == 0.0:
            return input_layer
        batch_size, channels, nt_half, _ = input_layer.shape
        t = torch.arange(nt_half, device=self.device).float()
        phase = 2 * np.pi * self.freq_offset * t
        rotation = torch.exp(1j * phase)
        rotation = rotation.view(1, 1, nt_half, 1).expand(batch_size, channels, nt_half, 1)
        return input_layer * rotation

    def rayleigh_noise_layer(self, input_layer, std, nt_half):
        """应用瑞利衰落信道：y = h * x + n，可选 CSI 估计。"""
        h_real = torch.normal(mean=0.0, std=1.0, size=input_layer.shape, device=self.device)
        h_imag = torch.normal(mean=0.0, std=1.0, size=input_layer.shape, device=self.device)
        h = (h_real + 1j * h_imag) / np.sqrt(2.0)
        noise = self.generate_complex_noise(input_layer.shape, std)
        y = input_layer * h + noise
        y = self.apply_frequency_offset(y, nt_half)
        if self.use_csi:
            h_safe = h + 1e-10 * (h == 0).float()
            return y / h_safe
        return y

    def apply_frequency_offset_only(self, input_layer, nt_half):
        """仅应用频率偏移。"""
        return self.apply_frequency_offset(input_layer, nt_half)

    def complex_forward(self, complex_input, chan_param=None, nt_half=8192):
        """在复数域应用信道效应。"""
        if self.use_std:
            std = chan_param if chan_param is not None else self.std
        else:
            snr = chan_param if chan_param is not None else self.snr_db
            std = np.sqrt(1.0 / (2 * 10 ** (snr / 10)))

        if self.chan_type == 'none':
            return complex_input
        elif self.chan_type == 'awgn':
            return self.gaussian_noise_layer(complex_input, std=std)
        elif self.chan_type == 'rayleigh':
            return self.rayleigh_noise_layer(complex_input, std=std, nt_half=nt_half)
        elif self.chan_type == 'frequency_offset':
            return self.apply_frequency_offset_only(complex_input, nt_half=nt_half)
        else:
            raise ValueError(f"不支持的信道类型: {self.chan_type}")

    def forward(self, input, chan_param=None):
        """
        前向传播，处理 Encoder 输出到 Decoder 输入。

        参数:
            input: 输入信号，形状 [B, 1, 128, 128]
            chan_param: SNR（dB）或 std，控制噪声强度

        返回:
            output: 信道输出，形状 [B, 1, 128, 128]
        """
        batch_size, channels, height, width = input.shape
        assert channels == 1 and height == 128 and width == 128, f"输入形状应为 [B, 1, 128, 128]，实际为 {input.shape}"

        # 转换为 IQ 格式
        nt_half = 8192
        input_flat = input.view(batch_size, 1, 128 * 128)  # [B, 1, 16384]
        input_iq = input_flat.view(batch_size, 1, nt_half, 2)  # [B, 1, 8192, 2]

        # 归一化信号功率
        scale = np.sqrt(nt_half / 2.0)
        input_norm = scale * torch.nn.functional.normalize(input_iq, p=2, dim=2)

        # 提取实部和虚部，生成复数信号
        real_part = input_norm[:, :, :, 0]  # [B, 1, 8192]
        imag_part = input_norm[:, :, :, 1]  # [B, 1, 8192]
        complex_input = real_part + 1j * imag_part
        complex_input = complex_input.unsqueeze(-1)  # [B, 1, 8192, 1]

        # 应用信道效应
        complex_output = self.complex_forward(complex_input, chan_param, nt_half=nt_half)

        # 分解复数输出
        output_real = torch.real(complex_output).squeeze(-1)  # [B, 1, 8192]
        output_imag = torch.imag(complex_output).squeeze(-1)  # [B, 1, 8192]
        output_iq = torch.stack([output_real, output_imag], dim=-1)  # [B, 1, 8192, 2]

        # 转换回 [B, 1, 128, 128]
        output_flat = output_iq.view(batch_size, 1, 128 * 128)  # [B, 1, 16384]
        output = output_flat.view(batch_size, 1, 128, 128)  # [B, 1, 128, 128]

        # 归一化输出范围
        output = self.norm(output)

        # print(f"Channel output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
        return output

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(32, 1, 128, 128, device=device)  # 模拟 Encoder 输出
    channel = Channel(channel_type='awgn', snr_db=10.0, freq_offset=0.0050, device=device).to(device)
    y = channel(x, chan_param=10.0)
    print("输入 shape:", x.shape)
    print("输出 shape:", y.shape)
    print(f"输出 range: [{y.min().item():.4f}, {y.max().item():.4f}]")
