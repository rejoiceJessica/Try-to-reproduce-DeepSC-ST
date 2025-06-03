import torch
import torch.nn as nn
from net.Modules import enframe, wav_norm

class Encoder(nn.Module):
    def __init__(self, frame_length=128, stride_length=128, num_frame=128, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.num_frame = num_frame
        self.frame_length = frame_length
        self.stride_length = stride_length
        self.device = torch.device(device)

        # 第一个 CNN: 3x3 卷积，输出通道 32
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2)

        # 第二个 CNN: 1x1 卷积，输出通道 1
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.bn2 = nn.BatchNorm2d(1)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2) # 替换 Tanh 为 ReLU

        self._init_weights()
        self.to(self.device)

    def _init_weights(self):
        """Initialize CNN weights using Kaiming initialization for LeakyReLU"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, _input):
        """
        输入: _input (B, 16384)
        输出: x (B, 1, 8192, 2), batch_mean (B, 1), batch_var (B, 1)
        """
        _input = _input.to(self.device)

        # 预处理
        _input, batch_mean, batch_var = wav_norm(_input)
        _input = enframe(_input, self.num_frame, self.frame_length, self.stride_length)  # (B, 128, 128)
        x = _input.unsqueeze(1)  # (B, 1, 128, 128)

        # CNN
        x = self.conv1(x)  # (B, 32, 128, 128)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)  # (B, 1, 128, 128)
        x = self.bn2(x)
        x = self.relu2(x)  # (B, 1, 128, 128)

        # # IQ 通道重塑
        # B, C, N, T = x.shape
        # x = x.view(B, C, N * T)  # (B, 1, 16384)
        # x = x.view(B, C, N * T // 2, 2)  # (B, 1, 8192, 2)

        # print(f"Encoder output range: [{x.min().item():.4f}, {x.max().item():.4f}]")
        return x, batch_mean, batch_var

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(16, 16384, device=device)
    model = Encoder(frame_length=128, stride_length=128, num_frame=128, device=device).to(device)
    output, batch_mean, batch_var = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Batch mean shape: {batch_mean.shape}")
    print(f"Batch var shape: {batch_var.shape}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")