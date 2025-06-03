import torch
import torch.nn as nn
import torch.nn.functional as F
from net.Modules import PatchEmbed, BasicLayer, deframe, wav_denorm

class KeepfeatModule(nn.Module):
    def __init__(self, in_channels=96, mid_channels=24, out_channels=96):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.act = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = self.act(x)
        return x

class UpsampleModule(nn.Module):
    def __init__(self, in_channels=96, out_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.act = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.norm = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv2(x)
        x = self.act(x)
        x = self.norm(x)
        return x

class LastModule(nn.Module):
    def __init__(self, in_channels=64, mid_channels=32, out_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.act = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        nn.init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        return x

class Decoder(nn.Module):
    def __init__(self, frame_length, num_frame, stride_length, patch_size=2, embed_dim=96, num_heads=8, window_size=8,  device='cpu'):
        super().__init__()
        self.frame_length = frame_length
        self.num_frame = num_frame
        self.stride_length = stride_length
        self.device = device

        self.shallow_conv = nn.Conv2d(1, embed_dim, kernel_size=3, stride=2, padding=1)
        self.shallow_act = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.patch_embed = PatchEmbed(img_size=(128, 128), patch_size=patch_size, in_chans=1, embed_dim=embed_dim)
        self.swin_blocks = BasicLayer(
            dim=embed_dim, input_resolution=(64, 64),  num_heads=num_heads, window_size=window_size)
        self.keepfeat = KeepfeatModule()
        self.upsample = UpsampleModule()
        self.last = LastModule()

        self.to(device)

    def forward(self, x, batch_mean=None, batch_var=None):
        shallow = self.shallow_conv(x)  # (B, 96, 64, 64)
        shallow = self.shallow_act(shallow)
        # print(f"Shallow branch output range: [{shallow.min().item():.4f}, {shallow.max().item():.4f}]")

        deep = self.patch_embed(x)  # (B, 4096, 96)
        deep = self.swin_blocks(deep)  # (B, 4096, 96)
        deep = deep.view(deep.shape[0], 64, 64, -1).permute(0, 3, 1, 2)  # (B, 96, 64, 64)
        deep = self.keepfeat(deep)  # (B, 96, 64, 64)
        # print(f"Deep branch output range: [{deep.min().item():.4f}, {deep.max().item():.4f}]")

        x = shallow + deep  # (B, 96, 64, 64)
        # print(f"Residual output range: [{x.min().item():.4f}, {x.max().item():.4f}]")

        x = self.upsample(x)  # (B, 64, 128, 128)
        # print(f"Upsample output range: [{x.min().item():.4f}, {x.max().item():.4f}]")

        x = self.last(x)  # (B, 1, 128, 128)
        # print(f"LastModule output range: [{x.min().item():.4f}, {x.max().item():.4f}]")

        x = x.squeeze(1)  # (B, 128, 128)
        x = deframe(x, self.num_frame, self.frame_length, self.stride_length)  # (B, 16384)
        x = wav_denorm(x, batch_mean, batch_var)  # (B, 16384)
        # print(f"Final output range: [{x.min().item():.4f}, {x.max().item():.4f}]")
        return x

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.rand(8, 1, 128, 128, device=device) * 2 - 1  # [-1, 1]
    batch_mean = torch.zeros(8, 1, device=device)
    batch_var = torch.ones(8, 1, device=device)
    model = Decoder(
        frame_length=128,
        num_frame=128,
        stride_length=128,
        device=device
    ).to(device)
    output = model(x, batch_mean, batch_var)
    print(f"Input shape: {x.shape}")
    print(f"Input range: [{x.min().item():.4f}, {x.max().item():.4f}]")
    print(f"Output shape: {output.shape}")