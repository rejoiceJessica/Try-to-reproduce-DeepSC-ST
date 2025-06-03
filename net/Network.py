
import torch
import torch.nn as nn
from random import choice
import logging
from net.Encoder import Encoder
from net.Channel import Channel
from net.Decoder import Decoder

class Network(nn.Module):
    def __init__(self, args=None, config=None):
        """
        初始化 DeepSC-TS 网络，包含编码器、信道和解码器。

        参数:
            args: 包含网络参数的对象（num_frame, frame_length, stride_length, channel_type, multiple_snr 等）。
            config: 包含设备和日志配置的对象（device, logger, pass_channel）。
        """
        super().__init__()
        self.config = config
        self.args = args or self._default_args()
        self.num_frame = self.args.num_frame
        self.frame_length = self.args.frame_length
        self.stride_length = self.args.stride_length

        self.encoder = Encoder(
            frame_length=self.frame_length,
            num_frame=self.num_frame,
            stride_length=self.stride_length,
            device=self.args.device
        )
        self.channel = Channel(
            channel_type=self.args.channel_type,
            snr_db=self.args.snr_db,
            freq_offset=self.args.freq_offset,
            use_csi=self.args.use_csi,
            use_std=self.args.use_std,
            device=self.args.device
        )
        self.decoder = Decoder(
            frame_length=self.frame_length,
            num_frame=self.num_frame,
            stride_length=self.stride_length,
            patch_size=2,
            embed_dim=96,  # 统一为 96，匹配 PatchEmbed 和 SwinBlock
            num_heads=8,
            window_size=8,  # 统一为 8，适配 input_resolution=(64, 64)

            device=self.args.device
        )

        self.distortion_loss = nn.MSELoss()
        self.squared_difference = nn.MSELoss(reduction='none')
        self.pass_channel = getattr(config, 'pass_channel', True) if config else True
        self.multiple_snr = [float(snr) for snr in self.args.multiple_snr.split(",")]

        # 日志配置
        self.logger = getattr(config, 'logger', None) if config else logging.getLogger(__name__)
        if self.logger:
            self.logger.info("网络配置：")
            self.logger.info(f"编码器：num_frame={self.num_frame}, frame_length={self.frame_length}, stride_length={self.stride_length}")
            self.logger.info(f"解码器：num_frame={self.num_frame}, frame_length={self.frame_length}, stride_length={self.stride_length}, num_swin_blocks=24")
            self.logger.info(f"信道：type={self.args.channel_type}, SNR={self.multiple_snr}, freq_offset={self.args.freq_offset}")
            self.logger.info(f"失真损失：MSE, 采样率={self.args.sample_rate}")

        self.to(self.args.device)

    def _default_args(self):
        """默认参数配置"""
        class Args:
            num_frame = 128
            frame_length = 128
            stride_length = 128
            channel_type = 'awgn'
            snr_db = 16.0
            freq_offset = 0.0050
            use_csi = False
            use_std = False
            multiple_snr = '-15,-10,-5,0,5'
            sample_rate = 8000
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return Args()

    def feature_pass_channel(self, feature, chan_param):
        """
        通过信道处理编码特征。

        参数:
            feature: 编码器输出特征，形状 [B, 1, 128, 128]。
            chan_param: 信道参数（SNR 或 std）。

        返回:
            noisy_feature: 信道输出特征，形状 [B, 1, 128, 128]。
        """
        noisy_feature = self.channel(feature, chan_param=chan_param)
        return noisy_feature

    def forward(self, input_audio, given_SNR=None, as_loss=True):
        """
        前向传播，处理输入音频到重建音频。

        参数:
            input_audio: 输入音频，形状 [B, 16384]。
            given_SNR: 指定 SNR（dB），若为 None 则随机选择。
            as_loss: 是否返回损失（True）或 MSE 值（False）。

        返回:
            recon_audio: 重建音频，形状 [B, 16384]。
            chan_param: 使用的信道参数（SNR 或 std）。
            mse: 均方误差均值。
            loss 或 mse_val: 失真损失（as_loss=True）或 MSE 值（as_loss=False）。
        """
        B, W = input_audio.shape
        device = input_audio.device
        assert input_audio.device.type == 'cuda', f"输入音频应为GPU，实际为{device}"
        assert W == 16384, f"输入音频长度应为16384，实际为{W}"

        if given_SNR is None:
            chan_param = choice(self.multiple_snr)
        else:
            chan_param = given_SNR

        # 编码器
        feature, batch_mean, batch_var = self.encoder(input_audio)
        assert feature.device.type == 'cuda', f"编码器输出设备不一致: {feature.device}"
        assert batch_mean.device.type == 'cuda', f"batch_mean 设备不一致: {batch_mean.device}"
        assert batch_var.device.type == 'cuda', f"batch_var 设备不一致: {batch_var.device}"
        assert feature.shape == (B, 1, 128, 128), f"编码器输出形状错误: {feature.shape}"
        assert batch_mean.shape == (B, 1), f"batch_mean 形状错误: {batch_mean.shape}"
        assert batch_var.shape == (B, 1), f"batch_var 形状错误: {batch_var.shape}"

        # 日志
        if self.logger:
            feature_power = torch.mean(feature ** 2).item()
            self.logger.debug(
                f"Encoder output: feature_shape={feature.shape}, "
                f"batch_mean_shape={batch_mean.shape}, batch_var_shape={batch_var.shape}, "
                f"feature_power={feature_power:.6f}, "
                f"feature_range=[{feature.min().item():.4f}, {feature.max().item():.4f}]"
            )

        # 信道
        if self.pass_channel:
            noisy_feature = self.feature_pass_channel(feature, chan_param)
            assert noisy_feature.device.type == 'cuda', f"信道输出设备不一致: {noisy_feature.device}"
            assert noisy_feature.shape == feature.shape, f"信道输出形状错误: {noisy_feature.shape}"
            if self.logger:
                noisy_power = torch.mean(noisy_feature ** 2).item()
                self.logger.debug(
                    f"Channel output: noisy_feature_shape={noisy_feature.shape}, "
                    f"noisy_power={noisy_power:.6f}, "
                    f"noisy_range=[{noisy_feature.min().item():.4f}, {noisy_feature.max().item():.4f}]"
                )
        else:
            noisy_feature = feature

        # 解码器
        recon_audio = self.decoder(noisy_feature, batch_mean, batch_var)
        assert recon_audio.device.type == 'cuda', f"解码器输出设备不一致: {recon_audio.device}"
        assert recon_audio.shape == (B, 16384), f"重建音频形状错误: {recon_audio.shape}"

        # 日志
        if self.logger:
            recon_power = torch.mean(recon_audio ** 2).item()
            self.logger.debug(
                f"Decoder output: recon_audio_shape={recon_audio.shape}, "
                f"recon_power={recon_power:.6f}, "
                f"recon_range=[{recon_audio.min().item():.4f}, {recon_audio.max().item():.4f}]"
            )

        # MSE
        mse = self.squared_difference(input_audio, recon_audio)

        # 梯度裁剪
        if self.training:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        # 返回
        if as_loss:
            loss = self.distortion_loss(input_audio, recon_audio)
            return recon_audio, chan_param, mse.mean(), loss
        else:
            mse_val = self.distortion_loss(input_audio, recon_audio)
            return recon_audio, chan_param, mse.mean(), mse_val

def test_network():
    """
    测试 DeepSC-TS 网络，验证 Encoder、Channel、Decoder 的功能，检查形状、范围。
    """
    import torch
    import logging

    # 日志配置
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # 参数配置
    class Args:
        def __init__(self, device):
            self.num_frame = 128
            self.frame_length = 128
            self.stride_length = 128
            self.channel_type = 'awgn'
            self.snr_db = 16.0
            self.freq_offset = 0.0050
            self.use_csi = False
            self.use_std = False
            self.multiple_snr = '-15,-10,-5,0,5'
            self.sample_rate = 8000
            self.device = device

    class Config:
        def __init__(self, device, logger):
            self.device = device
            self.logger = logger
            self.pass_channel = True
            self.norm = True

    args = Args(device)
    config = Config(device, logger)

    # 初始化模型
    model = Network(args, config).to(device)
    logger.info("Network 初始化完成")

    # 加载数据集
    try:
        data = torch.load('/home/robot/Github_Code/DeepSC-TS-2D/data/validset.pt', weights_only=False) / 2**15
        batch_size = 4
        data = data[:batch_size].to(device)
        logger.info(f"Dataset stats: Mean={data.mean().item():.4f}, Var={data.var().item():.4f}")
        logger.info(f"Dataset range: [{data.min().item():.4f}, {data.max().item():.4f}]")
    except Exception as e:
        logger.warning(f"加载数据集失败: {e}, 使用随机数据")
        batch_size = 16
        data = torch.rand(batch_size, 16384, device=device) * 0.9839 - 0.4790  # 模拟 [-0.4790, 0.5049]

    # 输入
    x = data
    snr_list = [float(snr) for snr in args.multiple_snr.split(",")]

    # 训练模式
    model.train()
    logger.info("\n训练模式：")
    for snr in snr_list:
        recon_audio, chan_param, mse, loss = model(x, snr, as_loss=True)
        logger.info(f"SNR: {chan_param} dB")
        logger.info(f"输入范围: [{x.min().item():.4f}, {x.max().item():.4f}]")
        logger.info(f"重建音频形状: {recon_audio.shape}")
        logger.info(f"重建范围: [{recon_audio.min().item():.4f}, {recon_audio.max().item():.4f}]")
        logger.info(f"MSE: {mse.item():.4f}")

    # 评估模式
    model.eval()
    logger.info("\n评估模式：")
    with torch.no_grad():
        for snr in snr_list:
            recon_audio, chan_param, mse, mse_val = model(x, snr, as_loss=False)
            logger.info(f"SNR: {chan_param} dB")
            logger.info(f"输入范围: [{x.min().item():.4f}, {x.max().item():.4f}]")
            logger.info(f"重建音频形状: {recon_audio.shape}")
            logger.info(f"重建范围: [{recon_audio.min().item():.4f}, {recon_audio.max().item():.4f}]")
            logger.info(f"MSE: {mse.item():.4f}")

if __name__ == '__main__':
    test_network()
