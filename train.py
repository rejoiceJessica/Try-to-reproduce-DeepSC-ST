import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.io import savemat
from net.Network import Network
from torch.utils.data import DataLoader, TensorDataset
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 系统配置
num_workers = os.cpu_count() or 4
logger.info(f"Number of CPU cores available: {num_workers}")

# CUDA 设置
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# 脚本目录
script_dir = os.path.dirname(os.path.abspath(__file__))

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="DeepSC-TS-2D Training")
    parser.add_argument("--sr", type=int, default=8000, help="Sample rate (Hz)")
    parser.add_argument("--num_frame", type=int, default=128, help="Number of frames")
    parser.add_argument("--frame_size", type=float, default=0.016, help="Frame duration (seconds)")
    parser.add_argument("--stride_size", type=float, default=0.016, help="Stride duration (seconds)")
    parser.add_argument("--trainset_pt_path", type=str, default=os.path.join(script_dir, "data", "trainset.pt"), help="Path to trainset.pt")
    parser.add_argument("--validset_pt_path", type=str, default=os.path.join(script_dir, "data", "validset.pt"), help="Path to validset.pt")
    parser.add_argument("--channel_types", type=str, default="awgn", help="Comma-separated channel types (e.g., awgn,rayleigh)")
    parser.add_argument("--multiple_snr", type=str, default="-5", help="Comma-separated SNR values in dB")
    parser.add_argument("--freq_offset", type=float, default=0.000, help="Frequency offset for Rayleigh channel")
    parser.add_argument("--num_epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    return parser.parse_args()

args = parse_args()
logger.info(f"Arguments: {vars(args)}")

# 计算帧长和步长
frame_length = int(args.sr * args.frame_size)  # 128
stride_length = int(args.sr * args.stride_size)  # 128
channel_types = args.channel_types.split(",")
multiple_snr = [float(snr) for snr in args.multiple_snr.split(",")]
logger.info(f"Training with channel types: {channel_types}")
logger.info(f"Training with SNR values: {multiple_snr}")

def load_pt_file(file_path):
    """加载 .pt 文件并归一化"""
    try:
        data = torch.load(file_path, weights_only=False)
        if not isinstance(data, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor, got {type(data)}")
        data = data.float() / 2**15  # 归一化到 [-0.5, 0.5]
        logger.info(f"Loaded {file_path}, shape: {data.shape}, range: [{data.min().item():.4f}, {data.max().item():.4f}]")
        return data
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        raise

def train():
    """训练 DeepSC-TS-2D 模型"""
    # 加载数据集
    train_data = load_pt_file(args.trainset_pt_path)
    valid_data = load_pt_file(args.validset_pt_path)

    train_dataset = TensorDataset(train_data)
    valid_dataset = TensorDataset(valid_data)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # 目录设置
    common_dir = os.path.join(script_dir, "train_data")
    saved_model_dir = os.path.join(common_dir, "saved_model")
    train_loss_dir = os.path.join(common_dir, "train")
    valid_loss_dir = os.path.join(common_dir, "valid")
    for dir_path in [saved_model_dir, train_loss_dir, valid_loss_dir]:
        os.makedirs(dir_path, exist_ok=True)

    class Config:
        device = device
        logger = logger
        pass_channel = True
        norm = True

    def train_step(_input, model, optimizer, snr_db):
        """单步训练"""
        model.train()
        _input = _input.to(device)
        assert _input.device.type == 'cuda', f"Input data not on GPU: {_input.device}"
        optimizer.zero_grad()
        try:
            recon_audio, chan_param, mse, loss = model(_input, given_SNR=snr_db, as_loss=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            return mse.item(), loss.item()
        except Exception as e:
            logger.error(f"Train step error: {e}")
            raise

    def valid_step(_input, model, snr_db):
        """单步验证"""
        model.eval()
        with torch.no_grad():
            _input = _input.to(device)
            assert _input.device.type == 'cuda', f"Input data not on GPU: {_input.device}"
            try:
                recon_audio, chan_param, mse, mse_val = model(_input, given_SNR=snr_db, as_loss=False)
                return mse_val.item()
            except Exception as e:
                logger.error(f"Valid step error: {e}")
                raise

    for channel_type in channel_types:
        for snr_db in multiple_snr:
            logger.info(f"*** Training: Channel={channel_type}, SNR={snr_db} dB ***")

            # 初始化模型
            model_args = argparse.Namespace(
                num_frame=args.num_frame,
                frame_length=frame_length,
                stride_length=stride_length,
                channel_type=channel_type,
                snr_db=snr_db,
                multiple_snr=args.multiple_snr,
                use_std=False,
                use_csi=False,
                freq_offset=args.freq_offset if channel_type == 'rayleigh' else 0.0,
                sample_rate=args.sr,
                device=device
            )
            model = Network(model_args, Config()).to(device)
            optimizer = optim.Adam(model.parameters(), lr=args.lr)

            logger.info(f"Model initialized: Channel={channel_type}, SNR={snr_db} dB")
            logger.info(f"Parameters: num_frame={args.num_frame}, frame_length={frame_length}, stride_length={stride_length}, freq_offset={model_args.freq_offset}")
            logger.info(f"Optimizer: Adam, lr={args.lr}")

            train_loss_all = []
            valid_loss_all = []
            best_valid_mse = float('inf')

            for epoch in range(args.num_epochs):
                train_loss_epoch = []
                train_mse = 0.0
                train_loss = 0.0
                start_time = time.time()

                # 训练
                for step, (_input,) in enumerate(train_loader):
                    mse_val, loss_val = train_step(_input, model, optimizer, snr_db)
                    train_loss_epoch.append(mse_val)
                    train_mse += mse_val
                    train_loss += loss_val

                train_mse /= (step + 1)
                train_loss /= (step + 1)
                train_loss_all.append(np.array(train_loss_epoch, dtype=np.float32))
                logger.info(
                    f"Channel={channel_type}, SNR={snr_db} dB, Epoch {epoch + 1}/{args.num_epochs}, "
                    f"Train MSE={train_mse:.6f}, Train Loss={train_loss:.6f}, Time={time.time() - start_time:.2f}s"
                )

                # 验证
                valid_loss_epoch = []
                valid_mse = 0.0
                start_time = time.time()

                for step, (_input,) in enumerate(valid_loader):
                    mse_val = valid_step(_input, model, snr_db)
                    valid_loss_epoch.append(mse_val)
                    valid_mse += mse_val

                valid_mse /= (step + 1)
                valid_loss_all.append(np.array(valid_loss_epoch, dtype=np.float32))
                logger.info(
                    f"Channel={channel_type}, SNR={snr_db} dB, Epoch {epoch + 1}/{args.num_epochs}, "
                    f"Valid MSE={valid_mse:.6f}, Time={time.time() - start_time:.2f}s"
                )

                # 保存最佳模型
                if valid_mse < best_valid_mse:
                    best_valid_mse = valid_mse
                    best_model_path = os.path.join(
                        saved_model_dir, f"snr_{int(snr_db)}_{channel_type}", f"best_model_snr_{int(snr_db)}_{channel_type}.pth"
                    )
                    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
                    torch.save(model.state_dict(), best_model_path)
                    logger.info(f"Saved best model to {best_model_path}")

                # 每 10 epoch 保存
                if (epoch + 1) % 10 == 0:
                    model_path = os.path.join(
                        saved_model_dir, f"snr_{int(snr_db)}_{channel_type}",
                        f"model_epoch_{epoch + 1}_{time.strftime('%Y%m%d_%H%M%S')}.pth"
                    )
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    torch.save(model.state_dict(), model_path)
                    logger.info(f"Saved model to {model_path}")

                    train_loss_file = os.path.join(train_loss_dir, f"train_mse_snr_{int(snr_db)}_{channel_type}.mat")
                    valid_loss_file = os.path.join(valid_loss_dir, f"valid_mse_snr_{int(snr_db)}_{channel_type}.mat")
                    savemat(train_loss_file, {"train_mse": np.array(train_loss_all, dtype=np.float32)})
                    savemat(valid_loss_file, {"valid_mse": np.array(valid_loss_all, dtype=np.float32)})
                    logger.info(f"Saved MSE to {train_loss_file} and {valid_loss_file}")

            # 保存最终模型
            final_model_path = os.path.join(
                saved_model_dir, f"snr_{int(snr_db)}_{channel_type}",
                f"final_model_snr_{int(snr_db)}_{channel_type}_{time.strftime('%Y%m%d_%H%M%S')}.pth"
            )
            os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
            torch.save(model.state_dict(), final_model_path)
            logger.info(f"Saved final model to {final_model_path}")

            # 保存最终 MSE
            train_loss_file = os.path.join(train_loss_dir, f"train_mse_snr_{int(snr_db)}_{channel_type}.mat")
            valid_loss_file = os.path.join(valid_loss_dir, f"valid_mse_snr_{int(snr_db)}_{channel_type}.mat")
            savemat(train_loss_file, {"train_mse": np.array(train_loss_all, dtype=np.float32)})
            savemat(valid_loss_file, {"valid_mse": np.array(valid_loss_all, dtype=np.float32)})
            logger.info(f"Saved final MSE to {train_loss_file} and {valid_loss_file}")

if __name__ == "__main__":
    train()