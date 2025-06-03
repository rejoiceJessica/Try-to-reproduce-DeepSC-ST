
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from pesq import pesq
import torchaudio
import logging
from net.Network import Network

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
    parser = argparse.ArgumentParser(description="DeepSC-TS-2D Model Evaluation")
    parser.add_argument("--sr", type=int, default=8000, help="Sample rate (Hz)")
    parser.add_argument("--num_frame", type=int, default=128, help="Number of frames")
    parser.add_argument("--frame_size", type=float, default=0.016, help="Frame duration (seconds)")
    parser.add_argument("--stride_size", type=float, default=0.016, help="Stride duration (seconds)")
    parser.add_argument("--validset_pt_path", type=str, default=os.path.join(script_dir, "data", "validset.pt"), help="Path to validset.pt")
    parser.add_argument("--channel_type", type=str, default="awgn", choices=["awgn", "rayleigh"], help="Channel type")
    parser.add_argument("--snr_min", type=int, default=-15, help="Minimum SNR (dB)")
    parser.add_argument("--snr_max", type=int, default=10, help="Maximum SNR (dB)")
    parser.add_argument("--snr_step", type=int, default=1, help="SNR step size (dB)")
    parser.add_argument("--model_path", type=str, default=os.path.join(script_dir, "train_data", "saved_model","snr_-5_awgn","model_epoch_110_20250531_061420.pth"), help="Path to model .pth file (default: best model for snr=0)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--freq_offset", type=float, default=0.0000, help="Frequency offset for Rayleigh channel")
    return parser.parse_args()

args = parse_args()
logger.info(f"Arguments: {vars(args)}")

# 计算帧长和步长
frame_length = int(args.sr * args.frame_size)  # 128
stride_length = int(args.sr * args.stride_size)  # 128
snr_list = list(range(args.snr_min, args.snr_max + 1, args.snr_step))
logger.info(f"Evaluating SNR values: {snr_list}")

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

def compute_sdr(ref, est):
    """计算 Signal-to-Distortion Ratio (SDR)"""
    signal = torch.sum(ref ** 2, dim=-1)
    noise = torch.sum((ref - est) ** 2, dim=-1)
    sdr = 10 * torch.log10(signal / (noise + 1e-8))
    return sdr.mean().item()

def evaluate_model():
    """评估模型性能"""
    # 加载验证集
    valid_data = load_pt_file(args.validset_pt_path)
    valid_dataset = TensorDataset(valid_data)
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # 模型配置
    class Config:
        device = device
        logger = logger
        pass_channel = True
        norm = True

    model_args = argparse.Namespace(
        num_frame=args.num_frame,
        frame_length=frame_length,
        stride_length=stride_length,
        channel_type=args.channel_type,
        snr_db=0.0,  # 默认值，实际 SNR 在循环中设置
        multiple_snr=','.join(map(str, snr_list)),
        use_std=False,
        use_csi=False,
        freq_offset=args.freq_offset if args.channel_type == 'rayleigh' else 0.0,
        sample_rate=args.sr,
        device=device
    )

    # 初始化模型
    model = Network(model_args, Config()).to(device)

    # 加载模型参数
    model_path = args.model_path
    if model_path is None:
        # 默认加载 snr=0 的最佳模型
        model_path = os.path.join(script_dir, "train_data", "saved_model", f"snr_0_{args.channel_type}", f"best_model_snr_0_{args.channel_type}.pth")
    if not os.path.exists(model_path):
        logger.error(f"Model path {model_path} does not exist")
        raise FileNotFoundError(f"Model path {model_path} does not exist")

    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        logger.info(f"Loaded model parameters from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model parameters: {e}")
        raise

    model.eval()
    results = []

    # 评估每个 SNR
    for snr_db in snr_list:
        mae_total = 0.0
        pesq_total = 0.0
        sdr_total = 0.0
        num_samples = 0

        with torch.no_grad():
            for step, (_input,) in enumerate(valid_loader):
                _input = _input.to(device)
                batch_size = _input.shape[0]

                try:
                    recon_audio, chan_param, mse, mse_val = model(_input, given_SNR=snr_db, as_loss=False)

                    # 计算 MAE
                    mae = torch.mean(torch.abs(_input - recon_audio)** 2).item()
                    mae_total += mae * batch_size

                    # 计算 PESQ
                    pesq_batch = 0.0
                    for i in range(batch_size):
                        ref = _input[i].cpu().numpy() * 2**15  # 反归一化
                        est = recon_audio[i].cpu().numpy() * 2**15
                        try:
                            pesq_score = pesq(args.sr, ref, est, 'nb')
                            pesq_batch += pesq_score
                        except Exception as e:
                            logger.warning(f"PESQ computation failed for sample {i}: {e}")
                            pesq_batch += 1.0  # 默认值
                    pesq_batch /= batch_size
                    pesq_total += pesq_batch * batch_size

                    # 计算 SDR
                    sdr = compute_sdr(_input, recon_audio)
                    sdr_total += sdr * batch_size

                    num_samples += batch_size

                except Exception as e:
                    logger.error(f"Evaluation step error at SNR {snr_db} dB: {e}")
                    raise

        # 平均指标
        mae_avg = mae_total / num_samples
        pesq_avg = pesq_total / num_samples
        sdr_avg = sdr_total / num_samples

        logger.info(
            f"SNR={snr_db} dB, MSE={mae_avg:.6f}, PESQ={pesq_avg:.3f}, SDR={sdr_avg:.3f}, Samples={num_samples}"
        )

        results.append({
            'SNR_dB': snr_db,
            'MAE': mae_avg,
            'PESQ': pesq_avg,
            'SDR': sdr_avg
        })

    # 保存结果
    results_df = pd.DataFrame(results)
    output_dir = os.path.join(script_dir, "train_data", "eval")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"results_{args.channel_type}.csv")
    results_df.to_csv(output_path, index=False)
    logger.info(f"Saved evaluation results to {output_path}")

if __name__ == "__main__":
    evaluate_model()
