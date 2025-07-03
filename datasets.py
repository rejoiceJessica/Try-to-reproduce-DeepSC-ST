
# make_ptrecords.py
import os
import random
import time
import argparse
import numpy as np
import torch
from scipy.io import wavfile

def parse_args():
    parser = argparse.ArgumentParser(description=".wav to PyTorch  .pt ")

    parser.add_argument("--sr", type=int, default=8000)
    parser.add_argument("--num_frame", type=int, default=128)
    parser.add_argument("--frame_size", type=float, default=0.016)
    parser.add_argument("--stride_size", type=float, default=0.016)

    parser.add_argument("--wav_path", type=str, default="/home/robot/reload_trainset")
    parser.add_argument("--save_path", type=str, default="/home/robot/data")

    parser.add_argument("--valid_percent", type=float, default=0.05)
    parser.add_argument("--trainset_filename", type=str, default="trainset.pt")
    parser.add_argument("--validset_filename", type=str, default="validset.pt")


    return parser.parse_args()

def slice_wav(wav_samples, window_size, threshold=0.015):
    slices = []
    num_samples = len(wav_samples)

    if num_samples < window_size:
        wav_samples = np.tile(wav_samples, (window_size // num_samples + 1))[:window_size]
        if np.mean(np.abs(wav_samples) / 2**15) >= threshold:
            slices.append(wav_samples.astype(np.int16))
    else:
        # pad to fit multiple of window size
        num_slices = (num_samples + window_size - 1) // window_size
        padded = np.tile(wav_samples, 2)[:num_slices * window_size]
        reshaped = padded.reshape(num_slices, window_size)
        for chunk in reshaped:
            if np.mean(np.abs(chunk) / 2**15) >= threshold:
                slices.append(chunk.astype(np.int16))
    return slices

def process_wavs(wav_files, window_size, threshold):
    all_slices = []
    for idx, wav_path in enumerate(wav_files):
        sr, samples = wavfile.read(wav_path)
        if sr != 8000 or samples.ndim != 1:
            continue
        slices = slice_wav(samples, window_size, threshold)
        all_slices.extend(slices)
        print(f" {idx+1}/{len(wav_files)}: {wav_path}", end="\r")
    return all_slices

def save_pt_dataset(data, filepath, batch_size):
    # Padding to multiple of batch_size
    while len(data) % batch_size != 0:
        data.append(random.choice(data))
    total = len(data)
    tensor_data = torch.tensor(np.array(data), dtype=torch.int16)
    torch.save(tensor_data, filepath)


def main():
    args = parse_args()
    os.makedirs(args.save_path, exist_ok=True)

    frame_len = int(args.sr * args.frame_size)
    stride_len = int(args.sr * args.stride_size)
    window_size = args.num_frame * stride_len + frame_len - stride_len
    batch_size = 32

    wav_files = [os.path.join(args.wav_path, f) for f in os.listdir(args.wav_path) if f.endswith('.wav')]
    random.shuffle(wav_files)

    split = int(len(wav_files) * (1 - args.valid_percent))
    train_files, valid_files = wav_files[:split], wav_files[split:]


    train_slices = process_wavs(train_files, window_size, threshold=0.015)
    save_pt_dataset(train_slices, os.path.join(args.save_path, args.trainset_filename), batch_size)


    valid_slices = process_wavs(valid_files, window_size, threshold=0.015)
    save_pt_dataset(valid_slices, os.path.join(args.save_path, args.validset_filename), batch_size)

    print(f" use {round(time.time(), 2)} s。")

if __name__ == "__main__":
    main()
