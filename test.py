"""
Test script for AMG-Fuse model.

Usage:
    python test.py \
        --ir_path /path/to/test/ir \
        --vi_path /path/to/test/vi \
        --save_path /path/to/results \
        --ckpt /path/to/checkpoint.pth
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import glob
import numpy as np
from tqdm import tqdm
from skimage import img_as_ubyte

from model.amgfuse import MSAKNet


def parse_args():
    parser = argparse.ArgumentParser(description='AMG-Fuse Testing')
    parser.add_argument('--ir_path', required=True, type=str, help='Path to IR images')
    parser.add_argument('--vi_path', required=True, type=str, help='Path to VI images')
    parser.add_argument('--save_path', required=True, type=str, help='Path to save fused results')
    parser.add_argument('--ckpt', required=True, type=str, help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load model
    print(f"Loading model from {args.ckpt}")
    model = MSAKNet(use_restormer=False).to(device)
    model = nn.DataParallel(model)
    checkpoint = torch.load(args.ckpt, map_location=device)

    # Handle different checkpoint formats
    if 'AWF' in checkpoint:
        model.load_state_dict(checkpoint['AWF'], strict=False)
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.eval()

    # Get image paths
    ir_paths = sorted(glob.glob(os.path.join(args.ir_path, '*')))
    vi_paths = sorted(glob.glob(os.path.join(args.vi_path, '*')))

    if len(ir_paths) == 0 or len(vi_paths) == 0:
        print("No images found!")
        return

    print(f"Found {len(vi_paths)} VI images and {len(ir_paths)} IR images. Starting fusion...")

    with torch.no_grad():
        for path_vi, path_ir in zip(tqdm(vi_paths), ir_paths):
            img_multiple_of = 8

            # Read VI image
            img_vi = cv2.imread(path_vi, cv2.IMREAD_COLOR)
            img_vi = cv2.cvtColor(img_vi, cv2.COLOR_BGR2RGB)
            if img_vi is not None and len(img_vi.shape) == 3:
                img_vi = torch.from_numpy(img_vi).float().div(255.).permute(2, 0, 1).unsqueeze(0).to(device)
            else:
                raise ValueError(f"Error processing img_vi from path: {path_vi}")

            # Read IR image
            img_ir = cv2.imread(path_ir, cv2.IMREAD_GRAYSCALE)
            if img_ir is not None and len(img_ir.shape) == 2:
                img_ir = torch.from_numpy(img_ir).float().div(255.).unsqueeze(0).unsqueeze(0).to(device)
            else:
                raise ValueError(f"Error processing img_ir from path: {path_ir}")

            # Get original size
            _, _, height, width = img_vi.shape

            # Pad to multiple of 8
            H = (height + img_multiple_of - 1) // img_multiple_of * img_multiple_of
            W = (width + img_multiple_of - 1) // img_multiple_of * img_multiple_of
            padh = H - height if height % img_multiple_of != 0 else 0
            padw = W - width if width % img_multiple_of != 0 else 0

            img_vi = F.pad(img_vi, (0, padw, 0, padh), 'reflect')
            img_ir = F.pad(img_ir, (0, padw, 0, padh), 'reflect')

            # Forward pass
            VI_Fea, out, IR_Fea, _, out_enc_level1_vi = model(img_vi, img_ir)

            restored = torch.clamp(out, 0, 1)

            # Remove padding
            restored = restored[:, :, :height, :width]
            restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
            restored = img_as_ubyte(restored[0])

            # Save output
            save_path = path_vi.replace(str(args.vi_path), str(args.save_path))
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, cv2.cvtColor(restored, cv2.COLOR_RGB2BGR))

    print(f"Fusion completed! Results saved to {args.save_path}")


if __name__ == "__main__":
    main()
