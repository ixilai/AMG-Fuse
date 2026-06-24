"""
Distributed training script for AMG-Fuse with WarmUp + CosineAnnealing LR scheduler.

Usage:
    Single GPU:
        python train.py --ir_path /path/to/ir --vi_path /path/to/vi ...

    Multi-GPU (Distributed):
        torchrun --nproc_per_node=4 train.py --ir_path /path/to/ir --vi_path /path/to/vi ...
"""
import os
import time
import datetime
import argparse
import math

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import numpy as np
import kornia
import matplotlib.pyplot as plt

from model.amgfuse import MSAKNet
from utils.loss import fusion_loss_vif
from dataloader.dataset import FusionDataset


class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Warm-up + Cosine Annealing scheduler (epoch-level).
    """
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=1e-6, last_epoch=-1):
        self.warmup_epochs = int(warmup_epochs)
        self.max_epochs = int(max_epochs)
        self.min_lr = float(min_lr)
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch + 1
        lrs = []
        for base_lr in self.base_lrs:
            if epoch <= self.warmup_epochs and self.warmup_epochs > 0:
                lr = base_lr * (epoch / float(self.warmup_epochs))
            else:
                cosine_epoch = max(0, epoch - self.warmup_epochs)
                cosine_total = max(1, self.max_epochs - self.warmup_epochs)
                cos_decay = 0.5 * (1 + math.cos(math.pi * cosine_epoch / float(cosine_total)))
                lr = self.min_lr + (base_lr - self.min_lr) * cos_decay
            lrs.append(lr)
        return lrs


def parse_args():
    parser = argparse.ArgumentParser(description="AMG-Fuse Training")

    # Dataset paths
    parser.add_argument('--ir_path', required=True, type=str, help='Path to degraded IR images')
    parser.add_argument('--vi_path', required=True, type=str, help='Path to degraded VI images')
    parser.add_argument('--gt_path', required=True, type=str, help='Path to clean VI images')
    parser.add_argument('--gt_ir_path', required=True, type=str, help='Path to clean IR images')
    parser.add_argument('--method_path', required=True, type=str, help='Path to pseudo GT from EMMA')

    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size per GPU')
    parser.add_argument('--patch_size', type=int, default=168, help='Training patch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--n_epochs', type=int, default=200, help='Total training epochs')
    parser.add_argument('--start_epoch', type=int, default=1, help='Start epoch')
    parser.add_argument('--clip_grad', type=float, default=1e-4, help='Gradient clipping threshold')

    # Scheduler
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Number of warmup epochs')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum LR after annealing')

    # Checkpoint
    parser.add_argument('--ckpt_path', type=str, default=None, help='Path to resume checkpoint')
    parser.add_argument('--restormer_ckpt', type=str, default=None,
                        help='Path to Restormer checkpoint for TDAS (e.g., derain/desnow/dehaze)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')

    # Dataloader
    parser.add_argument('--num_workers', type=int, default=8, help='Number of dataloader workers')

    return parser.parse_args()


def is_main_process():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def train_one_epoch(epoch, model, dataloader, optimizer, device, loss_fns, args):
    """Single epoch training loop."""
    model.train()
    running_losses = []

    if is_main_process():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.n_epochs}", ncols=100)
    else:
        pbar = dataloader

    for data_IR, data_VIS, data_GT, data_gt_ir, method in pbar:
        data_VIS = data_VIS.to(device, non_blocking=True)
        data_IR = data_IR.to(device, non_blocking=True)
        data_GT = data_GT.to(device, non_blocking=True)
        data_gt_ir = data_gt_ir.to(device, non_blocking=True)
        method = method.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Forward pass
        fea_vi, Final, fea_ir, res, resinput = model(data_VIS, data_IR)

        # Compute mask-guided features
        data_IR_expanded = data_gt_ir.expand_as(Final)
        eps = 1e-6
        denominator = data_GT - data_IR_expanded + method
        mask = (method - data_IR_expanded) / (denominator + eps)
        mask = torch.clamp(mask, 0, 1)
        mask = torch.sigmoid((mask - 0.5) * 5)
        vi_feat = mask * data_GT
        ir_feat = (1 - mask) * data_IR_expanded

        ir_feat_single_channel = ir_feat[:, 0, :, :]
        f_ir_single_channel = fea_ir[:, 0, :, :]

        l1_loss_fn = loss_fns['l1']

        # Mask-Guided Learning Loss (MGLS)
        Loss_con = 5 * (l1_loss_fn(vi_feat, fea_vi) + l1_loss_fn(ir_feat_single_channel, f_ir_single_channel))

        # Task-Coupled Degradation-Aware Loss (TDAS)
        Loss_res = 5 * l1_loss_fn(res, resinput)

        # Fusion loss
        fusion_loss_calc = fusion_loss_vif(device)
        loss__, loss_gradient, loss_l1, loss_SSIM, Loss_color = fusion_loss_calc(data_GT, data_gt_ir, Final)

        # Total loss
        loss = loss_SSIM + loss_l1 + loss_gradient + Loss_color + Loss_con + Loss_res

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad, norm_type=2)
        optimizer.step()

        running_losses.append(loss.item())

        if is_main_process():
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return float(np.mean(running_losses)) if len(running_losses) > 0 else 0.0


def save_checkpoint(model, epoch, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    state_dict = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
    ckpt = {'model': state_dict, 'epoch': epoch}
    path = os.path.join(save_dir, f'epoch_{epoch}.pth')
    torch.save(ckpt, path)
    return path


def main():
    args = parse_args()

    # Distributed setup
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0

    # Create model
    # Set use_restormer=True for training with TDAS (Task-Coupled Degradation-Aware Strategy)
    # Set use_restormer=False for inference or training without TDAS
    use_restormer = args.restormer_ckpt is not None
    model = MSAKNet(use_restormer=use_restormer).to(device)

    # Load Restormer checkpoint for TDAS (weather-specific: derain/desnow/dehaze)
    if use_restormer and args.restormer_ckpt:
        if is_main_process():
            print(f"Loading Restormer checkpoint from {args.restormer_ckpt}")
        restormer_checkpoint = torch.load(args.restormer_ckpt, map_location='cpu')
        model.res.load_state_dict(restormer_checkpoint["AWF"], strict=False)
        model.res.eval()
        for param in model.res.parameters():
            param.requires_grad = False
        if is_main_process():
            print("Restormer loaded successfully for TDAS")

    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # Load checkpoint
    if args.ckpt_path is not None:
        if is_main_process():
            print(f"Loading checkpoint from {args.ckpt_path}")
        state = torch.load(args.ckpt_path, map_location='cpu')
        model_state = model.module if isinstance(model, DDP) else model
        model_state.load_state_dict(state['model'], strict=False)

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = WarmupCosineAnnealingLR(optimizer, args.warmup_epochs, args.n_epochs, args.min_lr)

    # Loss functions
    l1_loss = nn.L1Loss().to(device)
    loss_fns = {'l1': l1_loss}

    # Dataset
    dataset = FusionDataset('train',
                           size=args.patch_size,
                           ir_path=args.ir_path,
                           vi_path=args.vi_path,
                           gt_path=args.gt_path,
                           gt_ir_path=args.gt_ir_path,
                           method_path=args.method_path)

    if dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=True)
    else:
        sampler = None

    dataloader = DataLoader(dataset,
                           batch_size=args.batch_size,
                           sampler=sampler,
                           shuffle=(sampler is None),
                           num_workers=args.num_workers,
                           pin_memory=True,
                           drop_last=False)

    # Training loop
    if is_main_process():
        print(f"Starting training for {args.n_epochs} epochs...")
        start_time = time.time()

    for epoch in range(args.start_epoch, args.n_epochs + 1):
        if dist.is_initialized():
            sampler.set_epoch(epoch)

        train_loss = train_one_epoch(epoch, model, dataloader, optimizer, device, loss_fns, args)
        scheduler.step()

        if is_main_process():
            lr_now = scheduler.get_last_lr()[0]
            elapsed = str(datetime.timedelta(seconds=int(time.time() - start_time)))
            print(f"Epoch [{epoch}/{args.n_epochs}] Loss: {train_loss:.6f} LR: {lr_now:.8f} Elapsed: {elapsed}")

            if epoch % 10 == 0 or epoch == args.n_epochs:
                ckpt_path = save_checkpoint(model, epoch, args.save_dir)
                print(f"Checkpoint saved: {ckpt_path}")

    if is_main_process():
        print("Training completed!")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
