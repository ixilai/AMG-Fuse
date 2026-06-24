"""
Dataset loader for AMG-Fuse training and testing.
"""
import os
import glob
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from . import transforms as T


def prepare_data_path(dataset_path):
    """Get all image files from directory."""
    data = glob.glob(os.path.join(dataset_path, "*.bmp"))
    data.extend(glob.glob(os.path.join(dataset_path, "*.tif")))
    data.extend(glob.glob(os.path.join(dataset_path, "*.jpg")))
    data.extend(glob.glob(os.path.join(dataset_path, "*.png")))
    data.sort()
    filenames = [os.path.basename(f) for f in data]
    return data, filenames


class FusionDataset(Dataset):
    """
    Multi-modal image fusion dataset.

    Args:
        split: 'train' or 'val'
        size: Patch size for training
        ir_path: Path to degraded IR images
        vi_path: Path to degraded VI images
        gt_path: Path to clean VI images
        gt_ir_path: Path to clean IR images
        method_path: Path to pseudo ground truth from EMMA
    """
    def __init__(self, split, size, ir_path=None, vi_path=None,
                 gt_path=None, gt_ir_path=None, method_path=None):
        super(FusionDataset, self).__init__()

        if split == 'train':
            self.filepath_vis, self.filenames_vis = prepare_data_path(vi_path)
            self.filepath_ir, self.filenames_ir = prepare_data_path(ir_path)
            self.filepath_gt, self.filenames_gt = prepare_data_path(gt_path)
            self.filepath_gt_ir, self.filenames_gt_ir = prepare_data_path(gt_ir_path)
            self.filepath_gt_fuse, self.filenames_gt_fuse = prepare_data_path(method_path)

            self.split = split
            self.length = min(len(self.filenames_vis), len(self.filenames_ir),
                            len(self.filenames_gt), len(self.filenames_gt_ir),
                            len(self.filenames_gt_fuse))

            self.transform = T.Compose([
                T.RandomCrop(size),
                T.RandomHorizontalFlip(0.5),
                T.RandomVerticalFlip(0.5),
                T.ToTensor()
            ])

        elif split == 'val':
            self.filepath_vis, self.filenames_vis = prepare_data_path(vi_path)
            self.filepath_ir, self.filenames_ir = prepare_data_path(ir_path)
            self.split = split
            self.length = min(len(self.filenames_vis), len(self.filenames_ir))

    def __getitem__(self, index):
        if self.split == 'train':
            vis_path = self.filepath_vis[index]
            ir_path = self.filepath_ir[index]
            gt_path = self.filepath_gt[index]
            gt_ir_path = self.filepath_gt_ir[index]
            gt_fuse_path = self.filepath_gt_fuse[index]

            image_vis = Image.open(vis_path).convert(mode='RGB')
            image_gt = Image.open(gt_path).convert(mode='RGB')
            image_ir = Image.open(ir_path).convert(mode='L')
            image_gt_ir = Image.open(gt_ir_path).convert(mode='L')
            image_gt_fuse = Image.open(gt_fuse_path).convert(mode='RGB')

            image_vis, image_gt, image_ir, image_gt_ir, image_gt_fuse = \
                self.transform(image_vis, image_gt, image_ir, image_gt_ir, image_gt_fuse)

            return (
                torch.tensor(image_ir),
                torch.tensor(image_vis),
                torch.tensor(image_gt),
                torch.tensor(image_gt_ir),
                torch.tensor(image_gt_fuse)
            )

        elif self.split == 'val':
            vis_path = self.filepath_vis[index]
            ir_path = self.filepath_ir[index]

            image_vis = np.array(Image.open(vis_path))
            image_ir = np.array(Image.open(ir_path).convert('L'))

            image_vis = (
                np.asarray(Image.fromarray(image_vis), dtype=np.float32).transpose((2, 0, 1)) / 255.0
            )
            image_ir = np.asarray(Image.fromarray(image_ir), dtype=np.float32) / 255.0
            image_ir = np.expand_dims(image_ir, axis=0)

            name = self.filenames_vis[index]
            return torch.tensor(image_vis), torch.tensor(image_ir), name

    def __len__(self):
        return self.length
