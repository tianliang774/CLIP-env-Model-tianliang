import math
import pickle
import shutil
import os
import sys
import tempfile
import threading
import time
import warnings
from copy import copy, deepcopy
from torch.utils.data import Dataset

import numpy as np
import torch
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

from torchvision.transforms import transforms, CenterCrop


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            path = os.path.join(root, fname)
            images.append(path)
    return images[:min(max_dataset_size, len(images))]


SIZE = {"ring": 512, "low2high": 512, "ct2mri": 256}


class ImageABDataset(Dataset):
    """
        The Dataset may have some subdatasets and each subdataset contain /A and /B
        The ideal path example: {project_name}/{task_name}/train/{"A" | "B"}/xxx.npy
    """

    def __init__(self, args):
        self.args = args
        self.root = args.data_root_path
        self.subsets = os.listdir(self.root)
        self.A_paths = []
        self.B_paths = []
        for sub in self.subsets:
            tdir = os.path.join(self.root, str(sub))
            tdir = os.path.join(tdir, args.phase)
            self.A_paths += sorted(make_dataset(os.path.join(tdir, "A")))
            self.B_paths += sorted(make_dataset(os.path.join(tdir, "B")))

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]
        name = A_path.split(os.sep)[-4]
        if A_path[-4:] != ".npy" and name in ["ring", "low2high", "ct2mri"]:
            size = SIZE[name]
            A = np.fromfile(A_path, dtype='float32').reshape(size, size)
            B = np.fromfile(B_path, dtype='float32').reshape(size, size)
        else:
            A = np.load(A_path)
            B = np.load(B_path)
        ratio = B.max() - B.min()
        m = B.mean()

        A = (A - A.mean()) / (A.max() - A.min())
        B = (B - B.mean()) / (B.max() - B.min())
        A = transforms.ToTensor()(A)
        B = transforms.ToTensor()(B)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'M': m, 'R': ratio, 'name': name}

    def __len__(self):
        return len(self.A_paths)


class InferDataset(Dataset):
    """
        The Dataset may have some subdatasets and each subdataset contain /A and /B
        The ideal path example: {project_name}/{task_name}/infer/xxx.npy
    """

    def __init__(self, args):
        self.args = args
        self.root = args.data_root_path
        self.subsets = os.listdir(self.root)
        self.A_paths = []
        for sub in self.subsets:
            tdir = os.path.join(self.root, str(sub))
            tdir = os.path.join(tdir, args.phase)
            self.A_paths += sorted(make_dataset(tdir))

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        name = A_path.split(os.sep)[-3]
        if name in ["ring", "low2high", "ct2mri"]:
            size = SIZE[name]
            A = np.fromfile(A_path, dtype='float32').reshape(size, size)
        else:
            A = np.load(A_path)

        ratio = A.max() - A.min()
        m = A.mean()

        A = (A - A.mean()) / (A.max() - A.min())
        A = transforms.ToTensor()(A)

        return {'A': A, 'A_paths': A_path, 'M': m, 'R': ratio, 'name': name}

    def __len__(self):
        return len(self.A_paths)


if __name__ == "__main__":
    class Args:
        def __init__(self):
            self.data_root_path = "testProject"
            self.phase = "train"
            self.device = "cuda"


    args = Args()
    dataset = ImageABDataset(args)

    for i, batch in enumerate(dataset):
        x, y, name = batch["A"].to(args.device), batch["B"].float().to(args.device), batch['name']
        target_project_name = "dinotestProject"
        size = x.shape[-1]
        if size == 256:
            res_size = 2
        elif size == 512:
            res_size = 4
        else:
            res_size = 0

        Aoutput = x[0][res_size:-res_size, res_size:-res_size].cpu().numpy()
        Boutput = y[0][res_size:-res_size, res_size:-res_size].cpu().numpy()
        Atarget = rf"{target_project_name}/{name}/{args.phase}/A"
        Btarget = rf"{target_project_name}/{name}/{args.phase}/B"

        if not os.path.exists(Atarget):
            os.makedirs(Atarget)
        if not os.path.exists(Btarget):
            os.makedirs(Btarget)
        np.save(os.path.join(Atarget, f"{i}.npy"), Aoutput)
        np.save(os.path.join(Btarget, f"{i}.npy"), Boutput)
        if i % 100 == 0:
            print(f"{i} of {len(dataset)} has been processed!")
            print(Boutput.shape)