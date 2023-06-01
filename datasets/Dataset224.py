import os
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from convert_data.convert import all_channels_image


class Dataset224(Dataset):
    def __init__(self, fd: bool, data_dir: str, pids: List[str], start: int, end: int):
        super().__init__()
        self.data = []
        self.labels = []
        self.fd = fd

        for pid in tqdm(pids):
            self.add(data_dir, pid, start, end)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i], self.labels[i]

    def _load_data(self, data_dir: str, pid: str, start: int, end: int):
        images = np.load(os.path.join(data_dir, pid, "images.npy"), mmap_mode='c')[start:end]
        gazes = np.load(os.path.join(data_dir, pid, "gazes.npy"),  mmap_mode='c')[start:end]

        return images, gazes

    def add(self, data_dir: str, pid: str, start: int, end: int):
        images, gazes = self._load_data(data_dir, pid, start, end)

        if self.fd:
            transformed_images = [all_channels_image(img) for img in images]
            images = np.stack(transformed_images, axis=0)

        # Transpose to images to (batch_size, channels, height, width) and convert the data to tensors
        images = images.transpose((0, 3, 1, 2))
        tensor_images = torch.from_numpy(images).float()
        tensor_gazes = torch.from_numpy(gazes).float()

        self.data.extend(tensor_images)
        self.labels.extend(tensor_gazes)
