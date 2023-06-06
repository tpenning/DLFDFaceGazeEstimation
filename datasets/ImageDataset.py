import os
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from converting.select_channels import select_channels


class ImageDataset(Dataset):
    def __init__(self, data_type: str, data_dir: str, pids: List[str], start: int, end: int):
        super().__init__()
        self.data = []
        self.labels = []
        self.data_type = data_type

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

        if self.data_type != "RGB":
            images = np.array(images)
            transformed_images = [select_channels(img, self.data_type) for img in images]
            images = np.stack(transformed_images, axis=0)

        # Transpose to images to (batch_size, channels, height, width) and convert the data to tensors
        images = images.transpose((0, 3, 1, 2))
        tensor_images = torch.from_numpy(images).float()
        tensor_gazes = torch.from_numpy(gazes).float()

        self.data.extend(tensor_images)
        self.labels.extend(tensor_gazes)
