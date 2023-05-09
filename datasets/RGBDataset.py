import os
from typing import List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


class RGBDataset(Dataset):
    def __init__(self, data_dir: str, pids: List[str], start: int, end: int):
        super().__init__()
        self.data = []
        self.labels = []

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        for pid in tqdm(pids):
            self.add(data_dir, pid, start, end)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i], self.labels[i]

    def add(self, data_dir: str, pid: str, start: int, end: int):
        images1 = np.load(os.path.join(data_dir, pid, "images.npy"))[start:end]
        images = [self.transform(Image.fromarray(img)) for img in images1]
        images = torch.stack(images, dim=0)
        gazes = np.array(np.load(os.path.join(data_dir, pid, "gazes.npy"))[start:end])
        gazes = torch.Tensor(gazes).float()

        if len(self.data) == 0:
            self.data = images
            self.labels = gazes
        else:
            self.data = torch.cat([self.data, images], dim=0)
            self.labels = torch.cat([self.labels, gazes], dim=0)
