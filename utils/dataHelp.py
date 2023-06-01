from torch.utils.data import random_split


def split_data(dataset, ratio: float):
    n = len(dataset)
    part = int(n * ratio)

    return random_split(dataset, [part, n - part])
