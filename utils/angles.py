import math
import torch


def convert_angle(py: torch.Tensor) -> torch.Tensor:
    pitch, yaw = py[:, 0], py[:, 1]
    x = -torch.cos(pitch) * torch.sin(yaw)
    y = -torch.sin(pitch)
    z = -torch.cos(pitch) * torch.cos(yaw)

    angle = torch.stack((x, y, z), dim=1)
    norm_angle = torch.nn.functional.normalize(angle, dim=1)
    return norm_angle


def angular_loss(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    cos_angles = torch.sum(v1 * v2, dim=1)
    angles = torch.acos(cos_angles) * 180.0 / torch.tensor(math.pi)
    avg_angle = torch.mean(angles)
    return avg_angle
