import torch

def get_device():
    # Use cuda gpu if available, else use cpu
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")