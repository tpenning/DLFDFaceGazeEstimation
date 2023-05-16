import torch.nn as nn
import torchvision.models as models
from models.GazeModel import GazeModel
from utils.device import get_device


class RGBGazeModelResNet18(GazeModel):
    def __init__(self, device=get_device()):
        super().__init__(device)
        self.name = "RGBGazeModelResNet18.pt"

        # Get the resnet model
        self.resnet = models.resnet18()
        # Change the output to 2 values
        self.resnet.fc = nn.Linear(512, 2)

        # Configure the device
        self.device = device

    def forward(self, image):
        image = image.to(self.device)

        # Run the image through the network
        image = self.resnet(image)

        return image
