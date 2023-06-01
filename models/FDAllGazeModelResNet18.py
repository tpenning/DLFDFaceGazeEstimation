import torch.nn as nn
import torchvision.models as models
from models.GazeModel import GazeModel
from utils.device import get_device


class FDAllGazeModelResNet18(GazeModel):
    def __init__(self, model_name: str, device=get_device()):
        super().__init__(device)
        self.name = model_name

        # Get the resnet model
        self.resnet = models.resnet18()
        # Change the first convolutional layer to accept (28, 28, 192) input
        self.resnet.conv1 = nn.Conv2d(192, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Change the output to 2 values
        self.resnet.fc = nn.Linear(512, 2)

        # Configure the device
        self.device = device

    def forward(self, image):
        image = image.to(self.device)

        # Run the image through the network
        image = self.resnet(image)

        return image
