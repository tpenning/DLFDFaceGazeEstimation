import torch.nn as nn
import torchvision.models as models
from models.GazeModel import GazeModel
from utils.device import get_device


class FDCSGazeModelResNet18(GazeModel):
    def __init__(self, model_name: str, input_channels: int, device=get_device()):
        super().__init__(device)
        self.name = model_name
        self.input_channels = input_channels

        # Get the ResNet18 model
        self.resnet = models.resnet18(pretrained=False)

        # Change the input channels of the first convolutional layer
        self.resnet.conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Change the output to 2 values
        self.resnet.fc = nn.Linear(512, 2)

        # Configure the device
        self.device = device

    def forward(self, image):
        image = image.to(self.device)

        # Run the image through the ResNet18 layers
        image = self.resnet(image)

        return image
