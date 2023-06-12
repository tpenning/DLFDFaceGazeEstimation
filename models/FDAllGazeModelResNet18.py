import torch.nn as nn
import torchvision.models as models
from models.GazeModel import GazeModel
from utils.device import get_device


class FDAllGazeModelResNet18(GazeModel):
    def __init__(self, model_name: str, input_channels: int, device=get_device()):
        # TODO: changed this class (from no input_channels, lower kernel size and additional layers)
        super().__init__(device)
        self.name = model_name
        self.input_channels = input_channels

        # Additional layers to gradually reduce the number of input channels
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(192, 128, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2))
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(128, 96, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2))

        # Get the ResNet18 model
        self.resnet = models.resnet18()

        # Change the input channels of the first convolutional layer in ResNet18
        self.resnet.conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # Change the output to 2 values
        self.resnet.fc = nn.Linear(512, 2)

        # Configure the device
        self.device = device

    def forward(self, image):
        image = image.to(self.device)

        # Additional layers
        # image = self.conv1(image)
        # image = self.conv2(image)

        # Run the image through the ResNet18 layers
        image = self.resnet(image)

        return image
