import torch.nn as nn
import torchvision.models as models
from models.GazeModel import GazeModel
from utils.device import get_device


class FDGazeModelResNet18(GazeModel):
    def __init__(self, model_name: str, input_channels: int, lc_hc: str, device=get_device()):
        super().__init__(device)
        self.name = model_name
        self.device = device

        # Set the channels for input and low channel of high channel version
        self.input_channels = input_channels
        self.channels = [64, 128, 256, 512]
        if lc_hc == "HC":
            self.channels = [channel * 2 for channel in self.channels]

        # Get the ResNet18 model
        self.resnet = models.resnet18()

        # Change the input channels, kernel size and stride of the first convolutional layer
        self.resnet.conv1 = nn.Conv2d(self.input_channels, self.channels[0], kernel_size=2, stride=1, padding=1, bias=False)

        # Configure the channels passed through the layers
        self.resnet.layer1.conv1 = nn.Conv2d(self.channels[0], self.channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.layer1.conv2 = nn.Conv2d(self.channels[0], self.channels[0], kernel_size=3, stride=1, padding=1, bias=False)

        self.resnet.layer2.conv1 = nn.Conv2d(self.channels[0], self.channels[1], kernel_size=3, stride=2, padding=1, bias=False)
        self.resnet.layer2.conv2 = nn.Conv2d(self.channels[1], self.channels[1], kernel_size=3, stride=1, padding=1, bias=False)

        self.resnet.layer3.conv1 = nn.Conv2d(self.channels[1], self.channels[2], kernel_size=3, stride=2, padding=1, bias=False)
        self.resnet.layer3.conv2 = nn.Conv2d(self.channels[2], self.channels[2], kernel_size=3, stride=1, padding=1, bias=False)

        self.resnet.layer4.conv1 = nn.Conv2d(self.channels[2], self.channels[3], kernel_size=3, stride=2, padding=1, bias=False)
        self.resnet.layer4.conv2 = nn.Conv2d(self.channels[3], self.channels[3], kernel_size=3, stride=1, padding=1, bias=False)

        # Change the output to 2 values
        self.resnet.fc = nn.Linear(self.channels[3], 2)

    def forward(self, image):
        image = image.to(self.device)

        # Run the image through the ResNet18 layers
        image = self.resnet(image)

        return image
