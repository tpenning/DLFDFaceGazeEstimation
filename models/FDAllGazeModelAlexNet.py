import torch.nn as nn
from models.GazeModel import GazeModel
from utils.device import get_device


class FDAllGazeModelAlexNet(GazeModel):
    def __init__(self, model_name: str, device=get_device()):
        super().__init__(device)
        self.name = model_name

        # Convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # Fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(2304, 4096),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.fc2 = nn.Sequential(
            nn.Linear(4096, 2))

        # Configure the device
        self.device = device

    def forward(self, image):
        image = image.to(self.device)

        # Convolutional layers
        image = self.conv1(image)
        image = self.conv2(image)
        image = self.conv3(image)
        image = self.conv4(image)
        image = self.conv5(image)

        # Fully connected layers
        image = image.reshape(image.size(0), -1)
        image = self.fc1(image)
        image = self.fc2(image)
        return image