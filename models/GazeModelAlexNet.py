import torch.nn as nn
from models.GazeModel import GazeModel
from utils.device import get_device


class GazeModelAlexNet(GazeModel):
    def __init__(self, model_name: str, lc_hc: str, input_channels=None, device=get_device()):
        super().__init__(device)
        self.name = model_name
        self.device = device

        # Set the variables for the model version to run
        self.input_channels = input_channels
        self.params = [3, 11, 4, 2, 3, 5, 2] if self.input_channels is None else [self.input_channels, 3, 1, 1, 2, 3, 1]
        self.channels = [96, 256, 384, 384, 256, 9216 if self.input_channels is None else 2304]
        if lc_hc == "HC":
            self.channels = [channel * 2 for channel in self.channels]

        # Convolutional layers changed to adapt to the small data size
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.params[0], out_channels=self.channels[0], kernel_size=self.params[1],
                      stride=self.params[2], padding=self.params[3]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=self.params[4], stride=2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.channels[0], out_channels=self.channels[1], kernel_size=self.params[5],
                      stride=1, padding=self.params[6]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=self.params[4], stride=2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=self.channels[1], out_channels=self.channels[2], kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=self.channels[2], out_channels=self.channels[3], kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=self.channels[3], out_channels=self.channels[4], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=self.params[4], stride=2))

        # Fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(self.channels[5], 4096),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.fc2 = nn.Sequential(
            nn.Linear(4096, 2))

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
