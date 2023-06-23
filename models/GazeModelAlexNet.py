import torch
import torch.nn as nn
from models.DynamicCSModel import DynamicCSModel
from models.GazeModel import GazeModel
from utils.device import get_device


class GazeModelAlexNet(GazeModel):
    def __init__(self, model_name: str, lc_hc: str, experiment: str, channel_regularization: float, input_channels=None, dynamic=False, device=get_device()):
        super().__init__(model_name, experiment, channel_regularization, dynamic, device)
        # Set the variables for the model version to run
        self.input_channels = input_channels
        self.params = [3, 11, 4, 2, 3, 5, 2] if self.input_channels is None else [self.input_channels, 3, 1, 1, 2, 3, 1]
        self.channels = [96, 256, 384, 384, 256, 9216 if self.input_channels is None else 2304]
        if lc_hc == "HC":
            self.channels = [channel * 2 for channel in self.channels]

        # Dynamic channel selection layers
        self.dynamic_cs = DynamicCSModel() if self.dynamic else None

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

        # Dynamic channel selection layers
        if self.dynamic:
            image, selected_amount = self.dynamic_cs(image)
        else:
            selected_amount = None

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

        # When running and FDD model add the selected amount to the output
        if self.dynamic:
            result = torch.cat((image, selected_amount), dim=1)
        else:
            result = image

        # TODO: remove nan check
        has_nan3 = torch.isnan(result).any().item()
        if has_nan3:
            print("has_nan3 start:")
            print(image)
            print("==============================")
            print(selected_amount)
            print("==============================")
            print(result)
            print("has_nan3 end:")
        self._check_nan_parameters()
        return result

    def _check_nan_parameters(self):
        for name, param in self.named_parameters():
            if torch.isnan(param).any():
                print(f"Parameter {name} contains nan values")
