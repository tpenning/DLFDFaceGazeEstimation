import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.device import get_device


class DynamicCSModel(nn.Module):
    def __init__(self, device=get_device()):
        super().__init__()
        # Configure the model
        self.name = "DynamicCSModel.pt"
        self.device = device
        self.to(device)

        # Change the shape from WxHxC to 1x1xC
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.conv1 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1)

        # Multiply the information to 2 layers, effectively the shape 1x1xCx2 that contain the probabilities
        self.conv2_1 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1, bias=True)
        self.conv2_2 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1, bias=True)

        # Layer used to set the lowest values to 0
        self.relu = nn.ReLU(inplace=True)

    def forward(self, image, temperature=1.0):
        image = image.to(self.device)

        # Channel information layers
        channel_information = self.avg_pool(image)
        # channel_information = self.conv1(channel_information)

        # Probability layers
        prob1 = self.conv2_1(channel_information)
        prob2 = self.conv2_2(channel_information)
        probs = torch.cat((prob1, prob2), dim=2)

        # First normalize the values of probs to the range [0, 5]
        # This step is to allow the gumbel noise to have an effective impact
        probs = torch.abs(probs)
        sum_probs = probs.sum(dim=2, keepdim=True)
        probs = probs / (sum_probs + 1e-8)
        probs *= 5

        # Apply gumbel softmax which gives us results roughly results in the range [0, 1]
        # The gumbel noise applies a random factor which can be corrected by the model by changing the ratio
        # The softmax approximates {0, 1}
        probs = F.gumbel_softmax(probs, tau=temperature, dim=2)
        selections, _ = torch.split(probs, 1, dim=2)

        # Set the lowest (around 0.09) values to zero to reduce computational cost
        selections = (selections * 1.1) - 0.1
        selections = self.relu(selections)

        # Get the amount of selected channels and select the channels for the image, then return both
        selected_amounts = torch.sum(selections, dim=(1, 2, 3), keepdim=True).squeeze(dim=2).squeeze(dim=2)
        selected_image = image * selections
        print(f"Selected channels: {torch.mean(selected_amounts).item()} out of 192")

        self._check_nan_parameters()
        return selected_image, selected_amounts

    def _check_nan_parameters(self):
        for name, param in self.named_parameters():
            if torch.isnan(param).any():
                print(f"Parameter {name} contains nan values")