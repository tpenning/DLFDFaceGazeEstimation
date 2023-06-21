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
        self.conv1 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1)

        # Multiply the information to 2 layers, effectively the shape 1x1xCx2 that contain the probabilities
        self.conv2_1 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1, bias=True)
        self.conv2_2 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1, bias=True)

    def forward(self, image, temperature=1.0):
        image = image.to(self.device)

        # Channel information layers
        channel_information = self.avg_pool(image)
        channel_information = self.conv1(channel_information)

        # Probability layers
        prob1 = self.conv2_1(channel_information)
        prob2 = self.conv2_2(channel_information)
        probs = torch.cat((prob1, prob2), dim=2)

        # First normalize the values of probs to the range [0, 6]
        # This step is to allow the gumbel noise to have an effective impact
        probs = torch.abs(probs)
        sum_probs = probs.sum(dim=2, keepdim=True)
        probs = probs / (sum_probs + 1e-8)
        probs *= 6

        # Apply gumbel softmax which gives us results roughly {0, 1}
        # The gumbel noise applies a random factor which can be corrected by the model by changing the ratio
        # Then the softmax with the very low temperature sets the greater value to 1 and the lower to 0
        probs = F.gumbel_softmax(probs, tau=temperature, dim=2)
        selections, _ = torch.split(probs, 1, dim=2)

        # Count the number of 1's
        selected_amount = torch.sum(selections).item() / selections.size(0)
        print(f"Selected channels: {selected_amount} out of 192")

        # Select the channels for the image
        selected_image = image * selections

        return selected_image
