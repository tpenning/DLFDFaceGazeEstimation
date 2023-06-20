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

        # Create 2 layers of the same shape, effectively now the shape is 1x1xCx2
        self.conv2_1 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1)
        self.conv2_2 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1)

    def forward(self, image):
        image = image.to(self.device)

        # Channel information layers
        channel_information = self.avg_pool(image)
        channel_information = self.conv1(channel_information)

        # Probability layers
        prob1 = self.conv2_1(channel_information)
        prob2 = self.conv2_2(channel_information)

        # Normalize the probabilities
        probabilities = torch.cat((prob1, prob2), dim=2)
        probabilities = F.softmax(probabilities, dim=2)
        prob1, prob2 = torch.split(probabilities, 1, dim=2)

        # Get the Bernoulli samples and select the channels from the image using it
        selections = torch.bernoulli(prob1)
        selected_image = image * selections

        # Count the number of 1's
        num_ones = torch.sum(selections == 1).item()
        print(f"Ones to zeros: {num_ones} - {192 - num_ones}")

        return selected_image


import numpy as np
from converting.select_channels import select_channels
if __name__ == "__main__":
    # Create an instance of the model
    model = DynamicCSModel()

    # Move the model to the same device as the input tensor
    device = get_device()
    model.to(device)

    # Specify the file path to the image .npy file and retrieve the image
    image_file_path = "../data/p00/images.npy"
    images = np.array([select_channels(np.load(image_file_path)[0], "FDD7")])  # Shape (1, 28, 28, 192)
    tensor = torch.from_numpy(images.transpose((0, 3, 1, 2))).float()

    # Forward pass
    output = model(tensor)

    # Check the shape of the output tensor
    print(tensor.shape)
    print(output.shape)
