import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.device import get_device


def custom_init_weights(layer):
    if isinstance(layer, nn.Conv2d):
        nn.init.uniform_(layer.weight, a=0.4, b=0.6)  # Initialize weights between 0.4 and 0.6
        nn.init.uniform_(layer.bias, a=-0.2, b=0.2)  # Initialize biases between -0.2 and 0.2


class DynamicCSModel(nn.Module):
    def __init__(self, threshold: float, device=get_device()):
        super().__init__()
        # Configure the model
        self.name = "DynamicCSModel.pt"
        self.device = device
        self.to(device)
        self.threshold = threshold

        # Change the shape from WxHxC to 1x1xC
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.conv1 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1)

        # Create 2 layers of the same shape, effectively now the shape is 1x1xCx2
        self.conv2_1 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1, bias=True)
        self.conv2_2 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1, bias=True)
        self.conv2_1.apply(custom_init_weights)
        self.conv2_2.apply(custom_init_weights)

        # ReLu layers to create a binary output
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, image, temperature=10.0):
        image = image.to(self.device)

        # Channel information layers
        channel_information1 = self.avg_pool(image)
        # channel_information2 = self.conv1(channel_information1)

        # Probability layers
        prob1 = self.conv2_1(channel_information1)
        prob2 = self.conv2_2(channel_information1)

        # Normalize the probabilities and add a random factor using Gumbel Softmax
        probs1 = torch.cat((prob1, prob2), dim=2)
        probs2 = F.gumbel_softmax(probs1, tau=temperature, dim=2)
        gumbel_results, _ = torch.split(probs2, 1, dim=2)

        # Create the binary outputs, the method is very complicated to be differentiable
        # selections = gumbel_results
        relu1 = 100 * F.relu(gumbel_results - self.threshold)
        relu2 = 100 * F.relu(self.threshold - gumbel_results)
        relus = torch.cat((relu1, relu2), dim=2)
        softmax_results = F.softmax(relus, dim=2)
        selections, _ = torch.split(softmax_results, 1, 2)

        # Count the number of 1's
        selected_amount = torch.sum(selections).item() / selections.size(0)
        print(f"Selected channels: {selected_amount} out of 192")

        # Select the channels for the image
        selected_image = image * selections

        return selected_image


import numpy as np
from converting.select_channels import select_channels

if __name__ == "__main__":
    # Create an instance of the model
    threshold = 0.7
    model = DynamicCSModel(threshold)

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
