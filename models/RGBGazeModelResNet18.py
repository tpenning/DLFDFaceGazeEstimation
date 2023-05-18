import torch.nn as nn
import torchvision.models as models
from models.GazeModel import GazeModel
from utils.device import get_device


class RGBGazeModelResNet18(GazeModel):
    def __init__(self, model_type, model_id, test_id, device=get_device()):
        super().__init__(model_type, model_id, device)
        self.name = f"RGBGazeModelResNet18{model_type}{model_id}_{test_id}.pt"

        # Get the resnet model
        self.resnet = models.resnet18()
        # Change the output to 2 values
        self.resnet.fc = nn.Linear(512, 2)

        # Configure the device
        self.device = device

    def forward(self, image):
        image = image.to(self.device)

        # Run the image through the network
        image = self.resnet(image)

        return image
