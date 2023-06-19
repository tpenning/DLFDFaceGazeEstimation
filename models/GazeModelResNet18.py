import torch.nn as nn
import torchvision.models as models
from models.GazeModel import GazeModel
from utils.device import get_device


class GazeModelResNet18(GazeModel):
    def __init__(self, model_name: str, lc_hc: str, experiment: str, input_channels=None, device=get_device()):
        super().__init__(model_name, experiment, device)
        # Set the variables for the model version to run
        self.input_channels = input_channels
        self.params = [3, 7, 2, 3, 2] if self.input_channels is None else [self.input_channels, 2, 1, 1, 1]
        self.channels = [64, 128, 256, 512]
        if lc_hc == "HC":
            self.channels = [channel * 2 for channel in self.channels]

        # Get the ResNet18 model
        self.resnet = models.resnet18()

        # Change the input channels, kernel size, and stride of the starting layers
        self.resnet.conv1 = nn.Conv2d(in_channels=self.params[0], out_channels=self.channels[0], kernel_size=self.params[1],
                                      stride=self.params[2], padding=self.params[3], bias=False)
        self.resnet.bn1 = nn.BatchNorm2d(self.channels[0])

        # Adjust the number of channels and stride in the residual blocks
        self._adjust_channels(self.resnet.layer1, self.channels[0], self.channels[0])
        self._adjust_channels(self.resnet.layer2, self.channels[0], self.channels[1], stride=self.params[4], downsample_stride=2)
        self._adjust_channels(self.resnet.layer3, self.channels[1], self.channels[2], stride=self.params[4], downsample_stride=2)
        self._adjust_channels(self.resnet.layer4, self.channels[2], self.channels[3], stride=self.params[4], downsample_stride=2)

        # Change the output to 2 values
        self.resnet.fc = nn.Linear(self.channels[3], 2)

    def forward(self, image):
        image = image.to(self.device)

        # Run the image through the network
        image = self.resnet(image)

        return image

    def _adjust_channels(self, layer, in_channels, out_channels, stride=1, downsample_stride=0):
        # Adjust the numbers of channels for the downsample part of the residual block
        if downsample_stride != 0:
            layer[0].downsample[0] = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            layer[0].downsample[1] = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # Adjust the number of channels for the rest of residual block
        for i in range(2):
            if i == 0:
                layer[i].conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                           kernel_size=3, stride=stride, padding=1, bias=False)
            else:
                layer[i].conv1 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                                           kernel_size=3, stride=1, padding=1, bias=False)
            layer[i].bn1 = nn.BatchNorm2d(out_channels)
            layer[i].conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                                       kernel_size=3, stride=1, padding=1, bias=False)
            layer[i].bn2 = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)


if __name__ == "__main__":
    model1 = GazeModelResNet18(model_name="resnet18", lc_hc="LC")
    for name, module in model1.named_modules():
        print(name, module)

    print("\n\n\n\n\n\n")
    model2 = models.resnet18()
    for name, module in model2.named_modules():
        print(name, module)
