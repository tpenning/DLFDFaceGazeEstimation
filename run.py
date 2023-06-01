import argparse
import re

from setups.RunConfig import RunConfig
from models.FDAllGazeModelAlexNet import FDAllGazeModelAlexNet
from models.FDAllGazeModelResNet18 import FDAllGazeModelResNet18
from models.FDGazeModelAlexNet import FDGazeModelAlexNet
from models.FDGazeModelResNet18 import FDGazeModelResNet18
from models.RGBGazeModelAlexNet import RGBGazeModelAlexNet
from models.RGBGazeModelResNet18 import RGBGazeModelResNet18
from setups.train import train
from setups.test import calibrate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-images',
                        '--images',
                        default=3000,
                        type=int,
                        required=False,
                        help="amount of images to use per person")

    parser.add_argument('-model',
                        '--model',
                        default="ResNet18",
                        type=str,
                        required=False,
                        help="what type of model to run")

    parser.add_argument('-data_type',
                        '--data_type',
                        default="RGB",
                        type=str,
                        required=False,
                        help="the type of date the model is running with")

    parser.add_argument('-model_id',
                        '--model_id',
                        type=str,
                        required=True,
                        help="id of the model")

    # Collect all the arguments
    args = parser.parse_args()

    # Create a RunConfig instance to access all the parameters in the run files
    config = RunConfig(args)

    # Create the correct type of model to run on
    model_name = f"{config.model}{config.data_type}{config.model_id}.pt"
    if config.data_type == "FD":
        if config.model == "AlexNet":
            model = FDGazeModelAlexNet(model_name)
        else:
            model = FDGazeModelResNet18(model_name)
    elif config.data_type == "FDAll":
        if config.model == "AlexNet":
            model = FDAllGazeModelAlexNet(model_name)
        else:
            model = FDAllGazeModelResNet18(model_name)
    else:
        if config.model == "AlexNet":
            model = RGBGazeModelAlexNet(model_name)
        else:
            model = RGBGazeModelResNet18(model_name)

    # Run train or calibrate based on the model id
    if re.search('[a-zA-Z]', args.model_id) is None:
        train(config, model)
    else:
        calibrate(config, model)
