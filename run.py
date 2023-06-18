import argparse
import re

from setups.RunConfig import RunConfig
from models.FDHCGazeModelAlexNet import FDHCGazeModelAlexNet
from models.FDHCGazeModelResNet18 import FDHCGazeModelResNet18
from models.FDLCGazeModelAlexNet import FDLCGazeModelAlexNet
from models.FDLCGazeModelResNet18 import FDLCGazeModelResNet18
from models.ColorGazeModelAlexNet import ColorGazeModelAlexNet
from models.ColorGazeModelResNet18 import ColorGazeModelResNet18
from setups.calibrate import calibrate
from setups.double import double
from setups.train import train


def main(config):
    # Run both models fully or just a single model with one stage
    if config.model == "double":
        double(config)
    else:
        # Create the correct type of model to run on
        model_name = f"{config.model}{config.data_type}{config.model_id}.pt"
        if config.data_type == "RGB" or config.data_type == "YCbCr":
            if config.model == "AlexNet":
                model = ColorGazeModelAlexNet(model_name)
            else:
                model = ColorGazeModelResNet18(model_name)
        # TODO: changed this method (from config.data_type == "FDAll")
        elif bool(re.search(r'A', config.data_type, re.IGNORECASE)):
            # Get the number of input channels
            input_channels = config.channel_selections[int(re.search(r'\d+', config.data_type).group())]

            if config.model == "AlexNet":
                model = FDHCGazeModelAlexNet(model_name, input_channels)
            else:
                model = FDHCGazeModelResNet18(model_name, input_channels)
        else:
            # Get the number of input channels
            input_channels = config.channel_selections[int(re.search(r'\d+', config.data_type).group())]

            if config.model == "AlexNet":
                model = FDLCGazeModelAlexNet(model_name, input_channels)
            else:
                model = FDLCGazeModelResNet18(model_name, input_channels)

        # Run train or calibrate based on the model id
        if re.search('[a-zA-Z]', args.model_id) is None:
            train(config, model)
        else:
            calibrate(config, model)


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
                        default="double",
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

    # Run the main method that will set everything up and run train, test or double
    main(config)
