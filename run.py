import argparse
import re

from setups.RunConfig import RunConfig
from models.FDGazeModelAlexNet import FDGazeModelAlexNet
from models.FDGazeModelResNet18 import FDGazeModelResNet18
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
        model_name = f"{config.model}{config.data}{config.lc_hc}{config.model_id}.pt"
        if config.data == "RGB" or config.data == "YCbCr":
            if config.model == "AlexNet":
                model = ColorGazeModelAlexNet(model_name, config.lc_hc)
            else:
                model = ColorGazeModelResNet18(model_name, config.lc_hc)
        else:
            # Get the number of input channels
            input_channels = config.channel_selections[int(re.search(r'\d+', config.data).group())]

            if config.model == "AlexNet":
                model = FDGazeModelAlexNet(model_name, input_channels, config.lc_hc)
            else:
                model = FDGazeModelResNet18(model_name, input_channels, config.lc_hc)

        # Run train or calibrate based on the model id
        if re.search('[a-zA-Z]', args.model_id) is None:
            train(config, [model])
        else:
            calibrate(config, [model])


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

    parser.add_argument('-data',
                        '--data',
                        default="RGB",
                        type=str,
                        required=False,
                        help="the type of data the model is running with")

    parser.add_argument('-lc_hc',
                        '--lc_hc',
                        default="LC",
                        type=str,
                        required=False,
                        help="whether the low channel or high channel version of the models should be used")

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
