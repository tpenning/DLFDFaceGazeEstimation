import argparse
import re

from setups.RunConfig import RunConfig
from models.GazeModelAlexNet import GazeModelAlexNet
from models.GazeModelResNet18 import GazeModelResNet18
from setups.calibrate import calibrate
from setups.inference import inference
from setups.train import train


def get_model(config):
    # Create the correct type of model to run on and the number of input channels (None for the color models)
    model_name = f"{config.model}{config.data}{config.lc_hc}{config.model_id}.pt"
    input_channels = None if config.data == "RGB" or config.data == "YCbCr" else \
        config.channel_selections[int(re.search(r'\d+', config.data).group())]

    if config.model == "AlexNet":
        return GazeModelAlexNet(model_name, config.lc_hc, input_channels=input_channels)
    else:
        return GazeModelResNet18(model_name, config.lc_hc, input_channels=input_channels)


def main(config):
    # Run train, calibrate, run inference or do all based on the model id and run
    if config.run == "single":
        if re.search('[a-zA-Z]', config.model_id) is None:
            train(config, get_model(config))
        else:
            calibrate(config, get_model(config))
    elif config.run == "inference":
        inference(config, get_model(config))
    else:
        # Run each step: training, calibration, inference for a model
        # Strip the model id of any characters and train the model
        config.model_id = re.sub("[^0-9]", "", config.model_id)
        train(config, get_model(config))

        # Add a training identifier to the model id and calibrate the model
        config.model_id += "A"
        calibrate(config, get_model(config))

        # Run inference on the model
        inference(config, get_model(config))


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
                        default="AlexNet",
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
                        help="id of the model that is being created or inference is run on")

    parser.add_argument('-run',
                        '--run',
                        default="full",
                        type=str,
                        required=False,
                        help="whether train/calibrate (single), inference or a full run is done")

    # Collect all the arguments
    args = parser.parse_args()

    # Create a RunConfig instance to access all the parameters in the run files
    config = RunConfig(args)

    # Run the main method that will set everything up and run train, test or double
    main(config)
