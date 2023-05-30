import argparse
import re

from configs.RunConfig import RunConfig
from train import train
from test import calibrate


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

    # Run train or calibrate based on the model id
    if re.search('[a-zA-Z]', args.model_id) is None:
        train(config)
    else:
        calibrate(config)
