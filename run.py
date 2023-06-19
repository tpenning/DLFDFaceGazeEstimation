import argparse
import numpy as np
import re
from tqdm import tqdm

from setups.RunConfig import RunConfig
from models.GazeModelAlexNet import GazeModelAlexNet
from models.GazeModelResNet18 import GazeModelResNet18
from setups.calibrate import calibrate
from setups.inference import inference
from setups.train import train
from utils.write_help import write_to_file


def get_model(config):
    # Create the correct type of model to run on and the number of input channels (None for the color models)
    model_name = f"{config.model}{config.data}{config.lc_hc}{config.model_id}.pt"
    input_channels = None if config.data == "RGB" or config.data == "YCbCr" else \
        config.channel_selections[int(re.search(r'\d+', config.data).group())]

    if config.model == "AlexNet":
        return GazeModelAlexNet(model_name, config.lc_hc, config.run, input_channels=input_channels)
    else:
        return GazeModelResNet18(model_name, config.lc_hc, config.run, input_channels=input_channels)


def one_run(config):
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


def multiple_runs(config):
    # Get the int value of the model and strip any characters on it
    model_id = int(re.sub("[^0-9]", "", config.model_id))

    for i in tqdm(range(4)):
        # Change the config for correct model to run
        if i < 2:
            config.model = "AlexNet"
        else:
            config.model = "ResNet18"
        if i % 2 == 0:
            config.lc_hc = "LC"
        else:
            config.lc_hc = "HC"

        for _ in tqdm(range(config.model_runs)):
            # Set the model id in the config and run full for the model (all is the same as full in one_run)
            config.model_id = str(model_id)
            one_run(config)

            # Update the model id for the next one
            model_id += 1

        # Add the averages over the runs in the experiment report files
        write_experiment_averages(config)


def write_experiment_averages(config):
    report_name = f"{config.model}{config.data}{config.lc_hc}.txt"
    filename = f"reports/reportExperiment{report_name}"

    # This list will contain the total added reported results, in order:
    categories = ["training time", "training accuracy", "calibration time", "calibration accuracy",
                  "inference time", "inference accuracy"]
    total_results = np.zeros(6)

    # Read the data from the file and add it correctly
    with open(filename, "r") as file:
        for line in file:
            # Check if the line is non-empty
            if line.strip():
                # Split the identifier and value
                information = line.strip().split(": ")

                # Decide on the result value based on the identifier
                if information[0].__contains__("training"):
                    result_index = 0
                elif information[0].__contains__("calibration"):
                    result_index = 2
                else:
                    result_index = 4
                if information[0].__contains__("accuracy"):
                    result_index += 1

                # Add the result to the correct value
                total_results[result_index] += float(information[1])

    # Average the results and write them to the file
    average_results = total_results / config.model_runs
    write_to_file(filename, "Average results over the experiment:")
    for i in range(len(average_results)):
        write_to_file(filename, f"Average {categories[i]}: {average_results[i]}")


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
                        help="whether train/calibrate (single), inference, a full run, or an experiment run is done")

    # Collect all the arguments
    args = parser.parse_args()

    # Create a RunConfig instance to access all the parameters in the run files
    _config = RunConfig(args)

    # Run the method that will set everything up and run the correct models
    if _config.run == "experiment":
        multiple_runs(_config)
    else:
        one_run(_config)
