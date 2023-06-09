import argparse
import numpy as np
import re
from tqdm import tqdm

from setups.RunConfig import RunConfig
from datasets.ImageDataset import ImageDataset
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
    dynamic = config.data.__contains__("FDD")

    if config.model == "AlexNet":
        return GazeModelAlexNet(model_name, config.lc_hc, config.run, config.channel_regularization,
                                input_channels=input_channels, dynamic=dynamic)
    else:
        return GazeModelResNet18(model_name, config.lc_hc, config.run, config.channel_regularization,
                                 input_channels=input_channels, dynamic=dynamic)


def one_run(config, training_data=None, testing_data=None):
    # Run train, calibrate, run inference or do all based on the model id and run
    if config.run == "single":
        if re.search('[a-zA-Z]', config.model_id) is None:
            if training_data is None:
                training_data = ImageDataset(config.data, config.data_dir, config.train_subjects, 0, config.images)
            train(config, training_data, get_model(config))
        else:
            if testing_data is None:
                testing_data = ImageDataset(config.data, config.data_dir, config.test_subjects, 0, config.images)
            calibrate(config, testing_data, get_model(config))
    elif config.run == "inference":
        if testing_data is None:
            testing_data = ImageDataset(config.data, config.data_dir, config.test_subjects, 0, config.images)
        inference(config, testing_data, get_model(config))
    else:
        # Create the datasets if they aren't provided from experiment
        if training_data is None:
            training_data = ImageDataset(config.data, config.data_dir, config.train_subjects, 0, config.images)
        if testing_data is None:
            testing_data = ImageDataset(config.data, config.data_dir, config.test_subjects, 0, config.images)

        # Run each step: training, calibration, inference for a model
        # Strip the model id of any characters and train the model
        config.model_id = re.sub("[^0-9]", "", config.model_id)
        train(config, training_data, get_model(config))

        # Add a training identifier to the model id and calibrate the model
        config.model_id += "A"
        calibrate(config, testing_data, get_model(config))

        # Run inference on the model
        inference(config, testing_data, get_model(config))


def experiment(config):
    # Get the int value of the model and strip any characters on it and create the datasets
    model_id = int(re.sub("[^0-9]", "", config.model_id))
    training_data = ImageDataset(config.data, config.data_dir, config.train_subjects, 0, config.images)
    testing_data = ImageDataset(config.data, config.data_dir, config.test_subjects, 0, config.images)

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
            one_run(config, training_data, testing_data)

            # Update the model id for the next one
            model_id += 1

        # Add the averages over the runs in the experiment report files
        write_experiment_averages(config)


def write_experiment_averages(config):
    report_name = f"{config.model}{config.data}{config.lc_hc}.txt"
    filename = f"reports/reportExperiment{report_name}"

    # This list will contain the total added reported results, in order:
    categories = 6
    category_names = ["training time", "training error", "calibration time", "calibration error",
                      "inference images per second", "inference error"]
    results = [[] for _ in range(categories)]

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
                if information[0].__contains__("error"):
                    result_index += 1

                # Add the result to the correct value
                results[result_index].append(float(information[1]))

    # Write the results to the file
    write_to_file(filename, "Final mean and standard deviation results over the experiment:")
    for i in range(categories):
        write_to_file(filename, f"Final {category_names[i]}: {np.mean(results[i])}, {np.std(results[i])}")


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
        experiment(_config)
    else:
        one_run(_config)
