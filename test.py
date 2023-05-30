import torch
import re

from torch.utils.data import DataLoader

from datasets.Dataset224 import Dataset224
from models.GazeModel224AlexNet import GazeModel224AlexNet
from models.GazeModel224ResNet18 import GazeModel224ResNet18
from utils.data_help import split_data


def calibrate(config):
    # Get the correct dataset
    if config.data_type == "FD":
        # TODO: Change these later for the channel selection frequency domain model
        data = Dataset224(True, config.data_dir, config.test_subjects, 0, config.images)
    else:
        data = Dataset224(config.data_type == "FDAll", config.data_dir, config.test_subjects, 0, config.images)

    # Split the dataset
    calibration_set, validation_set = split_data(data, config.calibration_size)

    # Data for calibration
    calibration_data = DataLoader(
        calibration_set,
        batch_size=config.batch_size,
        shuffle=True
    )

    # Data for validation
    validation_data = DataLoader(
        validation_set,
        batch_size=config.batch_size,
        shuffle=False
    )

    # Load the given model
    train_model_id = re.split('(\D+)', config.model_id)[0]
    trained_model_name = f"{config.model}{config.data_type}{train_model_id}.pt"
    tuned_model_name = f"{config.model}{config.data_type}{config.model_id}.pt"
    model_path = f"models/saves/{trained_model_name}"

    if config.data_type == "FD":
        # TODO: Change these later for the channel selection frequency domain model
        if config.model == "AlexNet":
            model = GazeModel224AlexNet(tuned_model_name)
        else:
            model = GazeModel224ResNet18(tuned_model_name)
    else:
        if config.model == "AlexNet":
            model = GazeModel224AlexNet(tuned_model_name)
        else:
            model = GazeModel224ResNet18(tuned_model_name)
    model.load_state_dict(torch.load(model_path))

    # Learning process
    model.freeze_bn_layers()
    model.learn(calibration_data, validation_data, config.calibration_epochs, config.learning_rate, config.saves_dir,
                True, config.model_id)
