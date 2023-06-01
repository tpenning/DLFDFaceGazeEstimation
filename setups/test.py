import torch
import re

from torch.utils.data import DataLoader

from datasets.ImageDataset import ImageDataset
from utils.dataHelp import split_data


def calibrate(config, model):
    # Split the dataset
    data = ImageDataset(config.data_type, config.data_dir, config.test_subjects, 0, config.images)
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

    # Load the given model with the trained model data
    train_model_id = re.split('(\D+)', config.model_id)[0]
    trained_model_name = f"{config.model}{config.data_type}{train_model_id}.pt"
    model_path = f"models/saves/{trained_model_name}"
    model.load_state_dict(torch.load(model_path))

    # Learning process
    model.freeze_bn_layers()
    model.learn(calibration_data, validation_data, config.calibration_epochs, config.learning_rate, config.saves_dir,
                True, config.model_id)
