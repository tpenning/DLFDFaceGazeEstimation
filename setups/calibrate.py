import torch
import re
from torch.utils.data import DataLoader

from utils.data_help import split_data


def calibrate(config, data, model):
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

    # Load the given model with the trained model data
    trained_model_name = re.sub(r'[A-Z]+\.pt$', '.pt', model.name)
    model_path = f"models/saves/{trained_model_name}"
    model.load_state_dict(torch.load(model_path))

    # Run the calibration process
    model.freeze_bn_layers()
    model.learn(calibration_data, validation_data, config.calibration_epochs, config.learning_rate,
                config.saves_dir, True, config.model_id)
