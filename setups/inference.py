import torch
from torch.utils.data import DataLoader

from datasets.ImageDataset import ImageDataset


def inference(config, model):
    # Data for inference
    inference_data = DataLoader(ImageDataset(config.data, config.data_dir, config.test_subjects, 0, config.images))

    # Load the given model with the calibrated model data
    model_path = f"models/saves/{model.name}"
    model.load_state_dict(torch.load(model_path))

    # Run the inference process
    model.freeze_bn_layers()
    # TODO: do this
    model.inference(inference_data)
