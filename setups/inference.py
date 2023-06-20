import torch
from torch.utils.data import DataLoader


def inference(config, data, model):
    # Data for inference
    inference_data = DataLoader(data)

    # Load the given model with the calibrated model data
    model_path = f"models/saves/{model.name}"
    model.load_state_dict(torch.load(model_path))

    # Run the inference process
    model.freeze_bn_layers()
    model.inference(inference_data, config.model_id)
