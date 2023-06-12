import re

from models.FDAllGazeModelAlexNet import FDAllGazeModelAlexNet
from models.FDAllGazeModelResNet18 import FDAllGazeModelResNet18
from models.FDCSGazeModelAlexNet import FDCSGazeModelAlexNet
from models.FDCSGazeModelResNet18 import FDCSGazeModelResNet18
from models.ColorGazeModelAlexNet import ColorGazeModelAlexNet
from models.ColorGazeModelResNet18 import ColorGazeModelResNet18
from setups.calibrate import calibrate
from setups.train import train


def double(config):
    # Get the model ids and retrieve the correct models
    model_id_int = int(config.model_id)
    model_ids = [config.model_id, config.model_id + 'A', str(model_id_int + 1), str(model_id_int + 1) + 'A']
    train_alex_net, calibrate_alex_net, train_res_net_18, calibrate_res_net_18 = get_models(config, model_ids)

    # Run the training models
    train(config, [train_alex_net, train_res_net_18], [model_ids[0], model_ids[2]])

    # Run the calibration models
    calibrate(config, [calibrate_alex_net, calibrate_res_net_18], [model_ids[1], model_ids[3]])


def get_models(config, model_ids):
    # Get all the 4 models to train/calibrate
    if config.data_type == "RGB" or config.data_type == "YCbCr":
        train_alex_net = ColorGazeModelAlexNet(f"AlexNet{config.data_type}{model_ids[0]}.pt")
        calibrate_alex_net = ColorGazeModelAlexNet(f"AlexNet{config.data_type}{model_ids[1]}.pt")
        train_res_net_18 = ColorGazeModelResNet18(f"ResNet18{config.data_type}{model_ids[2]}.pt")
        calibrate_res_net_18 = ColorGazeModelResNet18(f"ResNet18{config.data_type}{model_ids[3]}.pt")
    # TODO: changed this method (from config.data_type == "FDAll")
    elif bool(re.search(r'A', config.data_type, re.IGNORECASE)):
        # Get the number of input channels
        input_channels = config.channel_selections[int(re.search(r'\d+', config.data_type).group())]

        train_alex_net = FDAllGazeModelAlexNet(f"AlexNet{config.data_type}{model_ids[0]}.pt", input_channels)
        calibrate_alex_net = FDAllGazeModelAlexNet(f"AlexNet{config.data_type}{model_ids[1]}.pt", input_channels)
        train_res_net_18 = FDAllGazeModelResNet18(f"ResNet18{config.data_type}{model_ids[2]}.pt", input_channels)
        calibrate_res_net_18 = FDAllGazeModelResNet18(f"ResNet18{config.data_type}{model_ids[3]}.pt", input_channels)
    else:
        # Get the number of input channels
        input_channels = config.channel_selections[int(re.search(r'\d+', config.data_type).group())]

        train_alex_net = FDCSGazeModelAlexNet(f"AlexNet{config.data_type}{model_ids[0]}.pt", input_channels)
        calibrate_alex_net = FDCSGazeModelAlexNet(f"AlexNet{config.data_type}{model_ids[1]}.pt", input_channels)
        train_res_net_18 = FDCSGazeModelResNet18(f"ResNet18{config.data_type}{model_ids[2]}.pt", input_channels)
        calibrate_res_net_18 = FDCSGazeModelResNet18(f"ResNet18{config.data_type}{model_ids[3]}.pt", input_channels)

    return train_alex_net, calibrate_alex_net, train_res_net_18, calibrate_res_net_18
