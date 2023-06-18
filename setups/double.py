import re

from models.FDGazeModelAlexNet import FDGazeModelAlexNet
from models.FDGazeModelResNet18 import FDGazeModelResNet18
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
    if config.data == "RGB" or config.data == "YCbCr":
        train_alex_net = ColorGazeModelAlexNet(f"AlexNet{config.data}{config.lc_hc}{model_ids[0]}.pt", config.lc_hc)
        calibrate_alex_net = ColorGazeModelAlexNet(f"AlexNet{config.data}{config.lc_hc}{model_ids[1]}.pt", config.lc_hc)
        train_res_net_18 = ColorGazeModelResNet18(f"ResNet18{config.data}{config.lc_hc}{model_ids[2]}.pt", config.lc_hc)
        calibrate_res_net_18 = ColorGazeModelResNet18(f"ResNet18{config.data}{config.lc_hc}{model_ids[3]}.pt", config.lc_hc)
    else:
        # Get the number of input channels
        input_channels = config.channel_selections[int(re.search(r'\d+', config.data).group())]

        train_alex_net = FDGazeModelAlexNet(f"AlexNet{config.data}{config.lc_hc}{model_ids[0]}.pt", input_channels, config.lc_hc)
        calibrate_alex_net = FDGazeModelAlexNet(f"AlexNet{config.data}{config.lc_hc}{model_ids[1]}.pt", input_channels, config.lc_hc)
        train_res_net_18 = FDGazeModelResNet18(f"ResNet18{config.data}{config.lc_hc}{model_ids[2]}.pt", input_channels, config.lc_hc)
        calibrate_res_net_18 = FDGazeModelResNet18(f"ResNet18{config.data}{config.lc_hc}{model_ids[3]}.pt", input_channels, config.lc_hc)

    return train_alex_net, calibrate_alex_net, train_res_net_18, calibrate_res_net_18
