import torch
import argparse

from torch.utils.data import DataLoader

from datasets.RGBDataset import RGBDataset
from models.RGBGazeModelAlexNet import RGBGazeModelAlexNet
from models.RGBGazeModelResNet18 import RGBGazeModelResNet18
from utils.data_help import split_data

batch_size = 64
data_dir = "data/old_data"
saves_dir = "models/saves"


def main(args):
    # Fix the test_id
    test_id = str(args.test_id).zfill(2)

    # Split the dataset
    data = RGBDataset(data_dir, [f"p{test_id}"], args.calibration_images, args.person_images)
    calibration_set, validation_set = split_data(data, args.calibration_size)

    # Data for calibration
    calibration_data = DataLoader(
        calibration_set,
        batch_size=batch_size,
        shuffle=True
    )

    # Data for validation
    validation_data = DataLoader(
        validation_set,
        batch_size=batch_size,
        shuffle=False
    )

    # Load the given model
    model_path = f"models/saves/RGBGazeModel{args.model}_{test_id}.pt"
    model = RGBGazeModelAlexNet("Test", args.model_id, test_id) if args.model.startswith("AlexNet") else \
        RGBGazeModelResNet18("Test", args.model_id, test_id)
    model.load_state_dict(torch.load(model_path))

    # Learning process
    model.freeze_bn_layers()
    model.learn(calibration_data, validation_data, args.epochs, args.learning_rate, saves_dir,
                True, args.model_id, test_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-person_images',
                        '--person_images',
                        default=3000,
                        type=int,
                        required=False,
                        help="amount of images to use per person")

    parser.add_argument('-calibration_size',
                        '--calibration_size',
                        default=0.0333,
                        type=float,
                        required=False,
                        help="part of the test data to be used for calibration")

    parser.add_argument('-epochs',
                        '--epochs',
                        default=100,
                        type=int,
                        required=False,
                        help="number of epochs to train for")

    parser.add_argument('-learning_rate',
                        '--learning_rate',
                        default=0.0001,
                        type=float,
                        required=False,
                        help="learning rate of the model")

    parser.add_argument('-model',
                        '--model',
                        type=str,
                        required=True,
                        help="the type of model with id")

    parser.add_argument('-model_id',
                        '--model_id',
                        type=str,
                        required=True,
                        help="id of the model")

    parser.add_argument('-test_id',
                        '--test_id',
                        default=14,
                        type=int,
                        required=False,
                        help="id of the subject used for testing")

    args = parser.parse_args()
    main(args)
