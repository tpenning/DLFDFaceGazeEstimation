import torch
import argparse

from torch.utils.data import DataLoader

from datasets.RGBDataset import RGBDataset
from models.RGBGazeModelAlexNet import RGBGazeModelAlexNet
from models.RGBGazeModelResNet18 import RGBGazeModelResNet18


def main(args):
    # Data for calibration
    train_data = DataLoader(
        RGBDataset(args.data_dir, ["p14"], 0, args.calibration_images),
        batch_size=args.batch_size,
        shuffle=True
    )

    # Data for validation
    validation_data = DataLoader(
        RGBDataset(args.data_dir, ["p14"], args.calibration_images, args.person_images),
        batch_size=args.batch_size,
        shuffle=False
    )

    # Load the given model
    model_path = f"models/saves/RGBGazeModel{args.model}.pt"
    model = RGBGazeModelAlexNet(args.model_id) if args.model.startswith("AlexNet") else \
        RGBGazeModelResNet18(args.model_id)
    model.load_state_dict(torch.load(model_path))

    # Learning process
    model.learn(train_data, validation_data, args.epochs, args.learning_rate, args.filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-model',
                        '--model',
                        type=str,
                        required=True,
                        help="the type of model with id")

    parser.add_argument('-data_dir',
                        '--data_dir',
                        default="data",
                        type=str,
                        required=False,
                        help="path to the data directory")

    parser.add_argument('-person_images',
                        '--person_images',
                        default=3000,
                        type=int,
                        required=False,
                        help="amount of images to use per person")

    parser.add_argument('-batch_size',
                        '--batch_size',
                        default=64,
                        type=int,
                        required=False,
                        help="amount of images per batch")

    parser.add_argument('-calibration_images',
                        '--calibration_images',
                        default=100,
                        type=int,
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
                        default=0.00001,
                        type=float,
                        required=False,
                        help="learning rate of the model")

    parser.add_argument('-model_id',
                        '--model_id',
                        type=str,
                        required=True,
                        help="id of the model")

    args = parser.parse_args()
    main(args)
