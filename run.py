import math
import argparse

from torch.utils.data import DataLoader

from datasets.RGBDataset import RGBDataset
from models.RGBGazeModel import RGBGazeModel

learning_rate = 0.0005

def main(args):
    # Data for training
    train_data = DataLoader(
        RGBDataset(args.data_dir, [f"p{pid:02}" for pid in range(00, 14)], 0, args.person_images),
        batch_size=args.batch_size,
        shuffle=True
    )

    # Data for calibrating
    calibration_data = DataLoader(
        RGBDataset(args.data_dir, ["p14"], 0, math.floor(args.calibration_size * args.person_images)),
        batch_size=args.batch_size,
        shuffle=True
    )

    # Data for validation
    validation_data = DataLoader(
        RGBDataset(args.data_dir, ["p14"], math.floor(args.calibration_size * args.person_images), args.person_images),
        batch_size=args.batch_size,
        shuffle=True
    )

    # Learning process
    model = RGBGazeModel()
    model.learn(train_data, calibration_data, validation_data, args.train_epochs, args.calibration_epochs,
                args.learning_rate, str(args.fileid))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-data_dir',
                        '--data_dir',
                        default="data",
                        type=str,
                        required=False,
                        help="path to the data directory")

    parser.add_argument('-person_images',
                        '--person_images',
                        type=int,
                        required=True,
                        help="amount of images to use per person")

    parser.add_argument('-batch_size',
                        '--batch_size',
                        default=64,
                        type=int,
                        required=False,
                        help="amount of images per batch")

    parser.add_argument('-calibration_size',
                        '--calibration_size',
                        default=0.1,
                        type=int,
                        required=False,
                        help="part of the test data to be used for calibration")

    parser.add_argument('-train_epochs',
                        '--train_epochs',
                        default=20,
                        type=int,
                        required=False,
                        help="number of epochs for training")

    parser.add_argument('-calibration_epochs',
                        '--calibration_epochs',
                        default=100,
                        type=int,
                        required=False,
                        help="number of epochs for calibration")

    parser.add_argument('-learning_rate',
                        '--learning_rate',
                        default=0.00001,
                        type=float,
                        required=False,
                        help="learning rate of the model")

    parser.add_argument('-fileid',
                        '--fileid',
                        type=int,
                        required=True,
                        help="id for the results file")

    args = parser.parse_args()
    main(args)
