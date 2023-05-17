import argparse

from torch.utils.data import DataLoader

from datasets.RGBDataset import RGBDataset
from models.RGBGazeModelAlexNet import RGBGazeModelAlexNet
from models.RGBGazeModelResNet18 import RGBGazeModelResNet18


def main(args):
    # Data for training
    train_data = DataLoader(
        RGBDataset(args.data_dir, [f"p{pid:02}" for pid in range(00, 14)], 0, args.person_images),
        batch_size=args.batch_size,
        shuffle=True
    )

    # Data for validation
    validation_data = DataLoader(
        RGBDataset(args.data_dir, ["p14"], args.calibration_images, args.person_images),
        batch_size=args.batch_size,
        shuffle=False
    )

    # Learning process
    model = RGBGazeModelAlexNet(args.model_id) if args.model == "AlexNet" else RGBGazeModelResNet18(args.model_id)
    model.learn(train_data, validation_data, args.epochs, args.learning_rate, args.saves_dir, False, args.model_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-person_images',
                        '--person_images',
                        default=3000,
                        type=int,
                        required=False,
                        help="amount of images to use per person")

    parser.add_argument('-calibration_images',
                        '--calibration_images',
                        default=100,
                        type=int,
                        required=False,
                        help="part of the test data to be used for calibration")

    parser.add_argument('-batch_size',
                        '--batch_size',
                        default=64,
                        type=int,
                        required=False,
                        help="amount of images per batch")

    parser.add_argument('-epochs',
                        '--epochs',
                        default=20,
                        type=int,
                        required=False,
                        help="number of epochs to train for")

    parser.add_argument('-learning_rate',
                        '--learning_rate',
                        default=0.00001,
                        type=float,
                        required=False,
                        help="learning rate of the model")

    parser.add_argument('-data_dir',
                        '--data_dir',
                        default="data",
                        type=str,
                        required=False,
                        help="path to the data directory")

    parser.add_argument('-saves_dir',
                        '--saves_dir',
                        default="models/saves",
                        type=str,
                        required=False,
                        help="path to the model saves directory")

    parser.add_argument('-model',
                        '--model',
                        default="ResNet18",
                        type=str,
                        required=False,
                        help="what type of model to run")

    parser.add_argument('-model_id',
                        '--model_id',
                        type=str,
                        required=True,
                        help="id of the model")

    args = parser.parse_args()
    main(args)
