import argparse

from torch.utils.data import DataLoader

from datasets.RGBDataset import RGBDataset
from models.RGBGazeModelAlexNet import RGBGazeModelAlexNet
from models.RGBGazeModelResNet18 import RGBGazeModelResNet18

batch_size = 64
data_dir = "data"
saves_dir = "models/saves"


def main(args):
    # Data for training
    train_data = DataLoader(
        RGBDataset(data_dir, [f"p{pid:02}" for pid in range(00, 15) if pid != args.test_id], 0, args.person_images),
        batch_size=batch_size,
        shuffle=True
    )

    # Data for validation
    validation_data = DataLoader(
        RGBDataset(data_dir, [f"p{args.test_id}"], 0, args.person_images),
        batch_size=batch_size,
        shuffle=False
    )

    # Learning process
    model = RGBGazeModelAlexNet(args.model_id, args.test_id) if args.model == "AlexNet" else \
        RGBGazeModelResNet18(args.model_id, args.test_id)
    model.learn(train_data, validation_data, args.epochs, args.learning_rate, saves_dir,
                False, args.model_id, args.test_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-person_images',
                        '--person_images',
                        default=3000,
                        type=int,
                        required=False,
                        help="amount of images to use per person")

    parser.add_argument('-epochs',
                        '--epochs',
                        default=20,
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
                        default="ResNet18",
                        type=str,
                        required=False,
                        help="what type of model to run")

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
