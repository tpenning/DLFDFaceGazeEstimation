import math

from torch.utils.data import DataLoader

from datasets.RGBDataset import RGBDataset
from models.RGBGazeModel import RGBGazeModel

data_dir = "data"
batch_size = 64
person_images = 500
calibration_size = 0.1
train_epochs = 20
calibration_epochs = 100
learning_rate = 0.0005

def main():
    # Data for training
    train_data = DataLoader(
        RGBDataset(data_dir, [f"p{pid:02}" for pid in range(00, 14)], 0, person_images),
        batch_size=batch_size,
        shuffle=True
    )

    # Data for calibrating
    calibration_data = DataLoader(
        RGBDataset(data_dir, ["p14"], 0, math.floor(calibration_size * person_images)),
        batch_size=batch_size,
        shuffle=True
    )

    # Data for validation
    validation_data = DataLoader(
        RGBDataset(data_dir, ["p14"], math.floor(calibration_size * person_images), person_images),
        batch_size=batch_size,
        shuffle=True
    )

    # Learning process
    model = RGBGazeModel()
    model.learn(train_data, calibration_data, validation_data, train_epochs, calibration_epochs, learning_rate)


if __name__ == "__main__":
    main()
