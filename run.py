from torch.utils.data import DataLoader

from datasets.RGBDataset import RGBDataset
from models.RGBGazeModel import RGBGazeModel

def main():
    # Data for training
    train_data = DataLoader(
        RGBDataset("data", [f"p{pid:02}" for pid in range(00, 14)], 0, 40),
        batch_size=8,
        shuffle=True
    )

    # Data for evaluation
    eval_data = DataLoader(
        RGBDataset("data", ["p14"], 0, 40),
        batch_size=8,
        shuffle=True
    )

    # Learning process
    model = RGBGazeModel()
    model.learn(train_data, eval_data, 22, 0.0005)

if __name__ == "__main__":
    main()
