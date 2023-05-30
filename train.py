from torch.utils.data import DataLoader

from datasets.Dataset224 import Dataset224
from models.GazeModel224AlexNet import GazeModel224AlexNet
from models.GazeModel224ResNet18 import GazeModel224ResNet18


def train(config):
    if config.data_type == "FD":
        # TODO: Change these later for the channel selection frequency domain model
        train_dataset = Dataset224(True, config.data_dir, config.train_subjects, 0, config.images)
        validation_dataset = Dataset224(True, config.data_dir, config.test_subjects, 0, config.images)
    else:
        train_dataset = Dataset224(config.data_type == "FDAll", config.data_dir, config.train_subjects, 0, config.images)
        validation_dataset = Dataset224(config.data_type == "FDAll", config.data_dir, config.test_subjects, 0,
                                        config.images)

    # Data for training
    train_data = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )

    # Data for validation
    validation_data = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )

    # Decide on the model to run
    model_name = f"{config.model}{config.data_type}{config.model_id}.pt"
    if config.data_type == "FD":
        # TODO: Change these later for the channel selection frequency domain model
        if config.model == "AlexNet":
            model = GazeModel224AlexNet(model_name)
        else:
            model = GazeModel224ResNet18(model_name)
    else:
        if config.model == "AlexNet":
            model = GazeModel224AlexNet(model_name)
        else:
            model = GazeModel224ResNet18(model_name)

    # Run the learning process
    model.learn(train_data, validation_data, config.train_epochs, config.learning_rate, config.saves_dir,
                False, config.model_id)
