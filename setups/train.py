from torch.utils.data import DataLoader

from utils.data_help import split_data


def train(config, data, model):
    # Split the dataset
    training_set, validation_set = split_data(data, config.training_size)

    # Data for training
    train_data = DataLoader(
        training_set,
        batch_size=config.batch_size,
        shuffle=True
    )

    # Data for validation
    validation_data = DataLoader(
        validation_set,
        batch_size=config.batch_size,
        shuffle=False
    )

    # Run the training process
    model.learn(train_data, validation_data, config.train_epochs, config.learning_rate, config.saves_dir,
                False, config.model_id)
