from torch.utils.data import DataLoader

from datasets.ImageDataset import ImageDataset


def train(config, model):
    # Data for training
    train_data = DataLoader(
        ImageDataset(config.data_type, config.data_dir, config.train_subjects, 0, config.images),
        batch_size=config.batch_size,
        shuffle=True
    )

    # Data for validation
    validation_data = DataLoader(
        ImageDataset(config.data_type, config.data_dir, config.test_subjects, 0, config.images),
        batch_size=config.batch_size,
        shuffle=False
    )

    # Run the learning process
    model.learn(train_data, validation_data, config.train_epochs, config.learning_rate, config.saves_dir,
                False, config.model_id)
