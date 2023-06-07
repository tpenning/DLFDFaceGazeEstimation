import numpy as np

from converting.convert import dct_transform


def select_channels(image, model: str):
    # Convert the image to the frequency domain and concatenate the YCbCr components
    # The concatenating done below gives the intended result of Y then Cb then Cr in the last dimension
    dct_cubes = dct_transform(image)
    all_channels = np.squeeze(np.concatenate(np.split(dct_cubes, 3, axis=3), axis=2))

    # Select the desired channels and return them
    channels_to_select = get_channel_indices(model)
    selected_channels = all_channels[:, :, channels_to_select]
    return selected_channels


def get_channel_indices(model: str):
    # Decide on what channels to select based on the model
    if model == "FD1CS":
        y = np.array([0])
        cb = np.array([0])
        cr = np.array([0])
    elif model == "FD2CS":
        y = np.array([0, 1, 8])
        cb = np.array([0, 1, 8])
        cr = np.array([0, 1, 8])
    elif model == "FD3CS":
        y = np.array([0, 1, 2, 8, 9, 16])
        cb = np.array([0, 1, 8])
        cr = np.array([0, 1, 8])
    elif model == "FD4CS":
        y = np.array([0, 1, 2, 3, 8, 9, 10, 16, 17, 24])
        cb = np.array([0, 1, 3, 8, 24])
        cr = np.array([0, 1, 3, 8, 24])
    elif model == "FD5CS":
        y = np.array([0, 1, 2, 3, 8, 9, 10, 16, 17, 24])
        cb = np.array([0, 1, 2, 8, 9, 16])
        cr = np.array([0, 1, 2, 8, 9, 16])
    elif model == "FD6CS":
        y = np.array([0, 1, 2, 3, 4, 8, 9, 10, 11, 16, 17, 18, 24, 25, 32])
        cb = np.array([0, 1, 2, 3, 8, 9, 10, 16, 17, 24])
        cr = np.array([0, 1, 2, 3, 8, 9, 10, 16, 17, 24])
    else:
        y = np.arange(64)
        cb = np.arange(64)
        cr = np.arange(64)

    # Concatenate the channel indices to select and adjust them to fit
    channels_to_select = np.concatenate([y, cb + 64, cr + 128])
    return channels_to_select


if __name__ == "__main__":
    # Specify the file path to the image .npy file and retrieve one
    image_file_path = "../data/p00/images.npy"
    image = np.load(image_file_path)[0]
    selected_channels = select_channels(image, "FDAll")
