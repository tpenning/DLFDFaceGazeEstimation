import numpy as np

from converting.convert import dct_transform


# TODO: add selection method that accepts the data type argument and handles that in this file to simplify the dataset code


def select_all(image):
    # Get the dct cubes from the image
    dct_cubes = dct_transform(image)

    # Reshape the dct_cubes from (width, height, 64, 3) to (width, height, 192)
    fd_image = dct_cubes.reshape((dct_cubes.shape[0], dct_cubes.shape[1], 192))

    return fd_image


if __name__ == "__main__":
    # Specify the file path to the image .npy file and retrieve one
    image_file_path = "../data/p00/images.npy"
    image = np.load(image_file_path)[0]
    select_all(image)
