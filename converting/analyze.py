import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from convert import dct_transform, inverse_dct_transform, plot_original_reconstructed


def select_channels(dct_cubes, channel_indices):
    # Select the specified, the rest of the channels are "removed" by setting them to zeros
    # This will create a green image, this is chosen over specific values that would give
    # white or black for instance since there it is hard to see the effects of the chroma channels
    dct_cubes_selected = np.zeros(dct_cubes.shape)

    for i, indices in enumerate(channel_indices):
        if len(indices) > 0:
            dct_cubes_selected[:, :, indices, i] = dct_cubes[:, :, indices, i]

    return dct_cubes_selected


def analyze_dct_transform_single_selection(image):
    # The channels to select
    y_remove = [0, 1, 8]
    cb_remove = [0, 1, 8]
    cr_remove = [0, 1, 8]

    # Get the DCT transformed image
    dct_cubes = dct_transform(image)

    # Get the DCT transformed image with only the specified channels selected
    dct_cubes_selected = select_channels(dct_cubes, [y_remove, cb_remove, cr_remove])

    # Reconstruct the image with the channel selected dct cubes
    reconstructed_image = inverse_dct_transform(dct_cubes_selected)

    # Plot for the original and reconstructed image
    plot_original_reconstructed(image, reconstructed_image)


def generate_channel_plots(image):
    # Get the DCT transformed image
    dct_cubes = dct_transform(image)

    # Define the different channel selections, and set the size for the plots
    channel_selections = [
        [([i], [], []) for i in range(64)],  # First channel selected per index
        [([], [i], []) for i in range(64)],  # Second channel selected per index
        [([], [], [i]) for i in range(64)],  # Third channel selected per index
        [([i], [i], [i]) for i in range(64)]  # All channels selected per index
    ]
    size = 8

    # Generate four plots
    for plot_index, selection_list in tqdm(enumerate(channel_selections)):
        # Create a new figure for each plot
        plt.figure(plot_index + 1, figsize=(size, size))

        # Create subplots for each channel selection
        for subplot_index, selection in enumerate(selection_list):
            # Create a subplot for the current channel selection
            ax = plt.subplot(size, size, subplot_index + 1)
            ax.axis('off')

            # Get the DCT transformed image with the specified channels selected
            dct_cubes_selected = select_channels(dct_cubes, selection)

            # Reconstruct the image with the channel selected dct cubes
            reconstructed_image = inverse_dct_transform(dct_cubes_selected)

            # Plot the reconstructed image
            ax.imshow(reconstructed_image)

        # Add a title to the plot
        title = f"Plot {plot_index + 1}"
        plt.suptitle(title)

        # Adjust the spacing between subplots
        plt.tight_layout()

    # Show the plots
    plt.show()


if __name__ == "__main__":
    # Specify the file path to the image .npy file and retrieve the image
    image_file_path = "../data/p00/images.npy"
    image = np.load(image_file_path)[0]

    # Analyze the reconstructed image with the removed channels
    # analyze_dct_transform_single_selection(image)

    # Generate the channel plots
    generate_channel_plots(image)
