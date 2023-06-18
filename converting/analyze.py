import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from convert import dct_transform, inverse_dct_to_ycbcr, ycbcr_to_rgb, plot_original_reconstructed


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
    y_select = [0]
    cb_select = [0]
    cr_select = [0]

    # Get the DCT transformed image
    dct_cubes = dct_transform(image)

    # Get the DCT transformed image with only the specified channels selected
    dct_cubes_selected = select_channels(dct_cubes, [y_select, cb_select, cr_select])

    # Reconstruct the image with the channel-selected DCT cubes
    y, cb, cr = inverse_dct_to_ycbcr(dct_cubes_selected)
    reconstructed_image = ycbcr_to_rgb(y, cb, cr)

    # Plot for the original and reconstructed image
    plot_original_reconstructed(image, reconstructed_image)


def generate_channel_plots(image, pid):
    # Get the DCT transformed image
    dct_cubes = dct_transform(image)

    # Set plot values in variables
    # Various options can be used for the chroma channel mapping: "viridis", "RdYlBu", "bwr", "RdGy", "PuOr" and more
    titles = [f"{pid} Y frequency channels", f"{pid} Cb frequency channels", f"{pid} Cr frequency channels",
              f"{pid} Combined frequency channels"]
    color_schemes = ["gray", "viridis", "viridis", None]
    size = 8

    # Define the 4 plots for the Y, Cb, Cr and "combined" frequency channels
    for i in range(4):
        plt.figure(i, figsize=(size, size))

    # For each frequency channel index retrieve the channels information and add them to the plots
    for i in tqdm(range(64)):
        # Get the DCT transformed image for the current frequency index
        dct_cubes_selected = select_channels(dct_cubes, ([i], [i], [i]))

        # Reconstruct the image with the channel-selected DCT cubes
        # Retrieve the YCbCr components and the full image, then add all to a list
        y, cb, cr = inverse_dct_to_ycbcr(dct_cubes_selected)
        reconstructed_image = ycbcr_to_rgb(y, cb, cr)
        results = [y, cb, cr, reconstructed_image]

        for j in range(4):
            # Change the activate figure and create a subplot for the current image to display
            plt.figure(j)
            ax = plt.subplot(size, size, i + 1)
            ax.axis('off')

            # Plot the image with the correct color scheme
            ax.imshow(results[j], cmap=color_schemes[j])

            # Optional, line that adds the frequency index to each image to show the layout of the channels
            ax.text(0.5, 0.5, str(i), fontsize=20, color='white', ha='center', va='center', transform=ax.transAxes)

    # Adjust the spacing between subplots for each figure
    for i in range(4):
        # Change the activate figure and set the spacing
        plt.figure(i)
        plt.subplots_adjust(wspace=0.02, hspace=0.02)

        # Optional, add titles to the plots
        # plt.suptitle(titles[i], fontsize=20)

    # Show the plots
    plt.show()


if __name__ == "__main__":
    # Change the person ids that you want to analyze
    # for pid in tqdm([f"p{pid:02}" for pid in range(00, 14)]):
    for pid in tqdm(["p05"]):
        # Specify the file path to the image .npy file and retrieve the image
        image_file_path = f"../data/{pid}/images.npy"
        image = np.load(image_file_path)[0]

        # Analyze the reconstructed image with the removed channels
        # analyze_dct_transform_single_selection(image)

        # Generate the channel plots
        generate_channel_plots(image, pid)
