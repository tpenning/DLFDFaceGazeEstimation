import cv2
import matplotlib.pyplot as plt
import numpy as np


def dct_transform(image):
    # Convert image from RGB to YCrCb color space, watch the orders
    ycrcb_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

    # Split the YCrCb image into individual channels, watch the orders
    y, cr, cb = cv2.split(ycrcb_image)

    # Convert the channels to the appropriate data type for DCT transform
    y = y.astype(np.float32)
    cb = cb.astype(np.float32)
    cr = cr.astype(np.float32)

    # Divide each channel into non-overlapping blocks of size 8x8
    y_blocks = [np.split(row, image.shape[1] // 8, axis=1) for row in np.split(y, image.shape[0] // 8, axis=0)]
    cb_blocks = [np.split(row, image.shape[1] // 8, axis=1) for row in np.split(cb, image.shape[0] // 8, axis=0)]
    cr_blocks = [np.split(row, image.shape[1] // 8, axis=1) for row in np.split(cr, image.shape[0] // 8, axis=0)]

    # Apply DCT transform to each block
    y_dct_blocks = np.array([[cv2.dct(block) for block in row] for row in y_blocks])
    cb_dct_blocks = np.array([[cv2.dct(block) for block in row] for row in cb_blocks])
    cr_dct_blocks = np.array([[cv2.dct(block) for block in row] for row in cr_blocks])

    # Reshape the arrays
    y_dct_cube = y_dct_blocks.reshape(y_dct_blocks.shape[:2] + (-1,))
    cb_dct_cube = cb_dct_blocks.reshape(cb_dct_blocks.shape[:2] + (-1,))
    cr_dct_cube = cr_dct_blocks.reshape(cr_dct_blocks.shape[:2] + (-1,))

    # Stack the channels to form the DCT transformed image
    dct_cubes = np.stack([y_dct_cube, cb_dct_cube, cr_dct_cube], axis=-1)

    return dct_cubes


def inverse_dct_to_ycbcr(dct_cubes):
    # Split the channels of the DCT transformed image
    y_dct_cube, cb_dct_cube, cr_dct_cube = np.split(dct_cubes, 3, axis=-1)

    # Reshape the cubes
    y_dct_blocks = y_dct_cube.reshape(y_dct_cube.shape[:2] + (8, 8))
    cb_dct_blocks = cb_dct_cube.reshape(cb_dct_cube.shape[:2] + (8, 8))
    cr_dct_blocks = cr_dct_cube.reshape(cr_dct_cube.shape[:2] + (8, 8))

    # Apply inverse DCT transform to each block
    y_blocks = np.array([[cv2.idct(block.astype(np.float64)) for block in row] for row in y_dct_blocks])
    cb_blocks = np.array([[cv2.idct(block.astype(np.float64)) for block in row] for row in cb_dct_blocks])
    cr_blocks = np.array([[cv2.idct(block.astype(np.float64)) for block in row] for row in cr_dct_blocks])

    # Get the shape of the cubes and then reconstruct the channels
    num_rows, num_cols = dct_cubes.shape[:2]
    y = np.block([[y_blocks[i, j] for j in range(num_cols)] for i in range(num_rows)])
    cb = np.block([[cb_blocks[i, j] for j in range(num_cols)] for i in range(num_rows)])
    cr = np.block([[cr_blocks[i, j] for j in range(num_cols)] for i in range(num_rows)])

    # Convert the channels back to the appropriate data type
    y = y.astype(np.uint8)
    cb = cb.astype(np.uint8)
    cr = cr.astype(np.uint8)

    # Return the Y, Cb and Cr components
    return y, cb, cr


def ycbcr_to_rgb(y, cb, cr):
    # Merge the channels into a single YCrCb image, watch the orders
    ycrcb_image = cv2.merge([y, cr, cb])

    # Convert the YCrCb image to BGR color space and return it, watch the orders
    rgb_image = cv2.cvtColor(ycrcb_image, cv2.COLOR_YCrCb2BGR)
    return rgb_image


def test_dct_transform(image):
    # Get the DCT transformed image
    dct_cubes = dct_transform(image)

    # Reconstruct the original image from the transformed coefficients
    y, cb, cr = inverse_dct_to_ycbcr(dct_cubes)
    reconstructed_image = ycbcr_to_rgb(y, cb, cr)

    # Plot the original to reconstructed image and the ycbcr channels
    plot_original_reconstructed(image, reconstructed_image)
    plot_ycbcr(cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb))


def plot_original_reconstructed(original_image, reconstructed_image):
    # Define the plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Add the original image
    axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Add the reconstructed image
    axes[1].imshow(reconstructed_image)
    axes[1].set_title("Reconstructed Image")
    axes[1].axis("off")

    # Adjust the plot spacing and plot it
    plt.tight_layout()
    plt.show()


def plot_ycbcr(ycrcb_image):
    # Define the plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Add the Y channel
    axes[0].imshow(ycrcb_image[:, :, 0], cmap="gray")
    axes[0].set_title("Y Channel")
    axes[0].axis("off")

    # Add the Cb channel
    axes[1].imshow(ycrcb_image[:, :, 2], cmap="RdYlBu")
    axes[1].set_title("Cb Channel")
    axes[1].axis("off")

    # Add the Cr channel
    axes[2].imshow(ycrcb_image[:, :, 1], cmap="RdYlBu")
    axes[2].set_title("Cr Channel")
    axes[2].axis("off")

    # Adjust the plot spacing and plot it
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Specify the file path to the image .npy file and retrieve the image
    image_file_path = "../data/p00/images.npy"
    image = np.load(image_file_path)[0]

    # Test the DCT transform for a single image
    test_dct_transform(image)
