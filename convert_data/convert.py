import numpy as np
import cv2
import matplotlib.pyplot as plt


def dct_transform(image):
    # Convert image to YCbCr color space
    ycbcr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # Split the image into Y, Cb, and Cr channels
    y, cb, cr = cv2.split(ycbcr_image)

    # Apply 2D DCT on each channel
    y_dct = cv2.dct(y.astype(np.float32))
    cb_dct = cv2.dct(cb.astype(np.float32))
    cr_dct = cv2.dct(cr.astype(np.float32))

    # Reshape the Y DCT coefficients
    y_dct_reshaped = y_dct.reshape((y_dct.shape[0] // 8, y_dct.shape[1] // 8, 8, 8))

    # Reshape the Cb and Cr DCT coefficients
    cb_dct_reshaped = cb_dct.reshape((cb_dct.shape[0] // 8, cb_dct.shape[1] // 8, 8, 8))
    cr_dct_reshaped = cr_dct.reshape((cr_dct.shape[0] // 8, cr_dct.shape[1] // 8, 8, 8))

    # Combine the DCT coefficients into a single three-dimensional array
    dct_cubes = np.stack([y_dct_reshaped, cb_dct_reshaped, cr_dct_reshaped], axis=-1)

    return dct_cubes


def all_channels_image(image):
    dct_cubes = dct_transform(image)

    # Reshape the DCT cubes to the desired shape
    reshaped_cubes = dct_cubes.reshape((28, 28, 64, 3))

    # Permute the axes to obtain the final shape
    final_image = reshaped_cubes.transpose((0, 1, 3, 2)).reshape((224, 224, 3))

    return final_image


def inverse_dct_transform(dct_cubes):
    # Split the DCT cubes into individual channels
    y_dct_reshaped, cb_dct_reshaped, cr_dct_reshaped = np.split(dct_cubes, 3, axis=-1)

    # Reshape the DCT coefficients back to their original shape and change data type to np.float32
    y_dct = y_dct_reshaped.reshape((28 * 8, 28 * 8)).astype(np.float32)
    cb_dct = cb_dct_reshaped.reshape((28 * 8, 28 * 8)).astype(np.float32)
    cr_dct = cr_dct_reshaped.reshape((28 * 8, 28 * 8)).astype(np.float32)

    # Apply inverse 2D DCT on each channel
    y = cv2.idct(y_dct)
    cb = cv2.idct(cb_dct)
    cr = cv2.idct(cr_dct)

    # Merge the Y, Cb, and Cr channels into a single image
    ycbcr_image = cv2.merge([y, cb, cr])

    # Convert the image back to the BGR color space
    output_image = cv2.cvtColor(ycbcr_image.astype(np.uint8), cv2.COLOR_YCrCb2BGR)

    return output_image


def show_image(image):
    plt.imshow(image)
    plt.axis('off')
    plt.show()


# Allows testing the dct_transform and back for a single image
if __name__ == "__main__":
    # Specify the file path to the image .npy file
    image_file_path = "../data/p00/images.npy"

    # Load the image from the .npy file
    image = np.load(image_file_path)[0]

    # Apply the transform (DCT)
    _dct_cubes = dct_transform(image)
    transformed_image = all_channels_image(image)

    # Apply the inverse transform
    reversed_image = inverse_dct_transform(_dct_cubes)

    # Show the original image
    show_image(image)

    # Show the YCbCr image
    show_image(cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb))

    # Show the transformed image
    show_image(transformed_image)

    # Show the reversed image
    show_image(reversed_image)
