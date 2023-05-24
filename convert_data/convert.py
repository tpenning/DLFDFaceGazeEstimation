import numpy as np
import cv2


def load_image_from_npy(file_path):
    return np.load(file_path)[0]


def transform_image(image):
    # Convert image to YCbCr color space
    ycbcr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # Split the image into Y, Cb, and Cr channels
    y, cb, cr = cv2.split(ycbcr_image)

    # Convert the data type to floating-point
    y = y.astype(np.float64)
    cb = cb.astype(np.float64)
    cr = cr.astype(np.float64)

    # Apply 2D DCT on each channel
    y_dct = cv2.dct(cv2.dct(y))
    cb_dct = cv2.dct(cv2.dct(cb))
    cr_dct = cv2.dct(cv2.dct(cr))

    # Combine the DCT coefficients into a single three-dimensional array
    dct_cubes = np.stack([y_dct, cb_dct, cr_dct], axis=-1)

    return dct_cubes


def inverse_transform(dct_cubes):
    # Split the DCT cubes into individual channels
    y_dct, cb_dct, cr_dct = np.split(dct_cubes, 3, axis=-1)

    # Convert the data type to floating-point
    y_dct = y_dct.astype(np.float64)
    cb_dct = cb_dct.astype(np.float64)
    cr_dct = cr_dct.astype(np.float64)

    # Apply inverse 2D DCT on each channel
    y = cv2.idct(cv2.idct(y_dct, flags=cv2.DCT_INVERSE), flags=cv2.DCT_INVERSE)
    cb = cv2.idct(cv2.idct(cb_dct, flags=cv2.DCT_INVERSE), flags=cv2.DCT_INVERSE)
    cr = cv2.idct(cv2.idct(cr_dct, flags=cv2.DCT_INVERSE), flags=cv2.DCT_INVERSE)

    # Merge the Y, Cb, and Cr channels into a single image
    ycbcr_image = cv2.merge([y, cb, cr])

    # Convert the image back to the BGR color space
    output_image = cv2.cvtColor(ycbcr_image.astype(np.uint8), cv2.COLOR_YCrCb2BGR)

    return output_image


def show_image(image):
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Specify the file path to the image .npy file
    image_file_path = "../data/p00/images.npy"

    # Load the image from the .npy file
    image = load_image_from_npy(image_file_path)

    # Apply the transform (DCT)
    transformed_image = transform_image(image)

    # Apply the inverse transform
    reversed_image = inverse_transform(transformed_image)

    # Show the original image
    show_image(image)

    # Show the transformed image
    show_image(transformed_image)

    # Show the reversed image
    show_image(reversed_image)
