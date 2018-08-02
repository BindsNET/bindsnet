import cv2
import numpy as np


def gray_scale(image: np.ndarray) -> np.ndarray:
    # language=rst
    """
    Converts RGB image into grayscale.

    :param image: RGB image.
    :return: Gray-scaled image.
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def crop(image: np.ndarray, x1: int, x2: int, y1: int, y2: int) -> np.ndarray:
    # language=rst
    """
    Crops an image given coordinates of cropping box.

    :param image: 3-dimensional image.
    :param x1: Left x coordinate.
    :param x2: Right x coordinate.
    :param y1: Bottom y coordinate.
    :param y2: Top y coordinate.
    :return: Image cropped using coordinates (x1, x2, y1, y2).
    """
    return image[x1:x2, y1:y2, :]


def binary_image(image: np.ndarray) -> np.ndarray:
    # language=rst
    """
    Converts input image into black and white (binary)

    :param image: Gray-scaled image.
    :return: Black and white image.
    """
    return cv2.threshold(image, 0, 1, cv2.THRESH_BINARY)[1]


def subsample(image: np.ndarray, x: int, y: int) -> np.ndarray:
    # language=rst
    """
    Scale the image to (x, y).

    :param image: Image to be rescaled.
    :param x: Output value for ``image``'s x dimension.
    :param y: Output value for ``image``'s y dimension.
    :return: Re-scaled image.
    """
    return cv2.resize(image, (x, y))
