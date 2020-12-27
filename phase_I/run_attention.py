import numpy as np
from scipy import signal as sg
import scipy.ndimage as ndimage


def get_kernel() -> np.array:
    kernel = np.array([[-1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1],
                       [-1, -1, -1, -1, -1, -1,  1, 1,  1, -1, -1, -1, -1, -1, -1],
                       [-1, -1, -1, -1, -1,  1,  1, 1,  1,  1, -1, -1, -1, -1, -1],
                       [-1, -1, -1, -1,  1,  1,  1, 1,  1,  1,  1, -1, -1, -1, -1],
                       [-1, -1, -1,  1,  1,  1,  1, 1,  1,  1,  1,  1, -1, -1, -1],
                       [-1, -1,  1,  1,  1,  1,  1, 1,  1,  1,  1,  1,  1, -1, -1],
                       [-1,  1,  1,  1,  1,  1,  1, 1,  1,  1,  1,  1,  1,  1, -1],
                       [ 1,  1,  1,  1,  1,  1,  1, 0,  1,  1,  1,  1,  1,  1,  1],
                       [-1,  1,  1,  1,  1,  1,  1, 1,  1,  1,  1,  1,  1,  1, -1],
                       [-1, -1,  1,  1,  1,  1,  1, 1,  1,  1,  1,  1,  1, -1, -1],
                       [-1, -1, -1,  1,  1,  1,  1, 1,  1,  1,  1,  1, -1, -1, -1],
                       [-1, -1, -1, -1,  1,  1,  1, 1,  1,  1,  1, -1, -1, -1, -1],
                       [-1, -1, -1, -1, -1,  1,  1, 1,  1,  1, -1, -1, -1, -1, -1],
                       [-1, -1, -1, -1, -1, -1,  1, 1,  1, -1, -1, -1, -1, -1, -1],
                       [-1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1]], float)
    return kernel * (1 / 225)


def get_coordinate(c_image: np.ndarray, kernel: list) -> [list, list]:
    dim = 15
    filtered_image = ndimage.maximum_filter(sg.convolve2d(c_image, kernel, boundary='symm', mode='same'), size=dim)
    x, y = [], []
    height, width = filtered_image.shape

    for i in range(dim // 2, height, dim):
        for j in range(dim // 2, width, dim):
            if filtered_image[i, j] > 25:
                y.append(i)
                x.append(j)

    return x, y


def find_tfl_lights(c_image: np.ndarray, **kwargs) -> [list, list, list, list]:
    """
    Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement
    :param c_image: The image itself as np.uint8, shape of (H, W, 3)
    :param kwargs: Whatever config you want to pass in here
    :return: 4-tuple of x_red, y_red, x_green, y_green
    """
    kernel = get_kernel()

    red_x, red_y = get_coordinate(c_image[:, :, 0], kernel)
    green_x, green_y = get_coordinate(c_image[:, :, 1], kernel)

    return red_x, red_y, green_x, green_y

