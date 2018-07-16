import logging

import numpy as np


def zero_padding(tensor):
    """
    Pad the 2D tensor with zeros
    """

    xd = tensor.shape[0]

    tensor = np.hstack(
        (np.zeros((xd, 1)), tensor, np.zeros((xd, 1))))

    yd = tensor.shape[1]

    tensor = np.vstack(
        (np.zeros((1, yd)), tensor, np.zeros((1, yd))))

    return tensor


def convolve_2d(data, kernel):
    """
    Perform a 2D convole of data with the kernel
    """
    m, n = kernel.shape
    y, x = data.shape

    # Zero pad the data
    stacked_data = zero_padding(data)

    result = np.zeros((y, x))
    for i in range(y):
        for j in range(x):
            result[i][j] = np.sum(stacked_data[i:i+m, j:j+m]*kernel)

    return result


def convolve_tensors(data_tensor, kernel_tensor):
    # NOTE: Need to improve this

    channels = data_tensor.shape[1]
    number_of_data = data_tensor.shape[0]
    number_of_filters = kernel_tensor.shape[0]

    m, n = kernel_tensor.shape[2:4]
    y, x = data_tensor.shape[2:4]

    result = np.zeros((number_of_data, number_of_filters,
                       y, x), dtype=np.float32)

    # Pad the data with zeros
    pads = [(0, 0), (0, 0), (1, x), (1, y)]
    padded_tensor = np.pad(
        data_tensor, pads, mode="constant", constant_values=0)

    for f in range(number_of_filters):
        logging.debug("Convolution for filter {}".format(f+1))

        for c in range(channels):
            logging.debug(
                "Convolution for channel : {}".format(c+1))

            kernel = kernel_tensor[f, c, :, :]

            for i in range(y):
                for j in range(x):
                    # Process data one at a time
                    for index in range(number_of_data):
                        result[index, f, i, j] += np.sum(
                            padded_tensor[index, c, i:i+m, j:j+m]*kernel)
    return result
