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

    number_of_data = data_tensor.shape[0]
    number_of_filters = kernel_tensor.shape[0]

    m, n = kernel_tensor.shape[1:3]
    y, x = data_tensor.shape[1:3]

    result = np.zeros((number_of_filters, y, x), dtype=np.float32)

    for f in range(number_of_filters):
        logging.debug("Convolution for filter {}".format(f))

        kernel = kernel_tensor[f]

        for i in range(y):
            for j in range(x):
                # Process data one at a time
                for index in range(number_of_data):

                    # Zero pad the data
                    stacked_data = np.hstack(
                        (np.zeros((x, 1)), data_tensor[index], np.zeros((x, 1))))

                    stacked_data = np.vstack(
                        (np.zeros((1, y+2)), stacked_data, np.zeros((1, y+2))))

                    result[f, i, j] = np.sum(stacked_data[i:i+m, j:j+m]*kernel)
