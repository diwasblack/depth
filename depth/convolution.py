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
    # NOTE: Verify/Improve this

    number_of_data = data_tensor.shape[0]
    number_of_filters = kernel_tensor.shape[0]

    m, n = kernel_tensor.shape[2:4]
    x, y = data_tensor.shape[2:4]

    result = np.zeros((number_of_data, number_of_filters,
                       x, y), dtype=np.float32)

    # Pad the data with zeros
    pads = [(0, 0), (0, 0), (1, 1), (1, 1)]
    padded_tensor = np.pad(
        data_tensor, pads, mode="constant", constant_values=0)

    # Process data one at a time
    for index in range(number_of_data):
        logging.debug("Convolution for data n={}".format(index))

        for i in range(x):
            for j in range(y):
                data_block = padded_tensor[index, :, i:i+m, j:j+m]

                for f in range(number_of_filters):
                    convolution = np.multiply(data_block, kernel_tensor[f])
                    result[index, f, i, j] = np.sum(convolution)

    return result
