import logging

import numpy as np


def convolve2d(data_tensor, kernel_tensor):
    """
    Compute the 2D convolution of the data_tensor with kernel_tensor

    input:
    data_tensor = N * c * x * y tensor
    kernel_tensor = c * m * n

    output:
    a N * x' * y' tensor

    where,
    N = number of samples
    c = number of channels

    x, y = size of data
    m, n = filter size

    x' = x - m + 2 * p + 1
    y' = x - m + 2 * p + 1

    See:
    https://stackoverflow.com/questions/2448015/2d-convolution-using-python-and-numpy/42579291#42579291
    """

    if(data_tensor.shape[1] != kernel_tensor.shape[0]):
        raise Exception("Number of channels should be same")

    # Pad width to use for x and y axis
    pad_width = 1
    number_of_data = data_tensor.shape[0]
    channels = data_tensor.shape[1]

    x, y = data_tensor.shape[2:4]
    m, n = kernel_tensor.shape[1:3]

    conv_string = "{0}*{1}*{2}*{3} tensor with {1}*{4}*{5} tensor".format(
        number_of_data, channels, x, y, m, n)

    logging.debug("Performing convolution of " + conv_string)

    x = x - m + 2 * pad_width + 1
    y = y - m + 2 * pad_width + 1

    result = np.zeros((number_of_data, channels, x, y), dtype=np.float32)

    # Pad the data with zeros
    pads = [(0, 0), (0, 0), (pad_width, pad_width), (pad_width, pad_width)]
    padded_tensor = np.pad(
        data_tensor, pads, mode="constant", constant_values=0)

    # Process data one at a time
    for index in range(number_of_data):
        for i in range(x):
            for j in range(y):
                for c in range(channels):
                    data_block = padded_tensor[index, c, i:i+m, j:j+m]
                    convolution = np.multiply(data_block, kernel_tensor[c, :, :])
                    result[index, c, i, j] = np.sum(convolution)

    return result


def convolve_tensors(data_tensor, kernel_tensor):
    """
    Compute the 2D convolution of the data_tensor with kernel_tensor

    input:
    data_tensor = N * c * x * y tensor
    kernel_tensor = f * c * m * n

    where,
    N = number of samples
    c = number of channels
    f = number of filters

    x, y = size of data
    m, n = filter size

    output:
    a N * f * x * y tensor

    See:
    https://stackoverflow.com/questions/2448015/2d-convolution-using-python-and-numpy/42579291#42579291
    """

    if(data_tensor.shape[1] != kernel_tensor.shape[1]):
        raise Exception("Number of channels should be same")

    # Pad width to use for x and y axis
    pad_width = 1
    number_of_data = data_tensor.shape[0]
    number_of_filters = kernel_tensor.shape[0]
    channels = data_tensor.shape[1]

    m, n = kernel_tensor.shape[2:4]
    x, y = data_tensor.shape[2:4]

    conv_string = "{0}*{1}*{2}*{3} tensor with {4}*{1}*{5}*{6} tensor".format(
        number_of_data, channels, x, y, number_of_filters, m, n)

    logging.debug("Performing convolution of " + conv_string)

    x = x - m + 2 * pad_width + 1
    y = y - m + 2 * pad_width + 1

    result = np.zeros((number_of_data, number_of_filters,
                       x, y), dtype=np.float32)

    # Pad the data with zeros
    pads = [(0, 0), (0, 0), (pad_width, pad_width), (pad_width, pad_width)]
    padded_tensor = np.pad(
        data_tensor, pads, mode="constant", constant_values=0)

    # Process data one at a time
    for index in range(number_of_data):

        for i in range(x):
            for j in range(y):
                data_block = padded_tensor[index, :, i:i+m, j:j+m]

                for f in range(number_of_filters):
                    convolution = np.multiply(data_block, kernel_tensor[f])
                    result[index, f, i, j] = np.sum(convolution)

    logging.debug("Finished calculating convolution")
    return result
