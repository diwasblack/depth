import logging

import numpy as np


def convolve2d(data_tensor, kernel_tensor):
    """
    Compute the 2D convolution of the data_tensor with kernel_tensor

    input:
    data_tensor = N * c * x * y tensor
    kernel_tensor = c * m * n

    output:
    a N * c * x' * y' tensor

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

    if((len(data_tensor.shape) != 4)):
        raise Exception("Dimesion mismatch for input tensor")

    if(len(kernel_tensor.shape) == 4):
        input_channels = data_tensor.shape[1]
        kernel_channels = kernel_tensor.shape[1]
    elif(len(kernel_tensor.shape) == 3):
        input_channels = data_tensor.shape[1]
        kernel_channels = kernel_tensor.shape[0]
    else:
        raise Exception("Dimension mismatch for kernel tensor")

    if(input_channels != kernel_channels):
        raise Exception(
            "Number of channels for input and kernel should be same")

    if(len(kernel_tensor.shape) == 3):
        # Add new axis for the tensor
        # Needed to broadcast the kernel across each data sample
        kernel_tensor = kernel_tensor[np.newaxis, :, :, :]

    # Pad width to use for x and y axis
    pad_width = 1
    number_of_data = data_tensor.shape[0]
    channels = data_tensor.shape[1]

    x, y = data_tensor.shape[2:4]
    m, n = kernel_tensor.shape[2:4]

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

    for i in range(x):
        for j in range(y):
            data_block = padded_tensor[:, :, i:i+m, j:j+m]
            convolution = np.multiply(data_block, kernel_tensor)

            result[:, :, i, j] = np.sum(convolution, axis=(2, 3))

    return result
