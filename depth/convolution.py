import logging

import numpy as np


def convolve_tensors(data_tensor, kernel_tensor):
    """
    Compute the 2D convolution of the data_tensor with kernel_tensor

    data_tensor = N * c * x * y tensor
    kernel_tensor = f * c * m * n

    where,
    N = number of samples
    c = number of channels
    f = number of filters

    x, y = size of data
    m, n = filter size
    """

    # Pad width to use for x and y axis
    pad_width = 1
    number_of_data = data_tensor.shape[0]
    number_of_filters = kernel_tensor.shape[0]

    m, n = kernel_tensor.shape[2:4]
    x, y = data_tensor.shape[2:4]

    result = np.zeros((number_of_data, number_of_filters,
                       x, y), dtype=np.float32)

    # Pad the data with zeros
    pads = [(0, 0), (0, 0), (pad_width, pad_width), (pad_width, pad_width)]
    padded_tensor = np.pad(
        data_tensor, pads, mode="constant", constant_values=0)

    # NOTE: Verify/Improve this
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
