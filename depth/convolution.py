import numpy as np


def convolve2d(tensor1, tensor2):
    """
    Compute the 2D convolution of the tensor 1 with tensor 2
    """

    # Pad width to use for x and y axis
    pad_width = 1
    number_of_data = tensor1.shape[0]

    x, y = tensor1.shape[-2:]
    m, n = tensor2.shape[-2:]

    # For forward pass
    if(len(tensor1.shape) == 4 and len(tensor2.shape) == 4):
        if(tensor1.shape[0] != tensor2.shape[0]):
            filters = tensor2.shape[0]
            channels = tensor1.shape[1]
            tensor1 = tensor1[:, np.newaxis, :, :, :]
            tensor2 = tensor2[np.newaxis, :, :, :, :]
        else:
            raise Exception("Dimesion mismatch")

    # For delta for next layer
    elif(len(tensor1.shape) == 5 and len(tensor2.shape) == 4):
        filters = tensor2.shape[0]
        channels = tensor1.shape[2]
        tensor2 = tensor2[np.newaxis, :, :, :, :]

    # For gradient
    elif(len(tensor1.shape) == 4 and len(tensor2.shape) == 5):
        filters = tensor2.shape[1]
        channels = tensor1.shape[1]
        tensor1 = tensor1[:, np.newaxis, :, :, :]

    else:
        raise Exception("Unknown dimension received")

    number_of_data = tensor1.shape[0]

    x = x - m + 2 * pad_width + 1
    y = y - m + 2 * pad_width + 1

    result = np.zeros(
        (number_of_data, filters, channels, x, y), dtype=np.float32)

    # Pad the data with zeros
    pads = [(0, 0), (0, 0), (0, 0), (pad_width,
                                     pad_width), (pad_width, pad_width)]
    padded_tensor = np.pad(
        tensor1, pads, mode="constant", constant_values=0)

    for i in range(x):
        for j in range(y):
            data_block = padded_tensor[:, :, :, i:i+m, j:j+m]
            convolution = np.multiply(data_block, tensor2)

            result[:, :, :, i, j] = np.sum(convolution, axis=(3, 4))

    return result
