import numpy as np


class MaxPooling():
    """
    Implementation for the max pooling layer
    """

    def __init__(self, pool_size=(2, 2), strides=(2, 2), padding=0):
        self.filters = None
        self.samples = None
        self.pool_size = np.array(pool_size)
        self.padding = padding

        self.strides = np.array(strides)

        self.input_shape = None
        self.output_shape = None

        self.output_size = None

    def construct_layer(self, previous_layer=None):
        if(previous_layer is None):
            raise Exception("Received None for previous_layer")

        self.input_shape = previous_layer.get_output_shape()

        self.filters = self.input_shape[0]
        x = self.input_shape[1]
        y = self.input_shape[2]

        m = self.pool_size[0]
        n = self.pool_size[1]
        p = self.padding

        x_new = ((x - m + 2 * p) / self.strides[0]) + 1
        y_new = ((y - n + 2 * p) / self.strides[1]) + 1

        self.output_size = np.array([x_new, y_new], dtype=np.int)
        self.output_shape = np.array(
            [self.filters, x_new, y_new], dtype=np.int)

    def get_output_shape(self):
        return self.output_shape

    def forward_pass(self, input_data, store_values=False):
        """
        Compute the max pooling for forward layer
        """
        self.samples = input_data.shape[0]

        # Store the index of max pooling
        # Needs to be integer type
        max_map = np.zeros((self.samples, *self.output_shape, 2), dtype=np.int)
        max_values = np.zeros((self.samples, *self.output_shape))

        x, y = self.output_size
        m, n = self.pool_size
        stride_x = self.strides[0]
        stride_y = self.strides[1]

        for index in range(self.samples):
            for i in range(x):
                for j in range(y):
                    for f in range(self.filters):
                        i_index = stride_x * i
                        j_index = stride_y * j

                        data_block = input_data[
                            index, f, i_index:i_index+m, j_index:j_index+n]

                        # Calculate the max index in the pool
                        max_index = np.unravel_index(
                            np.argmax(data_block, axis=None), data_block.shape)

                        max_value = data_block[max_index[0], max_index[1]]

                        # Convert tuple to list
                        max_index = list(max_index)

                        # Store the max_index with offset
                        max_index[0] += i_index
                        max_index[1] += j_index

                        max_values[index, f, i, j] = max_value
                        max_map[index, f, i, j, :] = np.array(max_index)

        # Store the index of maximum value
        if(store_values):
            self.max_map = max_map

        return max_values

    def backprop(self, delta):
        new_delta = np.zeros((self.samples, *self.input_shape))

        x, y = self.output_size

        for index in range(self.samples):
            for i in range(x):
                for j in range(y):
                    for f in range(self.filters):
                        max_index = self.max_map[index, f, i, j]
                        delta_x = delta[index, f, i, j]
                        new_delta[index, f, max_index[0], max_index[1]] = delta_x

        return None, new_delta

    def get_regularized_cost(self):
        return 0
