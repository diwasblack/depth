import numpy as np

from .activations import relu, relu_derivative
from .convolution import convolve2d


class Convolution2D():
    """
    Implementation of 2D convolution layer
    """

    def __init__(self, filters, kernel_shape, input_shape=None, strides=(1, 1),
                 activation="relu", **kwargs):
        self.activation = activation
        self.kwargs = kwargs

        self.filters = filters
        self.kernel_shape = np.array(kernel_shape)
        self.strides = strides

        self.non_linear_activation = True

        self.input_shape = np.array(input_shape)
        self.image_size = np.array(input_shape[1:])
        self.output_shape = np.array([self.filters, *self.image_size])

        # Store the first moment and second moment
        self.first_moment = 0
        self.second_moment = 0

        self.regularizer = regularizer

    def construct_layer(self, previous_layer=None):
        if(previous_layer is not None):
            self.input_shape = previous_layer.get_output_shape()

        # Assumes the channel will be in the first position
        self.channels = self.input_shape[0]

        if(self.activation == "relu"):
            self.activation_function = relu
            self.activation_function_derivative = relu_derivative

        self.initialize_layer_weights()

    def initialize_layer_weights(self):
        input_units = self.input_shape.prod()

        if(self.activation in ["relu", "leakyrelu"]):
            variance = np.sqrt(2.0 / input_units)

        elif(self.activation in ["sigmoid", "tanh"]):
            variance = np.sqrt(1.0 / input_units)

        # Constuct a weights/kernel tensor
        self.weights = np.random.randn(
            self.filters, self.channels, *self.kernel_shape) * variance

    def get_output_shape(self):
        return self.output_shape

    def forward_pass(self, input_data, store_values=False):
        """
        Propagate the input data through the layer
        """

        self.samples = input_data.shape[0]

        # Perform the 2D convolution of the tensors
        z = np.zeros((self.samples, self.filters, *self.image_size))
        for f in range(self.filters):
            z_filter = convolve2d(input_data, self.weights[f])

            # Sum up the values from the channels/filters
            z[:, f, :, :] = np.sum(z_filter, axis=1)

        # Apply activation function
        activation_values = self.activation_function(z)

        if(store_values):
            self.input_values = input_data
            self.activation_values = activation_values

        return activation_values

    def backprop(self, delta):
        # Propagate delta through the activation function
        if(self.non_linear_activation):
            derivative_values = self.activation_function_derivative(
                self.activation_values)
            dloss_dz = np.multiply(delta, derivative_values)
        else:
            dloss_dz = np.copy(delta)

        gradient = np.zeros((
            self.samples, self.filters, self.channels, *self.kernel_shape),
            dtype=np.float32)

        delta = np.zeros((
            self.samples, self.channels, *self.image_size), dtype=np.float32)

        for f in range(self.filters):
            # Seperate the delta for each channel/filter
            dloss_dz_map = dloss_dz[:, f, :, :]

            # Repeat the delta across each input channel/filter
            dloss_dz_block = np.repeat(
                dloss_dz_map[:, np.newaxis, :, :], self.input_shape[0], axis=1)

            for i in range(self.samples):
                input_delta_conv = convolve2d(
                    self.input_values[i:i+1], dloss_dz_block[i])

                delta_weights_conv = convolve2d(
                    dloss_dz_block[i:i+1], self.weights[f])

                # Add delta from each filter
                delta[i] += delta_weights_conv[0]

                gradient[i, f, :, :, :] = input_delta_conv[0]

        # Average gradient across all samples
        # Use number of samples as normalization_factor for gradient
        gradient_avg = np.sum(gradient, axis=0) / self.samples

        # Cleanup memory
        self.input_values = None
        self.activation_values = None

        return gradient_avg, delta

    def get_regularized_cost(self):
        if(self.regularizer):
            return self.regularizer.get_cost(self.weights)
        else:
            return 0

    def update_weights(self, weight_update):
        # Update weight of the layer
        self.weights = self.weights - weight_update


class Flatten():
    def __init__(self):
        self.input_shape = None

    def construct_layer(self, previous_layer=None):
        if(previous_layer is None):
            raise Exception("Previous layer object is empty")

        self.input_shape = previous_layer.get_output_shape()

        units = self.input_shape.prod()
        self.output_shape = np.array([units, 1])
        self.output_units = units

    def forward_pass(self, input_data, store_values=False):
        self.samples = input_data.shape[0]

        # Flatten the data
        flattened_data = input_data.reshape(self.samples, self.output_units)

        # Transpose the data into column format and return
        return flattened_data.T

    def backprop(self, delta):
        # Truncate the delta for the bias
        delta = delta[1:, :]

        # Convert the delta in row wise format
        delta = delta.T

        # Reshape the flattened deltas
        delta = delta.reshape(self.samples, *self.input_shape)

        return None, delta

    def get_output_shape(self):
        return self.output_shape

    def get_regularized_cost(self):
        return 0
