import numpy as np

from .activations import relu, relu_derivative
from .convolution import convolve2d
from .layer_base import BaseLayer


class Convolution2D(BaseLayer):
    """
    Implementation of 2D convolution layer
    """

    def __init__(self, filters, kernel_shape, input_shape=None, strides=(1, 1),
                 activation="relu", regularizer=None, **kwargs):
        super().__init__()

        self.activation = activation
        self.kwargs = kwargs

        self.filters = filters
        self.kernel_shape = np.array(kernel_shape)
        self.strides = strides

        self.non_linear_activation = True

        self.input_shape = np.array(input_shape)
        self.image_size = None
        self.output_shape = None

        # Store the first moment and second moment
        self.first_moment = 0
        self.second_moment = 0

        # Store the input and activation values during the forward pass
        self.input_values = None
        self.activation_values = None

        # Store the regularizer to use with the layer
        self.regularizer = regularizer

    def initialize_layer_weights(self):
        input_units = self.input_shape.prod()

        # Calculate the variance to use for weights
        variance = self.get_weights_variance(input_units)

        # Constuct a weights/kernel tensor
        self.weights = np.random.randn(
            self.filters, self.channels, *self.kernel_shape) * variance

    def construct_layer(self, previous_layer=None):
        if(previous_layer is not None):
            self.input_shape = previous_layer.get_output_shape()

        # Calculate the output_shape
        self.image_size = np.array(self.input_shape[1:])
        self.output_shape = np.array([self.filters, *self.image_size])

        # Assumes the channel will be in the first position
        self.channels = self.input_shape[0]

        if(self.activation == "relu"):
            self.activation_function = relu
            self.activation_function_derivative = relu_derivative

        self.initialize_layer_weights()

    def forward_pass(self, input_data, store_values=False):
        """
        Propagate the input data through the layer
        """

        self.samples = input_data.shape[0]

        z = convolve2d(input_data, self.weights)

        # Sum up value from input channels/filters
        z = np.sum(z, axis=2)

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

        # Repeat the delta for each input channel/filter
        dloss_dz_repeat = np.repeat(
            dloss_dz[:, :, np.newaxis, :, :], self.input_shape[0], axis=2)

        # Convolve the delta with the weights
        delta_weights_conv = convolve2d(dloss_dz_repeat, self.weights)

        # Convolve the input values with delta
        gradient = convolve2d(self.input_values, dloss_dz_repeat)

        # Sum delta from each output filter
        delta = np.sum(delta_weights_conv, axis=1)

        # Average gradient across all samples
        layer_gradient = np.sum(gradient, axis=0) / self.samples

        # Add gradient from regularizer
        if(self.regularizer):
            layer_gradient += self.regularizer.get_derivative_value(
                self.weights
            )

        # Cleanup memory
        self.input_values = None
        self.activation_values = None

        return layer_gradient, delta

    def update_weights(self, weight_update):
        # Update weight of the layer
        self.weights = self.weights - weight_update


class Flatten(BaseLayer):
    def __init__(self):
        super().__init__()
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
