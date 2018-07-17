import logging

import numpy as np

from .activations import relu, relu_derivative
from .convolution import convolve_tensors, transposed_convolution, convolve2d


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
        self.output_shape = np.array([self.filters, *self.input_shape[1:]])

        # Assumes the channel will be in the first position
        self.channels = self.input_shape[0]

    def construct_layer(self):
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
        z = np.zeros((self.samples, self.filters, *self.input_shape[1:]))
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

        # Perform 2D convolution of input block with kernel_tensor
        gradient = convolve_tensors(self.input_values, dloss_dz)

        # Use number of samples as normalization_factor for gradient
        normalization_factor = self.input_values.shape[0]

        # Average gradient across all samples
        gradient_avg = np.sum(gradient, axis=0) / normalization_factor

        # Perform 2D convolution of delta with kernel_tensor
        delta = convolve_tensors(dloss_dz, self.kernel_tensor)

        # Cleanup memory
        self.input_values = None
        self.activation_values = None

        return gradient_avg, delta

    def get_regularized_cost(self):
        pass
