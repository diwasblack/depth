import logging

import numpy as np

from .activations import relu, relu_derivative
from .convolution import convolve_tensors


class Convolution2D():
    """
    Implementation of 2D convolution layer
    """

    def __init__(self, filters, kernel_shape, input_shape=None, strides=(1, 1), activation="relu"):
        self.activation = activation
        self.filters = filters
        self.kernel_shape = np.array(kernel_shape)
        self.strides = strides
        self.input_shape = np.array(input_shape)

        self.output_shape = np.array(self.filters, *self.input_shape[1:])

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

        # Perform the 2D convolution of the tensors
        z = convolve_tensors(input_data, self.weights)

        # Apply activation function
        activation = self.activation_function(z)

        return activation

    def backprop(self, delta, optimizer):
        pass

    def get_regularized_cost(self):
        pass
