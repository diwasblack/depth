import math

import numpy as np

from .activations import (
    hyperbolic_tangent, hyperbolic_tangent_derivative,
    relu, relu_derivative,
    leaky_relu, leaky_relu_derivative,
    sigmoid_function, sigmoid_function_derivative,
    softmax_function)


def HeWeightInitializer(input_units, output_units):
    """
    An implementation of He weight initialization

    See:
    https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf
    """

    return np.random.rand(output_units, input_units) * \
        math.sqrt(2.0 / input_units)


def XavierWeightInitializer(input_units, output_units):
    """
    An implementation of Xavier weight initialization

    See:
    http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
    """
    return np.random.rand(output_units, input_units) * \
        math.sqrt(1.0 / input_units)


class DenseLayerBase():
    """
    Base class for dense neural network layers
    """

    def __init__(self, input_units, output_units, regularizer=None,
                 activation="tanh", **kwargs):
        self.input_units = input_units
        self.output_units = output_units
        self.activation = activation
        self.kwargs = kwargs

        self.non_linear_activation = True

        # Store the result of previous gradient update
        self.previous_updates = None

        # Initialize the weights of the layer
        self.initialize_weights()

        # Construct the layer as per the configuration
        self.prepare_model()

        # Store the input values during the forward_pass
        self.input_values = None
        # Store the activation value calculated during the forward pass
        self.activation_values = None

        # Store the regularizer to use with the layer
        self.regularizer = regularizer

    def prepare_model(self):
        if(self.activation == "sigmoid"):
            self.activation_function = sigmoid_function
            self.activation_function_derivative = sigmoid_function_derivative

        elif(self.activation == "tanh"):
            self.activation_function = hyperbolic_tangent
            self.activation_function_derivative = hyperbolic_tangent_derivative

        elif(self.activation == "relu"):
            self.activation_function = relu
            self.activation_function_derivative = relu_derivative

        elif(self.activation == "leakyrelu"):
            alpha = self.kwargs.get("alpha", 0.0)
            # Assign relu function to be the activation function
            self.activation_function = leaky_relu(alpha=alpha)
            self.activation_function_derivative = leaky_relu_derivative(
                alpha=alpha)

        elif(self.activation == "leakyrelu"):
            # Assign sigmoid function to be the activation function
            self.activation_function = sigmoid_function
            self.activation_function_derivative = sigmoid_function_derivative

        elif(self.activation == "linear"):
            # Return the input as without modification
            self.activation_function = lambda x: x
            self.non_linear_activation = False

        elif(self.activation == "softmax"):
            self.activation_function = softmax_function
            self.non_linear_activation = False

    def initialize_weights(self):
        """
        Initialize the weights
        """

        # Select an initializer for the weights
        if(self.activation in ["relu", "leaky_relu"]):
            self.weights = HeWeightInitializer(
                self.input_units+1, self.output_units)

        elif(self.activation in ["tanh", "sigmoid"]):
            self.weights = XavierWeightInitializer(
                self.input_units+1, self.output_units)

        else:
            # Randomly initialize weights in the range [-0.5, 0.5]
            # Add bias unit to each layer
            self.weights = -0.5 + \
                np.random.rand(self.output_units, self.input_units+1)

    def forward_pass(self, input_matrix, store_values=True):
        """
        Layer method to compute the activation values during forward pass
        """

        # Augment input data to include activation from bias unit
        bias_units = np.ones((1, input_matrix.shape[1]))
        input_values = np.vstack((bias_units, input_matrix[:]))

        # Compute the linear combination of weight and layer input
        linear_combination = np.dot(self.weights, input_values)

        activation_values = self.activation_function(linear_combination)

        # Store values for backpropagation
        if(store_values):
            self.input_values = input_values
            self.activation_values = activation_values

        return activation_values

    def calcuate_dloss_dz(self, delta):
        if(not(self.non_linear_activation)):
            return delta

        derivative_values = self.activation_function_derivative(
            self.activation_values)

        dloss_dz = np.multiply(delta, derivative_values)

        return dloss_dz

    def backprop(self, delta, optimizer):
        """
        Propagate the delta through the layer to calculate delta for next
        layer and update the weight of current layer as well.
        """

        if(self.weights.shape[0] != delta.shape[0]):
            # Remove the delta for the bias
            delta = delta[1:, :]

        # Use number of samples as normalization_factor for gradient
        normalization_factor = self.input_values.shape[1]

        # Calculate gradient for activation function
        dloss_dz = self.calcuate_dloss_dz(delta)

        # Calculate the delta for the next layer before weight update
        delta = np.dot(self.weights.T, dloss_dz)

        # Calculate the gradient for current layer
        gradient = np.dot(dloss_dz, self.input_values.T) / normalization_factor

        # Add gradient from regularizer
        if(self.regularizer):
            regularizer_gradient = self.regularizer.get_derivative_value(
                self.weights[:, 1:])

            zeroes = np.zeros((self.weights.shape[0], 1))

            # Do not regularizer the bias weights
            regularizer_gradient = np.hstack((zeroes, regularizer_gradient))

            gradient += regularizer_gradient

        # Calculate the weight update for the layer
        weight_update = optimizer.get_updates(gradient, self.previous_updates)

        # Update weight of current layer
        self.weights = self.weights - weight_update

        # Store weight update
        self.previous_updates = np.copy(weight_update)

        # Clean up memory of input and activation values
        self.input_values = None
        self.activation_values = None

        return delta

    def get_regularized_cost(self):
        """
        Get the cost of the layer using the regularizer provided
        """

        if(self.regularizer):
            return self.regularizer.get_cost(self.weights)
        else:
            return 0


class DenseLayer():
    """
    An abstration of Dense layer model
    """

    def __init__(self, activation_function="tanh", units=64,
                 input_dimension=None, **kwargs):
        self.activation_function = activation_function
        self.units = units
        self.input_dimension = input_dimension

        self.kwargs = kwargs

    def get_layer_obj(self, previous_units=None):
        activation_function = self.activation_function
        units = self.units
        if(not(previous_units)):
            previous_units = self.input_dimension

        layer = DenseLayerBase(previous_units, units,
                               activation=activation_function, **self.kwargs)

        return layer
