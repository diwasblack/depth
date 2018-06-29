import math

import numpy as np


class DenseLayerBase():
    """
    Base class for dense neural network layers
    """

    def __init__(self, input_units, output_units, regularizer=None):
        self.input_units = input_units
        self.output_units = output_units

        # Store the result of previous gradient update
        self.previous_updates = None

        # Invoke the weight initializer
        self.initialize_weights(input_units, output_units)

        self.activation_function = None
        self.activation_function_derivative = None

        # Store the input values during the forward_pass
        self.input_values = None
        # Store the activation value calculated during the forward pass
        self.activation_values = None

        # Store the regularizer to use with the layer
        self.regularizer = regularizer

    def initialize_weights(self, input_units, output_units):
        """
        A default implementation of weight initializer
        """

        # Randomly initialize weights in the range [-0.5, 0.5]
        # Add bias unit to each layer
        self.weights = -0.5 + \
            np.random.rand(self.output_units, self.input_units+1)

    def forward_pass(self, input_matrix):
        """
        Layer method to compute the activation values during forward pass
        """

        # Augment input data to include activation from bias unit
        bias_units = np.ones((1, input_matrix.shape[1]))

        self.input_values = np.vstack((bias_units, input_matrix[:]))

        # Compute the linear combination of weight and layer input
        linear_combination = np.dot(self.weights, self.input_values)

        # Compute the activation value
        self.activation_values = self.activation_function(linear_combination)

        return self.activation_values

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


class NonLinearBackprop():
    def calcuate_dloss_dz(self, delta):
        derivative_values = self.activation_function_derivative(
            self.activation_values)

        dloss_dz = np.multiply(delta, derivative_values)

        return dloss_dz


class LinearBackprop():
    def calcuate_dloss_dz(self, delta):
        return delta


class XavierWeightInitializer():
    """
    An implementation of Xavier weight initialization

    See:
    http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
    """

    def initialize_weights(self, input_units, output_units):
        self.weights = np.random.rand(output_units, input_units+1) * \
                math.sqrt(1.0 / input_units)


class HeWeightInitializer():
    """
    An implementation of He weight initialization

    See:
    https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf
    """

    def initialize_weights(self, input_units, output_units):
        self.weights = np.random.rand(output_units, input_units+1) * \
                math.sqrt(2.0 / input_units)
