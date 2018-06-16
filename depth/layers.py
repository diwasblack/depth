import numpy as np

from .activations import (
    hyperbolic_tangent, hyperbolic_tangent_derivative,
    relu, relu_derivative,
    sigmoid_function, sigmoid_function_derivative,
    softmax_function
)


class LayerBase():
    """
    Base class for the neural network layers
    """

    def __init__(self, input_units, output_units, use_softmax=False):
        self.input_units = input_units
        self.output_units = output_units

        # Randomly initialize weights in the range [-0.5, 0.5]
        # Add bias unit to each layer
        self.weights = -0.5 + \
            np.random.rand(self.output_units, self.input_units+1)

        self.activation_function = None
        self.activation_function_derivative = None

        # Store the input values during the forward_pass
        self.input_values = None
        # Store the activation value calculated during the forward pass
        self.activation_values = None

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

    def update_weights(self, weight_update):
        """
        Update the weight of the layer
        """

        self.weights = self.weights - weight_update

    def backprop(self, delta, eta):

        if(self.weights.shape[0] != delta.shape[0]):
            # Remove the delta for the bias
            delta = delta[1:, :]

        dloss_dz = self.calcuate_dloss_dz(delta)

        # Calculate the delta for the next layer
        # Must be done before weight update
        new_delta = np.dot(self.weights.T, dloss_dz)

        # Calculate weight update
        weight_update = eta * np.dot(delta, self.input_values.T)

        self.update_weights(weight_update)

        # Clean up memory of input and activation values
        self.input_values = None
        self.activation_values = None

        return new_delta


class NonLinearBackprop():
    def calcuate_dloss_dz(self, delta):
        derivative_values = self.activation_function_derivative(
            self.activation_values)

        dloss_dz = np.multiply(delta, derivative_values)

        return dloss_dz


class TanhLayer(NonLinearBackprop, LayerBase):
    def __init__(self, *args):
        super().__init__(*args)
        # Assign hyperbolic_tangent function to be the activation function
        self.activation_function = hyperbolic_tangent
        self.activation_function_derivative = hyperbolic_tangent_derivative


class ReluLayer(NonLinearBackprop, LayerBase):
    def __init__(self, *args):
        super().__init__(*args)
        # Assign relu function to be the activation function
        self.activation_function = relu
        self.activation_function_derivative = relu_derivative


class SigmoidLayer(NonLinearBackprop, LayerBase):
    def __init__(self, *args):
        super().__init__(*args)
        # Assign sigmoid function to be the activation function
        self.activation_function = sigmoid_function
        self.activation_function_derivative = sigmoid_function_derivative


class LinearBackprop():
    def calcuate_dloss_dz(self, delta):
        return delta


class LinearLayer(LinearBackprop, LayerBase):
    def __init__(self, *args):
        super().__init__(*args)
        # Return the input as without modification
        self.activation_function = lambda x: x


class SoftmaxLayer(LinearBackprop, LayerBase):
    def __init__(self, *args):
        super().__init__(*args)
        self.activation_function = softmax_function
