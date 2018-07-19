import numpy as np

from .activations import (
    hyperbolic_tangent, hyperbolic_tangent_derivative,
    relu, relu_derivative,
    leaky_relu, leaky_relu_derivative,
    sigmoid_function, sigmoid_function_derivative,
    softmax_function)
from .initializers import (
    HeWeightInitializer, XavierWeightInitializer)
from .layer_base import BaseLayer


class DenseLayer(BaseLayer):
    """
    Base class for dense neural network layers
    """

    def __init__(self, units=32, input_dimension=None, regularizer=None,
                 activation="tanh", **kwargs):

        super().__init__()

        self.input_units = input_dimension
        self.output_units = units
        self.activation = activation
        self.kwargs = kwargs
        self.output_shape = np.array([self.output_units, 1])

        self.non_linear_activation = True

        # Store the first moment and second moment
        self.first_moment = 0
        self.second_moment = 0

        # Store the input and activation value during the forward pass
        self.input_values = None
        self.activation_values = None

        # Store the regularizer to use with the layer
        self.regularizer = regularizer

    def initialize_layer_weights(self):
        # Check if input and output units are properly initalized
        if(not(self.input_units)):
            raise Exception("Input units not provided")

        if(not(self.output_units)):
            raise Exception("Output units not provided")

        # Select an initializer for the weights
        if(self.activation in ["relu", "leakyrelu"]):
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

    def construct_layer(self, previous_layer=None):
        """
        Helper function to construct the layer as per the initialized values
        """

        if(previous_layer is not None):
            self.input_units = previous_layer.get_output_shape().prod()

        # Initialize the weights
        self.initialize_layer_weights()

        # Assign the activation function to use
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
        elif(self.activation == "linear"):
            # Return the input as without modification
            self.activation_function = lambda x: x
            self.non_linear_activation = False
        elif(self.activation == "softmax"):
            self.activation_function = softmax_function
            self.non_linear_activation = False
        else:
            raise Exception("Unknown activation function")

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

    def backprop(self, delta):
        """
        Propagate the delta through the layer to calculate delta for next
        layer and update the weight of current layer as well.
        """

        if(self.weights.shape[0] != delta.shape[0]):
            # Remove the delta for the bias
            delta = delta[1:, :]

        # Use number of samples as normalization_factor for gradient
        normalization_factor = self.input_values.shape[1]

        # Propagate delta through the activation function
        if(self.non_linear_activation):
            derivative_values = self.activation_function_derivative(
                self.activation_values)
            dloss_dz = np.multiply(delta, derivative_values)
        else:
            dloss_dz = np.copy(delta)

        # Calculate the delta for the next layer before weight update
        delta = np.dot(self.weights.T, dloss_dz)

        # Calculate the gradient for current layer
        gradient = np.dot(dloss_dz, self.input_values.T) / normalization_factor

        # Add gradient from regularizer
        if(self.regularizer):
            regularizer_gradient = self.regularizer.get_derivative_value(
                self.weights[:, 1:])

            # Do not regularizer the bias weights
            bias_gradient = np.zeros((self.weights.shape[0], 1))
            regularizer_gradient = np.hstack(
                (bias_gradient, regularizer_gradient))

            gradient += regularizer_gradient

        # Clean up memory of input and activation values
        self.input_values = None
        self.activation_values = None

        return gradient, delta

    def update_weights(self, weight_update):
        # Update weight of current layer
        self.weights = self.weights - weight_update
