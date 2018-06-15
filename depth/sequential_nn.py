import logging

import numpy as np

from .layers import (
    TanhLayer, ReluLayer, LinearLayer, SigmoidLayer, SoftmaxLayer
)

from .loss_functions import mean_squared_error, cross_entropy


class SequentialNeuralNet():
    """
    Implementation of sequential backpropagation neural network
    """

    def __init__(self, input_dimension):
        # List to hold the layers
        self.layers = []
        self.input_dimension = input_dimension
        self.learning_rate = None

        self.loss_function = None

    def add_layer(self, activation_function="tanh", units=64):
        if(not(self.layers)):
            previous_units = self.input_dimension
        else:
            previous_units = self.layers[-1].output_units

        if(activation_function == "tanh"):
            layer = TanhLayer(previous_units, units)

        elif(activation_function == "relu"):
            layer = ReluLayer(previous_units, units)

        elif(activation_function == "sigmoid"):
            layer = SigmoidLayer(previous_units, units)

        elif(activation_function == "linear"):
            layer = LinearLayer(previous_units, units)

        elif(activation_function == "softmax"):
            layer = SoftmaxLayer(previous_units, units)

        # Add layer to the list
        self.layers.append(layer)

    def compile(self, loss="mean_squared_error", learning_rate=0.001,
                error_threshold=0.001):
        self.output_dimension = self.layers[-1].output_units
        self.learning_rate = learning_rate

        self.error_threshold = error_threshold

        self.number_of_layers = len(self.layers)

        if(loss == "mean_squared_error"):
            self.loss_function = mean_squared_error
            self.loss_function_derivative = lambda x, y: x - y

        if(loss == "cross_entropy"):
            self.loss_function = cross_entropy
            self.loss_function_derivative = lambda x, y: x - y

    def forward_pass(self, input_matrix):
        output = np.copy(input_matrix)
        for layer in self.layers:
            output = layer.forward_pass(output)

        return output

    def backpropagation(self, delta):
        """
        Propagate delta through the layers
        """
        for layer in reversed(self.layers):
            # Propagate delta through layers
            delta = layer.backprop(delta, self.learning_rate)

    def train(self, input_matrix, target_matrix, logging_frequency=1000):
        number_of_iterations = 0

        while(True):
            # Propagate the input forward
            predicted_output = self.forward_pass(input_matrix)

            # Calculate delta at the final layer
            delta = self.loss_function_derivative(
                predicted_output, target_matrix)

            # Calculate the loss occured
            loss = self.loss_function(predicted_output, target_matrix)

            if(number_of_iterations % logging_frequency == 0):
                logging.info("Cost: {}".format(loss))

            if(loss < self.error_threshold):
                break

            # Update weights using backpropagation
            self.backpropagation(delta)

            number_of_iterations += 1

    def predict(self, input_matrix):
        return self.forward_pass(input_matrix)
