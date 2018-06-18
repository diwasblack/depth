import logging
import pickle
import gzip

import numpy as np

from .layers import (
    TanhLayer, ReluLayer, LinearLayer, SigmoidLayer, SoftmaxLayer
)
from .loss_functions import mean_squared_error, cross_entropy
from .helpers import vector_to_label
from .metrics import categorical_accuracy


class NeuralNet():
    """
    Implementation of sequential backpropagation neural network
    """

    def __init__(self):
        # List to hold the layers
        self.layers = []
        self.learning_rate = None

        self.loss_function = None
        self.loss_function_derivative = None

    def add_layer(self, activation_function="tanh", units=64, input_dimension=None):
        if(not(self.layers)):
            previous_units = input_dimension
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

    def dump_layer_weights(self, layers_filename):
        """
        Update layer weights periodically in a file
        """
        logging.info("Starting a backup of layers to a file")

        with gzip.open(layers_filename, "wb") as file:
            pickle.dump(self.layers, file)

        logging.info("Layer backup to the file completed")

    def load_layer_weights(self, layers_filename):
        """
        Load layer information from a file
        """

        logging.info("Trying to load layers from the file")

        with gzip.open(layers_filename, "rb") as file:
            self.layers = pickle.load(file)

        logging.info("Succefully loaded layers from file")

    def prediction_accuracy(self, predicted_output, target_matrix):
        """
        Convert the output probabilites and calculate the accuracy
        """

        # Convert output probabilites to labels
        predicted_labels = vector_to_label(predicted_output)

        # TODO find a better way
        # Convert target probabilites to labels
        target_labels = vector_to_label(target_matrix)

        # Calculate the accuracy
        accuracy = categorical_accuracy(predicted_labels, target_labels)

        return accuracy

    def train(self, input_matrix, target_matrix, max_iterations=1000,
              logging_frequency=1000, weight_backup_frequency=100,
              layers_filename="", training_logger=None
              ):
        """
        Train the neural network contructed

        Inputs:
        input_matrix: a (n * N) matrix
        target_matrix: a (m * N) matrix

        max_iterations: the maximum number of iterations
        logging_frequency: the frequency of logging the training cost
        weight_backup_frequency: the frequency of storing the weights to a file
        layers_filename: the file to use to store the layers

        training_costs: a list to hold the costs during the training
        training_accuracies: a list to hold the prediction accuracy during
            training

        """
        number_of_iterations = 1

        if(layers_filename):
            store_layers = True
        else:
            store_layers = False

        while(True):
            # Propagate the input forward
            predicted_output = self.forward_pass(input_matrix)

            # Calculate delta at the final layer
            delta = self.loss_function_derivative(
                predicted_output, target_matrix)

            loss = self.loss_function(predicted_output, target_matrix)
            accuracy = self.prediction_accuracy(
                predicted_output, target_matrix)

            log_message = "Loss: {}, Accuracy:{}".format(
                loss, accuracy)

            training_logger.info(log_message)

            if(number_of_iterations % logging_frequency == 0):
                logging.info("Cost: {}".format(loss))

            # Update weights using backpropagation
            self.backpropagation(delta)

            # NOTE dump layers only after backpropagation update
            if(store_layers and
                    number_of_iterations % weight_backup_frequency == 0):
                self.dump_layer_weights(layers_filename)

            if(loss < self.error_threshold or
                    number_of_iterations == max_iterations):
                break

            number_of_iterations += 1

    def predict(self, input_matrix):
        return self.forward_pass(input_matrix)
