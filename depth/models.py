import logging
import pickle
import gzip

import numpy as np

from .loss_functions import mean_squared_error, cross_entropy
from .helpers import vector_to_label
from .metrics import categorical_accuracy
from .optimizers import SGD


class Sequential():
    """
    Implementation of sequential backpropagation neural network
    """

    def __init__(self):
        # List to hold the layers
        self.layers = []

        self.loss_function = None
        self.loss_function_derivative = None

        # Function to use to transform the final output
        self.output_function = None

        self.optimizer = None

    def add_layer(self, layer):
        if(self.layers):
            previous_units = self.layers[-1].output_units
            layer.input_units = previous_units

        layer.construct_layer()

        # Add layer to the list
        self.layers.append(layer)

    def compile(self, loss="mean_squared_error", error_threshold=0.001,
                optimizer=None):
        self.output_dimension = self.layers[-1].output_units

        self.error_threshold = error_threshold

        self.number_of_layers = len(self.layers)

        if(loss == "mean_squared_error"):
            # NOTE for now assumes that the final layer to be linear
            self.loss_function = mean_squared_error
            self.loss_function_derivative = lambda x, y: x - y

        if(loss == "cross_entropy"):
            # NOTE for now assumes that the final layer to be softmax
            self.loss_function = cross_entropy
            self.loss_function_derivative = lambda x, y: x - y

        # Assign an optimizer to use for updating parameters
        if(not(optimizer)):
            self.optimizer = SGD()
        else:
            self.optimizer = optimizer

    def forward_pass(self, input_matrix, store_values=True):
        output = np.copy(input_matrix)
        for layer in self.layers:
            output = layer.forward_pass(output, store_values=store_values)

        return output

    def backpropagation(self, delta):
        """
        Propagate delta through the layers
        """
        for layer in reversed(self.layers):
            # Propagate delta through layers
            delta = layer.backprop(delta, self.optimizer)

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

    def get_regularization_cost(self):
        """
        Iterate through the layers to obtain the cost of the model
        """

        model_cost = 0

        for layer in self.layers:
            model_cost += layer.get_regularized_cost()

        return model_cost

    def train(self, input_matrix, target_matrix, max_iterations=1000,
              logging_frequency=100, update_frequency=100, decay_frequency=500,
              layers_filename="", training_logger=None):
        """
        Train the neural network contructed

        Inputs:
        input_matrix: a (n * N) matrix
        target_matrix: a (m * N) matrix

        max_iterations: the maximum number of iterations
        logging_frequency: the frequency of logging the training cost
        decay_frequency: the frequency to decay learning rate at
        update_frequency: the frequency of storing the weights to a file and decaying the learning
            rate
        layers_filename: the file to use to store the layers

        training_logger: the logger object to use for report training information

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

            # Add the regularied loss
            loss += self.get_regularization_cost()

            if(training_logger):
                accuracy = self.prediction_accuracy(
                    predicted_output, target_matrix)
                log_message = "Iteration:{}, Loss:{}, Accuracy:{}".format(
                    number_of_iterations, loss, accuracy)
                training_logger.info(log_message)
            else:
                if(number_of_iterations % logging_frequency == 0):
                    logging.info("Loss: {}".format(loss))

            # Update weights using backpropagation
            self.backpropagation(delta)

            if(number_of_iterations % update_frequency == 0):
                if(store_layers):
                    # NOTE dump layers only after backpropagation update
                    self.dump_layer_weights(layers_filename)

            # Decay the learning rate if needed
            if(number_of_iterations % decay_frequency == 0):
                # Decay the learning rate if needed
                self.optimizer.decay_learning_rate()

            if(loss < self.error_threshold or
                    number_of_iterations == max_iterations):

                # Dump layers information before exiting
                if(store_layers):
                    self.dump_layer_weights(layers_filename)
                break

            number_of_iterations += 1

    def predict(self, input_matrix):
        return self.forward_pass(input_matrix, store_values=False)
