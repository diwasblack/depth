import logging
import pickle
import gzip

import numpy as np

from .loss_functions import mean_squared_error, cross_entropy
from .helpers import vector_to_label
from .metrics import categorical_accuracy
from .optimizers import SGD
from .helpers import get_mini_batches


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
            layer.construct_layer(previous_layer=self.layers[-1])
        else:
            # Construct the layer
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

    def train(self, input_tensor, target_tensor, mini_batch_size=32,
              max_epochs=1000, logging_frequency=100, update_frequency=100,
              layers_filename="", training_logger=None):
        """
        Train the neural network contructed

        Inputs:
        input_tensor: a tensor to use as input
        target_tensor: target tensor

        max_iterations: the maximum number of iterations
        logging_frequency: the frequency of logging the training cost
        update_frequency: the frequency of storing the weights to a file
        layers_filename: the file to use to store the layers

        training_logger: the logger object to use for report training
            information
        """
        # Check if the model is ready for training
        if(not(self.optimizer)):
            raise Exception("Optimizer not assigned")

        if(layers_filename):
            store_layers = True
        else:
            store_layers = False

        iteration = 1
        for epoch in range(1, max_epochs+1):
            for batch in get_mini_batches(input_tensor, target_tensor,
                                          mini_batch_size):
                batch_input, batch_target = batch

                # Propagate the input forward
                predicted_output = self.forward_pass(batch_input)

                # Calculate delta at the final layer
                delta = self.loss_function_derivative(
                    predicted_output, batch_target)

                loss = self.loss_function(predicted_output, batch_target)

                # Add the regularied loss
                loss += self.get_regularization_cost()

                if(training_logger):
                    accuracy = self.prediction_accuracy(
                        predicted_output, batch_target)
                    log_message = "Epoch:{}, Iteration:{}, Loss:{}, Accuracy:{}".format(
                        epoch, iteration, loss, accuracy)
                    training_logger.info(log_message)
                else:
                    if(iteration % logging_frequency == 0):
                        logging.info("Loss: {}".format(loss))

                # Use backpropagation to propagate delta through layers
                for layer in reversed(self.layers):
                    gradient, delta, layer_delta = layer.backprop(delta)

                    if gradient is None:
                        continue

                    # Calculate the weight update for current layer
                    weight_update, first_moment, second_moment = self.optimizer.get_updates(
                        gradient, layer.first_moment, layer.second_moment,
                        time_step=iteration)

                    # Store the first moment and second_moment
                    layer.first_moment = first_moment
                    layer.second_moment = second_moment

                    # Update layer weights
                    layer.update_weights(weight_update)

                    if layer_delta is None:
                        continue

                    # Calculate the delta for bias
                    dloss_db = np.sum(layer_delta, axis=(2, 3))

                    # Average delta across samples
                    dloss_db = np.average(dloss_db, axis=0)

                    bias_update, bias_first_moment, bias_second_moment = self.optimizer.get_updates(
                        dloss_db, layer.bias_first_moment, layer.bias_second_moment,
                        time_step=iteration)

                    layer.bias_first_moment = bias_first_moment
                    layer.bias_second_moment = bias_second_moment

                    layer.update_biases(bias_update)

                if(iteration % update_frequency == 0):
                    if(store_layers):
                        training_logger.info("Storing layer weights")
                        # NOTE dump layers only after backpropagation update
                        self.dump_layer_weights(layers_filename)

                # Increase iteration count
                iteration += 1

    def predict(self, input_matrix):
        return self.forward_pass(input_matrix, store_values=False)
