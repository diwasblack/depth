import unittest

import numpy as np

from depth.helpers import one_hot_encoding
from depth.models import Sequential
from depth.regularizers import L2Regularizer


class TestClassification(unittest.TestCase):
    def setUp(self):
        self.number_of_samples = 50

        self.input_data_dimension = 10
        self.output_data_dimension = 3

        # Prepare dataset to use
        self.input_data = -0.5 + np.random.rand(
            self.input_data_dimension, self.number_of_samples)

        targets = np.random.randint(
            0, self.output_data_dimension, self.number_of_samples).reshape(-1)

        # Convert labels to one hot encoding
        self.target_data = one_hot_encoding(targets)


class TestReluLayer(TestClassification):
    def test_model_training(self):
        # Create nn object
        nn_object = Sequential()
        nn_object.add_layer(units=self.output_data_dimension, activation_function="relu",
                            input_dimension=self.input_data_dimension)
        nn_object.compile(loss="cross_entropy")

        # Train the neural network
        nn_object.train(self.input_data, self.target_data, max_iterations=1)


class TestLeakyReluLayer(TestClassification):
    def test_model_training(self):
        # Create nn object
        nn_object = Sequential()
        nn_object.add_layer(units=self.output_data_dimension, activation_function="leakyrelu",
                            input_dimension=self.input_data_dimension,
                            alpha=0.3)
        nn_object.compile(loss="cross_entropy")

        # Train the neural network
        nn_object.train(self.input_data, self.target_data, max_iterations=1)


class TestTanhLayer(TestClassification):
    def test_model_training(self):
        # Create nn object
        nn_object = Sequential()
        nn_object.add_layer(units=self.output_data_dimension, activation_function="tanh",
                            input_dimension=self.input_data_dimension)
        nn_object.compile(loss="cross_entropy")

        # Train the neural network
        nn_object.train(self.input_data, self.target_data, max_iterations=1)


class TestSigmoidLayer(TestClassification):
    def test_model_training(self):
        # Create nn object
        nn_object = Sequential()
        nn_object.add_layer(units=self.output_data_dimension, activation_function="sigmoid",
                            input_dimension=self.input_data_dimension)
        nn_object.compile(loss="cross_entropy")

        # Train the neural network
        nn_object.train(self.input_data, self.target_data, max_iterations=1)


class TestRegularizedClassification(TestClassification):
    def test_model_training(self):
        # Create a regularizer
        regularizer = L2Regularizer(1.0)

        # Create nn object
        nn_object = Sequential()
        nn_object.add_layer(
            units=self.output_data_dimension, activation_function="tanh",
            input_dimension=self.input_data_dimension, regularizer=regularizer
        )
        nn_object.compile(loss="cross_entropy")

        # Train the neural network
        nn_object.train(self.input_data, self.target_data, max_iterations=1)
