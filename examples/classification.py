import logging

import numpy as np

from depth.models import Sequential
from depth.layers import DenseLayer
from depth.helpers import one_hot_encoding, vector_to_label
from depth.metrics import categorical_accuracy


def main():
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)

    number_of_samples = 50

    input_data_dimension = 10
    output_data_dimension = 3

    nn_object = Sequential()
    nn_object.add_layer(DenseLayer(
        units=32, activation_function="tanh",
        input_dimension=input_data_dimension))
    nn_object.add_layer(DenseLayer(units=64, activation_function="tanh"))
    nn_object.add_layer(DenseLayer(
        units=output_data_dimension, activation_function="softmax"))
    nn_object.compile(loss="cross_entropy", error_threshold=0.001)

    input_data = -0.5 + np.random.rand(input_data_dimension, number_of_samples)
    targets = np.random.randint(
        0, output_data_dimension, number_of_samples).reshape(-1)

    # Convert labels to one hot encoding
    output_data = one_hot_encoding(targets)

    nn_object.train(input_data, output_data)

    predicted_output = nn_object.predict(input_data)
    predicted_labels = vector_to_label(predicted_output)

    print("Accuracy: {}%".format(100*categorical_accuracy(
        predicted_labels, targets)))


if __name__ == "__main__":
    main()
