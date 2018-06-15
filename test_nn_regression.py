import logging

import numpy as np

from sequential_nn import SequentialNeuralNet


def main():
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)

    number_of_samples = 50

    input_data_dimension = 10
    output_data_dimension = 3

    nn_object = SequentialNeuralNet(input_dimension=input_data_dimension)
    nn_object.add_layer(units=32, activation_function="relu")
    nn_object.add_layer(units=64)
    nn_object.add_layer(units=output_data_dimension, activation_function="linear")
    nn_object.compile(loss="mean_squared_error", error_threshold=0.001)

    input_data = -0.5 + np.random.rand(input_data_dimension, number_of_samples)
    output_data = np.random.rand(output_data_dimension, number_of_samples)

    nn_object.train(input_data, output_data)

    print(output_data)
    print(nn_object.predict(input_data))


if __name__ == "__main__":
    main()
