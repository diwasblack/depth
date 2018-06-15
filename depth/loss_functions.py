import numpy as np


def mean_squared_error(predicted_output, target_matrix):
    delta = predicted_output - target_matrix
    squared_errors = np.sum(np.square(delta), axis=0)
    return np.mean(squared_errors)


def cross_entropy(predicted_output, target_matrix):
    predicted_output_log = -np.log(predicted_output)

    loss_matrix = np.multiply(predicted_output_log, target_matrix)
    loss_matrix = np.sum(loss_matrix, axis=0)

    return np.mean(loss_matrix)
