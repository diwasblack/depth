import numpy as np


def categorical_accuracy(predicted_output, target_output):
    """
    Computes the accuracy score of predicted_output
    """
    correct_labels = 0

    number_of_samples = len(predicted_output)

    for i in range(0, number_of_samples):
        if(predicted_output[i] == target_output[i]):
            correct_labels += 1

    return correct_labels/number_of_samples
