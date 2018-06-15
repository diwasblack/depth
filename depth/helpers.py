import numpy as np


def one_hot_encoding(targets):
    """
    Convert the class labels to a one hot encoding

    target: a numpy array with element in range [0, num_labels] where
    num_labels+1 is the total number of labels in the data
    """

    # Convert targets to integer
    int_targets = targets.astype(int)

    output_dimension = np.max(int_targets) + 1

    return np.eye(output_dimension)[int_targets].T


def vector_to_label(input_matrix):
    """
    Convert the vector of probabilites to the class label
    """
    return np.argmax(input_matrix, axis=0)
