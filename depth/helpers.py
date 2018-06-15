import numpy as np


def one_hot_encoding(targets):
    """
    Convert the class labels to a one hot encoding

    target: a numpy array with element in range [0, num_labels] where
    num_labels+1 is the total number of labels in the data
    """

    output_dimension = np.max(targets) + 1

    return np.eye(output_dimension)[targets].T
