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


def train_test_split(input_matrix, output_matrix, test_size=0.33):
    data_size = input_matrix.shape[1]

    # Generate random permutations
    permutations = np.random.permutation(data_size)

    # Shuffle input and output matrix
    input_matrix = input_matrix[:, permutations]
    output_matrix = output_matrix[permutations]

    training_size = int((1 - test_size) * data_size)

    x_train = input_matrix[:, :training_size]
    x_test = input_matrix[:, training_size:]

    y_train = output_matrix[:training_size]
    y_test = output_matrix[training_size:]

    return x_train, x_test, y_train, y_test


def get_mini_batches(input_tensor, target_tensor, batch_size):
    """
    Iteratively return the mini batch to use
    """

    if(len(input_tensor.shape) == 2):
        samples = input_tensor.shape[1]
        permutations = np.random.permutation(samples)

        input_tensor = input_tensor[:, permutations]
        target_tensor = target_tensor[:, permutations]

        i = 0

        while(i <= samples):
            batch_input = input_tensor[:, i:i+batch_size]
            batch_target = target_tensor[:, i:i+batch_size]

            yield (batch_input, batch_target)
            i = i + batch_size

    elif(len(input_tensor.shape) == 4):
        samples = input_tensor.shape[0]
        permutations = np.random.permutation(samples)

        input_tensor = input_tensor[permutations]
        target_tensor = target_tensor[:, permutations]

        i = 0

        while(i <= samples):
            batch_input = input_tensor[i:i+batch_size]
            batch_target = target_tensor[:, i:i+batch_size]

            yield (batch_input, batch_target)
            i = i + batch_size

    else:
        raise Exception("Input tensor dimension not recognized")
