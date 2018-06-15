import numpy as np


def sigmoid_function(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_function_derivative(x):
    return x * (1.0 - x)


def bipolar_sigmoid_function(x):
    return 2.0 / (1.0 + np.exp(-x)) - 1.0


def bipolar_sigmoid_function_derivative(x):
    return (1.0 - np.square(x)) / 2.0


def hyperbolic_tangent(x):
    return np.tanh(x)


def hyperbolic_tangent_derivative(x):
    return 1.0 - np.square(x)


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    x_derivative = np.copy(x)
    x_derivative[x <= 0] = 0
    x_derivative[x > 0] = 1
    return x_derivative


def softmax_function(x):
    """
    Implementation for softmax function

    See:
    http://cs231n.github.io/linear-classify/#softmax
    """

    e_x = np.exp(x - np.max(x, axis=0))
    return e_x / e_x.sum(axis=0)
