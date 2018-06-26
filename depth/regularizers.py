import numpy as np


class L2Regularizer():
    def __init__(self, coefficient):
        self.coefficient = coefficient

    def get_cost(self, x):
        return self.coefficient * 0.5 * np.sum(np.square(x))

    def get_derivative_value(self, x):
        return self.coefficient * x
