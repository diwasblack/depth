import numpy as np


class SGD():
    """
    Implementation of stochastic gradient descent algorithm
    """

    def __init__(self, lr=0.01, decay=0.0):
        self.lr = lr
        self.decay = decay

    def get_updates(self, parameters):
        """
        Return the parameter update
        """
        parameter_updates = self.lr * parameters
        return parameter_updates

    def decay_learning_rate(self):
        """
        Decay learning rate by the factor provided
        """

        if(self.decay):
            self.lr = self.decay * self.lr
