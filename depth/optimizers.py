import numpy as np


class SGD():
    """
    Implementation of stochastic gradient descent algorithm
    """

    def __init__(self, lr=0.01, decay=0.0, momentum=0.0):
        self.lr = lr
        self.decay = decay
        self.momentum = momentum

    def get_updates(self, gradient, previous_update):
        """
        Return the parameter update
        """
        parameter_updates = self.lr * gradient

        if previous_update is not None:
            parameter_updates = parameter_updates + self.momentum *\
                previous_update

        return parameter_updates

    def decay_learning_rate(self):
        """
        Decay learning rate by the factor provided
        """

        if(self.decay):
            self.lr = self.decay * self.lr
