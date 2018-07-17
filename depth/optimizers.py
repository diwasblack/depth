import numpy as np


class SGD():
    """
    Implementation of stochastic gradient descent algorithm
    """

    def __init__(self, lr=0.01, decay=0.0, momentum=0.0):
        self.lr = lr
        self.decay = decay
        self.momentum = momentum

    def get_updates(self, gradient, first_moment, second_moment, **kwargs):
        """
        Return the parameter update
        """
        update = self.lr * gradient
        update += self.momentum * first_moment
        return update, first_moment, second_moment

    def decay_learning_rate(self):
        """
        Decay learning rate by the factor provided
        """

        if(self.decay):
            self.lr = self.decay * self.lr


class ADAM():
    """
    Implementation of ADAM algorithm for optimization

    See:
    https://arxiv.org/pdf/1412.6980.pdf
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=10e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def get_updates(self, gradient, first_moment, second_moment, **kwargs):
        """
        Return the update for the parameter
        """

        t = kwargs["time_step"]

        first_moment = self.beta1 * first_moment + (1 - self.beta1) * gradient
        second_moment = self.beta2 * second_moment + (1 - self.beta2) * np.square(
            gradient)

        first_moment_corrected = first_moment / (1 - pow(self.beta1, t))
        second_moment_corrected = second_moment / (1 - pow(self.beta2, t))

        update = self.lr * first_moment_corrected / \
            (np.sqrt(second_moment_corrected) + self.epsilon)

        return update, first_moment, second_moment
