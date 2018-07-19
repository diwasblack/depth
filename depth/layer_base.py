import numpy as np

from abc import ABC, abstractmethod


class BaseLayer(ABC):
    def __init__(self):
        self.regularizer = None
        self.activation = None

    def get_output_shape(self):
        """
        Return the shape of the output from this layer
        """
        return self.output_shape

    def get_weights_variance(self, input_units, output_units=None):
        """
        Return the variance to use with the weights
        """

        if(self.activation is None):
            raise Exception("Activation function is not assigned")

        # An implementation of He weight initialization
        # See:
        # https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf
        if(self.activation in ["relu", "leakyrelu"]):
            variance = np.sqrt(2.0 / input_units)
        # An implementation of Xavier weight initialization
        # See:
        # http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
        elif(self.activation in ["tanh", "sigmoid"]):
            variance = np.sqrt(1.0 / input_units)
        else:
            variance = 0.5

        return variance

    @abstractmethod
    def construct_layer(self, previous_layer=None):
        pass

    @abstractmethod
    def forward_pass(self, input_matrix, store_values=True):
        pass

    @abstractmethod
    def backprop(self, delta):
        pass

    def get_regularized_cost(self):
        """
        Get the cost of the layer using the regularizer provided
        """

        if(self.regularizer):
            return self.regularizer.get_cost(self.weights)
        else:
            return 0
