from abc import ABC, abstractmethod


class BaseLayer(ABC):
    def __init__(self):
        self.regularizer = None

    def get_output_shape(self):
        """
        Return the shape of the output from this layer
        """
        return self.output_shape

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
