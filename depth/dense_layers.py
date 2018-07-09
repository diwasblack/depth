from .dense import (
    DenseLayerBase, NonLinearBackprop, LinearBackprop, XavierWeightInitializer,
    HeWeightInitializer
)
from .activations import (
    hyperbolic_tangent, hyperbolic_tangent_derivative,
    relu, relu_derivative,
    leaky_relu, leaky_relu_derivative,
    sigmoid_function, sigmoid_function_derivative,
    softmax_function)


class TanhLayer(NonLinearBackprop, DenseLayerBase, XavierWeightInitializer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Assign hyperbolic_tangent function to be the activation function
        self.activation_function = hyperbolic_tangent
        self.activation_function_derivative = hyperbolic_tangent_derivative


class ReluLayer(NonLinearBackprop, DenseLayerBase, HeWeightInitializer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Assign relu function to be the activation function
        self.activation_function = relu
        self.activation_function_derivative = relu_derivative


class LeakyReluLayer(NonLinearBackprop, DenseLayerBase, HeWeightInitializer):
    def __init__(self, *args, alpha=0.3, **kwargs):
        super().__init__(*args, **kwargs)

        # Assign relu function to be the activation function
        self.activation_function = leaky_relu(alpha=alpha)
        self.activation_function_derivative = leaky_relu_derivative(
            alpha=alpha)


class SigmoidLayer(NonLinearBackprop, DenseLayerBase, XavierWeightInitializer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Assign sigmoid function to be the activation function
        self.activation_function = sigmoid_function
        self.activation_function_derivative = sigmoid_function_derivative


class LinearLayer(LinearBackprop, DenseLayerBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Return the input as without modification
        self.activation_function = lambda x: x


class SoftmaxLayer(LinearBackprop, DenseLayerBase):
    """
    An impementation of softmax layer that is to be used at output layer only
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.activation_function = softmax_function


class DenseLayer():
    def __init__(self, activation_function="tanh", units=64,
                 input_dimension=None, **kwargs):
        self.activation_function = activation_function
        self.units = units
        self.input_dimension = input_dimension

        self.kwargs = kwargs

    def get_layer_obj(self, previous_units=None):
        activation_function = self.activation_function
        units = self.units
        kwargs = self.kwargs

        if(not(previous_units)):
            previous_units = self.input_dimension

        if(activation_function == "tanh"):
            layer = TanhLayer(previous_units, units, **kwargs)

        elif(activation_function == "relu"):
            layer = ReluLayer(previous_units, units, **kwargs)

        elif(activation_function == "leakyrelu"):
            layer = LeakyReluLayer(previous_units, units, **kwargs)

        elif(activation_function == "sigmoid"):
            layer = SigmoidLayer(previous_units, units, **kwargs)

        elif(activation_function == "linear"):
            layer = LinearLayer(previous_units, units, **kwargs)

        elif(activation_function == "softmax"):
            layer = SoftmaxLayer(previous_units, units, **kwargs)

        else:
            raise Exception("Unknown layer name received")

        return layer
