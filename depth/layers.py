from .layers_base import (
    LayerBase, NonLinearBackprop, LinearBackprop, XavierWeightInitializer,
    HeWeightInitializer)
from .activations import (
    hyperbolic_tangent, hyperbolic_tangent_derivative,
    relu, relu_derivative,
    leaky_relu, leaky_relu_derivative,
    sigmoid_function, sigmoid_function_derivative,
    softmax_function)


class TanhLayer(NonLinearBackprop, LayerBase, XavierWeightInitializer):
    def __init__(self, *args):
        super().__init__(*args)
        # Assign hyperbolic_tangent function to be the activation function
        self.activation_function = hyperbolic_tangent
        self.activation_function_derivative = hyperbolic_tangent_derivative


class ReluLayer(NonLinearBackprop, LayerBase, HeWeightInitializer):
    def __init__(self, *args):
        super().__init__(*args)
        # Assign relu function to be the activation function
        self.activation_function = relu
        self.activation_function_derivative = relu_derivative


class LeakyReluLayer(NonLinearBackprop, LayerBase, HeWeightInitializer):
    def __init__(self, *args, alpha=0.3, **kwargs):
        super().__init__(*args, **kwargs)

        # Assign relu function to be the activation function
        self.activation_function = leaky_relu(alpha=alpha)
        self.activation_function_derivative = leaky_relu_derivative(alpha=alpha)


class SigmoidLayer(NonLinearBackprop, LayerBase, XavierWeightInitializer):
    def __init__(self, *args):
        super().__init__(*args)
        # Assign sigmoid function to be the activation function
        self.activation_function = sigmoid_function
        self.activation_function_derivative = sigmoid_function_derivative


class LinearLayer(LinearBackprop, LayerBase):
    def __init__(self, *args):
        super().__init__(*args)
        # Return the input as without modification
        self.activation_function = lambda x: x
