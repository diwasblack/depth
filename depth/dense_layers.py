from .dense import (
    DenseLayerBase, NonLinearBackprop, LinearBackprop, XavierWeightInitializer,
    HeWeightInitializer
)
from .activations import (
    hyperbolic_tangent, hyperbolic_tangent_derivative,
    relu, relu_derivative,
    leaky_relu, leaky_relu_derivative,
    sigmoid_function, sigmoid_function_derivative)


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
