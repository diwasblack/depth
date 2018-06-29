from .dense_layers import (
    TanhLayer, ReluLayer, LeakyReluLayer, SigmoidLayer, LinearLayer,
    SoftmaxLayer)


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
