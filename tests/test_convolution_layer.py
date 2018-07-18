import unittest

import numpy as np

from depth.layers import Convolution2D


class TestConvolution(unittest.TestCase):
    def setUp(self):
        convolution_layer = Convolution2D(5, (3,3), input_shape=(3, 32, 32))
        convolution_layer.construct_layer()

        convolution_layer.samples = 10
        convolution_layer.activation_values = np.random.randn(10, 5, 32, 32)
        convolution_layer.input_values = np.random.randn(10, 3, 32, 32)

        self.convolution_layer = convolution_layer

    def test_convolution_backprop(self):
        delta = np.ones((1, 5, 32, 32))
        self.assertIsNotNone(self.convolution_layer.backprop(delta))
