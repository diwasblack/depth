import unittest

import numpy as np

from depth.convolution import convolve_tensors


class TestConvolution(unittest.TestCase):
    def setUp(self):
        self.data_tensor = np.ones((1, 1, 5, 5))
        self.kernel_tensor = np.array([[[
            [0, 1, 0],
            [0, 0, 0],
            [0, -1, 0]]]])

    def test_2d_convolution(self):
        result = np.array([[[
            [-1, -1, -1, -1, -1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            ]]])

        convolution = convolve_tensors(self.data_tensor, self.kernel_tensor)
        self.assertTrue(np.array_equal(convolution, result))
