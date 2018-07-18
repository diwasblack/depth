import unittest

import numpy as np

from depth.convolution import convolve2d


class TestConvolution(unittest.TestCase):
    def test_convolve2d(self):
        data_tensor = np.ones((1, 1, 5, 5))
        kernel_tensor = np.array([[
            [0, 1, 0],
            [0, 0, 0],
            [0, -1, 0]]])

        result = np.array([[[
            [-1, -1, -1, -1, -1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            ]]])

        convolution = convolve2d(data_tensor, kernel_tensor)
        self.assertTrue(np.array_equal(convolution, result))
