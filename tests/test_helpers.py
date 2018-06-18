import unittest

import numpy as np

from depth.helpers import vector_to_label


class TestHelperMethods(unittest.TestCase):

    def test_vector_to_label(self):
        # Create a numpy array
        test_vector = np.array([
            [0.7, 0.0, 0.1],
            [0.1, 0.8, 0.2],
            [0.2, 0.2, 0.7],
        ])
        converted_labels = vector_to_label(test_vector)

        # Check if the labels are correct
        self.assertEqual(converted_labels[0], 0)
        self.assertEqual(converted_labels[1], 1)
        self.assertEqual(converted_labels[2], 2)
