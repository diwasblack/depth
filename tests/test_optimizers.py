import unittest

import numpy as np

from depth.optimizers import SGD


class TestSGD(unittest.TestCase):
    def test_get_updates(self):
        # Create a SGD optimizer object
        optimizer = SGD()

        gradients = np.array([
            [0.2, 0.3],
            [0.3, 1]
        ])

        self.assertIsNotNone(optimizer.get_updates(gradients, None))

    def test_decay_learning_rate(self):
        # Create a SGD optimizer object with decay
        optimizer = SGD(decay=0.1)

        previous_lr = optimizer.lr

        optimizer.decay_learning_rate()

        updated_lr = optimizer.lr
        self.assertNotEqual(previous_lr, updated_lr)
