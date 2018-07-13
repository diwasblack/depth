import math

import numpy as np


def HeWeightInitializer(input_units, output_units):
    """
    An implementation of He weight initialization

    See:
    https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf
    """

    return np.random.randn(output_units, input_units) * \
        math.sqrt(2.0 / input_units)


def XavierWeightInitializer(input_units, output_units):
    """
    An implementation of Xavier weight initialization

    See:
    http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
    """
    return np.random.randn(output_units, input_units) * \
        math.sqrt(1.0 / input_units)
