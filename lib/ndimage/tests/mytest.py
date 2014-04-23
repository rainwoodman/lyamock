
from __future__ import division, print_function, absolute_import

import math
import numpy
import numpy as np
from numpy import fft
from numpy.testing import assert_, assert_equal, assert_array_equal, \
        TestCase, run_module_suite, \
        assert_array_almost_equal, assert_almost_equal
import scipy.ndimage as ndimage

eps = 1e-12


def sumsq(a, b):
    return math.sqrt(((a - b)**2).sum())


class Test:
    def test_boundaries(self):
        "boundary modes"
        def shift(x):
            return (x[0] + 0.5,)

        data = numpy.array([1,2,3,4.])
        expected = {
#                    'constant': [1.5,2.5,3.5,-1,-1,-1,-1],
                    'wrap': [1.5,2.5,3.5,3.5,1.5,2.5,3.5],
#                    'mirror': [1.5,2.5,3.5,3.5,2.5,1.5,1.5],
#                    'nearest': [1.5,2.5,3.5,4,4,4,4]
}

        for mode in expected:
            assert_array_equal(expected[mode],
                               ndimage.geometric_transform(data,shift,
                                                           cval=-1,mode=mode,
                                                           output_shape=(7,),
                                                           order=1))

if __name__ == "__main__":
    run_module_suite()
