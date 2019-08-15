from typing import List

import numpy


class Cnn:

    @staticmethod
    def convolve(input: List[List[float]], filter: List[List[float]]) -> List[List[float]]:
        # Ref: https://stackoverflow.com/questions/43086557/convolve2d-just-by-using-numpy/43087507
        shape = filter.shape + tuple(numpy.subtract(input.shape, filter.shape) + 1)
        sub_matrices = numpy.lib.stride_tricks.as_strided(input, shape=shape, strides=input.strides * 2)
        output = numpy.einsum('ij,ijkl->kl', filter, sub_matrices)
        return output
