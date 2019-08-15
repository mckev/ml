from unittest import TestCase

import numpy

from classes.ml.cnn import Cnn


class TestCnn(TestCase):

    def test_convolve(self):
        # Ref: https://medium.com/datadriveninvestor/convolution-neural-networks-vs-fully-connected-neural-networks-8171a6e86f15
        input = numpy.array([
            [18, 54, 51, 239, 244, 188],
            [55, 121, 75, 78, 95, 88],
            [35, 24, 204, 113, 109, 221],
            [3, 154, 104, 235, 25, 130],
            [15, 253, 225, 159, 78, 233],
            [68, 85, 180, 214, 245, 0]
        ])
        filter = numpy.array([
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1]
        ])
        output = Cnn.convolve(input=input, filter=filter)
        self.assertEqual(output.tolist(), [
            [429, 505, 686, 856],
            [261, 792, 412, 640],
            [633, 653, 851, 751],
            [608, 913, 713, 657]
        ])
