import unittest
from typing import List

import numpy

from classes.ml.sgd import Sgd


class TestSgd(unittest.TestCase):

    def test_sigmoid_f(self):
        self.assertAlmostEqual(Sgd.sigmoid_f(-1.0), 0.2689414213699951)
        self.assertAlmostEqual(Sgd.sigmoid_f(0.0), 0.5)
        self.assertAlmostEqual(Sgd.sigmoid_f(1.0), 0.7310585786300049)

    def test_sigmoid(self):
        z = numpy.array([-1.0, 0.0, 1.0])
        output = Sgd.sigmoid(z).tolist()
        self.assertEqual(output, [0.2689414213699951, 0.5, 0.7310585786300049])

    def test_sigmoid_prime_f(self):
        self.assertAlmostEqual(Sgd.sigmoid_prime_f(-1.0), 0.19661193324148185)
        self.assertAlmostEqual(Sgd.sigmoid_prime_f(0.0), 0.25)
        self.assertAlmostEqual(Sgd.sigmoid_prime_f(1.0), 0.19661193324148185)

    def test_sigmoid_prime(self):
        z = numpy.array([-1.0, 0.0, 1.0])
        output = Sgd.sigmoid_prime(z).tolist()
        self.assertEqual(output, [0.19661193324148185, 0.25, 0.19661193324148185])

    def test_softmax_01(self):
        z = numpy.array([2.0, 1.0, 0.1])
        output = Sgd.softmax(z).tolist()
        self.assertEqual(output, [0.6590011388859679, 0.2424329707047139, 0.09856589040931818])
        self.assertAlmostEqual(sum(output), 1.0)

    def test_softmax_02(self):
        z = numpy.array([2.0, 1.0, 0.1, 0.0, -0.1, -1.0, -2.0])
        output = Sgd.softmax(z).tolist()
        self.assertEqual(output, [0.5424927875944123, 0.19957194353977034, 0.08113989717873449, 0.07341841506290936,
                                  0.0664317291218153, 0.027009125505036098, 0.00993610199732203])
        self.assertAlmostEqual(sum(output), 1.0)

    def test_learn_xnor_operations(self):
        sgd = Sgd(sizes=[2, 3, 1])

        print('Training...')
        for _ in range(500):
            sgd.update_network(input=numpy.array([[0], [0]]), expected_output=numpy.array([[1]]), mini_batch=2, eta=2.0)
            sgd.update_network(input=numpy.array([[0], [1]]), expected_output=numpy.array([[0]]), mini_batch=2, eta=2.0)
            sgd.update_network(input=numpy.array([[1], [0]]), expected_output=numpy.array([[0]]), mini_batch=2, eta=2.0)
            sgd.update_network(input=numpy.array([[1], [1]]), expected_output=numpy.array([[1]]), mini_batch=2, eta=2.0)

        print('Testing...')
        output1: List[float] = sgd.feed_forward(input=numpy.array([[0], [0]]))
        self.assertEqual(round(output1[0][0]), 1.0)
        output4: List[float] = sgd.feed_forward(input=numpy.array([[0], [1]]))
        self.assertEqual(round(output4[0][0]), 0.0)
        output3: List[float] = sgd.feed_forward(input=numpy.array([[1], [0]]))
        self.assertEqual(round(output3[0][0]), 0.0)
        output2: List[float] = sgd.feed_forward(input=numpy.array([[1], [1]]))
        self.assertEqual(round(output2[0][0]), 1.0)
