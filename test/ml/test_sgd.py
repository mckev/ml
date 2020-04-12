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
        z = numpy.array([2.0, 1.0, 0.1, -0.1, -1.0, -2.0])
        output = Sgd.softmax(z).tolist()
        self.assertEqual(output, [0.5854776270254112, 0.21538518224849038, 0.08756908025993568, 0.07169549902756336,
                                  0.02914921464456889, 0.010723396794030428])
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
