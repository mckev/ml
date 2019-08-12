import unittest
from typing import List

import numpy

from classes.ml.sgd import Sgd
from classes.mnist.mnist import Mnist


class TestSgdMnist(unittest.TestCase):

    def test_learn_handwriting(self):
        sgd = Sgd(sizes=[Mnist.IMAGE_SIZE * Mnist.IMAGE_SIZE, 30, 10])
        mnist_datas = Mnist.retrieve_mnist_datas(filename='../../classes/mnist/mnist.raw')

        print('Training...')
        for mnist_data in mnist_datas[:50000]:
            image_bytes = mnist_data['image_bytes']
            input = numpy.frombuffer(buffer=image_bytes, dtype='uint8').reshape((len(image_bytes), 1))
            normalized_input = input / 255
            number = mnist_data['number']
            expected_output = numpy.zeros(shape=(10, 1))
            expected_output[number] = 1.0
            sgd.update_network(input=normalized_input, expected_output=expected_output, mini_batch=10, eta=4.0)

        print('Testing...')
        correct = 0
        total = 0
        for mnist_data in mnist_datas[50000:]:
            image_bytes = mnist_data['image_bytes']
            input = numpy.frombuffer(buffer=image_bytes, dtype='uint8').reshape((len(image_bytes), 1))
            normalized_input = input / 255
            number = mnist_data['number']
            output: List[float] = sgd.feed_forward(input=normalized_input)
            output_index = numpy.argmax(output)
            if output_index == number:
                correct += 1
            total += 1
        percent_correct = 100 * correct / total
        print(f'Correct {correct} out of {total} ({percent_correct:.1f}% correct)')
        self.assertGreater(percent_correct, 80)
