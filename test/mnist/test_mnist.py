from unittest import TestCase

from classes.mnist.mnist import Mnist


class TestMnist(TestCase):

    def test_retrieve_mnist_datas(self):
        mnist_datas = Mnist.retrieve_mnist_datas(filename='../../classes/mnist/mnist.raw')
        self.assertEqual(len(mnist_datas), 70000)
        Mnist.draw_mnist_data(mnist_datas[9000])
