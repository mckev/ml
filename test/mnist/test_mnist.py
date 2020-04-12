import unittest

from classes.mnist.mnist import Mnist


class TestMnist(unittest.TestCase):

    def test_retrieve_mnist_datas(self):
        mnist_datas = Mnist.retrieve_mnist_datas(filename='../../classes/mnist/mnist.raw')
        self.assertEqual(len(mnist_datas), 70000)
        stdout = Mnist.show_mnist_data(mnist_datas[9000])
        self.assertEqual(stdout, [
            'Number: 6',
            '                            ',
            '                            ',
            '                            ',
            '                            ',
            '         *****              ',
            '        ******              ',
            '        ******              ',
            '       ****                 ',
            '       ****                 ',
            '       ***         *        ',
            '       ***      ******      ',
            '       ***     ********     ',
            '      ****    **********    ',
            '      ****   ************   ',
            '      ****  *****     ***   ',
            '      **** *****      ***   ',
            '      *********       ***   ',
            '      ********       ***    ',
            '       *******     ****     ',
            '       ****************     ',
            '        *************       ',
            '         *********          ',
            '           *****            ',
            '             *              ',
            '                            ',
            '                            ',
            '                            ',
            '                            '
        ])
