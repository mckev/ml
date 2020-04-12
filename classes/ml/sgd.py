import math
from typing import List

import numpy


class Sgd:
    """ Stochastic Gradient Descent (SGD) """

    # Ref: http://neuralnetworksanddeeplearning.com/chap1.html

    @staticmethod
    def sigmoid_f(z: float) -> float:
        return 1.0 / (1.0 + math.exp(-z))

    @staticmethod
    def sigmoid(z: List[float]) -> List[float]:
        return 1.0 / (1.0 + numpy.exp(-z))

    @staticmethod
    def sigmoid_prime_f(z: float) -> float:
        return Sgd.sigmoid_f(z) * (1 - Sgd.sigmoid_f(z))

    @staticmethod
    def sigmoid_prime(z: List[float]) -> List[float]:
        return Sgd.sigmoid(z) * (1 - Sgd.sigmoid(z))

    @staticmethod
    def softmax(z: List[float]) -> List[float]:
        # Typically used to turn logits into probabilities that sum to one
        return numpy.exp(z) / numpy.sum(numpy.exp(z))

    def __init__(self, sizes: List[int]):
        # The list "sizes" contains the number of neurons in the respective layers of the network. For example, if the list was [2, 3, 1] then it would be a three-layer network, with the first layer containing 2 neurons, the second layer 3 neurons, and the third layer 1 neuron.
        self.num_layers: int = len(sizes)
        # Note that the first layer (layer 0) is assumed to be an input layer, and by convention we won't set any biases for those neurons, since biases are only ever used in computing the outputs from later layers.
        self.biases: List[List[float]] = [numpy.random.randn(y, 1) for y in sizes[1:]]
        self.weights: List[List[List[float]]] = [numpy.random.randn(y, x) for (y, x) in zip(sizes[1:], sizes[:-1])]

    def feed_forward(self, input: List[float]) -> List[float]:
        # For example: sizes = {784, 30, 10}
        # First iteration: matrix 30x784 . matrix 784x1 = matrix 30x1
        # Second iteration: matrix 10x30 . matrix 30x1 = matrix 10x1
        a = input
        for (b, w) in zip(self.biases, self.weights):
            a = Sgd.sigmoid(numpy.dot(w, a) + b)
        return a

    def backprop(self, input: List[float], expected_output: List[float]):
        # Update nabla_b and nabla_w representing the gradient for the cost function C_x. nabla_b and nabla_w are layer-by-layer lists of numpy arrays, similar to self.biases and self.weights.
        nabla_b: List[List[float]] = [numpy.zeros(shape=b.shape) for b in self.biases]
        nabla_w: List[List[List[float]]] = [numpy.zeros(shape=w.shape) for w in self.weights]

        # Feed forward
        activation: List[float] = input
        activations: List[List[float]] = []
        activations.append(input)
        zs: List[List[float]] = []
        for b, w in zip(self.biases, self.weights):
            z: List[float] = numpy.dot(w, activation) + b
            zs.append(z)
            activation = Sgd.sigmoid(z)
            activations.append(activation)
        assert len(activations) == len(self.biases) + 1
        assert len(zs) == len(self.biases)

        # Backward pass
        delta: List[float] = (activations[-1] - expected_output) * Sgd.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = numpy.dot(delta, activations[-2].transpose())
        for L in range(2, self.num_layers):
            sp: List[float] = Sgd.sigmoid_prime(zs[-L])
            delta = numpy.dot(self.weights[-L + 1].transpose(), delta) * sp
            nabla_b[-L] = delta
            nabla_w[-L] = numpy.dot(delta, activations[-L - 1].transpose())
        return (nabla_b, nabla_w)

    def update_network(self, input: List[float], expected_output: List[float], mini_batch: int, eta: float):
        if not hasattr(self, 'batch_no'):
            self.batch_no: int = 0
            self.sum_nabla_b: List[List[float]] = [numpy.zeros(shape=b.shape) for b in self.biases]
            self.sum_nabla_w: List[List[List[float]]] = [numpy.zeros(shape=w.shape) for w in self.weights]
        self.batch_no += 1
        (nabla_b, nabla_w) = self.backprop(input, expected_output)
        self.sum_nabla_b = [nb + dnb for (nb, dnb) in zip(nabla_b, self.sum_nabla_b)]
        self.sum_nabla_w = [nw + dnw for (nw, dnw) in zip(nabla_w, self.sum_nabla_w)]
        if self.batch_no >= mini_batch:
            self.biases = [b - (eta / mini_batch) * nb for (b, nb) in zip(self.biases, self.sum_nabla_b)]
            self.weights = [w - (eta / mini_batch) * nw for (w, nw) in zip(self.weights, self.sum_nabla_w)]
            delattr(self, 'batch_no')
            delattr(self, 'sum_nabla_b')
            delattr(self, 'sum_nabla_w')
