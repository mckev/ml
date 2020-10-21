import sys
import unittest

import keras
import numpy

from classes.ml.genetic import Genetic
from classes.mnist.mnist import Mnist


class TestKerasGenetic(unittest.TestCase):
    """ Solve MNIST handwriting using Keras and Genetic Algorithm """

    @staticmethod
    def generate_model():
        # Ref: https://www.dlology.com/blog/how-to-choose-last-layer-activation-and-loss-function/

        # Simple model
        model = keras.models.Sequential()
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(30, activation='sigmoid'))
        model.add(keras.layers.Dense(10, activation='softmax'))
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    @staticmethod
    def count_correct_ratio(model, mnist_datas):
        """ Given a model, determine how well it is """
        inputs = []
        for mnist_data in mnist_datas:
            input = mnist_data['image_arr']
            inputs.append(input)
        # Most efficient shape would be (len(mnist_datas), 28, 28, 1) as input
        inputs = numpy.stack(inputs)

        guessed_outputs = model.predict(inputs)

        correct = 0
        for mnist_data, guessed_output in zip(mnist_datas, guessed_outputs):
            number = mnist_data['number']
            guessed_number = numpy.argmax(guessed_output)
            if guessed_number == number:
                correct += 1

        return correct / len(mnist_datas)

    def test_learn_handwriting(self):
        print('Pre-processing MNIST data...')
        input_shape = (28, 28)
        mnist_datas = Mnist.retrieve_mnist_datas(filename='../../classes/mnist/mnist.raw')
        for mnist_data in mnist_datas:
            image_bytes = mnist_data['image_bytes']
            mnist_data['image_arr'] = numpy.frombuffer(buffer=image_bytes, dtype='uint8').reshape(input_shape)
            mnist_data['image_arr'] = mnist_data['image_arr'] / 255

            number = mnist_data['number']
            mnist_data['number_arr'] = numpy.zeros(shape=(10,))
            mnist_data['number_arr'][number] = 1.0

        NUM_POPULATION = 20

        print('Generating model...')
        models = []
        for _ in range(NUM_POPULATION):
            model = TestKerasGenetic.generate_model()
            models.append(model)

        print('Evolution...')
        while True:
            best_model = None
            min_percent_correct = sys.maxsize
            max_percent_correct = 0
            for model in models:
                percent_correct = 100 * TestKerasGenetic.count_correct_ratio(model, mnist_datas[60000:])
                if percent_correct > max_percent_correct:
                    best_model = model
                max_percent_correct = max(max_percent_correct, percent_correct)
                min_percent_correct = min(min_percent_correct, percent_correct)
            print(f'Percent correct {min_percent_correct:.1f}% - {max_percent_correct:.1f}%')

            # Now use this best model as a template for the population
            models = []
            for _ in range(NUM_POPULATION):
                model = keras.models.clone_model(best_model)  # Notice this does not copy the internal weights
                weights = best_model.get_weights()
                # We do mutation
                for layer in weights:
                    Genetic.mutate(layer, prob_mutation=0.2, mu=0.0, sigma=0.1)
                model.set_weights(weights)
                models.append(model)
