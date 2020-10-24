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

        print('Generating individuals...')
        individuals = []
        for _ in range(NUM_POPULATION):
            model = TestKerasGenetic.generate_model()
            individuals.append({
                'model': model,
                'score': None
            })

        print('Evolution...')
        while True:
            max_percent_correct = 0
            min_percent_correct = sys.maxsize
            for individual in individuals:
                percent_correct = 100 * TestKerasGenetic.count_correct_ratio(individual['model'], mnist_datas[60000:])
                individual['score'] = percent_correct
                max_percent_correct = max(max_percent_correct, percent_correct)
                min_percent_correct = min(min_percent_correct, percent_correct)
            print(f'Percent correct {min_percent_correct:.1f}% - {max_percent_correct:.1f}%')
            individuals = sorted(individuals, key=lambda el: el['score'], reverse=True)

            # Now use the best individuals as template for the new generation
            new_individuals = []
            for n in range(NUM_POPULATION):
                new_model = keras.models.clone_model(individuals[0]['model'])  # This does not copy the internal weights
                if n < NUM_POPULATION / 2:
                    # Retain the best half of the population
                    weights = individuals[n]['model'].get_weights()
                else:
                    # For the other half of the population, we use the best one (i.e. individuals[0])
                    weights = individuals[0]['model'].get_weights()
                # Do mutation
                for layer in weights:
                    Genetic.mutate(layer, prob_mutation=0.05, scale=0.2)
                new_model.set_weights(weights)
                new_individuals.append({
                    'model': new_model,
                    'score': None
                })
            individuals = new_individuals
