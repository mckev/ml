import unittest

import keras
import numpy

from classes.mnist.mnist import Mnist


class TestKerasMnist(unittest.TestCase):
    """ Solve MNIST handwriting using Keras """

    @staticmethod
    def generate_model():
        # Ref: https://www.dlology.com/blog/how-to-choose-last-layer-activation-and-loss-function/

        # Simple model with 91% accuracy
        # model = keras.models.Sequential()
        # model.add(keras.layers.Flatten())
        # model.add(keras.layers.Dense(30, activation='sigmoid'))
        # model.add(keras.layers.Dense(10, activation='softmax'))
        # model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        # CNN model with 98% accuracy
        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(keras.layers.Conv2D(filters=32, kernel_size=(6, 6), activation='relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(128, activation='sigmoid'))
        model.add(keras.layers.Dropout(0.25))
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
        mnist_datas = Mnist.retrieve_mnist_datas(filename='../../classes/mnist/mnist.raw')
        for mnist_data in mnist_datas:
            image_bytes = mnist_data['image_bytes']
            mnist_data['image_arr'] = numpy.frombuffer(buffer=image_bytes, dtype='uint8').reshape(28, 28, 1)
            mnist_data['image_arr'] = mnist_data['image_arr'] / 255

            number = mnist_data['number']
            mnist_data['number_arr'] = numpy.zeros(shape=(10,))
            mnist_data['number_arr'][number] = 1.0

        print('Generating model...')
        model = TestKerasMnist.generate_model()

        print('Training...')
        inputs = []
        expected_outputs = []
        for mnist_data in mnist_datas[:50000]:
            input = mnist_data['image_arr']
            inputs.append(input)
            expected_output = mnist_data['number_arr']
            expected_outputs.append(expected_output)
        # Most efficient shape would be (50000, 28, 28, 1) as input, and (50000, 10) as expected output
        # numpy.stack() converts list of numpy arrays into a numpy array (with 1 dimension added)
        inputs = numpy.stack(inputs)
        expected_outputs = numpy.stack(expected_outputs)
        model.fit(inputs, expected_outputs, epochs=1)

        print('Predicting...')
        percent_correct = 100 * TestKerasMnist.count_correct_ratio(model, mnist_datas[50000:])
        print(f'Correct: {percent_correct:.1f}%')
        self.assertGreater(percent_correct, 97.5)
