import unittest

import keras
import numpy

from classes.mnist.mnist import Mnist


class TestKerasMnist(unittest.TestCase):
    """ Solve MNIST handwriting using Keras """

    model = None

    def setUp(self):
        print('Creating model...')
        # Ref: https://www.dlology.com/blog/how-to-choose-last-layer-activation-and-loss-function/

        # Simple model with 91% accuracy
        # model = keras.models.Sequential()
        # model.add(keras.layers.Flatten())
        # model.add(keras.layers.Dense(30, activation='sigmoid'))
        # model.add(keras.layers.Dense(10, activation='softmax'))
        # model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        # CNN model with 98.5% accuracy
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
        TestKerasMnist.model = model

    def test_learn_handwriting(self):
        model = TestKerasMnist.model
        mnist_datas = Mnist.retrieve_mnist_datas(filename='../../classes/mnist/mnist.raw')

        print('Training...')
        inputs = []
        expected_outputs = []
        for mnist_data in mnist_datas[:50000]:
            image_bytes = mnist_data['image_bytes']
            input = numpy.frombuffer(buffer=image_bytes, dtype='uint8').reshape(28, 28, 1)
            input = input / 255
            inputs.append(input)

            number = mnist_data['number']
            expected_output = numpy.zeros(shape=(10,))
            expected_output[number] = 1.0
            expected_outputs.append(expected_output)

        # Most efficient shape would be (50000, 28, 28, 1) as input, and (50000, 10) as expected output
        # numpy.stack() converts list of numpy arrays into a numpy array (with 1 dimension added)
        inputs = numpy.stack(inputs)
        expected_outputs = numpy.stack(expected_outputs)
        model.fit(inputs, expected_outputs, epochs=1)

        print('Predicting...')
        inputs = []
        for mnist_data in mnist_datas[50000:]:
            image_bytes = mnist_data['image_bytes']
            input = numpy.frombuffer(buffer=image_bytes, dtype='uint8').reshape(28, 28, 1)
            input = input / 255
            inputs.append(input)
        # Most efficient shape would be (20000, 28, 28, 1) as input
        inputs = numpy.stack(inputs)

        guessed_outputs = model.predict(inputs)
        correct = 0
        i = 0
        for mnist_data in mnist_datas[50000:]:
            number = mnist_data['number']
            guessed_number = numpy.argmax(guessed_outputs[i])
            if guessed_number == number:
                correct += 1
            i += 1
        total = i
        percent_correct = 100 * correct / total
        print(f'Correct {correct} out of {total} ({percent_correct:.1f}% correct)')
        self.assertGreater(percent_correct, 98)
