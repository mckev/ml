# Ref: https://keras.io/examples/mnist_cnn/

import keras
import numpy

# Ref: https://keras.io/examples/mnist_cnn/
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Convert x_train shape from (60000, 28, 28) into (60000, 28, 28, 1)
x_train_new = x_train.astype(numpy.float)
x_train_new /= 255
x_train_new = x_train_new.reshape(len(x_train), 28, 28, 1)

# Convert y_train shape from (60000,) into (60000, 10)
y_train_new = []
for number in y_train:
    y_new = numpy.zeros(shape=(10,))
    y_new[number] = 1.0
    y_train_new.append(y_new)
y_train_new = numpy.asarray(y_train_new)

print('Creating model...')
# Ref: https://www.dlology.com/blog/how-to-choose-last-layer-activation-and-loss-function/
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

print('Training...')
model.fit(x_train_new, y_train_new, batch_size=10, epochs=1)

print('Predicting...')
correct = 0
total = 0
for x, number in zip(x_test, y_test):
    x_new = x.astype(numpy.float)
    x_new /= 255
    x_new = x_new.reshape(1, 28, 28, 1)
    y_predict = model.predict(x_new)[0]  # predict() takes array, so we convert x into an array of size 1
    y_predict_index = numpy.argmax(y_predict)
    if y_predict_index == number:
        correct += 1
    total += 1
percent_correct = 100 * correct / total
print(f'Correct {correct} out of {total} ({percent_correct:.1f}% correct)')
