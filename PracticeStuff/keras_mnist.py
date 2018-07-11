"""
Neural network in Keras to classify handwritten digits (MNIST)
"""
# IMPORTS/DATA RETRIEVAL ###################################################################################
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

# FUNCTIONS THAT GET DATA/TRAIN "BASE" NETWORK #############################################################

# Function that gets training and testing sets of MNIST data ("flat" vectors, not suitable for conv. nets)
def getMNISTFlat():
    # Gets the MNIST data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Flatten 28x28 images into a vector of 784 pixels
    num_pixels = x_train.shape[1] * x_train.shape[2] #784, in this case, but this can be reused!
    x_train = x_train.reshape(x_train.shape[0], num_pixels).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], num_pixels).astype('float32')

    # Normalize the inputs, to make them between 0-1
    x_train = x_train / 255
    x_test = x_test / 255

    # One hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]

    return x_train, y_train, x_test, y_test


# Function that trains a "base" (normal, non-convolutional) neural net
def trainBaseModel(x_train, y_train, x_test, y_test, num_pixels=784, num_classes=10):
    model = Sequential()

    # Construct model layers
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu')) #Input
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax')) #Output
    model.summary()

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    results = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=20)

    # Evaluate the model
    print("\nBaseline Accuracy: {}%".format(100 * np.mean(results.history["val_acc"])))
    print("Baseline Peak Accuracy: {}%".format(100 * np.amax(results.history['val_acc'])))


# FUNCTIONS THAT GET DATA/TRAIN CONV. NET ##################################################################

# Function that gets data in multiple dimensions, for conv. net
def getMultiDimData():
    # Gets the MNIST data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape the data into 3 dimensions (pixel value, width, height)
    x_train = x_train.reshape(x_train.shape[0], 1, 28, 28).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 1, 28, 28).astype('float32')

    # Normalize inputs from 0-255 to 0-1
    x_train = x_train / 255
    x_test = x_test / 255

    # One hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]

    return x_train, y_train, x_test, y_test


# Function that trains a convolutional neural network for MNIST classification
def trainConvNet(x_train, y_train, x_test, y_test, num_classes=10):
    # Define model
    model = Sequential()

    # Conv layers
    model.add(Conv2D(32, (5,5), input_shape=(1,28,28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Flatten()) # Lets conv output be processed by densely connected layers
    # Densely connected layers
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    results = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=200)

    # Evaluate the model
    print("\nSimple Conv Net Avg. Accuracy: {}%".format(100 * np.mean(results.history["val_acc"])))
    print("Simple Conv Net Peak Accuracy: {}%".format(100 * np.amax(results.history['val_acc'])))


# Function that trains a large, high end, state of the art convolutional neural net
def trainLargeConvNet(x_train, y_train, x_test, y_test, num_classes=10):
    # Define model
    model = Sequential()

    # Conv layers
    model.add(Conv2D(30, (5,5), input_shape=(1,28,28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(15, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Flatten()) # Lets conv output be processed by densely connected layers
    # Densely connected layers
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    results = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=200)

    # Evaluate the model
    print("\nLarge Conv Net Avg. Accuracy: {}%".format(100 * np.mean(results.history["val_acc"])))
    print("Large Conv Net Peak Accuracy: {}%".format(100 * np.amax(results.history['val_acc'])))



# MAIN FUNCTION ############################################################################################

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = getMNISTFlat()
    trainBaseModel(x_train, y_train, x_test, y_test)
    # x_train, y_train, x_test, y_test = getMultiDimData()
    # trainLargeConvNet(x_train, y_train, x_test, y_test)
    # print("\nDone!\n")