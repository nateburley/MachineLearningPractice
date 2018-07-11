"""
This program uses a neural network to predict if someone will default on a credit card payment;
Dataset source: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
Keras source: keras.io

Results:
Current build at ~78.18010042161124% accuracy 
"""

import numpy as np
from keras.utils import to_categorical
from keras import models
from keras import layers
import csv
import random

# Loads the dataset from a file; returns a specified percentage of the dataset
def loadFile(filename, percentage):
    with open(filename) as RawData:
        data = csv.reader(RawData)
        dataset = list(data)
        print("Number of rows: {}".format(len(dataset)))
        for i in range(len(dataset)):
            dataset[i] = [float(x) for x in dataset[i]]
            del(dataset[i][0])
        # Only uses given percentage of dataset
        num_samples = percentage * len(dataset)
        while len(dataset) > num_samples:
            index = random.randrange(0, len(dataset))
            del(dataset[index])
        return dataset


# Function that extracts the features, and the labels (testing and training)
def getFeaturesAndLabels(dataset, split_ratio=1):
    train_size = int(100 * split_ratio)
    train_features = []
    test_features = []
    train_labels = []
    test_labels = []
    for i in range(len(dataset)):
        index = random.randint(1,101)
        if index <= train_size:
            train_labels.append(dataset[i][-1])
            del(dataset[i][-1])
            train_features.append(dataset[i])
        else:
            test_labels.append(dataset[i][-1])
            del(dataset[i][-1])
            test_features.append(dataset[i])
    
    # Converts the training/testing data/labels to numpy arrays
    train_features = np.asarray(train_features)
    train_labels = np.asarray(train_labels)
    test_features = np.asarray(test_features)
    test_labels = np.asarray(test_labels)

    # For debugging
    print("Train features shape: {}".format(train_features.shape))
    print("Train labels shape: {}".format(train_labels.shape))
    print("Test features shape: {}".format(test_features.shape))
    print("Test labels shape: {}".format(test_labels.shape))

    return train_features, train_labels, test_features, test_labels

def buildAndTrainNetwork(train_x, train_y, test_x, test_y):
    # Model defined
    model = models.Sequential()

    # Input layer
    model.add(layers.Dense(50, activation='relu', input_shape=(23,)))

    # Hidden layers
    # TODO: Play with other layers! Add some, drop some..... Also different activation functions
    model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
    model.add(layers.Dense(50, activation = 'relu'))
    model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
    model.add(layers.Dense(50, activation = 'relu'))
    model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
    model.add(layers.Dense(50, activation = 'relu'))

    # Output layer 
    model.add(layers.Dense(1, activation = 'sigmoid'))

    # Prints a summary of the construction of the model! (Layers, input/output shape, etc.)
    model.summary()

    # Compiles the model, which basically configures it for training
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Trains the model
    results = model.fit(train_x, train_y, epochs=5, batch_size=500, validation_data=(test_x, test_y))

    # Evaluates performance of the model
    print("\nValidation Accuracy: {}%".format(100 * np.mean(results.history["val_acc"])))

if __name__ == "__main__":
    datafile = 'sample_data.csv'
    dataset = loadFile(datafile, 1)
    train_x, train_y, test_x, test_y = getFeaturesAndLabels(dataset, 0.7)
    buildAndTrainNetwork(train_x, train_y, test_x, test_y)
    