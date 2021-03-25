"""
Contains functions that create and/or load a dataset

Author: Nathaniel M. Burley
Date: 25th March 2021
"""

import math
import numpy as np
from matplotlib import pyplot as plt



# Function that builds a dataset where the "rule" is the sine function
def buildSineDataset(num_train=100, num_test=25):
    # Build training dataset
    X_train = np.array([0.1 * x for x in range(0, num_train)])
    y_train = np.array([math.sin(x) for x in X_train])

    # Build testing dataset
    X_test = np.array([0.1 * x for x in range(num_train-1, num_train + num_test-1)])
    y_test = np.array([math.sin(x) for x in X_test])

    # For debugging
    print("Training shapes-- X: {} Y: {}".format(X_train.shape, y_train.shape))
    print("Testing shapes-- X: {} Y: {}".format(X_test.shape, y_test.shape))

    # Output our dataset
    return X_train, X_test, y_train, y_test


# Function that plots the dataset
def plotDataset(X_train, X_test, y_train, y_test):
    plt.plot(X_train, y_train, color="blue", label="Training Data")
    plt.plot(X_test, y_test, color="red", label="Testing Data")
    plt.legend()
    plt.show()
