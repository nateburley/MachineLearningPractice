"""
Contains class with neural network, and helper functions

Author: Nathaniel M. Burley
Date: 24th March 2021

TODO: First, make it my own by adding another hidden layer
TODO: Secondly, generalize it with loops so it works for n-layers
"""
# Imports and constants
import numpy as np



# Sigmoid function defined here. Fucking numpy...
def sigmoid(x):
    return 1/(1 + np.exp(-x))



class NeuralNetwork(object):
    def __init__(self):
        # Layers defined (hard coded for now. Will make variable later)
        self.input_layer_size = 1
        self.hidden_layer_size = 5
        self.output_layer_size = 1

        # Weights defined (hard coded for now. Will make list later)
        self.W1 = np.ones((self.input_layer_size, self.hidden_layer_size))
        self.W2 = np.ones((self.hidden_layer_size, self.output_layer_size))

        # Set an activation function
        self.activation_name = "sigmoid"

        # Set our learning rate
        self.learning_rate = 0.1

        # Number of training epochs
        self.num_epochs = 100


    # "Setter" function for the activation (so we can experiment with different ones)
    def setHyperParameters(self, act_name='sigmoid', learning_rate=0.1, num_epochs=100):
        self.activation_name = act_name
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
    

    # Function that displays the model's architecture
    def displayArchitecture(self):
        print("\n------------ NETWORK ARCHITECTURE ------------")
        print("Input layer size: {}".format(self.input_layer_size))
        print("Hidden layer size: {}".format(self.hidden_layer_size))
        print("Output layer size: {}".format(self.output_layer_size))
        print("Activation function: {}".format(self.activation_name))
        print("Training epochs: {}".format(self.num_epochs))
        print("Learning rate: {}".format(self.learning_rate))
        print("Weight matrix between input and hidden layer: \n{}".format(self.W1))
        print("Weight matrix between hidden and output layer: \n{}".format(self.W2))
        print("----------------------------------------------\n")

    
    # Function that returns the activation for a layer. Defaults to sigmoid
    def activationFunc(self, z):
        if self.activation_name == "sigmoid":
            return sigmoid(z)
        elif self.activation_name == "tanh":
            return np.tanh(z)
        elif self.activation_name == "relu":
            return np.relu(z)
        else:
            print("Activation function {} not available.Use setActivation with sigmoid, tanh, relu. \
                Defaulting to sigmoid.".format(self.activation_name))
            self.activation_name = "sigmoid"
            return sigmoid(z)
    
    # Function that returns the derivative of the activation function
    def activationFuncPrime(self, z):
        if self.activation_name == "sigmoid":
            sig = sigmoid(z)
            return sig * (1-sig)
        elif self.activation_name == "tanh":
            t = np.tanh(z)
            return 1-(t**2)
        elif self.activation_name == "relu":
            if np.relu(z) == 0: return 0.0
            else: return 1.0
        else:
            sig = sigmoid(z)
            return sig * (1-sig)


    # Function that computes the "forward" pass, returns a prediction
    def predict(self, X):
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.activationFunc(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.activationFunc(self.z3)
        return yHat
    

    # Function that predicts classes for multiple samples
    def predictClasses(self, X_test):
        y_pred = []

        # Make a prediction for each sample
        for x in X_test:
            y_pred.append(self.predict(x))
        
        # Convert to numpy array and return predictions
        y_pred = np.array(y_pred)
        return y_pred



    # Function that computes the cost for the given training input
    def costFunc(self, yHat, y):
        J = 0.5 * sum((y - yHat) ** 2)
        return J
    

    # Function that "backprops", i.e. computes cost derivative with respect to weights
    def costFuncPrime(self, X, y):
        # Feed data in and make a prediction
        self.yHat = self.predict(X)

        # Compute the error delta for the output layer
        delta3 = np.multiply(-(y-self.yHat), self.activationFuncPrime(self.z3))

        # Compute the derivative of the cost function with respect to W2
        dJdW2 = np.dot(self.a2.T, delta3)

        # Compute the error delta for the hidden layer
        delta2 = np.dot(delta3, self.W2.T) * self.activationFuncPrime(self.z2)

        # Compute the derivative of the cost function with respect to W1
        dJdW1 = np.dot(X.T, delta2)

        # Output the gradients
        return dJdW1, dJdW2
    

    # Function that updates the weights
    # TODO: Replace this with a 'gradient descent' and actually update the weights to train...
    def updateWeights(self, X, y):
        # Compute the gradients
        dJdW1, dJdW2 = self.costFuncPrime(X, y)

        # Update the weights by simply subtracting the gradients in a step of the learning rate
        self.W1 = self.W1 - (self.learning_rate * dJdW1)
        self.W2 = self.W2 - (self.learning_rate * dJdW2)

    
    # Function that trains on a complete dataset (or batch, I suppose. Multiple samples.)
    def train(self, X_train, y_train, num_epochs=200):
        for epoch in range(0, self.num_epochs):
            for X, y in zip(X_train, y_train):
                self.updateWeights(X, y)

