"""
Contains class with neural network, and helper functions

Author: Nathaniel M. Burley
Date: 24th March 2021

TODO: Fix the predict function, not working with second hidden layer...
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
        self.hidden_layer1_size = 5
        self.hidden_layer2_size = 5
        self.output_layer_size = 1

        # Weights defined (hard coded for now. Will make list later)
        self.W1 = 0.5 * np.ones((self.input_layer_size, self.hidden_layer1_size))
        self.W2 = 0.5 * np.ones((self.hidden_layer1_size, self.hidden_layer2_size))
        self.W3 = 0.5 * np.ones((self.hidden_layer2_size, self.output_layer_size))

        # Set an activation function
        self.activation_name = "sigmoid"

        # Set our learning rate
        self.learning_rate = 2
        self.learn_delta = 0.6  # Parameter that decreases learning rate as we converge

        # Number of training epochs
        self.num_epochs = 100

        ### PARAMETERS FOR CONJUGATE GRADIENT
        self.p1_0 = 0


    # "Setter" function for the activation (so we can experiment with different ones)
    def setHyperParameters(self, act_name='sigmoid', learning_rate=0.1, num_epochs=100):
        self.activation_name = act_name
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
    

    # Function that displays the model's architecture
    def displayArchitecture(self):
        print("\n------------ NETWORK ARCHITECTURE ------------")
        print("Input layer size: {}".format(self.input_layer_size))
        print("Hidden layer 1 size: {}".format(self.hidden_layer1_size))
        print("Hidden layer 2 size: {}".format(self.hidden_layer2_size))
        print("Output layer size: {}".format(self.output_layer_size))
        print("Activation function: {}".format(self.activation_name))
        print("Training epochs: {}".format(self.num_epochs))
        print("Learning rate: {}".format(self.learning_rate))
        print("Weight matrix between input and hidden layer: \n{}".format(self.W1))
        print("Weight matrix between hidden and output layer: \n{}".format(self.W2))
        print("Weight matrix between hidden and output layer: \n{}".format(self.W3))
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
        #print("Shape of z2: {}".format(self.z2.shape))
        self.a2 = self.activationFunc(self.z2)
        self.z3 = np.matmul(self.a2, self.W2)
        #print("Shape of z3: {}".format(self.z3.shape))
        self.a3 = self.activationFunc(self.z3)
        self.z4 = np.dot(self.a3, self.W3)  # Should this be dot? Or just multiply
        #print("Shape of z4: {}".format(self.z4.shape))  # Why the fuck is this 1,1...
        yHat = self.activationFunc(self.z4)
        yHat = yHat[0,0]
        #print("yHat shape: {}".format(yHat.shape))
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
        delta4 = np.multiply(-(y-self.yHat), self.activationFuncPrime(self.z4))
        #print("Delta4 shape: {}".format(delta4.shape))

        # Compute the derivative of the cost function with respect to W3
        dJdW3 = np.dot(self.a3.T, delta4)
        #print("dJdW3 shape: {}".format(dJdW3.shape))

        # Compute the error for the second hidden layer
        delta3 = np.dot(delta4, self.W3.T) * self.activationFuncPrime(self.z3)
        #print("Delta3 shape: {}".format(delta3.shape))

        # Compute the derivative with respect to W2
        dJdW2 = np.dot(self.a2.T, delta3)
        #print("dJdW2 shape: {}".format(dJdW2.shape))

        # Compute the error delta for the hidden layer
        delta2 = np.matmul(delta3, self.W2.T) * self.activationFuncPrime(self.z2)
        #print("Delta2 shape: {}".format(delta2.shape))

        # Compute the derivative of the cost function with respect to W1
        dJdW1 = np.dot(X.T, delta2)
        #print("dJdW1 shape: {}".format(dJdW1.shape))

        # Output the gradients
        return dJdW1, dJdW2, dJdW3
    

    # Function that updates the weights
    # TODO: Replace this with a 'gradient descent' and actually update the weights to train...
    def updateWeights(self, X, y):
        # Compute the gradients
        dJdW1, dJdW2, dJdW3 = self.costFuncPrime(X, y)

        # Update the weights by simply subtracting the gradients in a step of the learning rate
        self.W1 = self.W1 - (self.learning_rate * dJdW1)
        self.W2 = self.W2 - (self.learning_rate * dJdW2)
        self.W3 = self.W3 - (self.learning_rate * dJdW3)
    
    # Function that updates the weights with conjugate gradient


    
    # Function that trains on a complete dataset (or batch, I suppose. Multiple samples.)
    def train(self, X_train, y_train, num_epochs=100):
        for epoch in range(0, self.num_epochs):
            for X, y in zip(X_train, y_train):
                self.updateWeights(X, y)  # Update weights by taking step in gradient direction
                #self.learning_rate = self.learning_rate / (1 + (epoch ** self.learn_delta))


