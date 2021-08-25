"""
This file loads the dataset, creates an instance of our neural network class, and trains and evaluates
it on some data. 

Author: Nathaniel M. Burley
Date Created: 25th March 2021
"""

import NN
import data_helper



# Function that computes the average prediction error
def avgPredictionError(y_test, y_pred):
    delta = (NN.np.subtract(y_test, y_pred)) ** 2
    avg_delta = NN.np.mean(delta)
    return avg_delta
        


if __name__ == "__main__":
    # Create an instance of the NeuralNetwork class
    test_model = NN.NeuralNetwork()

    # Set hyperparameters to default
    test_model.setHyperParameters(act_name='tanh', learning_rate=0.1, num_epochs=100)

    # Display our model's architecture for debugging
    test_model.displayArchitecture()

    # Get our dataset (sin function, for now)
    X_train, X_test, y_train, y_test = data_helper.buildSineDataset()

    # Plot the dataset [FOR DEBUGGING]
    # data_helper.plotDataset(X_train, X_test, y_train, y_test)

    # Make predictions with initial weights (LOL this will be shit)
    y_pred = test_model.predictClasses(X_test)
    print("ERROR BEFORE TRAINING: {}".format(avgPredictionError(y_test, y_pred)))

    # Plot predictions before training
    data_helper.plotPredictions(X_test, y_test, y_pred)
    


    ###################################### TRAIN AND EVALUATE ######################################

    # Actually train on the training data!
    test_model.train(X_train, y_train, optimizer='CG')

    # Print architecture after training (to visualize weights)
    test_model.displayArchitecture()

    # Make new predictions with the trained neural net
    y_pred = test_model.predictClasses(X_test)
    print("ERROR AFTER {} EPOCHS: {}".format(test_model.num_epochs, avgPredictionError(y_test, y_pred)))

    # Plot predictions and the real dataset
    data_helper.plotPredictions(X_test, y_test, y_pred)