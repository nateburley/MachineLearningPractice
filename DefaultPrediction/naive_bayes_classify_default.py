import csv
import sys
import math
import random


"""
Naive Bayes Classifier for the predicition of credit card default
Dataset source: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
Naive Bayes walkthrough: https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
Results:
- Seems pretty shitty for this specific problem (this implementation anyway)
- Only around 50 percent accuracy or below
"""


# ################""" DATASET FORMATTING AND SEPARATING """####################


# Loads the dataset from a file TODO: Take in a number of datapoints to use
def loadFile(filename):
    with open(filename) as RawData:
        data = csv.reader(RawData)
        dataset = list(data)
        for i in range(len(dataset)):
            dataset[i] = [float(x) for x in dataset[i]]
            del(dataset[i][0])
            # print(dataset[i])
        return dataset


# Splits the dataset into training and testing sets
def splitDataset(dataset, split_ratio):
    trainSize = int(len(dataset) * split_ratio)
    train_set = []
    copy = list(dataset)
    while len(train_set) < trainSize:
        index = random.randrange(len(copy))
        train_set.append(copy.pop(index))
    return [train_set, copy]


# Separates the dataset by classes
def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated


# Returns the average of an array of numbers
def mean(numbers):
    return sum(numbers) / float(len(numbers))


# Returns the standard deviation of an array of numbers
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / \
        float(len(numbers) - 1)
    return math.sqrt(variance)


# Calculates the mean and standard deviation of each dataset attribute
def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del(summaries[-1])
    return summaries


# Summarizes the entire dataset by class
def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for class_value, instances in separated.items():
        summaries[class_value] = summarize(instances)
    return summaries


# ################""" PROBABILITY CALCULATIONS AND PREDICTIONS """##############


# Calculates the probability of a dataset attribute
def calculateProbability(x, the_mean, the_stddev):
    exponent = math.exp(-(math.pow(x - the_mean, 2) / (2 * math.pow(the_stddev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * the_stddev)) * exponent


# Calculates the probabilities of every class
def calculateClassProbabilities(summaries, input_vector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            the_mean, the_stddev = classSummaries[i]
            x = input_vector[i]
            probabilities[classValue] *= calculateProbability(x, the_mean, the_stddev)
    return probabilities


# Makes predictions, using calculated probabilities and Bayes' Theorem
def predict(summaries, input_vector):
    probabilities = calculateClassProbabilities(summaries, input_vector)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if ((best_label is None) or (probability > best_prob)):
            best_prob = probability
            best_label = class_value
    return best_label


# Makes a list of predictions for every test isinstance
def getPredictions(summaries, test_set):
    predictions = []
    for i in range(len(test_set)):
         results = predict(summaries, test_set[i])
         predictions.append(results)
    return predictions


# Calculates accuracy of predictions made on test set
def getAccuracy(test_set, predictions):
    correct = 0
    # print(predictions)
    for i in range(len(test_set)):
        if test_set[i][-1] == predictions[i]:
            correct += 1
    return (correct / float(len(test_set))) * 100.0


# ################""" MAIN FUNCTION """#########################################


if __name__ == "__main__":
    # Reads in data, formats
    filename = 'sample_data.csv'
    dataset = loadFile(filename)
    training_set, test_set = splitDataset(dataset, 0.8)
    # Prepares the data
    summaries = summarizeByClass(dataset)
    # Tests the model
    predictions = getPredictions(summaries, test_set)
    accuracy = getAccuracy(test_set, predictions)
    print('Accuracy: {}%'.format(accuracy))

"""
# SciKit Implentation:

from sklearn.naive_bayes import GaussianNB

# Function that trains naive bayes classifier (Gaussian distribution)
def trainNaiveBayes(train_features, train_labels):
    classifier = GaussianNB()
    classifier.fit(train_features, train_labels)
    return classifier


# Function that makes predictions with a naive bayes Classifier
def predictWithNaiveBayes(nb_classifier, test_features):
    predictions = nb_classifier.predict(test_features)
    return predictions
"""
