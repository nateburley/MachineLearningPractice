from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import csv
import random


"""
Decision Tree/ Random Forest Classifier for the predicition of credit card default
Dataset source: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
DT source: http://scikit-learn.org/stable/modules/tree.html
RF source: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
Results:
- Both seem to peak around 82.7 percent accuracy (max_depth=5, min_samples_leaf=9)
"""


# Loads the dataset from a file; returns a specified percentage of the dataset
def loadFile(filename, percentage):
    with open(filename) as RawData:
        data = csv.reader(RawData)
        dataset = list(data)
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
def getFeaturesAndLabels(dataset, split_ratio):
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
    return train_features, train_labels, test_features, test_labels


# Function that trains decision tree for classification
def trainDecisionTree(train_features, train_labels):
    classifier = tree.DecisionTreeClassifier(max_depth=5, min_samples_leaf=9)
    classifier.fit(train_features, train_labels)
    return classifier

# Function that trains random forest classifier
def trainRandomForest(train_features, train_labels):
    classifier = RandomForestClassifier(max_depth=7, random_state=0, verbose=1)
    classifier.fit(train_features, train_labels)
    return classifier


# Function that makes predictions with decision tree
def predictWithDecisionTree(classifier, test_features):
    predictions = classifier.predict(test_features)
    return predictions


# Function that makes predictions with random forest
def predictWithRandomForest(rf_classifier, test_features):
    predictions = rf_classifier.predict(test_features)
    return predictions


# Function that prints an array
def printArray(array):
    for i in range(len(array)):
        print(array[i])


#Function that computes the accuracy of the Classifier
def getAccuracy(predictions, test_labels, name):
    correct = 0
    for i in range(len(predictions)):
        if predictions[i] == test_labels[i]:
            correct += 1
    accuracy = float(correct / len(predictions))
    print("{} classification accuracy: {}%".format(name, accuracy * 100))


# Main
if __name__ == "__main__":
    for i in range(1, 11):
        print("{}%\ OF THE DATASET".format(i * 10))
        percentage = float(i) / float(10)
        dataset = loadFile("sample_data.csv", percentage)
        train_features, train_labels, test_features, test_labels = getFeaturesAndLabels(dataset, 0.8)
        print("Training features: {}   Training labels: {}".format(len(train_features), len(train_labels)))
        print("Test features: {}   Test labels: {}".format(len(test_features), len(test_labels)))
        # Trains the models
        classifier = trainDecisionTree(train_features, train_labels)
        rf_classifier = trainRandomForest(train_features, train_labels)
        # Makes predictions
        predictions = predictWithDecisionTree(classifier, test_features)
        rf_predictions = predictWithRandomForest(rf_classifier, test_features)
        # Scores the models
        getAccuracy(predictions, test_labels, 'Decision Tree')
        getAccuracy(rf_predictions, test_labels, 'Random Forest')
        print("\n")
