from sklearn import svm
import csv
import random


"""
Support Vector Machine (classifier) for the predicition of credit card default
Dataset source: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
SVM source: http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
"""


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



# Function that trains the classifier
def trainSVM(features, labels):
    clf = svm.SVC(verbose=True)
    clf.fit(features, labels)
    return clf


#Function that makes predictions (outputs array of predictions)
def predictWithSVM(classifier, test_features):
    predictions = classifier.predict(test_features)
    return predictions


# Function that prints an array
def printArray(array):
    for i in range(len(array)):
        print(array[i])


#Function that computes the accuracy of the Classifier
def getAccuracy(predictions, test_labels):
    correct = 0
    for i in range(len(predictions)):
        if predictions[i] == test_labels[i]:
            correct += 1
    accuracy = float(correct / len(predictions))
    print("Classification accuracy: {}%".format(accuracy * 100))



if __name__ == "__main__":
    dataset = loadFile("sample_data.csv")
    train_features, train_labels, test_features, test_labels = getFeaturesAndLabels(dataset, 0.8)
    print("Training features: {}   Training labels: {}".format(len(train_features), len(train_labels)))
    print("Test features: {}   Test labels: {}".format(len(test_features), len(test_labels)))
    classifier = trainSVM(train_features, train_labels)
    predictions = predictWithSVM(classifier, test_features)
    getAccuracy(predictions, test_labels)
