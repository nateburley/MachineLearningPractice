"""
Practice program for learning to make simple neural networks using Keras.
Source: https://towardsdatascience.com/how-to-build-a-neural-network-with-keras-e8faa33d0ae4
"""
# IMPORTS/DATA EXTRACTION ###################################################################################
import numpy as np
from keras.utils import to_categorical
from keras import models
from keras import layers

# Gets the data (imdb dataset)
from keras.datasets import imdb
(training_data, training_labels), (testing_data, testing_labels) = imdb.load_data(num_words=10000)
data = np.concatenate((training_data, testing_data), axis=0)
labels = np.concatenate((training_labels, testing_labels), axis=0)

# Vectorizes the data- takes every review, and fills it with zeroes so it's of size 10k
def vectorize(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

data = vectorize(data)
labels = np.array(labels).astype("float32")

# Make training and testing sets
test_x = data[:10000]
test_y = labels[:10000]
train_x = data[10000:]
train_y = labels[10000:]

# Explores the data; ensures successful import of data
# print("Categories: ", np.unique(labels))
# print("Number of unique words: ", len(np.unique(np.hstack(data))))

# length = [len(i) for i in data]
# print("Average review length: ", np.mean(length))
# print("Standard deviation: ", round(np.std(length)))

# Function that gets the original text of a review (decodes the numerical values of the review)
def printOriginalText(review):
    index = imdb.get_word_index()
    reverse_index = dict([(value, key) for (key, value) in index.items()])
    decoded = " ".join( [reverse_index.get(i - 3, "#") for i in data[0]] )
    print(decoded)

# print("First review label: {}".format(labels[0]))
# print("First review:")
# printOriginalText(data[0])


# NEURAL NETWORK CONSTRUCTION/TRAINING ######################################################################

# Model defined
model = models.Sequential()

# Input layer
model.add(layers.Dense(50, activation='relu', input_shape=(10000,)))

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
# TODO: Explore other parameters here (loss functions, optimizers, metrics, etc.)
# https://www.google.com/search?q=keras+model.compile&oq=keras+model.com&aqs=chrome.1.69i57j0l5.3119j0j1&sourceid=chrome&ie=UTF-8
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Trains the model
# TODO: Play with different batch sizes, epochs, etc.
results = model.fit(train_x, train_y, epochs=3, batch_size=500, validation_data=(test_x, test_y))

# Evaluates performance of the model
print("\nValidation Accuracy: {}%".format(100 * np.mean(results.history["val_acc"])))