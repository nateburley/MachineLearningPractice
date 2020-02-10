"""
This program contains a decision tree that will predict whether or not a given passenger would have survived
the Titanic. 

Results: 
Decision tree, impute NaN Age with median— 77.1%
Decision tree, impute NaN Age with mean— 70%

Data/Challenge: https://www.kaggle.com/c/titanic
Decision Tree Help: https://www.datacamp.com/community/tutorials/decision-tree-classification-python
Imputation Tips: https://machinelearningmastery.com/handle-missing-data-python/
                 https://scikit-learn.org/stable/modules/impute.html

Author: Nathaniel M. Burley
"""
# Import statements
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Declare our datafiles
train_file = "data/train.csv"
test_file = "data/test.csv"



############################################## READ AND PREPROCESS DATA #########################################

# Read in the data
train_df = pd.read_csv(train_file, header=0)
kaggle_df = pd.read_csv(test_file)

# Convert all rows to numerical; replace NaNs with the mean
train_df["Sex"] = np.where(train_df["Sex"] == "male", 0, 1)
train_df["Embarked"] = np.where(train_df["Embarked"] == "S", 0, 1)
train_df["Age"] = train_df.fillna(train_df.Age.median())

print(train_df.head(5))
print(train_df.columns)

# Remove unnecessary columns, create training and testing sets
x_df = train_df.drop(["Survived", "Name", "Ticket", "Cabin"], axis="columns")
y_df = train_df.Survived
x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.2)

# Print a heat map of correlations of the data
corr = train_df.corr()
plt.figure(figsize=(12,10))
cor = train_df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()
plt.savefig("titanic_correlation_matrix.png")



############################################## READ AND PREPROCESS DATA #########################################

tree_clf = DecisionTreeClassifier()
tree_clf = tree_clf.fit(x_train, y_train)
y_pred = tree_clf.predict(x_test)
print("Accuracy: {}".format(metrics.accuracy_score(y_test, y_pred)))