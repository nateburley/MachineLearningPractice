# Included files (figure out how to condense this shit. This is annoying)
import quandl as Q
Q.ApiConfig.api_key = "tS-9Qn82WHNUQwoysxTt"
import pandas as pd
import numpy as np
# cross_validation = model_selection
from sklearn import preprocessing, model_selection
from sklearn.linear_model import LinearRegression
import math
import datetime
import time
import pickle
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

# Gets Google stock prices from Quandl (https://www.quandl.com/data/WIKI-Wiki-EOD-Stock-Prices)
df = Q.get('WIKI/GOOGL')
# print(df.head()) #This print statement is to see the columns, which are used below

# Gets the columns we want to analyze; calculates high-low ratio and ratio of change
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume', ]]
df['HL_PCT'] = ((df['Adj. High'] - df['Adj. Close']) / df['Adj. Close']) * 100
df['PCT_Change'] = ((df['Adj. Close'] - df['Adj. Open']) /
                    df['Adj. Open']) * 100

# Filters the data to the desired columns
df = df[['Adj. Close', 'HL_PCT', 'PCT_Change', 'Adj. Volume', ]]

# Sets up our forecast (you can change later to forecast something else)
forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)
# Predicts "10 percent" out of the data frame
forecast_out = int(math.ceil(0.1 * len(df)))
# Label = Features that change 10% out
df['label'] = df[forecast_col].shift(-forecast_out)

# Prints the head, for shits and gigs
print(df.head())

#Define features and labels (features = X, Labels = Y)#########################
x = np.array(df.drop(['label'], 1))
# Sometimes skip this step, for stuff like high frequency trading
x = preprocessing.scale(x)
x_lately = x[-forecast_out:]  # This will be used to make predictions
x = x[:-forecast_out:]  # Here the x values are defined like normal

df.dropna(inplace=True)
y = np.array(df['label'])

x_train, x_test, y_train, y_test = model_selection.train_test_split(
    x, y, test_size=0.2)

#Making the classifier##########################################################
clf = LinearRegression()  # You can very easily change this algorithm
clf = clf.fit(x_train, y_train)
accuracy = clf.score(x_test, y_test)

# Saves the classifier to a file, so you don't have to retrain every. Single. Time.
with open('practice_regression_pickle.pickle', 'wb') as f:
    pickle.dump(clf, f)
    # To reuse the classifier, do something like this:
    # pickle_in = open('practice_regression_pickle.pickle', 'rv'), clf = pickle.load(pickle_in) (On separate lines of course)
    # Realistically, you would have a file that trains the classifier, and one that loads it and makes predictions

# These should be the same length, because math
print("Length of X: {}   Length of Y: {}".format(len(x), len(y)))
print("{} percent accuracy, {} days in advance".format(
    accuracy * 100, forecast_out))

#Predicting Stuff###############################################################
forecast_set = clf.predict(x_lately)
print(forecast_set, accuracy, forecast_out)
df['Forecast'] = np.nan

#Graphing the predictions#######################################################
last_date = df.iloc[-1].name
last_unix = time.mktime(last_date.timetuple())
next_unix = last_unix + 86400  # Number of seconds in a day

# Formats the data into actual dates for graphing
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400  # Move up 1 day
    df.loc[next_date] = [np.nan for j in range(len(df.columns) - 1)] + [i]

#Graphs the data and the regression predictions (predicted stock prices here)
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

"""
Here's the best fit slope coded by handed, with numpy arrays of x and y values

from statistics import mean
import numpy as np
import matplotlib as plt
from matplotlib import style

style.use('fivethirtyeight')

#Sample arrays of "random" data
x_values = np.array([1,2,3,4,5,6])
y_values = np.array([2,5,6,9,10,13])

#Performs the actual regression, and gets the slope of the best fit line
def best_fit_slope_and_intercept = (xs, ys):
    slope = ( (mean(xs) * mean(ys)) - mean(ys * xs) ) /
            ( (mean(xs) * mean(xs)) - mean(xs * xs) )
    y_int = mean(ys) - slope*mean(xs)
    return slope, y_int
slope, y_int = best_fit_slope_and_intercept(x_values, y_values)

#Prints out the slope of the best fit line (should be about 2 for this set)
print("Best fit slope: {}  Best fit Y intercept: {}").format(slope, y_int)

#Makes the actual regression line
regression_line = [(slope * x) + y_int for x in x_values]

#Graphs the points and the regression line
plt.scatter(x_values, y_values)
plt.plot(x_values, regression_line)
plt.show()

#Make a prediction
trial_x = 8
prediction_y = (slope * trial_x) + y_int

"""