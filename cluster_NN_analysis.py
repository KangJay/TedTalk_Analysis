import pandas as pd
from statsmodels.formula.api import ols
import statsmodels.api as sm
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

"""
P-values
number      0.038456
cause       0.069490
leisure     0.039940
focusfuture 0.05
death       0.043534
male        0.049835
"""

def linear_regression():
    c6 = pd.read_csv("./combined_cluster6.csv")
    y = c6[['views']]
    X = c6[['number', 'death', 'male', 'comments']]
    X = sm.add_constant(X)
    lr_model = sm.OLS(y,X).fit()
    print(lr_model.summary())

def nn_classify():
    c6 = pd.read_csv("./cluster0_modified.csv")
    avg_views =  np.mean(c6['views'].values.ravel())
    #med_views = np.median(c6['views'].values.ravel())
    is_successful = (c6['views'] > avg_views)
    c6['is_successful'] = is_successful
    y = c6['is_successful'].values
    X = c6.drop(labels=['views', 'is_successful', 'comments'], axis=1)
    input_dims = X.shape[1]
    X = X.values
    ss = StandardScaler()
    X = ss.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    nn = Sequential()
    nn.add(Dense(input_dims, activation='relu'))
    nn.add(Dense(70, activation='relu'))
    nn.add(Dense(55, activation='relu'))
    nn.add(Dense(30, activation='relu'))
    nn.add(Dense(9, activation='relu'))
    nn.add(Dense(1, activation='sigmoid')) #output layer
    nn.compile(loss='binary_crossentropy', metrics=['accuracy'])
    nn.fit(X_train, y_train, verbose=0, epochs=100)
    y_hat = nn.predict(X_test)
    y_hat = (y_hat > 0.5) 
    avg_acc = accuracy_score(y_test, y_hat)
    print(f"Average Accuracy for AVERAGE VIEWS: {avg_acc}")
    return avg_acc
    #print(f"Average Accuracy for MEDIAN VIEWS: {median_acc}")


if __name__ == "__main__":
    accuracies = []
    for i in range(30):
        accuracies.append(nn_classify())
    print(f"Average Accuracy: {sum(accuracies) / len(accuracies)}")

