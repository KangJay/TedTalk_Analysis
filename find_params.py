import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

cluster = pd.read_csv("./combined_cluster0.csv").drop(labels=['published_date',
                                                               'ratings',
                                                               'languages', 
                                                               #'Unnamed: 0',
                                                               'description'
                                                               ],axis=1)
y = cluster[['views']]
X = cluster.iloc[:, 2:-15]#.drop(labels=['comments', 'Unnamed: 0.1'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


def run_gbr():
    GBR = GradientBoostingRegressor()

    parameters={'learning_rate': [0.01,0.02,0.03,0.04],
                  'subsample'    : [0.9, 0.5, 0.2, 0.1],
                  'n_estimators' : [100,500,1000, 1500],
                  'max_depth'    : [4,6,8,10]
            }
    grid_GBR = GridSearchCV(estimator=GBR, param_grid = parameters, cv = 2, n_jobs=-1)
    grid_GBR.fit(X_train, y_train)


    print(" Results from Grid Search " )
    print("\n The best estimator across ALL searched params:\n",grid_GBR.best_estimator_)
    print("\n The best score across ALL searched params:\n",grid_GBR.best_score_)
    print("\n The best parameters across ALL searched params:\n",grid_GBR.best_params_)
"""
 The best score across ALL searched params:
 0.013275394234448346

 The best parameters across ALL searched params:
 {'learning_rate': 0.01, 'max_depth': 4, 'n_estimators': 100, 'subsample': 0.1}

"""


def filter_features():

    column_ranges = [(0,10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60),
                     (60, 70), (70, 80), (80, 90), (90, 95)]
    maps=[]
    for start, end in column_ranges:
        temp = X.iloc[:,start:end]
        temp[['views']] = y
        maps.append(temp)
    cor = maps[0].corr()
    print(cor)
    for df in maps: 
        plt.figure(figsize=(12,10))
        cor = df.corr()
        sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
        plt.show()

 
       
        
        #X[['views']] =y 
    
    #plt.figure(figsize=(12,10))
    #cor = X.corr()
    #sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    #plt.show()

if __name__ == "__main__":
    filter_features()
    X_1 = sm.add_constant(X)
    model = sm.OLS(y, X_1).fit()
    #print(type(model.pvalues))
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(model.pvalues)

        """
        'you', 'we', 'i', 'shehe', 'ppron', 'they', 'auxverb' have 
        the most significance to 'views'

        """
