import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

"""
Cluster 4 didn't have strong correlation with sentiment analytical metrics. 
Taking a quantitative approach by count vectorizing transcripts.
"""

df = pd.read_csv("./combined_cluster4.csv")
X = df[['leisure', 'ppron']]
y = df[['views']].values.ravel()
median_views = np.median(y) # What I'm going to be using to judge 'success'
#is_successful = [view > median_views for view in y] # List of which ted talks were 'successful' or not
cv = CountVectorizer()
transcripts = df.transcript.tolist()
#y = (y > median_views)
X = cv.fit_transform(transcripts)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# Scaling values. Vector sizes are too massive.
ss = StandardScaler(with_mean=False)
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)
#X_train = ss.fit_transform(X_train)
#X_test = ss.fit_transform(X_test)

is_successful = [view > median_views for view in y_test]
logmodel = LogisticRegression(max_iter=10000)
logmodel.fit(X_train, y_train)
predictions = (logmodel.predict(X_test) > median_views)
y_test = (y_test > median_views)
print(accuracy_score(predictions, y_test))
#https://scikit-learn.org/stable/modules/preprocessing.html


#transcripts = cv.fit_transform(X['transcript'])
#X.transcript = transcripts
#print(X.transcript)
#print(X['transcript'])
#X['transcript'] = cv.fit_transform(X['transcript'])




