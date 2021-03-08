import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


#######CLUSTER 7########

cluster7 = pd.read_csv("data/cluster7.csv")

################## film_date + duration + languages - Logistic Regression ##################

X = cluster7[['duration','film_date','languages']]
y = cluster7[['success']]

cl7_array = []

for i in range(0,20):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
    # build the model using training data
    logmodel = LogisticRegression(solver='liblinear')
    logmodel.fit(X_train, y_train)
    predictions = logmodel.predict(X_test)
    cl7_array.append(accuracy_score(y_test, predictions))
    
k1 = sum(cl7_array)/len(cl7_array)


#######CLUSTER 5########

cluster5 = pd.read_csv("data/cluster5.csv")

################## Title + duration + languages - Logistic Regression ##################

#----- use the whole result to run logistic regression -----

vectorizer = CountVectorizer()
transform_content = vectorizer.fit_transform(cluster5['title'])
transform_content_tdm = pd.DataFrame(transform_content.toarray(), columns=vectorizer.get_feature_names())
transform_content_tdm.to_csv("data/cluster5_vectors.csv")

X_selected_combined = pd.read_csv('data/cluster5_vectors.csv')
X_selected_combined[['duration']] = cluster5[['duration']]
X_selected_combined[['languages']] = cluster5[['languages']]
y_selected_combined = cluster5[['success']]

cl5_array = []
for i in range(0,20):

    X_train, X_test, y_train, y_test = train_test_split(X_selected_combined, y_selected_combined, test_size = 0.3)
    # build the model using training data
    logmodel = LogisticRegression(solver='liblinear')
    logmodel.fit(X_train, y_train)

    predictions = logmodel.predict(X_test)
    cl5_array.append(accuracy_score(y_test, predictions))

k2 = sum(cl5_array)/len(cl5_array)


print('Average Accuracy for cluster-7 over 20 runs is:')
print(k1)
# accuracy: ~71%
print('Average Accuracy for cluster-5 over 20 runs is:')
print(k2)
# accuracy: ~79%

