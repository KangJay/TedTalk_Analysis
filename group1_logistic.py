import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

data1 = pd.read_csv('group1_variables.csv')

vectorizer = CountVectorizer()
transform_content = vectorizer.fit_transform(data1['title'])
transform_content_tdm = pd.DataFrame(transform_content.toarray(), columns=vectorizer.get_feature_names())

################## Title + duration + languages - Logistic Regression ##################
#----- use the whole result to run logistic regression -----
X_combined = pd.read_csv('group1_vec.csv')
X_combined[['duration']] = data1[['duration']]
X_combined[['languages']] = data1[['languages']]
X_combined = X_combined.iloc[:, 1:]
y_combined = data1[['success']]

X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size = 0.3)

# build the model using training data
logmodel = LogisticRegression(solver='liblinear')
logmodel.fit(X_train, y_train)

predictions = logmodel.predict(X_test)

print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))

# accuracy: ~50%
#----- deleted meaningless columns (it, the, by, a ....and more) -----
X_selected_combined = pd.read_csv('group1_vec.csv')
X_selected_combined[['duration']] = data1[['duration']]
X_selected_combined[['languages']] = data1[['languages']]
y_selected_combined = data1[['success']]

X_train, X_test, y_train, y_test = train_test_split(X_selected_combined, y_selected_combined, test_size = 0.3)

# build the model using training data
logmodel = LogisticRegression(solver='liblinear')
logmodel.fit(X_train, y_train)

predictions = logmodel.predict(X_test)

print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))

# accuracy: ~80%

