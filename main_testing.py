import pandas as pd
import numpy as np
# from keras.models import Sequential
# from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.linear_model import LogisticRegression



# data = pd.read_csv('datasets/main_janice.csv')
# transcript = pd.read_csv('datasets/transcripts_liwc.csv')

# data = data[data['cluster_tags_10'] == 6]

# comb_data = transcript.merge(data, how='inner', left_on='Source (B)', right_on='url')
# comb_data.to_csv('combined_3.csv')

def log_reg(comb_data, cluster):
# create comparison measure
    avgview = np.mean(comb_data['views'])
    comb_data['above_avg'] = comb_data[['views']] > avgview

    X = comb_data[['social','family','friend','percept','see','hear','feel','power','reward','focuspast','focuspresent','focusfuture','leisure','home']]
    y = comb_data[['views']]

    ss = StandardScaler()
    X = ss.fit_transform(X)
    y = y.values.ravel()

    accuracy = 0
    for i in range(0, 100):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        lr = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=10000)
        lr.fit(X_train, y_train)
        pred = lr.predict(X_test) > avgview
        y_test = y_test > avgview
        accuracy += accuracy_score(pred, y_test)

    print(f'Cluster #{cluster} Accuracy: {accuracy/100}')

for i in range(2,3):
    data = pd.read_csv('datasets/main_janice.csv')
    transcript = pd.read_csv('datasets/transcripts_liwc.csv')

    data = data[data['cluster_tags_10'] == i]

    comb_data = transcript.merge(data, how='inner', left_on='Source (B)', right_on='url')
    log_reg(comb_data, i)

### making new dataset
#data = pd.read_excel("./main_janice.xlsx")

#cluster4 = data[data.cluster_tags_10 == 4]
#cluster6 = data[data.cluster_tags_10 == 6]

#cluster4.to_csv("./cluster4.csv")
#cluster6.to_csv("./cluster6.csv")

#cluster4 = pd.read_csv("./cluster4.csv")
#cluster6 = pd.read_csv("./cluster6.csv")
#cluster0 = pd.read_excel("./tedtalk_clusters.xlsx")
# data = pd.read_csv("datasets/main_janice.csv")
# cluster1 = data[data.cluster_tags_10 == 2]
# #print(cluster0)

# transcripts = pd.read_csv("datasets/transcripts_liwc.csv")
# combined0 = transcripts.merge(cluster1, how='inner', left_on='Source (B)', right_on='url')
# combined0.to_csv("datasets/combined_cluster2.csv")
