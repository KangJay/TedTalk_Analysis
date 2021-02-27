import pandas as pd

# data = pd.read_csv("ted_main.csv")
data = pd.read_csv("transcripts_liwc.csv")
values = data.loc[:, ["Source (A)", "Source (B)"]]

# data['transcript'] = data2[['transcript']]
data = data.drop(["Source (A)", "Source (B)"], axis=1)
# print(data.head())

# print(data.head())

### EM check groups
from sklearn.mixture import GaussianMixture

# data_em = data.loc[:, ['Tone', 'we', 'you', 'they', 'negate', 'quant', 'affect', 'social', 'family', 'health', 'work', 'informal']]
# data_em = data

# for i in range(1, 30):
#     model = GaussianMixture(n_components=i, init_params='random', max_iter=50)
#     model.fit(data_em)

#     yhat = model.predict(data_em)
#     # print(yhat)

#     print(f'{i} groups AIC: {model.aic(data_em)}')
#     print(f'{i} groups BIC: {model.bic(data_em)}')


### Clustering
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as sch

X = data.loc[:, ['Tone', 'we', 'you', 'they', 'negate', 'quant', 'affect', 'social', 'family', 'health', 'work', 'informal', 'Authentic']]
# X = data

# scaler = StandardScaler()
# scaler.fit_transform(X)

# times = 500
# accuracy = 0
# for i in range(times):
#     # KMeans Divisive clustering
#     kmeans = KMeans(n_clusters=4)
#     y_means = kmeans.fit_predict(X)

#     # centroids = kmeans.cluster_centers_
#     # print(centroids)
#     # print(y_means)
#     accuracy += accuracy_score(data['Disease'], y_means)
# print(f"Divisive Clustering Accuracy: {accuracy/times}")

kmeans = KMeans(n_clusters=6)
y_means = kmeans.fit_predict(X)

centroids = kmeans.cluster_centers_
# print(centroids)
print(y_means)
values['group'] = y_means
data['group'] = y_means


print(data['group'].value_counts())

for i in range(6):
    print(f"Group {i}: ")
    print(values.loc[values['group'] == i, ['Source (B)']].head())

    save = values.loc[values['group'] == i, ['Source (B)']]
    name = str(i) + "_data.csv"
    save.to_csv(name)
