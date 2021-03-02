import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer

main_url = 'https://raw.githubusercontent.com/KangJay/TedTalk_Analysis/main/ted_main.csv'
main = pd.read_csv(main_url, index_col = 0)

vectorizer = CountVectorizer()
transform_content = vectorizer.fit_transform(main['tags'])
transform_content_tdm = pd.DataFrame(transform_content.toarray(), columns=vectorizer.get_feature_names())

X = transform_content_tdm

scaler = StandardScaler()
scaler.fit_transform(X)

kmeans = KMeans(n_clusters = 10)
y_means = kmeans.fit_predict(X)

centroids = kmeans.cluster_centers_
print(centroids)
print(y_means)

main['cluster_tags_10'] = y_means
# main.to_csv('main.csv')
