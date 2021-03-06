import pandas as pd


#data = pd.read_excel("./main_janice.xlsx")

#cluster4 = data[data.cluster_tags_10 == 4]
#cluster6 = data[data.cluster_tags_10 == 6]

#cluster4.to_csv("./cluster4.csv")
#cluster6.to_csv("./cluster6.csv")

#cluster4 = pd.read_csv("./cluster4.csv")
#cluster6 = pd.read_csv("./cluster6.csv")
#cluster0 = pd.read_excel("./tedtalk_clusters.xlsx")
data = pd.read_excel("./tedtalk_clusters.xlsx")
cluster0 = data[data.cluster_tags_10 == 0]
#print(cluster0)

transcripts = pd.read_csv("./transcripts_liwc.csv")
combined0 = transcripts.merge(cluster0, how='inner', on='url')
combined0.to_csv("./combined_cluster0.csv")
#combined4 = transcripts[transcripts.url == cluster4.url]
#combined6 = transcripts[transcripts.url == cluster6.url]

#print(combined4)
