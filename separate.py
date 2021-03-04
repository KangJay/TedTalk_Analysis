import pandas as pd


#data = pd.read_excel("./main_janice.xlsx")

#cluster4 = data[data.cluster_tags_10 == 4]
#cluster6 = data[data.cluster_tags_10 == 6]

#cluster4.to_csv("./cluster4.csv")
#cluster6.to_csv("./cluster6.csv")

cluster4 = pd.read_csv("./cluster4.csv")
cluster6 = pd.read_csv("./cluster6.csv")
transcripts = pd.read_csv("./transcripts_liwc.csv")
combined6 = transcripts.merge(cluster6, how='inner', on='url')
combined6.to_csv("./combined_cluster6.csv")
#combined4 = transcripts[transcripts.url == cluster4.url]
#combined6 = transcripts[transcripts.url == cluster6.url]

#print(combined4)
