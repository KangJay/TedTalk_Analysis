import pandas as pd

data = pd.read_csv("datasets/main_janice.csv")

data = data.loc[data['cluster_tags_10'] == 1, ['tags', 'cluster_tags_10']]

print(f"Length of Data: {len(data)}")

worddict = {}
for wordlist in data['tags']:
    wordlist = wordlist.strip('][').split(', ')

    for word in wordlist:
        if word in worddict:
            worddict[word] += 1
        else:
            worddict[word] = 1

worddict = dict(sorted(worddict.items(), key=lambda item: item[1])) 
print(worddict)