import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import re, string
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction import text

"""
Will load in the 

"""
def load_data():
    # Line endings between Windows and Linux messed up my files. Converted to 
    # restore previous structure. Rest of you can probably use the csv file 
    # instead. This is the file Janice generated with clusters.
    all_data = pd.read_excel("./tedtalk_clusters.xlsx")
    clusters = []
    for i in range(10):
        clusters.append(all_data[all_data.cluster_tags_10 == i]['tags'].values.tolist())
    cluster_text_lists = {}
    for i in range(0, len(clusters)):
        cluster_text_lists[i] = []
        for line in clusters[i]:
            cluster_text_lists[i].append(line.split(","))
    return cluster_text_lists

def top_n_words(tag_list, n_words = 20):
    wordlist = (" ".join(tag_list)).split(" ")
    wordcount = Counter(wordlist)
    return wordcount.most_common(n_words)

def textcleaner(row):
    row = row.lower()
    #remove urls
    row  = re.sub(r'http\S+', '', row)
    #remove mentions
    row = re.sub(r"(?<![@\w])@(\w{1,25})", '', row)
    #remove hashtags
    row = re.sub(r"(?<![#\w])#(\w{1,25})", '',row)
    #remove other special characters
    row = re.sub('[^A-Za-z .-]+', '', row)
    #remove digits
    row = re.sub('\d+', '', row)
    row = row.strip(" ")
    return row

def clean_text(clusterlist): # List indices separate cluster number. Each has
    # object in the list has the text of the tags for that cluster
    textlists = []
    for i in range(0, len(clusterlist)):
        textlists.append([textcleaner(item) for sublist in clusterlist[i] for item in sublist])
    return textlists

def convert_to_top_words(textlists):
    for i in range(0, len(textlists)):
        textlists[i] = top_n_words(textlists[i], n_words=20)
    return textlists

def gen_wordclouds(wordcount_list):
    plt.rcParams['figure.figsize'] = [16, 10]
    full_names = ['0: Music/Performance', 
                  '1: Nature/Environment', 
                  '2: Global Issues', 
                  '3: Culture/Entertainment', 
                  '4: STEM/Tech Teaching', 
                  '5: Other', 
                  '6: Social & Global Improvement', 
                  '7: Alternative & Green Energy', 
                  '8: Biology/Healthcare', 
                  '9: Sociology/Humanity']
    wc = WordCloud(stopwords=text.ENGLISH_STOP_WORDS, background_color="white",
                       colormap="Dark2", max_font_size=150, random_state=42)
    for index, cluster in enumerate(wordcount_list):
        wc.generate(cluster)
        plt.subplot(4, 4, index+1)
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("on")
        plt.title(full_names[index])
    plt.show()
        #print(index)
        #wc.generate(cluster)
        #plt.subplot(4,)

if __name__ == "__main__":
    clusters = load_data()
    textlists = clean_text(clusters)
    for i in range(0, len(textlists)):
        textlists[i] = ' '.join(textlists[i])
    #while True: pass
    #text_counts = convert_to_top_words(textlists)
    gen_wordclouds(textlists)
