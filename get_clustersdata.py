#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 14:35:37 2021

@author: stlp
"""
import pandas as pd
import numpy as np
from collections import Counter
from collections import defaultdict



def top20(thislist):
    # First make a string out of the entire list
    BIGstr = " ".join(thislist)
    wordlist = BIGstr.split(" ")
    wordcount = Counter(wordlist)
    return(wordcount.most_common(20))

data = pd.read_excel("./tedtalk_clusters.xlsx")

cluster7 = data[data.cluster_tags_10 == 7]
cluster5 = data[data.cluster_tags_10 == 5]


# =============================================================================
# cluster7 = cluster7[['comments','duration','film_date','languages',
#                      'cluster_tags_10','views','title']]
# 
# cluster5 = cluster5[['comments','duration','film_date','languages',
#                      'cluster_tags_10','views','title']]
# 
# 
# cluster5['success'] = np.where(cluster5['views'] > cluster5['views'].mean(), 1, 0) 
# cluster7['success'] = np.where(cluster7['views'] > cluster7['views'].mean(), 1, 0) 
# 
# cluster5.to_csv("data/cluster5.csv")
# cluster7.to_csv("data/cluster7.csv")
# 
# =============================================================================





cluster5 = cluster5[['tags']]
textlist = cluster5['tags'].tolist()
#print(textlist)
#print(textlist)
#while True: pass
#textlist = [item for sublist in textlist for item in sublist]

x = []

for line in textlist:
    x.append(line.split(",")) 
#print(x)  
textlist1 = [item for sublist in x for item in sublist]
   
import re, string

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

cleaned_textlist = []

## list of all the tags
for t in textlist1:
    cleaned_textlist.append(textcleaner(t))
print(cleaned_textlist)
   
## top 20
print(top20(cleaned_textlist))
 

# =============================================================================
# for line in cleaned_textlist:
#     x.append(line.split(",")) 
# =============================================================================
    
#top20(cleaned_textlist)
    
#top20(cleaned_textlist)

        
