#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 14:35:37 2021

@author: stlp
"""
import pandas as pd
import numpy as np

data = pd.read_csv("data/main_janice.csv")

cluster7 = data[data.cluster_tags_10 == 7]
cluster5 = data[data.cluster_tags_10 == 5]


cluster7 = cluster7[['comments','duration','film_date','languages',
                     'cluster_tags_10','views','title']]

cluster5 = cluster5[['comments','duration','film_date','languages',
                     'cluster_tags_10','views','title']]


cluster5['success'] = np.where(cluster5['views'] > cluster5['views'].mean(), 1, 0) 
cluster7['success'] = np.where(cluster7['views'] > cluster7['views'].mean(), 1, 0) 

cluster5.to_csv("data/cluster5.csv")
cluster7.to_csv("data/cluster7.csv")






