import pandas as pd
import json
import re

import numpy as np



def columnizeRatings(inputFile, tagNum, outputFile):
    data = pd.read_csv(inputFile)
    dataTag = data.loc[data['cluster_tags_10'] == tagNum]
    target = dataTag[['title', 'views', 'ratings', 'comments']]
    
    # step 1: columnize Ratings
    tag_name_set = []
    for index, row in target.iterrows():
        print(index)
        ratings = str(row['ratings'])
        ratings1 = re.sub('\'', '\"', ratings)
        ratings2 = json.loads(ratings1)
        
        for ratings_dict in ratings2:
            if ratings_dict['name']:
                tag_name_set.append(ratings_dict['name'].strip())
        
    tag_name_set = set(tag_name_set)
    
    from collections import defaultdict
    m = defaultdict(list)
    print("length: ", len(tag_name_set))
    for tag_name in tag_name_set: # https://stackoverflow.com/questions/6649361/creating-a-new-list-for-each-for-loop
        m[tag_name]
    
    
    title_list = []
    views_list = []
    comments_list = []
        
    for index, row in target.iterrows():
        print(index)
        title_list.append(row['title'])
        views_list.append(row['views'])
        comments_list.append(row['comments'])
        
        ratings = str(row['ratings'])
        ratings1 = re.sub('\'', '\"', ratings)
        ratings2 = json.loads(ratings1)
        
        for ratings_dict in ratings2:
            m[ratings_dict['name']].append(ratings_dict['count'])
    
    df_dict = {}
    for key, val_list in m.items():
        print("key : ", key)
        print("val_list : ", val_list)
        df_dict[key] = val_list
        
    df_dict['title'] = title_list
    df_dict['comments'] = comments_list
    df_dict['views'] = views_list
    
    new_data = pd.DataFrame(df_dict)
    new_data.to_csv(outputFile)
    print("Success: columnizeRatings")


def binarizeViews(inputFile, outputFile):
    dataFrame = pd.read_csv(inputFile)
    views = dataFrame['views'].values
    views_mean = dataFrame['views'].mean()
    dataFrame[['views_binary']] = (views >= views_mean)
    dataFrame.to_csv(outputFile)
    print("Success: binarizeViews")
    
    
def runLogModel(inputFile):
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix
    
    rawData = pd.read_csv(inputFile)
    X = rawData.iloc[:,2:-4].values
    y = rawData[['views_binary']].values
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, shuffle=True)
    
    
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.fit_transform(X_test)
    
    # build the model using training data
    logmodel = LogisticRegression(solver='liblinear')
    logmodel.fit(X_train, y_train)
    
    predictions = logmodel.predict(X_test)
    
    print(accuracy_score(y_test, predictions))
    print(confusion_matrix(y_test, predictions))
            

if __name__ == "__main__":
    rawFile = "main_janice.csv"
    createdFile = "ratings8.csv"
    columnizeRatings(rawFile, 1, createdFile)
    binarizeViews(createdFile, createdFile)
    runLogModel(createdFile)

        
