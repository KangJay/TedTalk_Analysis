
def columnizeRatings(inputFile, tagNum, outputFile):
    import pandas as pd
    import json
    import re

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
    import pandas as pd
    import numpy as np
    
    dataFrame = pd.read_csv(inputFile)
    views = dataFrame['views'].values
    views_mean = dataFrame['views'].mean()
    dataFrame[['views_binary']] = (views >= views_mean)
    dataFrame.to_csv(outputFile)
    print("Success: binarizeViews")
    
    
def runLogModel(inputFile, times, outputFile):
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix
    
    rawData = pd.read_csv(inputFile)
    X = rawData.iloc[:,2:-4].values
    y = rawData[['views_binary']].values

    ss = StandardScaler()
    X = ss.fit_transform(X)  
    
    result = {}
    accuracy_list = []
    for i in range(times):
        print(i)
    
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, shuffle=True)
    
        # build the model using training data
        logmodel = LogisticRegression(solver='liblinear')
        logmodel.fit(X_train, y_train)
    
        predictions = logmodel.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        accuracy_list.append(round(accuracy, 2))
        print(confusion_matrix(y_test, predictions))

    avg = sum(accuracy_list) / len(accuracy_list)
    accuracy_list.append(round(max(accuracy_list), 2))
    accuracy_list.append(round(min(accuracy_list), 2))
    accuracy_list.append(round(avg, 2))
    result['LogModel'] = accuracy_list
        
    df = pd.DataFrame(result)
    df_new = df.rename(index={times : 'Max', times+1 : 'Min', times+2 : 'Avg'})
    df_new.to_csv(outputFile)

def runKNN(inputFile, times, numNeighbors, outputFile):
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix
    
    rawData = pd.read_csv(inputFile)
    X = rawData.iloc[:,2:-4].values
    y = rawData[['views_binary']].values

    ss = StandardScaler()
    X = ss.fit_transform(X)

    result = {}
    accuracy_list = []
    for i in range(times):
        print(i)     

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        knn = KNeighborsClassifier(n_neighbors=numNeighbors)
        knn.fit(X_train, y_train)        
        predictions = knn.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        accuracy_list.append(round(accuracy, 2))

    avg = sum(accuracy_list) / len(accuracy_list)
    accuracy_list.append(round(max(accuracy_list), 2))
    accuracy_list.append(round(min(accuracy_list), 2))
    accuracy_list.append(round(avg, 2))
    result['KNN'] = accuracy_list
        
    df = pd.DataFrame(result)
    df_new = df.rename(index={times : 'Max', times+1 : 'Min', times+2 : 'Avg'})
    df_new.to_csv(outputFile) 
    print("run kNN success!")


def runSVC(inputFile, times, kernelType, outputFile):
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix    
    from sklearn.svm import SVC

    rawData = pd.read_csv(inputFile)
    X = rawData.iloc[:,2:-4].values
    y = rawData[['views_binary']].values

    ss = StandardScaler()
    X = ss.fit_transform(X)

    result = {}
    accuracy_list = []
    for i in range(times):
        print(i)             
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = SVC(kernel=kernelType)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        accuracy_list.append(round(accuracy, 2))

    avg = sum(accuracy_list) / len(accuracy_list)
    accuracy_list.append(round(max(accuracy_list), 2))
    accuracy_list.append(round(min(accuracy_list), 2))
    accuracy_list.append(round(avg, 2))
    result['SVC'] = accuracy_list
        
    df = pd.DataFrame(result)
    df_new = df.rename(index={times : 'Max', times+1 : 'Min', times+2 : 'Avg'})
    df_new.to_csv(outputFile)         
        
    print("runSVC success!")
        
    
def corrMatrix(inputFile):
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    data = pd.read_csv(inputFile)
    X = data.iloc[:,2:-4]  #independent columns
    y = data[['views']]    #target column i.e price range

    data = pd.concat([X,y],axis=1)
    
    #get correlations of each features in dataset
    corrmat = data.corr()
    top_corr_features = corrmat.index
    print(top_corr_features)
    plt.figure(figsize=(20,20))
    #plot heat map
    g=sns.heatmap(data[top_corr_features].corr(), annot=True, cmap="RdYlGn")  
    
    
def runLinearRegress(inputFile):
    import pandas as pd
    import matplotlib.pyplot as plt
    import statsmodels.api as sm
    from sklearn.preprocessing import StandardScaler
    
    rawData = pd.read_csv(inputFile)

    X = rawData.iloc[:,2:-4].values
    y = rawData[['views']].values
    
    ss = StandardScaler()
    X = ss.fit_transform(X)
    X = sm.add_constant(X)
    y = ss.fit_transform(y)
    
    lr_model = sm.OLS(y, X).fit()
    print(lr_model.summary())
            

if __name__ == "__main__":
    rawFile = "main_janice.csv"
    createdFile = "ratings8.csv"
    outputFile = "accuracy_SVN_linear.csv"
    columnizeRatings(rawFile, 0, createdFile)
    binarizeViews(createdFile, createdFile)
    # runLogModel(createdFile, 20, outputFile)
    # runKNN(createdFile, 20, 3, outputFile)
    runSVC(createdFile, 20, "linear", outputFile)
    # corrMatrix(createdFile)

        
