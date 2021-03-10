import pandas as pd
import numpy as np

def chooseTag(tagNum, mainFile):
    
    main_janice = pd.read_csv("main_janice.csv")
    transcripts_liwc = pd.read_csv("transcripts_liwc.csv")
    
    main_janice = main_janice[["duration", "languages", "views", "url", "cluster_tags_10"]].reset_index(drop=True)
    transcripts_liwc = transcripts_liwc.iloc[:, 1:95].reset_index(drop=True)
    transcripts_liwc = transcripts_liwc.rename(columns={"Source (B)": "url"})
    
    data = main_janice.merge(transcripts_liwc, how='left', on='url')
    data = data.loc[data['cluster_tags_10'] == tagNum]
    data.to_csv(mainFile)
    print(data.head(5))

def binarizeViews(mainFile, outputFile):
    
    dataFrame = pd.read_csv(mainFile)
    views = dataFrame['views'].values
    views_mean = dataFrame['views'].mean()
    dataFrame[['views_binary']] = (views >= views_mean)
    dataFrame = dataFrame.dropna()
    dataFrame.to_csv(outputFile)
    print("Success: binarizeViews")
    
def corrMatrix(inputFile, beg, end):
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    data = pd.read_csv(inputFile)
    X = data.iloc[:,beg:end+1]  #independent columns
    y = data[['views']]    #target column i.e price range

    data = pd.concat([X,y],axis=1)
    
    #get correlations of each features in dataset
    corrmat = data.corr()
    top_corr_features = corrmat.index
    print(top_corr_features)
    plt.figure(figsize=(20,20))
    #plot heat map
    g=sns.heatmap(data[top_corr_features].corr(), annot=True, cmap="RdYlGn")      
    
def runSVC(inputFile, times, kernelType, outputFile):
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix    
    from sklearn.svm import SVC

    rawData = pd.read_csv(inputFile)
    X = rawData[["pronoun", "ipron", "anx", "AllPunc", "Quote"]].values
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
    print(df_new)
    df_new.to_csv(outputFile)         
        
    print("runSVC success!")

def runLogModel(inputFile, times, outputFile):
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix
    
    rawData = pd.read_csv(inputFile)
    X = rawData[["pronoun", "ipron", "anx", "AllPunc", "Quote"]].values
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
    X = rawData[["pronoun", "ipron", "anx", "AllPunc", "Quote"]].values
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
    
if __name__ == "__main__":
    mainFile = "combinedFile.csv"
    targetFile = "target9.csv"
    chooseTag(9, mainFile)
    binarizeViews(mainFile, targetFile)
    # corrMatrix(targetFile, 87, 98)
    # svcFile = "SVC_sigmoid9.csv"
    # runSVC(targetFile, 20, "sigmoid", svcFile)
    logModelFile = "LogModel9.csv"
    runLogModel(targetFile, 20, logModelFile)
    knnFile = "KnnFile9.csv"
    runKNN(targetFile, 20, 3, knnFile)