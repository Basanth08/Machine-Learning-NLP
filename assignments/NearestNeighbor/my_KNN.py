import pandas as pd
import numpy as np
from collections import Counter

class my_KNN:

    def __init__(self, n_neighbors=5, metric="minkowski", p=2):
        # metric = {"minkowski", "euclidean", "manhattan"}
        # p value only matters when metric = "minkowski"
        self.n_neighbors = int(n_neighbors)
        self.metric = metric
        self.p = p

    def fit(self, X, y):
        # X: pd.DataFrame, independent variables, float
        # y: list, np.array or pd.Series, dependent variables, int or str
        self.classes_ = list(set(list(y)))
        # write your code below
        self.X_train = X
        self.y_train = y
    

    def predict(self, X):
        # X: pd.DataFrame, independent variables, float
        # return predictions: list
        # write your code below
        predictions=[]
        for x in X.values:
            dist_list=[]
            for x_train in self.X_train.values:
                # Calculate the Euclidean distance
                Distance=np.power(np.sum(np.power(np.abs(x-x_train),self.p)),1/self.p)# By using the minkowski formula
                dist_list.append(Distance)# Adding the Distance to the list
            k_nearest_Indices=np.argsort(dist_list)[:self.n_neighbors]#Getting the indices of the Sorted Distance till the k 
            k_nearest_Labels=[self.y_train[i] for i in k_nearest_Indices]#Getting the labels for the same indices in the Distance
            most_common_Label=Counter(k_nearest_Labels).most_common(1)[0][0]
            predictions.append(most_common_Label)
        return predictions
        

    def predict_proba(self, X):
        # X: pd.DataFrame, independent variables, float
        # prob is a dict of prediction probabilities belonging to each categories
        # return probs = pd.DataFrame(list of prob, columns = self.classes_)
        # write your code below
        prob_list=[]
        for x in X.values:
            Dist_list=[]
            for x_train in self.X_train.values:
                Distance=np.power(np.sum(np.power(np.abs(x-x_train),self.p)),1/self.p)
                Dist_list.append(Distance)
            k_nearest_Indices=np.argsort(Dist_list)[:self.n_neighbors]
            k_nearest_Labels=[self.y_train[i] for i in k_nearest_Indices]
            class_count=Counter(k_nearest_Labels)
            prob_dict={j:class_count[j]/self.n_neighbors for j in self.classes_}
            prob_list.append(prob_dict)
            probs = pd.DataFrame(prob_list,columns=self.classes_)#returning into a dataframe
        return probs
