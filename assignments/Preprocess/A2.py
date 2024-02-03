import my_preprocess
import pandas as pd
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import random

##################################################
random.seed(42)
if __name__ == "__main__":
    # Loading the training data
    data_train = pd.read_csv("/Users/varagantibasanthkumar/Desktop/DSCI-633/assignments/data/Iris_train.csv")
   
    independent = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
    X = data_train[independent]
    y = data_train["Species"]
    # X_test = data_test[independent]
    
    print("Before normalization:")
    print("Min Sepal Length = " + str(np.min(X["SepalLengthCm"])))
    print("Mean Sepal Length = " + str(np.mean(X["SepalLengthCm"])))
    print("Max Sepal Length = " + str(np.max(X["SepalLengthCm"])))

    print("\nIris class distribution:")
    print(Counter(y))

    normalizers = ["Min-Max", "Standard_Score", "L1", "L2"]
    for norm in normalizers:
        print("Running " + norm + " normalizer")
    
        normalizer = my_preprocess.my_normalizer(norm=norm) 
        X_train_norm = normalizer.fit_transform(X)

        print("After normalization:")
        print("Min Sepal Length =", np.min(X_train_norm[:, 0]))
        print("Mean Sepal Length =", np.mean(X_train_norm[:, 0]))
        print("Max Sepal Length =", np.max(X_train_norm[:, 0]))

    # Preprocess (train)
    # normalizer = my_preprocess.my_normalizer(norm = "L2", axis = 1)
    # X_norm = normalizer.fit_transform(X)

    # Perform stratified sampling
        sample = my_preprocess.stratified_sampling(y, ratio = 0.5, replace = False)
        X_sample = X_train_norm[sample]
        y_sample = y[sample].to_numpy()

        print("\nSample class distribution:")
        print(Counter(y_sample))

        clf = DecisionTreeClassifier()
        clf.fit(X_sample, y_sample)
    
    # Load testing data
        data_test = pd.read_csv("/Users/varagantibasanthkumar/Desktop/DSCI-633/assignments/data/Iris_train.csv")
        X_test = data_test[independent]
    # Preprocess (test)
        X_test_norm = normalizer.transform(X_test)
    # Predict
    # predictions = clf.predict(X_test_norm)
        predictions = clf.predict(X_test_norm)
    
    # Output predictions on test data
        print("\nModel predictions:")
        print(predictions)
