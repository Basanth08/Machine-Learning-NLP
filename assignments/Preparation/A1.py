from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
#VbasanthKumar
#Myfirstassignment
if __name__ == "__main__":
    #  Load data
    data_train = pd.read_csv("/Users/varagantibasanthkumar/DSCI633/DSCI-633/fds/assignments/data/Iris_train.csv")
    #printing the 12th training point
    print(data_train.loc[11])
    # Printing the column "SepalWidthCm"
    print(data_train["SepalWidthCm"])
    # Printing the data points with "SepalWidthCm" < 2.5
    print(data_train[data_train["SepalWidthCm"]<2.5])
    # Separating independent variables and dependent variables
    independent = ["SepalLengthCm",	"SepalWidthCm",	"PetalLengthCm", "PetalWidthCm"]
    X = data_train[independent]
    Y = data_train["Species"]
    # Training the  model
    dsc = DecisionTreeClassifier()
    dsc.fit(X,Y)
    dsc.score(X,Y)
    # Load testing data
    data_test = pd.read_csv("/Users/varagantibasanthkumar/DSCI633/DSCI-633/fds/assignments/data/Iris_train.csv")
    X_test = data_test[independent]
    # Predict
    predictions = dsc.predict(X_test)
    # Predict probabilities
    probs = dsc.predict_proba(X_test)
    # Print results
    for i,pred in enumerate(predictions):
        print("%s\t%f" %(pred,max(probs[i])))
