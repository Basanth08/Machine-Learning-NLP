import time
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from project import my_model
sys.path.insert(0, '../..')
sys.path.append('assignments')
from Evaluation.my_evaluation import my_evaluation

def test(data):
    # Prepare the data for training and testing
    y = data["fraudulent"]
    X = data.drop(['fraudulent'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # Create an instance of the custom model class (my_model)
    clf = my_model()
    # Train the model using the training data
    # - Fit the model to the training features (X_train) and labels (y_train)
    # - The model learns the patterns and relationships in the training data
    clf.fit(X_train, y_train)
    # Make predictions on the testing data
    # - Use the trained model to predict the class labels for the testing features (X_test)
    predictions = clf.predict(X_test)
    # Evaluate the model's performance
    eval = my_evaluation(predictions, y_test)
    # Calculate the F1 score for the positive class (fraudulent)
    f1 = eval.f1(target=1)
    # Return the F1 score as the result of the test function
    return f1

if __name__ == "__main__":
    start = time.time()
    # Load data
    data = pd.read_csv("/Users/varagantibasanthkumar/Desktop/DSCI-633/assignments/data/job_train.csv")
    # Replace missing values with empty strings
    data = data.fillna("")
    f1 = test(data)
    print("This is the F1 Score.")
    print("F1 score: %f" % f1)
    runtime = (time.time() - start) / 60.0
    print("This is the Runtime.")
    print(runtime)