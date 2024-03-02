from my_evaluation import my_evaluation
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # Load training data
    data_train = pd.read_csv("/Users/varagantibasanthkumar/Desktop/DSCI-633/assignments/data/breast_cancer.csv")

    # Separate independent variables and dependent variables
    independent = ["age", "menopause", "tumor-size", "inv-nodes", "node-caps", "deg-malig", 
                   "breast", "breast-quad", "irradiat"]
    X = pd.get_dummies(data_train[independent], drop_first=True)
    y = data_train["Class"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Define the parameters for the models
    criteria = ["gini", "entropy"]
    max_depths = [2, 3, 4, 5]

    # Iterate over each combination of criterion and max depth
    for criterion in criteria:
        for max_depth in max_depths:
            # Initialize and train the DecisionTreeClassifier model
            clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)
            clf.fit(X_train, y_train)

            # Predict on testing data
            predictions = clf.predict(X_test)

            # Evaluate results
            metrics = my_evaluation(predictions, y_test)
            result = {}
            for target in clf.classes_:
                result[target] = {}
                result[target]["prec"] = metrics.precision(target)
                result[target]["recall"] = metrics.recall(target)
                result[target]["f1"] = metrics.f1(target)
            
            # Print model parameters and performance metrics
            print(f"Impurity Metric = {criterion}, Max Depth = {max_depth}")
            print(result)
            print()  # Adds a newline for better readability between models
