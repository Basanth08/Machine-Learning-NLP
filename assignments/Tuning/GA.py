import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from geneticalgorithm import geneticalgorithm as ga
from my_evaluation import my_evaluation

def f(X):
    # Load the Breast Cancer dataset
    df = pd.read_csv('/Users/varagantibasanthkumar/Desktop/DSCI-633/assignments/data/breast_cancer.csv')
    X_data = df.drop('Class', axis=1)
    y_data = df['Class']

    # Perform one-hot encoding on categorical features
    X_data = pd.get_dummies(X_data)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

    # Create a DecisionTreeClassifier with the given hyperparameters
    criterion = 'gini' if X[0] == 0 else 'entropy'
    max_depth = int(X[1])
    clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)

    # Train the classifier
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Calculate the macro F1 score using my_evaluation
    eval_obj = my_evaluation(y_pred, y_test)
    macro_f1 = eval_obj.f1(average='macro')

    # Negate the macro F1 score for maximization
    return -macro_f1

# Define the search space for the hyperparameters
varbound = np.array([[0, 1], [1, 12]])

# Create and run the Genetic Algorithm
model = ga(function=f, dimension=2, variable_type='int', variable_boundaries=varbound)
model.run()

# Print the best solution and objective function value
print("\nThe best solution found:")
print(model.output_dict['variable'])
print("\nObjective function:")
print(model.output_dict['function'])