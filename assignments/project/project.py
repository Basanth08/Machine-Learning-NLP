# Name: Basanth Varaganti
# Mail ID: bv8946@g.rit.edu
# For SVM Classifier
from sklearn.svm import SVC
# For Stratified k-Fold cross-validation
from sklearn.model_selection import StratifiedKFold
import pandas as pd
# For Hyperparameter Tuning
from sklearn.model_selection import RandomizedSearchCV 
# For Text Feature Extraction
from sklearn.feature_extraction.text import TfidfVectorizer

# Here we define the my_model class with two methods: fit and predict
class my_model():
    SVMFinalModel = None
    

    def fit(self, X, y):
        # Creating a TfidfVectorizer object to preprocess the text data
        # - Remove English stop words
        # - Normalize TF-IDF vectors using L2 norm
        self.preprocessor = TfidfVectorizer( stop_words='english', norm='l2', use_idf=True, smooth_idf=False )
        # Preprocess the text data by concatenating 'description', 'title', and 'location' columns
        # Apply fit_transform method to learn vocabulary, calculate TF-IDF weights, and transform text into a matrix of numerical features
        transform_variable = self.preprocessor.fit_transform(X [ "description" ] + ' ' + X[ "title" ]+' ' + X[ "location" ])

        # Define the hyperparameter search space for the SVM classifier
        HyperParameterGrid = {
            # Regularization parameter: controls the trade-off between training error and testing error
            'C': [ 0.1, 1, 10, 100 ] ,
            # Kernel function: specifies the kernel used in the SVM algorithm
            # - 'linear': suitable for linearly separable data
            # - 'rbf': radial basis function kernel, commonly used for non-linearly separable data
            # - 'poly': polynomial kernel, allows for modeling higher-order relationships
            'kernel': [ 'linear', 'rbf', 'poly' ] ,
            # Kernel coefficient: determines the influence of a single training example on the decision boundary
            'gamma': [ 'scale', 'auto' ]
        }

        # Initialize the SVM classifier
        svm_classifier = SVC()
        # Initialize the RandomizedSearchCV object for hyperparameter tuning
        # The classifier to be tuned svm_classifier

        param_search = RandomizedSearchCV(svm_classifier,
                                        param_distributions= HyperParameterGrid,# The hyperparameter search space 
                                        scoring='f1',  # The evaluation metric for assessing model performance
                                        cv=StratifiedKFold(n_splits=5), # The cross-validation strategy
                                        n_iter=10,   # The number of parameter settings to sample
                                        n_jobs=-1)  # Use all available CPU cores for parallel computation

        # Perform hyperparameter tuning using RandomizedSearchCV
        param_search.fit(transform_variable, y)

        # Getting the best optimized hyperparameters
        optimized_params = param_search.best_params_

        # Create the ultimate SVM model with the power of optimized hyperparameters
        # - Hyperparameters are the secret ingredients for peak performance
        # - Fine-tuned by the brilliant RandomizedSearchCV chef
        # - Ready to conquer the text classification universe
        self.SVMFinalModel = SVC(**optimized_params)

        # Train the optimized SVM model
        # - Input: transform_variable (TF-IDF matrix of text features), y (target labels)
        # - Fit the model to the training data to learn the underlying patterns
        # - The model adjusts its internal parameters to minimize the classification error
        # - Result: A trained SVM model ready to make predictions on new text data
        self.SVMFinalModel.fit(transform_variable, y)

        return

    def predict(self, X):
        # Preprocess the input text data
        # - Transform the text data using the trained TfidfVectorizer (self.preprocessor)
        # - Concatenate the "description", "title", and "location" columns
        # - Convert the text data into a matrix of TF-IDF features
        transformed_data = self.preprocessor.transform(X["description"] + ' ' + X["title"]+' ' + X["location"])
        # Make predictions using the trained SVM model
        # - Input: transformed_data (TF-IDF matrix of text features)
        # - Use the trained SVM model (self.SVM_final_model) to predict the class labels
        # - The model applies the learned decision boundary to classify the text samples
        predictions = self.SVMFinalModel.predict(transformed_data)
        # Return the predicted class labels
        return predictions