# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import classification_report, mean_squared_error, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# Load dataset
data = pd.read_csv('your_data.csv')

# Univariate Analysis
def univariate_analysis(data):
    print("Univariate Analysis:")
    for column in data.columns:
        plt.figure(figsize=(6,4))
        sns.histplot(data[column], kde=True)
        plt.title(f"Distribution of {column}")
        plt.show()

# Bivariate Analysis (Pair plot for classification/regression)
def bivariate_analysis(data, target_column):
    print("Bivariate Analysis:")
    sns.pairplot(data, hue=target_column)
    plt.show()

# Splitting data into features and target
X = data.drop(columns=['target_class', 'target_regression'])  # Replace with your feature columns
y_class = data['target_class']  # Classification target
y_regression = data['target_regression']  # Regression target

# Split for Classification
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.2, random_state=42)

# Split for Regression
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_regression, test_size=0.2, random_state=42)

# KNN Classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train_class, y_train_class)
y_pred_class = knn_classifier.predict(X_test_class)

# Classification Metrics
print("Classification Accuracy:", accuracy_score(y_test_class, y_pred_class))
print("Classification Report:")
print(classification_report(y_test_class, y_pred_class))

# Save the trained KNN classifier model
with open('knn_classifier.pkl', 'wb') as file:
    pickle.dump(knn_classifier, file)

# Load the saved KNN classifier model and predict
with open('knn_classifier.pkl', 'rb') as file:
    loaded_classifier = pickle.load(file)

# Example of passing new input data to the classifier
new_input_class = np.array([[1.5, 2.5, 3.0]])  # Replace with your new input data
class_prediction = loaded_classifier.predict(new_input_class)
print("Classification Prediction for new input:", class_prediction)

# KNN Regressor
knn_regressor = KNeighborsRegressor(n_neighbors=5)
knn_regressor.fit(X_train_reg, y_train_reg)
y_pred_reg = knn_regressor.predict(X_test_reg)

# Regression Metrics
print("Mean Squared Error for Regression:", mean_squared_error(y_test_reg, y_pred_reg))

# Save the trained KNN regressor model
with open('knn_regressor.pkl', 'wb') as file:
    pickle.dump(knn_regressor, file)

# Load the saved KNN regressor model and predict
with open('knn_regressor.pkl', 'rb') as file:
    loaded_regressor = pickle.load(file)

# Example of passing new input data to the regressor
new_input_reg = np.array([[1.5, 2.5, 3.0]])  # Replace with your new input data
regression_prediction = loaded_regressor.predict(new_input_reg)
print("Regression Prediction for new input:", regression_prediction)

# Call the analysis functions
univariate_analysis(data)
bivariate_analysis(data, 'target_class')  # Replace with your actual target column name for classification
