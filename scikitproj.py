# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load dataset (example: housing dataset)
# Replace this with your own dataset
# Assuming the dataset has features (X) and target variable (y)
dataset = pd.read_csv('housing_dataset.csv')

# Splitting the dataset into features (X) and target variable (y)
X = dataset.drop('target_variable_name', axis=1)  # Specify the name of the target variable
y = dataset['target_variable_name']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model (example: Linear Regression)
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Visualize the predictions
plt.scatter(y_test, y_pred)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("True Values vs Predictions")
plt.show()
