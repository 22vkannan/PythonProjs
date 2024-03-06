import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# loading data from CSV file
data = pd.read_csv(r"C:\Users\vishr\Downloads\realtor-data.zip.csv")

# delete rows with empty cells in 'G' (zip codes) and 'H' (house sizes)
data.dropna(subset=['zip_code', 'house_size'], inplace=True)

# get house sizes from column 'H' and zip codes from column 'G'
house_sizes = data['house_size'].values
zip_codes = data['zip_code'].values

# label encoder object for converting zip codes to numeric values
label_encoder = LabelEncoder()

# fit label encoder to zip codes to transform them into numeric values
numeric_zip_codes = label_encoder.fit_transform(zip_codes)

# replace original zip codes with numeric representations in the data frame
data['zip_code'] = numeric_zip_codes

# extract house sizes and house prices from the data
house_sizes = data['house_size'].values
house_prices = data['price'].values  # Replace 'price_column_name' with the actual name of the column containing house prices

# reshape house sizes to become a column vector
house_sizes = house_sizes.reshape(-1, 1)

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(house_sizes, house_prices, test_size=0.2, random_state=42)
# 20% of data for testing and rest for training
# random state makes sure data splitting is reproducible

# create and train the linear regression model
model = LinearRegression()  # create a linear regression model object
model.fit(X_train, y_train)  # train the model using training data

# make predictions on the testing set
y_pred = model.predict(X_test)  # make predictions on the test data

# calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
# mse calculates the difference between predicted and actual values
print("The mean squared error of the model is:", mse)

# plot the actual vs. predicted prices
plt.scatter(X_test, y_test, color='black')  # plot actual prices
plt.plot(X_test, y_pred, color='blue', linewidth=3)  # plot predicted prices
plt.xlabel('House Size')  # X axis label
plt.ylabel('House Price')  # y axis label
plt.title('House Price Prediction')  # plot title
plt.show()  # show plot
