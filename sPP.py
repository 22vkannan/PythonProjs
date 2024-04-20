# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

#loading historical stock price data
data = pd.read_csv('stock_data.csv')

#preprocessing data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

#feature engineering for stock price prediction
def create_features(data):
    #compute moving averages
    data['MA_5'] = data['Close'].rolling(window=5).mean()
    data['MA_10'] = data['Close'].rolling(window=10).mean()

    #moving averages
    data['EMA_5'] = data['Close'].ewm(span=5, adjust=False).mean()
    data['EMA_10'] = data['Close'].ewm(span=10, adjust=False).mean()

    #bollinger bands computation
    data['Rolling_Std'] = data['Close'].rolling(window=20).std()
    data['Upper_Band'] = data['MA_10'] + 2 * data['Rolling_Std']
    data['Lower_Band'] = data['MA_10'] - 2 * data['Rolling_Std']

    #RSI computation
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    RS = gain / loss
    data['RSI'] = 100 - (100 / (1 + RS))

    # Remove NaN values resulting from moving averages and exponential moving averages calculations
    data.dropna(inplace=True)

    return data


#applying feature engineering to stock data
data = create_features(data)

#updated data
print(data.head())


#making LSTM sequences
def create_sequences(data, sequence_length):
    X = []
    y = []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(X), np.array(y)

sequence_length = 10  #defining sequence length
X, y = create_sequences(scaled_data, sequence_length)

#training and testin gsets
train_size = int(len(data) * 0.8)
test_size = len(data) - train_size
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

#building LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=1))

#compiling model
model.compile(optimizer='adam', loss='mean_squared_error')

#training model
model.fit(X_train, y_train, epochs=10, batch_size=32)

#predicting future prices for stock
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)  # Inverse scaling

#evaluations of ytrue, ypred
y_true = np.array([100, 110, 105, 120, 125])  # True stock prices
y_pred = np.array([105, 112, 108, 118, 123])  # Predicted stock prices

#calculating MAE
mae = mean_absolute_error(y_true, y_pred)

#calculating MSE
mse = mean_squared_error(y_true, y_pred)

#calculating RMSE
rmse = np.sqrt(mse)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)


#printing predictinos
print("Predictions:", predictions)
