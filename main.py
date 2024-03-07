import numpy as np
import sys
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.utils import to_categorical

# Generate random text data
np.random.seed(42)  # Seed for reproducibility
num_chars = 10000  # Number of characters in the random text
chars = ['a', 'b', 'c', 'd', 'e']  # Possible characters in the text
text = ''.join(np.random.choice(chars, num_chars))  # Generate random text

# Preprocess text data
chars = sorted(list(set(text)))  # Get unique characters and sort them
char_to_int = dict((c, i) for i, c in enumerate(chars))  # Map characters to integers
int_to_char = dict((i, c) for i, c in enumerate(chars))  # Map integers to characters

# Create input and output sequences for training
seq_length = 100  # Length of input sequences
dataX = []  # List to store input sequences
dataY = []  # List to store output sequences

# Iterate through the text to create input/output sequences
for i in range(0, num_chars - seq_length, 1):
    seq_in = text[i:i + seq_length]  # Input sequence of characters
    seq_out = text[i + seq_length]  # Output character
    dataX.append([char_to_int[char] for char in seq_in])  # Map input characters to integers
    dataY.append(char_to_int[seq_out])  # Map output character to integer

# Reshape input sequences to be [samples, time steps, features]
X = np.reshape(dataX, (len(dataX), seq_length, 1))
# Normalize input data
X = X / float(len(chars))
# One-hot encode the output variable
y = to_categorical(dataY)

# Define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(y.shape[1], activation='softmax'))  # Output layer with softmax activation
model.compile(loss='categorical_crossentropy', optimizer='adam')  # Compile the model

# Generate text
start = np.random.randint(0, len(dataX)-1)  # Randomly select a starting point
pattern = dataX[start]  # Get the input sequence from the starting point
print("Seed:")
print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")  # Print the initial input sequence

# Generate characters
for i in range(1000):
    x = np.reshape(pattern, (1, len(pattern), 1))  # Reshape the input for the model
    x = x / float(len(chars))  # Normalize input data
    prediction = model.predict(x, verbose=0)  # Predict the next character
    index = np.argmax(prediction)  # Get the index of the predicted character
    result = int_to_char[index]  # Get the predicted character
    sys.stdout.write(result)  # Print the predicted character
    pattern.append(index)  # Add the predicted character to the input sequence
    pattern = pattern[1:len(pattern)]  # Update the input sequence by removing the first character
print("\nDone.")