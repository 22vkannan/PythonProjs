import numpy as np
import pandas as pd


#loading stock data
def load_data(C:\Users\vishr\Downloads\'hTB_.sd'):
    return pd.read_csv(C:\Users\vishr\Downloads\'hTB_.sd')


#moving average crossover strategy
def moving_average_crossover(data, short_window=50, long_window=200):
    signals = pd.DataFrame(index=data.index)
    signals['Signal'] = 0.0

    #computing short and long moving averages
    signals['Short_MA'] = data['Close'].rolling(window=short_window, min_periods=1).mean()
    signals['Long_MA'] = data['Close'].rolling(window=long_window, min_periods=1).mean()

    #generating buy signals
    signals.loc[signals['Short_MA'] > signals['Long_MA'], 'Signal'] = 1.0

    #generating sell signals
    signals.loc[signals['Short_MA'] < signals['Long_MA'], 'Signal'] = -1.0

    return signals


#RSI signal strategy
def rsi_signal(data, window=14):
    signals = pd.DataFrame(index=data.index)
    signals['Signal'] = 0.0

    #price differential calculator
    delta = data['Close'].diff()

    #gain and loss computations
    gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()

    #RSI calcualtions
    RS = gain / loss
    RSI = 100 - (100 / (1 + RS))

    #generating buy signals
    signals.loc[RSI < 30, 'Signal'] = 1.0

    #generating sell signals
    signals.loc[RSI > 70, 'Signal'] = -1.0

    return signals


#executing buy and sell orders
def execute_orders(data, signals):
    positions = pd.DataFrame(index=data.index).fillna(0.0)
    positions['Position'] = 0

    #buy signals
    positions.loc[signals['Signal'] == 1.0, 'Position'] = 1

    #sell signals
    positions.loc[signals['Signal'] == -1.0, 'Position'] = -1

    return positions


def main():
    file_path = 'stock_data.csv'
    data = load_data(file_path)

    #moving crossover strategy
    ma_signals = moving_average_crossover(data)

    #RSI signal strategy
    rsi_signals = rsi_signal(data)

    #combining signals and executing orders
    signals = ma_signals + rsi_signals
    positions = execute_orders(data, signals)

    print(positions)


if __name__ == "__main__":
    main()
