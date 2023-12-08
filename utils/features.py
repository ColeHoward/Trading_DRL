import numpy as np
import pandas as pd
import yfinance as yf
import torch



def dataframe_to_tensor(df, context_length=30):
    num_features = len(df.columns)
    tensor = torch.zeros((len(df), context_length, num_features))
    for i in range(context_length, len(df)):
        temp_tensor = torch.tensor(df.iloc[i - context_length:i].values)
        tensor[i] = temp_tensor

    return tensor


def extract_original_features(symbol, start_date, end_date, state_length) -> pd.DataFrame:
    df = yf.download(symbol, start=start_date, end=end_date)
    # df = preprocess_features(df)
    df.interpolate(method='linear', inplace=True)
    df['Open'] = df['Open'].diff()
    df['High'] = df['High'].diff()
    df['Low'] = df['Low'].diff()
    df['Close'] = df['Close'].diff()

    df.drop(columns=['Adj Close'], inplace=True)
    df.replace(0.0, np.nan, inplace=True)
    df.interpolate(method='linear', limit=5, limit_area='inside', inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.fillna(0, inplace=True)

    # normalize each column using min and max
    df = (df - df.min()) / (df.max() - df.min())
    return df


def preprocess_features(df) -> pd.DataFrame:
    df['Open'] = df['Open'].diff()
    df['High'] = df['High'].diff()
    df['Low'] = df['Low'].diff()
    df['Close'] = df['Close'].diff()

    df.drop(columns=['Adj Close'], inplace=True)
    df.replace(0.0, np.nan, inplace=True)
    df.interpolate(method='linear', limit=5, limit_area='inside', inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.fillna(0, inplace=True)

    # # normalize with min-max
    df['Open'] = (df['Open'] - df['Open'].min()) / (df['Open'].max() - df['Open'].min())
    df['High'] = (df['High'] - df['High'].min()) / (df['High'].max() - df['High'].min())
    df['Low'] = (df['Low'] - df['Low'].min()) / (df['Low'].max() - df['Low'].min())
    df['Close'] = (df['Close'] - df['Close'].min()) / (df['Close'].max() - df['Close'].min())
    df['Volume'] = (df['Volume'] - df['Volume'].min()) / (df['Volume'].max() - df['Volume'].min())

    return df


def extract_features(symbol, start_date, end_date, state_length) -> torch.Tensor:
    df = yf.download(symbol, start=start_date, end=end_date)
    df.interpolate(method='linear', inplace=True)

    # Setting the smoothing factor (alpha) and window size
    N = 14  # adjust value of N (window size) as needed
    alpha = 2 / (N + 1)

    df['EMA'] = calculate_ema(df['Close'], alpha)
    df['MFI'] = calculate_mfi(df)
    df['RSI'] = calculate_rsi(df)
    df['MACD'], df['Signal'] = calculate_macd(df)
    df['CCI'] = calculate_cci(df)
    df[['Tenkan_Sen', 'Kijun_Sen', 'Senkou_Span_A', 'Senkou_Span_B', 'Chikou_Span']] = calculate_ichimoku(df)

    df = preprocess_features(df)

    return dataframe_to_tensor(df)


def extract_feats(df):
    # Setting the smoothing factor (alpha) and window size
    N = 14  # adjust value of N (window size) as needed
    alpha = 2 / (N + 1)

    df['EMA'] = calculate_ema(df['Close'], alpha)
    df['MFI'] = calculate_mfi(df)
    df['RSI'] = calculate_rsi(df)
    df['MACD'], df['Signal'] = calculate_macd(df)
    df['CCI'] = calculate_cci(df)
    df[['Tenkan_Sen', 'Kijun_Sen', 'Senkou_Span_A', 'Senkou_Span_B', 'Chikou_Span']] = calculate_ichimoku(df)

    df = preprocess_features(df)
    return df



def calculate_ema(data, alpha):
    # Exponential Moving Average
    ema_values = [data.iloc[0]]

    for i in range(1, len(data)):
        ema = alpha * data.iloc[i] + (1 - alpha) * ema_values[-1]
        ema_values.append(ema)

    return pd.Series(ema_values, index=data.index)


def calculate_mfi(data, period=14):
    # money flow index
    money_flow = ((data['High'] + data['Low'] + data['Close']) / 3) * data['Volume']
    positive_flow = (money_flow * (data['Close'] > data['Close'].shift(1))).rolling(window=period, min_periods=1).sum()
    negative_flow = (money_flow * (data['Close'] < data['Close'].shift(1))).rolling(window=period, min_periods=1).sum()

    money_flow_ratio = positive_flow / negative_flow
    money_flow_index = 100 - (100 / (1 + money_flow_ratio))

    return money_flow_index


def calculate_rsi(data, window=14):
    # Relative Strength Index (RSI)
    # Calculating daily price changes
    delta = data['Close'].diff(1)

    # Separate gains (positive changes) and losses (negative changes)
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)

    # Calculating average gains and losses over the specified window
    avg_gains = gains.rolling(window=window, min_periods=1).mean()
    avg_losses = losses.rolling(window=window, min_periods=1).mean()

    # Relative Strength (RS)
    rs = avg_gains / avg_losses

    # RSI
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_macd(data, theta1=26, theta2=12, macd_period=9):
    # Moving Average Convergence Divergence (MACD)
    # Calculating EMAs
    ema_theta1 = data['Close'].ewm(span=theta1, adjust=False).mean()
    ema_theta2 = data['Close'].ewm(span=theta2, adjust=False).mean()

    # Calculating MACD line
    macd_line = ema_theta1 - ema_theta2

    # Calculating Signal line (Moving Average of MACD line)
    macd = macd_line.rolling(window=macd_period, min_periods=1).mean()

    return macd_line, macd



def calculate_cci(df, window=14):
    # Commodity Channel Index (CCI)
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    sma = typical_price.rolling(window=window).mean()
    mad = typical_price.rolling(window=window).apply(lambda x: abs(x - x.mean()).mean(), raw=True)
    cci = (typical_price - sma) / (0.015 * mad)

    return cci



def calculate_ichimoku(data):
    # Ichimoku Cloud
    # Tenkan-Sen (Conversion Line) - Calculate the average of the high and low prices for the last 9 periods
    nine_period_high = data['High'].rolling(window=9).max()
    nine_period_low = data['Low'].rolling(window=9).min()
    data['Tenkan_Sen'] = (nine_period_high + nine_period_low) / 2

    # Kijun-Sen (Base Line) - Calculate the average of the high and low prices for the last 26 periods
    twenty_six_period_high = data['High'].rolling(window=26).max()
    twenty_six_period_low = data['Low'].rolling(window=26).min()
    data['Kijun_Sen'] = (twenty_six_period_high + twenty_six_period_low) / 2

    # Calculate Senkou Span A (Leading Span A)
    data['Senkou_Span_A'] = ((data['Tenkan_Sen'] + data['Kijun_Sen']) / 2).shift(26)

    # Senkou Span B (Leading Span B) - Calculate the average of the high and low prices for the last 52 periods
    fifty_two_period_high = data['High'].rolling(window=52).max()
    fifty_two_period_low = data['Low'].rolling(window=52).min()
    data['Senkou_Span_B'] = ((fifty_two_period_high + fifty_two_period_low) / 2).shift(26)

    # Calculate Chikou Span (Lagging Span)
    data['Chikou_Span'] = data['Close'].shift(-26)

    return data[['Tenkan_Sen', 'Kijun_Sen', 'Senkou_Span_A', 'Senkou_Span_B', 'Chikou_Span']]



