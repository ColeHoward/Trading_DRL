import numpy as np
import yfinance as yf
from utils.TradingEnv import TradingEnv, TradingEnvArgs
import pandas as pd
from utils.features import extract_feats


class EnvGenerator:
    def __init__(self, env_args, **kwargs):
        self.start_date = env_args.start_date
        self.end_date = env_args.end_date
        self.env_args = env_args
        self.signal_lengths = kwargs.get("signal_lengths", [5])
        self.noise_stdevs = kwargs.get("noise_stdevs", [0])
        self.stretch_factors = kwargs.get("stretch_factors", [1])
        self.shifts = kwargs.get("shifts", [0])
        self.train_len = len(yf.download(env_args.ticker, start="2010-01-01", end="2018-01-01", interval="1d"))
        self.original_data = yf.download(env_args.ticker, start="2010-01-01", end="2020-01-01", interval="1d")

        self.index = 0
        self.trading_envs = []

    def signal_filter(self, old_data, length=10) -> pd.DataFrame:
        data = old_data.copy()
        data['Close'] = data['Close'].rolling(window=length).mean()
        data['Low'] = data['Low'].rolling(window=length).mean()
        data['High'] = data['High'].rolling(window=length).mean()
        data['Volume'] = data['Volume'].rolling(window=length).mean()
        for i in range(length):
            data['Close'].iloc[i] = old_data['Close'].iloc[i]
            data['Low'].iloc[i] = old_data['Low'].iloc[i]
            data['High'].iloc[i] = old_data['High'].iloc[i]
            data['Volume'].iloc[i] = old_data['Volume'].iloc[i]
        data['Open'] = data['Close'].shift(1)
        data['Open'].iloc[0] = old_data['Open'].iloc[0]
        return data

    def noise_addition(self, data, stdev) -> pd.DataFrame:
        # add gaussian noise to the time series
        augmented_data = data.copy()
        price = augmented_data['Close']
        volume = augmented_data['Volume']
        priceNoise = np.random.normal(0, stdev * (price / 100))
        volumeNoise = np.random.normal(0, stdev * (volume / 100))

        augmented_data['Close'] *= (1 + priceNoise / 100)
        augmented_data['Low'] *= (1 + priceNoise / 100)
        augmented_data['High'] *= (1 + priceNoise / 100)
        augmented_data['Adj Close'] *= (1 + priceNoise / 100)
        augmented_data['Volume'] *= (1 + volumeNoise / 100)
        augmented_data['Open'] = augmented_data['Close'].shift(1)

        return augmented_data

    def stretch(self, data, factor=1) -> pd.DataFrame:
        new_data = data.copy()

        returns = new_data['Close'].pct_change().ffill().bfill() * factor
        multiplier = (1 + returns).cumprod()

        # Apply the stretching operation to 'Close'
        new_data['Close'] = new_data['Close'].iloc[0] * multiplier

        # Adjust 'Low', 'High', and 'Open' based on the new 'Close' values
        new_data['Low'] = new_data['Close'] * new_data['Low'] / new_data['Close'].shift(1)
        new_data['High'] = new_data['Close'] * new_data['High'] / new_data['Close'].shift(1)
        new_data['Open'] = new_data['Close'].shift(1)

        # Handle the first row separately
        new_data['Low'].iloc[0] = data['Low'].iloc[0]
        new_data['High'].iloc[0] = data['High'].iloc[0]
        new_data['Open'].iloc[0] = data['Open'].iloc[0]

        return new_data

    def shift(self, data, shift=0) -> pd.DataFrame:
        new_data = data.copy()
        return new_data.iloc[shift:]

    def generate(self, data=None):
        if data is None:
            data = self.original_data
        data['Close'] = data['Adj Close']
        for shift in self.shifts:
            shifted = self.shift(data, shift)
            for length in self.signal_lengths:
                signal_filtered = self.signal_filter(shifted, length)
                for stdev in self.noise_stdevs:
                    noised = self.noise_addition(signal_filtered, stdev)
                    for factor in self.stretch_factors:
                        full_data = extract_feats(self.stretch(noised, factor))
                        train = full_data.iloc[:self.train_len]
                        test = full_data.iloc[self.train_len:]
                        self.trading_envs.append(TradingEnv(self.env_args, train, test))

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= len(self.trading_envs):
            raise StopIteration
        else:
            self.index += 1
            return self.trading_envs[self.index - 1]

    def __getitem__(self, index):
        return self.trading_envs[index]

    def __len__(self):
        return len(self.trading_envs)


def test():
    env_args = TradingEnvArgs(
        ticker='AAPL',
        principal=100_000,
        start_date='2010-01-01',
        end_date='2020-10-31',
        context_len=30,
        transaction_cost=0.0001,
        offset=0,
        init_strategy='long'
    )
    augmentation_args = dict(
        signal_lengths=[5, 10],
        noise_stdevs=[0.5, 1],
        stretch_factors=[1]
    )
    env_generator = EnvGenerator(env_args, **augmentation_args)
    env_generator.generate(env_generator.original_data)


if __name__ == "__main__":
    test()