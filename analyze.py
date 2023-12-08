import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys

STOCKS = [
    "AAPL",
    "AMZN",
    "BABA",
    "BIDU",
    "CCB",
    "DIA",
    "EWJ",
    "HSBC",
    "JPM",
    "KO",
    "NOK",
    "PG",
    "QQQ",
    "SIE",
    "SONY",
    "SPY",
    "TOY",
    "TSLA",
    "XOM"
]

def aggregate(tgt_stocks=None):
    if tgt_stocks is None:
        tgt_stocks = STOCKS
    sharpes, rois, sortinos, max_drawdowns, max_drawdown_durations, annualized_volatilities, profits = [], [], [], [], [], [], []
    for stock in tgt_stocks:
        filename = f"{stock}_test.csv"
        df = pd.read_csv(filename)

        # calculate sharpe ratio
        last_epoch = df[df['epoch'] == 29]
        last_episode = last_epoch.iloc[len(last_epoch) // 2:]
        # Calculate the daily returns of the portfolio
        daily_returns = last_episode['portfolio_value'].pct_change().dropna()

        # Calculate the average daily return and the standard deviation of daily returns
        average_daily_return = daily_returns.mean()
        std_dev_daily_returns = daily_returns.std()

        negative_returns = daily_returns[daily_returns < 0]
        downside_deviation = negative_returns.std()
        risk_free_rate = 0

        # Calculate ratios
        if average_daily_return != 0 and downside_deviation != 0:
            sharpe_ratio = np.sqrt(252) * (average_daily_return - risk_free_rate) / std_dev_daily_returns
            sortino_ratio = np.sqrt(252) * (average_daily_return - risk_free_rate) / downside_deviation
        else:
            sharpe_ratio = sortino_ratio = 0

        # Calculate Maximum Drawdown
        capital = last_episode['portfolio_value'].values
        trough = np.argmax(np.maximum.accumulate(capital) - capital)

        if trough != 0:
            peak = np.argmax(capital[:trough])
            max_drawdown = round(100 * (capital[peak] - capital[trough]) / capital[peak], 3)
            max_drawdown_duration = trough - peak
        else:
            max_drawdown = max_drawdown_duration = 0

        annualized_volatility = round(np.sqrt(252) * std_dev_daily_returns, 5)
        start, end = last_episode['portfolio_value'].iloc[0], last_episode['portfolio_value'].iloc[-1]
        profit = round((end - start) / start, 5)

        sharpes.append(sharpe_ratio)
        rois.append(profit)
        sortinos.append(sortino_ratio)
        max_drawdowns.append(max_drawdown)
        max_drawdown_durations.append(max_drawdown_duration)
        annualized_volatilities.append(annualized_volatility)
        profits.append(profit)

        print(stock, "&", round(sharpe_ratio, 5), "&", round(sortino_ratio, 5), "&", abs(max_drawdown), '&', max_drawdown_duration, '&', annualized_volatility, '&', profit, "\\\\")

    print('Average Sharpe Ratio:', np.mean(sharpes))
    print('Average ROI:', np.mean(rois))
    print('Average Sortino Ratio:', np.mean(sortinos))
    print('Average Max Drawdown:', np.mean(max_drawdowns))
    print('Average Max Drawdown Duration:', np.mean(max_drawdown_durations))
    print('Average Annualized Volatility:', np.mean(annualized_volatilities))
    print('Average Profit:', np.mean(profits))


def plot_price_and_pv(input_file, output_file, ticker):
    df = pd.read_csv(input_file)

    long_legend = mpatches.Patch(color='green', label='Long')
    short_legend = mpatches.Patch(color='red', label='Short')
    portfolio_df = df[df['epoch'] == 21]
    portfolio_df = portfolio_df.iloc[len(portfolio_df) // 2:]

    date_range = pd.date_range(start='2018-01-01', end='2020-01-01', periods=len(portfolio_df))
    portfolio_df['Date'] = date_range

    portfolio_df.set_index('Date', inplace=True)

    portfolio_df['previous_action'] = portfolio_df['position'].shift(1)
    changed_actions_df = portfolio_df[portfolio_df['position'] != portfolio_df['previous_action']]


    plt.figure(figsize=(12, 12))

    # Top plot (Portfolio Value)
    plt.subplot(2, 1, 1)
    plt.plot(portfolio_df.index, portfolio_df['portfolio_value'], label='Portfolio Value', color='blue')
    for date, row in changed_actions_df.iterrows():
        if row['position'] == 1:
            plt.scatter(date, row['portfolio_value'], color='green', marker='^')
        elif row['position'] == 0:
            plt.scatter(date, row['portfolio_value'], color='red', marker='v')

    plt.title('Portfolio Value Over Time with Changed Trading Actions')
    plt.ylabel('Portfolio Value')
    plt.legend(handles=[long_legend, short_legend])
    plt.grid(True)  # Remove x-axis labels

    # Bottom plot (Price)
    plt.subplot(2, 1, 2)
    plt.plot(portfolio_df.index, portfolio_df['price'], label='Price', color='blue')
    for date, row in changed_actions_df.iterrows():
        if row['position'] == 1:
            plt.scatter(date, row['price'], color='green', marker='^')
        elif row['position'] == 0:
            plt.scatter(date, row['price'], color='red', marker='v')

    plt.title(f'{ticker} Price Over Time with Changed Trading Actions')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(handles=[long_legend, short_legend])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()



def plot_rewards_vs_returns(ticker, output_file):
    df = pd.read_csv(f"{ticker}_test.csv")
    portfolio_df = df[df['epoch'] == 29]
    portfolio_df = portfolio_df.iloc[len(portfolio_df) // 2:]
    date_range = pd.date_range(start='2018-01-01', end='2020-01-01', periods=len(portfolio_df))
    portfolio_df['Date'] = date_range

    portfolio_df.set_index('Date', inplace=True)

    portfolio_df['previous_action'] = portfolio_df['position'].shift(1)
    changed_actions_df = portfolio_df[portfolio_df['position'] != portfolio_df['previous_action']]

    portfolio_df['portfolio_value_pct_change'] = portfolio_df['portfolio_value'].pct_change()

    plt.figure(figsize=(12, 12))

    # Reward Plot
    plt.subplot(2, 1, 1)
    plt.plot(portfolio_df.index, portfolio_df['reward'], label='Reward', color='orange')
    for date, row in changed_actions_df.iterrows():
        if row['position'] == 1:
            plt.scatter(date, row['reward'], color='green', marker='^')
        elif row['position'] == 0:
            plt.scatter(date, row['reward'], color='red', marker='v')

    plt.title('Rewards Over Time with Trading Actions')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)

    # Percentage Change in Portfolio Value Plot
    plt.subplot(2, 1, 2)
    plt.plot(portfolio_df.index, portfolio_df['portfolio_value_pct_change'],
             label='Percentage Change in Portfolio Value', color='blue')
    plt.title('Percentage Change in Portfolio Value Over Time')
    plt.xlabel('Date')
    plt.ylabel('Percentage Change')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()


def calculate_sharpe_ratios(df):
    sharpe_ratios = []
    for epoch in df['epoch'].unique():
        epoch_data = df[df['epoch'] == epoch]
        half_index = len(epoch_data) // 2
        # Calculating Sharpe ratio for each episode
        # Sharpe Ratio = (Mean Return - Risk Free Rate) / Standard Deviation of Return
        # Assuming Risk Free Rate as 0 (as in base paper)
        episode_1_returns = epoch_data.iloc[:half_index]['portfolio_value'].pct_change().dropna()
        episode_2_returns = epoch_data.iloc[half_index:]['portfolio_value'].pct_change().dropna()

        sharpe_ratio_1 = episode_1_returns.mean() / episode_1_returns.std() if episode_1_returns.std() != 0 else 0
        sharpe_ratio_2 = episode_2_returns.mean() / episode_2_returns.std() if episode_2_returns.std() != 0 else 0

        sharpe_ratios.append(np.sqrt(255) * sharpe_ratio_1)
        sharpe_ratios.append(np.sqrt(255) * sharpe_ratio_2)

    return sharpe_ratios

def extract_sharpe_ratio_ranges(df):
    ranges = []
    for epoch in df['epoch'].unique():
        epoch_data = df[df['epoch'] == epoch]
        half_index = len(epoch_data) // 2
        episode_1_std = epoch_data.iloc[:half_index]['portfolio_value'].pct_change().dropna().std()
        episode_2_std = epoch_data.iloc[half_index:]['portfolio_value'].pct_change().dropna().std()
        ranges.append(episode_1_std)
        ranges.append(episode_2_std)
    return ranges


def plot_train_test_sharpe(train_file, test_file, output_file):
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    train_sharpe_ratios = calculate_sharpe_ratios(train_df)
    test_sharpe_ratios = calculate_sharpe_ratios(test_df)

    train_sharpe_ranges = extract_sharpe_ratio_ranges(train_df)
    test_sharpe_ranges = extract_sharpe_ratio_ranges(test_df)

    plt.figure(figsize=(14, 7))

    plt.plot(range(1, 61), train_sharpe_ratios, label='Train Sharpe Ratio', color='blue', marker='o')
    plt.fill_between(range(1, 61),
                     np.array(train_sharpe_ratios) - np.array(train_sharpe_ranges),
                     np.array(train_sharpe_ratios) + np.array(train_sharpe_ranges),
                     color='blue', alpha=0.2)

    plt.plot(range(1, 61), test_sharpe_ratios, label='Test Sharpe Ratio', color='orange', marker='x')
    plt.fill_between(range(1, 61),
                     np.array(test_sharpe_ratios) - np.array(test_sharpe_ranges),
                     np.array(test_sharpe_ratios) + np.array(test_sharpe_ranges),
                     color='orange', alpha=0.2)

    plt.xlabel('Episode')
    plt.ylabel('Sharpe Ratio')
    plt.title('Sharpe Ratio per Episode (Train and Test Sets)')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)
    plt.show()



if __name__ == "__main__":
    tickers = []
    input_file = ""
    output_file = ""
    train_file, test_file = "", ""
    plot_train_test_sharpe(train_file, test_file, "sharpe_ticker.png")
    plot_rewards_vs_returns("PG", "rewards_returns.png")
    plot_price_and_pv("PG_test.csv", "price_pv.png", "PG")
    aggregate(tickers)