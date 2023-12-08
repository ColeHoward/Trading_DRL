from math import floor
import numpy as np
from dataclasses import dataclass
import torch
import yfinance as yf
from utils.features import extract_features, dataframe_to_tensor
from utils.colors import RESET, GREEN, BLUE, MAGENTA


@dataclass
class TradingEnvArgs:
    ticker: str
    principal: float
    start_date: str = "2010-01-01"
    end_date: str = "2020-01-01"
    state_len: int = 17
    context_len: int = 30
    transaction_cost: float = 0  # percentage of portfolio value
    offset: int = 0  # offset from start date
    init_strategy: str = 'long'  # not sure if they start with long or short
    batch_size: int = 32


class TradingEnv:
    def __init__(self, args: TradingEnvArgs, train_data=None, test_data=None):
        self.prices = self.get_prices(args.ticker, args.start_date, args.end_date)

        if train_data is None and test_data is None:
            self.train_state_data = self.get_state_data(args.ticker, "2010-01-01", "2018-01-01", args.context_len)
            self.test_state_data = self.get_state_data(args.ticker, "2018-01-01", "2020-01-01", args.context_len)
        elif train_data is not None and test_data is not None:
            self.train_state_data = dataframe_to_tensor(train_data)
            self.test_state_data = dataframe_to_tensor(test_data)
        else:
            raise ValueError("Invalid data input")

        self.strategy = None # long or short
        self.transaction_cost = args.transaction_cost
        self.cash = args.principal
        self.principal = args.principal
        self.portfolio_values = [args.principal]
        self.context_len = args.context_len
        self.start = args.offset if args.offset else 0 + self.context_len
        self.batch_size = args.batch_size
        self.ticker = args.ticker
        self.state_len = self.train_state_data.shape[2] + 2
        self.shares_held = 0
        self.shares_value = 0
        self.epsilon = 0.1
        self.actions = [0] * self.context_len
        self.rewards = [0] * self.context_len
        self.curr_step = self.start
        self.curr_price_idx = args.context_len
        self.actionspace = ["short", "long"]
        self.is_training = True

    def get_prices(self, ticker, start_date, end_date) -> np.ndarray:
        data = yf.download(ticker, start=start_date, end=end_date)
        prices = data['Close'].values
        return prices

    def get_state_data(self, ticker, start_date, end_date, context_length):
        return extract_features(ticker, start_date, end_date, context_length)

    def reset(self) -> torch.Tensor:
        # Reset the environment to the initial state
        self.strategy = None
        self.cash = self.principal
        self.shares_held = 0
        self.portfolio_values = [self.principal]
        self.curr_step = self.start
        self.rewards = [0] * self.context_len
        self.actions = [0] * self.context_len

        # Create a tensor for initial actions
        initial_actions = torch.tensor(self.actions).unsqueeze(1)  # Convert to tensor and add batch dimension
        initial_rewards = torch.tensor(self.rewards).unsqueeze(1)
        # Get the initial state and concatenate with initial actions
        if self.is_training:
            self.curr_price_idx = 0
            initial_state = self.train_state_data[self.curr_step]
        else:
            self.curr_price_idx = len(self.train_state_data)
            initial_state = self.test_state_data[self.curr_step]

        # Concatenate initial actions with the state
        initial_state = torch.cat((initial_state, initial_actions, initial_rewards), dim=1)

        return initial_state.float()

    def step(self, action: str) -> tuple:
        current_price = self.prices[self.curr_price_idx]
        prev_portfolio_value = self.portfolio_values[-1]
        action_num = 1 if action == "long" else -1
        self.actions.append(action_num)

        ########################################## CALCULATE ACTUAL REWARD ############################################
        # Execute Long Action
        if action == "long":
            if self.strategy == "short":  # short -> long
                # close short position
                self.cash -= self.shares_held * current_price * (1 + self.transaction_cost)
                # open long position
                self.shares_held = self.cash // (current_price * (1 + self.transaction_cost))
                self.cash -= self.shares_held * current_price * (1 + self.transaction_cost)
                self.shares_value = self.shares_held * current_price
                self.strategy = "long"

            elif self.strategy is None:  # no position -> long
                self.shares_held = self.cash // (current_price * (1 + self.transaction_cost))
                self.cash -= self.shares_held * current_price * (1 + self.transaction_cost)
                self.strategy = "long"
                self.shares_value = self.shares_held * current_price

            elif self.strategy == "long":  # long -> long
                self.shares_value = self.shares_held * current_price

        # Execute Short Action
        elif action == "short":
            if self.strategy == "long":  # long -> short
                # close out long position
                self.cash += self.shares_held * current_price * (1 - self.transaction_cost)
                self.shares_held = self.cash // (current_price * (1 + self.transaction_cost))
                # open short position
                self.cash += self.shares_held * current_price * (1 + self.transaction_cost)
                self.shares_value = -self.shares_held * current_price
                self.strategy = "short"

            elif self.strategy is None:  # no position -> short
                self.shares_held = self.cash // (current_price * (1 + self.transaction_cost))
                self.cash += self.shares_held * current_price * (1 + self.transaction_cost)
                self.shares_value = -self.shares_held * current_price
                self.strategy = "short"

            elif self.strategy == "short":  # short -> short

                lower_bound = self.computeLowerBound(self.cash, -self.shares_held, current_price)
                if lower_bound > 0:
                    shares_to_buy = min(floor(lower_bound), self.shares_held)
                    self.shares_held -= shares_to_buy
                    self.cash -= shares_to_buy * current_price * (1 + self.transaction_cost)
                    self.shares_value = -self.shares_held * current_price
                else:
                    self.shares_value = -self.shares_held * current_price

        new_portfolio_value = self.cash + self.shares_value
        self.portfolio_values.append(new_portfolio_value)
        if prev_portfolio_value != 0:
            reward = (new_portfolio_value - prev_portfolio_value) / prev_portfolio_value
        else:
            reward = 0

        self.rewards.append(reward)

        if self.is_training:
            done = 1 if self.curr_step >= (len(self.train_state_data) - 1) else 0
            done = done or self.curr_price_idx >= len(self.prices) - 1
            new_state = self.train_state_data[self.curr_step] if not done else None
        else:
            done = 1 if self.curr_step >= (len(self.test_state_data) - 1) else 0
            done = done or self.curr_price_idx >= len(self.prices) - 1
            new_state = self.test_state_data[self.curr_step] if not done else None

        # Convert recent actions to a tensor and concatenate with the new state
        if not done:
            actions_tensor = torch.tensor(self.actions[-self.context_len:]).unsqueeze(1)  # Convert to tensor and add batch dimension
            rewards_tensor = torch.tensor(self.rewards[-self.context_len:]).unsqueeze(1)
            actual_new_state = torch.cat((new_state, actions_tensor, rewards_tensor), dim=1).float()
        else:
            actual_new_state = None

        ######################################### CALCULATE OPPOSITE REWARD ###########################################
        opposite_action = self.opposite_action(action)

        # Simulate opposite action, but using the original state
        original_cash = self.cash
        original_shares_held = self.shares_held
        original_strategy = self.strategy
        temp_cash = original_cash
        temp_shares_held = original_shares_held
        temp_shares_value = original_shares_held * current_price if original_shares_held != 0 else 0

        # Simulate Long Action (based on the original state)
        if opposite_action == "long":
            if original_strategy == "short":  # short -> long
                temp_cash -= temp_shares_held * current_price * (1 + self.transaction_cost)
                temp_shares_held = temp_cash // (current_price * (1 + self.transaction_cost))
                temp_cash -= temp_shares_held * current_price * (1 + self.transaction_cost)
                temp_shares_value = temp_shares_held * current_price

            elif original_strategy is None:  # no position -> long
                temp_shares_held = temp_cash // (current_price * (1 + self.transaction_cost))
                temp_cash -= temp_shares_held * current_price * (1 + self.transaction_cost)
                temp_shares_value = temp_shares_held * current_price

        # Simulate Short Action (based on the original state)
        elif opposite_action == "short":
            if original_strategy == "long":  # long -> short
                temp_cash += temp_shares_held * current_price * (1 - self.transaction_cost)
                temp_shares_held = temp_cash // (current_price * (1 + self.transaction_cost))
                temp_cash += temp_shares_held * current_price * (1 + self.transaction_cost)
                temp_shares_value = -temp_shares_held * current_price

            elif original_strategy is None:  # no position -> short
                temp_shares_held = temp_cash // (current_price * (1 + self.transaction_cost))
                temp_cash += temp_shares_held * current_price * (1 + self.transaction_cost)
                temp_shares_value = -temp_shares_held * current_price

            elif original_strategy == "short":  # short -> short
                temp_lower_bound = self.computeLowerBound(temp_cash, -temp_shares_held, current_price)
                if temp_lower_bound > 0:
                    temp_shares_to_buy = min(floor(temp_lower_bound), temp_shares_held)
                    temp_shares_held -= temp_shares_to_buy
                    temp_cash -= temp_shares_to_buy * current_price * (1 + self.transaction_cost)
                    temp_shares_value = -temp_shares_held * current_price

        # Calculate the opposite portfolio value and reward
        opposite_portfolio_value = temp_cash + temp_shares_value
        if prev_portfolio_value != 0:
            opposite_reward = (opposite_portfolio_value - prev_portfolio_value) / prev_portfolio_value
        else:
            opposite_reward = 0

        if self.is_training:
            opp_new_state = self.train_state_data[self.curr_step] if not done else None
        else:
            opp_new_state = self.test_state_data[self.curr_step] if not done else None

        self.curr_price_idx += 1
        self.curr_step += 1
        if not done:
            opp_actions_tensor = torch.tensor(self.actions[-self.context_len:-1] + [-1 * action_num]).unsqueeze(1)
            opp_rewards_tensor = torch.tensor(self.rewards[-self.context_len:-1] + [opposite_reward]).unsqueeze(1)
            new_opp_state = torch.cat((opp_new_state, opp_actions_tensor, opp_rewards_tensor), dim=1).float()
        else:
            new_opp_state = None

        opp = {"state": new_opp_state, "reward": np.clip(opposite_reward, -1, 1)}

        return actual_new_state, np.clip(reward, -1, 1), done, opp


    def computeLowerBound(self, cash, num_shares, price):
        try:
            delta_vals = - cash - num_shares * price * (1 + self.epsilon) * (1 + self.transaction_cost)
            if delta_vals < 0:
                lower_bound = delta_vals / (
                                price * (2 * self.transaction_cost + (self.epsilon * (1 + self.transaction_cost)))
                            )
            else:
                lower_bound = delta_vals / (price * self.epsilon * (1 + self.transaction_cost))

        except:
            lower_bound = 0

        return lower_bound

    @staticmethod
    def opposite_action(action: str):
        if action == "long":
            return "short"
        if action == "short":
            return "long"

    def test_mode(self):
        self.is_training = False
        self.reset()
        self.curr_step = 0
        self.curr_price_idx = len(self.train_state_data)

    def train_mode(self):
        self.is_training = True
        self.reset()
        self.curr_step = 0
        self.curr_price_idx = 0

    def write_train_metrics(self, filepath):
        assert len(self.actions) == len(self.portfolio_values), "Actions and portfolio values length mismatch"
        with open(filepath, 'w') as f:
            f.write('t,price,portfolio_value,action\n')
            f.write(f'0,{self.prices[0]},100000,long\n')
            for i in range(1, len(self.portfolio_values) ):
                f.write(f"{i-1},{self.prices[i-1]},{self.portfolio_values[i-1]},{self.actions[i]}\n")

    def write_test_metrics(self, filepath):
        assert len(self.actions) == len(self.portfolio_values), "Actions and portfolio values length mismatch"
        with open(filepath, 'w') as f:
            f.write('t,price,portfolio_value,action\n')
            f.write(f'0,{self.prices[0]},100000,long\n')
            for i in range(1, len(self.portfolio_values)):
                p_idx = i + len(self.train_state_data)
                f.write(f"{i - 1},{self.prices[p_idx]},{self.portfolio_values[i - 1]},{self.actions[i]}\n")

    def __repr__(self):
        if self.is_training:
            label = "Training"
        else:
            label = "Testing"
        return f"{label} Portfolio Value: {GREEN}{self.portfolio_values[-1]:,.2f}{RESET}" + \
               f"cash: {MAGENTA}${self.cash:,.2f}{RESET}" + \
               f"shares_held_value: {BLUE}${self.shares_value: ,.2f}{RESET}"

    def __str__(self):
        if self.is_training:
            label = "Training"
        else:
            label = "Testing"
        return f"{label} Portfolio Value: {GREEN}{self.portfolio_values[-1]:,.2f}{RESET} " + \
               f"cash: {MAGENTA}${self.cash:,.2f}{RESET} " + \
               f"shares_held_value: {BLUE}${self.shares_value: ,.2f}{RESET}"



def test_trading_env():
    # Initialize the environment with test arguments
    env_args = TradingEnvArgs(
        ticker='AAPL',
        principal=100000,
        start_date='2020-01-01',
        end_date='2021-01-10',
        state_len=12,
        context_len=30,
        transaction_cost=0.01,
        offset=0,
        init_strategy='long'
    )
    env = TradingEnv(env_args)

    # Reset the environment and get the initial state
    initial_state = env.reset()
    assert initial_state.shape[1] == env.context_len, "Initial state shape mismatch"

    # Test a sequence of actions
    actions = ['long', 'short', 'long']
    for action in actions:
        next_state, reward, done, opposite_reward = env.step(action)

        # Check the shapes and types of step outputs
        assert isinstance(next_state, torch.Tensor), "Next state should be a torch.Tensor"
        assert isinstance(reward, float), "Reward should be a float"
        assert isinstance(done, bool), "Done should be a boolean"
        assert isinstance(opposite_reward, float), "Opposite reward should be a float"

        # Check the shape of the next state
        assert next_state.shape[1] == env.context_len, "Next state shape mismatch"

    print("PASSED: TradingEnv behaves as expected.")


if __name__ == "__main__":
    test_trading_env()



