from models.TDTransformer import TransformerDQN, TDTransformerArgs
from utils.TradingEnv import TradingEnv, TradingEnvArgs
import torch
from utils.colors import RESET, RED, YELLOW, CYAN


@torch.no_grad()
def evaluate(env, model, tracker, epoch, is_test=True, device="cpu", true_env=None):
    """Evaluate a RL agent on a given environment."""
    action_lookup = {0: 'short', 1: 'long'}
    model.test_mode()
    if is_test:
        env.test_mode()
        if true_env is not None:
            true_env.test_mode()
    state = env.reset()
    done = False
    action_values_count = {0: 0, 1: 0}
    tot_reward = 0
    count = 0
    while not done:
        count += 1
        action_values = model.main_pred(state.unsqueeze(0).to(device))
        action = action_values.argmax().item()
        action_values_count[action] += 1
        next_state, reward, done, _ = env.step(action_lookup[action])
        if true_env is not None:
            true_env.step(action_lookup[action])
        state = next_state
        if done:
            break

        next_reward = model.tgt_pred(next_state.unsqueeze(0).to(device)).max().item()
        tot_reward += reward + .4 * next_reward
        if true_env is None:
            if is_test:
                tracker.push(env.prices[env.curr_price_idx], env.portfolio_values[-1], env.cash, env.shares_value, action, next_reward, epoch, False)
            else:
                tracker.push(env.prices[env.curr_price_idx], env.portfolio_values[-1], env.cash, env.shares_value, action, next_reward, epoch, True)
        else:
            if is_test:
                tracker.push(true_env.prices[true_env.curr_price_idx], true_env.portfolio_values[-1], true_env.cash, true_env.shares_value, action, next_reward, epoch, False)
            else:
                tracker.push(true_env.prices[true_env.curr_price_idx], true_env.portfolio_values[-1], true_env.cash, true_env.shares_value, action, next_reward, epoch, True)
    if is_test:
        print('test action counts: ', YELLOW, action_values_count, 'test reward: ', CYAN, tot_reward / count, RESET)
        print(true_env if true_env is not None else env)
    else:
        print('train action counts: ', RED, action_values_count, 'train reward: ', CYAN, tot_reward / count, RESET)
        print(true_env if true_env is not None else env)

    model.train_mode()
    env.train_mode()
    if true_env is not None:
        true_env.train_mode()


if __name__ == "__main__":
    env_args = TradingEnvArgs(
        ticker='PG',
        principal=100000,
        start_date='2012-01-01',
        end_date='2019-12-31',
        context_len=30,
        transaction_cost=0.0001,
        offset=0,
        init_strategy='long'
    )

    env = TradingEnv(env_args)
    env.test_mode()

    model_args = TDTransformerArgs(
        dim=512,
        n_layers=2,
        head_dim=64,
        hidden_dim=512,
        n_heads=8,
        n_kv_heads=4,
        context_length=env.context_len,
        num_features=env.state_len,
        norm_eps=1e-6,
        action_dim=2,
        device="cpu"
    )

    tdqn = TransformerDQN(model_args)
    tdqn.load_models('models/tdqn')



