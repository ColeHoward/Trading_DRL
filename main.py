import sys
import torch
from utils.EnvGenerator import EnvGenerator
from utils.ExperienceReplay import ExperienceReplay
from utils.TradingEnv import TradingEnvArgs, TradingEnv
from utils.metric_tracker import MetricTracker
from models.TDTransformer import TDTransformerArgs, TransformerDQN
from train import train


def main(ticker):
    env_args = TradingEnvArgs(
        ticker=ticker,
        principal=100_000,
        start_date='2010-01-01',
        end_date='2020-01-01',
        context_len=30,
        transaction_cost=0.0001,
        offset=0,
        init_strategy='long'
    )
    augmentation_args = {
            "signal_lengths":[5, 10],
            "noise_stdevs":[1],
            "stretch_factors":[1]
    }

    true_env = TradingEnv(env_args)

    env_generator = EnvGenerator(env_args, **augmentation_args)
    env_generator.generate()


    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_args = TDTransformerArgs(
        dim=512,
        n_layers=3,
        head_dim=64,
        hidden_dim=512,
        n_heads=8,
        n_kv_heads=4,
        context_length=true_env.context_len,
        num_features=true_env.state_len,
        norm_eps=1e-6,
        action_dim=2,
        device=device
    )

    model = TransformerDQN(model_args).to(device)
    M = ExperienceReplay(24_000)

    tracker = MetricTracker(env_args.start_date, env_args.end_date)
    train(model, env_generator, true_env, device, tracker, M)



if __name__ == "__main__":
    main(sys.argv[1])








