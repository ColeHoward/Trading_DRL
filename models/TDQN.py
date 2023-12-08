import torch
import torch.nn as nn
from . import DQN
from dataclasses import dataclass


@dataclass
class TQDNModelArgs:
    contex_length: int = 30
    n_features: int = 13
    n_actions: int = 2
    dim: int = 512
    context_length: int = 30
    dropout: float = 0.2
    tgt_update_freq: int = 1000
    state_len: int = 5

model_args = TQDNModelArgs(state_len=18)

class TDQN(nn.Module):
    """📈"""
    def __init__(self, args: TQDNModelArgs):
        super(TDQN, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_actions = args.n_actions

        self.policy = DQN(args.contex_length, args.n_features,  args.n_actions, args.dim, args.dropout).to(self.device)
        self.tgt = DQN(args.contex_length, args.n_features, args.n_actions, args.dim, args.dropout).to(self.device)
        self.tgt.load_state_dict(self.policy.state_dict())
        self.tgt.eval()


    def main_pred(self, state) -> torch.Tensor:
        action_values = self.policy(state)
        return action_values

    def tgt_pred(self, state) -> torch.Tensor:
        action_values = self.tgt(state)
        return action_values

    def update_tgt(self) -> None:
        self.tgt.load_state_dict(self.policy.state_dict())

    def save_models(self, path) -> None:
        torch.save(self.policy.state_dict(), path + '_policy')
        torch.save(self.tgt.state_dict(), path + '_tgt')

    def load_models(self, path) -> None:
        self.policy.load_state_dict(torch.load(path + '_policy'))
        self.tgt.load_state_dict(torch.load(path + '_tgt'))
        self.tgt.eval()
        self.policy.eval()

    def train_mode(self):
        self.policy.train()

    def test_mode(self):
        self.policy.eval()




def test_dqn():
    input_dim = 30
    output_dim = 5
    hidden_dim = 512
    dropout_rate = 0.2
    batch_size = 1

    dqn = DQN(context_length=input_dim, num_features=17, n_out=output_dim, dim=hidden_dim, dropout=dropout_rate)

    dummy_input = torch.rand((batch_size, 30, 17))

    output = dqn(dummy_input)

    assert output.shape == (1, output_dim), "Output shape is incorrect"

    print("Test passed: DQN produces output of expected shape.")

if __name__ == "__main__":
    test_dqn()



