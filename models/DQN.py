import torch
import torch.nn as nn
import torch.nn.functional as F



class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)

        return output * self.weight

class Block(nn.Module):
    """Simple Linear Block with BatchNorm and Dropout"""
    def __init__(self, n_in, n_out, dropout=.2):
        super(Block, self).__init__()
        self.fc = nn.Linear(n_in, n_out)
        self.bn = RMSNorm(n_out)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, input):
        h = self.dropout(F.gelu(self.bn(self.fc(input))))
        return h

class DQN(nn.Module):
    """
    Vanilla DQN Network
    """
    def __init__(self, context_length, num_features, n_out, dim=512, dropout=.2):
        super(DQN, self).__init__()

        flattened_input_size = context_length * num_features
        self.net = nn.Sequential(
            Block(flattened_input_size, dim, dropout),
            Block(dim, dim, dropout),
            Block(dim, dim, dropout),
            Block(dim, dim, dropout),
            nn.Linear(dim, n_out)
        )

        nn.init.xavier_uniform_(self.net[-1].weight)

    def forward(self, input):
        if len(input.shape) != 3:
            print('input shape', input.shape)
            raise ValueError("Expected input shape: (batch_size, context_length, num_features)")

        batch_size, context_length, num_features = input.shape
        flattened_input = input.view(batch_size, -1)
        output = self.net(flattened_input)

        return output





def test_dqn():
    context_length = 30
    output_dim =10
    hidden_dim = 10
    dropout_rate = 0.2

    dqn = DQN(context_length=context_length, n_out=output_dim, dim=hidden_dim, dropout=dropout_rate)

    dummy_input = torch.rand((2, context_length))

    output = dqn(dummy_input)

    assert output.shape == (10, output_dim), f"Output shape is incorrect. Exepected shape: {10, 10}, received shape: " \
                                               + f"{(10, output_dim)}"

    print("PASSED: DQN produces output of expected shape.")


# Run the test
if __name__ == "__main__":
    test_dqn()














