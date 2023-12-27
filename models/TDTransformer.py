import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class TDTransformerArgs:
    dim: int
    n_layers: int
    head_dim: int
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    context_length: int
    num_features: int
    norm_eps: float
    device: str
    action_dim: int = 2


def precompute_freqs_cis(dim: int, end: int, theta: float = 100.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:, None, :]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
    return xq_out.type_as(xq).view(*xq.shape), xk_out.type_as(xk).view(*xk.shape)



class Attention(nn.Module):
    def __init__(self, args: TDTransformerArgs):
        super().__init__()
        self.args = args

        self.n_heads: int = args.n_heads  # number of query heads (or attention heads)
        self.n_kv_heads: int = args.n_kv_heads  # number of key/value heads

        self.repeats = self.n_heads // self.n_kv_heads  # number of times to repeat keys and values

        self.scale = self.args.head_dim ** -0.5
        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)

    def forward(self, x: torch.Tensor, freqs_cis) -> torch.Tensor:
        batch_size, context_length, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # Reshape for multi-head attention
        xq = xq.view(batch_size, context_length, self.n_heads, self.args.head_dim)
        xk = xk.view(batch_size, context_length, self.n_kv_heads, self.args.head_dim)
        xv = xv.view(batch_size, context_length, self.n_kv_heads, self.args.head_dim)

        # Apply rotary embeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # Repeat keys and values
        keys = xk.repeat_interleave(repeats=self.repeats, dim=2)
        vals = xv.repeat_interleave(repeats=self.repeats, dim=2)

        # Compute attention scores
        attention_scores = torch.matmul(xq, keys.transpose(-1, -2)) * self.scale

        # Compute attention weights and weighted sum of values
        attention_weights = F.softmax(attention_scores, dim=-1)
        weighted_values = torch.matmul(attention_weights, vals)

        weighted_values = weighted_values.view(batch_size, context_length, self.n_heads * self.args.head_dim)

        return self.wo(weighted_values)


class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim))

    def forward(self, x):
        # x.shape: (batch_size, context_length, dim)

        # Compute attention scores
        attn_scores = torch.matmul(x, self.query)  # (batch_size, context_length)
        attn_scores = F.softmax(attn_scores, dim=1)  # (batch_size, context_length)

        # weighted sum
        pooled_output = torch.sum(x * attn_scores.unsqueeze(-1), dim=1)  # (batch_size, dim)

        return pooled_output


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


class FeedForward(nn.Module):
    def __init__(self, args: TDTransformerArgs):
        super().__init__()

        self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=False)
        self.w2 = nn.Linear(args.hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, args.hidden_dim, bias=False)

    def forward(self, x) -> torch.Tensor:
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: TDTransformerArgs):
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args=args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.args = args

    def forward(self, x: torch.Tensor, freqs_cis) -> torch.Tensor:
        r = self.attention.forward(self.attention_norm(x), freqs_cis=freqs_cis)
        h = x + r
        r = self.feed_forward.forward(self.ffn_norm(h))
        out = h + r

        return out


class TDTransformer(nn.Module):
    def __init__(self, args: TDTransformerArgs):
        super().__init__()
        self.args = args
        self.n_layers = args.n_layers

        self.feature_embedding = nn.Linear(args.num_features, args.dim)
        # Transformer layers
        self.layers = torch.nn.ModuleList(
            [TransformerBlock(args=args) for _ in range(args.n_layers)]
        )

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.freqs_cis = precompute_freqs_cis(args.head_dim, args.context_length).to(args.device)
        self.attention_pool = AttentionPooling(args.dim)
        self.output = nn.Linear(args.dim, args.action_dim)

    def forward(self, state) -> torch.Tensor:
        batch_size, context_length, num_features = state.shape

        # Apply feature embeddings
        h = self.feature_embedding(state.view(-1, num_features)).view(batch_size, context_length, -1)

        # Apply rotary embeddings and pass through transformer layers
        for layer in self.layers:
            h = layer(h, freqs_cis=self.freqs_cis)

        h = self.attention_pool(self.norm(h))

        # return action values
        return self.output(h)


class GRUModel(nn.Module):
    def __init__(self, n_features, n_hidden, n_actions):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size=n_features, hidden_size=n_hidden, batch_first=True, num_layers=2)
        self.fc = nn.Linear(n_hidden, n_actions)
        self.attention_pool = AttentionPooling(n_hidden)

    def forward(self, x):
        # x shape: (batch_size, context_length, n_features)
        output, _ = self.gru(x)

        last_output = self.attention_pool(output)

        out = self.fc(last_output)

        return out



class TransformerDQN(nn.Module):
    def __init__(self, args: TDTransformerArgs):
        super().__init__()
        # self.policy = GRUModel(args.num_features, args.dim, args.action_dim)
        # self.tgt = GRUModel(args.num_features, args.dim, args.action_dim)
        self.policy = TDTransformer(args)
        self.tgt = TDTransformer(args)
        self.tgt.load_state_dict(self.policy.state_dict())
        self.tgt.eval()
        self.tau = 0.5

    def main_pred(self, state) -> torch.Tensor:
        action_values = self.policy(state)
        return action_values

    def tgt_pred(self, state) -> torch.Tensor:
        action_values = self.tgt(state.float())
        return action_values

    def update_tgt(self, tau=0.5) -> None:
        self.tau = self.tau ** 1.1
        tau = max(self.tau, .01)
        # Soft update the target network
        for target_param, policy_param in zip(self.tgt.parameters(), self.policy.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1 - tau) * target_param.data)

    def save_models(self, path) -> None:
        torch.save(self.policy.state_dict(), path + '_policy')
        torch.save(self.tgt.state_dict(), path + '_tgt')

    def load_models(self, path) -> None:
        self.policy.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        self.policy.eval()

    def train_mode(self) -> None:
        self.policy.train()

    def test_mode(self) -> None:
        self.policy.eval()


def test_td_transformer():
    args = TDTransformerArgs(
        dim=128,
        n_layers=2,
        head_dim=32,
        hidden_dim=128,
        n_heads=4,
        n_kv_heads=2,
        context_length=23,
        num_features=19,
        norm_eps=1e-6,
        action_dim=2
    )

    model = TDTransformer(args)
    state = torch.randn(17, args.context_length, args.num_features)
    action_values = model(state)
    print(action_values.shape)

if __name__ == "__main__":
    test_td_transformer()