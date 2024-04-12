import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32  # number of heads for the Q
    n_kv_heads: Optional[int] = None  # number of heads for the K and V
    vocab_size: int = -1  # This will be set when we load the tokenizer
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    # needed for KV-cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None


# 生成旋转矩阵
def precompute_theta_pos_frequencise(
    head_dim: int, seq_len: int, device: str = None, theta: float = 10000.0
) -> torch.Tensor:
    # as written in the paper, the dimension of the embedding must be even
    assert head_dim % 2 == 0, "head_dim must be even"
    # build the theta parameters
    # according to the formula: theta_i = 10000 ^ (-2(i-1)/dim) for i = 1, 2, ..., dim/2
    # shape: (head_dim / 2), 这一部分是公式中的(i-1)部分, 从0到dim/2-1
    theta_numerator = torch.arange(0, head_dim, 2).float()
    # shape: (head_dim / 2)
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)
    # construct the positions (the "m" parameter)
    # shape: (seq_len)
    m = torch.arange(seq_len, device=device)
    # multiply each theta by each position using the outer product
    # shape: (seq_len) outer_product* (head_dim / 2) --> (seq_len, head_dim / 2)
    freqs = torch.outer(m, theta).float()
    # 计算外积后，矩阵中的一行长这样： [m_1*theta_1, m_1*theta_2,..., m_1*theta_dim/2]
    # compute the complex representation of the frequencies int the polar form
    # polar form: c = R * exp(i * m * theta), where R = 1 as follows:
    # 极坐标表示，根据欧拉公式，可以转为 R*exp(i * m * theta) = R*cos(m * theta) + i * R*sin(m * theta)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    # 转为复数后，矩阵中的一行长这样：
    # [cos(m_1 * theta_1) + i*sin(m_1 * theta_1), cos(m_1 * theta_2) + i*sin(m_1 * theta_2), ..., cos(m_1 * theta_dim/2) + i*sin(m_1 * theta_dim/2)]
    # shape: (seq_len, head_dim / 2)
    return freqs_complex


# WARN: 这里维度变化的注释可能有点问题，需要进一步确认
def apply_rotary_embedding(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # (B, seq_len, H, head_dim) --> (B, seq_len, H, head_dim/2, 2)
    # x.shape[:-1] 获取除最后一个维度外的所有维度， '*'是解包运算符，用于解包这个元组，使其成为reshape函数的多个位置参数,
    # -1告诉reshape函数自动计算这个维度的大小，以保持总元素数量不变。最后，2指定了新的最后一个维度的大小。
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # (seq_len, head_dim/2) --> (1, seq_len, 1, head_dim/2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # (B, seq_len, H, head_dim/2, 2) * (1, seq_len, 1, head_dim/2) --> (B, seq_len, H, head_dim/2)
    x_rotated = x_complex * freqs_complex
    # (B, seq_len, H, head_dim/2) --> (B, seq_len, H, head_dim/2, 2)
    x_out = torch.view_as_real(x_rotated)
    # (B, seq_len, H, head_dim/2, 2) --> (B, seq_len, H, head_dim)
    x_out = x_out.flatten(*x.shape)
    return x_out.type_as(x).to(device)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps
        # the gamma parameter
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # (B, seq_len, dim) * (B, seq_len, 1) --> (B, seq_len, dim)
        # rsqrt: 1 / sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (dim) * (B, seq_len, dim) --> (B, seq_len, dim)
        return self.weight * self._norm(x)


def repeat_kv(x: torch.Tensor, n_repeat: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_repeat == 1:
        return x
    # (B, seq_len, n_kv_heads, head_dim) --> (B, seq_len, n_kv_heads * n_repeat, head_dim)
    return (
        x[:, :, :, None, :]
        .expand(batch_size, seq_len, n_kv_heads, n_repeat, head_dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_repeat, head_dim)
    )


class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super(SelfAttention, self).__init__()

        # indicates the number of heads for the K and V
        self.n_kv_heads = (
            args.n_kv_heads if args.n_kv_heads is not None else args.n_heads
        )
        # indicates the number of heads for the Q
        self.n_heads_q = args.n_heads
        # indicates how many times the K ans v shoule be repeated to match the number of heads for the Q
        self.n_repeat = self.n_heads_q // self.n_kv_heads
        # indicates the dim of each head
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros(
            args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim
        )
        self.cache_v = torch.zeros(
            args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim
        )

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        # seq_len = 1
        batch_size, seq_len, _ = x.shape  # (B, 1, dim)
        # (B, 1, dim) -> (B, 1, H_Q * head_dim)
        xq = self.wq(x)
        # (B, 1, dim) -> (B, 1, H_K * head_dim)
        xk = self.wk(x)
        # (B, 1, dim) -> (B, 1, H_V * head_dim)
        xv = self.wv(x)
        # (B, 1, H_Q * head_dim) -> (B, 1, H_Q, head_dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        # (B, 1, H_K * head_dim) -> (B, 1, H_K, head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        # (B, 1, H_V * head_dim) -> (B, 1, H_V, head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        # 对Q、K进行旋转位置编码，V不需要。 不改变shape
        xq = apply_rotary_embedding(xq, freqs_complex, device=x.device)
        xk = apply_rotary_embedding(xk, freqs_complex, device=x.device)
        # replace the entry in the cache for this token
        self.cache_k[:batch_size, start_pos : start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos : start_pos + seq_len] = xv

        # retrieve all the cached k, V so far
        # (B, seq_len_KV, H_KV, head_dim)
        keys = self.cache_k[:batch_size, 0 : start_pos + seq_len]
        values = self.cache_v[:batch_size, 0 : start_pos + seq_len]

        # repeat the keys and values to match the number of heads for the Q
        keys = repeat_kv(keys, self.n_repeat)
        values = repeat_kv(values, self.n_repeat)

        # (B, 1, H_Q, head_dim) --> (B, H_Q, 1, head_dim)
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # (B, H_Q, 1, head_dim) @ (B, H_Q, head_dim, seq_len_KV) --> (B, H_Q, 1, seq_len_KV)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        # (B, HQ, 1, seq_len_KV) @ (B, HQ, seq_len_KV, head_dim) --> (B, HQ, 1, head_dim)
        output = torch.matmul(scores, values)
        # (B, HQ, 1, head_dim) --> (B, 1, HQ, head_dim) --> (B, 1, HQ * head_dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        # (B, 1, dim) -->（B, 1, dim)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super(FeedForward, self).__init__()
        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        # round the hidden_dim to the nearest multiple of args.multiple_of
        hidden_dim = args.multiple_of * (
            (hidden_dim + args.multiple_of - 1) // args.multiple_of
        )

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        swish = F.silu(self.w1(x))
        x_V = self.w3(x)
        out = swish * x_V
        return self.w2(out)


class Block(nn.Module):
    def __init__(self, args: ModelArgs):
        super(Block, self).__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        # norm before self-attention and feed-forward
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor
    ) -> torch.Tensor:
        # (B, seq_len, dim) + (B, seq_len, dim) --> (B, seq_len, dim)
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_complex)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super(Transformer, self).__init__()
        assert args.vocab_size != -1, "vocab_size must be set"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(Block(args))
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size)

        self.freqs_complex = precompute_theta_pos_frequencise(
            args.dim // args.n_heads, args.max_seq_len * 2, device=args.device
        )

    def forward(self, tokens: torch.Tensor, start_pos: int):
        # (B, seq_len)
        batch_size, seq_len = tokens.shape
        assert (
            seq_len == 1
        ), "Only one token at a time is supported, because we use the KV-cache mechanism"

        # (B, seq_len) --> (B, seq_len, dim)
        h = self.tok_embeddings(tokens)

        # Retrieve the pairs (m, theta) corresponding to the positions [start_pos, start_pos+seq_len]
        freqs_complex = self.freqs_complex[start_pos : start_pos + seq_len]

        # consecutively apply all the layers
        for layer in self.layers:
            h = layer(h, freqs_complex)
        h = self.norm(h)
        output = self.output(h)
        return output
