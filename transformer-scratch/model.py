import math
from typing import Callable, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super(InputEmbedding, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # (batch, seq_len) -> (batch, seq_len, d_model)
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(p=dropout)

        # create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # create a vector of shape (seq_len), it could be a tensor with shape (seq_len, 1)
        # NOTE: unsequeeze 方法的作用是在张量的特定维度上添加一个长度为1的新维度
        position = torch.arange(0, seq_len, dtype=torch.float).unsequeeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # (d_model/2)
        # apply the sin to even positions
        pe[:, 0::2] = torch.sin(position * div_term)
        # apply the cos to odd positions
        pe[:, 1::2] = torch.cos(position * div_term)

        # add batch dim
        self.pe = pe.unsqueeze(0)  # (1, seq_len, d_model)

        # register the buffer to the module
        # NOTE: 通过register_buffer注册的张量会作为模块的一部分持久化保存, 不会参与梯度计算。
        #       这意味着它们不会在训练过程中被更新，适用于存储一些固定不变的数据。
        self.register_buffer("pe", self.pe)

    def forward(self, x: torch.Tensor):
        # add positional encoding to the input
        # positional encoding 不参与梯度计算
        x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(False)
        # (batch, seq_len, d_model)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(self, feature_size: int, eps: float = 1e-6):
        super().__init__()
        # eps is a small value added to the variance to avoid divide-by-zero
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(feature_size))  # multiplicative factor
        self.bias = nn.Parameter(torch.zeros(feature_size))  # additive factor

    def forward(self, x: torch.Tensor):
        # x: (batch, seq_len, hidden_size)
        # 我们希望对x的最后一个维度（即hidden_size维度）进行归一化操作，
        # 因此需要计算这个维度上的均值和标准差。 最后一个维度：dim=-1
        # keep the dimension for broadcasting
        mean = x.mean(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        # keep the dimension for broadcasting
        std = x.std(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        # normalize the input
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)  # W1, b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)  # W2, b2

    def forward(self, x: torch.Tensor):
        # (batch, seq_len, d_model) -> (batch, seq_len, d_ff)
        x = self.linear_1(x)
        x = self.dropout(F.relu(x))
        # (batch, seq_len, d_ff) -> (batch, seq_len, d_model)
        return self.linear_2(x)


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_head: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.h = num_head
        assert d_model % num_head == 0, "d_model should be divisible by num_head"
        self.d_k = d_model // num_head

        self.w_q = nn.Linear(d_model, self.d_k * self.h)  # Wq
        self.w_k = nn.Linear(d_model, self.d_k * self.h)  # Wk
        self.w_v = nn.Linear(d_model, self.d_k * self.h)  # Wv

        self.w_o = nn.Linear(self.d_k * self.h, d_model)  # Wo

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor = None,
        dropout: nn.Dropout = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        d_k = q.shape[-1]
        # (batch, h, seq_len, d_k) @ (batch, h, d_k, seq_len) --> (batch, h, seq_len, seq_len)
        attention_scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores.softmax(dim=-1)  # (batch, h, seq_len, seq_len) # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) @ (batch, h, seq_len, d_k) --> (batch, h, seq_len, d_k)
        attention_output = attention_scores @ v
        return attention_output, attention_scores

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        query = self.w_q(q)  # (batch, seq_len, d_model) --> (batch, seq_len, h * d_k)
        key = self.w_k(k)  # (batch, seq_len, d_model) --> (batch, seq_len, h * d_k)
        value = self.w_v(v)  # (batch, seq_len, d_model) --> (batch, seq_len, h * d_k)

        # (batch, seq_len, h * d_k) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        # 通过 transpose 将 h, seq_len 两个维度调换
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(
            1, 2
        )
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(
            1, 2
        )
        # compute attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(
            query, key, value, mask, self.dropout
        )
        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, h * d_k)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        # (batch, seq_len, h * d_k) --> (batch, seq_len, d_model)
        return self.w_o(x)


class ResidualConnection(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(
        self, x: torch.Tensor, sublayer: Callable[[torch.Tensor], torch.Tensor]
    ):
        # 这里的实现跟paper不同，这里先norm再sublayer，而paper是先sublayer再norm
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(
        self,
        self_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(2)]
        )

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor = None):
        x = self.residual_connection[0](
            x, lambda x: self.self_attention_block(x, x, x, src_mask)
        )
        x = self.residual_connection[1](x, lambda x: self.feed_forward_block(x))
        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        self_attention_block: MultiHeadAttentionBlock,
        cross_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(3)]
        )

    def forward(self, x, encoder_output, encoder_mask, decoder_mask):
        x = self.residual_connection[0](
            x, lambda x: self.self_attention_block(x, x, x, decoder_mask)
        )
        x = self.residual_connection[1](
            x,
            lambda x: self.cross_attention_block(
                x, encoder_output, encoder_output, encoder_mask
            ),
        )
        x = self.residual_connection[2](x, lambda x: self.feed_forward_block(x))
        return x


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, encoder_mask, decoder_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, encoder_mask, decoder_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return F.log_softmax(self.proj(x))


class Transformer(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: InputEmbedding,
        tgt_embed: InputEmbedding,
        src_pos: PositionalEncoding,
        tgt_pos: PositionalEncoding,
        projection_layer: ProjectionLayer,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)


def build_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_seq_len: int,
    tgt_seq_len: int,
    d_model: int = 512,
    num_head: int = 8,
    num_encoder_layer: int = 6,
    num_decoder_layer: int = 6,
    d_ff: int = 2048,
    dropout: float = 0.1,
) -> Transformer:
    # create embedding layers
    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)
    # create positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    # create encoder layers
    encoder_blocks = []
    for _ in range(num_encoder_layer):
        self_attention_block = MultiHeadAttentionBlock(d_model, num_head, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
    encoder = Encoder(nn.ModuleList(encoder_blocks))

    # create decoder layers
    decoder_blocks = []
    for _ in range(num_decoder_layer):
        self_attention_block = MultiHeadAttentionBlock(d_model, num_head, dropout)
        cross_attention_block = MultiHeadAttentionBlock(d_model, num_head, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(
            self_attention_block, cross_attention_block, feed_forward_block, dropout
        )
        decoder_blocks.append(decoder_block)
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # create projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # create transformer
    transformer = Transformer(
        encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer
    )
    # initialize parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_normal_(p)

    return transformer
