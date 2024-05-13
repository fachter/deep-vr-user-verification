import torch
from torch import nn


class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model, sequence_length):
        super(RelativePositionalEncoding, self).__init__()
        self.sequence_length = sequence_length
        self.embedding = nn.Embedding(sequence_length, d_model)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, dtype=torch.long, device=x.device)
        positions = positions.unsqueeze(0).expand(x.shape[0], -1)
        return x + self.embedding(positions)


class TimeAbsolutePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, sequence_length: int):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(sequence_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000)) / d_model)
        )
        pe = torch.zeros(1, sequence_length, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("positional_encoding", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.positional_encoding  # [:, :x.size(1)]
        return self.dropout(x)