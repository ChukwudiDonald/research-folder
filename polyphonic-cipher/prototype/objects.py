import torch.nn as nn
import torch


class CipherNoiseGRU(nn.Module):
    """Maps a sequence of tokens to a single class prediction"""
    def __init__(self,
                 embed_dim: int,
                 vocab_size: int,
                 hidden_size: int,
                 output_size: int,
                 pad_idx: int,
                 num_layers: int = 1,
                 dropout = 0.4):
        
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embed_dim,
                         hidden_size,
                         num_layers,
                         batch_first=True,
                         dropout=dropout,
                         bidirectional=True)
        self.fc = nn.Linear(hidden_size  * 2, output_size)  # Output size = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits shaped [batch_size, vocab_size]"""
        embedded = self.embedding(x)  # [B, L, D]
        gru_out, _ = self.gru(embedded)  # [B, L, H]
        # Take only the last output for many-to-one classification
        last_output = gru_out[:, -1, :]  # [B, H]
        logits = self.fc(last_output)  # [B, vocab_size]
        return logits
