import torch
import torch.nn as nn

# 假设 hidden_size = 512
layer_norm = nn.LayerNorm(512)

# 模拟输入: (batch_size, seq_len, hidden_dim)
x = torch.randn(4, 10, 512)
output = layer_norm(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        # Pre-LN: 在每个子层前使用 LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 1. 多头注意力部分 (Pre-LN)
        attn_output, _ = self.attention(x, x, x)
        x = x + self.dropout(attn_output)  # 残差连接
        x = self.norm1(x)                  # 子层后归一化

        # 2. 前馈网络部分 (Pre-LN)
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)    # 残差连接
        x = self.norm2(x)                  # 子层后归一化
        return x