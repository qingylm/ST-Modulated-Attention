import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from Logic.CoreAttention import SpacetimeAttentionWithCache
from Logic.SubsequentProcessing import LearnableSpacetimeNormalizer

class SpacetimeTransformerBlock(nn.Module):
    """单层时空Transformer块（注意力 + 前馈 + 残差）"""
    def __init__(self, d_model, d_space, d_time, num_heads, window_size, dropout=0.1):
        super().__init__()
        self.attn = SpacetimeAttentionWithCache(d_model, d_space, d_time, num_heads, window_size)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def reset_cache(self):
        self.attn.reset_cache()

    def forward(self, x, coords_norm):
        # 注意力
        attn_out, attn_weights = self.attn(x, coords_norm)
        x = self.norm1(x + self.dropout(attn_out))
        # 前馈
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x, attn_weights


class SpacetimeLM(nn.Module):
    """完整的时空语言模型（用于自回归训练）"""
    def __init__(self, vocab_size, d_model, d_space, d_time, num_heads,
                 window_size, num_layers=4, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.coord_normalizer = LearnableSpacetimeNormalizer()
        self.blocks = nn.ModuleList([
            SpacetimeTransformerBlock(d_model, d_space, d_time, num_heads, window_size, dropout)
            for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def reset_caches(self):
        """重置所有层的缓存（每个新序列开始前调用）"""
        for block in self.blocks:
            block.reset_cache()

    def forward(self, input_ids, coords_raw, attention_mask=None):
        """
        input_ids: [B, T]
        coords_raw: [B, T, 4]
        attention_mask: [B, T] (1=有效, 0=填充)
        """
        # 1. 嵌入
        x = self.token_embedding(input_ids)  # [B, T, D]
        x = self.dropout(x)

        # 2. 归一化坐标
        coords_norm = self.coord_normalizer(coords_raw)  # [B, T, 4]

        # 3. 逐层处理（注意：我们逐样本处理以维持缓存独立性）
        B, T = input_ids.size()
        outputs = []
        attn_weights_list = []
        for b in range(B):
            # 每个样本独立处理
            x_b = x[b:b+1]               # [1, T, D]
            coords_b = coords_norm[b:b+1] # [1, T, 4]
            for block in self.blocks:
                x_b, attn_w = block(x_b, coords_b)
                attn_weights_list.append(attn_w)  # 用于调试
            outputs.append(x_b)
        x = torch.cat(outputs, dim=0)  # [B, T, D]

        # 4. 输出头
        logits = self.lm_head(x)  # [B, T, vocab]

        # 如果提供了attention_mask，将填充位置的logits置为极小值（用于损失计算）
        if attention_mask is not None:
            # 只需在损失计算时处理，这里不修改logits本身
            pass

        return logits, attn_weights_list