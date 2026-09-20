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

    def forward(self, input_ids, coords_raw, attention_mask=None,
                collect_attn_weights=False):
        """
        input_ids: [B, T]
        coords_raw: [B, T, 4]
        attention_mask: [B, T] (1=有效, 0=填充)
        collect_attn_weights: 是否收集注意力权重用于调试。
            默认 False —— 收集会把 B*num_layers 个 [1,H,T,T] 张量累积成
            列表，每个都持有 autograd 图，属于纯显存浪费（训练时无人使用）。

        Returns:
            logits: [B, T, vocab]
            attn_weights_list: 调试用注意力权重列表（未收集时为 None）
            coords_norm: [B, T, 4] 归一化后的坐标

        coords_norm 必须一并返回: 它才是注意力掩码与 Minkowski logits 实际
        使用的坐标。物理正则损失若作用在 coords_raw 上，会因 coords_raw 是
        数据集常量（requires_grad=False）而不携带计算图，对模型无梯度贡献。

        缓存隔离: 每个样本前向之前都会重置所有层缓存。滑动窗口缓存是
        **单个序列**的状态，不是 batch 状态；若在样本之间复用，样本 b 的
        注意力会读到样本 0..b-1 的 K/V（跨样本上下文泄漏）。因此逐样本
        循环的语义必须是"每个样本各自从空缓存开始"。
        """
        # 1. 嵌入
        x = self.token_embedding(input_ids)  # [B, T, D]
        x = self.dropout(x)

        # 2. 归一化坐标（含可学习参数，必须保留在计算图中）
        coords_norm = self.coord_normalizer(coords_raw)  # [B, T, 4]

        # 3. 逐样本处理（每个样本拥有独立的缓存生命周期）
        B, T = input_ids.size()
        outputs = []
        attn_weights_list = [] if collect_attn_weights else None
        for b in range(B):
            # 关键: 样本之间必须清空缓存，否则会跨样本泄漏上下文。
            # reset 只是把引用置 None，开销可忽略。
            self.reset_caches()
            x_b = x[b:b + 1]                 # [1, T, D]
            coords_b = coords_norm[b:b + 1]  # [1, T, 4]
            for block in self.blocks:
                x_b, attn_w = block(x_b, coords_b)
                if attn_weights_list is not None:
                    attn_weights_list.append(attn_w)
            outputs.append(x_b)
        # 再清一次，避免最后一个样本的缓存被下一次 forward 误当作历史
        self.reset_caches()
        x = torch.cat(outputs, dim=0)  # [B, T, D]

        # 4. 输出头
        logits = self.lm_head(x)  # [B, T, vocab]

        # 如果提供了attention_mask，将填充位置的logits置为极小值（用于损失计算）
        if attention_mask is not None:
            # 只需在损失计算时处理，这里不修改logits本身
            pass

        return logits, attn_weights_list, coords_norm
