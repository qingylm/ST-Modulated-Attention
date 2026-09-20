from .LightConeMaskEngine import LightConeMaskEngine
from .MinkowskiLogitsCalculator import MinkowskiLogitsCalculator
from .SlidingWindowCache import SlidingWindowCache
import torch.nn as nn
import torch.nn.functional as F
import torch


class SpacetimeAttentionWithCache(nn.Module):
    """
    带滑动窗口缓存的完整时空注意力层
    """

    def __init__(self, d_model, d_space, d_time, num_heads, window_size=512):
        super().__init__()
        self.num_heads = num_heads
        self.d_space = d_space
        self.d_time = d_time

        # 标准 QKV 投影（但解耦空间和时间）
        self.W_q_s = nn.Linear(d_model, d_space * num_heads)
        self.W_q_t = nn.Linear(d_model, d_time * num_heads)
        self.W_k_s = nn.Linear(d_model, d_space * num_heads)
        self.W_k_t = nn.Linear(d_model, d_time * num_heads)
        self.W_v = nn.Linear(d_model, d_model * num_heads)  # V 保持标准语义

        # 核心计算部件
        self.logits_calc = MinkowskiLogitsCalculator(d_space, d_time)
        self.mask_engine = LightConeMaskEngine()

        # 滑动窗口缓存器
        self.cache = SlidingWindowCache(window_size)

        # 输出投影
        self.out_proj = nn.Linear(d_model * num_heads, d_model)

    def reset_cache(self):
        """在开始新序列前调用"""
        self.cache.reset()

    def forward(self, x, coords_current):
        """
        Args:
            x: [Batch, Curr_Seq, D_model] 当前输入的语义嵌入
            coords_current: [Batch, Curr_Seq, 4] 当前输入的时空坐标（已归一化）
        """
        Batch, Curr_Seq, _ = x.shape

        Q_s = self.W_q_s(x).view(Batch, Curr_Seq, self.num_heads, self.d_space).transpose(1, 2)
        Q_t = self.W_q_t(x).view(Batch, Curr_Seq, self.num_heads, self.d_time).transpose(1, 2)
        K_s = self.W_k_s(x).view(Batch, Curr_Seq, self.num_heads, self.d_space).transpose(1, 2)
        K_t = self.W_k_t(x).view(Batch, Curr_Seq, self.num_heads, self.d_time).transpose(1, 2)
        V = self.W_v(x).view(Batch, Curr_Seq, self.num_heads, -1).transpose(1, 2)

        cache_k_s, cache_k_t, cache_v, cache_coords = self.cache.update(K_s, K_t, V, coords_current)
        Total_Seq = cache_k_s.size(2)

        raw_logits = self.logits_calc(Q_s, cache_k_s, Q_t, cache_k_t)  # [B, H, Curr_Seq, Total_Seq]

        # 生成光锥掩码
        # 注意：mask_engine 接受 coords 并返回 [B, Total_Seq, Total_Seq]
        full_mask = self.mask_engine(cache_coords)  # [B, Total_Seq, Total_Seq]
        # 裁剪为 [B, Curr_Seq, Total_Seq]  (只取当前 query 对应的行)
        mask = full_mask[:, :Curr_Seq, :]  # [B, Curr_Seq, Total_Seq]
        # 扩展 head 维度
        mask = mask.unsqueeze(1)  # [B, 1, Curr_Seq, Total_Seq]

        masked_logits = raw_logits + mask
        attn_weights = F.softmax(masked_logits, dim=-1)
        context = torch.matmul(attn_weights, cache_v)
        context = context.transpose(1, 2).contiguous().view(Batch, Curr_Seq, -1)
        output = self.out_proj(context)

        return output, attn_weights