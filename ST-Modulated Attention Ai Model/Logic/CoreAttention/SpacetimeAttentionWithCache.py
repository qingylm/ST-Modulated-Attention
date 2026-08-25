from LightConeMaskingEngine import LightConeMaskEngine
from MinkowskiLogitsCalculator import MinkowskiLogitsCalculator
from SlidingWindowCache import SlidingWindowCache
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
            coords_current: [Batch, Curr_Seq, 4] 当前输入的时空坐标
        Returns:
            output: [Batch, Curr_Seq, D_model]
            attn_weights: [Batch, Heads, Curr_Seq, Total_Seq] 注意力权重（用于调试）
        """
        Batch, Curr_Seq, _ = x.shape

        # 1. 线性投影并重塑为多头格式
        Q_s = self.W_q_s(x).view(Batch, Curr_Seq, self.num_heads, self.d_space).transpose(1, 2)
        Q_t = self.W_q_t(x).view(Batch, Curr_Seq, self.num_heads, self.d_time).transpose(1, 2)
        K_s = self.W_k_s(x).view(Batch, Curr_Seq, self.num_heads, self.d_space).transpose(1, 2)
        K_t = self.W_k_t(x).view(Batch, Curr_Seq, self.num_heads, self.d_time).transpose(1, 2)
        V = self.W_v(x).view(Batch, Curr_Seq, self.num_heads, -1).transpose(1, 2)  # [B, H, Curr_Seq, D]

        # 2. 从缓存中获取历史的 K, V, Coords（关键步骤）
        # 注意：缓存的坐标和历史K/V一起被取出来
        cache_k_s, cache_k_t, cache_v, cache_coords = self.cache.update(K_s, K_t, V, coords_current)

        # 此时 cache_k_s 的序列维度 = 历史缓存长度 + Curr_Seq
        Total_Seq = cache_k_s.size(2)

        # 3. 计算完整注意力分数（Q 只针对当前序列，K 针对全部缓存序列）
        # 注意：Q_s 维度是 [B, H, Curr_Seq, D]，K_s 是 [B, H, Total_Seq, D]
        # 标准矩阵乘法会自动广播，结果 [B, H, Curr_Seq, Total_Seq]
        raw_logits = self.logits_calc(Q_s, cache_k_s, Q_t, cache_k_t)

        # 4. 生成光锥掩码（作用于全部缓存坐标）
        # coords 维度 [B, Total_Seq, 4]
        mask = self.mask_engine(cache_coords)  # [B, Total_Seq, Total_Seq]
        mask = mask.unsqueeze(1)  # [B, 1, Total_Seq, Total_Seq]

        # 重要：mask 需要裁剪，只保留 Q 对应当前序列的部分
        # 因为我们的 Q 只有 Curr_Seq 个，而 mask 是 Total_Seq x Total_Seq
        # 我们取 mask 的前 Curr_Seq 行（对应当前 Query），所有列对应全部缓存 Keys
        # 但由于我们直接使用了广播计算 raw_logits = Q(Curr_Seq) @ K(Total_Seq).T
        # 所以 raw_logits 是 [B, H, Curr_Seq, Total_Seq]
        # 对应的 mask 也必须是 [B, 1, Curr_Seq, Total_Seq]
        # 此处我们做一个简单的切片：保留 mask 的前 Curr_Seq 行
        mask = mask[:, :, :Curr_Seq, :]  # 确保维度对齐

        masked_logits = raw_logits + mask

        # 5. Softmax 和加权求和
        attn_weights = F.softmax(masked_logits, dim=-1)  # [B, H, Curr_Seq, Total_Seq]
        context = torch.matmul(attn_weights, cache_v)  # [B, H, Curr_Seq, D_model]

        # 6. 重塑并输出
        context = context.transpose(1, 2).contiguous().view(Batch, Curr_Seq, -1)
        output = self.out_proj(context)

        return output, attn_weights