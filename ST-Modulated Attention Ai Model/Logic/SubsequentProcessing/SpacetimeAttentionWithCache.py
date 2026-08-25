from LearnableSpacetimeNormalizer import LearnableSpacetimeNormalizer
import torch
import torch.nn as nn

class SpacetimeAttentionWithCache(nn.Module):
    def __init__(self, d_model, d_space, d_time, num_heads, window_size):
        super().__init__()
        # ... 原有的投影层、计算层、缓存初始化 ...

        # 新增：坐标预归一化层 (放在最前面)
        self.coord_normalizer = LearnableSpacetimeNormalizer(
            space_norm_type='rms',  # 空间用RMS，对旋转不变
            time_norm_type='minmax',  # 时间用MinMax，保留相对顺序
            clip_value=2.0  # 最大绝对值不超过2
        )

    def forward(self, x, coords_raw):
        """
        x: [Batch, Curr_Seq, D_model] 语义嵌入
        coords_raw: [Batch, Curr_Seq, 4] 原始时空坐标
        """
        Batch, Curr_Seq, _ = x.shape

        # ---- 1. 坐标归一化 ----
        coords_norm = self.coord_normalizer(coords_raw)  # [B, S, 4]

        # ---- 2. Q/K/V 投影（解耦空间和时间） ----
        Q_s = self.W_q_s(x).view(Batch, Curr_Seq, self.num_heads, self.d_space).transpose(1, 2)
        Q_t = self.W_q_t(x).view(Batch, Curr_Seq, self.num_heads, self.d_time).transpose(1, 2)
        K_s = self.W_k_s(x).view(Batch, Curr_Seq, self.num_heads, self.d_space).transpose(1, 2)
        K_t = self.W_k_t(x).view(Batch, Curr_Seq, self.num_heads, self.d_time).transpose(1, 2)
        V = self.W_v(x).view(Batch, Curr_Seq, self.num_heads, -1).transpose(1, 2)

        # ---- 3. 更新缓存（关键：K_s, K_t, V, coords_norm 必须已定义） ----
        cache_k_s, cache_k_t, cache_v, cache_coords = self.cache.update(K_s, K_t, V, coords_norm)

        # ---- 4. 后续注意力计算 ----
        raw_logits = self.logits_calc(Q_s, cache_k_s, Q_t, cache_k_t)
        mask = self.mask_engine(cache_coords)  # [B, Total_Seq, Total_Seq]
        mask = mask.unsqueeze(1)[:, :, :Curr_Seq, :]  # 对齐维度
        masked_logits = raw_logits + mask
        attn_weights = F.softmax(masked_logits, dim=-1)
        context = torch.matmul(attn_weights, cache_v)
        context = context.transpose(1, 2).contiguous().view(Batch, Curr_Seq, -1)
        output = self.out_proj(context)

        return output, attn_weights



def test_normalizer():
    normalizer = LearnableSpacetimeNormalizer(clip_value=2.0)
    coords_bad = torch.tensor([[[0.0, 0.0, 0.0, 0.0],
                                 [10.0, 0.0, 0.0, 0.0],
                                 [20.0, 0.0, 0.0, 0.0],
                                 [30.0, 0.0, 0.0, 0.0],
                                 [40.0, 0.0, 0.0, 0.0]]])
    coords_good = normalizer(coords_bad)
    print("归一化前:", coords_bad[0, :, 0])  # x: [0, 10, 20, 30, 40]
    print("归一化后:", coords_good[0, :, 0]) # x: 将被裁剪到 [-2, 2] 且尺度压缩


test_normalizer()