import torch
import torch.nn as nn


class LightConeMaskEngine(nn.Module):
    def __init__(self, init_bias=5.0):
        super().__init__()
        self.spacelike_bias = nn.Parameter(torch.tensor(init_bias))

    def forward(self, coords):
        # coords: [Batch, Seq, 4]  (x, y, z, t)
        x, y, z, t = coords[..., 0], coords[..., 1], coords[..., 2], coords[..., 3]

        # delta_t = t_query - t_key  (行 - 列)
        # (i,j) 位置表示查询 i 的时间减去键 j 的时间
        delta_t = t.unsqueeze(-1) - t.unsqueeze(-2)  # [B, Seq, Seq]
        delta_x = x.unsqueeze(-1) - x.unsqueeze(-2)
        delta_y = y.unsqueeze(-1) - y.unsqueeze(-2)
        delta_z = z.unsqueeze(-1) - z.unsqueeze(-2)

        space_dist_sq = delta_x ** 2 + delta_y ** 2 + delta_z ** 2
        interval_sq = space_dist_sq - delta_t ** 2  # ds^2

        # 因果掩码：键的时间 > 查询的时间 → 未来 → 屏蔽
        # 因为 delta_t = t_query - t_key，所以未来键对应 delta_t < 0
        causal_mask = delta_t < 0

        # 类空掩码：ds^2 > 0 且 非因果（即过去或同时）
        spacelike_mask = (interval_sq > 0) & (~causal_mask)

        mask_matrix = torch.zeros_like(delta_t, dtype=torch.float32)
        # 1. 硬因果掩蔽（绝对优先）
        mask_matrix = mask_matrix.masked_fill(causal_mask, -1e9)
        # 2. 类空软惩罚（只作用于非因果位置）
        penalty = torch.abs(self.spacelike_bias)
        mask_matrix = mask_matrix.masked_fill(spacelike_mask, -penalty)

        return mask_matrix