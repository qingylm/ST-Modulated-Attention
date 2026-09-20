import torch
import torch.nn as nn


class LightConeMaskEngine(nn.Module):
    """
    光锥掩码引擎：把 (x, y, z, t) 坐标上的因果性与闵可夫斯基类空关系
    转换为可加在注意力 logits 上的加性掩码。

    掩码语义（三档，因果屏蔽优先级最高）:
        1. 因果屏蔽  : 键在查询的"未来"(delta_t < 0)      -> -inf   （硬屏蔽）
        2. 类空惩罚  : 键在查询的"过去或同时"且 ds^2 > 0  -> -|bias|（软惩罚）
        3. 其他                                        ->  0

    为什么必须写两个互斥的 mask 条件:
        spacelike_mask 里的 `& (~causal_mask)` 是正确性关键，而不是冗余条件。
        它保证类空惩罚**永远不会**落在未来位置上，因此两个 masked_fill /
        torch.where 的作用范围彼此不相交，写入顺序不再影响结果。

        如果哪天有人"简化"成 `spacelike_mask = interval_sq > 0`（去掉
        ~causal_mask），那么 |Δ空间| > |Δt| 的未来位置就会被从硬屏蔽覆写成
        -|bias|，等于给自回归注意力开后门。这在跨角色切换的真实数据上会
        立刻发生（build_lccc_dataset.py 的 ROLE_SPACE 让 user 的 x=0、
        assistant 的 x=1）。TestLightConeMaskEngine.py 里有一条断言
        专门守住"因果 ∩ 类空 == 空集"这个不变量。
    """

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
        # `& (~causal_mask)` 与 causal_mask 的互斥性见类文档字符串。
        spacelike_mask = (interval_sq > 0) & (~causal_mask)

        # 软惩罚强度始终为正，保证只会降低注意力权重
        penalty = torch.abs(self.spacelike_bias)

        # 两档掩码互斥，先写软惩罚、再写硬屏蔽（顺序本身不影响结果，
        # 因为作用范围不相交；保持这个顺序是为了语义上突出"因果优先"）。
        mask_matrix = torch.where(
            spacelike_mask,
            -penalty.to(delta_t.dtype),
            torch.zeros((), dtype=delta_t.dtype, device=delta_t.device),
        )
        mask_matrix = torch.where(causal_mask, float("-inf"), mask_matrix)

        return mask_matrix
