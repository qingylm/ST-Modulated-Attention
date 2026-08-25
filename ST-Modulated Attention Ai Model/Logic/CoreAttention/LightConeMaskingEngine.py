import torch
import torch.nn as nn

class LightConeMaskEngine(nn.Module):
    """
    输入: 序列中每个Token的绝对时空坐标 (x, y, z, t)
    输出: 与 Logits 同维度的掩码矩阵 (加法掩码)
    """

    def __init__(self, init_bias=5.0):
        super().__init__()
        # 类空惩罚偏置，可学习，初始值较大
        self.spacelike_bias = nn.Parameter(torch.tensor(init_bias))

    def forward(self, coords):
        """
        coords: [Batch, Seq, 4]  (即 x, y, z, t)
        注意: 这里的 t 必须是绝对时间（例如对话轮次或挂钟时间归一化）
        """
        # 1. 解构坐标
        x, y, z, t = coords[..., 0], coords[..., 1], coords[..., 2], coords[..., 3]

        # 2. 计算两两之间的差异 (广播机制)
        # [B, Seq, 1] - [B, 1, Seq] -> [B, Seq, Seq]
        delta_t = t.unsqueeze(-1) - t.unsqueeze(-2)
        delta_x = x.unsqueeze(-1) - x.unsqueeze(-2)
        delta_y = y.unsqueeze(-1) - y.unsqueeze(-2)
        delta_z = z.unsqueeze(-1) - z.unsqueeze(-2)

        # 3. 计算时空间隔平方 (ds^2 = dx^2+dy^2+dz^2 - dt^2)
        space_dist_sq = delta_x ** 2 + delta_y ** 2 + delta_z ** 2
        interval_sq = space_dist_sq - delta_t ** 2  # 这就是闵可夫斯基模长

        # 4. 构建因果掩码 (Causal Mask)
        # 如果 delta_t < 0，表示查询 Token 在键 Token 之前（即看到了未来），必须掩蔽
        causal_mask = delta_t < 0

        # 5. 构建类空掩码 (Spacelike Mask)
        # 如果 interval_sq > 0，表示两事件无法通过光速联系
        spacelike_mask = interval_sq > 0

        # 6. 初始化掩码矩阵为 0 (不加任何干扰)
        # 注意这里仅处理 Logits 加法，因此 0 表示不改变原分数
        mask_matrix = torch.zeros_like(delta_t, dtype=torch.float32)

        # 7. 应用因果掩码 (硬掩码，设为负无穷)
        # 为了避免梯度爆炸，用极小的数代替 -inf，但在 Softmax 中效果等同于 -inf
        mask_matrix = mask_matrix.masked_fill(causal_mask, -1e9)

        # 8. 应用类空惩罚 (软掩码)
        # 只对类空位置减去偏置，而对类时位置保留原分数
        # 注意：偏置取绝对值确保惩罚为正
        penalty = torch.abs(self.spacelike_bias)
        mask_matrix = mask_matrix.masked_fill(spacelike_mask, -penalty)

        return mask_matrix