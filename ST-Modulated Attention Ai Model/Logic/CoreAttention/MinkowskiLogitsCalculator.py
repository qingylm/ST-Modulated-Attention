import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MinkowskiLogitsCalculator(nn.Module):
    """
    输入: 已经解耦投影的 Q_s, K_s (空间), Q_t, K_t (时间)
    输出: 未经过掩码的原始闵可夫斯基 Logits
    """

    def __init__(self, d_space, d_time, init_alpha=0.01):
        super().__init__()
        # 可学习的耦合系数，初始值极小，保证训练初期类似标准Attention，稳定收敛
        # 使用 Softplus 确保 alpha 永远 > 0
        self.alpha_param = nn.Parameter(torch.tensor(init_alpha).log())

        # 空间维度的缩放因子（沿用标准Transformer的 1/sqrt(d)）
        self.scale_space = 1.0 / math.sqrt(d_space)
        # 时间维度的缩放因子（单独处理，防止时间维度数值过大）
        self.scale_time = 1.0 / math.sqrt(d_time)

    @property
    def alpha(self):
        # 通过 Softplus 将参数映射到 (0, +∞)
        return F.softplus(self.alpha_param)

    def forward(self, Q_s, K_s, Q_t, K_t):
        """
        Q_s, K_s: [Batch, Heads, Seq, D_space]
        Q_t, K_t: [Batch, Heads, Seq, D_time]
        """
        # 1. 计算空间相似度 (欧氏点积)
        # 形状: [B, H, Seq, Seq]
        sim_space = torch.matmul(Q_s, K_s.transpose(-2, -1)) * self.scale_space

        # 2. 计算时间相似度 (普通点积，但物理意义不同)
        sim_time = torch.matmul(Q_t, K_t.transpose(-2, -1)) * self.scale_time

        # 3. 闵可夫斯基合成：空间相似度 - (alpha * 时间相似度)
        # 这就是将 (iw) 平方为负数的体现
        logits = sim_space - self.alpha * sim_time

        return logits