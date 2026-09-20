import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableSpacetimeNormalizer(nn.Module):
    """
    时空坐标归一化。

    clip_value 的选取很关键:
        它既是数值安全阀，也决定了空间的 RMS 归一化输出有多少"余量"。
        空间部分经 global_rms 归一化后 RMS 恒为 1，但**最大幅值取决于数据**
        —— 例如角色坐标 x∈{0,1}、y=z=0 时，RMS=0.375，x/RMS 最大达 2.667。
        若 clip_value 取 2.0，坐标在初始化时就贴到裁剪边界，而 clamp 在边界外
        梯度恒为 0，space_scale 会被永久冻结（实测梯度恰为 0）。
        取 4.0 时初始化不饱和，space_scale 可以正常学习（实测 1.0 -> 0.85）。
        调小该值前请确认归一化输出的实际幅值。
    """

    def __init__(self,
                 space_norm_type='global_rms',  # 跨序列统计，避免逐Token归一化抹平几何
                 time_norm_type='minmax',
                 clip_value=4.0,
                 eps=1e-8):
        super().__init__()
        self.eps = eps
        self.clip_value = clip_value

        # 空间维度：跨序列的缩放因子（标量，而非每个Token独立）
        self.space_scale = nn.Parameter(torch.ones(1))
        self.space_shift = nn.Parameter(torch.zeros(1))

        # 时间维度保持不变
        self.time_scale = nn.Parameter(torch.ones(1, 1, 1))
        self.time_shift = nn.Parameter(torch.zeros(1, 1, 1))

        self.space_norm_type = space_norm_type
        self.time_norm_type = time_norm_type

    def forward(self, coords, mask=None):
        x, y, z, t = coords[..., 0], coords[..., 1], coords[..., 2], coords[..., 3]
        space_part = torch.stack([x, y, z], dim=-1)  # [B, S, 3]

        # 【关键修改】空间归一化：跨 Seq 维度 (dim=1) 计算全局统计量
        if self.space_norm_type == 'global_rms':
            # 对整个序列的所有 Token 计算 RMS（忽略 Batch 和 3维特征）
            # 形状: [B, 1, 1]
            rms = torch.sqrt(torch.mean(space_part ** 2, dim=(1, 2), keepdim=True) + self.eps)
            space_normed = space_part / rms  # 所有 Token 共享同一个缩放因子
        elif self.space_norm_type == 'global_std':
            # 跨序列标准化
            mean = torch.mean(space_part, dim=(1, 2), keepdim=True)
            std = torch.std(space_part, dim=(1, 2), keepdim=True, unbiased=False) + self.eps
            space_normed = (space_part - mean) / std
        else:
            space_normed = space_part

        # 应用可学习缩放（此时标量乘以整个 Batch）
        space_normed = space_normed * self.space_scale + self.space_shift

        # 安全裁剪
        space_normed = torch.clamp(space_normed, -self.clip_value, self.clip_value)

        # 时间归一化（保持不变，依然按序列独立 Min-Max）
        t = t.unsqueeze(-1)
        if self.time_norm_type == 'minmax':
            t_min = torch.min(t, dim=1, keepdim=True)[0]
            t_max = torch.max(t, dim=1, keepdim=True)[0]
            t_range = (t_max - t_min) + self.eps
            t_normed = (t - t_min) / t_range
        else:
            t_normed = t
        t_normed = t_normed * self.time_scale + self.time_shift
        t_normed = torch.clamp(t_normed, -self.clip_value, self.clip_value)

        return torch.cat([space_normed, t_normed], dim=-1)


