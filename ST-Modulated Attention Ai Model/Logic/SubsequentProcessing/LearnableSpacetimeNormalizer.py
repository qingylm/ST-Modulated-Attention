import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableSpacetimeNormalizer(nn.Module):
    def __init__(self,
                 space_norm_type='global_rms',  # 改为 global_rms
                 time_norm_type='minmax',
                 clip_value=2.0,
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


