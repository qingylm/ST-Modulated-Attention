import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsRegularizationLoss(nn.Module):
    """
    物理正则化损失：用于约束模型学习到的时空坐标满足物理先验
    """

    def __init__(self, lambda_causal=0.1, lambda_time=0.01, lambda_norm=0.001):
        """
        Args:
            lambda_causal: 类空惩罚系数 (通常设得较大，如 0.1 ~ 0.5)
            lambda_time: 时间利用率惩罚系数 (鼓励时间发散)
            lambda_norm: 坐标范数惩罚系数 (防止数值爆炸)
        """
        super().__init__()
        self.lambda_causal = lambda_causal
        self.lambda_time = lambda_time
        self.lambda_norm = lambda_norm

    def forward(self, coords, mask=None):
        """
        Args:
            coords: [Batch, Seq_len, 4] 即 (x, y, z, t)
            mask:   [Batch, Seq_len] 有效Token掩码 (1=有效, 0=填充)，可选
        Returns:
            total_loss: 标量张量
            loss_dict:  包含各项分量的字典 (用于日志监控)
        """
        # ------------------------------------------------------------
        # 1. 提取坐标并计算两两时空间隔 Δs^2
        # ------------------------------------------------------------
        x, y, z, t = coords[..., 0], coords[..., 1], coords[..., 2], coords[..., 3]

        # 计算差值矩阵 [B, Seq, Seq]
        delta_t = t.unsqueeze(-1) - t.unsqueeze(-2)
        delta_x = x.unsqueeze(-1) - x.unsqueeze(-2)
        delta_y = y.unsqueeze(-1) - y.unsqueeze(-2)
        delta_z = z.unsqueeze(-1) - z.unsqueeze(-2)

        # 闵可夫斯基模长平方: ds^2 = dx^2+dy^2+dz^2 - dt^2
        interval_sq = (delta_x ** 2 + delta_y ** 2 + delta_z ** 2) - delta_t ** 2  # [B, Seq, Seq]

        # 构造有效的掩码矩阵 (忽略填充Token的对齐)
        if mask is not None:
            # mask: [B, Seq] -> 扩展为 [B, Seq, Seq]
            valid_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)  # 两两都为有效Token才计数
            # 忽略对角线 (i==i 时 interval_sq = 0，无意义，不参与损失)
            diag_mask = ~torch.eye(interval_sq.size(-1), device=interval_sq.device).bool()
            valid_mask = valid_mask & diag_mask.unsqueeze(0)
        else:
            # 无掩码时，仅忽略对角线
            diag_mask = ~torch.eye(interval_sq.size(-1), device=interval_sq.device).bool()
            valid_mask = diag_mask.unsqueeze(0)  # [1, Seq, Seq]

        # ------------------------------------------------------------
        # 2. 损失项 1: 因果惩罚 (类空抑制)
        #    正确公式: max(0, ds^2)，即只惩罚 ds^2 > 0 (类空) 的部分
        # ------------------------------------------------------------
        causal_penalty = F.relu(interval_sq)  # 等价于 max(0, interval_sq)
        # 应用掩码并求均值 (除以有效元素个数)
        num_valid = valid_mask.sum() + 1e-8  # 防止除零
        loss_causal = (causal_penalty * valid_mask).sum() / num_valid

        # ------------------------------------------------------------
        # 3. 损失项 2: 时间利用率惩罚
        #    鼓励模型的时间维度 t 在不同Token间有区分度 (方差大)
        #    公式: -Var(t)，越负表示方差越大，取负号后梯度会推动方差增大
        #    为了防止单条序列内方差波动太大，在 Batch 和 Seq 维度上计算总体方差
        # ------------------------------------------------------------
        if mask is not None:
            # 只统计有效Token的时间值
            valid_t = t * mask
            # 计算有效Token的方差 (无偏估计)
            var_t = torch.var(valid_t, dim=1, unbiased=False, keepdim=False)
        else:
            var_t = torch.var(t, dim=1, unbiased=False)

        # 取负号：模型为了最小化损失，会努力让方差变大
        loss_time = -var_t.mean()  # 标量

        # ------------------------------------------------------------
        # 4. 损失项 3: 坐标范数惩罚 (L2 正则)
        #    防止 x,y,z,t 数值无限增长
        # ------------------------------------------------------------
        loss_norm = (coords ** 2).mean()

        # ------------------------------------------------------------
        # 5. 加权求和
        # ------------------------------------------------------------
        total_loss = (self.lambda_causal * loss_causal +
                      self.lambda_time * loss_time +
                      self.lambda_norm * loss_norm)

        # 返回总损失和一个用于监控的字典
        loss_dict = {
            'loss_causal': loss_causal.item(),
            'loss_time': loss_time.item(),
            'loss_norm': loss_norm.item(),
            'total_physics': total_loss.item()
        }

        return total_loss, loss_dict