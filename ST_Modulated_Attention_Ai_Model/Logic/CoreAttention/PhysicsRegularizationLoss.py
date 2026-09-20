import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsRegularizationLoss(nn.Module):
    """
    物理正则化损失：用于约束模型学习到的时空坐标满足物理先验

    由三项组成:
        1. 因果/类空抑制: max(0, ds²)  —— 只惩罚 ds²>0 的类空对
        2. 时间利用率:    -Var(t)      —— 鼓励时间在不同 token 间有区分度
        3. 坐标范数:      mean(coords²) —— 防止数值爆炸

    mask 的 dtype 是安全的: 内部统一转成计算 dtype 后再相乘，不再依赖
    `&` 位运算，因此传 long / int8 / bool / float 掩码都能正常工作
    （此前 float 掩码会抛 `bitwise_and_cpu not implemented for 'Float'`）。
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

    @staticmethod
    def _as_valid_float(mask, coords):
        """
        把任意 dtype 的 mask 归一化为 [B, S] 的 0/1 浮点掩码。

        - 非张量输入: 转成张量（避免 list 直接参与运算）
        - bool 直接用；其余 dtype 先转 bool 再转计算 dtype
          （先转 bool 再转 float 是必要的: 若直接按原 dtype 转换，
           float 掩码里的 2.0 之类取值会变成权重而不是有效性指示）
        - dtype/device 跟随 coords，保证 AMP 下不会出现隐式升精度
        """
        if not isinstance(mask, torch.Tensor):
            mask = torch.as_tensor(mask, device=coords.device)
        if mask.dtype != torch.bool:
            mask = mask != 0
        return mask.to(device=coords.device, dtype=coords.dtype)

    def forward(self, coords, mask=None):
        """
        Args:
            coords: [Batch, Seq_len, 4] 即 (x, y, z, t)
            mask:   [Batch, Seq_len] 有效Token掩码 (非0=有效, 0=填充)，可选。
                    dtype 不限 (long/int/bool/float 均可)。
        Returns:
            total_loss: 标量张量
            loss_dict:  包含各项分量的字典 (用于日志监控，均为 python float)
        """
        # ------------------------------------------------------------
        # 0. 掩码归一化（防御: 任意 dtype / 非张量输入都能安全处理）
        # ------------------------------------------------------------
        valid = None
        if mask is not None:
            valid = self._as_valid_float(mask, coords)          # [B, S]
            if valid.dim() != 2:
                raise ValueError(
                    f"mask 期望形状 [Batch, Seq]，实际 {tuple(mask.shape)}"
                )
            if valid.shape != coords.shape[:2]:
                raise ValueError(
                    f"mask 形状 {tuple(valid.shape)} 与 coords 前两维 "
                    f"{tuple(coords.shape[:2])} 不一致"
                )

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

        seq_len = interval_sq.size(-1)
        # 忽略对角线 (i==i 时 interval_sq = 0，无意义，不参与损失)
        diag_mask = ~torch.eye(seq_len, dtype=torch.bool, device=interval_sq.device)
        diag_mask = diag_mask.unsqueeze(0)                      # [1, S, S]

        # 构造有效性权重 [B, S, S]。用乘法而非 `&`：
        # 既避免 dtype 限制，也天然支持非 0/1 的掩码取值。
        if valid is not None:
            pair_weight = valid.unsqueeze(-1) * valid.unsqueeze(-2)
            pair_weight = pair_weight * diag_mask.to(pair_weight.dtype)
        else:
            pair_weight = diag_mask.to(interval_sq.dtype).expand_as(interval_sq)

        # ------------------------------------------------------------
        # 2. 损失项 1: 因果惩罚 (类空抑制)
        #    正确公式: max(0, ds^2)，即只惩罚 ds^2 > 0 (类空) 的部分
        # ------------------------------------------------------------
        causal_penalty = F.relu(interval_sq)  # 等价于 max(0, interval_sq)

        num_valid = pair_weight.sum()
        if float(num_valid.detach()) <= 0.0:
            # 整个 batch 没有任何有效 token 对（例如全 padding）。
            # 此时任意"平均"都是无定义的，返回 0 而不是 NaN 或伪值。
            # sum() 保留在计算图中，反向传播不会报错。
            loss_causal = (causal_penalty * pair_weight).sum() * 0.0
        else:
            loss_causal = (causal_penalty * pair_weight).sum() / num_valid

        # ------------------------------------------------------------
        # 3. 损失项 2: 时间利用率惩罚
        #    鼓励模型的时间维度 t 在不同Token间有区分度 (方差大)
        #    公式: -Var(t)，取负号后梯度会推动方差增大
        #
        #    注意: 这里必须用**真正的 masked 统计量**。早期实现写成
        #    `torch.var(t * mask, dim=1)`，把 padding 位置当成 t=0 的真实
        #    取值，方差被系统性拉低（padding 越多拉得越低）。
        # ------------------------------------------------------------
        if valid is not None:
            n_valid = valid.sum(dim=1)                                  # [B]
            t_sum = (t * valid).sum(dim=1)                              # [B]
            t_mean = t_sum / n_valid.clamp(min=1.0)                     # [B]
            # 与均值的偏差同样只在有效位置统计（padding 位置置 0）
            dev = (t - t_mean.unsqueeze(1)) * valid
            var_t = (dev ** 2).sum(dim=1) / n_valid.clamp(min=1.0)
            # 有效 token 少于 2 个时方差无意义，置 0
            var_t = torch.where(n_valid >= 2, var_t, torch.zeros_like(var_t))
        else:
            var_t = torch.var(t, dim=1, unbiased=False)
        loss_time = -var_t.mean()  # 标量

        # ------------------------------------------------------------
        # 4. 损失项 3: 坐标范数惩罚 (L2 正则)
        #    防止 x,y,z,t 数值无限增长；只在有效 token 上统计
        # ------------------------------------------------------------
        if valid is not None:
            # 复用 valid 的逐元素权重，平均到每个坐标分量
            num_valid_entries = valid.sum() * coords.size(-1)
            loss_norm = ((coords ** 2) * valid.unsqueeze(-1)).sum() / \
                num_valid_entries.clamp(min=1.0)
        else:
            loss_norm = (coords ** 2).mean()

        # ------------------------------------------------------------
        # 5. 加权求和
        # ------------------------------------------------------------
        total_loss = (self.lambda_causal * loss_causal +
                      self.lambda_time * loss_time +
                      self.lambda_norm * loss_norm)

        # 返回总损失和一个用于监控的字典。
        # 数值项先 detach 再取标量: 日志取值不应参与反向传播。
        loss_dict = {
            'loss_causal': float(loss_causal.detach()),
            'loss_time': float(loss_time.detach()),
            'loss_norm': float(loss_norm.detach()),
            'total_physics': float(total_loss.detach()),
        }

        return total_loss, loss_dict
