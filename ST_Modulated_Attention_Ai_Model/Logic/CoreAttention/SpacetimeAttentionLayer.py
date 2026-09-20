from .LightConeMaskEngine import LightConeMaskEngine
from .MinkowskiLogitsCalculator import MinkowskiLogitsCalculator
import torch.nn as nn
import torch.nn.functional as F

class SpacetimeAttentionLayer(nn.Module):
    def __init__(self, d_space, d_time):
        super().__init__()
        self.logits_calc = MinkowskiLogitsCalculator(d_space, d_time)
        self.mask_engine = LightConeMaskEngine()

    def forward(self, Q_s, K_s, Q_t, K_t, coords):
        # 1. 计算原始闵可夫斯基分数
        raw_logits = self.logits_calc(Q_s, K_s, Q_t, K_t)

        # 2. 生成物理掩码矩阵
        # 注意：需要将 coords 扩展 Head 维度 (因为 mask 与 Head 无关)
        mask = self.mask_engine(coords)  # [B, Seq, Seq]
        mask = mask.unsqueeze(1)  # [B, 1, Seq, Seq] 广播给所有 Heads

        # 3. 叠加掩码
        masked_logits = raw_logits + mask

        # 4. 标准 Softmax 和 加权 V (V 仍使用标准语义投影，不包含坐标)
        attn_weights = F.softmax(masked_logits, dim=-1)
        # ... 乘以 V 得到最终输出

        return attn_weights



