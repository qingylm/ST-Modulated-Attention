from LightConeMaskingEngine import LightConeMaskEngine
from MinkowskiLogitsCalculator import MinkowskiLogitsCalculator
from SlidingWindowCache import SlidingWindowCache
import torch.nn as nn
import torch.nn.functional as F
import torch

# 假设你已将之前的类定义复制到当前环境中
# 如果没复制，请先把前文中的 MinkowskiLogitsCalculator, LightConeMaskEngine 粘贴过来

def test_physical_core():
    print("===== 测试 1: 物理核心逻辑 =====")

    # 1. 测试闵可夫斯基分数计算器
    calc = MinkowskiLogitsCalculator(d_space=4, d_time=2, init_alpha=0.1)

    # 构造模拟数据: Batch=2, Heads=3, Seq=5
    Q_s = torch.randn(2, 3, 5, 4)
    K_s = torch.randn(2, 3, 5, 4)
    Q_t = torch.randn(2, 3, 5, 2)
    K_t = torch.randn(2, 3, 5, 2)

    logits = calc(Q_s, K_s, Q_t, K_t)
    assert logits.shape == (2, 3, 5, 5), f"形状错误，期望 (2,3,5,5)，得到 {logits.shape}"
    print(f"✅ 闵可夫斯基分数形状正确: {logits.shape}")
    print(f"   alpha 初始值: {calc.alpha.item():.4f} (应接近 0.1)")

    # 2. 测试光锥掩码引擎
    mask_engine = LightConeMaskEngine(init_bias=5.0)

    # 构造有物理意义的坐标: 时间 t 从 0 到 4
    coords = torch.tensor([
        [0.0, 0.0, 0.0, 0.0],  # Token 0
        [1.0, 0.0, 0.0, 0.5],  # Token 1 (类空: 空间距离1, 时间差0.5 -> ds^2=0.75>0)
        [0.0, 0.0, 0.0, 2.0],  # Token 2 (类时: 空间距离0, 时间差2 -> ds^2=-4<0)
        [0.0, 0.0, 0.0, 3.0],  # Token 3
        [0.0, 0.0, 0.0, 4.0]  # Token 4
    ]).unsqueeze(0)  # [1, 5, 4]

    mask = mask_engine(coords)  # [1, 5, 5]

    # 验证因果掩码: 未来位置 (delta_t < 0) 应被设为 -1e9
    assert mask[0, 0, 1] < -1e8, f"Token0 看到 Token1 (未来) 未掩蔽! 值为 {mask[0, 0, 1]}"
    assert mask[0, 0, 2] < -1e8, "Token0 看到 Token2 未来未掩蔽!"
    assert mask[0, 4, 0] > -1e8, "Token4 看到 Token0 (过去) 被错误掩蔽!"  # 过去不应掩蔽

    # 验证类空惩罚: Token0 与 Token1 (类空) 应被减去 bias
    # 查看原始分数 (0,0) 对角线是自身忽略; 看 (0,1) 位置
    # 注意: mask[0,0,1] 由于是未来已被掩蔽为 -inf, 所以看 Token1 对 Token0 的关联 (索引 1,0)
    # 即查询是 Token1, 键是 Token0: delta_t = 0.5 - 0 = 0.5 (正), 类空惩罚生效
    assert mask[0, 1, 0] < -4.9, f"类空惩罚未生效! 期望接近 -5.0, 得到 {mask[0, 1, 0]}"

    # 验证类时: Token0 与 Token2 (delta_t=2, ds^2=-4) 不应有惩罚 (值为 0)
    assert mask[0, 2, 0] == 0.0, f"类时事件被错误惩罚! 值为 {mask[0, 2, 0]}"

    print("✅ 光锥掩码逻辑验证通过 (因果掩蔽 + 类空惩罚)")


test_physical_core()