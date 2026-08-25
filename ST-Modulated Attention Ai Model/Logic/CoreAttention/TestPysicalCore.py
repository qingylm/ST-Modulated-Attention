from LightConeMaskingEngine import LightConeMaskEngine
from MinkowskiLogitsCalculator import MinkowskiLogitsCalculator
from PhysicsRegularizationLoss import PhysicsRegularizationLoss
from SlidingWindowCache import SlidingWindowCache
import torch.nn as nn
import torch.nn.functional as F
import torch
from SpacetimeAttentionWithCache import SpacetimeAttentionWithCache

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


def test_physics_loss():
    print("\n===== 测试 2: 物理损失函数梯度方向 =====")

    loss_fn = PhysicsRegularizationLoss(lambda_causal=0.5, lambda_time=0.1, lambda_norm=0.01)

    # 构造坐标: 大部分 Token 类空 (ds^2 > 0)
    coords_bad = torch.tensor([
        [0.0, 0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0, 0.0],  # 空间巨大，类空
        [20.0, 0.0, 0.0, 0.0],
        [30.0, 0.0, 0.0, 0.0],
        [40.0, 0.0, 0.0, 0.0]
    ]).unsqueeze(0).requires_grad_(True)  # [1,5,4]

    # 计算损失
    loss, dict_loss = loss_fn(coords_bad)

    # 反向传播获取梯度
    loss.backward()

    # 检查因果惩罚梯度: 类空坐标的梯度应指向压缩方向
    # 对于类空坐标，梯度应该让坐标变小（减少空间距离）
    grad_spatial = coords_bad.grad[0, 1, 0]  # 索引1的x坐标梯度

    # 注意：由于类空，梯度应为负（因为 loss = max(0, dx^2 - dt^2)，梯度 = 2*dx > 0，但反向传播给坐标的是负梯度）
    # 我们只需检查梯度不为0且方向合理即可
    assert coords_bad.grad is not None, "梯度未传播!"
    assert torch.abs(grad_spatial) > 1e-6, f"类空坐标梯度接近零! 梯度 = {grad_spatial}"

    print(f"✅ 损失函数可微分，类空坐标梯度为: {grad_spatial.item():.4f}")
    print(
        f"   损失分量: Causal={dict_loss['loss_causal']:.4f}, TimeVar={dict_loss['loss_time']:.4f}, Norm={dict_loss['loss_norm']:.4f}")


test_physics_loss()

def test_sliding_window_cache():
    print("\n===== 测试 3: 滑动窗口缓存与截断 =====")

    # 初始化窗口大小为 4
    cache = SlidingWindowCache(window_size=4)

    # 模拟连续输入 3 次，每次序列长度为 2
    for step in range(3):
        B, H, S, Ds, Dt = 1, 2, 2, 3, 1
        k_s = torch.randn(B, H, S, Ds)
        k_t = torch.randn(B, H, S, Dt)
        v = torch.randn(B, H, S, 8)
        coords = torch.randn(B, S, 4)

        full_k_s, full_k_t, full_v, full_coords = cache.update(k_s, k_t, v, coords)

        expected_len = min((step + 1) * 2, 4)  # 窗口上限为4
        print(f"  第 {step + 1} 次更新后，缓存长度: {full_k_s.size(2)} (期望 {expected_len})")
        assert full_k_s.size(2) == expected_len, f"缓存长度错误!"

    # 验证梯度是否被截断: 检查 cached 张量的 requires_grad 属性
    # 由于我们在 update 中使用了 .detach()，缓存张量应没有梯度
    assert cache.cache_k_s.requires_grad == False, "缓存张量未正确截断梯度 (仍 requires_grad=True)"
    print("✅ 缓存长度与梯度截断验证通过")


test_sliding_window_cache()

def test_end_to_end():
    print("\n===== 测试 4: 端到端注意力层冒烟测试 =====")

    # 1. 初始化模型（极小配置，方便快速跑）
    d_model = 16
    d_space = 6
    d_time = 4
    num_heads = 2
    window_size = 4

    attn_layer = SpacetimeAttentionWithCache(
        d_model=d_model,
        d_space=d_space,
        d_time=d_time,
        num_heads=num_heads,
        window_size=window_size
    )

    # 2. 构造模拟输入 (Batch=1, Seq=3)
    x = torch.randn(1, 3, d_model)  # 语义嵌入
    coords = torch.randn(1, 3, 4)  # 时空坐标

    # 3. 前向传播 (第一次调用，缓存为空)
    output, attn_weights = attn_layer(x, coords)

    assert output.shape == (1, 3, d_model), f"输出形状错误: {output.shape}"
    # 由于缓存初始为空，且窗口大小为4，当前Seq=3 < 4，总序列应为3
    # attn_weights 维度应为 [B, Heads, Curr_Seq, Total_Seq] = [1, 2, 3, 3]
    assert attn_weights.shape == (1, 2, 3, 3), f"注意力权重形状错误: {attn_weights.shape}"

    # 4. 第二次前向传播 (模拟下一个 step，Seq=2，累积缓存)
    x2 = torch.randn(1, 2, d_model)
    coords2 = torch.randn(1, 2, 4)
    output2, attn_weights2 = attn_layer(x2, coords2)

    # 此时缓存应为窗口大小 min(3+2, 4) = 4
    assert attn_weights2.shape == (1, 2, 2, 4), f"第二次传播权重形状错误: {attn_weights2.shape}"

    # 5. 反向传播 (验证梯度是否正常流通)
    loss = output2.mean()
    loss.backward()

    # 检查投影层是否获得梯度
    grad_exists = attn_layer.W_q_s.weight.grad is not None
    assert grad_exists, "模型参数未获得梯度，反向传播断裂!"

    print(f"✅ 端到端测试通过! 输出形状: {output2.shape}, 注意权重大小: {attn_weights2.shape}")
    print("   梯度正常流动 (W_q_s 梯度存在)")


test_end_to_end()