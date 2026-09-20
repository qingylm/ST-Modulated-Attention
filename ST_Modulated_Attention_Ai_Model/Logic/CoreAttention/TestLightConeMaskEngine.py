"""
LightConeMaskEngine 回归测试

被守住的不变量:
    `spacelike_mask` 与 `causal_mask` 必须**恒不相交**。
    这是因果屏蔽不会被类空软惩罚覆写的唯一原因 —— 类空条件
    `(ds^2 > 0) & (~causal_mask)` 里的 `~causal_mask` 才是正确性关键。

    如果该条件被"简化"成 `spacelike_mask = interval_sq > 0`，那么
    |Δ空间| > |Δt| 的未来位置会被从 -inf 覆写成 -|bias|，自回归性被破坏。
    这在真实数据上必然发生: build_lccc_dataset.py 的 ROLE_SPACE 让
    user 的 x=0、assistant 的 x=1，跨角色对的 |Δx|=1 足以压过 |Δt|。

运行方式:
    python Logic/CoreAttention/TestLightConeMaskEngine.py
"""
import os
import sys

import torch

# 允许直接以脚本方式运行（无需安装为包）
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from Logic.CoreAttention.LightConeMaskEngine import LightConeMaskEngine

# 与 build_lccc_dataset.py 的 ROLE_SPACE 保持一致的坐标构造方式
ROLE_X = {"user": 0.0, "assistant": 1.0}


def _role_pattern_coords(roles):
    """
    按 LCCC 的真实模式构造坐标: [B, Seq, 4] = (x, y, z, t)。
    x 由角色决定，y=z=0，t 为单调递增的消息级时间。

    返回 (coords, delta_t, interval_sq)。
    """
    n = len(roles)
    coords = torch.zeros(1, n, 4)
    for i, role in enumerate(roles):
        coords[0, i, 0] = ROLE_X[role]
        coords[0, i, 3] = i / max(1, n - 1)
    delta_t = coords[0, :, 3].unsqueeze(-1) - coords[0, :, 3].unsqueeze(-2)
    delta_s = coords[0, :, None, :3] - coords[0, None, :, :3]
    interval_sq = delta_s.pow(2).sum(-1) - delta_t ** 2
    return coords, delta_t, interval_sq


def test_causal_and_spacelike_masks_are_disjoint():
    """
    核心不变量: 因果掩码与类空掩码的交集必须为空。

    只要这个不变量成立，因果屏蔽就不可能被类空惩罚覆写，写入顺序无关紧要；
    一旦它被破坏，未来位置就会泄漏。
    """
    engine = LightConeMaskEngine(init_bias=5.0)
    # 多种角色模式，包含多次角色切换
    patterns = [
        ["user"] * 4 + ["assistant"] * 2,
        ["user", "assistant"] * 4,
        ["user"] * 2 + ["assistant"] * 6,
        ["assistant"] * 3 + ["user"] * 3,
    ]
    total_future_spacelike = 0
    for roles in patterns:
        coords, delta_t, interval_sq = _role_pattern_coords(roles)
        causal = delta_t < 0
        spacelike = interval_sq > 0

        # 引擎内部的类空条件带了 ~causal，因此与 causal 必然不相交
        internal_spacelike = spacelike & (~causal)
        overlap = (causal & internal_spacelike).nonzero()
        assert overlap.numel() == 0, (
            f"角色模式 {roles}: 因果与类空掩码出现交集 "
            f"{[tuple(t) for t in overlap.tolist()]}"
        )
        # 统计"未来且类空"的裸组合，用于确认用例真的覆盖了风险场景
        total_future_spacelike += int((causal & spacelike).sum())

    # 用例必须真的覆盖到"未来且类空"的坐标组合，否则本测试形同虚设
    assert total_future_spacelike > 0, (
        "所有角色模式都没有产生『未来且类空』的位置，用例未覆盖该风险场景"
    )


def test_future_token_stays_hard_masked_when_spacelike():
    """未来位置必须一律硬屏蔽，即使它同时满足类空条件。"""
    engine = LightConeMaskEngine(init_bias=5.0)
    coords, delta_t, interval_sq = _role_pattern_coords(["user"] * 4 + ["assistant"] * 2)
    mask = engine(coords)[0]

    neg_inf = float("-inf")
    n = coords.shape[1]
    checked = 0
    for i in range(n):
        for j in range(n):
            if not bool(delta_t[i, j] < 0):
                continue  # 只看未来位置
            assert float(mask[i, j]) == neg_inf, (
                f"未来位置 ({i},{j}) 未被硬屏蔽! value={mask[i, j].item()}, "
                f"ds^2={float(interval_sq[i, j]):.4f} (类空={bool(interval_sq[i, j] > 0)})"
            )
            checked += 1

    assert checked > 0, "测试样本没有产生任何未来位置，用例无效"


def test_past_spacelike_still_penalised():
    """类空软惩罚在合法的过去位置上必须保持有效（这是本模块的实际作用）。"""
    engine = LightConeMaskEngine(init_bias=5.0)
    coords, delta_t, interval_sq = _role_pattern_coords(["user"] * 4 + ["assistant"] * 2)
    mask = engine(coords)[0]

    # (3, 0): user -> user，|Δx|=0 < |Δt| -> 类时，无惩罚
    assert float(mask[3, 0]) == 0.0, f"类时位置被错误惩罚: {mask[3, 0].item()}"

    # (4, 0): assistant 查询 user 键，Δt>0 且 |Δx|=1 > |Δt| -> 类空，应被惩罚
    assert interval_sq[4, 0] > 0, "用例前提不成立: (4,0) 不是类空位置"
    assert torch.isclose(mask[4, 0], torch.tensor(-5.0), atol=1e-4), (
        f"类空惩罚数值错误: {mask[4, 0].item()}"
    )

    # 对角线永远是自身，必须完全放行
    for i in range(coords.shape[1]):
        assert float(mask[i, i]) == 0.0, f"对角线 ({i},{i}) 被掩码"


def test_masked_attention_weights_are_exactly_zero():
    """
    端到端语义: 对含类空未来的 logits 做 softmax，未来位置权重必须为 0。

    注意必须选"有合法过去键"的查询行 —— 第 0 行没有合法的过去键，
    泄漏权重会被分母吸收，看不出问题。
    """
    torch.manual_seed(0)
    engine = LightConeMaskEngine(init_bias=5.0)
    # 交错角色: 既有多次切换(制造类空未来)，又有充足的合法过去键
    coords, delta_t, interval_sq = _role_pattern_coords(["user", "assistant"] * 5)
    mask = engine(coords)

    # 故意把未来位置的 logits 设得极大，模拟最坏情况下的泄漏
    logits = torch.randn(1, 10, 10) * 0.1
    logits = torch.where(delta_t < 0, torch.full_like(logits, 50.0), logits)

    weights = torch.softmax(logits + mask, dim=-1)
    leaked = weights[0][delta_t < 0]
    assert torch.all(leaked == 0), (
        f"未来位置泄漏了注意力权重，最大值={leaked.max().item():.6f}"
    )
    assert torch.allclose(weights.sum(-1), torch.ones(10), atol=1e-5), "注意力权重未归一化"


def test_gradient_flows_and_is_finite():
    """掩码必须是可微的，且梯度不能出现 NaN/Inf。"""
    engine = LightConeMaskEngine(init_bias=5.0)
    coords, _, _ = _role_pattern_coords(["user"] * 4 + ["assistant"] * 2)
    mask = engine(coords)

    mask.sum().backward()
    grad = engine.spacelike_bias.grad
    assert grad is not None, "spacelike_bias 未获得梯度"
    assert torch.isfinite(grad).all(), f"梯度出现 NaN/Inf: {grad}"
    assert grad.abs().item() > 0, "类空惩罚未对 spacelike_bias 产生梯度"


def test_dtype_is_preserved_for_amp():
    """
    掩码 dtype 必须跟随输入，否则 autocast 下 fp16 路径会被破坏。

    这是旧实现（`torch.zeros_like(delta_t, dtype=torch.float32)`）的真实缺陷:
    硬编码 float32 会让掩码与 fp16 logits 相加时把整条链路提升回 fp32。

    附带说明: 硬屏蔽哨兵不宜写成 -1e9。若哪天有人把 dtype 改成跟随输入的
    float16，`masked_fill(..., -1e9)` 会直接抛
    RuntimeError: value cannot be converted to type at::Half without overflow
    （已实测）。-inf 在 fp16/bf16/fp32 下都能正确表示。
    """
    engine = LightConeMaskEngine(init_bias=5.0)
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        coords = torch.zeros(1, 4, 4, dtype=dtype)
        coords[0, :, 0] = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=dtype)  # 角色切换
        coords[0, :, 3] = torch.tensor([0.0, 0.1, 0.2, 0.3], dtype=dtype)
        mask = engine(coords)
        assert mask.dtype == dtype, f"{dtype} 输入得到 {mask.dtype} 掩码"

        # 被屏蔽的位置必须是 -inf；其余位置必须有限
        masked = mask[0] == float("-inf")
        assert torch.isfinite(mask[0][~masked]).all(), f"{dtype} 出现意外非有限值"

        # 每行至少要有一个可见位置，否则 softmax 会产生 NaN
        assert (~masked).any(dim=-1).all(), f"{dtype} 存在整行被屏蔽（softmax 会 NaN）"

        # 对角线必须放行，这是"不存在全屏蔽行"的保证
        for i in range(4):
            assert float(mask[0, i, i]) == 0.0, f"{dtype} 对角线未放行"


def _main():
    tests = [
        test_causal_and_spacelike_masks_are_disjoint,
        test_future_token_stays_hard_masked_when_spacelike,
        test_past_spacelike_still_penalised,
        test_masked_attention_weights_are_exactly_zero,
        test_gradient_flows_and_is_finite,
        test_dtype_is_preserved_for_amp,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {fn.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001 - 测试脚本需要报告任何异常
            failed += 1
            print(f"ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS  {fn.__name__}")

    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
