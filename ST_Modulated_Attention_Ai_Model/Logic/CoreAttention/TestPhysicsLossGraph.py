"""
物理正则化 (P0-3) 回归测试

被守住的不变量:
    PhysicsRegularizationLoss 必须作用在**模型实际使用的坐标**上，
    即 SpacetimeLM 内部 coord_normalizer 的输出。若作用在 coords_raw 上，
    由于 coords_raw 是数据集常量（requires_grad=False），损失不携带计算图，
    对模型零梯度贡献 —— 物理正则形同虚设。

同时守住:
    - LearnableSpacetimeNormalizer.clip_value 必须相对归一化输出幅值留有余量，
      否则 clamp 会冻结 space_scale（见 test_clip_value_has_headroom_for_data_scale）

运行方式:
    python Logic/CoreAttention/TestPhysicsLossGraph.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from Logic.CoreAttention import PhysicsRegularizationLoss
from Logic.SpacetimeTransformer import SpacetimeLM


def _mk_model(vocab=64, d_model=16, layers=1):
    torch.manual_seed(0)
    model = SpacetimeLM(vocab_size=vocab, d_model=d_model, d_space=4, d_time=4,
                        num_heads=2, window_size=64, num_layers=layers, dropout=0.0)
    model.train()
    return model


def _mk_batch(B=2, T=8, vocab=64):
    ids = torch.randint(0, vocab, (B, T))
    coords = torch.zeros(B, T, 4)
    for i in range(T):
        coords[:, i, 0] = i % 2                     # 角色交错 -> 制造类空对
        coords[:, i, 3] = i / (T - 1)
    mask = torch.ones(B, T, dtype=torch.long)
    return ids, coords, mask


# ------------------------------------------------------------------ 核心
def test_forward_returns_coords_norm():
    """SpacetimeLM.forward 必须返回归一化坐标作为第三个值。"""
    model = _mk_model()
    ids, coords, mask = _mk_batch()
    out = model(ids, coords, mask)
    assert len(out) == 3, (
        f"forward 应返回 (logits, attn_weights_list, coords_norm)，实际 {len(out)} 个"
    )
    logits, _, coords_norm = out
    assert coords_norm.shape == coords.shape, (
        f"coords_norm 形状 {tuple(coords_norm.shape)} != {tuple(coords.shape)}"
    )


def test_coords_raw_has_no_graph_but_coords_norm_does():
    """核心: coords_raw 无图、coords_norm 有图 —— 所以损失必须用后者。"""
    model = _mk_model()
    ids, coords, mask = _mk_batch()
    phys = PhysicsRegularizationLoss()

    assert not coords.requires_grad, "测试前提不成立: coords_raw 不应带梯度"
    loss_raw, _ = phys(coords, mask=mask)
    assert not loss_raw.requires_grad, (
        "phys(coords_raw) 竟然带计算图 —— 说明 coords_raw 不再是常量"
    )

    _, _, coords_norm = model(ids, coords, mask)
    assert coords_norm.requires_grad, "coords_norm 丢失了计算图"
    loss_norm, _ = phys(coords_norm, mask=mask)
    assert loss_norm.requires_grad, (
        "phys(coords_norm) 不带计算图 —— 物理正则将无法训练模型"
    )


def test_physics_loss_reaches_normalizer():
    """phys(coords_norm) 必须对归一化层产生非零梯度（time_* 与 space_shift）。"""
    model = _mk_model()
    ids, coords, mask = _mk_batch()
    phys = PhysicsRegularizationLoss()

    _, _, coords_norm = model(ids, coords, mask)
    loss, _ = phys(coords_norm, mask=mask)
    loss.backward()

    norm = model.coord_normalizer
    for name, p in norm.named_parameters():
        assert p.grad is not None, f"归一化层 {name} 未获得梯度"
        assert torch.isfinite(p.grad).all(), f"归一化层 {name} 梯度非有限"

    # 至少 time_scale / time_shift / space_shift 必须有非零梯度
    for name in ["time_scale", "time_shift", "space_shift"]:
        g = dict(norm.named_parameters())[name].grad
        assert float(g.abs().sum()) > 0, f"归一化层 {name} 梯度恒为 0"


def test_space_scale_is_not_frozen_by_clamp():
    """
    已修复的问题（原 test_space_scale_is_frozen_by_clamp）:
        clip_value=2.0 时空间坐标在初始化即饱和（x/RMS 最大 2.667 > 2.0），
        clamp 在边界外梯度为 0，space_scale 被永久冻结（梯度恰为 0）。
        放宽到 4.0 后初始化不饱和，space_scale 可以正常学习。

    这条断言锁定"space_scale 必须收到非零梯度"这一修复结果。
    """
    model = _mk_model()
    ids, coords, mask = _mk_batch()
    phys = PhysicsRegularizationLoss()

    _, _, coords_norm = model(ids, coords, mask)
    loss, _ = phys(coords_norm, mask=mask)
    loss.backward()

    norm = model.coord_normalizer

    # 初始化不应饱和
    with torch.no_grad():
        saturated_frac = float(
            (coords_norm.detach().abs() >= norm.clip_value - 1e-6).float().mean()
        )
    assert saturated_frac == 0.0, (
        f"空间坐标在初始化就饱和了（{saturated_frac:.4f} 比例贴到 "
        f"clip_value={norm.clip_value}），space_scale 会被 clamp 冻结"
    )

    # 因此 space_scale 必须拿到非零梯度
    g = norm.space_scale.grad
    assert g is not None, "space_scale 未获得梯度"
    assert float(g.abs().sum()) > 0, (
        f"space_scale 梯度恒为 0 —— 又回到了被 clamp 冻结的状态 "
        f"(clip_value={norm.clip_value})"
    )


def test_clip_value_has_headroom_for_data_scale():
    """
    clip_value 必须显著大于归一化输出的自然幅值，否则 clamp 会变成事实上的
    硬约束而非安全阀。这里用生产级坐标（角色 x∈{0,1}）验证有余量。
    """
    from Logic.SubsequentProcessing import LearnableSpacetimeNormalizer

    norm = LearnableSpacetimeNormalizer()
    T = 64
    coords = torch.zeros(2, T, 4)
    bounds = [0, 20, 35, 47, 56, 61, 64]
    for m in range(len(bounds) - 1):
        lo, hi = bounds[m], bounds[m + 1]
        coords[:, lo:hi, 0] = m % 2
        coords[:, lo:hi, 3] = m / (len(bounds) - 2)

    with torch.no_grad():
        out = norm(coords)
        peak = float(out[..., :3].abs().max())

    assert peak < norm.clip_value, (
        f"归一化后空间峰值 {peak:.4f} 已触及 clip_value={norm.clip_value}，"
        f"space_scale 将无法学习"
    )
    # 至少 30% 余量，给 space_scale 留出学习空间
    assert peak <= norm.clip_value * 0.8, (
        f"clip_value={norm.clip_value} 相对峰值 {peak:.4f} 余量不足 20%"
    )


def test_physics_loss_changes_parameters_after_optimizer_step():
    """端到端: 只优化物理损失时，归一化层参数必须真的被更新。"""
    model = _mk_model()
    ids, coords, mask = _mk_batch()
    phys = PhysicsRegularizationLoss()

    before = {n: p.detach().clone()
              for n, p in model.coord_normalizer.named_parameters()}
    opt = torch.optim.SGD(model.coord_normalizer.parameters(), lr=0.1)

    for _ in range(5):
        _, _, coords_norm = model(ids, coords, mask)
        loss, _ = phys(coords_norm, mask=mask)
        opt.zero_grad()
        loss.backward()
        opt.step()

    changed = [n for n, p in model.coord_normalizer.named_parameters()
               if not torch.equal(p.detach(), before[n])]
    assert changed, "5 步优化后归一化层参数丝毫未变 —— 物理损失仍未接上计算图"


def test_full_loss_backward_is_finite():
    """交叉熵 + 物理损失联合反传，梯度必须有限（且 K/V 层也有梯度）。"""
    model = _mk_model()
    ids, coords, mask = _mk_batch()
    phys = PhysicsRegularizationLoss()

    logits, _, coords_norm = model(ids, coords, mask)
    ce = torch.nn.functional.cross_entropy(
        logits[..., :-1, :].reshape(-1, logits.size(-1)),
        ids[..., 1:].reshape(-1),
    )
    pl, _ = phys(coords_norm, mask=mask)
    (ce + pl).backward()

    for name in ["W_q_s", "W_k_s", "W_q_t", "W_k_t", "W_v"]:
        g = getattr(model.blocks[0].attn, name).weight.grad
        assert g is not None, f"{name} 无梯度"
        assert torch.isfinite(g).all(), f"{name} 梯度非有限"


def test_physics_loss_mask_dtype_robustness():
    """mask 为整型/bool 时物理损失应可正常计算。"""
    phys = PhysicsRegularizationLoss()
    coords = torch.randn(2, 6, 4)
    for dt in (torch.long, torch.int32, torch.bool):
        loss, _ = phys(coords, mask=torch.ones(2, 6, dtype=dt))
        assert torch.isfinite(loss), f"mask dtype={dt} 产生非有限损失"


def _main():
    tests = [
        test_forward_returns_coords_norm,
        test_coords_raw_has_no_graph_but_coords_norm_does,
        test_physics_loss_reaches_normalizer,
        test_space_scale_is_not_frozen_by_clamp,
        test_clip_value_has_headroom_for_data_scale,
        test_physics_loss_changes_parameters_after_optimizer_step,
        test_full_loss_backward_is_finite,
        test_physics_loss_mask_dtype_robustness,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {fn.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS  {fn.__name__}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
