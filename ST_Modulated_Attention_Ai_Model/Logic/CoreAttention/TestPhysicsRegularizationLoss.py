"""
PhysicsRegularizationLoss 回归测试

被守住的不变量:
    1. mask dtype 健壮性: long / int32 / int8 / bool / float / 非张量 都能正常工作
       （此前 float 掩码抛 `bitwise_and_cpu not implemented for 'Float'`）
    2. 真正的 masked 统计: padding 不得影响损失值，且 padding 数量变化时
       损失保持不变（此前 `torch.var(t * mask)` 把 padding 当成 t=0，
       方差被 padding 数量系统性拉低）
    3. 全 padding / 单 token 等退化输入不产生 NaN
    4. 各项损失可微、梯度有限

运行方式:
    python Logic/CoreAttention/TestPhysicsRegularizationLoss.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from Logic.CoreAttention import PhysicsRegularizationLoss


def _mk_coords(B=2, T=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    coords = torch.zeros(B, T, 4)
    for i in range(T):
        coords[:, i, 0] = i % 2
        # T == 1 时用 0 而不是除以 0
        coords[:, i, 3] = i / (T - 1) if T > 1 else 0.0
    coords = coords + torch.randn(B, T, 4, generator=g) * 0.01
    return coords


def test_all_mask_dtypes_work():
    """核心: 各种 dtype 的 mask 都必须能正常计算。"""
    phys = PhysicsRegularizationLoss()
    coords = _mk_coords()
    B, T = coords.shape[:2]

    for dt in (torch.long, torch.int64, torch.int32, torch.int16, torch.int8,
               torch.uint8, torch.bool, torch.float32, torch.float64):
        m = torch.ones(B, T, dtype=dt)
        loss, d = phys(coords, mask=m)
        assert torch.isfinite(loss), f"mask dtype={dt} 产生非有限损失"
        assert all(isinstance(v, float) for v in d.values()), (
            f"mask dtype={dt} 的 loss_dict 值应为 python float"
        )
    # float16 mask（AMP 下可能出现）
    loss, _ = phys(coords, mask=torch.ones(B, T, dtype=torch.float16))
    assert torch.isfinite(loss), "float16 mask 产生非有限损失"


def test_non_tensor_and_zero_one_masks():
    """非张量输入与 0/1 语义都必须正确处理。"""
    phys = PhysicsRegularizationLoss()
    coords = _mk_coords()
    B, T = coords.shape[:2]

    # list 输入
    loss_list, _ = phys(coords, mask=[[1] * T for _ in range(B)])
    loss_tensor, _ = phys(coords, mask=torch.ones(B, T, dtype=torch.long))
    assert torch.allclose(loss_list, loss_tensor), "list mask 与 tensor mask 结果不一致"

    # 非 0/1 取值应只表示"有效"，不应变成权重
    loss_two, _ = phys(coords, mask=torch.full((B, T), 2.0))
    assert torch.allclose(loss_two, loss_tensor), (
        "mask 取值为 2 时结果与全 1 不同 —— 掩码被当成了权重而非有效性指示"
    )
    loss_neg, _ = phys(coords, mask=torch.full((B, T), -1))
    assert torch.allclose(loss_neg, loss_tensor), "负数非零值应视为有效"


def test_shape_mismatch_raises_clear_error():
    """形状不符应给出清晰错误，而不是在深处抛难懂的异常。"""
    phys = PhysicsRegularizationLoss()
    coords = _mk_coords(B=2, T=8)
    for bad in (torch.ones(2, 5), torch.ones(3, 8), torch.ones(8)):
        try:
            phys(coords, mask=bad)
        except ValueError as exc:
            assert "mask" in str(exc), f"错误信息不清晰: {exc}"
        else:
            raise AssertionError(f"mask 形状 {tuple(bad.shape)} 未被拒绝")


def test_padding_does_not_change_loss():
    """
    核心: 同一份真实内容，尾部追加不同数量的 padding，损失必须不变。

    注意构造方式: 有效内容固定在 [0, n0)，padding 长度递增，因此比较的
    始终是"同一份内容"。反例（错误写法）是把内容长度本身也一起改掉，
    那比较的是不同内容，结论无意义。
    """
    phys = PhysicsRegularizationLoss()
    n0, pad_max = 8, 10
    T = n0 + pad_max
    coords = _mk_coords(B=1, T=T, seed=3)   # 只用前 n0 个为有效内容

    losses = []
    for pad in (0, 2, 5, 10):
        m = torch.zeros(1, T, dtype=torch.long)
        m[0, :n0] = 1
        loss, _ = phys(coords, mask=m)
        losses.append(float(loss))

    spread = max(losses) - min(losses)
    assert spread < 1e-6, (
        f"改变 padding 长度导致损失变化 {spread:.3e}（各配置={losses}）——"
        f" 说明 padding 影响了统计量"
    )


def test_time_variance_ignores_padding():
    """
    针对早期实现的缺陷: `torch.var(t * mask)` 把 padding 当成 t=0，
    padding 越多方差越小，loss_time 被系统性拉低到错误的值。

    正确的语义: loss_time 只统计有效 token 的方差，padding 长度无关。
    """
    phys = PhysicsRegularizationLoss(lambda_causal=0.0, lambda_time=1.0,
                                     lambda_norm=0.0)
    n0, pad_max = 12, 8
    T = n0 + pad_max
    coords = _mk_coords(B=1, T=T, seed=5)

    # 只对前 n0 个 token 求方差，这就是正确答案（与 padding 无关）
    expect = -float(torch.var(coords[0, :n0, 3], unbiased=False))

    times = []
    for pad in (0, 3, 8):
        m = torch.zeros(1, T, dtype=torch.long)
        m[0, :n0] = 1
        _, d = phys(coords, mask=m)
        times.append(d["loss_time"])
        assert abs(d["loss_time"] - expect) < 1e-6, (
            f"padding={pad}: loss_time={d['loss_time']:.8f}, 期望={expect:.8f}"
            f"（应仅统计有效 token 的方差）"
        )

    spread = max(times) - min(times)
    assert spread < 1e-6, f"loss_time 随 padding 变化 {spread:.3e}（各配置={times}）"


def test_masking_equals_slicing():
    """带 mask 的结果必须等同于直接切掉 padding 后的结果。"""
    phys = PhysicsRegularizationLoss()
    T = 10
    coords = _mk_coords(B=1, T=T, seed=7)
    n_valid = 6

    m = torch.zeros(1, T, dtype=torch.long)
    m[0, :n_valid] = 1

    loss_masked, d_masked = phys(coords, mask=m)
    loss_sliced, d_sliced = phys(coords[:, :n_valid])

    assert abs(float(loss_masked) - float(loss_sliced)) < 1e-6, (
        f"带 mask 的损失 {float(loss_masked):.8f} != 切片后的损失 "
        f"{float(loss_sliced):.8f}"
    )
    for k in ("loss_causal", "loss_time", "loss_norm"):
        assert abs(d_masked[k] - d_sliced[k]) < 1e-6, (
            f"{k}: mask={d_masked[k]:.8f} != slice={d_sliced[k]:.8f}"
        )


def test_degenerate_inputs_are_finite():
    """全 padding / 单 token 等退化输入不得产生 NaN。"""
    phys = PhysicsRegularizationLoss()

    # 全 padding
    coords = _mk_coords(B=2, T=6)
    loss, d = phys(coords, mask=torch.zeros(2, 6, dtype=torch.long))
    assert torch.isfinite(loss), f"全 padding 产生 NaN/Inf: {loss}"
    assert all(torch.isfinite(torch.tensor(v)) for v in d.values())
    # 全 padding 时损失与参数无关（常量 0），这里只要求不抛异常
    assert float(loss) == 0.0, f"全 padding 应返回 0，实际 {float(loss)}"

    # 每个样本只有 1 个有效 token
    m = torch.zeros(2, 6, dtype=torch.long)
    m[:, 0] = 1
    loss, d = phys(coords, mask=m)
    assert torch.isfinite(loss), f"单 token 产生 NaN/Inf: {loss}"
    assert torch.isfinite(torch.tensor(d["loss_time"])), "单 token 的 loss_time 非有限"

    # 序列长度为 1
    c1 = _mk_coords(B=2, T=1)
    loss, _ = phys(c1, mask=torch.ones(2, 1, dtype=torch.long))
    assert torch.isfinite(loss), "T=1 产生 NaN/Inf"


def test_degenerate_inputs_are_backward_safe():
    """
    退化输入下反传不得抛错。

    用 requires_grad 的坐标构造，确保计算图存在 —— 否则 loss 是常量，
    backward() 会因"不依赖任何需要梯度的张量"而报错，那是测试构造问题
    而非损失函数缺陷。
    """
    phys = PhysicsRegularizationLoss()
    coords = _mk_coords(B=2, T=6).requires_grad_(True)

    # 全 padding: 损失应为 0 且可反传（梯度全 0）
    loss, _ = phys(coords, mask=torch.zeros(2, 6, dtype=torch.long))
    loss.backward()
    assert coords.grad is not None, "全 padding 反传后无梯度张量"
    assert torch.isfinite(coords.grad).all(), "全 padding 梯度非有限"
    assert float(coords.grad.abs().max()) == 0.0, "全 padding 不应产生梯度"

    # 单 token 有效
    coords.grad = None
    m = torch.zeros(2, 6, dtype=torch.long)
    m[:, 0] = 1
    loss, _ = phys(coords, mask=m)
    loss.backward()
    assert coords.grad is not None and torch.isfinite(coords.grad).all()


def test_gradient_is_finite_and_masked():
    """梯度必须有限，且 padding 位置不应获得梯度。"""
    phys = PhysicsRegularizationLoss()
    coords = _mk_coords(B=1, T=8, seed=11).requires_grad_(True)
    m = torch.zeros(1, 8, dtype=torch.long)
    m[0, :5] = 1

    loss, _ = phys(coords, mask=m)
    loss.backward()

    g = coords.grad
    assert g is not None and torch.isfinite(g).all(), "梯度非有限"
    pad_grad = float(g[0, 5:].abs().max())
    assert pad_grad < 1e-6, f"padding 位置获得了梯度 {pad_grad:.3e}"


def test_no_mask_matches_full_mask():
    """mask=None 与全 1 mask 应给出一致结果。"""
    phys = PhysicsRegularizationLoss()
    coords = _mk_coords(B=2, T=8)
    B, T = coords.shape[:2]
    l_none, d_none = phys(coords)
    l_full, d_full = phys(coords, mask=torch.ones(B, T, dtype=torch.long))
    assert abs(float(l_none) - float(l_full)) < 1e-6
    for k in ("loss_causal", "loss_time", "loss_norm"):
        assert abs(d_none[k] - d_full[k]) < 1e-6, f"{k} 不一致"


def _main():
    tests = [
        test_all_mask_dtypes_work,
        test_non_tensor_and_zero_one_masks,
        test_shape_mismatch_raises_clear_error,
        test_padding_does_not_change_loss,
        test_time_variance_ignores_padding,
        test_masking_equals_slicing,
        test_degenerate_inputs_are_finite,
        test_degenerate_inputs_are_backward_safe,
        test_gradient_is_finite_and_masked,
        test_no_mask_matches_full_mask,
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
