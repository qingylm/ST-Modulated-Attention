"""
SlidingWindowCache / SpacetimeAttentionWithCache 回归测试

被守住的不变量（P0-5）:
    TBPTT 只应切断**历史**部分的梯度，**本 chunk** 的 K_s/K_t/V 必须留在
    计算图中。旧实现在缓存为空时 `return self.cache_*`（已 detach），
    导致首次 forward 的 W_k_s / W_k_t / W_v 完全拿不到梯度。

同时守住:
    - 拼接口径: k_s/k_t/v 沿 dim=2 拼接，coords 沿 dim=1 拼接
    - 截断后当前 chunk 仍在返回值内（否则梯度仍会丢）
    - 缓存长度受 window_size 约束
    - 已 detach 的历史不会把梯度带回更早的 chunk（TBPTT 语义本身）

运行方式:
    python Logic/CoreAttention/TestSlidingWindowCache.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from Logic.CoreAttention import SpacetimeAttentionWithCache
from Logic.CoreAttention.SlidingWindowCache import SlidingWindowCache

PROJ = ["W_q_s", "W_k_s", "W_q_t", "W_k_t", "W_v"]


def _mk_inputs(B=2, S=8, Ds=4, Dt=2, D=8, requires_grad=True):
    k_s = torch.randn(B, 2, S, Ds, requires_grad=requires_grad)
    k_t = torch.randn(B, 2, S, Dt, requires_grad=requires_grad)
    v = torch.randn(B, 2, S, D, requires_grad=requires_grad)
    coords = torch.randn(B, S, 4, requires_grad=requires_grad)
    return k_s, k_t, v, coords


# ---------------------------------------------------------------- 核心回归
def test_first_update_returns_graph_attached_tensors():
    """首次 update 返回的必须是入参本身（带图），而不是 detach 后的缓存。"""
    cache = SlidingWindowCache(window_size=64)
    k_s, k_t, v, coords = _mk_inputs()

    fk_s, fk_t, fv, fc = cache.update(k_s, k_t, v, coords)

    assert fk_s is k_s, "首次 update 未返回入参 k_s（说明返回了 detach 副本）"
    assert fk_t is k_t, "首次 update 未返回入参 k_t"
    assert fv is v, "首次 update 未返回入参 v"
    assert fc is coords, "首次 update 未返回入参 coords"
    for name, t in [("full_k_s", fk_s), ("full_k_t", fk_t), ("full_v", fv)]:
        assert t.requires_grad, f"{name} 丢失了计算图"

    # 缓存自身仍必须是 detach 的，否则 TBPTT 会退化成完整 BPTT
    for name, t in [("cache_k_s", cache.cache_k_s), ("cache_k_t", cache.cache_k_t),
                    ("cache_v", cache.cache_v), ("cache_coords", cache.cache_coords)]:
        assert not t.requires_grad, f"{name} 未 detach，TBPTT 会被破坏"


def test_attention_layer_gets_kv_gradients_on_first_forward():
    """端到端: 首次 forward 后 Q/K/V 所有投影层都必须有梯度。"""
    torch.manual_seed(7)
    B, T = 2, 8
    attn = SpacetimeAttentionWithCache(d_model=8, d_space=4, d_time=2,
                                       num_heads=2, window_size=1024)
    x = torch.randn(B, T, 8)
    coords = torch.zeros(B, T, 4)
    coords[:, :, 3] = torch.linspace(0, 1, T)

    out, _ = attn(x, coords)
    out.sum().backward()

    missing = [n for n in PROJ if getattr(attn, n).weight.grad is None]
    assert not missing, f"这些投影层拿不到梯度（P0-5 回归）: {missing}"

    # 与"绕开缓存"的同权重同数学前向对比，梯度量级应一致
    ref = SpacetimeAttentionWithCache(d_model=8, d_space=4, d_time=2,
                                      num_heads=2, window_size=1024)
    ref.load_state_dict(attn.state_dict())
    Q_s = ref.W_q_s(x).view(B, T, ref.num_heads, ref.d_space).transpose(1, 2)
    Q_t = ref.W_q_t(x).view(B, T, ref.num_heads, ref.d_time).transpose(1, 2)
    K_s = ref.W_k_s(x).view(B, T, ref.num_heads, ref.d_space).transpose(1, 2)
    K_t = ref.W_k_t(x).view(B, T, ref.num_heads, ref.d_time).transpose(1, 2)
    V = ref.W_v(x).view(B, T, ref.num_heads, -1).transpose(1, 2)
    lg = ref.logits_calc(Q_s, K_s, Q_t, K_t)
    mk = ref.mask_engine(coords).unsqueeze(1)
    ctx = torch.matmul(torch.softmax(lg + mk, dim=-1), V)
    ref.out_proj(ctx.transpose(1, 2).contiguous().view(B, T, -1)).sum().backward()

    for n in PROJ:
        g1 = getattr(attn, n).weight.grad
        g2 = getattr(ref, n).weight.grad
        assert torch.allclose(g1, g2, atol=1e-6), (
            f"{n} 梯度与不经缓存的参考实现不一致: "
            f"max diff = {float((g1 - g2).abs().max()):.3e}"
        )


def test_second_update_keeps_current_chunk_attached():
    """第二次 update: 历史 detach，但当前 chunk 仍带图。"""
    cache = SlidingWindowCache(window_size=64)
    cache.update(*_mk_inputs(S=4))

    k_s, k_t, v, coords = _mk_inputs(S=3)
    fk_s, fk_t, fv, fc = cache.update(k_s, k_t, v, coords)

    assert fk_s.shape[2] == 4 + 3, f"拼接长度错误: {fk_s.shape[2]}"
    assert fc.shape[1] == 4 + 3, f"coords 拼接长度错误: {fc.shape[1]}"

    # 当前 chunk 的后 3 个位置必须保留计算图 -> 对入参求导应得到非零梯度
    g = torch.autograd.grad(fk_s[:, :, -3:, :].sum(), k_s, allow_unused=True)[0]
    assert g is not None and float(g.abs().sum()) > 0, "当前 chunk 的 K_s 被 detach 了"

    # 历史部分的梯度已被切断
    gv = torch.autograd.grad(fv.sum(), v, allow_unused=True)[0]
    assert gv is not None and float(gv.abs().sum()) > 0, "当前 chunk 的 V 被 detach 了"


def test_truncation_keeps_current_chunk():
    """窗口截断后，当前 chunk 必须仍完整保留在返回值中（否则梯度会丢）。"""
    window = 5
    cache = SlidingWindowCache(window_size=window)
    cache.update(*_mk_inputs(S=4))                 # 历史 4 个 token

    S = 3
    k_s, k_t, v, coords = _mk_inputs(S=S)
    fk_s, fk_t, fv, fc = cache.update(k_s, k_t, v, coords)

    assert fk_s.shape[2] == window, f"截断后长度应为 {window}, 实际 {fk_s.shape[2]}"
    assert cache.cache_len == window

    # 末尾 S 个位置必须仍是当前 chunk（带图）
    g = torch.autograd.grad(fk_s[:, :, -S:, :].sum(), k_s, allow_unused=True)[0]
    assert g is not None and float(g.abs().sum()) > 0, (
        "截断把当前 chunk 的梯度切掉了"
    )


def test_history_gradient_is_blocked():
    """TBPTT 语义: 第二次调用不应把梯度回传到第一次调用的输入。"""
    cache = SlidingWindowCache(window_size=64)
    k_s1 = torch.randn(1, 2, 4, 4, requires_grad=True)
    k_t1 = torch.randn(1, 2, 4, 2, requires_grad=True)
    v1 = torch.randn(1, 2, 4, 8, requires_grad=True)
    c1 = torch.randn(1, 4, 4, requires_grad=True)
    cache.update(k_s1, k_t1, v1, c1)               # 写入历史

    k_s2 = torch.randn(1, 2, 3, 4, requires_grad=True)
    k_t2 = torch.randn(1, 2, 3, 2, requires_grad=True)
    v2 = torch.randn(1, 2, 3, 8, requires_grad=True)
    c2 = torch.randn(1, 3, 4, requires_grad=True)
    fk_s, _, fv, _ = cache.update(k_s2, k_t2, v2, c2)

    g_hist = torch.autograd.grad(fk_s.sum() + fv.sum(), k_s1, allow_unused=True)[0]
    assert g_hist is None, (
        "历史 chunk 的梯度未被切断（TBPTT 失效，显存会无界增长）"
    )
    # 当前 chunk 仍应有梯度
    g_cur = torch.autograd.grad(fk_s.sum(), k_s2, allow_unused=True)[0]
    assert g_cur is not None and float(g_cur.abs().sum()) > 0


def test_cache_len_and_reset():
    cache = SlidingWindowCache(window_size=10)
    cache.update(*_mk_inputs(S=4))
    assert cache.cache_len == 4
    cache.update(*_mk_inputs(S=4))
    assert cache.cache_len == 8
    cache.update(*_mk_inputs(S=4))
    assert cache.cache_len == 10, f"截断后应为 10, 实际 {cache.cache_len}"
    cache.reset()
    assert cache.cache_len == 0 and cache.cache_k_s is None


def test_forward_shapes_unchanged():
    """确认修复没有改变前向输出的形状（与原行为一致）。"""
    torch.manual_seed(0)
    attn = SpacetimeAttentionWithCache(d_model=16, d_space=6, d_time=4,
                                       num_heads=2, window_size=4)
    x = torch.randn(1, 3, 16)
    coords = torch.randn(1, 3, 4)
    out, w = attn(x, coords)
    assert out.shape == (1, 3, 16), f"输出形状错误: {out.shape}"
    assert w.shape == (1, 2, 3, 3), f"注意力形状错误: {w.shape}"

    x2 = torch.randn(1, 2, 16)
    coords2 = torch.randn(1, 2, 4)
    out2, w2 = attn(x2, coords2)
    assert out2.shape == (1, 2, 16)
    assert w2.shape == (1, 2, 2, 4), f"第二次注意力形状错误: {w2.shape}"


def _main():
    tests = [
        test_first_update_returns_graph_attached_tensors,
        test_attention_layer_gets_kv_gradients_on_first_forward,
        test_second_update_keeps_current_chunk_attached,
        test_truncation_keeps_current_chunk,
        test_history_gradient_is_blocked,
        test_cache_len_and_reset,
        test_forward_shapes_unchanged,
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
