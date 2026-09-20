"""
SpacetimeLM 逐样本缓存隔离 (P0-6) 回归测试

被守住的不变量:
    滑动窗口缓存是**单个序列**的状态，不是 batch 状态。SpacetimeLM 逐样本
    前向时必须为每个样本重置缓存；否则样本 b 的注意力会读到样本 0..b-1 的
    K/V，产生跨样本上下文泄漏（且泄漏程度随 batch 内位置变化，破坏训练一致性）。

    修复后: batch 前向的结果必须与"逐样本独立前向"逐元素相等。

同时守住:
    - collect_attn_weights 默认关闭，不累积持有计算图的权重张量
    - 开启该开关时行为可用（返回 B*num_layers 个张量）
    - forward 之后缓存被清空，不会把本 batch 当作下一次的历史

运行方式:
    python Logic/SpacetimeTransformer/TestCacheIsolation.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from Logic.SpacetimeTransformer import SpacetimeLM


def _mk_model(vocab=64, d_model=16, layers=2, T=8):
    torch.manual_seed(0)
    model = SpacetimeLM(vocab_size=vocab, d_model=d_model, d_space=4, d_time=4,
                        num_heads=2, window_size=64, num_layers=layers, dropout=0.0)
    model.eval()
    return model


def _mk_batch(B=4, T=8, vocab=64):
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, vocab, (B, T), generator=g)
    coords = torch.zeros(B, T, 4)
    for i in range(T):
        coords[:, i, 0] = i % 2
        coords[:, i, 3] = i / (T - 1)
    return ids, coords


def test_batch_matches_isolated_forward():
    """核心: batch 前向 == 逐样本独立前向（无跨样本泄漏）。"""
    model = _mk_model()
    ids, coords = _mk_batch()

    with torch.no_grad():
        model.reset_caches()
        batched, _, _ = model(ids, coords)

        # 逐样本独立前向，每次前先 reset（语义上的"正确参考"）
        per_sample = []
        for b in range(ids.shape[0]):
            model.reset_caches()
            out_b, _, _ = model(ids[b:b + 1], coords[b:b + 1])
            per_sample.append(out_b)
        isolated = torch.cat(per_sample, dim=0)

    diff = (batched - isolated).abs().flatten(1).max(dim=1).values
    worst = float(diff.max())
    assert worst < 1e-6, (
        f"batch 前向与逐样本前向不一致（跨样本缓存泄漏），最大差异 {worst:.3e}；"
        f" 各样本差异={[round(float(v), 8) for v in diff]}"
    )


def test_sample_content_does_not_affect_others():
    """更强的检验: 改动样本 0 的内容，不应影响其他样本的输出。"""
    model = _mk_model()
    ids, coords = _mk_batch()

    with torch.no_grad():
        model.reset_caches()
        base, _, _ = model(ids, coords)

        perturbed_ids = ids.clone()
        perturbed_ids[0] = torch.randint(0, 64, (ids.shape[1],),
                                        generator=torch.Generator().manual_seed(99))
        model.reset_caches()
        changed, _, _ = model(perturbed_ids, coords)

    # 样本 0 自身应变化
    assert not torch.allclose(base[0], changed[0], atol=1e-6), (
        "改动样本 0 的内容后其输出未变，测试无效"
    )
    # 其他样本必须完全不变
    for b in range(1, ids.shape[0]):
        d = float((base[b] - changed[b]).abs().max())
        assert d < 1e-6, (
            f"改动样本 0 导致样本 {b} 的输出变化 {d:.3e} —— 存在跨样本依赖"
        )


def test_cache_is_empty_after_forward():
    """forward 结束后所有层缓存必须为空（不残留本 batch 状态）。"""
    model = _mk_model()
    ids, coords = _mk_batch(B=2, T=8)
    with torch.no_grad():
        model(ids, coords)
    for i, block in enumerate(model.blocks):
        c = block.attn.cache
        assert c.cache_k_s is None, f"blocks[{i}] 缓存未清空（cache_k_s 非 None）"
        assert c.cache_v is None, f"blocks[{i}] 缓存未清空（cache_v 非 None）"
        assert c.cache_len == 0, f"blocks[{i}] cache_len={c.cache_len} != 0"


def test_attn_weights_not_collected_by_default():
    """默认不收集注意力权重（避免累积持有计算图的张量）。"""
    model = _mk_model()
    model.train()
    ids, coords = _mk_batch(B=4, T=8)
    _, attn_list, _ = model(ids, coords)
    assert attn_list is None, (
        f"默认应不收集注意力权重，实际返回 {type(attn_list).__name__} "
        f"(len={len(attn_list) if attn_list is not None else 0})"
    )


def test_attn_weights_collected_on_request():
    """显式开启时仍可用，数量为 B * num_layers。"""
    model = _mk_model(layers=2)
    ids, coords = _mk_batch(B=3, T=8)
    B, layers = ids.shape[0], len(model.blocks)
    _, attn_list, _ = model(ids, coords, collect_attn_weights=True)
    assert attn_list is not None, "开启开关后仍未收集"
    assert len(attn_list) == B * layers, (
        f"应收集 {B * layers} 个权重张量，实际 {len(attn_list)}"
    )


def test_training_step_still_works():
    """联合损失反传 + 优化步仍然正常（回归保护）。"""
    model = _mk_model()
    model.train()
    ids, coords = _mk_batch(B=4, T=8)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    logits, _, coords_norm = model(ids, coords)
    ce = torch.nn.functional.cross_entropy(
        logits[..., :-1, :].reshape(-1, logits.size(-1)),
        ids[..., 1:].reshape(-1),
    )
    ce.backward()
    for n in ["W_k_s", "W_v"]:
        g = getattr(model.blocks[0].attn, n).weight.grad
        assert g is not None and torch.isfinite(g).all(), f"{n} 梯度异常"
    opt.step()


def test_gradients_are_deterministic_across_batch_positions():
    """
    修复前，样本的梯度会因其在 batch 中的位置而不同（前面的样本看到更少历史）。
    这里检验同一样本在不同 batch 内位置上的梯度一致。
    """
    model = _mk_model()
    model.train()
    ids, coords = _mk_batch(B=4, T=8)

    def grad_for_sample(idx):
        model.zero_grad()
        logits, _, _ = model(ids, coords)
        # 只取该样本的损失
        loss = logits[idx].float().sum()
        loss.backward()
        return model.blocks[0].attn.W_v.weight.grad.detach().clone()

    # 单样本 batch
    model.zero_grad()
    single, _, _ = model(ids[2:3], coords[2:3])
    single.float().sum().backward()
    g_single = model.blocks[0].attn.W_v.weight.grad.detach().clone()

    # 四样本 batch 中同样只对样本 2 反传
    model.zero_grad()
    logits, _, _ = model(ids, coords)
    logits[2].float().sum().backward()
    g_in_batch = model.blocks[0].attn.W_v.weight.grad.detach().clone()

    d = float((g_single - g_in_batch).abs().max())
    assert d < 1e-6, (
        f"同一样本单独前向与置于 batch 中时梯度不一致（差异 {d:.3e}）"
        f" —— 样本行为依赖 batch 组成"
    )


def _main():
    tests = [
        test_batch_matches_isolated_forward,
        test_sample_content_does_not_affect_others,
        test_cache_is_empty_after_forward,
        test_attn_weights_not_collected_by_default,
        test_attn_weights_collected_on_request,
        test_training_step_still_works,
        test_gradients_are_deterministic_across_batch_positions,
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
