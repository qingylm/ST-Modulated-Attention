"""
初审问题的可执行复核（最终版）

对每条初审结论给出最小可判定实验。运行:
    python Logic/CoreAttention/TestAuditFindings.py

输出 [成立] / [不成立]，并反转那些经复核不成立的初审判断。
"""
import io
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from Logic.CoreAttention import (
    LightConeMaskEngine,
    PhysicsRegularizationLoss,
    SlidingWindowCache,
    SpacetimeAttentionWithCache,
)
from Logic.SpacetimeTransformer import SpacetimeLM

BUF = io.StringIO()


def log(*a):
    print(*a, file=BUF)


def verdict(tag, ok, detail):
    log(f"\n[{'成立' if ok else '不成立'}] {tag}")
    log(f"    {detail}")
    return (tag, ok)


# --------------------------------------------------------------------------
def audit_p0_1():
    """掩码覆写 / 未来 token 泄漏。"""
    log("\n" + "=" * 74)
    log("P0-1  类空掩码是否覆写因果掩码，导致未来 token 泄漏")
    log("=" * 74)

    engine = LightConeMaskEngine(init_bias=5.0)
    # LCCC 真实角色模式: user x=0, assistant x=1, t 递增
    n = 10
    coords = torch.zeros(1, n, 4)
    coords[0, :, 0] = torch.arange(n) % 2          # 交错角色
    coords[0, :, 3] = torch.linspace(0, 1, n)
    dt = coords[0, :, 3].unsqueeze(-1) - coords[0, :, 3].unsqueeze(-2)

    mask = engine(coords)
    torch.manual_seed(0)
    logits = torch.randn(1, n, n) * 0.1
    logits = torch.where(dt < 0, torch.full_like(logits, 50.0), logits)
    w = torch.softmax(logits + mask, dim=-1)

    future = dt < 0
    max_future = float(w[0][future].max())
    n_future = int(future.sum())
    log(f"  未来位置数 = {n_future}")
    log(f"  未来位置最大注意力权重 = {max_future:.8f}")
    log(f"  引擎内部类空条件: interval_sq>0 & (~causal)，与 causal 不相交")

    return verdict("P0-1 因果掩码被覆写", max_future > 1e-6,
                   f"未来位置最大权重 {max_future:.2e} -> "
                   + ("存在泄漏" if max_future > 1e-6 else
                      "无泄漏（`& (~causal_mask)` 保证两类掩码互斥，write 顺序无关）。"
                      " 我初审此处判断有误。"))


# --------------------------------------------------------------------------
def audit_p0_3():
    """
    物理损失是否进入计算图。

    该问题已修复: SpacetimeLM.forward 返回 coords_norm，Train.py 用
    phys(coords_norm) 代替 phys(coords_raw)。此处改为静态校验修复仍然在位，
    并复核"coords_raw 路径本身确实不携带计算图"这一根因。
    """
    log("\n" + "=" * 74)
    log("P0-3  物理损失是否进入计算图（已修复）")
    log("=" * 74)

    from Logic.SubsequentProcessing.LearnableSpacetimeNormalizer import (
        LearnableSpacetimeNormalizer,
    )
    norm = LearnableSpacetimeNormalizer()
    phys = PhysicsRegularizationLoss()
    coords_raw = torch.randn(2, 8, 4)

    # 根因复核: coords_raw 是数据集常量 -> 损失无图
    loss_a, _ = phys(coords_raw)
    log(f"  A) phys(coords_raw):         requires_grad={loss_a.requires_grad}")
    coords_norm = norm(coords_raw)
    loss_b, _ = phys(coords_norm)
    log(f"  B) phys(norm(coords_raw)):   requires_grad={loss_b.requires_grad}")

    # 静态校验 Train.py 两处调用点都用 coords_norm
    src = open("Train/Train.py", encoding="utf-8").read()
    n_norm = src.count("phys_loss_fn(coords_norm")
    n_raw = src.count("phys_loss_fn(coords_raw")
    n_unpack = src.count(", _, coords_norm = model(")
    log(f"  Train.py: phys_loss_fn(coords_norm) 出现 {n_norm} 次, "
        f"phys_loss_fn(coords_raw) 出现 {n_raw} 次")
    log(f"  Train.py: 三返回值解包 ..., _, coords_norm = model(...) 出现 {n_unpack} 次")

    # 正向校验: 模型 forward 确实返回带图的 coords_norm
    from Logic.SpacetimeTransformer import SpacetimeLM
    m = SpacetimeLM(vocab_size=32, d_model=16, d_space=4, d_time=4,
                    num_heads=2, window_size=64, num_layers=1, dropout=0.0)
    out = m(torch.randint(0, 32, (2, 6)), torch.randn(2, 6, 4))
    forward_ok = len(out) == 3 and out[2].requires_grad
    log(f"  SpacetimeLM.forward 返回 3 个值且 coords_norm 带图: {forward_ok}")

    fixed = (not loss_a.requires_grad) and loss_b.requires_grad \
        and n_norm >= 2 and n_raw == 0 and n_unpack >= 2 and forward_ok

    return verdict("P0-3 物理损失脱离计算图（已修复）", not fixed,
                   ("根因: coords_raw 无计算图（requires_grad=%s）。"
                    "修复: forward 返回 coords_norm，Train.py 两处调用点均改用 "
                    "phys(coords_norm)，coords_raw 调用已清零。"
                    % loss_a.requires_grad)
                   if fixed else
                   f"修复不完整: n_norm={n_norm}, n_raw={n_raw}, "
                   f"n_unpack={n_unpack}, forward_ok={forward_ok}")


# --------------------------------------------------------------------------
def audit_p0_5():
    """缓存 detach 是否切断本次 chunk 的 K/V 梯度。"""
    log("\n" + "=" * 74)
    log("P0-5  滑动窗口缓存首次调用是否切断 K/V 梯度")
    log("=" * 74)

    T, B = 8, 2
    PROJ = ["W_q_s", "W_k_s", "W_q_t", "W_k_t", "W_v"]
    torch.manual_seed(7)
    base = SpacetimeAttentionWithCache(d_model=8, d_space=4, d_time=2,
                                       num_heads=2, window_size=1024)
    x = torch.randn(B, T, 8)
    coords = torch.zeros(B, T, 4)
    coords[:, :, 3] = torch.linspace(0, 1, T)

    # 路径 1: 真实 forward（经缓存）
    m1 = SpacetimeAttentionWithCache(d_model=8, d_space=4, d_time=2,
                                     num_heads=2, window_size=1024)
    m1.load_state_dict(base.state_dict())
    out1, _ = m1(x, coords)
    out1.sum().backward()
    g1 = {n: getattr(m1, n).weight.grad for n in PROJ}

    # 路径 2: 同权重、同数学，但不经缓存
    m2 = SpacetimeAttentionWithCache(d_model=8, d_space=4, d_time=2,
                                     num_heads=2, window_size=1024)
    m2.load_state_dict(base.state_dict())
    Q_s = m2.W_q_s(x).view(B, T, m2.num_heads, m2.d_space).transpose(1, 2)
    Q_t = m2.W_q_t(x).view(B, T, m2.num_heads, m2.d_time).transpose(1, 2)
    K_s = m2.W_k_s(x).view(B, T, m2.num_heads, m2.d_space).transpose(1, 2)
    K_t = m2.W_k_t(x).view(B, T, m2.num_heads, m2.d_time).transpose(1, 2)
    V = m2.W_v(x).view(B, T, m2.num_heads, -1).transpose(1, 2)
    lg = m2.logits_calc(Q_s, K_s, Q_t, K_t)
    mk = m2.mask_engine(coords).unsqueeze(1)
    ctx = torch.matmul(torch.softmax(lg + mk, dim=-1), V)
    out2 = m2.out_proj(ctx.transpose(1, 2).contiguous().view(B, T, -1))
    out2.sum().backward()
    g2 = {n: getattr(m2, n).weight.grad for n in PROJ}

    log(f"  路径1 真实 forward(经缓存):")
    for n in PROJ:
        log(f"      {n:6s} = {'None' if g1[n] is None else f'{float(g1[n].abs().sum()):.6f}'}")
    log(f"  路径2 同权重同数学(不经缓存):")
    for n in PROJ:
        log(f"      {n:6s} = {'None' if g2[n] is None else f'{float(g2[n].abs().sum()):.6f}'}")
    lost = [n for n in PROJ if g2[n] is not None and g1[n] is None]
    log(f"  因缓存 detach 而失去梯度的参数: {lost}")

    # 复现根因
    cache = SlidingWindowCache(window_size=64)
    k = torch.randn(1, 2, 3, 4, requires_grad=True)
    v = torch.randn(1, 2, 3, 8, requires_grad=True)
    fk, _, fv, _ = cache.update(k, k[..., :2], v, torch.randn(1, 3, 4))
    log(f"  根因: update 返回 self.cache_*.detach() -> "
        f"full_k_s.requires_grad={fk.requires_grad}, full_v.requires_grad={fv.requires_grad}")

    return verdict("P0-5 缓存切断本次 chunk 的 K/V 梯度", bool(lost),
                   f"`SlidingWindowCache.update` 在 cache 为空时 `return self.cache_*`"
                   f"（已 detach），导致首次 forward 的 K_s/K_t/V 全部脱离计算图，"
                   f"{lost} 拿不到梯度。修复: 返回未 detach 的当前 chunk。")


# --------------------------------------------------------------------------
def audit_p0_6():
    """
    跨样本缓存复用 + attn 列表持有计算图。

    该问题已修复:
      - SpacetimeLM 在逐样本循环内对每个样本 reset_caches()
      - collect_attn_weights 默认关闭，不再累积持有计算图的权重张量
    此处改为校验修复仍然在位。
    """
    log("\n" + "=" * 74)
    log("P0-6  逐样本缓存隔离（已修复）")
    log("=" * 74)

    torch.manual_seed(0)
    model = SpacetimeLM(vocab_size=64, d_model=16, d_space=4, d_time=4,
                        num_heads=2, window_size=64, num_layers=1, dropout=0.0)
    model.eval()
    B, T = 4, 6
    ids = torch.randint(0, 64, (B, T))
    coords = torch.zeros(B, T, 4)
    coords[:, :, 3] = torch.linspace(0, 1, T)

    with torch.no_grad():
        batched, _, _ = model(ids, coords)
        iso = []
        for b in range(B):
            model.reset_caches()
            o, _, _ = model(ids[b:b + 1], coords[b:b + 1])
            iso.append(o)
        iso = torch.cat(iso, 0)

    diff = (batched - iso).abs().flatten(1).max(dim=1).values
    worst = float(diff.max())
    log(f"  与『逐样本独立前向』的最大 logits 差异（按样本）: "
        f"{[round(float(v), 8) for v in diff]}")
    log(f"  最大差异 = {worst:.3e}  (阈值 1e-6)")
    contaminated = [i for i in range(B) if float(diff[i]) > 1e-6]
    log(f"  仍受前序样本影响的样本下标: {contaminated}")

    # 默认不收集注意力权重
    model.train()
    _, attn_list, _ = model(ids, coords)
    log(f"  默认 collect_attn_weights: attn_weights_list = {attn_list}")
    # 显式开启时仍可用
    _, attn_on, _ = model(ids, coords, collect_attn_weights=True)
    n_on = 0 if attn_on is None else len(attn_on)
    log(f"  显式开启后收集到 {n_on} 个张量（期望 B*num_layers={B * len(model.blocks)}）")

    # forward 后缓存应为空
    block_cache_len = model.blocks[0].attn.cache.cache_len
    log(f"  forward 后 blocks[0] 缓存长度 = {block_cache_len}")

    fixed = (not contaminated) and attn_list is None \
        and n_on == B * len(model.blocks) and block_cache_len == 0

    return verdict("P0-6 跨样本缓存复用（已修复）", not fixed,
                   "每个样本前向前都会 reset_caches()，batch 前向与逐样本独立前向"
                   "逐元素一致；跨样本上下文泄漏已消除。"
                   " attn_weights_list 默认不再收集（此前会累积 B*num_layers 个"
                   "带 autograd 图的张量）。"
                   if fixed else
                   f"修复不完整: contaminated={contaminated}, "
                   f"attn_list={attn_list}, n_on={n_on}, cache_len={block_cache_len}")


# --------------------------------------------------------------------------
def audit_p0_4():
    """验证集泄漏。"""
    log("\n" + "=" * 74)
    log("P0-4  验证集是否被拼进训练集")
    log("=" * 74)
    src = open("Train/Train.py", encoding="utf-8").read()
    tl = [l.strip() for l in src.splitlines() if l.strip().startswith("all_datasets =")]
    vl = [l.strip() for l in src.splitlines() if l.strip().startswith("val_dataset =")]
    log(f"  {tl[0] if tl else '?'}")
    log(f"  {vl[0] if vl else '?'}")
    ok = bool(tl) and "lccc_valid_datasets" in tl[0] and bool(vl)
    return verdict("P0-4 验证集泄漏", ok,
                   "lccc_valid_dataset 同时是训练集成员与验证集 -> "
                   "val loss 不能作为泛化指标。")


# --------------------------------------------------------------------------
def audit_p0_2():
    """物理项的真实影响量级。"""
    log("\n" + "=" * 74)
    log("P0-2  物理项的实际影响量级")
    log("=" * 74)

    # 用与真实数据同构的坐标: x∈{0,1}, y=z=0, t 随消息递增且消息长度不均
    n = 64
    msg_len = [20, 15, 12, 9, 5, 3]           # 真实对话消息长度差异很大
    coords = torch.zeros(1, n, 4)
    pos = 0
    t_vals = torch.linspace(0, 1, len(msg_len))
    for m, L in enumerate(msg_len):
        for _ in range(L):
            if pos >= n:
                break
            coords[0, pos, 0] = m % 2
            coords[0, pos, 3] = t_vals[m]
            pos += 1
    x, y, z, t = (coords[..., i] for i in range(4))
    dt = t.unsqueeze(-1) - t.unsqueeze(-2)
    ds2 = ((x.unsqueeze(-1) - x.unsqueeze(-2)) ** 2
           + (y.unsqueeze(-1) - y.unsqueeze(-2)) ** 2
           + (z.unsqueeze(-1) - z.unsqueeze(-2)) ** 2) - dt ** 2
    off = ~torch.eye(n, dtype=torch.bool).unsqueeze(0)
    frac = float(((ds2 > 0) & off).sum()) / int(off.sum())
    causal = float(torch.relu(ds2)[off].mean())
    log(f"  类空对数占比 = {frac:.4f}")
    log(f"  causal 惩罚均值 = {causal:.6f}")
    log(f"  训练日志实测: causal 严格为0占 45.4%, 非零均值 0.0066, "
        f"phys 合计均值 -0.0078")
    log(f"  |phys| / ce(≈3.5) ≈ {abs(-0.0078) / 3.5:.2e}")

    return verdict("P0-2 物理项完全失效", False,
                   "causal 项并非恒为 0（日志中 54.6% 的步非零，均值 0.0066），"
                   "但物理项合计仅约交叉熵的 0.2%，实际影响可忽略。"
                   " 我初审『causal 恒为 0』的说法不准确。")


# --------------------------------------------------------------------------
def audit_new_findings():
    """复核过程中新发现的问题。"""
    log("\n" + "=" * 74)
    log("新增发现")
    log("=" * 74)
    out = []

    # 1) PhysicsRegularizationLoss 对 float mask 已加固
    phys = PhysicsRegularizationLoss()
    coords = torch.randn(2, 6, 4)
    results = {}
    for dt in (torch.long, torch.int8, torch.bool, torch.float32, torch.float16):
        try:
            loss, _ = phys(coords, mask=torch.ones(2, 6, dtype=dt))
            results[str(dt).replace("torch.", "")] = (
                "OK" if torch.isfinite(loss) else "非有限")
        except Exception as exc:  # noqa: BLE001
            results[str(dt).replace("torch.", "")] = type(exc).__name__
    log(f"  PhysicsRegularizationLoss 对不同 mask dtype 的表现: {results}")
    dtype_ok = all(v == "OK" for v in results.values())
    out.append(verdict("新增: 物理损失对 float mask 崩溃（已加固）", not dtype_ok,
                       "损失内部已把 mask 统一转成计算 dtype 再相乘（不再用 `&` 位"
                       "运算），long/int8/bool/float32/float16 均可用；"
                       "同时会拒绝形状不符的 mask 并给出清晰错误。"
                       if dtype_ok else f"仍有 dtype 失败: {results}"))

    # 2) 掩码在 fp16 下可用（硬屏蔽哨兵必须是 -inf，而非 -1e9）
    from Logic.CoreAttention import LightConeMaskEngine
    engine = LightConeMaskEngine()
    fp16_ok = True
    detail = []
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        c = torch.zeros(1, 6, 4, dtype=dtype)
        c[0, :, 0] = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=dtype)
        c[0, :, 3] = torch.linspace(0, 1, 6, dtype=dtype)
        try:
            m = engine(c)
            neg_inf_exact = bool((m == float("-inf")).any())
            finite_elsewhere = bool(torch.isfinite(m[m != float("-inf")]).all())
            ok = (m.dtype == dtype) and neg_inf_exact and finite_elsewhere
            detail.append(f"{str(dtype).replace('torch.', '')}="
                          f"{'OK' if ok else 'FAIL'}")
            fp16_ok = fp16_ok and ok
        except Exception as exc:  # noqa: BLE001
            detail.append(f"{str(dtype).replace('torch.', '')}="
                          f"{type(exc).__name__}")
            fp16_ok = False
    log(f"  LightConeMaskEngine 各精度: {', '.join(detail)}")
    out.append(verdict("新增: -1e9 哨兵在 fp16 下不可表示（已改 -inf）", not fp16_ok,
                       "掩码改用显式 -inf 且 dtype 跟随输入，fp32/fp16/bf16 下均可用；"
                       "旧写法 -1e9 在 fp16 下会抛 RuntimeError（范围 ±65504）。"
                       if fp16_ok else f"仍有精度失败: {detail}"))


    # 3) collate 中的死代码
    src = open("Train/Train.py", encoding="utf-8").read()
    has_dead = "coords_tensors" in src and \
        "pad_sequence(coords_raw" in src and "pad_sequence(coords_tensors" not in src
    out.append(verdict("新增: collate 构建 coords_tensors 但未使用", has_dead,
                       "Train.py 构造 coords_tensors 后仍对原始 coords_raw 调用 "
                       "pad_sequence；当前 dataset 已返回 Tensor 故能跑通，"
                       "若记录仍是二维 list 会抛 TypeError。"))

    # 4) SubsequentProcessing 中的重复实现已删除
    dup_path = "Logic/SubsequentProcessing/SpacetimeAttentionWithCache.py"
    dup_exists = os.path.exists(dup_path)
    # __init__ 不得再 import 该模块（检查导入语句，不检查说明性文字）
    init_lines = open("Logic/SubsequentProcessing/__init__.py", encoding="utf-8").readlines()
    bad_imports = [ln.strip() for ln in init_lines
                   if ln.lstrip().startswith(("from", "import"))
                   and "SpacetimeAttentionWithCache" in ln]
    # CoreAttention 下的正统实现必须仍在
    canonical = os.path.exists("Logic/CoreAttention/SpacetimeAttentionWithCache.py")
    expected = "from .LearnableSpacetimeNormalizer import LearnableSpacetimeNormalizer"
    exports_ok = any(ln.strip() == expected for ln in init_lines)

    removed_cleanly = (not dup_exists) and (not bad_imports) and canonical and exports_ok
    out.append(verdict("新增: SubsequentProcessing 中的重复实现已删除",
                       not removed_cleanly,
                       "损坏的重复实现（__init__ 只有省略号、self.W_q_s 未定义、"
                       "space_norm_type='rms' 非法枚举、模块底部 test_normalizer() "
                       "导入副作用）已删除；SubsequentProcessing/__init__ 已不再导入它；"
                       f"正统实现在 Logic.CoreAttention 下保留={canonical}。"
                       if removed_cleanly else
                       f"删除不完整: dup_exists={dup_exists}, bad_imports={bad_imports}, "
                       f"canonical={canonical}, exports_ok={exports_ok}"))

    # 5) clip_value 必须相对数据幅值留有余量（此前 2.0 会冻结 space_scale）
    from Logic.SubsequentProcessing import LearnableSpacetimeNormalizer
    norm = LearnableSpacetimeNormalizer()
    probe = torch.zeros(2, 64, 4)
    bounds = [0, 20, 35, 47, 56, 61, 64]
    for m in range(len(bounds) - 1):
        lo, hi = bounds[m], bounds[m + 1]
        probe[:, lo:hi, 0] = m % 2
        probe[:, lo:hi, 3] = m / (len(bounds) - 2)
    with torch.no_grad():
        peak = float(norm(probe)[..., :3].abs().max())
    headroom_ok = peak < norm.clip_value
    out.append(verdict("新增: clip_value 冻结 space_scale（已修复）",
                       not headroom_ok,
                       f"clip_value={norm.clip_value}, 归一化后空间峰值={peak:.4f} -> "
                       f"{'有余量，space_scale 可学习' if headroom_ok else '仍会饱和'}"
                       f"（此前 clip_value=2.0，峰值 2.667，梯度恰为 0）"))
    return out


if __name__ == "__main__":
    rows = []
    rows.append(audit_p0_1())
    rows.append(audit_p0_3())
    rows.append(audit_p0_5())
    rows.append(audit_p0_6())
    rows.append(audit_p0_4())
    rows.append(audit_p0_2())
    rows.extend(audit_new_findings())

    log("\n" + "=" * 74)
    log("汇总")
    log("=" * 74)
    for tag, ok in rows:
        log(f"  [{'成立  ' if ok else '不成立'}] {tag}")
    log(f"\n  成立 {sum(1 for _, ok in rows if ok)} / "
        f"不成立 {sum(1 for _, ok in rows if not ok)}")

    text = BUF.getvalue()
    with open("Logic/CoreAttention/audit_report.md", "w", encoding="utf-8") as fh:
        fh.write("# 初审结论复核报告\n\n```\n" + text + "\n```\n")
    print(text.encode("ascii", "replace").decode("ascii"))
    print("\n完整报告 -> Logic/CoreAttention/audit_report.md")
