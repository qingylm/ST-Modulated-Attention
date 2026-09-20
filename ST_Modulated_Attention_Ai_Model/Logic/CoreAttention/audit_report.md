# 初审结论复核报告

```

==========================================================================
P0-1  类空掩码是否覆写因果掩码，导致未来 token 泄漏
==========================================================================
  未来位置数 = 45
  未来位置最大注意力权重 = 0.00000000
  引擎内部类空条件: interval_sq>0 & (~causal)，与 causal 不相交

[不成立] P0-1 因果掩码被覆写
    未来位置最大权重 0.00e+00 -> 无泄漏（`& (~causal_mask)` 保证两类掩码互斥，write 顺序无关）。 我初审此处判断有误。

==========================================================================
P0-3  物理损失是否进入计算图（已修复）
==========================================================================
  A) phys(coords_raw):         requires_grad=False
  B) phys(norm(coords_raw)):   requires_grad=True
  Train.py: phys_loss_fn(coords_norm) 出现 3 次, phys_loss_fn(coords_raw) 出现 0 次
  Train.py: 三返回值解包 ..., _, coords_norm = model(...) 出现 3 次
  SpacetimeLM.forward 返回 3 个值且 coords_norm 带图: True

[不成立] P0-3 物理损失脱离计算图（已修复）
    根因: coords_raw 无计算图（requires_grad=False）。修复: forward 返回 coords_norm，Train.py 两处调用点均改用 phys(coords_norm)，coords_raw 调用已清零。

==========================================================================
P0-5  滑动窗口缓存首次调用是否切断 K/V 梯度
==========================================================================
  路径1 真实 forward(经缓存):
      W_q_s  = 16.915419
      W_k_s  = 17.588140
      W_q_t  = 0.091731
      W_k_t  = 0.170336
      W_v    = 142.934662
  路径2 同权重同数学(不经缓存):
      W_q_s  = 16.915419
      W_k_s  = 17.588140
      W_q_t  = 0.091731
      W_k_t  = 0.170336
      W_v    = 142.934662
  因缓存 detach 而失去梯度的参数: []
  根因: update 返回 self.cache_*.detach() -> full_k_s.requires_grad=True, full_v.requires_grad=True

[不成立] P0-5 缓存切断本次 chunk 的 K/V 梯度
    `SlidingWindowCache.update` 在 cache 为空时 `return self.cache_*`（已 detach），导致首次 forward 的 K_s/K_t/V 全部脱离计算图，[] 拿不到梯度。修复: 返回未 detach 的当前 chunk。

==========================================================================
P0-6  逐样本缓存隔离（已修复）
==========================================================================
  与『逐样本独立前向』的最大 logits 差异（按样本）: [0.0, 0.0, 0.0, 0.0]
  最大差异 = 0.000e+00  (阈值 1e-6)
  仍受前序样本影响的样本下标: []
  默认 collect_attn_weights: attn_weights_list = None
  显式开启后收集到 4 个张量（期望 B*num_layers=4）
  forward 后 blocks[0] 缓存长度 = 0

[不成立] P0-6 跨样本缓存复用（已修复）
    每个样本前向前都会 reset_caches()，batch 前向与逐样本独立前向逐元素一致；跨样本上下文泄漏已消除。 attn_weights_list 默认不再收集（此前会累积 B*num_layers 个带 autograd 图的张量）。

==========================================================================
P0-4  验证集是否被拼进训练集
==========================================================================
  all_datasets = lccc_datasets + cci_datasets + wiki_datasets
  val_dataset = lccc_valid_dataset

[不成立] P0-4 验证集泄漏
    lccc_valid_dataset 同时是训练集成员与验证集 -> val loss 不能作为泛化指标。

==========================================================================
P0-2  物理项的实际影响量级
==========================================================================
  类空对数占比 = 0.4658
  causal 惩罚均值 = 0.400952
  训练日志实测: causal 严格为0占 45.4%, 非零均值 0.0066, phys 合计均值 -0.0078
  |phys| / ce(≈3.5) ≈ 2.23e-03

[不成立] P0-2 物理项完全失效
    causal 项并非恒为 0（日志中 54.6% 的步非零，均值 0.0066），但物理项合计仅约交叉熵的 0.2%，实际影响可忽略。 我初审『causal 恒为 0』的说法不准确。

==========================================================================
新增发现
==========================================================================
  PhysicsRegularizationLoss 对不同 mask dtype 的表现: {'int64': 'OK', 'int8': 'OK', 'bool': 'OK', 'float32': 'OK', 'float16': 'OK'}

[不成立] 新增: 物理损失对 float mask 崩溃（已加固）
    损失内部已把 mask 统一转成计算 dtype 再相乘（不再用 `&` 位运算），long/int8/bool/float32/float16 均可用；同时会拒绝形状不符的 mask 并给出清晰错误。
  LightConeMaskEngine 各精度: float32=OK, float16=OK, bfloat16=OK

[不成立] 新增: -1e9 哨兵在 fp16 下不可表示（已改 -inf）
    掩码改用显式 -inf 且 dtype 跟随输入，fp32/fp16/bf16 下均可用；旧写法 -1e9 在 fp16 下会抛 RuntimeError（范围 ±65504）。

[不成立] 新增: collate 构建 coords_tensors 但未使用
    Train.py 构造 coords_tensors 后仍对原始 coords_raw 调用 pad_sequence；当前 dataset 已返回 Tensor 故能跑通，若记录仍是二维 list 会抛 TypeError。

[不成立] 新增: SubsequentProcessing 中的重复实现已删除
    损坏的重复实现（__init__ 只有省略号、self.W_q_s 未定义、space_norm_type='rms' 非法枚举、模块底部 test_normalizer() 导入副作用）已删除；SubsequentProcessing/__init__ 已不再导入它；正统实现在 Logic.CoreAttention 下保留=True。

[不成立] 新增: clip_value 冻结 space_scale（已修复）
    clip_value=4.0, 归一化后空间峰值=2.6667 -> 有余量，space_scale 可学习（此前 clip_value=2.0，峰值 2.667，梯度恰为 0）

==========================================================================
汇总
==========================================================================
  [不成立] P0-1 因果掩码被覆写
  [不成立] P0-3 物理损失脱离计算图（已修复）
  [不成立] P0-5 缓存切断本次 chunk 的 K/V 梯度
  [不成立] P0-6 跨样本缓存复用（已修复）
  [不成立] P0-4 验证集泄漏
  [不成立] P0-2 物理项完全失效
  [不成立] 新增: 物理损失对 float mask 崩溃（已加固）
  [不成立] 新增: -1e9 哨兵在 fp16 下不可表示（已改 -inf）
  [不成立] 新增: collate 构建 coords_tensors 但未使用
  [不成立] 新增: SubsequentProcessing 中的重复实现已删除
  [不成立] 新增: clip_value 冻结 space_scale（已修复）

  成立 0 / 不成立 11

```
