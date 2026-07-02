## AST base 下一次训练调参建议

本文档基于你给出的 3 次 AST 训练记录，以及当前训练代码 [main.py](file:///c:/Ethan/code/cdssHomeWork20260604/ADD-RSC-homework20260604/main.py) 和优化器逻辑 [misc.py](file:///c:/Ethan/code/cdssHomeWork20260604/ADD-RSC-homework20260604/util/misc.py#L60-L76)，总结下一次训练更值得优先尝试的调参方向。

目标不是“把所有参数都改一遍”，而是优先从**最可能提升 `Score` 的参数**入手。

---

## 1. 先看这 3 次训练的结论

你给出的 3 次结果可以先整理成下面这张表：

| 运行 | 关键参数 | 训练速度 | 最佳/当前表现 | 结论 |
| --- | --- | --- | --- | --- |
| A | `batch_size=64` `weighted_sampler` `weighted_loss` | 约 `110s/epoch` | `Best Score=46.88`，`Sp=41.02`，`Se=52.75` | **目前最好**，说明大 batch 更稳 |
| B | `batch_size=32` `weighted_sampler` `weighted_loss` | 约 `152s/epoch` | `Best Score=40.52`，`Sp=27.44`，`Se=53.60` | 比 64 差，主要差在 `Sp` |
| C | `batch_size=8` `weighted_sampler` `weighted_loss` | 约 `323s/epoch` | `Best Score=38.57`，`Sp=20.83`，`Se=56.30` | 最慢，且 `Sp` 最差 |

从这 3 次结果可以直接得出 4 个结论：

- `batch_size=64` 明显优于 `32/8`，而且训练更快。
- 当前问题不是 `Se` 太低，而是 **`Sp` 明显偏低**，也就是模型太容易把正常判成异常。
- `weighted_sampler + weighted_loss` 这套组合把异常召回拉起来了，但也把正常类压得比较狠。
- `--imagenet_pretrained` 对 AST **没有实际作用**，因为 [main.py](file:///c:/Ethan/code/cdssHomeWork20260604/ADD-RSC-homework20260604/main.py#L237-L248) 里这个参数只在 `resnet50` 分支被使用。

所以你下一次调参的核心方向应该是：

> **保住当前已经不错的 `Se`，重点想办法把 `Sp` 拉上来。**

---

## 2. 代码层面会影响调参判断的几个点

### 2.1 当前优化器是 `AdamW`

在 [set_optimizer()](file:///c:/Ethan/code/cdssHomeWork20260604/ADD-RSC-homework20260604/util/misc.py#L60-L76) 中，`optimizer='adam'` 实际对应：

```python
optimizer = optim.AdamW(...)
```

所以你现在不是普通 Adam，而是：

- `AdamW + cosine lr`

这类配置对 batch size 和 warmup 比较敏感。

### 2.2 默认学习率是 `5e-5`

见 [parse_args()](file:///c:/Ethan/code/cdssHomeWork20260604/ADD-RSC-homework20260604/main.py#L42-L61)：

```python
parser.add_argument('--learning_rate', type=float, default=5e-5)
```

这个学习率对 AST 来说偏保守，尤其是当你已经把 batch 提到 64 时，可以试着略微加大。

### 2.3 `loss_beta=0.5`

见 [parse_args()](file:///c:/Ethan/code/cdssHomeWork20260604/ADD-RSC-homework20260604/main.py#L104-L108)：

```python
parser.add_argument('--loss_beta', type=float, default=0.5)
```

训练时总损失是：

```python
loss = args.loss_beta * denoise_loss + (1 - args.loss_beta) * class_loss
```

这意味着目前：

- `denoise_loss` 和 `class_loss` 权重一半一半

如果当前表现是 `Se` 比较高、`Sp` 偏低，那么有可能：

- 去噪辅助分支约束偏强
- 再叠加 `weighted_sampler + weighted_loss`
- 导致模型整体更偏向“宁可判异常，也不要漏异常”

### 2.4 验证指标看的是 `Score`

best ckpt 在 [validate()](file:///c:/Ethan/code/cdssHomeWork20260604/ADD-RSC-homework20260604/main.py#L436-L439) 里按这个条件更新：

```python
if sc > best_acc[-1] and se > 5:
```

也就是说你调参时最该盯的是：

- `Sp`
- `Se`
- `Score`

不是单纯看 train acc。

---

## 3. 下一次训练最建议优先改什么

下面按“优先级从高到低”给建议。

### 建议 1：固定 `batch_size=64`，不要再退回 32 或 8

这是你这 3 次实验里最明确的结论。

原因：

- `64` 跑得最快
- `64` 的 `Sp` 最好
- `64` 的综合 `Score` 也是最好

所以后续 AST 调参，建议把：

```bash
--batch_size 64
```

作为基线固定住。

---

### 建议 2：去掉 `--imagenet_pretrained`

你现在的 AST 命令里带了：

```bash
--imagenet_pretrained
```

但这个参数只对 `resnet50` 分支生效，对 AST 没有效果。  
所以下次训练建议直接去掉，避免误判实验差异来源。

也就是说，AST 分支更干净的写法应是：

```bash
python main.py --model ast --audioset_ckpt ...
```

而不是再加 `--imagenet_pretrained`。

---

### 建议 3：先保留 `weighted_sampler`，优先尝试去掉 `weighted_loss`

你当前最大问题是：

- `Se` 已经不低
- `Sp` 明显偏低

这通常意味着模型对异常类“过度偏向”。  
而你目前同时开启了：

- `--weighted_sampler`
- `--weighted_loss`

这属于比较激进的“双重平衡”。

建议下一次优先试：

```bash
保留 --weighted_sampler
去掉 --weighted_loss
```

原因：

- `weighted_sampler` 仍能保证 batch 内类别更均衡
- 但去掉 `weighted_loss` 后，主损失不会再额外强化异常类
- 更有机会把 `Sp` 拉起来

这一步是我最推荐的下一组实验。

---

### 建议 4：把 `loss_beta` 从 `0.5` 下调到 `0.3` 或 `0.35`

当前总损失：

\[
loss = 0.5 \cdot denoise\_loss + 0.5 \cdot class\_loss
\]

如果模型现在偏向异常类，一个很合理的怀疑是：

- 去噪辅助分类分支影响偏大

所以建议下一次直接试：

```bash
--loss_beta 0.35
```

如果还觉得 `Sp` 不够，再试：

```bash
--loss_beta 0.30
```

这个调整的目标不是提升 train acc，而是：

- 让最终分类头 `class_loss` 更主导训练
- 减少 DDL 辅助头对决策边界的过强牵引

---

### 建议 5：大 batch 下加 `--warm`

你现在已经用到 `batch_size=64`。  
而代码里已经支持 warmup，见 [parse_args()](file:///c:/Ethan/code/cdssHomeWork20260604/ADD-RSC-homework20260604/main.py#L142-L150) 与 [warmup_learning_rate()](file:///c:/Ethan/code/cdssHomeWork20260604/ADD-RSC-homework20260604/util/misc.py#L50-L57)。

建议下一次在大 batch 配置下补上：

```bash
--warm
```

这样通常能让前期收敛更稳定，减少：

- 前几轮振荡
- 因 batch 变大带来的学习率不适配

---

### 建议 6：学习率优先试 `8e-5`，再试 `1e-4`

当前默认是：

```bash
--learning_rate 5e-5
```

而 batch 已经增大到 64，通常可以尝试略微提高学习率。  
建议顺序是：

1. 先试 `8e-5`
2. 如果训练稳定，再试 `1e-4`

不建议一下子改太大，因为当前模型已经能稳定训练到后期，并不是完全不收敛的问题。

所以建议优先：

```bash
--learning_rate 8e-5
```

---

### 建议 7：`early_stop_patience` 从 `4` 提到 `6`

你现在设置是：

```bash
--eval_freq 2 --early_stop_patience 4
```

这意味着：

- 连续 4 次验证不提升就停
- 实际相当于约 8 个 epoch 没提升就停

AST 的验证分数通常会有一定波动，特别是你还在用：

- weighted sampler
- denoise loss
- cosine lr

所以建议下一次改成：

```bash
--early_stop_patience 6
```

这样更不容易因为中期波动提前停掉。

---

## 4. 我最推荐的下一次训练命令

如果只让我选一条“下一次最值得试”的命令，我推荐：

```bash
python main.py \
  --model ast \
  --audioset_ckpt ./ast-finetuned-audioset-10-10-0.4593.pth \
  --weighted_sampler \
  --batch_size 64 \
  --learning_rate 8e-5 \
  --loss_beta 0.35 \
  --warm \
  --eval_freq 2 \
  --early_stop_patience 6 \
  --tag ast_bs64_ws_lr8e5_beta035
```

这条命令相对你当前 best 配置，核心变化只有 4 个：

- 去掉 `--weighted_loss`
- `lr: 5e-5 -> 8e-5`
- `loss_beta: 0.5 -> 0.35`
- 加 `--warm`

这是一个“改动不大，但最有希望拉高 `Sp` 和 `Score`”的组合。

---

## 5. 如果这条还不够好，第二组怎么试

如果上一组结果出现：

- `Sp` 提升了
- 但 `Se` 掉太多

那么第二组建议是把 `weighted_loss` 加回来，但继续保留较低的 `loss_beta`：

```bash
python main.py \
  --model ast \
  --audioset_ckpt ./ast-finetuned-audioset-10-10-0.4593.pth \
  --weighted_sampler \
  --weighted_loss \
  --batch_size 64 \
  --learning_rate 8e-5 \
  --loss_beta 0.30 \
  --warm \
  --eval_freq 2 \
  --early_stop_patience 6 \
  --tag ast_bs64_wswl_lr8e5_beta03
```

这组的思路是：

- 保留你当前对异常类的召回能力
- 但通过降低 `loss_beta`，减少 DDL 辅助损失对整体边界的影响

---

## 6. 如果你想做最小实验量的 3 组对照

为了不把实验矩阵开太大，我建议你下一轮只做这 3 组：

### 组 A：首推

```bash
python main.py --model ast --audioset_ckpt ./ast-finetuned-audioset-10-10-0.4593.pth --weighted_sampler --batch_size 64 --learning_rate 8e-5 --loss_beta 0.35 --warm --eval_freq 2 --early_stop_patience 6 --tag ast_A
```

目的：

- 优先看 `Sp` 能不能显著回升

### 组 B：如果 A 的 `Se` 掉太多

```bash
python main.py --model ast --audioset_ckpt ./ast-finetuned-audioset-10-10-0.4593.pth --weighted_sampler --weighted_loss --batch_size 64 --learning_rate 8e-5 --loss_beta 0.30 --warm --eval_freq 2 --early_stop_patience 6 --tag ast_B
```

目的：

- 兼顾 `Se`

### 组 C：如果 A 和 B 都不稳定

```bash
python main.py --model ast --audioset_ckpt ./ast-finetuned-audioset-10-10-0.4593.pth --weighted_sampler --batch_size 64 --learning_rate 5e-5 --loss_beta 0.35 --warm --eval_freq 2 --early_stop_patience 6 --tag ast_C
```

目的：

- 排除“lr 提高过快”的影响

这样你只要再跑 3 组，就能基本判断：

- 问题主要来自 `weighted_loss`
- 还是主要来自 `loss_beta`
- 还是学习率本身不合适

---

## 7. 不建议优先改的参数

下面这些参数，我不建议你下一轮优先动：

### `batch_size`

已经证明 `64` 最优，先固定。

### `denoise_d_model / denoise_num_heads / denoise_depth`

这是结构级参数，不是本轮最值得优先碰的。  
先把训练策略调顺，再考虑结构。

### `specaug_policy`

当前问题更像是类别偏置与损失权重问题，不像是增强不够。

### `optimizer`

当前 `AdamW + cosine` 是合理基线，先别动。

---

## 8. 你现在最该盯哪几个指标

下一次训练时，不要只盯：

- `Train accuracy`

更要重点盯这 3 个：

1. `Sp` 是否从 `41` 附近继续抬升
2. `Se` 是否还能稳在 `50+`
3. `Score` 是否突破 `46.88`

你当前最优结果其实说明：

- 模型已经学到异常了

所以再往上走，关键不是继续堆 `Se`，而是：

- **把正常类识别拉回来**

---

## 9. 一句话总结

基于你这 3 次 AST 训练结果，下一次最值得优先尝试的方向是：

> 固定 `batch_size=64`，去掉对 AST 无效的 `--imagenet_pretrained`，优先把 `weighted_sampler + weighted_loss` 调整为只保留 `weighted_sampler`，同时把 `loss_beta` 从 `0.5` 下调到 `0.35`、加上 `--warm`，并把学习率先试到 `8e-5`，重点观察 `Sp` 能否提升并推动 `Score` 超过当前 best 的 `46.88`。

