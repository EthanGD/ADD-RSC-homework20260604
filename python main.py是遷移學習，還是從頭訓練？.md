用户问的是 `python main.py` 是迁移学习还是从头训练。我需要查看 main.py 中的相关代码来回答这个问题。

从之前读取的 main.py 代码中，我可以看到：
1. 第 107 行：`parser.add_argument('--model', type=str, default='resnet50')` - 默认是 resnet50
2. 第 108 行：`parser.add_argument('--pretrained', action='store_true')` - pretrained 是可选参数
3. 第 116-118 行：关于 AST 的预训练权重配置
4. 第 211-236 行：set_model 函数中，如果 model 是 'ast' 且有 audioset_pretrained=True，会使用预训练的 AST 模型

所以：
- 默认 `python main.py` 是从头训练（resnet50，没有预训练权重）
- 加上 `--pretrained` 和 `--audioset_pretrained` 参数才是迁移学习（使用预训练的 AST 模型）
</think>

**默认 `python main.py` 是从头训练（ResNet50）**。

**加参数才是迁移学习**：

```bash
# 从头训练（默认）
python main.py --model resnet50

# 迁移学习（使用预训练 AST 权重）
python main.py --model ast --pretrained --audioset_pretrained
```

**区别**：

| 模式 | 参数 | 模型初始化 |
|------|------|------------|
| 从头训练 | `--model resnet50` | 随机初始化 |
| 迁移学习 | `--model ast --pretrained --audioset_pretrained` | 加载 AudioSet 预训练权重 |

**代码位置** (`main.py:213-223`)：
```python
if args.model == 'ast':
    model = ASTModel(
        input_fdim=int(args.h * args.resz),
        input_tdim=int(args.w * args.resz),
        label_dim=args.n_cls,
        audioset_pretrain=args.audioset_pretrained,  # True 时加载预训练
        pretrained_path=args.audioset_ckpt
    )
```





Train epoch 1, total time 2383.75, accuracy:52.64

 * S_p: 100.00, S_e: 0.00, Score: 50.00 (Best S_p: 0.00, S_e: 0.00, Score: 0.00)
 * Acc@1 51.84

這個日誌什麼意思


用户展示了训练 epoch 1 完成后的验证日志。我需要解释这个日志的含义，特别是为什么 Sp=100%, Se=0% 这种极端情况。
</think>

**这是 Epoch 1 的验证结果**，但模型**完全没学会**：

| 指标 | 值 | 含义 |
|------|-----|------|
| $S_p$ (Specificity) | 100.00% | 正常样本全预测对 |
| $S_e$ (Sensitivity) | 0.00% | 异常样本全预测错 |
| Score | 50.00% | 平均分数 |
| Acc@1 | 51.84% | 准确率 |

---

## 问题原因

**模型把所有样本都预测为"正常"（normal）**：
- 正常样本占多数 → $S_p = 100\%$
- 异常样本全漏掉 → $S_e = 0\%$

这是**类别不平衡**导致的典型问题。

---

## 解决方案

```bash
CUDA_VISIBLE_DEVICES=6 python main.py \
    --model ast \
    --pretrained \
    --audioset_ckpt ./ast-finetuned-audioset-10-10-0.4593.pth \
    --tag ast_add \
    --weighted_sampler \          # 使用加权采样器
    --weighted_loss \             # 使用加权损失
    --epochs 50
```

**或调整学习率**：
```bash
--learning_rate 1e-4  # 从 5e-5 提高到 1e-4
```

---

## 后续 epoch 应该看到

正常训练时，$S_e$ 应该逐渐上升：
```
Epoch 2: S_e: 15%, Score: 55%
Epoch 5: S_e: 30%, Score: 60%
Epoch 10: S_e: 40%, Score: 63%
```

**当前程序继续运行即可**，第 2 个 epoch 应该会开始学习。









用户问程序何时保存模型，我刚刚读取了 main.py 的保存逻辑部分，现在需要解释三种保存时机。
</think>

## 模型保存时机

### 1️⃣ **最佳模型保存**（`best_epoch_X.pth`）
**条件**：验证分数 Score 更新时（`save_bool = True`）

```python
# main.py:475-478
if save_bool:            
    save_file = os.path.join(args.save_folder, 'best_epoch_{}.pth'.format(epoch))
    save_model(...)
```

**触发条件**（`validate` 函数中）：
```python
# main.py:413-417
if sc > best_acc[-1] and se > 5:  # Score 提升 且 Se > 5%
    save_bool = True
    best_acc = [sp, se, sc]
    best_model = [...]  # 保存模型状态
```

---

### 2️⃣ **定期保存**（`epoch_X.pth`）
**条件**：每 `save_freq` 个 epoch（默认 100）

```python
# main.py:480-482
if epoch % args.save_freq == 0:
    save_file = os.path.join(args.save_folder, 'epoch_{}.pth'.format(epoch))
    save_model(...)
```

---

### 3️⃣ **最终最佳模型**（`best.pth`）
**条件**：训练结束后

```python
# main.py:484-489
save_file = os.path.join(args.save_folder, 'best.pth')
model.load_state_dict(best_model[0])  # 加载最佳状态
save_model(...)
```

---

## 保存位置

```
save/
└── icbhi_ast_ast_add/
    ├── best_epoch_1.pth    # Epoch 1 最佳
    ├── best_epoch_5.pth    # Epoch 5 最佳
    ├── epoch_100.pth       # 定期保存
    └── best.pth            # 最终最佳模型
```

**当前程序**：Epoch 1 验证后，如果 Score > 50% 且 Se > 5%，会保存 `best_epoch_1.pth`