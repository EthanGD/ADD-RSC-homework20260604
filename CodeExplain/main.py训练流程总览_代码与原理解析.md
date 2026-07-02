## `main.py` 训练流程总览：代码与原理解析

本文档从整体视角解释本项目 [main.py](file:///c:/Ethan/code/cdssHomeWork20260604/ADD-RSC-homework20260604/main.py) 的训练流程，包括：

- 参数解析
- 数据加载
- 模型组装
- 训练/验证
- best checkpoint 保存

---

## 1. 整体流程一句话概括

本项目的训练主线可以概括为：

> 先把 ICBHI 呼吸音切成单周期谱图样本，经 DDL 去噪后送入 `ResNet50` 或 `AST` backbone 提特征，再经线性分类头输出类别；训练时同时优化主分类损失和去噪辅助损失，验证时按 `Sp / Se / Score` 选择 best ckpt。

---

## 2. 入口函数做了什么

程序入口在 [main()](file:///c:/Ethan/code/cdssHomeWork20260604/ADD-RSC-homework20260604/main.py#L447-L533)。

主流程顺序如下：

1. `parse_args()` 解析命令行参数
2. 将参数保存到 `train_args.json`
3. 固定随机种子
4. `set_loader(args)` 构建 train / val DataLoader
5. `set_model(args)` 构建 DDL、backbone、classifier、loss、optimizer
6. 创建 AMP `GradScaler`
7. 进入 epoch 训练循环
8. 周期性验证并按 `Score` 保存 best ckpt
9. 训练结束后再把 best 权重写成 `best.pth`
10. 将最终结果更新到 `results.json`

---

## 3. `parse_args()`：训练配置从哪里来

参数定义在 [parse_args()](file:///c:/Ethan/code/cdssHomeWork20260604/ADD-RSC-homework20260604/main.py#L26-L166)。

大致分成 4 组：

### 3.1 训练控制参数

- `--epochs`
- `--learning_rate`
- `--optimizer`
- `--cosine`
- `--warm`
- `--eval_freq`
- `--early_stop_patience`

作用是控制：

- 学习率策略
- 是否 warmup
- 多久验证一次
- 是否提前停止

### 3.2 数据参数

- `--data_folder`
- `--batch_size`
- `--sample_rate`
- `--desired_length`
- `--n_mels`
- `--weighted_sampler`
- `--raw_augment`
- `--specaug_policy`

作用是控制：

- 音频采样率
- 周期裁剪长度
- fbank 维度
- 增强方式
- 类别不平衡采样

### 3.3 去噪模块参数

- `--denoise_d_model`
- `--denoise_num_heads`
- `--denoise_depth`
- `--loss_beta`

作用是控制：

- DDL 的特征维度、head 数
- 去噪辅助损失权重

### 3.4 backbone 参数

- `--model`：`resnet50` 或 `ast`
- `--imagenet_pretrained`
- `--audioset_pretrained`
- `--audioset_ckpt`

作用是控制：

- 具体使用哪个 backbone
- 是否加载预训练权重

---

## 4. `set_loader()`：数据进入训练前做了什么

数据加载逻辑在 [set_loader()](file:///c:/Ethan/code/cdssHomeWork20260604/ADD-RSC-homework20260604/main.py#L169-L216)。

### 4.1 固定输入尺寸

```python
args.h, args.w = 1024, 256
```

也就是说，项目里默认把谱图输入尺寸视为：

- 高：`1024`
- 宽：`256`

### 4.2 训练集和验证集 transform

训练集：

```python
ToTensor -> SpecAugment -> Resize(1024, 256)
```

验证集：

```python
ToTensor -> Resize(1024, 256)
```

所以：

- 训练集有谱图增强
- 验证集没有增强

### 4.3 构建 `ICBHIDataset`

```python
train_dataset = ICBHIDataset(train_flag=True, ...)
val_dataset = ICBHIDataset(train_flag=False, ...)
```

这个数据集内部会完成：

- 文件扫描
- train / test 划分
- 单周期切分
- 定长处理
- fbank 生成

### 4.4 是否启用加权采样

如果设置 `--weighted_sampler`，则用：

```python
WeightedRandomSampler
```

去减轻类别不平衡影响。

---

## 5. `set_model()`：训练图是怎么拼起来的

模型组装逻辑在 [set_model()](file:///c:/Ethan/code/cdssHomeWork20260604/ADD-RSC-homework20260604/main.py#L219-L302)。

### 5.1 设备与主卡

如果设置了 `--gpu_ids`，代码会把第一个 id 作为主卡：

```python
primary_device = torch.device(f'cuda:{device_ids[0]}')
torch.cuda.set_device(primary_device)
```

### 5.2 先建 DDL

```python
bias_denoise_encoder = DiffTransformerLayer(...)
```

DDL 是主干网络前的去噪前端。

### 5.3 再建 backbone

如果 `--model ast`：

```python
model = ASTModel(...)
classifier = deepcopy(model.mlp_head)
```

如果 `--model resnet50`：

```python
model = ResNet50(...)
classifier = nn.Linear(model.final_feat_dim, args.n_cls)
```

注意这里的设计是：

- backbone 只输出特征
- 最终分类由外部 `classifier` 完成

### 5.4 定义两种 loss

```python
criterion = nn.CrossEntropyLoss(...)
denoise_criterion = LabelSmoothingLoss(size=args.n_cls)
criterion = [criterion, denoise_criterion]
```

也就是说训练同时优化：

- 主分类损失 `class_loss`
- 去噪辅助损失 `denoise_loss`

### 5.5 多卡并行

当前只有 `model` 会包装成：

```python
torch.nn.DataParallel(model, ...)
```

而 `bias_denoise_encoder` 和 `classifier` 不做 DataParallel，因此主卡通常更吃显存。

### 5.6 优化器参数范围

```python
optim_params = list(model.parameters()) + list(bias_denoise_encoder.parameters()) + list(classifier.parameters())
```

说明优化器会同时更新：

- backbone
- DDL
- classifier

---

## 6. 单个 batch 的训练数据流

训练主逻辑在 [train()](file:///c:/Ethan/code/cdssHomeWork20260604/ADD-RSC-homework20260604/main.py#L305-L376)。

核心前向如下：

```python
images = images.squeeze(1)
denoise_images, denoise_feature = bias_denoise_encoder(images)
denoise_images = denoise_images.unsqueeze(1)
features = model(denoise_images)
output = classifier(features)
denoise_loss = criterion[1](denoise_feature, labels)
class_loss = criterion[0](output, labels)
loss = args.loss_beta * denoise_loss + (1 - args.loss_beta) * class_loss
```

这段代码体现了完整的数据流：

1. `images` 从 `[B, 1, H, W]` squeeze 成 `[B, H, W]`
2. 送入 DDL，得到：
   - `denoise_images`：去噪后特征
   - `denoise_feature`：DDL 辅助分类输出
3. 把 `denoise_images` 再 unsqueeze 回 `[B, 1, H, W]`
4. 送入 `ResNet50` 或 `AST`
5. backbone 输出特征向量
6. 外部 `classifier` 输出最终类别 logits
7. 计算主分类损失与去噪损失并加权求和

---

## 7. 为什么训练里有两个损失

训练总损失为：

\[
L = \beta L_{denoise} + (1-\beta)L_{class}
\]

对应 [main.py](file:///c:/Ethan/code/cdssHomeWork20260604/ADD-RSC-homework20260604/main.py#L336-L338)。

分工是：

- `class_loss`：约束最终分类结果
- `denoise_loss`：约束 DDL 辅助分支保留类别信息

这意味着 DDL 不是纯粹“滤波器”，而是一个：

- 带辅助监督的去噪前端

---

## 8. 训练时还有哪些细节

### 8.1 warmup 学习率

每个 batch 前会调用：

```python
warmup_learning_rate(...)
```

### 8.2 AMP 混合精度

使用：

```python
with torch.cuda.amp.autocast():
```

并配合：

```python
GradScaler
```

### 8.3 Moving Average 更新

如果 `args.ma_update` 为真，会在每个 iteration 后执行：

```python
update_moving_average(...)
```

这相当于对参数做一次 EMA 风格更新。

---

## 9. `validate()`：验证时看哪些指标

验证逻辑在 [validate()](file:///c:/Ethan/code/cdssHomeWork20260604/ADD-RSC-homework20260604/main.py#L379-L444)。

验证阶段同样会走：

```python
DDL -> backbone -> classifier
```

并计算：

- `loss`
- `Acc@1`
- `Sp`
- `Se`
- `Score`

其中 `Sp / Se / Score` 来自 [get_score()](file:///c:/Ethan/code/cdssHomeWork20260604/ADD-RSC-homework20260604/util/icbhi_util.py) 的统计逻辑。

对于 ICBHI 4 类任务，核心评价会把：

- `normal` 视为正常类
- `crackle / wheeze / both` 视为异常类

然后计算：

- `Sp`：正常识别率
- `Se`：异常识别率
- `Score = (Sp + Se) / 2`

---

## 10. best checkpoint 是按什么保存的

保存条件在 [validate()](file:///c:/Ethan/code/cdssHomeWork20260604/ADD-RSC-homework20260604/main.py#L436-L439)：

```python
if sc > best_acc[-1] and se > 5:
    save_bool = True
```

也就是说 best ckpt 不是按 loss 选，而是按：

- `Score` 提升
- 且 `Se > 5`

这个 `se > 5` 的门槛其实是在避免保存那种：

- `Sp` 很高
- `Se` 几乎为 0

的“全判 normal”坏模型。

---

## 11. epoch 循环做了什么

训练循环在 [main()](file:///c:/Ethan/code/cdssHomeWork20260604/ADD-RSC-homework20260604/main.py#L485-L525)。

每个 epoch 的逻辑是：

1. `adjust_learning_rate(...)`
2. 调用 `train(...)`
3. 根据 `eval_freq` 决定是否 `validate(...)`
4. 如果刷新 best，则保存 `best_epoch_x.pth`
5. 如果达到 `save_freq`，保存周期性 checkpoint
6. 如果触发 `early_stop_patience`，提前结束

训练结束后还会：

1. 把 best 权重重新加载回 `model / DDL / classifier`
2. 写出最终 `best.pth`

---

## 12. `--eval` 模式和训练模式有什么不同

如果传了 `--eval`，则不会进入训练循环，而是直接：

```python
validate(val_loader, ...)
```

此时程序只做：

- 加载模型
- 前向验证
- 输出评估指标

---

## 13. 从张量角度看完整训练主链路

以默认输入尺寸为例，整体张量流可以写成：

```text
[B, 1, 1024, 256]      dataset 输出谱图
-> [B, 1024, 256]      squeeze(1)
-> DDL
-> [B, 1024, 256]      denoise_images
-> [B, 1, 1024, 256]   unsqueeze(1)
-> backbone
-> [B, 2048]           ResNet50 时
   或 [B, 768]         AST 时
-> classifier
-> [B, 4]              最终 logits
```

同时 DDL 还会额外输出：

```text
denoise_feature: [B, 4]
```

供 `denoise_loss` 使用。

---

## 14. 训练流程背后的设计思想

这份 `main.py` 体现的设计思路很清楚：

- 数据预处理尽量统一成固定大小谱图
- DDL 专门负责“前端去噪”
- backbone 专门负责“高层特征提取”
- 外部 classifier 负责“最终分类”
- 用辅助去噪损失防止 DDL 把有用异常特征一起抹掉
- 用 `Sp / Se / Score` 而不是单纯 `Acc` 作为主要评估标准

这特别适合呼吸音这种：

- 类别不平衡明显
- 噪声干扰大
- 异常模式比较细微

的任务。

---

## 15. 当前实现里值得注意的点

### 15.1 训练和验证的 loss 口径不一致

训练时：

```python
loss = beta * denoise_loss + (1-beta) * class_loss
```

验证时：

```python
loss = denoise_loss + class_loss
```

所以 train loss 和 val loss 数值不能直接横向对比。

### 15.2 只有 backbone 做了 DataParallel

DDL 和 classifier 没有 DataParallel，因此主卡显存压力更大。

### 15.3 `denoise_depth` 当前并不等于“真实堆叠层数”

它传给 DDL 后主要影响 `MHDA` 里的 `lambda_init`，当前并没有自动堆叠多层 DDL block。

---

## 16. 一句话总结

可以把 [main.py](file:///c:/Ethan/code/cdssHomeWork20260604/ADD-RSC-homework20260604/main.py) 总结成：

> 它把 ICBHI 呼吸音样本组织成一个“DDL 去噪前端 + ResNet50/AST backbone + 外部分类头”的联合训练系统，并通过主分类损失与去噪辅助损失共同优化，最终按 `Sp / Se / Score` 选出最优模型。

