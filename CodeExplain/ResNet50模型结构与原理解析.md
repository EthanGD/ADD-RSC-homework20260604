## `ResNet50` 模型结构与原理解析

本文档解释本项目中 [ResNet50](file:///c:/Ethan/code/cdssHomeWork20260604/ADD-RSC-homework20260604/models/resnet.py) 的代码结构、各阶段组成、输入输出张量形状，以及它在本项目里的实际角色。

---

## 1. 先说结论

本项目里的 `ResNet50` 不是一个“自带最终分类层”的完整分类器，而是一个：

- 以 `torchvision.models.resnet.ResNet` 为基础
- 输入改为单通道谱图
- 去掉最后全连接层
- 只输出 `2048` 维特征向量

真正的最终分类，是在 [main.py](file:///c:/Ethan/code/cdssHomeWork20260604/ADD-RSC-homework20260604/main.py#L246-L248) 里额外接的：

```python
classifier = nn.Linear(model.final_feat_dim, args.n_cls)
```

---

## 2. 代码入口在哪里

定义在 [models/resnet.py](file:///c:/Ethan/code/cdssHomeWork20260604/ADD-RSC-homework20260604/models/resnet.py#L10-L47)：

```python
class ResNet50(torchvision.models.resnet.ResNet):
```

初始化关键代码：

```python
super().__init__(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], norm_layer=norm_layer)
del self.fc
self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
self.final_feat_dim = 2048
```

这 4 行已经把它的骨架定死了：

- block 类型：`Bottleneck`
- block 数量：`[3, 4, 6, 3]`
- 输入通道：`1`
- 输出特征：`2048`

其中 `[3, 4, 6, 3]` 正是标准 ResNet-50 的层配置。

---

## 3. ResNet50 的总体结构

从高层看，ResNet50 结构可以写成：

```text
conv1 -> bn1 -> relu -> maxpool
-> layer1
-> layer2
-> layer3
-> layer4
-> avgpool
-> flatten
```

对应 [resnet.py](file:///c:/Ethan/code/cdssHomeWork20260604/ADD-RSC-homework20260604/models/resnet.py#L31-L47)：

```python
x = self.conv1(x)
x = self.bn1(x)
x = self.relu(x)
x = self.maxpool(x)

x = self.layer1(x)
x = self.layer2(x)
x = self.layer3(x)
x = self.layer4(x)

x = self.avgpool(x)
x = torch.flatten(x, 1)
return x
```

所以它的功能很明确：

- 从输入谱图中提取层次化视觉特征
- 最终输出全局 pooled feature vector

---

## 4. 为什么它适合音频谱图

虽然 ResNet 最早是为自然图像设计的，但谱图本质上也是一种二维结构：

- 一个轴表示时间
- 一个轴表示频率
- 局部纹理和边缘模式具有语义

对于呼吸音谱图来说：

- wheeze 往往表现为持续窄带结构
- crackle 往往表现为短促瞬时纹理

2D CNN 非常适合提这类局部时频模式。

所以在本项目里，ResNet50 实际上就是：

- 把 Mel/fbank 谱图当成单通道图像来建模

---

## 5. 本项目相对标准 ResNet50 改了什么

### 5.1 输入从 3 通道 RGB 改成 1 通道谱图

标准图像 ResNet 的第一层是：

- `Conv2d(3, 64, 7x7, stride=2)`

这里改成：

```python
self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
```

原因很直接：

- 谱图只有 1 个通道

### 5.2 删掉原始 `fc`

```python
del self.fc
```

因为本项目想把 backbone 和 classifier 解耦：

- backbone 只负责抽特征
- 最终分类头由外部决定

### 5.3 可选 ImageNet 预训练

如果传 `--imagenet_pretrained`，代码会下载官方 ResNet50 权重，并把原本的 3 通道 `conv1.weight` 平均成 1 通道：

```python
state_dict["conv1.weight"] = w.mean(dim=1, keepdim=True)
```

这是一种很常见的“把 RGB 预训练迁移到灰度/谱图输入”的做法。

---

## 6. Bottleneck block 是什么

本模型继承自 torchvision 的标准 `Bottleneck` 结构。  
每个 bottleneck block 可以简化理解为：

```text
1x1 conv -> 3x3 conv -> 1x1 conv
+ shortcut
```

其设计思想是：

- 第一个 `1x1`：压缩/调整通道
- 中间 `3x3`：做主要空间建模
- 最后一个 `1x1`：扩展回更高维通道
- 再与 shortcut 相加

这样做的好处：

- 计算更省
- 网络更深
- 残差连接让优化更稳定

---

## 7. ResNet 的核心原理：残差学习

ResNet 的核心不是卷积本身，而是：

\[
y = F(x) + x
\]

也就是让网络学习“残差映射”而不是完整映射。

直觉上：

- 如果一层不需要大改动，直接学一个接近 0 的残差就行
- 深层网络训练更容易
- 梯度传播更稳定

这也是 ResNet 能堆到 50 层、101 层甚至更深的重要原因。

---

## 8. 各 stage 的 block 数量

本项目里：

```python
super().__init__(Bottleneck, [3, 4, 6, 3], ...)
```

意味着 4 个 stage 分别有：

- `layer1`: 3 个 bottleneck
- `layer2`: 4 个 bottleneck
- `layer3`: 6 个 bottleneck
- `layer4`: 3 个 bottleneck

这正是标准 ResNet50 的配置。

---

## 9. 输入输出张量形状怎么变化

结合本项目默认输入尺寸，ResNet50 输入通常来自：

- DDL 输出后重新 `unsqueeze(1)` 的谱图

即：

```text
[B, 1, 1024, 256]
```

下面按 stage 跟踪张量形状。

### 9.1 输入

```text
[B, 1, 1024, 256]
```

### 9.2 `conv1`

`7x7, stride=2, padding=3`

输出：

```text
[B, 64, 512, 128]
```

### 9.3 `maxpool`

标准 ResNet 的 `3x3 maxpool, stride=2`

输出：

```text
[B, 64, 256, 64]
```

### 9.4 `layer1`

`layer1` 不再降采样，通道扩到 256：

```text
[B, 256, 256, 64]
```

### 9.5 `layer2`

首个 block stride=2，空间减半，通道到 512：

```text
[B, 512, 128, 32]
```

### 9.6 `layer3`

再次降采样：

```text
[B, 1024, 64, 16]
```

### 9.7 `layer4`

再次降采样：

```text
[B, 2048, 32, 8]
```

### 9.8 `avgpool`

全局平均池化后：

```text
[B, 2048, 1, 1]
```

### 9.9 `flatten`

最终输出：

```text
[B, 2048]
```

这就是本项目里 `model.final_feat_dim = 2048` 的来源。

---

## 10. 这个 `[B, 2048]` 后面怎么用

在 [main.py](file:///c:/Ethan/code/cdssHomeWork20260604/ADD-RSC-homework20260604/main.py#L334-L338) 中：

```python
features = model(denoise_images)
output = classifier(features)
```

对于 ResNet50：

- `features` 形状是 `[B, 2048]`
- `classifier = Linear(2048, n_cls)`
- 所以 `output` 形状是 `[B, n_cls]`

在默认 4 分类下：

```text
[B, 2048] -> [B, 4]
```

---

## 11. 为什么不直接用 backbone 自带 `fc`

因为项目想统一两种 backbone 的接口：

- ResNet50 输出 `[B, 2048]`
- AST 输出 `[B, 768]`

然后统一用外部 `classifier` 做最终分类。

这样做有几个好处：

- 结构更统一
- backbone 和分类头解耦
- 更方便替换不同 backbone
- 更方便加载/复用 backbone 预训练权重

---

## 12. `track_bn` 是什么

初始化里定义了：

```python
def norm_layer(*args, **kwargs):
    return nn.BatchNorm2d(*args, **kwargs, track_running_stats=track_bn)
```

这个参数控制：

- BN 是否记录 running mean / running var

默认 `track_bn=True`，也就是正常 BatchNorm 行为。

---

## 13. 从感受野角度看 ResNet50 在谱图上的作用

随着网络层数加深，ResNet50 会逐步扩大感受野：

- 前层更关注局部时频纹理
- 中层开始组合局部模式
- 深层更关注全局呼吸周期结构

放到呼吸音任务上可以这样理解：

- 前层识别 crackle/wheeze 的局部纹理
- 中层识别连续异常模式
- 后层整合整个周期的结构信息

这也是 ResNet50 在谱图分类上通常表现稳定的原因。

---

## 14. 在本项目里的优点和局限

### 优点

- 结构成熟，训练稳定
- 对 2D 谱图建模直接有效
- 有 ImageNet 预训练可迁移
- 与 DDL 前端结合简单

### 局限

- 本质仍是 CNN，对长距离依赖不如 Transformer 显式
- 时间轴与频率轴的关系主要靠卷积堆叠隐式学习
- 输入分辨率大时计算量不小

---

## 15. 一句话总结

可以把本项目的 [ResNet50](file:///c:/Ethan/code/cdssHomeWork20260604/ADD-RSC-homework20260604/models/resnet.py) 概括成：

> 一个把单通道呼吸音谱图作为二维图像处理的标准残差卷积骨干网络，经过 `conv1 + 4 个 bottleneck stage + 全局平均池化` 后输出 `2048` 维特征，再交给外部线性分类头完成最终呼吸音类别判别。

