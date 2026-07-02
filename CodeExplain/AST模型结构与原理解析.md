## `AST` 模型结构与原理解析

本文档解释本项目中 [ASTModel](file:///c:/Ethan/code/cdssHomeWork20260604/ADD-RSC-homework20260604/models/ast.py) 的代码结构、核心原理，以及它在本项目中的输入输出数据流。

---

## 1. 先说结论

本项目里的 AST 是一个：

- 以 `DeiT base distilled patch16 384` 为基础
- 针对音频谱图重写 `PatchEmbed`
- 支持 AudioSet 预训练权重加载
- 最终输出 `768` 维特征向量

和本项目的 `ResNet50` 一样，它在这里也不是最终分类器本体，而是：

- 一个 feature extractor

真正分类仍由外部 `classifier` 完成，尽管 AST 内部保留了 `mlp_head`。

---

## 2. AST 的基本思想

AST 全称是 Audio Spectrogram Transformer。  
核心思想很简单：

> 把音频谱图切成很多小 patch，把每个 patch 当成一个 token，再像 Vision Transformer 一样做自注意力建模。

也就是说，AST 本质上是：

- 把谱图当作“音频图像”
- 用 Transformer 而不是 CNN 来建模时频关系

相比 CNN，Transformer 更擅长：

- 显式建模长距离依赖
- 让远距离时间位置之间直接交互
- 让不同频段之间直接建立关联

---

## 3. 代码入口在哪里

定义在 [models/ast.py](file:///c:/Ethan/code/cdssHomeWork20260604/ADD-RSC-homework20260604/models/ast.py#L29-L154)：

```python
class ASTModel(nn.Module):
```

它依赖一个自定义的 [PatchEmbed](file:///c:/Ethan/code/cdssHomeWork20260604/ADD-RSC-homework20260604/models/ast.py#L10-L26)：

```python
class PatchEmbed(nn.Module):
```

并在初始化时覆盖 timm 默认实现：

```python
timm.models.vision_transformer.PatchEmbed = PatchEmbed
```

这一步很关键，因为作者想让 ViT 能接收自定义音频谱图尺寸，而不是被原始图像尺寸限制住。

---

## 4. AST 的总体结构

从高层看，AST 的前向结构可以写成：

```text
input spectrogram
-> transpose
-> patch embedding
-> prepend cls/dist tokens
-> add positional embedding
-> transformer blocks
-> final norm
-> average(cls_token, dist_token)
-> 768-d feature
```

对应 [ASTModel.forward()](file:///c:/Ethan/code/cdssHomeWork20260604/ADD-RSC-homework20260604/models/ast.py#L133-L153)：

```python
x = x.transpose(2, 3)
B = x.shape[0]
x = self.v.patch_embed(x)

cls_tokens = self.v.cls_token.expand(B, -1, -1)
dist_token = self.v.dist_token.expand(B, -1, -1)
x = torch.cat((cls_tokens, dist_token, x), dim=1)
x = x + self.v.pos_embed
x = self.v.pos_drop(x)
for i, blk in enumerate(self.v.blocks):
    x = blk(x)
x = self.v.norm(x)
x = (x[:, 0] + x[:, 1]) / 2
return x
```

最终返回的是：

- `[B, 768]`

---

## 5. 自定义 `PatchEmbed` 做了什么

定义在 [PatchEmbed](file:///c:/Ethan/code/cdssHomeWork20260604/ADD-RSC-homework20260604/models/ast.py#L10-L26)：

```python
self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
```

forward：

```python
x = self.proj(x).flatten(2).transpose(1, 2)
```

这里的逻辑是：

1. 用一个 `Conv2d` 把输入谱图切成不重叠 patch
2. 每个 patch 线性映射成 `embed_dim=768`
3. 把二维 patch 网格摊平成一串 token

也就是说，patch embedding 实际上就是：

- “卷积版切 patch + 线性投影”

这是 ViT/DeiT 常见做法。

---

## 6. 本项目输入到 AST 的张量形状

从 [main.py](file:///c:/Ethan/code/cdssHomeWork20260604/ADD-RSC-homework20260604/main.py#L332-L335) 看，送入 backbone 前会先：

```python
denoise_images = denoise_images.unsqueeze(1)
features = model(denoise_images)
```

因此送入 AST 的张量通常是：

```text
[B, 1, 1024, 256]
```

而 AST forward 一上来会：

```python
x = x.transpose(2, 3)
```

变成：

```text
[B, 1, 256, 1024]
```

这说明当前实现里，进入 patch embed 前会交换后两个维度。  
因此代码里的“频率轴/时间轴”语义与注释不一定完全一致，但 patch 总数是一致的。

---

## 7. patch 是怎么切出来的

默认参数在 [ASTModel.__init__()](file:///c:/Ethan/code/cdssHomeWork20260604/ADD-RSC-homework20260604/models/ast.py#L41-L42)：

```python
fstride=16, tstride=16
```

并且 patch 大小固定为：

```python
kernel_size=(16, 16)
```

在本项目默认输入尺寸下：

- 输入到 patch embed 前大致是 `[B, 1, 256, 1024]`
- 用 `16x16` patch、stride 也是 `16x16`

所以 patch 网格大小是：

- 频率方向：`256 / 16 = 16`
- 时间方向：`1024 / 16 = 64`

patch 总数：

\[
16 \times 64 = 1024
\]

这也和代码中 AudioSet 位置编码 reshape 成 `16 x 64` 的逻辑一致：

```python
reshape(1, 768, 16, 64)
```

---

## 8. token 序列长度怎么来的

patch embedding 之后，`x` 的形状是：

```text
[B, 1024, 768]
```

然后再拼上两个特殊 token：

- `cls_token`
- `dist_token`

因此进入 transformer blocks 前，序列长度变为：

```text
[B, 1026, 768]
```

其中：

- 1024 个 patch token
- 1 个 class token
- 1 个 distillation token

---

## 9. 为什么有两个特殊 token

因为本项目底层用的是：

- `vit_deit_base_distilled_patch16_384`

这是 DeiT distilled 版本，所以除了常规 `cls_token` 外，还多了一个：

- `dist_token`

最后输出时，代码取二者平均：

```python
x = (x[:, 0] + x[:, 1]) / 2
```

因此最终全局表示不是只用 `cls_token`，而是：

- `cls_token` 和 `dist_token` 的平均

---

## 10. Transformer blocks 在做什么

AST 里的主干是：

```python
for i, blk in enumerate(self.v.blocks):
    x = blk(x)
```

这里的 `blk` 来自 timm 的 ViT/DeiT block，典型结构是：

```text
LayerNorm
-> Multi-Head Self-Attention
-> Residual
-> LayerNorm
-> MLP
-> Residual
```

原理上，它让每个 patch token 能与所有其他 patch token 直接交互。

对音频谱图来说，这意味着：

- 某一时刻的频带可直接关注较远时刻
- 不同频带之间可直接建立依赖

这就是 AST 相比 CNN 很大的特点。

---

## 11. 位置编码为什么重要

Transformer 本身不带空间位置信息，所以 AST 必须显式加入位置编码：

```python
x = x + self.v.pos_embed
```

否则模型只知道“有哪些 patch”，却不知道：

- patch 原来在谱图的哪个位置

而呼吸音谱图里，位置很重要：

- 不同时间位置可能对应吸气/呼气不同阶段
- 不同频率位置对应不同病理频段

所以位置编码决定了 AST 是否能理解时频布局。

---

## 12. 为什么 AST 要改位置编码

预训练 ViT/DeiT 的位置编码通常对应固定输入尺寸。  
而音频谱图尺寸往往和图像不同，所以 AST 里做了位置编码重建。

### 如果不使用 AudioSet 预训练

就直接新建一个可学习位置编码：

```python
new_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, embed_dim))
trunc_normal_(self.v.pos_embed, std=.02)
```

### 如果使用 AudioSet 预训练

则把原来的位置编码取出来，reshape 成二维网格，再双线性插值到当前 patch 网格大小：

```python
new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, 1024, 768).transpose(1, 2).reshape(1, 768, 16, 64)
new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
```

这一步的意义是：

- 尽量继承预训练模型的空间先验
- 同时适配新的谱图大小

---

## 13. AudioSet 预训练是怎么加载的

当 `audioset_pretrain=True` 时，代码会：

1. 先构造一个 `label_dim=527` 的 AST
2. 加载你提供的 AudioSet 权重
3. 只拷贝 shape 能匹配上的参数
4. 再把最终 `mlp_head` 改成当前任务需要的类别数

关键代码见 [ASTModel.__init__()](file:///c:/Ethan/code/cdssHomeWork20260604/ADD-RSC-homework20260604/models/ast.py#L82-L123)。

这说明本项目的 AST 迁移逻辑是：

- 底层 transformer 和 patch embedding 尽量继承 AudioSet 表征
- 最后任务头按当前 ICBHI 分类任务重新定义

---

## 14. `mlp_head` 在本项目里的角色

AST 内部定义了：

```python
self.mlp_head = nn.Sequential(
    nn.LayerNorm(self.original_embedding_dim),
    nn.Linear(self.original_embedding_dim, label_dim)
)
```

但在本项目训练主线里，真正使用方式是：

```python
classifier = deepcopy(model.mlp_head)
features = model(denoise_images)
output = classifier(features)
```

也就是说：

- AST backbone 本体 forward 只输出 `[B, 768]`
- `mlp_head` 被拷出来当外部分类头使用

这和 ResNet50 的设计风格保持一致。

---

## 15. AST 的输入输出形状总结

以本项目默认输入尺寸为例：

### 15.1 输入到 AST

```text
[B, 1, 1024, 256]
```

### 15.2 `transpose(2, 3)` 后

```text
[B, 1, 256, 1024]
```

### 15.3 patch embedding 后

```text
[B, 1024, 768]
```

### 15.4 拼接 `cls` 和 `dist` token 后

```text
[B, 1026, 768]
```

### 15.5 经过所有 transformer blocks 和 norm 后

仍然是：

```text
[B, 1026, 768]
```

### 15.6 取前两个 token 平均后

```text
[B, 768]
```

### 15.7 外部分类头后

若是 4 类任务：

```text
[B, 768] -> [B, 4]
```

---

## 16. AST 和 ResNet50 的核心差异

### ResNet50

- 基于卷积
- 擅长局部纹理提取
- 长距离依赖靠层层堆叠隐式获得

### AST

- 基于 self-attention
- patch 之间可全局交互
- 更擅长建模长程时频依赖

对呼吸音谱图来说，AST 的潜在优势在于：

- 某些异常模式的时间跨度较长
- 不同频带之间的协同关系重要
- Transformer 更容易直接看到全局结构

---

## 17. 当前实现里值得注意的点

### 17.1 timm 版本被锁死

代码里明确要求：

```python
assert timm.__version__ == '0.4.5'
```

说明实现对 timm 版本兼容性比较敏感。

### 17.2 forward 里做了轴交换

```python
x = x.transpose(2, 3)
```

这意味着代码注释中的 `input_fdim` / `input_tdim` 与实际进入 patch embed 时的轴含义要分开理解。

### 17.3 当前 stride 默认无重叠 patch

默认 `fstride=tstride=16`，而 patch 也是 `16x16`，因此 patch 之间默认不重叠。

### 17.4 backbone 只返回特征，不直接返回 logits

这和原始一些 AST 用法不同，本项目把分类头外置了。

---

## 18. 一句话总结

可以把本项目的 [ASTModel](file:///c:/Ethan/code/cdssHomeWork20260604/ADD-RSC-homework20260604/models/ast.py) 概括成：

> 它把单通道呼吸音谱图切成 `16x16` patch token，加入 `cls/dist` token 和位置编码后送入 DeiT-base Transformer blocks 做全局自注意力建模，最终输出 `768` 维全局音频表征，再由外部分类头完成呼吸音类别预测。

