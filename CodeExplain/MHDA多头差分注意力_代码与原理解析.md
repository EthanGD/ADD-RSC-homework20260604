## MHDA 多头差分注意力：代码与原理解析

本文档解释本项目中 MHDA（Multi-Head Differential Attention，多头差分注意力）的**代码实现**与其背后的**算法思想**。

对应代码位置：

- MHDA 主体实现：[adapt_diff_denoise.py](file:///c:/Ethan/code/cdssHomeWork20260604/ADD-RSC-homework20260604/models/adapt_diff_denoise.py#L153-L310)
- `lambda_init` 初始化函数：[lambda_init_fn()](file:///c:/Ethan/code/cdssHomeWork20260604/ADD-RSC-homework20260604/models/adapt_diff_denoise.py#L9-L10)
- MHDA 在 DDL 中的接入：[DiffTransformerLayer](file:///c:/Ethan/code/cdssHomeWork20260604/ADD-RSC-homework20260604/models/adapt_diff_denoise.py#L447-L506)

---

## 1. MHDA 在整个网络中的位置

在本项目里，MHDA 是差分去噪层（DDL）的核心模块。

在 [DiffTransformerLayer](file:///c:/Ethan/code/cdssHomeWork20260604/ADD-RSC-homework20260604/models/adapt_diff_denoise.py#L447-L506) 中：

```python
self.norm1 = RMSNorm(d_model)
self.attn = MultiHeadDifferentialAttention(d_model, num_heads, depth)
self.norm2 = RMSNorm(d_model)
self.ff = SwiGLU(d_model)
self.aff = AFNO1D(hidden_size=d_model)
```

forward 里它的调用顺序是：

```python
x1 = self.aff(x)
y = self.attn(self.norm1(x1)) + x1
z = self.ff(self.norm2(y)) + y
return z, x2
```

也就是说：

1. 先经过 AFF 做频域滤波
2. 再进入 MHDA 做差分注意力去噪
3. 再经过 SwiGLU 前馈网络

所以 MHDA 的角色不是单纯“提特征”，而是：

- 在注意力层面抑制噪声敏感变化
- 保留更稳定、更有判别性的时序依赖关系

---

## 2. 标准注意力和 MHDA 的区别

标准多头注意力（MHA）核心形式是：

\[
\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V
\]

它的问题是：

- 一路 attention 分布直接决定特征聚合
- 如果噪声影响了 `Q`、`K`，注意力图也会被噪声扰动

MHDA 的思路是：

- 不只算一张注意力图
- 而是算两张注意力图，再做差分

本项目代码中的核心形式是：

\[
A = \text{softmax}\left(\frac{Q_1K_1^\top}{\sqrt{d}}\right) -
\lambda \cdot
\text{softmax}\left(\frac{Q_2K_2^\top}{\sqrt{d}}\right)
\]

然后再乘上 `V`：

\[
O = A \cdot V
\]

也就是说：

- 第一条注意力支路提取“想保留的信息”
- 第二条注意力支路提取“想抵消的信息”
- 两者相减，从而实现一种“差分去噪”

---

## 3. 代码结构总览

MHDA 类定义在 [MultiHeadDifferentialAttention](file:///c:/Ethan/code/cdssHomeWork20260604/ADD-RSC-homework20260604/models/adapt_diff_denoise.py#L153-L310)：

```python
class MultiHeadDifferentialAttention(nn.Module):
```

主要包含：

- `W_q / W_k / W_v / W_o`：线性投影层
- `lambda_q1 / lambda_k1 / lambda_q2 / lambda_k2`：学习型差分系数参数
- `lambda_init`：与 depth 相关的初始偏置
- `rms_scale`：RMS 归一化的缩放参数

它的输入输出都是：

- 输入：`(batch, sequence_length, d_model)`
- 输出：`(batch, sequence_length, d_model)`

所以它是一个**保形注意力模块**，可以无缝插入 Transformer/DDL。

---

## 4. 为什么 Q、K、V 要投影到 `2 * d_head`

在初始化部分：

```python
self.W_q = nn.Linear(d_model, 2 * self.d_head * num_heads, bias=False)
self.W_k = nn.Linear(d_model, 2 * self.d_head * num_heads, bias=False)
self.W_v = nn.Linear(d_model, 2 * self.d_head * num_heads, bias=False)
```

相比标准 MHA，这里最明显的差别是：

- Q / K / V 都不是投到 `d_head`
- 而是投到 `2 * d_head`

原因是后面要把 Q 和 K 拆成两路：

```python
Q1, Q2 = Q.chunk(2, dim=-1)
K1, K2 = K.chunk(2, dim=-1)
```

也就是说：

- `Q1, K1` 负责第一张注意力图
- `Q2, K2` 负责第二张注意力图

这就是差分注意力的结构基础。

---

## 5. 输入如何变成多头形式

forward 一开始：

```python
Q = self.W_q(X)
K = self.W_k(X)
V = self.W_v(X)
```

然后 reshape：

```python
Q = Q.view(batch, N, self.num_heads, 2 * self.d_head).transpose(1, 2)
K = K.view(batch, N, self.num_heads, 2 * self.d_head).transpose(1, 2)
V = V.view(batch, N, self.num_heads, 2 * self.d_head).transpose(1, 2)
```

得到形状：

- `(batch, num_heads, N, 2*d_head)`

之后再切成两路：

```python
Q1, Q2 = Q.chunk(2, dim=-1)
K1, K2 = K.chunk(2, dim=-1)
```

每一路都是：

- `(batch, num_heads, N, d_head)`

这一步就正式形成了“**双路 attention 分支**”。

---

## 6. `lambda` 是怎么来的

差分注意力里最关键的系数是 `lambda`。  
代码中它不是固定常数，而是**可学习的**。

### 参数定义

```python
self.lambda_q1 = nn.Parameter(torch.empty(num_heads, self.d_head))
self.lambda_k1 = nn.Parameter(torch.empty(num_heads, self.d_head))
self.lambda_q2 = nn.Parameter(torch.empty(num_heads, self.d_head))
self.lambda_k2 = nn.Parameter(torch.empty(num_heads, self.d_head))
```

也就是说每个 head 都有一套自己的 `lambda` 参数。

### 计算方式

在 forward 中：

```python
lambda_q1_dot_k1 = torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()
lambda_q2_dot_k2 = torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()
lambda_val = torch.exp(lambda_q1_dot_k1) - torch.exp(lambda_q2_dot_k2) + self.lambda_init
```

数学上就是：

\[
\lambda =
\exp(q_{\lambda1}\cdot k_{\lambda1})
-
\exp(q_{\lambda2}\cdot k_{\lambda2})
+
\lambda_{init}
\]

### 为什么这样设计

这相当于给每个 head 一个可学习的门控系数：

- 控制第二路注意力要减掉多少
- 让模型自己学“抑制噪声”的强度

如果 `lambda` 很大：

- 第二条支路影响更强
- 差分抑制更激进

如果 `lambda` 较小：

- 更接近保守的差分

---

## 7. `lambda_init` 的作用是什么

初始化函数定义在 [lambda_init_fn()](file:///c:/Ethan/code/cdssHomeWork20260604/ADD-RSC-homework20260604/models/adapt_diff_denoise.py#L9-L10)：

```python
def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)
```

它和 `depth` 有关，直觉上是：

- 不同层深度允许不同初始差分强度
- 深层可能允许更强/更稳定的差分行为

不过你当前代码里 `depth` 虽然来自 `denoise_depth` 参数，但在实现上：

- 它只是用来生成一个 `lambda_init`
- 并不真的表示“堆叠了多少层 MHDA”

---

## 8. 两张注意力图是怎么计算的

代码中：

```python
scaling = 1 / math.sqrt(self.d_head)
A1 = torch.matmul(Q1, K1.transpose(-2, -1)) * scaling
A2 = torch.matmul(Q2, K2.transpose(-2, -1)) * scaling
attention1 = F.softmax(A1, dim=-1)
attention2 = F.softmax(A2, dim=-1)
```

这和标准注意力完全一致，只不过现在不是一张图，而是两张：

- `attention1`
- `attention2`

它们的形状都是：

- `(batch, num_heads, N, N)`

每一张都表示：

- 每个 token 对其他 token 的依赖权重

---

## 9. 差分注意力的核心公式在代码里是哪一行

核心就是这句：

```python
attention = attention1 - lambda_val * attention2
```

对应数学表达：

\[
\text{Attention}_{diff} = \text{Attention}_1 - \lambda \cdot \text{Attention}_2
\]

这是 MHDA 和普通 MHA 最本质的差别。

### 这个差分的直觉是什么

可以把它理解成：

- `attention1` 表示“主信息关注”
- `attention2` 表示“噪声敏感关注”或“干扰关注”
- `attention1 - lambda * attention2` 就是在做一种“保留主结构，抵消扰动”的过程

因此 MHDA 的目标不是单纯加强注意力，而是：

- **对噪声做显式的差分抑制**

---

## 10. 为什么后面还做了一次 LayerNorm

代码中有：

```python
norm_shape = (N,)
layer_norm = nn.LayerNorm(norm_shape, elementwise_affine=True).to(X.device)

attention_reshaped = attention.view(-1, *norm_shape)
attention_normalized = layer_norm(attention_reshaped)
attention_normalized = attention_normalized.view(attention.shape)
```

从代码意图看，这是想对差分后的 attention 做归一化，减少数值波动。

### 但要注意

当前实现里：

```python
O = torch.matmul(attention, V)
```

真正用于后续乘 `V` 的其实仍然是 **`attention`**，而不是 `attention_normalized`。

也就是说：

- 代码算出了 `attention_normalized`
- 但没有实际用它替换原始 `attention`

因此这段归一化目前更像是：

- 中间实验残留
- 或者未完成的稳定化逻辑

---

## 11. 差分注意力如何作用到 Value

标准注意力最后一步是：

\[
O = \text{Attention} \cdot V
\]

这里也是一样：

```python
O = torch.matmul(attention, V)
```

区别只是：

- 这里的 `attention` 已经不是普通 softmax 权重
- 而是差分后的权重图

这意味着最终输出不是：

- “直接按相似度聚合 value”

而是：

- “按差分后的选择性注意力聚合 value”

---

## 12. 为什么后面不是 LayerNorm，而是 RMS 风格归一化

得到 `O` 后，代码并没有直接拼接 heads，而是先做：

```python
O_reshaped = O.contiguous().view(batch * self.num_heads, N, 2 * self.d_head)
rms_norm = torch.sqrt(O_reshaped.pow(2).mean(dim=-1, keepdim=True) + self.eps)
O_normalized = (O_reshaped / rms_norm) * self.rms_scale
```

这就是一种 RMSNorm 思想：

- 只按均方根做缩放
- 不显式减去均值

### 为什么这么做

直觉上：

- 差分注意力可能带来更大的数值波动
- RMSNorm 计算更简单，稳定性也不错

之后还会乘上：

```python
O_normalized = O_normalized * (1 - self.lambda_init)
```

相当于再做一次强度缩放。

---

## 13. 最后如何回到原维度

多头输出先 reshape 回来：

```python
O_concat = O_normalized.transpose(1, 2).contiguous().view(batch, N, self.num_heads * 2 * self.d_head)
```

然后做最终线性投影：

```python
out = self.W_o(O_concat)
```

把维度重新投回：

- `(batch, N, d_model)`

所以 MHDA 最终输出和普通多头注意力一样，仍然可以作为 Transformer block 里的注意力模块替代品。

---

## 14. MHDA 在 DDL 里的残差结构

在 [DiffTransformerLayer.forward()](file:///c:/Ethan/code/cdssHomeWork20260604/ADD-RSC-homework20260604/models/adapt_diff_denoise.py#L475-L506) 中：

```python
y = self.attn(self.norm1(x1)) + x1
z = self.ff(self.norm2(y)) + y
```

说明 MHDA 不是裸用，而是放在标准的残差结构中：

- `Norm -> MHDA -> Add`

这有两个好处：

- 训练更稳定
- MHDA 学不好时不会完全毁掉原始特征

---

## 15. 用一句话理解 MHDA

可以把 MHDA 理解成：

> 不是只用一张 softmax 注意力图来聚合信息，而是同时学习两张注意力图，并用可学习系数做差分，从而压掉噪声敏感的关系，保留更稳定、更有诊断意义的依赖结构。

---

## 16. 它为什么适合呼吸音去噪

呼吸音分类中的难点是：

- 噪声很多：环境噪声、心音、设备噪声
- 有些噪声和病理特征可能会混叠

传统注意力会把所有显著相似性都学进去，可能连噪声关系也一起强化。  
MHDA 的设计则更偏向：

- 第一条支路：保留稳定、有效的模式
- 第二条支路：学习那些更像“干扰”的依赖
- 再通过差分把后者削弱

因此它特别适合论文中强调的目标：

- 在没有干净标签的情况下，提升对噪声的鲁棒性

---

## 17. 当前实现里几个值得注意的地方

### 17.1 `attention_normalized` 没有真正参与后续计算

代码做了：

```python
attention_normalized = ...
```

但后面仍然用的是：

```python
O = torch.matmul(attention, V)
```

说明这个归一化结果当前没有被实际利用。

### 17.2 `depth` 不是“层数”，只是影响 `lambda_init`

虽然参数名叫 `denoise_depth`，但在 MHDA 内：

- 它只影响 `lambda_init`
- 不会自动堆叠出多层 MHDA

### 17.3 `V` 没有拆成两路

Q 和 K 被拆成两路，但 V 没拆，仍然是统一的 value 投影：

```python
V = self.W_v(X)
```

这说明差分主要发生在：

- 注意力权重层面

而不是 value 表示层面。

---

## 18. 最后用更直觉的话说

如果把标准注意力想成：

- “看哪里重要，就把哪里聚合过来”

那么 MHDA 更像：

- “一边看哪里重要，一边看哪里可能是噪声关注点”
- “再把第二种关注从第一种关注里减掉”

所以它不是简单增强注意力，而是在做：

**注意力层面的去噪。**

