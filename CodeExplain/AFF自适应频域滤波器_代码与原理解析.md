## AFF 自适应频域滤波器：代码与原理解析

本文档解释本项目中 AFF（Adaptive Frequency Filter，自适应频域滤波器）的**代码实现**与其背后的**算法直觉**。

对应代码位置：

- AFF 主体实现：[adapt_diff_denoise.py](file:///c:/Ethan/code/cdssHomeWork20260604/ADD-RSC-homework20260604/models/adapt_diff_denoise.py#L13-L89)
- AFF 在去噪层中的接入：[DiffTransformerLayer](file:///c:/Ethan/code/cdssHomeWork20260604/ADD-RSC-homework20260604/models/adapt_diff_denoise.py#L447-L506)

---

## 1. AFF 在整个模型里的位置

在本项目里，AFF 不是单独拿出来分类，而是作为 `DiffTransformerLayer` 的第一步预处理：

```python
self.aff = AFNO1D(hidden_size=d_model)

def forward(self, x):
    x1 = self.aff(x)
    x2 = (x1[:, 0] + x1[:, 1]) / 2
    x2 = self.classifier(x2)
    y = self.attn(self.norm1(x1)) + x1
    z = self.ff(self.norm2(y)) + y
    return z, x2
```

也就是说，输入特征先经过 AFF 做一次**频域滤波/稀疏化去噪**，再送进后面的差分注意力（MHDA）和前馈网络。

可以把它理解成：

- AFF：先在频域里压噪声、保留有用频率模式
- MHDA：再在时序/特征交互层面继续抑制噪声扰动

---

## 2. AFF 的输入输出是什么

在 [AFNO1D.forward()](file:///c:/Ethan/code/cdssHomeWork20260604/ADD-RSC-homework20260604/models/adapt_diff_denoise.py#L41-L89) 里：

```python
B, N, C = x.shape
```

表示输入 `x` 的形状是：

- `B`：batch size
- `N`：序列长度
- `C`：通道数 / 特征维度

AFF 的输出形状与输入相同，最后通过残差连接：

```python
return x + bias
```

这说明 AFF 做的是一种**保形变换**：

- 不改变张量尺寸
- 只改变每个位置上的特征内容

---

## 3. 核心思想：为什么要转到频域

AFF 的关键思想是：

- 噪声和有用信号在频域上的表现 often 不同
- 在频域里更容易做“保留哪些频率、压制哪些频率”的操作

代码中第一步就是做实数 FFT：

```python
x = torch.fft.rfft(x, dim=1, norm="ortho")
```

这里是沿着 `dim=1` 做 1D FFT，也就是沿着序列长度 `N` 的方向变换。  
变换后，原始的时序/序列特征被映射成频域表示。

### 为什么是 `rfft`

因为输入是实数张量，`rfft` 会只保留非冗余频谱部分，因此输出长度是：

```python
N // 2 + 1
```

这也是后面频域张量 reshape 的依据。

---

## 4. 代码如何把频域特征分块处理

FFT 后代码做了 reshape：

```python
x = x.reshape(B, N // 2 + 1, self.num_blocks, self.block_size)
```

其中：

- `num_blocks=8`
- `block_size = hidden_size // num_blocks`

这一步的含义是：

- 把通道维 `C` 切成多个 block
- 每个 block 独立学习一组较小的频域线性变换

这么做的好处：

- 参数量更小
- 计算更便宜
- 比起对整个大通道矩阵做全连接变换，更稳定

这就是 AFNO 思路里的“block diagonal”近似做法。

---

## 5. 代码如何处理复数频谱

FFT 的输出是复数，因此要分别处理：

- 实部 `real`
- 虚部 `imag`

第一层变换：

```python
o1_real[:, :kept_modes] = F.relu(
    torch.einsum('...bi,bio->...bo', x[:, :kept_modes].real, self.w1[0]) -
    torch.einsum('...bi,bio->...bo', x[:, :kept_modes].imag, self.w1[1]) +
    self.b1[0]
)

o1_imag[:, :kept_modes] = F.relu(
    torch.einsum('...bi,bio->...bo', x[:, :kept_modes].imag, self.w1[0]) +
    torch.einsum('...bi,bio->...bo', x[:, :kept_modes].real, self.w1[1]) +
    self.b1[1]
)
```

这其实是在做一种**复数线性映射**。

如果把复数写成：

\[
z = a + ib
\]

那么复数乘法会自然对应到“实部和虚部的交叉组合”。  
这里的代码正是在模拟这种结构，只不过参数是可学习的。

第二层变换也同理：

```python
o2_real[:, :kept_modes] = ...
o2_imag[:, :kept_modes] = ...
```

所以 AFF 的频域滤波不是手工写死的，而是：

- **通过可学习参数 `w1/w2/b1/b2` 自动学出来的频域变换器**

---

## 6. `kept_modes`：为什么只处理一部分频率

代码里有：

```python
total_modes = N // 2 + 1
kept_modes = int(total_modes * self.hard_thresholding_fraction)
```

默认 `hard_thresholding_fraction=1`，意味着保留全部频率；  
如果把它设得更小，则只处理前一部分频率分量，其余频率相当于被硬截断。

这背后的直觉是：

- 低频区往往更重要
- 高频有时更容易带噪
- 只处理部分频谱可以进一步降计算量

不过在你当前代码默认值里，这个“硬阈值”功能实际上**没有真正截断掉频率**，因为它保留的是全部 modes。

---

## 7. `softshrink`：AFF 最像“滤波器”的地方

AFF 最关键的一行是：

```python
x = F.softshrink(x, lambd=self.sparsity_threshold)
```

这里默认：

- `sparsity_threshold = 0.01`

`softshrink` 的定义可理解为：

\[
S_\lambda(x)=
\begin{cases}
x-\lambda, & x>\lambda \\
0, & |x|\le \lambda \\
x+\lambda, & x<-\lambda
\end{cases}
\]

它的作用是：

- 小幅值分量直接压成 0
- 大幅值分量保留，但幅度略收缩

### 为什么这像去噪

很多噪声在频域里表现为：

- 幅值较小
- 分布分散
- 不够稳定

而真正有诊断价值的结构性频谱成分更可能：

- 幅值更明显
- 更有规律

所以 `softshrink` 会天然倾向于：

- 压掉弱小噪声分量
- 保留重要频率成分

这就是“自适应频域滤波器”里“滤波”的主要来源。

---

## 8. 为什么说它是“自适应”的

传统滤波器通常是：

- 固定截止频率
- 固定通带/阻带
- 不随输入变化

而 AFF 的“自适应”体现在：

- 频域变换参数 `w1/w2` 是训练学出来的
- 输入样本不同，经过线性映射和非线性之后，保留/抑制的频谱模式也不同

也就是说，它不是手工指定：

- “100Hz 以下保留，500Hz 以上砍掉”

而是让模型自己学：

- “哪些频域结构更可能是有用信号，哪些更可能是噪声”

因此它本质上是一个**数据驱动的频域滤波器**。

---

## 9. 从频域回到时域

经过复数重组后，代码执行：

```python
x = torch.view_as_complex(x)
x = x.reshape(B, N // 2 + 1, C)
x = torch.fft.irfft(x, n=N, dim=1, norm="ortho")
```

作用就是：

- 把处理后的频域信号重新还原回原始序列空间

换句话说，AFF 做的是：

1. 输入序列特征
2. 转到频域
3. 在频域里做可学习变换 + 稀疏化
4. 再变回原始特征空间

这是一个典型的“频域处理模块”。

---

## 10. 残差连接为什么重要

最后一行：

```python
return x + bias
```

这说明 AFF 不是“完全替换原特征”，而是：

- 在原特征基础上做一个频域修正

优点：

- 不容易把原本有用的信息完全破坏掉
- 训练更稳定
- 如果频域滤波学得不好，网络至少还能退化成“接近恒等映射”

这也是现代深度网络里很常见的设计。

---

## 11. 用一句话概括 AFF 的代码逻辑

可以把 AFF 概括成：

> 沿序列维度做 1D FFT，把输入映射到频域；对复数频谱按 block 做可学习线性变换；再通过 `softshrink` 把小幅值噪声压掉，最后 IFFT 回到原空间，并用残差连接保留原始信息。

---

## 12. 它和论文描述如何对应

从论文意图看，AFF 要解决的是：

- 背景噪声干扰
- 传统固定滤波器不够灵活
- 高频诊断特征不能被粗暴滤掉

而代码中的对应设计就是：

- `rfft / irfft`：进入频域操作
- `w1 / w2`：学习型频域掩码/变换
- `softshrink`：稀疏化与噪声抑制
- `residual`：避免过度破坏原始病理特征

因此这段实现虽然名字叫 `AFNO1D`，但在本项目语境里承担的角色就是论文中的 **AFF 自适应频域滤波器**。

---

## 13. 当前实现里值得注意的点

### 13.1 `self.scale = 0.02` 没有实际用到

在 [AFNO1D.__init__()](file:///c:/Ethan/code/cdssHomeWork20260604/ADD-RSC-homework20260604/models/adapt_diff_denoise.py#L20-L38) 里：

```python
self.scale = 0.02
```

但后续 forward 没有用它。  
这说明它可能是早期版本遗留变量。

### 13.2 `softshrink` 的阈值来自 `sparsity_threshold`

论文里常见会强调稀疏阈值的作用；本代码默认是 `0.01`。  
这个值越大，滤波越激进；越小，保留的频谱越多。

### 13.3 当前默认并没有裁掉频率

`hard_thresholding_fraction=1` 时，`kept_modes = total_modes`，所以没有真正只留“部分频率”。

---

## 14. 最后用直觉理解 AFF

你可以把 AFF 想成这样：

- 原始特征像一段混有噪声的呼吸音模式
- AFF 先把它拆成“不同频率的成分”
- 然后学会哪些频率模式更像噪声、哪些更像病理信号
- 把弱小、分散、不稳定的成分压掉
- 再把处理后的结果还原回去

所以它不是传统信号处理里的固定滤波器，而是一个：

**可学习、样本驱动、带稀疏约束的频域去噪模块。**

