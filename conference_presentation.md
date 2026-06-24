# ADD4RSC 会议报告 PPT

## 幻灯片 1：标题页
**呼吸声分类的自适应差分去噪**

Adaptive Differential Denoising for Respiratory Sounds Classification

- **作者**: Dong Gaoyang¹, Zhang Zhicheng², Sun Ping³*, Zhang Minghui¹*
- **单位**: ¹南昌大学信息工程学院, ²复旦大学计算机学院, ³南昌大学第一附属医院
- **发表**: Interspeech 2025
- **报告人**: [你的名字]

---

## 幻灯片 2：研究背景
**自动呼吸声分类（ARSC）的重要性**

- 无创诊断肺部疾病（哮喘、肺炎、COPD）
- 远程医疗和资源受限地区的早期筛查
- 通过分析数字听诊器或可穿戴传感器的听诊信号

**面临的挑战**

| 挑战 | 说明 |
|------|------|
| 数据稀缺 | 带注释的样本不足 |
| 噪声污染 | 肢体运动、心音、环境噪声、传感器伪影 |
| 传统方法局限 | 谱减法、维纳滤波无法保留高频病理特征 |

---

## 幻灯片 3：现有方法局限性
**顺序去噪→分类范式的问题**

```
传统流程：原始音频 → 显式去噪 → 分类
```

**三大局限**

1. **区域不匹配**：通用音频去噪器在非生物医学记录上训练
2. **病理特征丢失**：显式去噪模块无意中消除细微病理特征
3. **目标解耦**：级联结构阻碍噪声不变表示的联合优化

**传统滤波的缺陷**
- 巴特沃斯带通滤波：过度衰减 100-2000Hz 以外频率，丢失高频裂音（>2000Hz）
- 小波分解：依赖固定基函数导致瞬态特征丢失

---

## 幻灯片 4：核心贡献
**本文提出隐式去噪方法**

**三大创新**

1. **自适应频率滤波器（AFF）**
   - 可学习频谱掩码 + 软收缩
   - 消除噪声同时保留诊断高频成分

2. **差分降噪层（DDL）**
   - 差分注意力机制
   - 通过增强样本比较降低噪声变化

3. **偏置去噪损失**
   - 联合优化分类和鲁棒性
   - 无需干净标签

**结果**：ICBHI2017 达到 **65.53% Score**，提升 **1.99%**

---

## 幻灯片 5：方法框架
**ADD4RSC 整体架构**

```
输入音频 → 预处理 → 自适应频率滤波器 → 差分降噪层 → 分类器 → 输出
                    ↓                      ↓
              频域可学习掩码          多头差分注意力
                    ↓                      ↓
                 软收缩                  联合损失
```

**预处理步骤**
- 重采样到 16kHz
- 固定 8 秒长度（循环填充/截断）
- 幅度归一化
- Mel 谱图（64 滤波器，1024 窗口）

---

## 幻灯片 6：自适应频率滤波器（AFF）
**频域处理机制**

**快速傅里叶变换**
$$X(U, V) = \sum_{t=0}^{T-1} \sum_{f=0}^{F-1} X(t,f) e^{-2\pi i(Ut + Vf)}$$

**滤波公式**
$$\hat{X}(t,f) = F^{-1}[S_\alpha(M(X(U,V))) \odot X(U,V)]$$

**关键组件**
- $M(\cdot)$：可学习实例自适应掩码（线性层→ReLU→线性层）
- $S_\alpha(\cdot)$：软收缩 $S_\alpha(x) = \text{sign}(x)\max\{|x|-\alpha, 0\}$
- $\alpha = 0.02$：控制稀疏度

---

## 幻灯片 7：差分降噪层（DDL）
**多头差分注意力（MHDA）机制**

**注意力计算**
$$[Q_1; Q_2] = XW_Q, \quad [K_1; K_2] = XW_K, \quad V = XW_V$$

$$A = \left(\sigma\left(\frac{Q_1K_1^T}{\sqrt{d}}\right) - \lambda \cdot \sigma\left(\frac{Q_2K_2^T}{\sqrt{d}}\right)\right)V$$

**工作原理**
- 计算两个扰动表示的独立注意力图
- 相减过滤噪声敏感变化
- 保留稳定的高置信度特征
- $\lambda$：可学习比例因子

---

## 幻灯片 8：损失函数
**混合损失设计**

$$L = \beta L_{\text{Bias Denoise}} + (1-\beta)L_{\text{CE}}$$

**偏置去噪损失（标签平滑交叉熵）**
$$L_{\text{Bias Denoise}} = -\sum_i \left[y_c(1-\epsilon) + \frac{\epsilon}{C}\right] \log[\phi(\text{Norm}(p))]$$

**参数设置**
- $\beta = 0.5$：去噪与分类权重平衡
- $\epsilon = 0.2$：不确定性缓冲器
- $C$：分类总数
- $\phi$：1×1 卷积层

**优势**：仅依赖分类标签，无需干净参考信号

---

## 幻灯片 9：实验设置
**ICBHI 2017 数据集**

| 属性 | 值 |
|------|-----|
| 参与者 | 126 人 |
| 采样率 | 4kHz - 44.1kHz（重采样至 16kHz） |
| 持续时间 | 10-90 秒/记录 |
| 划分 | 60% 训练 / 40% 测试（患者不重叠） |

**训练配置**
- 框架：PyTorch 2.3.1
- GPU：Tesla V100
- 优化器：Adam（权重衰减 0.1）
- 学习率：5e-5
- 批次大小：8
- Epoch：50
- 骨干网：ResNet50 / AST-Base（AudioSet 预训练）

---

## 幻灯片 10：评估指标
**ICBHI 官方评估方法**

**特异性（Specificity）**
$$S_p = \frac{\text{正确识别的正常声音数}}{\text{总正常声音数}}$$

**敏感性（Sensitivity）**
$$S_e = \frac{\text{正确识别的异常声音数}}{\text{总异常声音数}}$$

**综合分数**
$$\text{Score} = \frac{S_e + S_p}{2}$$

**异常声音类别**
- Wheeze（喘息）
- Crackle（裂音）
- Wheeze & Crackle（两者）

---

## 幻灯片 11：与 SOTA 比较（CNN 方法）
**表 1 上半部分**

| CNN 模型 | $S_p$(%) | $S_e$(%) | Score(%) | 发表 |
|---------|----------|----------|----------|------|
| LungRN+NL | 63.20 | 41.32 | 52.26 | BioCAS 2019 |
| RespireNet | 72.30 | 40.10 | 56.20 | EMBC 2021 |
| CoTuning | 79.34 | 37.24 | 58.29 | IEEE TBE |
| SCL | 75.95 | 39.15 | 57.55 | WASPAA 2023 |
| **Ours (ResNet50)** | **83.76** | 34.18 | **58.97** | **Interspeech 2025** |

**提升**：比 CoTuning 高 **2.68%**

---

## 幻灯片 12：与 SOTA 比较（Transformer 方法）
**表 1 下半部分**

| Transformer 模型 | $S_p$(%) | $S_e$(%) | Score(%) | 发表 |
|-----------------|----------|----------|----------|------|
| AST Fine-tuning | 77.14 | 41.97 | 59.55 | Interspeech 2023 |
| Patch-Mix CL | 81.66 | 43.07 | 62.37 | Interspeech 2023 |
| M2D | 81.51 | 45.08 | 63.29 | IEEE TASLP |
| BTS | 81.40 | 45.67 | 63.54 | Interspeech 2024 |
| **Ours (AST)** | **85.13** | **45.94** | **65.53** | **Interspeech 2025** |

**提升**：比 BTS 高 **1.99%**，实现最高敏感性

---

## 幻灯片 13：消融实验
**各组件贡献分析（AST 骨干网）**

| 模型 | $S_p$(%) | $S_e$(%) | Score(%) |
|------|----------|----------|----------|
| w AFF | 82.47 | 44.47 | 63.47 |
| w DDL | 83.73 | 44.03 | 63.88 |
| w AFF+DDL | 84.41 | 44.78 | 64.60 |
| w AFF+$L_{\text{BiasDenoise}}$ | 83.76 | 44.62 | 64.19 |
| w DDL+$L_{\text{BiasDenoise}}$ | 84.54 | 44.32 | 64.43 |
| **Ours（完整）** | **85.13** | **45.94** | **65.53** |

**结论**：每个组件都至关重要，组合产生最佳性能

---

## 幻灯片 14：关键发现
**实验分析**

1. **AFF 模块**
   - $S_p$ 提升 2.66%
   - 有效抑制外源干扰

2. **DDL 模块**
   - 比 AFF 作用更大
   - 细化呼吸噪声，保留判别信号

3. **联合优化**
   - AFF+DDL 互补
   - $L_{\text{BiasDenoise}}$ 进一步细化特征

4. **跨架构优势**
   - CNN 和 Transformer 都有效
   - 证明方法与架构无关

---

## 幻灯片 15：代码与资源
**开源信息**

- **代码仓库**: https://github.com/deegy666/ADD-RSC
- **数据集**: ICBHI 2017 (https://bhichallenge.med.auth.gr/)
- **预训练模型**: Hugging Face (MIT/ast-finetuned-audioset-10-10-0.4593)

**依赖环境**
```
torch==2.0.1
torchaudio==2.0.1
cuda==11.7
```

---

## 幻灯片 16：未来工作
**研究方向**

1. 探索呼吸声事件检测任务的去噪技术
2. 扩展到多模态融合（文本、图像）
3. 实时部署优化
4. 跨数据集泛化能力验证

---

## 幻灯片 17：致谢
**感谢**

- 国家自然科学基金（No. 82260024）
- 南昌大学第一附属医院
- 南昌大学影像与视觉表征实验室
- patch-mix_contrastive_learning 社区
- FunASR 社区
- Adaptive frequency filters 项目

---

## 幻灯片 18：参考文献
**核心引用**

```bibtex
@article{dong2025adaptive,
  title={Adaptive Differential Denoising for Respiratory Sounds Classification},
  author={Dong, Gaoyang and Zhang, Zhicheng and Sun, Ping and Zhang, Minghui},
  journal={arXiv preprint arXiv:2506.02505},
  year={2025}
}
```

---

## 幻灯片 19：Q&A
**谢谢！欢迎提问**

联系方式：
- donggyncu@email.ncu.edu.cn
- zhangminghui@ncu.edu.cn
