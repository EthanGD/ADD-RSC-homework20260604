我来为你绘制 `MultiHeadDifferentialAttention` 的结构图。先让我查看完整的实现细节：


现在我来绘制详细的结构图：

```mermaid
graph TB
    subgraph Input
        X["输入 X<br/>(batch, N, d_model)"]
    end
    
    subgraph Linear_Projections["线性投影层"]
        Wq["W_q<br/>Linear(d_model → 2·d_head·h)"]
        Wk["W_k<br/>Linear(d_model → 2·d_head·h)"]
        Wv["W_v<br/>Linear(d_model → 2·d_head·h)"]
    end
    
    subgraph Reshape["重塑 & 多头分割"]
        Q["Q<br/>(batch, h, N, 2·d_head)"]
        K["K<br/>(batch, h, N, 2·d_head)"]
        V["V<br/>(batch, h, N, 2·d_head)"]
        
        Q1["Q₁<br/>(batch, h, N, d_head)"]
        Q2["Q₂<br/>(batch, h, N, d_head)"]
        K1["K₁<br/>(batch, h, N, d_head)"]
        K2["K₂<br/>(batch, h, N, d_head)"]
    end
    
    subgraph Attention1["注意力路径 1"]
        A1["A₁ = Q₁·K₁/√d<br/>(batch, h, N, N)"]
        Attn1["Softmax(A₁)<br/>(batch, h, N, N)"]
    end
    
    subgraph Attention2["注意力路径 2"]
        A2["A₂ = Q₂·K₂ᵀ/√d<br/>(batch, h, N, N)"]
        Attn2["Softmax(A₂)<br/>(batch, h, N, N)"]
    end
    
    subgraph Lambda["λ 可学习缩放因子"]
        L1["λ_q1 · λ_k1<br/>(num_heads,)"]
        L2["λ_q2 · λ_k2<br/>(num_heads,)"]
        LambdaVal["λ = exp(λ_q1·λ_k1) - exp(λ_q2·λ_k2) + λ_init<br/>(num_heads,)"]
    end
    
    subgraph Differential["差分注意力计算"]
        Diff["Attention = Softmax(A₁) - λ·Softmax(A₂)<br/>(batch, h, N, N)"]
        LayerNorm_Diff["LayerNorm<br/>(沿序列维度)"]
    end
    
    subgraph ApplyAttention["应用注意力到Value"]
        O["O = Attention · V<br/>(batch, h, N, 2·d_head)"]
        RMSNorm["RMSNorm + rms_scale<br/>(batch·h, N, 2·d_head)"]
        Scale["× (1 - λ_init)<br/>标量缩放"]
    end
    
    subgraph Output["输出投影"]
        Concat["拼接所有头<br/>(batch, N, h·2·d_head)"]
        Wo["W_o<br/>Linear(h·2·d_head → d_model)"]
        Out["输出<br/>(batch, N, d_model)"]
    end
    
    %% 连接关系
    X --> Wq
    X --> Wk
    X --> Wv
    
    Wq --> Q
    Wk --> K
    Wv --> V
    
    Q -->|"chunk(2)"| Q1
    Q -->|"chunk(2)"| Q2
    K -->|"chunk(2)"| K1
    K -->|"chunk(2)"| K2
    
    Q1 --> A1
    K1 --> A1
    A1 --> Attn1
    
    Q2 --> A2
    K2 --> A2
    A2 --> Attn2
    
    L1 --> LambdaVal
    L2 --> LambdaVal
    
    Attn1 --> Diff
    Attn2 --> Diff
    LambdaVal --> Diff
    
    Diff --> LayerNorm_Diff
    LayerNorm_Diff --> O
    
    O --> RMSNorm
    RMSNorm --> Scale
    Scale --> Concat
    
    Concat --> Wo
    Wo --> Out
    
    %% 样式定义
    classDef input fill:#1a1a2e,stroke:#0096d6,stroke-width:2px,color:#fff
    classDef linear fill:#2d1b69,stroke:#a05cf0,stroke-width:2px,color:#fff
    classDef attention fill:#1b4332,stroke:#4ecb71,stroke-width:2px,color:#fff
    classDef lambda fill:#4a2c2a,stroke:#ff8c42,stroke-width:2px,color:#fff
    classDef diff fill:#1e3a5f,stroke:#0096d6,stroke-width:2px,color:#fff
    classDef output fill:#2d1b69,stroke:#a05cf0,stroke-width:2px,color:#fff
    
    class X input
    class Wq,Wk,Wv,Wo linear
    class Q,K,V,Q1,Q2,K1,K2 linear
    class A1,Attn1,A2,Attn2 attention
    class L1,L2,LambdaVal lambda
    class Diff,LayerNorm_Diff diff
    class O,RMSNorm,Scale,Concat,Out diff
```

## 核心流程说明

### 1️⃣ **线性投影** (第 225-227 行)
- 输入 `X` 通过三个线性层得到 Q, K, V
- 每个投影到 `2 × d_head × num_heads` 维度

### 2️⃣ **多头分割** (第 232-239 行)
- 将 Q, K, V 重塑为 `(batch, num_heads, N, 2·d_head)`
- 使用 `chunk(2)` 沿最后一个维度分割成 Q₁/Q₂ 和 K₁/K₂

### 3️ **双路注意力计算** (第 258-270 行)
```python
A1 = softmax(Q1·K1ᵀ/√d)  # 注意力路径1
A2 = softmax(Q2·K2ᵀ/√d)  # 注意力路径2
Attention = A1 - λ·A2     # 差分注意力（核心！）
```

### 4️⃣ **λ 可学习缩放因子** (第 245-247 行)
```python
λ = exp(λ_q1·λ_k1) - exp(λ_q2·λ_k2) + λ_init
```
- 4个可学习参数矩阵控制两路注意力的权重平衡

### 5️⃣ **差分后处理** (第 272-301 行)
- LayerNorm（沿序列维度归一化）
- 应用注意力到 V：`O = Attention · V`
- RMSNorm（沿特征维度归一化）
- 标量缩放：`× (1 - λ_init)`

### 6️⃣ **输出投影** (第 305-308 行)
- 拼接所有头的输出
- 通过 `W_o` 线性层映射回 `d_model` 维度

---

##  与传统注意力的关键区别

| 传统注意力 | 差分注意力 (MHDA) |
|-----------|------------------|
| `softmax(QKᵀ)·V` | `(softmax(Q₁K₁ᵀ) - λ·softmax(Q₂K₂ᵀ))·V` |
| 单路 softmax | **双路差分** |
| 对噪声敏感 | **过滤噪声敏感变化** |
| 无法区分噪声/信号 | **保留高置信度特征** |

这个结构的核心创新就是通过 **两路注意力差分** 来抑制噪声扰动，同时保留对分类关键的稳定特征！