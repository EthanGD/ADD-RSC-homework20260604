音频频谱图变换器（Audio Spectrogram Transformer，简称 AST）是第一个用于音频分类的无卷积、纯注意力机制（Pure Attention-based）的深度学习模型。它由麻省理工学院（MIT）等机构的研究人员于 2021 年提出，直接应用了计算机视觉领域的 [Vision Transformer (ViT)](https://blog.csdn.net/qq_46079584/article/details/122515052) 架构来处理音频任务。 [1, 2, 3] 
## 核心工作原理
AST 的核心思想是将音频问题转化为图像问题。它的工作流程主要分为以下几个步骤： [4, 5] 

* 音频谱图化：将一维的原始音频波形通过短时傅里叶变换（STFT）转换成二维的梅尔频谱图（Mel Spectrogram），这个频谱图可以看作是一张单通道的“图像”。 [4, 5] 
* 分块切片（Patching）：像 ViT 处理图像一样，AST 将二维频谱图切割成一系列相互重叠的 16 × 16 小方块（Patches）。 [3] 
* 线性投影与位置编码：将每个小方块展平并通过线性层映射为一维向量，同时加入一维位置编码以保留音频在时间和频率上的顺序信息。 [3] 
* Transformer 编码器：将带有位置信息的向量序列输入纯 [Transformer Encoder](https://www.mindspore.cn/technology-blogs/zh/3797) 中，利用自注意力机制（Self-Attention）直接捕捉音频在长距离上的全局上下文特征。 [6] 

------------------------------
## AST 模型的核心优势
相比传统的音频处理模型，AST 具备三大核心特点：

* 完全无卷积（Convolution-Free）：传统音频模型（如 PANNs）极度依赖卷积神经网络（CNN），而 AST 摆脱了卷积的局部限制，从最底层开始就能捕捉全局信息。 [6, 7] 
* 强大的交叉领域知识迁移：由于其架构与 ViT 完全一致，AST 可以直接加载在 ImageNet 大规模图像数据集上预训练好的权重，这极大弥补了音频训练数据不足的问题。 [3] 
* 性能顶尖（SOTA）：在主流的音频分类和常规声音识别数据集（如 [AudioSet](https://aidoczh.com/hf/docs/transformers/v4.47.1/en/model_doc/audio-spectrogram-transformer.html)、ESC-50、Speech Commands）上，AST 的分类准确率均达到了当时的顶尖水平。 [2] 

------------------------------
## 主要应用场景
AST 及其衍生模型目前已被广泛应用于各种音频分析任务中：

* 环境音与事件分类：识别生活中的各种声音（如下雨声、汽车鸣笛、宠物叫声等）。
* 语音命令识别：对简短的语音控制指令进行精准分类。
* 乐器识别与音色匹配：分析音频中的乐器成分，甚至用于合成器参数的智能推断。 [2, 8] 

目前，您可以通过 Hugging Face Transformers 库非常方便地调用官方提供的 ASTModel 和 ASTForAudioClassification 开源模型进行开发和微调。 [5, 9] 
如果您对 AST 的具体实现感兴趣，需要我为您展示如何使用 Python 代码加载该模型进行音频分类，或者深入了解它的预训练细节吗？

[1] [https://zhuanlan.zhihu.com](https://zhuanlan.zhihu.com/p/660402940)
[2] [https://aidoczh.com](https://aidoczh.com/hf/docs/transformers/v4.47.1/en/model_doc/audio-spectrogram-transformer.html)
[3] [https://blog.csdn.net](https://blog.csdn.net/qq_46079584/article/details/122515052)
[4] [https://www.mindspore.cn](https://www.mindspore.cn/technology-blogs/zh/3797)
[5] [https://blog.csdn.net](https://blog.csdn.net/qq_42452134/article/details/136596925)
[6] [https://blog.csdn.net](https://blog.csdn.net/ggqyh/article/details/136116897)
[7] [https://hyper.ai](https://hyper.ai/cn/papers/2104.01778)
[8] [https://alphaxiv.org](https://alphaxiv.org/zh/overview/2407.16643v1)
[9] [https://huggingface.tw](https://huggingface.tw/docs/transformers/model_doc/audio-spectrogram-transformer)


https://sls.csail.mit.edu/publications/2021/YuanGong_Interspeech-2021.pdf