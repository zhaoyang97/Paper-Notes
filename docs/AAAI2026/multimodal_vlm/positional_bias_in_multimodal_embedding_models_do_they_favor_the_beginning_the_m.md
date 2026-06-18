---
title: >-
  [论文解读] Positional Bias in Multimodal Embedding Models: Do They Favor the Beginning, the Middle, or the End?
description: >-
  [AAAI 2026][信息检索/RAG][位置偏差] 本文首次系统研究多模态表示模型中的位置偏差现象，发现文本编码器倾向于偏好输入开头，而图像编码器在开头和结尾均表现偏好，并通过大量控制实验揭示该偏差源于位置编码方案、训练损失、上下文重要性和图文对训练的多因素共同作用。 领域现状：Transformer模型在NLP和视觉任…
tags:
  - "AAAI 2026"
  - "信息检索/RAG"
  - "位置偏差"
  - "CLIP"
  - "多模态表示学习"
  - "图文检索"
  - "注意力机制"
---

# Positional Bias in Multimodal Embedding Models: Do They Favor the Beginning, the Middle, or the End?

**会议**: AAAI 2026  
**arXiv**: [2511.11216](https://arxiv.org/abs/2511.11216)  
**代码**: [https://github.com/tiiuae/PosBias/](https://github.com/tiiuae/PosBias/)  
**领域**: 信息检索  
**关键词**: 位置偏差, CLIP, 多模态表示学习, 图文检索, 注意力机制

## 一句话总结
本文首次系统研究多模态表示模型中的位置偏差现象，发现文本编码器倾向于偏好输入开头，而图像编码器在开头和结尾均表现偏好，并通过大量控制实验揭示该偏差源于位置编码方案、训练损失、上下文重要性和图文对训练的多因素共同作用。

## 研究背景与动机

**领域现状**：Transformer模型在NLP和视觉任务上取得巨大成功，但研究发现模型捕获上下文信息的能力受到信息在输入序列中位置的影响——这就是"位置偏差"问题。Liu等人发现的"lost in the middle"现象表明模型常常优先关注开头和结尾而忽略中间内容。

**现有研究的不足**：
- 位置偏差研究主要集中在**文本生成**模型（LLM），对**表示学习**模型的关注较少
- 文本表示模型中已发现"dwelling at the beginning"偏差（Coelho et al.），但归因于倒金字塔写作风格
- **多模态模型中的位置偏差几乎未被报告**，尤其是图像端的偏差模式完全未知

**核心疑问**：
1. 多模态嵌入模型（如CLIP）中是否存在位置偏差？
2. 文本编码器和图像编码器的偏差模式是否相同？
3. 偏差的根本原因是什么——位置编码？训练数据？损失函数？模型架构？

**本文切入角度**：首先区分"上下文重要性"和"位置偏差"两个概念，然后通过精心设计的实验（将固定段落移动到不同位置并屏蔽其他内容）系统评估多模态模型中位置偏差的存在程度和模式，最后通过大量控制变量实验逐一排查可能的原因。

## 方法详解

### 整体框架

研究分为三个层次：(1) 上下文重要性分析——理解哪些区域本身更重要；(2) 位置偏差分析——检测偏差的存在和模式；(3) 偏差原因调查——通过控制实验验证或否定现有假说。

### 关键设计

1. **上下文重要性分析（Context Importance）**

    - **功能**：定位文本/图像中语义最重要的区域
    - **文本端**：将输入token均匀分段，每次只保留一段其余用padding mask，通过文本编码器获取表示，固定图像端计算检索准确率
    - **图像端**：同理，保留一个图像区域其余用CLIP RGB均值（[0.481, 0.458, 0.408]）遮蔽，固定文本端
    - **核心发现**：文本中第一段最重要（符合重要信息前置的写作习惯）；图像中中心区域最重要（主体通常居中，上下为背景/天空/地面）
    - **设计动机**：必须先理解"什么区域本身重要"才能判断"模型是因为内容还是因为位置而偏好某区域"

2. **位置偏差分析（Positional Bias）**

    - **文本端两种策略**：
        - **Text Perturbation**：将文本分成子文本，选一段移到不同位置，其他替换为Lorem Ipsum占位文本
        - **Token Masking**：将token分段，选一段移到不同位置，其他用padding token屏蔽
    - **图像端**：单个视觉段被隔离并移到不同空间位置，其他区域用CLIP RGB均值遮蔽
    - **关键设计选择**：与之前打乱多文档顺序的研究不同，本文隔离单个段并移动——因为CLIP的固定上下文窗口不允许同时放入完整内容
    - **设计动机**：目标不是测量绝对检索准确率，而是分析检索性能如何随位置变化而变化

3. **偏差原因调查**

    - **数据分布**：训练Shuffled Long-CLIP（打乱子句顺序），偏差仍存在但减弱（第一段从开头移到结尾的准确率下降0.199 vs 原版0.303），说明数据分布有贡献但不是唯一原因
    - **位置编码**：比较CLIP（绝对位置编码）vs TULIP（RoPE旋转位置编码），TULIP偏差更强（第一段从位置0移到2，准确率从0.66骤降到0.065），说明位置编码方案显著影响偏差
    - **文本长度**：即使短文本（COCO平均11.53 token）也存在偏差
    - **模型大小**：ViT-L/14比ViT-B/16偏差更强（变异系数更高）
    - **图像分辨率和Patch大小**：减小patch大小或增大分辨率有助于降低图像端偏差
    - **训练损失**：SigLIP（sigmoid损失）vs CLIP（softmax对比损失）偏差模式不同——SigLIP在图像端偏好开头而非两端
    - **模型架构**：CNN（CLIP-ResNet-50）也存在偏差，说明不是Transformer特有
    - **单模态vs多模态**：纯视觉ResNet无位置偏差，CLIP-ResNet有→图文对训练引入或放大了图像端偏差

## 实验关键数据

### 主实验（位置偏差存在性验证）

| 模型 | 文本端偏差模式 | 图像端偏差模式 | 数据集 |
|------|-------------|-------------|--------|
| Long-CLIP (ViT-B/16) | 开头偏好 | 开头+结尾偏好 | Urban1K |
| TULIP (ViT-L/14) | 更强的开头偏好 | 开头+结尾偏好 | Urban1K |
| Shuffled Long-CLIP | 减弱的开头偏好 | 开头+结尾偏好 | Urban1K |
| CLIP (ViT-B/16) | 开头偏好（较弱） | 开头+结尾偏好 | COCO |
| SigLIP-Base | 开头偏好 | 主要开头偏好 | COCO |
| CLIP-ResNet-50 | 开头偏好 | 开头+结尾偏好 | COCO |

### 消融实验（偏差原因控制变量）

| 控制因素 | 实验设置 | 结论 |
|---------|---------|------|
| 数据分布 | Long-CLIP vs Shuffled Long-CLIP | 有贡献但非唯一因素（偏差减弱但不消失） |
| 位置编码 | CLIP(绝对) vs TULIP(RoPE) | RoPE加剧偏差；绝对编码缓解但不消除 |
| 文本长度 | Long-CLIP(Urban1K) vs CLIP(COCO) | 短文本也存在偏差，但较弱 |
| 模型大小 | ViT-B/16 vs ViT-L/14 | 大模型偏差更强（变异系数更高） |
| 分辨率 | ViT-L/14 vs ViT-L/14@336 | 高分辨率有助降低图像端偏差 |
| 训练损失 | CLIP(softmax) vs SigLIP(sigmoid) | 不同损失导致不同偏差模式 |
| 架构 | ViT vs ResNet | CNN也有偏差，非Transformer特有 |
| 训练方式 | 纯视觉模型 vs CLIP视觉编码器 | 图文对训练引入/放大图像端偏差 |

**变异系数量化（长标题数据集Urban1K，图像端seg0-seg6）**：

| 模型 | seg0 | seg3 | seg6 | 说明 |
|------|------|------|------|------|
| Long-CLIP | 0.146 | 0.087 | 0.159 | 两端高 |
| TULIP | 0.220 | 0.107 | 0.186 | 两端高且更强 |
| Shuffled Long-CLIP | 0.185 | 0.089 | 0.144 | 打乱后减弱 |

### 关键发现

1. **位置偏差在多模态模型中普遍存在**：所有测试的模型、数据集、架构、训练设置中都观察到
2. **文本和图像的偏差模式不同**：
    - 文本编码器：**一致偏好开头**（将任何段移到开头都能提升检索准确率）
    - 图像编码器：**偏好开头和结尾**（形成U型分布），开头偏好更强
3. **图像端偏差尤其值得注意**：图像中心区域是最有语义意义的（上下文重要性实验证明），但模型却偏好开头和结尾——这明确证明了位置偏差的存在
4. **多模态训练引入图像端偏差**：纯视觉模型（ResNet、ViT）无明显偏差，但加入CLIP框架后偏差显著
5. **位置编码影响巨大但不是唯一因素**：即使Long-CLIP（绝对位置编码）比TULIP（RoPE）偏差更小，偏差仍然存在

## 亮点与洞察

1. **研究角度新颖**：首次将位置偏差分析从文本生成/文本表示扩展到多模态表示学习，填补了重要的研究空白
2. **上下文重要性与位置偏差的解耦**：这一区分非常关键——模型可能因为"开头内容确实重要"而非"偏好开头位置"表现出偏差，作者通过控制实验明确区分了两者
3. **实验设计严谨全面**：控制变量法逐一排查了数据分布、位置编码、损失函数、模型大小、分辨率、架构、训练方式等7个因素
4. **多模态训练引入图像偏差**：这是一个深刻洞察——纯视觉模型没有位置偏差，而CLIP训练后出现，说明图文对齐学习本身引入了新的归纳偏置
5. **发现RoPE加剧偏差**：反直觉——RoPE被设计来改善位置建模，但实际上在多模态表示学习中加剧了位置偏差

## 局限与展望

1. **仅分析了偏差，未提出缓解方法**：论文明确将缓解策略留作未来工作
2. **聚焦CLIP系列模型**：未评估LLaVA、BLIP-2等生成式多模态模型中的表示偏差
3. **评估指标较简单**：主要使用Recall@k和变异系数，更精细的偏差量化指标可能更有参考价值
4. **数据集有限**：仅使用Urban1K、DOCCI和COCO三个数据集
5. **因果分析不够深入**：虽然识别了多个贡献因素，但各因素的相对权重和交互效应未量化
6. **实际影响未充分讨论**：位置偏差对下游任务（RAG、zero-shot分类）的具体影响未评估

## 相关工作与启发

- **"Lost in the Middle"**（Liu et al.）：LLM中经典的位置偏差研究，本文将其扩展到表示模型
- **"Dwelling at the Beginning"**（Coelho et al.）：文本表示模型中的开头偏好，归因于对比学习训练，本文在多模态setting下验证了类似现象
- **Differential Transformer**（Ye et al.）：差分注意力机制减少生成模型中的注意力噪声，作者建议可适配到双向多模态表示
- 启发：CLIP作为许多高级多模态系统的后端（包括RAG），其位置偏差可能传递到下游系统
- 实际应用启发：在使用CLIP进行检索时，可能需要考虑如何放置关键信息以获得最佳效果；长文本caption应将关键描述放在开头

## 评分
- 新颖性: ⭐⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Do Retrieval Augmented Language Models Know When They Don't Know?](do_retrieval_augmented_language_models_know_when_they_dont_know.md)
- [\[AAAI 2026\] "As Eastern Powers, I Will Veto." : An Investigation of Nation-Level Bias of Large Language Models in International Relations](as_eastern_powers_i_will_veto_an_investigation_of_nation-level_bias_of_large_lan.md)
- [\[ICML 2026\] How can embedding models bind concepts?](../../ICML2026/information_retrieval/how_can_embedding_models_bind_concepts.md)
- [\[AAAI 2026\] PRECISE: Reducing the Bias of LLM Evaluations Using Prediction-Powered Ranking Estimation](precise_reducing_the_bias_of_llm_evaluations_using_prediction-powered_ranking_es.md)
- [\[ICML 2026\] LazyAttention: Efficient Retrieval-Augmented Generation with Deferred Positional Encoding](../../ICML2026/information_retrieval/lazyattention_efficient_retrieval-augmented_generation_with_deferred_positional_.md)

</div>

<!-- RELATED:END -->
