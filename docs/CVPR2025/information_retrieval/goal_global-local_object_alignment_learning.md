---
title: >-
  [论文解读] GOAL: Global-Local Object Alignment Learning
description: >-
  [CVPR 2025][信息检索/RAG][全局-局部对齐] 提出GOAL方法，通过局部图-句匹配（LISM）和Token相似性学习（TSL）两个模块增强CLIP对长文本描述的理解能力，在全局对齐的基础上引入局部语义对齐，大幅提升图文检索性能。 领域现状： CLIP通过在大规模图文对上进行对比学习，已经展现了强大的零样本迁移…
tags:
  - "CVPR 2025"
  - "信息检索/RAG"
  - "全局-局部对齐"
  - "CLIP微调"
  - "长文本图文检索"
  - "SAM分割"
  - "Token相似性学习"
---

# GOAL: Global-Local Object Alignment Learning

**会议**: CVPR 2025  
**arXiv**: [2503.17782](https://arxiv.org/abs/2503.17782)  
**代码**: [GitHub](https://perceptualai-lab.github.io/GOAL)  
**领域**: 信息检索  
**关键词**: 全局-局部对齐、CLIP微调、长文本图文检索、SAM分割、Token相似性学习

## 一句话总结

提出GOAL方法，通过局部图-句匹配（LISM）和Token相似性学习（TSL）两个模块增强CLIP对长文本描述的理解能力，在全局对齐的基础上引入局部语义对齐，大幅提升图文检索性能。

## 研究背景与动机

**领域现状**：
CLIP通过在大规模图文对上进行对比学习，已经展现了强大的零样本迁移能力。然而CLIP的训练数据主要是简短的图像描述（最多77个token），关注图像的高层概念。

**现有痛点**：
1. CLIP在处理较长、详细的文本描述时表现不佳，因为其统一嵌入空间针对简洁描述优化
2. 现有模型只做全局图-文匹配，将整张图和完整caption作为单一单元对齐，缺乏细粒度的局部对应关系
3. Long-CLIP虽然扩展了文本长度，但需要多模态LLM生成合成长描述，数据准备代价高

**核心矛盾**：
CLIP的全局对比学习范式无法捕获图像局部区域与文本中具体描述句之间的细粒度对应关系，导致处理长文本时丢失大量细节信息。

**本文目标**
如何高效地微调CLIP使其能理解长文本中的局部语义细节，同时保持全局对齐能力。

**切入角度**：
将"全局"定义为整张图或完整文本，"局部"定义为图像片段或文本中的单个句子，通过构建局部伪对并传播局部注意力来增强全局表示。

**核心 idea**：
用SAM分割图像、拆分句子来构建局部图-句伪对，然后通过Token级别的相似性学习将局部信息传播到全局表示中。

## 方法详解

### 整体框架

GOAL包含两个核心组件：(1) LISM（Local Image-Sentence Matching）用于从全局图像-长文本对中提取局部伪对；(2) TSL（Token Similarity-based Learning）通过匹配的局部对将局部注意力高效传播到全局Token表示中。整体目标函数结合全局对比、局部对比和TSL损失。

### 关键设计

1. **LISM（局部图-句匹配管道）**:
    - 功能：从一对图文数据中自动提取最佳的局部图像-句子匹配对
    - 核心思路：用SAM将图像分割成语义区域（过滤<1%面积的小区域），将caption按句子拆分；用CLIP编码器提取各区域和句子的CLS embedding，计算余弦相似度进行最大相似度匹配，选出最高相似度的局部对$(I_l, T_l)$
    - 设计动机：利用SAM的强分割能力和CLIP自身的匹配能力，无需额外标注即可获取高质量的局部对应关系

2. **TSL（Token相似性学习）**:
    - 功能：将局部语义信息传播到全局Token表示中
    - 核心思路：对于文本端，找到全局文本中对应局部句子的Token，平均池化后通过投影层映射，然后最大化与局部CLS embedding的相似度；对于视觉端，根据局部图像的bounding box找到全局图像中对应区域的patch token，平均池化后投影，最大化与局部CLS embedding的相似度
    - 设计动机：通过让全局Token的子集逼近对应局部的CLS表示，使编码器关注图像/文本中的关键局部元素

3. **位置编码插值**:
    - 功能：支持超过77 token的长文本处理
    - 核心思路：采用Long-CLIP的位置编码插值技术扩展文本编码器的最大序列长度
    - 设计动机：原始CLIP的77 token限制无法处理多句长描述

### 损失函数 / 训练策略

- 总损失：$\mathcal{L}_{total} = \lambda_{global}\mathcal{L}_{global} + \lambda_{local}\mathcal{L}_{local} + \lambda_{TSL}\mathcal{L}_{TSL}$
- $\lambda_{global}=1$, $\lambda_{local}=0.5$, $\lambda_{TSL}=1$
- $\mathcal{L}_{global}$ 和 $\mathcal{L}_{local}$ 是标准CLIP对比损失
- $\mathcal{L}_{TSL}$ 使用MSE损失最大化投影Token与对应局部CLS embedding的相似度
- 训练10 epochs，batch size 16，单张RTX 4090约1小时

## 实验关键数据

### 主实验

**DOCCI数据集（Original Test Set, ViT-L/14）**

| 方法 | T2I R@1 | T2I R@5 | I2T R@1 | I2T R@5 |
|------|---------|---------|---------|---------|
| Global fine-tuning | 74.00 | 93.84 | 73.55 | 93.94 |
| Local fine-tuning | 67.39 | 90.67 | 66.33 | 90.41 |
| w/o TSL | 74.75 | 94.31 | 74.55 | 94.37 |
| **GOAL** | **84.37** | **97.55** | **82.57** | **97.37** |

**DCI数据集（Original Test Set, ViT-L/14）**

| 方法 | T2I R@1 | T2I R@5 | I2T R@1 | I2T R@5 |
|------|---------|---------|---------|---------|
| Global fine-tuning | 65.73 | 84.24 | 65.73 | 86.04 |
| GOAL | **76.89** | **91.05** | **76.59** | **91.20** |

### 消融实验

**DOCCI数据集（ViT-B/16）**

| 设置 | Global | Local | TSL | T2I R@1 | I2T R@1 |
|------|--------|-------|-----|---------|---------|
| 仅全局 | ✓ | | | 72.41 | 72.04 |
| 仅局部 | | ✓ | | 65.82 | 65.73 |
| 全局+局部 | ✓ | ✓ | | 72.08 | 71.80 |
| GOAL完整 | ✓ | ✓ | ✓ | **79.47** | **79.43** |

### 关键发现

1. **TSL是核心贡献**：仅添加局部对比损失（w/o TSL）几乎不带来提升甚至略微下降，但加上TSL后性能大幅跃升（ViT-L/14 DOCCI: +12.87% R@1）
2. **局部对齐不能单独使用**：仅用局部对比损失性能显著低于全局，因为丢失了全局上下文
3. **在Global-Local联合测试集上同样有效**：GOAL在同时包含全局和局部查询的测试集上也显著优于基线（mAP@10提升3-6个百分点）
4. **资源友好**：单卡RTX 4090即可完成ViT-B/16训练，约1小时

## 亮点与洞察

1. **简洁有效**：方法设计非常简单，仅通过LISM管道和TSL损失两个模块就大幅提升了CLIP处理长文本的能力
2. **无额外标注**：利用SAM和CLIP自身能力自动构建局部伪对，无需人工标注
3. **TSL的关键洞察**：不是简单地在全局和局部之间加对比学习，而是让全局Token的子集"逼近"对应局部的表示，这种Token级别的对齐策略远比CLS级别的局部对比学习有效
4. **新Benchmark**：提出了三个针对长文本图文检索的新评测基准（DOCCI、DCI、Urban1k）

## 局限与展望

1. LISM管道仅选取一对最佳局部匹配，可能丢失其他有价值的局部对应关系
2. 依赖SAM进行图像分割，分割质量直接影响局部对的质量
3. 仅在3个数据集上验证，且数据规模较小（DOCCI仅约1万样本）
4. 未与大规模预训练方法（如CLOC）进行系统对比
5. 位置编码插值可能在极长文本上效果有限

## 相关工作与启发

- **CLIP/Long-CLIP**: 本文的基础模型和直接竞争对手
- **CLOC**: 大规模预训练方法，通过OWL-v2检测器建立局部对应，但需要20亿图文对
- **SAM**: 提供强大的零样本图像分割能力，是LISM管道的关键依赖
- 启发：局部语义对齐的关键不在于独立学习局部特征，而在于如何将局部信息有效传播到全局表示中

## 评分

- 新颖性: ⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Preserving Clusters in Prompt Learning for Unsupervised Domain Adaptation](preserving_clusters_in_prompt_learning_for_unsupervised_domain_adaptation.md)
- [\[ACL 2025\] Divide-Then-Align: Honest Alignment based on the Knowledge Boundary of RAG](../../ACL2025/information_retrieval/divide_then_align_rag_knowledge_boundary.md)
- [\[ACL 2025\] GainRAG: Preference Alignment in Retrieval-Augmented Generation through Gain Signal Synthesis](../../ACL2025/information_retrieval/gainrag_preference_alignment.md)
- [\[ACL 2026\] Learning to Extract Rational Evidence via Reinforcement Learning for Retrieval-Augmented Generation](../../ACL2026/information_retrieval/learning_to_extract_rational_evidence_via_reinforcement_learning_for_retrieval-a.md)
- [\[NeurIPS 2025\] Scaling Language-Centric Omnimodal Representation Learning](../../NeurIPS2025/information_retrieval/scaling_language-centric_omnimodal_representation_learning.md)

</div>

<!-- RELATED:END -->
