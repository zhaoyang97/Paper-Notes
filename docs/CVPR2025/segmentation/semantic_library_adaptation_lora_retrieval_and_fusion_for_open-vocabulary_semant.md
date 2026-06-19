---
title: >-
  [论文解读] Semantic Library Adaptation: LoRA Retrieval and Fusion for Open-Vocabulary Semantic Segmentation
description: >-
  [CVPR 2025][语义分割][开放词汇语义分割] SemLA 提出了一个无需训练的测试时域适应框架，通过构建基于 CLIP 索引的 LoRA 适配器库，在推理时根据输入图像与各域质心的嵌入距离动态检索和融合最相关的适配器，为开放词汇语义分割模型实现了即时、高效的域适应。 领域现状：开放词汇（OV）语义分割通过视觉-文本…
tags:
  - "CVPR 2025"
  - "语义分割"
  - "开放词汇语义分割"
  - "LoRA适配器"
  - "域适应"
  - "测试时适应"
  - "CLIP"
---

# Semantic Library Adaptation: LoRA Retrieval and Fusion for Open-Vocabulary Semantic Segmentation

**会议**: CVPR 2025  
**arXiv**: [2503.21780](https://arxiv.org/abs/2503.21780)  
**代码**: [https://thegoodailab.org/semla](https://thegoodailab.org/semla)  
**领域**: 分割 / 域适应  
**关键词**: 开放词汇语义分割, LoRA适配器, 域适应, 测试时适应, CLIP

## 一句话总结

SemLA 提出了一个无需训练的测试时域适应框架，通过构建基于 CLIP 索引的 LoRA 适配器库，在推理时根据输入图像与各域质心的嵌入距离动态检索和融合最相关的适配器，为开放词汇语义分割模型实现了即时、高效的域适应。

## 研究背景与动机

**领域现状**：开放词汇（OV）语义分割通过视觉-文本对齐实现像素级分类，可以用任意文本查询定义类别。CAT-Seg 等模型在零样本场景中展现了良好的灵活性。然而，训练域与测试域之间的分布偏移（域漂移）会显著降低性能——包括视觉外观偏移和词汇不对齐。

**现有痛点**：（1）传统域适应方法（UDA）通常只针对单一目标域，需要访问源域数据，过程缓慢且可能退化源域性能；（2）测试时适应方法（TTA）通常计算开销大，不适合实时应用；（3）数据虽然充足但异构——标签体系不同、标注风格不同、可能包含敏感信息——导致"数据充裕但模型脆弱"的悖论。

**核心矛盾**：开放词汇分割的灵活性和域适应的鲁棒性之间存在根本张力。模型需要对任意文本查询响应，同时还要适应千变万化的视觉域，而传统适应方法无法在保持 OV 能力的同时高效适应新域。

**本文目标**：实现 OV 语义分割的训练-free 测试时域适应——无需额外训练即可在推理时即时适应任意输入图像的域。

**切入角度**：受 LLM 社区中 LoRA 适配器+检索的启发（如 Hugging Face 的 adapter 生态），作者观察到可以为不同域训练轻量级 LoRA 适配器，用 CLIP 嵌入作为索引，在测试时根据输入图像的 CLIP 嵌入检索并融合最相关的适配器。这就像一个图书馆——想找的书可能不在，但可以从多本相关书中获取所需知识。

**核心 idea**：构建一个以 CLIP 质心索引的 LoRA 适配器库，测试时计算输入图像的 CLIP 嵌入与各域质心的距离，选取最近的 K 个适配器，根据距离加权融合（concat 方式），为每张输入图像构建一个定制化的模型。

## 方法详解

### 整体框架

SemLA 分为两个阶段：
1. **离线：构建 LoRA 适配器库**——为每个训练域训练一个 LoRA 适配器，计算该域图像的 CLIP 嵌入质心作为索引，组成库 $\mathcal{L} = \{(\mathbf{c}_i, \Delta\mathcal{W}_i)\}$
2. **在线：动态测试时适应**——对每张测试图像计算 CLIP 嵌入 → 检索最近的 K 个适配器 → 基于距离加权 → concat 方式融合 LoRA 权重 → 预测

### 关键设计

1. **CLIP 嵌入质心索引 (Domain Embeddings from CLIP)**:

    - 功能：为每个域的 LoRA 适配器创建一个紧凑的语义索引
    - 核心思路：对训练域 $\mathcal{D}_i$ 中每张图像计算 CLIP 图像嵌入 $\mathbf{e}_j = \text{CLIP}_\text{image}(\mathbf{x}_j)$，取平均得到质心 $\mathbf{c}_i = \frac{1}{N_i}\sum_{j=1}^{N_i} \mathbf{e}_j$。测试时同样计算输入图像的 CLIP 嵌入 $\mathbf{e}_t$，通过欧氏距离 $d_i = \|\mathbf{e}_t - \mathbf{c}_i\|_2$ 测量相似度
    - 设计动机：CLIP 的嵌入空间天然捕获了域的语义特征，质心计算简单高效，且不需要对 CLIP 进行任何训练或微调

2. **基于距离的适配器加权与 Concat 融合**:

    - 功能：将多个域的知识按相关性加权合并为一个定制化模型
    - 核心思路：选取 Top-K 个最近适配器（索引集 $\mathcal{K}$），用带温度 $\tau$ 的 softmax 计算权重 $w_i = \frac{\exp(1/(d_i \cdot \tau))}{\sum_{k \in \mathcal{K}} \exp(1/(d_k \cdot \tau))}$。融合采用 concatenation 方式：将各适配器的 $\mathbf{A}$ 矩阵乘以权重后纵向拼接，$\mathbf{B}$ 矩阵横向拼接，得到 $\Delta\mathbf{W}_\text{fused} = \mathbf{B}_\text{fused} \mathbf{A}_\text{fused}$
    - 设计动机：相比简单平均（Uniform merging），基于距离的加权能够让更相关的域贡献更大权重。Concat 方式保留了每个适配器的低秩结构，比直接加权平均 $\Delta\mathbf{W}$ 更好地保留了各域的独特知识

3. **库的可扩展性设计**:

    - 功能：随时添加新域的适配器，无需重新训练或影响现有适配器
    - 核心思路：新域只需计算质心 $\mathbf{c}_*$、训练 LoRA $\Delta\mathcal{W}_*$，然后 append 到库中即可 $\mathcal{L} = \mathcal{L} + (\mathbf{c}_*, \Delta\mathcal{W}_*)$
    - 设计动机：现实世界中新域会持续出现，库的增量式扩展避免了重新训练或全局重新优化

### 损失函数 / 训练策略

各域 LoRA 适配器使用标准的语义分割损失（交叉熵）在对应域数据上独立训练。训练时 CAT-Seg 原始权重冻结，仅优化 LoRA 参数。所有适配器使用相同的 rank $r$，确保融合时的维度一致性。

评估采用 leave-one-out 策略：评估某个域时，从库中移除对应适配器，确保模型从未见过目标域的直接知识。

## 实验关键数据

### 主实验

**20 域基准测试（CAT-Seg 骨干，leave-one-out, mIoU）：**

| 方法 | ACDC rain | ACDC fog | ACDC night | CS | BDD | ADE150 | IDD | h-mean |
|------|-----------|----------|------------|------|------|--------|------|--------|
| Zero-shot | 46.5 | 47.1 | 37.9 | 47.1 | 47.9 | 37.8 | 35.4 | 39.4 |
| Uniform merging | 67.4 | 69.7 | 50.0 | 62.2 | 58.2 | 37.3 | 38.8 | 51.9 |
| **SemLA** | **67.7** | **71.9** | **51.7** | **63.9** | **57.3** | **38.2** | **40.2** | **54.2** |
| Oracle (上界) | 70.9 | 70.0 | 51.6 | 67.5 | 60.1 | 54.0 | 64.3 | 61.1 |

### 消融实验

| 融合策略 | h-mean mIoU |
|---------|-------------|
| Zero-shot (无适应) | 39.4 |
| Uniform merging (均匀合并) | 51.9 |
| SemLA Late Fusion (输出级融合) | 52.1 |
| Uniform Late Fusion | 49.3 |
| **SemLA (权重级融合)** | **54.2** |

### 关键发现

- **SemLA 的 h-mean 比 Uniform merging 提升 2.27 个点**，且在大多数域上都有提升，证明了选择性融合优于全量均匀融合
- **在恶劣天气域上提升尤其明显**：ACDC fog +2.21, ACDC night +1.75, MUSES 系列普遍 +2~7 个点。这说明域特异性越强，SemLA 的选择性优势越大
- **有时超越 Oracle**：在 ACDC fog 和 ACDC night 上，SemLA 超越了使用目标域数据训练的单一适配器，说明多域知识融合可以带来互补增益
- **权重级融合优于输出级融合**：SemLA (54.2) > SemLA Late Fusion (52.1)，表明在参数空间融合比在预测空间融合更有效
- **CLIP 嵌入是有效的域导航器**：不需要额外训练检索器，CLIP 的零样本嵌入就能准确判断图像所属域

## 亮点与洞察

- **图书馆类比**贴切而深刻：适配器是"书"，CLIP 质心是"索引系统"，融合是"从多本相关书中综合知识"。这种框架将域适应从"训练问题"转化为"检索+融合问题"
- **可解释性是天然附带的**：通过观察哪些适配器被选中及其权重，可以理解模型为什么做出某个预测。这在医疗等敏感应用中非常有价值
- **数据隐私友好**：测试时无需访问任何训练数据，各域的数据始终留在本地，只共享 LoRA 参数和 CLIP 质心。这为联邦学习式的协作提供了可能
- **即插即用**：SemLA 不依赖特定骨干，可以应用于任何使用 CLIP 的 OV 分割模型

## 局限与展望

- **适配器库覆盖的域有限**：如果输入图像的域距离库中所有域都很远，融合效果可能很差。需要更大、更多样的库
- **CLIP 嵌入的域区分能力有上限**：某些视觉差异很大但语义相似的域（如白天vs夜间的同一场景）可能在 CLIP 空间中距离较近，导致选择不当
- **评估局限于 CAT-Seg**：虽然声称骨干无关，但主实验只在 CAT-Seg 上验证
- **质心表示过于粗糙**：一个均值向量难以捕获域内的多样性（如一个域包含多种子场景）
- 未来可以探索：更细粒度的域表示（如 GMM）、主动学习选择最有价值的新域进行训练、结合文本嵌入进行多模态检索

## 相关工作与启发

- **vs Uniform LoRA Merging (Model Soups 系列)**: Model Soups / AdapterSoup 均匀合并所有适配器，相当于让所有"书"权重一样。SemLA 根据相关性选择性融合，效果更好
- **vs LoraRetriever**: LoraRetriever 通过 instruction fine-tuning 训练检索器，SemLA 直接利用 CLIP 的零样本能力，更简单优雅
- **vs 测试时适应 (TTA)**: TTA 方法（如 entropy minimization）需要在测试时进行梯度更新，SemLA 完全免训练，只需一次 CLIP 推理和矩阵运算
- **vs UDA**: UDA 需要源域和目标域数据共存，且只适应一个目标域。SemLA 无需源数据、可同时适应任意域

## 评分

- 新颖性: ⭐⭐⭐⭐ 将 LoRA 检索+融合引入 OV 分割的域适应是新颖的组合，但各技术组件（LoRA、CLIP 检索）已有
- 实验充分度: ⭐⭐⭐⭐⭐ 20 域基准覆盖面极广，leave-one-out 评估严谨，Oracle 对比和消融充分
- 写作质量: ⭐⭐⭐⭐⭐ 图书馆类比贯穿始终，framework 图清晰，方法描述严谨
- 价值: ⭐⭐⭐⭐ 训练-free 和隐私保护特性使方法具有很强的实用性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Effective SAM Combination for Open-Vocabulary Semantic Segmentation](effective_sam_combination_for_open-vocabulary_semantic_segmentation.md)
- [\[CVPR 2025\] Exploring Simple Open-Vocabulary Semantic Segmentation](exploring_simple_open-vocabulary_semantic_segmentation.md)
- [\[CVPR 2025\] Universal Domain Adaptation for Semantic Segmentation](universal_domain_adaptation_for_semantic_segmentation.md)
- [\[CVPR 2025\] DPSeg: Dual-Prompt Cost Volume Learning for Open-Vocabulary Semantic Segmentation](dpseg_dual-prompt_cost_volume_learning_for_open-vocabulary_semantic_segmentation.md)
- [\[ICCV 2025\] Training-Free Class Purification for Open-Vocabulary Semantic Segmentation](../../ICCV2025/segmentation/training-free_class_purification_for_open-vocabulary_semantic_segmentation.md)

</div>

<!-- RELATED:END -->
