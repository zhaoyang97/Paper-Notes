---
title: >-
  [论文解读] Quantized Prompt for Efficient Generalization of Vision-Language Models
description: >-
  [ECCV 2024][多模态VLM][量化] 将量化误差视为一种正则化噪声，对VLM的可学习prompt进行极低比特量化（最低1-bit），在大幅减少存储开销（最高16倍压缩）的同时显著提升模型在未见类别上的泛化能力，QCoOp仅需0.26KB即超越大量SOTA方法。 大规模预训练视觉语言模型（如CLIP）在下游任务适配时…
tags:
  - "ECCV 2024"
  - "多模态VLM"
  - "量化"
  - "提示学习"
  - "泛化"
  - "视觉语言模型"
  - "参数高效微调"
---

# Quantized Prompt for Efficient Generalization of Vision-Language Models

**会议**: ECCV 2024  
**arXiv**: [2407.10704](https://arxiv.org/abs/2407.10704)  
**代码**: [GitHub](https://github.com)  
**领域**: 多模态VLM  
**关键词**: 量化, prompt tuning, 泛化, 视觉语言模型, 参数高效微调

## 一句话总结
将量化误差视为一种正则化噪声，对VLM的可学习prompt进行极低比特量化（最低1-bit），在大幅减少存储开销（最高16倍压缩）的同时显著提升模型在未见类别上的泛化能力，QCoOp仅需0.26KB即超越大量SOTA方法。

## 研究背景与动机
大规模预训练视觉语言模型（如CLIP）在下游任务适配时面临两大核心问题：**过拟合**和**灾难性遗忘**。现有方法（CoOp、CoCoOp、MaPLe等）通过prompt tuning进行参数高效微调，但随着方法日益复杂，存储和推理成本问题日益突出。

本文从一个关键观察出发：**适度的随机噪声可以抑制过拟合和灾难性遗忘**。作者进一步指出，量化误差本质上就是一种噪声，因此可以利用量化来正则化VLM。相比高斯噪声，量化误差更加可控，且量化本身还能大幅减少存储。这一独特视角将模型压缩与泛化增强有机统一。

核心矛盾在于：噪声过多会削弱模型的适配能力，噪声过少则无法提供有效正则化。因此需要精心设计量化方案，使量化误差维持在"适中"水平。

## 方法详解

### 整体框架
QPrompt方法基于对prompt权重分布特征的深入分析，采用K-Means聚类作为量化基础，结合归一化/反归一化和约束自适应聚类策略。整体pipeline：训练时通过STE（Straight-Through Estimator）传播梯度穿过量化操作；存储时将fp16参数转换为b-bit索引+码本。

### 关键设计

1. **噪声与泛化的关系分析**:

    - 功能：在prompt权重上添加不同强度的高斯噪声，观察base/new类准确率变化
    - 核心发现：训练过程中baseline的泛化能力持续下降而专化能力提升；适度噪声（如0.01）能在不显著损害已见类准确率的前提下提升未见类准确率
    - 设计动机：过强噪声（0.1）严重削弱适配能力，过弱噪声（0.001）无法正则化，只有适中噪声有益——这为量化方案的设计提供了理论依据

2. **Prompt权重分布特征分析**:

    - 功能：分析CoOp训练过程中prompt权重的分布变化
    - 核心发现：(1) 权重分布形状在整个训练过程中基本不变；(2) 方差在训练初期快速增大；(3) 几乎没有异常值；(4) 相邻阶段权重变化平缓
    - 设计动机：这些特征直接指导了量化方案的设计原则

3. **K-Means量化与归一化**:

    - 功能：使用K-Means聚类构建量化映射Q，将prompt权重映射到2^b个离散值
    - 核心思路：先对权重做归一化 $\hat{W} = \frac{W - \mu}{\sigma}$，在归一化空间中做K-Means聚类，再反归一化得到量化后的权重 $W_q = \sigma Q(\hat{W}) + \mu$
    - 设计动机：归一化消除了分布的平移和缩放变换的影响（因为训练中分布形状不变，主要变化来自方差）；K-Means不受异常值影响（因为prompt权重无异常值）

4. **约束自适应聚类（Constrained Adaptive Clustering, CAC）**:

    - 功能：动态控制K-Means码本的更新频率
    - 核心思路：不是每个iteration都重新聚类，而是设置最小更新间隔t，并通过KL散度衡量当前权重分布与缓存权重分布的差异。只有当KL散度超过阈值 $T_{KL}$ 时才重新聚类
    - 设计动机：(1) K-Means计算开销大，频繁执行降低训练效率；(2) 只有适度量化误差有助于泛化，持续最小化量化误差可能反而有害；(3) 相邻阶段权重变化平缓，频繁更新是徒劳的
    - KL散度计算：先将新旧权重映射到同一事件空间（K-Means聚类的索引空间），再计算索引概率分布间的KL散度

5. **存储优化**:

    - 功能：用b-bit索引+码本替代fp16参数
    - 存储量：$bN + 2^b \times 16$ bits，相比baseline的 $16N$ bits
    - 当b=1时，约16倍压缩

### 损失函数 / 训练策略
- 训练时使用STE直接穿过量化操作传播梯度：$\frac{\partial Q(x)}{\partial x} = x$
- 标准的CLIP分类loss：交叉熵损失基于图像-文本相似度
- 方法可集成到CoOp（得到QCoOp）或MaPLe（得到QMaPLe）中

## 实验关键数据

### 主实验：Base-to-New泛化（11个数据集平均）

| 方法 | 参数大小 | Base | New | H（调和均值） |
|------|---------|------|-----|--------------|
| CLIP | 0KB | 69.34 | 74.22 | 71.70 |
| CoOp | 4.1KB | 82.69 | 63.22 | 71.66 |
| CoCoOp | 70.8KB | 80.47 | 71.69 | 75.83 |
| ProGrad | 16.4KB | 82.79 | 68.55 | 75.00 |
| **QCoOp** | **0.26KB** | **80.68** | **74.44** | **77.43** |
| MaPLe | 7096KB | 82.28 | 75.14 | 78.55 |
| **QMaPLe** | **1774KB** | **83.02** | **75.57** | **79.12** |

### 消融实验

| 配置 | Base | New | H | 说明 |
|------|------|-----|---|------|
| K-Means only | 78.71 | 72.55 | 75.50 | 基础量化 |
| K-Means + Norm | 78.84 | 73.09 | 75.85 | 归一化提升new acc |
| K-Means + Norm + CAC | 78.24 | 74.02 | 76.07 | CAC进一步提升new acc |
| QAT | 80.72 | 72.35 | 76.31 | QAT优于PTQ |
| PTQ | 82.21 | 68.50 | 74.73 | PTQ泛化能力弱 |

### 关键发现
- QCoOp（0.26KB）比ProGrad（16.4KB）小63倍，但准确率更高，展示了极致的效率
- QMaPLe相比MaPLe仅需0.25倍存储空间，但H提升0.57%
- 在跨数据集迁移中，QCoOp在10个目标数据集中5个上取得最高准确率
- 在few-shot学习中，QCoOp在所有shot数下均超越CLIP、CoOp和CLIP-Adapter
- 基于SLIP模型验证：QCoOp的new accuracy（74.04%）远高于CoOp（46.60%），H从55.51%飙升至71.07%
- 增加量化比特数不一定带来更好结果：b=1（75.92%）≥ b=2（75.91%）≥ b=4（75.76%）

## 亮点与洞察
- **视角新颖**：首次将量化误差视为正则化工具而非需要最小化的缺陷，完全颠覆了传统量化的优化目标
- **理论与实践统一**：通过详尽的prompt权重分布分析推导出量化方案的设计原则，而非凭空设计
- **极致效率**：0.26KB的模型大小使得在极端资源受限设备上的适配成为可能
- **通用性强**：方法可直接集成到CoOp、MaPLe等多种现有方法中，获得一致提升
- **1-bit量化可行**：在传统场景中被认为是"激进"的1-bit量化，在prompt上反而效果最佳

## 局限与展望
- 量化位数的选择（b=1,2,4）目前是超参数，缺乏自适应选择机制
- KL散度阈值 $T_{KL}$ 和最小更新间隔t的设置缺乏理论指导
- 仅在分类任务上验证，未在检测、分割等下游任务上评估
- 只量化了prompt和部分线性层参数，未探索对backbone的量化

## 相关工作与启发
- **CoOp/CoCoOp/MaPLe**：prompt tuning的基础方法，QPrompt建立在其上进行量化增强
- **正则化与噪声**：Dropout、数据增强等传统正则化方法，本文扩展为权重空间的量化噪声正则化
- **K-Means量化**：传统上受异常值影响大，但prompt权重无异常值的特性使其重新可用
- 启发：模型压缩不一定以牺牲性能为代价，适当的"信息损失"反而可能增强泛化

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 将量化误差重新定义为正则化噪声的视角非常独特
- 实验充分度: ⭐⭐⭐⭐ 四个评估设置、11个数据集、多种基线对比、详尽消融
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，从观察到原则到设计的推导链完整
- 价值: ⭐⭐⭐⭐⭐ 在效率和效果上同时取得收益，实用性极强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] A Parameter-Efficient and Fine-Grained Prompt Learning for Vision-Language Models](../../ACL2025/multimodal_vlm/a_parameter-efficient_and_fine-grained_prompt_learning_for_vision-language_model.md)
- [\[ECCV 2024\] FlexAttention for Efficient High-Resolution Vision-Language Models](flexattention_for_efficient_high-resolution_vision-language_models.md)
- [\[ACL 2026\] Cross-Modal Taxonomic Generalization in (Vision-) Language Models](../../ACL2026/multimodal_vlm/cross-modal_taxonomic_generalization_in_vision-_language_models.md)
- [\[CVPR 2026\] Cluster-Aware Neural Collapse Prompt Tuning for Long-Tailed Generalization of Vision-Language Models](../../CVPR2026/multimodal_vlm/cluster-aware_neural_collapse_prompt_tuning_for_long-tailed_generalization_of_vi.md)
- [\[ECCV 2024\] Attention Prompting on Image for Large Vision-Language Models](attention_prompting_on_image_for_large_visionlanguage_models.md)

</div>

<!-- RELATED:END -->
