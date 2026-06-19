---
title: >-
  [论文解读] Alleviating Textual Reliance in Medical Language-guided Segmentation via Prototype-driven Semantic Approximation
description: >-
  [ICCV 2025][医学图像][医学图像分割] 提出ProLearn框架，首次通过原型驱动的语义近似（PSA）模块从根本上缓解医学语言引导分割对文本的依赖——仅需少量图文配对数据初始化原型空间，训练和推理均可无文本输入，在1%文本可用性下仍保持强劲性能（QaTa-COV19 Dice=0.857），且参数量比LLM方案减少1000倍，推理速度快100倍。
tags:
  - "ICCV 2025"
  - "医学图像"
  - "医学图像分割"
  - "语言引导分割"
  - "原型学习"
  - "文本依赖"
  - "语义近似"
---

# Alleviating Textual Reliance in Medical Language-guided Segmentation via Prototype-driven Semantic Approximation

**会议**: ICCV 2025  
**arXiv**: [2507.11055](https://arxiv.org/abs/2507.11055)  
**代码**: [https://github.com/ShuchangYe-bib/ProLearn](https://github.com/ShuchangYe-bib/ProLearn)  
**领域**: 医学图像分割 / 多模态学习  
**关键词**: 医学图像分割, 语言引导分割, 原型学习, 文本依赖, 语义近似

## 一句话总结
提出ProLearn框架，首次通过原型驱动的语义近似（PSA）模块从根本上缓解医学语言引导分割对文本的依赖——仅需少量图文配对数据初始化原型空间，训练和推理均可无文本输入，在1%文本可用性下仍保持强劲性能（QaTa-COV19 Dice=0.857），且参数量比LLM方案减少1000倍，推理速度快100倍。

## 研究背景与动机

深度学习推动了医学图像分割的发展，从U-Net到其变体（U-Net++、Attention U-Net、TransUNet等）已广泛应用。近年来，将临床报告文本作为辅助引导的多模态分割方法（如LViT、GuideSeg）展现出超越单模态方法的潜力，因为文本提供了病变的显式语义描述。

然而，语言引导分割存在固有的"文本依赖"问题：(1) **训练阶段**——大量医学分割数据集没有配对报告，大部分纯图像数据被浪费；(2) **推理阶段**——推理时需要配对报告限制了其仅能用于回顾性分析，而大多数临床场景（如术前规划、实时手术引导、诊断决策）中分割是在报告生成之前进行的。

前序工作SGSeg尝试用LLM（GPT-2、Llama3）在推理时从图像生成合成报告来弥补文本缺失，但引入了B级参数的LLM导致模型臃肿、推理缓慢，不适合边缘设备和实时应用，且训练阶段的文本依赖仍未解决。

本文的关键洞察是：语言引导分割中的关键引导并非完整临床报告（往往冗长且含不相关信息），而是其中与分割相关的特定语义特征。且医学报告的语义空间天然受限——临床报告遵循标准化医学术语，词汇相对封闭。基于这一洞察，可以用有限的原型来离散表示分割相关语义。

## 方法详解

### 整体框架
ProLearn = PSA模块 + Language-guided U-Net。PSA模块一次性从少量图文配对数据中初始化原型空间，此后训练和推理时通过"查询-响应"机制从图像特征查询原型空间获取近似的语义引导，无需文本输入。原型空间在训练中动态学习更新。

### 关键设计

1. **PSA初始化（一次性过程）**:

    - 功能：从可用的 $K$ 个图文配对样本中构建可查询的原型空间
    - 核心思路：
        - **代理标签提取**：用BioMedCLIP编码器提取图像特征 $e_i^I$ 和文本特征 $e_i^T$。用单独训练的Language-guided U-Net的交叉注意力权重筛选高注意力token（$\alpha_j > \tau$），得到分割相关的缩短句子 $T_i^{\text{selected}}$，编码为语义特征 $e_i^{\text{sem}}$
        - **层次聚类**：用HDBSCAN对 $e_i^{\text{sem}}$ 聚类为 $N$ 个代理标签，每个代理标签代表一种独特的分割相关语义
        - **原型空间构建**：在每个文本代理标签簇 $\mathcal{C}_i$ 内，用K-means对图像特征 $e_k^I$ 进一步聚为 $M$ 个子簇。取每个子簇最接近质心的样本（非质心本身，以减少异常值影响）作为原型对 $(q_{ij}, r_{ij})$，构建查询空间 $\mathcal{S}^Q$（图像原型）和响应空间 $\mathcal{S}^R$（文本原型），维度 $N \times M \times D$
    - 设计动机：两级聚类策略——先按文本语义分粗类，再按视觉特征分细类——使原型空间既紧凑又能表达比文本更丰富的视觉多样性

2. **PSA查询与响应机制**:

    - 功能：在训练和推理中为任意图像提供近似的语义引导
    - 核心思路：
        - **查询**：图像编码特征 $q^* = f_{\text{enc}}^I(I^*)$ 与查询空间所有原型计算余弦相似度 $s_{ij} = s(q^*, q_{ij})$，选择top-$k$最相似原型 $Q^*$
        - **响应**：找到 $Q^*$ 对应的文本响应原型 $R^*$，用softmax归一化的相似度加权求和：
       $$r^* = \sum_{r_i \in R^*} w_i r_i, \quad w_i = \frac{\exp(s(q^*, q_i))}{\sum_{q_j \in Q^*} \exp(s(q^*, q_j))}$$
        - 响应 $r^*$ 送入U-Net解码器引导分割
    - 设计动机：原型学习时间复杂度 $\mathcal{O}(1)$（查询固定大小的原型空间），vs LLM自回归生成的 $\mathcal{O}(n)$。仅1M参数 vs GPT-2的1.5B / Llama3的7B

3. **Language-guided U-Net**:

    - 功能：以PSA响应作为语义引导进行图像分割
    - 核心思路：标准U-Net编码器提取图像特征用于PSA查询，解码器接收PSA响应的近似语义特征引导解码分割掩码
    - 设计动机：复用已有的语言引导分割架构，PSA模块作为即插即用组件替代文本编码器

### 损失函数 / 训练策略
- 标准分割损失（Dice Loss + CE Loss）
- 原型空间在训练过程中动态学习更新（非固定）
- 可同时使用图文配对样本和纯图像样本训练

## 实验关键数据

### 主实验：不同文本可用性下的语言引导分割对比

| 数据集 | 模型 | 50%文本Dice | 10%文本Dice | 1%文本Dice |
|--------|------|------------|------------|-----------|
| QaTa-COV19 | LViT | 0.842 | 0.800 | 0.701 |
| QaTa-COV19 | GuideSeg | 0.863 | 0.840 | 0.733 |
| QaTa-COV19 | SGSeg | 0.864 | 0.842 | 0.731 |
| QaTa-COV19 | **ProLearn** | **0.867** | **0.858** | **0.857** |
| MosMedData+ | SGSeg | 0.746 | 0.695 | 0.345 |
| MosMedData+ | **ProLearn** | **0.754** | **0.742** | **0.722** |
| Kvasir-SEG | GuideSeg | 0.885 | 0.775 | 0.562 |
| Kvasir-SEG | **ProLearn** | **0.898** | **0.890** | **0.872** |

### 消融/效率对比

| 模型 | 参数量 | 推理时间 | 时间复杂度 |
|------|--------|---------|----------|
| ProLearn (PSA) | 1M | 4ms | $\mathcal{O}(1)$ |
| SGSeg (GPT-2) | 1.5B | 136ms | $\mathcal{O}(n)$ |
| SGSeg (Llama3) | 7B | 1.2s | $\mathcal{O}(n)$ |

### 关键发现
- 在1%文本可用性下，ProLearn几乎无性能降低（QaTa Dice: 0.867→0.857），而GuideSeg从0.863暴跌至0.733，SGSeg从0.864跌至0.731
- ProLearn甚至仅用1%文本就超越了所有使用100%图像数据的单模态方法（U-Net, U-Net++, Swin U-Net等）
- 比LLM方案参数减少1000倍、速度快100-300倍，适合边缘设备和实时应用
- 超参数（候选数k、原型数M）在较宽范围内性能稳定

## 亮点与洞察
- 关键洞察极具启发性：医学报告的语义空间是天然受限的（标准化术语+封闭词汇），因此可以用有限原型离散近似，无需每次重新生成文本
- PSA的"查询-响应"机制是一种优雅的跨模态桥接方案：图像查询→选择视觉最近原型→返回对应的文本语义，实现了无文本推理
- 两级聚类（HDBSCAN按文本语义 + K-means按视觉特征）的设计保证了原型空间既语义清晰又视觉多样
- 性能退化曲线（图5）直观展示了ProLearn vs 其他方法在文本递减时的鲁棒性差异

## 局限与展望
- 原型空间的初始化依赖BioMedCLIP的特征质量，对于BioMedCLIP覆盖不佳的医学领域可能效果折扣
- 代理标签提取需要先训练一个Language-guided U-Net来获取注意力权重，增加了pipeline复杂度
- 未探索原型空间的在线更新策略（如新数据到来时增量更新）
- 仅在COVID-19肺部和结肠息肉场景验证，更复杂的多器官/多病变场景有待探索

## 相关工作与启发
- **vs SGSeg**: SGSeg用LLM生成合成报告弥补推理时文本缺失，但训练仍依赖文本且引入B级参数LLM。ProLearn从根本上解耦了对文本的依赖，参数/速度优势巨大
- **vs LViT/GuideSeg**: 这些方法严格要求图文配对输入，文本不可用时性能急剧下降。ProLearn通过原型近似语义，使性能几乎不受文本可用性影响

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次用原型学习解决语言引导分割的文本依赖问题，洞察深刻
- 实验充分度: ⭐⭐⭐⭐ 三个数据集五种文本比例+效率对比+超参数分析
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义精准，motivation链完整，图示清晰
- 价值: ⭐⭐⭐⭐⭐ 具有重要的临床实用价值，解决了阻碍语言引导分割落地的核心瓶颈

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Boosting Vision Semantic Density with Anatomy Normality Modeling for Medical Vision-language Pre-training](boosting_vision_semantic_density_with_anatomy_normality_modeling_for_medical_vis.md)
- [\[CVPR 2025\] Evidential learning driven Breast Tumor Segmentation with Stage-divided Vision-Language Interaction](../../CVPR2025/medical_imaging/evidential_learning_driven_breast_tumor_segmentation_with_stage-divided_vision-l.md)
- [\[AAAI 2026\] Divide, Conquer and Unite: Hierarchical Style-Recalibrated Prototype Alignment for Federated Medical Segmentation](../../AAAI2026/medical_imaging/divide_conquer_and_unite_hierarchical_style-recalibrated_prototype_alignment_for.md)
- [\[NeurIPS 2025\] Magical: Medical Lay Language Generation via Semantic Invariance and Layperson-tailored Adaptation](../../NeurIPS2025/medical_imaging/magical_medical_lay_language_generation_via_semantic_invariance_and_layperson-ta.md)
- [\[CVPR 2026\] MedKCO: Medical Vision-Language Pretraining via Knowledge-Driven Cognitive Orchestration](../../CVPR2026/medical_imaging/medkco_medical_vision-language_pretraining_via_knowledge-driven_cognitive_orches.md)

</div>

<!-- RELATED:END -->
