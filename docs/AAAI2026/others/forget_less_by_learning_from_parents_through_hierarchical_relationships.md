---
title: >-
  [论文解读] Forget Less by Learning from Parents Through Hierarchical Relationships
description: >-
  [AAAI 2026][定制扩散模型] 本文提出 FLLP（Forget Less by Learning from Parents）框架，通过在双曲空间中建立概念之间的父子层次关系来缓解定制扩散模型（CDM）的灾难性遗忘，利用 Lorentz 流形的树状结构建模能力实现新概念学习时的知识保持和持续概念集成。
tags:
  - "AAAI 2026"
  - "定制扩散模型"
  - "灾难性遗忘"
  - "双曲空间"
  - "层次关系"
  - "概念学习"
---

# Forget Less by Learning from Parents Through Hierarchical Relationships

**会议**: AAAI 2026  
**arXiv**: [2601.01892](https://arxiv.org/abs/2601.01892)  
**代码**: 无  
**领域**: 持续学习 / 图像生成  
**关键词**: 定制扩散模型, 灾难性遗忘, 双曲空间, 层次关系, 概念学习

## 一句话总结

本文提出 FLLP（Forget Less by Learning from Parents）框架，通过在双曲空间中建立概念之间的父子层次关系来缓解定制扩散模型（CDM）的灾难性遗忘，利用 Lorentz 流形的树状结构建模能力实现新概念学习时的知识保持和持续概念集成。

## 研究背景与动机

**领域现状**：定制扩散模型（Custom Diffusion Models, CDMs）如 DreamBooth、Textual Inversion 等可以通过少量示例图像将新概念注入预训练扩散模型中，实现个性化图像生成。这使得用户可以用自己的宠物、特定物体等生成创意图像。

**现有痛点**：当需要顺序学习多个新概念时，CDM 会严重遭受灾难性遗忘——学习新概念会覆盖之前学到的概念表示。现有对抗遗忘的方法主要关注最小化概念间的干扰（如正交化、参数隔离），但忽略了概念之间可能存在的正向交互——相关概念的知识可以互相增强。

**核心矛盾**：现有方法将多概念学习视为冲突管理问题（如何避免新概念覆盖旧概念），而非协作机会（如何利用概念间的关系促进学习和保持）。

**本文目标**：（1）建模概念之间的层次结构关系；（2）利用"父概念"的知识指导"子概念"的学习；（3）在学习新概念的同时保持且增强旧概念。

**切入角度**：作者将概念组织为层次树结构——更通用的概念（如"狗"）作为更具体概念（如"我家的金毛"）的父节点。在双曲空间（Lorentz 流形）中嵌入这些概念，利用双曲空间天然适合建模树状层次结构的特性。

**核心 idea**：在双曲空间中定义概念的父子关系，让已学的"父概念"作为新"子概念"学习的锚点和指导，实现知识保持与新概念适应的双赢。

## 方法详解

### 整体框架

FLLP 在标准 CDM 训练 pipeline 基础上引入双曲空间的层次建模。输入是一系列待学习的概念（每个概念有少量示例图像），框架按顺序学习每个概念。核心增强在于：在潜在表示空间中将概念嵌入到 Lorentz 流形上，建立父子关系，并利用父概念的嵌入来约束和引导子概念的学习。

### 关键设计

1. **Lorentz 流形嵌入**:

    - 功能：为概念表示提供适合树状层次建模的几何空间。
    - 核心思路：将概念的文本嵌入或视觉嵌入映射到 Lorentz 流形 $\mathbb{H}^n$（双曲空间的一种模型）。在双曲空间中，距离度量自然反映了层次关系——越靠近原点的点越"通用"，越远离原点越"具体"。利用 Lorentz 距离 $d_L(u, v) = \text{arccosh}(-\langle u, v \rangle_L)$ 度量概念间的语义距离。
    - 设计动机：欧几里得空间中的嵌入无法有效建模层次关系——树状结构在欧空间中的嵌入失真严重。双曲空间的指数体积增长恰好匹配树结构的分支特性，可以低失真地嵌入层次结构。

2. **父子概念关系机制**:

    - 功能：利用已学概念的知识指导新概念的学习。
    - 核心思路：当学习新概念时，在双曲空间中找到与其最相关的已学概念作为"父概念"。新概念的初始嵌入从父概念出发通过指数映射（exponential map）微调得到。训练时添加父子距离约束——新概念不能偏离父概念太远，但也不能与父概念完全重合。
    - 设计动机：从父概念出发初始化利用了先验知识（如"金毛"继承了"狗"的大部分特征），减少了从头学习的负担。距离约束同时确保了父概念不被覆盖（不能太近）且新概念保持相关性（不能太远）。

3. **抗遗忘正则化**:

    - 功能：在双曲空间中约束概念嵌入的稳定性。
    - 核心思路：对已学概念的嵌入添加位移惩罚——旧概念在双曲空间中的位置不应因新概念的学习而大幅移动。同时保护扩散模型中与旧概念相关的关键参数（如交叉注意力层的key/value矩阵）。
    - 设计动机：即使有父子关系的保护，梯度更新仍可能间接影响旧概念的表示。额外的正则化提供了双重保障。

### 损失函数 / 训练策略

总损失包含：（1）标准 CDM 训练损失（扩散去噪损失）；（2）双曲距离约束损失（父子关系保持）；（3）旧概念嵌入稳定性正则化。训练按概念顺序进行，每个新概念的学习利用父概念的指导。

## 实验关键数据

### 主实验

在三个公开数据集和一个合成基准上评估。

| 数据集 | 指标 | FLLP | 之前SOTA | 提升 | 说明 |
|--------|------|------|----------|------|------|
| 公开数据集1 | 鲁棒性 | 最佳 | -- | 一致提升 | 多概念保持 |
| 公开数据集2 | 泛化性 | 最佳 | -- | 一致提升 | 新概念生成质量 |
| 公开数据集3 | 综合指标 | 最佳 | -- | 一致提升 | 整体表现 |
| 合成基准 | 遗忘率 | 最低 | -- | 显著降低 | 控制实验 |

### 消融实验

| 配置 | 性能 | 说明 |
|------|------|------|
| FLLP (Full) | 最佳 | 双曲嵌入+父子关系+正则化 |
| w/o 双曲空间 (欧几里得) | 下降 | 欧空间无法有效建模层次结构 |
| w/o 父子关系 | 下降 | 缺少知识传递机制 |
| w/o 正则化 | 旧概念退化 | 新概念学习影响旧概念 |

### 关键发现

- 双曲空间嵌入相比欧几里得空间嵌入带来了明显的性能提升，验证了层次几何结构的价值。
- 父子关系机制不仅防止遗忘，还通过知识传递提升了新概念的学习效率和生成质量。
- FLLP 在鲁棒性和泛化性上一致优于现有方法，说明"协作学习"比"冲突隔离"更有效。
- 合成基准上的控制实验清晰地展示了每个组件的贡献。

## 亮点与洞察

- **双曲空间建模概念层次**的创新将黎曼几何引入持续概念学习，是跨领域方法论迁移的优秀示例。
- **父子关系学习范式**将遗忘问题从"冲突管理"重新框定为"协作学习"，这一视角转变可以推广到其他持续学习场景。
- 方法支持概念的持续集成，不需要全部概念同时可用——对实际应用场景（用户逐步添加概念）友好。

## 局限与展望

- 双曲空间的计算（指数映射、对数映射等）比欧几里得空间更复杂，引入了额外计算开销。
- 父子关系的自动确定可能在语义模糊的概念之间不够准确。
- 论文假设概念之间确实存在有意义的层次关系，对完全无关的概念可能效果有限。
- 可以探索多叉树结构（多个父节点）而非简单的单父结构，以更好地建模复杂的概念关系。

## 相关工作与启发

- **vs DreamBooth**: DreamBooth 针对单概念定制，本文扩展到持续多概念学习。
- **vs C-LoRA / Custom Diffusion**: 这些方法通过参数隔离避免冲突，FLLP 通过层次关系实现协作——理念不同但可能互补。
- **vs 双曲嵌入方法（如 Poincaré embeddings）**: 本文将双曲嵌入从 NLP 和推荐系统领域迁移到生成模型的持续学习中，拓展了应用范围。

## 评分

- 新颖性: ⭐⭐⭐⭐ 双曲空间+父子概念关系的组合方案新颖独特
- 实验充分度: ⭐⭐⭐⭐ 四个数据集+完整消融
- 写作质量: ⭐⭐⭐⭐ 直觉清晰，几何解释优雅
- 价值: ⭐⭐⭐⭐ 为持续概念学习提供了新的方法论范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Break the Tie: Learning Cluster-Customized Category Relationships for Categorical Data Clustering](break_the_tie_learning_cluster-customized_category_relationships_for_categorical.md)
- [\[NeurIPS 2025\] Exploiting Task Relationships in Continual Learning via Transferability-Aware Task Embeddings](../../NeurIPS2025/others/exploiting_task_relationships_in_continual_learning_via_transferability-aware_ta.md)
- [\[AAAI 2026\] CAE: Hierarchical Semantic Alignment for Image Clustering](hierarchical_semantic_alignment_for_image_clustering.md)
- [\[ICML 2026\] Less Data, Faster Training: Repeating Smaller Datasets Speeds Up Learning via Sampling Biases](../../ICML2026/others/less_data_faster_training_repeating_smaller_datasets_speeds_up_learning_via_samp.md)
- [\[ICML 2026\] Polaris: Coupled Orbital Polar Embeddings for Hierarchical Concept Learning](../../ICML2026/others/polaris_coupled_orbital_polar_embeddings_for_hierarchical_concept_learning.md)

</div>

<!-- RELATED:END -->
