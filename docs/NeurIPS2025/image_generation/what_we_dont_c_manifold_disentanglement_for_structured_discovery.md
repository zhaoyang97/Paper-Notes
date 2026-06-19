---
title: >-
  [论文解读] What We Don't C: Manifold Disentanglement for Structured Discovery
description: >-
  [NeurIPS 2025][图像生成][流形解缠绕] 提出 WWDC（What We Don't C）方法，利用条件引导的潜在流匹配从已有 VAE 表征中去除已知信息，使未知特征在残余流形中更易被发现和访问，实现迭代式科学发现。 领域现状 领域现状：表征学习的核心挑战：如何在学到的表征中访问有意义的信息 现有痛点 现有痛点…
tags:
  - "NeurIPS 2025"
  - "图像生成"
  - "流形解缠绕"
  - "流匹配"
  - "VAE"
  - "分类器无关引导"
  - "结构化发现"
---

# What We Don't C: Manifold Disentanglement for Structured Discovery

**会议**: NeurIPS 2025  
**arXiv**: [2511.09433](https://arxiv.org/abs/2511.09433)  
**代码**: 有  
**领域**: 表征学习, 流匹配, 解缠绕  
**关键词**: 流形解缠绕, 流匹配, VAE, 分类器无关引导, 结构化发现

## 一句话总结
提出 WWDC（What We Don't C）方法，利用条件引导的潜在流匹配从已有 VAE 表征中去除已知信息，使未知特征在残余流形中更易被发现和访问，实现迭代式科学发现。

## 研究背景与动机

### 领域现状

**领域现状**：表征学习的核心挑战：如何在学到的表征中访问有意义的信息

### 现有痛点

**现有痛点**：标准解缠绕方法（如 β-VAE、对比学习）试图将所有变化因子分离到各自维度，但在复杂数据上难以实现

### 核心矛盾

**核心矛盾**：关键洞察：与其追求完全解缠绕，不如从已有表征中去除已知信息，让"未知"更易被发现

### 解决思路

**解决思路**：实际需求：在天文学等科学领域，主导信号（如星系形态类别）往往遮蔽了次要但重要的信号

## 方法详解

### 整体框架
- 使用已有的预训练 VAE 表征作为目标分布
- 训练条件流匹配模型，以已知特征作为引导条件
- 反向流映射到基分布时，已知特征被抑制，残余结构保留未知特征
- 迭代循环：发现新特征 → 加入条件 → 去除后继续探索

### 关键设计
1. **条件流匹配**：

    - 使用高斯最优传输路径在基分布（标准正态）和目标分布（VAE 潜在空间）间插值
    - 训练速度场 $u_t^\theta$ 近似最优传输轨迹
    - 损失：$\mathcal{L}_{CFM} = \mathbb{E}_{t,X_0,X_1} \|u_t^\theta(X_t) - (X_1-X_0)\|^2$

2. **分类器无关引导 (CFG)**：

    - 以概率 $p_{cfg}$ 随机丢弃条件信息（用空向量替代）
    - 推理时加权组合引导与非引导速度：$u_t^{CFG} = (1-\omega)u_t(x_1|x_t) + \omega u_t(x_1|x_t,y)$
    - 引导权重 ω 控制条件信息的去除程度

3. **反向流表征**：

    - 将 VAE 样本从 t=1 反向流到 t=0（基分布）
    - 由于引导流的最优传输性质，基分布保持全局结构但去除了条件信息
    - VAE 的 KL 约束使潜在空间近似高斯，与基分布自然对齐，减小结构扭曲

### 损失函数 / 训练策略
- 条件流匹配损失，$p_{cfg} \in [0.1, 0.2]$
- 对已有冻结 VAE 表征进行操作，无需重新训练 VAE
- 中点法 (midpoint method) 用于 ODE 求解
- 不同数据集使用不同 VAE 配置（MNIST: β=1e-4, z∈R^64; Galaxy10: β=1e-6, z∈R^(4×32×32)）

## 实验关键数据

### 2D 高斯实验


### 主实验

| 引导权重 ω | 类别互信息 (t=0) | 距离线性可解释性 R² (t=0) |
|-----------|---------------|---------------------|
| 0 (无引导) | 高 | ~0.3 |
| 0.5 | 中等 | ~0.6 |
| 1.0 (完全引导) | ~0 | ~1.0 |

### 彩色 MNIST 实验


### 消融实验

| 表征空间 | 数字分类准确率 | 蓝色回归 R² |
|---------|-------------|-----------|
| VAE (原始) | 高 | 低 |
| WWDC (引导去除数字+红+绿) | 显著降低 | **提升** |

### Galaxy10 实验
- 使用"Round"类作为引导目标，成功分离星系形态特征
- 残差图清晰显示了被去除的结构特征（旋臂、棒状结构等）
- 背景特征和成像伪影在引导过程中被完整保留

### 关键发现
- 引导权重 ω=1 可几乎完全去除条件信息的互信息
- 被去除的信息不会影响未引导特征的恢复：蓝色值在去除数字/红/绿后仍可被线性模型恢复
- VAE 潜在空间的近高斯性与流匹配基分布的自然对齐是方法有效性的关键
- 风格迁移能力：反向流后用不同条件前向流，可保持笔画宽度、位置、颜色等风格特征

## 亮点与洞察
- 核心思想极富创意：不追求"发现所有"，而是"去除已知，让未知浮现"
- 方法计算高效：复用已有 VAE，仅需训练轻量流匹配模型
- 迭代发现循环的愿景（图 1）：标注 → 条件引导 → 发现新特征 → 继续循环
- 在天文学中的应用场景具有真实影响：大规模巡天数据中复杂特征的系统性探索
- 理论分析简洁有效：从最优传输和 CFG 的角度解释了信息去除和保留的机制

## 局限与展望
- "去除干净"取决于流匹配模型的精度，实际中条件信息可能未完全去除
- 线性探测的评估方式对非线性特征可能不敏感
- Galaxy10 实验仅使用了简单的离散类别作为条件，连续特征（如红移）的效果未知
- 与其他表征学习方法（如 JEPA、对比学习）的 VAE 替代方案未探索
- 规模化到更大数据集和更高维潜在空间的表现有待验证

## 相关工作与启发
- "流形解缠绕"的概念定义与传统"维度解缠绕"形成有意义的区别
- CFG 在扩散/流模型中的应用从此有了新角色：不是为了增强生成质量，而是为了信息分离
- 对科学发现的系统化方法有重要启发：当你知道什么很重要时，主动去除可以帮助发现你不知道的
- 方法可直接应用于LSST等大规模天文巡天中的星系特征探索
- VAE 的近高斯潜在空间与流匹配基分布的天然对齐是方法成功的关键前提

## 评分
- 新颖性：⭐⭐⭐⭐⭐ （概念原创性极强）
- 技术贡献：⭐⭐⭐⭐ （方法简洁但有效）
- 实验充分度：⭐⭐⭐⭐ （从玩具到真实数据的完整验证链）
- 写作质量：⭐⭐⭐⭐⭐ （叙事流畅，图表出色）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Generative Model Inversion Through the Lens of the Manifold Hypothesis](generative_model_inversion_through_the_lens_of_the_manifold_hypothesis.md)
- [\[ICML 2025\] Local Manifold Approximation and Projection for Manifold-Aware Diffusion Planning](../../ICML2025/image_generation/local_manifold_approximation_and_projection_for_manifold-aware_diffusion_plannin.md)
- [\[NeurIPS 2025\] Diffusion-Based Electromagnetic Inverse Design of Scattering Structured Media](diffusion-based_electromagnetic_inverse_design_of_scattering_structured_media.md)
- [\[NeurIPS 2025\] Highlighting What Matters: Promptable Embeddings for Attribute-Focused Image Retrieval](highlighting_what_matters_promptable_embeddings_for_attribute-focused_image_retr.md)
- [\[NeurIPS 2025\] StelLA: Subspace Learning in Low-rank Adaptation using Stiefel Manifold](stella_subspace_learning_in_low-rank_adaptation_using_stiefel_manifold.md)

</div>

<!-- RELATED:END -->
