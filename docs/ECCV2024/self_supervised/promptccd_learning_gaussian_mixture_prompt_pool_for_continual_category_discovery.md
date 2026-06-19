---
title: >-
  [论文解读] PromptCCD: Learning Gaussian Mixture Prompt Pool for Continual Category Discovery
description: >-
  [ECCV 2024][自监督学习][持续类别发现] 提出PromptCCD框架，利用高斯混合模型（GMM）作为提示池，实现在无标签数据流中的持续新类别发现，同时缓解灾难性遗忘。 持续类别发现（Continual Category Discovery, CCD）：是一个新兴且极具挑战性的问题——在持续到来的无标签数据流中自动…
tags:
  - "ECCV 2024"
  - "自监督学习"
  - "持续类别发现"
  - "高斯混合模型"
  - "提示学习"
  - "灾难性遗忘"
  - "无标签类别"
---

# PromptCCD: Learning Gaussian Mixture Prompt Pool for Continual Category Discovery

**会议**: ECCV 2024  
**arXiv**: [2407.19001](https://arxiv.org/abs/2407.19001)  
**代码**: 有 (项目页面)  
**领域**: Self-supervised Learning  
**关键词**: 持续类别发现, 高斯混合模型, 提示学习, 灾难性遗忘, 无标签类别

## 一句话总结

提出PromptCCD框架，利用高斯混合模型（GMM）作为提示池，实现在无标签数据流中的持续新类别发现，同时缓解灾难性遗忘。

## 研究背景与动机

**持续类别发现（Continual Category Discovery, CCD）**是一个新兴且极具挑战性的问题——在持续到来的无标签数据流中自动发现新的类别，同时不遗忘已发现的类别。这个问题结合了两个已知困难问题的挑战：(1) **广义类别发现（GCD）**——在部分标注数据中发现新类别；(2) **持续学习**——在不遗忘旧知识的前提下学习新知识。

核心矛盾：在CCD设置中，数据以流的形式持续到来，每个阶段的数据都可能包含已知类别和未知新类别的样本，且完全无标签。模型需要同时做两件困难的事情：发现新类别的结构，以及保持对旧类别的记忆。即使在全监督的持续学习中，灾难性遗忘仍是未解决的问题，在完全无监督的CCD中更加严重。

此外，一个额外的挑战是：模型需要在不知道每个阶段新类别数量的情况下自动估计类别数。这要求方法不仅要能聚类，还要能自动确定聚类数量。

本文提出PromptCCD，利用基于高斯混合模型的提示池（GMP）作为核心机制，同时解决类别发现、遗忘防止和类别数量估计三个问题。

## 方法详解

### 整体框架

PromptCCD基于预训练的ViT特征提取器和可学习的提示池。在每个阶段：(1) 使用当前的提示池对新数据进行特征增强；(2) 使用GMM进行聚类以发现新类别；(3) 更新提示池以容纳新发现的类别；(4) 通过GMP的结构化记忆机制防止遗忘。

### 关键设计

1. **Gaussian Mixture Prompting (GMP) 模块**:
    - 功能：充当动态提示池，随时间更新以支持类别发现和遗忘防止
    - 核心思路：将提示池建模为高斯混合模型，每个高斯分量对应一个已发现的类别。每个分量存储均值（类别中心）、协方差（类别分布）和关联的提示向量。当新样本到来时，根据其与各分量的后验概率选择最匹配的提示进行特征增强
    - 设计动机：传统的提示池（如L2P、DualPrompt）使用离散的键值匹配，无法自然地扩展到新类别。GMM提供了概率化的框架，可以自然地添加新分量（新类别），同时保持对旧分量的记忆

2. **在线类别数量估计**:
    - 功能：在不知道真实类别数的情况下自动估计每个阶段的新类别数量
    - 核心思路：利用GMM的BIC（贝叶斯信息准则）或轮廓系数来自动确定最优的高斯分量数量。在每个新阶段，先用当前数据拟合不同分量数的GMM，选择BIC最优的分量数作为类别数估计
    - 设计动机：实际应用中类别数量未知是常态，该设计使方法更接近现实场景

3. **渐进式提示池更新策略**:
    - 功能：在发现新类别时扩展提示池，同时保护旧类别的知识
    - 核心思路：当发现新类别时，在GMM中添加新的高斯分量和对应的提示向量。旧分量的参数通过EMA更新缓慢漂移以适应数据分布变化，但核心结构保持不变。提示选择机制确保旧类别的样本始终能匹配到正确的旧提示
    - 设计动机：提示池的渐进扩展避免了"覆盖"旧知识，而EMA更新允许必要的参数微调

### 损失函数 / 训练策略

- 对比损失：利用自监督对比学习在特征空间中学习紧凑的类别表示
- GMM似然损失：最大化数据在当前GMM模型下的对数似然
- 正则化损失：约束提示更新幅度，防止遗忘
- 训练策略：每个阶段分两步——先进行特征提取和聚类，再更新提示池和GMM参数

## 实验关键数据

### 主实验

| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| CIFAR-100 (CCD) | All Acc | 显著领先 | GCD+持续学习基线 | +5-10% |
| ImageNet-100 (CCD) | All Acc | SOTA | 多个基线 | +3-8% |
| CUB-200 (CCD) | New Acc | 大幅领先 | GM方法 | +8-15% |
| 类别数量估计 | 估计误差 ↓ | 较低 | K-means+间隙统计 | 更准确 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 无GMP（固定提示） | 遗忘严重 | 旧类别性能显著下降 |
| GMM → K-means | 性能下降 | K-means无法提供类别数估计 |
| 无类别数估计（固定K） | 与真实K相关 | 性能对K的选择敏感 |
| 完整PromptCCD | 最优 | 三个组件互相配合 |

### 关键发现

- GMM提示池显著优于传统的键值对提示池在CCD任务上的表现
- 自动类别数量估计使方法摆脱了对先验知识的依赖
- 在"新类别发现"指标上的提升尤为显著，说明GMP有效促进了新类别的发现
- 方法在遗忘旧类别方面也表现良好，验证了提示池的记忆保持能力

## 亮点与洞察

- 将GMM与提示学习优雅地结合，每个高斯分量自然对应一个类别
- 同时解决了三个问题：类别发现、遗忘防止和类别数估计
- GMM的概率化框架比离散的提示匹配更灵活
- 标准化GCD到CCD的评估指标扩展，为后续工作提供了基准

## 局限与展望

- GMM假设类别分布近似高斯，对于复杂的多模态分布可能不够灵活
- 提示池随类别数增加而线性增长，长期运行可能面临内存压力
- BIC选择类别数的方法在小样本场景下可能不够稳定
- 可以探索层次化GMM来处理类别间的层级关系
- 跨模态（如文本辅助的视觉类别发现）是有价值的扩展

## 相关工作与启发

- **GCD**: Vaze et al.提出的广义类别发现，是CCD的单阶段版本
- **L2P / DualPrompt**: 提示学习在持续学习中的应用，但不支持无监督类别发现
- **DCCL**: 持续对比学习方法，但未使用提示池
- 启发：概率化模型（如GMM）在持续学习中的应用值得更多探索

## 评分

- 新颖性: ⭐⭐⭐⭐ GMM提示池的设计巧妙，CCD问题定义有价值
- 实验充分度: ⭐⭐⭐⭐ 多数据集评估，包含类别数估计实验
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，方法描述系统
- 价值: ⭐⭐⭐⭐ CCD是一个有实际意义的新方向，PromptCCD提供了强基线

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Is Parameter Isolation Better for Prompt-Based Continual Learning?](../../CVPR2026/self_supervised/is_parameter_isolation_better_for_prompt-based_continual_learning.md)
- [\[CVPR 2026\] Decouple Your Discovery and Memory in Continual Generalized Category Discovery](../../CVPR2026/self_supervised/decouple_your_discovery_and_memory_in_continual_generalized_category_discovery.md)
- [\[ICLR 2026\] Adaptive Gaussian Expansion for On-the-fly Category Discovery](../../ICLR2026/self_supervised/adaptive_gaussian_expansion_for_on-the-fly_category_discovery.md)
- [\[CVPR 2026\] Spectral Mixture-of-Experts for Continual Learning](../../CVPR2026/self_supervised/spectral_mixture-of-experts_for_continual_learning.md)
- [\[ECCV 2024\] Revisiting Supervision for Continual Representation Learning](revisiting_supervision_for_continual_representation_learning.md)

</div>

<!-- RELATED:END -->
