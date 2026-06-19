---
title: >-
  [论文解读] How Intrinsic Motivation Shapes Learned Representations in Decision Transformers: A Cognitive Interpretability Analysis
description: >-
  [NeurIPS 2025][可解释性][内在动机] 提出一个系统性的事后可解释性框架，分析内在动机（基于Random Network Distillation）如何塑造Elastic Decision Transformer的嵌入空间几何结构，揭示不同内在动机变体创造了根本不同的表示结构——EDT-SIL促进紧凑表示，EDT-TIL增强正交性——且嵌入属性与任务性能存在强烈的环境特异性相关。
tags:
  - "NeurIPS 2025"
  - "可解释性"
  - "内在动机"
  - "Transformer"
  - "嵌入分析"
  - "表示学习"
---

# How Intrinsic Motivation Shapes Learned Representations in Decision Transformers: A Cognitive Interpretability Analysis

**会议**: NeurIPS 2025  
**arXiv**: [2506.13958](https://arxiv.org/abs/2506.13958)  
**代码**: 暂无  
**领域**: 可解释性  
**关键词**: 内在动机, Decision Transformer, 可解释性, 嵌入分析, 表示学习

## 一句话总结

提出一个系统性的事后可解释性框架，分析内在动机（基于Random Network Distillation）如何塑造Elastic Decision Transformer的嵌入空间几何结构，揭示不同内在动机变体创造了根本不同的表示结构——EDT-SIL促进紧凑表示，EDT-TIL增强正交性——且嵌入属性与任务性能存在强烈的环境特异性相关。

## 研究背景与动机

Elastic Decision Transformer（EDT）结合了Transformer的序列建模能力与决策制定，通过动态历史长度调整实现高效的离线RL。将内在动机（如基于好奇心的RND）整合到EDT中已被经验性地证明可以提升性能，但**性能提升背后的表示机制仍不清楚**。

核心问题：内在动机到底是仅仅作为探索奖励发挥作用，还是在更深层次上**重塑了模型的内部表示结构**？理解这一机制对可解释RL至关重要，因为这些模型在高维空间中学习隐式状态表示，缺乏传统手工特征的可解释性。

本文的核心假设：**内在动机不仅仅是探索奖励加成，更是一种表示先验（representational prior），以类似生物神经系统的方式塑造嵌入空间的几何结构**。

## 方法详解

### 整体框架

基于EDT架构，比较三种模型变体：基线EDT、EDT-SIL（State Input Loss）和EDT-TIL（Transformer Input Loss），通过统计分析嵌入空间的几何属性来理解内在动机的机制性影响。

### 关键设计

1. **两种内在动机变体**：

    - **EDT-SIL**：内在损失直接作用于嵌入层的状态表示，$L_{\text{int}} = |f_{\text{pred}}(x_{\text{embed}}; \theta_{\text{pred}}) - f_{\text{target}}(x_{\text{embed}}; \theta_{\text{target}})|_2^2$。RND模块看到的是嵌入后的状态，因此内在信号直接影响状态嵌入层的学习，倾向于促进更紧凑的表示。
    - **EDT-TIL**：内在损失作用于Transformer输出表示，使内在信号同时影响嵌入层和Transformer层，能够塑造更连贯的序列表示，倾向于增强正交性。
    - 总损失：$L_{\text{overall}} = L_{\text{EDT}} + L_{\text{int}}$

2. **嵌入分析框架（三个关键指标）**：

    - **协方差迹（Covariance Trace）**：$\text{cov\_trace} = \text{Tr}(\text{Cov}(E))$，衡量嵌入维度上的总方差分布，反映表示空间捕获的总信息量
    - **L2范数**：$\text{l2\_norm} = \frac{1}{N}\sum_{i=1}^{N}|e_i|_2$，量化嵌入向量的平均幅度，反映表示紧凑性
    - **余弦相似度**：评估嵌入对之间的平均余弦相似度，反映表示正交性。低余弦相似度意味着状态表示更分散、更易区分

3. **定量关联分析**：计算嵌入指标与归一化性能分数之间的Pearson相关系数，识别每个环境-模型组合中最具预测性的指标。

### 损失函数 / 训练策略

- 使用D4RL基准的Medium和Medium-Replay数据集
- 评估四种连续控制任务：Ant、HalfCheetah、Hopper、Walker2d
- 每种配置使用5个随机种子，嵌入分析在3次重复上取平均
- RND模块使用3层预测网络（经系统性调参确定为最优配置）

## 实验关键数据

### 主实验

D4RL Medium数据集上的Human-Normalized Scores (HNS)：

| 模型 | Ant | HalfCheetah | Hopper | Walker2d |
|------|-----|-------------|--------|----------|
| EDT (Baseline) | 88.84±3.61 | **42.30**±0.14 | 57.49±3.81 | 68.50±2.03 |
| EDT-SIL | **90.49**±5.01 | 42.46±0.12 | 59.31±6.16 | 69.44±4.46 |
| EDT-TIL | 89.01±5.83 | 42.18±0.34 | **59.63**±2.35 | **73.50**±4.29 |

Medium-Replay数据集：

| 模型 | Ant | HalfCheetah | Hopper | Walker2d |
|------|-----|-------------|--------|----------|
| EDT | **85.51**±5.06 | 37.32±2.46 | 81.56±9.96 | 62.25±5.21 |
| EDT-SIL | 84.02±3.72 | 37.64±2.44 | **84.67**±4.80 | 57.21±8.54 |
| EDT-TIL | 83.72±4.13 | **38.60**±1.28 | 81.72±9.27 | **65.06**±3.81 |

### 消融实验

嵌入属性与性能的环境特异性相关分析：

| 环境 | 最强相关指标 | 相关系数 | EDT-SIL效果 | EDT-TIL效果 |
|------|-----------|---------|-----------|-----------|
| Ant | 协方差迹 | r = -0.907 | 降低迹（526→620对比基线） | 中等降低（573） |
| HalfCheetah | 协方差迹 | r = +0.850 | 略增（632） | 降低（563） |
| Hopper | 余弦相似度 | r = +0.658 | 增加（0.082） | **增加最多（0.117）** |
| Walker2d | 余弦相似度 | r = -0.950 | 增加（0.083） | **降低（0.073）** |

### 关键发现

- **EDT-SIL始终创造更紧凑的表示**：通过降低协方差迹和L2范数，在输入层压缩信息
- **EDT-TIL促进表示正交性**：通过变化余弦相似度优化状态可区分性，特别在Walker2d中效果显著（余弦相似度从0.081降至0.073，相关系数r = -0.950）
- **环境特异性**：不同环境的性能由不同的嵌入属性驱动——Ant靠方差控制，Walker2d靠正交性。这表明内在动机创造了与任务需求对齐的定制化表示结构
- **3层RND最优**：过少（1层）缺乏容量，过多（10层）导致过拟合或表示不稳定

## 亮点与洞察

- 首次从表示几何角度解释内在动机的作用机制，超越了"探索奖励"的简单理解
- 揭示了一个有趣的生物对应：EDT-SIL和EDT-TIL的互补机制类似于生物神经系统中不同处理阶段维持不同稳态机制的层级组织原则
- 内在动机作为"表示先验"的观点为设计更好的辅助损失函数提供了新的方向

## 局限与展望

- HNS提升幅度较小（百分之几的量级），统计显著性有待进一步验证
- 分析框架仅关注嵌入空间的几何属性，未使用显式的可解释性方法（如SHAP、Attention可视化）
- 仅分析了状态嵌入层，未扩展到Transformer输出和动作表示
- 未探索嵌入结构在训练过程中的时间演化，缺少动态视角
- 任务局限于D4RL连续控制，未涉及离散动作或多模态观测环境

## 相关工作与启发

- **Elastic Decision Transformer (EDT)**：通过动态历史长度实现灵活的离线RL策略
- **Random Network Distillation (RND)**：经典的内在动机方法，通过预测误差衡量新颖性
- **Allostatic Regulation**：生物学中的预测性适应概念，内在动机损失类似于变态调节器
- **启发**：辅助损失函数不仅通过梯度信号影响学习，还通过隐式地约束表示空间的几何结构发挥作用——这为设计更好的自监督/辅助任务提供了新思路

## 评分

- **新颖性**: ⭐⭐⭐⭐ 从表示几何角度分析内在动机是新颖的切入点
- **实验充分度**: ⭐⭐⭐ 环境和数据集有限，性能提升幅度小
- **写作质量**: ⭐⭐⭐⭐ 分析框架清晰，生物类比有启发性
- **价值**: ⭐⭐⭐⭐ 为理解辅助损失函数的表示层面影响提供了分析工具和见解

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] How Do Transformers Learn Implicit Reasoning?](how_do_transformers_learn_implicit_reasoning.md)
- [\[NeurIPS 2025\] nnterp: A Standardized Interface for Mechanistic Interpretability of Transformers](nnterp_a_standardized_interface_for_mechanistic_interpretability_of_transformers.md)
- [\[ICML 2026\] Cognitive Fatigue in Autoregressive Transformers: Formalization and Measurement](../../ICML2026/interpretability/cognitive_fatigue_in_autoregressive_transformers_formalization_and_measurement.md)
- [\[NeurIPS 2025\] CBMAS: Cognitive Behavioral Modeling via Activation Steering](cbmas_cognitive_behavioral_modeling_via_activation_steering.md)
- [\[CVPR 2025\] Prompt-CAM: Making Vision Transformers Interpretable for Fine-Grained Analysis](../../CVPR2025/interpretability/prompt-cam_making_vision_transformers_interpretable_for_fine-grained_analysis.md)

</div>

<!-- RELATED:END -->
