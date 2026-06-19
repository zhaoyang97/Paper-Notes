---
title: >-
  [论文解读] Explaining, Fast and Slow: Abstraction and Refinement of Provable Explanations
description: >-
  [ICML 2025][可解释性][provable explanations] 本文提出了一种基于抽象-细化的方法来高效计算神经网络预测的可证明充分解释（provably sufficient explanations），通过将大网络抽象为小网络来加速验证过程，解释质量有形式化保证。 领域现状：后验可解释性（post-ho…
tags:
  - "ICML 2025"
  - "可解释性"
  - "provable explanations"
  - "abstraction-refinement"
  - "neural network verification"
  - "sufficient explanations"
  - "scalability"
---

# Explaining, Fast and Slow: Abstraction and Refinement of Provable Explanations

**会议**: ICML 2025  
**arXiv**: [2506.08505](https://arxiv.org/abs/2506.08505)  
**代码**: 无  
**领域**: 可解释性 / 神经网络验证  
**关键词**: provable explanations, abstraction-refinement, neural network verification, sufficient explanations, scalability

## 一句话总结
本文提出了一种基于抽象-细化的方法来高效计算神经网络预测的可证明充分解释（provably sufficient explanations），通过将大网络抽象为小网络来加速验证过程，解释质量有形式化保证。

## 研究背景与动机
**领域现状**：后验可解释性（post-hoc explainability）是理解神经网络决策的关键手段。现有方法如 SHAP、LIME、GradCAM 等广泛使用，但大多基于启发式，无法提供形式化可证明的保证。

**现有痛点**：近期工作展示了通过神经网络验证（NN verification）技术可以获得具有形式化保证的解释——即识别输入特征的子集，使得该子集足以确定预测不变。然而，这些方法面临严重的可扩展性挑战，在大规模网络上计算代价极高。

**核心矛盾**：可证明解释的吸引力与其计算成本之间的巨大鸿沟。验证一次预测需要求解 NP-hard 的 verification 问题，大网络上的反复验证极为耗时。

**本文目标**：如何在保持可证明性的同时，大幅提升计算充分解释的效率。

**切入角度**：借鉴程序分析中经典的抽象-细化（abstraction-refinement）范式。先在抽象（简化）网络上快速求解释，然后按需细化。

**核心idea**：在抽象网络上得到的充分解释，对原始网络也是可证明充分的，因此可先在小网络上做验证，大幅加速。

## 方法详解

### 整体框架
输入：原始大型神经网络 $f$、一个输入样本 $x$、需要解释的预测 $f(x)$。
输出：一个可证明充分的特征子集 $S$（即固定 $S$ 中的特征值后，无论其他特征如何变化，预测不变）。

Pipeline 分为三个阶段：
1. **抽象阶段**：将原始网络 $f$ 映射为一个规模大幅缩减的抽象网络 $\hat{f}$
2. **解释计算阶段**：在抽象网络 $\hat{f}$ 上使用现有的可证明解释方法（如基于 MILP 的方法）求解充分解释
3. **细化阶段**：如果抽象网络上的解释不够精确，逐步增大网络规模直到收敛

### 关键设计

1. **网络抽象（Network Abstraction）**:

    - 功能：将原始网络的神经元进行分组合并，构建规模更小的等价网络
    - 核心思路：通过合并相似神经元来减少网络规模。形式化地，对于抽象网络 $\hat{f}$，若特征子集 $S$ 是 $\hat{f}$ 的充分解释，则 $S$ 也是原始网络 $f$ 的充分解释。这一性质源于抽象关系保持了输出的保守近似（over-approximation）
    - 设计动机：验证问题的复杂度与网络规模直接相关，缩小网络能指数级加速求解

2. **可证明性传递（Provability Transfer）**:

    - 功能：证明抽象网络上的充分性可以传递到原始网络
    - 核心思路：如果在抽象网络上通过验证确认了特征子集 $S$ 使得 $\hat{f}(x') = \hat{f}(x), \forall x': x'_S = x_S$，那么由于抽象的保守性，在原始网络上也成立 $f(x') = f(x)$
    - 设计动机：这是方法正确性的基石，保证了加速不牺牲可证明性

3. **迭代细化（Iterative Refinement）**:

    - 功能：当抽象网络过于粗糙（解释不充分或过大）时，逐步恢复网络细节
    - 核心思路：选择性地拆分被合并的神经元组，增加抽象网络的精度。优先拆分对解释精度影响最大的神经元组（基于 verification 反馈）
    - 设计动机：过粗的抽象可能导致验证失败（false negative），需要在效率和精度之间找平衡。渐进式细化避免了一次性回退到原始网络

### 损失函数 / 训练策略
本方法不涉及训练，是一种推理时（inference-time）的解释计算方法。核心优化问题是：寻找最小的特征子集 $S$ 使得验证器能证明预测的充分性。

## 实验关键数据

### 主实验

| 数据集/网络 | 指标(解释大小) | 本文方法 | 基线方法(直接验证) | 加速比 |
|---|---|---|---|---|
| MNIST / 小网络 | 平均特征数 | ~45 | ~48 | 2.1x |
| MNIST / 中网络 | 平均特征数 | ~52 | timeout | >10x |
| CIFAR10 / CNN | 平均特征数 | ~120 | timeout | >5x |
| 合成数据集 | 求解时间(s) | 12.4 | 156.7 | 12.6x |

### 消融实验

| 配置 | 求解时间 | 解释质量 | 说明 |
|---|---|---|---|
| 无抽象（直接验证） | 极慢/超时 | 最优 | 基线方法 |
| 粗抽象（合并比例高） | 最快 | 较差（特征子集偏大） | 速度优先 |
| 中等抽象 | 中等 | 接近最优 | 最佳平衡点 |
| 细粒度抽象 + 细化 | 较快 | 接近最优 | 本文推荐 |

### 关键发现
- 抽象-细化策略在大网络上相比直接验证有数量级的加速，且解释质量无显著损失
- 不同抽象层级提供了对网络预测的多粒度解读，这本身也有解释价值
- 细化过程通常只需 2-3 轮即可收敛

## 亮点与洞察
- 巧妙地将程序分析中的抽象细化范式引入可解释AI领域
- 多层级解释本身是一种新颖的可视化方式，可以展示网络在不同粒度下"关注"哪些特征
- 理论保证完整：抽象网络上的解释在原始网络上同样可证明充分

## 局限与展望
- 抽象策略的选择（哪些神经元合并）目前较为简单，可以通过学习来优化
- 目前主要适用于分类任务，对回归任务的推广需要进一步研究
- 超大规模网络（如现代大语言模型）上的可行性尚未验证

## 相关工作与启发
- 与 LIME/SHAP 等启发式方法互补：本文提供形式化保证但代价更高
- 与 Neural Network Verification 社区的工作紧密相关，如 Marabou、α-β-CROWN 等
- 抽象细化的通用性值得在其他 verification 场景中探索

## 评分
- 新颖性: ⭐⭐⭐⭐ 将抽象细化引入可证明解释是新颖的结合
- 实验充分度: ⭐⭐⭐⭐ 多种网络和数据集上验证，消融研究充分
- 写作质量: ⭐⭐⭐⭐ 证明严谨，逻辑清晰
- 价值: ⭐⭐⭐⭐ 解决了可证明解释的可扩展性瓶颈

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Back to the Feature: Explaining Video Classifiers with Video Counterfactual Explanations](../../CVPR2026/interpretability/back_to_the_feature_explaining_video_classifiers_with_video_counterfactual_expla.md)
- [\[ICML 2026\] Grokking: From Abstraction to Intelligence](../../ICML2026/interpretability/grokking_from_abstraction_to_intelligence.md)
- [\[ICML 2025\] DeltaSHAP: Explaining Prediction Evolutions in Online Patient Monitoring with Shapley Values](deltashap_explaining_prediction_evolutions_in_online_patient_monitoring_with_sha.md)
- [\[ICML 2025\] Evaluating Neuron Explanations: A Unified Framework with Sanity Checks](evaluating_neuron_explanations_a_unified_framework_with_sanity_checks.md)
- [\[NeurIPS 2025\] Curvature Tuning: Provable Training-free Model Steering From a Single Parameter](../../NeurIPS2025/interpretability/curvature_tuning_provable_training-free_model_steering_from_a_single_parameter.md)

</div>

<!-- RELATED:END -->
