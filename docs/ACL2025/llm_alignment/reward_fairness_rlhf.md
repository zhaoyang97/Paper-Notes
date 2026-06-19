---
title: >-
  [论文解读] Towards Reward Fairness in RLHF: From a Resource Allocation Perspective
description: >-
  [ACL 2025][LLM对齐][RLHF] 将 RLHF 中的长度偏差、类别偏差、社会偏差等多种奖励偏差统一定义为"奖励不公平"问题，借鉴资源分配理论提出 Fairness Regularization 和 Fairness Coefficient 两种偏差无关方法，分别应用于奖励模型训练和策略模型训练，在不针对特定偏差设计的前提下同时缓解多种偏差并提升对齐质量。
tags:
  - "ACL 2025"
  - "LLM对齐"
  - "RLHF"
  - "奖励公平性"
  - "资源分配"
  - "偏差缓解"
  - "偏好学习"
---

# Towards Reward Fairness in RLHF: From a Resource Allocation Perspective

**会议**: ACL 2025  
**arXiv**: [2505.23349](https://arxiv.org/abs/2505.23349)  
**代码**: [https://github.com/shoyua/Towards-Reward-Fairness](https://github.com/shoyua/Towards-Reward-Fairness)  
**领域**: LLM对齐  
**关键词**: RLHF, 奖励公平性, 资源分配, 偏差缓解, 偏好学习

## 一句话总结

将 RLHF 中的长度偏差、类别偏差、社会偏差等多种奖励偏差统一定义为"奖励不公平"问题，借鉴资源分配理论提出 Fairness Regularization 和 Fairness Coefficient 两种偏差无关方法，分别应用于奖励模型训练和策略模型训练，在不针对特定偏差设计的前提下同时缓解多种偏差并提升对齐质量。

## 研究背景与动机

- **领域现状**: RLHF 的核心假设是奖励模型（RM）能准确代理人类偏好。RM 按 Bradley-Terry 模型训练，对每个偏好对 $(x, y^w, y^l)$ 保证 $r(y^w) > r(y^l)$。然而，配对内的相对顺序正确并不意味着跨配对的绝对奖励分布合理——来自不同类别或长度的数据可能获得系统性不对等的奖励，进而误导策略模型的优化方向。
- **已有方法的不足**: 现有工作针对不同偏差提出不同的定制方案——长度偏差使用长度正则化（R-DPO 等），类别偏差使用模型集成或多样化偏好学习。这种"一个偏差一个方法"的碎片化策略不可扩展，且无法处理未知类型的偏差。
- **核心矛盾**: BT 模型只优化配对内 margin $r(y^w) - r(y^l)$，不约束跨配对的奖励绝对值分布。结果是不喜欢的回答 $y_1^l$ 的奖励可能高于另一对中喜欢的回答 $y_2^w$，当策略模型基于绝对奖励做最大化时就会朝错误方向优化。
- **本文动机**: 将各种奖励偏差视为"奖励不公平"的不同表现形式，从资源分配理论借鉴统一的公平性度量，在效用（utility）和公平（fairness）之间做权衡，提出 bias-agnostic 的通用解法。

## 方法详解

### 整体框架

将偏好学习建模为资源分配问题：奖励是待分配的"资源"，分配给不同的数据组（按长度/类别/人口统计划分）。优化目标从单一的效用最大化（$\max U(\mathbf{a})$，即标准 BT 损失）扩展为效用-公平联合优化，提出两种组合方式：加法形式的 Fairness Regularization 和乘法形式的 Fairness Coefficient。框架在两个场景下实例化：(1) 训练公平奖励模型（Fairness RM），(2) 训练公平策略模型（Fairness DPO）。

### 关键设计

1. **统一公平性度量函数**
    - **功能**: 提供一个满足连续性、齐次性、单调性三大性质的公平性指标族
    - **做法**: 采用 Lan et al. (2010) 提出的统一公平度量 $f_\tau(\mathbf{a}) = \text{sign}(1-\tau) \cdot \left[\sum_{i=1}^{n}\left(\frac{a_i}{\sum_j a_j}\right)^{1-\tau}\right]^{1/\tau}$，其中 $a_i = r_\phi(y^w_i) - r_\phi(y^l_i)$ 为第 $i$ 个偏好对的奖励 margin，$\tau$ 控制公平函数族的形状（$\tau = -1$ 退化为 Jain's Index）
    - **设计动机**: 齐次性保证公平度量与奖励尺度无关，可省去资源总量约束；统一参数 $\tau$ 允许从一族公平函数中灵活选择；实验证明对 $\tau$ 不敏感，各取值均优于基线

2. **Fairness Regularization（FR，加法公平正则）**
    - **功能**: 在训练目标中以正则项形式引入公平约束
    - **做法**: 损失函数为 $\mathcal{L}_{\text{FR}} = -\mathbb{E}[\log\sigma(a_i)] - \alpha \cdot f_\tau(\mathbf{a})$，第一项是标准 BT 效用损失，第二项是公平正则。超参 $\alpha$ 控制公平-效用权衡。对于 DPO 场景，$a_i$ 替换为隐式奖励 margin $\beta\log\frac{\pi_\theta(y^w|x)}{\pi_{\text{ref}}(y^w|x)} - \beta\log\frac{\pi_\theta(y^l|x)}{\pi_{\text{ref}}(y^l|x)}$
    - **设计动机**: 加法形式实现简单，公平项梯度独立于效用项，便于调节权衡；$\alpha=0.1$ 时在多个基准上取得最佳平衡

3. **Fairness Coefficient（FC，乘法公平系数）**
    - **功能**: 以乘法系数形式动态缩放效用损失
    - **做法**: 损失函数为 $\mathcal{L}_{\text{FC}} = -\mathbb{E}[\log\sigma(a_i)] \cdot f_\tau(\mathbf{a})^\gamma$，公平度量作为效用损失的权重系数。当奖励分布越不公平（$f_\tau$ 越小），整体损失越小——梯度自动将优化方向推向更公平的分布
    - **设计动机**: 乘法形式让公平性自适应调节——不公平时自动加大正则效果，公平时保持正常训练。无需手动调节 $\alpha$，只需设定 $\gamma$

### 两个应用场景

| 场景 | 目标 | 输入 | 分配向量 $a_i$ 定义 | 输出 |
|------|------|------|-------|------|
| Fairness RM（验证） | 训练公平奖励模型 | 偏好对 $(x, y^w, y^l)$ | $r_\phi(y^w) - r_\phi(y^l)$ | 公平的奖励打分器，用于 BoN 数据选择 |
| Fairness DPO（RL） | 训练公平策略模型 | 偏好对 + 参考模型 $\pi_{\text{ref}}$ | 隐式奖励 margin | 偏好对齐且公平的生成模型 |

## 实验关键数据

### 奖励模型公平性验证（LLaMA3-SFT 基座）

| 验证器 | Reward Bench Avg. | Chat | Chat Hard | Reasoning | Safety | HH-RLHF Avg. |
|--------|-------------------|------|-----------|-----------|--------|--------------|
| BT RM（基线） | 78.11 | 93.02 | 57.02 | 84.98 | 77.43 | 73.81 |
| FR RM | **78.38** | **94.41** | 57.02 | 83.86 | **78.24** | 73.55 |
| FC RM | 77.50 | **94.41** | 53.29 | **85.53** | 76.76 | 73.96 |

公平 RM 在精度不下降的前提下显著改善了 Helpful 与 Harmless 类别间的奖励分布一致性。

### 策略模型对齐质量

| 基座 | 方法 | AlpacaEval2 LC WR | AlpacaEval2 WR | MT-Bench |
|------|------|-------------------|----------------|----------|
| LLaMA3 | DPO | 16.71 | 14.23 | 6.46 |
| LLaMA3 | R-DPO（长度正则） | 20.87 | 11.16 | 6.48 |
| LLaMA3 | KTO | 19.44 | 16.64 | 6.64 |
| LLaMA3 | **FR DPO** | 20.48 | 15.74 | **6.70** |
| LLaMA3 | **FC DPO** | **21.10** | **16.96** | 6.58 |
| Qwen2.5 | DPO | 18.93 | 13.18 | 6.59 |
| Qwen2.5 | **FR DPO** | **21.05** | **15.25** | **7.24** |
| Qwen2.5 | **FC DPO** | 19.72 | 14.53 | 7.00 |

### 关键发现

- FR/FC DPO 在两个基座模型上均持续优于标准 DPO 和定制化的 R-DPO，验证了偏差无关方法的有效性
- 公平约束不仅不损害对齐质量，反而带来提升——因为减少偏差使模型学到更准确的偏好信号
- 公平 RM 在 BoN 数据选择中可用更少的采样达到相同性能，采样效率更高
- 在 CrowS-Pairs 社会偏差数据集上，FR/FC RM 的刻板印象与非刻板印象奖励分布差距显著缩小
- 消融实验表明 $\tau \in [-5, 10]$ 范围内所有公平函数均优于基线，方法对 $\tau$ 健壮；$\alpha = 0.1$ 为最优效用-公平权衡点

## 亮点与洞察

- **资源分配视角**是核心理论创新——将 RLHF 偏差与经济学公平分配理论桥接，提供了统一的分析框架。
- **偏差无关（bias-agnostic）**比定制方案更有实践意义——实际场景中偏差类型往往未知或混合存在。
- 揭示 BT 模型的结构性缺陷：配对内正确 ≠ 跨配对公平，这是策略模型被误导的根本原因。
- FC 方法的"即插即用"特性对已部署的 RLHF 系统有直接应用价值——无需重训奖励模型。

## 局限与展望

- 公平性度量需要预定义数据分组方式（按长度/类别/人口统计），分组质量影响效果
- $\alpha$ / $\gamma$ 超参仍需在验证集上调节
- 仅在 BT RM + DPO 上验证，未扩展到 PPO、GRPO 等其他 RL 算法
- 奖励不公平与 reward hacking 的关系未深入探讨

## 相关工作对比

- **vs R-DPO（长度正则）**: 只缓解长度偏差；本文通用缓解多种偏差，且效果更好（LC WR 21.10 vs 20.87）
- **vs ODIN（解耦奖励）**: 通过多维度解耦奖励缓解 hacking；本文从资源分配角度约束分布公平——视角互补
- **vs DPO 原始**: DPO 绕过显式 RM 但隐式拟合奖励，本文证明其隐式奖励同样存在不公平问题，FR/FC 可直接应用

## 评分

- **新颖性**: ⭐⭐⭐⭐ 资源分配视角统一理解奖励偏差新颖，公平度量族设计巧妙
- **实验充分度**: ⭐⭐⭐⭐ 覆盖 RM 验证 + DPO 策略 + 社会偏差 + 数据选择 + 消融，双基座验证
- **写作质量**: ⭐⭐⭐⭐ 理论推导严谨，从资源分配到公平度量到具体方法层层递进
- **实用价值**: ⭐⭐⭐⭐ FC 即插即用、对 $\tau$ 健壮、不牺牲效用，对 RLHF 实际部署有直接价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Reward Generalization in RLHF: A Topological Perspective](reward_generalization_in_rlhf_a_topological_perspective.md)
- [\[ACL 2025\] Aligning to What? Limits to RLHF Based Alignment](aligning_to_what_limits_to_rlhf_based_alignment.md)
- [\[ICML 2025\] Can RLHF be More Efficient with Imperfect Reward Models? A Policy Coverage Perspective](../../ICML2025/llm_alignment/can_rlhf_be_more_efficient_with_imperfect_reward_models_a_policy_coverage_perspe.md)
- [\[ACL 2025\] SEA: Low-Resource Safety Alignment for Multimodal Large Language Models via Synthetic Embeddings](sea_lowresource_safety_alignment_for_multimodal.md)
- [\[ACL 2025\] Rethinking Reward Model Evaluation Through the Lens of Reward Overoptimization](rethinking_reward_model_evaluation_through_the_lens_of_reward_overoptimization.md)

</div>

<!-- RELATED:END -->
