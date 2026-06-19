---
title: >-
  [论文解读] Dynamic and Generalizable Process Reward Modeling (DG-PRM)
description: >-
  [ACL 2025][LLM推理][process reward model] 提出DG-PRM框架，通过构建层次化奖励树动态存储和选择多维评估标准，结合Pareto支配估计识别多目标下的正负样本对，实现动态、可泛化的过程奖励建模。 - 问题定义：过程奖励模型（PRM）为LLM在复杂推理中的每个中间步骤提供密集奖励信号…
tags:
  - "ACL 2025"
  - "LLM推理"
  - "process reward model"
  - "reward tree"
  - "Pareto dominance"
  - "LLM-as-judge"
  - "dynamic evaluation"
---

# Dynamic and Generalizable Process Reward Modeling (DG-PRM)

**会议**: ACL 2025  
**arXiv**: [2507.17849](https://arxiv.org/abs/2507.17849)  
**代码**: 未公开  
**领域**: LLM推理/奖励建模  
**关键词**: process reward model, reward tree, Pareto dominance, LLM-as-judge, dynamic evaluation  

## 一句话总结

提出DG-PRM框架，通过构建层次化奖励树动态存储和选择多维评估标准，结合Pareto支配估计识别多目标下的正负样本对，实现动态、可泛化的过程奖励建模。

## 研究背景与动机

- **问题定义**：过程奖励模型（PRM）为LLM在复杂推理中的每个中间步骤提供密集奖励信号，对提升推理质量至关重要
- **启发式PRM的局限**：依赖人工制定的固定评估标准（如答案正确性），需要客观参考答案，跨领域泛化能力差，容易遭受reward hacking
- **生成式PRM的局限**：虽利用LLM-as-Judge提供反馈，但现有方法仅使用最终判断（正确/错误），忽略了判断文本中包含的丰富细节信息（如错误严重程度、错误类型）
- **核心观察**：LLM的评判反馈中包含丰富的多维指导信息（如逻辑一致性、计算准确性等），但当前方法对错误步骤统一赋予负奖励，无法区分不同错误的严重程度

## 方法详解

### 整体框架

DG-PRM包含三个核心模块：(1) 自动过程奖励设计——从LLM判断中提取多维评估标准并组织为层次化奖励树；(2) 动态过程奖励分配——根据每个步骤的内容从奖励树中动态选择相关标准进行评分；(3) 多目标奖励优化——使用Pareto支配识别正负样本对进行step-wise DPO训练。

### 关键设计

- **奖励树构建**：对正负输出对 $(y_+, y_-)$ 使用LLM Judge分析差异并提取评估标准 $R_{raw}$ → 过滤低质量标准 → 用文本编码器将标准映射到向量空间 → 通过增量层次聚类构建树结构 $\mathcal{T}$（粗粒度父节点 + 细粒度子节点），余弦距离低于阈值 $\xi$ 的标准合并去重
- **动态奖励分配**：评估步骤 $y^{(t)}$ 时，先从奖励树顶层选择相关父标准 → 分析函数 $\Phi$ 判断是否需要细粒度评估 → 用余弦距离匹配子节点标准（距离 < 阈值 $\zeta$ ）→ 引入滑动窗口 $\mu$ 利用前序步骤的奖励上下文信息
- **Pareto支配优化**：对同一步骤的多个候选输出，在多维奖励分数下计算Pareto前沿 → Pareto最优解为正样本，被支配的解为负样本 → 构造偏好对进行step-wise DPO训练

### 损失函数

基于DPO的step-wise优化目标：

$$\mathcal{L}_{\text{DG-PRM}}(\theta) = -\mathbb{E}_{(\hat{y}_+^{(t)}, \hat{y}_-^{(t)}) \in \mathbf{V}} \left[\log \sigma\left(\beta \Delta^{(t)}\right)\right]$$

其中 $\Delta^{(t)} = r_\theta^{(t)}(\hat{y}_+^{(t)}) - r_\theta^{(t)}(\hat{y}_-^{(t)})$，$r_\theta^{(t)}$ 为策略与参考策略的log-ratio。

## 实验

### 主实验（PRMBench）

| 模型 | Overall | Simplicity | Soundness Avg. | Sensitivity Avg. |
|------|---------|------------|----------------|------------------|
| Llemma-PRM800k-7B | 52.0 | 51.4 | 50.9 | 66.0 |
| RLHFlow-PRM-Mistral-8B | 54.4 | 46.7 | 57.5 | 68.5 |
| GPT-4o (Critic) | 66.8 | 59.7 | 70.9 | 75.8 |
| o1-mini (Critic) | 68.8 | 64.6 | 72.1 | 75.5 |
| DeepSeek-R1 (Critic) | 69.5 | 65.6 | 72.5 | 76.5 |
| **DG-PRM (o1-mini)** | **73.5** | **70.2** | **76.1** | - |

### 消融实验

| 组件 | 效果 |
|------|------|
| 去除奖励树（固定标准） | 性能显著下降，跨领域泛化变差 |
| 去除Pareto支配（随机选正负对） | 训练目标不清晰，性能下降 |
| 去除动态选择（使用所有标准） | 噪声标准干扰评分，性能下降 |
| 去除上下文窗口 | 失去跨步骤一致性信号 |

### 关键发现

- DG-PRM在PRMBench上显著超越所有开源判别式PRM和LLM-as-Critic方法
- 相比直接使用LLM做Critic，DG-PRM训练效率更高且泛化到OOD场景的能力更强
- 奖励树的层次化组织使得细粒度标准可以在不同领域间复用
- Pareto支配估计比简单的正/负二分法提供了更清晰的优化方向

## 亮点

- 首次系统性地利用LLM Judge反馈中的多维细节信息构建过程奖励
- 奖励树结构优雅地解决了评估标准的存储、去重和动态检索问题
- Pareto支配估计是处理多目标奖励信号的自然且有效的方案

## 局限性

- 奖励树的构建依赖高性能LLM（如GPT-4o/o1-mini）的判断质量，API调用成本较高
- 层次聚类的阈值参数（ξ、ζ）和滑动窗口大小 $\mu$ 需要手动调优
- 实验主要在数学推理和评估任务上验证，在代码生成、创意写作等其他推理场景的表现有待探索
- 奖励树随任务领域扩展可能变得庞大，检索效率会受影响
- Pareto支配在高维奖励空间中区分度可能下降（大量解互不支配）

## 相关工作

- 结果奖励模型（ORM）：Stiennon et al. 2020; Ouyang et al. 2022
- 过程奖励模型（PRM）：Lightman et al. 2024; Wang et al. 2024a（Math-Shepherd）
- LLM-as-Judge：Zheng et al. 2023（MT-Bench）; Kwon et al. 2023
- 多目标优化与Pareto：Miettinen 1999
- DPO：Rafailov et al. 2023

## 评分

| 维度 | 分数 |
|------|------|
| 新颖性 | ⭐⭐⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐⭐⭐ |
| 实验充分度 | ⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐⭐ |
| 总体推荐 | ⭐⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Efficient Process Reward Modeling via Contrastive Mutual Information](../../ACL2026/llm_reasoning/efficient_process_reward_modeling_via_contrastive_mutual_information.md)
- [\[ACL 2025\] An Efficient and Precise Training Data Construction Framework for Process-Supervised Reward Model in Mathematical Reasoning](an_efficient_and_precise_training_data_construction_framework_for_process-superv.md)
- [\[NeurIPS 2025\] Unlocking Multimodal Mathematical Reasoning via Process Reward Model](../../NeurIPS2025/llm_reasoning/unlocking_multimodal_mathematical_reasoning_via_process_reward_model.md)
- [\[ACL 2025\] Revisiting Self-Consistency from Dynamic Distributional Alignment Perspective on Answer Aggregation](revisiting_self-consistency_from_dynamic_distributional_alignment_perspective_on.md)
- [\[ACL 2025\] ProcessBench: Identifying Process Errors in Mathematical Reasoning](processbench_identifying_process_errors_in_mathematical_reasoning.md)

</div>

<!-- RELATED:END -->
