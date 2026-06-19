---
title: >-
  [论文解读] Test-Time Visual In-Context Tuning
description: >-
  [CVPR 2025][LLM 其他][上下文学习] 本文提出VICT（Visual In-Context Tuning），通过翻转任务提示和测试样本的角色并利用循环一致性损失，在测试时对视觉上下文学习模型（如Painter）进行单样本自适应，显著提升其在分布偏移下的泛化能力。 领域现状：视觉上下文学习（VICL）是计算机视…
tags:
  - "CVPR 2025"
  - "LLM 其他"
  - "上下文学习"
  - "测试时训练"
  - "循环一致性"
  - "分布偏移"
  - "Painter"
---

# Test-Time Visual In-Context Tuning

**会议**: CVPR 2025  
**arXiv**: [2503.21777](https://arxiv.org/abs/2503.21777)  
**代码**: [https://github.com/Jiahao000/VICT](https://github.com/Jiahao000/VICT)  
**领域**: LLM/NLP  
**关键词**: 上下文学习, 测试时训练, 循环一致性, 分布偏移, Painter

## 一句话总结
本文提出VICT（Visual In-Context Tuning），通过翻转任务提示和测试样本的角色并利用循环一致性损失，在测试时对视觉上下文学习模型（如Painter）进行单样本自适应，显著提升其在分布偏移下的泛化能力。

## 研究背景与动机

**领域现状**：视觉上下文学习（VICL）是计算机视觉的新范式，通过将输入-输出示例对和测试图像拼成网格，将各种视觉任务转化为图像修补（inpainting）问题。代表性方法Painter在多种任务上展示了少样本适应能力。

**现有痛点**：VICL模型在部署时被冻结，但测试分布通常与训练分布不同（如图像损坏）。实验发现Painter在分布偏移下泛化能力很差。更令人意外的是，即使提供来自测试分布的任务提示（one-shot设置），性能反而更差——这说明当前VICL模型的泛化能力严重不足。

**核心矛盾**：VICL模型需要在推理时适应新分布，但现有模型是冻结的，无法利用测试样本中蕴含的分布信息。而传统的测试时训练（TTT）方法需要依赖特定的自监督任务（如旋转预测、MAE重建），这些在VICL的多任务上下文中无法通用。

**本文目标**：设计一种任务无关的测试时训练方法，让VICL模型能在推理时利用单个测试样本自适应到新分布。

**切入角度**：VICL模型天然具备"给定提示预测输出"的能力。如果模型理解了测试分布，那它应该能从自己的预测出发，反向恢复原始的任务提示输出。

**核心 idea**：将预测的测试输出作为新的"提示"反馈给模型，要求模型恢复原始任务提示输出，形成循环一致性。这个信号天然存在于VICL框架中，不需要额外数据或标注，可用于任意任务。

## 方法详解

### 整体框架
给定任务提示对 $(x,y)$ 和测试输入 $x_t$，先构造网格 $I=(x,y,x_t,\varnothing)$ 让模型预测 $\hat{y}_t$。然后翻转角色，构造 $I'=(x,\varnothing,x_t,\hat{y}_t)$，让模型预测 $\hat{y}$。用 $\hat{y}$ 和真实 $y$ 之间的回归损失优化模型权重。每个测试样本独立优化（从预训练权重重新开始）。

### 关键设计

1. **循环一致性自监督信号**:

    - 功能：为测试时训练提供任务无关的监督信号
    - 核心思路：前向传播预测测试输出 $\hat{y}_t = f_\theta(x,y,x_t,\varnothing)$，然后角色翻转重建任务提示输出 $\hat{y} = f_\theta(x,\varnothing,x_t,\hat{y}_t)$，最小化 smooth-$\ell_1$ 损失 $\mathcal{L}(\hat{y}, y)$。关键洞察：如果模型真正适应了测试分布，它应能通过自己的预测恢复已知的任务提示
    - 设计动机：传统TTT方法（旋转预测、MAE）只适用于特定场景，而VICL的网格结构天然允许角色翻转，生成免费的监督信号

2. **权重重置策略**:

    - 功能：确保每个测试样本独立适应
    - 核心思路：每处理一个新测试输入就重置模型权重回预训练状态 $\theta_0$，避免对先前测试样本的过拟合累积
    - 设计动机：不假设测试样本来自同一分布，最大化灵活性

3. **零样本和单样本双模式**:

    - 功能：覆盖两种实际场景
    - 核心思路：零样本（任务提示来自训练分布/干净图像）和单样本（任务提示来自测试分布/损坏图像）两种设定。VICT在两种设定下都显著提升性能
    - 设计动机：实际部署中可能有也可能没有来自目标分布的标注示例

### 损失函数 / 训练策略
使用smooth-L1损失优化，遵循Painter的设计。每个测试样本优化少量步数后推理。模型基于Painter，使用其预训练权重作为起点。

## 实验关键数据

### 主实验

| 任务/数据集 | Painter (zero-shot) | VICT (zero-shot) | Painter (one-shot) | VICT (one-shot) |
|---|---|---|---|---|
| 深度估计 NYUv2-C (A.Rel↓) | 0.392 | 0.365 | 0.537 | 显著改善 |
| 语义分割 ADE20K-C | 性能下降 | 显著恢复 | 更差 | 显著改善 |
| 全景分割 COCO-C | 性能下降 | 显著恢复 | 更差 | 显著改善 |
| 图像去噪 SIDD-C | 性能下降 | 显著恢复 | 更差 | 显著改善 |
| 图像去雨 | 性能下降 | 显著恢复 | 更差 | 显著改善 |
| 低光增强 LoL-C | 性能下降 | 显著恢复 | 更差 | 显著改善 |

### 消融实验

| 配置 | 效果 | 说明 |
|---|---|---|
| 无TTT（纯Painter） | 基线 | 分布偏移下性能差 |
| one-shot提示（无TTT） | 比零样本更差 | VICL对新分布泛化差 |
| VICT零样本 | 显著提升 | 循环一致性有效 |
| VICT单样本 | 进一步提升 | 匹配分布的提示+TTT最优 |

### 关键发现
- Painter在15种图像损坏下性能严重下降，尤其是高斯噪声、脉冲噪声等
- 提供损坏域的任务提示（one-shot）反而比干净提示（zero-shot）更差，暴露了VICL的泛化缺陷
- VICT的零样本或单样本模式甚至能超越用更多few-shot损坏样本训练的Painter
- 该方法可推广到测试时处理未见过的任务（如从深度估计迁移到法线估计）

## 亮点与洞察
- **循环一致性的优雅应用**：利用VICL的网格结构天然支持角色翻转这一特性，零成本构造自监督信号。这个insight极为精妙——不需要任何外部预文本任务
- **揭示VICL的泛化缺陷**：首次系统评估VICL在分布偏移下的表现，发现"one-shot比zero-shot更差"的反直觉现象，对VICL社区有重要警示
- **通用性强**：6种视觉任务（从高层语义理解到底层图像处理），15种corruption，方法完全任务无关

## 局限与展望
- 每个测试样本都需要多步优化，推理成本显著增加
- 仅在Painter上验证，对更新的VICL模型（如LVM）的效果有待检验
- 循环一致性假设模型的前向预测足够合理才能提供有用的优化信号——严重损坏场景下此假设可能不成立
- 可探索批量级别的TTT策略以提高推理效率

## 相关工作与启发
- **vs TTT-MAE**: TTT-MAE使用MAE重建作为自监督信号，仅适用于分类任务；VICT利用循环一致性适用于任何密集视觉任务
- **vs Painter**: VICT在Painter基础上通过测试时调优显著提升泛化能力，且不需要额外训练数据
- 循环一致性思路可迁移到其他图像修补式框架（如语言指导的图像编辑）

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次将测试时训练引入VICL，循环一致性的insight非常优雅
- 实验充分度: ⭐⭐⭐⭐⭐ 6任务×15种corruption，零样本和单样本双模式
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图示直观
- 价值: ⭐⭐⭐⭐ 对VICL鲁棒性研究有重要推动，方法通用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] BEST-Route: Adaptive LLM Routing with Test-Time Optimal Compute](../../ICML2025/llm_nlp/best-route_adaptive_llm_routing_with_test-time_optimal_compute.md)
- [\[ACL 2025\] TestNUC: Enhancing Test-Time Computing Approaches and Scaling through Neighboring Unlabeled Data Consistency](../../ACL2025/llm_nlp/testnuc_enhancing_test-time_computing_approaches_and_scaling_through_neighboring.md)
- [\[ICLR 2026\] Best-of-∞: Asymptotic Performance of Test-Time LLM Ensembling](../../ICLR2026/llm_nlp/best-of-infinity_asymptotic_performance_of_test-time_llm_ensembling.md)
- [\[ACL 2025\] ChartLens: Fine-Grained Visual Attribution in Charts](../../ACL2025/llm_nlp/chartlens_fine-grained_visual_attribution_in_charts.md)
- [\[ACL 2025\] Exploring Explanations Improves the Robustness of In-Context Learning](../../ACL2025/llm_nlp/exploring_explanations_improves_the_robustness_of_in-context_learning.md)

</div>

<!-- RELATED:END -->
