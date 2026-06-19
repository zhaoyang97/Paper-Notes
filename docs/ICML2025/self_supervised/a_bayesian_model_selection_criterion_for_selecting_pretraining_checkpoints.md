---
title: >-
  [论文解读] A Bayesian Model Selection Criterion for Selecting Pretraining Checkpoints
description: >-
  [ICML 2025][自监督学习][贝叶斯模型选择] 引入"下游自由能"作为预训练检查点可适应性的贝叶斯模型选择准则，证明"预训练自由能"可作为其上界代理（无需下游数据），并实验验证大学习率/小 batch/高 momentum 通过降低预训练自由能改善下游迁移性能。 领域现状：基础模型的"预训练-微调"范式在 NLP 和…
tags:
  - "ICML 2025"
  - "自监督学习"
  - "贝叶斯模型选择"
  - "自由能"
  - "预训练检查点"
  - "迁移学习"
  - "损失景观"
  - "隐式正则化"
---

# A Bayesian Model Selection Criterion for Selecting Pretraining Checkpoints

**会议**: ICML 2025  
**arXiv**: [2410.05612](https://arxiv.org/abs/2410.05612)  
**代码**: 无  
**领域**: 迁移学习理论  
**关键词**: 贝叶斯模型选择, 自由能, 预训练检查点, 迁移学习, 损失景观, 隐式正则化

## 一句话总结

引入"下游自由能"作为预训练检查点可适应性的贝叶斯模型选择准则，证明"预训练自由能"可作为其上界代理（无需下游数据），并实验验证大学习率/小 batch/高 momentum 通过降低预训练自由能改善下游迁移性能。

## 研究背景与动机

**领域现状**：基础模型的"预训练-微调"范式在 NLP 和 CV 中占主导。预训练过程中会产生大量检查点，但选择哪个检查点进行微调一直缺乏理论指导——实践中主要靠经验启发式：用最终检查点、或认为大学习率更好。

**现有痛点**：(1) 缺乏原则性的检查点选择准则；(2) 已知经验规律（大学习率/小 batch 有利于迁移）缺乏统一理论解释；(3) 先前理论工作各有局限——Liu et al. 2023a 的 Hessian 迹缺乏形式化界，Galanti et al. 2022 的 neural collapse 缺乏实际正则化方法，且主要针对线性探针而非全参数微调。

**核心矛盾**：下游任务在预训练时未知——如何在不接触下游数据的情况下判断检查点的适应潜力？

**本文目标**：为检查点选择提供原则性理论框架，建立从"预训练特性"到"下游迁移性能"的可证明联系。

**切入角度**：利用贝叶斯统计中的边际似然（自由能）概念——限制在检查点邻域内——作为可适应性度量。

**核心 idea**：自由能低 = 检查点附近好参数浓度高 = 微调更容易成功。理论链条：下游测试误差 $\lesssim$ 下游自由能 $\lesssim$ 预训练自由能。

## 方法详解

### 整体框架

建立三层理论链条：

$$\text{下游贝叶斯测试误差} \lesssim \text{下游自由能} \lesssim \text{预训练自由能}$$

核心步骤：(1) 定义下游自由能作为检查点可适应性度量；(2) 定义预训练自由能作为仅用预训练数据可计算的代理；(3) 证明预训练自由能控制下游自由能（Proposition 5.1）；(4) 利用已知 SGD 隐式偏差验证。

### 关键设计

1. **下游自由能**：对检查点 $w^* = (v^*, \theta^*) \in U_0$，定义：

    $$\bar{F}^1(B_\gamma(w^*)) = -\log \int_{B_\gamma(w^*)} \exp\{-m K^1(w)\} \varphi(w) \, dw$$

    其中 $B_\gamma(w^*) = \{w = (v^*, \theta) : \|\theta - \theta^*\|^2 \leq 1/\gamma\}$ 是骨干参数邻域（冻结线性头），$K^1(w)$ 为下游测试损失。渐近展开 $\bar{F}^1 = mK^1(w^{*1}) + \lambda^1(w^*) \log m + O(\log\log m)$，其中 $\lambda^1$（local learning coefficient）衡量局部模型复杂度。高损失但低复杂度的检查点可能优于低损失但高复杂度——体现了贝叶斯的 Occam 剃刀。

2. **预训练自由能作为代理**：$F^0(B_\gamma(w^*); \beta) = -\log \int_{B_\gamma(w^*)} \exp\{-n\beta \hat{K}^0(w)\} \varphi(w) \, dw$，基于预训练训练损失 $\hat{K}^0$ 和逆温度 $\beta$。Proposition 5.1 在分布偏移条件下证明 $\bar{F}^1 \leq F^0 + \text{shift term}$——预训练自由能上界下游自由能。

3. **与 SGD 隐式正则化的联系**：Lau et al. (2025) 已证明大学习率、小 batch、高 momentum 隐式降低自由能中的 $\lambda^0$（local learning coefficient）。本文验证环路：降低预训练自由能的预训练超参数 → 更好的下游迁移性能。这提供了实际可操作的指导：调整预训练超参数即可间接优化检查点的可迁移性。

### 损失函数 / 训练策略

- 预训练和微调共享骨干 $\phi_\theta$，各有独立线性头 $v, u$
- 微调采用 limited fine-tuning（骨干更小学习率）
- 预训练/下游损失均为 KL 散度形式 $K^i(w) = \mathbb{E}_{r^i(x)} D_\text{KL}(r^i(y|x) \| p(y|x,w))$

## 实验关键数据

### 主实验：预训练自由能 vs 迁移准确率 ($R^2$)

| 超参数 | $R^2$ | 趋势 |
|-------|-------|------|
| 学习率 | 0.91 | 大lr → 低自由能 → 高迁移准确率 |
| Batch size | 0.87 | 小batch → 低自由能 → 高迁移准确率 |
| Momentum | 0.85 | 高momentum → 低自由能 → 高迁移准确率 |

### 与其他预训练指标的对比

| 预训练指标 | 与下游性能相关性 | 特点 |
|----------|---------------|------|
| 预训练损失 | 弱 | 低损失不等于好迁移 |
| Hessian 迹（平坦度） | 中等 | 相关但非因果 |
| Neural Collapse | 中等 | 缺乏显式正则化方法 |
| **预训练自由能** | **最强** | 同时捕获拟合和复杂度 |

### 消融实验

| 分析维度 | 结论 |
|---------|------|
| $mK^1(w^{*1})$（拟合） vs $\lambda^1 \log m$（复杂度） | 两者共同决定检查点质量，非单一因素 |
| 线性探针 vs 全参数微调 | 本文结果适用于全参数微调（先前工作主要考虑线性探针） |
| 多数据集验证 | CIFAR-10/100、多种架构（ResNet、ViT-Small）一致 |

### 关键发现

- 预训练自由能比所有其他候选指标与下游性能的相关性都强
- 统一解释了已知经验规律：大lr/小batch/高momentum → 降低自由能
- 自由能分解揭示检查点选择本质——不是追求最低损失，而是拟合与复杂度的平衡
- 可作为预训练过程中的在线监控指标

## 亮点与洞察

- 将经典贝叶斯框架首次引入迁移学习的检查点选择——旧理论的新应用
- 完整理论链条：预训练超参数 → 预训练自由能 → 下游自由能 → 下游性能
- 统一解释了 SGD 隐式正则化（平坦最小值）对迁移学习有益的深层原因
- 实际指导：预训练时监控自由能可在不知道下游任务的情况下选择更好的检查点

## 局限与展望

- 自由能的精确计算在高维中不可行——需要 MCMC 或变分推理近似，计算开销待评估
- Proposition 5.1 的分布偏移假设在现实中可能不严格成立
- 实验仅在小-中规模模型验证（ResNet、ViT-Small），未扩展到十亿参数级基础模型
- 自由能对参数缩放不具不变性——纯 ReLU 网络的参数缩放不改变输出但改变自由能，需要 batch norm/weight decay 打破此不变性

## 相关工作与启发

- Liu et al. (2023a)：Hessian 迹与迁移的经验关系，但缺乏形式化界
- Galanti et al. (2022)：neural collapse 链条，但缺乏显式正则化方法
- Munn et al. (2024)：geometric complexity→neural collapse，本文进一步推进
- Lau et al. (2025)：local learning coefficient 和 SGD 自由能正则化的理论基础

## 评分

⭐⭐⭐⭐ — 理论优美且有实际指导价值，将经典贝叶斯模型选择与迁移学习连接，统一解释多个已知经验规律。主要局限在实验规模偏小和自由能近似计算的实际可操作性。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Foundation Model Insights and a Multi-Model Approach for Superior Fine-Grained One-shot Subset Selection](foundation_model_insights_and_a_multi-model_approach_for_superior_fine-grained_o.md)
- [\[ICML 2025\] Griffin: Towards a Graph-Centric Relational Database Foundation Model](griffin_towards_a_graph-centric_relational_database_foundation_model.md)
- [\[ICML 2025\] What Has a Foundation Model Found? Using Inductive Bias to Probe for World Models](what_has_a_foundation_model_found_using_inductive_bias_to_probe_for_world_models.md)
- [\[CVPR 2025\] MAP: Unleashing Hybrid Mamba-Transformer Vision Backbone's Potential with Masked Autoregressive Pretraining](../../CVPR2025/self_supervised/map_unleashing_hybrid_mamba-transformer_vision_backbones_potential_with_masked_a.md)
- [\[ICLR 2026\] PonderLM: Pretraining Language Models to Ponder in Continuous Space](../../ICLR2026/self_supervised/ponderlm_pretraining_language_models_to_ponder_in_continuous_space.md)

</div>

<!-- RELATED:END -->
