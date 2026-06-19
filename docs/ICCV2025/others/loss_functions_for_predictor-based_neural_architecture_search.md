---
title: >-
  [论文解读] Loss Functions for Predictor-based Neural Architecture Search
description: >-
  [ICCV 2025][神经架构搜索] 首次对性能预测器中8种损失函数进行全面系统性研究，涵盖回归、排序和加权三大类，在5个搜索空间的13个任务上揭示了各类损失函数的特性与互补性，并提出分段损失（PW loss）组合方法PWLNAS，在多个基准上超越现有SOTA。 性能预测器是NAS中广泛使用的评估加速方法…
tags:
  - "ICCV 2025"
  - "神经架构搜索"
  - "性能预测器"
  - "损失函数"
  - "排序学习"
  - "加权损失"
---

# Loss Functions for Predictor-based Neural Architecture Search

**会议**: ICCV 2025  
**arXiv**: [2506.05869](https://arxiv.org/abs/2506.05869)  
**代码**: 无  
**领域**: 神经架构搜索  
**关键词**: 神经架构搜索, 性能预测器, 损失函数, 排序学习, 加权损失

## 一句话总结

首次对性能预测器中8种损失函数进行全面系统性研究，涵盖回归、排序和加权三大类，在5个搜索空间的13个任务上揭示了各类损失函数的特性与互补性，并提出分段损失（PW loss）组合方法PWLNAS，在多个基准上超越现有SOTA。

## 研究背景与动机

性能预测器是NAS中广泛使用的评估加速方法，其有效性受损失函数选择的关键影响：

**损失函数选择未被充分研究**：虽然已有MSE、排序损失等多种选择，但不同损失函数在不同搜索空间、不同训练数据量下的表现差异缺乏系统性研究

**单一损失函数各有局限**：回归损失擅长预测绝对精度但排序能力弱；排序损失擅长整体排序但难识别顶尖架构；加权损失关注高性能架构但小数据量下易过拟合

**实践指导缺失**：研究者在具体任务中如何选择合适的损失函数缺乏参考依据

## 方法详解

### 整体框架

本文为综合实验研究+新方法提出，核心工作包括：
1. 将损失函数分为回归、成对排序、列表排序、加权四大类共8种
2. 在5个搜索空间13个任务上用多种指标系统评估
3. 提出PWLNAS：基于发现的互补性设计分段损失函数

### 关键设计

1. **损失函数分类与选取**：

    - **回归损失**：MSE，最小化预测分数与真实精度的差异
    - **成对排序损失**：Hinge Ranking (HR)、Logistic Ranking (LR)、MSE+Sequence Ranking (MSE+SR)，关注架构对的相对排序
    - **列表排序损失**：ListMLE，优化预测排序列表与真实排序的一致性
    - **加权损失**：Exponential Weighted (EW)、MAPE、Weighted Approximate-Rank Pairwise (WARP)，高性能架构获得更高权重
    - 设计动机：覆盖预测器损失函数的主流范式，WARP为首次引入NAS领域

2. **多维度评估指标体系**：

    - **Kendall's Tau (τ)**：整体排序相关性
    - **Precision@T**：预测top-T%中实际属于top-T%的比例（越高越好）
    - **N@K**：预测top-K架构中实际最好的排名（越低越好）
    - 设计动机：NAS的核心目标是找到最好的架构，top-K指标比整体排序指标更重要

3. **分段损失PWLNAS（Piecewise Loss NAS）**：

    - 在预测器迭代训练的早期使用排序/回归损失做热身
    - 后期切换到加权损失以识别高性能架构
    - 具体组合依任务而定：NAS-Bench-201用HR→MAPE，NAS-Bench-101用ListMLE→WARP，DARTS用HR→MAPE
    - 设计动机：利用发现的互补性——排序损失在小数据量下更好，加权损失在充足数据下更好

### 损失函数 / 训练策略

所有实验使用统一的GCN基性能预测器。各损失函数使用不同级别的超参数进行公平调优。训练数据来自搜索空间中的随机采样子集，预测器在整个搜索空间上评估。所有结果取30次运行的平均值。

## 实验关键数据

### 主实验 - 各基准搜索结果（表格）

| 方法 | 损失 | NAS-Bench-201 C10 | NAS-Bench-201 C100 | NAS-Bench-201 IN-16 |
|------|------|-------------------|--------------------|--------------------|
| NASBOT | MSE | 6.36 | 28.62 | 54.12 |
| ReNAS | LR | 6.01 | 27.88 | 54.03 |
| NPENAS | MSE | 5.69 | 26.54 | 53.52 |
| **PWLNAS** | **PW** | **5.63** | **26.51** | **52.88** |
| Global Best | - | 5.63 | 26.49 | 52.69 |

| 方法 | 损失 | NAS-Bench-101 Test Err. |
|------|------|------------------------|
| BANANAS | MAPE | 5.92 |
| FlowerFormer | HR | 5.86 |
| NPENAS | MSE | 5.85 |
| PWLNAS-HR | HR | 5.83 |
| **PWLNAS-PW** | **PW** | **5.80** |

| 方法 | 损失 | DARTS Test Err. | Params |
|------|------|----------------|--------|
| GMAENAS | BPR | 2.50±0.03 | 3.6M |
| DCLP | ListMLE | 2.48±0.02 | 3.3M |
| **PWLNAS** | **PW** | **2.47±0.05** | **3.6M** |

### 消融实验 - 损失函数在不同条件下的表现（表格）

| 预测器骨架 | 损失 | N@10↓ | Ptop@0.5↑ | τ↑ |
|------------|------|-------|-----------|-----|
| AP (MLP) | MSE | 250.94 | 4.41 | 0.43 |
| AP (MLP) | HR | 23.58 | 22.15 | 0.65 |
| AP (MLP) | **ListMLE** | **22.74** | **24.15** | **0.66** |
| AP (MLP) | WARP | 113.20 | 9.36 | 0.43 |
| PINAT (Trans.) | MSE | 146.60 | 8.62 | 0.62 |
| PINAT (Trans.) | HR | 8.44 | 29.32 | 0.67 |
| PINAT (Trans.) | **WARP** | **3.78** | **38.71** | 0.65 |

### 关键发现

- **加权损失在充足数据下最佳**：WARP、MAPE在top-K指标（Precision@0.5、N@10）上全面领先
- **排序损失在极小数据下更优**：训练数据极少时，HR等排序损失优于加权损失，因后者会过度强调局部好架构
- **MSE在识别好架构上最差**：在大多数搜索空间上Precision@0.5最低
- **混合损失MSE+SR不如单一损失**：同时优化两种目标反而产生折中效果
- **简单骨架配排序损失，复杂骨架配加权损失**：MLP-based预测器用ListMLE最好，Transformer-based用WARP最好
- **GT权重优于排名权重**：在加权损失中使用架构的真实精度作为权重比使用排名更有效
- **更多训练数据不一定提升top-K能力**：回归和排序损失的top-K指标可能随数据增加反而下降
- **PW分段损失全面胜出**：在NAS-Bench-201、101、DARTS上均取得最低错误率

## 亮点与洞察

1. **首次系统研究**：对NAS性能预测器的损失函数进行了最全面的实证研究，具有重要参考价值
2. **发现了有价值的规律**：关于加权/排序损失的互补性、数据量与损失类型的交互效应等规律具有实践指导意义
3. **WARP的引入**：将多标签图像标注领域的WARP损失引入NAS，效果显著
4. **简洁有效的PW方法**：分段切换损失函数即可超越SOTA，实现简单但效果一致

## 局限与展望

- PW损失的切换阈值是固定的，依赖人工经验选择
- 未探索更灵活的损失组合方式（如逐步增加加权强度而非硬切换）
- 未考虑直接优化top-K指标的损失函数设计
- 训练数据的采样策略也很重要但未深入研究（随机采样可能不代表搜索空间分布）
- 在更大规模搜索空间（如超大型NAS-Bench或真实开放空间）的验证缺失
- 可探索根据搜索进度自适应选择损失函数的机制

## 相关工作与启发

- 与GATES（HR）、DCLP（ListMLE）、NAR-Former（MSE+SR）等工作形成系统对比
- 发现的规律可直接指导现有预测器方法的改进（只需更换损失函数即可提升）
- 损失函数的互补性观察可启发其他排序学习场景（如推荐系统、信息检索）
- 从信息论角度深入理解为何不同损失在不同数据量下表现截然不同是有价值的后续方向

## 评分

- **新颖性**: ⭐⭐⭐ 损失函数各自已有，核心贡献在于系统研究和PW组合，创新适中
- **实验充分度**: ⭐⭐⭐⭐⭐ 5个搜索空间、13个任务、多种预测器、多种训练数据量、30次平均，极其充分
- **写作质量**: ⭐⭐⭐⭐ 组织清晰有条理，发现总结准确，但图表密集
- **价值**: ⭐⭐⭐⭐ 为NAS社区提供了实用的损失函数选择指南，PW方法简简单但有效

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] HyperNAS: Enhancing Architecture Representation for NAS Predictor via Hypernetwork](../../CVPR2026/others/hypernas_enhancing_architecture_representation_for_nas_predictor_via_hypernetwor.md)
- [\[CVPR 2025\] Subnet-Aware Dynamic Supernet Training for Neural Architecture Search](../../CVPR2025/others/subnet-aware_dynamic_supernet_training_for_neural_architecture_search.md)
- [\[CVPR 2026\] InTrain: Intrinsic Trainability for Zero-Cost Neural Architecture Search](../../CVPR2026/others/intrain_intrinsic_trainability_for_zero-cost_neural_architecture_search.md)
- [\[CVPR 2025\] VKDNW: Training-free Neural Architecture Search through Variance of Knowledge of Deep Network Weights](../../CVPR2025/others/training-free_neural_architecture_search_through_variance_of_knowledge_of_deep_n.md)
- [\[ICCV 2025\] Joint Asymmetric Loss for Learning with Noisy Labels](joint_asymmetric_loss_for_learning_with_noisy_labels.md)

</div>

<!-- RELATED:END -->
