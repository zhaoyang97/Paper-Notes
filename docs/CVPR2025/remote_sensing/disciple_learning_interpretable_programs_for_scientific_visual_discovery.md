---
title: >-
  [论文解读] DiSciPLE: Learning Interpretable Programs for Scientific Visual Discovery
description: >-
  [CVPR 2025][遥感][可解释程序合成] 提出 DiSciPLE 框架，利用 LLM 引导的进化算法自动合成可解释的 Python 程序来分析视觉数据，在人口密度估计等科学任务上以比最近基线低 35% 的误差实现了 SOTA，且程序完全可解释。 领域现状：科学领域（遥感、生态学、气候科学等）大量使用视觉数据做预测（如…
tags:
  - "CVPR 2025"
  - "遥感"
  - "可解释程序合成"
  - "进化算法"
  - "LLM引导搜索"
  - "科学发现"
  - "遥感分析"
---

# DiSciPLE: Learning Interpretable Programs for Scientific Visual Discovery

**会议**: CVPR 2025  
**arXiv**: [2502.10060](https://arxiv.org/abs/2502.10060)  
**代码**: [https://disciple.cs.columbia.edu](https://disciple.cs.columbia.edu)  
**领域**: 遥感 / 可解释AI  
**关键词**: 可解释程序合成, 进化算法, LLM引导搜索, 科学发现, 遥感分析

## 一句话总结
提出 DiSciPLE 框架，利用 LLM 引导的进化算法自动合成可解释的 Python 程序来分析视觉数据，在人口密度估计等科学任务上以比最近基线低 35% 的误差实现了 SOTA，且程序完全可解释。

## 研究背景与动机

**领域现状**：科学领域（遥感、生态学、气候科学等）大量使用视觉数据做预测（如从卫星图估计人口密度），但科学家需要的不仅是准确预测，更需要理解预测背后的机制。概念瓶颈 (Concept Bottleneck) 等可解释模型虽然可读，但表达能力局限于简单的词袋模型。

**现有痛点**：ViperGPT 等 LLM 代码生成方法在标准视觉任务上有效，但在科学领域的新颖任务上失败——因为 LLM 缺乏特定领域知识。深度学习黑箱模型虽然准确但不可解释。符号回归方法无法处理高维视觉输入。

**核心矛盾**：可解释性与表达能力的trade-off——简单可解释模型（线性分类器）不够准确，准确的深度模型不可解释。

**本文目标** 如何自动发现既准确又可解释的程序来处理科学视觉数据。

**切入角度**：将程序搜索重构为进化算法问题，但用 LLM 替代传统的随机交叉和变异操作，利用 LLM 的编程能力和常识知识来更智能地搜索程序空间。

**核心 idea**：用 LLM 驱动的进化算法搜索可解释的 Python 程序，程序交叉神经网络基元（如开放词汇分割模型）与数学/逻辑操作。

## 方法详解

### 整体框架
输入：数据集 $\mathcal{D}$、评估指标 $\mathcal{M}$、原语函数库 $\mathcal{F}$（含开放词汇分割模型、数学/逻辑/图像操作）、任务文本描述。输出：可解释的 Python 程序 $P: X \to Y$。流程：LLM 零样本生成初始程序群体 → 基于适应度选择父代 → LLM 执行交叉/变异 → 程序评审(critic) → 程序简化(simplifier) → 迭代进化。

### 关键设计

1. **LLM 驱动的进化搜索**:

    - 功能：智能遍历程序空间
    - 核心思路：初始群体由 LLM 根据任务描述零样本生成（如"给定卫星图像，写一个函数估计人口密度"），保证起点不完全随机。交叉操作将两个父代程序和它们的得分一起交给 LLM，让其结合两者的优点生成新程序。变异操作以一定概率对程序做随机修改。LLM 的常识知识使交叉和变异比传统符号方法更有效
    - 设计动机：传统进化算法的随机搜索在高维程序空间中效率极低，LLM 的编程能力和常识可以大幅加速收敛

2. **程序评审 (Program Critic)**:

    - 功能：提供细粒度的分层评估反馈
    - 核心思路：不仅评估程序的整体得分，而是按数据分区（如不同土地利用类型）分层评估。将表现差的子集反馈给 LLM，引导其针对性改进。例如发现程序在城市区域表现好但农村区域差，就告知 LLM 重点改进农村区域的逻辑
    - 设计动机：整体得分无法告知 LLM 具体应该改进程序的哪个部分

3. **程序简化 (Program Simplifier)**:

    - 功能：保持可解释性，去除冗余
    - 核心思路：通过 AST（抽象语法树）分析去除死代码和冗余操作，通过回归权重阈值化去除对输出贡献微小的特征。确保进化过程中程序不会因不断叠加而变得臃肿不可读
    - 设计动机：进化过程会不断累积代码片段，不简化会导致程序越来越长越难理解

### 损失函数 / 训练策略
适应度函数使用任务特定指标（如人口密度用 L2-Log 误差）。群体大小 M，进化 T 代，每代对每个子代做交叉（必做）+ 变异（概率 $\rho_m$）+ 评审 + 简化。

## 实验关键数据

### 主实验

**人口密度估计（卫星图像）**:

| 方法 | L2-Log 误差↓ | 可解释 |
|------|------------|--------|
| 深度学习基线 | 0.3974 | ✗ |
| Concept Bottleneck | ~0.5 | ✓（有限） |
| LLM 零样本生成 | 0.84 | ✓ |
| **DiSciPLE** | **0.2607** | **✓** |

**34 项人口统计指标**:

| 方法 | L1 误差↓ |
|------|---------|
| 深度学习/均值基线 | 0.8527 |
| **DiSciPLE** | **0.8159** |

### 消融实验

| 配置 | L2-Log 误差↓ |
|------|------------|
| 基础版本（无 critic/simplifier） | 0.3159 |
| + 特征集扩展 | 0.2906 |
| + 程序评审 (critic) | 0.2873 |
| + 程序简化 (simplifier) | **0.2607** |
| 去除 LLM 常识 | 0.8401 |
| 去除问题上下文 | 0.4498 |

### 关键发现
- LLM 常识知识极其关键：去除后误差从 0.2607 暴增到 0.8401，说明 LLM 提供的领域直觉是搜索成功的基础
- 程序简化的贡献被低估：它不仅维护可解释性，还通过去除噪声特征实际提升了泛化能力
- DiSciPLE 在 OOD（分布外）泛化上显著优于深度模型，说明可解释程序学到了更鲁棒的规律
- 34 项人口统计指标实验表明方法具有广泛适用性，不局限于单一任务

## 亮点与洞察
- **LLM 作为智能搜索引擎**：用 LLM 替代传统进化算法的随机操作是一个非常优雅的设计——LLM 的编程能力和常识使得搜索效率远高于随机搜索
- **可解释性不牺牲准确性**：DiSciPLE 的程序比深度模型更准确（低 35% 误差），这颠覆了"可解释性必然损害性能"的成见
- **科学发现的民主化**：框架允许非 ML 专家通过提供任务描述和数据来自动发现视觉规律，降低了科学发现的技术门槛

## 局限与展望
- 依赖 LLM 的代码生成能力，程序质量受 LLM 水平限制
- 原语函数库需要人工设计，不同领域需要不同的原语集
- 进化搜索计算成本高（多次 LLM 调用 + 程序评估）
- 仅在遥感/人口统计领域验证，其他科学领域的适用性未知

## 相关工作与启发
- **vs ViperGPT/VisProg**: 这些方法直接用 LLM 零样本生成代码，在科学任务上效果很差（误差 0.84）。DiSciPLE 通过进化搜索迭代改进
- **vs Concept Bottleneck**: 概念瓶颈模型局限于简单线性函数，DiSciPLE 生成的程序可以包含复杂逻辑和嵌套调用
- **vs 符号回归**: 传统 SR 方法无法处理高维图像输入，DiSciPLE 通过引入视觉基础模型作为原语解决了这个问题
- 对科学发现自动化、AI for Science 有重要参考价值

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ LLM 引导进化搜索合成可解释程序是全新范式
- 实验充分度: ⭐⭐⭐⭐ 三个真实科学任务验证，消融详尽
- 写作质量: ⭐⭐⭐⭐ 动机强、问题清晰
- 价值: ⭐⭐⭐⭐⭐ 为 AI for Science 的可解释性开辟新方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] GeoViS: Geospatially Rewarded Visual Search for Remote Sensing Visual Grounding](../../CVPR2026/remote_sensing/geovis_geospatially_rewarded_visual_search_for_remote_sensing_visual_grounding.md)
- [\[CVPR 2025\] Meta-Learning Hyperparameters for Parameter Efficient Fine-Tuning](meta-learning_hyperparameters_for_parameter_efficient_fine-tuning.md)
- [\[ICML 2026\] The Perception-Physics Paradox: Probing Scientific Alignment with TC-Bench](../../ICML2026/remote_sensing/the_perception-physics_paradox_probing_scientific_alignment_with_tc-bench.md)
- [\[CVPR 2025\] Learning Occlusion-Robust Vision Transformers for Real-Time UAV Tracking](learning_occlusion-robust_vision_transformers_for_real-time_uav_tracking.md)
- [\[CVPR 2025\] Hierarchical Dual-Change Collaborative Learning for UAV Scene Change Captioning](hierarchical_dual-change_collaborative_learning_for_uav_scene_change_captioning.md)

</div>

<!-- RELATED:END -->
