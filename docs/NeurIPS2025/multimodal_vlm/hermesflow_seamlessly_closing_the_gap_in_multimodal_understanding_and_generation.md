---
title: >-
  [论文解读] HermesFlow: Seamlessly Closing the Gap in Multimodal Understanding and Generation
description: >-
  [NeurIPS 2025][多模态VLM][Pair-DPO] 首次揭示统一多模态大模型中理解能力普遍强于生成能力的现象，提出 HermesFlow 框架，通过同源偏好数据构建配对理解-生成偏好对，利用 Pair-DPO 和自博弈迭代优化，在不引入外部高质量数据的情况下同步提升理解与生成能力并缩小两者差距。
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "Pair-DPO"
  - "多模态"
  - "understanding-generation gap"
  - "self-play optimization"
  - "homologous preference data"
---

# HermesFlow: Seamlessly Closing the Gap in Multimodal Understanding and Generation

**会议**: NeurIPS 2025  
**arXiv**: [2502.12148](https://arxiv.org/abs/2502.12148)  
**代码**: [https://github.com/Gen-Verse/HermesFlow](https://github.com/Gen-Verse/HermesFlow)  
**领域**: 多模态VLM / 统一理解与生成 / 偏好对齐  
**关键词**: Pair-DPO, multimodal alignment, understanding-generation gap, self-play optimization, homologous preference data  

## 一句话总结
首次揭示统一多模态大模型中理解能力普遍强于生成能力的现象，提出 HermesFlow 框架，通过同源偏好数据构建配对理解-生成偏好对，利用 Pair-DPO 和自博弈迭代优化，在不引入外部高质量数据的情况下同步提升理解与生成能力并缩小两者差距。

## 研究背景与动机

**领域现状**：以 Show-o、Transfusion、Emu3 为代表的统一 MLLM 已经能用单个 Transformer 同时处理多模态理解和图像生成任务，展现出令人印象深刻的性能。

**现有痛点**：已有工作如 Liquid 和 MetaMorph 从数据层面研究理解与生成之间的协同关系，但忽视了一个关键现象——**理解能力系统性地强于生成能力**，两者之间存在显著的 gap。

**核心矛盾**：单纯增加理解或生成的训练数据不能按比例改善两方面性能，预训练后理解-生成能力不平衡的问题持续存在。现有 DPO 方法要么只优化理解（如 CSR），要么只优化生成（如 Emu3），无法联合改善。

**本文目标** 如何在不依赖外部高质量数据的情况下，同时提升理解和生成能力，并缩小两者之间的差距？

**切入角度**：利用 MLLM 自身——理解能力强于生成的特性，用更强的理解能力去评判生成质量，从同源输入数据中构建理解和生成的配对偏好数据。

**核心 idea**：Pair-DPO + 自博弈迭代优化，从同源数据出发构建配对偏好对，同时优化理解和生成。

## 方法详解

### 整体框架
输入：一组同源的 (image $x$, caption/prompt $y$) 数据对。流程分三步：(1) 构建同源偏好数据；(2) Pair-DPO 联合优化；(3) 自博弈迭代。输出：理解和生成能力同步提升、差距缩小的 MLLM。

### 关键设计

1. **理解偏好数据构建**：

    - 功能：为每张图像生成 $n$ 条不同 caption，通过 BERT 相似度打分筛选 winning/losing 样本
    - 核心思路：给定图像 $x$，MLLM 生成 $n$ 条 caption，计算每条与 ground-truth prompt $y$ 之间的 BERT 相似度 $s(y_k, y)$，最高分为 $y_w$，最低分为 $y_l$
    - 设计动机：captioning 任务能全面反映 MLLM 捕捉视觉特征的能力（包括物体属性、空间关系、细节），BERT 相似度提供了一个自动化的质量度量

2. **生成偏好数据构建**：

    - 功能：为每个 prompt 随机生成 $n$ 张图像，利用自我评判（self-VQA）筛选最优/最差
    - 核心思路：使用 TIFA 为 prompt $y$ 生成 $q$ 个 VQA 对 $\{(Q_i, A_i)\}$，对每张生成图像计算 VQA 准确率 $Acc(x_j) = \frac{1}{q}\sum_{i=1}^{q}\mathbb{I}(R_{j,i}=A_i)$，最高且>0.6 的为 $x_w$，最低的为 $x_l$
    - 设计动机：巧妙利用"理解强于生成"的特性——用 MLLM 自身更强的理解能力来评判生成质量，实现 self-critic

3. **Pair-DPO 联合优化**：

    - 功能：将理解和生成偏好数据配对，联合优化
    - 核心思路：损失函数为 $\mathcal{L}_{\text{Pair-DPO}}(\theta) = -\mathbb{E}[\log\sigma(\Delta_{Und}\cdot\Delta_{Gen})]$，其中 $\Delta_{Und}$ 和 $\Delta_{Gen}$ 分别是理解和生成的偏好差分。关键创新在于将两个模态的偏好差异**相乘**后取 sigmoid，使得两者同时满足时梯度最大
    - 设计动机：与分别做 DPO 不同，乘法耦合确保优化方向在同一语义空间中协调，不会顾此失彼

4. **自博弈迭代优化**：

    - 功能：多轮优化，每轮用优化后的模型重新生成候选，动态更新偏好对
    - 核心思路：第 $i$ 轮中，若新生成的最优 caption 超过上一轮 winning sample（$s(y_{\max}^i, y) > s(y_w^{i-1}, y)$），则用其替代 winning sample 并以旧 winning 作为 losing；否则以新最优替代旧 losing sample，提供更平滑的学习梯度
    - 设计动机：自适应的"提高标准"或"降低难度"策略，避免模型陷入停滞

### 损失函数 / 训练策略
- 使用 AdamW 优化器，学习率 2e-5，cosine schedule
- $\beta = 0.2$，batch size 4，训练 3000 步
- 基于 Show-o (1.3B)，同源数据来自 JourneyDB 的 5000 对
- 8×A100 GPUs

## 实验关键数据

### 主实验 — 理解

| 模型 | 参数量 | POPE↑ | MME↑ | Flickr30k↑ | VQAv2↑ | GQA↑ | MMMU↑ |
|------|--------|-------|------|------------|--------|------|-------|
| SEED-X | 17B | 84.2 | 1435.7 | 52.3 | - | 47.9 | 35.6 |
| Chameleon | 34B | - | - | 74.7 | 66.0 | - | - |
| Show-o | 1.3B | 80.0 | 1232.9 | 67.6 | 74.7 | 61.0 | 27.4 |
| **HermesFlow** | **1.3B** | **81.4** | **1249.7** | **69.2** | **75.3** | **61.7** | **28.3** |

### 主实验 — 生成 (GenEval)

| 方法 | 参数量 | Single Obj. | Two Obj. | Counting | Colors | Position | Overall |
|------|--------|-------------|----------|----------|--------|----------|---------|
| SD 2.1 | 865M | 0.97 | 0.50 | 0.46 | 0.80 | 0.07 | 0.49 |
| Show-o | 1.3B | 0.95 | 0.52 | 0.49 | 0.82 | 0.11 | 0.53 |
| Janus | 1.3B | 0.97 | 0.68 | 0.30 | 0.84 | 0.46 | 0.61 |
| **HermesFlow** | **1.3B** | **0.98** | **0.84** | **0.66** | **0.82** | **0.32** | **0.69** |

### 消融实验 — Pair-DPO vs DPO & 迭代次数

| 方法 | POPE↑ | MME↑ | MMMU↑ | GenEval Overall↑ | DPG-Bench↑ |
|------|-------|------|-------|------------------|------------|
| Show-o (baseline) | 80.0 | 1232.9 | 27.4 | 0.53 | 67.48 |
| DPO (仅理解) | 80.8 | 1242.2 | 27.8 | 0.58 | 67.88 |
| DPO (仅生成) | 80.5 | 1239.3 | 27.5 | 0.70 | 70.03 |
| Pair-DPO Iter.1 | 81.1 | 1246.7 | 28.0 | 0.68 | 70.19 |
| Pair-DPO Iter.3 | **81.4** | **1249.7** | **28.3** | **0.69** | **70.22** |

### 理解-生成差距量化

| 方法 | Understanding Score↑ | Generation Score↑ | Gap↓ |
|------|---------------------|-------------------|------|
| VILA-U (7B) | 0.646 | 0.477 | 0.169 |
| Janus (1.3B) | 0.599 | 0.417 | 0.182 |
| Show-o (1.3B) | 0.520 | 0.433 | 0.087 |
| **HermesFlow (1.3B)** | **0.533** | **0.497** | **0.036** |

### 关键发现
- Pair-DPO 单轮就能同时大幅提升理解和生成，效果优于分别做理解 DPO + 生成 DPO
- 第一轮迭代优化贡献最大，超过 2 轮后生成能力基本收敛，理解仍可微小提升
- 理解-生成 Gap 从 Show-o 的 0.087 缩小到 0.036（减少 59%）
- 生成偏好数据对采样数 $n$ 更敏感，$n$ 太小时噪声大、生成掉点严重

## 亮点与洞察
- **self-critic 机制**非常巧妙：利用"理解>生成"的不对称性，让模型用更强的理解能力评判自己的生成，不需要任何外部评价器
- **Pair-DPO 的乘法耦合**设计：通过 $\Delta_{Und} \cdot \Delta_{Gen}$ 的乘积形式，使得理解和生成在同一优化步内联动，而非独立优化
- **自博弈中的自适应标准调整**：当模型进步时提高标准（用新最优替代旧 winning），否则降低难度（用新最优替代旧 losing），保证持续有效学习
- **可迁移性**：Pair-DPO 框架理论上适用于任何统一理解-生成 MLLM（Janus、VILA-U 等）

## 局限与展望
- 仅在 Show-o (1.3B) 上验证，未扩展到更大模型或更多 backbone
- 同源数据量仅 5000 对，规模化后效果存疑
- 生成评估依赖 TIFA 生成的 VQA 质量，如果 VQA 不准则偏好数据有噪声
- 自博弈超过 2 轮后收益递减，生成质量趋于饱和，可能需要更精细的课程学习策略
- BERT 相似度作为理解偏好信号可能不够精确，可考虑替换为更强的语义评估模型

## 相关工作与启发
- **vs DPO (理解/生成分离)**：分别做理解 DPO 和生成 DPO 各自有效但不协调，Pair-DPO 通过乘法耦合将两者统一，效果更优
- **vs MetaMorph/Liquid**：这些工作从数据层面研究理解-生成协同，但没有明确的优化框架来缩小 gap
- **vs Emu3 (生成 DPO)**：Emu3 需要人工排序构建偏好数据，HermesFlow 完全自动化且同时优化理解

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次量化理解-生成 gap 并提出配对 DPO 优化框架，概念清晰但核心创新在于"组合"
- 实验充分度: ⭐⭐⭐⭐ 理解+生成多 benchmark 全面验证，消融清晰，但仅一个 backbone
- 写作质量: ⭐⭐⭐⭐ 动机阐述清楚，pipeline 图示直观，公式完整
- 价值: ⭐⭐⭐⭐ 作为统一 MLLM 的 post-training 对齐框架具有通用性，可推广到下一代模型

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Closing the Modality Gap Aligns Group-Wise Semantics](../../ICLR2026/multimodal_vlm/closing_the_modality_gap_aligns_group-wise_semantics.md)
- [\[ICCV 2025\] MetaMorph: Multimodal Understanding and Generation via Instruction Tuning](../../ICCV2025/multimodal_vlm/metamorph_multimodal_understanding_and_generation_via_instruction_tuning.md)
- [\[NeurIPS 2025\] UniTok: A Unified Tokenizer for Visual Generation and Understanding](unitok_a_unified_tokenizer_for_visual_generation_and_understanding.md)
- [\[NeurIPS 2025\] DanmakuTPPBench: A Multi-modal Benchmark for Temporal Point Process Modeling and Understanding](danmakutppbench_a_multimodal_benchmark_for_temporal_point_pr.md)
- [\[NeurIPS 2025\] Nautilus: A Large Multimodal Model for Underwater Scene Understanding](nautilus_a_large_multimodal_model_for_underwater_scene_understanding.md)

</div>

<!-- RELATED:END -->
