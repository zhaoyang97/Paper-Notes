---
title: >-
  [论文解读] Towards Scaling Laws for Symbolic Regression
description: >-
  [NeurIPS 2025][可解释性][符号回归] 首次系统研究符号回归（SR）中的缩放定律，证明基于 Transformer 的端到端 SR 在三个数量级的计算范围内遵循幂律缩放趋势，并给出最优 token-to-parameter ratio $\approx 15$、batch size 和学习率随模型规模增长的经验规律。
tags:
  - "NeurIPS 2025"
  - "可解释性"
  - "符号回归"
  - "缩放定律"
  - "Transformer"
  - "幂律"
  - "计算最优"
---

# Towards Scaling Laws for Symbolic Regression

**会议**: NeurIPS 2025  
**arXiv**: [2510.26064](https://arxiv.org/abs/2510.26064)  
**代码**: 无  
**领域**: 可解释性/符号回归  
**关键词**: 符号回归, 缩放定律, Transformer, 幂律, 计算最优

## 一句话总结

首次系统研究符号回归（SR）中的缩放定律，证明基于 Transformer 的端到端 SR 在三个数量级的计算范围内遵循幂律缩放趋势，并给出最优 token-to-parameter ratio $\approx 15$、batch size 和学习率随模型规模增长的经验规律。

## 研究背景与动机

- **符号回归**旨在从观测数据中发现底层数学表达式，兼具可解释性和泛化能力
- 近年来基于预训练 Transformer 的 SR 方法逐渐追平遗传编程方法，但**规模效应**几乎未被研究——现有工作参数量均未超过 $\sim 100$M
- 受 LLM 缩放定律（Kaplan et al., Hoffmann et al.）的启发，作者提出核心问题：**SR 是否存在类似的缩放定律？** 如果存在，能否指导下一代 SR 模型的设计？
- 现有工作主要在固定规模下调整训练细节，缺少对规模-性能关系的系统分析

## 方法详解

### 整体框架

采用端到端 encoder-decoder Transformer 架构，输入为表格数据（数值对），输出为 LaTeX 格式的数学表达式。整体流程：

1. **数据生成**：递归生成基础表达式集合 → 插入随机常数 + 采样数据集
2. **编码**：将表格中每个数值拆分为尾数+指数，投影到嵌入空间
3. **模型推理**：表格感知编码器（行/列双向注意力）→ 标准解码器自回归生成表达式
4. **评估**：采样 128 个表达式，取 $R^2$ 最高的作为预测

### 关键设计

1. **两步数据生成**：
    - 第一步：从变量 $\{x_1, x_2\}$ 出发，递归应用一元算子（exp, sin, neg, sqrt）和二元算子（+, -, ·, ÷）生成所有深度 ≤3 的表达式树，用 SymPy 做规范化和去重，得到 $|E|=100{,}000$ 个基础表达式
    - 第二步：对每个基础表达式采样 $k=3{,}600$ 个（表达式-数据集）对——随机插入整数常数（范围 -9 到 9，概率 $p=0.2$）并从高斯混合分布中采样 64 个数据点
    - 优势：避免传统方法中某些表达式过度采样的偏差，训练数据更干净

2. **表格感知编码器架构**：
    - 传统方法将每个输入点合并为单一嵌入；本文为表格中每个**单元格**生成独立嵌入
    - 尾数和指数分别上投影到嵌入维度后相加
    - 借鉴表格基础模型（TabPFN 等），在每层中同时执行**行注意力**（跨变量）和**列注意力**（跨数据点）
    - 解码器仅交叉注意目标单元格的更新嵌入

3. **端到端训练 pipeline**：
    - 直接输出完整表达式（含常数），无需 BFGS 后处理
    - 目标表达式以 LaTeX 字符串表示，常数逐位 tokenize
    - 不同模型规模共享同一数据生成和评估协议，保证缩放分析的公平性

### 损失函数 / 训练策略

- **损失函数**：标准交叉熵损失，预测 token 与真实表达式 token 之间
- **优化器**：AdamCPR（$\beta_1=0.9, \beta_2=0.98$），配合线性 warmup（前 5% steps）+ cosine annealing
- **FLOPs 估算**：$\text{FLOPs} \approx 6 \cdot (N_{enc} \cdot D_{in} + N_{dec} \cdot D_{out})$，其中 $N = N_{enc} + N_{dec}$ 为前馈参数数量
- **超参搜索策略**：对每个模型规模，在 token-to-parameter ratio = 20 下网格搜索 batch size 和 learning rate，找到最优配置后再扫描不同 ratio（5 到 80）

## 实验关键数据

### 主实验

五种模型规模（6.5M - 93M）的详细架构和最佳性能：

| 模型 | 维度 | 编码器层数 | 解码器层数 | 注意力头数 | 参数量 |
|------|------|-----------|-----------|-----------|--------|
| XS | 256 | 3 | 3 | 4 | 6.48M |
| S | 320 | 4 | 4 | 5 | 13.40M |
| M | 384 | 5 | 5 | 6 | 24.01M |
| L | 448 | 7 | 7 | 7 | 45.53M |
| XL | 512 | 11 | 11 | 8 | 93.08M |

各模型在最高计算预算下的最佳性能：

| 模型 | 最高 FLOPs | $\text{Acc}_{\text{solved}}$ | $\text{Acc}_{R^2>0.99}$ | 验证损失 |
|------|-----------|------|------|------|
| 6.5M | 7.20e+16 | 0.149 | 0.526 | 0.424 |
| 13.5M | 2.88e+17 | 0.271 | 0.667 | 0.312 |
| 24M | 9.81e+17 | 0.378 | 0.762 | 0.240 |
| 45.5M | 3.53e+18 | 0.519 | 0.835 | 0.168 |
| 93M | 1.47e+19 | 0.597 | 0.883 | 0.105 |

### 消融实验

- **Token-to-parameter ratio 扫描**：ratio 从 5 到 80，最优值约为 $\approx 15$，且随计算预算增大有轻微上升趋势，表明数据量应比模型参数增长稍快
- **Batch size 缩放**：最优 batch size 随模型规模增大——6.5M 用 32，13.5M 用 128，93M 用 256
- **Learning rate 缩放**：最优学习率随计算预算增大而增大（6.5M 和 24M 用 4.6e-4，93M 用 1.0e-3），这与 LLM 中学习率随规模下降的趋势**相反**

### 关键发现

1. **幂律缩放**：$\text{Acc}_{\text{solved}}$ 从最低计算预算的 $\sim 0.03$ 增长到最高的 $\sim 0.60$，遵循清晰的幂律趋势；外推预测在 $3.8 \times 10^{21}$ FLOPs 时可达 0.8
2. **$\text{Acc}_{R^2>0.99}$ 改善更快**：近似匹配比精确匹配容易得多，93M 模型已达 0.883
3. **无饱和迹象**：最大模型在最大计算预算下仍在持续改善，暗示进一步扩大规模可获得更好性能

## 亮点与洞察

- **首个 SR 缩放定律**：证明符号回归也服从类似 LLM 的幂律缩放，为该领域提供了全新的设计原则——不再靠"精巧技巧"，而是通过增加规模系统性提升性能
- **学习率趋势与 LLM 相反**：SR 中最优学习率随规模增大而增大，揭示不同任务的训练特性存在本质差异
- **Token-to-parameter ratio 的直接可用价值**：$\approx 15$ 的最优比值为实践者提供了直接的资源分配启发式
- **数据生成方法论**：递归生成 + 规范化去重的策略优于传统随机采样，保证了表达式空间的均匀覆盖
- **表格感知编码器**的行列双向注意力设计，是从表格基础模型领域借鉴的有效跨域迁移

## 局限与展望

- **表达式复杂度受限**：仅考虑 ≤2 个变量和小整数常数，现实 SR 任务通常涉及更多变量和浮点常数
- **单次 seed 训练**：由于计算限制，每个配置仅单 seed 训练，结果存在方差
- **计算范围有限**：三个数量级的范围对外推预测的可靠性有限
- **未与现有 SR 方法对比**：专注于缩放洞察，未验证是否能超越 GP 或其他深度 SR 方法
- **可改进方向**：
    - 扩展到更多变量、浮点常数和更复杂的算子集合
    - 验证端到端 SR + 改进数据生成 + 缩放是否能全面超越其他方法
    - 引入更大规模的训练（>100M 参数）验证外推预测

## 相关工作与启发

- **E2E**（Kamienny et al., NeurIPS 2022）：最接近的前身工作，首个端到端 Transformer SR，本文在此基础上将关注点从损失工程转向规模效应
- **Biggio et al.**：首次提出预训练 Transformer 做 SR，使用集合编码器 + BFGS 常数优化
- **Chinchilla**（Hoffmann et al.）：语言模型的计算最优缩放定律，本文的方法论直接借鉴
- **TabPFN**（Hollmann et al., Nature 2025）：表格基础模型，其行列注意力架构被本文采纳
- 本文的核心启发：**符号回归不应只追求精巧的训练技巧，而应像 LLM 一样系统性地利用规模带来的收益**

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首次将缩放定律研究引入符号回归，填补重要空白
- **技术深度**: ⭐⭐⭐ — 方法本身较为标准（Transformer + 合成数据），技术贡献主要在实验设计和分析
- **实验充分度**: ⭐⭐⭐⭐ — 5 种模型规模、3 个数量级计算范围、系统化超参扫描，但单 seed 是短板
- **实用价值**: ⭐⭐⭐⭐ — token-to-parameter ratio、batch size/lr 缩放趋势对 SR 从业者直接有用
- **表达清晰度**: ⭐⭐⭐⭐⭐ — 论文结构清晰，图表精心设计，关键发现突出

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Breaking the Simplification Bottleneck in Amortized Neural Symbolic Regression](../../ICML2026/interpretability/breaking_the_simplification_bottleneck_in_amortized_neural_symbolic_regression.md)
- [\[NeurIPS 2025\] Sloth: Scaling Laws for LLM Skills to Predict Multi-Benchmark Performance Across Families](sloth_scaling_laws_for_llm_skills_to_predict_multi-benchmark_performance_across_.md)
- [\[ICML 2025\] Ab Initio Nonparametric Variable Selection for Scalable Symbolic Regression with Large p](../../ICML2025/interpretability/ab_initio_nonparametric_variable_selection_for_scalable_symbolic_regression_with.md)
- [\[ICLR 2026\] Closing the Curvature Gap: Full Transformer Hessians and Their Implications for Scaling Laws](../../ICLR2026/interpretability/closing_the_curvature_gap_full_transformer_hessians_and_their_implications_for_s.md)
- [\[NeurIPS 2025\] Superposition Yields Robust Neural Scaling](superposition_yields_robust_neural_scaling.md)

</div>

<!-- RELATED:END -->
