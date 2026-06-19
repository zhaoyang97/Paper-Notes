---
title: >-
  [论文解读] Ab Initio Nonparametric Variable Selection for Scalable Symbolic Regression with Large p
description: >-
  [ICML2025][可解释性][符号回归] 提出 PAN+SR 框架，通过基于 BART 的非参数变量预筛选，将高维符号回归问题降维至低维子空间，使 19 种现有 SR 方法在高维场景下均获显著性能提升。 核心矛盾 核心矛盾：符号回归（SR）旨在发现可解释的数学表达式，但搜索空间随特征数 p 呈双指数增长…
tags:
  - "ICML2025"
  - "可解释性"
  - "符号回归"
  - "变量选择"
  - "非参数方法"
  - "BART"
  - "高维数据"
---

# Ab Initio Nonparametric Variable Selection for Scalable Symbolic Regression with Large p

**会议**: ICML2025  
**arXiv**: [2410.13681](https://arxiv.org/abs/2410.13681)  
**代码**: [GitHub - PAN_SR](https://github.com/mattsheng/PAN_SR)  
**领域**: 可解释性  
**关键词**: 符号回归, 变量选择, 非参数方法, BART, 高维数据

## 一句话总结
提出 PAN+SR 框架，通过基于 BART 的非参数变量预筛选，将高维符号回归问题降维至低维子空间，使 19 种现有 SR 方法在高维场景下均获显著性能提升。

## 研究背景与动机

### 核心矛盾

**核心矛盾**：符号回归（SR）旨在发现可解释的数学表达式，但搜索空间随特征数 p 呈双指数增长。现有 SR 方法几乎都只在 p≤10 下工作良好，无法扩展到现代科学中的高维数据集（p=102~459）。

### 解决思路

**本文目标**：与传统变量选择不同，SR 要求**假阴性率(FNR)接近零**——遗漏任何相关变量都会导致无法恢复真实函数。而假阳性仅增加计算负担，不影响正确性。这种不对称需求使标准 FDR 控制方法不适用。

### PySR 内置预选方法的失败

PySR 的随机森林特征预选效果不佳，其文档直接指出该选项"几乎不使用"。

## 方法详解

### 整体框架：PAN+SR（两阶段）
1. **PAN 预筛选**：从高维数据识别相关变量子集 Ŝ
2. **SR 求解**：将降维数据输入任意 SR 方法

### 关键设计 1：基于 BART 的变量重要性排名
独立运行 K=20 次 BART，对每棵后验采样的树计算各特征作为分裂变量的比例（VIP）。

### 关键设计 2：排名聚合策略
利用 K 次运行的**平均排名**（而非原始VIP值）：
- 相关特征聚类：排名集中在前端
- 无关特征聚类：排名集中在后端
- 用凝聚层次聚类(AHC)自动切分两组

### 设计优势
- 无需知道稀疏度 $p_0$
- 无可调阈值，完全数据驱动
- PAN 计算开销极低（平均仅 74.14 秒）

## 实验关键数据

### 黑盒回归问题（35个高维真实数据集）

| 指标 | PAN+SR 效果 |
|------|-----------|
| R² 改善 | 18/19 方法提升 |
| 训练时间 | AIFeynman 5x加速，uDSR 3x加速 |
| 模型复杂度 | 不增加或降低 |
| PAN开销 | 平均仅74.14秒 |

### 合成回归（100个高维Feynman方程，p=102~459）

| SR 方法 | 独立解率(%) | PAN+SR 解率(%) |
|---------|-----------|-------------|
| uDSR | 36.6 | **71.8** |
| AIFeynman | 0 (OOM) | **恢复运行** |
| Operon | 18.1 | **27.4** |
| DSR | 8.9 | **25.8** |
| GP-GOMEA | 18.2 | **24.1** |

### 消融：变量选择方法对比

| 方法 | TPR | PAN准则符合度 |
|------|-----|-------------|
| PAN (本文) | **~99%** | ✅ |
| BART-G.SE | 不足 | ❌ |
| Random Forest | 不佳 | ❌ |

## 亮点与洞察

1. 准确识别了 SR 变量选择中 FNR/FPR 的不对称需求，为后续研究奠基。
2. 方法极简："多次BART → 平均排名 → 层次聚类"三步，无超参无阈值。
3. 评估规模空前：138K核时计算量，19种方法×135个数据集×多种SNR。
4. 扩展的高维SRBench（102~459维）填补基准空白。

## 局限与展望

1. 极端噪声下(SNR=0.5, n=500)FNR>5%，PAN准则不再满足。
2. 高度相关特征集合的处理能力仍需提升。
3. 方法依赖 $p_0 \ll p$ 的稀疏性假设，非稀疏场景未讨论。
4. BART 在 p>1000 时效率可能下降。

## 相关工作与启发

- **SRBench**：标准SR基准，本文在其基础上扩展高维问题。
- **iBART**：迭代BART变量选择，是PAN设计灵感来源。
- 启发："先筛选后搜索"范式可推广到NAS、程序合成等组合爆炸问题。

## 补充分析

### SNR与样本量敏感性
- n=1000, SNR=∞：FNR≈0%，FPR≈0%，最优条件
- n=1000, SNR=10：FNR≈0%但FPR初增，解率受噪声限制降至0%
- n=500, SNR=0.5：FNR>5%，极端噪声下PAN准则失效
- 样本量影响小，SNR是主导因素

### 计算开销对比
PAN预筛选平均仅74s（黑盒）和325s（合成），K=20次BART可完全并行。对比下游SR（AIFeynman 71250s → PAN+AIFeynman 13997s = 5x加速），预筛选开销可忽略。

### 扩展SRBench基准
对Feynman方程每个相关特征生成s=50个同分布无关特征，总维度p=51·p₀（102~459维），加入8种信噪比。

## 评分
- 新颖性: ⭐⭐⭐⭐☆（4.0/5）
- 实验充分度: ⭐⭐⭐⭐⭐（5.0/5）— 迄今SR领域最大规模系统评估
- 写作质量: ⭐⭐⭐⭐⭐（5.0/5）
- 价值: ⭐⭐⭐⭐☆（4.0/5）— 对SR社区有直接实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Towards Scaling Laws for Symbolic Regression](../../NeurIPS2025/interpretability/towards_scaling_laws_for_symbolic_regression.md)
- [\[ICML 2026\] Breaking the Simplification Bottleneck in Amortized Neural Symbolic Regression](../../ICML2026/interpretability/breaking_the_simplification_bottleneck_in_amortized_neural_symbolic_regression.md)
- [\[ICML 2025\] Inference-Time Decomposition of Activations (ITDA): A Scalable Approach to Interpreting Large Language Models](inference-time_decomposition_of_activations_itda_a_scalable_approach_to_interpre.md)
- [\[ICML 2026\] Scalable Circuit Learning for Interpreting Large Language Models](../../ICML2026/interpretability/scalable_circuit_learning_for_interpreting_large_language_models.md)
- [\[NeurIPS 2025\] TangledFeatures: Robust Feature Selection in Highly Correlated Spaces](../../NeurIPS2025/interpretability/tangledfeatures_robust_feature_selection_in_highly_correlated_spaces.md)

</div>

<!-- RELATED:END -->
