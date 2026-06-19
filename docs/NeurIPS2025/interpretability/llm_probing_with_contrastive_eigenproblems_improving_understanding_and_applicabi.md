---
title: >-
  [论文解读] LLM Probing with Contrastive Eigenproblems: Improving Understanding and Applicability of CCS
description: >-
  [NeurIPS 2025][可解释性][CCS] 本文对无监督探测方法 CCS（Contrast-Consistent Search）进行了深入分析，提出将 CCS 重新表述为特征值问题（Contrastive Eigenproblems），获得闭式解和可解释的特征值，避免了 CCS 对随机初始化的敏感性，并自然扩展到多变量设置。
tags:
  - "NeurIPS 2025"
  - "可解释性"
  - "CCS"
  - "对比探测"
  - "特征值问题"
  - "机械可解释性"
  - "潜在知识发现"
---

# LLM Probing with Contrastive Eigenproblems: Improving Understanding and Applicability of CCS

**会议**: NeurIPS 2025  
**arXiv**: [2511.02089](https://arxiv.org/abs/2511.02089)  
**代码**: 待确认  
**领域**: 可解释性  
**关键词**: CCS, 对比探测, 特征值问题, 机械可解释性, 潜在知识发现  

## 一句话总结

本文对无监督探测方法 CCS（Contrast-Consistent Search）进行了深入分析，提出将 CCS 重新表述为特征值问题（Contrastive Eigenproblems），获得闭式解和可解释的特征值，避免了 CCS 对随机初始化的敏感性，并自然扩展到多变量设置。

## 研究背景与动机

大语言模型（LLM）在各种基准测试中表现出色，但其内部工作机制仍不透明。机械可解释性（Mechanistic Interpretability）旨在识别模型行为背后的机制以及模型使用的变量和编码方式。

CCS 是 Burns et al. (2023) 提出的无监督探测方法，用于检测语言模型是否在其内部激活中表示句子真假等二值特征。它的核心优势在于不依赖人工标注，不假设模型的真值判断与人类标签一致。然而 CCS 存在以下问题：

**两项损失函数的理解不充分**：CCS 包含一致性损失（consistency loss）和置信度损失（confidence loss），后者被认为仅用于避免退化解 $p(\mathbf{x}^+) = p(\mathbf{x}^-) = 0.5$，但实际作用远不止于此

**对随机初始化敏感**：在部分数据集上，CCS 的准确率随种子变化波动剧大（如 SNLI 准确率范围 49%~90%）

**无法诊断数据质量**：当对比数据未能成功隔离单一特征时，CCS 难以发现问题根源

**单变量限制**：原始 CCS 只能探测单个二值特征，难以扩展到多变量场景

## 方法详解

### 整体框架

本文的核心思路是将 CCS 的优化目标线性化，将其重新表述为特征值问题，从而获得闭式解。关键概念包括：

- **对比对（contrast pairs）**：$(X^+, X^-)$ 为一对语义相反的输入（如肯定句和否定句）
- **共性矩阵**（Commonality Matrix）：$\mathbf{C} = \mathbf{X}^- + \mathbf{X}^+$，捕获正负样本的共同特征
- **位移矩阵**（Displacement Matrix）：$\mathbf{D} = \mathbf{X}^- - \mathbf{X}^+$，捕获正负样本的差异特征

### 关键设计

**1. 相对对比一致性（Relative Contrast Consistency）**

作者发现 CCS 中置信度损失的真正作用是将探针偏向高方差方向（即数据的主成分方向）。如果某个方向上数据方差本身就很低，那么该方向上高一致性并不意味着模型真正编码了目标特征。因此需要关注的是**相对**一致性：

$$\text{目标} = \min_{\hat{\boldsymbol{\theta}}} \frac{\|\hat{\boldsymbol{\theta}}^\intercal (\mathbf{X}^+ + \mathbf{X}^-)\|}{\|\hat{\boldsymbol{\theta}}^\intercal \mathbf{X}^{+-}\|}$$

即在某方向上 $\mathbf{C}$ 的方差相对于整体数据 $\mathbf{X}^{+-}$ 的方差越小越好。

**2. 差分相对对比（DRC，Difference-Relative Contrast）**

将方差变化表示为差值的特征值问题：

$$(\mathbf{C}^\intercal \mathbf{C} - \mathbf{X}^{+-\intercal} \mathbf{X}^{+-}) \mathbf{n}_k = \lambda_k \mathbf{n}_k$$

$$(\mathbf{D}^\intercal \mathbf{D} - \mathbf{X}^{+-\intercal} \mathbf{X}^{+-}) \mathbf{t}_k = \mu_k \mathbf{t}_k$$

负 $\lambda_k$ 对应的方向 $\mathbf{n}_k$ 上共性矩阵方差小于整体方差，即该方向上对比特征差异被消除，适合作为分类方向。正 $\mu_k$ 对应的方向 $\mathbf{t}_k$ 上位移矩阵方差大于整体方差，即该方向编码了对比特征差异。

**3. 比率相对对比（RRC，Ratio-Relative Contrast）**

将方差变化表示为比值的广义特征值问题：

$$\mathbf{C}^\intercal \mathbf{C} \mathbf{n}_k = \lambda_k \mathbf{X}^{+-\intercal} \mathbf{X}^{+-} \mathbf{n}_k$$

特征值 $\lambda_k$ 给出的是方差的比率而非差值。两种问题最终归约为同一个对称特征值问题，因此 DRC 与 RRC 给出相同的特征向量基。

**4. 多变量扩展**

通过构造包含多组对比对的 $\mathbf{D}$ 矩阵（如同时变化极性和真值），可以在多个正交方向上获得对比一致的特征向量，从而同时探测多个二值特征。

### 损失函数 / 训练策略

本方法为闭式解，无需梯度下降训练。具体步骤：
1. 从 LLM 中层（论文使用 Llama-2-7B 第 16 层）提取激活
2. 对激活进行均值中心化
3. 可选使用 SVD 降维去除零空间
4. 构造 $\mathbf{C}$ 和 $\mathbf{D}$ 矩阵
5. 求解特征值问题，取最小/最大特征值对应的特征向量作为探针方向

## 实验关键数据

### 主实验

在 Llama-2-7B 第 16 层激活上，比较 CCS（30 个种子）与 DRC/RRC 的分类准确率：

| 数据集 | CCS min | CCS median | CCS max | DRC | RRC |
|--------|---------|------------|---------|-----|-----|
| comparisons | 100 | 100 | 100 | 100 | 100 |
| sp_en_trans | 99 | 99 | 99 | 98 | 98 |
| cities | 99 | 99 | 99 | 99 | 99 |
| amazon | 94 | 94 | 94 | 94 | 94 |
| imdb | 86 | 87 | 88 | 87 | 87 |
| ent_bank | 84 | 86 | 87 | 84 | 86 |
| snli | 49 | 86 | 90 | 82 | 73 |
| copa | 51 | 55 | 68 | 53 | 52 |
| rte | 46 | 50 | 61 | 50 | 50 |

### 消融实验

对 CCS 的两项损失进行消融，验证置信度损失的实际作用：

| 方法 | comparisons | sp_en_trans | cities | amazon | imdb |
|------|-------------|-------------|--------|--------|------|
| CCS 完整 | 100±0 | 99±0 | 99±0 | 93±0 | 87±0 |
| 仅 $\mathcal{L}_{conf}$ | 100±1 | 96±8 | 98±4 | 94±0 | 81±9 |
| 仅 $\mathcal{L}_{cons}$ | 66±11 | 64±13 | 70±14 | 67±9 | 60±6 |
| $\mathcal{L}_{cons}$+a1+a2 | 59±7 | 74±14 | 67±12 | 64±9 | 58±7 |

消融表明：置信度损失的作用远非仅仅避免退化解，它实际上将探针偏向高方差方向，这对于找到准确探针至关重要。

### 关键发现

1. **特征值分布可诊断数据质量**：当数据成功隔离单一对比特征时，顶部特征值明显突出（如 amazon 数据集）；当数据包含多个混合特征时，特征值分布更平坦（如 snli、copa）
2. **COPA 案例研究**：DRC 顶部特征向量编码的是**情感极性**而非句子真值。第二特征向量才编码真值，准确率达 70%，略优于 CCS 最优的 68%
3. **多变量实验**：在 cities 数据集上，DRC 前三个特征向量分别编码了真值、命题基础真值和极性，三个正交方向共同编码了真值与极性的交互关系

## 亮点与洞察

1. **理论洞察深刻**：证明了 CCS 中置信度损失的真实作用是偏向高方差方向，而非仅仅避免退化解。这一发现促使了"相对对比一致性"概念的提出
2. **闭式解替代梯度优化**：特征值分解完全避免了随机初始化问题，在 CCS 表现稳定的数据集上几乎完美匹配 CCS 准确率
3. **特征值的诊断能力**：特征值分布可以定量指示对比数据是否成功隔离了单一特征，这为数据质量评估提供了新工具
4. **多变量扩展自然优雅**：无需额外设计，特征值问题天然支持多变量探测，成功复现了真值-极性共享子空间的结果

## 局限与展望

1. **分类与干预方向的统一**：DRC/RRC 方法无法区分分类方向 $\mathbf{n}$ 和干预方向 $\mathbf{t}$（在特征相关时两者不同），未来需要设计能分离两种方向的方法
2. **在困难数据集上表现一般**：当对比数据质量差（如 copa、rte）时，方法性能仍不理想，但至少能诊断问题
3. **仅使用线性探针**：假设特征在潜在空间中线性编码，可能遗漏非线性编码的特征
4. **实验规模有限**：仅在 Llama-2-7B 一个模型上验证，需要在更多模型和规模上测试泛化性
5. **缺乏下游应用验证**：未展示特征值诊断在实际可解释性工作流中的应用价值

## 相关工作与启发

- **CCS 系列工作**：Burns et al. (2023) 提出原始 CCS；Farquhar et al. (2023) 证明 CCS 可能找到非真值特征；Belrose et al. (2024) 分析并扩展了 CRC-TPC
- **真值-极性交互**：Bürger et al. (2024) 发现真值和极性在共享子空间中编码，本文用更优雅的特征值方法复现了该结果
- **Fry et al. (2023)**：引入中点-位移损失，使用了与本文相同的 $\mathbf{C}$ 和 $\mathbf{D}$ 矩阵，但未进行相对化处理
- **真值-极性交互**：Bürger et al. (2024) 发现真值和极性在共享子空间中编码，本文用更优雅的特征值方法复现
- **Laurito et al. (2024)**：通过聚类+归一化消除不相关对比方向，与本文"寻找所有对比方向"的思路互补
- **激活干预**：Turner et al. (2024) 的 Activation Addition 方法利用类似的方向进行模型干预
- **启发**：特征值分解方法可推广到偏见检测、安全特征探测等场景；特征值分布可作为对比数据集设计质量的定量指标

## 评分

- 新颖性: ⭐⭐⭐⭐ 将 CCS 重新表述为特征值问题是高度原创的贡献，理论推导严谨优美
- 实验充分度: ⭐⭐⭐ 在 9 个数据集上验证覆盖多种场景，但仅用 Llama-2-7B 单一模型
- 写作质量: ⭐⭐⭐⭐⭐ 从实验观察→理论分析→方法设计→多变量扩展一气呵成，数学推导详尽清晰
- 价值: ⭐⭐⭐⭐ 为无监督探测提供了更深入的理论理解和实用的特征值诊断工具

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Improving Perturbation-based Explanations by Understanding the Role of Uncertainty Calibration](improving_perturbation-based_explanations_by_understanding_the_role_of_uncertain.md)
- [\[NeurIPS 2025\] URLs Help, Topics Guide: Understanding Metadata Utility in LLM Training](urls_help_topics_guide_understanding_metadata_utility_in_llm_training.md)
- [\[AAAI 2026\] Explainable Melanoma Diagnosis with Contrastive Learning and LLM-based Report Generation](../../AAAI2026/interpretability/explainable_melanoma_diagnosis_with_contrastive_learning_and_llm-based_report_ge.md)
- [\[NeurIPS 2025\] Understanding Prompt Tuning and In-Context Learning via Meta-Learning](understanding_prompt_tuning_and_in-context_learning_via_meta-learning.md)
- [\[ACL 2025\] Around the World in 24 Hours: Probing LLM Knowledge of Time and Place](../../ACL2025/interpretability/around_the_world_in_24_hours_probing_llm_knowledge_of_time_and_place.md)

</div>

<!-- RELATED:END -->
