---
title: >-
  [论文解读] Steer LLM Latents for Hallucination Detection
description: >-
  [ICML 2025][幻觉检测][steering vector] 提出 Truthfulness Separator Vector (TSV)，一种轻量级 steering vector，在推理时重塑 LLM 表示空间以增强真实与幻觉输出的分离，仅需 32 个标注样本即可接近全监督性能。 领域现状 领域现状：LLM 幻觉…
tags:
  - "ICML 2025"
  - "幻觉检测"
  - "steering vector"
  - "hallucination detection"
  - "optimal transport"
  - "伪标签"
  - "TSV"
---

# Steer LLM Latents for Hallucination Detection

**会议**: ICML 2025  
**arXiv**: [2503.01917](https://arxiv.org/abs/2503.01917)  
**代码**: -  
**领域**: 幻觉检测  
**关键词**: steering vector, hallucination detection, optimal transport, pseudo-labeling, TSV  

## 一句话总结

提出 Truthfulness Separator Vector (TSV)，一种轻量级 steering vector，在推理时重塑 LLM 表示空间以增强真实与幻觉输出的分离，仅需 32 个标注样本即可接近全监督性能。

## 研究背景与动机

### 领域现状

**领域现状**：LLM 幻觉是安全部署的重大隐患

### 核心矛盾

**核心矛盾**：现有基于潜在空间的方法依赖预训练 LLM 嵌入，但这些嵌入被优化为**语言连贯性**而非**事实准确性**

### 现有痛点

**现有痛点**：预训练嵌入中真实和幻觉内容重叠严重（见 T-SNE 可视化）

### 解决思路

**解决思路**：微调 LLM 计算昂贵且改变模型参数

### 补充说明

**补充说明**：核心问题**：如何在不修改模型参数的前提下重塑潜在空间以区分幻觉？

## 方法详解

### 1. Truthfulness Separator Vector (TSV)

定义可训练向量 $\mathbf{v} \in \mathbb{R}^d$，在推理时添加到中间层 $l$ 的隐状态：

$$\mathbf{h}^{(l)} \leftarrow \mathbf{h}^{(l)} + \lambda \mathbf{v}$$

其中 $\lambda$ 控制干预强度。TSV 跨所有 token 位置共享，通过后续非线性变换影响最终层嵌入。

### 2. 初始训练阶段

使用 von Mises-Fisher 分布建模最终层嵌入，类条件概率为：

$$p(c|\mathbf{r}^{\mathbf{v}}) = \frac{\exp(\kappa \boldsymbol{\mu}_c^\top \mathbf{r}^{\mathbf{v}})}{\sum_{c'} \exp(\kappa \boldsymbol{\mu}_{c'}^\top \mathbf{r}^{\mathbf{v}})}$$

其中 $\mathbf{r}^{\mathbf{v}}$ 为归一化后的最终嵌入，$\boldsymbol{\mu}_c$ 为类原型。

训练目标：最大化 exemplar set $\mathcal{D}_E$ 上的对数似然：

$$\mathcal{L} = -\frac{1}{|\mathcal{D}_E|}\sum_{i=1}^{|\mathcal{D}_E|}\sum_{c \in \mathcal{C}} q(c|\mathbf{r}_i^{\mathbf{v}}) \log p(c|\mathbf{r}_i^{\mathbf{v}})$$

### 3. 增强训练阶段

#### 基于最优传输的伪标签分配

对无标签数据 $\mathcal{D}_U$，通过 Sinkhorn 算法求解最优传输问题分配伪标签：

$$\min_{\mathbf{Q} \in [0,1]^{M \times 2}} -\sum_{m,c} \mathbf{Q}_{m,c} \log \mathbf{P}_{m,c} - \epsilon H(\mathbf{Q})$$

约束包括行和为 $1/M$（每个样本总概率为 1）和列和匹配类分布 $\mathbf{w}$。

#### 置信度筛选

仅选择预测不确定性最低的 $K$ 个伪标签样本加入训练：

$$\mathcal{D}_S = \{\mathcal{D}_U^j \mid j \in \text{TopK}_{i}(-\Omega_i)\}$$

其中 $\Omega_i$ 为交叉熵衡量的不确定性。

## 实验结果

### 主实验：TruthfulQA (AUROC)

| 方法 | LLaMA-3.1-8B |
|------|-------------|
| CCS | 58.1 |
| SAPLMA | 63.2 |
| Probing (supervised) | 71.3 |
| HaloScope | 71.4 |
| **TSV (32 exemplars)** | **84.2** |
| 全监督上界 | 85.5 |

- TSV 比 SOTA 提升 **+12.8%** AUROC
- 仅用 32 个标注样本即接近全监督上界 (84.2% vs 85.5%)

### 跨数据集泛化

在 TriviaQA 和 HaluEval 上测试在 TruthfulQA 训练的 TSV：
- 保持竞争力，展示良好的分布外泛化

### 消融实验

| 组件 | AUROC |
|------|-------|
| TSV (仅初始训练) | 79.8 |
| + OT 伪标签 | 82.5 |
| + 置信度筛选 | **84.2** |
| 无 TSV (直接用嵌入) | 71.3 |

- 每个组件都有明确贡献
- 最佳干预层：中间层（约第 16 层，32 层总共）

## 亮点与洞察

- 首次将 steering vector 用于幻觉**检测**（而非生成缓解），填补了重要空白
- 最优传输伪标签分配考虑了类不平衡，优于简单阈值方法
- 极少标注需求（32 个）即可达到近全监督性能，实用性极强
- 不修改模型参数，可在生成完成后再应用 TSV
- von Mises-Fisher 分布建模与 RMSNorm 后嵌入的单位范数特性完美匹配

## 局限与展望

- $\lambda$ 和干预层 $l$ 的选择需要在验证集上调优
- 对不同 LLM 可能需要重新训练 TSV
- 仅在封闭式 QA 任务上验证，对开放生成场景效果未知
- 伪标签的类分布先验 $\mathbf{w}$ 来自 exemplar set，可能不准确
- 理论上缺乏 TSV 为何能有效分离的深层解释

## 评分

⭐⭐⭐⭐⭐ — 方法轻量优雅、效果惊人，32 个标注样本接近全监督上界，在幻觉检测领域意义重大。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] MultiHaluDet: Multilingual Hallucination Detection via LLM Hidden State Probing](../../ACL2026/hallucination/multihaludet_multilingual_hallucination_detection_via_llm_hidden_state_probing.md)
- [\[ACL 2025\] HalluLens: LLM Hallucination Benchmark](../../ACL2025/hallucination/hallulens_llm_hallucination_benchmark.md)
- [\[ACL 2026\] Rethinking Evaluation for LLM Hallucination Detection: A Desiderata, A New RAG-based Benchmark, New Insights](../../ACL2026/hallucination/rethinking_evaluation_for_llm_hallucination_detection_a_desiderata_a_new_rag-bas.md)
- [\[ICML 2026\] Zero-source LLM Hallucination Detection with Human-like Criteria Probing](../../ICML2026/hallucination/zero-source_llm_hallucination_detection_with_human-like_criteria_probing.md)
- [\[ACL 2026\] 为什么 LLM 在结构化知识上产生幻觉：推理过程的机制分析](../../ACL2026/hallucination/why_llms_hallucinate_on_structured_knowledge_a_mechanistic_analysis_of_reasoning.md)

</div>

<!-- RELATED:END -->
