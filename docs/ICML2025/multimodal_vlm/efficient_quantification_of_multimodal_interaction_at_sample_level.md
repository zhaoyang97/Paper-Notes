---
title: >-
  [论文解读] Efficient Quantification of Multimodal Interaction at Sample Level
description: >-
  [ICML 2025][多模态VLM][多模态交互量化] 提出 LSMI（Lightweight Sample-wise Multimodal Interaction）估计器，首次实现了对真实世界连续分布数据的**逐样本级别**多模态交互（冗余、唯一性、协同）精确且高效的量化，并展示了其在数据分区、知识蒸馏和模型集成中的实用价值。
tags:
  - "ICML 2025"
  - "多模态VLM"
  - "多模态交互量化"
  - "部分信息分解"
  - "逐样本估计"
  - "冗余/唯一性/协同"
  - "熵估计"
---

# Efficient Quantification of Multimodal Interaction at Sample Level

**会议**: ICML 2025  
**arXiv**: [2506.17248](https://arxiv.org/abs/2506.17248)  
**代码**: [GeWu-Lab/LSMI_Estimator](https://github.com/GeWu-Lab/LSMI_Estimator)  
**领域**: 多模态VLM  
**关键词**: 多模态交互量化, 部分信息分解, 逐样本估计, 冗余/唯一性/协同, 熵估计

## 一句话总结

提出 LSMI（Lightweight Sample-wise Multimodal Interaction）估计器，首次实现了对真实世界连续分布数据的**逐样本级别**多模态交互（冗余、唯一性、协同）精确且高效的量化，并展示了其在数据分区、知识蒸馏和模型集成中的实用价值。

## 研究背景与动机

多模态信息由三种基本交互组成：**冗余**（Redundancy，模态间共享信息）、**唯一性**（Uniqueness，单个模态独有的信息）和**协同**（Synergy，仅当模态联合时才涌现的信息）。理解这些交互对于分析多模态系统的信息动态至关重要。

现有方法的局限性：

**部分信息分解（PID）** 框架主要针对离散分布定义交互，难以直接扩展到连续分布
2. 基于分布优化的方法（如 PID-Batch）只能在**整个数据集层面**量化交互，计算开销大且无法提供逐样本级别的细粒度分析
3. 已有的逐点（pointwise）PID 方法在连续分布上缺乏高效实用的方案

核心动机：不同样本的交互模式差异很大（如乐器图像和声音高度冗余，而"挠痒痒"需要多模态协同才能识别），逐样本分析能提供更精细的理解和更强的可解释性。

## 方法详解

### 整体框架

LSMI 的核心思路是：将多模态信息分解问题从"分布级别"下沉到"逐样本级别"，通过定义合理的逐点冗余度量，结合轻量级熵估计模型，高效计算每个样本的四种交互值 $r, u_1, u_2, s$。

整体流程（Algorithm 1）：

1. **输入**：双模态数据 $x_1, x_2$ 及目标 $y$；预训练的判别模型 $p(y|x_1,x_2), p(y|x_1), p(y|x_2)$
2. **训练熵估计器** $h_{\theta_1}, h_{\theta_2}$，分别对两个模态的数据分布进行熵估计
3. **计算逐样本熵** $h(x_1), h(x_2)$ 以及条件熵 $h(x_1|y), h(x_2|y)$
4. **计算冗余分量** $r^+, r^-$，得到冗余 $r = r^+ - r^-$
5. **计算逐点互信息** $i(x_1;y), i(x_2;y), i(x_1,x_2;y)$，由分解方程求解 $u_1, u_2, s$
6. **输出**：每个样本的 $r, u_1, u_2, s$

### 关键设计

#### 1. 基于冗余的逐点交互框架

将分布级分解方程扩展到逐点（event-level）：

$$i(x_1;y) = r + u_1, \quad i(x_2;y) = r + u_2$$
$$i(x_1, x_2; y) = r + u_1 + u_2 + s$$

关键挑战：四个未知量 $r, u_1, u_2, s$，三个方程，需要额外约束确定冗余 $r$。

#### 2. 信息分量分解解决负互信息问题

直接用逐点互信息 $i(x;y)$ 定义冗余存在问题：逐点互信息可以为负（当 $x$ 对 $y$ 提供误导信息时），破坏冗余分解框架要求的单调性。

**解决方案**：将互信息分解为两个非负分量：

$$i(x;y) = i^+(x;y) - i^-(x;y)$$

其中：
- $i^+(x;y) = h(x) = -\log p(x)$（自信息/惊讶度，始终非负）
- $i^-(x;y) = h(x|y) = -\log p(x|y)$（条件自信息，始终非负）

两个分量均满足单调性，可分别在格结构上进行冗余分解。

#### 3. 分量级冗余定义

在每个分量上用最小值操作定义冗余（集合论直觉：冗余不应超过任何单一来源的信息）：

$$r^+(x_1;x_2;y) = \min(i^+(x_1;y),\; i^+(x_2;y))$$
$$r^-(x_1;x_2;y) = \min(i^-(x_1;y),\; i^-(x_2;y))$$

最终冗余：

$$r(x_1;x_2;y) = r^+(x_1;x_2;y) - r^-(x_1;x_2;y)$$

确定 $r$ 后，$u_1, u_2, s$ 由分解方程唯一确定。

#### 4. 轻量级熵估计（KNIFE）

采用 KNIFE 差分熵估计器，利用可学习参数 $\theta$ 建模 $p_\theta(x)$：

$$\mathbb{E}[h_\theta(x)] = \mathbb{E}[h(x)] + D_{KL}(p(x) \| p_\theta(x)) \geq H(X)$$

通过最小化 KL 散度优化参数，获得紧的熵上界。负分量通过以下公式计算：

$$i^-(x_m;y) = h_{\theta_m}(x_m) - h(y) - \log p(y|x_m), \quad m \in \{1,2\}$$

### 损失函数 / 训练策略

- **熵估计器训练**：最小化 $\mathbb{E}[h_\theta(x)]$（即最小化估计分布与真实分布的 KL 散度）
- **判别模型**：使用标准分类损失预训练单模态模型 $p(y|x_1), p(y|x_2)$ 和多模态模型 $p(y|x_1,x_2)$
- 整个估计流程**无需联合分布建模**，仅需逐模态的熵映射 $\mathcal{X}_m \to \mathbb{R}^n$，复杂度远低于 PID-Batch 的联合分布建模 $\mathcal{X}_1 \times \mathcal{X}_2 \times \mathcal{Y} \to \mathbb{R}^n$

## 实验关键数据

### 主实验

**合成数据验证（电路逻辑）：**

| 方法 | XOR: R | XOR: S | OR: R | OR: U₁ | XOR+NOT: U₂ | XOR+NOT: S |
|------|--------|--------|-------|--------|-------------|-----------|
| PID-CVX | 0.000 | 0.692 | 0.210 | 0.001 | 0.338 | 0.346 |
| PID-Batch | 0.000 | 0.690 | 0.200 | 0.018 | 0.257 | 0.381 |
| **LSMI** | **0.000** | **0.691** | **0.215** | **0.001** | **0.336** | **0.347** |
| GT | 0.000 | 0.693 | 0.215 | 0.000 | 0.347 | 0.347 |

LSMI 在所有逻辑任务上均与 Ground Truth 高度一致，误差远小于 PID-Batch。

**真实数据集交互估计（与人类判断的一致性）：**

| 数据集 | 估计方法 | R | U₁ | U₂ | S |
|--------|---------|------|------|------|------|
| KS | LSMI | 3.28 | 0.11 | 0.00 | 0.03 |
| KS | PID-Batch | 3.16 | 0.02 | 0.19 | 0.01 |
| Food-101 | LSMI | 4.19 | 0.34 | 0.00 | 0.08 |
| CMU-MOSEI | LSMI | 0.02 | 0.12 | 0.01 | 0.24 |

LSMI 与人类注释在 Food-101 上的 Pearson 相关系数达到 **0.98（冗余）** 和 **0.95（文本唯一性）**。

**时间效率对比：**

| 数据集 | LSMI (s) | PID-Batch (s) | 加速比 |
|--------|----------|---------------|--------|
| KS | 454.4 | 1700.5 | 3.7× |
| CREMA-D | 667.1 | 3124.4 | 4.7× |
| UCF-101 | 426.1 | 5876.5 | 13.8× |
| Food-101 | 501.5 | 21928.0 | 43.7× |

类别数越多，LSMI 的效率优势越显著（Food-101 快 43.7 倍）。

### 消融实验

**融合阶段对学习到的交互影响（KS 数据集，层级式 Transformer）：**

| 融合层 $l$ | R | U₁ | U₂ | S | 总信息 |
|-----------|------|------|------|------|--------|
| 0（最早融合） | 1.238 | 0.737 | 0.000 | 1.445 | 3.420 |
| 2 | 1.975 | 1.093 | 0.000 | 0.355 | 3.423 |
| 4（最晚融合） | 2.335 | 0.907 | 0.000 | 0.181 | 3.423 |

**域偏移影响（ID vs OOD）：**

| 数据集 | 设置 | R | U₁ | U₂ | S | 总信息 |
|--------|------|------|------|------|------|--------|
| UCF | ID | 3.319 | 1.289 | 0.000 | 0.006 | 4.614 |
| UCF | OOD | 2.511 | 0.504 | 0.053 | 0.698 | 3.766 |
| KS | ID | 2.371 | 0.031 | 0.730 | 0.300 | 3.432 |
| KS | OOD | 1.864 | 0.083 | 0.386 | 0.559 | 2.892 |

### 关键发现

1. **早期融合促进协同，晚期融合促进冗余**：融合层 $l=0$ 时协同 $S=1.445$ 远高于冗余 $R=1.238$；$l=4$ 时冗余 $R=2.335$ 远高于协同 $S=0.181$，但总信息量基本不变
2. **OOD 数据更依赖协同**：OOD 场景下协同信息占比显著增加，说明面对陌生数据时模型需要更多跨模态互补
3. **类别级交互模式符合人类认知**：乐器类别（如弹风琴、演奏手风琴）高冗余；视觉相关类别（草地、雪地）倾向视觉唯一性；听觉相关类别（擤鼻子）倾向听觉唯一性；复杂识别（挠痒痒）依赖协同

## 亮点与洞察

1. **理论贡献扎实**：通过信息分量分解（$i^+, i^-$）巧妙绕开了逐点互信息可能为负导致的单调性破坏问题，定义了合理的逐样本冗余度量
2. **实用性强**：三个下游应用验证了逐样本交互估计的价值：
    - **冗余引导的数据分区**：高冗余子集微调 ImageBind 提升多模态对齐质量，低冗余子集帮助弱势模态学习
    - **交互引导的知识蒸馏**：根据 $r, u, s$ 的相对大小选择蒸馏策略（冗余/唯一→特征蒸馏，协同→输出蒸馏），优于直接蒸馏
    - **交互引导的模型集成**：即使添加低精度模型也能提升性能，因为不同模型关注不同交互模式
3. **效率极高**：无需联合分布建模，计算复杂度与类别数无关，在 Food-101 (101类) 上比 PID-Batch 快 43.7 倍

## 局限与展望

1. **两模态限制**：理论框架基于双模态 PID，多模态（≥3）场景只能采用逐对分析策略，缺乏统一的高阶交互分解
2. **依赖预训练模型质量**：交互估计依赖判别模型对真实分布的近似程度，模型欠拟合会影响估计精度
3. **负冗余的语义解释**：标签噪声实验中出现大量负信息值，其物理含义有待进一步探讨
4. **未探索动态融合**：可进一步研究如何在训练过程中动态利用逐样本交互信息来自适应调整融合策略

## 相关工作与启发

- **PID 理论**（Williams & Beer, 2010; Bertschinger et al., 2014）：提供了交互分解的基础框架
- **PID-Batch**（Liang et al., 2023b）：首个应用于复杂真实数据集的交互估计方法，但仅限于分布级别
- **KNIFE**（Pichler et al., 2022）：提供了高效的差分熵估计工具，是 LSMI 的计算基础
- **启发**：逐样本交互估计思路可拓展到更多领域——如多模态数据清洗（识别信息冲突样本）、课程学习（按交互复杂度排序训练样本）、主动学习（选择交互模式最多样的样本标注）

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 首次实现真实数据逐样本级别的多模态交互量化
- 实验充分度: ⭐⭐⭐⭐⭐ — 合成+真实数据，精度/效率/应用三方面验证
- 写作质量: ⭐⭐⭐⭐ — 理论推导清晰，但符号较多需要仔细阅读
- 价值: ⭐⭐⭐⭐⭐ — 为多模态学习提供了新的分析工具和实用指导

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Information-Theoretic Decomposition for Multimodal Interaction Learning](../../CVPR2026/multimodal_vlm/information-theoretic_decomposition_for_multimodal_interaction_learning.md)
- [\[ACL 2025\] MIRe: Enhancing Multimodal Queries Representation via Fusion-Free Modality Interaction](../../ACL2025/multimodal_vlm/mire_enhancing_multimodal_queries_representation_via_fusion-free_modality_intera.md)
- [\[ACL 2025\] Towards Storage-Efficient Visual Document Retrieval: An Empirical Study on Reducing Patch-Level Embeddings](../../ACL2025/multimodal_vlm/towards_storage-efficient_visual_document_retrieval_an_empirical_study_on_reduci.md)
- [\[CVPR 2026\] Multi-speaker Attention Alignment for Multimodal Social Interaction](../../CVPR2026/multimodal_vlm/multi-speaker_attention_alignment_for_multimodal_social_interaction.md)
- [\[CVPR 2025\] DynRefer: Delving into Region-level Multimodal Tasks via Dynamic Resolution](../../CVPR2025/multimodal_vlm/dynrefer_delving_into_region-level_multimodal_tasks_via_dynamic_resolution.md)

</div>

<!-- RELATED:END -->
