---
title: >-
  [论文解读] Perturb Your Data: Paraphrase-Guided Training Data Watermarking
description: >-
  [AAAI2026][LLM安全][训练数据水印] 提出SPECTRA——一种基于paraphrase采样的训练数据水印方法，通过LLM生成改写文本并利用Min-K%++评分选择与原文分数接近的paraphrase作为水印，在数据仅占训练语料0.001%的情况下，member与non-member的p-value差距稳定超过9个数量级。
tags:
  - "AAAI2026"
  - "LLM安全"
  - "训练数据水印"
  - "成员推断攻击"
  - "数据版权保护"
  - "LLM"
  - "paraphrasing"
  - "Min-K%++"
---

# Perturb Your Data: Paraphrase-Guided Training Data Watermarking

**会议**: AAAI2026  
**arXiv**: [2512.17075](https://arxiv.org/abs/2512.17075)  
**作者**: Pranav Shetty, Mirazul Haque, Petr Babkin, Zhiqiang Ma, Xiaomo Liu, Manuela Veloso (JPMorgan AI Research)  
**代码**: 未公开  
**领域**: AI安全  
**关键词**: 训练数据水印, 成员推断攻击, 数据版权保护, LLM, paraphrasing, Min-K%++  

## 一句话总结

提出SPECTRA——一种基于paraphrase采样的训练数据水印方法，通过LLM生成改写文本并利用Min-K%++评分选择与原文分数接近的paraphrase作为水印，在数据仅占训练语料0.001%的情况下，member与non-member的p-value差距稳定超过9个数量级。

## 背景与动机

### 数据版权危机

当代LLM依赖从互联网大规模抓取的文本进行预训练，这些语料常包含未经授权的受版权保护内容。近年来多起诉讼（如纽约时报诉OpenAI）聚焦于未授权使用付费内容训练模型的合法性问题。随着越来越多内容通过ChatGPT等LLM被消费，如果不确保内容创作者获得合理回报，LLM数据收集实践可能导致"extractive dead end"——创作者失去生产新内容的动力。

### 现有方法的不足

Membership Inference Attack (MIA) 是检测训练数据的主流方法，其核心思想是训练会改变模型对训练样本的输出概率。然而，MIA对member和non-member数据之间的微小分布偏移极为敏感，当两者来自同质分布时（去除时间伪影后），所有MIA方法退化至随机水平。STAMP方法通过KGW水印方案对文本进行多次重写来解决这个问题，但需要存储大量私有改写版本，且需要访问LLM的解码层（GPU资源昂贵），其member与non-member的p-value差距最多仅3个数量级，对于具有重大法律后果的场景不够充分。

### 核心洞察

现有MIA评分旨在度量模型训练前后loss surface的变化，但实践中很少能获取训练前的模型状态。SPECTRA的关键insight是：使用一个已知未在目标数据上训练的scoring model作为"训练前状态"的代理，从而可靠地检测因训练引起的变化。同时，通过精心设计的paraphrase采样策略，确保水印本身不引入任何假阳性信号。

## 核心问题

如何设计水印过程 $W$ 和统计检验 $T$，使得：(1) 若目标模型 $M$ 确实在水印数据 $D'$ 上训练过，$T$ 能以高置信度检测出来；(2) 若 $M$ 未在 $D'$ 上训练，$T$ 不会产生假阳性。要求整个过程仅需grey-box访问（获取token log probabilities），且水印数据占训练语料比例可低至0.001%。

## 方法详解

### 整体框架

SPECTRA分为两个阶段：**水印阶段**（发布前）和**验证阶段**（检测时）。

**水印阶段**：对数据集 $D$ 中每篇文档，用LLM生成多个paraphrase，用scoring model计算Min-K%++评分，通过精心设计的采样策略选择一个评分接近原文的paraphrase作为水印版本，构成 $D' = W(D)$。创作者保留原始 $D$，仅发布 $D'$。

**验证阶段**：分别在scoring model $M_S$ 和目标模型 $M_T$ 上计算原始文档与水印文档的Min-K%++评分比值，通过paired t-test检测 $M_T$ 是否在 $D'$ 上训练过。

### 关键设计1：Min-K%++ 评分

给定自回归模型 $M$ 和token序列 $x = (x_1, \ldots, x_n)$，定义归一化token log probability：

$$z(x_t; M) = \frac{\log P(x_t \mid x_{<t}; M) - \mu_{x_{<t}}}{\sigma_{x_{<t}}}$$

其中 $\mu_{x_{<t}} = \mathbb{E}_{z \sim P(\cdot|x_{<t}; M)}[\log P(z|x_{<t}; M)]$ 为期望log probability，$\sigma_{x_{<t}}$ 为对应标准差。Min-K%++评分取K%个最低值（最高surprisal）的平均：

$$f_{\text{Min-K\%++}}(x; M) = \frac{1}{|\text{min-K}(x)|} \sum_{x_t \in \text{min-K}(x)} z(x_t; M)$$

理论上，Min-K%++与loss landscape的Hessian trace的负值对应。训练通过最大似然直接降低训练样本处的曲率，导致Min-K%++评分上升。

### 关键设计2：Paraphrase采样策略

核心目标是选择Min-K%++评分接近原文的paraphrase，避免系统性偏移导致假阳性。对每个文档 $i$，计算paraphrase评分与原文评分的比值 $r_{ij} = s_{ij} / s_i^{(0)}$，按上下两侧分组：

- 权重定义：$w_{ij} = \exp(-\alpha |r_{ij} - 1|)$，$\alpha = 100$ 控制分布锐度
- 全局平衡：统计所有paraphrase均高于原文（集合 $\mathcal{B}$）和均低于原文（集合 $\mathcal{A}$）的文档数量，以反比概率 $\pi_+ = |\mathcal{A}| / (|\mathcal{A}| + |\mathcal{B}|)$ 选择上下侧
- 从对应侧按归一化权重进行categorical sampling

这种设计确保水印数据集的评分分布与原始数据保持一致，仅当模型确实训练过后才出现可检测的信号。

### 关键设计3：统计检验

定义score ratio：$r(x, x'; M) = f_{\text{Min-K\%++}}(x'; M) / f_{\text{Min-K\%++}}(x; M)$

零假设 $H_0$：$\mathbb{E}[r(x, x'; M_T)] = \mathbb{E}[r(x, x'; M_S)]$（$M_T$ 未在 $D'$ 上训练）

备择假设 $H_1$：$\mathbb{E}[r(x, x'; M_T)] < \mathbb{E}[r(x, x'; M_S)]$（$M_T$ 在 $D'$ 上训练过）

通过单侧paired t-test计算p-value，低p-value表示拒绝 $H_0$，即 $M_T$ 使用了水印数据。

## 实验关键数据

### 实验设置

- **Paraphraser**：Llama 3.1-405b，每篇生成10个paraphrase
- **Scoring model**：Pythia 2.8b-deduped（Pile数据集）/ OLMo-1b（PeS2o数据集）
- **Target model**：Pythia 410m + 5 billion tokens continued pretraining
- **水印数据**：每个数据集500个样本（每样本≤512 tokens），占训练语料<0.001%

### 主实验：p-value对比

| 方法 | PubMed M | PubMed NM | Wiki M | Wiki NM | HN M | HN NM | PeS2o M | PeS2o NM |
|------|----------|-----------|--------|---------|------|-------|---------|----------|
| LLM-DI | 0.06 | 0.48 | 0.02 | 0.44 | 0.49 | 0.35 | 0.02 | 0.17 |
| STAMP | 0.01 | 0.48 | 0.17 | 0.03 | 7E-4 | 0.15 | 0.15 | 0.46 |
| Maximum | 0.03 | 1.00 | 1.00 | 1.00 | 3E-6 | 1.00 | 0.95 | 1.00 |
| Random | 1E-7 | 8E-4 | 5E-9 | 2E-5 | 4E-27 | 0.10 | 1E-3 | 0.11 |
| **SPECTRA** | **1E-17** | 0.02 | **4E-19** | 0.02 | **3E-60** | 0.59 | **2E-12** | 3E-3 |

在严格阈值 $p < 10^{-4}$ 下，SPECTRA是唯一在所有4个数据集上都能正确检测membership且不产生假阳性的方法。Member与non-member的p-value差距稳定在 $>10^9$ 量级。

### MIA基线性能（500M vs 5B tokens训练）

| 方法 | Wiki 500M | HN 500M | PubMed 500M | Wiki 5B | HN 5B | PubMed 5B |
|------|-----------|---------|-------------|---------|-------|-----------|
| Loss | 0.71 | 0.73 | 0.63 | 0.55 | 0.54 | 0.52 |
| Min-K% | 0.76 | 0.79 | 0.65 | 0.56 | 0.55 | 0.52 |
| Min-K%++ | 0.85 | 0.84 | 0.72 | 0.55 | 0.55 | 0.51 |

Min-K%++在500M tokens训练时表现最优（AUC 0.72-0.85），但在5B tokens规模下退化至随机水平（AUC ≈0.5），这正是SPECTRA要解决的问题。

### Paraphrase质量

| 数据集 | PubMed | Wiki | HN | PeS2o |
|--------|--------|------|----|-------|
| P-SP | 0.88 | 0.93 | 0.76 | 0.93 |

人类评估（54篇文档，4位评估者）显示含义保留、结构保留和作者语调保留三个维度的平均分均超过4（满分5），结构保留在对话体文本（如Hackernews）上略低。

### Scoring Model鲁棒性

| 评分模型 | Spearman $\rho$ | Kendall $\tau$ |
|---------|----------------|----------------|
| OLMo-7b | 0.826 | 0.639 |
| Pythia-2.8b | 0.824 | 0.635 |
| Pythia-160m | 0.699 | 0.514 |
| Pythia-6.9b | 0.818 | 0.631 |

参数量≥2.8B的模型间排序相关性 $\rho > 0.8$，表明SPECTRA对scoring model选择具有鲁棒性。

## 亮点

- **极强的统计信号**：member与non-member的p-value差距稳定超过9个数量级，远超STAMP（3个数量级）和其他基线
- **无需解码层访问**：仅需grey-box（token log probabilities），不像STAMP需要修改LLM解码过程
- **无需non-member数据集**：直接比较scoring model与target model的评分比值，无需同域held-out数据
- **精巧的采样策略**：通过全局side-balance和指数衰减权重，既保持评分分布不变又保留训练后的可检测性
- **跨架构有效**：PeS2o实验中scoring model（OLMo-1b）与target model（Pythia 410m）架构不同仍有效

## 局限与展望

- **仅验证continued pretraining**：因计算资源限制，未在from-scratch训练场景下验证，实际部署中需要更大规模的验证
- **需要grey-box访问**：闭源商业模型通常不提供token log probabilities，需要第三方仲裁机制
- **仅适用于未发布数据**：水印必须在发布前嵌入，已发布的内容无法追溯保护
- **结构化/对话体文本效果较弱**：Hackernews的P-SP仅0.76，paraphraser对非标准文体的保真度下降
- **依赖强大的paraphraser**：使用Llama 3.1-405b生成paraphrase，成本较高；较小模型的效果未被充分探索
- **500样本的实用性**：需要至少100-150个样本才能达到显著水平，对小规模内容创作者可能不够友好

## 与相关工作的对比

- **vs STAMP**：STAMP需要KGW水印方案修改LLM解码层、存储大量私有改写版本，p-value差距仅3个数量级；SPECTRA无需解码层访问，差距超过9个数量级
- **vs MIA（Min-K%++等）**：传统MIA在大规模训练（5B tokens）下退化至随机水平，依赖non-member数据；SPECTRA通过主动水印创造可靠的检测信号
- **vs LLM-DI**：Dataset Inference在严格阈值下无法可靠检测membership；SPECTRA在所有数据集上都达到统计显著
- **vs Maximum采样**：贪心选择最高Min-K%++评分的paraphrase会导致预训练偏移过大，训练后信号不可区分；SPECTRA的平衡采样策略避免了这一问题
- **vs 后门水印**（如Winter Soldier）：后门方法插入特殊token影响文本可读性；SPECTRA生成自然的paraphrase，语义质量可验证

## 启发与关联

- **"代理模型"思路**具有广泛适用性：在无法获取模型训练前状态时，用同族或相似的预训练模型作为基准线进行差异检测，这一思路可以推广到其他模型审计场景
- **评分分布保持**的设计哲学值得借鉴：水印的关键不是最大化当前信号，而是保持训练前分布不变、仅让训练导致可检测的偏移——类似于密码学中"语义安全"的思想
- 该方法可能与**data attribution**、**model provenance**等研究方向结合，构建更完整的AI数据治理工具链
- 100-150个样本的最低需求暗示可以开发面向**小规模创作者联盟**的协作水印框架

## 评分

- 新颖性: ⭐⭐⭐⭐ — 将paraphrase采样与Min-K%++评分巧妙结合用于数据水印，采样策略的side-balance设计有独创性，但核心组件（Min-K%++、paraphrasing）均为已有技术
- 实验充分度: ⭐⭐⭐⭐ — 4个数据集、多个基线、消融实验（样本数、scoring model选择）和人类评估较完整，但仅在continued pretraining上验证
- 写作质量: ⭐⭐⭐⭐ — 问题建模清晰，方法描述直观，但部分符号定义分散，需要来回查阅
- 价值: ⭐⭐⭐⭐ — 解决了LLM训练数据版权保护的实际痛点，9个数量级的p-value差距为法律场景提供了强有力的统计证据

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Exploring Cross-Client Memorization of Training Data in Large Language Models for Federated Learning](../../ACL2026/llm_safety/exploring_cross-client_memorization_of_training_data_in_large_language_models_fo.md)
- [\[ACL 2025\] Robust Data Watermarking in Language Models by Injecting Fictitious Knowledge](../../ACL2025/llm_safety/robust_data_watermarking_in_language_models_by_injecting_fictitious_knowledge.md)
- [\[NeurIPS 2025\] Virus Infection Attack on LLMs: Your Poisoning Can Spread "VIA" Synthetic Data](../../NeurIPS2025/llm_safety/virus_infection_attack_on_llms_your_poisoning_can_spread_via_synthetic_data.md)
- [\[ACL 2026\] DualGuard: Dual-stream Large Language Model Watermarking Defense against Paraphrase and Spoofing Attack](../../ACL2026/llm_safety/dualguard_dual-stream_large_language_model_watermarking_defense_against_paraphra.md)
- [\[AAAI 2026\] AgentSense: Virtual Sensor Data Generation Using LLM Agents in Simulated Home Environments](agentsense_virtual_sensor_data_generation_using_llm_agents_i.md)

</div>

<!-- RELATED:END -->
