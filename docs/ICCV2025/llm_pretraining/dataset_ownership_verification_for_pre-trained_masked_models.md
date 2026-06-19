---
title: >-
  [论文解读] Dataset Ownership Verification for Pre-trained Masked Models
description: >-
  [ICCV 2025][预训练][数据集所有权验证] DOV4MM 提出了首个针对掩码预训练模型的数据集所有权验证方法，通过比较"见过"与"未见过"样本在嵌入空间中遮掩信息重构难度的差异，利用配对 t 检验判断黑盒模型是否使用了特定数据集进行预训练，在 10 种掩码图像模型和 4 种掩码语言模型上均实现 p 值远低于 0.05 的准确验证。
tags:
  - "ICCV 2025"
  - "预训练"
  - "数据集所有权验证"
  - "掩码建模"
  - "嵌入重构难度"
  - "假设检验"
  - "自监督学习"
---

# Dataset Ownership Verification for Pre-trained Masked Models

**会议**: ICCV 2025  
**arXiv**: [2507.12022](https://arxiv.org/abs/2507.12022)  
**代码**: [github.com/xieyc99/DOV4MM](https://github.com/xieyc99/DOV4MM)  
**领域**: LLM预训练  
**关键词**: 数据集所有权验证, 掩码建模, 嵌入重构难度, 假设检验, 自监督学习

## 一句话总结

DOV4MM 提出了首个针对掩码预训练模型的数据集所有权验证方法，通过比较"见过"与"未见过"样本在嵌入空间中遮掩信息重构难度的差异，利用配对 t 检验判断黑盒模型是否使用了特定数据集进行预训练，在 10 种掩码图像模型和 4 种掩码语言模型上均实现 p 值远低于 0.05 的准确验证。

## 研究背景与动机

高质量开源数据集是深度学习发展的基石，但面临被非法商业使用的威胁。数据集所有权验证（DOV）旨在检测某个可疑模型是否在特定数据集上训练过。现有方法存在如下局限：

**针对监督模型**：大多数 DOV 方法依赖数据点与决策边界的关系，无法适用于自监督模型；

**依赖后门水印**：在数据中注入水印会降低模型性能，且容易被水印移除技术对抗；

**不适用于掩码模型**：近期针对对比学习模型的 DOV 方法利用嵌入空间中的对比关系差距，但掩码建模（MAE、BEiT 等）与对比学习的代理任务差异显著，表征更难区分。

核心观察：**掩码预训练模型对"见过"的样本在嵌入空间中重构遮掩信息的难度显著低于"未见过"的样本。**

## 方法详解

### 整体框架

DOV4MM 在黑盒场景下操作（defender 只能通过 API 获取特征向量），包含三个关键步骤：

1. 将公开数据集随机划分为训练集 $\mathcal{D}_t$ 和验证集 $\mathcal{D}_v$，用 $\mathcal{D}_t$ 训练一个解码器；
2. 计算可疑模型在三个数据集上的相对嵌入重构难度；
3. 通过单侧配对 t 检验判断模型是否在 defender 的数据集上预训练。

### 关键设计

1. **嵌入重构难度（Embedding Reconstruction Difficulty）**：给定预训练掩码模型 $M$、输入空间掩码 $\boldsymbol{t}$ 和嵌入空间掩码 $\hat{\boldsymbol{t}}$、以及解码器 $M_d$，单个样本 $\boldsymbol{x}$ 的重构难度定义为：

$$R(\boldsymbol{x}, \boldsymbol{t}, \hat{\boldsymbol{t}}, M, M_d) = \frac{\|[M_d(\boldsymbol{e_t}) - \boldsymbol{e}] \odot (\boldsymbol{1} - \hat{\boldsymbol{t}})\|_2^2}{\|\boldsymbol{1} - \hat{\boldsymbol{t}}\|_1}$$

其中 $\boldsymbol{e} = M(\boldsymbol{x})$ 是完整嵌入，$\boldsymbol{e_t} = M(\boldsymbol{x} \odot \boldsymbol{t})$ 是遮掩后的嵌入。仅在遮掩位置处计算重构误差，反映缺失信息的重构难度。

2. **相对嵌入重构难度（Relative Embedding Reconstruction Difficulty）**：为放大见过/未见过样本的差异，引入相对指标。用训练集 $\mathcal{D}_t$ 作为基准，分别计算验证集 $\mathcal{D}_v$ 和私有集 $\mathcal{D}_{pvt}$ 相对于 $\mathcal{D}_t$ 的重构难度差：

$$\Delta\mathcal{R} = \{\overline{R'}_k - \overline{R}_k | k \in [1, K]\}$$

通过 $K=30$ 次随机采样，每次 $N=1024$ 个样本，得到成对差值序列。

3. **假设检验决策**：对 $\Delta\mathcal{R}_{vt}$（验证集相对难度）和 $\Delta\mathcal{R}_{pt}$（私有集相对难度）进行单侧配对 t 检验。原假设 $H_0$：$\Delta\mathcal{R}_{pt}$ 和 $\Delta\mathcal{R}_{vt}$ 的均值差 ≤ 0。若 p 值 < 0.05 则拒绝 $H_0$，判定模型非法使用了该数据集。核心逻辑是：如果模型确实在 $\mathcal{D}_{pub}$ 上训练过，那么 $\mathcal{D}_v$（属于 $\mathcal{D}_{pub}$ 的一部分）的重构难度应比 $\mathcal{D}_{pvt}$（从未见过的数据）更低。

### 训练策略

- 解码器 $M_d$：Transformer 架构（512 维、8 层、16 头），训练 50 epochs，batch size 64，学习率 1e-3；
- 遮掩策略：随机遮掩，遮掩率 75%；
- $\mathcal{D}_t$ 仅需 20,000 样本（仅 ImageNet-1K 的 ~3%），即可实现准确验证。

## 实验关键数据

### 主实验

**ImageNet-1K 子集上的分类验证能力**：

| 数据集 | 方法 | 灵敏度 | 特异度 | AUROC |
|--------|------|--------|--------|-------|
| ImageNet-50 | DI4SSL | 0.00 | 1.00 | 0.50 |
| ImageNet-50 | CTRL | 1.00 | 0.00 | 0.50 |
| ImageNet-50 | PartCrop | 0.00 | 0.22 | 0.39 |
| ImageNet-50 | **DOV4MM** | **1.00** | **1.00** | **1.00** |
| ImageNet-100 | **DOV4MM** | **1.00** | **1.00** | **1.00** |

**ImageNet-1K 上的 p 值结果（10 种 MIM 方法）**：

| 模型 | MIM方法 | IN-1K (非法) | Food101 (合法) | COCO (合法) | Places365 (合法) |
|------|---------|-------------|----------------|-------------|-----------------|
| ViT-B/16 | MAE | $10^{-5}$ ✓ | 0.99 ✓ | 0.98 ✓ | 0.99 ✓ |
| ViT-B/16 | BEiT v2 | $10^{-5}$ ✓ | 0.99 ✓ | 0.99 ✓ | 0.99 ✓ |
| ViT-L/16 | MAE | $10^{-6}$ ✓ | 0.99 ✓ | 0.99 ✓ | 0.99 ✓ |
| Swin-B | SimMIM | 0.03 ✓ | 0.99 ✓ | 0.98 ✓ | 0.98 ✓ |

所有 10 种 MIM 方法 + 4 种架构均成功验证，仅使用 3% 的 ImageNet-1K 数据。

### 消融实验

| 配置 | MAE p值 | CAE p值 | iBOT p值 | 说明 |
|------|---------|---------|----------|------|
| 解码器维度 128 | $10^{-5}$ | $10^{-3}$ | $10^{-3}$ | 均有效 |
| 解码器维度 1024 | $10^{-7}$ | 0.01 | $10^{-3}$ | 更大不一定更好 |
| 解码器层数 4 | $10^{-5}$ | $10^{-3}$ | $10^{-3}$ | 稳定 |
| 解码器层数 12 | $10^{-6}$ | 0.01 | $10^{-3}$ | 稳定 |
| 训练集大小 10k | $10^{-4}$ | 0.02 | 0.01 | 较少数据也有效 |
| 训练集大小 50k | $10^{-6}$ | $10^{-3}$ | $10^{-3}$ | 更多更好 |

### 关键发现

- DOV4MM 对解码器架构（维度、层数、头数）鲁棒，各种配置下 p 值均远低于 0.05；
- 仅需 3% 数据（~20k 样本）即可准确验证百万级数据集的所有权；
- 在 WikiText-103 上的 4 种掩码语言模型（BERT 等）也同样有效，验证了跨模态通用性。

## 亮点与洞察

1. **首创性**：第一个专门针对掩码预训练模型的数据集所有权验证方法，填补了重要空白；
2. **无需水印**：不修改原始数据集分布，避免了水印注入带来的性能降低和水印被移除的风险；
3. **极低数据需求**：仅需 3% 数据即可准确验证，相比 DI4SSL 需要推断整个数据集，计算成本大幅降低；
4. **统计严谨**：基于配对 t 检验的假设检验框架提供了严格的统计保证，而非简单的阈值判断；
5. **跨模态通用**：同时适用于视觉掩码模型（MAE、BEiT 等）和语言掩码模型（BERT 等）。

## 局限与展望

- 需要一个与可疑模型数据域无关的私有数据集 $\mathcal{D}_{pvt}$，若私有数据与公开数据过于相似，可能降低检测灵敏度；
- Swin-B/L 的 p 值（0.03）接近 0.05 阈值，对某些架构的鲁棒性有待提升；
- 黑盒场景假设要求可通过 API 获取嵌入向量（EaaS），如果仅提供分类结果则无法使用；
- 暂未测试在模型被微调（下游任务 fine-tuning）后是否仍能有效验证。

## 相关工作与启发

- 与数据集推断（DI4SSL）不同：DI4SSL 需推断整个数据集的似然分布，计算成本高；DOV4MM 仅需少量样本计算重构难度差异；
- 与成员推断（PartCrop）不同：PartCrop 直接使用高维表征做成员判断，包含大量冗余信息；DOV4MM 提取了最关键的"相对重构难度"指标；
- 核心思路可推广到其他生成式预训练范式（如扩散模型的去噪难度差异）。

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 首创性强，相对嵌入重构难度的概念直觉清晰、定义严谨
- **实验充分度**: ⭐⭐⭐⭐ 10种MIM+4种架构+语言模型跨模态验证+全面消融
- **写作质量**: ⭐⭐⭐⭐ 数学定义严谨，方法流程清晰
- **价值**: ⭐⭐⭐⭐ 数据安全领域的重要贡献，实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] LEANCODE: Understanding Models Better for Code Simplification of Pre-trained Large Language Models](../../ACL2025/llm_pretraining/leancode_understanding_models_better_for_code_simplification_of_pre-trained_larg.md)
- [\[ACL 2025\] Chinese Grammatical Error Correction With Pre-trained Models and Linguistic Clues](../../ACL2025/llm_pretraining/chinese_grammatical_error_correction_with_pre-trained_models_and_linguistic_clue.md)
- [\[NeurIPS 2025\] How Does Sequence Modeling Architecture Influence Base Capabilities of Pre-trained Language Models?](../../NeurIPS2025/llm_pretraining/how_does_sequence_modeling_architecture_influence_base_capabilities_of_pre-train.md)
- [\[ACL 2026\] SCRIPT: A Subcharacter Compositional Representation Injection Module for Korean Pre-Trained Language Models](../../ACL2026/llm_pretraining/script_a_subcharacter_compositional_representation_injection_module_for_korean_p.md)
- [\[AAAI 2026\] PrefixGPT: Prefix Adder Optimization by a Generative Pre-trained Transformer](../../AAAI2026/llm_pretraining/prefixgpt_prefix_adder_optimization_by_a_generative_pre-trained_transformer.md)

</div>

<!-- RELATED:END -->
