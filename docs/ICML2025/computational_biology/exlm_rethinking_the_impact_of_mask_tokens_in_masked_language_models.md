---
title: >-
  [论文解读] ExLM: Rethinking the Impact of [MASK] Tokens in Masked Language Models
description: >-
  [ICML 2025][计算生物][掩码语言模型] 本文首次系统分析了 MLM 中 [MASK] 对性能的影响，发现**语义损坏（corrupted semantics）**比**非真实token（unreal tokens）**的负面作用更大，据此提出 ExLM：通过将每个 [MASK] 扩展为多个隐状态并用转移矩阵建模依赖关系，有效缓解语义多模态性问题，在文本和分子建模任务上均取得显著提升。
tags:
  - "ICML 2025"
  - "计算生物"
  - "掩码语言模型"
  - "语义损坏"
  - "多模态性"
  - "状态扩展"
  - "DAG对齐"
---

# ExLM: Rethinking the Impact of [MASK] Tokens in Masked Language Models

**会议**: ICML 2025  
**arXiv**: [2501.13397](https://arxiv.org/abs/2501.13397)  
**代码**: 无（基于 Fairseq 实现）  
**领域**: 计算生物  
**关键词**: 掩码语言模型, 语义损坏, 多模态性, 状态扩展, DAG对齐

## 一句话总结

本文首次系统分析了 MLM 中 [MASK] 对性能的影响，发现**语义损坏（corrupted semantics）**比**非真实token（unreal tokens）**的负面作用更大，据此提出 ExLM：通过将每个 [MASK] 扩展为多个隐状态并用转移矩阵建模依赖关系，有效缓解语义多模态性问题，在文本和分子建模任务上均取得显著提升。

## 研究背景与动机

MLM（如 BERT）的预训练通过将输入 token 替换为 [MASK] 来学习上下文表征，但引入 [MASK] 会带来两个问题：

**非真实 token 问题**：上下文中出现大量预训练独有的 [MASK] 符号，与真实文本不一致

**语义损坏问题**：被mask的token导致上下文语义不完整，可能产生多义歧义

以往工作（ELECTRA、MAE-LM 等）主要关注第一个问题，但对语义损坏的影响缺乏系统研究。更关键的是，这两个因素与 mask ratio 耦合在一起，难以单独分析各自的影响。

本文的核心动机是：**解耦这两个因素，定量比较它们对 MLM 性能的影响，并基于发现设计更好的预训练方法**。

## 方法详解

### 整体框架

ExLM 的核心思路分为两步：

**第一步：分析实验（Repeated MLM）**——设计巧妙的解耦实验来量化语义损坏的影响。具体做法是在输入 MLM 之前，将每个 token 重复 k 次，然后按比例 p 进行 mask。此时：
- 非真实 token 的比例仍为 p
- 语义损坏的比例变为 p^k（因为只有当某个 token 的所有 k 个副本都被 mask 时才算语义损坏）

通过固定 p 变化 k，可以保持非真实 token 比例不变而改变语义损坏程度，反之亦然。实验结果（MNLI 任务）明确表明：
- 语义损坏比例不变时，mask ratio 从低到高，性能仅从 83.6 轻微下降到 82.8
- mask ratio 固定时，语义损坏比例增加，性能从 82.8 显著下降到 79.6

结论：**语义损坏对 MLM 性能的影响远大于非真实 token**。

**第二步：ExLM 方法设计**——基于上述发现，针对语义损坏导致的多模态性（multimodality）问题，设计增强上下文的 MLM。

### 关键设计

ExLM 包含两个核心组件：

#### 1. 状态扩展（States Expansion）

对输入中的每个 [MASK] token，将其 embedding 复制 k 份，形成扩展输入序列：

$$\mathbf{X'} = [\mathbf{e}_{x_1}, \mathbf{e}_{x_2}, \ldots, \mathbf{e}_{[\text{MASK}]}^{(1)}, \ldots, \mathbf{e}_{[\text{MASK}]}^{(k)}, \ldots, \mathbf{e}_{x_n}]$$

扩展后送入 Transformer Encoder 得到对应的隐状态。通过扩展状态，模型拥有更大的语义空间来捕获不同的可能语义，有效应对 intra-token multimodality（单个 token 的多义性）。

#### 2. 2D RoPE 位置编码

为区分同一 [MASK] 的不同克隆，引入二维旋转位置编码。原始位置 i 处的 [MASK] 的 k 个克隆被分配位置 $(i,1), (i,2), \ldots, (i,k)$，非 mask token 保持 $(j,0)$。第一维编码序列位置，第二维区分克隆索引。

#### 3. 转移矩阵（Transition Matrix）建模依赖

扩展状态之间的语义依赖建模为有向无环图（DAG）。具体通过 attention-like 计算得到转移矩阵 E：

$$\mathbf{E} = \text{softmax}\left(\frac{\mathbf{QK}^{\top}}{\sqrt{d}} + \mathbf{M}\right)$$

其中 Q = HW_Q, K = HW_K, M 是上三角 mask 矩阵确保 DAG 结构。每个状态还通过预测头计算 token 概率分布：

$$\mathbf{P} = \text{softmax}(\mathbf{H}\mathbf{W}_P^{\top})$$

转移矩阵有效捕获 inter-token multimodality（不同 mask token 之间的语义依赖），例如当第一个 [MASK] 是 "terrible" 时，第二个 [MASK] 更可能是 "sorry"。

### 损失函数 / 训练策略

#### States Alignment（状态对齐）

由于扩展后的隐状态数量多于目标 token 数量，需要确定状态与目标 token 的对齐关系。将此建模为 DAG 解码问题：

$$\mathcal{L}_{SA} = -\log P_{\theta}(\mathbf{Y}|\mathbf{X'}) = -\log \sum_{\mathbf{A} \in \Gamma} P_{\theta}(\mathbf{Y}, \mathbf{A}|\mathbf{X'})$$

使用动态规划高效求解，定义 $f_{i,u}$ 为所有到达状态 u 且已生成前 i 个目标 token 的路径累积概率：

$$f_{i,u} = \sum_{v < u} f_{i-1,v} \times \mathbf{E}_{v,u} \times \mathbf{P}_u(y_i)$$

最终目标为 $\mathcal{L}_{SA} = -\log f_{M,L}$。时间复杂度为 $O(M \times L^2)$，通过 CUDA 并行优化可进一步降至 $O(M)$。

预训练数据：文本使用 English Wikipedia + BookCorpus；分子使用 1900 万 SMILES。模型架构与 BERT-base 一致（12层，768维，128M 参数），使用 k=4 作为默认扩展数。

## 实验关键数据

### 主实验

**文本任务（GLUE + SQuAD 2.0, dev set）：**

| 模型 | MNLI-m/mm | QQP | QNLI | SST-2 | CoLA | RTE | MRPC | STS-B | MEAN | SQuAD EM | SQuAD F1 |
|------|-----------|-----|------|-------|------|-----|------|-------|------|----------|----------|
| BERT | 84.5/- | 91.3 | 91.7 | 93.2 | 58.9 | 68.6 | 87.3 | 89.5 | 83.1 | 73.7 | 76.3 |
| RoBERTa* | 85.9/85.8 | 91.6 | 92.3 | 93.7 | 64.3 | 75.5 | 88.7 | 89.5 | 85.2 | 78.3 | 81.5 |
| TUPE | 86.2/86.2 | 91.3 | 92.2 | 93.3 | 63.6 | 73.6 | 89.9 | 89.2 | 84.9 | - | - |
| **ExLM** | **86.9/86.7** | **92.0** | **93.1** | **93.9** | **64.6** | **78.8** | **89.6** | **90.5** | **86.2** | **82.0** | **84.6** |
| ExLM_LARGE | 87.8/87.5 | 92.2 | 93.8 | 94.5 | 65.3 | 79.1 | 90.4 | 91.2 | 86.9 | 82.6 | 85.0 |

ExLM 在 GLUE 8 项中 7 项最优，MEAN 从 RoBERTa 的 85.2 提升到 86.2（+1.0），SQuAD F1 从 81.5 提升到 84.6（+3.1）。

**分子属性预测（MoleculeNet, ROC-AUC）：**

| 模型 | BACE | BBBP | Tox21 | ToxCast | SIDER | ClinTox | MUV | Avg |
|------|------|------|-------|---------|-------|---------|-----|-----|
| D-MPNN | 80.9 | 71.0 | 75.9 | 57.0 | 78.6 | 90.6 | 65.5 | 74.2 |
| SMILES-BERT* | 77.8 | 68.6 | 75.1 | 61.2 | 75.1 | 89.8 | 64.9 | 73.2 |
| GraphMVP | 81.2 | 72.4 | 75.9 | 63.9 | 77.7 | 79.1 | 63.1 | 73.3 |
| **ExLM** | 79.6 | **72.8** | **78.2** | **64.5** | **78.8** | **91.6** | **66.9** | **76.1** |

ExLM 在 7 个数据集中 5 个最优，平均 76.1 显著超越同架构 SMILES-BERT（73.2，+2.9）。

### 消融实验

| 配置 | MNLI | QNLI | QQP | RTE | Avg | 说明 |
|------|------|------|-----|-----|-----|------|
| Vanilla MLM | 83.6 | 90.0 | 90.3 | 54.7 | 79.6 | 基线 |
| Vanilla MLM++ | 84.4 | 91.2 | 90.6 | 56.3 | 80.7 | 等训练成本 MLM |
| ExLM w/o Transitions | 83.8 | 90.9 | 91.1 | 55.6 | 80.4 | 去掉转移矩阵 |
| ExLM w/o 2D RoPE | 84.6 | 91.1 | 91.3 | 56.7 | 80.9 | 去掉 2D 位置编码 |
| ExLM w/ Sparse DAG | 84.4 | 91.2 | 91.3 | 56.9 | 81.0 | 稀疏 DAG |
| **ExLM** | **85.1** | **91.4** | **91.3** | **57.6** | **81.4** | 完整模型 |

### 关键发现

1. **转移矩阵 > 2D RoPE**：去掉转移矩阵的影响（-1.0 avg）大于去掉 2D RoPE（-0.5 avg），说明状态间依赖建模是核心
2. **效率合理**：ExLM (k=4) 训练时间约为 MLM 的 1.9 倍（104.2h vs 54.7h, A100），但等成本的 Vanilla MLM++ 仍低于 ExLM
3. **k=4 最优**：k 从 2 到 4 性能持续提升，k=8 时因输入过长导致轻微下降
4. **高 mask ratio 鲁棒**：ExLM 在高 mask ratio 下性能下降明显小于 MLM，验证了增强语义建模的效果
5. **熵分析**：ExLM 的预测熵显著低于 MLM，说明有效缓解了语义多模态性

## 亮点与洞察

- **精巧的解耦实验设计**：Repeated MLM 通过 token 重复巧妙地将 unreal tokens 和 corrupted semantics 解耦，是本文最具创新性的分析工具
- **从分析到方法的完整闭环**：先通过实验发现问题（语义损坏是主因），再针对性设计解决方案（状态扩展+依赖建模），逻辑链条清晰
- **跨领域验证**：在文本（GLUE/SQuAD）和分子（SMILES/MoleculeNet）两个差异很大的领域都验证了有效性，说明方法的通用性
- **Case study 直观有效**：DAG 可视化清楚展示了 ExLM 如何用不同状态捕获不同语义可能性及其依赖关系

## 局限与展望

1. **训练成本增加**：扩展状态导致序列变长，k=4 时训练时间接近 2 倍，难以直接 scale 到更大模型
2. **仅验证 BERT-scale**：未在更大规模模型（如 BERT-Large 以上）或更多预训练数据上验证
3. **推理时如何使用**？论文主要关注预训练阶段，fine-tuning 时扩展状态的处理方式未充分讨论
4. **仅适用于 Encoder MLM**：未探索在 decoder-only 或 encoder-decoder 架构中的适用性
5. **DAG 解码假设较强**：强制要求有向无环图结构，可能限制了对某些循环依赖的建模

## 相关工作与启发

- 与 ELECTRA（Clark, 2020）互补：ELECTRA 解决 unreal tokens 问题，ExLM 解决 corrupted semantics 问题，二者有潜在结合空间
- DA-Transformer（Huang et al., 2022）的 DAG + DP 范式从 NAT 迁移到 MLM 预训练，展示了跨任务方法迁移的可能
- 对 MAE 类视觉预训练也有启发：高 mask ratio 下的语义损坏问题同样存在，ExLM 的思路或可用于改进 MAE

## 评分

- 新颖性: ⭐⭐⭐⭐ — 解耦分析实验设计非常巧妙，方法整体是已有组件的新颖组合
- 实验充分度: ⭐⭐⭐⭐⭐ — 文本+分子双领域，消融/可视化/效率分析齐全
- 写作质量: ⭐⭐⭐⭐⭐ — 从分析到方法逻辑清晰，图表质量高
- 价值: ⭐⭐⭐⭐ — 对 MLM 预训练有深刻洞察，但受限于 encoder-only 范式的当前热度

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Steering Protein Language Models](steering_protein_language_models.md)
- [\[ICLR 2026\] How to Make the Most of Your Masked Language Model for Protein Engineering](../../ICLR2026/computational_biology/how_to_make_the_most_of_your_masked_language_model_for_protein_engineering.md)
- [\[NeurIPS 2025\] KLASS: KL-Guided Fast Inference in Masked Diffusion Models](../../NeurIPS2025/computational_biology/klass_kl-guided_fast_inference_in_masked_diffusion_models.md)
- [\[ICML 2025\] CFP-Gen: Combinatorial Functional Protein Generation via Diffusion Language Models](cfp-gen_combinatorial_functional_protein_generation_via_diffusion_language_model.md)
- [\[NeurIPS 2025\] Understanding and Enhancing Mask-Based Pretraining towards Universal Representations](../../NeurIPS2025/computational_biology/understanding_and_enhancing_mask-based_pretraining_towards_universal_representat.md)

</div>

<!-- RELATED:END -->
