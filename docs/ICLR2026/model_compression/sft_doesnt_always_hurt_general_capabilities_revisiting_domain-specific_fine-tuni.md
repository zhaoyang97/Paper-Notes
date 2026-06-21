---
title: >-
  [论文解读] SFT Doesn't Always Hurt General Capabilities: Revisiting Domain-Specific Fine-Tuning in LLMs
description: >-
  [ICLR2026][模型压缩][SFT] 本文系统性地重新审视了领域特定SFT对LLM通用能力的影响，发现**使用较小学习率即可大幅缓解通用能力退化**，并提出Token-Adaptive Loss Reweighting (TALR)方法通过自适应下调低概率token的损失权重进一步优化领域适配与通用能力之间的权衡。
tags:
  - "ICLR2026"
  - "模型压缩"
  - "SFT"
  - "领域微调"
  - "通用能力退化"
  - "学习率"
  - "token自适应重加权"
  - "持续学习"
  - "LLM"
---

# SFT Doesn't Always Hurt General Capabilities: Revisiting Domain-Specific Fine-Tuning in LLMs

**会议**: ICLR2026  
**arXiv**: [2509.20758](https://arxiv.org/abs/2509.20758)  
**代码**: 未开源  
**领域**: 模型压缩  
**关键词**: SFT, 领域微调, 通用能力退化, 学习率, token自适应重加权, 持续学习, LLM

## 一句话总结

本文系统性地重新审视了领域特定SFT对LLM通用能力的影响，发现**使用较小学习率即可大幅缓解通用能力退化**，并提出Token-Adaptive Loss Reweighting (TALR)方法通过自适应下调低概率token的损失权重进一步优化领域适配与通用能力之间的权衡。

## 背景与动机

1. **领域SFT是标准范式**：大语言模型在通用任务上表现优异，但在医疗、电商等专业领域仍需通过SFT注入领域知识以提升性能。
2. **通用能力退化被广泛报道**：多项研究指出在领域数据上SFT会严重损害数学推理、代码生成、指令遵循等通用能力，引发对SFT实用性的质疑。
3. **先前研究使用的学习率偏大**：已有工作多采用5e-6或2e-5等较大学习率，可能是退化现象被夸大的一个原因。
4. **data-oblivious设定更实际**：实际场景中通常无法获取预训练数据，因此不依赖原始数据的缓解策略更具价值。
5. **token层面分析缺失**：此前研究主要在样本或基准层面分析退化，缺少对训练数据中单个token学习难度的精细理解。
6. **缺乏理论支撑**：对于学习率为何影响通用能力退化程度，尚缺从信息论角度的形式化分析。

## 方法详解

### 整体框架

本文先用一个反直觉的经验发现立论——把领域SFT的学习率调小（如1e-6），通用能力退化能被大幅缓解而领域性能几乎不掉；再用信息论给这个现象一个形式化解释，把退化的根源定位到训练数据里少量"hard token"的梯度贡献；最后顺着这个根源提出 TALR（Token-Adaptive Loss Reweighting），在 data-oblivious（拿不到预训练数据）的现实设定下，用自适应权重压低 hard token 的损失，进一步改善领域适配与通用能力的权衡。

### 关键设计

**1. 小学习率即良好权衡：先证伪"大学习率更好"的惯例**

作者在 MedCalc（医疗计算）和 ESCI（电商分类）两个数据集上系统扫描学习率，得到与传统深度学习经验相反的结论：用较小学习率（如 1e-6）就能显著减少通用能力退化，而领域性能与大学习率几乎相当——也就是说先前文献里"SFT 严重损害通用能力"的结论，部分源于沿用了 5e-6、2e-5 这类偏大的学习率。进一步发现训练目标的构成也有影响：当监督信号仅含标签、不含 CoT 推理链时，能达到 Pareto 最优的学习率区间更宽，5e-6 也能表现良好。这两点共同把"学习率"从一个被忽视的超参提到了缓解退化的核心位置。

**2. 信息论视角的退化上界：把通用能力变化写成可量化的编码长度**

为了解释为什么小学习率能保住通用能力，作者把 LLM 看作一个数据压缩器，借助 token tree 与算术编码框架做形式化分析。核心结论是：模型从参数 $\theta_1$ 更新到 $\theta_2$ 时，对通用数据的预期编码长度变化恰等于两者 KL 散度之差，因而通用能力的损失可以被精确量化；在此基础上证明较小的分布更新步长 $\lambda$（对应小学习率）能压低通用性能退化的上界，这就为发现 1 提供了理论支撑。该分析还顺带解释了发现 2——当 hard token 数量减少时，"安全步长"的范围随之扩大，于是 label-only 训练能容忍更大的学习率。

**3. TALR：按 token 概率自适应重加权，压住退化的真正来源**

理论把退化的主要驱动力指向 hard token（模型当前概率很低的 token），TALR 便直接在损失层面下调这些 token 的权重，回避了"如何挑 hard token、阈值定多少、降权多少"这些手调难题。它被写成一个单纯形（simplex）上的约束优化：最小化加权损失再加一项熵正则，

$$\min_{\mathbf{w}\in\Delta_n}\ \sum_i w_i\,\ell_i(\theta)+\tau\sum_i w_i\log w_i,$$

其中第一项偏好低损失 token、熵正则防止权重过度集中。该问题有闭式解 $w_i^* \propto p_\theta(x_i)^{1/\tau}$——概率高的简单 token 拿到更大权重、概率低的 hard token 被自动压低，无需任何额外超参搜索。温度 $\tau$ 也不是手调的，而是取每个 batch 内 token 损失的中位数，随训练推进自动衰减，于是权重分布会动态变化：训练初期聚焦容易学的 token，随着模型进步再逐渐纳入原先的 hard token，整体呈现课程学习的动态（实验中 $p>0.2$ 的 token 占比从 Epoch 1 到 Epoch 2 稳步上升即印证了这一点）。工程上还有两处细节保证稳定：权重 $w_i$ 经 stop-gradient 计算、不参与反向传播，避免权重与损失互相耦合震荡；同时对权重设下界 $w_{\min}$ 做截断（$w_i\leftarrow\max(w_i,w_{\min})$），防止 hard token 被压成零权重而彻底学不到领域知识。

## 实验关键数据

### 表1：MedCalc基准 学习率1e-6下的领域/通用性能对比

| 方法 | Qwen2.5-3B 领域 | Qwen2.5-3B 通用 | Qwen3-4B 领域 | Qwen3-4B 通用 | 平均领域 | 平均通用 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| Standard (Ours) | 0.495 | 0.620 | 0.548 | 0.784 | 0.534 | 0.692 |
| L2-Reg | 0.490 | 0.621 | 0.469 | 0.796 | 0.506 | 0.697 |
| LoRA | 0.126 | 0.583 | 0.195 | 0.764 | 0.181 | 0.490 |
| Wise-FT | 0.195 | 0.629 | 0.143 | 0.788 | 0.198 | 0.727 |
| FLOW | 0.364 | 0.597 | 0.477 | 0.787 | 0.469 | 0.692 |
| **TALR (Ours)** | **0.481** | **0.648** | **0.489** | **0.788** | **0.501** | **0.717** |

小学习率下各方法差距不大，TALR在通用能力保持上最优。

### 表2：MedCalc基准 学习率5e-6下的领域/通用性能对比

| 方法 | 平均领域 | 平均通用 |
|------|:---:|:---:|
| Standard | 0.558 | 0.381 |
| L2-Reg | 0.555 | 0.395 |
| FLOW | 0.553 | 0.450 |
| **TALR (Ours)** | **0.542** | **0.502** |

大学习率下通用能力退化加剧，TALR优势最为显著——通用性能比Standard高出12个百分点。

### Token层面分析

- 绝大多数SFT训练token对LLM而言学习难度低（中位概率接近1.0），即使模型在该领域任务上zero-shot性能很差。
- 少量hard token主要出现在领域特有概念处（如临床换算因子），是性能瓶颈所在。
- TALR训练过程中p>0.2的token占比从Epoch 1到Epoch 2稳步增长，呈现课程学习动态。

## 亮点

- **挑战主流认知**：系统证明SFT并非总是显著损害通用能力，先前文献的夸大结论部分源于学习率选择不当。
- **理论与实践统一**：信息论分析不仅解释了经验现象，还直接指导了TALR方法的设计。
- **TALR设计优雅**：闭式解、无额外超参搜索（τ自适应）、stop-gradient保证稳定，实现简洁。
- **实用指南清晰**：(1) 优先使用小学习率；(2) 需更强平衡时采用TALR。

## 局限与展望

- **未完全消除退化**：包括TALR在内的所有方法均无法在大学习率下完全避免通用能力退化。
- **数据集有限**：仅在MedCalc和ESCI两个数据集上验证，未涵盖更多领域。
- **模型规模受限**：实验仅涉及3B-4B参数模型，未验证对更大模型或MoE架构的适用性。
- **最优学习率选择**：理论分析未给出如何自动选择最优学习率的实用准则。
- **计算资源限制**：作者承认因资源不足未能进行更大范围的实验验证。

## 与相关工作的对比

| 方法类别 | 代表工作 | 与本文关系 |
|----------|---------|-----------|
| L2正则化 | EWC, L2-Reg | 约束参数漂移，但效果有限 |
| 模型平均 | Wise-FT | 领域性能大幅下降，不适合领域差距大的场景 |
| LoRA | Hu et al. 2022 | 低秩约束导致领域性能严重不足 |
| 数据重加权 | FLOW | 基于样本级易难区分，本文提出更精细的token级方案 |
| 持续学习 | data-dependent方法 | 需要预训练数据，实际场景不可行 |

TALR在data-oblivious设定下实现了最佳的Pareto权衡。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 重新审视被忽视的学习率因素+信息论分析+token级自适应重加权
- 实验充分度: ⭐⭐⭐ — 多模型多设定验证充分，但数据集种类偏少
- 写作质量: ⭐⭐⭐⭐ — 逻辑清晰，理论与实验紧密衔接
- 价值: ⭐⭐⭐⭐ — 对LLM领域微调实践具有直接指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Sculpting Subspaces: Constrained Full Fine-Tuning in LLMs for Continual Learning](sculpting_subspaces_constrained_full_fine-tuning_in_llms_for_continual_learning.md)
- [\[CVPR 2026\] S2FT: Parameter-Efficient Fine-Tuning in Sparse Spectrum Domain](../../CVPR2026/model_compression/s2ft_parameter-efficient_fine-tuning_in_sparse_spectrum_domain.md)
- [\[AAAI 2026\] Consensus-Aligned Neuron Efficient Fine-Tuning Large Language Models for Multi-Domain Machine Translation](../../AAAI2026/model_compression/consensus-aligned_neuron_efficient_fine-tuning_large_language_models_for_multi-d.md)
- [\[ICLR 2026\] GradPruner: Gradient-guided Layer Pruning Enabling Efficient Fine-Tuning and Inference for LLMs](gradpruner_gradient-guided_layer_pruning_enabling_efficient_fine-tuning_and_infe.md)
- [\[ACL 2025\] DoMIX: An Efficient Framework for Exploiting Domain Knowledge in Fine-Tuning](../../ACL2025/model_compression/domix_an_efficient_framework_for_exploiting.md)

</div>

<!-- RELATED:END -->
