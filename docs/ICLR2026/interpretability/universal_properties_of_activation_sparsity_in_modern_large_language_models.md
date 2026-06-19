---
title: >-
  [论文解读] Universal Properties of Activation Sparsity in Modern Large Language Models
description: >-
  [ICLR2026][可解释性][activation sparsity] 对现代 LLM（GLU 架构 + SiLU/GELU）的激活稀疏性进行系统性研究，提出通用的 top-p 稀疏化框架和临界稀疏度（critical sparsity）指标，发现激活稀疏度随模型规模单调递增、输入稀疏化是最实用的免训练加速方案，并首次证明扩散型 LLM 也具有显著的激活稀疏性。
tags:
  - "ICLR2026"
  - "可解释性"
  - "activation sparsity"
  - "LLM acceleration"
  - "GLU architecture"
  - "critical sparsity"
  - "top-p sparsification"
  - "扩散模型"
---

# Universal Properties of Activation Sparsity in Modern Large Language Models

**会议**: ICLR2026  
**arXiv**: [2509.00454](https://arxiv.org/abs/2509.00454)  
**代码**: [GitHub](https://github.com/fszatkowski/activation-sparsity-benchmarking)  
**领域**: 可解释性  
**关键词**: activation sparsity, LLM acceleration, GLU architecture, critical sparsity, top-p sparsification, diffusion LLM

## 一句话总结
对现代 LLM（GLU 架构 + SiLU/GELU）的激活稀疏性进行系统性研究，提出通用的 top-p 稀疏化框架和临界稀疏度（critical sparsity）指标，发现激活稀疏度随模型规模单调递增、输入稀疏化是最实用的免训练加速方案，并首次证明扩散型 LLM 也具有显著的激活稀疏性。

## 研究背景与动机

**激活稀疏的历史**：ReLU 网络天然产生精确零激活，围绕此性质的效率优化、鲁棒性增强、可解释性分析已有大量工作。

**现代 LLM 的问题**：主流 LLM（Gemma3、LLaMA3、Qwen2.5）使用 GLU 架构 + SiLU/GELU 激活，不产生严格零值——ReLU 时代的方法无法直接迁移。

**现有方案碎片化**：
   - **改造方案**（将 SiLU 替换为 ReLU）需额外训练且可能损害模型质量
   - **近似稀疏方案**缺乏 ReLU 严格零值的原则性保证，需校准阈值，可能过拟合校准集
   - 不同方法分别针对 FFN 的输入、门控、中间激活，设计选择缺乏统一指导

**本文目标**：建立一个通用、简单、无需训练的框架来系统研究和利用现代 LLM 的激活稀疏性。

## 方法详解

### 整体框架

本文不提出新的稀疏化算法，而是建立一套统一、免训练的测量工具：用一个与架构无关的 top-p 规则把任意激活向量稀疏化，再用临界稀疏度这个挂钩实际性能的指标，去横向比较不同模型、不同 FFN 模块、不同生成范式能承受多大的稀疏度。所有结论都建立在 GLU 架构 $\mathcal{FFN}(x) = W_d\big((W_u x) \odot \sigma(W_g x)\big)$ 之上，这是 Gemma3、LLaMA3、Qwen2.5 等主流 LLM 共享的结构。

### 关键设计

**1. Top-p 稀疏化规则：用能量占比代替严格零值**

SiLU/GELU 不像 ReLU 那样产生精确零，传统按零值切分的稀疏化方法因此失效。Top-p 改从能量角度定义稀疏：对任意激活向量 $v \in \mathbb{R}^n$，只保留绝对值最大的若干条目，使被保留部分的 L1 能量占整体的比例达到 $p$，即 $\text{top-p}(v) = m_p \odot v$，其中掩码 $m_p = \arg\min_m \|m\|_0$ 在约束 $\|m \odot v\|_1 \geq p \cdot \|v\|_1$、$m \in \{0,1\}^n$ 下取最稀疏解；它诱导出稀疏度 $S_p(v) = \frac{1}{n}\sum_{i=1}^n \mathbb{1}(m_p^{(i)} = 0)$。这个规则之所以好用，是因为它不对架构做任何假设、也不需要训练或校准数据集，因而避开了近似稀疏方法常见的「过拟合校准集」问题；论文还实证它比 top-k 更可解释、随模型规模退化更平滑，因而比 top-k / max-p 更适合做跨模型的统一比较基准。

**2. 临界稀疏度：把「能稀疏多少」锚定到性能约束上**

单看某个固定 $p$ 下的稀疏度无法回答「这个模型到底能安全跳过多少计算」。本文把临界稀疏度（critical sparsity）定义为模型仍保持 $\geq 99\%$ 原始性能时所能达到的最大稀疏度，相当于把抽象的稀疏比例换成一个有实际意义的承受上限。正因为它绑定在性能上，不同规模、不同家族的模型才可以直接比较谁的冗余更多——后续「稀疏度随规模单调递增」「扩散模型比自回归更耐稀疏」等核心发现，都用这一指标来度量。

**3. 四类激活与三条加速路线：明确稀疏化能作用在哪、各自代价多大**

在 GLU FFN 内部，可被稀疏化的激活有四种，分别对应不同的加速收益与代价：

| 激活类型 | 定义 | 说明 |
|----------|------|------|
| 输入 $x$ | FFN 输入向量 | 稀疏后可同时加速三个线性层 |
| 上投影 $u$ | $W_u x$ | 无激活函数的线性投影 |
| 门控 $g$ | $\sigma(W_g x)$ | 经激活函数后的门控信号 |
| 中间 $i$ | $(W_u x) \odot \sigma(W_g x)$ | 逐元素乘积后的中间表示 |

这四个落点又对应三条互斥的加速路线，本文系统对比它们的取舍：**输入稀疏化**作用在 $x$ 上，无需任何预测器就能一并加速三个线性层，代价是 $x$ 本身没有天然稀疏性；**门控稀疏化**利用 $g$ 经激活函数后自带的压缩，但计算门控本身就占掉 FFN 约 1/3 成本；**预测器方法**直接预测中间激活 $i$ 的稀疏掩码，理论加速最高，却要额外训练一个预测器并承担近似误差。这张「落点—代价」地图，正是后文论证「输入稀疏化最实用」的依据。

## 实验关键数据

### 模型规模与临界稀疏度（Gemma3 系列）

| 模型 | 参数量 | 中间激活稀疏度 | 输入稀疏度 | 门控稀疏度 |
|------|--------|---------------|-----------|-----------|
| Gemma3-1B | 1B | ~50% | ~35% | ~35% |
| Gemma3-4B | 4B | ~55% | ~40% | ~40% |
| Gemma3-12B | 12B | ~62% | ~48% | ~48% |
| Gemma3-27B | 27B | ~70% | ~55% | ~55% |

**核心发现**：临界稀疏度随模型规模单调递增——更大的模型有更多冗余神经元可以安全跳过。

### 有效秩分析

有效秩（effective rank）随模型规模一致下降，表明大模型的激活表示更低秩、更冗余。但门控激活的有效秩与中间激活类似，虽然其经验稀疏化承受力更差——说明有效秩不足以完全刻画稀疏化鲁棒性。

### 跨模型家族趋势

| 模型家族 | 规模范围 | 临界稀疏度趋势 |
|---------|---------|---------------|
| Gemma3 | 1B–27B | 线性增长最明显 |
| LLaMA3.1/3.2 | 1B–70B | 一致增长，宽度/深度缩放较均匀 |
| Qwen2.5 | 0.5B–72B | 整体增长但较波动，维度增长不均匀 |

### 训练方式的影响

| 模型变体 | 临界稀疏度变化 |
|---------|--------------|
| 预训练 → 指令微调 | 大规模时 IT 模型稀疏度更高 |
| Qwen3-4B Instruct vs Thinking | 推理模型在 GSM8K 上更鲁棒，MMLU 上退化更快 |

### 扩散型 LLM（LLaDA-8B）首次分析

| 任务 | 中间激活临界稀疏度 | All-Inputs 临界稀疏度 |
|------|------------------|---------------------|
| MMLU | 69.46% | 62.72% |
| HumanEval | 81.25% | 77.89% |
| HellaSwag | 71.21% | 67.92% |
| MBPP | 66.67% | 59.18% |
| **平均** | **68.13%** | **56.79%** |

LLaDA-8B 的临界稀疏度显著高于同规模自回归 LLaMA3.1-8B——扩散模型的去噪特性使其对稀疏化引入的噪声更鲁棒。

### 扩散步内的时序稳定性

- 连续扩散步之间的 Jaccard 相似度稳定但不高（~0.6–0.7）
- 与初始步的漂移相似度快速下降——稀疏模式随去噪逐步变化
- 结论：扩散 LLM 的稀疏掩码**不能跨步复用**（与自回归模型在 prompt 后掩码可复用不同）

### MoE 模型分析（Qwen3-30B-A3B）

层内平均临界稀疏度稳定，但个别专家的稀疏度远超平均值。128 个专家中的异常值稀疏度甚至超过同等规模的稠密模型——MoE 专家同样普遍展现激活稀疏性。

## 亮点与洞察
- **"功能性稀疏是 LLM 的普遍性质"**：跨架构（GLU/MoE）、跨训练方式（PT/IT/Thinking）、跨生成范式（自回归/扩散）一致成立
- **输入稀疏化是最实用方案**：不需要预测器、不需要计算门控就能加速全部 FFN 模块——在研究的规模范围内门控并无优势
- **校准的风险**：临界稀疏度在不同任务间差异显著，基于校准数据集的阈值方法存在过拟合风险，应追求真正无数据的稀疏化方案
- **扩散 LLM 的潜力**：首次实证表明扩散型 LLM 的激活稀疏度高于自回归模型，但须针对扩散特性设计专用方法

## 局限与展望
- **仅关注 FFN**：未分析多头注意力中的激活稀疏性，虽然 FFN 主导长上下文以外的计算
- **加速上限有限**：激活稀疏加速约 1.3–1.5x，不如投机解码（~4x），应定位为互补技术
- **top-p 是稀疏度下界**：更复杂的层级/模块特定方法可能达到更高稀疏度
- **未提供具体加速实现**：论文聚焦于稀疏性的表征而非部署优化

## 相关工作与启发
- **vs Mirzadeh et al. (2024)**：先前改造 ReLU 方案需额外训练，本文证明免训练的 top-p 已达实用水平
- **vs Liu et al. (2025a/b)**：输入稀疏化加速方法的经验前提在本文得到系统验证
- **vs Song et al. (2024a) / Lee et al. (2024)**：门控稀疏化在研究规模范围内**不优于**输入稀疏化——重要的实践指南
- **启发**：随着模型持续增大，激活稀疏度持续增长，frontier 模型可能天然拥有 70%+ 的可利用稀疏度（Gemma3n 已开始在架构中集成稀疏感知层）

## 评分
- 新颖性: ⭐⭐⭐⭐ 统一框架 + 临界稀疏度定义 + 首次扩散 LLM 稀疏分析，但核心方法（top-p）简单
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖 Gemma3/LLaMA3/Qwen2.5 多规模 + PT/IT/Thinking + MoE + 扩散模型，9 个基准
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图表信息量大，结论明确
- 价值: ⭐⭐⭐⭐ 为 LLM 激活稀疏加速提供了全面的基础参考，实用指导意义强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Towards Atoms of Large Language Models](../../ICML2026/interpretability/towards_atoms_of_large_language_models.md)
- [\[ICLR 2026\] ZeroTuning: Unlocking the Initial Token's Power to Enhance Large Language Models Without Training](zerotuning_unlocking_the_initial_tokens_power_to_enhance_large_language_models_w.md)
- [\[ACL 2026\] Model Internal Sleuthing: Finding Lexical Identity and Inflectional Features in Modern Language Models](../../ACL2026/interpretability/model_internal_sleuthing_finding_lexical_identity_and_inflectional_features_in_m.md)
- [\[ACL 2026\] Compositional Steering of Large Language Models with Steering Tokens](../../ACL2026/interpretability/compositional_steering_of_large_language_models_with_steering_tokens.md)
- [\[ACL 2026\] Knowledge Vector of Logical Reasoning in Large Language Models](../../ACL2026/interpretability/knowledge_vector_of_logical_reasoning_in_large_language_models.md)

</div>

<!-- RELATED:END -->
