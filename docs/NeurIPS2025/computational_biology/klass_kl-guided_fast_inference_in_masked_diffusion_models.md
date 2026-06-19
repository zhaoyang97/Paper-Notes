---
title: >-
  [论文解读] KLASS: KL-Guided Fast Inference in Masked Diffusion Models
description: >-
  [NeurIPS 2025 Spotlight][计算生物][扩散模型] 提出 KLASS（KL-Adaptive Stability Sampling），一种无需训练的采样方法，利用 token 级别的 KL 散度和置信度来识别稳定 token 并行解码，在掩码扩散模型上实现最高 2.78× 加速且不损失甚至提升生成质量。
tags:
  - "NeurIPS 2025 Spotlight"
  - "计算生物"
  - "扩散模型"
  - "KL Divergence"
  - "加速采样"
  - "Token稳定性"
  - "并行解码"
---

# KLASS: KL-Guided Fast Inference in Masked Diffusion Models

**会议**: NeurIPS 2025 Spotlight  
**arXiv**: [2511.05664](https://arxiv.org/abs/2511.05664)  
**代码**: [GitHub](https://github.com/shkim0116/KLASS)  
**领域**: 计算生物  
**关键词**: Masked Diffusion Models, KL Divergence, 加速采样, Token稳定性, 并行解码

## 一句话总结

提出 KLASS（KL-Adaptive Stability Sampling），一种无需训练的采样方法，利用 token 级别的 KL 散度和置信度来识别稳定 token 并行解码，在掩码扩散模型上实现最高 2.78× 加速且不损失甚至提升生成质量。

## 研究背景与动机

**领域现状**：掩码扩散模型（MDM）在语言生成、图像生成、分子生成等任务上展现了竞争力，如 LLaDA、DREAM 等大规模模型已具备推理能力。

**现有痛点**：MDM 的采样过程依赖迭代去掩码，通常采用固定的 Top-k 或随机采样策略，每步仅解掩很少 token，导致推理速度慢。

**核心矛盾**：加速采样（一次解掩更多 token）与生成质量之间存在权衡——过早解掩不稳定的 token 会降低准确率。

**本文目标**：如何在不需要额外训练或外部 planner 的前提下，安全地并行解掩多个 token 以加速生成。

**切入角度**：利用模型自身的信号（KL 散度 + 置信度）来判断 token 是否"稳定"，稳定的 token 可以安全地提前解掩。

**核心 idea**：KL 散度低 + 置信度高的 token 是"稳定"的，可被并行解掩；不正确的 token 在扩散过程中无法保持动态稳定性。

## 方法详解

### 整体框架

KLASS 在标准掩码扩散模型的反向采样过程中，每一步计算所有被掩码 token 的两个指标：置信度分数和 KL 分数，然后自适应地选择满足条件的 token 批量解掩。

### 关键设计

1. **置信度分数 (Confidence Score)**：定义为模型预测分布中最大概率值：
    $\text{conf}_t^i = \max_v p_t^i(v)$
   高置信度表明模型对该 token 的估计更确定。

2. **KL 分数 (KL Score)**：定义为相邻时间步的预测分布之间的 KL 散度：
    $d_t^i = D_{\text{KL}}(p_t^i \| p_{t+1}^i)$
   低 KL 分数表明模型对该 token 的估计在时间上是一致的、稳定的。实验发现正确预测的 token 普遍具有更低的 KL 分数。

3. **稳定 token 选择**：给定历史长度 $n$、KL 阈值 $\epsilon_{\text{KL}}$ 和置信度阈值 $\tau$，稳定 token 集合为：
    $S_t = \{i \mid \forall k \in \{1,...,n\}, D_{\text{KL}}(p_{t+k-1}^i \| p_{t+k}^i) < \epsilon_{\text{KL}} \wedge \text{conf}_t^i > \tau\}$

4. **解掩规则**：若存在稳定 token（$S_t \neq \emptyset$），全部解掩；否则退回到 Top-$u$ 置信度解掩作为 fallback。

### 理论支撑

**Proposition 5.3**：对于一个良好训练的模型，如果某个 token 当前的预测是错误的，那么随着上下文逐步揭示，它的预测分布必然会发生显著变化（平均 KL 散度有下界）。这意味着**不正确的 token 无法保持动态稳定**，因此 KLASS 通过延迟解掩不稳定 token 来避免错误。

### 计算开销

KL 计算是轻量级的后处理操作，无需额外前向传播。实验显示内存开销 < 1.57%，延迟开销 < 0.21%。

## 实验关键数据

### 主实验：推理任务

| Method | Model | MATH Acc↑ | Steps↓ | GSM8K Acc↑ | Steps↓ | HumanEval Acc↑ | Steps↓ |
|--------|-------|-----------|--------|------------|--------|----------------|--------|
| Top-1 | LLaDA | 31.4 | 256 | 75.13 | 256 | 39.63 | 256 |
| Confidence | LLaDA | 31.6 | 96.5 | 75.21 | 74.4 | 37.80 | 54.4 |
| **KLASS** | **LLaDA** | **33.8** | **128.6** | **76.50** | **98.6** | **40.85** | **92.0** |
| Top-1 | DREAM | 37.97 | 256 | 79.55 | 256 | 58.53 | 256 |
| Confidence | DREAM | 41.80 | 95.1 | 73.67 | 74.8 | 50.00 | 52.5 |
| **KLASS** | **DREAM** | **43.20** | **149.7** | **79.43** | **155.7** | **59.35** | **74.9** |

KLASS 在几乎所有任务上都优于标准 Top-1 解码，同时步数减少 40-70%，wall-clock 加速最高 2.78×。

### 文本生成

| Method | MAUVE↑ | LLaMA2 PPL↓ | LLaMA3 PPL↓ | GPT-2 PPL↓ |
|--------|--------|-------------|-------------|------------|
| MDLM | 0.115 | 30.88 | 54.15 | 51.78 |
| **KLASS** | **0.179** | **26.94** | **49.19** | **45.50** |

### 图像生成 (MMaDA)

| Method | Steps | FID↓ | IS↑ |
|--------|-------|------|-----|
| Confidence | 16 | 34.48 | 75.72 |
| **KLASS** | **16** | **30.48** | **93.07** |

### 消融实验

| 解掩策略 | MATH Acc↑ | Steps↓ |
|----------|-----------|--------|
| Single (conf) | 31.2 | 256 |
| Single (KL) | 29.0 | 256 |
| **Parallel (KLASS)** | **33.8** | **128.6** |

并行解掩稳定 token 比仅选单个 token 效果更好，且步数更少。

### 关键发现

- KL 分数和置信度的组合是关键——单用任一标准都不够好
- 正确预测的 token 始终具有更低的 KL 分数（Figure 1b 实证验证）
- KLASS 在文本、图像、分子三种模态上均有效，证明其通用性

## 亮点与洞察

- **无需训练的通用采样器**：KLASS 不修改模型参数，纯粹利用模型自身推理过程中的统计信号
- **理论与实践结合**：Proposition 5.3 从理论上解释了为什么 KL 散度可以区分正确/错误 token
- **跨模态通用性**：同一方法适用于语言、图像、分子生成，这在扩散模型采样器中很少见
- **加速且提升质量**：KLASS 不仅加速推理，还在多个 benchmark 上提升了准确率，打破了速度-质量的传统权衡

## 局限与展望

- 超参数（KL 阈值、置信度阈值）需要针对不同模型和任务调整，虽然作者声称不敏感但仍需 grid search
- 目前仅在掩码扩散模型上验证，尚未扩展到其他离散扩散模型（如 uniform noise schedule）
- 历史长度 $n$ 的选择对性能有影响，更长历史带来更多缓存开销
- 在更大规模模型上的效果有待验证

## 相关工作与启发

- 与 Fast-dLLM、Dimple、EB-Sampler 等并发工作相比，KLASS 独特之处在于不仅用置信度，还引入 KL 散度作为"动态稳定性"的度量
- 对 token 级别推理质量的细粒度监控思路可以启发其他离散生成模型的采样策略设计
- KL 散度的"先验"校验机制有潜力扩展到自回归模型的投机解码中

## 评分

- 新颖性: ⭐⭐⭐⭐ KL散度+置信度双条件筛选稳定token的思路新颖，但核心idea相对简洁
- 实验充分度: ⭐⭐⭐⭐⭐ 涵盖文本推理、文本生成、图像生成、分子生成四种模态，消融充分
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰，理论部分简洁有力，图表设计直观
- 价值: ⭐⭐⭐⭐ 即插即用的加速方法，实用价值高，但受限于掩码扩散模型的生态

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] SpecMER: Fast Protein Generation with K-mer Guided Speculative Decoding](specmer_fast_protein_generation_with_k-mer_guided_speculative_decoding.md)
- [\[NeurIPS 2025\] Remasking Discrete Diffusion Models with Inference-Time Scaling](remasking_discrete_diffusion_models_with_inference-time_scaling.md)
- [\[ICLR 2026\] Ultra-Fast Language Generation via Discrete Diffusion Divergence Instruct](../../ICLR2026/computational_biology/ultra-fast_language_generation_via_discrete_diffusion_divergence_instruct.md)
- [\[NeurIPS 2025\] Uncertainty-Aware Multi-Objective Reinforcement Learning-Guided Diffusion Models for 3D De Novo Molecular Design](uncertainty-aware_multi-objective_reinforcement_learning-guided_diffusion_models.md)
- [\[NeurIPS 2025\] Consistent Sampling and Simulation: Molecular Dynamics with Energy-Based Diffusion Models](consistent_sampling_and_simulation_molecular_dynamics_with_energy-based_diffusio.md)

</div>

<!-- RELATED:END -->
