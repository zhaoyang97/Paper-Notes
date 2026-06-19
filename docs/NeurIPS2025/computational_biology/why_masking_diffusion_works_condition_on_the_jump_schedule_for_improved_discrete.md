---
title: >-
  [论文解读] Why Masking Diffusion Works: Condition on the Jump Schedule for Improved Discrete Diffusion
description: >-
  [NeurIPS 2025][计算生物][离散扩散模型] 揭示了掩码扩散模型优越性的根本原因——它内建了已知的跳转时间分布，由此提出Schedule-Conditioned Diffusion (SCUD)框架，将此优势推广到任何离散扩散模型，结合结构化前向过程在图像和蛋白质数据上超越掩码扩散。 离散扩散模型通过逐步逆转加噪…
tags:
  - "NeurIPS 2025"
  - "计算生物"
  - "离散扩散模型"
  - "掩码扩散"
  - "跳转调度"
  - "SCUD"
  - "蛋白质生成"
---

# Why Masking Diffusion Works: Condition on the Jump Schedule for Improved Discrete Diffusion

**会议**: NeurIPS 2025  
**arXiv**: [2506.08316](https://arxiv.org/abs/2506.08316)  
**代码**: [GitHub](https://github.com/AlanNawzadAmin/SCUD)  
**领域**: 计算生物  
**关键词**: 离散扩散模型, 掩码扩散, 跳转调度, SCUD, 蛋白质生成

## 一句话总结

揭示了掩码扩散模型优越性的根本原因——它内建了已知的跳转时间分布，由此提出Schedule-Conditioned Diffusion (SCUD)框架，将此优势推广到任何离散扩散模型，结合结构化前向过程在图像和蛋白质数据上超越掩码扩散。

## 研究背景与动机

离散扩散模型通过逐步逆转加噪的马尔可夫过程来生成离散数据（文本、图像像素、蛋白质序列），在生物序列设计等任务上已达到SOTA。

一个令人困惑的现象是：**尽管理论上结构化前向过程（如将相近token以更高概率相互转换）应该比简单的均匀/掩码过程更好，但实践中最简单的掩码扩散（masking diffusion）始终表现最优**。之前的工作因此放弃了前向过程设计，将注意力转向采样和扩展。

本文的核心洞察在于：连续和离散马尔可夫过程之间存在一个根本差异——**离散马尔可夫过程通过固定速率的不连续跳转演化**，掩码扩散之所以优越，是因为它内建了已知的跳转时间分布 $p(S)$，只需要学习"跳转到哪里"而不需要同时学习"何时跳转"。

实验观察也证实了这一点：经典离散扩散模型的反向过程在"何时跳转"上与前向过程存在可检测的差距，而掩码扩散没有这一误差。

## 方法详解

### 整体框架

SCUD (Schedule-Conditioned Diffusion) 将ELBO分解为"何时跳转"和"跳转到哪里"两部分，将已知的跳转时间分布内建到模型中（设 $q(S) = p(S)$），从而只需学习在每个事件处"跳转到哪里"。

### 关键设计

1. **ELBO的When/Where分解**

   传统ELBO写成关于瞬时t的积分形式。本文引入"跳转调度" $S = \{t_1, t_2, \ldots, t_M\}$，将ELBO重新分解为：

    $\text{ELBO} = \underbrace{E_{p} \log \frac{q_\theta((x_t)_t | x_1, S)}{p((x_t)_t | x_0, x_1, S)}}_{\text{where to jump}} - \underbrace{\text{KL}(p(S) \| q_\theta(S))}_{\text{when to jump}} - E_{p(S,x_0)} \text{KL}(p(x_1|S,x_0) \| q_\theta(x_1|S)) + C$

   第一项衡量在已知跳转时间下前向/反向过程是否在跳转目标上匹配；第二项衡量跳转时间是否匹配；第三项衡量前向过程的收敛程度。

   **核心操作**：令 $q(S) = p(S)$，第二项KL变为0，只需优化第一项——学习"去哪里"。

2. **基于事件调度的SCUD模型**

   对一般的无穷小生成器 $\mathcal{L}$，通过引入事件速率 $r$ 和转移核 $K$，使得 $\mathcal{L} = r(K - I)$。每个事件间隔服从 $\text{Exp}(r)$，事件发生时按 $K$ 的行分布跳转（可能跳转到自身，即不变）。

   反向过程通过"在每个事件处预测前一个状态"来参数化：
    $q_\theta(\text{pr}(x_t^d) | x_t, s_t)$
   其中 $s_t$ 是到时间 $t$ 发生的事件数（DD维向量），比单一时间 $t$ 提供了**更细粒度**的噪声信息。

   SCUD损失的高效形式：
    $-E_{t \sim \text{Unif}(0,1)} E_{p(x_t, x_0, S)} \frac{\beta_t}{\int_0^t \beta_s ds} \sum_d s_t^d \text{KL}(p(\text{pr}(x_t^d) | x_t^d, s_t^d, x_0^d) \| q_\theta(\text{pr}(x_t^d) | x_t, s_t))$

3. **与掩码/经典扩散的统一关系**

   通过参数 $\gamma$ 控制调度条件化的程度：$r = \gamma^{-1} r^*$

    - 当 $\gamma = 1 - 1/D$ 且 $\mathcal{L}$ 为均匀过程时，SCUD**恰好等价于掩码扩散**——掩码指示器 $m_t^d = \mathbb{I}[s_t^d > 0]$ 即为条件化信息
    - 当 $\gamma \to 0$ 时，事件数趋于无穷，$s_t$ 输入近似于 $t$，SCUD**退化为经典离散扩散(SEDD)**

   这一统一关系**解释了**掩码扩散为何总是比均匀扩散好：掩码扩散内建了跳转调度信息。

### 损失函数 / 训练策略

在架构上，SCUD将原本注入时间 $t$ 的加法层替换为注入 $s_t$（DD维向量）的FiLM层。对图像使用D3PM的logistic参数化，使相近像素值有相近概率。对蛋白质使用BLOSUM突变矩阵作为前向过程。训练细节尽量与基线保持一致，确保性能差异来自调度条件化。

## 实验关键数据

### 主实验——图像 (CIFAR-10, B=256)

| 方法 | 前向过程 | BPD (bits/dim) ↓ |
|------|---------|----------------|
| D3PM Gaussian | 结构化 | 3.44 |
| τLDR Uniform | 均匀 | 3.41 |
| MD4 Masking | 掩码 | 3.32 |
| **SCUD Uniform** | 均匀 | 3.32 |
| **SCUD Gaussian** | 结构化 | **3.26** |

### 主实验——蛋白质 (UniRef50)

| 方法 | 前向过程 | Perplexity ↓ |
|------|---------|-------------|
| D3PM BLOSUM | 结构化 | 8.25 |
| D3PM Masking | 掩码 | 6.29 |
| Classical BLOSUM (re-impl) | 结构化 | 6.18 |
| Masking (re-impl) | 掩码 | 6.22 |
| **SCUD Uniform** | 均匀 | 6.13 |
| **SCUD BLOSUM** | 结构化 | **5.91** |

### 语言建模 (LM1B)

| 方法 | 前向过程 | Perplexity ↓ |
|------|---------|-------------|
| D3PM Masking | 掩码 | 76.9 |
| D3PM Graph | 图结构 | 149.5 |
| SCUD Masking | 掩码 | 37.82 |
| **SCUD Graph** | 图结构 | **37.63** |

### 关键发现

1. **解释了掩码扩散的优越性**：SCUD均匀过程 ($\gamma$ 从0到1扫描) 平滑插值了经典均匀扩散和掩码扩散的性能，证实调度条件化是掩码扩散成功的关键
2. **结构化前向过程+调度条件化 > 掩码**：在图像上SCUD Gaussian比掩码扩散好0.06 BPD，在蛋白质上SCUD BLOSUM比掩码好0.31 perplexity——之前被认为无用的结构化先验被调度条件化"解锁"
3. **计算开销极小**：SCUD与经典扩散的运行时间差异在10%以内，且在语言建模(B=30522)上SCUD允许使用经典方法计算上不可行的稀疏图结构前向过程
4. **掩码之所以优于高斯**：不是因为掩码前向过程更好，而是因为调度条件化的收益大于结构化先验的收益；SCUD同时拥有两者则最优

## 亮点与洞察

- 这篇工作的核心贡献是**理论洞察**：将离散扩散模型的训练目标分解为"何时"和"去哪里"，并证明掩码扩散等价于完全条件化均匀过程的跳转调度
- SCUD框架统一了掩码扩散与经典离散扩散，$\gamma$ 参数提供了一个连续的设计空间
- 结果具有广泛的实用影响：研究者不再需要默认选择掩码前向过程，而可以设计结合领域先验的结构化前向过程
- 稀疏图结构在语言建模中的应用展示了SCUD的计算优势

## 局限与展望

- 本文聚焦于模型拟合（ELBO/likelihood），未深入探讨对采样质量的影响
- 条件化信息 $S$ 的选择存在权衡：过多信息使去噪任务更难且 $p(x_1|S,x_0)$ 可能不收敛
- 结构化前向过程的设计仍需领域知识，本文仅测试了高斯和BLOSUM，未探索自动学习前向过程
- 在文本任务上结构化前向过程的提升不如图像和蛋白质显著

## 相关工作与启发

- 证明了掩码扩散并非"最优前向过程"，而是"最优参数化"的一个实例，为离散扩散设计空间重新打开大门
- SCUD可与离散流匹配方法结合（附录E中有初步讨论）
- 对蛋白质设计领域有直接应用价值：BLOSUM矩阵编码了进化突变先验，SCUD使其优势得以体现
- 未来可探索为每种突变类型分别计数的更细粒度调度

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 核心理论洞察深刻且优雅，统一了两类看似不同的方法
- 实验充分度: ⭐⭐⭐⭐ 覆盖图像/蛋白质/文本三种模态，但采样质量评估不足
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导严谨，直觉解释清晰，图表说明力强
- 价值: ⭐⭐⭐⭐⭐ 对离散扩散模型设计范式有深远影响

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Constrained Discrete Diffusion](constrained_discrete_diffusion.md)
- [\[NeurIPS 2025\] Split Gibbs Discrete Diffusion Posterior Sampling](split_gibbs_discrete_diffusion_posterior_sampling.md)
- [\[NeurIPS 2025\] Remasking Discrete Diffusion Models with Inference-Time Scaling](remasking_discrete_diffusion_models_with_inference-time_scaling.md)
- [\[ICML 2025\] GenMol: A Drug Discovery Generalist with Discrete Diffusion](../../ICML2025/computational_biology/genmol_a_drug_discovery_generalist_with_discrete_diffusion.md)
- [\[NeurIPS 2025\] Fractional Diffusion Bridge Models](fractional_diffusion_bridge_models.md)

</div>

<!-- RELATED:END -->
