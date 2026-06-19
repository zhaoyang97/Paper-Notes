---
title: >-
  [论文解读] On the Information Processing of One-Dimensional Wasserstein Distances with Finite Samples
description: >-
  [AAAI 2026][计算生物][Wasserstein距离] 本文通过Poisson过程框架，解析刻画了一维Wasserstein距离在有限样本下同时编码概率密度函数的逐点密度差异（rate difference）和支撑差异（support difference）的能力，并在神经脉冲数据和氨基酸接触频率数据上验证了其实际价值。
tags:
  - "AAAI 2026"
  - "计算生物"
  - "Wasserstein距离"
  - "有限样本"
  - "Poisson过程"
  - "速率编码"
  - "支撑差异"
---

# On the Information Processing of One-Dimensional Wasserstein Distances with Finite Samples

**会议**: AAAI 2026  
**arXiv**: [2511.12881](https://arxiv.org/abs/2511.12881)  
**代码**: [GitHub](https://github.com/cheongjae/one-dim-wasserstein)  
**领域**: 计算生物  
**关键词**: Wasserstein距离, 有限样本, Poisson过程, 速率编码, 支撑差异

## 一句话总结

本文通过Poisson过程框架，解析刻画了一维Wasserstein距离在有限样本下同时编码概率密度函数的逐点密度差异（rate difference）和支撑差异（support difference）的能力，并在神经脉冲数据和氨基酸接触频率数据上验证了其实际价值。

## 研究背景与动机

Wasserstein距离（也称Earth Mover's Distance）通过计算样本间的传输代价来度量两个概率分布之间的差异。由于其依赖于数据空间中的传输距离，Wasserstein距离在捕捉分布的**支撑差异**方面具有天然优势——这也是它在生成模型（如WGAN）中被广泛采用的原因之一。

然而，当两个分布的支撑高度重叠、但逐点密度存在显著差异时，Wasserstein距离是否以及如何准确识别这些密度差异，尤其是在**有限样本**条件下的解析刻画，仍不清楚。这一问题在以下情境中尤为关键：

**神经科学中的编码之争**：大脑是通过速率编码（rate coding）还是时间编码（temporal coding）来传递信息，一直是神经科学的经典争论。Wasserstein距离能否同时捕获这两种信息模式？

**与KL散度的比较**：KL散度能够恰当量化密度差异（rate difference），但对支撑差异过度敏感（甚至可能为无穷大）。Wasserstein距离被认为可以避免这种过度敏感性，但它能否可靠地处理密度差异？

**有限样本的实际需求**：在实际应用中，我们总是只有有限个样本，渐近分析给出的洞察不够充分。

本文旨在填补这一理论空白：证明一维Wasserstein距离在有限样本下能够同时编码速率差异和支撑差异，并阐明两者如何协调。

## 方法详解

### 整体框架

作者选择**Poisson过程**作为可处理的分析框架。Poisson过程仅由速率参数 $\lambda > 0$ 定义，该参数直接控制事件频率。通过对Poisson过程生成的有限样本序列分析其经验Wasserstein距离，可以将速率差异和支撑差异的影响清晰地隔离出来。

一维1-Wasserstein距离在经验分布 $\hat{\mu}_N$ 和 $\hat{\nu}_N$ 之间有闭式解：

$$W(\hat{\mu}_N, \hat{\nu}_N) = \frac{1}{N}\sum_{k=1}^{N}|x_k - y_k|$$

其中 $x_k$ 和 $y_k$ 分别是两个序列的第 $k$ 个有序样本。

### 关键设计

#### 1. 速率差异编码（Rate Difference Encoding）

考虑两个速率分别为 $\lambda_1$ 和 $\lambda_2$ 的Poisson过程（$\lambda_1 < \lambda_2$），各生成 $N$ 个事件时间。第 $k$ 个事件时间服从Erlang分布：$x_k \sim p(x_k; k, \lambda_1)$。

**核心定理（Proposition 3.1）**：第 $k$ 对样本之间的期望距离为：

$$\mathbb{E}[|x_k - y_k|] = \frac{\lambda_1 + \lambda_2}{2\lambda_1\lambda_2}\mathbb{E}_{i\sim P(i|2k,p)}[|i-(2k-i)|]$$

其中 $p = \lambda_1/(\lambda_1+\lambda_2)$，$P(i|2k,p)$ 是参数为 $2k$ 和 $p$ 的二项分布。

该表达式的关键特征：
- 完全由速率 $\lambda_1$ 和 $\lambda_2$ 表示，呈现出对称性
- 在调和平均数 $\frac{2\lambda_1\lambda_2}{\lambda_1+\lambda_2}=C$ 固定的约束下，$\mathbb{E}[|x_k-y_k|]$ 在 $\lambda_1=\lambda_2$ 时取最小值
- 当两速率相等时（即无密度差异），Wasserstein距离最小化

**渐近行为（Proposition 3.2）**：当 $k \to \infty$ 时，归一化距离 $s_k = |x_k-y_k|/k$ 满足：

$$\lim_{k\to\infty}\mathbb{E}[s_k] = \frac{1}{\lambda_1} - \frac{1}{\lambda_2}, \quad \lim_{k\to\infty}\text{Var}[s_k] = 0$$

这说明大样本下期望Wasserstein距离可近似为 $\frac{N+1}{2}\left(\frac{1}{\lambda_1}-\frac{1}{\lambda_2}\right)$，直接反映了逆速率差。

#### 2. 支撑差异编码（Support Difference Encoding）

将一个Poisson过程的事件时间平移 $\Delta t \geq 0$，分析 $\mathbb{E}[|x_1+\Delta t - y_1|]$：

$$\mathbb{E}[|x_1+\Delta t - y_1|] = e^{-\lambda_2\Delta t}\cdot(\text{纯速率项}) + \Delta t + (1-e^{-\lambda_2\Delta t})\left(\frac{1}{\lambda_1}-\frac{1}{\lambda_2}\right)$$

- 当 $\Delta t = 0$ 时，退化为纯速率编码
- 当 $\Delta t \to \infty$ 时，简化为 $\Delta t + \frac{1}{\lambda_1} - \frac{1}{\lambda_2}$，平移项主导
- 通过 $e^{-\lambda_2\Delta t}$ 因子自然平衡速率与支撑信息的权重

#### 3. 时变速率扩展

对于非齐次Poisson过程（时变速率 $\mu(t)$），通过变量替换 $x_k \mapsto u = m(x_k)$（其中 $m(x)=\int_0^x \mu(t)dt$），期望距离可以转化为涉及逆累积速率函数的双重积分。当 $|m^{-1}(u)-n^{-1}(v)|u^{k-1}v^{l-1}$ 的双重Laplace变换可解析处理时，理论框架可扩展至更一般情形，包括Sliced Wasserstein距离。

### 损失函数 / 训练策略

本文为理论分析工作，不涉及模型训练。实验中用到的分类模型（如FCN、ResNet、InceptionTime、XceptionTime）使用SGD优化交叉熵损失，但这非本文核心贡献。

## 实验关键数据

### 主实验：合成数据中的速率/支撑差异预测

| 特征类型 | $R^2$（$\log r_1$） | $R^2$（$\log r_2$） | $R^2$（$|\Delta t|$） |
|---------|-------------------|-------------------|---------------------|
| Directed Hausdorff | 43.7±0.5 | 43.9±0.3 | 70.4±0.3 |
| Bin-Wise JS散度 | 64.0±0.4 | 68.4±0.3 | 70.3±0.1 |
| **样本传输代价** | **81.5±0.1** | **81.9±0.2** | **98.9±0.0** |

样本传输代价在速率和支撑差异预测任务中全面超越Hausdorff距离和JS散度特征。

### 视网膜神经节细胞刺激分类

| 方法 | Retina-All | Retina14 | Retina23 |
|------|-----------|----------|----------|
| FCN (仅ISI) | 0.945 | 0.962 | 0.925 |
| FCN + SD1 | 0.951 | 0.971 | 0.931 |
| FCN + SD2 | 0.945 | 0.968 | 0.935 |
| XceptionTime (仅ISI) | 0.944 | 0.970 | 0.930 |
| XceptionTime + SD1 | 0.947 | 0.979 | 0.932 |

在ISI特征基础上加入样本传输距离特征（SD1/SD2）可显著提升刺激类型分类的AUC。

### 消融实验：氨基酸嵌入

| 度量方法 | Kendall's τ（Top-10疏水性） | Kendall's τ（全部残基） |
|---------|---------------------------|----------------------|
| KL散度嵌入 | 0.582 | 0.731 |
| **Wasserstein嵌入** | **0.722** | **0.807** |

Wasserstein距离嵌入的径向排序与已知疏水性排序的相关性显著高于KL散度。

### 关键发现

1. **Wasserstein距离在有限样本下可靠编码速率差异**：期望Wasserstein距离在速率相等时取最小值，证明了密度差异的可辨识性
2. **速率与支撑信息的自然协调**：通过指数衰减因子 $e^{-\lambda_2\Delta t}$ 平滑过渡
3. **互补性**：Wasserstein距离提供了与KL散度和Hausdorff距离互补的信息视角
4. **实际效用**：在神经科学（脉冲序列解码）和分子生物学（氨基酸接触频率）中展示了跨领域适用性

## 亮点与洞察

- **理论优雅性**：将Wasserstein距离的信息处理能力分解为速率编码和支撑编码两个正交维度，并通过Poisson过程获得闭式解析表达
- **跨学科桥梁**：将最优传输理论与神经科学中的speed率编码vs时间编码之争联系起来，Wasserstein距离天然地统一了两种信息
- **Isomap嵌入实验设计巧妙**：通过在人类神经脉冲数据上的嵌入可视化，直观展示了Wasserstein距离如何在同一空间中同时编码时间平移和速率变化
- **氨基酸实验的生物学意义**：CYS（最强疏水性氨基酸）在Wasserstein嵌入中位于最外缘，而在KL嵌入中位置偏内，说明Wasserstein距离更好地捕获了长程接触导致的速率差异

## 局限与展望

1. **仅限一维分析**：虽然讨论了向Sliced Wasserstein距离的扩展，但核心理论推导严格限于一维情形
2. **Poisson假设较强**：实际数据（如神经脉冲）并不完全符合Poisson假设，虽然实验表明结论在非Poisson设定下仍有效，但缺乏理论保证
3. **等样本数假设**：理论推导假设两个序列样本数相同（$N=N$），不等样本数情形的推广仅简要提及
4. **高维推广**：如何将这些精确的信息处理性质推广到高维Wasserstein距离是开放问题
5. **计算复杂度**：未讨论有限样本Wasserstein距离计算在大规模数据上的可扩展性

## 相关工作与启发

- 本文的理论贡献可以为**生成模型中的Wasserstein距离应用**提供更深层的理论支撑，特别是在理解WGAN等方法为何能有效捕获分布差异方面
- 对**Sliced Wasserstein距离**的讨论暗示了可以将本文的信息处理分析扩展到高维场景，这对于近年来大量使用sliced Wasserstein的方法（如图像生成、分布匹配）具有潜在指导意义
- 在**神经科学**方面，本文为spike train分析提供了一个新的理论工具，可能启发新的神经信号解码方法

## 评分

- **理论深度**: ★★★★★ — 解析推导严格完整，从Poisson过程到Wasserstein距离的连接优雅
- **实验充分性**: ★★★★☆ — 合成实验+两个真实应用领域，但下游任务提升幅度有限
- **创新性**: ★★★★☆ — 填补了有限样本Wasserstein距离信息处理能力的理论空白
- **实用性**: ★★★☆☆ — 理论洞察有深度，但直接应用场景相对有限
- **综合**: 7.5/10

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] One Small Step with Fingerprints, One Giant Leap for De Novo Molecule Generation from Mass Spectra](../../NeurIPS2025/computational_biology/one_small_step_with_fingerprints_one_giant_leap_for_de_novo_molecule_generation_.md)
- [\[ICML 2026\] Neural Estimation of Pairwise Mutual Information in Masked Discrete Sequence Models](../../ICML2026/computational_biology/neural_estimation_of_pairwise_mutual_information_in_masked_discrete_sequence_mod.md)
- [\[ICML 2025\] WGFormer: An SE(3)-Transformer Driven by Wasserstein Gradient Flows for Molecular Generation](../../ICML2025/computational_biology/wgformer_an_se3-transformer_driven_by_wasserstein_gradient_flows_for_molecular_g.md)
- [\[NeurIPS 2025\] Manipulating 3D Molecules in a Fixed-Dimensional E(3)-Equivariant Latent Space](../../NeurIPS2025/computational_biology/manipulating_3d_molecules_in_a_fixed-dimensional_e3-equivariant_latent_space.md)
- [\[NeurIPS 2025\] Benchmarking Agentic Systems in Automated Scientific Information Extraction with ChemX](../../NeurIPS2025/computational_biology/benchmarking_agentic_systems_in_automated_scientific_information_extraction_with.md)

</div>

<!-- RELATED:END -->
