---
title: >-
  [论文解读] Back to Square Roots: An Optimal Bound on the Matrix Factorization Error for Multi-Epoch Differentially Private SGD
description: >-
  [ICLR 2026][AI安全][differential privacy] 提出 Banded Inverse Square Root (BISR) 矩阵分解方法，通过对逆相关矩阵（而非相关矩阵本身）施加带状结构，首次在多轮参与差分隐私 SGD 中实现渐近最优的分解误差界，并配套低存储优化变体 BandInvMF。
tags:
  - "ICLR 2026"
  - "AI安全"
  - "differential privacy"
  - "matrix factorization"
  - "DP-SGD"
  - "multi-epoch participation"
  - "banded factorization"
  - "optimal error bounds"
---

# Back to Square Roots: An Optimal Bound on the Matrix Factorization Error for Multi-Epoch Differentially Private SGD

**会议**: ICLR 2026  
**arXiv**: [2505.12128](https://arxiv.org/abs/2505.12128)  
**代码**: 无（使用 jax-privacy 库进行基线比较）  
**领域**: AI安全 / 差分隐私  
**关键词**: differential privacy, matrix factorization, DP-SGD, multi-epoch participation, banded factorization, optimal error bounds

## 一句话总结
提出 Banded Inverse Square Root (BISR) 矩阵分解方法，通过对逆相关矩阵（而非相关矩阵本身）施加带状结构，首次在多轮参与差分隐私 SGD 中实现渐近最优的分解误差界，并配套低存储优化变体 BandInvMF。

## 研究背景与动机
**领域现状**：矩阵分解机制（Matrix Factorization Mechanism）是差分隐私训练中通过注入相关噪声来提升模型效用的重要方法，已被 Google 用于生产级 on-device 语言模型训练。

**现有痛点**：在多轮训练（multi-epoch）中，同一数据点被多次使用，需要刻画分解误差与参与次数的关系。但现有上下界之间存在显著差距——Banded Square Root (BSR) 的误差界中对带宽 $p$ 的依赖是隐式的，无法判断其是否最优。

**核心矛盾**：理论上不清楚多轮参与下分解误差的最优增长率是什么，实践中缺少既高效又有显式误差刻画的分解方法。

**本文目标** 给出多轮参与下矩阵分解误差的紧界（tight bound），并提供一个计算高效、理论最优的显式分解方法。

**切入角度**：不是像 BSR 那样让相关矩阵 $C$ 带状化，而是让 $C^{-1}$ 带状化——这一视角转换带来了显式误差刻画和高效实现的双重优势。

**核心 idea**：在逆相关矩阵上施加带状结构，使噪声注入可通过卷积高效实现，同时获得关于带宽的显式最优误差界。

## 方法详解

### 整体框架
矩阵分解机制把要发布的工作负载矩阵 $A$ 拆成 $A = BC$，在私有端注入相关噪声后得到无偏估计 $\widehat{AX} = B(CX + Z)$，整套方法的好坏全压在分解误差 $\mathcal{E}(B,C)$ 上——它由 $\|B\|_F$ 与 $C$ 的列灵敏度共同决定。已有的 BSR 直接对相关矩阵 $C$ 做带状截断，误差里对带宽的依赖被埋在闭式解里、看不出是否最优。本文反过来让 $C^{-1}$ 带状化：先给出兼具显式误差界与卷积式高效实现的 BISR 分解（连同它在训练时只需一个噪声缓冲区的落地算法），再用"配套下界 + 匹配上界"这一对结果证明它在多轮参与下渐近最优，最后补一个更省存储的数值优化变体 BandInvMF。下面三个设计分别对应这三件事。

### 关键设计

**1. BISR 分解：让逆相关矩阵带状，把噪声注入变成一次卷积**

痛点在 BSR：它直接把相关矩阵 $C$ 截断成带状，结果误差对带宽 $p$ 的依赖隐式埋在闭式解里、既看不出是否最优、训练时还得在线解优化。BISR 换了个落点：从工作负载 $A$ 出发先取其正定平方根 $C$（满足 $C^2 = A$），再把它的逆 $C^{-1}$ 截断为 $p$-带状矩阵，最后重新求逆得到 $C^p$，给出分解 $A = B^p C^p$，其中 $B^p = A (C^p)^{-1}$（Definition 1）。这一步带来两个连锁好处。其一是**显式可分析**：带状落在 $C^{-1}$ 上后，分解出的两个因子的范数与灵敏度都能写成闭式，从而能精确刻画误差对带宽的依赖（这是后面证最优的前提）。其二是**实现廉价**：$C^{-1}$ 带状意味着 $(C^p)^{-1}Z$ 等价于把噪声序列与 $p$ 个固定系数做一次卷积，可用 FFT 加速；落到训练里（Algorithm 1）就是维护一个长度 $p$ 的噪声缓冲区，第 $i$ 步按 $\hat{x}_i = x_i + \zeta \sum_{t=0}^{\min(p,i)-1} c_t Z_{i-t}$ 把最近 $p$ 个噪声样本加权叠加进梯度，流式场景只需存 $p$ 行噪声。因为只触碰梯度的线性变换，它天然兼容 momentum 与 weight decay，存储和计算都远低于需要在线求解优化问题的 BandMF。

**2. 上下界对齐：先钉死最优增长率，再证 BISR 贴着它走**

要谈"最优"，得先知道任何分解都逃不掉的误差下限，再证 BISR 能达到。本文把这两半凑成一对。下半部分（Theorem 3）用概率方法构造参与向量并界定其范数，给出关于参与次数 $k$、矩阵规模 $n$ 的紧下界：无 weight decay（$\alpha = 1$）时为 $\Omega(\sqrt{k}\log n + k)$，带 weight decay（$\alpha < 1$）时收紧为 $\Omega_\alpha(\sqrt{k})$——这第一次明确了多轮参与下误差关于 $\sqrt{k}$ 的"平方根"标度，也是标题 "Back to Square Roots" 的来历。上半部分（Theorem 4 + Corollary 1）借助设计 1 给出的显式因子范数，写出 BISR 误差对带宽 $p$、$k$、$n$、分离参数 $b$ 的显式上界；把带宽取到最优后，这个上界在阶的意义上正好贴住 Theorem 3 的下界。上下界一对齐，BISR 就被证明在多轮参与场景下渐近最优，闭合了 BSR 遗留的上下界差距。
> ⚠️ 最优带宽 $p^*$ 的具体表达式以原文为准（论文给出"BISR 所需带宽比 BSR 的 $p=b$ 更小"，实践中按经验优化 $p^*$）。

**3. BandInvMF：放弃闭式解、改用数值优化的省存储变体**

BISR 的系数来自平方根的闭式构造，在有限规模下不一定是字面最优、且面向低存储场景仍可再压。BandInvMF 保留"逆矩阵带状 + Toeplitz"这套结构，但把系数交给数值优化直接拟合而非套闭式公式，并以 BISR 的系数作初始化（收敛很快）。收益是在低存储 regime 下误差能进一步逼近 state-of-the-art，同时仍只需缓存少量系数、实现简单高效。
> ⚠️ BandInvMF 的具体误差率与收敛步数以原文为准。

## 实验关键数据

### 表1：CIFAR-10 测试精度（(9, 10⁻⁵)-DP，10 epochs）

| 方法 | Epoch 1 | Epoch 5 | Epoch 10 |
|------|---------|---------|----------|
| DP-SGD (Amp.) | 12.7±2.2 | 39.8±1.2 | 44.6±0.7 |
| BSR (Amp.) | 28.3±0.7 | 48.0±2.0 | 49.8±0.3 |
| **BISR (Amp.)** | **32.3±0.7** | **52.8±2.0** | **61.8±0.3** |
| Band-MF (Amp.) | 27.7±2.0 | 46.8±0.8 | 50.0±0.4 |
| Band-Inv-MF (Amp.) | 23.6±2.8 | 48.6±1.0 | 57.4±1.2 |
| DP-SGD (Non-Amp.) | 19.5±3.0 | 37.7±1.2 | 39.0±0.7 |
| **BISR (Non-Amp.)** | **31.8±1.5** | **51.1±1.0** | **56.2±0.2** |

### 表2：RMSE 比较（矩阵分解误差，n=16384）

| 方法 | k=4, α=1,β=0 | k=16, α=1,β=0 | k=16, α=1,β=0.9 |
|------|---------------|----------------|------------------|
| BSR | 与 BISR 相当 | 明显差于 BISR | 差于 BISR |
| BLT | 与 BISR 相当 | 与 BISR 相当 | 仅支持 prefix-sum |
| BandMF | 略优（小矩阵） | 略优但不可扩展 | 计算成本过高 |
| **BISR** | **最优或接近最优** | **显著优于 BSR** | **一致性最佳** |

> BISR 在 k=16 高参与次数下优势尤为明显；BandMF 虽 RMSE 略低但不可扩展至 n>4096

## 亮点与洞察
- **视角转换的力量**：从"让 $C$ 带状化"转为"让 $C^{-1}$ 带状化"，获得了显式误差刻画——这种看似微小的结构改变带来了理论突破。
- **理论与实践闭环**：BISR 同时实现了理论最优性（上下界匹配）和实践竞争力（与 BLT/BandMF 精度相当），且实现极其简单（卷积操作）。
- **低存储 regime 的洞察**：RMSE 更低不等于模型精度更高——Band-Inv-MF 的 RMSE 优于 BISR，但两者训练精度相近，说明分解误差与模型效用的关系非简单单调。
- **实用性突出**：仅需 $p$ 个系数的卷积，存储和计算成本远低于需要求解优化问题的 BandMF。

## 局限性
1. **渐近最优 ≠ 有限规模最优**：BISR 在渐近意义上最优，但有限矩阵大小下 BandMF 等数值优化方法仍可能略优。
2. **常数因子未优化**：上下界匹配在阶（order）的意义上，常数项的差距尚未完全消除。
3. **RMSE-精度脱节**：更低的分解 RMSE 不一定转化为更高的模型精度，特别是在使用 amplification by subsampling 时。
4. **BLT 比较受限**：BLT 仅实现了 prefix-sum 矩阵，无法在 momentum/weight decay 设置下对比。

## 补充实验：IMDB 情感分析（BERT-base）
- 在 (9, 10⁻⁵)-DP 下微调 BERT-base，BISR 在 amplified 和 non-amplified 设置下均优于 BSR 和 Band-MF
- BISR (Amplified) 10 epoch 后显著领先 DP-SGD，体现矩阵分解机制的优势
- 低存储 regime 下 Band-Inv-MF 与 BISR 精度接近，但 BISR 无需优化求解

## 相关工作
- **矩阵分解机制**：Choquette-Choo et al. (2023a) 定义了多轮参与下的最优分解问题；BLT (Dvijotham et al., 2024) 提供了 buffer-based 方法；BandMF (McKenna, 2025) 通过数值优化求解最优带状分解。
- **平方根分解**：Henzinger et al. (2024) 提出，Kalinin & Lampert (2024) 扩展为 BSR 并建立了首个上下界，但带宽依赖不显式。
- **隐私会计**：本文使用 MCMC accountant (Choquette-Choo et al., 2024b) 和 bins-and-balls 子采样 (Chua et al., 2025) 进行隐私分析。
- **联邦学习中的 MF**：Zhang et al. (2025) 和 Bienstock et al. (2025) 将矩阵分解扩展到联邦学习场景。

## 评分
- **创新性**: ★★★★☆ — 逆矩阵带状化的视角转换优雅且有效，配套的理论紧界具有重要贡献。
- **实用性**: ★★★★☆ — 实现简单高效，卷积操作可并行化，已有 JAX 实现。
- **理论深度**: ★★★★★ — 闭合了多轮参与分解误差的理论差距，上下界渐近匹配。
- **实验充分性**: ★★★★☆ — RMSE 和训练精度双重评估，覆盖多种优化器设置和数据集，但大规模 LLM 实验缺失。
- **表达清晰度**: ★★★★☆ — 数学推导严谨，算法描述清晰，Figure 1 的可视化很直观。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Unified Privacy Guarantees for Decentralized Learning via Matrix Factorization](unified_privacy_guarantees_for_decentralized_learning_via_matrix_factorization.md)
- [\[ICLR 2026\] Skirting Additive Error Barriers for Private Turnstile Streams](skirting_additive_error_barriers_for_private_turnstile_streams.md)
- [\[AAAI 2026\] An Improved Privacy and Utility Analysis of Differentially Private SGD with Bounded Domain and Smooth Losses](../../AAAI2026/ai_safety/an_improved_privacy_and_utility_analysis_of_differentially_p.md)
- [\[NeurIPS 2025\] Differentially Private Bilevel Optimization: Efficient Algorithms with Near-Optimal Rates](../../NeurIPS2025/ai_safety/differentially_private_bilevel_optimization_efficient_algorithms_with_near-optim.md)
- [\[ICML 2026\] PRISM: Gauge-Invariant Tangent-Space Differentially Private LoRA](../../ICML2026/ai_safety/prism_gauge-invariant_tangent-space_differentially_private_lora.md)

</div>

<!-- RELATED:END -->
