---
title: >-
  [论文解读] POET-X: Memory-efficient LLM Training by Scaling Orthogonal Transformation
description: >-
  [ICML2026][预训练][内存高效训练] POET-X 把训练稳定但又慢又费显存的 POET（正交等价变换、谱保持优化器）做了一整套系统级提速降存：通过输入中心化重构、置换内核加速、块对角批并行、半存储 CNP 与 Triton 融合，相比原版 POET 实现 3× 显存下降、8× 速度提升，让单张 H100 就能预训练 8B~13B 的 LLM，而 AdamW 在同等设置下直接 OOM。
tags:
  - "ICML2026"
  - "预训练"
  - "内存高效训练"
  - "正交等价变换"
  - "谱保持"
  - "稀疏训练"
  - "CUDA内核"
---

# POET-X: Memory-efficient LLM Training by Scaling Orthogonal Transformation

**会议**: ICML2026  
**arXiv**: [2603.05500](https://arxiv.org/abs/2603.05500)  
**代码**: spherelab.ai/poetx （项目主页）  
**领域**: LLM预训练 / LLM效率  
**关键词**: 内存高效训练, 正交等价变换, 谱保持, 稀疏训练, CUDA内核  

## 一句话总结
POET-X 把训练稳定但又慢又费显存的 POET（正交等价变换、谱保持优化器）做了一整套系统级提速降存：通过输入中心化重构、置换内核加速、块对角批并行、半存储 CNP 与 Triton 融合，相比原版 POET 实现 3× 显存下降、8× 速度提升，让单张 H100 就能预训练 8B~13B 的 LLM，而 AdamW 在同等设置下直接 OOM。

## 研究背景与动机
**领域现状**：训练大模型既烧算力又常常不稳定。POET（reParameterized Orthogonal Equivalence Training）把每个权重重参数化为 $\bm{W}_{RP}=\bm{R}\bm{W}_0\bm{P}$——$\bm{W}_0$ 是固定的随机权重，$\bm{R}$、$\bm{P}$ 是可训练的正交矩阵。因为正交变换保持奇异值不变（谱保持），且在 Gaussian 初始化下超球能量可证地小，POET 训练非常稳定，是个有吸引力的优化框架。

**现有痛点**：POET 虽然稳，但显存效率很差、跑得比 Adam 慢得多，根因是它要做大量大规模矩阵乘法。更矛盾的是，POET 优化的正交矩阵本身被约束成稀疏（块对角）结构、**参数效率**很高，但原始实现里这种稀疏性根本没体现到**显存占用**上——它甚至比 AdamW 还费显存，因为要为反向传播存下变换后的权重 $\bm{W}_{RP}$ 等中间激活。结果 POET 难以扩到 3B 以上，实用性受限。

**核心矛盾**：参数效率高 ≠ 显存效率高。POET 卡在「理论上很稀疏、工程上很臃肿」的鸿沟上——稀疏的正交矩阵没被高效地计算和存储。

**本文目标**：把正交等价变换这件事**做到可扩展**，逐一分析 POET 前向/反向里每一步计算的显存和运行时开销，逐个优化，从而把「参数效率」真正兑现成「显存效率」。

**核心 idea**：不改 POET 的数学（保留谱保持、稳定性），只重写它的「算法工程」——把 weight-centric 计算改成 input-centric（matrix-free），用定制 CUDA/Triton 内核消除冗余的矩阵构造、置换和激活存储，让 POET 的显存逼近 LoRA 级、运行时逼近 Adam 级。

## 方法详解

### 整体框架
POET-X 建立在 **block-stochastic POET** 之上（它对权重矩阵各维度的更新覆盖更均衡，即使可训练参数很少也能保证均衡更新，这对显存效率很关键）。每一步把正交矩阵 $\bm{R}_i$ 参数化为「行置换 → 块对角正交矩阵 → 列置换」的夹心结构 $\bm{R}_i=\bm{\Psi}_i^\top\,\mathrm{Diag}(\tilde{\bm{G}}^1_i,\dots)\,\bm{\Psi}_i$，$\bm{P}_i$ 同理，然后周期性地把它们乘进权重 $\bm{W}_i=\bm{R}_i\bm{W}_{i-1}\bm{P}_i$。整篇方法的核心挑战就一句话：**怎么在满足正交约束的前提下，又快又省显存地完成这些乘法**。POET-X 没有引入新的多阶段流水线，而是沿着「一次前向 $\bm{z}=\bm{G}_P^\top\bm{W}\bm{G}_R^\top\bm{x}$」这条固定计算链，从计算形式、置换、块对角构造、正交参数化、激活存储五个维度逐个开刀。下面四个关键设计正是这条链上的四类优化。

### 关键设计

**1. 输入中心化（matrix-free）重构：把权重乘法拆成矩阵-向量乘法，干掉激活存储**

原版 POET 直接对权重操作 $\bm{W}\leftarrow\bm{R}_i\bm{W}\bm{P}_i$（weight-centric），复杂度 $\mathcal{O}(nm^2)$，而且算 $\bm{R}_i$、$\bm{P}_i$ 的梯度都要访问 $\bm{W}$，显存随之上去。受解大规模线性系统的 matrix-free 方法启发，POET-X 把更新改写成 input-centric 形式：左边 $\bm{P}_i^\top\bm{W}^\top\bm{R}_i^\top\bm{x}$ 需要两次矩阵-矩阵乘 + 一次矩阵-向量乘，右边 $\bm{P}_i^\top(\bm{W}^\top(\bm{R}_i^\top\bm{x}))$ 则变成三次矩阵-向量乘，从内到外逐步收缩。这样把 POET 变成一串线性映射，避免存储与权重矩阵绑定的中间激活。难点在于：不同于只有一个正交矩阵 $\bm{R}_i$ 的正交微调，POET-X 在 $\bm{W}$ 左边还多了个 $\bm{P}_i$，算 $\bm{P}_i$ 的梯度仍要访问 $\bm{W}$——这个非平凡之处由后面的置换与检查点设计一起化解。

**2. 置换算子的加速与缩减：用 CUDA 索引映射替代显式矩阵，并把 4 次置换砍成 2 次**

把完整推理写开是 $\bm{z}=\bm{\Phi}_n\bm{G}_P^\top\bm{\Phi}_n^\top\bm{W}\bm{\Phi}_m\bm{G}_R^\top\bm{\Phi}_m^\top\bm{x}$，里面有四个置换矩阵。**加速**上，作者不显式构造置换矩阵，而是实现定制 CUDA 算子做索引映射——例如 $\bm{\Psi}_m\bm{W}\equiv\bm{W}'$ 等价于 $(\bm{W}')_{i,:}=\bm{W}_{\pi_p(i),:}$，只需存一个置换索引集、按规定顺序访问权重即可，前向反向都能用这个双射，最高带来 20× 加速。**缩减**上，作者发现 input-centric 前向里的 4 次置换有 2 次可以提前合并进权重：$\bm{\Phi}_n^\top\bm{W}\bm{\Phi}_m$ 在优化 $\bm{G}_P$、$\bm{G}_R$ 的内循环里 $\bm{W}$ 固定，可预先算好，省掉重复置换，再带来约 1.1~1.8× 提速。

**3. 块对角批并行 + 高效 CNP：不构造大稀疏矩阵，正交参数化只存一半还融合内核**

这一组针对「正交矩阵的构造与正交化」降存提速。**块对角批并行**：原版要先把 $\bm{G}_P=\mathrm{Diag}(\tilde{\bm{G}}^1_P,\dots)$ 这种大而稀疏的块对角矩阵显式拼出来再做乘法，但块对角矩阵的乘法只在各块内发生，根本不必拼全矩阵——POET-X 把每个块当独立矩阵做 batch-wise 乘法，既省显存（最高省 31%）又快（约 2.3×）。**高效 CNP**：POET 用 Cayley-Neumann Parameterization 把斜对称矩阵 $\bm{Q}=-\bm{Q}^\top$ 近似成正交矩阵，$k=3$ 时 $\bm{G}\approx\bm{I}+2\bm{Q}+2\bm{Q}^2+2\bm{Q}^3+\bm{Q}^4$。POET-X 只存斜对称矩阵的上三角（参数量从 $b^2$ 降到 $b(b-1)/2$），把 POET 相关显存直接砍半；再把 CNP 重排成 $\bm{G}\approx2(\bm{Q}+\bm{Q}^2+\bm{Q}^2\!\cdot\!\bm{Q})+\bm{Q}^2\!\cdot\!\bm{Q}^2+\bm{I}$，发现所有下游计算只依赖 $\bm{Q}$ 和 $\bm{Q}^2$——于是用 Triton 内核把这两个张量一次性载入共享内存、就地算高阶项并求和，前向反向都用这种 kernel fusion，带来 2~3× 加速。

**4. 梯度检查点与量化训练：按需重算激活，进一步压到 PEFT 级显存并支持低比特**

简化后的前向是三次矩阵乘 mm1: $\bm{a}=\bm{G}_R^\top\bm{x}$、mm2: $\bm{b}=\bm{W}\bm{a}$、mm3: $\bm{z}=\bm{G}_P^\top\bm{b}$。逐个分析反向需要存哪些激活后发现：算 $\nabla_{\bm{G}_P}$ 需要存中间张量 $\bm{b}\in\mathbb{R}^{N\times m}$；而 mm2、mm1 因为 $\bm{W}$ 无梯度、$\bm{x}$ 是原始输入，几乎不需额外存。据此给出两个变体：$\text{POET-X}_{\text{fast}}$ 走标准 Autograd、存那一份 $\bm{b}$，速度更快；$\text{POET-X}_{\text{mem}}$ 用梯度检查点在反向时按需重算 $\bm{b}$，显存最省。更进一步，因为有了自定义 CUDA 内核，$\text{POET-X}_{\text{mem}}$ 可扩展成 $\text{POET-XQ}$：只存低比特量化权重、用时即时反量化，从不在显存里保留高精度权重的激活，从而支持内存高效的量化训练。

### 损失函数 / 训练策略
POET-X 不改训练目标，纯粹是 POET 优化器的工程实现优化。所有前向/反向（含批并行、CNP、置换）都用定制 Triton/CUDA 内核实现，对 GPU 显存访问和计算做细粒度控制。块大小 $b$（如 256/512）是主要超参，决定可训练参数量与显存的权衡。

## 实验关键数据

### 主实验
单层 profiling 上，相比原版 POET 单次前向+反向延迟从 10.59ms 降到 $\text{POET-X}_{\text{fast}}$ 的 1.38ms、$\text{POET-X}_{\text{mem}}$ 的 1.89ms，整体相对 POET 实现 3× 显存下降、8× 速度提升。Llama-8B 单卡预训练（$L_{\max}=256$、5B tokens）的验证困惑度对比：

| 方法 | 可训练参数(M) | 显存(G) | Val PPL |
|------|--------------|---------|---------|
| AdamW | 2764.47 | 81.03 | 12.69 |
| Muon (Kimi) | 2764.47 | 70.94 | 11.45 |
| APOLLO | 2764.47 | 80.60 | 12.97 |
| GaLore | 2764.47 | 74.50 | 14.88 |
| POET-X (b=256) | **366.64** | 60.58 | 12.76 |
| POET-X (b=512) | 570.06 | **68.52** | **12.05** |

POET-X 用约 1/5~1/8 的可训练参数和显著更低的显存，PPL 还优于 AdamW（12.05 vs 12.69），仅次于 Muon。

### 量化训练对比
$\text{POET-XQ}$（8-bit）在更省显存的同时困惑度明显优于同为 8-bit 的 APOLLO / GaLore：

| 方法 | 参数(M) | 显存(G) | Val PPL |
|------|---------|---------|---------|
| 8-bit APOLLO | 2764.47 | 66.37 | 20.49 |
| 8-bit GaLore | 2764.47 | 66.28 | 17.74 |
| POET-XQ (b=256) | 366.64 | **51.66** | 16.21 |
| POET-XQ (b=512) | 570.06 | 60.65 | **14.78** |

### 关键发现
- **稀疏性终于兑现成显存效率**：原版 POET 比 AdamW 还费显存（栽在存 $\bm{W}_{RP}$ 等激活上），POET-X 的两个变体都降到 PEFT 级显存足迹，这是「单卡训 8B~13B」成为可能的根因。
- **各项内核优化叠加效果显著**：定制置换 CUDA 算子最高 20× 加速、CNP Triton 融合 2~3×、块对角批并行约 2.3× 且省显存、置换缩减约 1.1~1.8×——逐点优化累积成整体 8× 提速。
- **fast vs mem 是计算-显存权衡**：$\text{POET-X}_{\text{fast}}$ 因参数效率高，反向延迟可比肩普通线性层；$\text{POET-X}_{\text{mem}}$ 用重算换最省显存，并且是支持量化训练的前提。

## 亮点与洞察
- **「matrix-free 重构」是核心解锁点**：把 weight-centric 的 $\bm{R}\bm{W}\bm{P}$ 改成 input-centric 的三次矩阵-向量乘，避免存与权重绑定的激活——这个把「先乘矩阵」改成「先作用于向量」的思路，对任何需要左右夹乘大矩阵的算法都有借鉴价值。
- **置换不必造矩阵、只需索引映射**：用 $(\bm{W}')_{i,:}=\bm{W}_{\pi(i),:}$ 这种双射索引代替显式置换矩阵乘法，零额外显存还 20× 提速，是个干净利落、可复用的工程 trick。
- **CNP 重排暴露「只依赖 $\bm{Q}$ 和 $\bm{Q}^2$」**：把多项式重排后发现前向反向的所有高阶项都能从这两个张量算出，于是一次载入共享内存做 kernel fusion——把数学结构和 GPU 内存层级对齐的典型范例。

## 局限与展望
- POET-X 不改 POET 的数学本质，所以它继承了 POET 的归纳偏置（谱保持、固定 $\bm{W}_0$），其性能上限受 POET 框架本身约束；PPL 上仍略逊于 Muon（12.05 vs 11.45）。
- 大量收益来自定制 CUDA/Triton 内核，意味着对硬件和实现高度耦合，在非 NVIDIA 平台或新架构上的可移植性、维护成本是现实问题。
- 论文主打单卡可训 8B~13B 的「显存胜利」，但跨多卡分布式扩展、与张量/流水并行的协同、超长序列下的表现等大规模训练细节展开有限。
- 量化变体 $\text{POET-XQ}$ 困惑度相对全精度仍有明显退化（如 b=512 从 12.05 升到 14.78），低比特下的精度-显存权衡还有打磨空间。

## 相关工作与启发
- **vs 原版 POET**：数学完全一致（谱保持、稳定性不变），POET-X 是纯工程层面的重写——3× 省显存、8× 提速，把 POET 从「3B 以上跑不动」推到「单卡训 13B」。
- **vs AdamW**：AdamW 在单卡 8B 设置直接 OOM，POET-X 用更少参数和更低显存达到更优 PPL，是其有力替代。
- **vs GaLore / APOLLO（低秩/投影类省存优化器）**：它们靠低秩投影压优化器状态，POET-X 走正交稀疏 + 内核优化路线，在全精度和 8-bit 量化下 PPL 都更优、显存更省。
- **vs LoRA（PEFT）**：POET-X 达到 LoRA 级的显存效率，但做的是带谱保持的**预训练**而非微调，且训完 $\bm{R}$、$\bm{P}$ 可合并进权重、无推理开销。

## 评分
- 新颖性: ⭐⭐⭐⭐ 数学不新（沿用 POET），但系统级把稀疏正交变换做到可扩展、matrix-free + 内核融合的组合很扎实
- 实验充分度: ⭐⭐⭐⭐ 单层 profiling + 8B/13B 预训练 + 量化变体 + 逐项内核消融齐全；分布式扩展验证偏少
- 写作质量: ⭐⭐⭐⭐ 五个优化维度脉络清晰、公式和内核分析到位，但工程细节密集、对读者门槛较高
- 价值: ⭐⭐⭐⭐⭐ 让单卡预训练十亿级 LLM 可行，对算力受限的研究者和工程落地价值很高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Scaling with Collapse: Efficient and Predictable Training of LLM Families](../../ICLR2026/llm_pretraining/scaling_with_collapse_efficient_and_predictable_training_of_llm_families.md)
- [\[ACL 2026\] SAGE: Sign-Adaptive Gradient for Memory-Efficient LLM Optimization](../../ACL2026/llm_pretraining/sage_sign-adaptive_gradient_for_memory-efficient_llm_optimization.md)
- [\[ICML 2026\] AC-ODM: Actor–Critic Online Data Mixing for Sample-Efficient LLM Pretraining](ac-odm_actor--critic_online_data_mixing_for_sample-efficient_llm_pretraining.md)
- [\[ICLR 2026\] Beyond URLs: Metadata Diversity and Position for Efficient LLM Pretraining](../../ICLR2026/llm_pretraining/beyond_urls_metadata_diversity_and_position_for_efficient_llm_pretraining.md)
- [\[ICML 2025\] Scaling Inference-Efficient Language Models](../../ICML2025/llm_pretraining/scaling_inference-efficient_language_models.md)

</div>

<!-- RELATED:END -->
