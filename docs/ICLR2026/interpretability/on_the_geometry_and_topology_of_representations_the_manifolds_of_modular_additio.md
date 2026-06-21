---
title: >-
  [论文解读] On The Geometry and Topology of Representations: the Manifolds of Modular Addition
description: >-
  [ICLR 2026][可解释性][模块加法] 本文用「把同频率的一整簇神经元当成一个流形来看」的视角，证明了此前被认为「学到完全不同电路（Clock vs Pizza）」的多种网络，其实在第一层都学到**同一类环面/向量加法圆盘流形**，并用闭式公式 + 拓扑数据分析在数百个网络上统计验证，从而修复了被 Zhong et al. (2023) 当作反例的「普适性假设」。
tags:
  - "ICLR 2026"
  - "可解释性"
  - "模块加法"
  - "机制可解释性"
  - "普适性假设"
  - "拓扑数据分析"
  - "表示流形"
---

# On The Geometry and Topology of Representations: the Manifolds of Modular Addition

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=2olkCiSELH](https://openreview.net/forum?id=2olkCiSELH)  
**代码**: 待确认  
**领域**: 可解释性 / 机制可解释性 / 表示几何  
**关键词**: 模块加法、机制可解释性、普适性假设、拓扑数据分析、表示流形

## 一句话总结
本文用「把同频率的一整簇神经元当成一个流形来看」的视角，证明了此前被认为「学到完全不同电路（Clock vs Pizza）」的多种网络，其实在第一层都学到**同一类环面/向量加法圆盘流形**，并用闭式公式 + 拓扑数据分析在数百个网络上统计验证，从而修复了被 Zhong et al. (2023) 当作反例的「普适性假设」。

## 研究背景与动机
**领域现状**：机制可解释性（mechanistic interpretability）希望把神经网络拆成可理解的「电路」(circuit)，并寄希望于两个支柱假设：**普适性假设**（结构相似、数据相似的网络会学到相似电路）和**流形假设**（表示学习本质是为数据找一个低维流形）。模块加法 $(a+b)\bmod n = c$ 因为是循环群乘法、数据不可线性分、又被研究透彻，成了 toy 可解释性的标准试验台。

**现有痛点**：Zhong et al. (2023) 在 Nanda et al. (2023) 的 transformer 上插值「均匀注意力 ↔ 可学习注意力」，声称网络学到了两种**互不相交**的电路——均匀注意力学 Pizza（向量加法），可学习注意力学 Clock（角度求和），并给出 distance irrelevance / gradient symmetricity 两个指标来区分它们。这等于给普适性假设提供了一个反例：同样的数据、同样的任务，不同架构学到「毫无共性」的电路。

**核心矛盾**：如果这个反例成立，后果很严重——它意味着大模型可能在权重里对同一任务同时学了一大堆互不相交的电路，那么「识别可泛化的可解释原理」这件事在组合上就近乎绝望。问题是：Clock 和 Pizza 真的是两种本质不同的算法吗，还是只是同一结构的不同投影被两套指标人为切开了？

**本文目标**：(1) 用闭式数学刻画这些网络第一层到底学了什么流形；(2) 证明 Clock/Pizza/MLP 等架构在拓扑和几何上等价；(3) 给出能在数千个网络上统计验证的可计算工具。

**切入角度**：作者不再像以往那样逐个解释单个神经元/权重，而是**把属于同一学习表示（同一关键频率 $f$）的所有神经元聚成一簇，当成一个整体实体来看**——这一簇神经元的预激活点集就构成一个**流形**，于是可以搬来拓扑学（Betti 数、持续同调）的工具来刻画它。

**核心 idea**：在 McCracken et al. (2025) 验证过的「simple neuron」模型下，第一层流形的结构**只由两个相位 $(\phi_L,\phi_R)$ 的联合分布决定**；据此可证明流形几乎必然是环面 $T^2$ 或它的线性投影（向量加法圆盘 = pizza），而 Clock 这种二阶角度求和结构在该假设下根本不会自然出现。

## 方法详解

### 整体框架
任务固定为 $n=59$ 的模块加法。所有架构都先用一个共享可学习嵌入矩阵把 $a,b$ 映射到 $E_a,E_b\in\mathbb{R}^{128}$，之后处理方式不同：MLP-Add 直接把 $E_a+E_b$ 送进 MLP；MLP-Concat 把拼接 $E_a\oplus E_b$ 送进 MLP；可学习注意力（被称为 Clock，记为 Attention 1.0）和均匀/常数注意力（被称为 Pizza，记为 Attention 0.0）则先过一层自注意力再进 MLP。

本文的分析管线是：**先用「simple neuron」模型给第一层预激活一个闭式形式 → 用对称性定理把一整簇神经元的预激活矩阵分解出 rank-2（圆盘）或 rank-4（环面）的因子，从而预测流形几何 → 再设计两套可统计的工具（相位对齐分布 PAD + Betti 数分布）在 703 个训练好的网络上实测，验证理论预测的流形确实普遍出现**。其骨架是「理论刻画 → 大规模统计验证」，并非传统串行 pipeline，所以不画流程图，而是用定理和公式讲清。

关键对象是频率簇 $f$ 在第 $\ell$ 层的预激活流形与 logit 流形：

$$M^{\text{pre}}_{\ell,f} := \{ h^{\text{pre}}_{\ell,f}(a,b) : (a,b)\in\mathbb{Z}_n^2 \}, \qquad M^{\text{logit}}_{f} := \{ l_f(a,b) : (a,b)\in\mathbb{Z}_n^2 \}.$$

### 关键设计

**1. Simple neuron 模型 + 一整簇神经元当流形看**

以往工作盯着单个神经元/权重，很难看出不同架构间的共性。本文借用 McCracken et al. (2025) 已验证的事实：第一层神经元绝大多数是「simple neuron」，其预激活是关于 $a$ 和关于 $b$ 两个正弦的线性叠加

$$N(a,b) = \cos(2\pi f a/n + \phi_L) + \cos(2\pi f b/n + \phi_R),$$

即一个 simple neuron 的全部自由度只剩相位对 $(\phi_L,\phi_R)$。作者把所有「关键频率」同为 $f$ 的神经元（用对每个神经元的 $n\times n$ 预激活矩阵做 2D 离散傅里叶变换来判定其主频）聚成一簇，把每个神经元的预激活矩阵拉平后按列堆叠，得到 $n^2\times|\text{cluster}\,f|$ 的**neuron-cluster 预激活矩阵**。这样研究对象就从「单个神经元」升格为「一簇神经元张成的点集 = 流形」，几何/拓扑工具才能用上。

**2. 相位分布决定流形：环面/圆盘二分定理**

模块加法是交换的，直觉上交换 $a,b$ 不应改变输出，因此人们会期望两个相位之间存在某种对称。本文把这个直觉形式化为定理 4.1：设频率簇有 $m\ge2$ 个神经元，矩阵 $X\in\mathbb{R}^{p^2\times m}$ 的元素 $X_{(a,b),i}=\cos(\theta_a+\phi_i^L)+\cos(\theta_b+\phi_i^R)$（$\theta_a=2\pi f a/p$），并假设 $\phi_i^L,\phi_i^R$ 同分布、其联合分布 $\mu_i^{a,b}$ 支撑集有正测度。则几乎必然有两种情况：

- **完美相位相关**（$\phi_i^L\equiv\phi_i^R$）：$X$ 有 rank-2 分解 $X=V^{\text{disc}}W$，其中 $V^{\text{disc}}_{(a,b)}=(\cos\theta_a+\cos\theta_b,\ \sin\theta_a+\sin\theta_b)^\top$，恰好就是图 1 的**向量加法圆盘（pizza）**；
- **相位独立**：$X$ 有 rank-4 分解 $X=V^{\text{torus}}W$，其中 $V^{\text{torus}}_{(a,b)}=(\cos\theta_a,\sin\theta_a,\cos\theta_b,\sin\theta_b)^\top$，恰好编码**环面 $T^2$**。

关键洞察是：圆盘只是环面的线性投影 $(x_1,x_2,x_3,x_4)\mapsto(x_1+x_3,\,x_2+x_4)$，所以**环面是更一般的结构，pizza 圆盘是它的降秩版本**。一个直接推论（Remark 4.2）是：Zhong et al. 主张的 Clock（角度求和、需要二阶交互）在定理 4.1 的假设下**根本不可能出现**——它理论上可行，但不会自然学到。这就把「Clock vs Pizza 是两种不同算法」的论点从根上拆掉了：差别只在相位是否完美相关，而非两类电路。

**3. Phase Alignment Distribution (PAD) + MMD/torus distance 的可统计验证**

定理把「判别流形」化简成了「判别相位是否对齐」，于是只需统计相位分布即可大规模验证。作者定义 **PAD**：一个 $\mathbb{Z}_n\times\mathbb{Z}_n$ 上的分布，采样方式是——随机种子训练一个网络、均匀采一个神经元、返回令该神经元激活最大的输入对 $(a,b)$（相位也可用「激活质心」估计，两种估计给出定性一致的 PAD）。PAD 直观刻画「学到的相位有多频繁落在 $a=b$ 对角线上」，即相位有多对齐。为定量比较不同架构的 PAD，作者用**最大均值差异 MMD**（有可计算的无偏样本估计）度量分布间距离；并补充提出 **torus distance**——点 $(a,b)$ 在环面上到 $a=b$ 线的离散图距离，用其直方图区分模型。这套工具让作者能在 703 个一隐层网络上跑统计，而不是手工解释少数几个网络。

**4. Betti 数分布刻画多层网络的拓扑**

PAD 主要针对一隐层网络；对多层网络，作者转向拓扑数据分析，用持续同调（Ripser 库）估计每层（及 logits）某频率簇神经元集合的 **Betti 数向量** $(\beta_0,\beta_1,\beta_2)$ 的分布：$\beta_0$ 数连通分量、$\beta_1$ 数环、$\beta_2$ 数被曲面包住的空洞。参考值：圆盘是 $(1,0,0)$，圆是 $(1,1,0)$，2-环面是 $(1,2,1)$。用 Betti 数分布就能在统计意义上判断某层结构更像圆盘、环面还是圆，从而验证「不同架构在做拓扑等价的计算」「logit 层普遍收敛到圆环（annulus）」这两个论断。需要留意一个坑：在 logits 处偶尔检出「圆盘」其实多是持续同调难以发现小半径空洞造成的假象。

## 实验关键数据

实验在 703 个训练好的一隐层网络上展开，覆盖 MLP-Add、Attention 0.0 (Pizza)、Attention 1.0 (Clock)、MLP-Concat 四种架构。

### 主实验：第一层表示 vs 参考流形相似度（CKA / RSM）

| 参考流形 | 指标 | MLP-Concat | MLP-Add | Attn 0.0 | Attn 1.0 |
|----------|------|-----------|---------|----------|----------|
| 圆盘（向量加法） | CKA | 0.707 | **0.998** | **0.988** | **0.974** |
| 圆盘（向量加法） | RSM | 0.578 | **0.998** | **0.986** | **0.972** |
| 环面 | CKA | **0.994** | 0.706 | 0.699 | 0.689 |
| 圆（Clock） | CKA | ~0 | ~0 | ~0 | 0.012 |

结论：MLP-Add、Pizza、Clock 三者第一层几乎完全对齐到**圆盘**（CKA≈0.97–0.998），与环面参考的对齐都只有 0.69–0.71；而 MLP-Concat 反过来强对齐**环面**（CKA 0.994）。所有架构第一层与「圆（Clock）」参考的 CKA 都接近 0——即没有任何网络在第一层学到 Zhong et al. 所谓的 Clock 角度求和结构。

### logit 层 vs 参考流形

| 参考流形 | 指标 | MLP-Concat | MLP-Add | Attn 0.0 | Attn 1.0 |
|----------|------|-----------|---------|----------|----------|
| 圆（Clock） | CKA | **0.986** | **0.926** | **0.940** | **0.941** |
| 圆盘 | CKA | ~0 | 0.037 | 0.002 | 0.002 |

四种架构的 logits 全部高度对齐到**圆/圆环**（CKA 0.93–0.99），说明无论第一层是圆盘还是环面，最终都收敛到同一个 logit 流形。

### PAD / MMD 与关键发现
- PAD 直方图显示 MLP-Add、Attention 0.0、Attention 1.0 都高度集中在 $a=b$ 对角线，MLP-Concat 则几乎沿对角线均匀铺开（off-diagonal）。
- Attention 0.0 与 1.0 的 PAD 在 MMD 下极其接近（两种相位估计下分别为 **0.0237** 和 **0.0181**），统计显著性 $p\approx0$；MLP-Add 居中靠近二者，MLP-Concat 与所有人都强烈分离。这直接说明被声称「学不同电路」的 Pizza 与 Clock 其实是几乎相同的分布。
- Betti 数结果：MLP-Add、Attention 0.0、Attention 1.0 在拓扑上等价；MLP-Concat 看似不同，实则**更高效**——环面本身已经带有把正确答案投到 logits 所需的「洞」，过一次非线性即可，故只是同一类计算的低开销实现。
- 后 ReLU 激活沿 $a=b$ 对角线集中、并随 $|a-b|$ 平滑衰减——这个「对角依赖」此前被 Zhong et al. 当作 Pizza 的**定义性特征**，本文发现它在 MLP-Add 和 Clock 里同样出现，进一步说明所谓 Clock/Pizza 之分不成立。

## 亮点与洞察
- **把「电路解释」从单神经元抬升到「流形」层面**：聚簇 + 2D-DFT 定频 + 闭式因子分解，这套「整簇当一个实体」的视角是可迁移的——它让原本只能逐元素解释的机制问题，变成了可以用拓扑/几何工具批量统计的问题。
- **用一条「圆盘是环面的线性投影」把对立解释统一**：最 aha 的地方在于，Clock/Pizza/MLP 的差异被压缩成「相位是否完美相关」这一个二元开关，而非两套独立算法；普适性假设由此被「救」了回来。
- **把流形判别化简为相位分布判别**，再配 MMD/Betti 这类有可计算估计的统计量，使「在数百上千个网络上验证」成为可能，而不是手工解释几个网络——这是方法论上的关键提速。
- 提出的 torus distance 和 PAD 都是轻量、可复用的「表示相似性」度量，可推广到其他群运算/toy 任务的电路比较。

## 局限与展望
- 全部实验固定在 $n=59$ 的循环群模块加法这一 toy 设定，是否能外推到更一般的群运算、真实大模型的子任务仍是开放问题。
- 核心结论依赖「simple neuron 模型 + 相位同分布、支撑集正测度」这组假设；虽然在第一层经验上几乎总成立，但后续层会出现 degree-1 与 degree-2 正弦的混合，定理 4.1 的干净二分主要刻画第一层。
- 持续同调对**小半径空洞**不敏感，会在 logits 处误判出「圆盘」，作者靠手工复核纠正——大规模自动化时这类拓扑估计噪声需要更稳健的处理。
- 文章主要分析一隐层网络的 PAD；多层网络只用 Betti 分布间接刻画，逐层如何「迭代旋转+线性投影」最终到 logit 圆环的细粒度过程仍可进一步展开。

## 相关工作与启发
- **vs Zhong et al. (2023)**：他们用 distance irrelevance / gradient symmetricity 两个指标把网络切成 Clock 与 Pizza 两类，断言学到不相交电路。本文从相位分布的闭式刻画出发，证明这两类在第一层是同一圆盘流形的不同投影、PAD 在 MMD 下几乎重合，从而否定了「不同架构学不同电路」的反例地位。
- **vs Nanda et al. (2023)**：他们把所有层的神经元都建模为 degree-2 三角多项式来解释 grokking。本文沿用 McCracken et al. (2025) 的修正——第一层主要是 degree-1 的 simple neuron，后层才需 degree-2——并在此基础上推出流形的闭式结构。
- **vs McCracken et al. (2025)**：他们用抽象证明 MLP 与 transformer 收敛到一个近似中国剩余定理的分治算法；本文接力，给出第一层表示流形（环面/圆盘）的精确闭式，并用 TDA 把「表示几何普适」这件事在统计上量化验证。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 把对立的 Clock/Pizza 解释用「环面 ↔ 其线性投影」统一，并首次指出环面表示的存在，视角新。
- 实验充分度: ⭐⭐⭐⭐ 703 个网络 + CKA/RSM/PAD/MMD/Betti 多工具交叉验证扎实，但限于 $n=59$ 的单一 toy 任务。
- 写作质量: ⭐⭐⭐⭐ 定理与统计验证衔接清晰，部分拓扑/相位记号较密、对非专业读者门槛偏高。
- 价值: ⭐⭐⭐⭐⭐ 直接修复了被当作普适性假设反例的关键案例，对机制可解释性的方法论有较强示范意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Temporal Geometry of Deep Networks: Hyperbolic Representations of Training Dynamics for Intrinsic Explainability](temporal_geometry_of_deep_networks_hyperbolic_representations_of_training_dynami.md)
- [\[ICLR 2026\] Mixture of Cognitive Reasoners: Modular Reasoning with Brain-Like Specialization](mixture_of_cognitive_reasoners_modular_reasoning_with_brain-like_specialization.md)
- [\[ICLR 2026\] The Geometry of Reasoning: Flowing Logics in Representation Space](the_geometry_of_reasoning_flowing_logics_in_representation_space.md)
- [\[ICLR 2026\] Diagnosing Generalization Failures from Representational Geometry Markers](diagnosing_generalization_failures_from_representational_geometry_markers.md)
- [\[ICLR 2026\] Cross-Modal Redundancy and the Geometry of Vision-Language Embeddings](cross-modal_redundancy_and_the_geometry_of_vision-language_embeddings.md)

</div>

<!-- RELATED:END -->
