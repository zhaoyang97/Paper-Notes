---
title: >-
  [论文解读] Maximum Coverage in Turnstile Streams with Applications to Fingerprinting Measures
description: >-
  [ICML2025][流算法 / 子模优化 / 隐私风险度量][maximum coverage] 首次在 turnstile 流模型（支持任意插入/删除）下给出最大覆盖问题的单遍流算法，空间 $\tilde{O}(d/\varepsilon^3)$、更新时间 $\tilde{O}(1)$，并将其推广到隐私指纹识别（fingerprinting）场景，实验比先前方法快 210×。
tags:
  - "ICML2025"
  - "流算法 / 子模优化 / 隐私风险度量"
  - "maximum coverage"
  - "turnstile streaming"
  - "linear sketch"
  - "fingerprinting"
  - "frequency moment"
  - "submodular maximization"
---

# Maximum Coverage in Turnstile Streams with Applications to Fingerprinting Measures

**会议**: ICML2025  
**arXiv**: [2504.18394](https://arxiv.org/abs/2504.18394)  
**代码**: 无  
**领域**: 流算法 / 子模优化 / 隐私风险度量  
**关键词**: maximum coverage, turnstile streaming, linear sketch, fingerprinting, frequency moment, submodular maximization

## 一句话总结

首次在 turnstile 流模型（支持任意插入/删除）下给出最大覆盖问题的单遍流算法，空间 $\tilde{O}(d/\varepsilon^3)$、更新时间 $\tilde{O}(1)$，并将其推广到隐私指纹识别（fingerprinting）场景，实验比先前方法快 210×。

## 研究背景与动机

**最大覆盖问题（Maximum Coverage）**：给定 $d$ 个集合（来自大小为 $n$ 的全集），选出 $k$ 个集合使其并集覆盖最多元素。经典贪心算法可达到紧的 $(1-1/e)$ 近似比，但需 $O(knd)$ 时间和 $O(nd)$ 空间，在大规模数据上不可行。

**流模型的需求**：
- 已有工作 [BEM17, MV19] 仅支持 **insertion-only** 流（只允许添加元素），不支持删除
- 实际场景（如隐私数据集动态更新）需要 **turnstile 流**——同时支持插入与删除
- 本文首次给出 turnstile 模型下的最大覆盖单遍算法

**指纹识别（Fingerprinting）的应用**：在隐私审计中，需找出 $k$ 个特征使得用户被重标识（re-identification）的风险最大。这可以归约到最大覆盖或子模最大化问题。先前 [GÁC16] 的算法需 $O(nd)$ 空间和 $O(knd)$ 时间。

## 方法详解

### 核心思路：线性草图（Linear Sketch）

将输入表示为 $n \times d$ 矩阵 $\boldsymbol{A}$（$A_{ij} \neq 0$ 表示元素 $i$ 属于集合 $j$）。用随机草图矩阵 $\boldsymbol{S}$（$r \times n$）压缩输入：

$$\boldsymbol{S}(\boldsymbol{A} + c_{ij}) = \boldsymbol{S}\boldsymbol{A} + \boldsymbol{S} c_{ij}$$

线性性保证插入/删除都能高效更新草图，无需存储完整矩阵。

### 最大覆盖算法（Max-Coverage-LS）

**第一步：子采样缩小全集。** 利用 [MV19] 的技术对行做随机子采样，得到 $\boldsymbol{A}'$，使 $\text{OPT} = O(k \log d / \varepsilon^2)$。此步仅损失 $\varepsilon/4$ 因子的近似质量。

**第二步：哈希分桶 + 非零条目采样。** 对 $\boldsymbol{A}'$ 的行用 $t = O(\log(d/\varepsilon))$ 个独立哈希函数分到 $b = O(k \log d / \varepsilon^2)$ 个桶中。每个桶内，最多保留 $O(d \log(1/\varepsilon)/(\varepsilon k))$ 个非零条目。

**第三步：构造小矩阵 $\boldsymbol{A}_*$。** 按随机排列处理行，逐行从存储的非零条目中取出，直到 $\boldsymbol{A}_*$ 含 $\tilde{O}(d/\varepsilon^3)$ 个非零条目为止。

**第四步：贪心求解。** 在 $\boldsymbol{A}_*$ 上运行标准贪心算法（或任意 $1-1/e$ 近似算法），得到最终解。

关键引理保证：大行（$\geq d/k$ 个非零条目）和小行的条目均能被正确恢复。

### 子模最大化框架（Theorem 3）

为解决一般指纹识别，设计了更通用的子模最大化框架：若单调子模函数 $f$ 可被线性草图以 $(1 \pm \gamma)$ 近似估计（空间 $O(s)$），则可在 $O(sk)$ 空间下得到 $(1-1/e-\varepsilon)$ 近似。

### 频率矩补集估计（Theorem 4）

设计新型线性草图估计 $n^p - F_p$（$p \geq 2$），其中 $F_p = \sum_i f_i^p$ 是 $p$ 阶频率矩：

$$\text{草图大小} = \tilde{O}(\gamma^{-2/(p-1)}), \quad \text{近似比} = (1 \pm \gamma^{1/(p-1)})$$

此结果具有独立理论价值，且被用于实例化子模框架以解决一般指纹识别。

## 理论结果总览

| 问题 | 近似比 | 空间 | 更新时间 | 先前最优 |
|------|--------|------|----------|----------|
| 最大覆盖（turnstile） | $(1-1/e-\varepsilon)$ | $\tilde{O}(d/\varepsilon^3)$ | $\tilde{O}(1)$ | 仅 insertion-only |
| 目标指纹识别 | $(1-1/e-\varepsilon)$ | $\tilde{O}(d/\varepsilon^3)$ | $\tilde{O}(1)$ | $O(nd)$ 空间 [GÁC16] |
| 一般指纹识别 | $(1-1/e-\varepsilon)$ | $\tilde{O}(dk^3/\varepsilon^2)$ | $\tilde{O}(k^3/\varepsilon^2)$ | $O(nd)$ 空间 [GÁC16] |
| $n^p - F_p$ 估计 | $(1 \pm \gamma^{1/(p-1)})$ | $\tilde{O}(\gamma^{-2/(p-1)})$ | $\tilde{O}(\gamma^{-2/(p-1)})$ | 新结果 |

**下界匹配**：空间 $\tilde{O}(d/\varepsilon^3)$ 接近 $\Omega(d/\varepsilon^2)$ 下界 [Ass17]，近似比 $(1-1/e)$ 是最优的（假设 P ≠ NP）[Fei98]。

## 实验关键数据

- **目标指纹识别**：相比 [GÁC16] 加速 **49×**，准确率接近
- **一般指纹识别**：相比 [GÁC16] 加速 **210×**，准确率接近
- 在两个不同数据集上验证了实用性
- 一般指纹识别算法还可作为**降维技术**用于 $k$-means 等聚类算法的特征选择，效率提升的同时准确率损失极小

## 亮点与洞察

1. **首个 turnstile 流下的最大覆盖算法**：突破了先前仅支持 insertion-only 流的限制，polylog 更新时间极为高效
2. **线性草图的通用性**：算法本身是线性草图，直接适用于分布式（coordinator 模型）和并行计算场景
3. **$n^p - F_p$ 的新估计器**：频率矩补集的线性草图是独立重要的理论贡献，补充了经典 $F_p$ 估计（AMS99 等）的研究
4. **从理论到应用的完整链条**：最大覆盖 → 子模最大化框架 → 频率矩草图 → 指纹识别，层层递进
5. **实验速度提升显著**（210×），证明了理论算法的实际可用性

## 局限与展望

- 一般指纹识别的空间含 $k^3$ 因子，当 $k$ 较大时开销增大，能否降低对 $k$ 的依赖？
- 近似比中的 $\varepsilon$ 精度与空间的 trade-off（$1/\varepsilon^3$）较陡，是否存在更优 sketch？
- 实验仅验证了指纹识别场景，未在信息检索、影响力最大化等其他经典最大覆盖应用上测试
- 后处理阶段的贪心算法仍需 $O(k \cdot |\boldsymbol{A}_*|)$ 时间，虽然 $|\boldsymbol{A}_*|$ 已被压缩
- 频率矩补集估计的误差形式 $\gamma^{1/(p-1)}$ 随 $p$ 增大而放大，高阶矩的精度有限

## 相关工作与启发

- **[BEM17]** Bateni et al.：insertion-only 流下 $(1-1/e-\varepsilon)$ 近似，空间 $\tilde{O}(d/\varepsilon^3)$，本文以此为基础
- **[MV19]** McGregor & Vu：set-arrival 流下的子采样框架
- **[GÁC16]** Gulyás et al.：指纹识别的贪心算法，本文大幅改进其效率
- **[KNW10]** Kane, Nelson, Woodruff：$L_0$ 草图，$O(1)$ 更新时间
- **[AMS99]** Alon, Matias, Szegedy：频率矩估计的开创性工作
- **[CCF04]** CountSketch：本文使用的频率估计基础工具

## 评分

- 新颖性: ⭐⭐⭐⭐ （首个 turnstile 最大覆盖算法 + 频率矩补集新草图）
- 理论深度: ⭐⭐⭐⭐⭐ （多层嵌套的 sketch 设计与严格证明）
- 实验充分度: ⭐⭐⭐ （仅指纹识别场景，数据集有限）
- 写作质量: ⭐⭐⭐⭐ （结构清晰，从贡献到细节层次分明）
- 实用价值: ⭐⭐⭐⭐ （210× 加速，适用于实时隐私监控）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Improved Generalization Bounds for Transductive Learning by Transductive Local Complexity and Its Applications](improved_generalization_bounds_for_transductive_learning_by_transductive_local_c.md)
- [\[ICML 2026\] Simple Algorithms for Bad Triangle Transversals with Applications to Correlation Clustering](../../ICML2026/learning_theory/simple_algorithms_for_bad_triangle_transversals_with_applications_to_correlation.md)
- [\[ICML 2026\] Optimal Design for Multinomial Logit Model with Applications to Best Assortment Identification](../../ICML2026/learning_theory/optimal_design_for_multinomial_logit_model_with_applications_to_best_assortment_.md)
- [\[ICLR 2026\] An Improved Model-free Decision-estimation Coefficient with Applications in Adversarial MDPs](../../ICLR2026/learning_theory/an_improved_model-free_decision-estimation_coefficient_with_applications_in_adve.md)
- [\[ICML 2026\] Conditional KRR: Injecting Unpenalized Features into Kernel Methods with Applications to Kernel Thresholding](../../ICML2026/learning_theory/conditional_krr_injecting_unpenalized_features_into_kernel_methods_with_applicat.md)

</div>

<!-- RELATED:END -->
