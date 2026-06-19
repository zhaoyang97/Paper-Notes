---
title: >-
  [论文解读] Efficient Optimization with Orthogonality Constraint: a Randomized Riemannian Submanifold Method
description: >-
  [ICML2025][正交约束优化] 提出随机黎曼子流形下降方法 (RSDM)，通过将每步更新限制在随机低维子流形上，将正交约束优化中 retraction 操作的复杂度从 $O(np^2)$ 降至 $O(r^3)$，同时保持与全空间黎曼梯度下降相匹配的总计算复杂度。 正交约束优化 $\min_{X \in \mathbb{…
tags:
  - "ICML2025"
  - "正交约束优化"
  - "Riemannian优化"
  - "Stiefel流形"
  - "随机子流形"
  - "坐标下降"
---

# Efficient Optimization with Orthogonality Constraint: a Randomized Riemannian Submanifold Method

**会议**: ICML2025  
**arXiv**: [2505.12378](https://arxiv.org/abs/2505.12378)  
**代码**: [andyjm3/RSDM](https://github.com/andyjm3/RSDM)  
**领域**: 其他/优化  
**关键词**: 正交约束优化, Riemannian优化, Stiefel流形, 随机子流形, 坐标下降

## 一句话总结

提出随机黎曼子流形下降方法 (RSDM)，通过将每步更新限制在随机低维子流形上，将正交约束优化中 retraction 操作的复杂度从 $O(np^2)$ 降至 $O(r^3)$，同时保持与全空间黎曼梯度下降相匹配的总计算复杂度。

## 研究背景与动机

正交约束优化 $\min_{X \in \mathbb{R}^{n \times p}: X^\top X = I_p} F(X)$ 广泛出现在 PCA、深度网络训练、大模型微调等场景。黎曼优化是求解此类问题的标准框架，但其核心操作——retraction（将切空间更新映射回流形）——计算代价至少为 $O(np^2)$（涉及 QR 分解、极分解、Cayley 变换等非标准线性代数运算），在大规模问题中成为严重瓶颈。

已有的坐标下降方法（如仅更新少量行/列）要么在 GPU 上迭代次数过多导致运行时间差，要么涉及难以求解的子问题。Infeasible 方法虽避免了 retraction，但不严格保持正交约束。本文的动机是：**能否在每次迭代中只在低维子流形上做 retraction，从根本上降低代价，同时保持收敛性？**

## 方法详解

### 核心框架：正交群参数化 + 随机子流形

**关键思想**：利用正交群 $\mathcal{O}(n)$ 对 Stiefel 流形 $\mathrm{St}(n,p)$ 的传递性，将更新参数化为 $X_{k+1} = U_k X_k$，其中 $U_k \in \mathcal{O}(n)$。进而将 $U_k$ 限制在由随机正交矩阵 $P_k$ 定义的 $r$ 维子流形上：

$$U_k(Y) = P_k^\top \begin{bmatrix} Y & 0 \\ 0 & I_{n-r} \end{bmatrix} P_k, \quad Y \in \mathcal{O}(r)$$

这样，原本在 $\mathcal{O}(n)$ 上的优化被转化为在 $\mathcal{O}(r)$ 上的小规模优化，retraction 代价从 $O(np^2)$ 降至 $O(r^3)$。

### 算法流程 (RSDM)

每次迭代：
1. **采样**随机正交矩阵 $P_k \in \mathcal{O}(n)$
2. **计算子流形梯度**：$\mathrm{grad}\,\tilde{F}_k(I_r) = P_k(r) \cdot \mathrm{grad}\,F_k(I_n) \cdot P_k(r)^\top$
3. **子流形更新**：$Y_k = \mathrm{Retr}_{I_r}(-\eta\,\mathrm{grad}\,\tilde{F}_k(I_r))$（仅 $r \times r$ retraction）
4. **迭代更新**：$X_{k+1} = U_k(Y_k) X_k$

### 两种采样策略

| 策略 | 采样方式 | 每迭代复杂度 | 特点 |
|------|---------|------------|------|
| **均匀正交采样 (RSDM-O)** | 从 Haar 测度采样 $P_k(r) \in \mathbb{R}^{r \times n}$ | $O(npr)$ | 高概率收敛界更紧 |
| **均匀置换采样 (RSDM-P)** | 无放回采样 $r$ 个索引 | $O(nr^2)$ | 采样更快，但高概率界多 $\binom{n}{r}$ 因子 |

**有趣联系**：当 $r=2$ 且用置换采样时，RSDM 退化为经典的 Givens 旋转坐标下降，因此本方法严格推广了黎曼坐标下降。

### 收敛性保证

**期望收敛**（两种采样一致）：

- 非凸：$\min_i \mathbb{E}[\|\mathrm{grad}\,F(X_i)\|^2] \leq O\!\left(\frac{n^2}{r^2 k}\right)$
- PL 条件下线性收敛：$O\!\left(\exp\!\left(-\frac{r^2}{n^2}k\right)\right)$

**总复杂度**：迭代数 $O(n^2 r^{-2} \epsilon^{-2})$ × 每步 $O(nr^2)$ = $O(n^3 \epsilon^{-2})$，在 $p = \Theta(n)$ 时匹配 RGD 的 $O(np^2 \epsilon^{-2})$。

### 推广能力

方法可直接推广至正交群的商流形（Grassmann 流形、Flag 流形等），因为算法完全在 $\mathcal{O}(n)$ 上定义。

## 实验关键数据

### 主实验

在 Procrustes 问题和 PCA 问题上进行基准测试，基线包括 RGD、RCD、TSD、PCAL、Landing。

| 实验 | 设置 | RSDM 表现 |
|------|------|----------|
| Procrustes | 多种 $(n,p)$ | 与最优基线竞争性持平 |
| PCA | 多种 $(n,p)$ | **收敛速度最快** |
| PCA $(2000,1500)$ | 变化 $r$ | 对 $r$ 的选择鲁棒 |
| PCA $(2000,1500)$ | 变化随机种子 | 对随机性鲁棒 |

### 消融实验

| 消融因素 | 发现 |
|---------|------|
| 子流形维度 $r$ | $r$ 从小到大均有效，存在效率-精度平衡点 |
| RSDM-O vs RSDM-P | 正交采样实际运行更稳定，置换采样每步更快 |
| 不同 retraction | 默认 QR-based；方法对 retraction 类型不敏感 |
| 随机种子 | RSDM 相对 RGD 的优势在不同种子下稳定保持 |

### 关键发现

- 在 $p$ 接近 $n$ 的大规模问题（retraction 代价主导）中，RSDM 相比 RGD **运行时间优势显著**
- **RSDM-P** 每步更快但需要更多迭代；**RSDM-O** 每步稍慢但收敛更好
- 理论预测的 $O(n^2/r^2)$ 缩放关系在实验中得到验证

## 亮点与洞察

1. **优雅的参数化**：通过正交群传递作用 + 块对角嵌入，将流形上的大规模 retraction 问题完美分解为小规模问题
2. **统一框架**：坐标下降（$r=2$）是特例，首次给出了从坐标下降到全空间梯度下降的连续谱
3. **统一的分析**：正交采样和置换采样在期望收敛率上完全一致（$r(r-1)/n(n-1)$），区别仅体现在高概率界
4. **可扩展性**：方法自然推广到 Grassmann、Flag 等商流形，无需修改算法
5. **实际有效**：PyTorch 实现，GPU 友好，代码开源

## 局限与展望

1. **总复杂度匹配仅在 $p = \Theta(n)$ 时**：当 $p \ll n$ 时，retraction 本身不是瓶颈，RSDM 的优势减弱
2. **高概率界中置换采样的 $\binom{n}{r}$ 因子**：理论上很可能是松的，但目前无法改进
3. **子流形维度 $r$ 的选择**缺乏自适应策略，目前依赖手动设置
4. **未讨论二阶/方差缩减/动量**版本的 RSDM，可能进一步加速收敛
5. 实验主要在合成任务和 PCA 上验证，**缺少大规模深度学习场景**（如正交约束微调 LLM）的实验

## 相关工作与启发

- **Stiefel 流形优化教材**：Absil et al. (2008), Boumal (2023) — 本文方法论基础
- **坐标下降**：Shalit & Chechik (2014), Han et al. (2024a), Yuan (2023) — RSDM 的 $r=2$ 特例
- **Infeasible 方法**：Gao et al. (2019), Ablin & Peyré (2022) — 放弃正交可行性换取效率
- **欧氏随机子空间方法**：Kozak et al. (2021) — RSDM 的直接灵感来源，本文将其推广到流形
- **正交约束微调**：Qiu et al. (2023), Liu et al. (2024) — 潜在应用场景
- **启发**：将"大问题分解到随机子空间"的思路可能适用于其他流形（如固定秩矩阵、正定锥）

## 评分

- 新颖性: ⭐⭐⭐⭐ — 正交群参数化+随机子流形是新颖且自然的组合
- 实验充分度: ⭐⭐⭐ — 合成任务验证充分，但缺少大规模实际应用
- 写作质量: ⭐⭐⭐⭐⭐ — 理论推导完整，框架清晰，包含统一性分析
- 价值: ⭐⭐⭐⭐ — 为大规模正交约束优化提供了可扩展的新工具

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Exploiting Similarity for Computation and Communication-Efficient Decentralized Optimization](exploiting_similarity_for_computation_and_communication-efficient_decentralized_.md)
- [\[ICML 2025\] Provably Cost-Sensitive Adversarial Defense via Randomized Smoothing](provably_cost-sensitive_adversarial_defense_via_randomized_smoothing.md)
- [\[ICML 2025\] Randomized Dimensionality Reduction for Euclidean Maximization and Diversity Measures](randomized_dimensionality_reduction_for_euclidean_maximization_and_diversity_mea.md)
- [\[AAAI 2026\] GDBA Revisited: Unleashing the Power of Guided Local Search for Distributed Constraint Optimization](../../AAAI2026/others/gdba_revisited_unleashing_the_power_of_guided_local_search_for_distributed_const.md)
- [\[ACL 2025\] TACLR: A Scalable and Efficient Retrieval-Based Method for Industrial Product Attribute Value Identification](../../ACL2025/others/taclr_a_scalable_and_efficient_retrieval-based_method_for_industrial_product_att.md)

</div>

<!-- RELATED:END -->
