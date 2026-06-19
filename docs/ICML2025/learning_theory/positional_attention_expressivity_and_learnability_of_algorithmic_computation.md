---
title: >-
  [论文解读] Positional Attention: Expressivity and Learnability of Algorithmic Computation
description: >-
  [ICML2025][others (Transformer 理论 / 算法推理)][注意力机制] 提出 **Positional Transformer**——注意力权重仅由位置编码决定、与输入数据无关的 Transformer 变体，证明其保持了与 MPC 并行计算模型等价的表达力（仅增加 $O(\log n)$ 深度代价），并在算法任务上展现出显著更优的分布外泛化能力。
tags:
  - "ICML2025"
  - "others (Transformer 理论 / 算法推理)"
  - "注意力机制"
  - "Transformer"
  - "算法执行"
  - "并行计算模型"
  - "泛化界"
---

# Positional Attention: Expressivity and Learnability of Algorithmic Computation

**会议**: ICML2025  
**arXiv**: [2410.01686](https://arxiv.org/abs/2410.01686)  
**代码**: 待确认  
**领域**: others (Transformer 理论 / 算法推理)  
**关键词**: Positional Attention, Transformer 表达力, 算法执行, 并行计算模型, 泛化界

## 一句话总结

提出 **Positional Transformer**——注意力权重仅由位置编码决定、与输入数据无关的 Transformer 变体，证明其保持了与 MPC 并行计算模型等价的表达力（仅增加 $O(\log n)$ 深度代价），并在算法任务上展现出显著更优的分布外泛化能力。

## 研究背景与动机

Transformer 在执行算法任务（排序、前缀和、最小值等）方面展现了强大能力，近期研究将其与并行计算模型（如 MPC）建立了等价关系。

一个关键观察：许多并行算法中，处理器间的**通信模式仅依赖处理器编号（位置），与被处理的数据无关**。例如二叉树归约求最小值时，通信拓扑是固定的二叉树结构。

基于此，作者提出一个自然问题：如果将 Transformer 的注意力权重限制为仅依赖位置编码，是否仍保持足够的表达力？这种约束能否带来泛化优势？

## 方法详解

### Positional Transformer 架构

标准 Transformer 第 $\ell$ 层的注意力矩阵由输入计算：

$$A^{(\ell,h)}(X) = \mathrm{softmax}\left((X W_Q^{(\ell,h)}) \cdot (X W_K^{(\ell,h)})^\top\right)$$

**Positional Transformer** 将注意力权重解耦为仅依赖位置编码 $P$：

$$A^{(\ell,h)} = \mathrm{softmax}\left((P W_Q^{(\ell,h)}) \cdot (P W_K^{(\ell,h)})^\top\right)$$

其中 $P \in \mathbb{R}^{n \times d_P}$ 是固定的位置编码矩阵，**跨层不变**。Value 矩阵仍由前一层输出计算，因此信息的**聚合方式**固定而**被聚合内容**动态更新。每层结构：

$$F^{(\ell)}(X) = \Phi^{(\ell)}\left(\left(\bigoplus_{h=1}^{H} A^{(\ell,h)} X W_V^{(\ell,h)}\right) W_O^{(\ell)} \oplus X\right)$$

### 表达力结果（Theorem 5.1）

**核心定理**：对于 $R$ 轮 MPC（$N$ 台机器、本地内存 $s$），存在具有 $n = N+1$ 个节点、$2R\lceil \log N \rceil$ 层、$2s$ 个注意力头的 Positional Transformer 可以任意精度逼近该 MPC 实例。

证明分两步：

1. 引入代理模型 **PCOC**（Parallel Computation with Oracle Communication），该模型仅允许在静态网络上通信，利用 Beneš 网络模拟 MPC 的动态通信
2. 证明 Positional Transformer 可模拟 PCOC：注意力机制模拟节点间通信，MLP 通过万能逼近定理实现本地计算

**推论（Remark 5.1）**：标准 Transformer（$N$ 节点、$L$ 层）可被 Positional Transformer（$O(N^2)$ 节点、$O(L \log N)$ 层）模拟。额外的对数深度代价是静态通信限制的固有代价。

### 泛化界（Theorem 6.1）

对有界范数的 $L$ 层 $H$ 头 Positional Transformer 函数类 $\mathcal{F}$，在 $m$ 个样本上，以概率 $\geq 1-\delta$，对所有 $f \in \mathcal{F}$：

$$|\mathsf{risk}(f) - \widehat{\mathsf{risk}}(f)| \leq \tilde{O}\left((H L_\sigma B_2)^{cL} B_{2,1} \sqrt{\frac{\log(Hdmn)}{m}} + \sqrt{\frac{\log(1/\delta)}{m}}\right)$$

**关键优势**：与标准 Transformer 的泛化界相比，**不含** $B_{QK}^{O(L)}$ 项。标准 Transformer 因注意力权重依赖输入，其泛化界会指数依赖 Query-Key 矩阵的谱范数。Positional Transformer 消除了这一因素。

**权衡**：某些任务中 Positional Transformer 可能需要更多层（如 $O(\log n)$），这会通过 $(H L_\sigma B_2)^{cL}$ 因子增加样本复杂度。是否获益取决于具体任务。

## 实验关键数据

### 五类算法任务

累积和 (Cumulative Sum)、累积最小值 (Cumulative Min)、累积中位数 (Cumulative Median)、排序 (Sorting)、累积最大子数组和 (Cumulative Max Subarray Sum)。

### 分布内性能（Figure 2）

- 输入长度 $n=8$，训练样本从 5K 到 50K
- **两种架构在所有任务上 loss 均随样本增加单调下降**，验证可学习性
- Positional Transformer 与标准 Transformer 分布内性能相当

### k-hop 归纳头任务（Figure 3）

- 标准 Transformer 仅需 3 层即接近零 loss
- Positional Transformer 需要更多层才能匹配，**验证了理论上的额外对数深度代价**
- 该任务天然需要数据依赖的动态通信，不符合 positional attention 的动机

### 分布外泛化（Figure 4 — 核心结果）

训练范围 $[-2, 2]$，测试扩展至 $[-2c, 2c]$（$c > 1$）：

| 任务 | 标准 Transformer OOD | Positional Transformer OOD |
|------|---------------------|---------------------------|
| 累积和 | loss 随 $c$ 增大急剧上升 | **loss 保持稳定** |
| 累积最小值 | 显著退化 | **几乎不变** |
| 累积中位数 | 退化 | **稳定** |
| 排序 | 退化 | **稳定** |
| 最大子数组和 | 退化 | **稳定** |

Positional Transformer 在所有五个算法任务上**OOD 泛化显著优于标准 Transformer**。

### k-hop 归纳头 OOD（Figure 5）

- **两种架构均表现不佳**
- 验证了该任务需要动态通信的假说

### 混合类型输入实验（Figure 6）

输入包含文本类别标签和数值，测试条件推理 + 模式匹配 + 聚合计算：

- 训练数值范围 $[0, 5]$，测试扩展至 $[0, 5c]$
- Positional Transformer **在 min、sum、multi-task(min+max) 三种聚合操作上均优于标准 Transformer**
- 甚至优于 fine-tuned GPT-2 的 OOD 性能

## 亮点与洞察

- **算法对齐假说**的实证支持：当目标算法的通信拓扑与输入数据无关时，positional attention 的归纳偏置与任务天然对齐，带来更好的泛化
- **理论-实验闭环**：表达力等价 → 可学习性保证 → OOD 优势，三层递进论证
- **简洁而深刻的架构改动**：仅将 Q/K 输入从 $X$ 换为 $P$，就消除了泛化界中的指数依赖项
- 通过 PCOC 代理模型优雅地桥接了静态通信限制与 MPC 的动态通信

## 局限与展望

- **固定位置编码**限制了对更长输入的泛化（长度泛化仍是开放问题）
- OOD 理论分析不够精细，现有 OOD 泛化界不足以刻画两种架构的差异
- Positional Transformer 在需要**动态通信**的任务（如归纳头）上表现不佳
- 模拟标准 Transformer 需要 $O(N^2)$ 节点的二次开销（可能存在更高效策略）
- 实验仅涉及数值/简单文本算法任务，未探索更复杂的推理（如图算法、程序执行）

## 相关工作与启发

- **Sanford et al., 2024**：证明标准 Transformer 可模拟 MPC（本文直接构建于此上）
- **Edelman et al., 2022**：标准 Transformer 的泛化界（本文改进了 QK 范数依赖）
- **Engelmayer et al., 2023**：使用并行算法中间标签指导训练
- **Kazemnejad et al., 2023**：研究不同位置编码对任务的影响
- 启发：positional attention 的思想可推广到 GNN 领域——在消息传递拓扑固定时，图注意力权重可仅依赖结构位置

## 评分

- 新颖性: ⭐⭐⭐⭐ (位置注意力概念简洁但洞察深刻，PCOC 代理模型巧妙)
- 实验充分度: ⭐⭐⭐⭐ (5 个算法任务 + 归纳头 + 混合输入，ID/OOD 全面覆盖)
- 写作质量: ⭐⭐⭐⭐⭐ (理论-实验-直觉三位一体，Figure 1 极为清晰)
- 价值: ⭐⭐⭐⭐ (为理解注意力机制中位置与内容的角色提供了重要理论框架)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] On the Learnability of Test-Time Adaptation: A Recovery Complexity Perspective](../../ICML2026/learning_theory/on_the_learnability_of_test-time_adaptation_a_recovery_complexity_perspective.md)
- [\[ICLR 2026\] Algorithmic Guarantees for Distilling Supervised and Offline RL Datasets](../../ICLR2026/learning_theory/algorithmic_guarantees_for_distilling_supervised_and_offline_rl_datasets.md)
- [\[ICML 2025\] Improved Generalization Bounds for Transductive Learning by Transductive Local Complexity and Its Applications](improved_generalization_bounds_for_transductive_learning_by_transductive_local_c.md)
- [\[ICML 2025\] Near Optimal Best Arm Identification for Clustered Bandits](near_optimal_best_arm_identification_for_clustered_bandits.md)
- [\[ICML 2025\] On Fine-Grained Distinct Element Estimation](on_fine-grained_distinct_element_estimation.md)

</div>

<!-- RELATED:END -->
