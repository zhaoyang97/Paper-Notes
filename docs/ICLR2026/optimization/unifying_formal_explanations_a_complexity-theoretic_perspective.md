---
title: >-
  [论文解读] Unifying Formal Explanations: A Complexity-Theoretic Perspective
description: >-
  [ICLR 2026][优化/理论][可解释AI] 提出统一框架将充分理由和对比理由（局部/全局、概率/非概率）归结为对统一概率值函数的最小化问题，揭示全局值函数具有单调性、子模性/超模性等组合优化关键性质，从而证明全局解释在多项式时间内可计算——即使对应的局部解释是 NP-hard 的。 可解释性的两种基本形式 充分理由（…
tags:
  - "ICLR 2026"
  - "优化/理论"
  - "可解释AI"
  - "计算复杂性"
  - "充分理由"
  - "对比理由"
  - "子模/超模函数"
---

# Unifying Formal Explanations: A Complexity-Theoretic Perspective

**会议**: ICLR 2026  
**arXiv**: [2602.18160](https://arxiv.org/abs/2602.18160)  
**领域**: 优化  
**关键词**: 可解释AI, 计算复杂性, 充分理由, 对比理由, 子模/超模函数

## 一句话总结

提出统一框架将充分理由和对比理由（局部/全局、概率/非概率）归结为对统一概率值函数的最小化问题，揭示全局值函数具有单调性、子模性/超模性等组合优化关键性质，从而证明全局解释在多项式时间内可计算——即使对应的局部解释是 NP-hard 的。

## 研究背景与动机

### 可解释性的两种基本形式

**充分理由（Sufficient Reasons）**：固定特征子集 $S$ 的值后，模型预测不变——回答"什么足以支持这个预测？"

**对比理由（Contrastive Reasons）**：修改特征子集 $S$ 的值后，模型预测改变——回答"什么改变可以翻转预测？"

越小的理由越有解释价值，因此**最小性**是核心追求。

### 从局部到全局，从确定到概率

literature 中的解释问题沿两个维度扩展：

- **局部 vs 全局**：针对单个预测 vs 针对模型在整个输入空间的行为
- **非概率 vs 概率**：绝对保证 vs 以概率 $\delta$ 保证

**已知困难结果**：

| 设置 | 充分理由 | 对比理由 |
|------|----------|----------|
| 决策树（局部非概率） | NP-hard | P |
| 神经网络（局部非概率） | $\Sigma_2^P$-hard | NP-hard |
| 决策树（局部概率） | NP-hard | NP-hard |

这些令人沮丧的复杂性结果引出核心问题：**是否存在某种解释形式在保持有意义保证的同时可以高效计算？**

## 方法详解

### 整体框架

本文把"找解释"全部归约成同一个组合优化问题：在某个值函数 $v$ 下，寻找满足质量门槛的最小特征子集 $S$，即 $\min |S|$ s.t. $v(S) \geq \delta$。充分理由 / 对比理由、局部 / 全局、概率 / 非概率这四个维度的差异，全部被吸收进 $v$ 的具体形式里——一旦确定了 $v$，问题的难易就完全由 $v$ 的组合优化结构（是否单调、是否子模/超模）决定，而本文的核心发现就是：把"局部"换成"全局"会让 $v$ 从毫无结构变得高度规整，从而把 NP-hard 的解释问题拉回多项式时间。

### 关键设计

**1. 统一值函数：把四类解释塞进一个最小化模板**

不同解释形式的本质区别只在"固定哪些特征、问什么概率"。固定子集 $S$ 内的值、问预测保持的概率，就是充分理由；固定补集 $\bar S$ 的值、问预测翻转的概率，就是对比理由。再把单点 $\boldsymbol{x}$ 换成对数据分布求期望，就从局部升级到全局。四个值函数因此一字排开：局部充分 $v_{\text{suff}}^\ell(S) = \Pr_{\boldsymbol{z} \sim \mathcal{D}}(f(\boldsymbol{z}) = f(\boldsymbol{x}) \mid \boldsymbol{z}_S = \boldsymbol{x}_S)$，全局充分 $v_{\text{suff}}^g(S) = \mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}}[v_{\text{suff}}^\ell(S)]$；局部对比 $v_{\text{con}}^\ell(S) = \Pr_{\boldsymbol{z}}(f(\boldsymbol{z}) \neq f(\boldsymbol{x}) \mid \boldsymbol{z}_{\bar S} = \boldsymbol{x}_{\bar S})$，全局对比 $v_{\text{con}}^g(S) = \mathbb{E}_{\boldsymbol{x}}[v_{\text{con}}^\ell(S)]$。统一模板的价值在于：后续所有复杂性结论都可以只针对 $v$ 的性质来证明，一次推导覆盖全部设置。

**2. 三大组合优化性质：用结构换算法**

决定问题难易的不是值函数的语义，而是三条组合性质。单调性 $v(S \cup \{i\}) \geq v(S)$ 意味着加特征只增不减，它让"贪心地逐个删特征"能稳定收敛到子集最小解。超模性 $v(S \cup \{i\}) - v(S) \leq v(S' \cup \{i\}) - v(S')$（$S \subseteq S'$）说边际贡献递增，子模性则相反、边际贡献递减——子模性正是 Wolsey (1982) 那类贪心算法获得对数近似保证的前提。这一步把抽象的解释问题翻译成了组合优化里成熟的"工具能不能用"的问题：有单调就有精确贪心，有子模就有近似保证。

**3. 局部到全局的"相变"：期望抹平了不规则性**

把这三条性质逐一核对四个值函数，得到全文最反直觉的结果——局部值函数在任何设置下都不满足任何一条性质，而对单个预测的概率取数据分布期望后，全局值函数立刻全部变得单调，且充分理由变超模、对比理由变子模，恰好对偶。

| 性质 | 局部充分 $v_{\text{suff}}^\ell$ | 全局充分 $v_{\text{suff}}^g$ | 局部对比 $v_{\text{con}}^\ell$ | 全局对比 $v_{\text{con}}^g$ |
|------|:---:|:---:|:---:|:---:|
| 单调性 | ✗ | ✓ | ✗ | ✓ |
| 超模性 | ✗ | ✓（独立分布） | ✗ | ✗ |
| 子模性 | ✗ | ✗ | ✗ | ✓（独立分布） |

这不是巧合，而是期望运算的平滑效应：单个 $\boldsymbol{x}$ 上添加一个特征可能让概率剧烈跳变（破坏单调与子模），但对整个分布取期望后这些跳变被抹平，规整结构随之浮现。也正是这层平滑，让"模型整体上依赖哪些特征"这种全局问题，反而比"这条预测依赖哪些特征"的局部问题更容易回答。

**4. 两个贪心算法：把性质兑现成可计算解**

有了结构就能给出算法。求子集最小解释用自顶向下的 Algorithm 1：从全特征集出发，每次删掉删除后值函数最高的特征 $j = \arg\max_{i \in S} v(S \setminus \{i\})$，再令 $S \leftarrow S \setminus \{j\}$；单调性保证它在全局设置下精确收敛，而局部设置因缺单调性无此保证。求基数最小解释用自底向上的 Algorithm 2：从空集出发，每次加入增益最大的特征 $j = \arg\max_{i \notin S} v(S \cup \{i\})$，再令 $S \leftarrow S \cup \{j\}$；子模/超模性在此提供近似界——全局对比靠子模拿到 $O(\ln|D|)$ 近似，全局充分则借 Shi et al. (2021) 的有界曲率技术拿到常数因子近似。

## 实验关键数据

本文为纯理论工作，核心贡献是复杂性分析结果。

### 复杂性结果总结

| 解释类型 | 局部（任意模型） | 全局（经验分布） |
|----------|:---:|:---:|
| 子集最小充分理由 | NP-hard（即使决策树） | **P** |
| 子集最小对比理由 | NP-hard（神经网络/树集成） | **P** |
| 基数最小充分理由近似 | 无有界近似 | $O\left(\frac{1}{1-k^f} + \ln\frac{v([n])}{\min_i v(\{i\})}\right)$ |
| 基数最小对比理由近似 | 无有界近似 | $O\left(\ln\frac{v([n])}{\min_i v(\{i\})}\right)$ |

### 近似保证对比

| 设置 | 全局 | 局部 |
|------|------|------|
| 子集最小 | 多项式时间精确解 | NP-hard（即使决策树+均匀分布） |
| 基数最小（对比） | $O(\ln |D|)$ 近似 | 无有界近似（即使 $|D|=1$） |
| 基数最小（充分） | 常数因子近似 | 无有界近似（即使 $|D|=1$） |

### 关键理论发现

1. 虽然非概率全局充分理由是唯一的（Bassan et al. 2024），但概率设置下可有指数多个子集最小全局充分理由：$\Theta(2^n/\sqrt{n})$
2. 全局对比理由的子模性可直接使用 Wolsey (1982) 经典贪心保证
3. 全局充分理由的超模性需要利用 Shi et al. (2021) 的有界曲率技术
4. 经验分布下的近似保证仅关于数据集大小 $|D|$ 对数增长

## 亮点与洞察

1. **统一框架极具优雅性**：将四大维度（充分/对比 × 局部/全局 × 概率/非概率 × 子集/基数最小）统一到一个值函数最小化任务中
2. **局部与全局的"相变"**：局部值函数完全缺乏结构（非单调、非子模、非超模），而全局值函数恰好具备所有有用性质——这不是巧合，而是期望运算的平滑效应
3. **充分与对比的对偶性**：一个超模一个子模，反映了"固定特征保持预测"与"变动特征改变预测"的对称关系
4. **实用意义深远**：全局解释可以高效计算意味着"这个模型在整体上依赖哪些特征"这个问题比"这个预测依赖哪些特征"更容易回答
5. **理论贡献涵盖可解释性谱系**：结果适用于决策树、神经网络和树集成，覆盖从"可解释"到"黑箱"的全谱模型

## 局限性

1. 纯理论工作，缺乏实验验证——贪心算法的实际运行效率和解释质量未知
2. 子模/超模性质仅在特征独立假设下成立，现实中特征往往高度相关
3. 经验分布的近似保证中，数据集大小 $|D|$ 作为隐含常数可能实际上很大
4. 全局解释的语义可能与用户真正关心的局部解释需求不完全匹配
5. 未讨论值函数计算本身的复杂度——对于神经网络，即使是单次推理也可能很慢

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 统一框架和局部/全局性质反差的发现极具原创性
- **实验**: ⭐⭐ — 纯理论工作，无实验支撑
- **写作**: ⭐⭐⭐⭐⭐ — 定义严谨、定理层次分明、证明路线清晰
- **价值**: ⭐⭐⭐⭐⭐ — 为可解释 AI 的复杂性理论奠定了重要基础

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Directional Sheaf Hypergraph Networks: Unifying Learning on Directed and Undirected Hypergraphs](directional_sheaf_hypergraph_networks_unifying_learning_on_directed_and_undirect.md)
- [\[ICLR 2026\] Neural Networks Learn Generic Multi-Index Models Near Information-Theoretic Limit](neural_networks_learn_generic_multi-index_models_near_information-theoretic_limi.md)
- [\[ICML 2026\] Learning Dynamics of Zeroth-Order Optimization: A Kernel Perspective](../../ICML2026/optimization/learning_dynamics_of_zeroth-order_optimization_a_kernel_perspective.md)
- [\[ICML 2025\] Learning Mixtures of Experts with EM: A Mirror Descent Perspective](../../ICML2025/optimization/learning_mixtures_of_experts_with_em_a_mirror_descent_perspective.md)
- [\[ICML 2026\] Rethinking the Flow-Based Gradual Domain Adaptation: A Semi-Dual Optimal Transport Perspective](../../ICML2026/optimization/rethinking_the_flow-based_gradual_domain_adaptation_a_semi-dual_optimal_transpor.md)

</div>

<!-- RELATED:END -->
