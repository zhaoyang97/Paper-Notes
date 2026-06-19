---
title: >-
  [论文解读] Exploring Large Action Sets with Hyperspherical Embeddings using von Mises-Fisher Sampling
description: >-
  [ICML2025][强化学习][大动作空间探索] 提出 vMF-exp，通过在超球面上采样 von Mises-Fisher 分布向量再做最近邻检索，实现对大规模动作集（百万级）的可扩展探索，理论证明在均匀分布假设下渐近等价于 Boltzmann 探索，并成功部署于 Deezer 音乐推荐系统。 核心问题 在强化学习中…
tags:
  - "ICML2025"
  - "强化学习"
  - "大动作空间探索"
  - "von Mises-Fisher 分布"
  - "超球面嵌入"
  - "Boltzmann 探索"
  - "推荐系统"
---

# Exploring Large Action Sets with Hyperspherical Embeddings using von Mises-Fisher Sampling

**会议**: ICML2025  
**arXiv**: [2507.00518](https://arxiv.org/abs/2507.00518)  
**代码**: [deezer/vMF-exploration](https://github.com/deezer/vMF-exploration)  
**领域**: 强化学习  
**关键词**: 大动作空间探索, von Mises-Fisher 分布, 超球面嵌入, Boltzmann 探索, 推荐系统

## 一句话总结

提出 vMF-exp，通过在超球面上采样 von Mises-Fisher 分布向量再做最近邻检索，实现对大规模动作集（百万级）的可扩展探索，理论证明在均匀分布假设下渐近等价于 Boltzmann 探索，并成功部署于 Deezer 音乐推荐系统。

## 研究背景与动机

### 核心问题

在强化学习中，当动作空间非常大（如推荐系统中的百万级歌曲目录）时，如何高效地进行动作探索是一个关键难题。现有方法面临三重困境：

**随机/ε-贪心探索**：可扩展但忽略嵌入信息，不满足顺序保持性（P3），对大量不相关动作赋予等概率

**Boltzmann 探索（B-exp）**：满足顺序保持和不受限半径，但需计算所有 $n$ 个动作的 softmax 概率，复杂度 $\mathcal{O}(n)$ 不可扩展

**截断 Boltzmann 探索（TB-exp）**：通过 ANN 先检索 $m \ll n$ 个候选再做 softmax（即 Wolpertinger 架构），可扩展但人为限制了探索半径

### 三个期望性质

论文形式化了理想探索策略应同时满足的三个性质：

- **P1 可扩展性**：采样时间不超过 ANN 检索时间（关于 $n$ 的亚线性复杂度）
- **P2 不受限半径**：所有动作的探索概率均非零，不被设计限制在特定邻域内
- **P3 顺序保持性**：$\langle V, X_i \rangle > \langle V, X_j \rangle \Rightarrow P(i|V) > P(j|V)$，即嵌入相似度越高被选中概率越大

现有方法均无法同时满足 P1+P2+P3，这是本文的核心动机。

### 应用场景

以 Deezer 等音乐流媒体的"灵感歌单"推荐为例：用户选择一首歌后，系统需从百万级歌曲中逐步推荐下一首。RL 框架允许根据用户反馈（点赞/跳过）动态调整策略，但需要高效探索整个候选空间。

## 方法详解

### 问题设定

- $n$ 个动作，每个动作 $i$ 用单位范数嵌入向量 $X_i \in \mathcal{S}^{d-1}$ 表示
- 状态向量 $V \in \mathcal{S}^{d-1}$（同样归一化到超球面）
- 可用 ANN 搜索引擎在亚线性时间内检索最近邻

### vMF-exp 算法（两步）

**Step 1**：给定状态向量 $V$，从 von Mises-Fisher 分布 $\text{vMF}(\kappa, V)$ 中采样一个扰动向量 $\tilde{V}$：

$$f_{\text{vMF}}(\tilde{V} | \kappa, V, d) = C_d(\kappa) \cdot e^{\kappa \langle V, \tilde{V} \rangle}$$

其中 $C_d(\kappa) = \frac{\kappa^{d/2-1}}{(2\pi)^{d/2} I_{d/2-1}(\kappa)}$，$I_{d/2-1}$ 是修正 Bessel 函数。

**Step 2**：用 ANN 搜索引擎检索 $\tilde{V}$ 在嵌入空间中的最近邻动作 $i_{\tilde{V}}^\star$ 进行探索。

### 超参数 $\kappa$ 的作用

- $\kappa = 0$：vMF 退化为超球面均匀分布，等价于随机探索
- $\kappa$ 越大：采样越集中在 $V$ 附近，探索范围缩小
- $\kappa$ 越小：采样越分散，探索范围扩大

### 为什么有效——Voronoï 镶嵌视角

每个动作 $X_i$ 在超球面上拥有一个 Voronoï 胞腔 $\mathcal{S}_{\text{Voronoï}}(X_i | \mathcal{X}_n)$。vMF-exp 选中动作 $i$ 的概率等于 vMF 分布在该胞腔上的积分：

$$P_{\text{vMF-exp}}(i | V, \mathcal{X}_n, \kappa) = \int_{\tilde{V} \in \mathcal{S}_{\text{Voronoï}}(X_i)} f_{\text{vMF}}(\tilde{V} | \kappa, V, d) \, d\tilde{V}$$

这个概率由两个因素决定：(1) $X_i$ 与 $V$ 的相似度（vMF 密度在该区域的平均值）；(2) Voronoï 胞腔面积（反映该动作与其他动作的差异程度）。

### 性质验证

| 性质 | 随机探索 | B-exp | TB-exp | vMF-exp |
|------|---------|-------|--------|---------|
| P1 可扩展 | ✓ | ✗ | ✓ | ✓ |
| P2 不受限半径 | ✓ | ✓ | ✗ | ✓ |
| P3 顺序保持 | ✗ | ✓ | ✓ | ✓* |

*P3 在嵌入均匀分布假设下渐近满足。

## 理论分析

### 核心定理（Proposition 4.1）

在 $\mathcal{X}_n \sim \mathcal{U}(\mathcal{S}^{d-1})$（均匀分布假设）下：

$$\lim_{n \to +\infty} \frac{P_{\text{B-exp}}(a | n, d, V, \kappa)}{P_{\text{vMF-exp}}(a | n, d, V, \kappa)} = 1$$

即当动作数趋于无穷时，vMF-exp 与 B-exp 对每个动作赋予相同的探索概率。

### 渐近逼近表达式

两种方法共享的零阶近似 $P_0$：

$$P_0(a | n, d, V, \kappa) = \frac{f_{\text{vMF}}(A | V, \kappa) \cdot \mathcal{A}(\mathcal{S}^{d-1})}{n}$$

- B-exp 的收敛速度：$P_{\text{B-exp}} = P_0 + o(1/(n\sqrt{n}))$
- vMF-exp 的收敛速度：$P_{\text{vMF-exp}} = P_0 + \mathcal{O}(1/n^{1+2/(d-1)})$（$d > 2$）

### 高维修正（Proposition 4.4）

当维度 $d$ 较大（$d \geq 20$）时，需要一阶修正项 $P_1$ 才能更好逼近 vMF-exp 的真实概率。修正项的符号取决于 $\langle V, A \rangle$：与 $V$ 相似的动作被 vMF-exp 采样的概率略低于 B-exp，反之略高——即 vMF-exp 在高维下倾向于更充分地探索。

## 实验关键数据

### 蒙特卡洛模拟

- 对不同 $(d, \kappa, \langle V, A \rangle)$ 组合进行 800 万次重复实验
- $d$ 较小时（$d \leq 8$），$P_{\text{vMF-exp}}$ 与 $P_{\text{B-exp}}$ 几乎不可区分
- $d \geq 16$ 时，一阶近似 $P_1$ 明显优于零阶近似 $P_0$
- 所有结果与理论预测一致

### GloVe 真实数据验证

- 使用 100 万 GloVe 词嵌入向量（归一化后）
- 尽管不满足 i.i.d. 均匀分布假设，理论近似仍然准确
- 验证了 vMF-exp 同时满足 P1、P2、P3

### Deezer 生产环境部署

- 已在全球音乐流媒体 Deezer 的"灵感歌单"推荐系统中部署数月
- 探索百万级候选歌曲，采样延迟仅几毫秒
- 通过全球 A/B 测试验证了正向效果
- 证实了方法在真实大规模系统中的可扩展性和实用性

## 亮点与洞察

1. **巧妙的连续-离散桥接**：将离散动作空间的探索问题转化为连续超球面上的 vMF 采样 + 最近邻检索，完全绕开了 softmax 的 $O(n)$ 瓶颈
2. **理论严谨性**：不仅证明了渐近等价，还给出了不同维度下的收敛速率和高维修正公式
3. **Voronoï 几何直觉**：通过 Voronoï 镶嵌提供了直观的概率解释——密度低的区域动作有更大的胞腔因此被更多探索，这是一种有益的"稀疏区偏好"
4. **工业验证**：从理论到蒙特卡洛到真实数据再到生产部署的完整验证链路
5. **与 Thompson Sampling 的联系**：vMF-exp 与 Thompson Sampling 存在有趣的结构相似性（在后验分布上采样再做贪心选择）

## 局限与展望

1. **理论保证仅限均匀分布假设**：实际嵌入通常不满足 i.i.d. 均匀假设（如聚类结构），虽然实验显示理论近似依然有效，但缺乏严格保证
2. **高维收敛较慢**：$d$ 增大时，vMF-exp 逼近 B-exp 需要更大的 $n$，二阶误差项衰减速率为 $\mathcal{O}(1/n^{2/(d-1)})$
3. **ANN 误差未深入分析**：理论假设精确最近邻，极大规模下 ANN 的近似误差可能影响探索质量
4. **$\kappa$ 调参**：如何针对具体应用自适应选择 $\kappa$ 没有详细讨论
5. **未考虑聚类嵌入**：推荐系统嵌入常呈聚类结构（如音乐流派），在此情况下 Voronoï 胞腔面积分布不均，P3 的满足程度有待研究

## 相关工作与启发

- **Wolpertinger 架构**：先用 ANN 检索再做策略，vMF-exp 可视为其更优雅的替代方案
- **方向统计学**：将 vMF 分布从方向统计学引入 RL 探索是本文的关键创新
- **Thompson Sampling**：vMF-exp 与 Thompson Sampling 共享"采样-再贪心"的哲学
- **推荐系统 RL**：YouTube 等平台已采用 TB-exp，本文提供了无需截断的理论保证方案

## 评分

- 新颖性: ⭐⭐⭐⭐ — 将 vMF 分布引入大动作空间探索的想法简洁而新颖
- 实验充分度: ⭐⭐⭐⭐⭐ — 理论+模拟+真实公开数据+工业部署四层验证
- 写作质量: ⭐⭐⭐⭐⭐ — 问题定义清晰，理论推导严谨，表述流畅
- 价值: ⭐⭐⭐⭐ — 对大规模推荐系统等场景有直接应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] UME-R1: Exploring Reasoning-Driven Generative Multimodal Embeddings](../../ICLR2026/reinforcement_learning/ume-r1_exploring_reasoning-driven_generative_multimodal_embeddings.md)
- [\[ICML 2025\] Enhancing Decision-Making of Large Language Models via Actor-Critic](enhancing_decision-making_of_large_language_models_via_actor-critic.md)
- [\[ICML 2025\] Demystifying the Paradox of Importance Sampling with an Estimated History-Dependent Behavior Policy in Off-Policy Evaluation](demystifying_the_paradox_of_importance_sampling_with_an_estimated_history-depend.md)
- [\[ICML 2025\] Fast and Robust: Task Sampling with Posterior and Diversity Synergies for Adaptive Decision-Makers in Randomized Environments](fast_and_robust_task_sampling_with_posterior_and_diversity_synergies_for_adaptiv.md)
- [\[ICML 2025\] Action-Dependent Optimality-Preserving Reward Shaping (ADOPS)](action-dependent_optimality-preserving_reward_shaping.md)

</div>

<!-- RELATED:END -->
