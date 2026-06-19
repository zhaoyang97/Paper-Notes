---
title: >-
  [论文解读] Zeroth-Order Optimization Finds Flat Minima
description: >-
  [NeurIPS 2025][强化学习][零阶优化] 首次从理论上证明标准零阶优化（两点梯度估计）具有隐式正则化效果——收敛到Hessian迹最小的平坦极小值（flat minima），在凸且充分光滑条件下给出了$T = \mathcal{O}(d^4/\epsilon^2)$的收敛复杂度保证。 问题背景 零阶优化（Zero…
tags:
  - "NeurIPS 2025"
  - "强化学习"
  - "零阶优化"
  - "隐式正则化"
  - "Hessian迹"
  - "平坦极小值"
  - "语言模型微调"
  - "锐度感知最小化"
---

# Zeroth-Order Optimization Finds Flat Minima

**会议**: NeurIPS 2025  
**arXiv**: [2506.05454](https://arxiv.org/abs/2506.05454)  
**作者**: Liang Zhang (ETH Zurich / MPI-IS), Bingcong Li (ETH Zurich), Kiran Koshy Thekumparampil (Amazon), Sewoong Oh (UW), Michael Muehlebach (MPI-IS), Niao He (ETH Zurich)  
**代码**: [Liang137/FlatZero](https://github.com/Liang137/FlatZero)  
**领域**: 强化学习  
**关键词**: 零阶优化, 隐式正则化, Hessian迹, 平坦极小值, 语言模型微调, 锐度感知最小化  

## 一句话总结

首次从理论上证明标准零阶优化（两点梯度估计）具有隐式正则化效果——收敛到Hessian迹最小的平坦极小值（flat minima），在凸且充分光滑条件下给出了$T = \mathcal{O}(d^4/\epsilon^2)$的收敛复杂度保证。

## 研究背景与动机

### 问题背景
零阶优化（Zeroth-Order Optimization）在梯度不可用或计算代价高的场景中广泛应用，包括黑盒对抗攻击、强化学习策略搜索和大语言模型微调。Malladi et al. (2023) 展示了零阶方法可在单张A100 GPU上微调300亿参数模型，而梯度方法需要8张。标准两点梯度估计器通过有限差分近似梯度：

$$g_\lambda(x,u) = \frac{f(x+\lambda u) - f(x - \lambda u)}{2\lambda} u, \quad u \sim \mathcal{N}(0, I_d)$$

### 已有工作的不足
- 现有零阶优化理论聚焦于收敛到**任意驻点**，对收敛到**哪种解**缺乏刻画
- 平坦极小值（flat minima）的研究几乎都围绕一阶方法（SGD、SAM等），零阶方法的隐式正则化效果被忽视
- Ahn et al. (2024) 定义了局部的平坦极小值概念（梯度流极限点处Hessian迹的驻点），非全局最优
- 零阶方法能在高维LLM微调中取得合理性能的原因（维度依赖可被Hessian迹替代）缺少理论解释

### 核心动机
零阶优化实际最小化的是平滑函数$f_\lambda(x) = \mathbb{E}_{u}[f(x+\lambda u)]$。通过Taylor展开：

$$f_\lambda(x) = f(x) + \frac{\lambda^2}{2} \mathrm{Tr}(\nabla^2 f(x)) + o(\lambda^2)$$

这暗示零阶优化隐式地将Hessian迹作为正则项。本文将这一直觉形式化，给出严格的收敛复杂度分析。

## 方法详解

### 核心定义

**平坦极小值（Definition 3.1）**：设$\mathcal{X}^* = \arg\min_x f(x)$为最优解集。$x^*$是平坦极小值当且仅当$x^* \in \arg\min_{x \in \mathcal{X}^*} \mathrm{Tr}(\nabla^2 f(x))$，即在所有最优解中Hessian迹最小。

**近似平坦极小值（Definition 3.2）**：$\hat{x}$是$(\epsilon_1, \epsilon_2)$-近似平坦极小值，若满足：
- $f(\hat{x}) - \min_x f(x) \leq \epsilon_1$（函数值接近最优）
- $\mathrm{Tr}(\nabla^2 f(\hat{x})) - \min_{x \in \mathcal{X}^*} \mathrm{Tr}(\nabla^2 f(x)) \leq \epsilon_2$（Hessian迹接近最小）

### 理论分析框架

核心思想是分析正则化损失$F(x) = f(x) + \frac{\lambda^2}{2} \mathrm{Tr}(\nabla^2 f(x))$的收敛性，而非将Hessian迹项简单视为要控制的偏差。

**假设条件**：
- **Assumption 3.3（光滑性）**：$f(x)$三阶连续可微，满足一阶（$L_1$）、二阶（$L_2$）、三阶（$L_3$）光滑条件
- **Assumption 3.4（凸性）**：$f(x)$在$\mathbb{R}^d$上凸

**关键引理（Lemma 3.5）**：建立$F(x)$与$f_\lambda(x)$的逼近关系，以及两点估计器二阶矩的上界：

$$|f_\lambda(x) - F(x)| \leq \frac{L_3}{24}\lambda^4(d+4)^2$$

$$\mathbb{E}\|g_\lambda(x,u)\|^2 \leq 2(d+6)\|\nabla F(x)\|^2 + \mathcal{O}(\lambda^4 d^4)$$

此结果需要计算8阶高斯矩（Isserlis定理），涉及105个组合项，远超传统分析所需的4阶矩（3个项）。

### 主定理（Theorem 1）

在假设3.3和3.4下，Algorithm 1取步长$\eta = 1/(8(d+6)L_1)$，平滑参数$\lambda^2 \leq \sqrt{2}L_1/(d^{3/2}L_3)$，有：

$$\mathbb{E}[F(x_\tau) - \min_x F(x)] \leq \frac{8(d+6)L_1\|x_0 - x_F^*\|^2}{T} + \mathcal{O}(\lambda^4 d^3)$$

### 推论（Corollary 2）：平坦极小值收敛复杂度

取$\lambda^2 = \mathcal{O}(\epsilon/d^3)$，$T = \mathcal{O}(d^4/\epsilon^2)$，Algorithm 1输出$(\mathcal{O}(\epsilon/d^2), \epsilon)$-近似平坦极小值：
- 函数值误差：$\mathbb{E}[f(x_\tau) - f^*] \leq \mathcal{O}(\epsilon/d^2)$
- Hessian迹误差：$\mathbb{E}[\mathrm{Tr}(\nabla^2 f(x_\tau)) - \min_{x\in\mathcal{X}^*}\mathrm{Tr}(\nabla^2 f(x))] \leq \epsilon$

### 与标准分析的对比

| 分析类型 | $\lambda^2$选取 | 迭代次数$T$ | 函数值保证 | Hessian迹保证 |
|---------|---------------|------------|----------|-------------|
| 标准分析 [Nesterov & Spokoiny] | $\mathcal{O}(\epsilon/d)$ | $\mathcal{O}(d/\epsilon)$ | $\leq \epsilon$ | 无 |
| **本文** | $\mathcal{O}(\epsilon/d^3)$ | $\mathcal{O}(d^4/\epsilon^2)$ | $\leq \mathcal{O}(\epsilon/d^2)$ | $\leq \epsilon$ |

本文用更多迭代换取了对Hessian迹的收敛保证，且在函数值误差上也更精细。

## 实验关键数据

### 实验1：凸二分类任务

在LIBSVM数据集上，使用过参数化SVM（squared hinge loss）和逻辑回归，特征维度从$d$扩展到$D=10000$。

| 模型 | 数据集 | 方法 | 训练损失 | 测试准确率 | Hessian迹趋势 |
|------|-------|------|---------|----------|-------------|
| SVM | a5a ($N$=6414, $d$=123) | GD | 收敛至接近0 | ~81% | 不变 |
| SVM | a5a | ZO ($\lambda$=0.005) | 收敛至接近0 | ~81% | **持续下降** |
| SVM | w5a ($N$=9888, $d$=300) | GD | 收敛至接近0 | ~97% | 不变 |
| SVM | w5a | ZO ($\lambda$=0.005) | 收敛至接近0 | ~97% | **持续下降** |
| Logistic | a5a | GD | 收敛 | ~82% | 不变 |
| Logistic | a5a | ZO ($\lambda$=0.01) | 收敛 | ~82% | **持续下降** |

关键观察：GD和ZO在训练损失和测试准确率上表现相当，但ZO持续降低Hessian迹，收敛到更平坦的解。

### 实验2：RoBERTa-Large语言模型微调

在RoBERTa-Large (355M参数) 上进行few-shot文本分类微调，使用全批量训练以排除mini-batch噪声影响。

| 数据集 | K (每类样本) | 方法 | 训练损失 | 测试准确率 | Hessian迹趋势 |
|-------|------------|------|---------|----------|-------------|
| SST-2 | 32 | GD | 较低 | ~91% | 下降 |
| SST-2 | 32 | ZO | 稍高 | ~89% | **下降** |
| SST-2 | 256 | GD | 较低 | ~93% | 下降 |
| SST-2 | 256 | ZO | 稍高 | ~92% | **下降** |
| SST-5 | 32 | GD | 较低 | ~48% | 下降 |
| SST-5 | 32 | ZO | 稍高 | ~45% | **下降** |
| TREC | 32 | GD | 较低 | ~74% | 下降 |
| TREC | 32 | ZO | 稍高 | ~70% | **下降** |

在LLM微调中，GD和ZO均降低了Hessian迹。GD的隐式正则化被认为来自大学习率效应，而ZO的行为与本文理论预测一致。实验还验证了Hessian迹远小于维度$d$，支持了prior work中解释ZO在高维中有效性的假设。

## 亮点

- **开创性理论贡献**：首次证明标准零阶优化收敛到全局意义的平坦极小值，而非仅局部驻点，建立了新的分析框架
- **优雅的隐式正则化机制**：揭示两点估计器自然地将$\mathrm{Tr}(\nabla^2 f(x))$编码为正则项，无需额外算法修改即可获得类似SAM的锐度最小化效果
- **分析技术深度**：需要计算高斯随机向量的8阶矩（105个组合项），显著推进了零阶优化的理论分析工具
- **广泛适用性**：结论可扩展到其他满足$\mathbb{E}[uu^\top]=I_d$的无偏估计器（单点估计器、球面均匀分布等），且分析框架可推广到一阶方法（SAM、SGD）

## 局限与展望

- **凸性假设**：主定理仅适用于凸函数，深度学习中的非凸优化需要额外假设（如局部凸性），全局非凸情况下平坦极小值的定义本身也需修改
- **高阶光滑性假设**：要求三阶连续可微且$L_1$、$L_2$、$L_3$光滑，限制了对非光滑问题（如ReLU网络）的适用性
- **维度依赖**：收敛复杂度$\mathcal{O}(d^4/\epsilon^2)$中$d^4$的依赖较重，虽然这是为了同时保证Hessian迹收敛
- **仅考虑标准两点估计器**：是否能通过算法修改（如结合一阶信息、方差缩减）改善复杂度仍为开放问题
- **LLM微调实验中GD也降低Hessian迹**：未能在非凸场景中展示ZO相对GD在平坦性上的独特优势

## 与相关工作的对比

- **Nesterov & Spokoiny (2017)**：建立了零阶优化收敛到任意驻点的经典理论，本文在此基础上进一步刻画收敛到平坦极小值
- **Ahn et al. (2024)**：定义了局部平坦极小值（梯度流极限下Hessian迹的驻点），提出两种梯度算法。本文采用全局定义，用零阶方法实现，无需梯度和Hessian计算
- **Wen et al. (2023, SAM)**：证明SAM在batch size=1时最小化Hessian迹。本文表明标准零阶优化无需特殊设计即可达到类似效果
- **Malladi et al. (2023, MeZO)**：展示了零阶方法微调LLM的实际可行性。本文为其有效性提供了隐式正则化的理论解释
- **Zhang et al. (2024)**：受$f_\lambda$启发提出显式最小化平滑损失的梯度方法。本文证明零阶方法已隐式完成此目标
- **Zheng et al. (2025), Tang et al. (2024)**：将零阶与SAM结合以显式寻找平坦极小值。本文关注标准零阶方法已有的隐式效果

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 首次建立零阶优化→平坦极小值的理论联系，开辟新研究方向
- 实验充分度: ⭐⭐⭐⭐ — 覆盖玩具例子、凸分类任务、LLM微调三层次，但非凸场景下GD也有类似行为减弱了区分度
- 写作质量: ⭐⭐⭐⭐⭐ — 从直觉到形式化层层递进，Example 2.1和Proposition 2.2的Warm-up设计精巧
- 价值: ⭐⭐⭐⭐ — 理论有深度，为零阶优化和LLM微调提供了新的理解角度，但凸性限制降低了实际指导价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] FAB: A First-Order AB-based Gradient Algorithm for Distributed Bilevel Optimization over Time-Varying Directed Graphs](../../ICML2026/reinforcement_learning/fab_a_first-order_ab-based_gradient_algorithm_for_distributed_bilevel_optimizati.md)
- [\[NeurIPS 2025\] Online Optimization for Offline Safe Reinforcement Learning](online_optimization_for_offline_safe_reinforcement_learning.md)
- [\[NeurIPS 2025\] Complexity Scaling Laws for Neural Models using Combinatorial Optimization](complexity_scaling_laws_for_neural_models_using_combinatorial_optimization.md)
- [\[AAAI 2026\] First-Order Representation Languages for Goal-Conditioned RL](../../AAAI2026/reinforcement_learning/first-order_representation_languages_for_goal-conditioned_rl.md)
- [\[NeurIPS 2025\] Sequential Monte Carlo for Policy Optimization in Continuous POMDPs](sequential_monte_carlo_for_policy_optimization_in_continuous_pomdps.md)

</div>

<!-- RELATED:END -->
