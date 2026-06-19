---
title: >-
  [论文解读] Online Linear Regression with Paid Stochastic Features
description: >-
  [AAAI2026][在线线性回归] 研究了在线线性回归中特征被噪声污染、学习者可以**付费降低噪声强度**的新问题设定，证明了已知噪声协方差时最优遗憾率为 $\widetilde{\mathcal{O}}(\sqrt{T})$、未知时为 $\widetilde{\mathcal{O}}(T^{2/3})$，并给出匹配的下界，所有界关于时间 $T$ 的依赖都是阶最优的。
tags:
  - "AAAI2026"
  - "在线线性回归"
  - "噪声特征"
  - "付费特征"
  - "遗憾界"
  - "协方差估计"
  - "矩阵鞅集中"
---

# Online Linear Regression with Paid Stochastic Features

**会议**: AAAI2026  
**arXiv**: [2511.08073](https://arxiv.org/abs/2511.08073)  
**代码**: 无  
**领域**: 在线学习 / 线性回归  
**关键词**: 在线线性回归, 噪声特征, 付费特征, 遗憾界, 协方差估计, 矩阵鞅集中

## 一句话总结

研究了在线线性回归中特征被噪声污染、学习者可以**付费降低噪声强度**的新问题设定，证明了已知噪声协方差时最优遗憾率为 $\widetilde{\mathcal{O}}(\sqrt{T})$、未知时为 $\widetilde{\mathcal{O}}(T^{2/3})$，并给出匹配的下界，所有界关于时间 $T$ 的依赖都是阶最优的。

## 研究背景与动机

### 在线学习中的噪声特征问题

在经典的在线线性回归中，学习者顺序观测到 i.i.d. 的特征向量 $x_t \in \mathbb{R}^d$ 并预测输出 $y_t = x_t^\top \theta^*$。绝大多数工作假设特征 $x_t$ 可被完美观测，但现实中我们通常只能获取 $x_t$ 的噪声版本。例如：

- **实验测量误差**：通过物理实验获取的特征不可避免地带有测量噪声
- **用户行为采样**：用户的上下文信息仅依赖其行为的随机采样
- **隐私保护机制**：特征被差分隐私机制（如 Laplace/Gaussian 噪声）有意扰动

### 付费降低噪声的场景

在很多场景中，学习者可以通过**分配更多资源**来降低噪声强度：

**实验设备升级**：租用更精密的仪器、雇佣更有经验的人员

**延长测量时间**：在更长时间窗口内平均多次原始采样

**隐私补偿**：向用户提供付费服务以换取较低的隐私保护水平（即较少的噪声）

**数据收集**：收集更多统计信息后再做决策

降低噪声有利于预测，但也会产生成本（时间、金钱或服务回馈）。因此，学习者需要**最优平衡预测误差与支付成本**。

### 核心问题定义

学习者观测到带噪声的特征 $\hat{x}_t(c_t) = x_t + n_t(c_t)$，其中 $c_t \in [0,1]$ 是学习者在第 $t$ 轮选择的支付金额，$n_t(c)$ 是零均值噪声，其协方差矩阵 $\Sigma_n(c)$ 随支付增加而递减（即更高支付 → 更小噪声）。学习者的目标是最小化**遗憾**：

$$\text{Reg}(T) = \mathbb{E}\left[\sum_{t=1}^T \left((\hat{y}_t(\hat{x}_t) - y_t)^2 + \lambda c_t\right)\right] - T \cdot \ell^*$$

其中 $\ell^*$ 是最优线性预测器在最优支付水平下的期望损失。

## 方法详解

### 整体框架

论文将问题分为两个设定处理：

1. **已知噪声协方差设定**（Section 4）：学习者知道所有支付水平 $c$ 对应的噪声协方差 $\Sigma_n(c)$
2. **未知噪声协方差设定**（Section 5）：学习者不知道支付如何影响噪声水平

两个设定的最优遗憾率存在本质差异：已知情况下为 $\widetilde{\mathcal{O}}(\sqrt{T})$，未知情况下退化为 $\widetilde{\mathcal{O}}(T^{2/3})$。

### 关键设计一：协方差修正的信息共享（已知设定）

已知设定的核心洞察是：由于噪声协方差已知，在不同支付水平下收集的样本可以**共享信息**。具体而言，损失函数 $\ell(c, \nu)$ 中唯一依赖噪声协方差的项是 $\nu^\top \Sigma_{\hat{x}}(c) \nu$。论文构造了一个**协方差修正估计器**：

$$\hat{\Sigma}_{\hat{x}}(c) = \sum_{s=1}^t \left(\hat{x}_s(c_s) \hat{x}_s(c_s)^\top + \Sigma_n(c) - \Sigma_n(c_s)\right)$$

这个估计器利用了所有历史样本（无论它们在哪个支付水平下被观测），通过添加修正项 $\Sigma_n(c) - \Sigma_n(c_s)$ 将不同支付水平的观测"对齐"到目标支付水平 $c$。其期望恰好等于 $t \cdot \Sigma_{\hat{x}}(c)$，是无偏的。

基于此修正估计器，定义经验损失：

$$\hat{L}_t^{kc}(c, \nu) = \sum_{s=1}^t \left((\hat{x}_s^\top \nu - y_s)^2 + \nu^\top(\Sigma_n(c) - \Sigma_n(c_s))\nu + \lambda c\right)$$

信息共享的关键优势是**无需探索不同支付水平**——在任何历史支付序列下，估计器都能为所有支付水平提供有效估计，从而可以贪心地选择最优支付。

### 关键设计二：乐观探索策略（未知设定）

未知噪声协方差时，无法进行跨支付水平的信息共享，只能利用**在同一支付水平下收集的样本**估计该水平的损失。这引发了经典的**探索-利用困境**：

- **探索需求**：必须在不同支付水平上收集足够样本以识别最优支付 $c^*$
- **利用需求**：尽量多地在当前最优估计的支付水平上运行

论文采用 **UCB 风格的乐观策略**：

1. 将连续支付空间离散化为网格 $\{k/K\}_{k \in [K]}$
2. 对每个网格点维护经验损失 $\hat{L}_t^{uc}(k/K, \nu)$
3. 选择使**乐观损失**（经验平均损失减去置信宽度）最小的支付水平：

$$k_{t-1} \in \arg\min_{k \in [K]} \left\{\frac{\hat{L}_{t-1}^{uc}(k/K, \hat{\nu}_{t-1}(k))}{N_{t-1}(k/K)} - 9S^2 \sqrt{\frac{8\beta_t^2 \ln \frac{3dt(t+1)}{\delta}}{N_{t-1}(k/K)}}\right\}$$

其中 $N_{t-1}(k/K)$ 是该支付水平的历史选择次数。观测不足的支付水平获得更大的置信减量，鼓励探索。

### 理论分析

#### 定理 1（已知协方差上界）

通过矩阵鞅集中不等式证明经验损失关于所有支付水平和线性预测器的**一致收敛**（Proposition 2），进而推导 Algorithm 1 的遗憾界：

$$\text{Reg}(T) = \widetilde{\mathcal{O}}\left(S^2(R^2 d + S^2)\sqrt{T} + \lambda\right)$$

#### 定理 2（已知协方差下界）

构造一维反例：噪声方差在 $c < 1/2$ 时固定为 $\mathcal{N}(0,1)$，在 $c \geq 1/2$ 时为零。利用特征方差的微小差异（$1 \pm \epsilon$）使得最优支付在 $c=0$ 和 $c=1/2$ 之间切换，将问题归约为双臂老虎机，证明 $\Omega(\sqrt{T})$ 的下界。

#### 定理 3（未知协方差上界）

通过离散化网格 + UCB 探索，证明 Algorithm 2 的遗憾界：

$$\text{Reg}(T) = \widetilde{\mathcal{O}}\left((S^2(R^2 d + S^2))^{2/3} \lambda^{1/3} T^{2/3}\right)$$

网格大小取 $K = \lceil T^{1/3} \lambda^{2/3} / (S^2(R^2 d + S^2))^{2/3} \rceil$ 时最优平衡离散化误差与探索代价。

#### 定理 4（未知协方差下界）

精心构造 $K$ 个噪声方差曲线，每条在一个互不重叠的小区间（宽度 $\approx 1/K$）上偏离基准函数 $f(c) = \frac{1-c}{1+c}$（该函数使所有支付水平的最优损失恒等于 $1/2$）。利用鸽巢原理和信息论工具证明 $\Omega(T^{2/3})$ 的下界，技巧类似 Lipschitz 老虎机的经典下界构造。

## 实验关键数据

本文为纯理论贡献，实验部分以理论界的比较为主。

### 表 1：遗憾率总结

| 设定 | 上界 | 下界 | 关于 $T$ 是否阶最优 |
|------|------|------|---------------------|
| 已知噪声协方差 | $\widetilde{\mathcal{O}}(\sqrt{T})$ | $\Omega(\sqrt{T})$ | ✅ 阶最优（忽略对数因子） |
| 未知噪声协方差 | $\widetilde{\mathcal{O}}(T^{2/3})$ | $\Omega(T^{2/3})$ | ✅ 阶最优（忽略对数因子） |
| 无噪声经典设定 | $\Theta(\ln T)$ | $\Theta(\ln T)$ | ✅ 精确 |

### 表 2：两个设定下算法的关键对比

| 特性 | Algorithm 1（已知） | Algorithm 2（未知） |
|------|---------------------|---------------------|
| 信息共享 | ✅ 跨支付水平共享 | ❌ 仅同支付水平 |
| 探索机制 | 无需（贪心即可） | UCB 风格乐观探索 |
| 支付网格大小 $K$ | $\lceil \lambda T \rceil$ | $\lceil T^{1/3} \lambda^{2/3} / (\cdot)^{2/3} \rceil$ |
| 正则化 | 需要（确保凸性） | 不需要（天然凸） |
| 遗憾率 | $\widetilde{\mathcal{O}}(\sqrt{T})$ | $\widetilde{\mathcal{O}}(T^{2/3})$ |
| 每步计算 | $K$ 个凸优化问题 | $K$ 个凸优化问题 |

## 亮点与洞察

1. **协方差修正是全文核心技术**：通过已知的噪声协方差差 $\Sigma_n(c) - \Sigma_n(c_s)$ 来修正经验协方差估计，使得不同支付水平的观测可以被统一利用。这一简洁的代数技巧同时消除了探索需求并保持了估计的无偏性。

2. **$\sqrt{T}$ 与 $\ln T$ 的鸿沟**：经典无噪声在线线性回归的最优遗憾为 $\Theta(\ln T)$，而即使噪声协方差完全已知，付费特征问题的最优遗憾也指数级地退化到 $\Theta(\sqrt{T})$。下界构造揭示了根本原因：噪声方差的不连续性使得支付选择本质上变成了老虎机问题。

3. **一侧 Lipschitz 性质的巧妙利用**：论文证明最优损失 $\ell^*(c)$ 是 $\lambda$-单侧 Lipschitz 的——即增加支付永远不会让损失显著增加（因为更高支付总是降低预测误差）。这个性质使得网格离散化的误差可控，且不需要对噪声协方差施加 Lipschitz 假设。

4. **避免 Lipschitz bandit 的维度诅咒**：直接将问题归约为 Lipschitz bandit 会得到 $\widetilde{\mathcal{O}}(T^{(d+2)/(d+3)})$ 的遗憾率（$d+1$ 维空间），但论文通过将线性预测器优化与支付选择解耦，只在一维支付空间上离散化，保持了 $T^{2/3}$ 的速率，与维度无关。

5. **矩阵鞅集中的非平凡应用**：论文将 Tropp (2012) 的矩阵鞅集中不等式推广到异方差设定（不同时刻的噪声协方差不同），是理论工具层面的有意义贡献。

## 局限性

1. **维度依赖的紧性未确定**：所有下界构造基于一维实例，$d$ 维下的最优依赖关系（上界中的 $d$ 因子是否可改进）仍然是开放问题。

2. **计算效率不理想**：每个时间步需对 $K$ 个网格点求解凸优化问题，$K$ 可达 $\Theta(T)$ 量级（已知设定），总计算量为 $O(T^2 \cdot \text{poly}(d))$，对在线场景偏重。论文也指出自适应网格或连续支付优化是值得探索的方向。

3. **仅限线性预测器**：最优非线性预测器需要计算 $\mathbb{E}[x_t | \hat{x}_t]^\top \theta^*$，依赖完整的分布信息且通常计算不可行。论文选择与线性基准比较是合理的，但限制了结果的适用范围。

4. **缺乏实证实验**：作为纯理论论文，没有数值模拟来展示算法在有限样本下的实际表现，也没有与现有方法的经验比较。

5. **噪声模型假设较强**：要求噪声为零均值独立的 sub-Gaussian 分布，且支付-噪声映射满足单调性。某些应用场景中噪声可能与特征相关，或支付与噪声的关系更加复杂。

## 相关工作

- **错误变量模型（Errors-in-Variables）**：经典统计回归中的噪声特征问题，主要研究 $\theta^*$ 的估计而非在线预测，且通常假设高维稀疏性。
- **Cesa-Bianchi et al. (2011)**：最直接的前驱工作，研究含噪声特征的在线线性回归，但没有支付/成本概念，且遗憾基准是无噪声特征下的最优预测器。
- **van den Heuvel &"; Kamvar (2023)**：在二分类设定下研究了付费降噪，证明了 $d^2 (\ln T) \sqrt{T}$ 的遗憾界，但限于独立二元特征噪声，技术不适用于相关噪声的回归问题。
- **Lipschitz Bandits (Kleinberg 2004; Bubeck et al. 2011)**：一维问题也可达 $\widetilde{\mathcal{O}}(T^{2/3})$，本文的下界构造深受此方向启发，但问题结构本质不同（非 Lipschitz，且线性预测器部分可精确优化）。
- **隐私与激励机制设计（Ghosh et al. 2011; Hsu et al. 2014）**：研究如何设计付费机制激励用户出售隐私数据，但侧重估计精度最优化而非在线预测中的遗憾最小化。

## 评分

⭐⭐⭐⭐ (4/5)

**理由**：论文提出了一个新颖且有实际动机的在线学习问题设定——付费降低特征噪声。理论结果干净利落：两个设定下均给出了关于 $T$ 阶最优的上下界，协方差修正和一侧 Lipschitz 性质等技术贡献也具有独立价值。主要不足在于缺乏实验验证、维度依赖的最优性未解决、以及计算效率问题。整体上是一篇扎实的理论贡献。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Information-Computation Tradeoffs for Noiseless Linear Regression with Oblivious Contamination](../../NeurIPS2025/others/information-computation_tradeoffs_for_noiseless_linear_regression_with_oblivious.md)
- [\[ICML 2026\] Inference of Online Newton Methods with Nesterov's Accelerated Sketching](../../ICML2026/others/inference_of_online_newton_methods_with_nesterovs_accelerated_sketching.md)
- [\[AAAI 2026\] DeToNATION: Decoupled Torch Network-Aware Training on Interlinked Online Nodes](detonation_decoupled_torch_network-aware_training_on_interlinked_online_nodes.md)
- [\[ICML 2026\] Cascaded Flow Matching for Heterogeneous Tabular Data with Mixed-Type Features](../../ICML2026/others/cascaded_flow_matching_for_heterogeneous_tabular_data_with_mixed-type_features.md)
- [\[ICML 2026\] Adaptive Multi-Round Allocation with Stochastic Arrivals](../../ICML2026/others/adaptive_multi-round_allocation_with_stochastic_arrivals.md)

</div>

<!-- RELATED:END -->
