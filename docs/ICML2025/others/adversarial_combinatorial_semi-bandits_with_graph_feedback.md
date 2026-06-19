---
title: >-
  [论文解读] Adversarial Combinatorial Semi-bandits with Graph Feedback
description: >-
  [ICML 2025][组合半臂赌博机] 本文将图反馈（graph feedback）引入对抗组合半臂赌博机（combinatorial semi-bandits）框架，提出 OSMD-G 算法，建立了最优遗憾（regret）界 $\widetilde{\Theta}(S\sqrt{T} + \sqrt{\alpha S T})$，其中 $S$ 是组合决策大小，$\alpha$ 是反馈图的独立数，关键技术在于利用随机化轮换舍入（randomized swap rounding）实现负相关采样。
tags:
  - "ICML 2025"
  - "组合半臂赌博机"
  - "图反馈"
  - "对抗在线学习"
  - "遗憾界"
  - "负相关采样"
---

# Adversarial Combinatorial Semi-bandits with Graph Feedback

**会议**: ICML 2025  
**arXiv**: [2502.18826](https://arxiv.org/abs/2502.18826)  
**代码**: 无  
**领域**: 其他（在线学习 / Bandit 理论）  
**关键词**: 组合半臂赌博机, 图反馈, 对抗在线学习, 遗憾界, 负相关采样

## 一句话总结

本文将图反馈（graph feedback）引入对抗组合半臂赌博机（combinatorial semi-bandits）框架，提出 OSMD-G 算法，建立了最优遗憾（regret）界 $\widetilde{\Theta}(S\sqrt{T} + \sqrt{\alpha S T})$，其中 $S$ 是组合决策大小，$\alpha$ 是反馈图的独立数，关键技术在于利用随机化轮换舍入（randomized swap rounding）实现负相关采样。

## 研究背景与动机

### 问题起源

组合半臂赌博机（combinatorial semi-bandits）是经典多臂赌博机的推广，在多平台在线广告、推荐系统、网页优化、在线最短路径等场景中有广泛应用。每轮中，学习者从 $K$ 个臂中选择 $S$ 个，观察被选臂的奖励作为反馈。

### 现有反馈结构的局限

已有工作只区分两种极端反馈场景：
- **完全信息（full information）**：每轮观察所有 $K$ 个臂的奖励，最优 regret 为 $\widetilde{\Theta}(S\sqrt{T})$
- **半臂反馈（semi-bandit）**：每轮仅观察被选 $S$ 个臂的奖励，最优 regret 为 $\widetilde{\Theta}(\sqrt{KST})$

但实际问题中存在丰富的中间信息结构。例如：

**在线广告竞拍**：平台公布中标价，学习者可推算比自己出价高的所有出价的反事实收益

**在线推荐**：语义相似的物品（如两款纸巾）之间存在信息关联——用户点击一个很可能也会点击另一个

这些额外信息在经典 semi-bandit 反馈中被完全忽略，亟需一个通用框架来刻画。

### 本文目标

用有向反馈图 $G = ([K], E)$ 统一描述反馈结构：选择臂 $a$ 后，可观察所有出邻居 $N_{\text{out}}(a)$ 的奖励。完全图对应完全信息，仅含自环对应 semi-bandit。本文要：
1. 在一般反馈图下建立 minimax 最优 regret
2. 设计近似最优算法

## 方法详解

### 整体框架

本文提出 **OSMD-G**（Online Stochastic Mirror Descent with Graphs），是经典 OSMD 算法在图反馈设定下的扩展。算法流程如下：

**输入**：时间范围 $T$，决策集 $\mathcal{A}$，臂集 $[K]$，组合预算 $S$，反馈图 $G$，截断参数 $\epsilon$，学习率 $\eta$，镜像映射 $F$

1. **初始化**：$x^1 = \arg\min_{x \in \text{Conv}_\epsilon(\mathcal{A})} F(x)$
2. **每轮 $t = 1, \ldots, T$**：
    - 通过随机化轮换舍入（Algorithm 2）从 $x^t$ 采样决策 $v^t \in \mathcal{A}$，满足 $\mathbb{E}[v^t] = x^t$ 且负相关
    - 执行 $v^t$，观察图反馈 $\{r_i^t : i \in N_{\text{out}}(v^t)\}$
    - 构造无偏奖励估计量 $\tilde{r}_a^t$
    - 镜像下降更新：$w^{t+1} = \nabla F^*(\nabla F(x^t) + \eta \tilde{r}^t)$
    - 投影回约束集：$x^{t+1} = \arg\min_{x \in \text{Conv}_\epsilon(\mathcal{A})} D_F(x, w^{t+1})$

### 关键设计

#### 1. 奖励估计量

对每个臂 $a \in [K]$，基于图反馈构造无偏估计量：

$$\tilde{r}_a^t = \frac{\sum_{i \in [K]: i \to a} \mathbf{1}[v_i^t = 1]}{\sum_{i \in [K]: i \to a} x_i^t} \cdot r_a^t$$

其中分子是实际观察到臂 $a$ 的指示变量之和，分母是其期望值。由于 $\mathbb{E}[v_i^t] = x_i^t$，此估计量无偏：$\mathbb{E}[\tilde{r}_a^t] = r_a^t$。

当 $G$ 仅含自环时，该估计退化为经典 semi-bandit 的估计 $\tilde{r}_a^t = v_a^t r_a^t / x_a^t$。

#### 2. 负相关采样（核心技术贡献）

这是本文最关键的创新。OSMD-G 要求采样分布 $p^t$ 同时满足两个条件：

**条件 1（均值）**：$\forall i \in [K], \; \mathbb{E}_{v \sim p}[v_i] = x_i$

**条件 2（负相关）**：$\forall i \neq j, \; \mathbb{E}_{v \sim p}[v_i v_j] \leq x_i x_j$

即任意两臂之间负相关。作者通过引用 Chekuri et al. (2009) 的 matroid 理论结果（Lemma 1），证明对完整决策集 $\mathcal{A}$ 上任意 $x \in \text{Conv}(\mathcal{A})$，总存在满足上述条件的分布 $p$，且可通过**随机化轮换舍入（randomized swap rounding）**高效采样。

**为什么负相关至关重要？** 在 regret 分析的关键步骤中，需要控制估计量方差：

$$\mathbb{E}\left[\left(\sum_{i: i \to a} v_i^t\right)^2\right]$$

- **若无负相关约束**：只能朴素地上界为 $S \cdot \mathbb{E}[\sum_{i:i \to a} v_i^t]$，导致 regret 中出现 $\sqrt{S\alpha}$ 因子
- **有负相关约束**：可将二阶矩分解为平方均值 + 个体方差之和，消除 $O(K^2)$ 个协方差项，将因子从 $S\alpha$ 降低为 $S + \alpha$

具体推导：

$$\mathbb{E}\left[\left(\sum_{i: i \to a} v_i^t\right)^2\right] = \left(\sum_{i: i \to a} x_i^t\right)^2 + \text{Var}\left(\sum_{i: i \to a} v_i^t\right) \leq \left(\sum_{i: i \to a} x_i^t\right)^2 + \sum_{i: i \to a} x_i^t$$

这一步关键利用了 $\text{Cov}(v_i^t, v_j^t) \leq 0$，使方差可以分解为各臂方差之和。

#### 3. 随机化轮换舍入算法

给定目标 $x = \sum_{i=1}^N w_i v_i$（作为 $\mathcal{A}$ 中决策的凸组合），算法通过如下过程生成满足负相关的随机决策：

1. 初始化 $u \leftarrow v_1$
2. 对 $i = 1, \ldots, N-1$：取 $c \leftarrow v_{i+1}$，计算累积权重 $\beta_i = \sum_{j=1}^i w_j$
3. 当 $u \neq c$ 时：找到 $a \in u \setminus c$ 和 $a' \in c \setminus u$，使得交换后仍可行
4. 以概率 $\beta_i / (\beta_i + w_{i+1})$ 执行 $u \leftarrow u - \{a\} + \{a'\}$，否则执行 $c \leftarrow c - \{a'\} + \{a\}$

该算法的关键是利用决策集的**交换性质**（exchange property）：对任意 $u, c \in \mathcal{A}$，存在可以交换的臂对使得结果仍可行。完整决策集 $\mathcal{A}$ 和 matroid 结构都满足此性质。

#### 4. 镜像映射选择与学习率设置

采用负熵镜像映射 $F(x) = \sum_{i=1}^K (x_i \log x_i - x_i)$，此时：
- 对偶空间 $D^* = \mathbb{R}^K$
- 更新简化为：$w^{t+1} = x^t \circ \exp(\eta \tilde{r}^t)$（逐坐标乘性更新）
- Bregman 散度为 KL 散度

最优参数选取：
- 学习率：$\eta = \sqrt{\frac{2S \log(K/S)}{(S + 4\alpha \log(4K^2 T/\alpha))T}}$
- 截断率：$\epsilon = \frac{1}{KT}$

### 损失函数 / 训练策略

本文是理论分析驱动的工作，不涉及传统意义上的损失函数和训练。核心目标是最小化 regret：

$$\mathbb{E}[\mathsf{R}(\pi)] = \mathbb{E}\left[\max_{v_* \in \mathcal{A}} \sum_{t=1}^T \langle v_* - v^t, r^t \rangle\right]$$

**Regret 上界分析**（Theorem 3）将 regret 分解为三项：
1. **截断误差**：$\epsilon K T$（由凸化约束集的截断引入）
2. **初始 Bregman 散度**：$S \log(K/S) / \eta$（衡量初始点到最优解的"距离"）
3. **梯度估计方差**：$\frac{\eta}{2}(S + 4\alpha \log(4K/(\epsilon\alpha))) T$

代入最优参数后得到最终上界：

$$\mathbb{E}[\mathsf{R}] \leq S\sqrt{T \log(K/S)} + 2\sqrt{\alpha S T \log(K/S) \log(4K^2 T/\alpha)}$$

## 实验关键数据

本文为纯理论工作，无数值实验，核心成果以定理形式呈现。

### 主实验

本文的"主实验"即为 minimax regret 的上下界匹配。下表汇总了不同反馈设定下的最优 regret（忽略 polylog 因子）：

| 反馈结构 | $\alpha$ 值 | 最优 Regret | 算法 | 来源 |
|:---|:---|:---|:---|:---|
| 完全信息（$G$ 为完全图） | $\alpha = 1$ | $\widetilde{\Theta}(S\sqrt{T})$ | OSMD | Koolen et al. (2010) |
| 一般图反馈 | 一般 $\alpha$ | $\widetilde{\Theta}(S\sqrt{T} + \sqrt{\alpha S T})$ | **OSMD-G** | **本文** |
| 半臂反馈（$G$ 仅含自环） | $\alpha = K$ | $\widetilde{\Theta}(\sqrt{KST})$ | OSMD | Audibert et al. (2014) |
| 全臂反馈 | — | $\widetilde{\Theta}(\sqrt{KS^3T})$ | — | Cohen et al. (2017) |

### 消融实验

通过理论分析对比不同设计选择的影响：

| 配置 | Regret 上界 | 说明 |
|:---|:---|:---|
| OSMD-G + 负相关采样（完整决策集 $\mathcal{A}$） | $\widetilde{O}(S\sqrt{T} + \sqrt{\alpha S T})$ | 最优，本文主结果 |
| OSMD-G + 仅均值约束（无负相关） | $\widetilde{O}(\sqrt{S\alpha T})$ | 次优，Theorem 4 |
| 一般决策子集 $\mathcal{A}_0$ + 负相关不可行 | $\widetilde{\Theta}(\sqrt{S\alpha T})$ | 不可避免的退化 |
| 弱可观测图 + 探索后利用 (ETC) | $\widetilde{O}(S T^{2/3} + \delta^{1/3} S^{2/3} T^{2/3})$ | Theorem 5 |
| 时变图 $\{G_t\}$ | $\widetilde{O}(S\sqrt{T} + \sqrt{S \sum_{t=1}^T \alpha_t})$ | 自然扩展 |

### 关键发现

1. **Regret 平滑插值**：当 $\alpha$ 从 1 变化到 $K$，regret 从 $S\sqrt{T}$（完全信息）平滑过渡到 $\sqrt{KST}$（semi-bandit），首次揭示了中间区域的精确行为
2. **负相关是必要的**：仅满足均值约束（$\mathbb{E}[v^t] = x^t$）的 OSMD 方案被证明是次优的（$\sqrt{S\alpha T}$ vs $S\sqrt{T} + \sqrt{\alpha ST}$）
3. **决策集结构影响根本**：从完整决策集 $\mathcal{A}$ 到特定子集 $\mathcal{A}_{\text{partition}}$，最优 regret 会从 $\widetilde{\Theta}(S\sqrt{T} + \sqrt{\alpha ST})$ 升至 $\widetilde{\Theta}(\sqrt{S\alpha T})$，说明负相关采样的可行性直接决定了问题难度
4. **下界构造基于独立集分割**：将最大独立集均匀分为 $S$ 个子集，每个子集构造一个多臂赌博机子问题，通过 KL 散度的链式法则和 Pinsker 不等式得到 regret 下界

## 亮点与洞察

1. **优雅的统一框架**：用反馈图的独立数 $\alpha$ 一个参数就刻画了从完全信息到 semi-bandit 的完整谱系，regret 公式 $S\sqrt{T} + \sqrt{\alpha ST}$ 极为简洁
2. **负相关 ≠ 显然**：虽然 $\|v\|_1 = S$ 的约束直觉上暗示负相关，但作者给出具体反例说明朴素分布可能不满足——例如 $x = (1, 0.8, 0.2)$, $S=2$ 时唯一解是确定性地选 $\{1,2\}$ 或 $\{1,3\}$
3. **理论工具的跨领域迁移**：核心引理（Lemma 1）来自 matroid 理论中的 dependent randomized rounding (Chekuri et al., 2009)，将其成功应用于 bandit 分析是一个巧妙的跨领域连接
4. **时变图的自然扩展**：算法无需修改即可处理时变反馈图 $\{G_t\}$，只需将证明中固定的 $\alpha$ 替换为 $\alpha_t$

## 局限与展望

1. **一般决策子集**：当 $\mathcal{A}_0 \subsetneq \mathcal{A}$ 且不满足交换性质时，负相关不可实现，算法退化。对一般 $\mathcal{A}_0$ 的最优算法仍为开放问题
2. **弱可观测图**：仅在随机奖励设定下给出 ETC 算法的 $T^{2/3}$ 界，对抗设定下的弱可观测图尚未解决
3. **计算复杂度**：投影步骤 $\arg\min_{x \in \text{Conv}_\epsilon(\mathcal{A})} D_F(x, w^{t+1})$ 的效率取决于决策集结构，未详细讨论
4. **纯理论无实验**：缺乏数值验证，实际场景中 $\alpha$ 的估计和算法的有限时间性能未知
5. **非线性 payoff 的推广**：当前框架仅限线性 payoff $\langle v, r^t \rangle$，对非线性奖励函数的扩展留待未来工作

## 相关工作与启发

- **Alon et al. (2015)**：多臂赌博机 + 图反馈的开创性工作，证明了 $\widetilde{\Theta}(\sqrt{\alpha T})$ 的最优 regret
- **Audibert et al. (2014)**：组合 semi-bandit 的经典结果，建立了 $\widetilde{\Theta}(\sqrt{KST})$ 最优界
- **Chekuri et al. (2009)**：matroid 上的 dependent randomized rounding，本文 Lemma 1 的直接来源
- **Wen et al. (2024, NeurIPS)**：同一作者在上下文 bandit + 图反馈方向的前序工作
- **Kocák & Carpentier (2023)**：揭示了小 $T$ 区间 regret 的精确形态（$T^{1/2}$ 与 $T^{2/3}$ 的混合）

本文为上下文 bandit、combinatorial optimization 与图结构学习的交叉领域提供了理论基石，后续可考虑将图反馈引入更复杂的组合优化场景（如子模优化、路由问题等）。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 首次将图反馈引入组合 semi-bandit 并完整刻画 minimax regret
- 实验充分度: ⭐⭐⭐ — 纯理论工作，无数值实验
- 写作质量: ⭐⭐⭐⭐⭐ — 结构清晰，证明思路自然，示例直观
- 价值: ⭐⭐⭐⭐⭐ — 解决了重要的开放问题，建立了领域基准结果

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Semi-Supervised Regression with Heteroscedastic Pseudo-Labels](../../NeurIPS2025/others/semi-supervised_regression_with_heteroscedastic_pseudo-labels.md)
- [\[ACL 2025\] Learning to Reason from Feedback at Test-Time](../../ACL2025/others/learning_to_reason_from_feedback_at_test-time.md)
- [\[ICML 2025\] Provably Cost-Sensitive Adversarial Defense via Randomized Smoothing](provably_cost-sensitive_adversarial_defense_via_randomized_smoothing.md)
- [\[NeurIPS 2025\] Put CASH on Bandits: A Max K-Armed Problem for Automated Machine Learning](../../NeurIPS2025/others/put_cash_on_bandits_a_max_k-armed_problem_for_automated_machine_learning.md)
- [\[NeurIPS 2025\] Semi-infinite Nonconvex Constrained Min-Max Optimization](../../NeurIPS2025/others/semi-infinite_nonconvex_constrained_min-max_optimization.md)

</div>

<!-- RELATED:END -->
