---
title: >-
  [论文解读] Algorithmic Guarantees for Distilling Supervised and Offline RL Datasets
description: >-
  [ICLR2026][学习理论][样本复杂度] 本文给"数据集蒸馏"补上了第一套**无需训练模型**的可证明算法保证：对线性回归，证明只要用 $\tilde O(d^2)$ 个随机采样的回归器做凸的损失匹配，就能得到一份让任意有界线性模型 MSE 几乎不变的合成数据集，并给出匹配的 $\Omega(d^2)$ 下界；进一步把方法搬到离线 RL，用 Bellman 损失匹配得到同类保证，并在玩具 RL 环境上验证。
tags:
  - "ICLR2026"
  - "学习理论"
  - "数据集蒸馏"
  - "离线强化学习"
  - "样本复杂度"
  - "损失匹配"
  - "Bellman 损失"
  - "反集中不等式"
---

# Algorithmic Guarantees for Distilling Supervised and Offline RL Datasets

**会议**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=zueIXg5FP5](https://openreview.net/forum?id=zueIXg5FP5)  
**代码**: https://github.com/google-deepmind/loss_matching_dataset_distillation  
**领域**: 学习理论 / 数据集蒸馏 / 离线强化学习  
**关键词**: 数据集蒸馏, 样本复杂度, 损失匹配, Bellman 损失, 反集中不等式

## 一句话总结
本文给"数据集蒸馏"补上了第一套**无需训练模型**的可证明算法保证：对线性回归，证明只要用 $\tilde O(d^2)$ 个随机采样的回归器做凸的损失匹配，就能得到一份让任意有界线性模型 MSE 几乎不变的合成数据集，并给出匹配的 $\Omega(d^2)$ 下界；进一步把方法搬到离线 RL，用 Bellman 损失匹配得到同类保证，并在玩具 RL 环境上验证。

## 研究背景与动机
**领域现状**：数据集蒸馏（dataset distillation, DD）的目标是把一个大训练集 $D_{\text{train}}$ 压成一个很小的合成集 $D_{\text{syn}}$，使得"在 $D_{\text{syn}}$ 上训练出来的模型"和"在 $D_{\text{train}}$ 上训练出来的模型"性能几乎一样。主流做法是各种"匹配"思路——匹配 loss 梯度、匹配特征分布、匹配训练轨迹，通常嵌在一个 bi-level 优化里（一边训模型、一边优化合成集），或者对一组固定采样的网络做匹配。

**现有痛点**：这些方法绝大多数是**启发式**，只有经验上的增益，缺乏理论保证。少数给出保证的工作也都带着很强的前提：Chen et al. (2024) 对线性岭回归证明了效率，但**假设已经有了训练集上的最优模型**；Nguyen et al. (2021) 不训模型，但**只证明了"给定合成标签时合成特征向量会收敛"**，不是真正的性能保证。对离线 RL，现状更不令人满意——已有 DD 方法基本走**行为克隆（BC）**路线（Lei et al. 2024、Light et al. 2024），既丢掉了离线数据里本来就有的 reward 和 next-state 信息，又往往只在专家策略生成的数据上才好用，而且 Lei et al. 的分析同样需要一个**训练好的策略模型**。

**核心矛盾**：到目前为止，**没有任何"不训练模型就可证明高效"的 DD 算法**——无论监督学习还是离线 RL。所有保证要么靠"已有最优模型"这根拐杖，要么只证了半截（特征收敛而非性能）。

**本文目标**：(1) 给监督回归设计一个**不训模型**、目标函数**凸**、且有性能保证的 DD 算法，并刻画它到底需要多少随机采样模型（上界+下界）；(2) 把同一套思路推广到离线 RL，用能利用 reward/next-state 的 Bellman 损失替代 BC。

**切入角度**：作者沿用 Zhao & Bilen (2023) 那种"对一组**固定的随机采样网络**做匹配、完全不训练模型"的范式，但把匹配对象换成**损失值本身**（loss matching），而不是梯度或特征。关键洞察是：对线性回归，损失差 $L^{\text{err}}_{\text{mse}}$ 关于合成数据点是**凸函数**，可以高效求解；而它关于随机线性回归器的权重是一个 4 次多项式，于是可以用**多项式的反集中（anti-concentration）不等式**（Carbery-Wright）来论证"少数随机回归器就能逼住所有回归器"。

**核心 idea**：用"对 $\tilde O(d^2)$ 个随机采样模型做凸的损失匹配"代替"bi-level 训模型"，把 DD 变成一个有样本复杂度保证、可证明紧的算法问题。

## 方法详解

### 整体框架
本文不是一个网络架构，而是两套**算法 + 配套的样本复杂度定理**。两套算法共享同一个骨架，可以一句话概括：

> 从一个固定分布里**随机采样 $k$ 个模型**（监督场景是线性回归器，RL 场景是线性 Q 值预测器），然后**只优化合成数据集 $D_{\text{syn}}$**，让它在这 $k$ 个采样模型上的损失尽量贴近训练集上的损失。整个过程**不训练任何模型**。

形式化地，监督回归要求对所有有界线性回归器 $f\in\mathcal F_0$（$\|v\|_2\le 1$）满足

$$L^{\text{err}}_{\text{mse}}(D^{\text{sup}}_{\text{train}}, D^{\text{sup}}_{\text{syn}}, f) := \big(L_{\text{mse}}(D^{\text{sup}}_{\text{train}}, f) - L_{\text{mse}}(D^{\text{sup}}_{\text{syn}}, f)\big)^2 \le \varepsilon.$$

离线 RL 则把 MSE 换成 Bellman 损失，对所有线性 Q 值预测器 $f\in\mathcal Q_0$ 要求 $L^{\text{err}}_{\text{Bell}}\le\varepsilon$，其中

$$L_{\text{Bell}}(D^{\text{orl}}, f) = \mathbb E_{(s,a,r,s')\leftarrow D^{\text{orl}}}\Big[\big(f(s,a) - r - \gamma \max_{a'\in A} f(s',a')\big)^2\Big].$$

难点在于：我们要保证的是"对**所有** $f$ 损失都几乎不变"，而算法只在**有限个采样 $f$** 上做匹配。整篇论文的技术含量就在于证明"少数随机采样的 $f$ 足以覆盖全体 $f$"，并刻画这个"少数"到底是多少。下面四个关键设计就是这套框架的四块拼图。

### 关键设计

**1. 损失匹配 + 随机采样回归器：把 DD 变成一个不训模型的凸问题**

针对"现有 DD 要么 bi-level 训模型、要么只有半截保证"的痛点，作者的做法是：固定采样 $k$ 个回归器 $g_1,\dots,g_k$（每个 $g$ 由 $r\sim\mathcal N(0, 1/(d{+}1))$ 采样得到），然后求解

$$D^{\text{sup}}_{\text{syn}} = \arg\min_{D_{\text{syn}}} \sum_{j=1}^k L^{\text{err}}_{\text{mse}}(D^{\text{sup}}_{\text{train}}, D_{\text{syn}}, g_j)\quad(\text{Algorithm 1}).$$

关键性质是：单个 $L^{\text{err}}_{\text{mse}}(D_{\text{train}}, D_{\text{syn}}, g)$ 关于 $D_{\text{syn}}$ 的点是**凸函数**（用同质化技巧把标签并进特征：$\zeta(x,y)=(x_1,\dots,x_d,y)$，使损失写成 $\mathbb E[(r^\top\zeta)^2]$ 的差），因此整个目标也凸、可高效全局优化。这与旧方法的根本区别在于：旧方法的合成集质量依赖"训出来的模型好不好"，而这里合成集是一个干净凸问题的解，质量由下面的定理直接背书。

**2. $\tilde O(d^2)$ 上界与 $\Omega(d^2)$ 下界：证明采样数刚好够、且不能更少**

设计 1 留下的核心问题是"采样几个回归器才够"。Theorem 4.1 给出上界：当 $k = O(d^2\log(d(B{+}b)/\Delta)\log(1/\delta))$ 时，以概率 $1-\delta$，只要找到的 $D_{\text{syn}}$ 在这 $k$ 个采样回归器上把损失匹配到 $\Delta' = \Delta k / O(d^2)$ 以内，就能保证**对所有** $f\in\mathcal F_0$ 都有 $L^{\text{err}}_{\text{mse}}\le\Delta$。证明走反证法：若存在某个 $h$ 使损失差 $>\Delta$，则随机 $g$ 上的损失差 $\upsilon^2$ 是 Gaussian 变量的 4 次多项式且 $\mathbb E[\upsilon^2]>\Delta/O(d^2)$，用 **Carbery-Wright 反集中**得到 $\Pr[\upsilon^2>\Delta/O(d^2)]\ge 1/3$，再用 Chernoff + 对 $\mathcal F$ 和 $D_{\text{syn}}$ 建 $\varepsilon$-net 做并集界，凑出 $k=O(d^2\cdots)$。

更漂亮的是 Theorem 4.2 给出**匹配下界**：存在一个 $(d{+}1)$ 点的训练集，使得对**任意** $<q(q{+}1)/2$（$q=d{+}1$）个线性回归器，都能构造一个 $D_{\text{syn}}$ 让这些回归器损失全相等、却存在另一个回归器 $f_0$ 损失差 $\ge 1/(4q^2)$。其几何核心是：对称 $q\times q$ 矩阵构成 $q(q{+}1)/2$ 维线性空间，回归器少于这个数时必有非零对称矩阵 $A$ 与所有 $vv^\top$ 正交，于是令 $\mathbb E_{D_{\text{train}}}[xx^\top]=I$、$\mathbb E_{D_{\text{syn}}}[xx^\top]=I+A$ 即可。两条合起来证明 $\tilde O(d^2)$ **在多对数因子内是紧的**。此外附录还给出合成集**大小**的下界 $m\ge d+1$ 并证明可达。值得一提的是，定理里随机量正比于 $\log(1/\Delta)$，比矩阵 sketching 技术 $1/\Delta$ 的依赖更省。

**3. Bellman 损失匹配：把保证搬进离线 RL，但要为 max 项付出指数代价**

针对"离线 RL 的 DD 只会 BC、丢掉 reward 和 next-state"的痛点，作者把匹配对象换成 Bellman 损失，对一组随机采样的线性 Q 值预测器 $(f_j,\lambda_j)$（$\lambda$ 是给 reward 加的尺度因子）做同样的损失匹配（Algorithm 2）。这样 reward 和 next-state 自然进入目标，无需训练任何策略模型。Theorem 4.3 给出保证：采样 $k=\exp(O(d\log d))\cdot\text{poly}(\cdots)$ 个预测器，把 $\hat L^{\text{err}}_{\text{Bell}}$ 匹配到位即可保证对所有 $f\in\mathcal Q_0$ 的 $L^{\text{err}}_{\text{Bell}}\le\Delta$。

代价为什么从 $d^2$ 跳到指数？因为 Bellman 损失含一个 $\max_{a'} f(s',a')$ 项，**不能**写成预测器权重的低次多项式，于是设计 2 里那套多项式反集中失效，证明改走"条件化（conditioning）"论证，最坏情况下需要 $\exp(O(d\log d))$ 个采样预测器。作者也指出：暴力对 $\mathcal Q_0$ 取 net 反而要 $\exp(d)$ 个，且实践中远少于此即可（实验里只用 $k=20$）。

**4. 可分解特征映射：用一个自然假设把 RL 的指数代价压回 $\tilde O(d^2)$**

设计 3 的指数代价太肉疼，作者观察到：实践中状态-动作特征常常是**可分解**的，即 $\phi(s,a)=\phi_1(s)+\phi_2(a)$（最常见的就是把 state embedding 和 one-hot 动作拼接，DDPG 等深度 RL 默认就这么干）。在这个假设下，Theorem 4.4 证明只需 $k=\tilde O(d^2)$ 个采样预测器——条件是优化额外满足一阶矩约束 $\mathbb E_{D_{\text{syn}}}[(\phi_1(s),\phi_2(a),r,\phi_1(s'))]=\mathbb E_{D_{\text{train}}}[\cdots]$。当 $\phi_1,\phi_2$ 还是线性映射时，这个约束是关于 $D_{\text{syn}}$ 的线性约束，整个优化重新变成**凸**问题。技术上靠的是分解结构让条件化论证能利用线性预测器的 Gaussian 反集中，把采样预测器逼进一个有非平凡概率的子集。这给出"未来怎么把保证推广到神经网络"的一个路标：只要能证明非线性预测器输出关于权重的 Lipschitz 性 + 反集中，论证就能搬过去。

### 损失函数 / 训练策略
本文没有"训练损失"——算法不训模型，唯一被优化的是合成数据集 $D_{\text{syn}}$ 自身。监督场景目标是 $\sum_j L^{\text{err}}_{\text{mse}}(D_{\text{train}}, D_{\text{syn}}, g_j)$（凸，直接凸优化求解，实验用 $D_{\text{rand}}$ 初始化）；RL 场景目标是带尺度因子的 $\sum_j \hat L^{\text{err}}_{\text{Bell}}(D_{\text{train}}, D_{\text{syn}}, f_j, \lambda_j)$，可分解线性特征下加一阶矩约束后亦凸。评测时才在 $D_{\text{syn}}$ 上训一个真模型（监督：同质线性模型；RL：用 Fitted-Q Iteration 训 2 层网络得到策略）。

## 实验关键数据

实验目的是**验证理论**而非刷 SOTA，因此用的是经典小数据集和玩具 RL 环境。

### 主实验

| 场景 | 数据集/环境 | 合成集大小 $N_{\text{syn}}$ | 采样模型数 $k$ | 结果（vs baselines） |
|------|-------------|----------------------------|----------------|----------------------|
| 监督回归 | 红/白葡萄酒质量、Boston Housing | $\{20,50,100\}$ | 100 | $D_{\text{syn}}$ 上训的线性模型测试 MSE ≈ 全量 $D_{\text{train}}$，远好于 $D_{\text{rand}}$，优于或持平 leverage-score 子采样 $D_{\text{lev}}$ |
| 离线 RL | Cartpole / Mountain Car / Acrobot | $N_{\text{syn}}$ 多档 | 20 | $D_{\text{syn}}$ 策略显著优于 $D_{\text{rand}}$；Cartpole、Acrobot 上随 $N_{\text{syn}}$ 增大可达到或超过 $D_{\text{train}}$ |

监督实验取 10 次试验的平均测试损失（Figure 1），直接对应并验证了 Theorem 4.1；RL 用 Fitted-Q Iteration 训 2 层网络（线性 Q 值在 FQI 下表现差，故用小网络），策略各评测 10 次（Figure 2）。

### 消融 / 关键观测

| 配置/对照 | 现象 | 说明 |
|-----------|------|------|
| $D_{\text{syn}}$ vs $D_{\text{rand}}$ | $D_{\text{syn}}$ 大幅领先 | 损失匹配优化确实学到了有用的合成数据，不是随机子采样 |
| $D_{\text{syn}}$ vs $D_{\text{lev}}$（监督） | 优于或持平 | 在同质线性回归上击败经典的 leverage-score 降维 |
| 很小的 $k$（监督 100 / RL 20） | 仍然有效 | 实践所需采样数远小于最坏情况理论界（尤其 RL 的指数界） |
| 线性 vs 非线性 | 用 2 层网络也 work | 保证虽对线性模型证明，经验上推广到神经网络 |

### 关键发现
- 理论保证虽然只覆盖**线性模型**，但实验里换成 2 层神经网络算法依然有效，说明分析抓住了问题本质。
- RL 最坏情况要 $\exp(O(d\log d))$ 个采样预测器，但实践只用 $k=20$ 就够——这与"特征常被映射到比输入更低的维度 + 可分解结构"的现实相符。
- **失败案例（作者主动承认）**：在 D4RL 这类标准离线 RL benchmark 上方法**不 work**。原因是 D4RL 专门设计来区分 FQI（迭代最小化 Bellman 损失）和 CQL 这类更先进算法，而本文的蒸馏是围绕 Bellman 损失匹配设计的，对 D4RL 不适配；要上 D4RL 得把蒸馏推广到 CQL 那种更复杂的损失。

## 亮点与洞察
- **第一套"无需训模型"的 DD 性能保证**：以往的理论保证都要拐杖（已有最优模型，或只证特征收敛），本文把"不训模型 + 性能保证 + 凸可解"三者第一次凑齐，这是真正补空白。
- **上界配下界，证明紧到多对数因子**：$\tilde O(d^2)$ 上界 + $\Omega(d^2)$ 下界，下界用"对称矩阵空间维数 = $q(q{+}1)/2$"这个干净的线性代数事实构造，非常优雅，告诉你"采样数再省也省不动了"。
- **把统计技巧迁进 DD**：用 Carbery-Wright 多项式反集中证"少数随机模型逼住全体模型"，这套"反集中 → net → 并集界"的模板可复用到其它"对一族随机探针做匹配"的问题。
- **诚实的负结果**：主动报告 D4RL 上失败并解释机理（Bellman 匹配 vs CQL），比只报喜的论文更可信，也给后续工作指明了"换损失函数"的方向。
- **可迁移思路**：设计 4 给出的"Lipschitz + 反集中 ⇒ 推广到非线性预测器"是一条明确的技术路线，谁能证出某类神经网络的权重反集中，就能把这套保证搬到深度模型。

## 局限与展望
- **只对线性模型有证明**：监督和 RL 的保证都建立在线性回归器/线性 Q 值预测器上，神经网络版本只有经验结果，缺理论。作者把"证明某类网络的多项式逼近 + 权重反集中"列为未来方向。
- **RL 最坏情况指数复杂度**：一般特征映射下要 $\exp(O(d\log d))$ 个采样预测器，只有在"可分解特征"这个（虽自然但有限制的）假设下才回到 $\tilde O(d^2)$。
- **D4RL 上失效**：方法绑定 Bellman 损失匹配，无法直接处理 CQL 等需要更复杂损失/保守正则的设置，因此在主流离线 RL benchmark 上不适用。
- **实验规模偏小**：监督是 UCI 级小数据集、RL 是 Cartpole/Acrobot 玩具环境，主要为验证理论，离"大规模实用蒸馏"还有距离。
- **改进思路**：把 Bellman 匹配换成 CQL 风格的保守损失再做蒸馏与分析；或寻找特定 MDP 的几何性质，给一般状态-动作 embedding 拿到多项式级保证。

## 相关工作与启发
- **vs Chen et al. (2024)**：同样针对（核/线性）岭回归给 DD 的高效算法保证，但他们**假设已有训练集上的最优模型**；本文**不训任何模型**，保证性质上更强。
- **vs Nguyen et al. (2021)**：同属"不训模型"的隐式损失匹配，但他们**只证明给定合成标签下合成特征向量收敛**，不是性能保证；本文直接给出"任意有界模型损失几乎不变"的性能保证。
- **vs Zhao & Bilen (2023)**：本文沿用其"对固定随机采样网络做匹配、不训模型"的范式，但把匹配对象从特征分布换成**损失值**，并补上理论保证。
- **vs Light et al. (2024) / Lei et al. (2024)（离线 RL DD）**：两者都走 BC 路线——Light 匹配 BC 损失梯度，Lei 用 action-value 加权 BC 但**需要训练好的 Q 预测器**；本文改用 **Bellman 损失匹配**，能利用 reward/next-state、且不训模型。
- **vs 矩阵 sketching / leverage-score 子采样（Drineas et al. 2006 等）**：这些经典降维方法有保证但局限于线性/凸模型，且样本量正比 $1/\Delta$；本文样本量正比 $\log(1/\Delta)$ 更省，且 DD 框架可经验性地用于神经网络，实验中亦优于 leverage-score。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 第一套无需训模型的 DD 性能保证，上界配紧下界，并首次给离线 RL 的 Bellman 匹配蒸馏以理论
- 实验充分度: ⭐⭐⭐ 实验用于验证理论且诚实报告 D4RL 失败，但规模偏小、未与近期 RL DD 方法在大 benchmark 上对比
- 写作质量: ⭐⭐⭐⭐ 问题定义、定理、证明思路、技术 overview 层次清晰，理论与算法对应明确
- 价值: ⭐⭐⭐⭐ 给 DD 这一长期偏经验的方向打下可证明的地基，并给"推广到非线性"留出清晰路标

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Towards a Sharp Analysis of Offline Policy Learning for f-Divergence-Regularized Contextual Bandits](towards_a_sharp_analysis_of_offline_policy_learning_for_f-divergence-regularized.md)
- [\[ICLR 2026\] On the Computational Limits of AI4S-RL：A Unified $\varepsilon$-$N$ Analysis](on_the_computational_limits_of_ai4s-rl_a_unified_varepsilon-n_analysis.md)
- [\[ICLR 2026\] Larger Datasets Can Be Repeated More: A Theoretical Analysis of Multi-Epoch Scaling in Linear Regression](larger_datasets_can_be_repeated_more_a_theoretical_analysis_of_multi-epoch_scali.md)
- [\[ICLR 2026\] Reshaping Reasoning in LLMs: A Theoretical Analysis of RL Training Dynamics through Pattern Selection](reshaping_reasoning_in_llms_a_theoretical_analysis_of_rl_training_dynamics_throu.md)
- [\[ICML 2026\] Formalizing Learning from Language Feedback with Provable Guarantees](../../ICML2026/learning_theory/formalizing_learning_from_language_feedback_with_provable_guarantees.md)

</div>

<!-- RELATED:END -->
