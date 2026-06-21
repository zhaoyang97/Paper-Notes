---
title: >-
  [论文解读] Displacement-Resistant Extensions of DPO with Nonconvex $f$-Divergences
description: >-
  [ICLR 2026][LLM对齐][DPO] 发现 f-DPO 的可解性不需要 f 凸（仅需 $\lim_{t\to 0^+} f'(t) = -\infty$），进一步证明 $\arg\min f(t) \geq 1$ 是抵抗概率位移的必要条件，由此提出 SquaredPO（$f(t) = \frac{1}{2}(\log t)^2$，非凸），在保持性能的同时显著缓解 winner 概率下降问题。
tags:
  - "ICLR 2026"
  - "LLM对齐"
  - "DPO"
  - "f-divergence"
  - "likelihood displacement"
  - "preference optimization"
  - "SquaredPO"
---

# Displacement-Resistant Extensions of DPO with Nonconvex $f$-Divergences

**会议**: ICLR 2026  
**arXiv**: [2602.06788](https://arxiv.org/abs/2602.06788)  
**代码**: 无  
**领域**: LLM对齐 / 偏好优化  
**关键词**: DPO, f-divergence, likelihood displacement, preference optimization, SquaredPO

## 一句话总结
发现 f-DPO 的可解性不需要 f 凸（仅需 $\lim_{t\to 0^+} f'(t) = -\infty$），进一步证明 $\arg\min f(t) \geq 1$ 是抵抗概率位移的必要条件，由此提出 SquaredPO（$f(t) = \frac{1}{2}(\log t)^2$，非凸），在保持性能的同时显著缓解 winner 概率下降问题。

## 研究背景与动机
**领域现状**：DPO 及其变体是 LLM 对齐的主流方法，本质上是在 RLHF 目标中用 KL 散度约束策略偏离参考模型。Wang et al. (2024) 将 KL 推广为 f-divergence，但仅限于凸 f。

**现有痛点**：DPO 存在"概率位移"（probability displacement）现象——训练过程中 winner 和 loser 的概率都趋近零。这导致过训练时性能急剧下降，是 DPO 最广为诟病的实际问题。

**核心矛盾**：KL 散度对应的 $f_{KL}(t) = t\log t$，其 $\arg\min = e^{-1} < 1$，这在理论上决定了 DPO 必然导致 winner 概率下降至少 $e^{-1}$ 倍。凸 f-divergence 类中很难找到同时满足可解性和抗位移的 f。

**本文目标**（1）f-DPO 的可解性条件到底是什么？（2）哪些 f 能从理论上防止概率位移？（3）能否设计一个同时可解且抗位移的损失？

**切入角度**：放弃凸性要求，在更广的函数类中寻找满足两个条件的 f。

**核心 idea**：用 $f(t) = \frac{1}{2}(\log t)^2$（非凸、抗位移）替换 $f(t) = t\log t$（凸、会位移），得到理论更优的 SquaredPO 损失。

## 方法详解

### 整体框架
论文的目标不是再造一个对齐算法，而是回答一个更基础的问题：f-DPO 这一大类损失的行为，到底由 f 的什么性质决定。起点是广义 RLHF 目标 $\max_{\pi_\theta} \mathbb{E}[r(x,y)] - \beta D_f[\pi_\theta \| \pi_{ref}]$；把它在 f-divergence 约束下闭式求解，再代入 Bradley-Terry 偏好模型，就得到 f-DPO 损失 $-\log\sigma(\beta f'(\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}) - \beta f'(\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}))$。值得注意的是损失里出现的是 f 的导数 $f'$、而不是 f 本身——后面所有结论都挂在 $f'$（以及 f 最小值位置）的性质上。顺着这条线，论文分三步推进：先问什么样的 f 能让上面的推导成立（可解性），再问在可解的 f 里哪些能避免 winner 概率塌缩（抗位移），最后给出一个同时满足两者的具体 f。下面三个关键设计正对应这三步。

> 这是一篇纯理论/损失函数论文（DPO 的 f-divergence 推广 + 一个新损失），没有多阶段管线或多模块协同，框架就是"目标→闭式解→两个 f 条件→具体 f"的推导链，因此**不画 Mermaid 框架图**；三者一致改由整体框架与三个设计逐条对齐保证。

### 关键设计

**1. DPO-Inducing 条件：可解性其实不需要凸性**

要把 RLHF 目标化成 DPO 那样的闭式偏好损失，前提是 f-divergence 约束下的最优策略存在且良定义。过去的 f-DPO 工作（Wang et al., 2024）默认 f 必须凸，本文 Corollary 1 把这个门槛精确地替换掉：f 是 DPO-inducing 当且仅当 $\lim_{t\to 0^+} f'(t) = -\infty$。直觉上，$f'$ 在 0 附近趋于负无穷，等价于要求最优策略对任何 response 都赋予严格正的概率（否则梯度会把某些 response 推到概率 0，破坏可解性）。凸性只是保证这一点的充分条件之一，并非必要——这一步把可用的 f 从凸函数类一下子放宽到所有满足该极限条件的函数，为引入非凸 f 打开了空间。

**2. Displacement-Resistant 条件：$\arg\min f \geq 1$ 是抗位移的必要门槛**

概率位移指训练中 winner 概率不升反降。Lemma 2 给出它的成因刻画：若 $\arg\min_{t \geq 0} f(t) < 1$，则最优策略对 in-sample response 的概率必然被压到低于 $c \cdot \pi_{ref}$，也就是必然位移；反过来，抗位移的必要条件是 $\arg\min f(t) \geq 1$。把 DPO 代进来就一目了然：它对应 $f_{KL}(t) = t\log t$，最小值在 $t = e^{-1} < 1$，所以 DPO 在理论层面就注定会把 winner 概率往下压。

更深一层的原因来自 Lemma 1——f-DPO 名义上解的是完整 RLHF 问题 (5)，但它同时也解一个退化问题 (7)，后者的正则化只覆盖出现在训练数据里的 in-sample response，对 out-of-sample 行为毫无约束。于是模型可以毫无代价地把概率质量挪到没见过的 response 上，winner 概率随之下降。位移因此不是实现 bug，而是损失结构里的数学必然；而 $\arg\min f \geq 1$ 之所以能挡住它，正是因为它要求 f 在 $t < 1$（即概率被压低于参考模型）时付出更大代价，从而堵住这条挪走概率质量的通道。

**3. SquaredPO：用 $f(t)=\frac{1}{2}(\log t)^2$ 同时踩中两个条件**

把前两条当作设计约束，问题变成"找一个 f 同时满足 $\lim_{t\to 0^+} f'(t) = -\infty$ 和 $\arg\min f(t) \geq 1$"。论文给出的解是 $f(t) = \frac{1}{2}(\log t)^2$：它是非凸函数，但 $\lim_{t\to 0^+} f'(t) = -\infty$（可解），且 $\arg\min f(t) = 1$（抗位移），两个条件都恰好成立。代入 f-DPO 损失后，它等价于一个"自适应 $\beta$ 的 DPO"，其有效系数

$$\beta_\theta(y,x) = \beta \Big/ \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}.$$

含义很直接：当 winner 概率 $\pi_\theta/\pi_{ref}$ 往下掉时，分母变小，$\beta_\theta$ 自动放大，正则化随之加强，把进一步下降的趋势顶回去——这正是抗位移条件在训练动态里的具体表现。和 SimPO、$\beta$-DPO 这类同样改 $\beta$ 的方法相比，区别在于它们的自适应是启发式的（SimPO 的 $\beta$ 只随长度变化、训练中固定，$\beta$-DPO 还要额外引入超参数），而 SquaredPO 的自适应 $\beta$ 是从 f-divergence 理论自然推导出来的，不引入任何新超参数。

## 实验关键数据

### 概率位移缓解

| 指标 | SquaredPO | DPO |
|------|-----------|-----|
| Epoch 1 chosen log-ratio 中位数 | 更高（位移更小） | 更低（位移更严重） |
| 单调下降的 winner 占比（4 epoch） | **4.21%** | **99.63%** |

关键发现：DPO 中 99.63% 的 winner 概率一旦在第 1 个 epoch 下降，后续每个 epoch 都继续下降（单调下降）。SquaredPO 将这一比例降至 4.21%。

### 过训练鲁棒性（TL;DR Win Rate vs Base Model）

| Epochs | SquaredPO | χPO | DPO |
|--------|-----------|-----|-----|
| 1 | 50.8% | 51.2% | 51.8% |
| 2 | 50.6% | 48.9% | 45.0% |
| 4 | **51.0%** | 48.3% | **34.7%** |

DPO 在 4 epoch 后 win rate 降至 34.7%（严重过训练），SquaredPO 保持 51.0%。

### 标准基准（1 epoch）

| 方法 | AlpacaEval LC↑ | AlpacaEval WR↑ | MT-Bench↑ |
|------|---------------|----------------|-----------|
| SquaredPO | 29.2 | 24.5 | 7.924 |
| DPO | 29.6 | 24.8 | 7.925 |

性能基本持平，但 SquaredPO 未调超参（使用 DPO 默认值）。

## 亮点与洞察
- **从理论推导出的"自适应 $\beta$"**：SquaredPO 的核心直觉极为简单——当 winner 概率下降时自动加大正则化。但这不是启发式设计，而是从 f-divergence 理论中自然推导出来的。
- **Lemma 1 的深刻揭示**：f-DPO 同时解决完整问题和退化问题，意味着所有 f-DPO 变体都具有对 out-of-sample 行为缺乏约束的结构性缺陷。位移不是 bug，而是数学上的必然。
- **99.63% 单调下降**：首次报告 DPO 中 winner 概率单调下降的现象，这比之前"平均概率下降"的报告更精确和令人震惊。

## 局限与展望
- 仅在单一数据集（TL;DR）和单一模型（Llama-3-8B）上验证并使用 LoRA。
- Displacement-resistant 条件被证明是必要条件，但不是充分条件——满足条件并不保证完全消除位移。
- SquaredPO 在第 1 个 epoch 略逊于 DPO，超参数（$\beta$）未针对 SquaredPO 调优。
- 仅探索了一个具体的 f（$(\log t)^2/2$），还有许多满足两个条件的 f 值得探索。

## 相关工作与启发
- **vs DPO (Rafailov et al., 2023)**：DPO 是 $f_{KL}(t) = t\log t$ 的特例，$\arg\min = e^{-1}$，理论上必然位移。SquaredPO 用 $f(t) = \frac{1}{2}(\log t)^2$ 保证 $\arg\min = 1$，从根上解决。
- **vs χPO (Huang et al., 2025)**：χPO 也用 f-DPO 框架（$\chi^2$ 散度），有过训练鲁棒性但不如 SquaredPO。本文理论更一般（覆盖所有 f），χPO 只分析一个特例。
- **vs SimPO/β-DPO**：这些方法用启发式自适应 $\beta$，SquaredPO 的自适应 $\beta$ 从理论推导，无额外超参。
- **vs RCPO (Beyond Pairwise)**：RCPO 关注偏好数据格式（pairwise → ranked），SquaredPO 关注正则化的数学性质。两者正交，可以组合。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 完全刻画 DPO-inducing 条件 + 首次提出 displacement-resistant 条件，理论贡献深刻
- 实验充分度: ⭐⭐⭐ 仅一个数据集/模型，但位移分析很详尽
- 写作质量: ⭐⭐⭐⭐⭐ 理论结构清晰，定义→引理→定理的逻辑链完美，Venn 图直观
- 价值: ⭐⭐⭐⭐ 为 DPO 类方法提供了设计原则（两个条件），对未来偏好优化研究有指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Alignment-Weighted DPO: A Principled Reasoning Approach to Improve Safety Alignment](alignment-weighted_dpo_a_principled_reasoning_approach_to_improve_safety_alignme.md)
- [\[ICLR 2026\] Why DPO is a Misspecified Estimator and How to Fix It](why_dpo_is_misspecified_estimator.md)
- [\[ICLR 2026\] Token-Importance Guided Direct Preference Optimization (TI-DPO)](token-importance_guided_direct_preference_optimization.md)
- [\[ICML 2025\] DPO Meets PPO: Reinforced Token Optimization for RLHF](../../ICML2025/llm_alignment/dpo_meets_ppo_reinforced_token_optimization_for_rlhf.md)
- [\[ACL 2026\] S2H-DPO: Hardness-Aware Preference Optimization for Vision-Language Models](../../ACL2026/llm_alignment/s2h-dpo_hardness-aware_preference_optimization_for_vision-language_models.md)

</div>

<!-- RELATED:END -->
