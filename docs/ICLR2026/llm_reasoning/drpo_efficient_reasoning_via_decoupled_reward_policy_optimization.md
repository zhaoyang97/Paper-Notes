---
title: >-
  [论文解读] DRPO: Efficient Reasoning via Decoupled Reward Policy Optimization
description: >-
  [ICLR 2026][Reasoning][efficient reasoning] 诊断出 GRPO 在加入长度惩罚后的根本缺陷——正确但冗长的回答可能获得负优势值从而被错误惩罚——提出 DRPO 将正负样本的奖励信号解耦，确保长度惩罚只在正确回答组内归一化，在 1.5B 模型上实现 77% 长度缩减仅 1.1% 性能损失（对比基线 68% 缩减 4.3% 损失）。
tags:
  - "ICLR 2026"
  - "Reasoning"
  - "efficient reasoning"
  - "overthinking"
  - "GRPO"
  - "length penalty"
  - "reinforcement-learning"
---

# DRPO: Efficient Reasoning via Decoupled Reward Policy Optimization

**会议**: ICLR 2026  
**arXiv**: [2510.04474](https://arxiv.org/abs/2510.04474)  
**代码**: [https://github.com/Optimization-AI/DRPO](https://github.com/Optimization-AI/DRPO)  
**领域**: LLM推理  
**关键词**: efficient reasoning, overthinking, GRPO, length penalty, reinforcement-learning

## 一句话总结
诊断出 GRPO 在加入长度惩罚后的根本缺陷——正确但冗长的回答可能获得负优势值从而被错误惩罚——提出 DRPO 将正负样本的奖励信号解耦，确保长度惩罚只在正确回答组内归一化，在 1.5B 模型上实现 77% 长度缩减仅 1.1% 性能损失（对比基线 68% 缩减 4.3% 损失）。

## 研究背景与动机
**领域现状**：大推理模型（DeepSeek-R1 等）通过 GRPO 训练获得强推理能力，但存在严重的 overthinking 问题——回答"2+3=?"也需要生成 ~1000 个 token。

**现有痛点**：现有 RL 方法通过在奖励中加入长度惩罚来鼓励简洁推理（如 RLOO-LP, ALP, HAPO），但几乎都导致显著的性能下降。

**核心矛盾**：GRPO 的 group-relative 优势函数在混合长度惩罚后会将正确但冗长的回答的优势推到负值——模型被误导为把有效推理当作负样本来惩罚。例如：6 个回答中 3 个正确，加入长度惩罚后第 3 个正确回答的优势从 +1 变为 -0.17。

**本文目标**：如何在缩短推理长度的同时最小化性能损失？

**切入角度**：将学习信号的计算从正负样本混合改为分离——正确回答的长度惩罚只在正确回答组内归一化，永远不会产生负学习信号。

**核心 idea**：解耦正负样本的奖励归一化，让长度惩罚只减弱（而不翻转）正确回答的学习信号。

## 方法详解

### 整体框架
DRPO 要解决的是"加了长度惩罚就掉点"这件事，而它的判断是：问题出在 GRPO 把正负样本混在一起归一化，所以索性换一套不混合的目标函数。它不搭在 GRPO 上，而是搭在判别式优化框架 DisCO 上，把目标拆成互不干扰的两项——正样本项用一套基于长度奖励的闭式权重做加权似然（短而正确的回答权重更大），负样本项用 log-sum-exp 聚合难负样本。长度惩罚只活在正样本项里，于是"鼓励简洁"再也不会反过来惩罚正确推理。

### 关键设计

**1. 诊断 GRPO 的负优势陷阱：长度惩罚为什么会反过来罚正确推理**

GRPO 用组内相对优势 $A(o_i|q) = \frac{r(o_i) - \text{mean}(\{r_j\})}{\text{std}(\{r_j\})}$，把同一道题采样出的正确与错误回答放进一个组里一起做标准化。只用 0/1 对错奖励时这没问题；可一旦把长度惩罚掺进 $r$，一个正确但偏长的回答就可能掉到组均值以下、优势变成负数——模型于是被告知"这次正确推理是负样本，要往下压"。背景里那个 6 选 3 的例子（某个正确回答的优势从 $+1$ 翻成 $-0.17$）正是这么发生的。论文进一步点明：这不是某个实现的 bug，而是 RLOO-LP、ALP、HAPO 等一切建立在相对优势 / 混合归一化之上的方法共有的结构性缺陷——只要奖励是"对错 + 长度"的复合信号，正负样本一起归一化就必然有概率把正确信号翻负。

**2. 解耦正负归一化：把长度惩罚关进正样本组内部**

既然病根是"正负混在一起归一化"，DRPO 就把正确回答的学习信号锁在正确回答组内部单独归一化，不再和错误样本相互拉扯。它先解一个 KL 正则化的最优正样本分布

$$P_q^* = \arg\max_P \mathbb{E}_{o\sim P}[r_l(o)] - \lambda D_{KL}(P, \pi_{old}^+)$$

其中 $r_l(o) = 1 - |o|/C$ 是越短得分越高的长度奖励。这个目标有闭式解 $P_q^*(o) \propto \pi_{old}^+(o|q)\exp(r_l(o)/\lambda)$，于是每个正样本拿到一个只在正样本集合内归一化的权重

$$\omega(o|q) = \frac{\exp(r_l(o)/\lambda)}{\mathbb{E}_{o\sim\pi^+}\exp(r_l(o)/\lambda)}.$$

关键在于 $\omega$ 恒为正：短回答 $r_l$ 大、权重大，长回答权重小，但再长也只是趋近 0、绝不翻负。这样"鼓励简洁"就从"把长的正确回答踢成负样本"变成了"在正确回答之间重新分配权重"，超参 $\lambda$ 平滑地控制长度与准确率的取舍（$\lambda\to\infty$ 退化为均匀权重、不做长度控制）。

**3. 判别式目标：正负两项彻底分开写**

有了只作用于正样本的权重 $\omega$，DRPO 索性不搭 GRPO，而是搭在判别式优化框架 DisCO 上，把目标拆成互不干扰的两项：正样本项 $\mathbb{E}_{o\sim\pi^+}\,\omega(o|q)\,s_\theta(o,q)$ 用 $\omega$ 做加权似然，专门抬高短而正确的回答的分数 $s_\theta$；负样本项 $-\tau\log\mathbb{E}_{o'\sim\pi^-}\exp(s_\theta(o',q)/\tau)$ 用 log-sum-exp 聚合，自动把梯度压到最难的负样本上。外层再套一个信赖域约束 $D_{KL}(\pi_{old}\,\|\,\pi_\theta) \leq \delta$ 稳住更新步长。因为正负彻底解耦，长度惩罚只活在正样本项里，再也碰不到对错误样本的判别；当 $\lambda \to +\infty$ 时正样本权重退化为均匀，整个目标就还原成不带长度惩罚的 DisCO。

### 损失函数 / 训练策略
基于 DisCO 框架，约束用 penalty 函数处理。在 DeepScaleR-Preview-Dataset（40.3K 数学题）上训练 1000 步，生成预算 8K token，每题采样 8 个回答。

## 实验关键数据

### 主实验（AES: Accuracy Efficiency Score）

| 方法 | 模型 | Pass@1 | Length | AES |
|------|------|--------|--------|-----|
| RLOO-LP | 1.5B | 0.567 | 2531 | -0.129 |
| ALP | 1.5B | 0.606 | 3494 | -0.387 |
| HAPO | 1.5B | 0.534 | 1791 | -0.519 |
| **DRPO** | **1.5B** | **0.624** | **1527** | **+0.178** |
| RLOO-LP | 7B | 0.692 | 2649 | -0.033 |
| **DRPO** | **7B** | **0.714** | **1502** | **+0.249** |

### 关键发现
- DRPO 是唯一在所有模型规模（1.5B/7B/8B）上都获得正 AES 的方法——所有基线在多数设置下 AES 为负
- 1.5B 模型在 GSM8K 上：DRPO 77% 长度缩减仅 1.1% 性能损失 vs 基线 68% 缩减 4.3% 损失
- 7B 模型：DRPO 51% 长度缩减仅 2.6% 性能损失 vs RLOO-LP 38% 缩减 7.1% 损失
- $\lambda$ 可以平滑控制长度-准确率 trade-off：$\lambda\to\infty$ 无长度控制，$\lambda\to 0$ 最大化长度惩罚
- 在非数学推理任务（K&K 逻辑谜题）上也有效

## 亮点与洞察
- **"正负解耦"是核心洞察**：GRPO 的问题不在长度惩罚本身，而在于正负样本混合归一化。解耦后问题自然解决——这是一个清晰优雅的诊断和修复
- **闭式最优分布的理论优美性**：不需要额外训练奖励模型或采集数据，直接从 RLHF 的 KL 正则化框架推导出加权方案
- **对所有 GRPO 变体的通用诊断**：论文指出 RLOO、REINFORCE 等所有相对优势方法在复合奖励下都有此问题——DRPO 的解耦原则具有通用性

## 局限与展望
- 基于 DisCO 框架而非 GRPO，可能需要更多工程适配
- 长度奖励 $r_l(o) = 1 - |o|/C$ 是简单线性的，更复杂的长度-质量关系可能需要非线性设计
- 仅在数学推理上验证，代码推理、科学推理等其他领域需要确认
- 生成预算限制在 8K token，对超长推理的效果未知

## 相关工作与启发
- **vs GRPO + 长度惩罚（RLOO-LP/ALP/HAPO）**: 这些方法都受限于混合归一化导致的负优势问题，DRPO 通过解耦彻底解决
- **vs DisCO**: DRPO 在 DisCO 基础上引入了长度奖励的闭式加权方案，是 DisCO 在高效推理方向的自然扩展
- **vs L1-max / ShorterBetter**: 这些方法用不同机制控制长度，但也面临性能-效率 trade-off。DRPO 的 AES 持续最优
- **vs VIP (Adaptive Rollout)**: VIP 在采样前优化计算分配，DRPO 在训练目标上优化学习信号——两者互补

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 诊断清晰、解决方案优雅、闭式解理论完备
- 实验充分度: ⭐⭐⭐⭐⭐ 3 个模型、6 个基线、4 个难度级别、AES 定量对比
- 写作质量: ⭐⭐⭐⭐⭐ Figure 1 的诊断直观易懂，理论推导干净
- 价值: ⭐⭐⭐⭐⭐ 解决了高效推理训练中的核心矛盾，实用性极强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Reference-guided Policy Optimization for Molecular Optimization via LLM Reasoning](reference-guided_policy_optimization_for_molecular_optimization_via_llm_reasonin.md)
- [\[ICLR 2026\] PEAR: Phase Entropy Aware Reward for Efficient Reasoning](pear_phase_entropy_aware_reward_for_efficient_reasoning.md)
- [\[ICLR 2026\] HiPO: Self-Hint Policy Optimization for RLVR](hipo_self-hint_policy_optimization_for_rlvr.md)
- [\[ICLR 2026\] Slow-Fast Policy Optimization: Reposition-Before-Update for LLM Reasoning](slow-fast_policy_optimization_reposition-before-update_for_llm_reasoning.md)
- [\[ICLR 2026\] Adaptive Social Learning via Mode Policy Optimization for Language Agents](adaptive_social_learning_via_mode_policy_optimization_for_language_agents.md)

</div>

<!-- RELATED:END -->
