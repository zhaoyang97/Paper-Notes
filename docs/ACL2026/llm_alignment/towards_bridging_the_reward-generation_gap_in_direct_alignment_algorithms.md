---
title: >-
  [论文解读] Towards Bridging the Reward-Generation Gap in Direct Alignment Algorithms
description: >-
  [ACL 2026 Findings][LLM对齐][直接对齐算法] 本文识别了直接对齐算法（DAAs）中的"奖励-生成鸿沟"——训练目标与自回归解码动态之间的不匹配，提出 POET（Prefix-Oriented Equal-length Training），通过将偏好响应对截断为较短者长度来隐式约束 token 级 MDP 在所有时间步上收敛，在 AlpacaEval 2 上最高提升 11.8 个百分点。
tags:
  - "ACL 2026 Findings"
  - "LLM对齐"
  - "直接对齐算法"
  - "前缀重要性"
  - "等长训练"
  - "奖励-生成鸿沟"
  - "DPO/SimPO"
---

# Towards Bridging the Reward-Generation Gap in Direct Alignment Algorithms

**会议**: ACL 2026 Findings  
**arXiv**: [2506.09457](https://arxiv.org/abs/2506.09457)  
**代码**: [GitHub](https://github.com/sustech-nlp/POET)  
**领域**: LLM 对齐 / 偏好优化  
**关键词**: 直接对齐算法, 前缀重要性, 等长训练, 奖励-生成鸿沟, DPO/SimPO

## 一句话总结

本文识别了直接对齐算法（DAAs）中的"奖励-生成鸿沟"——训练目标与自回归解码动态之间的不匹配，提出 POET（Prefix-Oriented Equal-length Training），通过将偏好响应对截断为较短者长度来隐式约束 token 级 MDP 在所有时间步上收敛，在 AlpacaEval 2 上最高提升 11.8 个百分点。

## 研究背景与动机

**领域现状**：DPO 和 SimPO 等直接对齐算法（DAAs）已成为 RLHF 的高效替代方案。DAAs 通过隐式奖励函数直接在偏好数据集上优化，无需显式奖励模型和强化学习。

**现有痛点**：(1) DAAs 可能在增大偏好-非偏好奖励差距的同时降低偏好响应的绝对奖励；(2) 更高的偏好奖励和更大的奖励差距并不一定带来更好的生成质量；(3) DAAs 的隐式奖励对每个 token 赋予相同权重，忽略了前缀 token 在自回归生成中的关键重要性。

**核心矛盾**：DAAs 在序列级别优化 $r(x, y_w) \gg r(x, y_l)$，但无法保证在前缀级别也满足 $r(x, y_{w,<k}) \gg r(x, y_{l,<k})$。而自回归生成中早期 token 的错误会通过暴露偏差累积放大，前缀质量决定了整体生成质量。

**本文目标**：从 token 级 MDP 视角分析 DAAs 的局限性，并设计方法弥合训练目标与生成动态之间的鸿沟。

**切入角度**：实证观察发现前缀 token 的熵显著高于后续 token，但其对数概率被大量后续 token 的均值稀释，导致 DAAs 无法充分关注前缀的质量差异。

**核心 idea**：将偏好响应对截断为等长（取较短者长度），使 DAAs 的训练目标隐式约束在所有时间步上收敛，从而增强对前缀 token 的关注。

## 方法详解

### 整体框架

POET 要弥合的是直接对齐算法（DAAs）里的"奖励-生成鸿沟"：DPO/SimPO 在序列级别优化 $r(x,y_w)\gg r(x,y_l)$，却管不住前缀级别——而自回归生成中早期 token 的错误会被暴露偏差累积放大。它的做法极简：输入一对偏好响应 $(y_w,y_l)$，把两者都截断到较短者长度 $k=\min(|y_w|,|y_l|)$，中间产物是一批等长的偏好对，输出就是在这批等长数据上跑标准 DPO 或 SimPO。方法不动任何优化目标，因此兼容所有 DAAs。

### 关键设计

**1. 等长子轨迹 BT 模型的理论基础：证明截断不改变最优策略**

POET 的合法性首先要回答一个问题——把响应截短了，训练出来的策略会不会偏？作者定义等长子轨迹 BT 模型 $p_k^*(y_{w,\leq k} \succeq y_{l,\leq k})$，其中显式包含了截断点之后的最优状态值函数 $V^*$。定理 1 证明：从这个等长子轨迹模型导出的最优策略，与从原始序列级 BT 模型导出的最优策略完全等价。

这给出了严格的理论保证——截断本身不会改变最优策略，而在多种截断长度上训练，反而能为前缀 token 提供更细粒度的奖励信号，正好对症 DAAs 对前缀关注不足的毛病。

**2. 前缀质量差异的实证验证：确认截断后偏好排序仍然成立**

理论保证之外，还要确认用全序列偏好标签去监督截断后的等长对是否安全。作者在 1000 个样本上，对不同前缀长度 $k$ 计算偏好-非偏好的前缀质量差 $\Delta Q(k) = Q(y_{w,\leq k}) - Q(y_{l,\leq k})$。结果显示质量差在前缀非常早期就出现并随长度增长，但边际收益递减，截断后偏好排序高度一致（98.5%）。

正因为截断后偏好排序几乎不变，用全序列偏好标签训练等长对才是安全的——这个实证验证是 POET 可行性的关键支撑，否则截断会引入大量标签噪声。

**3. POET 数据增强策略：用最短长度截断，无额外超参数**

具体取 $k = \min(|y_w|, |y_l|)$，意味着一对响应里总有一个保持完整，只截掉较长者的后缀。由于数据集中不同样本的 $k$ 值各不相同，训练隐式地在多种截断长度上进行，从而约束 DAAs 在所有 MDP 时间步上收敛、自然地把注意力拉回前缀质量。

这种设计有三重好处：通用兼容任何 DAAs；不引入任何额外超参数；只截后缀因此把数据噪声风险压到最低——较短响应长度以上的后缀对整体质量影响本就很小。

### 损失函数 / 训练策略

不修改 DPO/SimPO 的优化目标，仅改变输入数据。按 Meng et al. (2024) 的超参数设置训练。支持 Base（从 SFT 模型开始）和 Instruct（从指令微调模型开始）两种设置。

## 实验关键数据

### 主实验

**AlpacaEval 2 & Arena-Hard 指令遵循评估**

| 方法 | Mistral-7B LC% | Llama-3-8B LC% | Llama-3-Inst LC% | Gemma-2-9B LC% |
|------|----------------|----------------|-------------------|----------------|
| DPO | 12.9 | 16.9 | 65.9 | 78.4 |
| DPO + POET | **24.7** (+11.8) | **28.4** (+11.5) | **70.4** (+4.5) | **79.7** (+1.3) |
| SimPO | 20.0 | 28.0 | 68.1 | 78.5 |
| SimPO + POET | **24.2** (+4.2) | **33.8** (+5.8) | **70.1** (+2.0) | **80.1** (+1.6) |

### 消融实验

**截断策略对比（Mistral-7B, AlpacaEval 2 LC%）**

| 截断策略 | 保留 25% | 50% | 75% | 100% |
|---------|---------|-----|-----|------|
| 原始长度 | 14.1 | 17.2 | 16.2 | 12.9 |
| POET 长度 | 23.5 | 24.9 | 26.7 | 24.7 |

### 关键发现

- POET 在所有 4 个模型 × 2 个 DAA 的 8 种设置中一致提升 AlpacaEval 2 LC，最高 +11.8 个百分点
- 消融实验证明等长截断策略（POET Len.）在所有保留比例下都显著优于按原始长度截断，说明等长本身是关键
- POET 不增加对齐税——在 HuggingFace Open Leaderboard 下游任务上保持或略微提升
- 与 token 级方法对比：POET 显著优于 SamPO 和 D2PO，后者甚至在 SimPO 上适得其反
- 安全对齐评估也显示安全率的显著提升

## 亮点与洞察

- 问题识别精准——"奖励-生成鸿沟"抓住了 DAAs 的一个根本性问题：序列级优化目标与自回归生成动态的不匹配
- 方法极简但有效——不修改目标、不加超参数、仅截断数据就能大幅提升，说明问题的根源确实在数据形式而非优化算法
- 前缀质量差异的实证分析（Figure 2）是全文最有说服力的部分——用 oracle 模型验证了"截断后偏好排序几乎不变"这一关键假设

## 局限与展望

- 等长截断依赖"较短响应长度以上的后缀对质量影响小"这一假设，在极端不对称长度的样本中可能不成立
- 理论保证假设存在最优状态值函数 $V^*$，实际中无法计算
- 仅在 DPO 和 SimPO 上验证，未扩展到 IPO、KTO 等其他 DAAs
- 截断后偏好排序虽然一致性高（91.4%-98.5%），但仍有一定噪声引入

## 相关工作与启发

- **vs SamPO (Lu et al., 2024)**: SamPO 随机子集 token 计算奖励来减轻长度偏见，但不聚焦前缀；POET 聚焦前缀且在 DPO/SimPO 上都有效
- **vs D2PO (Shao et al., 2025)**: D2PO 用指数衰减权重强调前缀，但衰减因子引入额外超参数且在 SimPO 上反效果；POET 无超参数且一致有效
- **vs Token-level DPO (Rafailov et al., 2024b)**: 理论上 token-level DPO 可学到最优策略，但实际中只有序列级偏好标签，无法直接训练

## 评分

- 新颖性: ⭐⭐⭐⭐ 问题识别清晰，方法虽简单但理论动机和实证验证充分
- 实验充分度: ⭐⭐⭐⭐⭐ 4 模型 × 2 DAA × 详细消融 + token 级方法对比 + 安全评估
- 写作质量: ⭐⭐⭐⭐⭐ 从理论到实证到方法的逻辑链严密
- 价值: ⭐⭐⭐⭐⭐ 即插即用、无超参数、兼容所有 DAAs，实用价值极高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] MPO: Multilingual Safety Alignment via Reward Gap Optimization](../../ACL2025/llm_alignment/mpo_multilingual_safety_alignment.md)
- [\[ACL 2026\] BACH-V: Bridging Abstract and Concrete Human-Values in Large Language Models](bach-v_bridging_abstract_and_concrete_human-values_in_large_language_models.md)
- [\[ACL 2026\] RbtAct: Rebuttal as Supervision for Actionable Review Feedback Generation](rbtact_rebuttal_as_supervision_for_actionable_review_feedback_generation.md)
- [\[CVPR 2026\] Bridging Human Evaluation to Infrared and Visible Image Fusion](../../CVPR2026/llm_alignment/bridging_human_evaluation_to_infrared_and_visible_image_fusion.md)
- [\[ICLR 2026\] Is On-Policy Data always the Best Choice for Direct Preference Optimization-based LM Alignment?](../../ICLR2026/llm_alignment/is_on-policy_data_always_the_best_choice_for_direct_preference_optimization-base.md)

</div>

<!-- RELATED:END -->
