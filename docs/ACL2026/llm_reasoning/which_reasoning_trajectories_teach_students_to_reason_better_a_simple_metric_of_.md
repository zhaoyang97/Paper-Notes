---
title: >-
  [论文解读] Which Reasoning Trajectories Teach Students to Reason Better? A Simple Metric of Informative Alignment
description: >-
  [ACL 2026][LLM推理][知识蒸馏] 提出 Rank-Surprisal Ratio (RSR) 指标，通过联合衡量推理轨迹对学生模型的"信息量"和"对齐度"来评估训练数据适配性，在 5 个学生模型和 11 个教师模型的组合中与训练后性能达到平均 0.86 的 Spearman 相关性，并成功应用于轨迹选择和教师选择。
tags:
  - "ACL 2026"
  - "LLM推理"
  - "知识蒸馏"
  - "推理轨迹"
  - "数据选择"
  - "链式思维"
  - "大语言模型"
---

# Which Reasoning Trajectories Teach Students to Reason Better? A Simple Metric of Informative Alignment

**会议**: ACL 2026  
**arXiv**: [2601.14249](https://arxiv.org/abs/2601.14249)  
**代码**: [GitHub](https://github.com/UmeanNever/RankSurprisalRatio)  
**领域**: 模型压缩  
**关键词**: 知识蒸馏, 推理轨迹, 数据选择, 链式思维, 大语言模型

## 一句话总结

提出 Rank-Surprisal Ratio (RSR) 指标，通过联合衡量推理轨迹对学生模型的"信息量"和"对齐度"来评估训练数据适配性，在 5 个学生模型和 11 个教师模型的组合中与训练后性能达到平均 0.86 的 Spearman 相关性，并成功应用于轨迹选择和教师选择。

## 研究背景与动机

**领域现状**：长链式思维（Long CoT）轨迹已成为从大型推理模型向小型学生模型蒸馏推理能力的主要手段，通过 SFT 让学生模型学习教师的推理过程。

**现有痛点**：实验反复证明，更强的教师模型（如 671B 的 DeepSeek-R1）并不一定能训练出更好的学生。数据与学生模型之间的"适配性"是决定蒸馏效果的关键因素，但现有方法主要依赖学生模型对数据的对数概率（log-probability）来衡量适配性，倾向于选择学生已经熟悉的高概率轨迹，忽略了那些真正有学习价值的数据。

**核心矛盾**：信息量与对齐度之间的权衡——太熟悉的数据没有学习价值，太陌生的数据又学不会。这呼应了心理学中"最近发展区"（Zone of Proximal Development）的概念：最有效的学习材料应当略超出学习者当前能力范围但又不至于完全无法理解。

**本文目标**：设计一个简单、可解释的指标来度量推理轨迹对特定学生模型的适配性，兼顾信息量和对齐度。

**切入角度**：作者观察到有效的轨迹呈现一种特殊模式——其 token 在学生模型下的绝对概率很低（高 surprisal，说明不是学生会生成的内容），但在词表排序中仍然排名靠前（低 rank，说明仍在学生理解范围内）。这种"绝对不熟悉但相对熟悉"的特征恰好平衡了信息量和对齐度。

**核心 idea**：用 token 排名与 surprisal 的比值（RSR）来衡量轨迹适配性——RSR 越低，说明轨迹既有信息量又与学生对齐。

## 方法详解

### 整体框架

整个方法分三步：(1) 对给定推理轨迹，用学生模型做一次前向传播，获取每个 token 的概率分布；(2) 计算每个 token 的 surprisal（负对数似然）和 rank（在词表中的排名）；(3) 将轨迹级别的 RSR 定义为平均 rank 与平均 surprisal 的比值，用于评估轨迹适配性。RSR 仅需一次前向传播，不需要额外的验证器或测试数据。

### 关键设计

**1. 双模态分布理论模型：为什么"绝对不熟悉但相对熟悉"才是好教材**

如果只看 surprisal，会把所有学生没见过的内容都当成有价值；如果只看 rank，又会把学生本就熟悉的高频内容选进来。要把"有信息量且对齐"和"有信息量但不对齐"区分开，单一维度不够。作者把学生模型的 token 预测分布建模成一个双模态混合分布：主模态 $Z_A$ 是学生的主导生成模式（高概率、低 rank，即它本来就会写的东西），次模态 $Z_B$ 是偏离主模态但仍落在学生知识范围内的模式（低概率，但 rank 依然靠前）。

有效的教师轨迹应当对应 $Z_B$——绝对概率低（学生不会主动生成，所以有信息量），但排名相对高（学生仍能理解，所以对齐）。模拟实验印证了这个划分：$Z_B$ 类轨迹的 RSR 最低（1.30），而真正错位、学生学不动的 $Z_C$ 类轨迹 RSR 最高（2.23）。这正好解释了为什么 RSR 要用 rank 与 surprisal 的比值——只有两者一起看，才能把 $Z_B$ 从 $Z_C$ 里分出来。

**2. Surprisal 加权聚合：把 token 级比值稳稳地汇成轨迹级指标**

token 级 RSR 定义为 $\text{RSR}_\text{token} = \text{Rank}(t_k) / \text{Surprisal}(t_k)$，但若直接对所有 token 取算术平均，高概率 token 的 surprisal 趋近零会让比值爆炸，数值极不稳定。作者改用 surprisal 加权平均，经数学推导后形式出人意料地简洁：

$$\text{RSR}(\mathbf{x}) = \frac{\sum_k \text{Rank}(t_k)}{\sum_k \text{Surprisal}(t_k)}$$

即"平均 rank 除以平均 surprisal"，既消除了除零爆炸，又自然地让 surprisal 高的 token（对学生学习影响更大）在指标里占更大权重。这一步是整个指标的命脉：消融里去掉加权改回逐 token 平均，相关性从 0.856 暴跌到 0.391。

**3. Rank 截断：把极端陌生 token 的噪声压平**

极端不熟悉的 token，其 rank 可能一路飙到词表大小（如 128K）。但对学生来说，rank 是 5 万还是 12 万本质上没有区别——都是"完全不会"，保留这些巨大的原始值只会给指标注入噪声。作者因此把 rank 截断到 $r_{max}$（默认 100），即取 $\min(\text{Rank}(t_k), r_{max})$。消融显示去掉截断后相关性从 0.856 降到 0.700，而 $r_{max}$ 在 100~500 区间内结果都稳健，说明这道截断既必要又不挑参数。

### 损失函数 / 训练策略

RSR 本身不是训练方法，而是数据选择指标。应用时，对候选轨迹计算 RSR，选择 RSR 最低的轨迹用于 SFT 训练。训练本身使用标准的监督微调损失。

## 实验关键数据

### 主实验（相关性分析）

| 指标 | Qwen-3-14B | LLaMA-3.1-8B | Qwen-2.5-7B | Qwen-3-4B | Qwen-2.5-3B | 平均 |
|------|-----------|-------------|------------|----------|-----------|------|
| Teacher Params | 0.04 | 0.34 | 0.20 | 0.02 | 0.26 | 0.01 |
| Avg-Surprisal | 0.24 | 0.42 | 0.55 | 0.55 | 0.70 | 0.49 |
| GRACE | 0.25 | 0.58 | 0.66 | 0.75 | 0.69 | 0.59 |
| **RSR (本文)** | **0.85** | **0.85** | **0.92** | **0.82** | **0.85** | **0.86** |

### 消融实验

| 配置 | 平均相关性 | 变化 |
|------|---------|------|
| RSR ($r_{max}=100$) | 0.856 | - |
| 无 rank 截断 | 0.700 | -0.156 |
| 无加权平均 (Avg-RSRtoken) | 0.391 | -0.465 |
| 过滤平均 (top 30%) | 0.793 | -0.064 |
| $r_{max}=500$ | 0.822 | -0.034 |
| 仅用 200 样本 | 0.864 | +0.007 |

### 关键发现
- RSR 在所有 5 个学生模型上都显著优于所有对比指标，平均 Spearman 相关性 0.86，次优方法（Rule-based Quality）仅 0.65
- surprisal 加权是最关键的设计，去掉后相关性暴跌 0.465
- RSR 对样本量不敏感，仅用 200 样本（原来的 4%）即可达到同等效果
- 在轨迹选择任务中，RSR 选出的数据训练效果可达甚至超过暴力搜索所有教师的最优结果

## 亮点与洞察

- **"绝对不熟悉+相对熟悉"的洞察极为精炼**：将信息量和对齐度的矛盾转化为 surprisal 和 rank 两个维度的对比，这一观察不仅指导了 RSR 设计，还为理解蒸馏数据选择提供了新视角
- **数学形式的优雅简化**：从 token 级加权平均出发，最终推导出极其简洁的"平均 rank / 平均 surprisal"形式，计算成本极低（单次前向传播）且可解释
- **迁移潜力大**：RSR 的核心思想——衡量数据对模型的"可学习性"——可以推广到任何 SFT 场景的数据选择，不限于推理任务

## 局限与展望

- 目前仅在数学推理任务上验证，尚未系统测试代码生成、通用对话等场景
- RSR 需要对每条轨迹做前向传播计算排名，对于超大规模数据集（百万级）的计算成本仍不可忽视
- 仅考虑了单条轨迹的适配性，未建模轨迹之间的多样性和互补性（子集选择场景）
- 作者在 Discussion 中提到 RSR 可用于非 CoT 数据和子集选择，但实验支持不足

## 相关工作与启发

- **vs Avg-Surprisal（Zhang et al.）**: 仅用概率衡量适配性，倾向选择学生已熟悉的数据，相关性仅 0.49；RSR 通过引入 rank 维度解决了"信息量盲区"
- **vs GRACE（Li et al.）**: 基于梯度的方法，需要额外计算梯度且相关性 0.59；RSR 更简单（仅需前向传播）且效果更好
- **vs Influence Score**: 受 influence function 启发，但在部分学生上表现不稳定（相关性波动大）；RSR 跨学生表现一致

## 评分

- 新颖性: ⭐⭐⭐⭐ 核心洞察清晰优雅，但本质上是两个已有指标的组合
- 实验充分度: ⭐⭐⭐⭐⭐ 5个学生×11个教师的大规模实验，消融详尽，两个下游应用验证
- 写作质量: ⭐⭐⭐⭐⭐ 从观察到理论到实验的论证逻辑极其流畅
- 价值: ⭐⭐⭐⭐ 对推理蒸馏数据选择有直接实用价值，但适用范围待扩展

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] LLM Reasoning as Trajectories: Step-Specific Representation Geometry and Correctness Signals](llm_reasoning_as_trajectories_step-specific_representation_geometry_and_correctn.md)
- [\[ACL 2026\] Do Not Step Into the Same River Twice: Learning to Reason from Trial and Error](do_not_step_into_the_same_river_twice_learning_to_reason_from_trial_and_error.md)
- [\[ACL 2026\] Chain-of-Thought as a Lens: Evaluating Structured Reasoning Alignment between Human Preferences and Large Language Models](chain-of-thought_as_a_lens_evaluating_structured_reasoning_alignment_between_hum.md)
- [\[ACL 2026\] Discovering a Shared Logical Subspace: Steering LLM Logical Reasoning via Alignment of Natural-Language and Symbolic Views](discovering_a_shared_logical_subspace_steering_llm_logical_reasoning_via_alignme.md)
- [\[NeurIPS 2025\] Martingale Score: An Unsupervised Metric for Bayesian Rationality in LLM Reasoning](../../NeurIPS2025/llm_reasoning/martingale_score_an_unsupervised_metric_for_bayesian_rationality_in_llm_reasonin.md)

</div>

<!-- RELATED:END -->
