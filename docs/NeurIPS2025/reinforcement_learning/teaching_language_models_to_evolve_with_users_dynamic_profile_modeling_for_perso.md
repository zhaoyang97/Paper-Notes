---
title: >-
  [论文解读] Teaching Language Models to Evolve with Users: Dynamic Profile Modeling for Personalized Alignment
description: >-
  [NeurIPS 2025][强化学习][个性化对齐] 将个性化对话对齐建模为多轮马尔可夫决策过程，提出 RLPA 框架，让 LLM 通过与模拟用户的在线交互学习动态推断和维护用户画像，并据此生成个性化回复。 大语言模型的对齐技术（如 RLHF）已让模型很好地服务于"大众化"的偏好，但现实场景中每个用户的需求、目标、风格都不…
tags:
  - "NeurIPS 2025"
  - "强化学习"
  - "个性化对齐"
  - "用户画像建模"
  - "多轮对话"
  - "冷启动"
---

# Teaching Language Models to Evolve with Users: Dynamic Profile Modeling for Personalized Alignment

**会议**: NeurIPS 2025  
**arXiv**: [2505.15456](https://arxiv.org/abs/2505.15456)  
**代码**: [GitHub](https://github.com/XingYuSSS/RLPA)  
**领域**: 强化学习  
**关键词**: 个性化对齐, 用户画像建模, 强化学习, 多轮对话, 冷启动

## 一句话总结

将个性化对话对齐建模为多轮马尔可夫决策过程，提出 RLPA 框架，让 LLM 通过与模拟用户的在线交互学习动态推断和维护用户画像，并据此生成个性化回复。

## 研究背景与动机

大语言模型的对齐技术（如 RLHF）已让模型很好地服务于"大众化"的偏好，但现实场景中每个用户的需求、目标、风格都不同。现有的个性化方案存在两类根本局限：

**基于提示词的方法**（profile/RAG 注入）虽然实现简单，但只能做浅层个性化——模板化的信息注入既无法灵活表达用户偏好，也受上下文窗口限制而无法维护长期记忆。**离线优化方法**（SFT/DPO）依赖大规模标注数据，在冷启动场景下不可行，且静态训练使得模型无法在交互中实时适应用户的动态变化。

核心矛盾在于：个性化本质上是一个**动态演化**的过程——用户画像需要在对话中"从无到有"逐步构建（冷启动）并持续更新（长期维护），而现有方法都是静态的。本文的切入角度是：将个性化对齐建模为**多轮 MDP**，用强化学习让模型在与用户的在线交互中学会推断、维护和利用用户画像。

## 方法详解

### 整体框架

RLPA（Reinforcement Learning for Personalized Alignment）将个性化对话形式化为一个五元组 MDP $(\mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \gamma)$：

- **状态** $s_t = \{u_1, r_1, \dots, u_t\}$：当前对话历史
- **动作** $a_t$：模型在第 $t$ 轮生成的回复 $r_t$
- **转移** $\mathcal{T}$：模拟用户根据对话历史生成下一条消息 $u_{t+1}$
- **奖励** $R_t$：由 Profile Reward + Response Reward 组成的双层奖励

目标是学习对话策略 $\pi(a_t|s_t)$ 以最大化：

$$J(\pi) = \mathbb{E}_{\pi}\left[\sum_{t=1}^{T} \gamma^{t-1} R_t\right]$$

### 关键设计

1. **模拟用户模型（Simulated User）**：用一个 LLM（GPT-4o-mini）作为模拟用户，其行为由预设的用户画像 $\mathcal{P} = \{p_1, p_2, \dots, p_n\}$（如偏好、性格、目标）约束。关键设计是**渐进暴露**——模拟用户被指示在多轮对话中逐步透露画像信息，而非一次性全部暴露。这迫使对话模型通过多轮推理来积累和完善对用户画像的理解。设计动机是模拟真实冷启动场景。

2. **Profile Reward（画像奖励）**：引导模型准确推断用户画像。用户画像采用 slot-value 结构化格式。每轮对话后，模型输出当前画像估计 $\hat{\mathcal{P}}_t$，与真实画像 $\mathcal{P}$ 做 slot 级别匹配：

$$R_t^{\text{profile}} = \frac{2 \cdot |\hat{\mathcal{P}}_t \cap \mathcal{P}|}{|\hat{\mathcal{P}}_t| + |\mathcal{P}|}$$

本质上是 F1-score，同时惩罚遗漏和错误预测，鼓励模型渐进式推断完整画像。

3. **Response Reward（回复奖励）**：确保生成的回复忠实反映推断出的画像。由外部奖励模型（GPT-4o-mini）从四个维度（偏好表达、风格一致性、目标对齐、人格连贯性）打分，并结合五个二元质量标准：

$$R_t^{\text{response}} = N \cdot R \cdot L \cdot G \cdot F$$

其中 N=自然度、R=相关性、L=逻辑一致性、G=参与度、F=信息量。乘法形式确保只有全面达标的回复才能获得正向奖励。

### 损失函数 / 训练策略

总奖励为双层奖励之和：$R_t = R_t^{\text{profile}} + R_t^{\text{response}}$。使用 PPO 算法优化策略。基于 OpenRLHF + vLLM 框架在 8×A100 80GB 上训练 Qwen-2.5-3B-Instruct，得到 Qwen-RLPA。

## 实验关键数据

### 主实验

在 ALOE benchmark 两个设置（Vanilla: 同格式泛化 / Extended: 跨格式泛化）上评估：

| 模型 | 方法 | Vanilla AVG↑ | Vanilla N-R²↑ | Extended AVG↑ | Extended N-R²↑ |
|------|------|-------------|---------------|--------------|----------------|
| GPT-4o | Self-Critic | 74.59 | 0.380 | 54.66 | 0.070 |
| Claude-3.5-Sonnet | Self-Critic | 69.19 | 0.792 | 35.62 | 0.266 |
| Qwen-2.5-3B | SFT | 44.32 | 0.628 | 24.38 | 0.159 |
| Qwen-2.5-3B | DPO | 45.27 | 0.389 | 27.26 | 0.037 |
| **Qwen-2.5-3B** | **RLPA** | **73.38** | **0.855** | **52.74** | **0.498** |

3B 模型通过 RLPA 训练后，对齐分数超过 GPT-4o 和 Claude-3.5，且 N-R² 远高于所有基线（0.855 vs 0.380/0.500），表明回复-画像一致性更强。

### 消融实验

| 配置 | AVG↑ | N-IR↑ | N-R²↑ | 说明 |
|------|------|-------|-------|------|
| RLPA (完整) | 73.38 | 0.090 | 0.855 | 双层奖励协同最优 |
| 仅 Profile Reward | 45.54 | 0.015 | 0.019 | 能推断画像但难以反映在回复中 |
| 仅 Response Reward | 66.19 | 0.088 | 0.524 | 回复个性化但画像不稳定 |

### 关键发现

- **长期稳定性**：在 70 轮长对话中，画像推断分数从第 1 轮的 9.26 稳步上升到第 50 轮的 52.54，接近理论上限，证明 RLPA 支持长期画像维护。
- **偏好冲突处理**：在第 6 轮引入偏好突变时，RLPA 仅短暂下降后迅速恢复（82.43@k=9），而 DPO 从 71.62 暴跌至 25.68 且难以恢复。
- **推理效率**：与 DeepSeek-R1、QwQ-32B 等推理型 LLM 相比，Qwen-RLPA 用更少的推理 token 获得更高的对齐分数（QwQ-32B 每轮 300+ token 但效果更差）。

## 亮点与洞察

- 将**个性化对齐**重新定义为动态 MDP 而非静态拟合问题，抓住了个性化的本质——"演化性"。
- 双层奖励设计精巧：Profile Reward 做"显式画像监督"，Response Reward 做"隐式行为监督"，二者互补缺一不可。
- 3B 小模型通过 RLPA 训练可以媲美甚至超越 GPT-4o 等大模型，说明"正确的训练范式"比"模型规模"更重要。

## 局限与展望

- 目前假设单用户单线程交互，未考虑多用户、多会话、跨领域场景。
- 模拟用户与真实用户之间可能存在行为分布差距（sim-to-real gap）。
- 长期对齐的理论收敛性质尚未充分研究。
- 依赖 GPT-4o-mini 作为奖励模型，成本和可控性值得探讨。

## 相关工作与启发

- 与 persona-based dialogue 关联但更进一步：不是给定画像生成回复，而是从对话中推断画像。
- RLPA 的思路可迁移到其他需要"从交互中学习用户偏好"的场景，如推荐系统、教育 AI。
- 双层奖励的设计模式（过程奖励 + 结果奖励）与数学推理 RL 中的 PRM/ORM 设计思路相通。

## 评分

- **新颖性**: ⭐⭐⭐⭐☆ — 将个性化对齐建模为 MDP+RL 的思路新颖，双层奖励设计有创意
- **实验充分度**: ⭐⭐⭐⭐⭐ — 消融、长期稳定性、偏好冲突、推理效率等分析全面
- **写作质量**: ⭐⭐⭐⭐☆ — 结构清晰，图示直观，但部分细节在附录中
- **价值**: ⭐⭐⭐⭐☆ — 对个性化 LLM 有实际参考价值，3B 打败 GPT-4o 的结果很有说服力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Training Language Models to Reason Efficiently](training_language_models_to_reason_efficiently.md)
- [\[NeurIPS 2025\] Checklists Are Better Than Reward Models For Aligning Language Models](checklists_are_better_than_reward_models_for_aligning_langua.md)
- [\[NeurIPS 2025\] MMaDA: Multimodal Large Diffusion Language Models](mmada_multimodal_large_diffusion_language_models.md)
- [\[NeurIPS 2025\] The Burden of Interactive Alignment with Inconsistent Preferences](the_burden_of_interactive_alignment_with_inconsistent_preferences.md)
- [\[NeurIPS 2025\] Reinforcing the Diffusion Chain of Lateral Thought with Diffusion Language Models](reinforcing_the_diffusion_chain_of_lateral_thought_with_diffusion_language_model.md)

</div>

<!-- RELATED:END -->
