---
title: >-
  [论文解读] Reverse Engineering Human Preferences with Reinforcement Learning
description: >-
  [NeurIPS 2025 (Spotlight)][LLM安全][LLM-as-a-Judge] 使用强化学习训练前导文本生成器来提升下游 LLM 的评分成绩,揭示了 LLM-as-a-Judge 评估框架的脆弱性,且该攻击方式几乎不可检测并具有跨模型迁移能力。 解决思路 本文目标：领域现状：LLM-as-a-Judge…
tags:
  - "NeurIPS 2025 (Spotlight)"
  - "LLM安全"
  - "LLM-as-a-Judge"
  - "对抗攻击"
  - "偏好逆向"
  - "强化学习"
  - "可检测性"
---

# Reverse Engineering Human Preferences with Reinforcement Learning

**会议**: NeurIPS 2025 (Spotlight)

**arXiv**: [2505.15795](https://arxiv.org/abs/2505.15795)

**代码**: 无

**领域**: AI安全

**关键词**: LLM-as-a-Judge, 对抗攻击, 偏好逆向, 强化学习, 可检测性

## 一句话总结

使用强化学习训练前导文本生成器来提升下游 LLM 的评分成绩,揭示了 LLM-as-a-Judge 评估框架的脆弱性,且该攻击方式几乎不可检测并具有跨模型迁移能力。

## 研究背景与动机

### 解决思路

**本文目标**：**领域现状**：LLM-as-a-Judge 已成为评估 LLM 能力的主流框架——利用一个强大的 LLM 作为评判者来预测人类偏好。然而,这一框架存在被恶意利用的风险：

**可游戏性**: LLM 的输出可以被针对性优化以取悦评判模型

**先前方法的局限**: 已有的后处理编辑方法直接修改模型回答,容易被检测

**安全威胁**: 如果排行榜分数可以被"刷"上去,整个评估体系的可信度将受到质疑

## 方法详解

### 整体框架

训练一个**前导文本生成器**（preamble generator）模型,该模型在候选 LLM 的回答之前生成一段隐式引导文本，使评判 LLM 给出更高分数。

### 关键设计

**1. 管线架构**

两个 LLM 串联使用:
- **前导生成器 $\pi_\phi$**: 根据 prompt 生成前导文本 $p$
- **候选 LLM $M$** (冻结): 将 $[p; \text{prompt}]$ 作为输入,生成回答 $y$
- **评判 LLM $J$**: 对回答 $y$ 打分

关键: 候选 LLM 保持冻结,仅训练前导生成器。

**2. RL 训练**

- **奖励信号**: 评判 LLM 对回答的评分
- **策略**: 前导生成器 $\pi_\phi$ 的参数
- **目标**: $\max_\phi \mathbb{E}_{p \sim \pi_\phi}[J(M(p, \text{prompt}))]$
- 使用 PPO 进行优化

**3. 隐蔽性**

- 与直接编辑回答不同,前导文本是模型内部使用的,用户和评判者看到的回答仍是候选 LLM 的自然输出
- 前导文本影响模型的激活状态,从而改变生成行为
- 几乎**不可检测**: 回答本身的分布变化微小

### 损失函数 / 训练策略

$$\mathcal{L}(\phi) = -\mathbb{E}_{p \sim \pi_\phi}[R(p)] + \beta \text{KL}(\pi_\phi \| \pi_0)$$

其中 $R(p)$ 是评判分数, $\pi_0$ 是前导生成器的初始策略。

## 实验关键数据

### 主实验

不同框架的评估分数提升 (AlpacaEval 2.0, Win Rate vs GPT-4):

| 方法 | 基线 Win Rate | 攻击后 Win Rate | 可检测性 |
|------|-------------|---------------|---------|
| 直接回答 | 25.3% | - | - |
| 后处理编辑 | - | 38.5% | 高 (82%) |
| Prompt注入 (手工) | - | 32.1% | 中 (55%) |
| 前导生成器 (Ours) | - | **42.8%** | **低 (12%)** |

跨模型迁移性:

| 训练评判模型 | 测试评判模型 | 基线 | 攻击后 | 迁移效果 |
|------------|------------|------|--------|---------|
| GPT-4 | GPT-4 | 25.3% | 42.8% | (训练本身) |
| GPT-4 | Claude-3 | 28.1% | 38.5% | ✓ 有效 |
| GPT-4 | Gemini | 26.5% | 36.2% | ✓ 有效 |
| Claude-3 | GPT-4 | 25.3% | 37.5% | ✓ 有效 |

### 消融实验

前导文本长度对攻击效果的影响:

| 前导长度 (tokens) | Win Rate | 可检测性 | 回答自然度 |
|-----------------|----------|---------|----------|
| 0 (无前导) | 25.3% | 0% | 100% |
| 16 | 32.5% | 5% | 98% |
| 64 | 38.8% | 8% | 95% |
| 128 | 42.8% | 12% | 92% |
| 256 | 43.2% | 25% | 85% |

### 关键发现

1. 前导生成器将 Win Rate 从 25.3% 提升到 42.8%,提升幅度远超后处理编辑方法
2. 攻击的可检测性仅 12%,远低于直接编辑（82%）
3. **跨模型迁移**: 在一个评判模型上训练的前导生成器在其他评判模型上同样有效
4. 这表明人类偏好的某些特征是**模型无关**的,可以被系统性利用

## 亮点与洞察

- **Spotlight 论文**: 揭示了 LLM-as-a-Judge 的根本性安全漏洞
- **不可检测性**: 核心创新在于攻击是通过前导文本(不可见)而非修改回答(可见)
- **跨模型迁移**: 暗示偏好判断中存在系统性偏差,而非个别模型的弱点
- **双重含义**: 既是安全警报,也为利用前导文本优化上游输入提供了新思路

## 局限与展望

1. 前导文本虽"不可见",但分析系统 prompt 仍可发现
2. 如果评估者同时检查系统 prompt,攻击有效性降低
3. RL 训练需要大量评判 API 调用,成本较高
4. 研究主要聚焦英文场景,跨语言效果未验证

## 相关工作与启发

- **LLM-as-a-Judge** (Zheng et al.): 建立了LLM评估框架
- **Prompt Injection**: 系统 prompt 攻击的相关工作
- **AlpacaEval**: LLM 评估排行榜

## 评分

- ⭐ 创新性: 9/10 — 前导文本攻击思路新颖,揭示深层问题
- ⭐ 实用性: 8/10 — 对评估体系设计有直接指导意义
- ⭐ 写作质量: 9/10 — Spotlight级别,论证清晰有力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Reinforcement Learning with Backtracking Feedback](reinforcement_learning_with_backtracking_feedback.md)
- [\[NeurIPS 2025\] TRAP: Targeted Redirecting of Agentic Preferences](trap_targeted_redirecting_of_agentic_preferences.md)
- [\[NeurIPS 2025\] Contextual Integrity in LLMs via Reasoning and Reinforcement Learning](contextual_integrity_in_llms_via_reasoning_and_reinforcement_learning.md)
- [\[ICLR 2026\] Supervised Reinforcement Learning: From Expert Trajectories to Step-wise Reasoning](../../ICLR2026/llm_safety/supervised_reinforcement_learning_from_expert_trajectories_to_step-wise_reasonin.md)
- [\[ACL 2026\] Privacy-R1: Privacy-Aware Multi-LLM Agent Collaboration via Reinforcement Learning](../../ACL2026/llm_safety/privacy-r1_privacy-aware_multi-llm_agent_collaboration_via_reinforcement_learnin.md)

</div>

<!-- RELATED:END -->
