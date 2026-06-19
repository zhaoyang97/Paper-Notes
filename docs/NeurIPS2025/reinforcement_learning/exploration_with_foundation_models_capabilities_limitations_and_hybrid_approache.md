---
title: >-
  [论文解读] Exploration with Foundation Models: Capabilities, Limitations, and Hybrid Approaches
description: >-
  [NeurIPS 2025][强化学习][foundation models] 系统评测 LLM/VLM 在经典 RL 探索任务（bandit、Gridworld、Atari）上的零样本能力，发现 VLM 存在"知行差距"（knowing-doing gap）——高层推理正确但低层控制失败，并提出简单的 VLM-RL 混合框架在理想条件下可显著加速学习。
tags:
  - "NeurIPS 2025"
  - "强化学习"
  - "foundation models"
  - "exploration"
  - "reinforcement-learning"
  - "VLM"
  - "knowing-doing gap"
---

# Exploration with Foundation Models: Capabilities, Limitations, and Hybrid Approaches

**会议**: NeurIPS 2025  
**arXiv**: [2509.19924](https://arxiv.org/abs/2509.19924)  
**代码**: 无  
**领域**: 强化学习  
**关键词**: foundation models, exploration, reinforcement-learning, VLM, knowing-doing gap

## 一句话总结

系统评测 LLM/VLM 在经典 RL 探索任务（bandit、Gridworld、Atari）上的零样本能力，发现 VLM 存在"知行差距"（knowing-doing gap）——高层推理正确但低层控制失败，并提出简单的 VLM-RL 混合框架在理想条件下可显著加速学习。

## 研究背景与动机

RL 中的探索问题在稀疏奖励环境下极具挑战性。基础模型（LLM/VLM）具备强大的语义先验和推理能力，能否用于改善探索效率？

现有研究的不足：

**评测范围窄**：MAB 实验仅关注复杂 prompt 技巧，未研究简单指令措辞的影响

**环境层次不全**：缺乏从简单（bandit）到复杂（Atari）的系统性渐进评测

**失败模式不明**：VLM 在视觉环境中为何失败？是理解问题还是执行问题？

本文通过**三级递进评测**（MAB → Gridworld → Atari）系统化回答这些问题，并通过定性分析揭示失败的根本原因。

## 方法详解

### 整体框架

三级评测体系：
1. **多臂 Bandit**（隔离探索-利用权衡）：对比隐式 vs. 显式 prompt 对 LLM 探索行为的影响
2. **Gridworld**（引入状态转移和记忆需求）：测试 LLM 在确定性/随机环境中的空间导航能力
3. **Atari 硬探索游戏**（高维视觉输入 + 稀疏奖励）：评测 GPT-4o 的零样本游戏能力
4. **混合框架**：VLM 定期接管 PPO agent 的控制，作为语义探索引导

### 关键设计

**Prompt 设计的对比实验**：
- 隐式（v1）：*"Your goal is to maximize the total reward by pulling the arm with the highest probability"* → 要求 LLM 自行推断需要探索
- 显式（v2）：*"Your goal is to maximize the total reward by finding out which arm has the highest probability"* → 直接指示探索

**Atari 的时序信息处理**：引入 $m=6$ 步的帧间隔（而非连续 4 帧），增加时序多样性，帮助 VLM 推断运动方向。对所有游戏使用统一的 minimal prompt。

**混合算法**：
- PPO agent 以概率 $\epsilon$ 被 VLM 接管控制，持续 $T$ 步
- VLM 作为"语义探索器"将 agent 引导到有前途的状态区域
- PPO 从新状态恢复标准 on-policy 学习

### 损失函数 / 训练策略

- 混合框架使用标准 PPO 损失
- VLM 零样本推理，无训练
- 对比基线：PPO + RND（Random Network Distillation）作为强探索基线
- 评测指标：累积奖励、遗憾、学习曲线

## 实验关键数据

### MAB 实验

| 模型 | 隐式 prompt (v1) | 显式 prompt (v2) | UCB | Thompson Sampling |
|------|-----------------|-----------------|-----|-------------------|
| GPT-3.5 | 高遗憾 | 中等遗憾 | 低遗憾 | 低遗憾 |
| GPT-4 | 中等遗憾 | **接近最优** | 低遗憾 | 低遗憾 |
| Gemini 1.0 | 高遗憾 | 中等遗憾 | — | — |
| Gemini 1.5 | 中等遗憾 | 中等偏低 | — | — |

次优性差距分析（GPT-4 显式 prompt）：

| $\Delta$ | GPT-4 vs. UCB/TS |
|----------|-------------------|
| 0.6 | 竞争力强 |
| 0.4 | 竞争力强 |
| 0.2 | **明显落后** |

### Atari 零样本实验

| 游戏 | GPT-4o | RB 250K | RB 2.5M | RB 25M | 人类 |
|------|--------|---------|---------|--------|------|
| Freeway | **21** | 8 | 32 | 32 | 29.6 |
| Gravitar | **500** | 64 | 199 | 2405 | 3351 |
| Montezuma | **0** | 0 | 50 | 544 | 4753 |
| Pitfall | -158 | -26 | -7 | -7 | 6464 |
| Private Eye | -1000 | 503 | 125 | 1573 | 69571 |
| Solaris | **600** | 681 | 1137 | 2093 | 12326 |
| Venture | 0 | 8 | 20 | 1513 | 1188 |

### Gridworld 结果

| 设置 | Action Only | Simple Plan | Focused Plan | PPO/RecPPO |
|------|------------|-------------|--------------|-------------|
| 确定性 | LLM 表现良好 | LLM 优秀 | LLM 优秀 | 收敛慢 |
| 随机（部分可观测）| **严重退化** | 有改善 | 有改善 | 最终收敛 |

### 混合框架实验（Freeway）

| 方法 | 100K 步后得分 | 收敛速度 |
|------|-------------|---------|
| Vanilla PPO | ~5 | 慢 |
| PPO + RND | ~15 | 中等 |
| **PPO + VLM** | **~25** | **快** |

### 关键发现

1. **显式 prompt 显著改善探索**：LLM 不会自行推断探索需求，需要明确指示
2. **Knowing-doing gap**：VLM 在 Freeway 识别"向上走"，在 Gravitar 识别敌人并开火（+250分），但在需要精确时序控制的游戏中完全失败
3. **失败模式分类**：
    - **精确控制失败**：Montezuma（正确推理"拿钥匙"但无法执行跳跃）
    - **自我识别失败**：Venture（无法识别粉色方块是玩家角色）
    - **时序推理失败**：Pitfall（理解"跳过坑"但时机把握不对）
4. **混合框架在理想条件下有效**：在 VLM 策略正确且控制简单的 Freeway 中，显著超越 PPO+RND

## 亮点与洞察

- **"知行差距"的精确刻画**：不是 VLM 不理解游戏，而是无法将理解转化为精确的低层动作——这是当前 VLM 作为自主 agent 的根本瓶颈
- **渐进式评测设计**：从 MAB（无状态转移）到 Gridworld（空间推理）到 Atari（视觉+稀疏奖励），层层递进地暴露 FM 能力边界
- **诚实的实验设计**：混合框架选择 Freeway（VLM 已知表现好的游戏）作为上界分析，明确声明不是通用解决方案
- **实用启示**：FM 更适合作为 RL 的"语义加速器"而非端到端控制器

## 局限与展望

- 混合框架仅在 Freeway 一个游戏验证，泛化性未知
- VLM 推理成本高（每步调用 GPT-4o），与样本效率的权衡未量化
- Atari 评测仅用 GPT-4o 一个模型，未对比开源 VLM
- 缺乏自适应介入机制——何时让 VLM 接管、何时交还给 RL 的决策应基于不确定性
- 未探索 VLM 作为奖励塑造器或状态抽象器的替代整合方式

## 相关工作与启发

- **Atari-GPT**（Waytowich et al.）：评测 VLM 在密集奖励 Atari 中的表现，本文聚焦稀疏奖励硬探索游戏
- **BALROG**（Paglieri et al.）：在 NetHack 中发现知行差距，本文在 Atari 中独立验证了同一现象
- **TextAtari**（Li et al.）：去除视觉瓶颈后 LLM 推理能力大幅提升 → 验证低层控制是主要瓶颈
- **Intelligent Go-Explore**（Lu et al.）：用 GPT-4 替代手工启发式选择回访状态，比本文的直接控制方式更成功
- **Motif**（Klissarov et al.）：LLM 作为内在奖励函数而非直接控制器 → 更好地利用了语义理解能力

## 评分

- **新颖性**: 7/10 — 系统评测有价值，但知行差距概念已有先例
- **实验充分度**: 7/10 — 渐进评测设计好，混合框架验证不够充分
- **实用性**: 6/10 — 混合框架过于简单，更多作为概念验证
- **写作质量**: 8/10 — 结构清晰，定性分析生动有图

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Foundation Models as World Models: A Foundational Study in Text-Based GridWorlds](foundation_models_as_world_models_a_foundational_study_in_text-based_gridworlds.md)
- [\[NeurIPS 2025\] Hybrid Latent Reasoning via Reinforcement Learning](hybrid_latent_reasoning_via_reinforcement_learning.md)
- [\[ICLR 2026\] Optimistic Task Inference for Behavior Foundation Models](../../ICLR2026/reinforcement_learning/optimistic_task_inference_behavior_models.md)
- [\[NeurIPS 2025\] Decoder-Hybrid-Decoder Architecture for Efficient Reasoning with Long Generation](decoderhybriddecoder_architecture_for_efficient_reasoning_wi.md)
- [\[NeurIPS 2025\] Interactive and Hybrid Imitation Learning: Provably Beating Behavior Cloning](interactive_and_hybrid_imitation_learning_provably_beating_behavior_cloning.md)

</div>

<!-- RELATED:END -->
