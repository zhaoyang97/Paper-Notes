---
title: >-
  [论文解读] TowerMind: A Tower Defence Game Learning Environment and Benchmark for LLM as Agents
description: >-
  [AAAI 2026 Oral][强化学习][LLM 智能体] 提出 TowerMind，一个基于塔防游戏的轻量级多模态环境，用于评估 LLM 的长期规划和决策能力，揭示了当前 LLM 与人类专家之间仍存在显著性能差距（最佳模型仅达人类专家 42% 的得分），并识别出规划验证不足、缺乏多终态思维、动作空间利用不充分等行为缺陷。
tags:
  - "AAAI 2026 Oral"
  - "强化学习"
  - "LLM 智能体"
  - "塔防游戏"
  - "实时策略游戏"
  - "benchmark"
  - "多模态评估"
---

# TowerMind: A Tower Defence Game Learning Environment and Benchmark for LLM as Agents

**会议**: AAAI 2026 Oral  
**arXiv**: [2601.05899](https://arxiv.org/abs/2601.05899)  
**代码**: [https://github.com/tb6147877/TowerMind](https://github.com/tb6147877/TowerMind)  
**领域**: 强化学习  
**关键词**: LLM 智能体, 塔防游戏, 实时策略游戏, benchmark, 多模态评估

## 一句话总结

提出 TowerMind，一个基于塔防游戏的轻量级多模态环境，用于评估 LLM 的长期规划和决策能力，揭示了当前 LLM 与人类专家之间仍存在显著性能差距（最佳模型仅达人类专家 42% 的得分），并识别出规划验证不足、缺乏多终态思维、动作空间利用不充分等行为缺陷。

## 研究背景与动机

**LLM 作为智能体的核心能力评估需求**：LLM 跨领域应用（医疗、办公自动化、设计）均依赖两项基础能力——**长期规划**（将高层任务分解为子目标序列）和**决策**（将子目标转化为可执行动作）。

**RTS 游戏是理想的测试平台**：实时策略游戏同时需要宏观战略规划和微观战术适应，天然适合评估这两项能力。然而现有 RTS 游戏环境存在问题：

**计算需求过高**：基于 StarCraft II 的 TextStarCraft II、LLM-PySC2 等需要约 30GB 磁盘空间、2GB 内存和专用 GPU

**缺乏文本观测支持**：轻量级 RTS 环境如 ELF、DeepRTS、Gym-μRTS 不支持文本观测和动作接口，与 LLM 不兼容

**塔防游戏的优势**：共享 RTS 核心机制但仅防守预定义敌人波次，避免对手不可预测性的干扰，支持更孤立地评估 LLM 的规划与决策能力。固定的塔位选项和预定义路线便于分析 LLM 的策略选择。

## 方法详解

### 整体框架

TowerMind 是一个基于 Unity 游戏引擎构建的塔防游戏环境，通过 Unity ML-Agents Toolkit 扩展为 AI 环境，遵循 OpenAI Gym 标准接口。核心特点：

- **轻量级**：仅需 0.15GB 磁盘和内存，CPU 即可运行（vs SC2LE 的 30GB+GPU）
- **多模态观测**：支持像素图像（512×512×3）、文本（JSON 格式）、结构化游戏状态三种观测
- **幻觉评估**：同时评估得分（能力）和有效动作率（可靠性）
- **高度可定制**：提供图形化关卡编辑器

### 关键设计

#### 1. 游戏机制设计

TowerMind 的地图定义在以 (0,0) 为中心、边长为 6 的正方形区域内，包含：

- **道路**：红/蓝方向曲线，引导敌人移动，由 2D 坐标路径点序列表示
- **塔位**：预定义的建塔位置，部分远离道路的塔位作为"误导性塔位"
- **三种塔**：弓箭塔（单体高伤）、法师塔（范围攻击）、骑士塔（召唤近战骑士）
- **英雄单位**：可精细控制移动和技能的强力单位
- **15 种敌人**：不同血量、速度、攻击力，部分有特殊能力（如兽人巫师可禁用附近塔）
- **战争迷雾**：白色云状区域随机移动，引入部分可观测性

设计动机：通过道路形状多样性、塔位分布差异、敌人组合变化以及战争迷雾，创造丰富的决策空间，无法用单一固定策略通关。

#### 2. 混合动作空间

动作表示为三维向量 $a = (x, y, c)$：

- $(x, y) \in [-3.0, 3.0]$：连续空间坐标
- $c \in \{0, 1, 2, \ldots, 11\}$：离散动作类型（建塔、升级、出售、部署骑士增援等 12 种动作）

仅符合游戏规则和当前状态的动作为"有效动作"，否则为"无效动作"不予执行。这一设计使得**有效动作率**成为衡量 LLM 幻觉的直接指标。

#### 3. 难度量化系统

定义关卡 $l$ 的难度为：$D(l) = d_r(l) + d_t(l) + d_e(l) + d_{re}(l)$

- 道路难度 $d_r$：道路数量/最大道路数量
- 塔位难度 $d_t$：塔位数量/最大塔位数量
- 敌人难度 $d_e$：敌人类型比 + 每波平均敌人数比
- 资源难度 $d_{re}$：综合初始金币、金币掉落量、塔出售回收比

设计动机：提供量化的难度建模方法，使研究者能对比不同关卡的评估意义。

#### 4. 评估指标设计

- **得分（Score）**：每个到达玩家基地的敌人扣 -1.0 分，范围 [-20, 0]
- **有效动作率**：有效动作数 / 总动作数，范围 [0, 1]

两个指标分别评估 LLM 的**能力**和**可靠性**（幻觉程度），独立衡量"正确性"和"有效性"。

### 损失函数 / 训练策略

对 RL 算法（Ape-X DQN 和 PPO），使用稀疏奖励信号（敌人到达基地时 -1.0），每 16 游戏步执行一次动作（约 187 APM）。训练使用 1 亿环境步数。

对 LLM 评估，采用零样本提示策略，所有模型使用相同提示，分语言模态和视觉-语言模态两种设置。

## 实验关键数据

### 主实验

**LLM 得分表现（归一化至人类专家基线）**：

| 模型 | Lv1 | Lv2 | Lv3 | Lv4 | Lv5 | 平均 |
|------|-----|-----|-----|-----|-----|------|
| Claude 3.7 Sonnet (L) | 0.62 | 0.51 | 0.40 | 0.24 | 0.15 | **0.38** |
| GPT-4.1 (VL) | 0.63 | 0.56 | 0.44 | **0.32** | 0.15 | **0.42** |
| Claude 3.7 Sonnet (VL) | **0.67** | **0.58** | **0.45** | 0.20 | **0.16** | 0.41 |
| Gemini-2.5-Pro (L) | 0.52 | 0.42 | 0.31 | 0.11 | 0.01 | 0.27 |
| Qwen 2.5-VL 72B (L) | 0.47 | 0.36 | 0.21 | 0.00 | 0.00 | 0.21 |
| Llama 3.2 90B (L) | 0.42 | 0.32 | 0.19 | 0.12 | 0.00 | 0.21 |
| Qwen 2.5-VL 7B (L) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

**LLM 有效动作率表现（归一化至人类专家基线）**：

| 模型 | Lv1 | Lv2 | Lv3 | Lv4 | Lv5 | 平均 |
|------|-----|-----|-----|-----|-----|------|
| GPT-4.1 (L) | **0.92** | 0.89 | 0.88 | 0.84 | 0.75 | 0.86 |
| Gemini-2.5-Pro (L) | 0.91 | **0.90** | **0.89** | 0.83 | **0.82** | **0.87** |
| Claude 3.7 Sonnet (VL) | 0.85 | **0.85** | **0.83** | **0.80** | **0.79** | **0.82** |
| Qwen 2.5-VL 7B (L) | 0.11 | 0.05 | 0.03 | 0.01 | 0.01 | 0.04 |
| Random 基线 | 0.25 | 0.25 | 0.24 | 0.24 | 0.22 | 0.24 |

### 消融实验 / RL 基准

**RL 算法评估结果**：

| 配置 | Lv1-5 表现 | 说明 |
|------|-----------|------|
| Ape-X DQN (像素) | 部分解决简单关卡 | 1 亿步后仍显著低于人类专家 |
| PPO (像素) | 部分解决简单关卡 | 表明 TowerMind 是一个具有挑战性的 RL 环境 |
| PPO (结构化状态) | 略优于像素输入 | 结构化观测提供更有效的信息 |

### 关键发现

1. **LLM 与人类专家差距巨大**：最佳 LLM（Claude 3.7 Sonnet VL）仅达人类专家约 42% 的得分，最难关卡差距超过 84%
2. **视觉输入改善性能**：除 Llama 3.2 外，所有模型在视觉-语言模态下均优于纯语言模态
3. **开源 LLM 幻觉严重**：Qwen 2.5-VL 7B 和 Llama 3.2 11B 的有效动作率甚至低于随机基线
4. **难度增加加剧幻觉**：更难关卡的游戏元素更多，导致提示更长，挑战模型生成的稳定性
5. **三大行为缺陷**：
    - **规划验证不足**：LLM 会在无法攻击敌人的误导性塔位上建塔，尽管有足够信息可推理出这些位置无效
    - **缺乏多终态思维**：人类专家会同时完成多个目标（如收集金币同时攻击敌人），LLM 从未展现此行为
    - **动作空间利用不充分**：忽视升级塔（金币充足时）、派骑士到空白区域、在无敌人时使用英雄技能

## 亮点与洞察

1. **轻量级设计的实用价值**：0.15GB vs 30GB 的巨大差异使大规模并行评估成为可能
2. **幻觉评估的巧妙设计**：将无效动作率作为幻觉的直接度量，无需额外标注
3. **误导性塔位的实验设计**：通过放置远离道路的塔位，测试 LLM 是否能进行空间推理来识别无效选项
4. **"正确但无效"的洞察**：LLM 的有效动作率与得分之间的差距，类似于问答任务中"技术正确但实际无用"的问题
5. **关卡编辑器的可扩展性**：使研究者可以创建自定义关卡，支持多样化研究需求和降低数据污染风险

## 局限与展望

1. **塔防 vs 完整 RTS 的代表性**：塔防仅是 RTS 的子类，缺少对抗性（PvP）评估
2. **零样本评估的局限**：未探索 few-shot 或 fine-tuned LLM 在 TowerMind 上的表现
3. **RL 基准不够深入**：仅测试了两种基础 RL 算法，未包含更先进的方法
4. **文本观测的信息量**：JSON 格式的文本观测可能过于冗长，影响 LLM 的处理效率
5. **静态敌人波次**：当前敌人配置是固定的，未来可加入更多随机性和动态调整

## 相关工作与启发

- **SC2LE / TextStarCraft II / LLM-PySC2**：基于星际争霸的重量级 RTS 环境
- **ELF / DeepRTS / Gym-μRTS**：轻量级 RTS 环境但不支持文本
- **AGENTBENCH** (Liu et al., 2023)：LLM 在交互式环境中的综合评估基准
- **ReAct / AutoGPT**：LLM 规划与推理框架
- 启发：LLM 评估应从"静态正确性"转向"交互式有效性"，TowerMind 为此提供了实用平台

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首个面向 LLM 评估的塔防游戏环境，弥补了轻量级 RTS 环境的空白
- 实验充分度: ⭐⭐⭐⭐ — 覆盖了 7 个 LLM + 2 个 RL 算法 + 人类基线，分析深入
- 写作质量: ⭐⭐⭐⭐⭐ — 结构清晰，图表丰富，定量与定性分析结合良好
- 价值: ⭐⭐⭐⭐ — 为 LLM 智能体评估提供了实用且可扩展的新工具

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] DPEPO: Diverse Parallel Exploration Policy Optimization for LLM-based Agents](../../ACL2026/reinforcement_learning/dpepo_diverse_parallel_exploration_policy_optimization_for_llm-based_agents.md)
- [\[ICLR 2026\] Toward a Dynamic Stackelberg Game-Theoretic Framework for Agent-Based Conversational AI Defense Against LLM Jailbreaking](../../ICLR2026/reinforcement_learning/toward_a_dynamic_stackelberg_game-theoretic_framework_for_agent-based_conversat.md)
- [\[ICML 2026\] Game of Thought: Robust Information Seeking with Large Language Models Using Game Theory](../../ICML2026/reinforcement_learning/game_of_thought_robust_information_seeking_with_large_language_models_using_game.md)
- [\[ICLR 2026\] Don't Just Fine-tune the Agent, Tune the Environment](../../ICLR2026/reinforcement_learning/dont_just_fine-tune_the_agent_tune_the_environment.md)
- [\[ICLR 2026\] Learning to Orchestrate Agents in Natural Language with the Conductor](../../ICLR2026/reinforcement_learning/learning_to_orchestrate_agents_in_natural_language_with_the_conductor.md)

</div>

<!-- RELATED:END -->
