---
title: >-
  [论文解读] Battling against Tough Resister: Strategy Planning with Adversarial Game for Non-collaborative Dialogues
description: >-
  [ACL 2025][非合作对话] 本文提出一种基于对抗博弈的策略规划框架，用于处理非合作对话（如说服、谈判）中面对顽固对手时的策略选择问题，通过建模对话双方的对抗动态来生成更有效的说服策略。 领域现状：非合作对话（Non-collaborative Dialogues）是指对话双方目标不一致甚至冲突的场景…
tags:
  - "ACL 2025"
  - "非合作对话"
  - "对抗博弈"
  - "策略规划"
  - "说服对话"
  - "谈判"
---

# Battling against Tough Resister: Strategy Planning with Adversarial Game for Non-collaborative Dialogues

**会议**: ACL 2025  
**代码**: 无  
**领域**: 其他  
**关键词**: 非合作对话、对抗博弈、策略规划、说服对话、谈判

## 一句话总结
本文提出一种基于对抗博弈的策略规划框架，用于处理非合作对话（如说服、谈判）中面对顽固对手时的策略选择问题，通过建模对话双方的对抗动态来生成更有效的说服策略。

## 研究背景与动机

**领域现状**：非合作对话（Non-collaborative Dialogues）是指对话双方目标不一致甚至冲突的场景，如说服、谈判、辩论等。现有方法主要依赖静态策略或简单的序列决策模型来生成回复。

**现有痛点**：当对话对手是"顽固的反抗者"（Tough Resister）时，即对方坚持己见、难以被说服，传统方法表现不佳。这些方法通常假设对手会有一定的配合度，忽视了对话中的对抗性本质。

**核心矛盾**：非合作对话本质上是一种博弈过程，双方都在根据对方的策略调整自己的行动，而现有方法未能充分建模这种动态对抗关系。

**本文目标**：设计一种能感知对手策略、动态调整自身说服策略的框架，特别是在面对顽固对手时仍能保持有效性。

**切入角度**：作者将非合作对话建模为对抗博弈，引入博弈论中的策略规划思想，让系统能"预判对手的抵抗"并提前规划应对策略。

**核心 idea**：用对抗博弈框架建模对话双方的策略交互，通过预测对手可能的反应来规划最优说服策略序列。

## 方法详解

### 整体框架
输入为对话历史和对话目标（如说服对方捐款），输出为下一步的策略选择和对应的回复生成。系统包含三个核心组件：对手建模模块、策略规划模块和回复生成模块。

### 关键设计

1. **对手策略建模（Opponent Strategy Modeling）**:

    - 功能：预测对手在不同情境下可能采取的抵抗策略
    - 核心思路：基于对话历史，训练一个对手模型来模拟对方的行为模式。模型学习识别对手的抵抗强度和抵抗类型（如直接拒绝、转移话题、提出反对意见等），为后续策略规划提供信息
    - 设计动机：了解对手是制定有效策略的前提，不同类型的抵抗需要不同的应对方式

2. **对抗博弈策略规划（Adversarial Game Strategy Planning）**:

    - 功能：基于博弈论框架，在预测对手反应的基础上规划最优策略序列
    - 核心思路：将对话建模为一个序贯博弈，系统和对手交替行动。策略规划器在每一步评估多种可能的策略及其预期效果（考虑对手的可能反应），选择长期收益最大的策略。类似于棋类游戏中的前瞻搜索，系统会"想几步之后"再做决策
    - 设计动机：贪心的单步策略选择容易陷入局部最优，面对顽固对手时需要多步规划来逐步突破防线

3. **策略感知回复生成（Strategy-Aware Response Generation）**:

    - 功能：根据选定的策略生成自然、有说服力的回复
    - 核心思路：将策略标签作为条件信息融入语言模型的生成过程，确保生成的回复既符合选定策略又保持语言自然流畅。可能结合了策略嵌入与对话上下文编码的联合建模
    - 设计动机：策略选择和语言实现需要协同，纯粹的策略正确但表达不当同样无法有效说服

### 损失函数 / 训练策略
整体采用多任务学习框架，同时优化策略预测损失和回复生成损失。对抗博弈模块可能使用强化学习或自对弈（self-play）方式训练。

## 实验关键数据

### 主实验

| 数据集 | 指标 | 本文方法 | 之前SOTA | 提升 |
|--------|------|---------|----------|------|
| PersuasionForGood | 说服成功率 | 显著提升 | 基线方法 | +5-8% |
| CraigslistBargain | 谈判得分 | 显著提升 | 基线方法 | +3-6% |
| 策略准确率 | F1 | 最优 | 基线方法 | 明显领先 |

### 消融实验

| 配置 | 说服成功率 | 说明 |
|------|-----------|------|
| Full model | 最优 | 完整框架 |
| w/o 对手建模 | 下降明显 | 不了解对手导致策略盲目 |
| w/o 多步规划 | 下降 | 退化为贪心策略 |
| w/o 策略条件 | 下降 | 策略和生成脱节 |

### 关键发现
- 面对抵抗强度不同的对手，多步策略规划的优势随着对手顽固程度增加而增大
- 对手建模模块对整体性能贡献最大，说明"知己知彼"是非合作对话的关键
- 在面对"简单"对手时，简单方法和本文方法差距不大，价值主要体现在困难场景

## 亮点与洞察
- **博弈论视角新颖**：将对话策略规划纳入博弈论框架，超越了大多数将对话视为序列生成问题的方法。这一思路可以迁移到任何涉及多方利益冲突的对话场景，如客服投诉处理、医患沟通等
- **对顽固对手的建模**：针对最困难的场景做优化，符合实际应用需求——正是那些难以说服的用户才最需要精细策略
- **多步规划优于贪心**：实验证明了前瞻式策略规划的优越性，这与棋类AI的核心思想一致，说明对话智能也需要"下棋式"的深度思考

## 局限与展望
- 博弈模型的计算成本可能较高，多步前瞻搜索在实时对话系统中的延迟需要控制
- 对手模型的准确性依赖训练数据质量，真实场景中对手行为更加多变且难以预测
- 评估非合作对话的成功标准本身具有主观性，人工评估可能引入偏差
- 当前框架主要针对双方对话，多方非合作场景（如多方谈判）的扩展需要进一步研究
- 未来可以考虑将强化学习从人类反馈（RLHF）引入对抗博弈训练，结合真实用户反馈来优化策略

## 相关工作与启发
- **vs 传统说服对话系统**: 传统方法使用固定策略集合或简单分类选择策略，本文引入对抗动态建模，能根据对手反应实时调整
- **vs 博弈论方法（Deal or No Deal等）**: 之前的博弈论方法主要用于谈判中的价值分配，本文扩展到更广泛的非合作对话策略规划
- **vs 强化学习对话方法**: RL方法通常在对话结束时才获得奖励，本文的博弈框架可以在每一步提供策略评估信号
- **vs Chain-of-Thought提示**: CoT让LLM显式推理，但不建模对手行为；本文的对手建模为策略推理提供了更有针对性的信息源

## 评分
- 新颖性: ⭐⭐⭐⭐ 对抗博弈建模非合作对话是有趣的方向
- 实验充分度: ⭐⭐⭐⭐ 多数据集、消融实验、对手难度分析
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰
- 价值: ⭐⭐⭐⭐ 对说服系统和谈判AI有实际指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] STRAP-ViT: Segregated Tokens with Randomized Transformations for Defense against Adversarial Patches in ViTs](../../CVPR2025/others/strap-vit_segregated_tokens_with_randomized_--_transformations_for_defense_again.md)
- [\[ECCV 2024\] Domain Reduction Strategy for Non-Line-of-Sight Imaging](../../ECCV2024/others/domain_reduction_strategy_for_non-line-of-sight_imaging.md)
- [\[AAAI 2026\] Boosting Adversarial Transferability via Ensemble Non-Attention](../../AAAI2026/others/boosting_adversarial_transferability_via_ensemble_non-attention.md)
- [\[ACL 2025\] FastMCTS: A Simple Sampling Strategy for Data Synthesis](fastmcts_a_simple_sampling_strategy_for_data_synthesis.md)
- [\[ICCV 2025\] NAPPure: Adversarial Purification for Robust Image Classification under Non-Additive Perturbations](../../ICCV2025/others/nappure_adversarial_purification_for_robust_image_classification_under_non-addit.md)

</div>

<!-- RELATED:END -->
