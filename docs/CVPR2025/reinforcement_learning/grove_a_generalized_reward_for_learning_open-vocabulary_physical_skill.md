---
title: >-
  [论文解读] GROVE: A Generalized Reward for Learning Open-Vocabulary Physical Skill
description: >-
  [CVPR 2025][强化学习][开放词汇物理技能] 本文提出GROVE框架，利用LLM生成物理约束+VLM评估动作语义的互补方式构建广义奖励函数，并通过Pose2CLIP轻量映射器跳过渲染直接将姿态投影到语义空间，实现了开放词汇物理技能学习，比现有方法训练速度快8.4倍同时动作自然度提升22.2%。
tags:
  - "CVPR 2025"
  - "强化学习"
  - "开放词汇物理技能"
  - "广义奖励函数"
  - "LLM约束生成"
  - "VLM动作评估"
---

# GROVE: A Generalized Reward for Learning Open-Vocabulary Physical Skill

**会议**: CVPR 2025  
**arXiv**: [2504.04191](https://arxiv.org/abs/2504.04191)  
**代码**: 无  
**领域**: 强化学习  
**关键词**: 开放词汇物理技能, 广义奖励函数, LLM约束生成, VLM动作评估, 强化学习

## 一句话总结
本文提出GROVE框架，利用LLM生成物理约束+VLM评估动作语义的互补方式构建广义奖励函数，并通过Pose2CLIP轻量映射器跳过渲染直接将姿态投影到语义空间，实现了开放词汇物理技能学习，比现有方法训练速度快8.4倍同时动作自然度提升22.2%。

## 研究背景与动机

**领域现状**：在仿真环境中学习多样化的物理技能（如跑步、跳舞、搬运等）是具身AI的核心挑战。当前强化学习方法主要依赖两种奖励设计范式：人工设计的奖励函数和基于示范的奖励。

**现有痛点**：人工设计的奖励函数缺乏跨任务的可扩展性——每个新任务都需要专家手动调参。基于示范的方法（如动作捕捉数据驱动的模仿学习）则难以泛化到训练分布之外的新任务。这两种范式都无法实现"开放词汇"——即用自然语言描述任意物理技能就能让智能体学会。

**核心矛盾**：开放词汇意味着无法预先为所有可能的任务设计奖励或收集示范，需要一种能从任务描述自动生成奖励信号的通用机制。但物理技能的学习既需要满足精确的物理约束（如"抬腿到30度"），也需要保证动作的整体自然度和语义正确性。

**本文目标**：设计一个不需要人工工程或任务特定示范的广义奖励框架，使仿真智能体能够通过自然语言描述学习开放词汇的物理技能。

**切入角度**：LLM擅长将任务描述分解为精确的物理约束（但不擅长评估视觉效果），VLM擅长评估动作的整体自然度和语义（但不擅长生成精确约束）——两者互补。

**核心 idea**：构建LLM-VLM协同的奖励系统：LLM生成物理约束，VLM评估动作语义，两者通过迭代反馈自我改进，并用Pose2CLIP跳过昂贵的渲染实现高效训练。

## 方法详解

### 整体框架
GROVE的流程分为三个阶段：(1) LLM约束生成阶段——给定任务的自然语言描述，LLM输出一组物理约束（关节角度、位置、速度等的目标值和权重）；(2) VLM迭代优化阶段——用VLM评估当前约束生成的动作视频，反馈给LLM进行约束修正；(3) 高效训练阶段——用Pose2CLIP替代渲染，将智能体姿态直接映射到CLIP语义空间计算奖励。

### 关键设计

1. **LLM物理约束生成**:

    - 功能：将自然语言任务描述转化为可执行的物理约束集合
    - 核心思路：给LLM（如GPT-4）提供仿真环境的API描述（可用关节列表、物理量范围等），让它根据任务描述生成约束代码。每个约束包含目标物理量、目标值、容忍范围和权重。例如"做俯卧撑"会被分解为：手掌接触地面约束、肘部角度周期性变化约束、躯干保持水平约束等
    - 设计动机：LLM拥有广泛的世界知识，能理解"俯卧撑"等高层描述并转化为物理约束，但它缺乏视觉判断力——不知道生成的约束是否真的导致自然的动作

2. **VLM反馈迭代优化**:

    - 功能：通过视觉反馈循环持续改进LLM生成的约束质量
    - 核心思路：用当前约束训练RL智能体，渲染其动作视频，交给VLM（如GPT-4V）评估动作自然度和任务完成度。VLM的反馈（如"手臂伸展不够"、"下蹲时膝盖角度过大"）被回传给LLM用于修正约束。这个过程迭代多轮，形成自我改进循环
    - 设计动机：单靠LLM生成的约束通常不完美——可能遗漏关键约束或权重设置不当。VLM提供了"视觉常识"来弥补LLM的"想象力不足"

3. **Pose2CLIP轻量映射器**:

    - 功能：将智能体的关节姿态直接映射到CLIP语义空间，避免昂贵的渲染步骤
    - 核心思路：训练一个小型MLP网络，输入为智能体的关节角度向量，输出为CLIP图像空间的特征向量。训练数据通过对齐姿态-渲染图像-CLIP特征的三元组获得。推理时直接用Pose2CLIP输出的特征与任务描述的CLIP文本特征计算相似度作为语义奖励分量
    - 设计动机：在RL训练中，每步都渲染图像然后跑CLIP的计算开销极大。Pose2CLIP将这个过程简化为一次MLP前向传播，训练速度提升8.4倍

### 损失函数 / 训练策略
总奖励函数为LLM物理约束奖励分量和Pose2CLIP语义奖励分量的加权组合。RL算法使用PPO。Pose2CLIP的训练损失为预测特征与真实CLIP特征的余弦相似度损失。

## 实验关键数据

### 主实验
在多种具身形态（人形、四足等）上评估开放词汇物理技能学习：

| 方法 | 动作自然度 | 任务完成度 | 训练速度 | 说明 |
|------|----------|----------|---------|------|
| GROVE (本文) | **最高 (+22.2%)** | **最高 (+25.7%)** | **最快 (8.4x)** | LLM+VLM+Pose2CLIP |
| Text2Reward | 中等 | 中等 | 慢 | 仅LLM生成奖励 |
| Eureka | 中等 | 较高 | 慢 | LLM+进化搜索 |
| UniHSI | 较低 | 较低 | 中等 | 基于示范 |

### 消融实验

| 配置 | 动作自然度 | 任务完成度 | 说明 |
|------|----------|----------|------|
| Full GROVE | 最高 | 最高 | 完整模型 |
| w/o VLM反馈 | 下降显著 | 下降明显 | 仅LLM单次生成约束，质量不稳定 |
| w/o Pose2CLIP | 相当 | 相当 | 用渲染+CLIP替代，精度相似但速度慢8.4x |
| w/o LLM约束 | 最低 | 最低 | 仅用VLM评分做奖励，缺乏精确物理约束 |

### 关键发现
- LLM和VLM的协同是关键：单独使用任何一个都显著差于两者结合。LLM提供精度，VLM提供语义常识
- Pose2CLIP在几乎不损失精度的前提下实现了8.4倍加速，是一个理想的渲染替代方案
- VLM反馈迭代通常在2-3轮后收敛，更多轮次的边际收益递减
- GROVE在不同具身形态间表现一致，证明了框架的泛化性

## 亮点与洞察
- **LLM-VLM互补架构**的设计思路非常巧妙：LLM擅长结构化推理但缺少视觉判断，VLM有视觉常识但难以输出精确约束，两者完美互补。这种模式可以迁移到其他"需要精确+自然"的任务中
- **Pose2CLIP**的"跳过渲染"思路值得关注：在RL训练中渲染是常见瓶颈，用轻量映射器直接将低维状态投影到语义空间是一个通用的加速策略
- **自我改进循环**（VLM反馈→LLM修正→重新训练→VLM再评估）是一个很有潜力的范式，类似于"AI设计AI奖励"的元学习思路

## 局限与展望
- 当前仅在仿真环境中验证，sim-to-real转移效果未探索
- VLM反馈迭代需要实际训练RL智能体并渲染视频，初始化成本较高
- 对于高度物理精确的任务（如体操），LLM生成的约束精度可能不足
- Pose2CLIP的训练数据收集需要渲染，不同环境/形态需要重新训练映射器
- 未探索更复杂的多智能体协作物理技能场景

## 相关工作与启发
- **vs Text2Reward**: Text2Reward仅用LLM单次生成奖励代码，缺少视觉反馈闭环，GROVE通过VLM迭代优化显著提升了约束质量
- **vs Eureka**: Eureka使用进化搜索优化LLM生成的奖励，但搜索过程缓慢且不利用视觉信息。GROVE的VLM反馈机制更高效且信息更丰富
- **vs UniHSI**: UniHSI依赖人体交互示范数据，GROVE完全不需要任何示范，扩展性更强

## 评分
- 新颖性: ⭐⭐⭐⭐ LLM+VLM协同奖励设计和Pose2CLIP加速都有新意
- 实验充分度: ⭐⭐⭐⭐ 多种具身形态和任务的验证，消融全面
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，动机充分
- 价值: ⭐⭐⭐⭐ 为开放词汇具身技能学习提供了实用框架

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Self-Improving Skill Learning for Robust Skill-based Meta-Reinforcement Learning](../../ICLR2026/reinforcement_learning/self-improving_skill_learning_for_robust_skill-based_meta-reinforcement_learning.md)
- [\[ICLR 2026\] Single Index Bandits: Generalized Linear Contextual Bandits with Unknown Reward Functions](../../ICLR2026/reinforcement_learning/single_index_bandits_generalized_linear_contextual_bandits_with_unknown_reward_f.md)
- [\[NeurIPS 2025\] Certifying Stability of Reinforcement Learning Policies using Generalized Lyapunov Functions](../../NeurIPS2025/reinforcement_learning/certifying_stability_of_reinforcement_learning_policies_using_generalized_lyapun.md)
- [\[ICLR 2026\] From Verifiable Dot to Reward Chain: Harnessing Verifiable Reference-based Rewards for RL of Open-ended Generation](../../ICLR2026/reinforcement_learning/from_verifiable_dot_to_reward_chain_harnessing_verifiable_reference-based_reward.md)
- [\[NeurIPS 2025\] DeepDiver: Adaptive Search Intensity Scaling via Open-Web Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/deepdiver_adaptive_search_intensity_scaling_via_open-web_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
