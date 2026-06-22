---
title: >-
  ACL2025 强化学习论文汇总 · 8篇论文解读
description: >-
  8篇ACL2025的强化学习方向论文解读，涵盖强化学习、LLM、模型压缩、对话系统、Agent等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ACL2025"
  - "强化学习"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "模型压缩"
  - "对话系统"
  - "Agent"
item_list:
  - u: "align-slm_textless_spoken_language_models_with_reinforcement_learning_from_ai_fe/"
    t: "Align-SLM: Textless Spoken Language Models with Reinforcement Learning from AI Feedback"
  - u: "bypass_back-propagation_optimization-based_structural_pruning_for_large_language/"
    t: "Bypass Back-propagation: Optimization-based Structural Pruning for Large Language Models via Policy Gradient"
  - u: "eierl_dialogue_policy/"
    t: "An Efficient Task-Oriented Dialogue Policy: Evolutionary Reinforcement Learning Injected by Elite Individuals"
  - u: "learning_to_generate_structured_output_with_schema_reinforcement_learning/"
    t: "Learning to Generate Structured Output with Schema Reinforcement Learning"
  - u: "llm-enhanced_self-evolving_reinforcement_learning_for_multi-step_e-commerce_paym/"
    t: "LLM-Enhanced Self-Evolving Reinforcement Learning for Multi-Step E-Commerce Payment Fraud Risk Detection"
  - u: "maporl_multi-agent_post-co-training_for_collaborative_large_language_models_with/"
    t: "MAPoRL: Multi-Agent Post-Co-Training for Collaborative Large Language Models with Reinforcement Learning"
  - u: "prompt-based_personality_profiling_reinforcement_learning_for_relevance_filterin/"
    t: "Prompt-based Personality Profiling: Reinforcement Learning for Relevance Filtering"
  - u: "treerl_tree_search_rl/"
    t: "TreeRL: LLM Reinforcement Learning with On-Policy Tree Search"
item_total: 8
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎮 强化学习

**💬 ACL2025** · **8** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (25)](../../CVPR2026/reinforcement_learning/index.md) · [🔬 ICLR2026 (400)](../../ICLR2026/reinforcement_learning/index.md) · [💬 ACL2026 (46)](../../ACL2026/reinforcement_learning/index.md) · [🧪 ICML2026 (110)](../../ICML2026/reinforcement_learning/index.md) · [🤖 AAAI2026 (58)](../../AAAI2026/reinforcement_learning/index.md) · [🧠 NeurIPS2025 (143)](../../NeurIPS2025/reinforcement_learning/index.md)

🔥 **高频主题：** 强化学习 ×7 · LLM ×4

**[Align-SLM: Textless Spoken Language Models with Reinforcement Learning from AI Feedback](align-slm_textless_spoken_language_models_with_reinforcement_learning_from_ai_fe.md)**

:   本文提出 Align-SLM 框架，首次将偏好优化（DPO + RLAIF）应用于纯语音语言模型（无文本注入），通过 LLM 自动评估生成的语音续写质量构建偏好数据，结合课程学习迭代提升 SLM 的语义理解能力，在 ZeroSpeech 和 StoryCloze 等基准上达到 SLM 的 SOTA。

**[Bypass Back-propagation: Optimization-based Structural Pruning for Large Language Models via Policy Gradient](bypass_back-propagation_optimization-based_structural_pruning_for_large_language.md)**

:   本文提出一种基于策略梯度的LLM结构化剪枝方法，通过在概率空间中学习伯努利剪枝掩码来直接优化剪枝模型的损失函数，全程无需对LLM本身进行反向传播，仅需前向推理即可完成剪枝优化。

**[An Efficient Task-Oriented Dialogue Policy: Evolutionary Reinforcement Learning Injected by Elite Individuals](eierl_dialogue_policy.md)**

:   首次将进化强化学习（ERL）应用于任务导向对话策略任务，提出 EIERL 方法结合 EA 的全局探索与 DRL 的局部优化，并通过精英个体注入（EII）机制解决 EA 在自然语言大搜索空间中进化缓慢的问题，在 4 个数据集上实现了更高效的探索-利用平衡。

**[Learning to Generate Structured Output with Schema Reinforcement Learning](learning_to_generate_structured_output_with_schema_reinforcement_learning.md)**

:   提出 SchemaBench 基准（约4万条 JSON schema）和 Schema Reinforcement Learning (SRL) 训练框架，通过细粒度 schema 验证器提供密集奖励信号，结合 Thoughts of Structure (ToS) 推理机制，将 LLM 的复杂 JSON 生成准确率提升高达16%，同时不损害通用推理能力。

**[LLM-Enhanced Self-Evolving Reinforcement Learning for Multi-Step E-Commerce Payment Fraud Risk Detection](llm-enhanced_self-evolving_reinforcement_learning_for_multi-step_e-commerce_paym.md)**

:   将电商支付欺诈检测建模为多步 MDP，用 LLM（Mixtral/LLaMA/Gemma）通过进化算法自动生成和优化 RL 奖励函数，在 eBay 真实交易数据上比人工设计奖励函数和传统 SL 基线显著提升 dollar-wise precision。

**[MAPoRL: Multi-Agent Post-Co-Training for Collaborative Large Language Models with Reinforcement Learning](maporl_multi-agent_post-co-training_for_collaborative_large_language_models_with.md)**

:   提出 MAPoRL——一种基于多智能体强化学习的后训练范式，通过让多个 LLM 在辩论框架中共同训练（co-training），配合验证器评分和协作激励机制，显著提升多 LLM 协作的效果，并展现出跨任务的泛化能力。

**[Prompt-based Personality Profiling: Reinforcement Learning for Relevance Filtering](prompt-based_personality_profiling_reinforcement_learning_for_relevance_filterin.md)**

:   提出RL-Profiler方法，用强化学习训练一个帖子相关性过滤器（SelNet），从用户Profile的大量帖子中筛选出与人格特征相关的少量帖子，再交给LLM零样本预测人格，在大幅减少上下文长度的同时保持接近使用全部帖子的预测效果。

**[TreeRL: LLM Reinforcement Learning with On-Policy Tree Search](treerl_tree_search_rl.md)**

:   提出 TreeRL，将基于熵引导的树搜索（EPTree）直接集成到 LLM 的 on-policy 强化学习训练中，通过在高不确定性 token 处分叉来扩展推理路径多样性，并利用树结构提供的全局+局部优势作为过程监督信号，在数学和代码推理任务上超过传统的多链采样 RL。
