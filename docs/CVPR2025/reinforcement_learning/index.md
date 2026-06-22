---
title: >-
  CVPR2025 强化学习论文汇总 · 5篇论文解读
description: >-
  5篇CVPR2025的强化学习方向论文解读，涵盖强化学习、压缩/编码等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "CVPR2025"
  - "强化学习"
  - "论文解读"
  - "论文笔记"
  - "压缩/编码"
item_list:
  - u: "calf_communication_aware_distributed_rl/"
    t: "CALF: Communication-Aware Learning Framework for Distributed Reinforcement Learning"
  - u: "gazing_at_rewards_eye_movements_as_a_lens_into_human_and_ai_decision-making_in_h/"
    t: "Gazing at Rewards: Eye Movements as a Lens into Human and AI Decision-Making in Hybrid Visual Foraging"
  - u: "grove_a_generalized_reward_for_learning_open-vocabulary_physical_skill/"
    t: "GROVE: A Generalized Reward for Learning Open-Vocabulary Physical Skill"
  - u: "skillmimic_learning_basketball_interaction_skills_from_demonstrations/"
    t: "SkillMimic: Learning Basketball Interaction Skills from Demonstrations"
  - u: "thinking_in_streaming_video/"
    t: "ThinkStream: Thinking in Streaming Video"
item_total: 5
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎮 强化学习

**📷 CVPR2025** · **5** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (25)](../../CVPR2026/reinforcement_learning/index.md) · [🔬 ICLR2026 (400)](../../ICLR2026/reinforcement_learning/index.md) · [💬 ACL2026 (46)](../../ACL2026/reinforcement_learning/index.md) · [🧪 ICML2026 (110)](../../ICML2026/reinforcement_learning/index.md) · [🤖 AAAI2026 (58)](../../AAAI2026/reinforcement_learning/index.md) · [🧠 NeurIPS2025 (143)](../../NeurIPS2025/reinforcement_learning/index.md)

**[CALF: Communication-Aware Learning Framework for Distributed Reinforcement Learning](calf_communication_aware_distributed_rl.md)**

:   本文提出 CALF 框架，通过在 RL 训练中注入可配置的网络延迟、抖动和丢包模型，使策略在部署到真实分布式边缘设备时性能退化降低约 3-4 倍，揭示网络条件是 sim-to-real 转移中被忽视的重要维度。

**[Gazing at Rewards: Eye Movements as a Lens into Human and AI Decision-Making in Hybrid Visual Foraging](gazing_at_rewards_eye_movements_as_a_lens_into_human_and_ai_decision-making_in_h.md)**

:   提出Visual Forager（VF）模型，通过目标特征调制、目标价值调制和ViT-based Actor-Critic决策网络模拟人类混合视觉搜索任务中的眼动策略，在归一化得分上达到72.6%（人类87.4%），扫视大小仅差0.01°（4.06° vs 人类4.05°），首次揭示目标价值和出现率如何联合影响人类搜索决策。

**[GROVE: A Generalized Reward for Learning Open-Vocabulary Physical Skill](grove_a_generalized_reward_for_learning_open-vocabulary_physical_skill.md)**

:   本文提出GROVE框架，利用LLM生成物理约束+VLM评估动作语义的互补方式构建广义奖励函数，并通过Pose2CLIP轻量映射器跳过渲染直接将姿态投影到语义空间，实现了开放词汇物理技能学习，比现有方法训练速度快8.4倍同时动作自然度提升22.2%。

**[SkillMimic: Learning Basketball Interaction Skills from Demonstrations](skillmimic_learning_basketball_interaction_skills_from_demonstrations.md)**

:   提出 SkillMimic，一个纯数据驱动的框架，通过统一的 HOI 模仿奖励（特别是创新的接触图奖励）从动捕数据中学习多样的篮球交互技能，并通过高层控制器组合技能实现连续得分等复杂长程任务。

**[ThinkStream: Thinking in Streaming Video](thinking_in_streaming_video.md)**

:   提出 ThinkStream，采用 Watch-Think-Speak 范式实现流式视频的实时连续推理，通过 RCSM（推理压缩流式记忆）将推理 trace 作为紧凑语义锚点替代旧视觉 token，配合 Streaming RLVR 训练策略，在保持低延迟/低内存的同时超越现有在线视频模型。
