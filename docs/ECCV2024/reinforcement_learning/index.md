---
title: >-
  ECCV2024 强化学习论文汇总 · 3篇论文解读
description: >-
  3篇ECCV2024的强化学习方向论文解读，涵盖多模态、强化学习等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ECCV2024"
  - "强化学习"
  - "论文解读"
  - "论文笔记"
  - "多模态"
item_list:
  - u: "adaglimpse_active_visual_exploration_with_arbitrary_glimpse_position_and_scale/"
    t: "AdaGlimpse: Active Visual Exploration with Arbitrary Glimpse Position and Scale"
  - u: "octopus_embodied_vision-language_programmer_from_environmental_feedback/"
    t: "Octopus: Embodied Vision-Language Programmer from Environmental Feedback"
  - u: "visual_grounding_for_object-level_generalization_in_reinforcement_learning/"
    t: "Visual Grounding for Object-Level Generalization in Reinforcement Learning"
item_total: 3
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎮 强化学习

**🎞️ ECCV2024** · **3** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (25)](../../CVPR2026/reinforcement_learning/index.md) · [🔬 ICLR2026 (400)](../../ICLR2026/reinforcement_learning/index.md) · [💬 ACL2026 (46)](../../ACL2026/reinforcement_learning/index.md) · [🧪 ICML2026 (110)](../../ICML2026/reinforcement_learning/index.md) · [🤖 AAAI2026 (58)](../../AAAI2026/reinforcement_learning/index.md) · [🧠 NeurIPS2025 (143)](../../NeurIPS2025/reinforcement_learning/index.md)

**[AdaGlimpse: Active Visual Exploration with Arbitrary Glimpse Position and Scale](adaglimpse_active_visual_exploration_with_arbitrary_glimpse_position_and_scale.md)**

:   提出AdaGlimpse，利用Soft Actor-Critic强化学习从连续动作空间中选择任意位置和尺度的glimpse，结合弹性位置编码的ViT编码器实现多任务（重建/分类/分割）的主动视觉探索，以仅6%像素超越了使用18%像素的SOTA方法。

**[Octopus: Embodied Vision-Language Programmer from Environmental Feedback](octopus_embodied_vision-language_programmer_from_environmental_feedback.md)**

:   提出 Octopus，一个具身视觉-语言编程模型，通过生成可执行代码来连接高层规划与底层操控，并引入 Reinforcement Learning with Environmental Feedback (RLEF) 训练方案来提升决策质量。

**[Visual Grounding for Object-Level Generalization in Reinforcement Learning](visual_grounding_for_object-level_generalization_in_reinforcement_learning.md)**

:   利用视觉语言模型 (MineCLIP) 的 visual grounding 能力生成目标物体的 confidence map，通过奖励设计和任务表征两条路径将 VLM 知识迁移到强化学习中，实现对未见物体和指令的零样本泛化。
