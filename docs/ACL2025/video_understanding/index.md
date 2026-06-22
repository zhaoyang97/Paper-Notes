---
title: >-
  ACL2025 视频理解论文汇总 · 8篇论文解读
description: >-
  8篇ACL2025的视频理解方向论文解读，涵盖目标跟踪、多模态、问答、对话系统、推理、对抗鲁棒等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ACL2025"
  - "视频理解"
  - "论文解读"
  - "论文笔记"
  - "目标跟踪"
  - "多模态"
  - "问答"
  - "对话系统"
  - "推理"
  - "对抗鲁棒"
item_list:
  - u: "a_thousand_words_paint_a_picture_multimodal_goal_tracking_for_grounded_social_in/"
    t: "A Thousand Words Paint a Picture: Multimodal Goal Tracking for Grounded Social Intelligence"
  - u: "addressing_blind_guessing_calibration_of_selection_bias_in_multiple-choice_quest/"
    t: "Addressing Blind Guessing: Calibration of Selection Bias in Multiple-Choice Question Answering by Video Language Models"
  - u: "attention-seeker_dynamic_self-attention_scoring_for_unsupervised_key-frame_extra/"
    t: "Attention-Seeker: Dynamic Self-Attention Scoring for Unsupervised Key-Frame Extraction"
  - u: "from_teacher_to_student_tracking_memorization_through_model_distillation/"
    t: "From Teacher to Student: Tracking Memorization Through Model Distillation"
  - u: "generative_frame_sampler_for_long_video_understanding/"
    t: "Generative Frame Sampler for Long Video Understanding"
  - u: "improving_dialogue_state_tracking_through_combinatorial_search_for_in-context_ex/"
    t: "Improving Dialogue State Tracking through Combinatorial Search for In-Context Examples"
  - u: "raven_robust_advertisement_video_violation_temporal_grounding_via_reinforcement_/"
    t: "RAVEN: Robust Advertisement Video Violation Temporal Grounding via Reinforcement Reasoning"
  - u: "sparse-to-dense_a_free_lunch_for_lossless_acceleration_of_video_understanding_in/"
    t: "Sparse-to-Dense: A Free Lunch for Lossless Acceleration of Video Understanding in LLMs"
item_total: 8
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📹 视频理解

**💬 ACL2025** · **8** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (187)](../../CVPR2026/video_understanding/index.md) · [🔬 ICLR2026 (48)](../../ICLR2026/video_understanding/index.md) · [🧪 ICML2026 (17)](../../ICML2026/video_understanding/index.md) · [🤖 AAAI2026 (27)](../../AAAI2026/video_understanding/index.md) · [🧠 NeurIPS2025 (39)](../../NeurIPS2025/video_understanding/index.md) · [📹 ICCV2025 (56)](../../ICCV2025/video_understanding/index.md)

🔥 **高频主题：** 目标跟踪 ×3

**[A Thousand Words Paint a Picture: Multimodal Goal Tracking for Grounded Social Intelligence](a_thousand_words_paint_a_picture_multimodal_goal_tracking_for_grounded_social_in.md)**

:   本文提出了一个多模态目标追踪框架，通过结合视觉与语言线索来推理社交场景中参与者的隐含目标，从而提升模型对社交情境的理解能力（即"落地的社会智能"）。

**[Addressing Blind Guessing: Calibration of Selection Bias in Multiple-Choice Question Answering by Video Language Models](addressing_blind_guessing_calibration_of_selection_bias_in_multiple-choice_quest.md)**

:   首次系统性研究视频语言模型（VLM）在多选题回答中的选项选择偏差问题，通过任务分解分析偏差来源，提出BOLD后处理校准技术，在减少偏差的同时提升模型性能。

**[Attention-Seeker: Dynamic Self-Attention Scoring for Unsupervised Key-Frame Extraction](attention-seeker_dynamic_self-attention_scoring_for_unsupervised_key-frame_extra.md)**

:   本文提出Attention-Seeker方法，通过动态地分析Transformer模型中自注意力层的注意力得分分布，无需任何监督信号即可从视频中提取最具代表性的关键帧，在多个视频摘要基准数据集上超越了现有的无监督方法。

**[From Teacher to Student: Tracking Memorization Through Model Distillation](from_teacher_to_student_tracking_memorization_through_model_distillation.md)**

:   系统研究了知识蒸馏（KD）对大语言模型记忆化行为的影响，发现蒸馏不仅能压缩模型，还能显著降低对训练数据的逐字记忆风险——其中反向 KL 蒸馏（RKLD/MiniLLM）将记忆化比例从 SFT 的 65.4% 降至最低 6.0%。

**[Generative Frame Sampler for Long Video Understanding](generative_frame_sampler_for_long_video_understanding.md)**

:   提出 GenS，一个基于 VideoLLM 的生成式帧采样模块，用自然语言输出question-aware的相关帧时间段和置信度分数，作为即插即用模块在 LongVideoBench/MLVU/HourVideo 上为多种 VideoLLM 带来 2-4 个点的一致提升。

**[Improving Dialogue State Tracking through Combinatorial Search for In-Context Examples](improving_dialogue_state_tracking_through_combinatorial_search_for_in-context_ex.md)**

:   提出 CombiSearch 方法，通过组合式评分为对话状态追踪（DST）选择最优 in-context 示例组合，在仅用 5% 训练数据的情况下超越所有使用 100% 数据的 baseline，理想设置下 JGA 上界比传统方法高 12%。

**[RAVEN: Robust Advertisement Video Violation Temporal Grounding via Reinforcement Reasoning](raven_robust_advertisement_video_violation_temporal_grounding_via_reinforcement_.md)**

:   本文提出RAVEN框架，将课程强化学习与多模态LLM结合，通过分层奖励机制和渐进式训练策略，实现广告视频违规内容的精确时序定位和类别预测，无需显式推理标注数据即可激发涌现推理能力。

**[Sparse-to-Dense: A Free Lunch for Lossless Acceleration of Video Understanding in LLMs](sparse-to-dense_a_free_lunch_for_lossless_acceleration_of_video_understanding_in.md)**

:   基于Video-LLM中注意力分数的稀疏性观察，提出Sparse-to-Dense (StD)解码策略，用top-K稀疏注意力模型作为draft model快速生成候选token，再用全注意力模型并行验证，实现最高1.94倍的无损加速，且无需额外训练或架构修改。
