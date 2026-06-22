---
title: >-
  AAAI2026 预训练论文汇总 · 9篇论文解读
description: >-
  9篇AAAI2026的预训练方向论文解读，涵盖 LLM、对齐/RLHF、少样本学习、压缩/编码等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "AAAI2026"
  - "预训练"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "对齐/RLHF"
  - "少样本学习"
  - "压缩/编码"
item_list:
  - u: "beyond_cosine_similarity_magnitude-aware_clip_for_no-reference_image_quality_ass/"
    t: "Beyond Cosine Similarity: Magnitude-Aware CLIP for No-Reference Image Quality Assessment"
  - u: "elspr_evaluator_llm_training_data_self-purification_on_non-transitive_preference/"
    t: "ELSPR: Evaluator LLM Training Data Self-Purification on Non-Transitive Preferences"
  - u: "granalign_granularity-aware_alignment_framework_for_zero-shot_video_moment_retri/"
    t: "GranAlign: Granularity-Aware Alignment Framework for Zero-Shot Video Moment Retrieval"
  - u: "learning_procedural-aware_video_representations_through_state-grounded_hierarchy/"
    t: "Learning Procedural-aware Video Representations through State-Grounded Hierarchy Unfolding"
  - u: "no-regret_strategy_solving_in_imperfect-information_games_via_pre-trained_embedd/"
    t: "No-Regret Strategy Solving in Imperfect-Information Games via Pre-Trained Embedding"
  - u: "perspective_from_a_broader_context_can_room_style_knowledge_help_visual_floorpla/"
    t: "Perspective from a Broader Context: Can Room Style Knowledge Help Visual Floorplan Localization?"
  - u: "prefixgpt_prefix_adder_optimization_by_a_generative_pre-trained_transformer/"
    t: "PrefixGPT: Prefix Adder Optimization by a Generative Pre-trained Transformer"
  - u: "rectified_noise_a_generative_model_using_positive-incentive_noise/"
    t: "Rectified Noise: A Generative Model Using Positive-incentive Noise"
  - u: "trace_a_generalizable_drift_detector_for_streaming_data-driven_optimization/"
    t: "TRACE: A Generalizable Drift Detector for Streaming Data-Driven Optimization"
item_total: 9
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📚 预训练

**🤖 AAAI2026** · **9** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (5)](../../CVPR2026/llm_pretraining/index.md) · [🔬 ICLR2026 (79)](../../ICLR2026/llm_pretraining/index.md) · [💬 ACL2026 (12)](../../ACL2026/llm_pretraining/index.md) · [🧪 ICML2026 (27)](../../ICML2026/llm_pretraining/index.md) · [🧠 NeurIPS2025 (51)](../../NeurIPS2025/llm_pretraining/index.md) · [📹 ICCV2025 (9)](../../ICCV2025/llm_pretraining/index.md)

**[Beyond Cosine Similarity: Magnitude-Aware CLIP for No-Reference Image Quality Assessment](beyond_cosine_similarity_magnitude-aware_clip_for_no-reference_image_quality_ass.md)**

:   提出 MA-CLIP，发现并利用 CLIP 图像特征的**幅度信息**作为感知质量的互补线索，结合余弦相似度实现无需训练的自适应双线索融合图像质量评估。

**[ELSPR: Evaluator LLM Training Data Self-Purification on Non-Transitive Preferences](elspr_evaluator_llm_training_data_self-purification_on_non-transitive_preference.md)**

:   ELSPR 将 LLM 评估器的成对偏好建模为锦标赛图，通过强连通分量 (SCC) 识别非传递偏好，提出归一化有向图结构熵指标，并基于图重构过滤有问题的训练数据——过滤后的评估器非传递性降低 13.8%、结构熵降低 0.088，且丢弃数据的人类一致性仅 34.4%（vs 保留数据 52.6%）。

**[GranAlign: Granularity-Aware Alignment Framework for Zero-Shot Video Moment Retrieval](granalign_granularity-aware_alignment_framework_for_zero-shot_video_moment_retri.md)**

:   提出一个无需训练的粒度感知对齐框架GranAlign，通过将查询重写为简化版和细化版并分别匹配无关/感知查询的视频描述，解决了零样本视频时刻检索中语义粒度不匹配的核心难题，在QVHighlights上mAP@avg提升3.23%。

**[Learning Procedural-aware Video Representations through State-Grounded Hierarchy Unfolding](learning_procedural-aware_video_representations_through_state-grounded_hierarchy.md)**

:   提出 Task-Step-State（TSS）三层语义框架，在传统的任务-步骤层次中引入"状态"作为视觉锚定层，并设计渐进式预训练策略（Task→Step→State→Step→Task）逐步展开 TSS 层次，在 COIN 和 CrossTask 数据集上的任务识别、步骤识别和步骤预测任务上全面超越 SOTA。

**[No-Regret Strategy Solving in Imperfect-Information Games via Pre-Trained Embedding](no-regret_strategy_solving_in_imperfect-information_games_via_pre-trained_embedd.md)**

:   提出 Embedding CFR 算法，将不完美信息博弈中的信息集映射到连续低维嵌入空间（而非离散聚类），在相同空间开销下实现更快的可利用性收敛和更高质量的策略求解。

**[Perspective from a Broader Context: Can Room Style Knowledge Help Visual Floorplan Localization?](perspective_from_a_broader_context_can_room_style_knowledge_help_visual_floorpla.md)**

:   提出利用房间风格知识（通过无监督聚类预训练获得的 room discriminator）来消除视觉楼层平面图定位中因重复结构导致的歧义，在 Gibson 和 Structured3D 两个标准基准上取得 SOTA 性能。

**[PrefixGPT: Prefix Adder Optimization by a Generative Pre-trained Transformer](prefixgpt_prefix_adder_optimization_by_a_generative_pre-trained_transformer.md)**

:   提出PrefixGPT，将前缀加法器优化建模为序列生成问题，通过定制的GPT模型预训练学习设计规则后用RL微调生成优化设计，在面积-延迟乘积(ADP)上取得SOTA且对初始化不敏感。

**[Rectified Noise: A Generative Model Using Positive-incentive Noise](rectified_noise_a_generative_model_using_positive-incentive_noise.md)**

:   提出 Rectified Noise（ΔRN），通过正向激励噪声（π-noise）框架学习一组有益噪声并注入预训练 Rectified Flow 模型的速度场中，以仅 0.39% 的额外参数在 ImageNet-1k 上将 FID 从 10.16 降低到 9.05。

**[TRACE: A Generalizable Drift Detector for Streaming Data-Driven Optimization](trace_a_generalizable_drift_detector_for_streaming_data-driven_optimization.md)**

:   提出TRACE，一种基于注意力序列学习的可迁移概念漂移检测器，通过统计特征标记化和双注意力编码器学习跨任务可迁移的漂移模式，能泛化到未见过的数据集，并作为即插即用模块嵌入流式数据驱动优化算法。
