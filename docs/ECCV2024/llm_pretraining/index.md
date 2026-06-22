---
title: >-
  ECCV2024 预训练论文汇总 · 8篇论文解读
description: >-
  8篇ECCV2024的预训练方向论文解读，涵盖少样本学习、异常检测、布局/合成等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ECCV2024"
  - "预训练"
  - "论文解读"
  - "论文笔记"
  - "少样本学习"
  - "异常检测"
  - "布局/合成"
item_list:
  - u: "cross-domain_learning_for_video_anomaly_detection_with_limited_supervision/"
    t: "Cross-Domain Learning for Video Anomaly Detection with Limited Supervision"
  - u: "dragapart_learning_a_part-level_motion_prior_for_articulated_objects/"
    t: "DragAPart: Learning a Part-Level Motion Prior for Articulated Objects"
  - u: "i_canapost_believe_itaposs_not_scene_flow/"
    t: "I Can't Believe It's Not Scene Flow!"
  - u: "learning_to_obstruct_few-shot_image_classification_over_restricted_classes/"
    t: "Learning to Obstruct Few-Shot Image Classification over Restricted Classes"
  - u: "plan_posture_and_go_towards_open-vocabulary_text-to-motion_generation/"
    t: "Plan, Posture and Go: Towards Open-Vocabulary Text-to-Motion Generation"
  - u: "prelar_world_model_pre-training_with_learnable_action_representation/"
    t: "PreLAR: World Model Pre-training with Learnable Action Representation"
  - u: "prompting_language-informed_distribution_for_compositional_zero-shot_learning/"
    t: "Prompting Language-Informed Distribution for Compositional Zero-Shot Learning"
  - u: "scaling_backwards_minimal_synthetic_pre-training/"
    t: "Scaling Backwards: Minimal Synthetic Pre-training?"
item_total: 8
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📚 预训练

**🎞️ ECCV2024** · **8** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (5)](../../CVPR2026/llm_pretraining/index.md) · [🔬 ICLR2026 (79)](../../ICLR2026/llm_pretraining/index.md) · [💬 ACL2026 (12)](../../ACL2026/llm_pretraining/index.md) · [🧪 ICML2026 (27)](../../ICML2026/llm_pretraining/index.md) · [🤖 AAAI2026 (9)](../../AAAI2026/llm_pretraining/index.md) · [🧠 NeurIPS2025 (51)](../../NeurIPS2025/llm_pretraining/index.md)

🔥 **高频主题：** 少样本学习 ×2

**[Cross-Domain Learning for Video Anomaly Detection with Limited Supervision](cross-domain_learning_for_video_anomaly_detection_with_limited_supervision.md)**

:   提出弱监督跨域学习（CDL）框架，通过不确定性驱动的伪标签机制将无标注外部视频整合到训练中，显著提升视频异常检测的跨域泛化能力。

**[DragAPart: Learning a Part-Level Motion Prior for Articulated Objects](dragapart_learning_a_part-level_motion_prior_for_articulated_objects.md)**

:   DragAPart 提出了一种以拖拽为交互接口的图像生成器，能够响应部件级别的交互（如开关抽屉/门），而非仅仅移动整个物体。通过新的合成数据集 Drag-a-Move、多分辨率拖拽编码和域随机化策略，模型在仅用合成数据训练的情况下能良好泛化到真实图像和未见类别。

**[I Can't Believe It's Not Scene Flow!](i_canapost_believe_itaposs_not_scene_flow.md)**

:   揭示现有场景流方法在行人等小目标上的灾难性失败被现有评估指标所掩盖，提出类别感知且速度归一化的Bucket Normalized EPE评估协议，以及一个简单但SOTA的TrackFlow基线（检测器+跟踪器生成场景流），在行人运动描述上实现1.5倍提升。

**[Learning to Obstruct Few-Shot Image Classification over Restricted Classes](learning_to_obstruct_few-shot_image_classification_over_restricted_classes.md)**

:   提出 Learning to Obstruct (LTO) 算法，通过类似 MAML 的元学习方式修改预训练 backbone 参数，使其成为特定受限类别的"差初始化"，从而阻碍少样本分类方法在受限类上的微调效果，同时保持其他类别的正常性能。

**[Plan, Posture and Go: Towards Open-Vocabulary Text-to-Motion Generation](plan_posture_and_go_towards_open-vocabulary_text-to-motion_generation.md)**

:   本文提出 PRO-Motion 分治框架，将文本到动作生成分解为三个阶段：LLM 驱动的动作规划（Plan）、基于脚本的姿态扩散生成（Posture）、以及全身平移旋转估计（Go），通过降低各阶段的复杂度实现了开放词汇的高质量动作生成。

**[PreLAR: World Model Pre-training with Learnable Action Representation](prelar_world_model_pre-training_with_learnable_action_representation.md)**

:   本文提出PreLAR，在无动作标签的视频上进行世界模型预训练时，通过从相邻帧编码隐式动作表示并设计动作-状态一致性损失来弥合无动作预训练与有动作微调之间的差距，显著提升了下游视觉控制任务的样本效率。

**[Prompting Language-Informed Distribution for Compositional Zero-Shot Learning](prompting_language-informed_distribution_for_compositional_zero-shot_learning.md)**

:   本文提出 PLID 方法，利用 LLM 生成的句子级类别描述构建语言知识驱动的高斯分布，配合视觉-语言原语分解和随机 logit 融合，在组合零样本学习（CZSL）任务上取得 SOTA。

**[Scaling Backwards: Minimal Synthetic Pre-training?](scaling_backwards_minimal_synthetic_pre-training.md)**

:   提出 1p-frac——仅用单个分形图像的微小扰动即可实现与 ImageNet-1k 级别可比的预训练效果，挑战了"预训练需要大规模数据集"的常规认知，揭示预训练本质可能更接近权重初始化而非视觉概念学习。
