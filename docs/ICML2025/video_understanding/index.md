---
title: >-
  ICML2025 视频理解论文汇总 · 4篇论文解读
description: >-
  4篇ICML2025的视频理解方向论文解读，收录 Fine-Grained Captioning of Lon、MoMa、Scaling Video-Language Models等。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "ICML2025"
  - "视频理解"
  - "论文解读"
  - "论文笔记"
item_list:
  - u: "fine-grained_captioning_of_long_videos_through_scene_graph_consolidation/"
    t: "Fine-Grained Captioning of Long Videos through Scene Graph Consolidation"
  - u: "moma_modulating_mamba_for_adapting_image_foundation_models_to_video_recognition/"
    t: "MoMa: Modulating Mamba for Adapting Image Foundation Models to Video Recognition"
  - u: "scaling_video-language_models_to_10k_frames_via_hierarchical_differential_distil/"
    t: "Scaling Video-Language Models to 10K Frames via Hierarchical Differential Distillation"
  - u: "unifying_specialized_visual_encoders_for_video_language_models/"
    t: "Unifying Specialized Visual Encoders for Video Language Models"
item_total: 4
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📹 视频理解

**🧪 ICML2025** · **4** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (187)](../../CVPR2026/video_understanding/index.md) · [🔬 ICLR2026 (48)](../../ICLR2026/video_understanding/index.md) · [🧪 ICML2026 (17)](../../ICML2026/video_understanding/index.md) · [🤖 AAAI2026 (27)](../../AAAI2026/video_understanding/index.md) · [🧠 NeurIPS2025 (39)](../../NeurIPS2025/video_understanding/index.md) · [📹 ICCV2025 (56)](../../ICCV2025/video_understanding/index.md)

**[Fine-Grained Captioning of Long Videos through Scene Graph Consolidation](fine-grained_captioning_of_long_videos_through_scene_graph_consolidation.md)**

:   提出 SGVC 框架，通过将视频各段的文本描述解析为场景图、用 Hungarian 算法迭代合并为统一图表示、再用轻量图到文本解码器生成视频级描述，以极低计算开销实现了超越 LLM-based 方法的零样本长视频描述性能。

**[MoMa: Modulating Mamba for Adapting Image Foundation Models to Video Recognition](moma_modulating_mamba_for_adapting_image_foundation_models_to_video_recognition.md)**

:   提出 MoMa 框架，通过序列调制操作 (SeqMod) 将 Mamba 的线性复杂度 SSM 以 scale-bias 方式注入冻结的 CLIP Transformer，实现高效全时空动态建模，在多个视频识别基准上以更少计算量达到 SOTA 水平。

**[Scaling Video-Language Models to 10K Frames via Hierarchical Differential Distillation](scaling_video-language_models_to_10k_frames_via_hierarchical_differential_distil.md)**

:   ViLaMP 提出差分蒸馏 (Differential Distillation) 原则，通过层次化的帧级差分关键帧选择 (DKS) 和 patch 级差分特征融合 (DFM) 两种机制实现"混合精度"视频处理——关键帧保留全部视觉 token，非关键帧压缩为单个 token，成功在单张 A100 GPU 上处理长达 10K 帧（约 2.7 小时）的超长视频。

**[Unifying Specialized Visual Encoders for Video Language Models](unifying_specialized_visual_encoders_for_video_language_models.md)**

:   MERV 提出了多编码器视频表示方法，将四种专长不同的视觉编码器（DINOv2、ViViT、SigLIP、LanguageBind）通过时空对齐和跨注意力融合整合到单一 VideoLLM 中，在视频推理基准上比基线 Video-LLaVA 提升最高 4.62%，并验证了不同编码器的互补专长。
