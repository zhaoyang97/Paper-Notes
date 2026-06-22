---
title: >-
  ECCV2024 视频生成论文汇总 · 14篇论文解读
description: >-
  14篇ECCV2024的视频生成方向论文解读，涵盖扩散模型、视频生成、超分辨率等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ECCV2024"
  - "视频生成"
  - "论文解读"
  - "论文笔记"
  - "扩散模型"
  - "超分辨率"
item_list:
  - u: "blazebvd_make_scale-time_equalization_great_again_for_blind_video_deflickering/"
    t: "BlazeBVD: Make Scale-Time Equalization Great Again for Blind Video Deflickering"
  - u: "draganything_motion_control_for_anything_using_entity_representation/"
    t: "DragAnything: Motion Control for Anything using Entity Representation"
  - u: "dreammotion_space-time_self-similar_score_distillation_for_zero-shot_video_editi/"
    t: "DreamMotion: Space-Time Self-Similar Score Distillation for Zero-Shot Video Editing"
  - u: "evaluating_text-to-visual_generation_with_image-to-text_generation/"
    t: "Evaluating Text-to-Visual Generation with Image-to-Text Generation"
  - u: "exploring_pre-trained_text-to-video_diffusion_models_for_referring_video_object_/"
    t: "Exploring Pre-trained Text-to-Video Diffusion Models for Referring Video Object Segmentation"
  - u: "freeinit_bridging_initialization_gap_in_video_diffusion_models/"
    t: "FreeInit: Bridging Initialization Gap in Video Diffusion Models"
  - u: "kalman-inspired_feature_propagation_for_video_face_super-resolution/"
    t: "Kalman-Inspired Feature Propagation for Video Face Super-Resolution"
  - u: "magdiff_multi-alignment_diffusion_for_high-fidelity_video_generation_and_editing/"
    t: "MagDiff: Multi-Alignment Diffusion for High-Fidelity Video Generation and Editing"
  - u: "mofa-video_controllable_image_animation_via_generative_motion_field_adaptions_in/"
    t: "MOFA-Video: Controllable Image Animation via Generative Motion Field Adaptions in Frozen Image-to-Video Diffusion Model"
  - u: "physdreamer_physics-based_interaction_with_3d_objects_via_video_generation/"
    t: "PhysDreamer: Physics-Based Interaction with 3D Objects via Video Generation"
  - u: "realviformer_investigating_attention_for_real-world_video_super-resolution/"
    t: "RealViformer: Investigating Attention for Real-World Video Super-Resolution"
  - u: "sv3d_novel_multi-view_synthesis_and_3d_generation_from_a_single_image_using_late/"
    t: "SV3D: Novel Multi-view Synthesis and 3D Generation from a Single Image using Latent Video Diffusion"
  - u: "vfusion3d_learning_scalable_3d_generative_models_from_video_diffusion_models/"
    t: "VFusion3D: Learning Scalable 3D Generative Models from Video Diffusion Models"
  - u: "videoshop_localized_semantic_video_editing_with_noise-extrapolated_diffusion_inv/"
    t: "Videoshop: Localized Semantic Video Editing with Noise-Extrapolated Diffusion Inversion"
item_total: 14
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎬 视频生成

**🎞️ ECCV2024** · **14** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (182)](../../CVPR2026/video_generation/index.md) · [🔬 ICLR2026 (97)](../../ICLR2026/video_generation/index.md) · [💬 ACL2026 (4)](../../ACL2026/video_generation/index.md) · [🧪 ICML2026 (32)](../../ICML2026/video_generation/index.md) · [🤖 AAAI2026 (11)](../../AAAI2026/video_generation/index.md) · [🧠 NeurIPS2025 (23)](../../NeurIPS2025/video_generation/index.md)

🔥 **高频主题：** 扩散模型 ×7 · 视频生成 ×3 · 超分辨率 ×2

**[BlazeBVD: Make Scale-Time Equalization Great Again for Blind Video Deflickering](blazebvd_make_scale-time_equalization_great_again_for_blind_video_deflickering.md)**

:   提出 BlazeBVD，利用经典 Scale-Time Equalization (STE) 在光照直方图空间提取 deflickering 先验（滤波光照图、曝光图、闪烁帧索引），将复杂的视频时空学习简化为 2D 空间网络逐帧处理 + 轻量 3D 时序一致性网络，在盲视频去闪烁任务上实现 SOTA 质量且推理速度比基线快 10 倍以上。

**[DragAnything: Motion Control for Anything using Entity Representation](draganything_motion_control_for_anything_using_entity_representation.md)**

:   提出DragAnything，利用扩散模型的隐空间特征作为实体表征（Entity Representation）来实现实体级运动控制，解决了现有轨迹驱动方法仅拖拽像素而无法精确控制目标对象运动的问题，在VIPSeg上实现SOTA的FVD/FID指标，用户研究中运动控制投票超出DragNUWA 26%。

**[DreamMotion: Space-Time Self-Similar Score Distillation for Zero-Shot Video Editing](dreammotion_space-time_self-similar_score_distillation_for_zero-shot_video_editi.md)**

:   提出基于分数蒸馏（Score Distillation）的零样本视频编辑框架DreamMotion，通过时空自相似性正则化在注入目标外观的同时保持原始视频的结构和运动完整性，适用于级联和非级联视频扩散模型。

**[Evaluating Text-to-Visual Generation with Image-to-Text Generation](evaluating_text-to-visual_generation_with_image-to-text_generation.md)**

:   提出VQAScore，利用VQA模型替代CLIP来评估文本-视觉生成质量，在复杂组合性提示上大幅超越CLIPScore，并发布GenAI-Bench基准。

**[Exploring Pre-trained Text-to-Video Diffusion Models for Referring Video Object Segmentation](exploring_pre-trained_text-to-video_diffusion_models_for_referring_video_object_.md)**

:   本文首次探索预训练文本到视频（T2V）扩散模型的视觉特征用于视频理解任务，提出 VD-IT 框架，通过文本引导的图像投影和视频特定噪声预测两项关键设计，从固定的 T2V 扩散模型中提取具有优越时序语义一致性的视觉特征，在 R-VOS 四大基准上超越了使用判别式预训练视频骨干网络（如 Video Swin Transformer）的 SOTA 方法。

**[FreeInit: Bridging Initialization Gap in Video Diffusion Models](freeinit_bridging_initialization_gap_in_video_diffusion_models.md)**

:   发现视频扩散模型存在训练-推理初始化差异（训练时低频信息泄露导致初始噪声具有时序相关性，而推理时使用无相关的高斯噪声），提出 FreeInit 通过迭代精炼初始噪声的时空低频成分来弥合该差异，显著提升视频生成的时序一致性。

**[Kalman-Inspired Feature Propagation for Video Face Super-Resolution](kalman-inspired_feature_propagation_for_video_face_super-resolution.md)**

:   本文提出 KEEP 框架，借鉴卡尔曼滤波原理在隐空间中递归融合前帧先验与当前帧观测，实现视频人脸超分辨率中面部细节的高保真恢复与时序一致性，在 VFHQ 数据集上 PSNR 超过此前最优方法 0.8 dB。

**[MagDiff: Multi-Alignment Diffusion for High-Fidelity Video Generation and Editing](magdiff_multi-alignment_diffusion_for_high-fidelity_video_generation_and_editing.md)**

:   提出首个统一视频生成与编辑的多对齐扩散模型 MagDiff，通过主体驱动对齐、自适应提示对齐和高保真对齐三种策略，在单一无微调框架中同时实现高质量视频生成与编辑。

**[MOFA-Video: Controllable Image Animation via Generative Motion Field Adaptions in Frozen Image-to-Video Diffusion Model](mofa-video_controllable_image_animation_via_generative_motion_field_adaptions_in.md)**

:   提出 MOFA-Video，通过设计多个领域感知运动场适配器（MOFA-Adapter）为冻结的视频扩散模型（SVD）添加可控运动能力，支持手绘轨迹、人脸关键点等多种控制信号及其组合，实现开放域可控图像动画。

**[PhysDreamer: Physics-Based Interaction with 3D Objects via Video Generation](physdreamer_physics-based_interaction_with_3d_objects_via_video_generation.md)**

:   利用视频生成模型中隐含的物理动力学先验，为静态3D高斯对象估计空间变化的杨氏模量材料场，从而实现物理合理的交互式3D动力学合成。

**[RealViformer: Investigating Attention for Real-World Video Super-Resolution](realviformer_investigating_attention_for_real-world_video_super-resolution.md)**

:   本文系统研究了空间注意力和通道注意力在真实世界视频超分辨率（RWVSR）中的行为差异，发现通道注意力对退化伪影更鲁棒但会导致特征冗余，据此提出了带有改进通道注意力（ICA）和通道注意力融合（CAF）模块的 RealViformer，以更少的参数和更快的速度达到 SOTA。

**[SV3D: Novel Multi-view Synthesis and 3D Generation from a Single Image using Latent Video Diffusion](sv3d_novel_multi-view_synthesis_and_3d_generation_from_a_single_image_using_late.md)**

:   提出SV3D，将图像到视频扩散模型适配为多视图合成和3D生成，利用视频模型的泛化能力和多视图一致性，并引入显式相机控制。

**[VFusion3D: Learning Scalable 3D Generative Models from Video Diffusion Models](vfusion3d_learning_scalable_3d_generative_models_from_video_diffusion_models.md)**

:   提出利用预训练视频扩散模型（EMU Video）作为多视图数据引擎，通过微调使其生成3D一致的多视图视频，从而构建约300万合成数据训练前馈式3D生成模型VFusion3D，实现从单张图片秒级生成3D资产，用户偏好率超过90%。

**[Videoshop: Localized Semantic Video Editing with Noise-Extrapolated Diffusion Inversion](videoshop_localized_semantic_video_editing_with_noise-extrapolated_diffusion_inv.md)**

:   提出Videoshop——一种免训练的局部语义视频编辑方法，用户可通过任意图像编辑工具修改视频首帧，系统基于噪声外推扩散反演和隐变量归一化技术，自动将编辑传播到所有帧，同时保持语义、空间和时序一致性，在10个指标上超越6个基线方法。
