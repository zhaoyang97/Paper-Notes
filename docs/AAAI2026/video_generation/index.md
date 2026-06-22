---
title: >-
  AAAI2026 视频生成论文汇总 · 11篇论文解读
description: >-
  11篇AAAI2026的视频生成方向论文解读，涵盖视频生成、扩散模型、动态场景、布局/合成、RAG、对抗鲁棒等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "AAAI2026"
  - "视频生成"
  - "论文解读"
  - "论文笔记"
  - "扩散模型"
  - "动态场景"
  - "布局/合成"
  - "RAG"
  - "对抗鲁棒"
item_list:
  - u: "3d4d_an_interactive_editable_4d_world_model_via_3d_video_generation/"
    t: "3D4D: An Interactive Editable 4D World Model via 3D Video Generation"
  - u: "dreamrunner_fine-grained_compositional_story-to-video_genera/"
    t: "DreamRunner: Fine-Grained Compositional Story-to-Video Generation with Retrieval-Augmented Motion Adaptation"
  - u: "filmweaver_weaving_consistent_multi-shot_videos_with_cache-guided_autoregressive/"
    t: "FilmWeaver: Weaving Consistent Multi-Shot Videos with Cache-Guided Autoregressive Diffusion"
  - u: "genvidbench_a_6-million_benchmark_for_ai-generated_video_detection/"
    t: "GenVidBench: A 6-Million Benchmark for AI-Generated Video Detection"
  - u: "mask2iv_interaction-centric_video_generation_via_mask_trajectories/"
    t: "Mask2IV: Interaction-Centric Video Generation via Mask Trajectories"
  - u: "mofu_scale-aware_modulation_and_fourier_fusion_for_multi-subject_video_generatio/"
    t: "MoFu: Scale-Aware Modulation and Fourier Fusion for Multi-Subject Video Generation"
  - u: "motioncharacter_fine-grained_motion_controllable_human_video_generation/"
    t: "MotionCharacter: Fine-Grained Motion Controllable Human Video Generation"
  - u: "omnivdiff_omni_controllable_video_diffusion_for_generation_and_understanding/"
    t: "OmniVDiff: Omni Controllable Video Diffusion for Generation and Understanding"
  - u: "phased_one-step_adversarial_equilibrium_for_video_diffusion_models/"
    t: "Phased One-Step Adversarial Equilibrium for Video Diffusion Models"
  - u: "seeing_the_unseen_zooming_in_the_dark_with_event_cameras/"
    t: "Seeing the Unseen: Zooming in the Dark with Event Cameras"
  - u: "spherediff_tuning-free_360_static_and_dynamic_panorama_generation_via_spherical_/"
    t: "SphereDiff: Tuning-free Omnidirectional Panoramic Image and Video Generation via Spherical Latent Representation"
item_total: 11
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎬 视频生成

**🤖 AAAI2026** · **11** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (182)](../../CVPR2026/video_generation/index.md) · [🔬 ICLR2026 (97)](../../ICLR2026/video_generation/index.md) · [💬 ACL2026 (4)](../../ACL2026/video_generation/index.md) · [🧪 ICML2026 (32)](../../ICML2026/video_generation/index.md) · [🧠 NeurIPS2025 (23)](../../NeurIPS2025/video_generation/index.md) · [📹 ICCV2025 (49)](../../ICCV2025/video_generation/index.md)

🔥 **高频主题：** 视频生成 ×6 · 扩散模型 ×3

**[3D4D: An Interactive Editable 4D World Model via 3D Video Generation](3d4d_an_interactive_editable_4d_world_model_via_3d_video_generation.md)**

:   提出 3D4D，一个集成 WebGL 和 Supersplat 渲染的交互式 4D 可视化框架，通过四个后端模块（3D重建、图像生视频、视频分帧、4D场景生成）将静态图片和文本转化为可实时交互的 4D 场景，并引入 VLM 引导的注视点渲染策略在保持语义一致性的同时实现 60fps 实时交互。

**[DreamRunner: Fine-Grained Compositional Story-to-Video Generation with Retrieval-Augmented Motion Adaptation](dreamrunner_fine-grained_compositional_story-to-video_genera.md)**

:   提出 DreamRunner 框架，通过 LLM 双层规划 + 检索增强运动先验学习 + 时空区域3D注意力模块(SR3AI)，实现细粒度可控的多角色多事件故事视频生成。

**[FilmWeaver: Weaving Consistent Multi-Shot Videos with Cache-Guided Autoregressive Diffusion](filmweaver_weaving_consistent_multi-shot_videos_with_cache-guided_autoregressive.md)**

:   提出 FilmWeaver 框架，通过双层缓存（Shot Cache + Temporal Cache）引导自回归扩散模型，实现任意长度、跨镜头一致性的多镜头视频生成。

**[GenVidBench: A 6-Million Benchmark for AI-Generated Video Detection](genvidbench_a_6-million_benchmark_for_ai-generated_video_detection.md)**

:   提出 GenVidBench——首个 678 万级 AI 生成视频检测数据集，具备跨源（cross-source）和跨生成器（cross-generator）特性，覆盖 11 种 SOTA 视频生成器，并提供丰富的语义标注。

**[Mask2IV: Interaction-Centric Video Generation via Mask Trajectories](mask2iv_interaction-centric_video_generation_via_mask_trajectories.md)**

:   提出 Mask2IV，一个两阶段解耦框架——先预测交互者和物体的 mask 运动轨迹，再基于轨迹生成视频——实现了无需密集 mask 标注的、以交互为中心的可控视频生成，支持人-物交互和机器人操作两个场景。

**[MoFu: Scale-Aware Modulation and Fourier Fusion for Multi-Subject Video Generation](mofu_scale-aware_modulation_and_fourier_fusion_for_multi-subject_video_generatio.md)**

:   提出 MoFu，通过 Scale-Aware Modulation（LLM 引导的尺度感知调制）和 Fourier Fusion（基于 FFT 的排列不变特征融合）两个核心模块，同时解决多主体视频生成中的**尺度不一致**和**排列敏感性**两大挑战，并构建了 MoFu-1M 训练数据集和 MoFu-Bench 评测基准。

**[MotionCharacter: Fine-Grained Motion Controllable Human Video Generation](motioncharacter_fine-grained_motion_controllable_human_video_generation.md)**

:   提出 MotionCharacter 框架，通过将运动解耦为动作类型和运动强度两个独立可控维度，实现高保真人体视频生成中的细粒度运动控制和身份一致性保持。

**[OmniVDiff: Omni Controllable Video Diffusion for Generation and Understanding](omnivdiff_omni_controllable_video_diffusion_for_generation_and_understanding.md)**

:   提出 OmniVDiff，一个统一的可控视频扩散框架，通过将多种视觉模态（RGB、深度、分割、Canny）在颜色空间中联合建模，并引入自适应模态控制策略（AMCS），在单一扩散模型中同时支持文本条件生成、X 条件生成和视频理解三种任务，在 VBench 上达到 SOTA。

**[Phased One-Step Adversarial Equilibrium for Video Diffusion Models](phased_one-step_adversarial_equilibrium_for_video_diffusion_models.md)**

:   提出 V-PAE（Video Phased Adversarial Equilibrium），通过**稳定性预热 + 统一对抗均衡**两阶段蒸馏框架，将大规模视频扩散模型（如 Wan2.1-I2V-14B）压缩至单步生成，实现 100 倍加速，在 VBench-I2V 上平均质量超越已有加速方法 5.8%。

**[Seeing the Unseen: Zooming in the Dark with Event Cameras](seeing_the_unseen_zooming_in_the_dark_with_event_cameras.md)**

:   提出首个事件驱动低光视频超分（LVSR）框架 RetinexEVSR，通过 Retinex 启发的双向融合策略（RBF）——先用光照图引导事件特征去噪（IEE），再用增强后的事件特征恢复反射率细节（ERE），在 SDSD 基准上实现 2.95dB 增益且运行时间减少 65%。

**[SphereDiff: Tuning-free Omnidirectional Panoramic Image and Video Generation via Spherical Latent Representation](spherediff_tuning-free_360_static_and_dynamic_panorama_generation_via_spherical_.md)**

:   本文提出 SphereDiff，定义球面隐空间表示（Fibonacci Lattice 均匀分布）替代传统等距矩形投影，结合动态采样算法和畸变感知加权平均，无需微调即可利用 SANA/LTX Video 等预训练扩散模型生成无缝、低畸变的360度全景图像和视频。
