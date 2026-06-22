---
title: >-
  ICML2025 视频生成论文汇总 · 7篇论文解读
description: >-
  7篇ICML2025的视频生成方向论文解读，涵盖扩散模型、视频生成、多模态、对抗鲁棒等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2025"
  - "视频生成"
  - "论文解读"
  - "论文笔记"
  - "扩散模型"
  - "多模态"
  - "对抗鲁棒"
item_list:
  - u: "asymrnr_video_diffusion_transformers_acceleration_with_asymmetric_reduction_and_/"
    t: "AsymRnR: Video Diffusion Transformers Acceleration with Asymmetric Reduction and Restoration"
  - u: "ca2-vdm_efficient_autoregressive_video_diffusion_model_with_causal_generation_an/"
    t: "Ca2-VDM: Efficient Autoregressive Video Diffusion Model with Causal Generation and Cache Sharing"
  - u: "data-juicer_sandbox_a_feedback-driven_suite_for_multimodal_data-model_co-develop/"
    t: "Data-Juicer Sandbox: A Feedback-Driven Suite for Multimodal Data-Model Co-development"
  - u: "diffusion_adversarial_post-training_for_one-step_video_generation/"
    t: "Diffusion Adversarial Post-Training for One-Step Video Generation"
  - u: "how_far_is_video_generation_from_world_model_a_physical_law_perspective/"
    t: "How Far is Video Generation from World Model: A Physical Law Perspective"
  - u: "mimicmotion_high-quality_human_motion_video_generation_with_confidence-aware_pos/"
    t: "MimicMotion: High-Quality Human Motion Video Generation with Confidence-aware Pose Guidance"
  - u: "riflex_a_free_lunch_for_length_extrapolation_in_video_diffusion_transformers/"
    t: "RIFLEx: A Free Lunch for Length Extrapolation in Video Diffusion Transformers"
item_total: 7
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎬 视频生成

**🧪 ICML2025** · **7** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (182)](../../CVPR2026/video_generation/index.md) · [🔬 ICLR2026 (97)](../../ICLR2026/video_generation/index.md) · [💬 ACL2026 (4)](../../ACL2026/video_generation/index.md) · [🧪 ICML2026 (32)](../../ICML2026/video_generation/index.md) · [🤖 AAAI2026 (11)](../../AAAI2026/video_generation/index.md) · [🧠 NeurIPS2025 (23)](../../NeurIPS2025/video_generation/index.md)

🔥 **高频主题：** 扩散模型 ×4 · 视频生成 ×3

**[AsymRnR: Video Diffusion Transformers Acceleration with Asymmetric Reduction and Restoration](asymrnr_video_diffusion_transformers_acceleration_with_asymmetric_reduction_and_.md)**

:   提出 AsymRnR——一种免训练的视频 DiT 加速方法，基于注意力中不同组件（Q/K/V）、不同层、不同去噪步骤的冗余程度不同的观察，非对称地削减 token 以实现无损加速。

**[Ca2-VDM: Efficient Autoregressive Video Diffusion Model with Causal Generation and Cache Sharing](ca2-vdm_efficient_autoregressive_video_diffusion_model_with_causal_generation_an.md)**

:   提出 Ca2-VDM，通过因果生成（Causal Generation）和缓存共享（Cache Sharing）两大设计，消除自回归视频扩散模型中条件帧的冗余计算，将计算复杂度从二次降至线性，生成 80 帧视频速度比基线快 2.5 倍，同时保持 SOTA 级生成质量。

**[Data-Juicer Sandbox: A Feedback-Driven Suite for Multimodal Data-Model Co-development](data-juicer_sandbox_a_feedback-driven_suite_for_multimodal_data-model_co-develop.md)**

:   提出 Data-Juicer Sandbox 沙箱套件，通过"探测-分析-精炼"(Probe-Analyze-Refine) 工作流，在低成本小规模实验中系统探索数据处理算子 (OP) 与模型性能的交互关系，将获得的数据配方迁移到大规模场景，在 VBench 排行榜取得第一名。

**[Diffusion Adversarial Post-Training for One-Step Video Generation](diffusion_adversarial_post-training_for_one-step_video_generation.md)**

:   提出对抗式后训练（Adversarial Post-Training, APT）框架，通过在扩散模型预训练后引入对抗训练阶段，实现单步生成高质量视频（2秒、1280×720、24fps），模型名为Seaweed-APT。

**[How Far is Video Generation from World Model: A Physical Law Perspective](how_far_is_video_generation_from_world_model_a_physical_law_perspective.md)**

:   通过构建严格遵循经典力学定律的2D物理模拟视频数据集，系统性评估视频生成模型是否能从纯视觉数据中发现物理规律，揭示当前模型仅能记忆训练分布内的模式而无法泛化到新的物理条件。

**[MimicMotion: High-Quality Human Motion Video Generation with Confidence-aware Pose Guidance](mimicmotion_high-quality_human_motion_video_generation_with_confidence-aware_pos.md)**

:   基于 Stable Video Diffusion 构建姿态引导人体视频生成框架，通过将姿态估计置信度编码进引导信号、对高置信手部区域放大训练损失、以及位置感知的渐进式潜变量融合三项设计，在 TikTok 数据集上 FID-VID 达 9.3（前最优 12.4），同时支持任意长度平滑视频生成。

**[RIFLEx: A Free Lunch for Length Extrapolation in Video Diffusion Transformers](riflex_a_free_lunch_for_length_extrapolation_in_video_diffusion_transformers.md)**

:   通过系统分析RoPE位置编码中各频率分量的角色，发现存在一个"固有频率"主导外推时的时间重复行为，提出仅降低该频率使其在外推后保持单周期的最小化方案RIFLEx，在CogVideoX-5B和HunyuanVideo上实现无训练2×高质量视频外推。
