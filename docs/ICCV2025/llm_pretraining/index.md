---
title: >-
  ICCV2025 预训练论文汇总 · 9篇论文解读
description: >-
  9篇ICCV2025的预训练方向论文解读，涵盖域适应、对抗鲁棒、扩散模型等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICCV2025"
  - "预训练"
  - "论文解读"
  - "论文笔记"
  - "域适应"
  - "对抗鲁棒"
  - "扩散模型"
item_list:
  - u: "aceg_improving_generalization_of_scene_coordinate_regression/"
    t: "ACE-G: Improving Generalization of Scene Coordinate Regression Through Query Pre-Training"
  - u: "conststyle_robust_domain_generalization_with_unified_style_transformation/"
    t: "ConstStyle: Robust Domain Generalization with Unified Style Transformation"
  - u: "dataset_ownership_verification_for_pre-trained_masked_models/"
    t: "Dataset Ownership Verification for Pre-trained Masked Models"
  - u: "eta_energy-based_test-time_adaptation_for_depth_completion/"
    t: "ETA: Energy-based Test-time Adaptation for Depth Completion"
  - u: "flow_to_the_mode_mode-seeking_diffusion_autoencoders_for_state-of-the-art_image_/"
    t: "FlowMo: Flow to the Mode — Mode-Seeking Diffusion Autoencoders for State-of-the-Art Image Tokenization"
  - u: "image_intrinsic_scale_assessment_bridging_the_gap_between_quality_and_resolution/"
    t: "Image Intrinsic Scale Assessment: Bridging the Gap Between Quality and Resolution"
  - u: "make_your_training_flexible_towards_deployment-efficient_video_models/"
    t: "Make Your Training Flexible: Towards Deployment-Efficient Video Models"
  - u: "synchronization_of_multiple_videos/"
    t: "Synchronization of Multiple Videos"
  - u: "syncity_training-free_generation_of_3d_worlds/"
    t: "SynCity: Training-Free Generation of 3D Worlds"
item_total: 9
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📚 预训练

**📹 ICCV2025** · **9** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (5)](../../CVPR2026/llm_pretraining/index.md) · [🔬 ICLR2026 (79)](../../ICLR2026/llm_pretraining/index.md) · [💬 ACL2026 (12)](../../ACL2026/llm_pretraining/index.md) · [🧪 ICML2026 (27)](../../ICML2026/llm_pretraining/index.md) · [🤖 AAAI2026 (9)](../../AAAI2026/llm_pretraining/index.md) · [🧠 NeurIPS2025 (51)](../../NeurIPS2025/llm_pretraining/index.md)

**[ACE-G: Improving Generalization of Scene Coordinate Regression Through Query Pre-Training](aceg_improving_generalization_of_scene_coordinate_regression.md)**

:   将场景坐标回归器拆分为「场景无关的Transformer」和「场景特定的map code」，通过在数万场景上进行交替的mapping/query预训练，显著提升SCR方法在光照、视角变化下的泛化能力，同时保持轻量化的计算开销。

**[ConstStyle: Robust Domain Generalization with Unified Style Transformation](conststyle_robust_domain_generalization_with_unified_style_transformation.md)**

:   提出ConstStyle框架，通过构建一个理论驱动的"统一域"（Unified Domain），在训练时将所有样本风格对齐到该统一域，测试时将未见域样本部分投影到统一域，有效缩小域间差距并提升泛化性能。

**[Dataset Ownership Verification for Pre-trained Masked Models](dataset_ownership_verification_for_pre-trained_masked_models.md)**

:   DOV4MM 提出了首个针对掩码预训练模型的数据集所有权验证方法，通过比较"见过"与"未见过"样本在嵌入空间中遮掩信息重构难度的差异，利用配对 t 检验判断黑盒模型是否使用了特定数据集进行预训练，在 10 种掩码图像模型和 4 种掩码语言模型上均实现 p 值远低于 0.05 的准确验证。

**[ETA: Energy-based Test-time Adaptation for Depth Completion](eta_energy-based_test-time_adaptation_for_depth_completion.md)**

:   提出ETA方法，利用能量模型量化深度预测属于源域分布的可能性，并在测试时通过最小化目标域预测的能量值来引导预训练深度补全模型适配到新环境，在室外和室内场景平均比先前SOTA分别提升6.94%和10.23%。

**[FlowMo: Flow to the Mode — Mode-Seeking Diffusion Autoencoders for State-of-the-Art Image Tokenization](flow_to_the_mode_mode-seeking_diffusion_autoencoders_for_state-of-the-art_image_.md)**

:   提出 FlowMo，一种基于 Transformer 的扩散自编码器 (diffusion autoencoder)，通过两阶段训练（mode-matching 预训练 + mode-seeking 后训练），首次实现扩散自编码器在 ImageNet-1K 离散图像 tokenization 上的 SOTA 性能，无需使用卷积、对抗损失、2D 空间对齐 latent 或从其他 tokenizer 蒸馏。

**[Image Intrinsic Scale Assessment: Bridging the Gap Between Quality and Resolution](image_intrinsic_scale_assessment_bridging_the_gap_between_quality_and_resolution.md)**

:   本文定义了图像内在尺度（IIS）这一新概念——即图像展现最高感知质量的最大缩放比例，并提出 IISA 任务、构建了 785 张图像的数据集，以及基于弱标签的 WIISA 训练策略，在多个 NR-IQA 方法上一致提升了 IIS 预测性能。

**[Make Your Training Flexible: Towards Deployment-Efficient Video Models](make_your_training_flexible_towards_deployment-efficient_video_models.md)**

:   本文提出Flux——一种使视频模型训练灵活化的数据增强工具，通过灵活采样网格+组动态token选择，使单一模型在不同计算预算下都能高效工作；并提出Token Optimization新测试范式，在1/4 token下即可匹配前SOTA性能，节省约90%计算。

**[Synchronization of Multiple Videos](synchronization_of_multiple_videos.md)**

:   提出 Temporal Prototype Learning (TPL)，一个基于原型的视频同步框架，从预训练模型提取的高维嵌入中构建共享的紧凑1D表征，通过学习统一的原型序列锚定关键动作阶段来对齐多个视频，首次解决了生成式AI视频的同步问题。

**[SynCity: Training-Free Generation of 3D Worlds](syncity_training-free_generation_of_3d_worlds.md)**

:   SynCity 提出了一种无需训练/优化的方法，通过精心设计的 prompt engineering 策略组合预训练的语言模型、2D图像生成器和3D生成器（TRELLIS），以 tile-by-tile 的方式自回归生成大规模、高质量、可自由导航的3D世界。
