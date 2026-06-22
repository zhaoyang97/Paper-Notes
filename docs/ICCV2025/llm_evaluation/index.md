---
title: >-
  ICCV2025 LLM评测论文汇总 · 27篇论文解读
description: >-
  27篇ICCV2025的 LLM 评测方向论文解读，涵盖少样本学习、推理、布局/合成、扩散模型、对抗鲁棒等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICCV2025"
  - "LLM 评测"
  - "论文解读"
  - "论文笔记"
  - "少样本学习"
  - "推理"
  - "布局/合成"
  - "扩散模型"
  - "对抗鲁棒"
item_list:
  - u: "3dsrbench_a_comprehensive_3d_spatial_reasoning_benchmark/"
    t: "3DSRBench: A Comprehensive 3D Spatial Reasoning Benchmark"
  - u: "a_conditional_probability_framework_for_compositional_zerosh/"
    t: "A Conditional Probability Framework for Compositional Zero-shot Learning"
  - u: "a_real-world_display_inverse_rendering_dataset/"
    t: "A Real-world Display Inverse Rendering Dataset"
  - u: "batclip_bimodal_online_test-time_adaptation_for_clip/"
    t: "BATCLIP: Bimodal Online Test-Time Adaptation for CLIP"
  - u: "combinative_matching_for_geometric_shape_assembly/"
    t: "Combinative Matching for Geometric Shape Assembly"
  - u: "degradation-modeled_multipath_diffusion_for_tunable_metalens_photography/"
    t: "Degradation-Modeled Multipath Diffusion for Tunable Metalens Photography"
  - u: "discontinuity-aware_normal_integration_for_generic_central_camera_models/"
    t: "Discontinuity-aware Normal Integration for Generic Central Camera Models"
  - u: "discopatch_taming_adversarially-driven_batch_statistics_for_improved_out-of-dist/"
    t: "DisCoPatch: Taming Adversarially-driven Batch Statistics for Improved Out-of-Distribution Detection"
  - u: "dista-net_dynamic_closely-spaced_infrared_small_target_unmixing/"
    t: "DISTA-Net: Dynamic Closely-Spaced Infrared Small Target Unmixing"
  - u: "few-shot_pattern_detection_via_template_matching_and_regression/"
    t: "Few-Shot Pattern Detection via Template Matching and Regression"
  - u: "forcennet_foreground-centric_network_for_document_image_rectification/"
    t: "ForCenNet: Foreground-Centric Network for Document Image Rectification"
  - u: "generative_zoo/"
    t: "Generative Zoo"
  - u: "hiero_understanding_the_hierarchy_of_human_behavior_enhances_reasoning_on_egocen/"
    t: "HiERO: Understanding the Hierarchy of Human Behavior Enhances Reasoning on Egocentric Videos"
  - u: "imbalance_in_balance_online_concept_balancing_in_generation_models/"
    t: "Imbalance in Balance: Online Concept Balancing in Generation Models"
  - u: "intersyn_interleaved_learning_for_dynamic_motion_synthesis_in_the_wild/"
    t: "InterSyn: Interleaved Learning for Dynamic Motion Synthesis in the Wild"
  - u: "lay2story_extending_diffusion_transformers_for_layout-togglable_story_generation/"
    t: "Lay2Story: Extending Diffusion Transformers for Layout-Togglable Story Generation"
  - u: "neural_multi-view_self-calibrated_photometric_stereo_without_photometric_stereo_/"
    t: "Neural Multi-View Self-Calibrated Photometric Stereo without Photometric Stereo Cues"
  - u: "odp-bench_benchmarking_out-of-distribution_performance_prediction/"
    t: "ODP-Bench: Benchmarking Out-of-Distribution Performance Prediction"
  - u: "omnidiff_a_comprehensive_benchmark_for_fine-grained_image_difference_captioning/"
    t: "OmniDiff: A Comprehensive Benchmark for Fine-grained Image Difference Captioning"
  - u: "on_the_robustness_tradeoff_in_fine-tuning/"
    t: "On the Robustness Tradeoff in Fine-Tuning"
  - u: "phatnet_a_physics-guided_haze_transfer_network_for_domain-adaptive_real-world_im/"
    t: "PHATNet: A Physics-guided Haze Transfer Network for Domain-adaptive Real-world Image Dehazing"
  - u: "rethinking_few_shot_clip_benchmarks_a_critical_analysis_in_the_inductive_setting/"
    t: "Rethinking Few Shot CLIP Benchmarks: A Critical Analysis in the Inductive Setting"
  - u: "sketchsplat_3d_edge_reconstruction_via_differentiable_multi-view_sketch_splattin/"
    t: "SketchSplat: 3D Edge Reconstruction via Differentiable Multi-view Sketch Splatting"
  - u: "spectral_sensitivity_estimation_with_an_uncalibrated_diffraction_grating/"
    t: "Spectral Sensitivity Estimation with an Uncalibrated Diffraction Grating"
  - u: "streammind_unlocking_full_frame_rate_streaming_video_dialogue_through_event-gate/"
    t: "StreamMind: Unlocking Full Frame Rate Streaming Video Dialogue through Event-Gated Cognition"
  - u: "supercharging_floorplan_localization_with_semantic_rays/"
    t: "Supercharging Floorplan Localization with Semantic Rays"
  - u: "svtrv2_ctc_beats_encoder-decoder_models_in_scene_text_recognition/"
    t: "SVTRv2: CTC Beats Encoder-Decoder Models in Scene Text Recognition"
item_total: 27
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📊 LLM 评测

**📹 ICCV2025** · **27** 篇论文解读

📌 **同领域跨会议浏览：** [🔬 ICLR2026 (131)](../../ICLR2026/llm_evaluation/index.md) · [💬 ACL2026 (96)](../../ACL2026/llm_evaluation/index.md) · [🧪 ICML2026 (40)](../../ICML2026/llm_evaluation/index.md) · [🤖 AAAI2026 (16)](../../AAAI2026/llm_evaluation/index.md) · [🧠 NeurIPS2025 (38)](../../NeurIPS2025/llm_evaluation/index.md) · [🧪 ICML2025 (22)](../../ICML2025/llm_evaluation/index.md)

🔥 **高频主题：** 少样本学习 ×3 · 推理 ×2 · 布局/合成 ×2 · 扩散模型 ×2 · 对抗鲁棒 ×2

**[3DSRBench: A Comprehensive 3D Spatial Reasoning Benchmark](3dsrbench_a_comprehensive_3d_spatial_reasoning_benchmark.md)**

:   提出首个全面的3D空间推理基准3DSRBench，包含2,772个人工标注的VQA对（12种问题类型），通过平衡数据分布和新型FlipEval策略实现鲁棒评估，揭示SOTA LMM（包括GPT-4o、Gemini）在3D空间推理上远落后于人类水平（≈52% vs 95.7%），且在非常规视角下性能显著退化。

**[A Conditional Probability Framework for Compositional Zero-shot Learning](a_conditional_probability_framework_for_compositional_zerosh.md)**

:   提出条件概率框架（CPF），将组合识别概率分解为对象似然 p(o|x) 和属性条件似然 p(a|o,x) 两部分，通过文本增强对象学习和对象引导属性学习两个模块显式建模属性-对象依赖关系，在三个 CZSL 基准上全面超越 SOTA。

**[A Real-world Display Inverse Rendering Dataset](a_real-world_display_inverse_rendering_dataset.md)**

:   本文构建了首个基于LCD显示器-相机系统的真实世界逆渲染数据集，包含16个不同材质物体在OLAT照明模式下的立体偏振图像及高精度几何真值，并提出了一个简单有效的显示器逆渲染基线方法，超越了现有逆渲染方法。

**[BATCLIP: Bimodal Online Test-Time Adaptation for CLIP](batclip_bimodal_online_test-time_adaptation_for_clip.md)**

:   提出BATCLIP，一种针对CLIP的双模态在线测试时自适应（TTA）方法，通过同时适应视觉编码器和文本编码器的LayerNorm参数，引入投影匹配损失和类间可分性损失来增强图文特征对齐和类别区分度，在CIFAR-10C/100C/ImageNet-C上达到SOTA效果。

**[Combinative Matching for Geometric Shape Assembly](combinative_matching_for_geometric_shape_assembly.md)**

:   提出组合匹配（Combinative Matching）方法，同时建模互锁部件的"表面形状一致性"和"体积占用相反性"两大属性，通过等变网络学习方向对齐、形状匹配与占用匹配三个目标，大幅减少几何组装中的局部歧义。

**[Degradation-Modeled Multipath Diffusion for Tunable Metalens Photography](degradation-modeled_multipath_diffusion_for_tunable_metalens_photography.md)**

:   提出DMDiff框架，利用预训练扩散模型的自然图像先验，通过正/中/负三路径多提示扩散策略和空间变化退化感知注意力（SVDA）模块，实现毫米级超透镜相机的高保真可调图像重建，在多项指标上超越现有方法。

**[Discontinuity-aware Normal Integration for Generic Central Camera Models](discontinuity-aware_normal_integration_for_generic_central_camera_models.md)**

:   提出一种支持显式不连续性建模和通用中心相机模型的法线积分新方法，通过局部平面性假设建立法线与光线方向之间的约束，在标准法线积分基准上达到 SOTA，并首次直接处理通用中心相机（如鱼眼、全景相机）。

**[DisCoPatch: Taming Adversarially-driven Batch Statistics for Improved Out-of-Distribution Detection](discopatch_taming_adversarially-driven_batch_statistics_for_improved_out-of-dist.md)**

:   提出DisCoPatch框架，利用对抗性VAE中BatchNorm对批统计量的内在偏向性来区分ID和OOD样本，通过推理时将同一图像的多个patch组成batch来保证分布一致性，在协变量偏移OOD检测（ImageNet-1K(-C) 95.5% AUROC）和近分布OOD检测（95.0% AUROC）上达到SOTA，模型仅25MB且延迟低一个数量级。

**[DISTA-Net: Dynamic Closely-Spaced Infrared Small Target Unmixing](dista-net_dynamic_closely-spaced_infrared_small_target_unmixing.md)**

:   DISTA-Net提出动态深度展开网络，将ISTA稀疏重建中的非线性变换和阈值参数从静态改为根据输入自适应生成，实现密集红外小目标的首个深度学习解混方法，并建立了包含数据集、评估指标和工具包的首个开源生态。

**[Few-Shot Pattern Detection via Template Matching and Regression](few-shot_pattern_detection_via_template_matching_and_regression.md)**

:   本文提出TMR方法，通过经典模板匹配结合支持条件化边界框回归，实现了对任意模式（包括非物体级模式）的小样本检测，同时引入RPINE数据集覆盖更广泛的重复模式，在多个基准上超越现有FSCD方法并展现出强大的跨数据集泛化能力。

**[ForCenNet: Foreground-Centric Network for Document Image Rectification](forcennet_foreground-centric_network_for_document_image_rectification.md)**

:   提出以前景为中心的文档矫正网络ForCenNet，通过前景标签生成、掩码引导Transformer解码器和曲率一致性损失三大创新，仅需无畸变图像即可高效训练，在DocUNet、DIR300、WarpDoc、DocReal四个基准上达到SOTA。

**[Generative Zoo](generative_zoo.md)**

:   提出一种利用条件图像生成模型（FLUX + ControlNet）合成动物 3D 姿态和形状训练数据的可扩展流水线，生成百万级 GenZoo 数据集，仅用合成数据训练即在真实世界基准上达到 SOTA。

**[HiERO: Understanding the Hierarchy of Human Behavior Enhances Reasoning on Egocentric Videos](hiero_understanding_the_hierarchy_of_human_behavior_enhances_reasoning_on_egocen.md)**

:   提出 HiERO，一种弱监督的层次化图架构，通过对齐视频片段与叙述文本来学习功能性活动线索的层次结构，使视频片段特征编码多尺度的行为依赖关系，在程序学习任务的零样本评估中大幅超越全监督方法（EgoProceL 上 F1 提升 +12.5%），在视频-文本对齐基准上也达到了 SOTA。

**[Imbalance in Balance: Online Concept Balancing in Generation Models](imbalance_in_balance_online_concept_balancing_in_generation_models.md)**

:   通过精心设计的因果实验揭示了数据分布（而非模型规模或数据量）是扩散模型概念组合能力的决定性因素，并提出 IMBA Loss——一种在线的、概念级别的均衡损失函数，通过条件与无条件分布差异（IMBA 距离）自适应调整 token 级损失权重，只需几行代码修改即可显著提升模型的多概念生成能力。

**[InterSyn: Interleaved Learning for Dynamic Motion Synthesis in the Wild](intersyn_interleaved_learning_for_dynamic_motion_synthesis_in_the_wild.md)**

:   提出 InterSyn 框架，通过交错学习策略（Interleaved Learning）将单人与多人动作在统一序列中联合建模，配合相对协调精修（REC）模块，生成更自然、更协调的人体交互动作，在 InterHuman 测试集上 FID 较 FreeMotion 降低 6.1%，R Precision Top-1 提升 2.8%。

**[Lay2Story: Extending Diffusion Transformers for Layout-Togglable Story Generation](lay2story_extending_diffusion_transformers_for_layout-togglable_story_generation.md)**

:   Lay2Story 提出布局可切换的故事生成任务，构建了超 100 万张高分辨率图像的 Lay2Story-1M 数据集，并基于 DiT 架构设计全局-主体双分支框架，在一致性、语义相关性和美学质量上全面超越现有方法。

**[Neural Multi-View Self-Calibrated Photometric Stereo without Photometric Stereo Cues](neural_multi-view_self-calibrated_photometric_stereo_without_photometric_stereo_.md)**

:   提出一种端到端的神经逆渲染框架，从多视图变化光照图像中联合恢复几何、空间变化反射率和光照参数，无需光源标定或中间光度立体线索（如法线图），超越了现有的分阶段 MVPS 方法。

**[ODP-Bench: Benchmarking Out-of-Distribution Performance Prediction](odp-bench_benchmarking_out-of-distribution_performance_prediction.md)**

:   构建了首个全面的OOD性能预测基准ODP-Bench，涵盖29个OOD数据集、10种预测算法和1,444个预训练模型，揭示现有算法在合成corruption上表现较好但在自然分布偏移上普遍失效的关键发现。

**[OmniDiff: A Comprehensive Benchmark for Fine-grained Image Difference Captioning](omnidiff_a_comprehensive_benchmark_for_fine-grained_image_difference_captioning.md)**

:   提出包含324个多样场景（真实+3D合成）的细粒度图像差异描述数据集 OmniDiff，并设计即插即用的多尺度差异感知（MDP）模块嵌入 MLLM 构建 M3Diff 模型，在 OmniDiff 及多个公开基准上取得 SOTA。

**[On the Robustness Tradeoff in Fine-Tuning](on_the_robustness_tradeoff_in_fine-tuning.md)**

:   首次系统研究微调过程中对抗鲁棒性与准确率的权衡关系，在231个模型、7种微调策略和6个数据集上揭示：(1)微调初期鲁棒性先升后降；(2)不同PEFT策略和任务复杂度导致不同的Pareto前沿；(3)OOD鲁棒性不存在类似权衡而是紧跟准确率变化。

**[PHATNet: A Physics-guided Haze Transfer Network for Domain-adaptive Real-world Image Dehazing](phatnet_a_physics-guided_haze_transfer_network_for_domain-adaptive_real-world_im.md)**

:   提出物理引导的雾迁移网络PHATNet，通过将大气散射模型（ASM）扩展到潜空间来解耦和迁移雾模式，生成域自适应的微调数据集，使去雾模型在测试时有效适应未见过的真实世界雾场景。

**[Rethinking Few Shot CLIP Benchmarks: A Critical Analysis in the Inductive Setting](rethinking_few_shot_clip_benchmarks_a_critical_analysis_in_the_inductive_setting.md)**

:   指出现有 CLIP 少样本分类基准因 CLIP 预训练时已见过测试数据集而实际是"部分转导设置"，提出基于 unlearning 的归纳基准评估方案，并设计了一种在新基准下稳定 SOTA 的少样本分类方法。

**[SketchSplat: 3D Edge Reconstruction via Differentiable Multi-view Sketch Splatting](sketchsplat_3d_edge_reconstruction_via_differentiable_multi-view_sketch_splattin.md)**

:   提出 SketchSplat，将 3D 边缘表示为参数化 sketch（直线+Bézier曲线），通过从 sketch 采样高斯点进行可微渲染来直接优化边缘参数，同时提出自适应拓扑控制和改进的 2D 边缘检测器，在 CAD 数据集上实现 SOTA 的准确性、完整性和紧凑性。

**[Spectral Sensitivity Estimation with an Uncalibrated Diffraction Grating](spectral_sensitivity_estimation_with_an_uncalibrated_diffraction_grating.md)**

:   提出一种使用未标定衍射光栅片估计相机光谱灵敏度的实用方法，通过联合估计光谱灵敏度和光栅效率，仅需一次已知光谱光源拍摄即可获得准确的闭式解，性能显著优于传统色卡方法且设备成本不到5美元。

**[StreamMind: Unlocking Full Frame Rate Streaming Video Dialogue through Event-Gated Cognition](streammind_unlocking_full_frame_rate_streaming_video_dialogue_through_event-gate.md)**

:   StreamMind 提出"事件门控 LLM 调用"范式替代现有的"逐帧 LLM 调用"，通过在视频编码器和 LLM 之间插入认知门控网络（Cognition Gate），仅在查询相关事件发生时才调用 LLM，配合基于状态空间方法的事件保持特征提取器（EPFE）实现常量感知成本，在单张 A100 上达到 **100 fps** 的流式视频处理速度。

**[Supercharging Floorplan Localization with Semantic Rays](supercharging_floorplan_localization_with_semantic_rays.md)**

:   提出一种语义感知的平面图定位框架，将语义光线预测与深度光线融合为结构-语义概率体，配合由粗到细策略，在两个标准数据集上实现了2-3倍的性能提升。

**[SVTRv2: CTC Beats Encoder-Decoder Models in Scene Text Recognition](svtrv2_ctc_beats_encoder-decoder_models_in_scene_text_recognition.md)**

:   提出 SVTRv2，通过多尺寸resize策略（MSR）、特征重排模块（FRM）和语义引导模块（SGM）三大设计，让 CTC 模型首次在多场景基准上全面超越编码器-解码器方法，同时保持推理速度优势。
