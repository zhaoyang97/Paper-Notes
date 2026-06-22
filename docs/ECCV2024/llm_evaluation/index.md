---
title: >-
  ECCV2024 LLM评测论文汇总 · 19篇论文解读
description: >-
  19篇ECCV2024的 LLM 评测方向论文解读，涵盖对抗鲁棒、对齐/RLHF、压缩/编码、翻译、域适应、人脸/视线等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "ECCV2024"
  - "LLM 评测"
  - "论文解读"
  - "论文笔记"
  - "对抗鲁棒"
  - "对齐/RLHF"
  - "压缩/编码"
  - "翻译"
  - "域适应"
  - "人脸/视线"
item_list:
  - u: "colormnet_a_memory-based_deep_spatial-temporal_feature_propagation_network_for_v/"
    t: "ColorMNet: A Memory-based Deep Spatial-Temporal Feature Propagation Network for Video Colorization"
  - u: "deep_cost_ray_fusion_for_sparse_depth_video_completion/"
    t: "Deep Cost Ray Fusion for Sparse Depth Video Completion"
  - u: "distribution_alignment_for_fully_test-time_adaptation_with_dynamic_online_data_s/"
    t: "Distribution Alignment for Fully Test-Time Adaptation with Dynamic Online Data Streams"
  - u: "eliminating_warping_shakes_for_unsupervised_online_video_stitching/"
    t: "Eliminating Warping Shakes for Unsupervised Online Video Stitching"
  - u: "evsign_sign_language_recognition_and_translation_with_streaming_events/"
    t: "EvSign: Sign Language Recognition and Translation with Streaming Events"
  - u: "gradient-regularized_out-of-distribution_detection/"
    t: "Gradient-Regularized Out-of-Distribution Detection"
  - u: "image-feature_weak-to-strong_consistency_an_enhanced_paradigm_for_semi-supervise/"
    t: "Image-Feature Weak-to-Strong Consistency: An Enhanced Paradigm for Semi-Supervised Learning"
  - u: "imaging_interiors_an_implicit_solution_to_electromagnetic_inverse_scattering_pro/"
    t: "Imaging Interiors: An Implicit Solution to Electromagnetic Inverse Scattering Problems"
  - u: "instance-dependent_noisy-label_learning_with_graphical_model_based_noise-rate_es/"
    t: "Instance-dependent Noisy-label Learning with Graphical Model Based Noise-rate Estimation"
  - u: "learn_from_the_learnt_source-free_active_domain_adaptation_via_contrastive_sampl/"
    t: "Learn from the Learnt: Source-Free Active Domain Adaptation via Contrastive Sampling and Visual Persistence"
  - u: "merlin_single-shot_material_estimation_and_relighting_for_photometric_stereo/"
    t: "MERLiN: Single-Shot Material Estimation and Relighting for Photometric Stereo"
  - u: "ogni-dc_robust_depth_completion_with_optimization-guided_neural_iterations/"
    t: "OGNI-DC: Robust Depth Completion with Optimization-Guided Neural Iterations"
  - u: "petface_a_large-scale_dataset_and_benchmark_for_animal_identification/"
    t: "PetFace: A Large-Scale Dataset and Benchmark for Animal Identification"
  - u: "r2-bench_benchmarking_the_robustness_of_referring_perception_models_under_pertur/"
    t: "R²-Bench: Benchmarking the Robustness of Referring Perception Models under Perturbations"
  - u: "sigma_sinkhorn-guided_masked_video_modeling/"
    t: "SIGMA: Sinkhorn-Guided Masked Video Modeling"
  - u: "spherical_linear_interpolation_and_text-anchoring_for_zero-shot_composed_image_r/"
    t: "Spherical Linear Interpolation and Text-Anchoring for Zero-shot Composed Image Retrieval"
  - u: "sync_from_the_sea_retrieving_alignable_videos_from_large-scale_datasets/"
    t: "Sync from the Sea: Retrieving Alignable Videos from Large-Scale Datasets"
  - u: "versatile_incremental_learning_towards_class_and_domain-agnostic_incremental_lea/"
    t: "Versatile Incremental Learning: Towards Class and Domain-Agnostic Incremental Learning"
  - u: "visfocus_prompt-guided_vision_encoders_for_ocr-free_dense_document_understanding/"
    t: "VisFocus: Prompt-Guided Vision Encoders for OCR-Free Dense Document Understanding"
item_total: 19
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📊 LLM 评测

**🎞️ ECCV2024** · **19** 篇论文解读

📌 **同领域跨会议浏览：** [🔬 ICLR2026 (131)](../../ICLR2026/llm_evaluation/index.md) · [💬 ACL2026 (96)](../../ACL2026/llm_evaluation/index.md) · [🧪 ICML2026 (40)](../../ICML2026/llm_evaluation/index.md) · [🤖 AAAI2026 (16)](../../AAAI2026/llm_evaluation/index.md) · [🧠 NeurIPS2025 (38)](../../NeurIPS2025/llm_evaluation/index.md) · [📹 ICCV2025 (27)](../../ICCV2025/llm_evaluation/index.md)

🔥 **高频主题：** 对抗鲁棒 ×2

**[ColorMNet: A Memory-based Deep Spatial-Temporal Feature Propagation Network for Video Colorization](colormnet_a_memory-based_deep_spatial-temporal_feature_propagation_network_for_v.md)**

:   提出 ColorMNet，一种基于记忆机制的时空特征传播网络，通过预训练大视觉模型引导的特征提取（PVGFE）、基于记忆的特征传播（MFP）和局部注意力（LA）三个模块，在显著降低 GPU 显存消耗（仅需 1.9G）的同时实现了优于 SOTA 的视频上色效果。

**[Deep Cost Ray Fusion for Sparse Depth Video Completion](deep_cost_ray_fusion_for_sparse_depth_video_completion.md)**

:   本文提出 RayFusion 框架，通过在 cost volume 上沿射线方向施加 self-attention 和 cross-attention 实现时序融合，以仅 1.15M 参数在 KITTI、VOID、ScanNetV2 三个数据集上全面超越或持平 SOTA 稀疏深度补全方法。

**[Distribution Alignment for Fully Test-Time Adaptation with Dynamic Online Data Streams](distribution_alignment_for_fully_test-time_adaptation_with_dynamic_online_data_s.md)**

:   提出分布对齐（DA）损失将测试时特征分布拉回源域分布，配合域偏移检测机制，在非 i.i.d. 动态数据流和连续域偏移场景下大幅超越现有 TTA 方法。

**[Eliminating Warping Shakes for Unsupervised Online Video Stitching](eliminating_warping_shakes_for_unsupervised_online_video_stitching.md)**

:   定义了视频拼接中的"warping shake"新问题（图像拼接扩展到视频时非重叠区域的时域抖动），提出StabStitch首个无监督在线视频拼接框架，通过拼接轨迹生成与平滑同时实现视频拼接和稳定，达到实时28.2ms/帧。

**[EvSign: Sign Language Recognition and Translation with Streaming Events](evsign_sign_language_recognition_and_translation_with_streaming_events.md)**

:   首次构建面向连续手语识别（CSLR）和手语翻译（SLT）任务的事件相机基准数据集 EvSign，并提出基于稀疏Transformer的高效框架，在仅0.34% FLOPs和44.2%参数量下达到与SOTA RGB方法可比或更优的性能。

**[Gradient-Regularized Out-of-Distribution Detection](gradient-regularized_out-of-distribution_detection.md)**

:   提出 GReg/GReg+，通过正则化 OOD 评分函数的输入梯度范数来学习评分流形的局部平滑性，并结合基于能量评分的聚类采样策略选取高信息量辅助样本，在 CIFAR 和 ImageNet OOD 检测基准上取得 SOTA。

**[Image-Feature Weak-to-Strong Consistency: An Enhanced Paradigm for Semi-Supervised Learning](image-feature_weak-to-strong_consistency_an_enhanced_paradigm_for_semi-supervise.md)**

:   本文提出 IFMatch，在传统图像级弱到强一致性范式基础上引入特征级扰动并构建三分支结构，通过置信度策略区分朴素/困难样本，在多个 SSL 基准上显著提升已有方法（如 FixMatch、FreeMatch 等）的性能。

**[Imaging Interiors: An Implicit Solution to Electromagnetic Inverse Scattering Problems](imaging_interiors_an_implicit_solution_to_electromagnetic_inverse_scattering_pro.md)**

:   提出基于隐式神经表示（INR）的电磁逆散射问题（EISP）求解方案，通过将散射体的相对介电常数建模为连续隐式表示并在前向框架中优化，有效避免了逆估计的困难和离散化导致的低分辨率问题。

**[Instance-dependent Noisy-label Learning with Graphical Model Based Noise-rate Estimation](instance-dependent_noisy-label_learning_with_graphical_model_based_noise-rate_es.md)**

:   本文提出一种基于概率图模型的噪声率估计方法，可自动估计训练集标签噪声率，并利用估计值指导样本选择策略的课程设计，可无缝集成到 DivideMix、InstanceGM 等 SOTA 噪声标签学习方法中，在合成和真实世界基准上提升其分类精度。

**[Learn from the Learnt: Source-Free Active Domain Adaptation via Contrastive Sampling and Visual Persistence](learn_from_the_learnt_source-free_active_domain_adaptation_via_contrastive_sampl.md)**

:   提出 LFTL（Learn from the Learnt）框架，通过对比主动采样（CAS）和视觉持久性引导适应（VPA）两个核心模块，在无源数据、极少量目标标注（≤5%）的条件下实现高效域适应，在 VisDA-C 上仅用 1% 标注即达到 87.4% 准确率。

**[MERLiN: Single-Shot Material Estimation and Relighting for Photometric Stereo](merlin_single-shot_material_estimation_and_relighting_for_photometric_stereo.md)**

:   提出单阶段注意力沙漏网络MERLiN，从单张图像联合估计空间变化BRDF参数并进行物理正确的重打光，首次利用重打光图像驱动光度立体方法实现单图法向估计，弥合了Shape from Shading与Photometric Stereo之间的鸿沟。

**[OGNI-DC: Robust Depth Completion with Optimization-Guided Neural Iterations](ogni-dc_robust_depth_completion_with_optimization-guided_neural_iterations.md)**

:   提出 OGNI-DC，通过"优化引导的神经迭代"（OGNI）框架，结合 ConvGRU 迭代精炼深度梯度场和可微深度积分器（DDI）来实现深度补全，同时达到 SOTA 精度和强泛化能力。

**[PetFace: A Large-Scale Dataset and Benchmark for Animal Identification](petface_a_large-scale_dataset_and_benchmark_for_animal_identification.md)**

:   构建了包含13个动物科、319个品种、257,484个个体（超100万张图像）的大规模动物面部识别数据集PetFace，并建立了已见个体重识别和未见个体验证两套基准测试，为动物非侵入式自动识别提供基础设施。

**[R²-Bench: Benchmarking the Robustness of Referring Perception Models under Perturbations](r2-bench_benchmarking_the_robustness_of_referring_perception_models_under_pertur.md)**

:   提出 R²-Bench，一个系统评估指代感知模型（RPM）在各种扰动下鲁棒性的综合基准，包含完整的扰动分类体系、通用的扰动合成工具箱和基于 LLM 的自动化评估代理 R²-Agent，覆盖五大关键任务，揭示了当前 RPM 在噪声条件下的脆弱性。

**[SIGMA: Sinkhorn-Guided Masked Video Modeling](sigma_sinkhorn-guided_masked_video_modeling.md)**

:   本文提出 SIGMA，通过引入投影网络将 masked video modeling 的重建目标从像素级升级为可学习的深层特征聚类分配，利用 Sinkhorn 算法的最优传输实施高熵正则化避免坍缩，在 10 个数据集 3 个 benchmark 上全面超越 VideoMAE 等 SOTA 方法。

**[Spherical Linear Interpolation and Text-Anchoring for Zero-shot Composed Image Retrieval](spherical_linear_interpolation_and_text-anchoring_for_zero-shot_composed_image_r.md)**

:   提出 Slerp-based ZS-CIR 方法，通过球面线性插值（Slerp）直接融合 VLP 模型的图像和文本嵌入构造组合查询表示，配合 Text-Anchored-Tuning (TAT) 用 LoRA 微调图像编码器缩小模态间隙，在 CIRR/CIRCO/FashionIQ 上达到 SOTA。

**[Sync from the Sea: Retrieving Alignable Videos from Large-Scale Datasets](sync_from_the_sea_retrieving_alignable_videos_from_large-scale_datasets.md)**

:   提出可对齐视频检索（Alignable Video Retrieval, AVR）任务，通过 DRAQ 对齐质量指标从大规模视频数据库中识别并检索出最适合与查询视频进行时序对齐的视频，同时提出特征上下文化方法提升对齐性能。

**[Versatile Incremental Learning: Towards Class and Domain-Agnostic Incremental Learning](versatile_incremental_learning_towards_class_and_domain-agnostic_incremental_lea.md)**

:   首次定义 Versatile Incremental Learning (VIL) 场景——后续任务的类别或领域增量类型未知，并提出 ICON 框架，通过 CAST 损失控制学习方向避免与历史任务冲突、IC 增量分类器动态扩展输出节点处理跨域同类覆写问题，在三个基准上全面超越现有 CIL/DIL 方法。

**[VisFocus: Prompt-Guided Vision Encoders for OCR-Free Dense Document Understanding](visfocus_prompt-guided_vision_encoders_for_ocr-free_dense_document_understanding.md)**

:   VisFocus提出了一种提示引导的视觉编码方法用于OCR-free文档理解：通过将用户提示（prompt）直接注入视觉编码器的patch merging层（ViLMA层），配合局部掩码提示建模（LMPM）预训练任务，使视觉编码器学会聚焦于与提示相关的文本区域，在多个文档VQA基准上达到同规模SOTA。
