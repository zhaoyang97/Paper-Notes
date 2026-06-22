---
title: >-
  ICCV2025 遥感论文汇总 · 11篇论文解读
description: >-
  11篇ICCV2025的遥感方向论文解读，涵盖遥感、对抗鲁棒、导航、对齐/RLHF等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICCV2025"
  - "遥感"
  - "论文解读"
  - "论文笔记"
  - "对抗鲁棒"
  - "导航"
  - "对齐/RLHF"
item_list:
  - u: "astroloc_robust_space_to_ground_image_localizer/"
    t: "AstroLoc: Robust Space to Ground Image Localizer"
  - u: "citynav_a_large-scale_dataset_for_real-world_aerial_navigation/"
    t: "CityNav: A Large-Scale Dataset for Real-World Aerial Navigation"
  - u: "geodistill_geometry-guided_self-distillation_for_weakly_supervised_cross-view_lo/"
    t: "GeoDistill: Geometry-Guided Self-Distillation for Weakly Supervised Cross-View Localization"
  - u: "geoexplorer_active_geo-localization_with_curiosity-driven_exploration/"
    t: "GeoExplorer: Active Geo-Localization with Curiosity-Driven Exploration"
  - u: "information-bottleneck_driven_binary_neural_network_for_change_detection/"
    t: "Information-Bottleneck Driven Binary Neural Network for Change Detection"
  - u: "pan-crafter_learning_modality-consistent_alignment_for_pan-sharpening/"
    t: "Pan-Crafter: Learning Modality-Consistent Alignment for Pan-Sharpening"
  - u: "rs-vheat_heat_conduction_guided_efficient_remote_sensing_foundation_model/"
    t: "RS-vHeat: Heat Conduction Guided Efficient Remote Sensing Foundation Model"
  - u: "skysense_v2_a_unified_foundation_model_for_multi-modal_remote_sensing/"
    t: "SkySense V2: A Unified Foundation Model for Multi-Modal Remote Sensing"
  - u: "smarties_spectrum-aware_multi-sensor_auto-encoder_for_remote_sensing_images/"
    t: "SMARTIES: Spectrum-Aware Multi-Sensor Auto-Encoder for Remote Sensing Images"
  - u: "towards_a_unified_copernicus_foundation_model_for_earth_vision/"
    t: "Towards a Unified Copernicus Foundation Model for Earth Vision"
  - u: "wildsat_learning_satellite_image_representations_from_wildlife_observations/"
    t: "WildSAT: Learning Satellite Image Representations from Wildlife Observations"
item_total: 11
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🛰️ 遥感

**📹 ICCV2025** · **11** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (63)](../../CVPR2026/remote_sensing/index.md) · [🔬 ICLR2026 (11)](../../ICLR2026/remote_sensing/index.md) · [🧪 ICML2026 (3)](../../ICML2026/remote_sensing/index.md) · [🤖 AAAI2026 (7)](../../AAAI2026/remote_sensing/index.md) · [🧠 NeurIPS2025 (12)](../../NeurIPS2025/remote_sensing/index.md) · [🧪 ICML2025 (7)](../../ICML2025/remote_sensing/index.md)

🔥 **高频主题：** 遥感 ×5

**[AstroLoc: Robust Space to Ground Image Localizer](astroloc_robust_space_to_ground_image_localizer.md)**

:   提出AstroLoc，首个利用30万张人工标注宇航员照片进行训练的太空对地定位模型，通过查询-卫星配对损失和无监督挖掘技术学习鲁棒的地球表面特征表征，在recall@1上平均提升35%，recall@100持续超过99%，已在实际中完成50万+照片的定位。

**[CityNav: A Large-Scale Dataset for Real-World Aerial Navigation](citynav_a_large-scale_dataset_for_real-world_aerial_navigation.md)**

:   构建了首个面向真实城市环境的大规模空中视觉语言导航数据集 CityNav（32,637 条人类演示轨迹，覆盖 4.65 km²），并提出地理语义地图（GSM）辅助表示，显著提升基线模型的导航性能。

**[GeoDistill: Geometry-Guided Self-Distillation for Weakly Supervised Cross-View Localization](geodistill_geometry-guided_self-distillation_for_weakly_supervised_cross-view_lo.md)**

:   提出GeoDistill框架，通过基于视场角（FoV）遮挡的教师-学生自蒸馏范式增强局部判别性特征学习，在弱监督条件下（仅需粗略GPS标注）实现稳健的跨视角定位，性能提升超过10%且可即插即用于不同定位框架。

**[GeoExplorer: Active Geo-Localization with Curiosity-Driven Exploration](geoexplorer_active_geo-localization_with_curiosity-driven_exploration.md)**

:   提出 GeoExplorer，一个结合目标导向和好奇心驱动内在奖励的主动地理定位（AGL）智能体，通过联合动作-状态动力学建模和好奇心探索实现更鲁棒的 UAV 搜索策略，在未知目标和环境中展现出优越的泛化能力。

**[Information-Bottleneck Driven Binary Neural Network for Change Detection](information-bottleneck_driven_binary_neural_network_for_change_detection.md)**

:   提出 BiCD，首个专为变化检测设计的二值神经网络，通过信息瓶颈（IB）原理引导的辅助目标模块提升 BNN 的特征表示能力和可分离性，在街景和遥感变化检测数据集上达到 BNN 领域的 SOTA，同时实现 30× 内存压缩和 2.5× 推理加速。

**[Pan-Crafter: Learning Modality-Consistent Alignment for Pan-Sharpening](pan-crafter_learning_modality-consistent_alignment_for_pan-sharpening.md)**

:   PAN-Crafter 提出模态一致性对齐框架，通过模态自适应重建（MARs）和跨模态对齐感知注意力（CM3A）显式处理 PAN 和 MS 图像的跨模态错位问题，在多个遥感基准数据集上达到 SOTA，且推理速度比扩散模型快 **1110×**。

**[RS-vHeat: Heat Conduction Guided Efficient Remote Sensing Foundation Model](rs-vheat_heat_conduction_guided_efficient_remote_sensing_foundation_model.md)**

:   首次将物理热传导过程引入遥感基础模型，提出 RS-vHeat，用热传导算子（HCO）替代注意力机制来建模遥感图像中的局部区域相关性，在 4 个任务 10 个数据集上取得优异性能的同时，相比注意力基线减少 84% 显存、24% FLOPs、提升 2.7 倍吞吐量。

**[SkySense V2: A Unified Foundation Model for Multi-Modal Remote Sensing](skysense_v2_a_unified_foundation_model_for_multi-modal_remote_sensing.md)**

:   本文提出SkySense V2，使用单一统一Transformer骨干网络处理高分辨率光学/多光谱/SAR三种遥感模态数据，通过自适应Patch合并、模态特异性Prompt Token和基于Query的语义聚合对比学习（QSACL）进行预训练，仅用665M参数（相比前作SkySense的1.26B）在16个数据集7种任务上平均提升1.8分。

**[SMARTIES: Spectrum-Aware Multi-Sensor Auto-Encoder for Remote Sensing Images](smarties_spectrum-aware_multi-sensor_auto-encoder_for_remote_sensing_images.md)**

:   提出 SMARTIES，一个统一的传感器无关遥感基础模型，通过光谱感知投影将异构传感器数据映射到共享空间，结合跨传感器 token 混合和掩码重建进行自监督预训练，在单模态和多模态任务上超越专用传感器模型，并可泛化到预训练未见过的传感器。

**[Towards a Unified Copernicus Foundation Model for Earth Vision](towards_a_unified_copernicus_foundation_model_for_earth_vision.md)**

:   构建了涵盖所有主要Copernicus Sentinel任务的统一地球观测基础模型体系，包括1870万对齐图像的Copernicus-Pretrain数据集、支持任意光谱/非光谱传感器的Copernicus-FM模型、以及覆盖15个层级化下游任务的Copernicus-Bench评估基准。

**[WildSAT: Learning Satellite Image Representations from Wildlife Observations](wildsat_learning_satellite_image_representations_from_wildlife_observations.md)**

:   提出 WildSAT，利用公民科学平台上的数百万地理标记野生动物观测数据，通过对比学习将卫星图像、物种位置和文本描述对齐，显著提升遥感图像表征质量，并支持零样本文本检索。
