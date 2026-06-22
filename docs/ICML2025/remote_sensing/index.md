---
title: >-
  ICML2025 遥感论文汇总 · 7篇论文解读
description: >-
  7篇ICML2025的遥感方向论文解读，涵盖多模态、推理、自监督学习、时序预测、遥感等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2025"
  - "遥感"
  - "论文解读"
  - "论文笔记"
  - "多模态"
  - "推理"
  - "自监督学习"
  - "时序预测"
item_list:
  - u: "causal_foundation_models_disentangling_physics_from_instrument_properties/"
    t: "Causal Foundation Models: Disentangling Physics from Instrument Properties"
  - u: "explora_parameter-efficient_extended_pre-training_to_adapt_vision_transformers_u/"
    t: "ExPLoRA: Parameter-Efficient Extended Pre-Training to Adapt Vision Transformers under Domain Shifts"
  - u: "high-resolution_live_fuel_moisture_content_lfmc_maps_for_wildfire_risk_from_mult/"
    t: "High-Resolution Live Fuel Moisture Content (LFMC) Maps for Wildfire Risk from Multimodal Earth Observation Data"
  - u: "lighthouse_fast_and_precise_distance_to_shoreline_calculations_from_anywhere_on_/"
    t: "LIGHTHOUSE: Fast and Precise Distance to Shoreline Calculations from Anywhere on Earth"
  - u: "mapeval_a_map-based_evaluation_of_geo-spatial_reasoning_in_foundation_models/"
    t: "MapEval: A Map-Based Evaluation of Geo-Spatial Reasoning in Foundation Models"
  - u: "neural_augmented_kalman_filters_for_road_network_assisted_gnss_positioning/"
    t: "Neural Augmented Kalman Filters for Road Network Assisted GNSS Positioning"
  - u: "resampling_augmentation_for_time_series_contrastive_learning_application_to_remo/"
    t: "Resampling Augmentation for Time Series Contrastive Learning: Application to Remote Sensing"
item_total: 7
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🛰️ 遥感

**🧪 ICML2025** · **7** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (63)](../../CVPR2026/remote_sensing/index.md) · [🔬 ICLR2026 (11)](../../ICLR2026/remote_sensing/index.md) · [🧪 ICML2026 (3)](../../ICML2026/remote_sensing/index.md) · [🤖 AAAI2026 (7)](../../AAAI2026/remote_sensing/index.md) · [🧠 NeurIPS2025 (12)](../../NeurIPS2025/remote_sensing/index.md) · [📹 ICCV2025 (11)](../../ICCV2025/remote_sensing/index.md)

**[Causal Foundation Models: Disentangling Physics from Instrument Properties](causal_foundation_models_disentangling_physics_from_instrument_properties.md)**

:   提出因果驱动的基础模型，通过双编码器架构和结构化对比学习从天文时间序列中解耦物理信号和仪器效应，利用自然存在的观测三元组（同一目标不同仪器/同一仪器不同目标），在低数据场景下显著优于单一潜空间方法。

**[ExPLoRA: Parameter-Efficient Extended Pre-Training to Adapt Vision Transformers under Domain Shifts](explora_parameter-efficient_extended_pre-training_to_adapt_vision_transformers_u.md)**

:   提出 ExPLoRA，通过解冻 1-2 个 ViT block 并对其余层施加 LoRA，以参数高效的方式在目标域上继续自监督预训练，在遥感等域偏移场景下以 <10% 参数量超越从头全量预训练的 SOTA。

**[High-Resolution Live Fuel Moisture Content (LFMC) Maps for Wildfire Risk from Multimodal Earth Observation Data](high-resolution_live_fuel_moisture_content_lfmc_maps_for_wildfire_risk_from_mult.md)**

:   利用预训练多模态地球观测模型 Galileo 微调生成 10 米分辨率的活体燃料含水量（LFMC）地图，相比随机初始化模型 RMSE 降低 20%+，并通过 2025 年洛杉矶野火案例验证了管线的实用性。

**[LIGHTHOUSE: Fast and Precise Distance to Shoreline Calculations from Anywhere on Earth](lighthouse_fast_and_precise_distance_to_shoreline_calculations_from_anywhere_on_.md)**

:   提出了一个全球10米分辨率的海岸线数据集和毫秒级查询库 Lighthouse，通过融合 ESA WorldCover 与 OpenStreetMap 数据，结合分层 BallTree + 球面 Voronoi 索引实现仅需1 CPU/2GB RAM的实时海岸距离查询，精度比已有数据集提升100倍以上。

**[MapEval: A Map-Based Evaluation of Geo-Spatial Reasoning in Foundation Models](mapeval_a_map-based_evaluation_of_geo-spatial_reasoning_in_foundation_models.md)**

:   提出 MapEval 基准，通过 700 道涵盖文本、API 和视觉三类任务的多选题，系统评估 30 个基础模型在地图场景下的地理空间推理能力，发现最强模型准确率不超过 67%，且所有模型落后人类表现 20% 以上。

**[Neural Augmented Kalman Filters for Road Network Assisted GNSS Positioning](neural_augmented_kalman_filters_for_road_network_assisted_gnss_positioning.md)**

:   提出用时序图神经网络（TGNN）将开源道路网络信息集成到 GNSS 卡尔曼滤波中——TGNN 在图结构上预测最可能的道路段并动态估计其不确定性，在真实城市数据中 P95 定位误差从 77.23m 降至 55.02m（降幅 29%）。

**[Resampling Augmentation for Time Series Contrastive Learning: Application to Remote Sensing](resampling_augmentation_for_time_series_contrastive_learning_application_to_remo.md)**

:   论文提出一种面向时间序列对比学习的重采样增强（resampling augmentation），通过“上采样 + 不相交子序列抽取 + 对齐回原时间轴”构造正样本对，在多项 SITS 农业分类任务上优于常见增强策略，并在 S2-Agri100 上取得领先结果。
