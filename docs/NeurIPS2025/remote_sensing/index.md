---
title: >-
  NeurIPS2025 遥感论文汇总 · 12篇论文解读
description: >-
  12篇NeurIPS2025的遥感方向论文解读，涵盖时序预测、遥感、强化学习等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "NeurIPS2025"
  - "遥感"
  - "论文解读"
  - "论文笔记"
  - "时序预测"
  - "强化学习"
item_list:
  - u: "c3po_cross-view_cross-modality_correspondence_by_pointmap_prediction/"
    t: "C3PO: Cross-View Cross-Modality Correspondence by Pointmap Prediction"
  - u: "chamaevit_unifying_channelaware_masked_autoencoders_and_mult/"
    t: "ChA-MAEViT: Unifying Channel-Aware Masked Autoencoders and Multi-Channel Vision Transformers for Improved Cross-Channel Learning"
  - u: "cloud4d_estimating_cloud_properties_at_a_high_spatial_and_temporal_resolution/"
    t: "Cloud4D: Estimating Cloud Properties at a High Spatial and Temporal Resolution"
  - u: "connecting_the_dots_a_machine_learning_ready_dataset_for_ionospheric_forecasting/"
    t: "Connecting the Dots: A Machine Learning Ready Dataset for Ionospheric Forecasting Models"
  - u: "ecocast_a_spatio-temporal_model_for_continual_biodiversity_and_climate_risk_fore/"
    t: "EcoCast: A Spatio-Temporal Model for Continual Biodiversity and Climate Risk Forecasting"
  - u: "geolink_empowering_remote_sensing_foundation_model_with_openstreetmap_data/"
    t: "GeoLink: Empowering Remote Sensing Foundation Model with OpenStreetMap Data"
  - u: "greenhyperspectra_a_multi-source_hyperspectral_dataset_for_global_vegetation_tra/"
    t: "GreenHyperSpectra: A Multi-Source Hyperspectral Dataset for Global Vegetation Trait Prediction"
  - u: "mass_conservation_on_rails_--_rethinking_physics-informed_learning_of_ice_flow_v/"
    t: "Mass Conservation on Rails – Rethinking Physics-Informed Learning of Ice Flow Vector Fields"
  - u: "orbitzoo_real_orbital_systems_challenges_for_reinforcement_learning/"
    t: "OrbitZoo: Real Orbital Systems Challenges for Reinforcement Learning"
  - u: "ortholoc_uav_6-dof_localization_and_calibration_using_orthographic_geodata/"
    t: "OrthoLoC: UAV 6-DoF Localization and Calibration Using Orthographic Geodata"
  - u: "rscc_a_large-scale_remote_sensing_change_caption_dataset_for_disaster_events/"
    t: "RSCC: A Large-Scale Remote Sensing Change Caption Dataset for Disaster Events"
  - u: "scaling_image_geo-localization_to_continent_level/"
    t: "Scaling Image Geo-Localization to Continent Level"
item_total: 12
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🛰️ 遥感

**🧠 NeurIPS2025** · **12** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (63)](../../CVPR2026/remote_sensing/index.md) · [🔬 ICLR2026 (11)](../../ICLR2026/remote_sensing/index.md) · [🧪 ICML2026 (3)](../../ICML2026/remote_sensing/index.md) · [🤖 AAAI2026 (7)](../../AAAI2026/remote_sensing/index.md) · [📹 ICCV2025 (11)](../../ICCV2025/remote_sensing/index.md) · [🧪 ICML2025 (7)](../../ICML2025/remote_sensing/index.md)

🔥 **高频主题：** 时序预测 ×2 · 遥感 ×2

**[C3PO: Cross-View Cross-Modality Correspondence by Pointmap Prediction](c3po_cross-view_cross-modality_correspondence_by_pointmap_prediction.md)**

:   构建了包含 90K 地面照片-平面图对（597 个场景、153M 像素级对应和 85K 相机位姿）的 C3 数据集，揭示现有对应模型在跨视角跨模态（如地面照片 vs. 平面图）场景下的局限性，通过在该数据上训练可将最佳方法的 RMSE 降低 34%。

**[ChA-MAEViT: Unifying Channel-Aware Masked Autoencoders and Multi-Channel Vision Transformers for Improved Cross-Channel Learning](chamaevit_unifying_channelaware_masked_autoencoders_and_mult.md)**

:   提出ChA-MAEViT，通过动态通道-patch联合掩码、记忆token、混合token融合和通道感知解码器四大组件增强多通道图像（MCI）的跨通道特征学习，在卫星和显微三大数据集上平均超越SOTA 3.0-21.5%。

**[Cloud4D: Estimating Cloud Properties at a High Spatial and Temporal Resolution](cloud4d_estimating_cloud_properties_at_a_high_spatial_and_temporal_resolution.md)**

:   首个基于地面多视角相机的学习框架，通过单应性引导的2D-to-3D Transformer重建四维（3D空间+时间）云液态水含量分布，在25m空间/5s时间分辨率下实现了相对雷达<10%的误差，比卫星观测提升了一个数量级的时空分辨率。

**[Connecting the Dots: A Machine Learning Ready Dataset for Ionospheric Forecasting Models](connecting_the_dots_a_machine_learning_ready_dataset_for_ionospheric_forecasting.md)**

:   作为2025 NASA Heliolab的成果，本文构建了首个全面的ML-ready电离层预测数据集，将太阳动力学观测站（SDO）极紫外辐照度嵌入、太阳风参数、行星际磁场、地磁活动指数、JPL稠密TEC全球电离层图、Madrigal稀疏TEC、太阳通量指数以及轨道力学参数等7大类异构数据源统一对齐到一致的时间-空间结构中，并在此基础上训练了包括LSTM、球面神经算子（SFNO）和GraphCast在内的多种时空预测架构，实现了对全球垂直总电子含量（vTEC）在安静和地磁活跃条件下长达12小时的自回归预测，超越了持续性基线。

**[EcoCast: A Spatio-Temporal Model for Continual Biodiversity and Climate Risk Forecasting](ecocast_a_spatio-temporal_model_for_continual_biodiversity_and_climate_risk_fore.md)**

:   提出EcoCast，融合卫星遥感（Sentinel-2）、气候再分析（ERA5）和公民科学观测（GBIF）数据的Transformer时空序列模型，通过12个月环境特征序列预测下月物种出现概率，在非洲5种鸟类分布预测上F1宏平均从Random Forest的0.31提升至0.65，并设计了基于EWC的持续学习框架以适应数据更新。

**[GeoLink: Empowering Remote Sensing Foundation Model with OpenStreetMap Data](geolink_empowering_remote_sensing_foundation_model_with_openstreetmap_data.md)**

:   GeoLink将OpenStreetMap矢量数据直接融入遥感基础模型预训练，通过异构GNN编码OSM数据并设计多粒度跨模态学习目标（区域-图像级对比 + 对象-patch级融合），在127万样本对上高效预训练后，7个分类和4个分割/变化检测benchmark全面超越现有RS FM。

**[GreenHyperSpectra: A Multi-Source Hyperspectral Dataset for Global Vegetation Trait Prediction](greenhyperspectra_a_multi-source_hyperspectral_dataset_for_global_vegetation_tra.md)**

:   GreenHyperSpectra构建了一个包含14万+多源高光谱植被样本的预训练数据集，横跨近端、航空和卫星三种平台，通过半监督和自监督方法（MAE、GAN、RTM-AE）训练的标签高效回归模型在7种植物性状预测上全面超越全监督基线，特别是在标签稀缺和分布外场景中优势显著。

**[Mass Conservation on Rails – Rethinking Physics-Informed Learning of Ice Flow Vector Fields](mass_conservation_on_rails_--_rethinking_physics-informed_learning_of_ice_flow_v.md)**

:   提出散度无关神经网络（dfNN），通过流函数的辛梯度从架构上精确保证质量守恒（散度恒为零），结合方向引导学习策略，在南极Byrd冰川冰通量插值中显著优于软约束PINNs和无约束NN。

**[OrbitZoo: Real Orbital Systems Challenges for Reinforcement Learning](orbitzoo_real_orbital_systems_challenges_for_reinforcement_learning.md)**

:   本文提出OrbitZoo，一个基于工业级Orekit轨道动力学库构建的多智能体RL环境，支持碰撞规避、霍曼转移、星座协调等真实轨道任务，通过PettingZoo接口实现标准化MARL训练，并在Starlink真实星历数据验证中达到低误差组24米RMSE（16.6小时传播）。

**[OrthoLoC: UAV 6-DoF Localization and Calibration Using Orthographic Geodata](ortholoc_uav_6-dof_localization_and_calibration_using_orthographic_geodata.md)**

:   OrthoLoC构建了首个面向正射地理数据（DOP+DSM）的大规模UAV 6-DoF定位基准数据集，包含16425张真实UAV图像覆盖德国和美国47个区域，并引入AdHoP（自适应单应性预处理）匹配改进技术，在不修改特征匹配器的情况下将匹配性能提升95%、平移误差降低63%。

**[RSCC: A Large-Scale Remote Sensing Change Caption Dataset for Disaster Events](rscc_a_large-scale_remote_sensing_change_caption_dataset_for_disaster_events.md)**

:   构建了RSCC——首个大规模灾害感知遥感变化描述数据集（62,351对灾前/灾后图像+详细变化描述），覆盖地震/洪水/野火等31个全球事件，利用QvQ-Max视觉推理模型生成高质量标注，并建立了全面的基准评测体系。

**[Scaling Image Geo-Localization to Continent Level](scaling_image_geo-localization_to_continent_level.md)**

:   混合方法结合分类学习的原型和航拍图像嵌入，在覆盖西欧43.3万平方公里上实现200m内68%+、100m内59.2%的定位率，首次在大陆规模实现此精度。
