---
title: >-
  ICLR2026 遥感论文汇总 · 11篇论文解读
description: >-
  11篇ICLR2026的遥感方向论文解读，涵盖遥感、多模态、Agent、扩散模型、时序预测、推理等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICLR2026"
  - "遥感"
  - "论文解读"
  - "论文笔记"
  - "多模态"
  - "Agent"
  - "扩散模型"
  - "时序预测"
  - "推理"
item_list:
  - u: "earth-agent_unlocking_the_full_landscape_of_earth_observation_with_agents/"
    t: "Earth-Agent: Unlocking the Full Landscape of Earth Observation with Agents"
  - u: "mars_-_a_foundational_map_auto-regressor/"
    t: "MARS - A Foundational Map Auto-Regressor"
  - u: "measuring_the_intrinsic_dimension_of_earth_representations/"
    t: "Measuring the Intrinsic Dimension of Earth Representations"
  - u: "mora_mobility_as_the_backbone_for_geospatial_representation_learning_at_scale/"
    t: "MoRA: Mobility as the Backbone for Geospatial Representation Learning at Scale"
  - u: "object_fidelity_diffusion_for_remote_sensing_image_generation/"
    t: "Object Fidelity Diffusion for Remote Sensing Image Generation"
  - u: "satdreamer360_multiview-consistent_generation_of_ground-level_scenes_from_satell/"
    t: "SatDreamer360: Multiview-Consistent Generation of Ground-Level Scenes from Satellite Imagery"
  - u: "selvabox_a_highresolution_dataset_for_tropical_tree_crown_detection/"
    t: "SelvaBox: A high-resolution dataset for tropical tree crown detection"
  - u: "tamms_change_understanding_and_forecasting_in_satellite_image_time_series_with_t/"
    t: "TAMMs: Change Understanding and Forecasting in Satellite Image Time Series with Temporal-Aware Multimodal Models"
  - u: "task-free_adaptive_meta_black-box_optimization/"
    t: "Task-free Adaptive Meta Black-box Optimization"
  - u: "terrafm_a_scalable_foundation_model_for_unified_multisensor_earth_observation/"
    t: "TerraFM: A Scalable Foundation Model for Unified Multisensor Earth Observation"
  - u: "towards_faithful_reasoning_in_remote_sensing_a_perceptually-grounded_geospatial_/"
    t: "Towards Faithful Reasoning in Remote Sensing: A Perceptually-Grounded Geospatial Chain-of-Thought for Vision-Language Models"
item_total: 11
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🛰️ 遥感

**🔬 ICLR2026** · **11** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (63)](../../CVPR2026/remote_sensing/index.md) · [🧪 ICML2026 (3)](../../ICML2026/remote_sensing/index.md) · [🤖 AAAI2026 (7)](../../AAAI2026/remote_sensing/index.md) · [🧠 NeurIPS2025 (12)](../../NeurIPS2025/remote_sensing/index.md) · [📹 ICCV2025 (11)](../../ICCV2025/remote_sensing/index.md) · [🧪 ICML2025 (7)](../../ICML2025/remote_sensing/index.md)

🔥 **高频主题：** 遥感 ×4 · 多模态 ×2

**[Earth-Agent: Unlocking the Full Landscape of Earth Observation with Agents](earth-agent_unlocking_the_full_landscape_of_earth_observation_with_agents.md)**

:   Earth-Agent是首个基于MCP工具生态的地球观测Agent框架，统一了RGB和光谱遥感数据，通过动态调用104个专家工具实现跨模态、多步骤、定量时空推理，配套提出的Earth-Bench基准包含248个专家任务和13,729张图像，实验证明Earth-Agent远超通用Agent和遥感MLLM。

**[MARS - A Foundational Map Auto-Regressor](mars_-_a_foundational_map_auto-regressor.md)**

:   把矢量地图（点 / 折线 / 多边形）当作一种"语言"，用一个统一的视觉编码器 + 自回归解码器端到端生成道路网络与建筑轮廓，无需任何分割后处理，配套发布了迄今最大的多类地图数据集 MAP-3M（约 3M 张图）。

**[Measuring the Intrinsic Dimension of Earth Representations](measuring_the_intrinsic_dimension_of_earth_representations.md)**

:   首次系统度量地理隐式神经表示（Geographic INR）的内在维度（ID），发现256-512维嵌入的真实ID仅2-10维；冻结嵌入空间的高ID与好的下游性能正相关，而监督任务头激活空间的低ID与高性能正相关，揭示了「代表性 vs 任务对齐」的双重机制。

**[MoRA: Mobility as the Backbone for Geospatial Representation Learning at Scale](mora_mobility_as_the_backbone_for_geospatial_representation_learning_at_scale.md)**

:   MoRA 把人类移动（mobility）图当作多模态融合的"骨架锚点"，用 CLIP 式非对称对比学习把 POI、卫星影像、人口统计三种辅助模态对齐到十亿边级移动图上，在 9 个社会经济下游任务上以 128 维表征平均超越 SOTA 12.9%，并首次给出地理空间表示学习的标度律证据。

**[Object Fidelity Diffusion for Remote Sensing Image Generation](object_fidelity_diffusion_for_remote_sensing_image_generation.md)**

:   OF-Diff 用类别标签直接提取遥感目标的"形状掩码先验"来约束扩散生成，再用一个"在线蒸馏"框架把含真实图像信息的混合特征蒸馏进只依赖形状的解码器，使得推理时**不再需要真实图像参考**也能生成高保真、布局一致的遥感图，最后用 DDPO 强化微调进一步对齐真实分布，下游检测中飞机/船/车等类别 mAP 提升 4–8%。

**[SatDreamer360: Multiview-Consistent Generation of Ground-Level Scenes from Satellite Imagery](satdreamer360_multiview-consistent_generation_of_ground-level_scenes_from_satell.md)**

:   SatDreamer360 从单张卫星图像和预设地面相机轨迹出发，用三平面场景表示、逐像素射线注意力和全景极线约束时序注意力，在扩散模型中生成几何对齐且跨帧一致的 360° 地面全景序列，并在新构建的 VIGOR++ 基准上优于 Sat2Density、ControlS2S 和 EscherNet。

**[SelvaBox: A high-resolution dataset for tropical tree crown detection](selvabox_a_highresolution_dataset_for_tropical_tree_crown_detection.md)**

:   SelvaBox 构建了目前最大规模的开放热带森林高分辨率无人机 RGB 树冠检测数据集，并用统一的多分辨率检测基准证明：高分辨率输入、DINO-Swin 检测器和跨数据集训练能显著提升热带树冠检测的域内与零样本泛化表现。

**[TAMMs: Change Understanding and Forecasting in Satellite Image Time Series with Temporal-Aware Multimodal Models](tamms_change_understanding_and_forecasting_in_satellite_image_time_series_with_t.md)**

:   提出 TAMMs——首个统一框架，在单一 MLLM-扩散架构中联合执行卫星图像时间序列的时序变化描述（TCD）和未来图像预测（FSIF），通过时序适配模块（TAM）唤醒冻结 MLLM 的时序推理能力，并通过语义融合控制注入（SFCI）机制将变化理解转化为生成控制信号。

**[Task-free Adaptive Meta Black-box Optimization](task-free_adaptive_meta_black-box_optimization.md)**

:   提出 ABOM——一种无需预定义训练任务的自适应元黑盒优化器，通过将进化算子（选择、交叉、变异）参数化为可微注意力模块，在优化过程中利用自生成数据在线更新参数，在合成基准和无人机路径规划上实现零样本竞争性能。

**[TerraFM: A Scalable Foundation Model for Unified Multisensor Earth Observation](terrafm_a_scalable_foundation_model_for_unified_multisensor_earth_observation.md)**

:   TerraFM 面向多传感器地球观测数据，把 Sentinel-1 SAR 与 Sentinel-2 光学影像当作同一地点的天然增强视图，通过模态专属 patch embedding、逐位置 cross-attention 融合和面向长尾地表覆盖的 dual-centering DINO 训练，在 GEO-Bench 与 Copernicus-Bench 的分类和分割任务上取得了强泛化表现。

**[Towards Faithful Reasoning in Remote Sensing: A Perceptually-Grounded Geospatial Chain-of-Thought for Vision-Language Models](towards_faithful_reasoning_in_remote_sensing_a_perceptually-grounded_geospatial_.md)**

:   本文提出"感知接地的地理空间思维链"（Geo-CoT），让遥感 VLM 把分析过程拆成"规划→接地取证→综合"三步、每一步都用边界框把论断锚定到具体像素区域；通过构建 38 万条结构化推理数据集 Geo-CoT380k + 两阶段对齐（SFT 灌输认知结构、GRPO 精炼忠实度），训出的 RSThinker 在视觉定位、计数、检测、描述、VQA 等十余个遥感任务上大幅领先现有 SOTA。
