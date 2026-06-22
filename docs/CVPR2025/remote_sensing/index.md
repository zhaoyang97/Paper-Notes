---
title: >-
  CVPR2025 遥感论文汇总 · 11篇论文解读
description: >-
  11篇CVPR2025的遥感方向论文解读，涵盖遥感、动态场景、对话系统、目标跟踪、对抗鲁棒、少样本学习等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "CVPR2025"
  - "遥感"
  - "论文解读"
  - "论文笔记"
  - "动态场景"
  - "对话系统"
  - "目标跟踪"
  - "对抗鲁棒"
  - "少样本学习"
item_list:
  - u: "dense_dispersed_structured_light_for_hyperspectral_3d_imaging_of_dynamic_scenes/"
    t: "Dense Dispersed Structured Light for Hyperspectral 3D Imaging of Dynamic Scenes"
  - u: "disciple_learning_interpretable_programs_for_scientific_visual_discovery/"
    t: "DiSciPLE: Learning Interpretable Programs for Scientific Visual Discovery"
  - u: "earthdial_turning_multi-sensory_earth_observations_to_interactive_dialogues/"
    t: "EarthDial: Turning Multi-sensory Earth Observations to Interactive Dialogues"
  - u: "hierarchical_dual-change_collaborative_learning_for_uav_scene_change_captioning/"
    t: "Hierarchical Dual-Change Collaborative Learning for UAV Scene Change Captioning"
  - u: "joint_and_streamwise_distributed_mimo_satellite_communications_with_multi-antenn/"
    t: "Joint and Streamwise Distributed MIMO Satellite Communications with Multi-Antenna Ground Users"
  - u: "learning_occlusion-robust_vision_transformers_for_real-time_uav_tracking/"
    t: "Learning Occlusion-Robust Vision Transformers for Real-Time UAV Tracking"
  - u: "meta-learning_hyperparameters_for_parameter_efficient_fine-tuning/"
    t: "Meta-Learning Hyperparameters for Parameter Efficient Fine-Tuning"
  - u: "metaspectra_a_compact_broadband_metasurface_camera_for_snapshot_hyperspectral_im/"
    t: "MetaSpectra+: A Compact Broadband Metasurface Camera for Snapshot Hyperspectral+ Imaging"
  - u: "mfoghub_bridging_multi-regional_and_multi-satellite_data_for_global_marine_fog_d/"
    t: "MFogHub: Bridging Multi-Regional and Multi-Satellite Data for Global Marine Fog Detection and Forecasting"
  - u: "sgformer_satellite-ground_fusion_for_3d_semantic_scene_completion/"
    t: "SGFormer: Satellite-Ground Fusion for 3D Semantic Scene Completion"
  - u: "think_and_answer_me_benchmarking_and_exploring_multi-entity_reasoning_grounding_/"
    t: "Think and Answer ME: Benchmarking and Exploring Multi-Entity Reasoning Grounding in Remote Sensing"
item_total: 11
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🛰️ 遥感

**📷 CVPR2025** · **11** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (63)](../../CVPR2026/remote_sensing/index.md) · [🔬 ICLR2026 (11)](../../ICLR2026/remote_sensing/index.md) · [🧪 ICML2026 (3)](../../ICML2026/remote_sensing/index.md) · [🤖 AAAI2026 (7)](../../AAAI2026/remote_sensing/index.md) · [🧠 NeurIPS2025 (12)](../../NeurIPS2025/remote_sensing/index.md) · [📹 ICCV2025 (11)](../../ICCV2025/remote_sensing/index.md)

🔥 **高频主题：** 遥感 ×4

**[Dense Dispersed Structured Light for Hyperspectral 3D Imaging of Dynamic Scenes](dense_dispersed_structured_light_for_hyperspectral_3d_imaging_of_dynamic_scenes.md)**

:   提出 Dense Dispersed Structured Light（DDSL）方法，利用廉价衍射光栅薄膜（<\$20）+ 立体 RGB 相机 + RGB 投影仪，设计光谱复用 DDSL 图案大幅减少所需投影帧数，实现 6.6fps 实时高光谱 3D 成像，光谱分辨率 15.5nm FWHM，深度误差 4mm。

**[DiSciPLE: Learning Interpretable Programs for Scientific Visual Discovery](disciple_learning_interpretable_programs_for_scientific_visual_discovery.md)**

:   提出 DiSciPLE 框架，利用 LLM 引导的进化算法自动合成可解释的 Python 程序来分析视觉数据，在人口密度估计等科学任务上以比最近基线低 35% 的误差实现了 SOTA，且程序完全可解释。

**[EarthDial: Turning Multi-sensory Earth Observations to Interactive Dialogues](earthdial_turning_multi-sensory_earth_observations_to_interactive_dialogues.md)**

:   提出 EarthDial，一个专为地球观测 (EO) 数据设计的对话式视觉语言模型，支持多光谱 (SAR/NIR/红外)、多时序和多分辨率遥感影像的统一理解，基于 1111 万条指令微调数据集，在 44 个下游数据集上超越现有遥感 VLM。

**[Hierarchical Dual-Change Collaborative Learning for UAV Scene Change Captioning](hierarchical_dual-change_collaborative_learning_for_uav_scene_change_captioning.md)**

:   提出 UAV 场景变化描述（UAV-SCC）新任务及 HDC-CL 框架，通过动态自适应布局 Transformer 建模移动视角下的图像对重叠/非重叠区域，结合层级跨模态方向一致性校准增强视角偏移方向感知，并构建了专用基准数据集。

**[Joint and Streamwise Distributed MIMO Satellite Communications with Multi-Antenna Ground Users](joint_and_streamwise_distributed_mimo_satellite_communications_with_multi-antenn.md)**

:   研究多 LEO 卫星联合服务多天线地面用户的分布式 MIMO 下行通信，提出联合传输与流式传输两种模式：前者通过 WMMSE 迭代优化预编码器最大化和频谱效率，后者通过匈牙利算法的流-卫星关联减少前传开销，实现性能与前传负载的灵活权衡。

**[Learning Occlusion-Robust Vision Transformers for Real-Time UAV Tracking](learning_occlusion-robust_vision_transformers_for_real-time_uav_tracking.md)**

:   提出 ORTrack 框架，通过基于空间 Cox 过程的随机遮罩来学习遮挡鲁棒的 ViT 特征表征（训练时加遮罩约束、推理时零开销），并设计自适应特征蒸馏方法将大模型压缩为轻量级学生模型 ORTrack-D，在多个无人机跟踪基准上实现 SOTA 精度与实时速度的最佳平衡。

**[Meta-Learning Hyperparameters for Parameter Efficient Fine-Tuning](meta-learning_hyperparameters_for_parameter_efficient_fine-tuning.md)**

:   MetaPEFT提出了一种元学习框架，将PEFT中的离散位置选择和连续缩放因子统一为可微分的调制器（modulator），通过双层优化自动搜索最优的PEFT超参数配置，在遥感和自然图像的长尾分布适应任务上取得SOTA。

**[MetaSpectra+: A Compact Broadband Metasurface Camera for Snapshot Hyperspectral+ Imaging](metaspectra_a_compact_broadband_metasurface_camera_for_snapshot_hyperspectral_im.md)**

:   提出 MetaSpectra+，一种基于超表面-折射混合光学的紧凑多功能相机，通过双层超表面独立控制各通道色散/曝光/偏振，在约 250nm 可见光带宽内实现快照式高光谱+HDR 或高光谱+偏振联合成像，重建精度在基准数据集上达到 SOTA。

**[MFogHub: Bridging Multi-Regional and Multi-Satellite Data for Global Marine Fog Detection and Forecasting](mfoghub_bridging_multi-regional_and_multi-satellite_data_for_global_marine_fog_d.md)**

:   MFogHub 构建了首个多区域（15个沿海区域）多卫星（6颗地球同步卫星）的全球海雾检测与预测数据集，包含超过68000个高分辨率样本和11600+像素级标注，通过16个基线模型的大规模实验揭示了区域差异和卫星变化对模型泛化能力的影响。

**[SGFormer: Satellite-Ground Fusion for 3D Semantic Scene Completion](sgformer_satellite-ground_fusion_for_3d_semantic_scene_completion.md)**

:   首次将卫星影像引入 3D 语义场景补全(SSC)任务，提出双分支框架 SGFormer，通过地面视角引导的卫星特征校正和自适应融合策略，有效解决因视觉遮挡导致的场景补全不完整问题。

**[Think and Answer ME: Benchmarking and Exploring Multi-Entity Reasoning Grounding in Remote Sensing](think_and_answer_me_benchmarking_and_exploring_multi-entity_reasoning_grounding_.md)**

:   构建遥感多实体推理定位基准 ME-RSRG（首个显式标注主体-客体角色的遥感定位数据集），提出 Entity-Aware Reasoning (EAR) 框架，结合 SFT 冷启动与实体感知奖励驱动的 GRPO 优化，实现结构化推理链输出和主-客体联合定位，Qwen2.5-VL 系列在 EAR 优化后 mAcc@0.5 提升超 10%。
