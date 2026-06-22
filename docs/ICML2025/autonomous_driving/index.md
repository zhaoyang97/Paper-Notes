---
title: >-
  ICML2025 自动驾驶论文汇总 · 10篇论文解读
description: >-
  10篇ICML2025的自动驾驶方向论文解读，涵盖点云、强化学习、Agent等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2025"
  - "自动驾驶"
  - "论文解读"
  - "论文笔记"
  - "点云"
  - "强化学习"
  - "Agent"
item_list:
  - u: "dont_be_so_negative_score-based_generative_modeling_with_oracle-assisted_guidanc/"
    t: "Don't be so Negative! Score-based Generative Modeling with Oracle-assisted Guidance"
  - u: "drivegpt_scaling_autoregressive_behavior_models_for_driving/"
    t: "DriveGPT: Scaling Autoregressive Behavior Models for Driving"
  - u: "geometry-to-image_synthesis-driven_generative_point_cloud_registration/"
    t: "Geometry-to-Image Synthesis-Driven Generative Point Cloud Registration"
  - u: "goirl_graph-oriented_inverse_reinforcement_learning_for_multimodal_trajectory_pr/"
    t: "GoIRL: Graph-Oriented Inverse Reinforcement Learning for Multimodal Trajectory Prediction"
  - u: "hierarchical_and_collaborative_llm-based_control_for_multi-uav_motion_and_commun/"
    t: "Hierarchical and Collaborative LLM-Based Control for Multi-UAV Motion and Communication in Integrated Terrestrial and Non-Terrestrial Networks"
  - u: "hybrid_quantum-classical_multi-agent_pathfinding/"
    t: "Hybrid Quantum-Classical Multi-Agent Pathfinding"
  - u: "infocons_identifying_interpretable_critical_concepts_in_point_clouds_via_informa/"
    t: "InfoCons: Identifying Interpretable Critical Concepts in Point Clouds via Information Theory"
  - u: "r3dm_enabling_role_discovery_and_diversity_through_dynamics_models_in_multi-agen/"
    t: "R3DM: Enabling Role Discovery and Diversity Through Dynamics Models in Multi-agent Reinforcement Learning"
  - u: "safemap_robust_hd_map_construction_from_incomplete_observations/"
    t: "SafeMap: Robust HD Map Construction from Incomplete Observations"
  - u: "sphinx_structural_prediction_using_hypergraph_inference_network/"
    t: "SPHINX: Structural Prediction using Hypergraph Inference Network"
item_total: 10
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🚗 自动驾驶

**🧪 ICML2025** · **10** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (157)](../../CVPR2026/autonomous_driving/index.md) · [🔬 ICLR2026 (50)](../../ICLR2026/autonomous_driving/index.md) · [🧪 ICML2026 (8)](../../ICML2026/autonomous_driving/index.md) · [🤖 AAAI2026 (56)](../../AAAI2026/autonomous_driving/index.md) · [🧠 NeurIPS2025 (47)](../../NeurIPS2025/autonomous_driving/index.md) · [📹 ICCV2025 (91)](../../ICCV2025/autonomous_driving/index.md)

🔥 **高频主题：** 点云 ×2 · 强化学习 ×2 · Agent ×2

**[Don't be so Negative! Score-based Generative Modeling with Oracle-assisted Guidance](dont_be_so_negative_score-based_generative_modeling_with_oracle-assisted_guidanc.md)**

:   提出 Gen-neG 方法，通过迭代地在扩散模型的合成数据上训练贝叶斯最优分类器并用其引导采样，将生成分布从约束违规区域引导至正支撑域。关键创新在于正确处理类先验概率的重要性采样，交通场景生成中碰撞+越界率从 29.3% 降至 5.6%。

**[DriveGPT: Scaling Autoregressive Behavior Models for Driving](drivegpt_scaling_autoregressive_behavior_models_for_driving.md)**

:   提出 DriveGPT，一个 1.4B 参数的自回归 Transformer 驾驶行为模型，在 1.2 亿真实驾驶片段上训练（比现有最大数据集多 50x），首次系统建立驾驶行为建模的数据/模型/计算缩放定律，验证数据是性能瓶颈，在规划和 WOMD 预测任务上超越 SOTA。

**[Geometry-to-Image Synthesis-Driven Generative Point Cloud Registration](geometry-to-image_synthesis-driven_generative_point_cloud_registration.md)**

:   提出 Generative Point Cloud Registration 新范式，设计 DepthMatch-ControlNet 和 LiDARMatch-ControlNet 两个配准专用可控 2D 生成模型，从纯几何点云对生成跨视图一致的 RGB 图像对，通过几何-颜色特征融合即插即用地提升现有 3D 配准方法，在 3DMatch/ScanNet/Dur360BEV 上验证有效。

**[GoIRL: Graph-Oriented Inverse Reinforcement Learning for Multimodal Trajectory Prediction](goirl_graph-oriented_inverse_reinforcement_learning_for_multimodal_trajectory_pr.md)**

:   首次将最大熵逆强化学习框架与向量化场景表示相融合，提出 GoIRL 轨迹预测框架：通过可学习的 Feature Adaptor 将图特征聚合到网格空间以适配 IRL，再用层级参数化轨迹生成器（Bézier曲线+精细化模块）和 MCMC 概率融合机制实现多模态轨迹预测，在 Argoverse 和 nuScenes 上达到 SOTA 并展现出相比监督模型显著更强的泛化能力。

**[Hierarchical and Collaborative LLM-Based Control for Multi-UAV Motion and Communication in Integrated Terrestrial and Non-Terrestrial Networks](hierarchical_and_collaborative_llm-based_control_for_multi-uav_motion_and_commun.md)**

:   提出一种基于 LLM 的层次化协作控制框架，通过 HAPS 端部署的元控制器 LLM 和 UAV 端部署的边缘控制器 LLM 的双层协同，实现多 UAV 在 3D 空中高速公路场景下的运动规划与通信接入联合优化。

**[Hybrid Quantum-Classical Multi-Agent Pathfinding](hybrid_quantum-classical_multi-agent_pathfinding.md)**

:   提出首个最优混合量子-经典MAPF算法QP和QCP，将MAPF的路径选择问题转化为可在量子硬件上求解的QUBO子问题，通过冲突图+列生成框架实现理论最优性，在真实量子硬件上验证可行性。

**[InfoCons: Identifying Interpretable Critical Concepts in Point Clouds via Information Theory](infocons_identifying_interpretable_critical_concepts_in_point_clouds_via_informa.md)**

:   提出 InfoCons 框架，将信息瓶颈（IB）原理应用于点云模型解释——通过学习一个注意力瓶颈网络来分解点云为不同重要性的 3D 概念，引入可学习的无偏先验替代固定先验，在保证对模型预测忠实（faithfulness）的同时生成概念连贯（conceptual cohesion）的解释。

**[R3DM: Enabling Role Discovery and Diversity Through Dynamics Models in Multi-agent Reinforcement Learning](r3dm_enabling_role_discovery_and_diversity_through_dynamics_models_in_multi-agen.md)**

:   提出 R3DM 框架，通过最大化智能体角色、历史轨迹与未来预期行为之间的互信息，利用动力学模型驱动的内在奖励实现角色多样性与协调性的平衡，在 SMAC/SMACv2 环境中将胜率提升最高 20%。

**[SafeMap: Robust HD Map Construction from Incomplete Observations](safemap_robust_hd_map_construction_from_incomplete_observations.md)**

:   SafeMap 提出了一个即插即用的鲁棒高精地图构建框架，通过高斯采样视角重建（G-PVR）和蒸馏式 BEV 校正（D-BEVC）两个模块，在相机视角缺失的不完整观测条件下仍能准确构建矢量化高精地图。

**[SPHINX: Structural Prediction using Hypergraph Inference Network](sphinx_structural_prediction_using_hypergraph_inference_network.md)**

:   提出SPHINX无监督超图推断模型——将超边发现建模为序列化软聚类问题，用k-subset可微采样产生离散稀疏超图结构，可插入任意超图神经网络，在合成数据上超图重建达90%重叠率、在NBA轨迹预测和3D物体分类上超越现有方法。
