---
title: >-
  ECCV2026 机器人/具身智能论文汇总 · 15篇论文解读
description: >-
  15篇ECCV2026的机器人/具身智能方向论文解读，涵盖多模态、导航、机器人等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ECCV2026"
  - "机器人/具身智能"
  - "论文解读"
  - "论文笔记"
  - "多模态"
  - "导航"
  - "机器人"
item_list:
  - u: "a_scalar_per_patch_from_pre-trained_vits_enables_fast_moving_navigation_in_the_r/"
    t: "A scalar per patch from pre-trained ViTs enables fast moving navigation in the real world"
  - u: "affogato_open-vocabulary_affordance_grounding_with_automated_data_generation_at_/"
    t: "Affogato: Open-Vocabulary Affordance Grounding with Automated Data Generation at Scale"
  - u: "autospeed_annotation-free_stage-adaptive_motion_speed_learning_for_robot_manipul/"
    t: "AutoSpeed: Annotation-Free Stage-Adaptive Motion Speed Learning for Robot Manipulation"
  - u: "e-vla_event-augmented_vision-language-action_model_for_dark_and_blurred_scenes/"
    t: "E-VLA: Event-Augmented Vision-Language-Action Model for Dark and Blurred Scenes"
  - u: "empathic_robots_exploring_the_role_of-empathy_in_vision-language_action_models/"
    t: "MuSix: Multi-scale Mixture of World Models for Embodied Agents in Evolving Environments"
  - u: "libero-safety_a_comprehensive_benchmark_for_physical_and_semantic_safety_in_visi/"
    t: "LIBERO-Safety: A Comprehensive Benchmark for Physical and Semantic Safety in Vision-Language-Action Models"
  - u: "navwm_a_unified_navigation_world_model_for_foresight-driven_planning/"
    t: "NavWM: A Unified Navigation World Model for Foresight-Driven Planning"
  - u: "policytrim_boosting_intrinsic_policy_efficiency_of_vision-language-action_models/"
    t: "PolicyTrim: Boosting Intrinsic Policy Efficiency of Vision-Language-Action Models"
  - u: "pondering_the_way_spatial-perceiving_world_action_model_for_embodied_navigation/"
    t: "Pondering the Way: Spatial-perceiving World Action Model for Embodied Navigation"
  - u: "pose_anything_anywheremodel-free_object_poses_from_arbitrary_references/"
    t: "Pose Anything Anywhere: Model-free Object Poses from Arbitrary References"
  - u: "revisiting_parameter_redundancy_in_vision-language-action_models_insights_from_v/"
    t: "Revisiting Parameter Redundancy in Vision-Language-Action Models: Insights from VLM-to-VLA Adaptation"
  - u: "roboatlas_contextual_active_slam/"
    t: "RoboAtlas: Contextual Active SLAM"
  - u: "towards_generalizable_robotic_manipulation_in_dynamic_environments/"
    t: "Towards Generalizable Robotic Manipulation in Dynamic Environments"
  - u: "vla_knows_its_limits_adaptive_execution_horizons_for_robot_policies/"
    t: "VLA Knows Its Limits: Adaptive Execution Horizons for Robot Policies"
  - u: "zr0_dense_embodied_chain_of_thought_vla/"
    t: "ZR-0: Training Vision-Language-Action Models with Dense Embodied Chain-of-Thought Supervision"
item_total: 15
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🤖 机器人/具身智能

**🎞️ ECCV2026** · **15** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (146)](../../CVPR2026/robotics/index.md) · [🔬 ICLR2026 (162)](../../ICLR2026/robotics/index.md) · [💬 ACL2026 (11)](../../ACL2026/robotics/index.md) · [🧪 ICML2026 (53)](../../ICML2026/robotics/index.md) · [🤖 AAAI2026 (30)](../../AAAI2026/robotics/index.md) · [🧠 NeurIPS2025 (75)](../../NeurIPS2025/robotics/index.md)

🔥 **高频主题：** 多模态 ×5 · 导航 ×3 · 机器人 ×3

**[A scalar per patch from pre-trained ViTs enables fast moving navigation in the real world](a_scalar_per_patch_from_pre-trained_vits_enables_fast_moving_navigation_in_the_r.md)**

:   这是一项在真实办公楼里跑了 966 次真机导航（累计 24km、0.7 m/s 快速移动）的大规模实证研究，系统对比了 DinoV2/DinoV3/VC-1/AM-Radio/Dune 等冻结视觉编码器在纯 RGB 点目标导航上的表现，并提出一种极简投影层 PAP（Pure Attention Projection）：把每个图像 patch 压成一个标量注意力值喂给下游 agent，既不掉导航性能，又让「可通行区域 / 障碍 / affordance」这类可解释热力图自发涌现。

**[Affogato: Open-Vocabulary Affordance Grounding with Automated Data Generation at Scale](affogato_open-vocabulary_affordance_grounding_with_automated_data_generation_at_.md)**

:   Affogato通过串联Gemma3、Molmo、MobileSAM三个基础模型构建全自动标注管线，从Objaverse的15万3D资产中生成了750K规模的开放词汇affordance热力图数据集（Affogato-750K），并设计跨模态统一架构Espresso-3D/2D验证了该数据集作为通用监督信号的有效性——预训练后在3D和2D affordance grounding基准上一致提升，尤其在unseen类别上增益最大。

**[AutoSpeed: Annotation-Free Stage-Adaptive Motion Speed Learning for Robot Manipulation](autospeed_annotation-free_stage-adaptive_motion_speed_learning_for_robot_manipul.md)**

:   AutoSpeed 提出一种模型无关的训练框架，在不需要阶段或速度标注的情况下，通过将不同速度下的未来轨迹作为候选优化目标、用复合代价函数（预测误差 vs 预测时域）选择最优候选，让视觉运动策略端到端地学会在简单阶段加速执行、在精细阶段减速执行，在多个仿真和真实机器人任务上同时提升成功率和执行效率。

**[E-VLA: Event-Augmented Vision-Language-Action Model for Dark and Blurred Scenes](e-vla_event-augmented_vision-language-action_model_for_dark_and_blurred_scenes.md)**

:   E-VLA 将事件相机流通过最近计数窗口转化为类帧累积表示，再以零参数叠加或 13M 分层适配器两种轻量方式融入 SmolVLA 视觉编码管线，使机器人在 20 lux 极低照度下 Pick-Place 成功率从 0% 跃升至 90%，在 1000 ms 严重运动模糊下从 0% 恢复至 25%，并在 OOD 泛化测试中仅靠 200 lux 训练数据即在 20 lux 保持 45%。

**[MuSix: Multi-scale Mixture of World Models for Embodied Agents in Evolving Environments](empathic_robots_exploring_the_role_of-empathy_in_vision-language_action_models.md)**

:   提出 MuSix 框架，通过建构水平理论（CLT）启发的经验距离驱动两阶段尺度感知路由，显式解耦世界模型的"尺度确定"与"模型选择"；同时引入尺度依赖的遗忘率和门控跨尺度知识传递，让低层即时知识快速刷新、高层抽象知识持续稳定，实现具身智能体在动态环境中的多尺度自适应演化。

**[LIBERO-Safety: A Comprehensive Benchmark for Physical and Semantic Safety in Vision-Language-Action Models](libero-safety_a_comprehensive_benchmark_for_physical_and_semantic_safety_in_visi.md)**

:   本文提出 LIBERO-Safety，一个通过参数化场景定义语言（UBDDL）程序化生成安全性关键场景、以 keypose 驱动的大规模无碰撞演示数据管线（19664 条）、以及五维安全任务分类体系（75 个任务）来系统评估 VLA 模型物理安全与语义安全的全面基准，揭示了当前 VLA 模型在泛化能力与安全执行之间存在根本性张力——高多样性训练虽能提升安全轨迹质量，但任务成功率仍受制于次优轨迹合成与语义错配。

**[NavWM: A Unified Navigation World Model for Foresight-Driven Planning](navwm_a_unified_navigation_world_model_for_foresight-driven_planning.md)**

:   NavWM 提出一个统一导航世界模型，在共享的双向 Mamba 主干上联合学习潜在世界推理（深度/语义场景抽象）、锚点式多模态轨迹预测和 Flow Matching 条件视觉生成，使得世界模型本身能充当"视觉预见"闭环规划器——对每条候选轨迹模拟未来观测并选出最优路径——在五个机器人数据集上离线生成质量（PSNR 14.17→17.34）和零样本导航成功率（44%）均达到 SOTA。

**[PolicyTrim: Boosting Intrinsic Policy Efficiency of Vision-Language-Action Models](policytrim_boosting_intrinsic_policy_efficiency_of_vision-language-action_models.md)**

:   PolicyTrim 提出两阶段 GRPO 后训练框架，通过动态执行窗口探索扩展可靠动作块长度、配合节省步数奖励与群锚定稳定性正则消除冗余物理步骤，在不改架构、不增示范数据的前提下将 VLA 机器人端到端部署速度提升最高 5.83 倍。

**[Pondering the Way: Spatial-perceiving World Action Model for Embodied Navigation](pondering_the_way_spatial-perceiving_world_action_model_for_embodied_navigation.md)**

:   SWAM 提出将视觉导航中的轨迹规划与未来观测生成合二为一，用单次扩散推理从起始/目标 RGB 图像直接联合生成中间 RGB-D 帧序列和对应的行动轨迹，摒弃了传统"采样-验证"两阶段范式，在精度、效率和零样本泛化上全面超越基线。

**[Pose Anything Anywhere: Model-free Object Poses from Arbitrary References](pose_anything_anywheremodel-free_object_poses_from_arbitrary_references.md)**

:   PANY提出统一的无模型6D位姿估计框架，将多视图几何推理与跨视图对应学习联合端到端训练，从任意数量的稀疏参考图（含可选无位姿辅助视图）中直接估计未见物体的6D位姿，无需CAD模型或预注册，在YCB-V、LM-O等基准上大幅超越此前方法。

**[Revisiting Parameter Redundancy in Vision-Language-Action Models: Insights from VLM-to-VLA Adaptation](revisiting_parameter_redundancy_in_vision-language-action_models_insights_from_v.md)**

:   本文重新审视VLA模型的参数冗余问题，发现现有"剪枝后微调恢复"的范式掩盖了对关键参数的误删；作者提出以VLM到VLA adaption过程中的参数变化量(ΔW)作为冗余判据，通过受控剪枝诊断实验揭示不同模块的ΔW信号具有强烈异质性，据此设计多模块联合剪枝方案，在OpenVLA和π0.5上实现12%-30%参数量缩减且无需任何后剪枝恢复即保持约90%原始性能。

**[RoboAtlas: Contextual Active SLAM](roboatlas_contextual_active_slam.md)**

:   RoboAtlas 提出了一种上下文感知的主动 SLAM 框架，通过上下文多臂赌博机在几何探索、全局语义地图推理和第一人称 VLM 推理三个专家策略间自适应切换，在 GOAT-Bench 上以 90.6% 的成功率达成 SOTA，并在真实 Unitree Go2 机器人上 1800+ m² 环境中实现了 100% 任务成功率。

**[Towards Generalizable Robotic Manipulation in Dynamic Environments](towards_generalizable_robotic_manipulation_in_dynamic_environments.md)**

:   针对现有 VLA 只会抓静止物体、面对运动目标就失灵的痛点，本文构建了含 11.7 万条轨迹、35 个任务的动态操作基准 DOMINO，并提出用「历史光流 + 物体级未来特征预测」武装 VLA 的 PUMA 架构，在动态任务上把成功率绝对提升 6.3%。

**[VLA Knows Its Limits: Adaptive Execution Horizons for Robot Policies](vla_knows_its_limits_adaptive_execution_horizons_for_robot_policies.md)**

:   本文发现 flow-based VLA 的执行步长（每个动作块里真正执行几步）存在一个「先升后降」的最优点，并揭示动作 self-attention 里的「首尾动作汇聚点」编码了模型的预测极限，据此提出 AutoHorizon——一种零训练、几乎零开销的测试时方法，为每个动作块动态估计执行步长。

**[ZR-0: Training Vision-Language-Action Models with Dense Embodied Chain-of-Thought Supervision](zr0_dense_embodied_chain_of_thought_vla.md)**

:   ZR-0 通过密集的 Embodied Chain-of-Thought (ECoT) 推理监督对齐跨机器人本体的表征，采用预训练 VLM (System 2) 生成结构化推理 + DiT 动作专家 (System 1) 流匹配产出的双流架构，在 60M 帧的大规模预训练语料上训练后在 LIBERO、RoboTwin 2.0、RoboCasa 及真实 xArm 上达到 SOTA。
