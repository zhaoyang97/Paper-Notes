---
title: >-
  ECCV2026 VLMReasoning论文汇总 · 18篇论文解读
description: >-
  18篇ECCV2026的 VLM Reasoning 方向论文解读，涵盖推理、多模态、机器人等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ECCV2026"
  - "VLM Reasoning"
  - "论文解读"
  - "论文笔记"
  - "推理"
  - "多模态"
  - "机器人"
item_list:
  - u: "are_video_reasoning_models_ready_to_go_outside/"
    t: "ROVA: Are Video Reasoning Models Ready to Go Outside?"
  - u: "dense_reward_for_multi-view_3d_reasoning_with_global_maps_and_local_views/"
    t: "Dense Reward for Multi-View 3D Reasoning with Global Maps and Local Views"
  - u: "e-tts_a_new_embodied_test-time_scaling_framework_for_robotic_manipulation/"
    t: "E-TTS: A New Embodied Test-Time Scaling Framework for Robotic Manipulation"
  - u: "egovita_learning_to_plan_and_verify_for_egocentric_video_reasoning/"
    t: "EgoVITA: Learning to Plan and Verify for Egocentric Video Reasoning"
  - u: "gaia_a_data_flywheel_system_for_training_gui_test-time_scaling_critic_models/"
    t: "GAIA: A Data Flywheel System for Training GUI Test-Time Scaling Critic Models"
  - u: "generative_lane_topology_reasoning_via_autoregressive_model_with_geometry_prior/"
    t: "Generative Lane Topology Reasoning via Autoregressive Model with Geometry Prior"
  - u: "hylar_hybrid_latent_reasoning_with_decoupled_policy_optimization/"
    t: "HyLaR: Hybrid Latent Reasoning with Decoupled Policy Optimization"
  - u: "information-regularized_attention_for_visual-centric_reasoning/"
    t: "Information-Regularized Attention for Visual-Centric Reasoning"
  - u: "latent_visual_diffusion_reasoning_with_monte_carlo_tree_search/"
    t: "Latent Visual Diffusion Reasoning with Monte Carlo Tree Search"
  - u: "monosr_open-vocabulary_spatial_reasoning_from_monocular_images/"
    t: "MonoSR: Open-Vocabulary Spatial Reasoning on Monocular Images"
  - u: "on_test-time_scaling_for_vision-language_models/"
    t: "On Test-Time Scaling for Vision-Language Models"
  - u: "roadbench_benchmarking_mllms_on_fine-grained_spatial_understanding_and_reasoning/"
    t: "RoadBench: Benchmarking MLLMs on Fine-Grained Spatial Understanding and Reasoning under Urban Road Scenarios"
  - u: "scale_attention_head_scaling_as_a_minimal_adapter_for_spatial_reasoning_in_visio/"
    t: "ScAle: Attention Head Scaling as a Minimal Adapter for Spatial Reasoning in Vision–Language Models"
  - u: "sciir_a_large-scale_training_dataset_and_benchmark_for_scientific_image_reasonin/"
    t: "SciIR: A Large-scale Training Dataset and Benchmark for Scientific Image Reasoning Generation"
  - u: "sentry_sam2-enhanced_neighbor-aware_and_temporally_reasoned_memory_for_visual_tr/"
    t: "SENTRY: SAM2-Enhanced Neighbor-Aware and Temporally Reasoned Memory for Visual Tracking"
  - u: "towards_spatial_trace_with_reasoning_in_vision-language_models_for_robotics/"
    t: "Towards Spatial Trace with Reasoning in Vision-Language Models for Robotics"
  - u: "videosearch-r1_iterative_video_retrieval_and_reasoning_via_soft_query_refinement/"
    t: "VideoSearch-R1: Iterative Video Retrieval and Reasoning via Soft Query Refinement"
  - u: "what_if_emulative_simulation_with_world_models_for_situated_reasoning/"
    t: "What if? Emulative Simulation with World Models for Situated Reasoning"
item_total: 18
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧠 VLM Reasoning

**🎞️ ECCV2026** · **18** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (150)](../../CVPR2026/vlm_reasoning/index.md) · [🔬 ICLR2026 (112)](../../ICLR2026/vlm_reasoning/index.md) · [💬 ACL2026 (32)](../../ACL2026/vlm_reasoning/index.md) · [🧪 ICML2026 (31)](../../ICML2026/vlm_reasoning/index.md) · [🤖 AAAI2026 (10)](../../AAAI2026/vlm_reasoning/index.md) · [🧠 NeurIPS2025 (30)](../../NeurIPS2025/vlm_reasoning/index.md)

🔥 **高频主题：** 推理 ×14 · 多模态 ×3 · 机器人 ×2

**[ROVA: Are Video Reasoning Models Ready to Go Outside?](are_video_reasoning_models_ready_to_go_outside.md)**

:   ROVA 提出了一套鲁棒视频推理训练框架，通过结构化时空扰动生成、自反思难度感知在线课程、以及基于 GRPO 的双分支对齐来提升 VLM 在真实世界扰动（天气、遮挡、光照、相机抖动）下的推理鲁棒性。同时提出 PVRBench——首个系统性注入真实扰动的具身视频推理 benchmark，覆盖 9K+ 视频和 51K+ 问答对。

**[Dense Reward for Multi-View 3D Reasoning with Global Maps and Local Views](dense_reward_for_multi-view_3d_reasoning_with_global_maps_and_local_views.md)**

:   针对多视角 3D 视觉问答里 MLLM「只拿最终答案对错做稀疏监督、跨视角推理不一致、视角选择乱跳」的顽疾，本文把任务显式拆成「建全局俯视认知地图 → 按问题规划看图顺序 → 转成自我中心视角作答」三段，用冻结的 3D 视觉基础模型（VGGT+SAM3）生成的几何一致伪地图当全局奖励、用基准元数据推出的参考轨迹当局部奖励，在 GRPO 下做轨迹级优化，让一个 3B 模型在 MindCube 上从 37.8 提到 66.5。

**[E-TTS: A New Embodied Test-Time Scaling Framework for Robotic Manipulation](e-tts_a_new_embodied_test-time_scaling_framework_for_robotic_manipulation.md)**

:   E-TTS 是一个即插即用的具身测试时缩放框架，在推理时对 VLA 模型的推理过程和动作进行联合采样、历史感知验证和反馈引导迭代精炼，无需额外训练或数据即可将多种基座模型在仿真和真实场景中的成功率最高提升 33.14%。

**[EgoVITA: Learning to Plan and Verify for Egocentric Video Reasoning](egovita_learning_to_plan_and_verify_for_egocentric_video_reasoning.md)**

:   EgoVITA 将第一人称视频推理显式分解为"自我规划 + 外部视角验证"两阶段推理，用 GRPO 强化学习配合两个密集奖励信号（ACMG 预测性跨模态对齐 + 置信度验证偏好优化）训练，在仅 52k 样本下 EgoBlind 超 Qwen2.5-VL-7B +7.7 点，同时外部视角视频理解性能零退化。

**[GAIA: A Data Flywheel System for Training GUI Test-Time Scaling Critic Models](gaia_a_data_flywheel_system_for_training_gui_test-time_scaling_critic_models.md)**

:   本文提出 GAIA 数据飞轮系统，通过让真实 GUI agent 执行任务来采集正/负动作样本，训练一个直觉式（非推理式）裁判模型 ICM；ICM 在测试时以 Best-of-N 方式从 agent 采样的多个候选动作中选出概率最高的正确动作，再用这些选中的动作扩充数据飞轮训练更强的 ICM-r2，形成自我进化的良性循环。

**[Generative Lane Topology Reasoning via Autoregressive Model with Geometry Prior](generative_lane_topology_reasoning_via_autoregressive_model_with_geometry_prior.md)**

:   TopoGPT 将车道拓扑推理从"检测-关联"的判别式范式转为自回归序列生成范式，在 330 万大规模地图场景上预训练车道图的几何先验后，通过流匹配感知适配器将对齐到 BEV 图像条件，在 OpenLane-V2 上以 lane-level +6.4、point-level +11.6 的显著优势全面超越此前方法，且生成的车道图在端点几何一致性和结构完整性上大幅改善。

**[HyLaR: Hybrid Latent Reasoning with Decoupled Policy Optimization](hylar_hybrid_latent_reasoning_with_decoupled_policy_optimization.md)**

:   HyLaR 通过引入控制 token 在文本离散 token 和连续视觉潜在向量之间无缝切换，实现混合离散-连续推理；针对混合动作空间的高效强化学习，提出 DePO（解耦策略优化），利用 vMF 球面建模和解耦信任域裁剪解决标准 RL 的几何与方差不匹配问题，在细粒度感知和通用多模态理解基准上显著超越现有方法。

**[Information-Regularized Attention for Visual-Centric Reasoning](information-regularized_attention_for_visual-centric_reasoning.md)**

:   本文提出 Information-Regularized Attention (IRA)，在 VLM 的 attention 模块内对视觉 value states 施加基于信息瓶颈原理的随机正则化——通过变分推断框架注入数据依赖的层级噪声，让模型在端到端全参数微调中学会主动控制视觉信息的注入量，从而同时改善幻觉、视觉定位和灾难性遗忘问题。

**[Latent Visual Diffusion Reasoning with Monte Carlo Tree Search](latent_visual_diffusion_reasoning_with_monte_carlo_tree_search.md)**

:   提出 LVDR 框架，将细粒度技能评估中从不确定逐步收敛到确定的认知过程建模为潜空间扩散去噪，并用关键点引导的 Monte Carlo Tree Search 提取可解释的推理轨迹，在运动与手术四个数据集上取得超越 SOTA 的评分精度同时输出透明的决策依据。

**[MonoSR: Open-Vocabulary Spatial Reasoning on Monocular Images](monosr_open-vocabulary_spatial_reasoning_from_monocular_images.md)**

:   MonoSR 构建了首个大规模开放世界单目空间推理数据集（100万+ QA对，覆盖室内/室外/物体中心三大场景共98个语义类别），通过四阶段可观察性过滤机制保证每对QA均可从单张RGB图像回答，并系统性评测了7个开源/闭源VLM以及多种辅助信息对空间推理能力的影响。

**[On Test-Time Scaling for Vision-Language Models](on_test-time_scaling_for_vision-language_models.md)**

:   这是第一篇系统研究「LLM 上的测试时扩展（test-time scaling）方法能否直接搬到大型视觉语言模型（LVLM）上」的实证工作，横跨 13 个模型、9 种扩展策略、6 个 benchmark，颠覆了「小模型不受益」的旧结论——小而强的指令模型受益最大（最高约 +30%，可达到甚至超过大模型），同时揭示了「算力过量反而伤害感知任务」和「视觉信息在推理链早期就被编码进文本 token」两个关键现象。

**[RoadBench: Benchmarking MLLMs on Fine-Grained Spatial Understanding and Reasoning under Urban Road Scenarios](roadbench_benchmarking_mllms_on_fine-grained_spatial_understanding_and_reasoning.md)**

:   RoadBench提出首个以道路标线为核心的城市场景细粒度空间理解与推理基准，包含BEV卫星图和FPV车载图双视角下的8个分级任务共3040个人工核验测试用例；对20个主流MLLM的系统评估表明，现有模型在细粒度空间理解上严重不足——最佳模型在BEV车道计数上F1仅0.355，在车道方向识别上甚至不敌基于交通常识的规则基线，揭示了MLLM从"看见"到"推理"之间的巨大能力鸿沟。

**[ScAle: Attention Head Scaling as a Minimal Adapter for Spatial Reasoning in Vision–Language Models](scale_attention_head_scaling_as_a_minimal_adapter_for_spatial_reasoning_in_visio.md)**

:   ScAle提出一种极轻量的VLM空间推理适配方法：在完全冻结的backbone中，为每层每个注意力头和MLP输出学习一个bounded scalar（通过tanh约束在(1-s_max, 1+s_max)范围内），仅对last token的激活值做乘法缩放，以约1K可训练参数在SpatialEval上实现最高134.1%的相对精度提升，参数效率比LoRA高2500倍以上。

**[SciIR: A Large-scale Training Dataset and Benchmark for Scientific Image Reasoning Generation](sciir_a_large-scale_training_dataset_and_benchmark_for_scientific_image_reasonin.md)**

:   本文以皮尔士符号学三分法（Icon/Index/Symbol）为理论框架，构建了 SciIR-82k（80k+ 科学图文对 + 逆向推理链标注）和 SciIR-Bench（原子检查表驱动的细粒度评估基准），并通过微调得到 Qwen-Image-SciIR，将 SciIR-Bench 最终得分从 35% 提升至 43%，系统性地揭示了当前 T2I 模型在科学推理上的严重缺陷。

**[SENTRY: SAM2-Enhanced Neighbor-Aware and Temporally Reasoned Memory for Visual Tracking](sentry_sam2-enhanced_neighbor-aware_and_temporally_reasoned_memory_for_visual_tr.md)**

:   SENTRY是一个免训练、即插即用的"写前精炼"(refine-before-write)模块，在SAM2系列跟踪器写入记忆前用短时域循环一致性验证每个候选掩码的时序合理性，替代原有基于置信度的记忆更新机制，集成到五个强基线后在九个benchmark上一致提点，在LaSOT/LaSOText/GOT-10k/VOT20/VOT22/DiDi上达到零样本新SOTA，且SAM2-L版本在A100上仍保持32.8 FPS的实时速度。

**[Towards Spatial Trace with Reasoning in Vision-Language Models for Robotics](towards_spatial_trace_with_reasoning_in_vision-language_models_for_robotics.md)**

:   提出 RoboTracer——一个 3D 感知的 VLM，用可回归监督的 scale decoder 与可插拔几何输入的 universal spatial encoder 学会「3D 空间指代 + 度量测量」，再用带度量敏感过程奖励的 GRPO 强化微调把它拔高到多步、度量落地的空间轨迹推理，在自建 TraceSpatial-Bench 上比 Gemini-2.5-Pro 高 36 个点，并能直接接运动规划驱动 UR5、G1 人形等真实机器人。

**[VideoSearch-R1: Iterative Video Retrieval and Reasoning via Soft Query Refinement](videosearch-r1_iterative_video_retrieval_and_reasoning_via_soft_query_refinement.md)**

:   VideoSearch-R1 提出了一个将视频检索与时间定位统一在迭代交互循环中的 agentic 框架，用连续隐空间的"软查询优化"替代传统的文本级查询重写，通过 GRPO 联合优化检索与推理，在 VCMR 三项基准上达到最优。

**[What if? Emulative Simulation with World Models for Situated Reasoning](what_if_emulative_simulation_with_world_models_for_situated_reasoning.md)**

:   WanderDream 是首个大规模"模拟探索"（emulative simulation）数据集，包含 15.8K 全景轨迹视频和 158K 问答对，让 agent 无需物理移动就能通过世界模型想象前往目标情景的完整路径并回答空间"what-if"问题。
