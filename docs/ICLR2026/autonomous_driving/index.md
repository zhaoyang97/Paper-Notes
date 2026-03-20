---
search:
  exclude: true
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🚗 自动驾驶

**🔬 ICLR2026** · 共 **14** 篇

**[SMART-R1: Advancing Multi-agent Traffic Simulation via R1-Style Reinforcement Fine-Tuning](advancing_multi-agent_traffic_simulation_via_r1-style_reinforcement_fine-tuning.md)**

:   SMART-R1 首次将 R1 风格的强化微调（RFT）引入多智能体交通仿真，提出 Metric-oriented Policy Optimization (MPO) 算法和"SFT-RFT-SFT"迭代训练策略，在 WOSAC 2025 排行榜上以 0.7858 的 Realism Meta 分数取得第一名。

**[Astra: General Interactive World Model with Autoregressive Denoising](astra_general_interactive_world_model_with_autoregressive_denoising.md)**

:   提出 Astra，一个通用交互式世界模型，通过自回归去噪框架在预训练视频扩散模型上实现动作条件化的长程视频预测，引入 ACT-Adapter（动作注入）、噪声增强历史记忆（缓解视觉惯性）和 Mixture of Action Experts（统一多异构动作模态），在自动驾驶、机器人操控和场景探索等多场景上实现 SOTA 的保真度和动作跟随能力。

**[BridgeDrive: Diffusion Bridge Policy for Closed-Loop Trajectory Planning in Autonomous Driving](bridgedrive_diffusion_bridge_policy_for_closed-loop_trajectory_planning_in_auton.md)**

:   BridgeDrive 提出用扩散桥（diffusion bridge）替代截断扩散来实现锚点引导的自动驾驶轨迹规划，保证前向/反向过程的理论对称性，在 Bench2Drive 闭环评估中成功率达到 74.99%（PDM-Lite）和 89.25%（LEAD），分别超越前 SOTA 7.72% 和 2.45%。

**[DrivingGen: A Comprehensive Benchmark for Generative Video World Models in Autonomous Driving](drivinggen_a_comprehensive_benchmark_for_generative_video_world_models_in_autono.md)**

:   DrivingGen 提出首个面向自动驾驶视频世界模型的综合性基准，包含跨天气/地域/时间/复杂场景的多样化评估数据集和四维度评估指标体系（分布、质量、时序一致性、轨迹对齐），对 14 个 SOTA 模型的评测揭示了通用模型与驾驶专用模型之间的核心权衡。

**[EgoDex: Learning Dexterous Manipulation from Large-Scale Egocentric Video](egodex_learning_dexterous_manipulation_from_large-scale_egocentric_video.md)**

:   Apple 使用 Vision Pro 采集了 829 小时的第一人称视频 + 3D 手部关节追踪数据（EgoDex），覆盖 194 种桌面操作任务，并在此数据集上系统评估了模仿学习策略（BC/DDPM/FM + Transformer），为灵巧操作的扩展训练提供了迄今最大规模的数据基础。

**[MARC: Memory-Augmented RL Token Compression for Efficient Video Understanding](marc_memory-augmented_rl_token_compression_for_efficient_video_un.md)**

:   提出 MARC 框架，通过"先检索再压缩"策略——用 Visual Memory Retriever (VMR) 选出与查询最相关的视频片段，再用 Compression GRPO (C-GRPO) 将 64 帧教师模型的推理能力蒸馏到仅用 1 帧 token 的学生模型——实现视觉 token 95% 压缩，GPU 显存降低 72%，推理延迟降低 23.9%，性能几乎无损（42.20 vs 42.21）。

**[Multi-Head Low-Rank Attention (MLRA)](multi-head_low-rank_attention.md)**

:   提出 Multi-Head Low-Rank Attention (MLRA)，通过将 MLA 的单一 latent head 分解为多个可独立分片的 latent head，并对各分支注意力输出求和，实现原生 4-way 张量并行支持，在保持 SOTA 性能的同时获得 2.8× 的解码加速。

**[NeMo-map: Neural Implicit Flow Fields for Spatio-Temporal Motion Mapping](nemo-map_neural_implicit_flow_fields_for_spatio-temporal_motion_mapping.md)**

:   提出 NeMo-map——基于神经隐式函数的连续时空动态地图，将空间-时间坐标直接映射为半包裹高斯混合模型（SWGMM）参数，消除传统方法的空间离散化和时间分段限制，在真实行人追踪数据上实现更低 NLL 和更平滑的速度分布。

**[ResWorld: Temporal Residual World Model for End-to-End Autonomous Driving](resworld_temporal_residual_world_model_for_end-to-end_autonomous_driving.md)**

:   ResWorld 提出时序残差世界模型（TR-World），通过计算 BEV 场景表征的时序残差来提取动态物体信息（无需检测/跟踪），避免对静态区域的冗余建模，结合未来引导轨迹优化（FGTR）模块利用预测的未来 BEV 特征修正规划轨迹，在 nuScenes 和 NAVSIM 上达到 SOTA 规划性能。

**[SEAL: Segment Any Events with Language](segment_any_events_with_language.md)**

:   首次提出开放词汇事件实例分割（OV-EIS）任务，设计 SEAL 框架通过多模态层次语义引导（MHSG）和轻量多模态融合网络，在仅使用事件-图像对（无密集标注）的情况下，实现事件流的多粒度（实例级+部件级）语义分割，大幅领先所有基线方法且推理速度最快。

**[SiMO: Single-Modality-Operable Multimodal Collaborative Perception](simo_single-modality-operable_multimodal_collaborative_perceptio.md)**

:   提出 SiMO 框架，通过 LAMMA 融合模块和 PAFR 训练策略，首次在多智能体协同感知中实现任意模态缺失（特别是 LiDAR 失效仅有相机可用时）下仍可正常工作的多模态感知系统，类似并联电路——只要有一条通路就能工作。

**[SPACeR: Self-Play Anchoring with Centralized Reference Models](spacer_self-play_anchoring_with_centralized_reference_models.md)**

:   SPACeR 提出"类人自博弈"框架，用预训练的 tokenized 自回归运动模型作为集中式参考策略，通过对数似然奖励和 KL 散度约束引导去中心化自博弈 RL 策略向人类驾驶分布对齐，在 WOSAC 上超越纯自博弈方法，同时推理速度比模仿学习快 10 倍、参数量小 50 倍。

**[ST4VLA: Spatially Guided Training for Vision-Language-Action Models](st4vla_spatially_guided_training_for_vision-language-action_models.md)**

:   提出 ST4VLA，通过两阶段空间引导训练（spatial grounding pre-training + spatially guided action post-training），将 VLM 的空间先验显式注入 VLA 策略学习，在 SimplerEnv 上将 Google Robot 成功率从 66.1% 提升至 84.6%，WidowX 从 54.7% 提升至 73.2%，达到 SOTA。

**[Steerable Adversarial Scenario Generation through Test-Time Preference Alignment (SAGE)](steerable_adversarial_scenario_generation_through_test-time_preference_alignment.md)**

:   SAGE 将自动驾驶对抗场景生成重构为多目标偏好对齐问题，通过训练两个偏好专家模型并在推理时通过权重插值实现对抗性与真实性之间的连续可控权衡，无需重新训练即可生成从温和到激进的全谱场景，显著提升闭环训练效果。
