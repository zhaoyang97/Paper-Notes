<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎬 视频理解

**📷 CVPR2026** · 共 **15** 篇

**[A4VL: A Multi-Agent Perception-Action Alliance for Efficient Long Video Reasoning](a4vl_multiagent_long_video_reasoning.md)**

:   提出 A4VL，一个多 Agent 感知-行动探索联盟框架用于高效长视频推理——多个 VLM Agent 在多轮循环中进行查询特定的感知线索提取（找到最相关的视频片段）和行动探索（生成答案、交叉审查、达成共识或重新探索），在 5 个 VideoQA 基准上超越 18 个现有 VLM 和 10 个长视频推理专用方法，且推理延迟显著更低。

**[AutoGaze: Attend Before Attention — Efficient and Scalable Video Understanding via Autoregressive Gazing](autogaze_attend_before_attention_efficient_video.md)**

:   提出AutoGaze——在ViT/MLLM处理视频之前，用一个轻量模块自回归地选择最少的多尺度patch，减少4x-100x视觉token，加速最高19x，支持1K帧4K视频并在VideoMME达67.0%。

**[Beyond Single-Sample: Reliable Multi-Sample Distillation for Video Understanding](beyond_singlesample_reliable_multisample_distillat.md)**

:   提出 R-MSD 框架，通过每输入采样 K 个教师响应构建教师池，结合任务自适应质量匹配（封闭题质量加权、开放题均匀配对）和在线判别器对抗蒸馏，解决视频 LVLM 黑盒蒸馏中单样本监督不可靠的问题。

**[EgoPointVQA: Gesture-Based Egocentric Video Question Answering](egopointvqa_gesture_based_egocentric_video_qa.md)**

:   提出 EgoPointVQA 数据集（4000 合成 + 400 真实第一人称视频）和 HINT 方法，通过 3D 手部关键点编码为手势意图 token 并与视觉 token 交织输入 MLLM，使模型能理解用户指向手势并回答指示性问题，HINT-14B 达到 68.1% 准确率，超越 InternVL3-14B 6.6 个百分点。

**[Enhancing Accuracy of Uncertainty Estimation in Appearance-based Gaze Tracking](enhancing_accuracy_of_uncertainty_estimation_in_ap.md)**

:   提出基于等保序回归的后校准(post-hoc calibration)方法，仅用50个标定样本即可修正视线追踪模型在域偏移下的不确定性估计失准，并引入CPE(Coverage Probability Error)指标替代EUC正确评估不确定性质量——校准后CPE从8%-45%降至~5%，95%置信区间覆盖率从16%-67%提升至86%-89%。

**[FC-Track: Overlap-Aware Post-Association Correction for Online Multi-Object Tracking](fctrack_overlapaware_postassociation_correction_fo.md)**

:   提出轻量后关联校正框架 FC-Track，通过 IoA 触发的外观更新抑制和局部检测-轨迹错配重分配，将长期身份切换比例从 36.86% 降至 29.55%，同时保持 MOT17/MOT20 上的 SOTA 水平。

**[FlashMotion: Few-Step Controllable Video Generation with Trajectory Guidance](flashmotion_fewstep_controllable_video_generation.md)**

:   提出 FlashMotion 三阶段训练框架——先训轨迹 adapter、再蒸馏少步生成器、最后用扩散+对抗混合目标微调 adapter——在少步推理下实现高质量轨迹可控视频生成，并发布 FlashBench 评估基准。

**[Real-World Point Tracking with Verifier-Guided Pseudo-Labeling](realworld_point_tracking_with_verifierguided_pseud.md)**

:   提出一个可学习的Verifier元模型，通过逐帧评估多个预训练tracker预测的可靠性来生成高质量伪标签，实现合成数据到真实世界的高效域适应，在四个真实世界点跟踪基准上达到SOTA。

**[FlexHook: Rethinking Two-Stage Referring-by-Tracking in RMOT](rethinking_twostage_referringbytracking_in_referri.md)**

:   FlexHook重新激活了两阶段RBT(Referring-by-Tracking)范式：用C-Hook从backbone直接采样目标特征(替代双编码)并注入语言条件线索，用PCD(成对对应解码器)替代CLIP余弦相似度做主动对应建模，首次让两阶段方法全面超越一阶段RMOT的SOTA——Refer-KITTI-V2上HOTA从10.32(iKUN)提升到42.53，训练仅1.91小时(2×4090)。

**[SAVA-X: Ego-to-Exo Imitation Error Detection via Scene-Adaptive View Alignment and Bidirectional Cross View Fusion](savax_egotoexo_imitation_error_detection_via_scene.md)**

:   提出Align-Fuse-Detect框架SAVA-X，通过Gumbel Top-K自适应采样去冗余、场景自适应视角嵌入缩小域差距、双向交叉注意力融合互补语义，在EgoMe数据集上Mean AUPRC达22.36，超越最强baseline +13.56%。

**[Semantic Satellite Communications for Synchronized Audiovisual Reconstruction](semantic_satellite_communications_for_synchronized.md)**

:   提出LLM驱动的自适应多模态语义卫星通信系统，通过双流生成架构(V2A/A2V)+动态知识库更新+GPT-4o决策代理，实现比强制更新基线节省约50%带宽的高保真同步音视频重建。

**[Stay in your Lane: Role Specific Queries with Overlap Suppression Loss for Dense Video Captioning](stay_in_lane_role_query_dense_video_captioning.md)**

:   ROS-DVC通过为DETR-based密集视频描述设计角色专用查询初始化（分离定位和描述查询）+跨任务对比对齐损失+重叠抑制损失，在YouCook2上无需预训练即达到CIDEr 39.18的SOTA，超越使用GPT-2的DDVC。

**[StreamingTOM: Streaming Token Compression for Efficient Video Understanding](streamingtom_streaming_token_compression_video.md)**

:   针对流式视频 VLM 面临的因果性（无法访问未来帧）和累积性（token 无界增长）两个约束，提出 StreamingTOM——一个免训练、即插即用的两阶段框架，通过因果时序缩减（减少 pre-LLM prefill）和在线量化记忆（4-bit KV-cache 存储+按需检索反量化），实现 15.7× KV-cache 压缩比、较 SOTA LiveVLM 降低 1.2× 峰值内存和 2× 更快 TTFT，在离线基准平均 63.8% 和流式基准 RVS 55.8% 达到免训练方法 SOTA。

**[TrajTok: 学习轨迹Token实现更好的视频理解](trajtok_trajectory_token_video_understanding.md)**

:   提出TrajTok——首个端到端可微的轨迹视频tokenizer，通过隐式时空聚类将视频编码为物体轨迹token，无需外部分割/跟踪管线，在分类、检索和长视频QA上全面超越patch-based方法。

**[VideoChat-M1: Collaborative Policy Planning for Video Understanding via Multi-Agent Reinforcement Learning](videochatm1_collaborative_policy_planning_for_vide.md)**

:   VideoChat-M1 提出了多智能体协作策略规划（CPP）范式 + 多智能体强化学习（MARL）训练框架，让 4 个异构 VLM agent 动态生成和更新工具调用策略来理解视频，在 LongVideoBench 上超过 Gemini 2.5 Pro 3.6%，超过 GPT-4o 15.6%。
