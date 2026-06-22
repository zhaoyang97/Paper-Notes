---
title: >-
  ICML2026 视频理解论文汇总 · 17篇论文解读
description: >-
  17篇ICML2026的视频理解方向论文解读，涵盖目标跟踪、推理、语音、压缩/编码、LLM、异常检测等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2026"
  - "视频理解"
  - "论文解读"
  - "论文笔记"
  - "目标跟踪"
  - "推理"
  - "语音"
  - "压缩/编码"
  - "LLM"
  - "异常检测"
item_list:
  - u: "avtrack_audio-visual_tracking_in_human-centric_complex_scenes/"
    t: "AVTrack: Audio-Visual Tracking in Human-centric Complex Scenes"
  - u: "foresee-to-ground_from_predictive_temporal_perception_to_evidence-driven_reasoni/"
    t: "Foresee-to-Ground: From Predictive Temporal Perception to Evidence-Driven Reasoning"
  - u: "metaphorvu_towards_metaphorical_video_understanding/"
    t: "MetaphorVU: Towards Metaphorical Video Understanding"
  - u: "omnisift_modality-asymmetric_token_compression_for_efficient_omni-modal_large_la/"
    t: "OmniSIFT: Modality-Asymmetric Token Compression for Efficient Omni-modal Large Language Models"
  - u: "privacy-aware_video_anomaly_detection_through_orthogonal_subspace_projection/"
    t: "Privacy-Aware Video Anomaly Detection through Orthogonal Subspace Projection"
  - u: "proact-vl_a_proactive_videollm_for_real-time_ai_companions/"
    t: "ProAct-VL: A Proactive VideoLLM for Real-Time AI Companions"
  - u: "relo_reinforcement_learning_to_localize_for_visual_object_tracking/"
    t: "RELO: Reinforcement Learning to Localize for Visual Object Tracking"
  - u: "return_of_frustratingly_easy_unsupervised_video_domain_adaptation/"
    t: "Return of Frustratingly Easy Unsupervised Video Domain Adaptation"
  - u: "revisiting_uncertainty_on_evidential_learning_for_partially_relevant_video_retri/"
    t: "Revisiting Uncertainty: On Evidential Learning for Partially Relevant Video Retrieval"
  - u: "skelhcc_a_hyperbolic_clip-driven_cache_adaptation_framework_for_skeleton-based_o/"
    t: "SkelHCC: A Hyperbolic CLIP-Driven Cache Adaptation Framework for Skeleton-based One-Shot Action Recognition"
  - u: "slap_the_semantic_least_action_principle_for_variational_video-language_modeling/"
    t: "SLAP: The Semantic Least Action Principle for Variational Video-Language Modeling"
  - u: "storm_segment_track_and_object_re-localization_from_a_single_image/"
    t: "STORM: Segment, Track, and Object Re-Localization from a Single Image"
  - u: "unified_multimodal_visual_tracking_with_dual_mixture-of-experts/"
    t: "Unified Multimodal Visual Tracking with Dual Mixture-of-Experts"
  - u: "video-mtr_reinforced_multi-turn_reasoning_for_long_video_understanding/"
    t: "Video-MTR: Reinforced Multi-Turn Reasoning for Long Video Understanding"
  - u: "videoseal_mitigating_evidence_misalignment_in_agentic_long_video_understanding_b/"
    t: "VideoSEAL: Mitigating Evidence Misalignment in Agentic Long Video Understanding by Decoupling Answer Authority"
  - u: "videotemp-o3_harmonizing_temporal_grounding_and_video_understanding_in_agentic_t/"
    t: "VideoTemp-o3: Harmonizing Temporal Grounding and Video Understanding in Agentic Thinking"
  - u: "vscd_video-based_scene_change_detection_in_unaligned_scenes/"
    t: "VSCD：无对齐场景的视频场景变化检测"
item_total: 17
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📹 视频理解

**🧪 ICML2026** · **17** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (187)](../../CVPR2026/video_understanding/index.md) · [🔬 ICLR2026 (48)](../../ICLR2026/video_understanding/index.md) · [🤖 AAAI2026 (27)](../../AAAI2026/video_understanding/index.md) · [🧠 NeurIPS2025 (39)](../../NeurIPS2025/video_understanding/index.md) · [📹 ICCV2025 (56)](../../ICCV2025/video_understanding/index.md) · [🧪 ICML2025 (4)](../../ICML2025/video_understanding/index.md)

🔥 **高频主题：** 目标跟踪 ×4 · 推理 ×2

**[AVTrack: Audio-Visual Tracking in Human-centric Complex Scenes](avtrack_audio-visual_tracking_in_human-centric_complex_scenes.md)**

:   提出 AVTrack 数据集和 AVTracker 基线方法，针对复杂人体中心场景下的音视频实例分割与跟踪（AVIS）任务，通过定义 8 种挑战条件构建高难度评测基准，并设计三阶段局部-全局分治框架（ASR 分段聚合 → 局部说话人定位 → 全局身份关联），在 HOTA 指标上超越现有最优方法约 8 个百分点。

**[Foresee-to-Ground: From Predictive Temporal Perception to Evidence-Driven Reasoning](foresee-to-ground_from_predictive_temporal_perception_to_evidence-driven_reasoni.md)**

:   Foresee-to-Ground (F2G) 把视频时序定位（VTG）从直接时间戳回归重构为「识别-测量」两阶段问题——先用预测性时序感知 + 跨度证据编码器构建候选事件证据池，再用 LLM 在选中事件的约束下精确生成边界，使 Charades-STA R@0.7 提升 4.1 个点、ActivityNet 提升 6.7 个点。

**[MetaphorVU: Towards Metaphorical Video Understanding](metaphorvu_towards_metaphorical_video_understanding.md)**

:   本文提出首个隐喻视频理解基准 MetaphorVU-Bench（860 视频 + 8 类隐喻分类法）和增强方法 MetaphorBoost——通过 54K 节点 / 200K 边的隐喻知识图谱作为外部认知支架，定量揭示 MLLM 在隐喻视频上的核心瓶颈是"跨域映射缺失"而非视觉识别错误，最优模型相比人类（83.4）仍差 17 个点。

**[OmniSIFT: Modality-Asymmetric Token Compression for Efficient Omni-modal Large Language Models](omnisift_modality-asymmetric_token_compression_for_efficient_omni-modal_large_la.md)**

:   本文指出现有 Omni-LLM token 压缩方法对音频和视频"对称"处理是次优的，提出 OmniSIFT——先用时空显著性剪掉视频冗余得到"视觉锚点"，再用这些锚点引导音频选择的两阶段非对称压缩框架，仅引入 4.85M 额外参数就在 Qwen2.5-Omni-7B 上保留 25% token 时一致超过现有压缩基线甚至原模型。

**[Privacy-Aware Video Anomaly Detection through Orthogonal Subspace Projection](privacy-aware_video_anomaly_detection_through_orthogonal_subspace_projection.md)**

:   作者提出 OPL（Orthogonal Projection Layer）和加强版 G-OPL，用一个 QR 分解出来的可学习正交子空间，在视频异常检测特征空间中显式投影掉"任务无关变量"和"人脸隐私分量"，同时引入 SSC/ARD/PD/FPD 四个隐私感知指标，在保持/提升 VAD AUC 的前提下让线性 SVM 探针对面部预测的准确率显著下降。

**[ProAct-VL: A Proactive VideoLLM for Real-Time AI Companions](proact-vl_a_proactive_videollm_for_real-time_ai_companions.md)**

:   ProAct-VL 通过分块输入-输出范式 + 轻量级 FLAG 决策头 + 过渡感知损失函数，使视频大语言模型在流式输入下能自主决定**何时响应**并生成短片段评论，同时实现 ~1 秒低延迟与强主动性——在游戏解说任务上响应时机 TimeDiff 仅 1.20 秒、触发 F1 = 63.25%，全面超越 GPT-4o 等离线模型。

**[RELO: Reinforcement Learning to Localize for Visual Object Tracking](relo_reinforcement_learning_to_localize_for_visual_object_tracking.md)**

:   RELO 把视觉单目标跟踪中"哪里是目标"这件事重构成一个空间特征图上的 MDP,把每个空间位置当作 action,用 actor-critic + IoU/AUC 直接奖励替换掉传统的手工中心热图监督,并配合"先 warmup 回归 + 层对齐时序 token 传播"两个稳定化设计,在 LaSOText 上以 57.5% AUC 拿到 SOTA。

**[Return of Frustratingly Easy Unsupervised Video Domain Adaptation](return_of_frustratingly_easy_unsupervised_video_domain_adaptation.md)**

:   本文提出 MetaTrans——一个"令人沮丧地简单"的无监督视频域适应（UVDA）方法，通过双流 Transformer 的时空特征相减来解耦空间和时间域差异，仅用两个基础损失（监督 + 域对抗）即可超过 SOTA 复杂方法，并把超参搜索成本从指数级压到线性级。

**[Revisiting Uncertainty: On Evidential Learning for Partially Relevant Video Retrieval](revisiting_uncertainty_on_evidential_learning_for_partially_relevant_video_retri.md)**

:   本文针对 Partially Relevant Video Retrieval (PRVR) 中"短查询 vs 长视频"导致的查询歧义与时间稀疏监督问题，提出基于 Dirichlet 分布的层次证据学习框架 Holmes，在视频间用三重原则区分精确/多义/欠定查询并自适应校准标签，在视频内用带 dustbin 的柔性最优传输获得稠密对齐，在 ActivityNet/Charades/TVR 三个数据集上取得 SOTA。

**[SkelHCC: A Hyperbolic CLIP-Driven Cache Adaptation Framework for Skeleton-based One-Shot Action Recognition](skelhcc_a_hyperbolic_clip-driven_cache_adaptation_framework_for_skeleton-based_o.md)**

:   SkelHCC 把 CLIP 搬到 Hyperbolic 空间，显式按"关节 → 身体部分 → 全身"三粒度对齐骨骼-语言表示，并用 LLM 生成的身体部位重要性掩码做无训练的多粒度投票缓存推理，在 NTU120 单样本动作识别上比 SOTA 提升 9%，可训参数只有 0.5M。

**[SLAP: The Semantic Least Action Principle for Variational Video-Language Modeling](slap_the_semantic_least_action_principle_for_variational_video-language_modeling.md)**

:   SLAP 把"经典力学的最小作用量原理"搬到视频语义流形上，把稀疏采样视频的缺帧补全建模为 Riemannian 流形上的两点边界值问题——用语义动力学替代概率生成来强制物体持久性，在隧道遮挡测试上准确率 83.9%（超扩散模型 12 个点）且推理加速 177×。

**[STORM: Segment, Track, and Object Re-Localization from a Single Image](storm_segment_track_and_object_re-localization_from_a_single_image.md)**

:   STORM 提出"一张参考图就能跑"的 6D 位姿跟踪框架：用层级化空间融合注意力 HSFA 做参考-查询特征对齐（产出分割掩膜 + SAM3D 网格），再训一个 BCE 二分类的 Tracking Verifier，把其 logit 取负当作能量分数 $E=-g_\theta$，连续 $L=3$ 帧超阈值就触发自动重定位，从而在 LM-O / YCB-V 上把无标注 6D 跟踪精度推到接近 ground-truth 掩膜上限。

**[Unified Multimodal Visual Tracking with Dual Mixture-of-Experts](unified_multimodal_visual_tracking_with_dual_mixture-of-experts.md)**

:   OneTrackerV2 把 RGB / RGB+D / RGB+T / RGB+E / RGB+N 五种跟踪任务统一在一个网络里端到端训练，靠 Meta Merger 做模态融合、Dual MoE 把"时空匹配"与"模态融合"两类异质特征显式拆到 T-MoE 与 M-MoE，并用 dissimilarity loss + router clustering 保证它们不塌成同一子空间。

**[Video-MTR: Reinforced Multi-Turn Reasoning for Long Video Understanding](video-mtr_reinforced_multi-turn_reasoning_for_long_video_understanding.md)**

:   Video-MTR 是一个基于强化学习的**多轮推理**框架——通过**门控双层奖励机制**引导 MLLM 迭代选择关键视频片段，仅用 **8K 数据**实现长视频理解的 SOTA 性能，对标方法需要 257K~440 万样本（数据效率提升两个数量级）。

**[VideoSEAL: Mitigating Evidence Misalignment in Agentic Long Video Understanding by Decoupling Answer Authority](videoseal_mitigating_evidence_misalignment_in_agentic_long_video_understanding_b.md)**

:   VideoSEAL 发现现有 agentic 长视频 QA 系统存在「答对但没看到证据」的失配问题，并把根因归结为「coupled agent 把规划和回答权混在一起」，提出 planner-inspector 解耦框架：planner 负责长视距证据搜寻、inspector 持有独占回答权并在像素级证据充分时才放行，在 LVBench 上把准确率从 48.2% 拉到 55.1%（↑20.5%）且 LongVideoBench 从 52.2% 升至 62.0%。

**[VideoTemp-o3: Harmonizing Temporal Grounding and Video Understanding in Agentic Thinking](videotemp-o3_harmonizing_temporal_grounding_and_video_understanding_in_agentic_t.md)**

:   VideoTemp-o3 是统一的 Agent 视频理解框架——通过**冷启动 SFT 的统一掩码策略** + **可感知奖惩的 IoU 奖励**联合建模视频时间定位与问答，在长视频理解中实现高质量的多轮迭代定位与精准回答，超长视频（> 20 分钟）mIoU 15.6% 超过 Gemini-2.5-Pro 的 14.8%。

**[VSCD：无对齐场景的视频场景变化检测](vscd_video-based_scene_change_detection_in_unaligned_scenes.md)**

:   本文引入 VSCD 任务——通过查询中心的多参考模型，在无约束相机运动和强烈视点失配条件下，利用时间一致性、补丁级对应和置信度加权融合，逐像素检测两段不同时间记录的同一环境视频中的物体级变化。
