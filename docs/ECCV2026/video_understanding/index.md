---
title: >-
  ECCV2026 视频理解论文汇总 · 12篇论文解读
description: >-
  12篇ECCV2026的视频理解方向论文解读，涵盖目标跟踪、域适应、模型压缩、语音等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ECCV2026"
  - "视频理解"
  - "论文解读"
  - "论文笔记"
  - "目标跟踪"
  - "域适应"
  - "模型压缩"
  - "语音"
item_list:
  - u: "bridging_videoqa_and_video-guided_agentic_tasks_via_generalized_keyframe_extract/"
    t: "Bridging VideoQA and Video-Guided Agentic Tasks via Generalized Keyframe Extraction"
  - u: "egoexo-con_exploring_view-invariant_video_temporal_understanding/"
    t: "EgoExo-Con: Exploring View-Invariant Video Temporal Understanding"
  - u: "hiedg_a_hierarchical_discrete_geometry-guided_framework_for_multi-animal_trackin/"
    t: "HieDG: A Hierarchical Discrete Geometry-Guided Framework for Multi-Animal Tracking"
  - u: "mass_motion-aligned_selective_scan_for_refinement_in_flow-based_video_frame_inte/"
    t: "MASS: Motion-Aligned Selective Scan for Refinement in Flow-Based Video Frame Interpolation"
  - u: "mavfusion_efficient_infrared_and_visible_video_fusion_via_motion-aware_sparse_in/"
    t: "MAVFusion: Efficient Infrared and Visible Video Fusion via Motion-Aware Sparse Interaction"
  - u: "onpoint_offline-to-online_multi-level_distillation_for_point-supervised_online_t/"
    t: "OnPoint: Offline-to-Online Multi-Level Distillation for Point-Supervised Online Temporal Action Localization"
  - u: "pa-vad_diffusion-based_pseudo-only_video_anomaly_detection_via_domain-aligned_me/"
    t: "PA-VAD: 基于扩散模型的纯伪异常视频异常检测"
  - u: "sfdatrack_generalized_source-free_domain_adaptive_tracking_under_adverse_weather/"
    t: "SFDATrack: Generalized Source-Free Domain Adaptive Tracking Under Adverse Weather Conditions"
  - u: "sink-token-aware_pruning_for_fine-grained_video_understanding_in_efficient_video/"
    t: "Sink-Token-Aware Pruning for Fine-Grained Video Understanding in Efficient Video LLMs"
  - u: "towards_long-form_spatio-temporal_video_grounding/"
    t: "Towards Long-Form Spatio-Temporal Video Grounding"
  - u: "triangular_consistency_as_a_universal_constraint_for_learning_optical_flow/"
    t: "Triangular Consistency as a Universal Constraint for Learning Optical Flow"
  - u: "visual_dubbing_pipeline_with_diffusion_models_for_video_dubbing/"
    t: "HoliDubber: Holistic Video Dubbing for Complex Acoustic Scenes via Text-Guided Audio Synthesis"
item_total: 12
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📹 视频理解

**🎞️ ECCV2026** · **12** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (187)](../../CVPR2026/video_understanding/index.md) · [🔬 ICLR2026 (48)](../../ICLR2026/video_understanding/index.md) · [🧪 ICML2026 (17)](../../ICML2026/video_understanding/index.md) · [🤖 AAAI2026 (27)](../../AAAI2026/video_understanding/index.md) · [🧠 NeurIPS2025 (39)](../../NeurIPS2025/video_understanding/index.md) · [📹 ICCV2025 (56)](../../ICCV2025/video_understanding/index.md)

🔥 **高频主题：** 目标跟踪 ×2

**[Bridging VideoQA and Video-Guided Agentic Tasks via Generalized Keyframe Extraction](bridging_videoqa_and_video-guided_agentic_tasks_via_generalized_keyframe_extract.md)**

:   本文提出 TASKER，一种将关键帧提取形式化为图搜索问题的通用算法——MLLM 同时评估任务相关性（缺什么信息）和场景动态（哪里变化大）来指导搜索方向，配合双路置信度投票决定何时终止，在 EgoSchema 和 NExT-QA 上分别以仅约 15% 的帧数超越此前最佳基线 2.0% 和 1.8%，并配套发布 VG-GUI-Bench 基准以评估模型从视频教程中学习操作步骤并迁移到 GUI agent 任务的能力。

**[EgoExo-Con: Exploring View-Invariant Video Temporal Understanding](egoexo-con_exploring_view-invariant_video_temporal_understanding.md)**

:   本文提出 EgoExo-Con 基准（1,148 对同步 ego-exo 视频 + 2,269 条人工精炼的时序查询），首次系统评估 Video-LLM 在不同视角下时序理解的一致性，发现现有模型跨视角一致性仅勉强达到单视角性能的一半；并提出 View-GRPO 强化学习框架，通过语义对齐与结构一致性双重推理奖励，显著提升跨视角时序理解的鲁棒性和一致性。

**[HieDG: A Hierarchical Discrete Geometry-Guided Framework for Multi-Animal Tracking](hiedg_a_hierarchical_discrete_geometry-guided_framework_for_multi-animal_trackin.md)**

:   HieDG 将多动物追踪中不稳定的连续几何信号（位置、尺度、速度）通过两级残差码本离散化为结构化 token，再与视觉特征融合注入 query-based 追踪器，在 AnimalTrack / BFT / BuckTales 三个动物追踪基准上取得 SOTA 关联性能（HOTA、AssA、IDF1 显著提升），并在 DanceTrack / SportsMOT 上验证了泛化性。

**[MASS: Motion-Aligned Selective Scan for Refinement in Flow-Based Video Frame Interpolation](mass_motion-aligned_selective_scan_for_refinement_in_flow-based_video_frame_inte.md)**

:   MASS 将视频帧插值中 SSM 的特征扫描从静态空间网格重新定义为沿光流轨迹的动态序列建模，通过可学习非线性路径积分和速度感知自适应扫描，在保持高效计算的同时显著提升大位移和复杂遮挡场景下的插值质量，在 SNU-FILM Extreme 子集上以 26.04 dB PSNR 超越此前最优方法 0.38 dB。

**[MAVFusion: Efficient Infrared and Visible Video Fusion via Motion-Aware Sparse Interaction](mavfusion_efficient_infrared_and_visible_video_fusion_via_motion-aware_sparse_in.md)**

:   MAVFusion 提出运动感知稀疏交互机制，将红外-可见光视频融合的跨帧跨模态交互按光流先导的运动区域检测拆分为"静态背景弱交互 + 动态区域稀疏全局注意力"两条路径，在三个基准上达到 SOTA 融合质量的同时，计算量仅为现有视频融合方案的 5% 左右。

**[OnPoint: Offline-to-Online Multi-Level Distillation for Point-Supervised Online Temporal Action Localization](onpoint_offline-to-online_multi-level_distillation_for_point-supervised_online_t.md)**

:   OnPoint 提出离线到在线的多级蒸馏框架，用仅需单帧点标注的离线教师模型生成伪段标签、帧级类激活序列（CAS）和窗口级动作预期信号，通过实例级、帧级、窗口级三层蒸馏注入严格在线的学生模型，辅以动作性校准注意力解码和锚点级原始点监督来稳定训练，在五个数据集上一致超越强基线（THUMOS 上平均 mAP 提升 4.8%，最高 +7.0%），首次打通了点监督在线时序动作定位（POTAL）这一新任务。

**[PA-VAD: 基于扩散模型的纯伪异常视频异常检测](pa-vad_diffusion-based_pseudo-only_video_anomaly_detection_via_domain-aligned_me.md)**

:   PA-VAD 提出一个无需任何真实异常视频的框架，通过 CLIP 引导的初始帧选择和 VLM 提示词精炼驱动视频扩散模型合成类感知的伪异常片段，并设计领域对齐正则化模块（DARM）抑制伪异常在特征空间中的幅度偏差，在 ShanghaiTech (98.2% AUC)、UCF-Crime (82.5%) 和 XD-Violence (95.1%) 上超越 UVAD SOTA，甚至超过部分使用真实异常视频的 WVAD 方法。

**[SFDATrack: Generalized Source-Free Domain Adaptive Tracking Under Adverse Weather Conditions](sfdatrack_generalized_source-free_domain_adaptive_tracking_under_adverse_weather.md)**

:   SFDATrack 提出首个面向恶劣天气下视觉跟踪的源数据无关域自适应框架，通过均值教师结构中的双向交互 Mamba 模块和超球面原型投影模块，在不访问任何源域数据的情况下鲁棒适应多种天气条件，在合成与真实恶劣天气跟踪基准上取得最佳性能。

**[Sink-Token-Aware Pruning for Fine-Grained Video Understanding in Efficient Video LLMs](sink-token-aware_pruning_for_fine-grained_video_understanding_in_efficient_video.md)**

:   本文发现注意力剪枝中高注意力但无语义的"sink token"是细粒度视频理解崩溃的关键障碍，提出 SToP 方法——定义跨帧 sink score 量化 token 的 sink 倾向，并分别注入空间剪枝（STSP，降低 sink token 保留优先级）和时序剪枝（STTP，提高 sink token 被剪概率），在 10% 极端保留率下将幻觉和组合推理的性能损失大幅收窄（如 FastVid 从 15.69% 降至 6.32%，VisionZip 从 16.79% 降至 6.87%）。

**[Towards Long-Form Spatio-Temporal Video Grounding](towards_long-form_spatio-temporal_video_grounding.md)**

:   本文定义并首次探索长视频时空定位（LF-STVG）问题，提出 ART-STVG——一种记忆增强的自回归Transformer架构，将视频视为流式输入逐帧处理，通过空间/时间记忆库加选择性检索策略和级联时空解码设计，在不依赖额外长视频训练数据的前提下，在 1-5 分钟级别的长视频定位任务上大幅超越现有方法。

**[Triangular Consistency as a Universal Constraint for Learning Optical Flow](triangular_consistency_as_a_universal_constraint_for_learning_optical_flow.md)**

:   本文从位移场可复合的几何第一性原理出发，提出三角一致性（triangular consistency）——一种与架构、监督类型和数据集无关的通用光流约束，通过对三帧构成的最小三角形施加组合一致性损失，在有监督、无监督和自监督域适应三种训练范式下均取得一致提升。

**[HoliDubber: Holistic Video Dubbing for Complex Acoustic Scenes via Text-Guided Audio Synthesis](visual_dubbing_pipeline_with_diffusion_models_for_video_dubbing.md)**

:   HoliDubber 提出首个通过单条文本提示词统一生成语音与背景环境音效的视频配音框架，采用 patch 级前瞻式视听跨注意力融合与自回归扩散 Transformer 架构，在保证高唇形同步性的同时实现复杂声场的端到端合成。
