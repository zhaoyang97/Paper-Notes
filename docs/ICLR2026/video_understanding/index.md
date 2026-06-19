---
title: >-
  ICLR2026 视频理解论文汇总 · 14篇论文解读
description: >-
  14篇ICLR2026的视频理解方向论文解读，涵盖推理、LLM、异常检测等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICLR2026"
  - "视频理解"
  - "论文解读"
  - "论文笔记"
  - "推理"
  - "LLM"
  - "异常检测"
item_list:
  - u: "a_training-free_framework_for_long_video_understanding_via_video-query-options_s/"
    t: "A Training-Free Framework for Long Video Understanding via Video-Query-Options Similarity"
  - u: "air_enabling_adaptive_iterative_and_reasoning-based_frame_selection_for_video_qu/"
    t: "A.I.R.: Adaptive, Iterative, and Reasoning-based Frame Selection For Video Question Answering"
  - u: "beyond_static_vision_scene_dynamic_field_unlocks_intuitive_physics_understanding/"
    t: "Beyond Static Vision: Scene Dynamic Field Unlocks Intuitive Physics Understanding in Multi-modal Large Language Models"
  - u: "cambrian-s_towards_spatial_supersensing_in_video/"
    t: "Cambrian-S: Towards Spatial Supersensing in Video"
  - u: "carebench_a_fine-grained_benchmark_for_video_captioning_and_retrieval/"
    t: "CaReBench: A Fine-grained Benchmark for Video Captioning and Retrieval"
  - u: "flashvid_efficient_video_large_language_models_via_training-free_tree-based_spat/"
    t: "FlashVID: Efficient Video Large Language Models via Training-free Tree-Based Spatiotemporal Token Merging"
  - u: "floc_facility_location-based_efficient_visual_token_compression_for_long_video_u/"
    t: "FLoC: Facility Location-Based Efficient Visual Token Compression for Long Video Understanding"
  - u: "from_vicious_to_virtuous_cycles_synergistic_representation_learning_for_unsuperv/"
    t: "From Vicious to Virtuous Cycles: Synergistic Representation Learning for Unsupervised Video Object-Centric Learning"
  - u: "language-guided_open-world_video_anomaly_detection_under_weak_supervision/"
    t: "Language-guided Open-world Video Anomaly Detection under Weak Supervision"
  - u: "lets_split_up_zero-shot_classifier_edits_for_fine-grained_video_understanding/"
    t: "Let's Split Up: Zero-Shot Classifier Edits for Fine-Grained Video Understanding"
  - u: "map_the_flow_revealing_hidden_pathways_of_information_in_videollms/"
    t: "Map the Flow: Revealing Hidden Pathways of Information in VideoLLMs"
  - u: "steering_and_rectifying_latent_representation_manifolds_in_frozen_multi-modal_ll/"
    t: "Steering and Rectifying Latent Representation Manifolds in Frozen Multi-Modal LLMs for Video Anomaly Detection"
  - u: "video-ktr_reinforcing_video_reasoning_via_key_token_attribution/"
    t: "Video-KTR: Reinforcing Video Reasoning via Key Token Attribution"
  - u: "videonsa_native_sparse_attention_scales_video_understanding/"
    t: "VideoNSA: Native Sparse Attention Scales Video Understanding"
item_total: 14
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📹 视频理解

**🔬 ICLR2026** · **14** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (187)](../../CVPR2026/video_understanding/index.md) · [🧪 ICML2026 (17)](../../ICML2026/video_understanding/index.md) · [🤖 AAAI2026 (27)](../../AAAI2026/video_understanding/index.md) · [🧠 NeurIPS2025 (39)](../../NeurIPS2025/video_understanding/index.md) · [📹 ICCV2025 (56)](../../ICCV2025/video_understanding/index.md) · [🧪 ICML2025 (4)](../../ICML2025/video_understanding/index.md)

🔥 **高频主题：** 推理 ×2 · LLM ×2 · 异常检测 ×2

**[A Training-Free Framework for Long Video Understanding via Video-Query-Options Similarity](a_training-free_framework_for_long_video_understanding_via_video-query-options_s.md)**

:   针对小时级长视频塞不进多模态大模型上下文的问题，本文提出一套**无需训练**的输入侧框架：用视频-文本检索模型给每个视频片段打相关性分，再据此**自适应加密采样**（AFS）、**动态分配分辨率**（DRA），并让 MLLM 自己生成候选答案融进检索 query（VQOS）来精修相关性估计，在 5 个长视频基准上把 LLaVA-Video 和 Qwen2.5-VL 平均提了 3~5 个点。

**[A.I.R.: Adaptive, Iterative, and Reasoning-based Frame Selection For Video Question Answering](air_enabling_adaptive_iterative_and_reasoning-based_frame_selection_for_video_qu.md)**

:   提出 A.I.R.，一种无需训练的自适应-迭代-推理驱动帧选择框架，通过两阶段策略（GMM 自适应初始采样 + 迭代式 VLM 精细分析）解决 VideoQA 中轻量模型（CLIP）相似度不准确和 VLM 分析成本爆炸的双重困境，在最坏情况下也仅需分析 72 帧（vs 基线 128 帧），同时显著提升多个长视频 benchmark 性能。

**[Beyond Static Vision: Scene Dynamic Field Unlocks Intuitive Physics Understanding in Multi-modal Large Language Models](beyond_static_vision_scene_dynamic_field_unlocks_intuitive_physics_understanding.md)**

:   这篇论文先用 Next Frame Selection（下一帧选择）和 Temporal Coherence Verification（时序一致性判别）两个"低层"诊断任务，揭示当前 MLLM 连流体这类连续介质的直觉物理动态都看不懂；再提出 Scene Dynamic Field（SDF）——把物理模拟器算出的粒子速度映射成蓝色梯度图当视觉提示，配合多任务微调，让 Qwen2-VL / GLM-4.1V 在流体任务上最高涨 20.7%，并能迁移到布料、沙、烟雾等未见物理域。

**[Cambrian-S: Towards Spatial Supersensing in Video](cambrian-s_towards_spatial_supersensing_in_video.md)**

:   本文提出"空间超感知（spatial supersensing）"这一从被动任务驱动转向主动世界建模的范式：先用 VSI-SUPER 基准证明暴力扩长上下文（包括 Gemini-2.5 和自训的 Cambrian-S）在任意长视频上的空间回忆与计数任务上彻底失效，再用一个自监督的"潜帧预测"头把预测误差（"惊讶"）当作控制信号去驱动记忆管理与事件分割，从而在长视频空间任务上大幅超过强商业基线。

**[CaReBench: A Fine-grained Benchmark for Video Captioning and Retrieval](carebench_a_fine-grained_benchmark_for_video_captioning_and_retrieval.md)**

:   CaReBench 用 1000 个人工标注、字幕长达 200+ 词且**显式拆成空间/时间两份**的视频，搭起一个能同时考视频细粒度字幕（captioning）和检索（retrieval）的 benchmark，配套两个新指标 ReBias 与 CapST 专门量化 VLM 的时空偏置，并顺手给出一个把字幕和检索统一进单个 MLLM 的两阶段 SFT 基线 CARE。

**[FlashVID: Efficient Video Large Language Models via Training-free Tree-Based Spatiotemporal Token Merging](flashvid_efficient_video_large_language_models_via_training-free_tree-based_spat.md)**

:   提出 FlashVID，一个免训练的视频大语言模型推理加速框架，通过树状时空 token 合并（TSTM）联合建模空间和时间冗余，仅保留 10% 的视觉 token 就能保持 LLaVA-OneVision 99.1% 的性能，并能将 Qwen2.5-VL 的输入帧数提升 10 倍。

**[FLoC: Facility Location-Based Efficient Visual Token Compression for Long Video Understanding](floc_facility_location-based_efficient_visual_token_compression_for_long_video_u.md)**

:   提出 FLoC，基于设施选址函数（facility location function）的视觉 token 压缩框架，通过子模优化在给定预算下快速选择兼具代表性和多样性的 token 子集，实现无训练、模型无关、查询无关的长视频理解 token 压缩。

**[From Vicious to Virtuous Cycles: Synergistic Representation Learning for Unsupervised Video Object-Centric Learning](from_vicious_to_virtuous_cycles_synergistic_representation_learning_for_unsuperv.md)**

:   发现 slot-based 目标中心学习中编码器（产生尖锐但有噪声的注意力图）与解码器（产生空间一致但模糊的重建掩码）之间的恶性循环，提出同步对比学习目标和 slot 正则化预热策略将其转化为良性循环，在 MOVi 和 YouTube-VIS 上大幅提升物体发现性能。

**[Language-guided Open-world Video Anomaly Detection under Weak Supervision](language-guided_open-world_video_anomaly_detection_under_weak_supervision.md)**

:   提出语言引导的开放世界视频异常检测范式LaGoVAD，通过将异常定义建模为随机变量并以自然语言形式输入，结合动态视频合成和对比学习正则化策略，在七个数据集上实现零样本SOTA性能。

**[Let's Split Up: Zero-Shot Classifier Edits for Fine-Grained Video Understanding](lets_split_up_zero-shot_classifier_edits_for_fine-grained_video_understanding.md)**

:   提出了"类别拆分"(Category Splitting)新任务，通过挖掘视频分类器权重中的潜在组合结构，在零样本条件下将粗粒度动作类别拆分为细粒度子类别，无需重训或额外数据。

**[Map the Flow: Revealing Hidden Pathways of Information in VideoLLMs](map_the_flow_revealing_hidden_pathways_of_information_in_videollms.md)**

:   首次用机制可解释性工具（Attention Knockout + Logit Lens）系统逆向工程VideoLLM的时序推理过程，揭示出"早中层跨帧交互→中层视频-语言整合→中后层答案生成"的三阶段信息流蓝图，并证明仅保留42%注意力边即可几乎无损保持VideoQA性能。

**[Steering and Rectifying Latent Representation Manifolds in Frozen Multi-Modal LLMs for Video Anomaly Detection](steering_and_rectifying_latent_representation_manifolds_in_frozen_multi-modal_ll.md)**

:   提出 SteerVAD 框架，在完全冻结的多模态大语言模型 (MLLM) 内部，通过识别"潜在异常专家"注意力头并用层次化元控制器动态操控其表示流形，仅用 1% 训练数据即实现免调优视频异常检测的 SOTA。

**[Video-KTR: Reinforcing Video Reasoning via Key Token Attribution](video-ktr_reinforcing_video_reasoning_via_key_token_attribution.md)**

:   提出 Video-KTR，一种模态感知的策略塑造框架，通过反事实分析识别视觉感知型、时序敏感型和高熵 Token 三类关键 Token，仅对这些 Token 执行选择性强化学习更新，在多个视频推理基准上达到 SOTA（Video-Holmes 42.7%，超越 GPT-4o）。

**[VideoNSA: Native Sparse Attention Scales Video Understanding](videonsa_native_sparse_attention_scales_video_understanding.md)**

:   本文提出 VideoNSA，将 Native Sparse Attention（NSA）引入视频语言模型，通过压缩、选择和滑动窗口三分支动态门控的混合稀疏注意力机制，在仅使用 3.6% 注意力预算的条件下实现 128K token 的视频理解，在长视频理解、时序推理和空间理解任务上全面超越 token 压缩和无训练稀疏注意力基线。
