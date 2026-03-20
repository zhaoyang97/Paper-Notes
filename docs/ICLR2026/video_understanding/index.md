---
search:
  exclude: true
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎬 视频理解

**🔬 ICLR2026** · 共 **12** 篇

**[A.I.R.: Adaptive, Iterative, and Reasoning-based Frame Selection For Video Question Answering](air_enabling_adaptive_iterative_and_reasoning-based_frame_selection_for_video_qu.md)**

:   提出 A.I.R.，一种无需训练的自适应-迭代-推理驱动帧选择框架，通过两阶段策略（GMM 自适应初始采样 + 迭代式 VLM 精细分析）解决 VideoQA 中轻量模型（CLIP）相似度不准确和 VLM 分析成本爆炸的双重困境，在最坏情况下也仅需分析 72 帧（vs 基线 128 帧），同时显著提升多个长视频 benchmark 性能。

**[BindWeave: Subject-Consistent Video Generation via Cross-Modal Integration](bindweave_subject-consistent_video_generation_via_cross-modal_integration.md)**

:   BindWeave 用多模态大语言模型（MLLM）替代传统的浅层融合机制来解析多主体复杂文本指令，生成主体感知的隐状态作为 DiT 的条件信号，结合 CLIP 语义特征和 VAE 细粒度外观特征，实现高保真、主体一致的视频生成。

**[Emergence of Superposition: Unveiling the Training Dynamics of Chain of Continuous Thought](emergence_of_superposition_unveiling_the_training_dynamics_of_chain_of_continuou.md)**

:   从理论上分析了两层 Transformer 在有向图可达性问题上使用连续 Chain-of-Thought（Coconut）训练时的训练动力学，揭示了"叠加态"（superposition）机制如何自然涌现：index-matching logit 先增长后有界，从而在探索与利用之间取得平衡。

**[FlashVID: Efficient Video Large Language Models via Training-free Tree-Based Spatiotemporal Token Merging](flashvid_efficient_video_large_language_models_via_training-free_tree-based_spat.md)**

:   提出 FlashVID，一个免训练的视频大语言模型推理加速框架，通过树状时空 token 合并（TSTM）联合建模空间和时间冗余，仅保留 10% 的视觉 token 就能保持 LLaVA-OneVision 99.1% 的性能，并能将 Qwen2.5-VL 的输入帧数提升 10 倍。

**[FLoC: Facility Location-Based Efficient Visual Token Compression for Long Video Understanding](floc_facility_location-based_efficient_visual_token_compression_for_long_video_u.md)**

:   提出 FLoC，基于设施选址函数（facility location function）的视觉 token 压缩框架，通过子模优化在给定预算下快速选择兼具代表性和多样性的 token 子集，实现无训练、模型无关、查询无关的长视频理解 token 压缩。

**[From Vicious to Virtuous Cycles: Synergistic Representation Learning for Unsupervised Video Object-Centric Learning](from_vicious_to_virtuous_cycles_synergistic_representation_learning_for_unsuperv.md)**

:   发现 slot-based 目标中心学习中编码器（产生尖锐但有噪声的注意力图）与解码器（产生空间一致但模糊的重建掩码）之间的恶性循环，提出同步对比学习目标和 slot 正则化预热策略将其转化为良性循环，在 MOVi 和 YouTube-VIS 上大幅提升物体发现性能。

**[GOT-Edit: Geometry-Aware Generic Object Tracking via Online Model Editing](got-edit_geometry-aware_generic_object_tracking_via_online_model_editing.md)**

:   通过零空间约束的在线模型编辑，将 VGGT 提供的 3D 几何信息融入 2D 通用目标跟踪器中，在保持语义判别力的同时增强几何感知能力，在遮挡和背景杂乱场景中显著提升跟踪性能。

**[JavisDiT: Joint Audio-Video Diffusion Transformer with Hierarchical Spatio-Temporal Prior Synchronization](javisdit_joint_audio-video_diffusion_transformer_with_hierarchical_spatio-tempor.md)**

:   提出 JavisDiT，基于 DiT 架构的音视频联合生成模型，通过层级化时空同步先验估计器（HiST-Sypo）实现细粒度的音视频时空对齐；同时构建了新基准 JavisBench（10K 复杂场景样本）和新评估指标 JavisScore。

**[Lumos-1: On Autoregressive Video Generation with Discrete Diffusion from a Unified Model Perspective](lumos-1_on_autoregressive_video_generation_with_discrete_diffusion_from_a_unifie.md)**

:   提出 Lumos-1，一个基于 LLM 架构的统一视频生成模型：通过 MM-RoPE（分布式多模态 RoPE）解决视觉时空编码问题，通过 AR-DF（自回归离散扩散强迫）解决帧间损失不均衡问题，仅用 48 GPU 训练即可在 GenEval、VBench-I2V 和 VBench-T2V 上达到竞争力水平。

**[CAPO: Curvature-Aware Policy Optimization for Sample-Efficient RL in LLM Reasoning](stabilizing_policy_gradients_for_sample-efficient_reinforcement_learning_in_llm_.md)**

:   CAPO 通过建模优化景观的二阶几何（仅在 LM head 最后一层计算曲率），实现 token 级别的数据筛选——拒绝会导致策略崩溃的更新，使 LLM 推理 RL 训练在激进超参数下仍保持稳定，样本效率提升 30 倍。

**[Stop Tracking Me! Proactive Defense Against Attribute Inference Attack in LLMs](stop_tracking_me_proactive_defense_against_attribute_inference_attack_in_llms.md)**

:   TRACE-RPS 提出统一防御框架应对 LLM 属性推断攻击：TRACE 通过注意力+推理链精准定位隐私泄露文本元素做细粒度匿名化，RPS 通过轻量后缀优化诱导模型拒绝推断，将属性推断准确率从约 50% 降至 5% 以下。

**[TTOM: Test-Time Optimization and Memorization for Compositional Video Generation](ttom_test-time_optimization_and_memorization_for_compositional_video_generation.md)**

:   提出 TTOM 框架，在推理时通过优化新增参数将视频生成模型的注意力与 LLM 生成的时空布局对齐，并用参数记忆机制保存历史优化上下文支持复用，在 T2V-CompBench 上相对提升 34%（CogVideoX）和 14%（Wan2.1）。
