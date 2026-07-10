---
title: >-
  ECCV2026 视频生成论文汇总 · 24篇论文解读
description: >-
  24篇ECCV2026的视频生成方向论文解读，涵盖视频生成、扩散模型、压缩/编码、语音、对齐/RLHF等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ECCV2026"
  - "视频生成"
  - "论文解读"
  - "论文笔记"
  - "扩散模型"
  - "压缩/编码"
  - "语音"
  - "对齐/RLHF"
item_list:
  - u: "ahoy_animatable_humans_under_occlusion_from_youtube_videos_with_gaussian_splatti/"
    t: "AHOY! Animatable Humans under Occlusion from YouTube Videos with Gaussian Splatting and Video Diffusion Priors"
  - u: "ar-copo_align_autoregressive_video_generation_with_contrastive_policy_optimizati/"
    t: "AR-CoPO: Align Autoregressive Video Generation with Contrastive Policy Optimization"
  - u: "avtok_1d_unified_tokenization_for_holistic_audio-video_generation/"
    t: "AVTok: 1D Unified Tokenization for Holistic Audio-Video Generation"
  - u: "controllable_egocentric_video_generation_via_occlusion-aware_sparse_3d_hand_join/"
    t: "Controllable Egocentric Video Generation via Occlusion-Aware Sparse 3D Hand Joints"
  - u: "customx_unified_character_action_and_scene_customization_in_video_world_models/"
    t: "CustomX: Unified Character, Action, and Scene Customization in Video World Models"
  - u: "emosh_expressive_motion_and_shape_disentanglement_for_human_animation/"
    t: "EMOSH: Expressive Motion and Shape Disentanglement for Human Animation"
  - u: "event-driven_video_generation/"
    t: "Event-Driven Video Generation"
  - u: "geonvs_geometry_grounded_video_diffusion_for_novel_view_synthesis/"
    t: "GeoNVS: Geometry Grounded Video Diffusion for Novel View Synthesis"
  - u: "latsearch_latent_reward-guided_search_for_faster_inference-time_scaling_in_video/"
    t: "LatSearch: Latent Reward-Guided Search for Faster Inference-Time Scaling in Video Diffusion"
  - u: "learning_transferable_dynamics_priors_from_action_to_world_modeling/"
    t: "Learning Transferable Dynamics Priors from Action to World Modeling"
  - u: "liveedit_towards_real-time_diffusion-based_streaming_video_editing/"
    t: "LiveEdit: Towards Real-Time Diffusion-Based Streaming Video Editing"
  - u: "lumos-nexus_efficient_frequency_bridging_with_homogeneous_latent_space_for_video/"
    t: "Lumos-Nexus: Efficient Frequency Bridging with Homogeneous Latent Space for Video Unified Generation"
  - u: "memlearner_learning_to_query_context_memory_for_video_world_models/"
    t: "MemLearner: Learning to Query Context Memory for Video World Models"
  - u: "mmcontrol_unified_multi-modal_control_for_joint_audio-video_generation/"
    t: "MMControl: Unified Multi-Modal Control for Joint Audio-Video Generation"
  - u: "next-frame_decoding_for_ultra-low-bitrate_image_compression_with_video_diffusion/"
    t: "Next-Frame Decoding for Ultra-Low-Bitrate Image Compression with Video Diffusion Priors"
  - u: "phygdpo_physics-aware_groupwise_direct_preference_optimization_for_physically_co/"
    t: "PhyGDPO: Physics-Aware Groupwise Direct Preference Optimization for Physically Consistent Text-to-Video Generation"
  - u: "physics_question_scene_graph_fine-grained_evaluation_of_physical_plausibility_in/"
    t: "Physics Question Scene Graph: Fine-grained Evaluation of Physical Plausibility in Text-to-Video Generation"
  - u: "physrag_enhancing_physics-awareness_in_video_generation_via_retrieval-augmented_/"
    t: "PhysRAG: Enhancing Physics-Awareness in Video Generation via Retrieval-Augmented Generation"
  - u: "refalign_representation_alignment_for_reference-to-video_generation/"
    t: "RefAlign: Representation Alignment for Reference-to-Video Generation"
  - u: "salt_self-consistent_distribution_matching_with_cache-aware_training_for_fast_vi/"
    t: "Salt: Self-Consistent Distribution Matching with Cache-Aware Training for Fast Video Generation"
  - u: "sift_self-imagination_fine-tuning_for_physically_plausible_motion_in_video_diffu/"
    t: "SIFT: Self-Imagination Fine-Tuning for Physically Plausible Motion in Video Diffusion Models"
  - u: "streamedit_training-free_video_editing_via_few-step_streaming_video_generation/"
    t: "StreamEdit: Training-Free Video Editing via Few-Step Streaming Video Generation"
  - u: "taming_text-to-sounding_video_generation_via_advanced_modality_condition_and_int/"
    t: "Taming Text-to-Sounding Video Generation via Advanced Modality Condition and Interaction"
  - u: "your_data_manifold_is_secretly_a_reward_model_shell-lcc_for_text-to-video_genera/"
    t: "Your Data Manifold is Secretly a Reward Model: Shell-LCC for Text-to-Video Generation"
item_total: 24
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎬 视频生成

**🎞️ ECCV2026** · **24** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (182)](../../CVPR2026/video_generation/index.md) · [🔬 ICLR2026 (97)](../../ICLR2026/video_generation/index.md) · [💬 ACL2026 (4)](../../ACL2026/video_generation/index.md) · [🧪 ICML2026 (32)](../../ICML2026/video_generation/index.md) · [🤖 AAAI2026 (11)](../../AAAI2026/video_generation/index.md) · [🧠 NeurIPS2025 (23)](../../NeurIPS2025/video_generation/index.md)

🔥 **高频主题：** 视频生成 ×13 · 扩散模型 ×6 · 压缩/编码 ×3 · 语音 ×2 · 对齐/RLHF ×2

**[AHOY! Animatable Humans under Occlusion from YouTube Videos with Gaussian Splatting and Video Diffusion Priors](ahoy_animatable_humans_under_occlusion_from_youtube_videos_with_gaussian_splatti.md)**

:   AHOY 从严重遮挡的单目 YouTube 视频中重建完整可动画的 3DGS 化身——先用 DensePose+FLUX+多视角扩散粗建 canonical 化身，再通过 LoRA 微调的 Wan 2.2 视频扩散模型做 RF-Inversion 生成多视角"幻觉"监督视频，最后用 map-pose/LBS-pose 解耦吸收生成数据的不一致性，训练出带姿态依赖形变的高保真高斯图化身。

**[AR-CoPO: Align Autoregressive Video Generation with Contrastive Policy Optimization](ar-copo_align_autoregressive_video_generation_with_contrastive_policy_optimizati.md)**

:   AR-CoPO 提出分块级对比策略优化框架来对齐少步自回归视频生成器：通过分叉机制在随机选取的 chunk 处构造初始噪声邻域候选、局部 GRPO 更新实现信用分配，绕开了 SDE 式 GRPO 与一致性模型近确定性动力学之间的根本性不匹配；同时设计半在策略训练范式利用固定参考回放和 ratio clipping 信任域来提升文本语义对齐质量，避免纯探索导致运动质量崩溃，在 Self-Forcing 上实现 VBench 和 VideoAlign 双基准联合提升。

**[AVTok: 1D Unified Tokenization for Holistic Audio-Video Generation](avtok_1d_unified_tokenization_for_holistic_audio-video_generation.md)**

:   AVTok 提出首个统一的音视频 tokenizer，用双流 Transformer 架构（共享编解码器 + 模态特定可学习查询）将音视频对联合编码到单个 1D 离散隐空间，配合 VFAL 分层训练策略和表示对齐学习，在重建质量和下游自回归生成任务（A2V / V2A / 联合生成）上均达到领先水平，且参数量和计算量远小于主流双分支方案。

**[Controllable Egocentric Video Generation via Occlusion-Aware Sparse 3D Hand Joints](controllable_egocentric_video_generation_via_occlusion-aware_sparse_3d_hand_join.md)**

:   本文提出用稀疏3D手部关节点作为显式控制信号驱动第一人称视频生成，通过遮挡移除的上下文聚合、3D深度加权的遮挡感知传播和3D几何嵌入三个关键设计，在WAN 2.1基础上以仅~20k参数的轻量模块实现精细的手物交互视频控制，性能大幅超越现有2D轨迹和隐式姿态方法。

**[CustomX: Unified Character, Action, and Scene Customization in Video World Models](customx_unified_character_action_and_scene_customization_in_video_world_models.md)**

:   CustomX将3DGS场景、多视角角色与文本指令统一建模为条件自回归视频生成问题，用户提供任意场景与角色后可通过自然语言驱动角色执行上百种动作，在生成质量、角色一致性和动作控制成功率上全面超越基础模型与专用世界模型。

**[EMOSH: Expressive Motion and Shape Disentanglement for Human Animation](emosh_expressive_motion_and_shape_disentanglement_for_human_animation.md)**

:   EMOSH 利用具有丰富表达能力的三维参数化人体模型 EHM 作为控制信号，通过显式的形状-姿态参数解耦从根本上消除体型泄漏问题，结合粗到细的混合运动注入和空间对齐条件机制，在保持精细表情和手势控制的同时实现高保真的人体动画生成。

**[Event-Driven Video Generation](event-driven_video_generation.md)**

:   EVD 给预训练视频 DiT 加了一个轻量 token-level event head，并在训练和采样时用事件门控约束 latent update 只发生在交互真正活跃的区域，从而减少物体提前动、接触缺失、支撑关系跳变和事件后漂移等视频生成动态错误。

**[GeoNVS: Geometry Grounded Video Diffusion for Novel View Synthesis](geonvs_geometry_grounded_video_diffusion_for_novel_view_synthesis.md)**

:   GeoNVS 提出即插即用的 GS-Adapter，把参考视角的扩散特征"抬升"到 3D 高斯里再渲染回目标视角，在**特征空间**而非图像输入端注入几何先验，从而在相机可控性和几何一致性上大幅超越 SEVA / CameraCtrl，平移误差最多降 2 倍、Chamfer 距离最多降 7 倍。

**[LatSearch: Latent Reward-Guided Search for Faster Inference-Time Scaling in Video Diffusion](latsearch_latent_reward-guided_search_for_faster_inference-time_scaling_in_video.md)**

:   LatSearch 训练了一个能对**去噪轨迹中间态潜在向量**（而非解码后视频）直接打分的潜在奖励模型，并用它驱动「奖励引导重采样 + 剪枝（RGRP）」在潜在空间做推理时搜索，从而在视频扩散上做到与 SOTA 相当甚至更好的质量，同时把运行时间最多减少 79%。

**[Learning Transferable Dynamics Priors from Action to World Modeling](learning_transferable_dynamics_priors_from_action_to_world_modeling.md)**

:   提出 A2World，在大规模（215.6 万条轨迹、20+ 机械臂形态）真实机器人操控数据上预训练一个以动作为条件的多视角扩散世界模型，学得的动作→动态先验可适配为长程仿真器（A2World-sim）或指令驱动策略（A2World-policy），在 LIBERO 基准和真实机器人上均达到领先性能。

**[LiveEdit: Towards Real-Time Diffusion-Based Streaming Video Editing](liveedit_towards_real-time_diffusion-based_streaming_video_editing.md)**

:   LiveEdit 通过三阶段渐进蒸馏将双向扩散 Transformer 的编辑能力迁移到因果流式编辑器（4 步推理），并利用自注意力层在背景区域的高时序冗余设计 AR 掩码缓存，在保留双向模型编辑质量的同时实现 12.66 FPS 的实时推理。

**[Lumos-Nexus: Efficient Frequency Bridging with Homogeneous Latent Space for Video Unified Generation](lumos-nexus_efficient_frequency_bridging_with_homogeneous_latent_space_for_video.md)**

:   Lumos-Nexus 提出一种训练高效的两阶段统一视频生成框架：训练时仅微调轻量生成器吸收 VLM 语义，推理时通过统一渐进频率桥接（UPFB）将生成责任从轻量生成器逐步移交给大容量预训练生成器，在共享同质潜空间中实现粗到细的视频合成，兼顾推理驱动的语义准确性与高保真视觉质量。

**[MemLearner: Learning to Query Context Memory for Video World Models](memlearner_learning_to_query_context_memory_for_video_world_models.md)**

:   MemLearner 提出一种基于可学习查询令牌（Q tokens）的自适应上下文记忆机制，让视频世界模型在生成长视频时能端到端地学会"从历史帧里查什么信息有用"，而非依赖人工规则检索关键帧。Q tokens 作为 C（context）和 P（predicted）之间的信息桥，在预训练 Video DiT 内部通过 3D 注意力完成上下文查询，配合浅层查询+深层生成的分层策略和注意力裁剪大幅降低计算开销；在遮挡和动态物体场景下，PSNR 比 CaM 提升 1.38 dB，LPIPS 降低 0.057。

**[MMControl: Unified Multi-Modal Control for Joint Audio-Video Generation](mmcontrol_unified_multi-modal_control_for_joint_audio-video_generation.md)**

:   MMControl 提出首个面向联合音视频生成的多模态统一控制框架，通过 MMCU 将参考图像、参考音频、深度图和姿态序列等异构控制信号统一编码，经双流旁路架构非侵入式注入冻结的 Joint DiT 骨干，并在推理时以模态专属引导缩放因子独立调节视觉与声学控制强度，在身份一致性、音色保真度和结构对齐上全面超越现有单模态控制方法。

**[Next-Frame Decoding for Ultra-Low-Bitrate Image Compression with Video Diffusion Priors](next-frame_decoding_for_ultra-low-bitrate_image_compression_with_video_diffusion.md)**

:   本文提出 NeFIC，将超低码率图像压缩（ULB-IC）中的生成式解码重新解释为从 compact anchor 到最终重建图像的虚拟时间演化过程，利用预训练视频扩散模型（VDM）作为时序先验做下一帧预测，并通过两阶段训练（区间适配 + 单步生成旁路）实现高质量、高效率的极低码率图像压缩。

**[PhyGDPO: Physics-Aware Groupwise Direct Preference Optimization for Physically Consistent Text-to-Video Generation](phygdpo_physics-aware_groupwise_direct_preference_optimization_for_physically_co.md)**

:   PhyGDPO 提出了一种面向物理合理视频生成的成组直接偏好优化框架，用真实视频作为必胜样本、Plackett-Luce 成组概率模型替代两两对比以捕捉全局偏好，配合 VLM 引导的物理奖励重加权和 LoRA 切换参考机制，在 14B 参数量级下高效实现了优于 Sora2 和 Veo3.1 的物理一致性。

**[Physics Question Scene Graph: Fine-grained Evaluation of Physical Plausibility in Text-to-Video Generation](physics_question_scene_graph_fine-grained_evaluation_of_physical_plausibility_in.md)**

:   PQSG 将视频物理合理性评估分解为三层层次化问题图（Object -> Action -> Physics），由 VLM 自动生成带依赖关系的问题并逐节点回答，在 FinePhyEval 数据集上实现了比现有指标更高的人类判断相关性，且能精确定位视频在哪一维度（对象/动作/物理）违反物理规律。

**[PhysRAG: Enhancing Physics-Awareness in Video Generation via Retrieval-Augmented Generation](physrag_enhancing_physics-awareness_in_video_generation_via_retrieval-augmented_.md)**

:   PhysRAG 将检索增强生成（RAG）引入视频扩散模型：先从手工构建的物理视频数据库中检索与输入 prompt 物理属性最相关的参考视频，再用 VideoMAE V2 编码其时空特征，最后通过一组可学习查询（learnable queries）作为信息瓶颈选择性提取物理先验并注入 DiT 去噪过程，在 PhyGenBench 物理常识基准上以平均分 0.58 达到 SOTA，同时推理开销仅增加 1.24%。

**[RefAlign: Representation Alignment for Reference-to-Video Generation](refalign_representation_alignment_for_reference-to-video_generation.md)**

:   RefAlign 提出参考对齐损失（RA loss），在训练时将 DiT 参考分支的中间特征显式对齐到视觉基础模型（VFM）的语义空间——正项拉近同主体特征保证身份一致性，负项推远不同主体特征增强语义判别力——推理时移除对齐模块实现零额外开销，在 OpenS2V-Eval 上 TotalScore 达到 SOTA（60.42%），有效缓解了参考-to-视频生成中的 copy-paste 伪影和多主体混淆。

**[Salt: Self-Consistent Distribution Matching with Cache-Aware Training for Fast Video Generation](salt_self-consistent_distribution_matching_with_cache-aware_training_for_fast_vi.md)**

:   Salt 提出 SC-DMD 框架，通过半群缺陷正则化（semigroup-defect regularizer）修复分布匹配蒸馏（DMD）在多步推理中的组合性缺陷，并结合缓存感知混合步数训练进一步提升自回归视频生成的稳定性，在 2-4 NFE 的极低推理预算下显著提升生成质量。

**[SIFT: Self-Imagination Fine-Tuning for Physically Plausible Motion in Video Diffusion Models](sift_self-imagination_fine-tuning_for_physically_plausible_motion_in_video_diffu.md)**

:   SIFT 提出自想象微调范式，抛弃真实视频输入、从纯高斯噪声出发让扩散模型仅凭 LLM 生成的文本提示"想象"并生成视频，结合双分类器运动感知判别监督和渐进式困难样本回放策略，在不依赖运动解耦标注数据的情况下，显著提升 Wan 和 CogVideoX 两个主干模型生成视频的物理合理性与运动解耦能力，VLM 和人类评估均一致优于 SFT 和 VideoREPA 等基线。

**[StreamEdit: Training-Free Video Editing via Few-Step Streaming Video Generation](streamedit_training-free_video_editing_via_few-step_streaming_video_generation.md)**

:   StreamEdit 把免训练视频编辑从主流的「数据到数据」范式重新表述成「源条件下的噪声到目标」生成问题，直接嫁接到预训练的流式视频生成器（Self Forcing / LongLive）上，用双分支少步采样 + 自注意力桥 + 交叉注意力接地/增强 + 源导向引导，在 5-15 步内实现比现有方法更快、结构保持更好、编辑更可靠的任意长度视频编辑。

**[Taming Text-to-Sounding Video Generation via Advanced Modality Condition and Interaction](taming_text-to-sounding_video_generation_via_advanced_modality_condition_and_int.md)**

:   针对「从文本同时生成画面 + 同步音效」（T2SV）任务，本文用一个双 agent 的字幕改写框架（CRR）给视频塔和音频塔分别喂互不干扰的模态纯净字幕，再用双塔扩散 BridgeDiT 以双向交叉注意力（DCA）在两塔之间搭桥交换特征，在三个基准上多数指标取得 SOTA。

**[Your Data Manifold is Secretly a Reward Model: Shell-LCC for Text-to-Video Generation](your_data_manifold_is_secretly_a_reward_model_shell-lcc_for_text-to-video_genera.md)**

:   本文提出 Shell-LCC，通过将高质量 SFT 数据的补丁级潜在流形建模为各向同性的壳状高密度区域，从中提取稠密、可微且免费的几何奖励信号，在不引入外部奖励模型或人类标注的前提下显著改善 T2V 生成的低层失真（模糊、过度平滑、运动拖影等）。
