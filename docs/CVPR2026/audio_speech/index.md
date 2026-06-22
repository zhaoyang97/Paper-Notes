---
title: >-
  CVPR2026 音频/语音论文汇总 · 22篇论文解读
description: >-
  22篇CVPR2026的音频/语音方向论文解读，涵盖语音、多模态、对齐/RLHF、扩散模型等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "CVPR2026"
  - "音频/语音"
  - "论文解读"
  - "论文笔记"
  - "语音"
  - "多模态"
  - "对齐/RLHF"
  - "扩散模型"
item_list:
  - u: "amuse_audio-visual_benchmark_and_alignment_framework_for_agentic_multi-speaker_u/"
    t: "AMUSE: Audio-Visual Benchmark and Alignment Framework for Agentic Multi-Speaker Understanding"
  - u: "audiostory_generating_long-form_narrative_audio_with_large_language_models/"
    t: "AudioStory: Generating Long-Form Narrative Audio with Large Language Models"
  - u: "babyvlm-v2_toward_developmentally_grounded_pretraining_and_benchmarking_of_visio/"
    t: "BabyVLM-V2: Toward Developmentally Grounded Pretraining and Benchmarking of Vision Foundation Models"
  - u: "cleaning_the_pool_progressive_filtering_of_unlabeled_pools_in_deep_active_learni/"
    t: "Cleaning the Pool: Progressive Filtering of Unlabeled Pools in Deep Active Learning"
  - u: "echoes_over_time_unlocking_length_generalization_in_video-to-audio_generation_mo/"
    t: "Echoes Over Time: Unlocking Length Generalization in Video-to-Audio Generation Models"
  - u: "echofoley_event-centric_hierarchical_control_for_video_grounded_creative_sound_g/"
    t: "EchoFoley: Event-Centric Hierarchical Control for Video Grounded Creative Sound Generation"
  - u: "foleydirector_fine-grained_temporal_steering_for_video-to-audio_generation_via_s/"
    t: "FoleyDirector: Fine-Grained Temporal Steering for Video-to-Audio Generation via Structured Scripts"
  - u: "gem-tfl_bridging_weak_and_full_supervision_for_forgery_localization_through_em-g/"
    t: "GEM-TFL: Bridging Weak and Full Supervision for Forgery Localization"
  - u: "hear_what_you_see_video-to-audio_generation_with_diffusion_transformer_and_seman/"
    t: "Hear What You See: Video-to-Audio Generation with Diffusion Transformer and Semantic-Temporal Alignment-Ranked Direct Preference Optimization"
  - u: "hierarchical_codec_diffusion_for_video-to-speech_generation/"
    t: "Hierarchical Codec Diffusion for Video-to-Speech Generation"
  - u: "how_far_can_we_go_with_synthetic_data_for_audio-visual_sound_source_localization/"
    t: "How Far Can We Go With Synthetic Data for Audio-Visual Sound Source Localization?"
  - u: "infinityhuman_towards_long-term_audio-driven_human_animation/"
    t: "InfinityHuman: Towards Long-Term Audio-Driven Human Animation"
  - u: "omni-mmsi_toward_identity-attributed_social_interaction_understanding/"
    t: "Omni-MMSI: Toward Identity-Attributed Social Interaction Understanding"
  - u: "omniret_efficient_and_high-fidelity_omni_modality_retrieval/"
    t: "OmniRet: Efficient and High-Fidelity Omni Modality Retrieval"
  - u: "omnisonic_towards_universal_and_holistic_audio_generation_from_video_and_text/"
    t: "OmniSonic: Towards Universal and Holistic Audio Generation from Video and Text"
  - u: "pavas_physics-aware_video-to-audio_synthesis/"
    t: "PAVAS: Physics-Aware Video-to-Audio Synthesis"
  - u: "pushing_the_frontier_of_audiovisual_perception_with_large-scale_multimodal_corre/"
    t: "Pushing the Frontier of Audiovisual Perception with Large-Scale Multimodal Correspondence Learning"
  - u: "save_speech-aware_video_representation_learning_for_video-text_retrieval/"
    t: "SAVE: Speech-Aware Video Representation Learning for Video-Text Retrieval"
  - u: "tape_task-adaptive_prototype_evolution_in_audio-language_models_for_fully_few-sh/"
    t: "TAPE: Task-Adaptive Prototype Evolution in Audio-Language Models for Fully Few-shot Class-incremental Audio Classification"
  - u: "tri-subspaces_disentanglement_for_multimodal_sentiment_analysis/"
    t: "Tri-Subspaces Disentanglement for Multimodal Sentiment Analysis"
  - u: "unim_a_unified_any-to-any_interleaved_multimodal_benchmark/"
    t: "UniM: A Unified Any-to-Any Interleaved Multimodal Benchmark"
  - u: "unlocking_strong_supervision_a_data-centric_study_of_general-purpose_audio_pre-t/"
    t: "Unlocking Strong Supervision: A Data-Centric Study of General-Purpose Audio Pre-Training Methods"
item_total: 22
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎵 音频/语音

**📷 CVPR2026** · **22** 篇论文解读

📌 **同领域跨会议浏览：** [🔬 ICLR2026 (79)](../../ICLR2026/audio_speech/index.md) · [💬 ACL2026 (70)](../../ACL2026/audio_speech/index.md) · [🧪 ICML2026 (36)](../../ICML2026/audio_speech/index.md) · [🤖 AAAI2026 (31)](../../AAAI2026/audio_speech/index.md) · [🧠 NeurIPS2025 (47)](../../NeurIPS2025/audio_speech/index.md) · [📹 ICCV2025 (11)](../../ICCV2025/audio_speech/index.md)

🔥 **高频主题：** 语音 ×14 · 多模态 ×3 · 对齐/RLHF ×2 · 扩散模型 ×2

**[AMUSE: Audio-Visual Benchmark and Alignment Framework for Agentic Multi-Speaker Understanding](amuse_audio-visual_benchmark_and_alignment_framework_for_agentic_multi-speaker_u.md)**

:   本文提出 AMUSE——一个面向「多说话人、对话密集」场景的音视频 Benchmark（6 个 agentic 任务 × 零样本/引导/agentic 三种评测模式），揭示了 GPT-4o、Qwen3-Omni 等主流 MLLM 在「谁在说、何时说、跨场景因果」上的系统性短板；并配套提出 RAFT 对齐框架（反思式奖励 + 选择性参数适配），用极少标注就把开源模型在该 Benchmark 上的准确率最高提升 39.52%（相对）。

**[AudioStory: Generating Long-Form Narrative Audio with Large Language Models](audiostory_generating_long-form_narrative_audio_with_large_language_models.md)**

:   AudioStory 把 LLM 的叙事推理和 DiT 扩散音频生成器拼成一个端到端框架，先让 LLM 把复杂指令拆成带时间戳的子事件、再逐段生成短音频拼成长篇叙事音频，靠"语义 token + 残差 token"两路解耦桥接保证段内对齐与跨段连贯，能稳定生成最长 150 秒的多场景音频故事。

**[BabyVLM-V2: Toward Developmentally Grounded Pretraining and Benchmarking of Vision Foundation Models](babyvlm-v2_toward_developmentally_grounded_pretraining_and_benchmarking_of_visio.md)**

:   提出BabyVLM-V2框架，从婴儿第一视角的SAYCam纵向语料构建三种格式预训练数据（768K图像对+181K视频对+63K交错序列），设计基于NIH Baby Toolbox®的DevCV Toolbox（10个发育认知任务），从零训练的紧凑模型在部分数学任务上超越GPT-4o，首次系统探索人工发育智能(ADI)。

**[Cleaning the Pool: Progressive Filtering of Unlabeled Pools in Deep Active Learning](cleaning_the_pool_progressive_filtering_of_unlabeled_pools_in_deep_active_learni.md)**

:   提出 Refine 集成主动学习方法，通过两阶段策略——渐进过滤（多策略迭代精炼无标签池）+ 覆盖选择（从精炼池中选择多样性高价值样本）——在不预知最佳策略的情况下一致超越单一 AL 策略和现有集成方法。

**[Echoes Over Time: Unlocking Length Generalization in Video-to-Audio Generation Models](echoes_over_time_unlocking_length_generalization_in_video-to-audio_generation_mo.md)**

:   提出 MMHNet，一种基于层级结构和非因果 Mamba-2 的多模态层级网络，实现了在短片段（8秒）上训练、在长视频（5分钟以上）上生成高质量对齐音频的长度泛化能力，在 UnAV100 和 LongVale 基准上大幅超越现有方法。

**[EchoFoley: Event-Centric Hierarchical Control for Video Grounded Creative Sound Generation](echofoley_event-centric_hierarchical_control_for_video_grounded_creative_sound_g.md)**

:   针对现有视频配音模型「视觉主导、听不懂文本指令、做不了细粒度编辑」的问题，本文提出 EchoFoley 任务（用符号化「发声事件」表示 + 三层控制粒度），配套 6k 规模密标注 benchmark，并设计了 training-free 的 agentic 框架 EchoVidia（slow-fast thinking + 动作池），在可控性上比最强 baseline 提升约 40.7%、感知质量提升 12.5%。

**[FoleyDirector: Fine-Grained Temporal Steering for Video-to-Audio Generation via Structured Scripts](foleydirector_fine-grained_temporal_steering_for_video-to-audio_generation_via_s.md)**

:   FoleyDirector 在预训练 DiT 类 V2A 生成器（MMAudio）上挂一个可插拔适配器，用"导演脚本"式的逐秒文本（Structured Temporal Scripts）补足视觉线索、实现按时间段精确控制声音何时出现，并用双流并行渲染画内/画外声，在 DirectorBench 上把控制力 F1 从 0.2451 提到 0.4819，同时几乎不损伤原模型音质。

**[GEM-TFL: Bridging Weak and Full Supervision for Forgery Localization](gem-tfl_bridging_weak_and_full_supervision_for_forgery_localization_through_em-g.md)**

:   提出 GEM-TFL，通过两阶段分类-回归框架弥合弱监督与全监督之间的差距，用 EM 分解二元标签为多维潜在属性、训练无关的时序一致性精化、图扩散提案精化三大模块，在弱监督时序伪造定位上平均 mAP 提升 4-8%。

**[Hear What You See: Video-to-Audio Generation with Diffusion Transformer and Semantic-Temporal Alignment-Ranked Direct Preference Optimization](hear_what_you_see_video-to-audio_generation_with_diffusion_transformer_and_seman.md)**

:   VisioSonic 用「CLIP 低帧率语义 + Synchformer 高帧率时序」双路条件喂给一个 video-text-audio 共注意力扩散 Transformer 做整流流匹配生成无声视频的配音，再用全自动、无需人工标注的 STAR-DPO 偏好优化把语义和时序对齐进一步拉满——以 151M 可训练参数（同类最少）拿到 VGGSound 上最强的分布匹配与音视频同步。

**[Hierarchical Codec Diffusion for Video-to-Speech Generation](hierarchical_codec_diffusion_for_video-to-speech_generation.md)**

:   HiCoDiT 把"哑视频→语音"这件事重新拆成**沿 RVQ 离散 token 层级逐层生成**的掩码扩散任务——低层 token 负责内容与音色、由唇动和身份引导，高层 token 负责韵律、由表情通过双尺度 AdaLN 调制，从而在 LRS2/LRS3 上跨数据集零训练就拿下自然度、可懂度和唇同步的领先成绩。

**[How Far Can We Go With Synthetic Data for Audio-Visual Sound Source Localization?](how_far_can_we_go_with_synthetic_data_for_audio-visual_sound_source_localization.md)**

:   本文提出首个用 text-to-X 生成模型批量造数据来训练声源定位（SSL）模型的可扩展框架，证明纯合成数据能与真实数据打平、用合成图像替换有噪的真实中间帧能"提纯"训练集，而真实+合成混合训练在单源定位、音视分割、交互式定位三类任务上全面刷到 SOTA。

**[InfinityHuman: Towards Long-Term Audio-Driven Human Animation](infinityhuman_towards_long-term_audio-driven_human_animation.md)**

:   InfinityHuman 提出"先低分辨率出动作、再姿态引导精炼"的 coarse-to-fine 框架，用与外观解耦、抗时间退化的姿态序列 + 首帧视觉锚点来对抗长视频中的身份漂移和色偏，并引入手部专属奖励反馈学习修正手部畸变，在 EMTD/HDTF 上把长时音频驱动全身动画的画质、身份保持、手部准确度和唇音同步全面刷到 SOTA。

**[Omni-MMSI: Toward Identity-Attributed Social Interaction Understanding](omni-mmsi_toward_identity-attributed_social_interaction_understanding.md)**

:   提出 Omni-MMSI 任务——从原始音视频输入（而非预处理的 oracle 社交线索）理解多人社交交互，并设计 Omni-MMSI-R 参考引导流水线，通过工具生成身份归因社交线索 + 链式思维推理实现准确的社交交互理解。

**[OmniRet: Efficient and High-Fidelity Omni Modality Retrieval](omniret_efficient_and_high-fidelity_omni_modality_retrieval.md)**

:   提出首个支持文本-视觉-音频三模态组合查询的统一检索模型 OmniRet，通过共享媒体重采样器（Shared Media Resampler）提升计算效率，并引入注意力切片 Wasserstein 池化（ASWP）保留细粒度信息，在 13 个检索任务上取得 12 项领先。

**[OmniSonic: Towards Universal and Holistic Audio Generation from Video and Text](omnisonic_towards_universal_and_holistic_audio_generation_from_video_and_text.md)**

:   提出 Universal Holistic Audio Generation (UniHAGen) 任务和 OmniSonic 框架，通过 TriAttn-DiT 架构的三路交叉注意力和 MoE 门控机制，首次实现同时生成屏幕内/屏外环境声和人声的统一音频合成，在新构建的 UniHAGen-Bench 上全面超越 SOTA。

**[PAVAS: Physics-Aware Video-to-Audio Synthesis](pavas_physics-aware_video-to-audio_synthesis.md)**

:   PAVAS 在潜在扩散的视频转音频（V2A）框架里显式注入「物体级质量 + 速度」两个物理量：用 VLM 估质量、用分割 + 动态三维重建估速度，再通过一个零初始化残差的 Phy-Adapter 把这些物理线索灌进扩散 Transformer，让生成的声音强度/衰减真正随物理动力学变化，并在自建的 VGG-Impact 基准上把物理一致性（APCC-∆）从 0.5+ 降到 0.378。

**[Pushing the Frontier of Audiovisual Perception with Large-Scale Multimodal Correspondence Learning](pushing_the_frontier_of_audiovisual_perception_with_large-scale_multimodal_corre.md)**

:   PEAV（Perception Encoder Audiovisual）是 Meta 提出的「音-视-文」统一对比编码器家族：靠一个两阶段合成字幕数据引擎为 O(100M) 音视频对造出高质量的音频/视觉/音视频三类字幕，再用最多十组跨模态对比损失把音频、视频、文本对齐到同一空间，在声音、音乐、语音、视频四类零样本基准上全面刷新 SOTA（如 AudioCaps T→A 从 35.4 提到 45.8 R@1，VGGSound 分类 36.0→47.1），还首次让「语音→转写文本」检索从近 0 做到 85.6。

**[SAVE: Speech-Aware Video Representation Learning for Video-Text Retrieval](save_speech-aware_video_representation_learning_for_video-text_retrieval.md)**

:   提出 SAVE 方法，通过添加专用语音分支（Whisper ASR + CLIP 文本编码器）和 soft-ALBEF 视觉-音频早期对齐策略，实现语音感知的视频表示学习，在五个视频-文本检索基准上全面超越 SOTA。

**[TAPE: Task-Adaptive Prototype Evolution in Audio-Language Models for Fully Few-shot Class-incremental Audio Classification](tape_task-adaptive_prototype_evolution_in_audio-language_models_for_fully_few-sh.md)**

:   针对"基础阶段和增量阶段都只有极少样本"的全小样本类增量音频分类（FFCAC），TAPE 不去微调 CLAP 的文本端，而是冻结其音频编码器、只学一个把音频投影到正交参考点空间的线性 Task-Adapter 来抗遗忘，并在推理阶段用低熵 query 样本动态修正类原型来抗过拟合，三个数据集上把平均准确率从 54.93% 拉到 82.76%。

**[Tri-Subspaces Disentanglement for Multimodal Sentiment Analysis](tri-subspaces_disentanglement_for_multimodal_sentiment_analysis.md)**

:   提出 TSD 框架，将多模态特征显式分解为全局共享/成对共享/模态专属三个互补子空间，并通过子空间感知跨注意力融合模块自适应整合三层信息，在 CMU-MOSI/MOSEI 上全面 SOTA。

**[UniM: A Unified Any-to-Any Interleaved Multimodal Benchmark](unim_a_unified_any-to-any_interleaved_multimodal_benchmark.md)**

:   提出首个统一的任意到任意交错多模态基准 UniM（31K 样本、7 种模态、30 个领域），配套三维评估体系和基于可追溯推理的智能体基线 UniMA，揭示现有 MLLM 在交错多模态范式下的严重不足。

**[Unlocking Strong Supervision: A Data-Centric Study of General-Purpose Audio Pre-Training Methods](unlocking_strong_supervision_a_data-centric_study_of_general-purpose_audio_pre-t.md)**

:   本文通过系统的数据中心实验证明音频预训练性能主要由标签/监督质量驱动而非模型设计，提出 Unified Tag System (UTS) 将语音、音乐、环境音统一到 800-3k 标签的高粒度词表中，UTS 训练的模型用 5 倍更少的数据在语音（VoxCeleb2）和音乐（MusicCaps）等域外任务上超越 AudioSet 基线。
