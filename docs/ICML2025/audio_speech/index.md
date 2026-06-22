---
title: >-
  ICML2025 音频/语音论文汇总 · 15篇论文解读
description: >-
  15篇ICML2025的音频/语音方向论文解读，涵盖语音、对话系统、少样本学习、扩散模型等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2025"
  - "音频/语音"
  - "论文解读"
  - "论文笔记"
  - "语音"
  - "对话系统"
  - "少样本学习"
  - "扩散模型"
item_list:
  - u: "aligning_spoken_dialogue_models_from_user_interactions/"
    t: "Aligning Spoken Dialogue Models from User Interactions"
  - u: "binauralflow_a_causal_and_streamable_approach_for_high-quality_binaural_speech_s/"
    t: "BinauralFlow: A Causal and Streamable Approach for High-Quality Binaural Speech Synthesis with Flow Matching Models"
  - u: "bridging_the_language_gap_synthetic_voice_diversity_via_latent_mixup_for_equitab/"
    t: "Bridging the Language Gap: Synthetic Voice Diversity via Latent Mixup for Equitable Speech Recognition"
  - u: "do_not_mimic_my_voice_speaker_identity_unlearning_for_zero-shot_text-to-speech/"
    t: "Do Not Mimic My Voice: Speaker Identity Unlearning for Zero-Shot Text-to-Speech"
  - u: "etta_elucidating_the_design_space_of_text-to-audio_models/"
    t: "ETTA: Elucidating the Design Space of Text-to-Audio Models"
  - u: "flam_frame-wise_language-audio_modeling/"
    t: "FLAM: Frame-Wise Language-Audio Modeling"
  - u: "impact_iterative_mask-based_parallel_decoding_for_text-to-audio_generation_with_/"
    t: "IMPACT: Iterative Mask-based Parallel Decoding for Text-to-Audio Generation with Diffusion Modeling"
  - u: "long-form_speech_generation_with_spoken_language_models/"
    t: "Long-Form Speech Generation with Spoken Language Models"
  - u: "musecontrollite_multifunctional_music_generation_with_lightweight_conditioners/"
    t: "MuseControlLite: Multifunctional Music Generation with Lightweight Conditioners"
  - u: "ntpp_generative_speech_language_modeling_for_dual-channel_spoken_dialogue_via_ne/"
    t: "NTPP: Generative Speech Language Modeling for Dual-Channel Spoken Dialogue via Next-Token-Pair Prediction"
  - u: "omniaudio_generating_spatial_audio_from_360-degree_video/"
    t: "OmniAudio: Generating Spatial Audio from 360-Degree Video"
  - u: "one_wave_to_explain_them_all_a_unifying_perspective_on_feature_attribution/"
    t: "One Wave To Explain Them All: A Unifying Perspective On Feature Attribution"
  - u: "sortformer_a_novel_approach_for_permutation-resolved_speaker_supervision_in_spee/"
    t: "Sortformer: A Novel Approach for Permutation-Resolved Speaker Supervision in Speech-to-Text Systems"
  - u: "sounding_that_object_interactive_object-aware_image_to_audio_generation/"
    t: "Sounding that Object: Interactive Object-Aware Image to Audio Generation"
  - u: "teaching_physical_awareness_to_llms_through_sounds/"
    t: "Teaching Physical Awareness to LLMs through Sounds"
item_total: 15
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎵 音频/语音

**🧪 ICML2025** · **15** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (22)](../../CVPR2026/audio_speech/index.md) · [🔬 ICLR2026 (79)](../../ICLR2026/audio_speech/index.md) · [💬 ACL2026 (70)](../../ACL2026/audio_speech/index.md) · [🧪 ICML2026 (36)](../../ICML2026/audio_speech/index.md) · [🤖 AAAI2026 (31)](../../AAAI2026/audio_speech/index.md) · [🧠 NeurIPS2025 (47)](../../NeurIPS2025/audio_speech/index.md)

🔥 **高频主题：** 语音 ×11 · 对话系统 ×2

**[Aligning Spoken Dialogue Models from User Interactions](aligning_spoken_dialogue_models_from_user_interactions.md)**

:   首次为全双工语音对话模型（Moshi）设计完整的偏好对齐框架，从15万+条真实用户语音对话中自动构建内容+时序两类偏好对，通过仅在文本token上做DPO-LN对齐，QA平均提升3.1%、安全性提升6.9%，并通过人类评估确认多轮对话质量的改善。

**[BinauralFlow: A Causal and Streamable Approach for High-Quality Binaural Speech Synthesis with Flow Matching Models](binauralflow_a_causal_and_streamable_approach_for_high-quality_binaural_speech_s.md)**

:   提出 BinauralFlow，一个基于条件 Flow Matching 的流式双耳语音合成框架，通过因果 U-Net 架构和连续推理管线实现高保真、可流式生成的双耳音频，感知测试中 42% 的混淆率表明生成结果几乎无法与真实录音区分。

**[Bridging the Language Gap: Synthetic Voice Diversity via Latent Mixup for Equitable Speech Recognition](bridging_the_language_gap_synthetic_voice_diversity_via_latent_mixup_for_equitab.md)**

:   本文提出 LatentVoiceMix，在语音转换模型 Diff-HierVC 的说话人风格编码器潜在空间中进行 mixup 插值，生成具有新颖声音特征的合成语音数据用于增强 ASR 训练，在低资源语言 Wolof 上取得了优于波形增强、频谱增强和标准语音转换的 WER 改善效果。

**[Do Not Mimic My Voice: Speaker Identity Unlearning for Zero-Shot Text-to-Speech](do_not_mimic_my_voice_speaker_identity_unlearning_for_zero-shot_text-to-speech.md)**

:   首次提出零样本TTS中的说话人身份遗忘任务，设计了Teacher-Guided Unlearning (TGU) 框架，通过引入随机性使模型"忘记"目标说话人的声纹特征，同时保持对其他说话人的高质量语音合成能力，并提出 spk-ZRF 指标量化遗忘效果。

**[ETTA: Elucidating the Design Space of Text-to-Audio Models](etta_elucidating_the_design_space_of_text-to-audio_models.md)**

:   ETTA 通过大规模系统性实验阐明了文本到音频(TTA)模型的设计空间（数据、架构、训练目标、采样策略），并基于分析结论构建了当前公开数据下最优的 TTA 模型。

**[FLAM: Frame-Wise Language-Audio Modeling](flam_frame-wise_language-audio_modeling.md)**

:   提出 FLAM，一个帧级音频-语言对比模型，通过文本依赖的 logit 偏置校正和百万级合成 SED 数据集，实现开放词汇声音事件的精确时间定位，同时保持全局检索和零样本分类性能。

**[IMPACT: Iterative Mask-based Parallel Decoding for Text-to-Audio Generation with Diffusion Modeling](impact_iterative_mask-based_parallel_decoding_for_text-to-audio_generation_with_.md)**

:   提出 IMPACT 框架，将迭代掩码并行解码（MGM）与潜在扩散模型（LDM）结合，在连续潜在空间中进行文本到音频生成，以轻量 MLP 扩散头替代重型注意力层，同时引入无条件预训练阶段，在 AudioCaps 上取得 FD/FAD 指标 SOTA 且推理速度与最快的 MAGNET-S 相当。

**[Long-Form Speech Generation with Spoken Language Models](long-form_speech_generation_with_spoken_language_models.md)**

:   提出 SpeechSSM，首个能在单次解码会话中学习和生成长达 16 分钟语音的 textless 语音语言模型，利用 Griffin 混合 SSM 架构实现常量内存解码和无限上下文，并引入 LibriSpeech-Long 评估基准和新的嵌入/LLM 评判指标。

**[MuseControlLite: Multifunctional Music Generation with Lightweight Conditioners](musecontrollite_multifunctional_music_generation_with_lightweight_conditioners.md)**

:   提出 MuseControlLite，通过在解耦交叉注意力层中引入旋转位置编码（RoPE），以仅 85M 可训练参数（比 ControlNet 少 6.75 倍）实现对文本到音乐生成的精确时变条件控制，同时首次统一支持音乐属性控制与音频修复/续写。

**[NTPP: Generative Speech Language Modeling for Dual-Channel Spoken Dialogue via Next-Token-Pair Prediction](ntpp_generative_speech_language_modeling_for_dual-channel_spoken_dialogue_via_ne.md)**

:   提出 Next-Token-Pair Prediction (NTPP) 范式，首次用 decoder-only 架构对双通道语音对话进行 speaker-independent 联合分布建模，实现更自然的轮次转换、更低的推理延迟和更强的说话人无关性。

**[OmniAudio: Generating Spatial Audio from 360-Degree Video](omniaudio_generating_spatial_audio_from_360-degree_video.md)**

:   提出 OmniAudio 框架，首次实现从 360 度全景视频生成 First-order Ambisonics (FOA) 空间音频，通过 coarse-to-fine 自监督预训练和双分支视频编码架构，在自建的 Sphere360 数据集上取得 SOTA 性能。

**[One Wave To Explain Them All: A Unifying Perspective On Feature Attribution](one_wave_to_explain_them_all_a_unifying_perspective_on_feature_attribution.md)**

:   提出 Wavelet Attribution Method (WAM)，将特征归因从像素域迁移到小波域，利用小波系数的空间-尺度局部性为音频、图像、体数据提供统一且更具结构信息的模型解释。

**[Sortformer: A Novel Approach for Permutation-Resolved Speaker Supervision in Speech-to-Text Systems](sortformer_a_novel_approach_for_permutation-resolved_speaker_supervision_in_spee.md)**

:   提出 Sortformer——一个基于编码器的说话人日志模型，通过 Sort Loss 按说话人到达时间排序来解决排列问题，替代或辅助传统的排列不变损失（PIL），并设计正弦核函数将说话人标签注入 ASR 编码器，使多说话人 ASR 训练可直接使用标准交叉熵损失，在 LibriSpeechMix 上实现 2-mix/3-mix 相对误差降低 30%/25%。

**[Sounding that Object: Interactive Object-Aware Image to Audio Generation](sounding_that_object_interactive_object-aware_image_to_audio_generation.md)**

:   提出一种交互式对象感知音频生成模型，通过多模态点积注意力在训练时学习图像区域与声音的关联，在测试时用 SAM 分割掩码替代注意力权重，允许用户通过点击选择图像中的视觉对象来生成对应的声音。

**[Teaching Physical Awareness to LLMs through Sounds](teaching_physical_awareness_to_llms_through_sounds.md)**

:   提出 ACORN 框架，通过基于物理的声学通道仿真器生成大规模训练数据，配合同时捕获幅度和相位信息的音频编码器，教会 LLM 从声音中理解物理世界现象。
