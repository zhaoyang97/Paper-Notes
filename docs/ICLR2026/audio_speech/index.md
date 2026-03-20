<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎵 音频/语音

**🔬 ICLR2026** · 共 **7** 篇

**[AC-Foley: Reference-Audio-Guided Video-to-Audio Synthesis with Acoustic Transfer](ac-foley_reference-audio-guided_video-to-audio_synthesis_with_acoustic_transfer.md)**

:   提出 AC-Foley，一种参考音频引导的视频到音频合成框架，通过两阶段训练（声学特征学习+时序适应）和多模态条件流匹配实现了细粒度音色控制、音色迁移和零样本音效生成，在音频质量和声学保真度上显著优于现有方法。

**[FlexiCodec: A Dynamic Neural Audio Codec for Low Frame Rates](flexicodec_a_dynamic_neural_audio_codec_for_low_frame_rates.md)**

:   提出 FlexiCodec，通过 ASR 特征引导的动态帧率合并策略，在 3–12.5Hz 超低帧率下实现高质量语音编解码，同时保持优异的语义信息保留能力。

**[Incentive-Aligned Multi-Source LLM Summaries](incentive-aligned_multi-source_llm_summaries.md)**

:   将博弈论中的多任务 peer prediction 机制引入 LLM 多源摘要管线，提出 Truthful Text Summarization (TTS) 框架：通过 leave-one-out 交叉构造评价声明集、提取每个来源对声明的立场、用 informative agreement 评分来源可靠性并过滤不可靠来源后重新摘要，理论上证明"如实报告是效用最大策略"，实验中有效抵御 prompt injection、虚假信息源和协同攻击。

**[Latent Speech-Text Transformer](latent_speech_text_transformer.md)**

:   提出 Latent Speech-Text Transformer (LST)，将离散语音 token 聚合为更高层级的"潜在语音 patch"作为自回归单元，对齐语音和文本的序列建模粒度，在 speech HellaSwag 上获得 +6.5% 绝对提升，且增益随模型规模（420M→7B）持续增长，同时降低 ASR/TTS 推理计算成本。

**[MMSU: A Massive Multi-task Spoken Language Understanding and Reasoning Benchmark](mmsu_a_massive_multi-task_spoken_language_understanding_and_reasoning_benchmark.md)**

:   提出 MMSU（5000 条音频 QA、47 个任务），首个系统融合语言学理论的语音理解与推理基准，评测 22 个 SpeechLLM，发现现有模型在音韵感知和复杂推理上仍存在显著差距。

**[Query-Guided Spatial-Temporal-Frequency Interaction for Music Audio-Visual Question Answering](query-guided_spatial-temporal-frequency_interaction_for_music_audio-visual_quest.md)**

:   提出 QSTar 框架，通过在整个处理流程中嵌入问题引导（Query Guidance），并引入空间-时序-频域三维度交互模块（特别是利用频谱特征区分音色），显著提升了音乐场景下的音频-视觉问答（Music AVQA）性能。

**[RedTeamCUA: Realistic Adversarial Testing of Computer-Use Agents in Hybrid Web-OS Environments](redteamcua_adversarial_testing_agents.md)**

:   构建首个混合 Web-OS 环境的 CUA 红队测试框架 RedTeamCUA 和 864 个测试用例的 RTC-Bench，系统评估 9+ 前沿 CUA 对间接 prompt injection 的脆弱性，发现所有 CUA 均可被攻击（最高 ASR 83%），且能力越强的模型越危险——攻击尝试率（AR）远高于成功率（ASR）意味着模型能力提升将直接转化为更高的攻击成功率。
