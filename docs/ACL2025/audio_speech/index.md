<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎵 音频/语音

**💬 ACL2025** · 共 **10** 篇

**[Finding A Voice: Exploring the Potential of African American Dialect and Voice Generation for Chatbots](aae_voice_chatbot.md)**

:   研究将非裔美式英语（AAE）整合到聊天机器人中的效果——开发文本和语音 AAE 聊天机器人并用 AAE 说话者评估，发现文本 AAE 聊天机器人常表现不佳（方言生成不够自然），但语音聊天机器人结合非裔声音和 AAE 元素时用户体验更好，揭示了语言个性化的复杂性。

**[AI4Reading: Chinese Audiobook Interpretation System Based on Multi-Agent Collaboration](ai4reading_chinese_audiobook_interpretation_system_based_on_multi-agent_collabor.md)**

:   提出 AI4Reading，一个基于 11 个专业化 LLM Agent 协作的中文有声书解读系统，通过主题分析、案例扩展、编辑润色、口语化改写和整合修订等阶段自动生成解读稿，并用 TTS 合成音频，在解读脚本质量（简洁性、完整性、准确性、连贯性）上超过专业人工解读平台樊登读书。

**[Benchmarking Open-ended Audio Dialogue Understanding for Large Audio-Language Models](audio_dialogue_benchmark.md)**

:   提出 ADU-Bench，一个包含 20,000+ 开放式音频对话的综合基准，覆盖 3 种通用场景、12 项技能、9 种语言和 4 类歧义处理，首次系统评估大型音频语言模型（LALM）的音频对话理解能力，在 16 个模型上的实验揭示了现有 LALM 在数学符号、角色扮演、多语言和语音歧义处理上的显著不足。

**[Autoregressive Speech Synthesis without Vector Quantization](autoregressive_speech_synthesis_without_vq.md)**

:   MELLE 提出了一种基于连续 mel-spectrogram 帧的自回归语言模型 TTS 方法，通过回归损失 + 变分推断采样模块 + spectrogram flux loss 直接预测连续频谱帧，避免了向量量化带来的保真度损失和采样鲁棒性问题，单阶段模型即可达到与人类水平相当的语音合成质量。

**[ControlSpeech: Towards Simultaneous and Independent Zero-shot Speaker Cloning and Zero-shot Language Style Control](controlspeech_zero_shot.md)**

:   ControlSpeech 是首个同时实现零样本音色克隆和零样本语言风格控制的TTS系统，通过离散编解码器空间中的解耦表示和风格混合语义密度（SMSD）模块解决了风格控制中的多对多问题。

**[GigaSpeech 2: An Evolving, Large-Scale and Multi-domain ASR Corpus for Low-Resource Languages](gigaspeech2_low_resource_asr.md)**

:   GigaSpeech 2 构建了一个约 30,000 小时的大规模低资源语言（泰语、印尼语、越南语）ASR 语料库，通过自动化爬取-转录-精炼管线从无标注 YouTube 视频生成高质量伪标签，训练的模型仅用 10% 参数量即可将 WER 比 Whisper large-v3 降低 25%-40%。

**[Sparsify: Learning Sparsity for Effective and Efficient Music Performance Question Answering](sparsify_music_avqa.md)**

:   Sparsify 提出三层稀疏化策略（稀疏掩码+自适应稀疏合并+关键子集选择）用于音乐表演视听问答（Music AVQA），在 MUSIC-AVQA 和 v2.0 两个 benchmark 上达到 SOTA（81.75%/81.30%），训练时间减少 28.32%，25% 数据即保持 74% 的全量性能。

**[SpeechIQ: Speech-Agentic Intelligence Quotient Across Cognitive Levels in Voice Understanding by Large Language Models](speechiq_speechagentic_intelligence_quotient_across_cognitive.md)**

:   提出 SpeechIQ，一个基于 Bloom 认知分类学的层次化语音理解评估框架，从 Remember（WER）、Understand（语义相似度）、Apply（QA 准确率）三个层次综合评估语音 LLM 的智能水平，发现级联 ASR+LLM 系统在同规模下优于端到端多模态模型。

**[T2A-Feedback: Improving Basic Capabilities of Text-to-Audio Generation via Fine-grained AI Feedback](t2a_feedback_audio_gen.md)**

:   提出三个细粒度 AI 音频评分管线（事件出现/事件顺序/声学和谐质量）替代人工标注构建大规模音频偏好数据集 T2A-FeedBack（41K提示+249K音频），用偏好调优增强 TTA 模型的基础能力，在简单（AudioCaps）和复杂（T2A-EpicBench）场景下都显著提升多事件音频生成质量。

**[In-the-wild Audio Spatialization with Flexible Text-guided Localization](tas_audio_spatialization.md)**

:   提出 TAS（Text-guided Audio Spatialization）框架，用灵活的文本提示（3D 空间位置描述或声源间相对位置描述）引导潜在扩散模型将单声道音频转换为双耳音频，构建了 376K 样本的 SpatialTAS 数据集，在模拟和真实录制数据上均超越现有方法，并基于 Llama-3.1-8B 开发了空间语义一致性评估模型。
