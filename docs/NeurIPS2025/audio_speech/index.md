<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎵 音频/语音

**🧠 NeurIPS2025** · 共 **16** 篇

**[A Controllable Examination for Long-Context Language Models](a_controllable_examination_for_longcontext_language_models.md)**

:   提出LongBioBench，通过生成虚构传记作为可控的needle和haystack，构建满足"无缝上下文、可控设置、可靠评估"三大原则的长上下文LLM评估框架，测试18个模型后揭示当前LCLM在检索能力尚可的情况下推理和可信性仍有显著短板。

**[A TRIANGLE Enables Multimodal Alignment Beyond Cosine Similarity](a_triangle_enables_multimodal_alignment_beyond_cosine_simila.md)**

:   TRIANGLE提出用三模态嵌入向量端点构成的三角形面积作为相似度度量，替代传统的两两余弦相似度，实现视频-音频-文本的联合对齐，在视频检索任务上比VAST提升最高9个R@1点。

**[Accelerate Creation of Product Claims Using Generative AI](accelerate_creation_of_product_claims_using_generative_ai.md)**

:   开发 Claim Advisor 平台，利用 LLM 的 in-context learning 和 LoRA 微调加速消费品产品宣称的搜索、生成、优化和排序，通过模仿 MaxDiff 研究方法论让微调的 Phi-3 14B 模型在宣称排序上超越 GPT-4o（仅用 1 个示例 vs GPT 的 100 个示例），三轮迭代后 100% 的生成宣称达到"高吸引力"级别。

**[AdaptDel: Adaptable Deletion Rate Randomized Smoothing for Certified Robustness](adaptdel_adaptable_deletion_rate_randomized_smoothing_for_ce.md)**

:   提出AdaptDel方法，将随机平滑(randomized smoothing)中的固定删除率扩展为**自适应删除率**，根据输入长度等属性动态调整删除概率，在编辑距离攻击下实现认证鲁棒性的巨大提升（认证区域基数提升最高30个数量级）。

**[Associative Syntax and Maximal Repetitions Reveal Context-Dependent Complexity in Animal Vocalizations](associative_syntax_and_maximal_repetitions_reveal_context-dependent_complexity_i.md)**

:   提出基于"关联句法"和"最大重复"的信息论框架分析动物发声序列的结构复杂度，发现动物发声（如鲸鱼歌声）展现出上下文依赖的复杂句法结构，超越了简单的马尔可夫假设。

**[AudSemThinker: Enhancing Audio-Language Models through Reasoning over Semantics of Sound](audsemthinker_enhancing_audio-language_models_through_reasoning_over_semantics_o.md)**

:   AudSemThinker 为音频语言模型引入结构化语义推理框架——定义 9 类声音语义描述符（谁/什么/如何/何时/何地等），在 Qwen2.5-Omni-7B 上通过 SFT + GRPO（含可验证奖励和长度约束）训练产生 \<think\>\<semantic_elements\>\<answer\> 三阶段输出，MMAU 基准达 66.70%（超越 Audio-Reasoner 61.71% 和 Qwen2.5-Omni 65.60%）。

**[Benchmarking Egocentric Multimodal Goal Inference for Assistive Wearable Agents](benchmarking_egocentric_multimodal_goal_inference_for_assist.md)**

:   Meta 提出 WAGIBench，一个针对可穿戴辅助智能体的多模态目标推断基准，包含 348 名参与者的 3,477 条第一视角录制（29小时），涵盖视觉/音频/数字/纵向四种模态，人类准确率 93% vs 最佳 VLM 84%（MCQ），生成式评估中模型仅 55% 时间产生相关目标，揭示了当前 VLM 在实际可穿戴场景中的显著差距。

**[BNMusic: Blending Environmental Noises into Personalized Music](bnmusic_blending_environmental_noises_into_personalized_music.md)**

:   提出 BNMusic，一个两阶段框架将环境噪声融合到个性化生成音乐中：第一阶段通过 mel-spectrogram 的 outpainting + inpainting 生成与噪声节奏对齐的音乐，第二阶段利用听觉掩蔽理论自适应放大音乐信号以降低噪声感知，无需额外训练，在 EPIC-SOUNDS 和 ESC-50 上显著优于 baseline。

**[Can LLMs Outshine Conventional Recommenders? A Comparative Evaluation](can_llms_outshine_conventional_recommenders_a_comparative_evaluation.md)**

:   提出RecBench全面评估框架，在5个数据集上对比17个LLM和10个传统DLRM，发现LLM推荐器准确率提升5-170%但推理速度慢10-100×，传统DLRM+LLM特征组合以20×更快速度达到LLM~95%的性能，揭示了LLM-as-RS的实际部署不可行性。

**[Embedding Alignment In Code Generation For Audio](embedding_alignment_in_code_generation_for_audio.md)**

:   通过双MLP+对比学习将代码嵌入和音频嵌入映射到公共空间，使LLM生成的音频代码能隐式捕捉音乐相似性（CKA从0.090提升到0.590）。

**[Eurospeech A Multilingual Speech Corpus](eurospeech_a_multilingual_speech_corpus.md)**

:   构建EuroSpeech多语言语音数据集：从欧洲议会录音中自动化提取61K小时、覆盖22种语言的高质量语音数据，ASR WER降低41.8%。

**[Generating Physically Sound Designs From Text And A Set Of Physical Constraints](generating_physically_sound_designs_from_text_and_a_set_of_physical_constraints.md)**

:   联合优化视觉目标（CLIP文本对齐）和物理目标（可微分FEM结构约束），生成满足工程要求且包含文本指定特征的结构设计。

**[Instance-Specific Test-Time Training for Speech Editing in the Wild](instance-specific_test-time_training_for_speech_editing_in_the_wild.md)**

:   提出面向野外语音编辑的实例特定测试时训练方法：在推理前利用未编辑区域的真实声学特征做直接监督、编辑区域通过时长约束和音素预测辅助损失做间接监督，对模型进行实例级自适应微调，有效缓解编辑边界的带宽不连续问题，并支持通过 mask 长度调整精确控制语速，在野外 benchmark 上主客观评估均超越现有系统。

**[Multi-head Temporal Latent Attention](multi-head_temporal_latent_attention.md)**

:   MTLA 在 MLA 低秩潜在维度压缩基础上，用超网络动态融合时序相邻的 KV 向量，实现 KV 缓存在特征维度和时序维度的双重压缩，配合 stride-aware 因果 mask 保证训练-推理一致性，在语音翻译等任务上达到 4.29× 加速和 6.58× 内存降低，质量持平甚至略优于标准 MHA。

**[Perceptually Aligning Representations of Music via Noise-Augmented Autoencoders](perceptually_aligning_representations_of_music_via_noise-augmented_autoencoders.md)**

:   证明在自编码器训练中对潜变量加噪（noise-augmented latent training）配合感知损失，能使编码空间形成"感知层次结构"——感知最显著的音乐特征（如音高）编码在最粗粒度的潜在结构中，而次要特征（如音色细节）编码在细粒度结构中。这种对齐改善了潜在扩散解码下的音乐惊奇感估计和 EEG 脑响应预测。

**[Seeing Sound, Hearing Sight: Uncovering Modality Bias and Conflict of AI Models in Sound Localization](seeing_sound_hearing_sight_uncovering_modality_bias_and_conflict_of_ai_models_in.md)**

:   系统性地揭示了AI声源定位(SSL)模型存在严重视觉偏见——在视听冲突时降到随机水平，提出神经科学启发的EchoPin模型（HRTF滤波+耳蜗图+立体声），在AudioCOCO数据集上大幅超越现有方法并展现出类人的水平>垂直定位精度偏差。
