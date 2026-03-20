<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎵 音频/语音

**🧠 NeurIPS2025** · 共 **9** 篇

**[A Controllable Examination for Long-Context Language Models](a_controllable_examination_for_longcontext_language_models.md)**

:   提出LongBioBench，通过生成虚构传记作为可控的needle和haystack，构建满足"无缝上下文、可控设置、可靠评估"三大原则的长上下文LLM评估框架，测试18个模型后揭示当前LCLM在检索能力尚可的情况下推理和可信性仍有显著短板。

**[A TRIANGLE Enables Multimodal Alignment Beyond Cosine Similarity](a_triangle_enables_multimodal_alignment_beyond_cosine_simila.md)**

:   TRIANGLE提出用三模态嵌入向量端点构成的三角形面积作为相似度度量，替代传统的两两余弦相似度，实现视频-音频-文本的联合对齐，在视频检索任务上比VAST提升最高9个R@1点。

**[Accelerate Creation of Product Claims Using Generative AI](accelerate_creation_of_product_claims_using_generative_ai.md)**

:   开发 Claim Advisor 平台，利用 LLM 的 in-context learning 和 LoRA 微调加速消费品产品宣称的搜索、生成、优化和排序，通过模仿 MaxDiff 研究方法论让微调的 Phi-3 14B 模型在宣称排序上超越 GPT-4o（仅用 1 个示例 vs GPT 的 100 个示例），三轮迭代后 100% 的生成宣称达到"高吸引力"级别。

**[AdaptDel: Adaptable Deletion Rate Randomized Smoothing for Certified Robustness](adaptdel_adaptable_deletion_rate_randomized_smoothing_for_ce.md)**

:   提出AdaptDel方法，将随机平滑(randomized smoothing)中的固定删除率扩展为**自适应删除率**，根据输入长度等属性动态调整删除概率，在编辑距离攻击下实现认证鲁棒性的巨大提升（认证区域基数提升最高30个数量级）。

**[Benchmarking Egocentric Multimodal Goal Inference for Assistive Wearable Agents](benchmarking_egocentric_multimodal_goal_inference_for_assist.md)**

:   Meta 提出 WAGIBench，一个针对可穿戴辅助智能体的多模态目标推断基准，包含 348 名参与者的 3,477 条第一视角录制（29小时），涵盖视觉/音频/数字/纵向四种模态，人类准确率 93% vs 最佳 VLM 84%（MCQ），生成式评估中模型仅 55% 时间产生相关目标，揭示了当前 VLM 在实际可穿戴场景中的显著差距。

**[BNMusic: Blending Environmental Noises into Personalized Music](bnmusic_blending_environmental_noises_into_personalized_music.md)**

:   提出 BNMusic，一个两阶段框架将环境噪声融合到个性化生成音乐中：第一阶段通过 mel-spectrogram 的 outpainting + inpainting 生成与噪声节奏对齐的音乐，第二阶段利用听觉掩蔽理论自适应放大音乐信号以降低噪声感知，无需额外训练，在 EPIC-SOUNDS 和 ESC-50 上显著优于 baseline。

**[Embedding Alignment In Code Generation For Audio](embedding_alignment_in_code_generation_for_audio.md)**

:   通过双MLP+对比学习将代码嵌入和音频嵌入映射到公共空间，使LLM生成的音频代码能隐式捕捉音乐相似性（CKA从0.090提升到0.590）。

**[Eurospeech A Multilingual Speech Corpus](eurospeech_a_multilingual_speech_corpus.md)**

:   构建EuroSpeech多语言语音数据集：从欧洲议会录音中自动化提取61K小时、覆盖22种语言的高质量语音数据，ASR WER降低41.8%。

**[Generating Physically Sound Designs From Text And A Set Of Physical Constraints](generating_physically_sound_designs_from_text_and_a_set_of_physical_constraints.md)**

:   联合优化视觉目标（CLIP文本对齐）和物理目标（可微分FEM结构约束），生成满足工程要求且包含文本指定特征的结构设计。
