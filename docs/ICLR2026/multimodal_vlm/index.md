---
search:
  exclude: true
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 👁️ 多模态 VLM

**🔬 ICLR2026** · 共 **18** 篇

**[A-TPT: Angular Diversity Calibration Properties for Test-Time Prompt Tuning of Vision-Language Models](a-tpt_angular_diversity_calibration_properties_for_test-time_prompt_tuning_of_vi.md)**

:   提出 A-TPT 框架，通过最大化归一化文本特征在单位超球面上的最小成对角距离来促进角度多样性，解决测试时提示调优 (TPT) 中 VLM 预测过度自信导致的校准不良问题，在自然分布偏移和医学数据集上均优于现有 TPT 校准方法。

**[Adaptive Debiasing Tsallis Entropy for Test-Time Adaptation](adaptive_debiasing_tsallis_entropy_for_test-time_adaptation.md)**

:   提出将 Tsallis 熵（SE 的广义形式）引入 VLM 的 Test-Time Adaptation，并进一步发展为自适应去偏 Tsallis 熵（ADTE），为每个类别定制去偏参数 $q^l$，在不引入分布特定超参数的情况下比 Shannon 熵选择更可靠的高置信视图，在 ImageNet 及其 5 个变体和 10 个跨域 benchmark 上均超越 SOTA。

**[AgilePruner: An Empirical Study of Attention and Diversity for Adaptive Visual Token Pruning in LVLMs](agilepruner_an_empirical_study_of_attention_and_diversity_for_adaptive_visual_to.md)**

:   通过 erank（有效秩）和注意力熵的系统性实证分析，揭示了视觉 token 剪枝中注意力方法和多样性方法的互补特性——注意力方法抑制幻觉但覆盖有限，多样性方法覆盖全面但易引入幻觉——并据此提出基于图像复杂度自适应切换剪枝策略的 AgilePruner，在 9 个 benchmark 上表现稳健。

**[AQuA: Toward Strategic Response Generation for Ambiguous Visual Questions](aqua_toward_strategic_response_generation_for_ambiguous_visual_questions.md)**

:   提出 AQuA，首个按模糊度细粒度分级（4 级）的视觉问答数据集（7.2K 样本），为每级定义最优回应策略（直接回答/推断/列举/请求澄清），发现 GPT-5 和 Gemini 在模糊 VQA 上都过度自信地直接回答，通过 SFT+GRPO 训练的 3B 模型反而能超越闭源大模型的策略适应能力。

**[BioCAP: Exploiting Synthetic Captions Beyond Labels in Biological Foundation Models](biocap_exploiting_synthetic_captions_beyond_labels_in_biological_foundation_mode.md)**

:   提出 BioCAP，通过用 MLLM 生成 wiki 知识引导的合成描述性 caption（而非仅用物种标签）来训练生物学多模态基础模型，在 10 个物种分类 benchmark 上比 BioCLIP 平均提升 8.8%，在文本-图像检索任务上提升 21.3%。

**[Bongard-RWR+: Real-World Representations of Fine-Grained Concepts in Bongard Problems](bongard-rwr_real-world_representations_of_fine-grained_concepts_in_bongard_probl.md)**

:   构建 Bongard-RWR+，一个包含 5400 个 Bongard 问题的 benchmark，使用 VLM 流水线（Pixtral-12B + Flux.1-dev）自动生成真实感图像来表示抽象概念，系统评估揭示 SOTA VLM 在辨别细粒度视觉概念（如轮廓、旋转、角度）时表现挣扎，准确率低至 19%。

**[Bootstrapping MLLM for Weakly-Supervised Class-Agnostic Object Counting (WS-COC)](bootstrapping_mllm_for_weakly-supervised_class-agnostic_object_counting.md)**

:   提出 WS-COC，首个基于 MLLM 的弱监督类无关目标计数框架，通过分而治之的对话微调（逐步缩小计数范围）、比较排序优化（学习图像间相对计数关系）和全局-局部计数增强三个策略，仅用图像级计数标注即可匹敌甚至超越全监督方法。

**[Breaking the Limits of Open-Weight CLIP: An Optimization Framework for Self-supervised Fine-tuning of CLIP](breaking_the_limits_of_open-weight_clip_an_optimization_framework_for_self-super.md)**

:   本文提出 TuneCLIP，一个自监督微调（SSFT）框架，通过两阶段设计——先恢复优化器统计量（OSR）消除冷启动偏差，再用带margin的铰链全局对比损失（HGCL）缓解假负样本过度惩罚——在不使用任何标签的条件下持续提升已有开源 CLIP 模型的通用性能，在 ImageNet 及变体上提升最高 +2.5%，在 DataComp 基准上提升 +1.2%。

**[Cross-Modal Redundancy and the Geometry of Vision-Language Embeddings](cross-modal_redundancy_and_the_geometry_of_vision-language_embeddings.md)**

:   提出 Iso-Energy 假设（真正跨模态共享的概念在不同模态中应具有相同的平均激活能量），并设计 Aligned SAE 作为分析工具，揭示 VLM 嵌入空间中双模态原子承载跨模态对齐信号、单模态原子完全解释模态间隙的几何结构。

**[Grounding-IQA: Grounding Multimodal Language Models for Image Quality Assessment](grounding-iqa_grounding_multimodal_language_model_for_image_quality_assessment.md)**

:   将空间定位（referring + grounding）与图像质量评估结合，构建 GIQA-160K 数据集训练多模态 LLM 生成带有边界框的质量描述和空间 VQA，在细粒度质量感知上显著优于通用 MLLM。

**[GTR-Bench: Evaluating Geo-Temporal Reasoning in Vision-Language Models](gtr-bench_evaluating_geo-temporal_reasoning_in_vision-language_mod.md)**

:   提出 GTR-Bench，一个面向大规模摄像头网络中移动目标地理时空推理的新基准，评估发现最强模型 Gemini-2.5-Pro（34.9%）远落后于人类水平（78.61%），揭示了当前 VLM 在时空上下文利用失衡、时序预测能力弱、地图-视频对齐能力不足三大缺陷。

**[Leveraging Data to Say No: Memory Augmented Plug-and-Play Selective Prediction](leveraging_data_to_say_no_memory_augmented_plug-and-play_selective_prediction.md)**

:   提出 MA-PaPSP 框架，通过外部检索数据集构建代理嵌入（k-NN 加权平均降低表示方差）+ 对比归一化评分（改善校准），无训练地为任意 VLM 提供可靠的"拒绝回答"能力，在图像描述、图文匹配、分类的选择性预测上全面优于 PaPSP 和 LLM-as-judge 基线。

**[Look Carefully: Adaptive Visual Reinforcements in Multimodal Large Language Models for Hallucination Mitigation](look_carefully_adaptive_visual_reinforcements_in_multimodal_large_language_model.md)**

:   提出 AIR（Adaptive vIsual Reinforcement）框架，通过原型距离的 token 精简 + 最优传输引导的 patch 选择性增强，在推理时无训练地减少 MLLM 幻觉（LLaVA-1.5-7B CHAIR_S: 22→18.4，POPE 准确率 +5.3%），同时保持多模态通用能力。

**[Multimodal Classification via Total Correlation Maximization](multimodal_classification_via_total_correlation_maximization.md)**

:   从信息论角度分析多模态分类中的模态竞争问题，提出 TCMax 损失函数通过最大化多模态特征与标签之间的总相关性（Total Correlation），同时兼顾联合学习、单模态学习和跨模态对齐三重目标，在多个音视频/图文分类基准上超越 SOTA。

**[RAVENEA: A Benchmark for Multimodal Retrieval-Augmented Visual Culture Understanding](ravenea_a_benchmark_for_multimodal_retrieval-augmented_visual_culture_understand.md)**

:   构建首个评估多模态检索增强文化理解的基准 Ravenea，包含 1868 个实例和 11396 篇人工排序的 Wikipedia 文档，覆盖 8 个国家 11 个类别，评估 7 个多模态检索器和 17 个 VLM，发现文化感知的 RAG 可在 cVQA 上平均提升 6%、cIC 上提升 11%。

**[Shuffle-R1: Efficient RL Framework for Multimodal Large Language Models via Data-centric Dynamic Shuffle](shuffle-r1_efficient_rl_framework_for_multimodal_large_language_models_via_data-.md)**

:   提出 Shuffle-R1 框架，通过 Pairwise Trajectory Sampling（选取高对比度轨迹对）和 Advantage-based Batch Shuffle（按优势值重分配训练批次），解决 RL 训练中的 Advantage Collapsing 和 Rollout Silencing 两大效率瓶颈，在 Geo3K 上比 baseline 提升 22%，MathVerse 上超越 GPT-4o。

**[ThinkOmni: Lifting Textual Reasoning to Omni-modal Scenarios via Guidance Decoding](thinkomni_lifting_textual_reasoning_to_omni-modal_scenarios_via_guidance_decodin.md)**

:   提出 ThinkOmni 无训练框架，利用纯文本大推理模型(LRM)在解码时引导全模态 LLM(OLLM)，通过 Stepwise Contrastive Scaling 自适应平衡感知与推理信号，MathVista 达 70.2%、MMAU 达 75.5%，匹配或超越 RFT 方法。

**[Visual Symbolic Mechanisms: Emergent Symbol Processing in Vision Language Models](visual_symbolic_mechanisms_vlm.md)**

:   发现 VLM 内部涌现了一套三阶段符号处理机制（ID retrieval → ID selection → feature retrieval），利用内容无关的空间位置索引（position IDs）来解决视觉绑定问题，并证明绑定错误可直接追溯到这些机制的失败。
