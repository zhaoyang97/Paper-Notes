---
title: >-
  ICLR2026 预训练论文汇总 · 35篇论文解读
description: >-
  35篇ICLR2026的预训练方向论文解读，涵盖 LLM、扩散模型、对齐/RLHF、对抗鲁棒等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICLR2026"
  - "预训练"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "扩散模型"
  - "对齐/RLHF"
  - "对抗鲁棒"
item_list:
  - u: "a_law_of_data_reconstruction_for_random_features_and_beyond/"
    t: "A Law of Data Reconstruction for Random Features (and Beyond)"
  - u: "accessible_realistic_and_fair_evaluation_of_positive-unlabeled_learning_algorith/"
    t: "Accessible, Realistic, and Fair Evaluation of Positive-Unlabeled Learning Algorithms"
  - u: "adept_continual_pretraining_via_adaptive_expansion_and_dynamic_decoupled_tuning/"
    t: "ADEPT: Continual Pretraining via Adaptive Expansion and Dynamic Decoupled Tuning"
  - u: "any-order_flexible_length_masked_diffusion/"
    t: "Any-Order Flexible Length Masked Diffusion"
  - u: "avey-b/"
    t: "Avey-B：把无注意力架构改造成双向编码器"
  - u: "beyond_length_quantifying_long-range_information_for_long-context_llm_pretrainin/"
    t: "Beyond Length: Quantifying Long-Range Information for Long-Context LLM Pretraining Data"
  - u: "beyond_masks_efficient_flexible_diffusion_language_models_via_deletion-insertion/"
    t: "Beyond Masks: Efficient, Flexible Diffusion Language Models via Deletion-Insertion Processes"
  - u: "beyond_multi-token_prediction_pretraining_llms_with_future_summaries/"
    t: "Beyond Multi-Token Prediction: Pretraining LLMs with Future Summaries"
  - u: "beyond_urls_metadata_diversity_and_position_for_efficient_llm_pretraining/"
    t: "Beyond URLs: Metadata Diversity and Position for Efficient LLM Pretraining"
  - u: "block-sample_mac-bayes_generalization_bounds/"
    t: "Block-Sample MAC-Bayes Generalization Bounds"
  - u: "can_small_training_runs_reliably_guide_data_curation_rethinking_proxy-model_prac/"
    t: "Can Small Training Runs Reliably Guide Data Curation? Rethinking Proxy-Model Practice"
  - u: "chammi-75_pre-training_multi-channel_models_with_heterogeneous_microscopy_images/"
    t: "CHAMMI-75: Pre-training multi-channel models with heterogeneous microscopy images"
  - u: "common_corpus_ethical_data_for_llm_pretraining/"
    t: "Common Corpus: The Largest Collection of Ethical Data for LLM Pre-Training"
  - u: "common_corpus_the_largest_collection_of_ethical_data_for_llm_pre-training/"
    t: "Common Corpus: The Largest Collection of Ethical Data for LLM Pre-Training"
  - u: "conditioned_initialization_for_attention/"
    t: "Conditioned Initialization for Attention"
  - u: "deconstructing_positional_information_from_attention_logits_to_training_biases/"
    t: "Deconstructing Positional Information: From Attention Logits to Training Biases"
  - u: "emergent_misalignment_is_easy_narrow_misalignment_is_hard/"
    t: "Emergent Misalignment is Easy, Narrow Misalignment is Hard"
  - u: "explaining_grokking_and_information_bottleneck_through_neural_collapse_emergence/"
    t: "Explaining Grokking and Information Bottleneck through Neural Collapse Emergence"
  - u: "fictionalqa_a_dataset_for_studying_memorization_and_knowledge_acquisition/"
    t: "FictionalQA: A Dataset for Studying Memorization and Knowledge Acquisition"
  - u: "identifying_and_evaluating_inactive_heads_in_pretrained_llms/"
    t: "Identifying and Evaluating Inactive Heads in Pretrained LLMs"
  - u: "imagine_how_to_change_explicit_procedure_modeling_for_change_captioning/"
    t: "Imagine How To Change: Explicit Procedure Modeling for Change Captioning"
  - u: "implicit_bias_and_loss_of_plasticity_in_matrix_completion_depth_promotes_low-ran/"
    t: "Implicit Bias and Loss of Plasticity in Matrix Completion: Depth Promotes Low-Rank"
  - u: "intrinsic_training_dynamics_of_deep_neural_networks/"
    t: "Intrinsic Training Dynamics of Deep Neural Networks"
  - u: "lossless_vocabulary_reduction_for_auto-regressive_language_models/"
    t: "Lossless Vocabulary Reduction for Auto-Regressive Language Models"
  - u: "moma_a_modular_deep_learning_framework_for_material_property_prediction/"
    t: "MoMa: A Simple Modular Deep Learning Framework for Material Property Prediction"
  - u: "polynomial_trigonometric_and_tropical_activations/"
    t: "Polynomial, trigonometric, and tropical activations"
  - u: "pre-training_llm_without_learning_rate_decay_enhances_supervised_fine-tuning/"
    t: "Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning"
  - u: "predicting_training_re-evaluation_curves_enables_effective_data_curriculums_for_/"
    t: "Predicting Training Re-evaluation Curves Enables Effective Data Curriculums"
  - u: "recon_robust_symmetry_discovery_via_explicit_canonical_orientation_normalization/"
    t: "RECON: Robust symmetry discovery via Explicit Canonical Orientation Normalization"
  - u: "scaling_with_collapse_efficient_and_predictable_training_of_llm_families/"
    t: "Scaling with Collapse: Efficient and Predictable Training of LLM Families"
item_total: 35
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📚 预训练

**🔬 ICLR2026** · **35** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (5)](../../CVPR2026/llm_pretraining/index.md) · [🧪 ICML2026 (27)](../../ICML2026/llm_pretraining/index.md) · [💬 ACL2026 (12)](../../ACL2026/llm_pretraining/index.md) · [🤖 AAAI2026 (9)](../../AAAI2026/llm_pretraining/index.md) · [🧠 NeurIPS2025 (51)](../../NeurIPS2025/llm_pretraining/index.md) · [📹 ICCV2025 (9)](../../ICCV2025/llm_pretraining/index.md)

🔥 **高频主题：** LLM ×7 · 扩散模型 ×2

**[A Law of Data Reconstruction for Random Features (and Beyond)](a_law_of_data_reconstruction_for_random_features_and_beyond.md)**

:   从信息论和代数角度证明随机特征模型中存在数据重构定律：当参数量 $p \gg dn$（$d$ 为数据维度，$n$ 为样本数）时，训练数据可被完整重构，并通过投影损失优化方法在 RF、两层网络和 ResNet 上验证了该阈值的普适性。

**[Accessible, Realistic, and Fair Evaluation of Positive-Unlabeled Learning Algorithms](accessible_realistic_and_fair_evaluation_of_positive-unlabeled_learning_algorith.md)**

:   提出首个 PU 学习统一基准，系统解决两个关键问题：(1) 用代理准确率和代理 AUC 实现无负样本的模型选择；(2) 发现并通过将正样本并入无标签集的简单校准方法解决单样本设置下的内部标签偏移问题，使双样本算法在单样本评估中得到公平比较。

**[ADEPT: Continual Pretraining via Adaptive Expansion and Dynamic Decoupled Tuning](adept_continual_pretraining_via_adaptive_expansion_and_dynamic_decoupled_tuning.md)**

:   ADEPT 发现 LLM 各层、各参数单元对"通用能力"的贡献是高度不均的，于是只复制那些对通用域最不重要的层来腾出新容量，并在这些扩展层内部按单元重要性分配不对称学习率，从而在数学/医学领域持续预训练中既注入新知识又几乎不损伤通用能力——只调 15% 参数、不到 50% 训练时间，却比全参 CPT 在通用基准上高 5.76%、领域基准上高 5.58%。

**[Any-Order Flexible Length Masked Diffusion](any-order_flexible_length_masked_diffusion.md)**

:   本文提出 FlexMDM，一种能在生成过程中**插入新 token、从而建模变长序列**的掩码扩散模型，它在理论上保留了掩码扩散"任意顺序并行解码"的能力，困惑度与定长掩码扩散持平但长度分布拟合显著更好，并且只需 16 张 H100 三天就能把预训练好的 LLaDA-8B 改造成变长模型，在 GSM8K（58%→67%）和代码填空（52%→65%）上明显提升。

**[Avey-B：把无注意力架构改造成双向编码器](avey-b.md)**

:   Avey-B 把原本自回归的无注意力架构 Avey 改造成 BERT 式双向编码器：去掉因果掩码、把静态权重和动态相似度解耦成交替层、给动态层加行归一化、再在 ranker 里塞一个神经压缩器，结果在 token 分类和信息检索上稳超 BERT/RoBERTa/ModernBERT/NeoBERT，且预训练 token 量比 ModernBERT 少约 11×、在 96K 长度上吞吐快 ModernBERT 3.38×。

**[Beyond Length: Quantifying Long-Range Information for Long-Context LLM Pretraining Data](beyond_length_quantifying_long-range_information_for_long-context_llm_pretrainin.md)**

:   针对"长文本 ≠ 长依赖"这个被忽视的事实，提出 LongFilter——用同一个语言模型在长/短上下文下对每个 token 的预测分布做对比，量化"扩展上下文带来的信息增益"，据此筛掉那些虽然很长但其实只靠局部就能预测的样本；用筛后的数据继续预训练 LLaMA-3-8B（8K→64K），在 HELMET、LongBench、RULER 上平均提升 2 分以上，且约一半数据量即可达到等效效果。

**[Beyond Masks: Efficient, Flexible Diffusion Language Models via Deletion-Insertion Processes](beyond_masks_efficient_flexible_diffusion_language_models_via_deletion-insertion.md)**

:   DID 把扩散语言模型的「掩码-去掩码」彻底换成「删除-插入」两条连续时间马尔可夫链：前向把 token 逐个删到空序列、后向从空序列逐个插回去，再配一套基于「插入分数」的 DISE 训练目标和并行动态规划，既扔掉了占一半算力的 `<MASK>`/`<PAD>` token，又天然支持变长和生成中自纠错，定长/变长两种设定下训练加速最高 3.42×、推理加速最高 3.79×。

**[Beyond Multi-Token Prediction: Pretraining LLMs with Future Summaries](beyond_multi-token_prediction_pretraining_llms_with_future_summaries.md)**

:   这篇论文提出 **未来摘要预测（Future Summary Prediction, FSP）**：在标准的下一 token 预测之外挂一个辅助头，让模型预测对**长程未来序列的紧凑摘要**（而不是逐个预测未来若干 token），并给出两种摘要构造方式——手工的词袋摘要（FSP-BoW）和用反向语言模型蒸馏出来的学习式摘要（FSP-RevLM）；3B/8B 大规模预训练实验显示它在数学、推理、代码任务上稳定超过 NTP 与多 token 预测（MTP），数学任务上最高提升约 4–5 个百分点。

**[Beyond URLs: Metadata Diversity and Position for Efficient LLM Pretraining](beyond_urls_metadata_diversity_and_position_for_efficient_llm_pretraining.md)**

:   这篇论文系统地拓宽了"元数据条件化加速 LLM 预训练"的设计空间：除了已知有效的 URL 前置，作者发现**细粒度**的质量分数与领域信息同样能加速训练，并提出"后置元数据作为辅助预测任务"和"可学习元 token"两种新机制，再用逐层探针揭示这些信号如何重塑潜在表征。

**[Block-Sample MAC-Bayes Generalization Bounds](block-sample_mac-bayes_generalization_bounds.md)**

:   提出块样本MAC-Bayes泛化界（mean approximately correct），将训练数据划分为J个块后用各块条件下的KL散度之和替代整体KL散度，在确定性学习算法（如均值估计）等原始PAC-Bayes界为空（vacuous）的场景下仍能给出有限、有意义的泛化误差界，并证明了该界的高概率版本在一般情况下不可行。

**[Can Small Training Runs Reliably Guide Data Curation? Rethinking Proxy-Model Practice](can_small_training_runs_reliably_guide_data_curation_rethinking_proxy-model_prac.md)**

:   这篇论文指出前沿团队普遍依赖的"用小代理模型、固定超参比较数据配方"的做法存在致命缺陷——数据集排名会被学习率的微小变化翻转，作者提出用极小学习率（$10^{-5}\sim10^{-6}$）训练代理模型作为简单补丁，并在 23 个数据配方上把代理（GPT2-125M）到目标模型（Pythia-1B）的排名 Spearman 相关性从 $<0.75$ 提升到 $>0.95$。

**[CHAMMI-75: Pre-training multi-channel models with heterogeneous microscopy images](chammi-75_pre-training_multi-channel_models_with_heterogeneous_microscopy_images.md)**

:   构建 CHAMMI-75——最大的异构多通道显微镜图像预训练数据集（280 万图像，75 个来源，25 种通道类型，16 种物种），证明成像模态多样性是提升多通道模型泛化能力的关键因素，训练的 MorphEm 模型在 7 个 benchmark 中 6 个达到 SOTA。

**[Common Corpus: The Largest Collection of Ethical Data for LLM Pre-Training](common_corpus_ethical_data_for_llm_pretraining.md)**

:   构建 Common Corpus——约 2 万亿 token 的最大规模合法授权 LLM 预训练数据集，覆盖 6 大集合（政府/文化/科学/代码/Web/语义），多语言（含低资源语言），所有数据均为无版权或宽松许可来源，配有完整数据溯源和多阶段过滤管道，已被 Anthropic 等行业领导者采用。

**[Common Corpus: The Largest Collection of Ethical Data for LLM Pre-Training](common_corpus_the_largest_collection_of_ethical_data_for_llm_pre-training.md)**

:   作者发布了 **Common Corpus**——目前规模最大（约 2 万亿 token）、且**全部由公共领域或开放许可内容**构成的 LLM 预训练数据集，覆盖多语言、多领域、跨数百年时间跨度，并配套开源了一整套 OCR 纠错 / 文本分割 / PII 脱敏 / 毒性过滤工具；用它训出的 350M 与 1.2B 小模型在多语言基准上与同量级闭源数据训练的模型表现相当，证明"合规也能训出好模型"。

**[Conditioned Initialization for Attention](conditioned_initialization_for_attention.md)**

:   这篇论文从理论上把注意力层的优化稳定性归因到其 Jacobian 的条件数，进而提出"条件化初始化"——把 value 矩阵初始化成矩形单位阵、把 query/key 矩阵初始化成半正交阵（两者条件数都为 1），从而在训练起点收紧 Jacobian 条件数的上界，在图像分类、检测分割、语言建模、长序列等多种 Transformer 任务上一致地加速收敛（快 20–30%）并提升泛化。

**[Deconstructing Positional Information: From Attention Logits to Training Biases](deconstructing_positional_information_from_attention_logits_to_training_biases.md)**

:   提出基于 Toeplitz 矩阵的统一分析框架，将位置编码分为加法（Absolute/T5/ALiBi）和乘法（RoPE）两类；通过合成任务发现 RoPE 在位置敏感任务上优势显著但存在"单头沉积模式"（single-head deposit pattern）——浅层几乎所有位置推理集中于单个注意力头；理论证明该模式是 RoPE 乘法结构的固有属性。

**[Emergent Misalignment is Easy, Narrow Misalignment is Hard](emergent_misalignment_is_easy_narrow_misalignment_is_hard.md)**

:   研究发现在窄域有害数据上微调会造成广域错位（emergent misalignment），因为"通用错位"比"仅在特定域错位"是更简单高效的参数空间解——通用解的参数范数更小且对噪声更稳定。

**[Explaining Grokking and Information Bottleneck through Neural Collapse Emergence](explaining_grokking_and_information_bottleneck_through_neural_collapse_emergence.md)**

:   通过 Neural Collapse 的视角统一解释 Grokking（延迟泛化）和 Information Bottleneck（压缩阶段）两大训练后期现象，证明群体类内方差的收缩是两者的共同关键因素，并揭示训练损失收敛与 Neural Collapse 发生存在由 weight decay 控制的不同时间尺度。

**[FictionalQA: A Dataset for Studying Memorization and Knowledge Acquisition](fictionalqa_a_dataset_for_studying_memorization_and_knowledge_acquisition.md)**

:   提出 FictionalQA 数据集及生成管线，通过合成关于虚构事件的 webtext 风格文档和 QA 对，在受控环境下研究 LLM 训练中事实记忆与逐字记忆的双重过程，发现更多样的表面形式有助于知识获取而简洁的结构化列表反而最不利于泛化。

**[Identifying and Evaluating Inactive Heads in Pretrained LLMs](identifying_and_evaluating_inactive_heads_in_pretrained_llms.md)**

:   系统评估12种评分函数识别LLM中不活跃注意力头，发现基于头输出范数的评分函数（AHON LN）比传统注意力权重指标更能跨模型家族一致地识别不活跃头，14个模型上平均超过12%的头可被置零而保持MMLU精度在1%以内。

**[Imagine How To Change: Explicit Procedure Modeling for Change Captioning](imagine_how_to_change_explicit_procedure_modeling_for_change_captioning.md)**

:   提出 ProCap 框架，将变化描述从静态图像对比较重新定义为动态过程建模：第一阶段通过帧插值和掩码重建训练过程编码器学习时空变化动力学，第二阶段用可学习过程查询隐式推断变化过程，在三个数据集上超越 SOTA。

**[Implicit Bias and Loss of Plasticity in Matrix Completion: Depth Promotes Low-Rank](implicit_bias_and_loss_of_plasticity_in_matrix_completion_depth_promotes_low-ran.md)**

:   通过分析深度矩阵分解（深度线性网络）在矩阵补全任务中的梯度流动力学，证明了耦合动力学是深度网络低秩隐式偏差的关键机制，且深度≥3的网络除对角初始化外必然展现耦合，从而解释了深度模型为何能避免可塑性损失。

**[Intrinsic Training Dynamics of Deep Neural Networks](intrinsic_training_dynamics_of_deep_neural_networks.md)**

:   本文研究深度神经网络梯度流训练中，参数空间的轨迹何时可以被"提升"到低维本征空间并表示为内禀的黎曼梯度流，提出了基于守恒律的内禀可恢复性（intrinsic recoverability）准则，并将结果推广到任意深度的 ReLU 网络和线性网络。

**[Lossless Vocabulary Reduction for Auto-Regressive Language Models](lossless_vocabulary_reduction_for_auto-regressive_language_models.md)**

:   提出**无损词表缩减（LVR）**的理论框架，通过嵌套分词（nested tokenization）将任意自回归语言模型精确转换为使用任意子词表的等价模型，并基于**最大公共词表（MCV）**实现不同分词方案语言模型之间的高效集成，在 GSM8K、MATH、翻译等多个任务上验证了方法的有效性。

**[MoMa: A Simple Modular Deep Learning Framework for Material Property Prediction](moma_a_modular_deep_learning_framework_for_material_property_prediction.md)**

:   提出 MoMa 模块化材料属性预测框架，先在多任务上训练专用模块并集中存储为 MoMa Hub，再通过表示驱动的无训练自适应模块组合算法（AMC）为下游任务定制模型，在 17 个数据集上平均超越最强基线 14%。

**[Polynomial, trigonometric, and tropical activations](polynomial_trigonometric_and_tropical_activations.md)**

:   系统探索基于正交基（Hermite多项式、Fourier三角基）和热带化（tropicalization）的可学习激活函数族，通过方差保持初始化解决多项式激活的梯度爆炸/消失问题，在GPT-2和ConvNeXt上成功替代GELU实现有效训练。

**[Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning](pre-training_llm_without_learning_rate_decay_enhances_supervised_fine-tuning.md)**

:   提出 Warmup-Stable-Only (WSO) 学习率调度策略——在预训练中完全去掉学习率衰减阶段，虽然预训练指标较差，但在 SFT 后一致性地超越所有衰减策略，通过损失景观分析揭示 WSO 保持更平坦的极小值区域是其优势根源。

**[Predicting Training Re-evaluation Curves Enables Effective Data Curriculums](predicting_training_re-evaluation_curves_enables_effective_data_curriculums_for_.md)**

:   提出训练再评估曲线（TREC）诊断工具，通过分析训练完成后模型在各时间步训练数据上的损失来指导高质量数据的最优放置位置，并证明 TREC 形状可通过 AdamW 的隐式 EMA 系数预测，无需实际训练即可设计数据课程。

**[RECON: Robust symmetry discovery via Explicit Canonical Orientation Normalization](recon_robust_symmetry_discovery_via_explicit_canonical_orientation_normalization.md)**

:   提出 RECON，一种类-姿态无关的正则化方向归一化方法，通过简单的右平移（right translation）修正任意训练过程中产生的正则化表示，实现无监督的实例级对称性发现、OOD 姿态检测以及即插即用的测试时正则化层。

**[Scaling with Collapse: Efficient and Predictable Training of LLM Families](scaling_with_collapse_efficient_and_predictable_training_of_llm_families.md)**

:   证明 LLM 家族的训练损失曲线在优化超参数与数据预算匹配时会“崩塞”到同一条通用曲线上，并利用这一现象实现两个实用应用：(1) 偏离崩塞作为训练病理的早期诊断信号，(2) 崩塞曲线的可预测性实现大规模超参调优的早停。

**[SemHiTok: A Unified Image Tokenizer via Semantic-Guided Hierarchical Codebook](semhitok_a_unified_image_tokenizer_via_semantic-guided_hierarchical_codebook_for.md)**

:   提出SemHiTok——通过语义引导层次codebook(SGHC)统一理解和生成的tokenizer：预训练语义codebook上建像素子codebook，结构和训练解耦(分阶段优化)避免联合训练的语义-像素冲突，LLaVA设定下离散tokenizer中理解和重建都SOTA。

**[Steering Language Models with Weight Arithmetic](steering_language_models_with_weight_arithmetic.md)**

:   提出对比式权重引导（Contrastive Weight Steering），通过对正/负行为微调模型的权重差来提取行为方向向量，直接修改模型权重实现行为控制，在谄媚性、恶意性和拒绝性实验中比激活引导（Activation Steering）具有更好的泛化能力和一致性。

**[Token-level Data Selection for Safe LLM Fine-tuning](token-level_data_selection_for_safe_llm_fine-tuning.md)**

:   提出 TOSS（Token-level data Selection for Safe LLM fine-tuning），首个 token 级别的数据选择框架,通过安全退化模型和效用导向模型之间的损失差评估每个 token 的安全风险，实现比样本级方法更优的安全-效用权衡。

**[Understanding and Improving Shampoo and SOAP via Kullback-Leibler Minimization](understanding_and_improving_shampoo_and_soap_via_kullback-leibler_minimization.md)**

:   从 KL 散度最小化角度重新解释 Shampoo 和 SOAP 的结构化二阶矩估计，揭示其固有局限，并提出 KL-Shampoo 和 KL-SOAP 两种实用方案，在无需 Adam grafting 的情况下匹配或超越原始方法。

**[Understanding the Emergence of Seemingly Useless Features in Next-Token Predictors](understanding_the_emergence_of_seemingly_useless_features_in_next-token_predicto.md)**

:   通过将训练梯度信号分解为 direct、pre-cached 和 circuit sharing 三种成分，解释了为什么 NTP 训练的 Transformer 会学到对预测当前下一token"无用"的特征，并在 OthelloGPT、小型语言模型和预训练 LLM（Gemma 2）上验证了这一框架的解释力。
