---
title: >-
  ICLR2026 预训练论文汇总 · 79篇论文解读
description: >-
  79篇ICLR2026的预训练方向论文解读，涵盖 LLM、扩散模型、对齐/RLHF、对抗鲁棒等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
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
  - u: "autoregressive_models_rival_diffusion_models_at_any-order_generation/"
    t: "Autoregressive Models Rival Diffusion Models at Any-Order Generation"
  - u: "avey-b/"
    t: "Avey-B：把无注意力架构改造成双向编码器"
  - u: "beyond_length_quantifying_long-range_information_for_long-context_llm_pretrainin/"
    t: "Beyond Length: Quantifying Long-Range Information for Long-Context LLM Pretraining Data"
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
  - u: "conditioned_initialization_for_attention/"
    t: "Conditioned Initialization for Attention"
  - u: "deconstructing_positional_information_from_attention_logits_to_training_biases/"
    t: "Deconstructing Positional Information: From Attention Logits to Training Biases"
  - u: "distilled_pretraining_a_modern_lens_of_data_in-context_learning_and_test-time_sc/"
    t: "Distilled Pretraining: A modern lens of Data, In-Context Learning and Test-Time Scaling"
  - u: "dual-objective_language_models_training_efficiency_without_overfitting/"
    t: "Dual-objective Language Models: Training Efficiency Without Overfitting"
  - u: "duet_optimizing_llm_training_data_mixtures_via_noisy_feedback_from_unseen_downst/"
    t: "DUET: Optimizing LLM Training Data Mixtures via Noisy Feedback from Unseen, Downstream Evaluation Tasks"
  - u: "dynamic_chunking_for_end-to-end_hierarchical_sequence_modeling/"
    t: "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling"
  - u: "emergent_misalignment_is_easy_narrow_misalignment_is_hard/"
    t: "Emergent Misalignment is Easy, Narrow Misalignment is Hard"
  - u: "energy-based_transformers_are_scalable_learners_and_thinkers/"
    t: "Energy-Based Transformers are Scalable Learners and Thinkers"
  - u: "explaining_grokking_and_information_bottleneck_through_neural_collapse_emergence/"
    t: "Explaining Grokking and Information Bottleneck through Neural Collapse Emergence"
  - u: "fictionalqa_a_dataset_for_studying_memorization_and_knowledge_acquisition/"
    t: "FictionalQA: A Dataset for Studying Memorization and Knowledge Acquisition"
  - u: "fone_precise_single-token_number_embeddings_via_fourier_features/"
    t: "FoNE: Precise Single-Token Number Embeddings via Fourier Features"
  - u: "gneissweb_preparing_high_quality_data_for_llms_at_scale/"
    t: "GneissWeb: Preparing High Quality Data for LLMs at Scale"
  - u: "how_learning_rate_decay_wastes_your_best_data_in_curriculum-based_llm_pretrainin/"
    t: "How Learning Rate Decay Wastes Your Best Data in Curriculum-Based LLM Pretraining"
  - u: "how_text_quality_interventions_reshape_neural_scaling_laws_for_llms_empirical_st/"
    t: "How Text Quality Interventions Reshape Neural Scaling Laws for LLMs: Empirical Study"
  - u: "how_to_train_data-efficient_llms/"
    t: "How to Train Data-Efficient LLMs"
  - u: "identifying_and_evaluating_inactive_heads_in_pretrained_llms/"
    t: "Identifying and Evaluating Inactive Heads in Pretrained LLMs"
  - u: "imagine_how_to_change_explicit_procedure_modeling_for_change_captioning/"
    t: "Imagine How To Change: Explicit Procedure Modeling for Change Captioning"
  - u: "implicit_bias_and_loss_of_plasticity_in_matrix_completion_depth_promotes_low-ran/"
    t: "Implicit Bias and Loss of Plasticity in Matrix Completion: Depth Promotes Low-Rank"
item_total: 79
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📚 预训练

**🔬 ICLR2026** · **79** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (5)](../../CVPR2026/llm_pretraining/index.md) · [💬 ACL2026 (12)](../../ACL2026/llm_pretraining/index.md) · [🧪 ICML2026 (27)](../../ICML2026/llm_pretraining/index.md) · [🤖 AAAI2026 (9)](../../AAAI2026/llm_pretraining/index.md) · [🧠 NeurIPS2025 (51)](../../NeurIPS2025/llm_pretraining/index.md) · [📹 ICCV2025 (9)](../../ICCV2025/llm_pretraining/index.md)

🔥 **高频主题：** LLM ×17 · 扩散模型 ×5

**[A Law of Data Reconstruction for Random Features (and Beyond)](a_law_of_data_reconstruction_for_random_features_and_beyond.md)**

:   从信息论和代数角度证明随机特征模型中存在数据重构定律：当参数量 $p \gg dn$（$d$ 为数据维度，$n$ 为样本数）时，训练数据可被完整重构，并通过投影损失优化方法在 RF、两层网络和 ResNet 上验证了该阈值的普适性。

**[Accessible, Realistic, and Fair Evaluation of Positive-Unlabeled Learning Algorithms](accessible_realistic_and_fair_evaluation_of_positive-unlabeled_learning_algorith.md)**

:   提出首个 PU 学习统一基准，系统解决两个关键问题：(1) 用代理准确率和代理 AUC 实现无负样本的模型选择；(2) 发现并通过将正样本并入无标签集的简单校准方法解决单样本设置下的内部标签偏移问题，使双样本算法在单样本评估中得到公平比较。

**[ADEPT: Continual Pretraining via Adaptive Expansion and Dynamic Decoupled Tuning](adept_continual_pretraining_via_adaptive_expansion_and_dynamic_decoupled_tuning.md)**

:   ADEPT 发现 LLM 各层、各参数单元对"通用能力"的贡献是高度不均的，于是只复制那些对通用域最不重要的层来腾出新容量，并在这些扩展层内部按单元重要性分配不对称学习率，从而在数学/医学领域持续预训练中既注入新知识又几乎不损伤通用能力——只调 15% 参数、不到 50% 训练时间，却比全参 CPT 在通用基准上高 5.76%、领域基准上高 5.58%。

**[Autoregressive Models Rival Diffusion Models at Any-Order Generation](autoregressive_models_rival_diffusion_models_at_any-order_generation.md)**

:   本文提出 **A3（Any-order Any-subset Autoregressive modeling）**，把扩散语言模型的"任意顺序、任意子集"灵活性重新装回自回归框架——通过分组式因子分解保留 AR 的多层依赖建模能力，再用双流注意力 + 渐进式课程把预训练 AR 模型平滑改造成任意顺序生成器，在用更少训练数据的前提下全面超过同规模扩散语言模型。

**[Avey-B：把无注意力架构改造成双向编码器](avey-b.md)**

:   Avey-B 把原本自回归的无注意力架构 Avey 改造成 BERT 式双向编码器：去掉因果掩码、把静态权重和动态相似度解耦成交替层、给动态层加行归一化、再在 ranker 里塞一个神经压缩器，结果在 token 分类和信息检索上稳超 BERT/RoBERTa/ModernBERT/NeoBERT，且预训练 token 量比 ModernBERT 少约 11×、在 96K 长度上吞吐快 ModernBERT 3.38×。

**[Beyond Length: Quantifying Long-Range Information for Long-Context LLM Pretraining Data](beyond_length_quantifying_long-range_information_for_long-context_llm_pretrainin.md)**

:   针对"长文本 ≠ 长依赖"这个被忽视的事实，提出 LongFilter——用同一个语言模型在长/短上下文下对每个 token 的预测分布做对比，量化"扩展上下文带来的信息增益"，据此筛掉那些虽然很长但其实只靠局部就能预测的样本；用筛后的数据继续预训练 LLaMA-3-8B（8K→64K），在 HELMET、LongBench、RULER 上平均提升 2 分以上，且约一半数据量即可达到等效效果。

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

**[Conditioned Initialization for Attention](conditioned_initialization_for_attention.md)**

:   这篇论文从理论上把注意力层的优化稳定性归因到其 Jacobian 的条件数，进而提出"条件化初始化"——把 value 矩阵初始化成矩形单位阵、把 query/key 矩阵初始化成半正交阵（两者条件数都为 1），从而在训练起点收紧 Jacobian 条件数的上界，在图像分类、检测分割、语言建模、长序列等多种 Transformer 任务上一致地加速收敛（快 20–30%）并提升泛化。

**[Deconstructing Positional Information: From Attention Logits to Training Biases](deconstructing_positional_information_from_attention_logits_to_training_biases.md)**

:   提出基于 Toeplitz 矩阵的统一分析框架，将位置编码分为加法（Absolute/T5/ALiBi）和乘法（RoPE）两类；通过合成任务发现 RoPE 在位置敏感任务上优势显著但存在"单头沉积模式"（single-head deposit pattern）——浅层几乎所有位置推理集中于单个注意力头；理论证明该模式是 RoPE 乘法结构的固有属性。

**[Distilled Pretraining: A modern lens of Data, In-Context Learning and Test-Time Scaling](distilled_pretraining_a_modern_lens_of_data_in-context_learning_and_test-time_sc.md)**

:   这篇论文系统拆解了"用蒸馏做预训练（DPT）"在现代 LLM 范式下的得失：发现蒸馏能显著增强测试时缩放（pass@k 多样性），却同时损害上下文学习（削弱归纳头），并用一个 bigram 沙盒证明这两件相反的事其实出自同一机制——蒸馏只对高熵分布有帮助、对低熵确定性映射无益甚至有害，最后给出 token routing 等可落地的预训练设计建议。

**[Dual-objective Language Models: Training Efficiency Without Overfitting](dual-objective_language_models_training_efficiency_without_overfitting.md)**

:   在**不改动任何模型结构**的前提下，把自回归（AR）和掩码扩散（masked-diffusion, MD）两种训练目标用一个权重 $\alpha$ 线性混合到同一个 Transformer 上训练，让模型同时拥有 AR 的训练高效率和 MD 的抗过拟合能力；作者训了 50 个 470M 模型系统地扫出了在不同数据重复次数下的最优 $\alpha$，结论是「任何设置下混合都比单目标好」。

**[DUET: Optimizing LLM Training Data Mixtures via Noisy Feedback from Unseen, Downstream Evaluation Tasks](duet_optimizing_llm_training_data_mixtures_via_noisy_feedback_from_unseen_downst.md)**

:   DUET 面对"评测任务数据看不见、只能拿到多轮粗糙噪声反馈"的现实场景，用"全局贝叶斯优化调数据域配比 + 局部影响函数挑高质量样本"交替迭代的方式优化 LLM 训练数据混合，并给出收敛性证明，在多个语言任务上显著超过 DoReMi、LESS 等需要细粒度数据信息的方法。

**[Dynamic Chunking for End-to-End Hierarchical Sequence Modeling](dynamic_chunking_for_end-to-end_hierarchical_sequence_modeling.md)**

:   本文提出 H-Net，一个用可学习的「动态分块（Dynamic Chunking, DC）」机制取代 BPE 分词的层级序列模型：网络在 byte 级输入上自动学会在哪里切 chunk、压缩到什么粒度，全程端到端可微，单级 H-Net 在算力/数据对齐下就超过了基于 BPE 的强 Transformer，两级 H-Net 还能匹敌两倍大小的 token 级模型。

**[Emergent Misalignment is Easy, Narrow Misalignment is Hard](emergent_misalignment_is_easy_narrow_misalignment_is_hard.md)**

:   研究发现在窄域有害数据上微调会造成广域错位（emergent misalignment），因为"通用错位"比"仅在特定域错位"是更简单高效的参数空间解——通用解的参数范数更小且对噪声更稳定。

**[Energy-Based Transformers are Scalable Learners and Thinkers](energy-based_transformers_are_scalable_learners_and_thinkers.md)**

:   本文把"预测"重新表述为"对一个学到的验证器（能量函数）做梯度下降优化"，提出一类可扩展的能量模型 Energy-Based Transformers (EBTs)，让模型仅靠无监督预训练就涌现出跨模态、跨任务的 System 2 思考能力（动态分配算力 + 自我验证），在语言与图像上同时超越 Transformer++ 和 DiT。

**[Explaining Grokking and Information Bottleneck through Neural Collapse Emergence](explaining_grokking_and_information_bottleneck_through_neural_collapse_emergence.md)**

:   通过 Neural Collapse 的视角统一解释 Grokking（延迟泛化）和 Information Bottleneck（压缩阶段）两大训练后期现象，证明群体类内方差的收缩是两者的共同关键因素，并揭示训练损失收敛与 Neural Collapse 发生存在由 weight decay 控制的不同时间尺度。

**[FictionalQA: A Dataset for Studying Memorization and Knowledge Acquisition](fictionalqa_a_dataset_for_studying_memorization_and_knowledge_acquisition.md)**

:   提出 FictionalQA 数据集及生成管线，通过合成关于虚构事件的 webtext 风格文档和 QA 对，在受控环境下研究 LLM 训练中事实记忆与逐字记忆的双重过程，发现更多样的表面形式有助于知识获取而简洁的结构化列表反而最不利于泛化。

**[FoNE: Precise Single-Token Number Embeddings via Fourier Features](fone_precise_single-token_number_embeddings_via_fourier_features.md)**

:   FoNE 用一组不同周期的正余弦（Fourier 特征）把任意数字直接映射成**单个 token** 的嵌入，每位数字只占 2 维，从而绕过分词碎片化与频率偏差；一个 38M 的从零训练 Transformer 在加减乘上就能超过微调的 Llama-3.2-1B，并且是唯一在十万级测试样本上达到 100% 准确率的方法。

**[GneissWeb: Preparing High Quality Data for LLMs at Scale](gneissweb_preparing_high_quality_data_for_llms_at_scale.md)**

:   GneissWeb 用「分片精确子串去重 + 一组互补的新颖质量过滤器集成」从 15T 的 FineWeb 蒸馏出约 10T 高质量 token，让 7B 模型在 11 个基准上平均超过 FineWeb 训练版 2.73 个百分点，填补了「<5T 小而精」和「>15T 大而糙」之间的空白。

**[How Learning Rate Decay Wastes Your Best Data in Curriculum-Based LLM Pretraining](how_learning_rate_decay_wastes_your_best_data_in_curriculum-based_llm_pretrainin.md)**

:   作者指出"按质量升序排数据的课程学习"与"学习率衰减"天然冲突——高质量数据被故意放在末尾，却正好撞上学习率衰减到最低、更新步长最小的阶段，于是好数据被白白浪费；通过"温和衰减 + 用模型平均替代衰减"两招，在 1.5B 模型 / 30B token 上仅靠重排数据就把标准 benchmark 平均分相对随机打乱提升了 1.64%。

**[How Text Quality Interventions Reshape Neural Scaling Laws for LLMs: Empirical Study](how_text_quality_interventions_reshape_neural_scaling_laws_for_llms_empirical_st.md)**

:   作者构建了 23 个不同数据质量干预的数据集 QualityPajama，训练了 2000+ 个模型来系统测量「过滤 / 去重 / LLM 改写」如何改变神经缩放律的全部五个参数，发现数据干预同时改变缩放律的系数和指数（不像架构改动只改系数），导致计算最优的 token/参数比可跨数量级波动，从而把缩放律分析确立为评估数据策略的原则性框架。

**[How to Train Data-Efficient LLMs](how_to_train_data-efficient_llms.md)**

:   本文系统对比 22 种数据筛选策略对 LLM 预训练的影响，提出基于指令微调 LLM 直接打质量分的 **Ask-LLM** 和基于核密度估计做覆盖采样的 **Density**，发现质量筛选（Ask-LLM）即便只保留 10% 数据也能超过全量训练并收敛快 70%，而覆盖采样通常只能"追平"全量。

**[Identifying and Evaluating Inactive Heads in Pretrained LLMs](identifying_and_evaluating_inactive_heads_in_pretrained_llms.md)**

:   系统评估12种评分函数识别LLM中不活跃注意力头，发现基于头输出范数的评分函数（AHON LN）比传统注意力权重指标更能跨模型家族一致地识别不活跃头，14个模型上平均超过12%的头可被置零而保持MMLU精度在1%以内。

**[Imagine How To Change: Explicit Procedure Modeling for Change Captioning](imagine_how_to_change_explicit_procedure_modeling_for_change_captioning.md)**

:   提出 ProCap 框架，将变化描述从静态图像对比较重新定义为动态过程建模：第一阶段通过帧插值和掩码重建训练过程编码器学习时空变化动力学，第二阶段用可学习过程查询隐式推断变化过程，在三个数据集上超越 SOTA。

**[Implicit Bias and Loss of Plasticity in Matrix Completion: Depth Promotes Low-Rank](implicit_bias_and_loss_of_plasticity_in_matrix_completion_depth_promotes_low-ran.md)**

:   通过分析深度矩阵分解（深度线性网络）在矩阵补全任务中的梯度流动力学，证明了耦合动力学是深度网络低秩隐式偏差的关键机制，且深度≥3的网络除对角初始化外必然展现耦合，从而解释了深度模型为何能避免可塑性损失。

**[Intrinsic Training Dynamics of Deep Neural Networks](intrinsic_training_dynamics_of_deep_neural_networks.md)**

:   本文研究深度神经网络梯度流训练中，参数空间的轨迹何时可以被"提升"到低维本征空间并表示为内禀的黎曼梯度流，提出了基于守恒律的内禀可恢复性（intrinsic recoverability）准则，并将结果推广到任意深度的 ReLU 网络和线性网络。

**[Joint Selection for Large-Scale Pre-Training Data via Policy Gradient-based Mask Learning](joint_selection_for_large-scale_pre-training_data_via_policy_gradient-based_mask.md)**

:   把万亿 token 级预训练数据选择重新表述成"可学习掩码"问题，用分组策略梯度同时优化质量与多样性两类指标，比贪心算法快 98.9%，从 15T 的 FineWeb 选出 1.5T 的 FineWeb-Mask，在 1.5B/7B 模型上分别带来 3.2%/1.9% 的平均提升。

**[Late-to-Early Training: 让 LLM 更早学到后期知识，从而更快更好](late-to-early_training_let_llms_learn_earlier_so_faster_and_better.md)**

:   LET 用一个小很多（最多 10×）的开源预训练模型的**末层表示**去对齐目标大模型**早期训练步的早层表示**，让大模型在预训练初期就"提前"学到后期才会形成的知识，从而在 1.4B/7B 上实现约 1.6× 加速且下游准确率提升近 5%。

**[Learned Meta-Tokens for Language Modeling](learned_meta-tokens_for_language_modeling.md)**

:   在预训练时向序列里随机注入一批可学习的 **meta-token**，并配一个只在 meta-token 之间流动的稀疏 **meta-attention**，让这些 token 把前文压缩"缓存"成内容锚点，从而用 <100B token 的小模型就实现到 2× 上下文窗口的长度泛化，并给出"meta-token 锐化位置编码"的信息论解释。

**[Learning Facts at Scale with Active Reading](learning_facts_at_scale_with_active_reading.md)**

:   让模型自己为每篇文档生成一组"学习策略"（释义、自测、知识联想、类比……）再据此合成多样化训练数据，从而把一份封闭知识高效地刻进参数里——8B 的 WikiExpert 在 SimpleQA 上反超 405B 的 Llama 和 236B 的 DeepSeekV2。

**[LLM-JEPA: Large Language Models Meet Joint Embedding Predictive Architectures](llm-jepa_large_language_models_meet_joint_embedding_predictive_architectures.md)**

:   把视觉里大获成功的 JEPA（联合嵌入预测架构）首次搬到 LLM 上：在标准 next-token 重构损失之外，再加一项"用 Text 的嵌入预测 Code 的嵌入"的隐空间目标，在不牺牲生成能力、且抗过拟合的前提下，跨四个模型家族、四个数据集显著超过标准微调与预训练。

**[LLM Pretraining with Continuous Concepts](llm_pretraining_with_continuous_concepts.md)**

:   这篇论文提出 CoCoMix，在标准下一词预测之外让模型预测由 SAE 抽取并按归因筛选出的高层概念，再把这些概念压缩成连续向量插入 Transformer 隐状态序列，从而在语言建模、下游推理和可控生成上比普通 NTP 与知识蒸馏更高效。

**[Lossless Vocabulary Reduction for Auto-Regressive Language Models](lossless_vocabulary_reduction_for_auto-regressive_language_models.md)**

:   提出**无损词表缩减（LVR）**的理论框架，通过嵌套分词（nested tokenization）将任意自回归语言模型精确转换为使用任意子词表的等价模型，并基于**最大公共词表（MCV）**实现不同分词方案语言模型之间的高效集成，在 GSM8K、MATH、翻译等多个任务上验证了方法的有效性。

**[MrRoPE: Mixed-radix Rotary Position Embedding](mrrope_mixed-radix_rotary_position_embedding.md)**

:   本文从「进制（radix）转换」的视角重新审视 RoPE，提出统一框架 MrRoPE，把 PI / NTK / YaRN 等一众外推方法都解释为不同的混合进制转换策略，并据此设计出无需微调的 MrRoPE-Pro（渐进式进制转换），在 128K 长上下文上把 YaRN 的检索/对话精度翻倍。

**[Nemotron-CC-Math: A 133 Billion-Token-Scale High Quality Math Pretraining Dataset](nemotron-cc-math_a_133_billion-token-scale_high_quality_math_pretraining_dataset.md)**

:   用「lynx 布局渲染 + 轻量 LLM 清洗」的领域无关流水线，从 Common Crawl 中可靠抽取并标准化数学/代码内容，构建出迄今最高质量的开源数学预训练语料 Nemotron-CC-Math（133B token），在数学、代码、通用知识任务上全面超越 FineMath、MegaMath、OpenWebMath。

**[Next-ToBE: Probabilistic Next Token-Bag Exploitation for Activating Anticipatory Capacity in LLMs](next-tobe_probabilistic_next_token-bag_exploitation_for_activating_anticipatory_.md)**

:   Next-ToBE 把标准 NTP 的 one-hot 目标换成一个覆盖未来窗口内多个 token 的"软目标袋分布"，不加任何额外参数就激活了 LLM 本就潜藏的"前瞻规划"能力，在数学/代码/常识推理上稳定超过 MTP 等强基线。

**[Not All Documents Are What You Need for Extracting Instruction Tuning Data](not_all_documents_are_what_you_need_for_extracting_instruction_tuning_data.md)**

:   针对"从网络语料里抽取指令微调 QA 数据又贵又脏"的问题，本文提出 EQUAL：先用对比学习把文档和 QA 的特征空间对齐再聚类，再把每个文档簇当成多臂老虎机的一条臂、用最优传输分数衡量"这个簇能产出多贴近目标分布的 QA"，迭代地"选簇—抽取—更新"，从而把抽取成本降低 5–10 倍、同时下游准确率反升约 2.5%。

**[OptimSyn: Influence-Guided Rubrics Optimization for Synthetic Data Generation](optimsyn_influence-guided_rubrics_optimization_for_synthetic_data_generation.md)**

:   OptimSyn 把"为合成数据写 rubric（生成规则）"从专家手工活变成一个可学习的策略：用基于梯度的影响力分数衡量每条合成 QA 对目标模型训练的真实贡献，再把这个分数当成奖励，用 GRPO 训练一个 rubric 生成器，在人文社科和医疗两类知识密集领域稳定刷出比主流开源 SFT 语料更好的下游精度。

**[Polynomial, trigonometric, and tropical activations](polynomial_trigonometric_and_tropical_activations.md)**

:   系统探索基于正交基（Hermite多项式、Fourier三角基）和热带化（tropicalization）的可学习激活函数族，通过方差保持初始化解决多项式激活的梯度爆炸/消失问题，在GPT-2和ConvNeXt上成功替代GELU实现有效训练。

**[Pre-training Limited Memory Language Models with Internal and External Knowledge](pre-training_limited_memory_language_models_with_internal_and_external_knowledge.md)**

:   LMLM（Limited Memory Language Model）在**预训练阶段**就把实体级事实查询调用插进语料、并把检索回来的事实值从损失里 mask 掉，逼模型学会「该查就查」而不是死记硬背，结果 382M 的小模型在事实精度上能逼近 LLaMA2-7B，还能靠改数据库一键遗忘。

**[Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning](pre-training_llm_without_learning_rate_decay_enhances_supervised_fine-tuning.md)**

:   提出 Warmup-Stable-Only (WSO) 学习率调度策略——在预训练中完全去掉学习率衰减阶段，虽然预训练指标较差，但在 SFT 后一致性地超越所有衰减策略，通过损失景观分析揭示 WSO 保持更平坦的极小值区域是其优势根源。

**[Pre-training under Infinite Compute](pre-training_under_infinite_compute.md)**

:   当算力远超网页数据时，作者用「重正则化 + 模型集成 + 联合参数/集成缩放 + 蒸馏」把固定 200M token 的预训练损失压到渐近值 3.17，相比标准配方节省 5.17× 数据，并把集成蒸馏进 8× 更小的学生模型仍保留 83% 收益。

**[Predicting Training Re-evaluation Curves Enables Effective Data Curriculums](predicting_training_re-evaluation_curves_enables_effective_data_curriculums_for_.md)**

:   提出训练再评估曲线（TREC）诊断工具，通过分析训练完成后模型在各时间步训练数据上的损失来指导高质量数据的最优放置位置，并证明 TREC 形状可通过 AdamW 的隐式 EMA 系数预测，无需实际训练即可设计数据课程。

**[Pretraining Scaling Laws for Generative Evaluations of Language Models](pretraining_scaling_laws_for_generative_evaluations_of_language_models.md)**

:   本文为「生成式评测」（数学解题等可验证二值奖励、按 pass@k 打分的任务）提出并系统对比了三套预训练缩放定律——分别以**预训练算力**、**参数量+训练 token**、**黄金参考解的对数似然**为自变量去拟合并外推 pass@k；揭示了采样次数 $k$ 是一个能调控缩放行为与可预测性的新杠杆，发现「黄金参考似然」定律的参数在近 5 个数量级上异常稳定，并从理论上证明算力定律就是参数+token 定律的「计算最优包络」。

**[Pretraining with Hierarchical Memories: Separating Long-Tail and Common Knowledge](pretraining_with_hierarchical_memories_separating_long-tail_and_common_knowledge.md)**

:   本文提出在预训练阶段给一个小的「锚模型」挂上一个超大的「分层参数化记忆库」：根据输入文档先做层次聚类路由，只取回约 10% 的记忆参数拼到锚模型上，让锚模型专注通用知识与推理、记忆库专门吸收长尾世界知识；万亿 token 实验显示，160M 锚模型 + 18M 取回记忆（从 4.6B 记忆库取）能追平参数量 2 倍以上的常规模型。

**[Programming by Backprop: An Instruction is Worth 100 Examples when Finetuning LLMs](programming_by_backprop_an_instruction_is_worth_100_examples_when_finetuning_llm.md)**

:   论文提出 Programming by Backprop（PBB）——一套两阶段训练课程，让 LLM 仅凭训练数据里的"声明式指令"（如一段 Python 源码或一组语法规则）就把对应的可执行行为"编译"进权重，无需提供执行示例；实验证明一条指令最多能顶 100 条执行样例，且这一现象对数据治理与安全有直接含义。

**[RECON: Robust symmetry discovery via Explicit Canonical Orientation Normalization](recon_robust_symmetry_discovery_via_explicit_canonical_orientation_normalization.md)**

:   提出 RECON，一种类-姿态无关的正则化方向归一化方法，通过简单的右平移（right translation）修正任意训练过程中产生的正则化表示，实现无监督的实例级对称性发现、OOD 姿态检测以及即插即用的测试时正则化层。

**[Reformulation for Pretraining Data Augmentation](reformulation_for_pretraining_data_augmentation.md)**

:   针对优质预训练语料不够用、反复重复又会掉点的困境，本文提出 **MGA（Massive Genre-Audience reformulation）**：用一个轻量 3.3B MoE 小模型，自适应地为每篇原始文档生成多组「体裁-受众」对，再据此把同一篇文档改写成 5 个风格各异但事实一致的新版本，最终把 195B 优质 token 扩成 770B 合成 token（3.9× 扩张），在 134M–13B 模型上都跑出比「数据重复 / 上采样」更好的 N/D 双向 scaling。

**[Rethinking Data Curation in LLM Training: Online Reweighting Offers Better Generalization than Offline Methods](rethinking_data_curation_in_llm_training_online_reweighting_offers_better_genera.md)**

:   这篇论文把 LLM 的数据筛选（selection）、数据混合（mixing）统一成一个"在线重加权"问题，提出 ADAPT——在训练过程中用样本与验证集的语义相似度动态调整每个样本的逐样本学习率，不删一条数据，几乎零额外开销，却在指令微调和预训练上都比离线筛选/混合方法获得更强的跨基准泛化。

**[Revisiting the Scaling Properties of Downstream Metrics in Large Language Model Training](revisiting_the_scaling_properties_of_downstream_metrics_in_large_language_model_.md)**

:   本文挑战「下游 benchmark 准确率不可预测」的成见，提出**直接**从训练 FLOPs 建模下游准确率的两参数幂律 $-\log Q = A/C^{\alpha}$，并扩展到不同 token-参数比与重复采样 pass@k；在最大 17B 参数、350B token 的网格实验上证明它比经典的「先预测代理指标再映射到准确率」的两阶段法外推更准、更稳。

**[Rewriting Pre-training Data Boosts LLM Performance in Math and Code](rewriting_pre-training_data_boosts_llm_performance_in_math_and_code.md)**

:   本文不靠"过滤丢弃"而是用一个 70B 大模型把公开代码/数学语料"重写干净再保留"，构建出 SwallowCode（≈16.1B token）与 SwallowMath（≈2.3B token）两个数据集；在固定 50B token 预算的继续预训练里，让 Llama-3.1-8B 在 HumanEval 上 pass@1 提升 +17.0、GSM8K 提升 +12.4，证明"数据质量"才是代码与数学能力的根本瓶颈。

**[Scaling Behavior of Discrete Diffusion Language Models](scaling_behavior_of_discrete_diffusion_language_models.md)**

:   这篇论文系统研究了离散扩散语言模型（DLM）在不同噪声类型下的标度律：通过一套用信噪比（SNR）参数化、可在掩码扩散与均匀扩散之间平滑插值的统一扩散框架，并仔细调好 batch size 与学习率，作者发现 DLM 的标度行为强依赖噪声类型——均匀扩散在数据受限场景下更"省数据、吃参数"，最终把均匀扩散模型扩到 10B 参数 / $10^{22}$ FLOPs，验证其标度律可与自回归模型（ALM）竞争。

**[Scaling Laws Revisited: Modeling the Role of Data Quality in Language Model Pretraining](scaling_laws_revisited_modeling_the_role_of_data_quality_in_language_model_pretr.md)**

:   本文在经典 Chinchilla 缩放定律里塞进一个无量纲的数据质量参数 $Q \in (0,1]$，得到 $L(N,D,Q)=A/N^\alpha + B/(D^\beta Q^\gamma) + E$，并在机器翻译和因果语言建模上系统注入噪声做受控实验，证明损失随数据质量可预测地下降、且高质量数据能换取更小的模型与更少的算力。

**[Scaling with Collapse: Efficient and Predictable Training of LLM Families](scaling_with_collapse_efficient_and_predictable_training_of_llm_families.md)**

:   证明 LLM 家族的训练损失曲线在优化超参数与数据预算匹配时会“崩塞”到同一条通用曲线上，并利用这一现象实现两个实用应用：(1) 偏离崩塞作为训练病理的早期诊断信号，(2) 崩塞曲线的可预测性实现大规模超参调优的早停。

**[Selective Rotary Position Embedding](selective_rotary_position_embedding.md)**

:   本文从理论上论证「强召回 = 旋转 + 衰减」缺一不可，指出线性注意力恰恰缺了 softmax 隐式做的那部分「旋转」，于是提出 **Selective RoPE**——一个输入依赖、可学习、能旋转任意角度并与衰减门无缝复合的旋转位置编码，用 RoPE trick 高效实现为一层复数门控线性注意力，在合成召回任务和 370M/1.3B 语言建模上以极小代价提升了召回、表达力和困惑度。

**[SemHiTok: A Unified Image Tokenizer via Semantic-Guided Hierarchical Codebook](semhitok_a_unified_image_tokenizer_via_semantic-guided_hierarchical_codebook_for.md)**

:   提出SemHiTok——通过语义引导层次codebook(SGHC)统一理解和生成的tokenizer：预训练语义codebook上建像素子codebook，结构和训练解耦(分阶段优化)避免联合训练的语义-像素冲突，LLaVA设定下离散tokenizer中理解和重建都SOTA。

**[Seq vs Seq: An Open Suite of Paired Encoders and Decoders](seq_vs_seq_an_open_suite_of_paired_encoders_and_decoders.md)**

:   作者训练了一套从 17M 到 1B、配对的 encoder-only 与 decoder-only 模型（ETTIN suite），二者用**完全相同**的数据、架构和训练配方，只差「目标函数 + 注意力方向」；在公平对比下既各自刷到同尺寸开放数据 SOTA，又证明：分类/检索任务上 encoder 碾压 decoder，生成任务反之，而且**靠继续训练把一种模型改造成另一种（cross-objective）始终补不平这个差距**。

**[Should We Still Pretrain Encoders with Masked Language Modeling?](should_we_still_pretrain_encoders_with_masked_language_modeling.md)**

:   作者用 38 个 210M~1B 的模型、超 1.5 万次微调跑了一场严格受控的对照实验，回答"还该不该用 MLM 预训练编码器"——结论是 MLM 在文本表示任务上整体仍更强，但 CLM 更省数据、微调更稳，因此**先 CLM 再 MLM 的两阶段策略**（尤其是直接拿现成 CLM 解码器继续 MLM）在固定算力下能拿到最优编码器。

**[Soft-Masked Diffusion Language Models](soft-masked_diffusion_language_models.md)**

:   针对掩码扩散语言模型（MDLM）解码时"保留 mask 还是替换成预测 token"这种二元决策会丢掉预测信息的问题，本文提出 **soft-masking（SM）**：把保留下来的 `[MASK]` 嵌入与上一步 top-k 预测 token 的嵌入做一个置信度加权的凸组合，让部分信息跨步传播，仅增加 3 个可训练参数，就在小模型从头训练、预训练续训、以及 Dream-7B/Dream-Coder-7B 微调上稳定提升了困惑度、MAUVE 和代码生成准确率，尤其在低算力（少解码步数 / 高吞吐）场景增益显著。

**[SPICE: Submodular Penalized Information–Conflict Selection for Efficient Large Language Model Training](spice_submodular_penalized_informationconflict_selection_for_efficient_large_lan.md)**

:   SPICE 指出"基于 Fisher 信息的贪心数据选择"在实践中崩得比理论快的元凶是**样本间梯度冲突**，用一个 ε-分解把"偏离理想子模性的程度"量化成冲突统计量，进而提出一个"信息增益 − 冲突惩罚"的冲突感知贪心选择器：在 LLaMA2-7B / Qwen2-7B 上只用 10% 数据、20 GPU-hours，就在 8 个 benchmark 上追平甚至超过全量微调与 6 个基线。

**[ssToken: Self-modulated and Semantic-aware Token Selection for LLM Fine-tuning](sstoken_self-modulated_and_semantic-aware_token_selection_for_llm_fine-tuning.md)**

:   ssToken 在 LLM 监督微调中做 token 级数据筛选：用模型自己的历史 checkpoint 替代外部参考模型算「回顾式超额损失」（自调制信号），再叠加一个基于注意力的语义重要性指标，两路正交信号加权融合后只对 top-ρ 的 token 计损失，在 3B–14B 模型上比全量微调最高提升 4.3%、比已有 token 选择方法最高提升 2.8%，且几乎不增加训练开销。

**[Steering Language Models with Weight Arithmetic](steering_language_models_with_weight_arithmetic.md)**

:   提出对比式权重引导（Contrastive Weight Steering），通过对正/负行为微调模型的权重差来提取行为方向向量，直接修改模型权重实现行为控制，在谄媚性、恶意性和拒绝性实验中比激活引导（Activation Steering）具有更好的泛化能力和一致性。

**[StochasTok: Improving Fine-Grained Subword Understanding in LLMs](stochastok_improving_fine-grained_subword_understanding_in_llms.md)**

:   StochasTok 在分词之后加一个极轻量的后处理步骤——按概率随机把 token 拆成词表里等价的更小 token 对，让 LLM 在预训练时"看见"token 内部结构，从而在数字母、找子串、多位数加法等细粒度子词任务上大幅超越确定性分词与 BPE-dropout，且可热插拔到任意训练阶段。

**[Synthetic Bootstrapped Pretraining](synthetic_bootstrapped_pretraining.md)**

:   SBP（Synthetic Bootstrapped Pretraining）先从预训练语料里挖出语义相近的文档对、训练一个"给定 $d_1$ 生成相关 $d_2$"的条件合成器，再把它铺到整个语料上合成一大批新文档与真实数据联合预训练；在算力对齐的 3B / 6B、1T token 设定下，它稳定超过强复读基线，并能补回 oracle（拥有 20 倍新数据）增益的最多约 60%。

**[Task-Aware Data Selection via Proxy-Label Enhanced Distribution Matching for LLM Fine-Tuning](task-aware_data_selection_via_proxy-label_enhanced_distribution_matching_for_llm.md)**

:   针对"给定一个小目标集、要从大语料池里挑出最相关指令数据来微调 LLM"这一任务，本文指出现有方法只对齐输入特征 $X$ 是不够的，提出用 LLM 推断**代理标签** $Y$、把问题重构成联合分布 $P(X,Y)$ 对齐，并配一条"标注→聚类传播→LLM 评分过滤→增量采样"的四步流水线 TADS，从 300K 池子里只选 10K 样本微调 LLaMA-3.1-8B，效果即可媲美甚至超过 LESS、TSDS 等 SOTA。

**[The Diffusion Duality, Chapter II: Ψ-Samplers](the_diffusion_duality_chapter_ii_psi-samplers.md)**

:   针对均匀态离散扩散（USDM）在大采样步数下质量不升反而饱和的问题，本文提出一族「叠加后验」Ψ-posterior 及其 Ψ-sampler（预测-纠正采样器），把 ReMDM 等纠正方法推广到任意噪声先验，让 USDM 的文本/图像生成质量随采样步数持续改善；同时给出一套用 top-k 顺序统计量近似 softmax 的高效课程，把训练显存降 33%、时间降 25%。

**[Time is a Feature: Exploiting Temporal Dynamics in Diffusion Language Models](time_is_a_feature_exploiting_temporal_dynamics_in_diffusion_language_models.md)**

:   作者发现扩散语言模型（dLLM）在去噪过程中常常"中途答对、最后又改错"（时序振荡），于是把被丢弃的中间步预测当成信号来用：一个免训练的**时序自一致投票**在单条采样轨迹内跨步投票选出最稳定答案，一个**时序一致性强化**用"负时序语义熵"作无标签奖励做 GRPO 后训练，二者在四个数学推理基准上分别带来约 1.5% 与最高 25.3% 的提升。

**[TNT: Improving Chunkwise Training for Test-Time Memorization](tnt_improving_chunkwise_training_for_test-time_memorization.md)**

:   本文提出 TNT 训练范式，用「分层记忆 + 周期性状态重置」打破非线性 RNN 的序列依赖以实现大规模上下文并行，再用一个轻量微调阶段把局部记忆切换到小 chunk，从而把 Titans 类深度记忆模型的训练速度提升至多 17×、同时还提升了精度。

**[Token-level Data Selection for Safe LLM Fine-tuning](token-level_data_selection_for_safe_llm_fine-tuning.md)**

:   提出 TOSS（Token-level data Selection for Safe LLM fine-tuning），首个 token 级别的数据选择框架,通过安全退化模型和效用导向模型之间的损失差评估每个 token 的安全风险，实现比样本级方法更优的安全-效用权衡。

**[Train on Validation (ToV): Fast Data Selection with Applications to Fine-Tuning](train_on_validation_tov_fast_data_selection_with_applications_to_fine-tuning.md)**

:   ToV 把"在训练池上估计每个样本对验证损失的影响"这件事，通过一阶泰勒展开揭示的训练-验证对称性，**反转**成"先在小验证集上微调一步、再看训练池里每个样本的损失变化最大"——只用前向损失评估、不需要逐样本梯度或 Hessian，就能在指令微调与 NER 上以 2–6× 的速度选出比 LESS 更好的微调数据。

**[Understanding and Improving Shampoo and SOAP via Kullback-Leibler Minimization](understanding_and_improving_shampoo_and_soap_via_kullback-leibler_minimization.md)**

:   从 KL 散度最小化角度重新解释 Shampoo 和 SOAP 的结构化二阶矩估计，揭示其固有局限，并提出 KL-Shampoo 和 KL-SOAP 两种实用方案，在无需 Adam grafting 的情况下匹配或超越原始方法。

**[Understanding the Emergence of Seemingly Useless Features in Next-Token Predictors](understanding_the_emergence_of_seemingly_useless_features_in_next-token_predicto.md)**

:   通过将训练梯度信号分解为 direct、pre-cached 和 circuit sharing 三种成分，解释了为什么 NTP 训练的 Transformer 会学到对预测当前下一token"无用"的特征，并在 OthelloGPT、小型语言模型和预训练 LLM（Gemma 2）上验证了这一框架的解释力。

**[Unveiling Downstream Performance Scaling of LLMs: A Clustering-Based Perspective](unveiling_downstream_performance_scaling_of_llms_a_clustering-based_perspective.md)**

:   本文提出 Clustering-On-Difficulty（COD）框架：先按"难度 scaling 特征"把评测样本聚类、筛掉不可外推的簇，再用一条新推导的下游性能 scaling law 对每个簇做 compute-性能外推，最后用一个平滑映射把"可预测子集"的精度还原到完整评测集——在 70B 模型的 8 个主流 benchmark 上把平均预测误差压到 1.55%。

**[What Scales in Cross-Entropy Scaling Law?](what_scales_in_cross-entropy_scaling_law.md)**

:   这篇论文把交叉熵损失精确拆解成「误差熵（Error-Entropy）+ 自对齐（Self-Alignment）+ 置信度（Confidence）」三项，用 32 个跨 5 个数量级的模型实验证明：真正随模型规模呈幂律下降的只有误差熵，另外两项基本不随规模变化——这解释了为什么交叉熵缩放定律在小模型上很准、在超大模型上却会失效。
