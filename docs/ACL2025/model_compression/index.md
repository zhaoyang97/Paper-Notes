---
title: >-
  ACL2025 模型压缩论文汇总 · 78篇论文解读
description: >-
  78篇ACL2025的模型压缩方向论文解读，涵盖模型压缩、LLM、压缩/编码、知识蒸馏、对抗鲁棒、布局/合成等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ACL2025"
  - "模型压缩"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "压缩/编码"
  - "知识蒸馏"
  - "对抗鲁棒"
  - "布局/合成"
item_list:
  - u: "500xcompressor_generalized_prompt_compression_for_large_language_models/"
    t: "500xCompressor: Generalized Prompt Compression for Large Language Models"
  - u: "accurate_kv_cache_quantization_with_outlier_tokens_tracing/"
    t: "Accurate KV Cache Quantization with Outlier Tokens Tracing"
  - u: "aligndistil_token_level_alignment/"
    t: "AlignDistil: Token-Level Language Model Alignment as Adaptive Policy Distillation"
  - u: "apb_distributed_long_context/"
    t: "APB: Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks across GPUs"
  - u: "assigning_distinct_roles_to_quantized_and_low-rank_matrices_toward_optimal_weigh/"
    t: "Assigning Distinct Roles to Quantized and Low-Rank Matrices Toward Optimal Weight Decomposition"
  - u: "basic_reading_distillation/"
    t: "Basic Reading Distillation"
  - u: "beamlora_beam_constraint_lora/"
    t: "BeamLoRA: Beam-Constraint Low-Rank Adaptation"
  - u: "beyond_logits_aligning_feature_dynamics_for_effective_knowledge_distillation/"
    t: "Beyond Logits: Aligning Feature Dynamics for Effective Knowledge Distillation"
  - u: "beyond_text_compression_tokenizers/"
    t: "Beyond Text Compression: Evaluating Tokenizers Across Scales"
  - u: "bf16_or_death_quantization_tradeoffs/"
    t: "\"Give Me BF16 or Give Me Death\"? Accuracy-Performance Trade-Offs in LLM Quantization"
  - u: "blockpruner_fine-grained_pruning_for_large_language_models/"
    t: "BlockPruner: Fine-grained Pruning for Large Language Models"
  - u: "bone_soups_multi_objective_gen/"
    t: "Bone Soups: A Seek-and-Soup Model Merging Approach for Controllable Multi-Objective Generation"
  - u: "brainecho_semantic_brain_signal_decoding_through_vector-quantized_spectrogram_re/"
    t: "BrainECHO: Semantic Brain Signal Decoding through Vector-Quantized Spectrogram Reconstruction for Whisper-Enhanced Text Generation"
  - u: "cami_a_counselor_agent_supporting_motivational_interviewing_through_state_infere/"
    t: "CAMI: A Counselor Agent Supporting Motivational Interviewing through State Inference and Topic Exploration"
  - u: "capture_key_cot_distillation/"
    t: "Capture the Key in Reasoning to Enhance CoT Distillation Generalization"
  - u: "cfsp_an_efficient_structured_pruning_framework_for_llms_with_coarse-to-fine_acti/"
    t: "CFSP: An Efficient Structured Pruning Framework for LLMs with Coarse-to-Fine Activation Information"
  - u: "claimpkg_enhancing_claim_verification_via_pseudo-subgraph_generation_with_lightw/"
    t: "ClaimPKG: Enhancing Claim Verification via Pseudo-Subgraph Generation with Lightweight Specialized LLM"
  - u: "cola_collaborative_low-rank_adaptation/"
    t: "CoLA: Collaborative Low-Rank Adaptation"
  - u: "compact_and_compressible_representations_for_llms_using_structured_sparse_decom/"
    t: "Compact and Compressible Representations for LLMs Using Structured Sparse Decomposition"
  - u: "compression_in_transformer_language_models_has_a_surprising_relationship_with_pe/"
    t: "Compression in Transformer Language Models Has a Surprising Relationship with Performance"
  - u: "dac_prompt_compression/"
    t: "DAC: A Dynamic Attention-aware Approach for Task-Agnostic Prompt Compression"
  - u: "data_laundering_artificially_boosting_benchmark_results_through_knowledge_distil/"
    t: "Data Laundering: Artificially Boosting Benchmark Results through Knowledge Distillation"
  - u: "deepsolution_boosting_complex_engineering_solution_design_via_tree-based_explora/"
    t: "DeepSolution: Boosting Complex Engineering Solution Design via Tree-based Exploration and Bi-point Thinking"
  - u: "direct_behavior_optimization_unlocking_the_potential_of_lightweight_llms/"
    t: "Direct Behavior Optimization: Unlocking the Potential of Lightweight LLMs"
  - u: "disentangling_the_roles_of_representation_and_selection_in_data_pruning/"
    t: "Disentangling the Roles of Representation and Selection in Data Pruning"
  - u: "domix_an_efficient_framework_for_exploiting/"
    t: "DoMIX: An Efficient Framework for Exploiting Domain Knowledge in Fine-Tuning"
  - u: "drpruning_robust_pruning/"
    t: "DRPruning: Efficient Large Language Model Pruning through Distributionally Robust Optimization"
  - u: "eac_moe_expert_aware_compression/"
    t: "EAC-MoE: Expert-Selection Aware Compressor for Mixture-of-Experts Large Language Models"
  - u: "efficient_long_context_language_model_retrieval_with_compression/"
    t: "Efficient Long Context Language Model Retrieval with Compression"
  - u: "efficient_opamp_adaptation_for_zoom_attention_to_golden_contexts/"
    t: "Efficient OpAmp Adaptation for Zoom Attention to Golden Contexts"
item_total: 78
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📦 模型压缩

**💬 ACL2025** · **78** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (108)](../../CVPR2026/model_compression/index.md) · [🔬 ICLR2026 (239)](../../ICLR2026/model_compression/index.md) · [💬 ACL2026 (59)](../../ACL2026/model_compression/index.md) · [🧪 ICML2026 (116)](../../ICML2026/model_compression/index.md) · [🤖 AAAI2026 (60)](../../AAAI2026/model_compression/index.md) · [🧠 NeurIPS2025 (143)](../../NeurIPS2025/model_compression/index.md)

🔥 **高频主题：** 模型压缩 ×20 · LLM ×19 · 压缩/编码 ×15 · 知识蒸馏 ×4 · 对抗鲁棒 ×4

**[500xCompressor: Generalized Prompt Compression for Large Language Models](500xcompressor_generalized_prompt_compression_for_large_language_models.md)**

:   提出 500xCompressor，将最多约 500 个自然语言 token 压缩为最少 1 个特殊 token 的 KV 值，实现 6x 到 480x 的压缩比，仅增加约 0.25% 的参数，LLM 在压缩后保留 62.26%-72.89% 的原始能力，显著超越 ICAE 基线。

**[Accurate KV Cache Quantization with Outlier Tokens Tracing](accurate_kv_cache_quantization_with_outlier_tokens_tracing.md)**

:   发现 KV Cache 的 outlier channel 中存在少量异常 token 偏离先前假设的均匀分布，提出 OTT（Outlier Tokens Tracing）方法，在量化过程中动态追踪并排除这些 token，在 2-bit 量化下实现 6.4x 内存压缩和 2.3x 吞吐提升，同时显著提高精度。

**[AlignDistil: Token-Level Language Model Alignment as Adaptive Policy Distillation](aligndistil_token_level_alignment.md)**

:   AlignDistil 证明了 RLHF 目标函数与 token 级蒸馏过程的理论等价性，并据此设计了一种简单的蒸馏方法：用 DPO 模型和反向 DPO 模型的 logit 分布线性组合构造教师分布，配合 token 自适应外推机制实现 token 级奖励优化，在 AlpacaEval 2.0、MT-Bench 和 Arena-Hard 上优于现有方法且收敛更快。

**[APB: Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks across GPUs](apb_distributed_long_context.md)**

:   APB 提出了一种分布式长上下文推理框架，通过在序列并行框架中引入本地 KV cache 压缩和跨 GPU 传递压缩上下文块的机制，在不损失任务性能的前提下实现了相比 FlashAttn/RingAttn/StarAttn 分别高达 9.2x/4.2x/1.6x 的 prefill 加速。

**[Assigning Distinct Roles to Quantized and Low-Rank Matrices Toward Optimal Weight Decomposition](assigning_distinct_roles_to_quantized_and_low-rank_matrices_toward_optimal_weigh.md)**

:   提出ODLRI (Outlier-Driven Low-Rank Initialization)，为联合量化+低秩优化(Q+LR)框架中的低秩分量赋予明确角色——捕获激活异常值敏感权重，使量化分量处理更平滑的残差，在Llama2/3和Mistral的2-bit极端量化场景下持续降低困惑度和提升零样本精度。

**[Basic Reading Distillation](basic_reading_distillation.md)**

:   本文提出基础阅读蒸馏（BRD），通过让教师LLM在通用语料上生成基础阅读行为数据（包括NER和问答），训练小型学生模型模仿这些行为，使564M参数的小模型在不接触下游任务数据的情况下就能在多种NLP任务上达到或超过20倍大的教师模型性能。

**[BeamLoRA: Beam-Constraint Low-Rank Adaptation](beamlora_beam_constraint_lora.md)**

:   BeamLoRA 发现 LoRA 模块中不同 rank 的重要性存在显著差异且随训练动态演变，受 beam search 启发，提出在训练过程中动态评估 rank 重要性、剪枝不重要的 rank 并将参数空间扩展给重要 rank，在固定总 rank 下提升性能，在三个基座模型的 12 个数据集上持续优于 LoRA 及其变体。

**[Beyond Logits: Aligning Feature Dynamics for Effective Knowledge Distillation](beyond_logits_aligning_feature_dynamics_for_effective_knowledge_distillation.md)**

:   本文提出一种超越 logit 匹配的知识蒸馏方法，通过对齐教师和学生模型在训练过程中的特征变化动态（而非静态特征快照），实现更有效的知识转移，显著提升了 NLP 任务上的蒸馏效果。

**[Beyond Text Compression: Evaluating Tokenizers Across Scales](beyond_text_compression_tokenizers.md)**

:   本文系统评估了 6 种 tokenizer 在 350M 和 2.7B 参数模型上的影响，发现 tokenizer 选择对英文任务影响极小但对多语言任务（如机器翻译）有显著且跨尺度一致的影响，并提出了基于 Zipf 定律的新型内在评估指标，比文本压缩率能更好地预测多语言场景下的下游性能。

**["Give Me BF16 or Give Me Death"? Accuracy-Performance Trade-Offs in LLM Quantization](bf16_or_death_quantization_tradeoffs.md)**

:   这是迄今最全面的 LLM 量化实证研究，在 Llama-3.1 全系列（8B/70B/405B）上对 FP8/INT8/INT4 进行了超过 50 万次评估，发现 FP8 几乎无损、INT8 仅降 1-3%、INT4 出奇地有竞争力，并给出了不同部署场景的量化格式选择建议。

**[BlockPruner: Fine-grained Pruning for Large Language Models](blockpruner_fine-grained_pruning_for_large_language_models.md)**

:   提出 BlockPruner，将 Transformer 层分解为 MHA 和 MLP 两个最小残差块，基于困惑度评估块重要性并通过迭代搜索进行细粒度剪枝，实现比层级剪枝更优的压缩效果。

**[Bone Soups: A Seek-and-Soup Model Merging Approach for Controllable Multi-Objective Generation](bone_soups_multi_objective_gen.md)**

:   提出 Bone Soup 模型合并方法，通过先构造"骨架奖励"（多目标奖励的组合）训练骨架模型、再用对称循环矩阵映射确定合并系数，解决了 Rewarded Soup 中单目标模型合并的次优性问题，在三个多目标生成任务上实现更好的 Pareto 前沿和可控性。

**[BrainECHO: Semantic Brain Signal Decoding through Vector-Quantized Spectrogram Reconstruction for Whisper-Enhanced Text Generation](brainecho_semantic_brain_signal_decoding_through_vector-quantized_spectrogram_re.md)**

:   提出 BrainECHO 三阶段框架（自编码—对齐—微调），通过向量量化离散表示将脑信号映射到 Mel 频谱图空间，再借助 Whisper 完成非侵入式脑信号到文本的高质量解码。

**[CAMI: A Counselor Agent Supporting Motivational Interviewing through State Inference and Topic Exploration](cami_a_counselor_agent_supporting_motivational_interviewing_through_state_infere.md)**

:   本文提出CAMI（Counselor Agent for Motivational Interviewing），一个基于动机式访谈（MI）原则的咨询Agent，通过STAR框架（状态推断、话题探索、回复生成）来引导来访者产生改变谈话（change talk），在自动化和人工评估中均优于现有方法。

**[Capture the Key in Reasoning to Enhance CoT Distillation Generalization](capture_key_cot_distillation.md)**

:   提出 EDIT（mistakE-Driven key reasonIng step distillaTion），通过构造正确/错误配对的 dual CoTs 数据，利用最小编辑距离算法定位关键推理步骤，并以 token 级细粒度损失函数引导小模型聚焦学习这些关键步骤，而非简单模仿教师的推理形式。

**[CFSP: An Efficient Structured Pruning Framework for LLMs with Coarse-to-Fine Activation Information](cfsp_an_efficient_structured_pruning_framework_for_llms_with_coarse-to-fine_acti.md)**

:   本文提出CFSP框架，利用粗粒度（block间）和细粒度（block内）的激活信息作为重要性准则指导LLM的结构化剪枝，仅需一次前向传播即可完成剪枝，在多个模型和稀疏度预算上优于现有方法。

**[ClaimPKG: Enhancing Claim Verification via Pseudo-Subgraph Generation with Lightweight Specialized LLM](claimpkg_enhancing_claim_verification_via_pseudo-subgraph_generation_with_lightw.md)**

:   提出 ClaimPKG 框架，通过轻量级专用 LLM 将文本声明转换为伪子图表示，再从知识图谱中检索相关子图作为证据，最终由通用 LLM 进行推理验证，在 FactKG 数据集上比 SOTA 高出 9%-12% 准确率。

**[CoLA: Collaborative Low-Rank Adaptation](cola_collaborative_low-rank_adaptation.md)**

:   提出 CoLA，一种灵活的 LoRA 架构，打破矩阵 A 和 B 之间的固定数量约束（#A=M, #B=N），并设计三种协作策略（全协作/随机协作/启发式协作），结合扩展的 PiSSA 初始化，在低样本场景下显著优于现有 PEFT 方法。

**[Compact and Compressible Representations for LLMs Using Structured Sparse Decomposition](compact_and_compressible_representations_for_llms_using_structured_sparse_decom.md)**

:   本文提出一种结构化稀疏分解方法，将LLM权重矩阵分解为低秩部分和结构化稀疏部分的组合，实现高压缩比的同时保持模型性能，使大模型在资源受限环境下高效部署成为可能。

**[Compression in Transformer Language Models Has a Surprising Relationship with Performance](compression_in_transformer_language_models_has_a_surprising_relationship_with_pe.md)**

:   本文从信息论角度研究Transformer语言模型中压缩（权重的可压缩性）与模型性能之间的关系，发现了一个反直觉的现象：在一定范围内，更容易被压缩的模型反而具有更好的泛化性能，这与最小描述长度（MDL）原理的预测一致。

**[DAC: A Dynamic Attention-aware Approach for Task-Agnostic Prompt Compression](dac_prompt_compression.md)**

:   DAC 提出动态注意力感知的 prompt 压缩方法，通过融合信息熵和注意力分数作为 token 重要性度量，并动态感知压缩过程中的熵偏移来进行细粒度压缩，在 LongBench 上比 SOTA 方法提升平均 1.33 分。

**[Data Laundering: Artificially Boosting Benchmark Results through Knowledge Distillation](data_laundering_artificially_boosting_benchmark_results_through_knowledge_distil.md)**

:   本文揭示了知识蒸馏可被滥用来人为提高基准测试分数的漏洞——通过"数据洗白"（Data Laundering）方法，将教师模型在测试集上学到的知识通过看似合法的中间训练步骤隐蔽地传递给学生模型，使一个2层BERT即可在GPQA上达到73.94%（接近OpenAI o1的77.30%），而该模型并未真正学会推理。

**[DeepSolution: Boosting Complex Engineering Solution Design via Tree-based Exploration and Bi-point Thinking](deepsolution_boosting_complex_engineering_solution_design_via_tree-based_explora.md)**

:   提出面向复杂工程方案设计的新基准 SolutionBench 和新系统 SolutionRAG，通过树搜索探索+双视角思维（设计-审查交替）在 RAG 框架下逐步生成满足多约束的可靠工程方案，在 8 个工程领域达到 SOTA。

**[Direct Behavior Optimization: Unlocking the Potential of Lightweight LLMs](direct_behavior_optimization_unlocking_the_potential_of_lightweight_llms.md)**

:   提出 DeBoP 范式，将轻量级 LLM（LwLLM）的行为优化转化为对离散执行序列的优化，通过无梯度蒙特卡洛树搜索（MCTS）自动寻找最优 demonstration，使 LLaMA3-8B 在多数任务上超越 GPT-3.5 并减少约 60% 计算时间。

**[Disentangling the Roles of Representation and Selection in Data Pruning](disentangling_the_roles_of_representation_and_selection_in_data_pruning.md)**

:   本文将数据剪枝（data pruning）系统性地拆解为"数据表示"和"选择算法"两个独立维度，通过理论分析和大规模实验发现：表示质量（尤其是训练梯度）对剪枝效果起决定性作用，而不同选择算法在不同场景下各有优劣，且常常偏离其设计目标。

**[DoMIX: An Efficient Framework for Exploiting Domain Knowledge in Fine-Tuning](domix_an_efficient_framework_for_exploiting.md)**

:   提出 DoMIX，将各领域知识用独立 LoRA 模块存储后通过对角初始化的 bridge 矩阵在微调时灵活组合利用，在持续领域适应预训练场景下减少 58% 预训练时间和 87% GPU 内存，同时性能超越 SOTA。

**[DRPruning: Efficient Large Language Model Pruning through Distributionally Robust Optimization](drpruning_robust_pruning.md)**

:   DRPruning 将分布稳健优化（DRO）引入 LLM 结构化剪枝，通过 scaling law 预测各领域最终 loss 作为参考、动态调整训练数据分布来平衡剪枝后各领域性能，在单语和多语设置下分别以 -5.59% PPL 和 +2.95% 下游任务的提升超越 Sheared LLaMA。

**[EAC-MoE: Expert-Selection Aware Compressor for Mixture-of-Experts Large Language Models](eac_moe_expert_aware_compression.md)**

:   EAC-MoE 深入分析 MoE 模型的专家选择特性，提出两个互补模块——量化时通过逐层校准路由器缓解 expert-shift 问题（QESC），推理时基于专家选择频率动态剪枝不重要专家（PESF），在 4 个 MoE 模型上实现显著的内存压缩和推理加速且精度损失极小。

**[Efficient Long Context Language Model Retrieval with Compression](efficient_long_context_language_model_retrieval_with_compression.md)**

:   提出 CoLoR（Compression for Long context Retrieval），通过偏好优化和长度正则化联合训练段落压缩模型，在保持长上下文语言模型检索性能提升 6% 的同时将上下文长度压缩 1.91 倍。

**[Efficient OpAmp Adaptation for Zoom Attention to Golden Contexts](efficient_opamp_adaptation_for_zoom_attention_to_golden_contexts.md)**

:   受运算放大器（OpAmp）电路启发，提出 OpAmp Adaptation 方法通过 adapter 高效改造预训练 Transformer 的注意力机制，在噪声上下文场景下让 LLM 更精准聚焦于 golden document，Qwen2.5-OpAmp-72B 在多个噪声上下文基准上超越 DeepSeek-V3 和 GPT-4o。

**[EfficientQAT: Efficient Quantization-Aware Training for Large Language Models](efficientqat.md)**

:   EfficientQAT 提出两阶段 QAT 框架——先逐块训练所有参数（Block-AP）提供良好初始化，再端到端训练量化参数（E2E-QP）捕获跨块交互，在单张 A100 上 41 小时完成 Llama-2-70B 的 2-bit 量化，精度仅降 3 点。

**[Explaining Puzzle Solutions in Natural Language: An Exploratory Study on 6×6 Sudoku](explaining_puzzle_solutions_in_natural_language_an_exploratory_study_on_6x6_sudo.md)**

:   评估五个LLM在求解和解释6×6数独谜题上的能力，发现即使o1-preview能解出65%的题目，其推理解释在忠实性、清晰度和教育价值方面仍严重不足。

**[Towards Robust and Efficient Federated Low-Rank Adaptation with Heterogeneous Clients](federated_lora_heterogeneous.md)**

:   提出 LoRA-A²（Low Rank Adaptation with Alternating freeze and Adaptive rank selection），通过交替冻结 A/B 矩阵解决联邦 LoRA 聚合不一致问题，并结合自适应秩选择机制在大幅压缩上传参数量（最高减少 99.8%）的同时保持鲁棒性，尤其在低秩+高数据异构场景下显著优于现有方法。

**[FedEx-LoRA: Exact Aggregation for Federated and Efficient Fine-Tuning of Large Language Models](fedex_lora_federated_exact_aggregation.md)**

:   FedEx-LoRA 发现联邦学习中独立平均 LoRA 的 A 和 B 矩阵会导致不精确的全局更新（"乘积的均值≠均值的乘积"），通过在冻结权重矩阵中加入残差误差项实现精确聚合，在多个推理和 NLU 任务上一致优于 FedIT 和 FFA-LoRA。

**[Flipping Knowledge Distillation: Leveraging Small Models' Expertise to Enhance LLMs in Text Matching](flipping_kd_small_to_large.md)**

:   本文提出"反向知识蒸馏"范式——让 LLM 从微调过的小模型学习文本匹配的领域专家知识，通过将 decoder-only LLM 重新解释为 encoder-decoder 架构（用 LoRA 的压缩矩阵做 encoder）并设计 Margin-aware Contrastive Loss 来对齐表示相似度。

**[A Silver Bullet or a Compromise for Full Attention? A Comprehensive Study of Gist Token-based Context Compression](gist_token_context_compression.md)**

:   对基于 Gist Token 的上下文压缩方法进行全面系统研究，发现细粒度 KV Cache 架构在 RAG/QA 等任务上接近无损，但在精确回忆任务上存在明显差距，并识别出三种关键失败模式和两种有效改进策略。

**[Graph-guided Cross-composition Feature Disentanglement for Compositional Zero-shot Learning](graph-guided_cross-composition_feature_disentanglement_for_compositional_zero-sh.md)**

:   DCDA 提出图引导的跨组合特征解耦方案，通过双适配器（L-Adapter 用于文本端 GNN 特征聚合、V-Adapter 用于视觉端跨注意力解耦）注入冻结 CLIP，在组合零样本学习任务上显著超越现有方法。

**[GSQ-Tuning: Group-Shared Exponents Integer in Fully Quantized Training for LLMs On-Device Fine-tuning](gsq-tuning_group-shared_exponents_integer_in_fully_quantized_training_for_llms_o.md)**

:   GSQ-Tuning 提出了一种基于"组共享指数整数"（Group-Shared Exponents Integer）格式的全量化微调框架，在推理和训练中完全消除浮点运算，结合 LoRA 适配器实现了精度接近 BF16 微调、内存降低 1.85x、功耗降低 5x、芯片面积缩小 11x 的边端 LLM 微调方案。

**[IAM: Efficient Inference through Attention Mapping between Different-scale LLMs](iam_efficient_inference_through_attention_mapping_between_different-scale_llms.md)**

:   发现不同规模 LLM 的注意力矩阵具有高度相似性，提出 IAM 框架——在 prefill 阶段建立小模型与大模型注意力头之间的余弦相似度映射，decode 阶段用小模型的注意力矩阵替代大模型映射层的注意力计算，实现 KV cache 减少 22% 和推理加速 11%，且与现有 KV cache 压缩方法正交。

**[L4Q: Parameter Efficient Quantization-Aware Fine-Tuning on Large Language Models](l4q_parameter_efficient_quantization_aware_finetuning.md)**

:   提出 L4Q，将量化感知训练 (QAT) 与 LoRA 深度整合：先合并权重与LoRA参数再统一量化，通过定制反向传播路径消除权重梯度存储开销，实现联合优化量化与微调参数，在4-bit和3-bit量化下显著超越现有方法。

**[Language Models Resist Alignment: Evidence From Data Compression](language_models_resist_alignment.md)**

:   本文从压缩理论视角提出LLM的"弹性"(elasticity)概念，证明模型在受到微调扰动时压缩率变化与数据集大小成反比——因为预训练数据远大于对齐数据，对齐效果被优先"遗忘"，这从信息论角度根本性地解释了为什么LLM对齐如此脆弱。

**[Unveiling Language-Specific Features in Large Language Models via Sparse Autoencoders](language_specific_features.md)**

:   利用 Sparse Autoencoders (SAEs) 分析多语言 LLM 的内部表示，发现存在强烈的语言特定 SAE features，这些 features 不仅与语言特有 token 相关还与语言上下文相关，消融它们只影响对应语言能力，且多个语言 features 之间存在协同效应；进一步利用这些 features 增强 steering vectors 实现对生成语言的精确控制。

**[Towards the Law of Capacity Gap in Distilling Language Models](law_of_capacity_gap_distilling_language_models.md)**

:   揭示了语言模型蒸馏中的"容量差距定律"——最优教师模型的参数量与学生模型成线性关系（约 2.5 倍），将 LLM 蒸馏中的"不可能三角"转化为可解问题，并据此成功蒸馏出 3B 的 MiniMA 模型。

**[Limited-Resource Adapters Are Regularizers, Not Linguists](limited-resource_adapters_are_regularizers_not_linguists.md)**

:   本文将 adapter souping（权重平均）与交叉注意力微调结合用于低资源克里奥尔语机器翻译，发现虽然方法带来了显著提升（最高 +8 BLEU），但语言关联性与 adapter 性能无有意义的协变关系——随机初始化的未训练 adapter 表现同样优秀，表明 adapter 在此设定下的作用本质是**参数正则化而非语言信息迁移**。

**[LLMSR@XLLM25: Less is More: Enhancing Structured Multi-Agent Reasoning via Quality-Guided Distillation](llmsrxllm25_less_is_more_enhancing_structured_multi-agent_reasoning_via_quality-.md)**

:   本文提出 Less is More 框架，在仅有 24 个标注样本的极端低资源条件下，通过逆向提示归纳、GPT-4o 增强的检索式推理合成和双阶段奖励引导过滤三个阶段，蒸馏出高质量的结构化推理数据来微调 LLaMA3-8B 多智能体系统，在 XLLM@ACL2025 共享任务中获得第三名。

**[LongReD: Mitigating Short-Text Degradation of Long-Context Large Language Models via Restoration Distillation](longred_mitigating_short-text_degradation_of_long-context_large_language_models_.md)**

:   本文系统分析了长上下文LLM在短文本任务上性能退化的两个原因（分布漂移和灾难性遗忘），并提出LongReD方法，通过短文本蒸馏和短到长蒸馏两个训练目标来最小化扩展模型与原始模型之间的分布差异，在保持长文本建模能力的同时将短文本性能保留至原始模型的99.4%。

**[Low-Rank Interconnected Adaptation across Layers](low-rank_interconnected_adaptation_across_layers.md)**

:   提出 Lily（Low-rank Interconnected Adaptation across Layers），通过将 LoRA 的 A/B 适配器跨层解耦并互联共享，配合数据依赖的路由机制，在相同或更少参数下实现高秩权重更新，在多模态、多架构、多规模场景中均优于 LoRA。

**[MaCP: Minimal yet Mighty Adaptation via Hierarchical Cosine Projection](macp_minimal_yet_mighty_adaptation_via_hierarchical_cosine_projection.md)**

:   本文提出 MaCP——一种基于离散余弦变换（DCT）的参数高效微调方法，通过将权重变化投影到余弦频域并分层选择最关键的频率分量，在极低参数量（比 LoRA 少 99.7%）下实现了优于或媲美现有 PEFT 方法的性能。

**[Magnet: Multi-turn Tool-use Data Synthesis and Distillation via Graph Translation](magnet_multi-turn_tool-use_data_synthesis_and_distillation_via_graph_translation.md)**

:   提出 Magnet 框架，基于函数依赖图的随机游走和节点操作（Insert/Merge/Split）构建高质量多轮 Function Calling 训练轨迹，结合基于提示的上下文蒸馏生成正负对比轨迹进行 SFT + mDPO 训练，使 14B 模型 Magnet-14B-mDPO 在 BFCL-v3 上达到 68.01（排名第 4），在多轮场景上大幅超越教师模型 Gemini-1.5-pro-002。

**[Memorization: A Close Look at Books](memorization_a_close_look_at_books.md)**

:   系统研究 Llama 3 系列模型对完整书籍的记忆化程度，发现书籍提取率与其流行度（训练数据重复度代理）高度正相关，并通过 LoRA 微调揭示指令微调的抗反刍缓解措施仅涉及极少量集中在底层 transformer block 的权重变化。

**[MoQAE: Mixed-Precision Quantization for Long-Context LLM Inference via Mixture of Quantization-Aware Experts](moqae_mixed_precision_kv_cache.md)**

:   MoQAE 创造性地将不同量化比特宽度配置视为 MoE 中的"专家"，通过轻量路由器学习每个 chunk 的最优量化策略，结合路由冻结和路由共享机制，在几乎不损失精度的情况下大幅减少长上下文推理的 KV cache 内存。

**[MoRE: A Mixture of Low-Rank Experts for Adaptive Multi-Task Learning](more_a_mixture_of_low-rank_experts_for_adaptive_multi-task_learning.md)**

:   提出 MoRE (Mixture of Low-Rank Experts)，将 LoRA 中的不同秩视为不同专家，通过自适应秩选择器为每个任务动态选择最合适的秩，配合对比学习优化的任务嵌入和平衡数据采样策略，使用单个 LoRA 模块实现高效的多任务微调。

**[mPLUG-DocOwl2: High-resolution Compressing for OCR-free Multi-page Document Understanding](mplug_docowl2_doc_compress.md)**

:   提出布局感知的High-resolution DocCompressor模块，用全局低分辨率视觉特征作为query、子图特征作为key/value进行分组交叉注意力，将每张高分辨率文档图片从数千tokens压缩至324 tokens，配合三阶段训练框架在多页文档理解上达到SOTA且First Token Latency降低50%以上。

**[One QuantLLM for ALL: Fine-tuning Quantized LLMs Once for Efficient Deployments](one_quantllm_for_all_fine-tuning_quantized_llms_once_for_efficient_deployments.md)**

**[Unraveling LoRA Interference: Orthogonal Subspaces for Robust Model Merging](osrm_lora_merging_orthogonal.md)**

:   OSRM 发现 LoRA 模型合并失败的根因是参数与数据分布的交互干扰（而非仅仅是参数冲突），提出在微调前通过数据协方差矩阵的特征分解来初始化 LoRA 矩阵 A，使其子空间与其他任务的数据分布正交，从而在合并时最小化跨任务干扰，在 8 个数据集、5 个模型上显著提升合并性能。

**[Outlier-Safe Pre-Training for Robust 4-Bit Quantization of Large Language Models](outlier-safe_pre-training_for_robust_4-bit_quantization_of_large_language_models.md)**

:   OSP（Outlier-Safe Pre-Training）框架通过三项创新——Muon 优化器（消除特权基方向）、Single-Scale RMSNorm（防止通道放大）和可学习嵌入投影层（重分布嵌入层激活），在预训练阶段主动防止异常值形成，训练的 1.4B 模型在 1T tokens 上实现近零超额峰度（0.04 vs 标准模型的 1818.56），在激进4-bit量化下平均分 35.7（Adam 为 26.5），仅 2% 训练开销。

**[C3A: Parameter-Efficient Fine-Tuning via Circular Convolution](parameter-efficient_fine-tuning_via_circular_convolution.md)**

:   提出 C3A 方法用循环卷积算子替代 LoRA 的低秩矩阵分解实现参数高效微调，核心优势是矩阵秩与参数量解耦——可用少量参数实现高秩适配，同时通过 FFT 保持与 LoRA 相当的计算和内存效率，在多种微调任务上一致超越 LoRA 及其变体。

**[Pre-training Distillation for Large Language Models: A Design Space Exploration](pre-training_distillation_for_large_language_models_a_design_space_exploration.md)**

:   系统性地探索大语言模型预训练蒸馏（Pre-training Distillation）的设计空间，从 logits 处理、损失函数选择、scaling law 和 offline/online logits 四个维度进行广泛实验，找到更优配置并得出有价值的结论。

**[Predicting Through Generation: Why Generation Is Better for Prediction](predicting_through_generation_why_generation_is_better_for_prediction.md)**

:   本文从信息论角度证明了token级生成比pooled表示保留更多互信息，提出PredGen框架通过scheduled sampling和task adapter解决生成式预测中的exposure bias和格式不匹配问题，并设计了Writer-Director Alignment Loss统一生成与预测目标。

**[Prompt Candidates, then Distill: A Teacher-Student Framework for LLM-driven Data Annotation](prompt_distill_teacher_student.md)**

:   提出CanDist框架，借鉴人类面对不确定性时的"模糊规避"心理，引导LLM输出多个候选标签而非单一标签(候选标注)，再通过分布精炼(Distribution Refinery)策略蒸馏到小语言模型(SLM)获得最终标注，从理论到实验证明候选标注蒸馏优于单一标注。

**[PTQ1.61: Push the Real Limit of Extremely Low-Bit Post-Training Quantization Methods for Large Language Models](ptq161_low_bit_quantization.md)**

:   提出 PTQ1.61，首个将 LLM 权重有效压缩到真正 sub-2-bit（1.61-bit）的后训练量化方法，通过一维结构化掩码（仅增加 0.0002-bit 开销）、分块缩放因子优化和量化预处理三项技术实现 SOTA 性能。

**[Quaff: Quantized Parameter-Efficient Fine-Tuning under Outlier Spatial Stability Hypothesis](quaff_quantized_peft.md)**

:   本文提出 Outlier Spatial Stability Hypothesis (OSSH)——微调期间激活异常通道的空间位置保持稳定——并基于此设计了 Quaff 框架，通过目标动量缩放仅处理少量不变的异常通道，实现 1.73× 延迟降低和 30% 内存节省，同时在 GPQA 上精度还提升了 0.6%。

**[Quantification of Large Language Model Distillation](quantification_of_large_language_model_distillation.md)**

:   本文提出了两种互补的LLM蒸馏量化方法——身份一致性评估（ICE）和响应相似性评估（RSE），通过越狱攻击挖掘模型身份信息泄露和多粒度响应相似性来衡量模型的蒸馏程度，发现大多数知名LLM（除Claude、Doubao和Gemini外）都表现出较高的蒸馏程度。

**[Revisiting LoRA through the Lens of Parameter Redundancy: Spectral Encoding Helps](revisiting_lora_through_the_lens_of_parameter_redundancy_spectral_encoding_helps.md)**

:   本文系统研究了 LoRA 微调中的参数冗余问题，发现降低密度冗余不会损害表达能力（稀疏性质），并提出 SeLoRA——利用频谱变换（Fourier/Wavelet）从稀疏频谱子空间重参数化 LoRA 矩阵，以更少参数实现更优性能，且可即插即用地集成到多种 LoRA 变体中。

**[Spectra 1.1: Scaling Laws and Efficient Inference for Ternary Language Models](scaling_laws_and_efficient_inference_for_ternary_language_models.md)**

:   本文系统研究三值语言模型（TriLM）的缩放规律，发现 TriLM 从增加训练数据中获益远大于增加参数量，基于此训练了在 1.2T token 上预训练的 Spectra-1.1 模型族（1B/2B/3B），并提出 1.6-bit 和 2-bit 权重打包方案及 TriRun GPU 内核，实现最高 8 倍的推理加速。

**[Sci-LoRA: Mixture of Scientific LoRAs for Cross-Domain Lay Paraphrasing](sci-lora_mixture_of_scientific_loras_for_cross-domain_lay_paraphrasing.md)**

:   提出 Sci-LoRA——一种混合多领域 LoRA 的框架，通过对比学习训练文本编码器+动态权重生成器+LoRA 融合模块，在无需领域标签的情况下实现跨12个学科领域的科学文本通俗化改写，在5个数据集10个指标上超越 SOTA。

**[SCOPE: Optimizing Key-Value Cache Compression in Long-context Generation](scope_optimizing_key-value_cache_compression_in_long-context_generation.md)**

:   提出 SCOPE 框架，针对长上下文生成任务分别优化 prefill 和 decoding 阶段的 KV 缓存压缩策略——prefill 阶段保留完整缓存以维持理解能力，decoding 阶段采用滑动窗口选择 heavy hitters，并通过 adaptive 和 discontinuous 策略进一步优化内存和传输效率。

**[Mitigating Selection Bias with Node Pruning and Auxiliary Options](selection_bias_node_pruning.md)**

:   提出 Bias Node Pruning (BNP) 和 Auxiliary Option Injection (AOI) 两种互补方法，通过定位并剪除模型输出层中 0.002% 的偏差参数（白盒）与注入"I don't know"辅助选项（黑盒通用），从内外两端同时缓解 LLM 在多选题中的选择偏差，同时提出分布级偏差度量 CKLD，组合方法在 Llama-3 上将 ARC-Challenge 准确率从 52.3% 提升至 65.3%。

**[Sparse Logit Sampling: Accelerating Knowledge Distillation in LLMs](sparse_logit_sampling_accelerating_knowledge_distillation_in_llms.md)**

:   证明了朴素的 Top-K 稀疏知识蒸馏会产生有偏估计，提出基于重要性采样的 Random Sampling Knowledge Distillation (RSKD) 方法，提供无偏梯度估计，仅需存储极度稀疏的 logits，训练开销仅比交叉熵增加不到 10%，性能与全量蒸馏持平。

**[State-offset Tuning: State-based Parameter-Efficient Fine-Tuning for State Space Models](state_offset_tuning_ssm_peft.md)**

:   针对 SSM（如 Mamba）提出 State-offset Tuning，一种新的"状态基"PEFT 方法家族，通过在每个时间步直接注入可训练的状态偏移量 $h'$ 替代 Prefix-Tuning 的虚拟 token，解决了 prompt-based 方法在 SSM 上表达能力受限的问题，在更少参数量下持续优于 LoRA 和 Prefix-Tuning。

**[STUN: Structured-Then-Unstructured Pruning for Scalable MoE Pruning](stun_moe_pruning.md)**

:   STUN 提出"先结构化后非结构化"的两阶段 MoE 剪枝范式：第一阶段利用路由权重的行为相似性聚类冗余专家，以 $O(1)$ GPU 前向传播完成专家级剪枝；第二阶段在剩余专家内做非结构化权重剪枝，两者协同在 480B Snowflake Arctic 上以 40% 稀疏度几乎无性能损失。

**[TaDA: Training-free recipe for Decoding with Adaptive KV Cache Compression and Mean-centering](tada_training-free_recipe_for_decoding_with_adaptive_kv_cache_compression_and_me.md)**

:   提出 TaDA——无需训练的 KV cache 压缩方法，通过对 K/V 激活做 head 维度均值中心化后量化偏差（而非原始激活），自动消除离群值问题，配合逐层自适应量化精度搜索，将 KV cache 压缩至原始 16 位的 27% 同时保持接近基线的精度。

**[TeamLoRA: Boosting Low-Rank Adaptation with Expert Collaboration and Competition](teamlora_boosting_low-rank_adaptation_with_expert_collaboration_and_competition.md)**

:   提出 TeamLoRA，通过非对称协作模块（共享A矩阵+多个专家B矩阵的"插件式"组织）和基于Shapley值的竞争模块来优化Multi-LoRA架构，在多任务学习中实现了更好的效果-效率平衡——训练时间比MoELoRA减少30%，推理速度提升40%，同时性能更优。

**[Trans-PEFT: Transferable Parameter-Efficient Fine-Tuning on Evolving Base Models](trans_peft_transferable.md)**

:   Trans-PEFT 发现基座模型更新（如 Qwen2→Qwen2.5）主要改变 FFN 层的任务知识存储而较少影响 Attention 层的任务模式，据此提出层内知识掩码和跨层知识丢弃两种策略，使在旧版本上训练的 PEFT 模块可直接迁移到新版本而不需重新微调，性能提升可达 30%。

**[UniICL: An Efficient ICL Framework Unifying Compression, Selection, and Generation](uniicl_icl_framework.md)**

:   提出 UniICL 框架，用**一个冻结的 LLM** 同时完成 demonstration 压缩（compress→virtual tokens）、demonstration 选择（基于压缩后的 virtual token 相似度排序）和最终响应生成三个任务，仅需 17M 可训练参数（projection layer + learnable embedding），配合 Demonstration Bank 缓存机制避免重复压缩，实现 12× 压缩率下从 4-shot 扩展到 64-shot ICL（24GB 显存内），在多个 out-of-domain 数据集上超越 AutoCompressor、ICAE、LLMLingua 等基线。

**[UniQuanF: Unifying Uniform and Binary-coding Quantization for Accurate Compression of Large Language Models](uniquanf_unified_quantization.md)**

:   UniQuanF 统一了均匀量化（UQ,表现力弱但优化性强）和二进制编码量化（BCQ,表现力强但优化性差）的优势，通过统一初始化、局部周期映射和统一定理，实现无额外部署开销的高精度 LLM 量化，在 GSM8K 上提升最高 4.60%。

**[Wanda++: Pruning Large Language Models via Regional Gradients](wanda_pruning_large_language_models_via_regional_gradients.md)**

:   提出 Wanda++——基于 decoder block 级别区域梯度的轻量级 LLM 剪枝框架，通过区域梯度评分（RGS）改进剪枝准则 + 区域优化（RO）最小化稠密/稀疏块输出差异，在 2:4 稀疏下 WikiText 困惑度较 Wanda 最高降低 32%，单 H100 GPU 10 分钟内完成 7B 模型剪枝。

**[Who Taught You That? Tracing Teachers in Model Distillation](who_taught_you_that_tracing_teachers_in_model_distillation.md)**

:   本文提出"教师模型归因"新问题：给定一个蒸馏后的学生模型，能否从候选教师中识别出其训练教师？发现 n-gram 相似度和困惑度不可靠，但词性（PoS）句法模板能提供有效的教师识别信号。
