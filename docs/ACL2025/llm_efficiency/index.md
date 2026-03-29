<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚡ LLM 效率

**💬 ACL2025** · 共 **21** 篇

**[Extending LLM Context Window with Adaptive Grouped Positional Encoding: A Training-Free Method](adaptive_grouped_pe_context_window.md)**

:   提出 AdaGroPE（Adaptive Grouped Positional Encoding），一种无需训练的即插即用方法，通过让位置复用次数随距离递增式增长、并根据输入序列长度动态调整位置编码映射，将 LLM 上下文窗口外推到远超预训练长度，在多个 benchmark 上达到 SOTA 甚至超过原生长上下文模型。

**[CLaSp: In-Context Layer Skip for Self-Speculative Decoding](clasp_self_speculative_decoding.md)**

:   CLaSp 提出一种无需训练的自推测解码方法，通过动态规划算法在每个验证步骤后根据上下文动态调整跳层策略，利用上一次验证的完整隐状态作为目标来选择最优跳层集合，在 LLaMA3 系列上实现 1.3-1.7× 加速且不改变生成分布。

**[CNNSum: Exploring Long-Context Summarization with Large Language Models in Chinese Novels](cnnsum_exploring_long-context_summarization_with_large_language_models_in_chines.md)**

:   构建了 CNNSum——基于中文小说的多尺度长文本摘要基准（695 样本，16k-128k tokens），通过人工标注确保质量，系统测评了 20+ 个 LLM，发现高级 LLM 倾向生成主观评述导致摘要模糊、小模型性价比更高、Base 版微调效果优于 Chat 版，且用短文本数据微调即可显著提升长文本摘要能力。

**[Dynamic Chunking and Selection for Reading Comprehension of Ultra-Long Context in Large Language Models](dynamic_chunking_and_selection_for_reading_comprehension_of_ultra-long_context_i.md)**

:   提出 Dynamic Chunking and Selection (DCS)，通过基于语义相似度的动态分块和问题感知分类器的块选择，解决长文本固定分块导致的语义断裂问题，在 12 个长文本 QA 数据集上以 Llama3 为基座实现 single-hop 平均 35.50（+28.6%）和 multi-hop 平均 29.07（+20.0%）的提升，且在 256k token 输入下保持鲁棒。

**[FlashBack: Efficient Retrieval-Augmented Language Modeling for Fast Inference](flashbackefficient_retrieval-augmented_language_modeling_for_long_context_infere.md)**

:   针对检索增强语言模型(RALM)中因检索内容前置(prepending)导致 KV cache 反复重算的推理效率问题，提出 FlashBack，将检索内容后置(appending)以保留输入的 KV cache，并用 Marking Token + LoRA 微调适配新的上下文模式，在 Llama 2-7B 上实现最高 4 倍推理加速且 perplexity 持平。

**[GigaChat Family: Efficient Russian Language Modeling Through Mixture of Experts Architecture](gigachat_family_efficient_russian_language_modeling_through_mixture_of_experts_a.md)**

:   介绍 GigaChat 系列——首个从头为俄语设计并预训练的 MoE 架构 LLM 家族，包含 20B 总参数/3.3B 激活参数的基座和指令微调模型，在俄语 benchmark 上达到同规模 SOTA，训练速度是同量级 dense 模型的 2 倍，推理延迟降低 40%。

**[Graph of Records: Boosting Retrieval Augmented Generation for Long-context Summarization with Graphs](gor_rag_long_context_summary.md)**

:   提出 Graph of Records（GoR），将 LLM 历史响应与检索文本块构建为图结构，用 GNN 学习节点间的语义和逻辑关联，配合 BERTScore 自监督训练目标，在四个长文本全局摘要数据集上比检索基线提升 8-19%（ROUGE 指标）。

**[GradOT: Training-free Gradient-preserving Offsite-tuning for Large Language Models](gradot_offsite_tuning.md)**

:   从优化理论角度首次系统分析 Offsite-tuning 问题，提出梯度保持压缩分数（GCS），并基于此设计了 GradOT 方法，对 MHA 使用动态秩分解（DRD）、对 MLP 使用选择性通道剪枝（SCP），在免训练条件下同时实现性能保持和隐私保护。

**[KV-Latent: Dimensional-level KV Cache Reduction with Frequency-aware Rotary Positional Embedding](kv_latent_cache_reduction.md)**

:   提出KV-Latent范式，通过对KV头的维度进行降采样将其映射到潜空间，仅需不到1%预训练量的额外训练即可恢复性能，在LLaMA-3-8B上KV Cache减少50%（dqk=dvo=64→50%缓存），同时TTFT延迟降低8%。

**[LADM: Long-context Training Data Selection with Attention-based Dependency Measurement](ladm_long_context_data.md)**

:   提出 LADM 框架，利用注意力机制的内在检索能力来度量长上下文数据中的跨 span 依赖关系，从大规模预训练语料中高效筛选高质量长上下文训练数据，仅用 1B token 持续预训练即可显著提升多种 LLM 的长上下文能力。

**[Literary Evidence Retrieval via Long-Context Language Models](literary_evidence_retrieval_via_long-context_language_models.md)**

:   构建文学证据检索 benchmark，要求模型给定完整小说文本和文学评论摘录后生成缺失的引用，Gemini Pro 2.5 达 62.5% 准确率超过人类专家(55%)，但最佳开源模型仅 29.1%，揭示了巨大能力差距。

**[What Really Matters in Many-Shot Attacks? An Empirical Study of Long-Context Vulnerabilities in LLMs](many_shot_attacks_long_context.md)**

:   系统分析 Many-Shot Jailbreaking（MSJ）攻击的关键因素，发现上下文长度是攻击成功的决定性因素，而内容的有害性、主题、格式几乎不重要——即使重复安全内容、随机无意义文本（Lorem Ipsum）都能在长上下文下突破模型安全对齐。

**[Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](native_sparse_attention.md)**

:   DeepSeek提出NSA——一种原生可训练的稀疏注意力机制，通过"压缩+选择+滑动窗口"的层次化稀疏策略和硬件对齐的Triton kernel设计，在27B参数模型上实现了超越Full Attention的性能，同时在64k序列上获得前向9倍、解码11.6倍的加速。

**[On Many-Shot In-Context Learning for Long-Context Evaluation](on_many-shot_in-context_learning_for_long-context_evaluation.md)**

:   深入研究 many-shot ICL 用于长上下文语言模型评估，提出 Sample Learning Ratio 指标区分 SSL 和 ASL 任务，构建 ManyICLBench 基准全面评测 12 个 LCLM。

**[Ref-Long: Benchmarking the Long-Context Referencing Capability of Long-Context Language Models](ref-long_benchmarking_the_long-context_referencing_capability_of_long-context_la.md)**

:   提出 Ref-Long benchmark，从"引用定位"（给定 key 识别哪些文档引用了它并返回索引）这一被忽视的维度评估长上下文模型，包含 3 个子集（合成→真实）共 4300 个任务；发现即使 GPT-4o 在 Multi-Hard-24K 上 ExAcc 仅 19%，远低于人类 92%，且 prompt 工程和专项微调均无法根本解决该问题。

**[MegaBeam: Scaling Context, Not Parameters](scaling_context_not_parameters_training_a_compact_7b_language_model_for_efficien.md)**

:   提出四阶段继续预训练策略将 Mistral-7B 的上下文长度扩展到 512K，7B 模型在 RULER-128K 上超越 GPT-4-1106 和 Llama-3.1-70B，是首个在 512K BABILong 上不用 RAG 就达到 35% 的开源模型。

**[SEAL: Scaling to Emphasize Attention for Long-Context Retrieval](seal_scaling_to_emphasize_attention_for_long-context_retrieval.md)**

:   通过剪枝分析发现特定注意力头与长上下文检索高度相关，提出 SEAL（可学习标量/向量缩放注意力头/通道），仅需 ~1024 个额外参数 + 50 个合成样本训练，即可将 LongChat-7B 在 31K 上的检索准确率从 0.32 提升到 0.88，且缩放参数可离线合并到权重中实现零推理开销。

**[Sliding Windows Are Not the End: Exploring Full Ranking with Long-Context Large Language Models](sliding_windows_full_ranking.md)**

:   本文系统研究了长上下文LLM在段落排序中的应用，提出用 full ranking（一次性排序所有段落）替代传统滑动窗口策略，并设计了多轮滑动窗口标签构造方法和重要性感知损失函数来微调 full ranking 模型，在效率提升约30-65%的同时实现了排序效果的全面超越。

**[SpindleKV: A Novel KV Cache Reduction Method Balancing Both Shallow and Deep Layers](spindlekv_layered_kv_cache.md)**

:   SpindleKV 提出分层处理 KV cache 压缩的策略——深层使用注意力驱动的 token eviction（利用稀疏注意力），浅层使用基于相似性学习的 codebook 替换（利用 token 间高相似度），并解决了 GQA 兼容性问题，实现 50% KV cache 缩减而不损失性能。

**[How to Train Long-Context Language Models (Effectively)](train_long_context_effectively.md)**

:   本文系统研究了如何通过持续预训练和监督微调（SFT）有效训练长上下文语言模型，提出了包括数据配比、训练长度缩放等一系列关键发现，最终训练出的 ProLong-8B 模型仅用 Llama-3.1 5% 的长上下文训练数据量即在 128K 长度上达到同规模最优性能。

**[What are the Essential Factors in Crafting Effective Long Context Multi-Hop Instruction Datasets? Insights and Best Practices](what_are_the_essential_factors_in_crafting_effective_long_context_multi-hop_inst.md)**

:   提出多智能体交互式多跳生成（MIMG）框架，通过质量验证、单跳问题生成、多问题采样和多跳合并四个模块，系统性地合成高质量长上下文多跳指令数据，训练后模型平均提升7.54%，甚至超越更大规模人工标注数据集。
