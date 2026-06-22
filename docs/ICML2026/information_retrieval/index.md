---
title: >-
  ICML2026 信息检索/RAG论文汇总 · 26篇论文解读
description: >-
  26篇ICML2026的信息检索/RAG 方向论文解读，涵盖 RAG、对抗鲁棒、对齐/RLHF等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2026"
  - "信息检索/RAG"
  - "论文解读"
  - "论文笔记"
  - "RAG"
  - "对抗鲁棒"
  - "对齐/RLHF"
item_list:
  - u: "blitzrank_principled_zero-shot_ranking_agents_with_tournament_graphs/"
    t: "BlitzRank: Principled Zero-shot Ranking Agents with Tournament Graphs"
  - u: "care_class-adaptive_expert_consensus_for_reliable_learning_with_long-tailed_nois/"
    t: "CARE: Class-Adaptive Expert Consensus for Reliable Learning with Long-Tailed Noisy Labels"
  - u: "graph-r1_towards_agentic_graphrag_framework_via_end-to-end_reinforcement_learnin/"
    t: "Graph-R1: Towards Agentic GraphRAG Framework via End-to-end Reinforcement Learning"
  - u: "hgmem_hypergraph-based_working_memory_to_improve_multi-step_rag_for_long-context/"
    t: "HGMem: Hypergraph-based Working Memory to Improve Multi-step RAG for Long-Context Complex Relational Modeling"
  - u: "hierarchical_abstract_tree_for_cross-document_retrieval-augmented_generation/"
    t: "Hierarchical Abstract Tree for Cross-Document Retrieval-Augmented Generation"
  - u: "how_can_embedding_models_bind_concepts/"
    t: "How can embedding models bind concepts?"
  - u: "lare_low-attention_region_encoding_for_text-image_retrieval/"
    t: "LARE: Low-Attention Region Encoding for Text–Image Retrieval"
  - u: "lazyattention_efficient_retrieval-augmented_generation_with_deferred_positional_/"
    t: "LazyAttention: Efficient Retrieval-Augmented Generation with Deferred Positional Encoding"
  - u: "lemur_learned_multi-vector_retrieval/"
    t: "LEMUR: Learned Multi-Vector Retrieval"
  - u: "less_is_more_elevating_rag_via_performance-driven_context_compression/"
    t: "Less Is More: Elevating RAG via Performance-Driven Context Compression"
  - u: "linguistic_nepotism_trading-off_quality_for_language_preference_in_multilingual_/"
    t: "Linguistic Nepotism: Trading-off Quality for Language Preference in Multilingual RAG"
  - u: "ml-embed_inclusive_and_efficient_embeddings_for_a_multilingual_world/"
    t: "ML-Embed: Inclusive and Efficient Embeddings for a Multilingual World"
  - u: "pariskv_fast_and_drift-robust_kv-cache_retrieval_for_long-context_llms/"
    t: "ParisKV: Fast and Drift-Robust KV-Cache Retrieval for Long-Context LLMs"
  - u: "predictive_prefetching_for_retrieval-augmented_generation/"
    t: "Predictive Prefetching for Retrieval-Augmented Generation"
  - u: "ranking_free_rag_replacing_re-ranking_with_selection_in_rag_for_sensitive_domain/"
    t: "Ranking-Free RAG: Replacing Re-Ranking with Selection in RAG for Sensitive Domains"
  - u: "real_resolving_knowledge_conflicts_in_knowledge-intensive_visual_question_answer/"
    t: "REAL: Resolving Knowledge Conflicts in Knowledge-Intensive Visual Question Answering via Reasoning-Pivot Alignment"
  - u: "reliable_ai_needs_to_externalize_implicit_knowledge_a_human-ai_collaboration_per/"
    t: "Position: Reliable AI Needs to Externalize Implicit Knowledge: A Human-AI Collaboration Perspective"
  - u: "reseek_a_self-correcting_framework_for_search_agents_with_instructive_rewards/"
    t: "ReSeek: A Self-Correcting Framework for Search Agents with Instructive Rewards"
  - u: "retriever_portfolios_a_principled_approach_to_adaptive_rag/"
    t: "Retriever Portfolios: A Principled Approach to Adaptive RAG"
  - u: "seeing_to_generalize_how_visual_data_corrects_binding_shortcuts/"
    t: "Seeing to Generalize: How Visual Data Corrects Binding Shortcuts"
  - u: "self-augmenting_retrieval_for_diffusion_language_models/"
    t: "Self-Augmenting Retrieval for Diffusion Language Models"
  - u: "temporal_preference_optimization_for_unsupervised_retrieval/"
    t: "Temporal Preference Optimization for Unsupervised Retrieval"
  - u: "through_the_stealth_lens_attention-aware_defenses_against_poisoning_in_rag/"
    t: "Through the Stealth Lens: Attention-Aware Defenses Against Poisoning in RAG"
  - u: "understand_and_accelerate_memory_processing_pipeline_for_disaggregated_llm_infer/"
    t: "Understand and Accelerate Memory Processing Pipeline for Large Language Model Inference"
  - u: "understanding_lora_as_knowledge_memory_an_empirical_analysis/"
    t: "Understanding LoRA as Knowledge Memory: An Empirical Analysis"
  - u: "vector_linking_via_cross-model_local_isometric_consistency/"
    t: "基于跨模型局部等距一致性的向量链接"
item_total: 26
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔍 信息检索/RAG

**🧪 ICML2026** · **26** 篇论文解读

📌 **同领域跨会议浏览：** [🔬 ICLR2026 (81)](../../ICLR2026/information_retrieval/index.md) · [💬 ACL2026 (73)](../../ACL2026/information_retrieval/index.md) · [🤖 AAAI2026 (21)](../../AAAI2026/information_retrieval/index.md) · [🧠 NeurIPS2025 (25)](../../NeurIPS2025/information_retrieval/index.md) · [📹 ICCV2025 (5)](../../ICCV2025/information_retrieval/index.md) · [🧪 ICML2025 (6)](../../ICML2025/information_retrieval/index.md)

🔥 **高频主题：** RAG ×9 · 对抗鲁棒 ×2 · 对齐/RLHF ×2

**[BlitzRank: Principled Zero-shot Ranking Agents with Tournament Graphs](blitzrank_principled_zero-shot_ranking_agents_with_tournament_graphs.md)**

:   提出基于锦标赛图（tournament graph）的零样本重排序框架 BlitzRank，通过将每次 $k$-wise 比较产生的 $\binom{k}{2}$ 个偏好对累积到全局偏好图中并利用传递闭包推断额外排序关系，在 14 个基准、5 个 LLM oracle 上实现 Pareto 最优——在匹配或超越现有方法精度的同时减少 25–40% token 消耗。

**[CARE: Class-Adaptive Expert Consensus for Reliable Learning with Long-Tailed Noisy Labels](care_class-adaptive_expert_consensus_for_reliable_learning_with_long-tailed_nois.md)**

:   提出 CARE 框架，利用 VLM 的文本嵌入、图像特征和原始标签三路互补专家，通过类别自适应 Top-$K$ 共识机制实现长尾噪声标签场景下的可靠标签矫正，在合成与真实基准上一致超越 SOTA 最高 3.0%。

**[Graph-R1: Towards Agentic GraphRAG Framework via End-to-end Reinforcement Learning](graph-r1_towards_agentic_graphrag_framework_via_end-to-end_reinforcement_learnin.md)**

:   Graph-R1 把 GraphRAG 重写成"知识超图环境 + 多轮 think–query–retrieve–answer 智能体 + 结果导向 GRPO"的端到端 RL 框架，用更轻量的 n 元超图构建和双路超边检索 + RRF 融合，在 6 个标准 RAG 数据集上把 7B 模型的 F1 从 Search-R1 的 46.19 拉到 57.82。

**[HGMem: Hypergraph-based Working Memory to Improve Multi-step RAG for Long-Context Complex Relational Modeling](hgmem_hypergraph-based_working_memory_to_improve_multi-step_rag_for_long-context.md)**

:   本文把多步 RAG 中的 working memory 从"扁平的事实列表"重构成一张**超图**——每条超边就是一个可被 update / insert / merge 的记忆点，借助超边天然连接 $n\geq 2$ 个实体的能力，让记忆能在交互过程中持续合并低阶事实成高阶概念，从而显著提升需要"全局意义构建"的长上下文问答性能。

**[Hierarchical Abstract Tree for Cross-Document Retrieval-Augmented Generation](hierarchical_abstract_tree_for_cross-document_retrieval-augmented_generation.md)**

:   Ψ-RAG 用"合并—坍缩"式的层次聚类替换 RAPTOR 的 k-means 来构建跨文档抽象树，并配上一个具备多轮重写能力的检索回答 Agent 与稀疏 BM25 混合索引，让 Tree-RAG 第一次能在语料级、跨文档多跳问答上追平甚至超越 Graph-RAG，平均 F1 比 RAPTOR 高 25.9%、比 HippoRAG 2 高 7.4%。

**[How can embedding models bind concepts?](how_can_embedding_models_bind_concepts.md)**

:   本文把 "embedding 模型为什么不会绑定概念" 形式化成 "binding function 的复杂度问题"：通过几何分析证明 CLIP 的场景嵌入可加性分解成对象与概念之和（解释了单模态可探测、跨模态却失败），并在受控 Transformer 上证明当数据覆盖足够时，模型会学到一个由概念间**乘性交互**主导的低复杂度 binding，从而实现对未见对象组合的系统性泛化。

**[LARE: Low-Attention Region Encoding for Text–Image Retrieval](lare_low-attention_region_encoding_for_text-image_retrieval.md)**

:   LARE 是一个**免训练**的文本-图像检索框架：它把视觉编码器内部「低注意力」的区域单独抠出来再编码，用置信度门控的方式补进全局相似度里，从而在拥挤、含小目标/稀有目标的密集场景里把 CLIP/SigLIP 这类双编码器的检索召回明显拉高，而在常规数据集上几乎不掉点。

**[LazyAttention: Efficient Retrieval-Augmented Generation with Deferred Positional Encoding](lazyattention_efficient_retrieval-augmented_generation_with_deferred_positional_.md)**

:   LazyAttention 把 RoPE 位置编码从 KV 缓存写入阶段推迟到 attention kernel 内部 on-the-fly 完成，让同一份物理 KV 副本可以被任意 logical 位置复用，在 skewed RAG 工作负载上比 SOTA Block-Attention 减少 1.37× TTFT、提升 1.40× 吞吐，且生成质量基本无损。

**[LEMUR: Learned Multi-Vector Retrieval](lemur_learned_multi-vector_retrieval.md)**

:   Lemur 将多向量相似性搜索转化为监督学习问题，用一个两层 MLP 将 token 级嵌入映射到低维潜空间，再利用现有单向量 ANNS 索引完成检索，比 PLAID/MUVERA 等方法快一个数量级。

**[Less Is More: Elevating RAG via Performance-Driven Context Compression](less_is_more_elevating_rag_via_performance-driven_context_compression.md)**

:   CORE-RAG 用"性能即奖励"的 GRPO 强化学习训练一个 1.5B 小压缩器，把检索到的 top-k 文档压成 ~3% 长度的摘要，结果不仅没掉点反而在 4 个 QA benchmark 上比满上下文 RAG 平均提升 3.3 EM。

**[Linguistic Nepotism: Trading-off Quality for Language Preference in Multilingual RAG](linguistic_nepotism_trading-off_quality_for_language_preference_in_multilingual_.md)**

:   这篇论文设计了一套用模型内部信号（下一 token 引用预测概率）来测量多语言 RAG 中"语言偏好"的可控方法，发现六个开源大模型在长文生成时系统性地偏爱引用英文文档，并且在英文文档不相关时也照引不误——语言本身比文档相关性更能左右模型的引用选择。

**[ML-Embed: Inclusive and Efficient Embeddings for a Multilingual World](ml-embed_inclusive_and_efficient_embeddings_for_a_multilingual_world.md)**

:   ML-Embed 把 Matryoshka 思想从一维 (representation 维度) 扩展到**三维** —— 在 embedding 参数 (MEL)、模型深度 (MLL)、表征维度 (MRL) 上**全栈嵌套训练**, 同时构建 282 种自然语言 + 40 种编程语言、5000 万样本的多语训练集, 推出 140M-8B 一族开源模型, 在 17 个 MTEB benchmark 上 9 个拿第一, 波兰语 +22.89, 越南语 +6.88.

**[ParisKV: Fast and Drift-Robust KV-Cache Retrieval for Long-Context LLMs](pariskv_fast_and_drift-robust_kv-cache_retrieval_for_long-context_llms.md)**

:   ParisKV 通过把 key/query 归一化并随机正交旋转到单位超球上、用"数据无关的解析质心"代替从 prefill 学习出来的质心，再叠加一个 GPU 原生的"碰撞投票 + 4-bit 量化重排"两阶段检索 + UVA 按需取 KV，在百万 token 上下文上把 Top-$k$ KV 检索的解码延迟相比 MagicPIG/PQCache 降低 17–44×，并在 9 个长生成任务里 7 个达到或超过 full attention 精度。

**[Predictive Prefetching for Retrieval-Augmented Generation](predictive_prefetching_for_retrieval-augmented_generation.md)**

:   通过学习 transformer 隐状态/注意力中"早于不确定性 8–16 token 出现的语义前兆"，本文用 RetrievalPredictor + ContextMonitor + QueryGenerator 三件套把 RAG 的检索从同步阻塞改造为预测式异步预取，在 HotpotQA 等基准上把端到端延迟降低 43.5%、TTFT 降低 62.4%，同时答案质量保持在同步 RAG 1% 以内。

**[Ranking-Free RAG: Replacing Re-Ranking with Selection in RAG for Sensitive Domains](ranking_free_rag_replacing_re-ranking_with_selection_in_rag_for_sensitive_domain.md)**

:   本文提出 METEORA，用 DPO 训练的"理由生成器 + 统计肘部检测 + 同框架 Verifier"三件套，把 RAG 中不可解释、依赖 top-$k$ 的 re-ranker 整段替换掉，在 6 个敏感领域数据集上同时拿到更高召回、80% 的证据量削减和 4.4× 的对抗鲁棒性提升。

**[REAL: Resolving Knowledge Conflicts in Knowledge-Intensive Visual Question Answering via Reasoning-Pivot Alignment](real_resolving_knowledge_conflicts_in_knowledge-intensive_visual_question_answer.md)**

:   本文提出 REAL 框架，用"Reasoning-Pivot"（推理链中必须依赖外部证据才能补全的原子节点/边）重新定义 KI-VQA 中的知识冲突，并通过 RPA-SFT 训练 pivot 感知的冲突判别器 + RPGD 训练免费的对比解码策略，在 E-VQA / InfoSeek / A-OKVQA 上分别取得 +3.8% / +1.6% / +3.6% 的提升。

**[Position: Reliable AI Needs to Externalize Implicit Knowledge: A Human-AI Collaboration Perspective](reliable_ai_needs_to_externalize_implicit_knowledge_a_human-ai_collaboration_per.md)**

:   本文是一篇 ICML 立场论文,主张当前所有 AI 可靠性方法 (RAG / 自一致性 / RLHF / Agent Memory) 都只能验证显式知识,而 AI 真正强大的能力来自训练数据里 80-95% 未被人类正式记录的"隐式知识",作者提出 Knowledge Objects (KOs) 作为基础设施——把 AI 隐式推理外化成人类可检查、可验证、可背书的结构化产物,从而让一次人类验证的成本在群体中长期复利。

**[ReSeek: A Self-Correcting Framework for Search Agents with Instructive Rewards](reseek_a_self-correcting_framework_for_search_agents_with_instructive_rewards.md)**

:   ReSeek 给 RL-trained 搜索 agent 增加一个 JUDGE 动作 + 用 BGE-reranker 计算"理想判断"作为过程奖励,使 agent 能在每次检索后软性"屏蔽"无效信息并重新查询;同时提出 FictionalHot 这一基于虚构实体的抗污染评测,Qwen2.5-7B 上平均 EM 达到 0.377,比 ZeroSearch 高 +3.1。

**[Retriever Portfolios: A Principled Approach to Adaptive RAG](retriever_portfolios_a_principled_approach_to_adaptive_rag.md)**

:   本文把 RAG 中"选哪个 retriever"重新表述为一个 best-of-$k$ 组合优化问题，从 360 个候选 retriever 里离线贪心挑出一个互补的 size-$k$ 组合（portfolio），并训练一个轻量对比学习路由器在线把每个 query 分发给组合里的 top-$\ell$ 个成员，在 4 个 QA 基准上同时打过单 retriever 和 Vendi-RAG 类推理时调参方法，并显著降低 token 和延迟成本。

**[Seeing to Generalize: How Visual Data Corrects Binding Shortcuts](seeing_to_generalize_how_visual_data_corrects_binding_shortcuts.md)**

:   本文用一个"颜色-形状-item"受控合成检索任务复现了"VLM 在纯文本任务上超过其 base LLM"的奇怪现象，并用机制可解释性证明：图像训练让模型把变量绑定策略从"位置捷径"切换到"语义符号匹配"，这一切换在重新接回纯文本后被保留下来，使 OOD 检索准确率从 37.2% 提升到 69.5%；在真实 Qwen2/2.5/3 家族上也观察到一致的"symbolic/positional 比例上升"。

**[Self-Augmenting Retrieval for Diffusion Language Models](self-augmenting_retrieval_for_diffusion_language_models.md)**

:   利用扩散语言模型在去噪过程中对所有位置同时给出的"暂定预测"作为前瞻信号，在每一步去噪时用这些尚未提交的 token 重新检索证据，提出训练无关、检索器无关的动态 RAG 框架 SARDI，在 5 个多跳 QA 基准上同时打败扩散和自回归的训练无关检索基线，且吞吐量最高快 8 倍。

**[Temporal Preference Optimization for Unsupervised Retrieval](temporal_preference_optimization_for_unsupervised_retrieval.md)**

:   本文提出 TPOUR，把 DPO 式偏好学习搬到检索的**时间维度**上，让无监督密集检索器在语义相近但时间错位的文档里优先选出"时间对齐"的那一篇，并用时间向量插值零成本泛化到没训练过的年份。

**[Through the Stealth Lens: Attention-Aware Defenses Against Poisoning in RAG](through_the_stealth_lens_attention-aware_defenses_against_poisoning_in_rag.md)**

:   本文指出现有 RAG 投毒攻击虽然能用少量恶意段落操纵 LLM 输出，但**并非真正隐蔽**——成功的低预算攻击必然会让模型把注意力过度集中在恶意段落上，因此作者用每段落归一化注意力分数 NPAS 和基于其方差的 AV Filter 把异常段落筛掉，在 4 数据集 × 5 LLM × 5 攻击的设定下把 RACC 比 Certified Robust RAG 最高拉高 20%。

**[Understand and Accelerate Memory Processing Pipeline for Large Language Model Inference](understand_and_accelerate_memory_processing_pipeline_for_disaggregated_llm_infer.md)**

:   本文把现代 LLM 长上下文推理中的稀疏注意力、RAG、压缩上下文记忆等优化统一为四阶段 "Prepare Memory → Compute Relevancy → Retrieval → Apply to Inference" 内存处理流水线，定量证明该流水线占整体延迟 22%-97% 且各阶段计算特性高度异构，并据此提出 GPU-FPGA 异构系统：把规则/算密集操作留 GPU、把稀疏/不规则/访存密集操作 offload 到 FPGA，在 MI210 + Alveo U55C 上取得最多 2.2× 端到端加速和 4.7× 能耗下降。

**[Understanding LoRA as Knowledge Memory: An Empirical Analysis](understanding_lora_as_knowledge_memory_an_empirical_analysis.md)**

:   作者用 PhoneBook 与新构造的 PaperQA 基准做系统实证审计，把 LoRA 看作可独立训练 / 加载 / 组合的知识记忆单元，定量给出"秩 → 容量 → 效率 → 多模块组合 → 与 RAG/ICL 互补"全链路的设计准则。

**[基于跨模型局部等距一致性的向量链接](vector_linking_via_cross-model_local_isometric_consistency.md)**

:   论文提出向量链接问题——在黑盒约束下通过发现两个不同编码器产生的嵌入云之间的对象对应关系。核心观察是独立训练的对比学习编码器在短距离内保持局部等距一致（相似度保留 up to 缩放因子），基于此提出多视图几何哈希自举框架，只需 15-30 个种子对即可恢复 79-90% 的重叠对象。
