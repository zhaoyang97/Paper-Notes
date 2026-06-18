---
title: >-
  ICLR2026 信息检索/RAG论文汇总 · 28篇论文解读
description: >-
  28篇ICLR2026的信息检索/RAG 方向论文解读，涵盖 RAG、LLM、推理等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICLR2026"
  - "信息检索/RAG"
  - "论文解读"
  - "论文笔记"
  - "RAG"
  - "LLM"
  - "推理"
item_list:
  - u: "amemgym_interactive_memory_benchmarking_for_assistants_in_long-horizon_conversat/"
    t: "AMemGym: Interactive Memory Benchmarking for Assistants in Long-Horizon Conversations"
  - u: "attributing_response_to_context_a_jensen-shannon_divergence_driven_mechanistic_s/"
    t: "Attributing Response to Context: A Jensen-Shannon Divergence Driven Mechanistic Study of Context Attribution in Retrieval-Augmented Generation"
  - u: "attribution-guided_decoding/"
    t: "Attribution-Guided Decoding"
  - u: "bayesian_attention_mechanism_a_probabilistic_framework_for_positional_encoding_a/"
    t: "Bayesian Attention Mechanism: A Probabilistic Framework for Positional Encoding and Context Length Extrapolation"
  - u: "beyond_rag_vs_long-context_learning_distraction-aware_retrieval_for_efficient_kn/"
    t: "Beyond RAG vs. Long-Context: Learning Distraction-Aware Retrieval for Efficient Knowledge Grounding"
  - u: "btzsc_a_benchmark_for_zero-shot_text_classification_across_cross-encoders_embedd/"
    t: "BTZSC: A Benchmark for Zero-Shot Text Classification Across Cross-Encoders, Embedding Models, Rerankers and LLMs"
  - u: "digging_deeper_learning_multi-level_concept_hierarchies/"
    t: "Digging Deeper: Learning Multi-Level Concept Hierarchies"
  - u: "embedding-based_context-aware_reranker/"
    t: "Embedding-Based Context-Aware Reranker"
  - u: "fine-tuning_with_rag_for_improving_llm_learning_of_new_skills/"
    t: "Fine-tuning with RAG for Improving LLM Learning of New Skills"
  - u: "flow_of_spans_generalizing_language_models_to_dynamic_span-vocabulary_via_gflown/"
    t: "Flow of Spans: Generalizing Language Models to Dynamic Span-Vocabulary via GFlowNets"
  - u: "g-reasoner_foundation_models_for_unified_reasoning_over_graph-structured_knowled/"
    t: "G-reasoner: Foundation Models for Unified Reasoning over Graph-structured Knowledge"
  - u: "hierarchical_concept-based_interpretable_models/"
    t: "Hierarchical Concept-based Interpretable Models"
  - u: "hume_measuring_the_human-model_performance_gap_in_text_embedding_tasks/"
    t: "HUME: Measuring the Human-Model Performance Gap in Text Embedding Tasks"
  - u: "hybrid_deep_searcher_scalable_parallel_and_sequential_search_reasoning/"
    t: "Hybrid Deep Searcher: Scalable Parallel and Sequential Search Reasoning"
  - u: "judges_verdict_a_comprehensive_analysis_of_llm_judge_capability_through_human_ag/"
    t: "Judge's Verdict: A Comprehensive Analysis of LLM Judge Capability Through Human Agreement"
  - u: "leveraging_data_to_say_no_memory_augmented_plug-and-play_selective_prediction/"
    t: "Leveraging Data to Say No: Memory Augmented Plug-and-Play Selective Prediction"
  - u: "lightretriever_a_llm-based_text_retrieval_architecture_with_extremely_faster_que/"
    t: "LightRetriever: A LLM-based Text Retrieval Architecture with Extremely Faster Query Inference"
  - u: "mapping_semantic_syntactic_relationships_with_geometric_rotation/"
    t: "Mapping Semantic & Syntactic Relationships with Geometric Rotation"
  - u: "on_the_wings_of_imagination_conflicting_script-based_multi-role_framework_for_hu/"
    t: "On the Wings of Imagination: Conflicting Script-based Multi-role Framework for Humor Caption Generation"
  - u: "q_rag_long_context_multi_step_retrieval/"
    t: "Q-RAG: Long Context Multi-Step Retrieval via Value-Based Embedder Training"
  - u: "query-level_uncertainty_in_large_language_models/"
    t: "Query-Level Uncertainty in Large Language Models"
  - u: "raee_a_robust_retrieval-augmented_early_exit_framework_for_efficient_inference/"
    t: "RAEE: A Robust Retrieval-Augmented Early Exit Framework for Efficient Inference"
  - u: "reftool_reference-guided_tool_creation_for_knowledge-intensive_reasoning/"
    t: "RefTool: Reference-Guided Tool Creation for Knowledge-Intensive Reasoning"
  - u: "revela_dense_retriever_learning_via_language_modeling/"
    t: "Revela: Dense Retriever Learning via Language Modeling"
  - u: "summaries_as_centroids_for_interpretable_and_scalable_text_clustering/"
    t: "Summaries as Centroids for Interpretable and Scalable Text Clustering"
  - u: "synthworlds_controlled_parallel_worlds_for_disentangling_reasoning_and_knowledge/"
    t: "SynthWorlds: Controlled Parallel Worlds for Disentangling Reasoning and Knowledge in Language Models"
  - u: "tokmem_one-token_procedural_memory_for_large_language_models/"
    t: "TokMem: One-Token Procedural Memory for Large Language Models"
  - u: "your_language_model_secretly_contains_personality_subnetworks/"
    t: "Your Language Model Secretly Contains Personality Subnetworks"
item_total: 28
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔍 信息检索/RAG

**🔬 ICLR2026** · **28** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (22)](../../ICML2026/information_retrieval/index.md) · [💬 ACL2026 (70)](../../ACL2026/information_retrieval/index.md) · [🤖 AAAI2026 (21)](../../AAAI2026/information_retrieval/index.md) · [🧠 NeurIPS2025 (25)](../../NeurIPS2025/information_retrieval/index.md) · [📹 ICCV2025 (5)](../../ICCV2025/information_retrieval/index.md) · [🧪 ICML2025 (6)](../../ICML2025/information_retrieval/index.md)

🔥 **高频主题：** RAG ×5 · LLM ×5 · 推理 ×4

**[AMemGym: Interactive Memory Benchmarking for Assistants in Long-Horizon Conversations](amemgym_interactive_memory_benchmarking_for_assistants_in_long-horizon_conversat.md)**

:   提出AMemGym——首个支持on-policy交互式评估的长程对话记忆基准环境，通过结构化数据采样（用户画像→状态演化→个性化问答）驱动LLM模拟用户进行角色扮演，揭示了off-policy评估的排名偏差问题，并系统诊断了RAG/长上下文/Agent记忆系统的write/read/utilization三阶段失败模式。

**[Attributing Response to Context: A Jensen-Shannon Divergence Driven Mechanistic Study of Context Attribution in Retrieval-Augmented Generation](attributing_response_to_context_a_jensen-shannon_divergence_driven_mechanistic_s.md)**

:   提出ARC-JSD方法，通过计算完整上下文与逐句消融上下文下的响应分布的Jensen-Shannon散度，在无需微调、梯度计算或代理模型的情况下实现高效精准的RAG上下文归因，并结合Logit Lens进行机制分析，定位负责上下文归因的注意力头和MLP层，通过门控操作降低约39%的幻觉率。

**[Attribution-Guided Decoding](attribution-guided_decoding.md)**

:   提出AGD解码策略，在每步生成时从高概率候选token中选择对用户指定"兴趣区域"（ROI）归因得分最高的token，将归因方法从被动分析工具转变为主动生成引导工具，在指令遵循和事实性任务上均取得显著提升。

**[Bayesian Attention Mechanism: A Probabilistic Framework for Positional Encoding and Context Length Extrapolation](bayesian_attention_mechanism_a_probabilistic_framework_for_positional_encoding_a.md)**

:   将位置编码重新表述为贝叶斯注意力机制中的先验分布，统一了 NoPE（均匀先验）和 ALiBi（拉普拉斯先验），并提出广义高斯先验（GGD-BAM），仅增加 384 个参数即可在 500 倍训练长度上实现完美的 passkey 检索。

**[Beyond RAG vs. Long-Context: Learning Distraction-Aware Retrieval for Efficient Knowledge Grounding](beyond_rag_vs_long-context_learning_distraction-aware_retrieval_for_efficient_kn.md)**

:   提出 LDAR（Learning Distraction-Aware Retrieval），一个轻量级自适应检索器，通过学习基于查询-段落相似度分布选择段落的连续区间（band），在平衡信息覆盖与干扰段落影响的同时，以约一半的 token 用量超越长上下文方法的性能。

**[BTZSC: A Benchmark for Zero-Shot Text Classification Across Cross-Encoders, Embedding Models, Rerankers and LLMs](btzsc_a_benchmark_for_zero-shot_text_classification_across_cross-encoders_embedd.md)**

:   提出 BTZSC 基准（22 个数据集），首次在统一零样本协议下系统比较 NLI 交叉编码器、嵌入模型、Reranker 和指令微调 LLM 四大模型家族（共 38 个模型），发现 Qwen3-Reranker-8B 以 macro F1=0.72 取得新 SOTA，嵌入模型在精度-延迟权衡上最优。

**[Digging Deeper: Learning Multi-Level Concept Hierarchies](digging_deeper_learning_multi-level_concept_hierarchies.md)**

:   本文提出 Multi-Level Concept Splitting（MLCS）将概念分裂过程从单层递归扩展到多层，仅用顶层概念标注就能自动发现任意深度的概念层级树，并设计 Deep-HiCEMs 架构来表示和利用这些深层层级，实现多粒度的测试时概念干预。

**[Embedding-Based Context-Aware Reranker](embedding-based_context-aware_reranker.md)**

:   提出 EBCAR，一个基于嵌入空间的轻量级重排序框架，通过文档 ID 嵌入和段落位置编码引入结构信息，结合共享全注意力 + 专用掩码注意力的混合机制实现跨段落推理，在 ConTEB 基准上以 126M 参数达到最优平均 nDCG@10，推理速度比 LLM 重排器快 150 倍以上。

**[Fine-tuning with RAG for Improving LLM Learning of New Skills](fine-tuning_with_rag_for_improving_llm_learning_of_new_skills.md)**

:   提出将 RAG 从推理时的永久依赖转化为训练时的教师信号：从 agent 失败中提取 hint、用 hint 增强的教师生成更优轨迹、然后移除 hint 蒸馏到学生模型，使学生内化检索增益而无需运行时 RAG，在 ALFWorld 达到 91% 成功率（基线 79%），WebShop 分数达 72（基线 61）。

**[Flow of Spans: Generalizing Language Models to Dynamic Span-Vocabulary via GFlowNets](flow_of_spans_generalizing_language_models_to_dynamic_span-vocabulary_via_gflown.md)**

:   提出 FoSS，首次将 GFlowNets 引入 span 级别语言模型，通过构建 DAG 结构的状态空间代替传统 token-by-token 的树形结构，实现更灵活多样的文本生成，MAUVE 分数最高提升 12.5%。

**[G-reasoner: Foundation Models for Unified Reasoning over Graph-structured Knowledge](g-reasoner_foundation_models_for_unified_reasoning_over_graph-structured_knowled.md)**

:   提出 G-reasoner，通过 QuadGraph 四层统一图接口将异构知识源标准化，训练 34M 参数的 GNN 图基础模型联合推理图拓扑和文本语义，配合 LLM 在 6 个基准上全面超越 SOTA GraphRAG 方法。

**[Hierarchical Concept-based Interpretable Models](hierarchical_concept-based_interpretable_models.md)**

:   HiCEMs引入层级概念嵌入模型，通过Concept Splitting方法在预训练CEM的嵌入空间中自动发现细粒度子概念（无需额外标注），构建层级概念结构，使模型能在不同粒度层次进行测试时概念干预以提升任务性能。

**[HUME: Measuring the Human-Model Performance Gap in Text Embedding Tasks](hume_measuring_the_human-model_performance_gap_in_text_embedding_tasks.md)**

:   提出 HUME 人类评估框架，在 MTEB 的 16 个数据集（重排序/分类/聚类/STS）上系统测量人类表现，发现人类总体排名第 4（77.6 vs 模型最佳 80.1），揭示模型"超人"表现多出现在人类一致性最低的任务上，并评估 9 个 LLM 作为标注代理的可行性。

**[Hybrid Deep Searcher: Scalable Parallel and Sequential Search Reasoning](hybrid_deep_searcher_scalable_parallel_and_sequential_search_reasoning.md)**

:   提出 HybridDeepSearcher，通过构建 HDS-QA 数据集训练大语言推理模型（LRM）区分可并行化和顺序依赖的搜索查询，在 FanOutQA 上 F1 提升 +15.9、BrowseComp 子集上提升 +11.5，同时显著降低推理延迟并展示出一致的测试时搜索扩展能力。

**[Judge's Verdict: A Comprehensive Analysis of LLM Judge Capability Through Human Agreement](judges_verdict_a_comprehensive_analysis_of_llm_judge_capability_through_human_ag.md)**

:   提出 Judge's Verdict Benchmark——基于"相关性过滤 → Cohen's Kappa 人类相似性测试"的两步评估框架，对 54 个 LLM 评委进行系统评测，筛选出 27 个 Tier 1 评委（23 个人类相似型 + 4 个超一致型），核心发现是相关性高不等于一致性高，需要用 Kappa + z-score 才能真正衡量 LLM 评委质量。

**[Leveraging Data to Say No: Memory Augmented Plug-and-Play Selective Prediction](leveraging_data_to_say_no_memory_augmented_plug-and-play_selective_prediction.md)**

:   提出 MA-PaPSP 框架，通过外部检索数据集构建代理嵌入（k-NN 加权平均降低表示方差）+ 对比归一化评分（改善校准），无训练地为任意 VLM 提供可靠的"拒绝回答"能力，在图像描述、图文匹配、分类的选择性预测上全面优于 PaPSP 和 LLM-as-judge 基线。

**[LightRetriever: A LLM-based Text Retrieval Architecture with Extremely Faster Query Inference](lightretriever_a_llm-based_text_retrieval_architecture_with_extremely_faster_que.md)**

:   提出 LightRetriever，一种极端不对称的LLM检索架构：文档端保留完整LLM编码器，查询端完全去除深度建模——稠密检索仅需嵌入查表+平均，稀疏检索仅需token计数——实现查询编码1000倍加速、端到端10倍吞吐提升，同时保持95%的检索性能。

**[Mapping Semantic & Syntactic Relationships with Geometric Rotation](mapping_semantic_syntactic_relationships_with_geometric_rotation.md)**

:   提出RISE（Rotor-Invariant Shift Estimation）方法，利用Clifford代数的rotor将话语级语义-句法变换（否定、条件化、礼貌化）表示为单位超球面上的一致旋转操作，在7种语言×3种嵌入模型×3种变换的系统实验中证实这些旋转可跨语言和跨模型迁移（77%-95%保持率），首次将线性表示假说从词级扩展到跨语言话语级并推广到弯曲流形上的测地线结构。

**[On the Wings of Imagination: Conflicting Script-based Multi-role Framework for Humor Caption Generation](on_the_wings_of_imagination_conflicting_script-based_multi-role_framework_for_hu.md)**

:   提出 HOMER 框架，基于 GTVH 幽默理论构建三角色 LLM 协作机制（冲突脚本提取器 + 层次想象器 + 标题生成器），通过显式建模脚本对立、多视角联想链与笑话数据库检索构建想象树来扩展创意空间，在 New Yorker 漫画基准上以 GPT-4o 为底座平均提升 ~7%，人类评估也显著优于所有基线。

**[Q-RAG: Long Context Multi-Step Retrieval via Value-Based Embedder Training](q_rag_long_context_multi_step_retrieval.md)**

:   将多步检索建模为 MDP，用基于值的 RL（soft Q-learning）微调 **embedder 而非 LLM**，Q 函数设计为状态嵌入和动作嵌入的内积（理论证明为万能近似器），结合 RoPE 相对位置编码实现时序推理，在单卡 A100 上训练 12 小时，4K 训练泛化到 1M+ token 上下文，RULER 基准达到近乎完美的 NIAH 性能。

**[Query-Level Uncertainty in Large Language Models](query-level_uncertainty_in_large_language_models.md)**

:   提出Query-Level Uncertainty概念，通过Internal Confidence方法在生成前（单次前向传播）估计LLM能否回答给定查询，无需训练即可实现高效的自适应推理（RAG触发/模型级联/弃权）。

**[RAEE: A Robust Retrieval-Augmented Early Exit Framework for Efficient Inference](raee_a_robust_retrieval-augmented_early_exit_framework_for_efficient_inference.md)**

:   提出 RAEE，一种无需训练分类器的检索增强早退框架，通过检索语义相似样本的退出信息来动态确定最优退出层，不仅加速推理还能纠正模型错误预测，实现加速与性能提升的双赢。

**[RefTool: Reference-Guided Tool Creation for Knowledge-Intensive Reasoning](reftool_reference-guided_tool_creation_for_knowledge-intensive_reasoning.md)**

:   提出 RefTool 框架基于外部参考资料（教材、知识片段）自动创建可执行 Python 工具，解决了现有工具创建方法依赖 LLM 内在知识在专业领域失败的问题，在因果推理、物理和化学任务上平均超过已有方法 12.3%。

**[Revela: Dense Retriever Learning via Language Modeling](revela_dense_retriever_learning_via_language_modeling.md)**

:   提出 Revela，通过 in-batch attention 机制将检索器学习融入语言建模——NTP 不仅依赖本序列上下文，还依赖批内其他序列（由检索器相似度加权），无需标注 query-document 对即可训练强大的密集检索器。

**[Summaries as Centroids for Interpretable and Scalable Text Clustering](summaries_as_centroids_for_interpretable_and_scalable_text_clustering.md)**

:   提出 k-NLPmeans 和 k-LLMmeans，通过在 k-means 迭代中周期性地用文本摘要替换数值质心（summary-as-centroid），在保持 k-means 标准目标的同时实现可解释的聚类原型，且 LLM 调用量与数据集大小无关。

**[SynthWorlds: Controlled Parallel Worlds for Disentangling Reasoning and Knowledge in Language Models](synthworlds_controlled_parallel_worlds_for_disentangling_reasoning_and_knowledge.md)**

:   构建结构完全相同但实体分别映射到真实/合成名称的平行语料库，通过对比两个"平行世界"上的任务表现来量化 LLM 的参数化知识优势差距（Knowledge Advantage Gap），发现即使有 RAG 和 CoT 增强，该差距依然持续存在。

**[TokMem: One-Token Procedural Memory for Large Language Models](tokmem_one-token_procedural_memory_for_large_language_models.md)**

:   提出 TokMem，将可复用的任务程序编译为单个可训练记忆 token，既作为程序索引又作为生成控制信号，无需长 prompt 即可高效调用 1000+ 任务程序，且支持无遗忘的持续扩展。

**[Your Language Model Secretly Contains Personality Subnetworks](your_language_model_secretly_contains_personality_subnetworks.md)**

:   本文提出通过激活引导的剪枝（activation-guided pruning）从预训练 LLM 中提取人格专用子网络，无需任何训练即可实现高效的人格切换，并引入对比剪枝策略增强对立人格间的参数分离。
