---
title: >-
  ACL2026 推荐系统论文汇总 · 22篇论文解读
description: >-
  22篇ACL2026的推荐系统方向论文解读，涵盖推荐系统、个性化生成、对话系统、LLM、推理、RAG等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ACL2026"
  - "推荐系统"
  - "论文解读"
  - "论文笔记"
  - "个性化生成"
  - "对话系统"
  - "LLM"
  - "推理"
  - "RAG"
item_list:
  - u: "bridging_language_and_items_for_retrieval_and_recommendation_benchmarking_llms_a/"
    t: "Bridging Language and Items for Retrieval and Recommendation: Benchmarking LLMs as Semantic Encoders"
  - u: "clusterrag_cluster-based_collaborative_filtering_for_personalized_retrieval-augm/"
    t: "ClusterRAG: Cluster-Based Collaborative Filtering for Personalized Retrieval-Augmented Generation"
  - u: "culinary_crossroads_a_rag_framework_for_enhancing_diversity_in_cross-cultural_re/"
    t: "Culinary Crossroads: A RAG Framework for Enhancing Diversity in Cross-Cultural Recipe Adaptation"
  - u: "decisive_guiding_user_decisions_with_optimal_preference_elicitation_from_unstruc/"
    t: "Decisive: Guiding User Decisions with Optimal Preference Elicitation from Unstructured Documents"
  - u: "from_past_to_path_masked_history_learning_for_next-item_prediction_in_generative/"
    t: "From Past To Path: Masked History Learning for Next-Item Prediction in Generative Recommendation"
  - u: "from_recall_to_forgetting_benchmarking_long-term_memory_for_personalized_agents/"
    t: "From Recall to Forgetting: Benchmarking Long-Term Memory for Personalized Agents"
  - u: "graphlora_structure-aware_low-rank_adaptation_for_large_language_model_recommend/"
    t: "GraphLoRA: Structure-Aware Low-Rank Adaptation for Large Language Model Recommendation"
  - u: "harpo_hierarchical_agentic_reasoning_for_user-aligned_conversational_recommendat/"
    t: "HARPO: Hierarchical Agentic Reasoning for User-Aligned Conversational Recommendation"
  - u: "horizon_a_benchmark_for_in-the-wild_user_behaviour_modeling/"
    t: "HORIZON: A Benchmark for in-the-wild User Behaviour Modeling"
  - u: "hsuga_llm-enhanced_recommendation_with_hierarchical_semantic_understanding_and_g/"
    t: "HSUGA: LLM-Enhanced Recommendation with Hierarchical Semantic Understanding and Group-Aware Alignment"
  - u: "icebreaker_for_conversational_agents_breaking_the_first-message_barrier_with_per/"
    t: "IceBreaker for Conversational Agents: Breaking the First-Message Barrier with Personalized Starters"
  - u: "intent-driven_semantic_id_generation_for_grounded_conversational_news_recommenda/"
    t: "Intent-Driven Semantic ID Generation for Grounded Conversational News Recommendation"
  - u: "learning_to_retrieve_user_history_and_generate_user_profiles_for_personalized_pe/"
    t: "Learning to Retrieve User History and Generate User Profiles for Personalized Persuasiveness Prediction"
  - u: "memrec_collaborative_memory-augmented_agentic_recommender_system/"
    t: "MemRec: Collaborative Memory-Augmented Agentic Recommender System"
  - u: "mirroring_users_towards_building_preference-aligned_user_simulator_with_user_fee/"
    t: "Mirroring Users: Towards Building Preference-aligned User Simulator with User Feedback in Recommendation"
  - u: "personalizing_llms_with_binary_feedback_a_preference-corrected_optimization_fram/"
    t: "Personalizing LLMs with Binary Feedback: A Preference-Corrected Optimization Framework"
  - u: "quality_over_clicks_intrinsic_quality-driven_iterative_reinforcement_learning_fo/"
    t: "Quality Over Clicks: Intrinsic Quality-Driven Iterative RL for Cold-Start E-Commerce Query Suggestion"
  - u: "rerec_reasoning-augmented_llm-based_recommendation_assistant_via_reinforcement_f/"
    t: "ReRec: Reasoning-Augmented LLM-based Recommendation Assistant via Reinforcement Fine-tuning"
  - u: "sensejudge_human-centric_preference-driven_judgment_framework/"
    t: "SenseJudge: Human-Centric Preference-Driven Judgment Framework"
  - u: "what_makes_an_ideal_quote_recommending_34unexpected_yet_rational34_quotations_vi/"
    t: "What Makes an Ideal Quote? Recommending \"Unexpected yet Rational\" Quotations via Novelty"
  - u: "what_makes_llms_effective_sequential_recommenders_a_study_on_preference_intensit/"
    t: "What Makes LLMs Effective Sequential Recommenders? A Study on Preference Intensity and Temporal Context"
  - u: "where_and_what_reasoning_dynamic_and_implicit_preferences_in_situated_conversati/"
    t: "Where and What: Reasoning Dynamic and Implicit Preferences in Situated Conversational Recommendation"
item_total: 22
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎁 推荐系统

**💬 ACL2026** · **22** 篇论文解读

📌 **同领域跨会议浏览：** [🔬 ICLR2026 (24)](../../ICLR2026/recommender/index.md) · [🧪 ICML2026 (11)](../../ICML2026/recommender/index.md) · [🤖 AAAI2026 (27)](../../AAAI2026/recommender/index.md) · [🧠 NeurIPS2025 (24)](../../NeurIPS2025/recommender/index.md) · [🧪 ICML2025 (17)](../../ICML2025/recommender/index.md) · [💬 ACL2025 (7)](../../ACL2025/recommender/index.md)

🔥 **高频主题：** 推荐系统 ×12 · 个性化生成 ×5 · 对话系统 ×4 · LLM ×3 · 推理 ×3

**[Bridging Language and Items for Retrieval and Recommendation: Benchmarking LLMs as Semantic Encoders](bridging_language_and_items_for_retrieval_and_recommendation_benchmarking_llms_a.md)**

:   本文发布 Amazon Reviews 2023 大规模数据集（570M reviews / 48M items）并基于它构建 BLaIR 基准，覆盖序列推荐 / 协同过滤 / 商品搜索 (短 query + 复杂 query) 三大场景，benchmark 了 11 个顶尖 LLM 作为语义编码器，发现它们在 BLaIR 上的排名与 MTEB 几乎不相关（Spearman -0.476），并指出推荐场景对语义编码器有独特要求。

**[ClusterRAG: Cluster-Based Collaborative Filtering for Personalized Retrieval-Augmented Generation](clusterrag_cluster-based_collaborative_filtering_for_personalized_retrieval-augm.md)**

:   ClusterRAG 把协同过滤引入个性化 RAG：先用用户历史文档构建用户表示并用 HDBSCAN 聚类，再从目标用户和相似用户中分层检索 profile 文档组成 prompt，在 LaMP 多任务基准上使 hybrid 模式全面优于 vanillaRAG、LaMP-IPA、ROPG 和 CFRAG。

**[Culinary Crossroads: A RAG Framework for Enhancing Diversity in Cross-Cultural Recipe Adaptation](culinary_crossroads_a_rag_framework_for_enhancing_diversity_in_cross-cultural_re.md)**

:   作者发现标准 RAG 在创意任务上"给了多样上下文也产出不多样"，于是设计 plug-and-play 的 CARRIAGE：查询重写 + diversity-aware MMR 重排 + sliding-window 动态上下文 + 对比性上下文注入，把"上下文多样性"真正传导到"输出多样性"，在西班牙语跨国菜谱适配上同时改善 lexical/semantic/ingredient diversity 与 CultureScore，对 closed-book LLM 达到 Pareto efficiency。

**[Decisive: Guiding User Decisions with Optimal Preference Elicitation from Unstructured Documents](decisive_guiding_user_decisions_with_optimal_preference_elicitation_from_unstruc.md)**

:   提出 DECISIVE 交互式决策框架，通过从非结构化文档中提取客观选项评分矩阵，结合贝叶斯偏好推断自适应选择成对比较问题高效学习用户潜在偏好向量，在最小化用户交互负担的同时实现透明个性化推荐，决策准确率比强基线提升最高 20%。

**[From Past To Path: Masked History Learning for Next-Item Prediction in Generative Recommendation](from_past_to_path_masked_history_learning_for_next-item_prediction_in_generative.md)**

:   提出掩码历史学习（MHL）训练框架，通过在生成式推荐的自回归训练中加入掩码历史重建辅助任务，结合熵引导的自适应掩码策略和课程学习调度器，使模型从仅预测"下一个是什么"转向理解"为什么形成这条路径"，在三个数据集上显著超越SOTA。

**[From Recall to Forgetting: Benchmarking Long-Term Memory for Personalized Agents](from_recall_to_forgetting_benchmarking_long-term_memory_for_personalized_agents.md)**

:   本文提出Memora基准和FAMA指标，将长期记忆评估从浅层事实检索扩展到跨越数周至数月的记忆整合与突变处理，揭示现有LLM和记忆agent在处理频繁知识更新时的系统性失败。

**[GraphLoRA: Structure-Aware Low-Rank Adaptation for Large Language Model Recommendation](graphlora_structure-aware_low-rank_adaptation_for_large_language_model_recommend.md)**

:   现有 LLM 推荐要么把协同信息塞进 prompt、要么把预训练好的静态嵌入注入 LoRA 权重，都把结构当成"读一遍"的静态输入；GraphLoRA 把一个可训练的图消息传递网络嵌进 LoRA 瓶颈（down-projection $\mathbf{A}$ 和 up-projection $\mathbf{B}$ 之间），让协同拓扑在参数空间里动态传播、直接引导参数更新，仅增 ~1.67% 参数就在 ML-1M、Amazon-Book 上超过 CoRA 等 SOTA。

**[HARPO: Hierarchical Agentic Reasoning for User-Aligned Conversational Recommendation](harpo_hierarchical_agentic_reasoning_for_user-aligned_conversational_recommendat.md)**

:   提出 HARPO 框架，将对话推荐重新定义为以推荐质量为优化目标的结构化决策问题，通过层次化偏好学习、基于价值网络的树搜索推理、虚拟工具操作和多智能体精炼四大组件，在 ReDial、INSPIRED 和 MUSE 三个基准上显著超越现有方法。

**[HORIZON: A Benchmark for in-the-wild User Behaviour Modeling](horizon_a_benchmark_for_in-the-wild_user_behaviour_modeling.md)**

:   本文提出 HORIZON，首个全开源的大规模跨领域长期推荐基准，基于 Amazon Reviews 合并构建包含 54M 用户和 35M 商品的统一交互历史，设计了沿时间轴和用户维度解耦的四象限评估协议，揭示了 BERT4Rec 等模型在分布内表现强劲但在时序外推和未见用户场景下显著退化的现象，且 LLM 在用户行为建模上并未一致优于专用架构。

**[HSUGA: LLM-Enhanced Recommendation with Hierarchical Semantic Understanding and Group-Aware Alignment](hsuga_llm-enhanced_recommendation_with_hierarchical_semantic_understanding_and_g.md)**

:   HSUGA 把 LLM 增强序列推荐的两个核心环节拆开来打补丁：用"阶段式 + 四类原子编辑（Add/Delete/Update/Retain）"的 HSU 模块把长交互序列的语义抽取做稳，再用按活跃度分组（20% 头部 / 80% 长尾）的 GAA 自蒸馏对齐解决长尾用户欠监督、活跃用户过对齐的问题，在 Steam/Fashion/Beauty 三个数据集 + GRU4Rec/BERT4Rec/SASRec 三个 backbone 上即插即用都涨点。

**[IceBreaker for Conversational Agents: Breaking the First-Message Barrier with Personalized Starters](icebreaker_for_conversational_agents_breaking_the_first-message_barrier_with_per.md)**

:   本文提出 IceBreaker，通过两步"握手"——共鸣感知兴趣蒸馏捕获触发兴趣 + 交互导向启动语生成配合个性化偏好对齐——解决对话智能体的"首条消息壁垒"，在全球最大对话产品之一的 A/B 测试中提升用户活跃天数 +1.84‰ 和点击率 +94.25‰。

**[Intent-Driven Semantic ID Generation for Grounded Conversational News Recommendation](intent-driven_semantic_id_generation_for_grounded_conversational_news_recommenda.md)**

:   本文提出 NewsRec-Chat，把对话式新闻推荐从"先检索再生成"反转为"先生成 SID 再模糊匹配"，靠两阶段 SID 对齐 + GPT-4 CoT 蒸馏让 7B 模型直接生成层级 Semantic ID 前缀并与当日新闻池模糊匹配，腾讯新闻平台上 152K 开放生成空间里取得 12.4% L1（4× 随机）、0% 幻觉，并通过 Profile-Aware Dual-Signal Reasoning 让 0 历史用户达到 18.0% L1（其他基线 0%）。

**[Learning to Retrieve User History and Generate User Profiles for Personalized Persuasiveness Prediction](learning_to_retrieve_user_history_and_generate_user_profiles_for_personalized_pe.md)**

:   本文提出 ReCAP 框架，通过可训练的查询生成器和用户画像生成器，从用户历史记录中检索与说服相关的信息并构建上下文感知的用户画像，显著提升个性化说服力预测的效果。

**[MemRec: Collaborative Memory-Augmented Agentic Recommender System](memrec_collaborative_memory-augmented_agentic_recommender_system.md)**

:   MemRec 用一个**轻量级 LLM 专门管理一张动态"协同记忆图"**（把多个 user 与 item 的语义记忆通过交互边相连），然后把蒸馏后的"协同记忆面（facets）"喂给重量级推理 LLM 做最终推荐；通过"Curate-then-Synthesize"压噪 + 异步 $O(1)$ 标签传播更新，在 4 个 benchmark 上 H@1 相对 SOTA i2Agent 提升 **+15% 到 +29%**，数据稀疏用户上更是相对 Vanilla LLM 提升 **+91.4%**。

**[Mirroring Users: Towards Building Preference-aligned User Simulator with User Feedback in Recommendation](mirroring_users_towards_building_preference-aligned_user_simulator_with_user_fee.md)**

:   作者把推荐系统里的"用户反馈日志"重写成一个 LLM 能理解的"用户记忆 + 曝光列表"统一仿真场景，再用 EKB 消费者决策模型生成显式的 chain-of-thought 决策过程作为"clarification"，通过不确定性分解 + 拒绝采样蒸馏出 10K 高质量 SFT/DPO 数据，让 3B 的 Llama 用户模拟器在 8 个领域的真实用户行为预测上超过 GPT-5 和 Gemini-2.5-Flash。

**[Personalizing LLMs with Binary Feedback: A Preference-Corrected Optimization Framework](personalizing_llms_with_binary_feedback_a_preference-corrected_optimization_fram.md)**

:   这篇论文提出 C-BPO，把目标用户历史当作正反馈、其他用户历史当作带噪未标注负反馈，并用 PU 学习校正“偏好重叠”带来的误惩罚，从而让 LLM 学到用户独特偏好而不压制通用任务能力。

**[Quality Over Clicks: Intrinsic Quality-Driven Iterative RL for Cold-Start E-Commerce Query Suggestion](quality_over_clicks_intrinsic_quality-driven_iterative_reinforcement_learning_fo.md)**

:   提出 Cold-EQS，一个面向冷启动电商场景的查询建议框架，利用可回答性、事实准确性和信息增益作为内在质量奖励，通过迭代强化学习持续优化查询建议质量，在线 chatUV 提升 6.81%。

**[ReRec: Reasoning-Augmented LLM-based Recommendation Assistant via Reinforcement Fine-tuning](rerec_reasoning-augmented_llm-based_recommendation_assistant_via_reinforcement_f.md)**

:   本文提出 ReRec，一个基于强化微调（RFT）的推荐助手框架，通过双图增强的奖励塑形提供细粒度奖励信号、推理感知的优势估计对推理步骤进行差异化监督、以及在线课程调度器动态调整训练难度，使 LLM 能处理复杂的多步推理推荐查询，在 RecBench+ 基准上显著超越现有方法。

**[SenseJudge: Human-Centric Preference-Driven Judgment Framework](sensejudge_human-centric_preference-driven_judgment_framework.md)**

:   提出 SenseJudge，一种基于显式人类偏好的可定制化 LLM 判断框架，配合真实多轮对话基准 SenseBench，在个性化评判任务中平均准确率比基线高 16.08%，模型排名与真实人类排名一致。

**[What Makes an Ideal Quote? Recommending "Unexpected yet Rational" Quotations via Novelty](what_makes_an_ideal_quote_recommending_34unexpected_yet_rational34_quotations_vi.md)**

:   NOVELQR 提出了一个新颖性驱动的引用推荐框架，通过生成式标签代理构建深层语义知识库实现语义理性检索，并用 token 级新颖性估计器缓解自回归续写偏差，在双语基准上显著提升推荐质量。

**[What Makes LLMs Effective Sequential Recommenders? A Study on Preference Intensity and Temporal Context](what_makes_llms_effective_sequential_recommenders_a_study_on_preference_intensit.md)**

:   本文揭示现有 LLM 推荐系统的二元偏好建模丢失了偏好强度和时间上下文两个关键信息，提出 RecPO 框架通过自适应奖励边际将这两个因素纳入偏好优化，在五个数据集上显著超越 S-DPO 等基线。

**[Where and What: Reasoning Dynamic and Implicit Preferences in Situated Conversational Recommendation](where_and_what_reasoning_dynamic_and_implicit_preferences_in_situated_conversati.md)**

:   SiPeR 通过场景转换估计（"Where"）和贝叶斯逆推理（"What"）两个机制，解决情景对话推荐中用户偏好随环境动态变化且常常隐式表达的挑战，在 SIMMC 2.1 和 SCREEN 上分别提升 10.9% 和 10.6%。
