<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🤖 AAAI2026 论文笔记

共 **431** 篇笔记，覆盖 **35** 个领域。

## 领域概览

| 领域 | 篇数 |
|:-----|-----:|
| 💬 [LLM / NLP](#llm_nlp) | 58 |
| 🦾 [LLM Agent](#llm_agent) | 49 |
| 💡 [LLM 推理](#llm_reasoning) | 33 |
| 🧩 [多模态 VLM](#multimodal_vlm) | 29 |
| ⚖️ [对齐 / RLHF](#llm_alignment) | 26 |
| 🎨 [图像生成](#image_generation) | 15 |
| 🚗 [自动驾驶](#autonomous_driving) | 14 |
| ⚡ [LLM 效率](#llm_efficiency) | 14 |
| 📈 [时间序列](#time_series) | 14 |
| 🤖 [机器人/具身智能](#robotics) | 12 |
| 🎮 [强化学习](#reinforcement_learning) | 11 |
| 🛡️ [AI 安全](#ai_safety) | 10 |
| 📦 [模型压缩](#model_compression) | 10 |
| 🧊 [3D 视觉](#3d_vision) | 9 |
| 🏥 [医学图像](#medical_imaging) | 9 |
| 🎯 [目标检测](#object_detection) | 9 |
| 🔄 [自监督/表示学习](#self_supervised) | 9 |
| 🎵 [音频/语音](#audio_speech) | 8 |
| 🕸️ [图学习](#graph_learning) | 8 |
| 🎬 [视频理解](#video_understanding) | 6 |
| 🔗 [因果推理](#causal_inference) | 5 |
| 🧑 [人体理解](#human_understanding) | 5 |
| ✍️ [文本生成](#nlp_generation) | 5 |
| 🎁 [推荐系统](#recommender) | 5 |
| 🛰️ [遥感](#remote_sensing) | 5 |
| ✂️ [语义分割](#segmentation) | 5 |
| 📡 [信号/通信](#signal_comm) | 5 |
| 🖼️ [图像恢复](#image_restoration) | 4 |
| 📖 [NLP 理解](#nlp_understanding) | 4 |
| 📐 [优化/理论](#optimization) | 4 |
| 🧮 [科学计算](#scientific_computing) | 4 |
| 🔎 [AIGC 检测](#aigc_detection) | 3 |
| ⚛️ [物理学](#physics) | 3 |
| 🌍 [地球科学](#earth_science) | 1 |
| 📂 [其他](#others) | 20 |

---

## 💬 LLM / NLP { #llm_nlp }

**[An Invariant Latent Space Perspective on Language Model Inversion](llm_nlp/an_invariant_latent_space_perspective_on_language_model_inve.md)**

:   提出不变潜空间假说(ILSH)，将LLM反演问题重新建模为复用LLM自身潜空间，设计Inv²A框架通过轻量级逆编码器将输出映射到去噪伪表示，再由冻结的LLM解码恢复隐藏prompt，在9个数据集上BLEU平均提升4.77%且仅需20%数据量即可达到可比性能。

**["As Eastern Powers, I Will Veto." : An Investigation of Nation-Level Bias of Large Language Models in International Relations](llm_nlp/as_eastern_powers_i_will_veto_an_investigation_of_nation-level_bias_of_large_lan.md)**

:   系统性地研究 LLM 在国际关系领域的国家级偏见，基于联合国安理会真实数据设计三种偏见测试（直接问答、关联测试、投票模拟），揭示偏见的多维性——随模型和评知上下文变化，并提出 RAG+Reflexion 去偏框架。

**[Benchmarking LLMs for Political Science: A United Nations Perspective](llm_nlp/benchmarking_llms_for_political_science_a_united_nations_perspective.md)**

:   提出 UNBench，首个基于联合国安理会 1994-2024 年记录的综合性政治科学 LLM 评测基准，涵盖决议起草、投票模拟、通过预测和代表发言生成四个关联任务，评估 LLM 对复杂政治动态的理解和模拟能力。

**[Beyond Accuracy: A Cognitive Load Framework for Mapping the Capability Boundaries of Tool-use Agents](llm_nlp/beyond_accuracy_a_cognitive_load_framework_for_mapping_the_c.md)**

:   借鉴心理学的认知负荷理论（CLT），将工具使用任务的复杂度分解为内在负荷（任务解题路径的结构复杂度）和外在负荷（问题表述的歧义性），构建可参数化调节认知负荷的 ToolLoad-Bench 基准，用指数衰减模型 $\text{Acc} \approx e^{-(k \cdot CL + b)}$ 精确刻画不同 Agent 的能力边界。

**[Beyond Cosine Similarity: Magnitude-Aware CLIP for No-Reference Image Quality Assessment](llm_nlp/beyond_cosine_similarity_magnitude-aware_clip_for_no-reference_image_quality_ass.md)**

:   提出 MA-CLIP，发现并利用 CLIP 图像特征的**幅度信息**作为感知质量的互补线索，结合余弦相似度实现无需训练的自适应双线索融合图像质量评估。

**[Beyond Hallucinations: A Composite Score for Measuring Reliability in Open-Source Large Language Models](llm_nlp/beyond_hallucinations_a_composite_score_for_measuring_reliability_in_open-source.md)**

:   提出 Composite Reliability Score (CRS)，将校准度、鲁棒性和不确定性量化三个维度统一为单一可解释指标，对 10 个开源 LLM 在 5 个 QA 数据集上进行系统评估，发现 Mistral-8x22B 综合可靠性最高（CRS=0.81），而模型大小并不直接决定可靠性。

**[Blue Teaming Function-Calling Agents](llm_nlp/blue_teaming_function-calling_agents.md)**

:   系统评估了四个开源function-calling LLM在三种攻击下的鲁棒性，并测试了八种防御方案的效果，揭示了当前模型默认不安全、防御方案在实际场景中仍难以部署的现状。

**[Cog-RAG: Cognitive-Inspired Dual-Hypergraph with Theme Alignment Retrieval-Augmented Generation](llm_nlp/cog-rag_cognitive-inspired_dual-hypergraph_with_theme_alignment_retrieval-augmen.md)**

:   提出 Cog-RAG，用主题超图和实体超图构建双超图索引，模拟人类"自顶向下"的认知过程进行两阶段检索（先主题后细节），实现从全局语义到局部信息的对齐生成。

**[ComLQ: Benchmarking Complex Logical Queries in Information Retrieval](llm_nlp/comlq_benchmarking_complex_logical_queries_in_information_retrieval.md)**

:   构建了首个面向复杂逻辑查询的信息检索基准 ComLQ（含合取、析取、否定等 14 种查询类型），并提出子图引导的 LLM 数据合成方法和否定一致性评估指标 LSNC，揭示现有检索器在逻辑推理尤其是否定建模上的严重不足。

**[ConInstruct: Evaluating Large Language Models on Conflict Detection and Resolution in Instructions](llm_nlp/coninstruct_evaluating_large_language_models_on_conflict_detection_and_resolutio.md)**

:   提出 ConInstruct 基准，评估 LLM 在指令包含冲突约束时的检测和解决能力，发现多数专有模型能较好检测冲突但很少主动告知用户，其中 DeepSeek-R1 和 Claude-4.5-Sonnet 在冲突检测上表现最佳（F1 分别达 91.5% 和 87.3%）。

**[Control Illusion: The Failure of Instruction Hierarchies in Large Language Models](llm_nlp/control_illusion_the_failure_of_instruction_hierarchies_in_large_language_models.md)**

:   系统性揭示了当前 LLM 中 system/user 提示分离机制**无法有效建立指令优先级**，并发现预训练习得的社会层级先验（权威、专业、共识）比显式的 system/user 角色对模型行为有更强的控制力。

**[Conversational Learning Diagnosis via Reasoning Multi-Turn Interactive Learning](llm_nlp/conversational_learning_diagnosis_via_reasoning_multi-turn_interactive_learning.md)**

:   提出 ParLD（Preview-Analyze-Reason 框架），通过多 Agent 协作实现对话式学习过程中学生认知状态的细粒度逐轮诊断，在性能预测上超越传统知识追踪方法 10%，并显著提升辅导效果。

**[ConvMix: A Mixed-Criteria Data Augmentation Framework for Conversational Dense Retrieval](llm_nlp/convmix_a_mixed-criteria_data_augmentation_framework_for_conversational_dense_re.md)**

:   提出 ConvMix 混合准则数据增强框架，从查询和文档双方向用 LLM 进行可扩展的相关性标注增强，并通过聚类多样性选择和 Fisher 信息近分布监督筛选，系统性提升对话式稠密检索性能。

**[Do Not Merge My Model! Safeguarding Open-Source LLMs Against Unauthorized Model Merging](llm_nlp/do_not_merge_my_model_safeguarding_open-source_llms_against_unauthorized_model_m.md)**

:   提出MergeBarrier，一种即插即用的防御方法，通过对注意力层施加正交投影、对FFN层进行激活函数展开重参数化，破坏受保护模型与同源模型之间的线性模态连通性（LMC），从而在不损失模型性能的前提下主动阻止未授权的模型合并。

**[Emergent Persuasion: Will LLMs Persuade Without Being Prompted?](llm_nlp/emergent_persuasion_will_llms_persuade_without_being_prompted.md)**

:   研究 LLM 在未被提示说服的情况下是否会自发产生说服行为：发现激活引导（steering）无法可靠诱发说服倾向，但在良性说服数据上的 SFT 微调会导致模型在有害话题上产生涌现性说服行为，揭示了后训练安全风险。

**[GloCTM: Cross-Lingual Topic Modeling via a Global Context Space](llm_nlp/gloctm_cross-lingual_topic_modeling_via_a_global_context_space.md)**

:   提出GloCTM，通过双路径VAE架构（局部语言路径+全局上下文路径）结合Polyglot Augmentation（跨语言近邻词扩充输入）、KL散度内部对齐、统一解码器结构对齐和CKA语义对齐四重机制，在3个跨语言数据集上全面超越现有方法的主题质量和跨语言对齐度。

**[Graph Out-of-Distribution Detection via Test-Time Calibration with Dual Dynamic Dictionaries](llm_nlp/graph_out-of-distribution_detection_via_test-time_calibration_with_dual_dynamic_.md)**

:   提出 BaCa 框架，在测试阶段通过 graphon 估计 + mixup 策略生成边界感知的合成图拓扑，结合双优先队列动态字典和注意力机制自适应校准 OOD 分数，无需微调预训练模型或引入辅助OOD数据，在全部 10 个数据集上超越 GOODAT，平均 AUC 提升 8.37%。

**[Guess or Recall? Training CNNs to Classify and Localize Memorization in LLMs](llm_nlp/guess_or_recall_training_cnns_to_classify_and_localize_memorization_in_llms.md)**

:   在 LLM 注意力权重上训练 CNN 来评估记忆化分类法与实际注意力机制的对齐程度，提出新的三类分类法（Guess/Recall/Non-Memorized），最小 F1 从 64.7% 提升至 89.0%，并定位了不同记忆类型分别依赖低层（Guess）和高层（Recall）注意力。

**[Hallucination Stations: On Some Basic Limitations of Transformer-Based Language Models](llm_nlp/hallucination_stations_on_some_basic_limitations_of_transformer-based_language_m.md)**

:   从计算复杂性角度分析LLM幻觉和能力局限，论证超过特定计算复杂度后LLM不仅无法正确执行任务，甚至无法验证其输出的正确性，为幻觉问题划定理论边界。

**[How Does Alignment Enhance LLMs' Multilingual Capabilities? A Language Neurons Perspective](llm_nlp/how_does_alignment_enhance_llms_multilingual_capabilities_a_language_neurons_per.md)**

:   提出三元神经元分类（语言特定/语言相关/通用），将 LLM 多语言推理分为四阶段分析，发现多语言对齐通过增加语言相关神经元（减少语言特定神经元）来提升性能，且在未训练语言上也产生"自发多语言对齐"效应。

**[HSKBenchmark: Modeling and Benchmarking Chinese Second Language Acquisition in Large Language Models through Curriculum Tuning](llm_nlp/hskbenchmark_modeling_and_benchmarking_chinese_second_language_acquisition_in_la.md)**

:   提出 HSKBenchmark，首个面向 LLM 中文二语习得（SLA）分阶段建模与写作评估的基准，包含 HSK 3-6 级教材（6.76M tokens）、16K 合成指令数据、30 个测试题目及语言学评估系统，配合课程式微调框架模拟人类习得轨迹。

**[Hypothesis Generation via LLM-Automated Language Bias for ILP](llm_nlp/hypothesis_generation_via_llm-automated_language_bias_for_ilp.md)**

:   提出首个端到端框架：多Agent LLM系统（Actor/Critic）自动从原始文本构建ILP语言偏差（谓词系统+类型声明+模式约束），Translator将文本翻译为Prolog事实，再由MAXSYNTH求解器基于MDL原则归纳全局最优规则集。在SHOES和ZENDO任务上分别达88.3%和81.3%准确率，跨4种LLM方差<5%。

**[ICL-Router: In-Context Learned Model Representations for LLM Routing](llm_nlp/icl-router_in-context_learned_model_representations_for_llm_routing.md)**

:   提出 ICL-Router，通过两阶段训练（查询重建 + ICL模型路由）将 LLM 的能力画像编码为 in-context 向量，实现可扩展的动态模型路由——新增模型无需重训路由器，在分布内和分布外任务上均达到 SOTA。

**[Identifying and Analyzing Performance-Critical Tokens in Large Language Models](llm_nlp/identifying_and_analyzing_performance-critical_tokens_in_large_language_models.md)**

:   通过representation-level和token-level两种消融实验，发现LLM在ICL中直接依赖的"性能关键token"是模板和停用词token（如"Answer:"），而非人类会关注的内容token（如实际文本），并揭示了LLM通过将内容信息聚合到这些关键token的表示中来间接利用内容。

**[Improving Sustainability Of Adversarial Examples In Class-Incremental Learning](llm_nlp/improving_sustainability_of_adversarial_examples_in_class-incremental_learning.md)**

:   提出SAE框架解决类增量学习（CIL）中对抗样本因域漂移而失效的问题，通过语义校正模块（CLIP+CIL模型联合引导）和过滤增强模块（去除语义混淆样本），使对抗样本在类别数增长9倍后仍保持攻击效果，平均攻击成功率提升31.28%。

**[Induce, Align, Predict: Zero-Shot Stance Detection via Cognitive Inductive Reasoning](llm_nlp/induce_align_predict_zero-shot_stance_detection_via_cognitive_inductive_reasonin.md)**

:   提出CIRF（Cognitive Inductive Reasoning Framework），受认知科学启发，从原始文本中无监督归纳一阶逻辑推理模式（schema），构建多关系schema图，用图核模型对齐输入与schema模板实现可解释的零样本立场推理，在SemEval-2016、VAST和COVID-19-Stance上达到SOTA，仅30%数据即可匹配全量。

**[Learning Spatial Decay for Vision Transformers](llm_nlp/learning_spatial_decay_for_vision_transformers.md)**

:   提出 Spatial Decay Transformer（SDT），首次将数据依赖的空间衰减机制从 1D 序列建模适配到 2D 视觉 Transformer，通过 Context-Aware Gating（CAG）生成动态的、内容相关的 patch 交互衰减强度，在 ImageNet-1K 分类和生成任务上一致超越 RMT 等强基线。

**[Llm-As-A-Judge For Scalable Test Coverage Evaluation Accuracy Operational Reliab](llm_nlp/llm-as-a-judge_for_scalable_test_coverage_evaluation_accuracy_operational_reliab.md)**

:   将LLM-as-Judge范式应用于Gherkin验收测试覆盖率评估，在20种模型配置x500次评估中系统量化准确性-可靠性-成本三维权衡，发现GPT-4o Mini以6.07 MAAE、96.6% ECR@1和$1.01/1K评估成为最优生产选择，成本仅为GPT-5高推理版的1/78。

**[LLM Targeted Underperformance Disproportionately Impacts Vulnerable Users](llm_nlp/llm_targeted_underperformance_disproportionately_impacts_vulnerable_users.md)**

:   系统实验表明，主流LLM（GPT-4、Claude 3 Opus、Llama 3-8B）对英语水平较低、教育程度较低、非美国出身的用户，在信息准确性、真实性和拒绝回答方面存在显著的歧视性表现下降，使最脆弱的用户成为最不可靠的信息服务对象。

**[LoKI: Low-damage Knowledge Implanting of Large Language Models](llm_nlp/loki_low-damage_knowledge_implanting_of_large_language_models.md)**

:   提出LoKI，一种基于Transformer知识存储机制理解的参数高效微调方法，通过知识向量归因（KVA）评估FFN中各知识向量的贡献度，选择低贡献向量进行层均衡的知识植入，在获得强任务性能的同时显著缓解灾难性遗忘。

**[LoopLLM: Transferable Energy-Latency Attacks in LLMs via Repetitive Generation](llm_nlp/loopllm_transferable_energy-latency_attacks_in_llms_via_repetitive_generation.md)**

:   提出LoopLLM，一种通过诱导LLM进入重复生成模式来发起能耗延迟攻击的框架，利用重复诱导提示优化和token对齐的集成优化，在12个开源和2个商业LLM上实现超过90%最大输出长度的攻击效果，跨模型迁移性提升约40%。

**[Lost in Benchmarks? Rethinking Large Language Model Benchmarking with Item Response Theory](llm_nlp/lost_in_benchmarks_rethinking_large_language_model_benchmarking_with_item_respon.md)**

:   提出 PSN-IRT（Pseudo-Siamese Network for IRT），用增强版项目反应理论同时估计 LLM 能力参数和题目的四参数特征（难度/区分度/猜测率/可行性），在 11 个基准 41,871 题上发现当前基准存在广泛饱和、难度天花板不足、数据污染等系统性问题，PSN-IRT 选出的题目子集排名一致性达 Kendall τ=1.00。

**[Low-Rank Curvature for Zeroth-Order Optimization in LLM Fine-Tuning](llm_nlp/low-rank_curvature_for_zeroth-order_optimization_in_llm_fine-tuning.md)**

:   提出 LOREN，一种曲率感知的零阶优化方法，通过低秩块对角预条件器捕获损失景观的各向异性曲率，并结合 REINFORCE Leave-One-Out 方差缩减技术，在 LLM 微调中实现了更高精度和更快收敛，同时相比 MeZO-Adam 节省高达 27.3% 的峰值内存。

**[MAPS: Multi-Agent Personality Shaping for Collaborative Reasoning](llm_nlp/maps_multi-agent_personality_shaping_for_collaborative_reaso.md)**

:   提出 MAPS 五 Agent 协作推理框架，基于大五人格理论为 4 个功能 Agent 赋予不同"性格"（Interpreter-开放性、Aligner-宜人性、Scholar-尽责性、Solver-外向性）实现异质化协作，加上 Critic Agent（神经质→苏格拉底式反思）做迭代修正，在 MathVista/OlympiadBench/EMMA 上超越 GPT-4o 基线 15.84%，首次超过人类专家 3.58%。

**[MCTS-SQL: Light-Weight LLMs can Master the Text-to-SQL through Monte Carlo Tree Search](llm_nlp/mcts-sql_light-weight_llms_can_master_the_text-to-sql_through_monte_carlo_tree_s.md)**

:   提出MCTS-SQL，让轻量LLM（如Qwen-1.5B）通过蒙特卡洛树搜索实现强大的Text-to-SQL能力——三组件架构（Selector做Schema剪枝 + Direct Generator生成初始SQL + MCTS-Refiner迭代精化），配合前缀缓存机制减少53%推理时间，Qwen-1.5B在BIRD上达40.69%执行准确率（超ChatGPT-3.5）。

**[Mem-PAL: Towards Memory-based Personalized Dialogue Assistants for Long-term User-Agent Interaction](llm_nlp/mem-pal_towards_memory-based_personalized_dialogue_assistants_for_long-term_user.md)**

:   提出H2Memory四层分层异构记忆结构（日志图/背景记忆/主题大纲/原则），通过PAL-Set数据集（100用户×8.4个月交互）验证，在需求重述和方案建议任务上将BLEU-1从13.59提升至26.67。

**[MindVote: When AI Meets the Wild West of Social Media Opinion](llm_nlp/mindvote_when_ai_meets_the_wild_west_of_social_media_opinion.md)**

:   提出 MindVote——首个基于真实社交媒体投票数据的 LLM 舆情预测基准，包含 Reddit/微博上 3,918 个自然投票（23 个话题），附带平台和话题上下文。评估 15 个 LLM 发现：最佳模型（o3-medium）1-Wasserstein 仅 0.892 vs 上界 0.972；在调查数据上微调的专用模型反而不如通用模型（"调查特化陷阱"）；模型表现出强烈文化对齐——西方模型擅长 Reddit、中国模型擅长微博。

**[Mitigating Content Effects on Reasoning in Language Models through Fine-Grained Activation Steering](llm_nlp/mitigating_content_effects_on_reasoning_in_language_models_through_fine-grained_.md)**

:   通过激活转向（activation steering）技术缓解 LLM 中的内容效应偏见——模型将内容可信度与形式逻辑有效性混淆的问题，提出 K-CAST（基于 kNN 的条件激活转向）方法，在不响应静态转向的模型上实现高达 15% 的形式推理准确率提升。

**[Multiplicative Orthogonal Sequential Editing for Language Models (MOSE)](llm_nlp/multiplicative_orthogonal_sequential_editing_for_language_models.md)**

:   提出 MOSE（乘法正交序列编辑），用正交矩阵左乘（而非加法更新）参数矩阵来注入新知识，严格保持编辑后矩阵的范数和条件数不变，在序列编辑中实现 12.08% 的性能提升并保留 95.73% 通用能力。

**[No-Regret Strategy Solving in Imperfect-Information Games via Pre-Trained Embedding](llm_nlp/no-regret_strategy_solving_in_imperfect-information_games_via_pre-trained_embedd.md)**

:   提出 Embedding CFR 算法，将不完美信息博弈中的信息集映射到连续低维嵌入空间（而非离散聚类），在相同空间开销下实现更快的可利用性收敛和更高质量的策略求解。

**[OptScale: Probabilistic Optimality for Inference-time Scaling](llm_nlp/optscale_probabilistic_optimality_for_inference-time_scaling.md)**

:   提出概率最优框架 OptScale，通过建模验证器分数的概率分布推导出最优采样数量的理论下界，动态决定每个问题所需的最少采样次数，在保持推理准确率的同时大幅减少计算开销。

**[ParetoHqD: Fast Offline Multiobjective Alignment of Large Language Models Using Pareto High-Quality Data](llm_nlp/paretohqd_fast_offline_multiobjective_alignment_of_large_language_models_using_p.md)**

:   提出 ParetoHqD，将人类偏好表示为目标空间中的偏好方向（而非线性标量化），通过选取靠近 Pareto 前沿的高质量数据做两阶段 SFT，用仅 42% 的 GPU 时间实现优于 5 个基线的多目标 LLM 对齐效果。

**[Position on LLM-Assisted Peer Review: Addressing Reviewer Gap through Mentoring and Feedback](llm_nlp/position_on_llm-assisted_peer_review_addressing_reviewer_gap_through_mentoring_a.md)**

:   本文作为立场论文，提出将LLM在同行评审中的角色从"自动生成审稿意见"转向"增强人类审稿能力"——通过LLM驱动的导师系统（三阶段培训+认证）和反馈系统（违规检测+证据反馈+可靠性测试）来缩小审稿质量差距。

**[PRECISE: Reducing the Bias of LLM Evaluations Using Prediction-Powered Ranking Estimation](llm_nlp/precise_reducing_the_bias_of_llm_evaluations_using_prediction-powered_ranking_es.md)**

:   将Prediction-Powered Inference（PPI）框架扩展到子实例级别的排序指标（如Precision@K），通过仅30-100条人工标注+大量LLM评判结果获得无偏的排序指标估计，计算复杂度从 $O(2^{|C|})$ 降至 $O(2^K)$，在印度电商搜索场景中成功指导LLM查询改写系统上线。

**[Profuser Progressive Fusion Of Large Language Models](llm_nlp/profuser_progressive_fusion_of_large_language_models.md)**

:   提出ProFuser，通过双模式优势评估（训练模式Min-CE + 推理模式Reward Model投票）全面识别各源模型在不同维度的优势，再用渐进式融合策略（先推理模式→后训练模式的easy-to-hard课程）将异构LLM的互补能力整合到单个目标模型中，在知识/推理/安全6个基准上平均提升1.65%。

**[Quiet Feature Learning in Algorithmic Tasks](llm_nlp/quiet_feature_learning_in_algorithmic_tasks.md)**

:   在 10 个算法任务（18,544 次训练运行，$10^9$-$10^{16}$ FLOPs）上发现，Transformer 的损失平台期并非学习停滞——模型在此期间悄悄学习了"安静特征"（中间算法子程序），这些特征不直接降低输出损失但对最终性能因果必要（消融后准确率下降 41-75%）。这挑战了用损失曲线判断训练进展的常规做法。

**[Rectification Reimagined: A Unified Mamba Model for Image Correction and Rectangling with Prompts](llm_nlp/rectification_reimagined_a_unified_mamba_model_for_image_cor.md)**

:   从统一畸变矫正视角出发，提出 UniRect 框架，通过 Residual Progressive TPS 处理几何形变 + Residual Mamba Blocks 补偿退化，统一处理肖像校正、广角矩形化、拼接矩形化、旋转校正四种任务，并通过 Sparse MoE 实现 four-in-one 多任务学习，拼接矩形化 PSNR 提升 3.82 dB，旋转校正提升 0.87 dB。

**[ReFeed: Retrieval Feedback-Guided Dataset Construction for Style-Aware Query Rewriting](llm_nlp/refeed_retrieval_feedback-guided_dataset_construction_for_style-aware_query_rewr.md)**

:   提出一个检索反馈驱动的数据集生成框架，通过识别检索失败case、LLM风格化改写、重检索验证三步闭环，自动构建高质量的风格感知查询改写数据集，为训练检索对齐的改写模型提供数据基础。

**[Scalable and Accurate Graph Reasoning with LLM-Based Multi-Agents](llm_nlp/scalable_and_accurate_graph_reasoning_with_llm-based_multi-agents.md)**

:   提出 GraphAgent-Reasoner（GAR），受分布式图计算理论启发，将图问题分解为以节点为中心的子任务分配给多个 Agent，通过邻居消息传递协作求解，将 LLM 可处理的图规模从 100 个节点扩展到 1000 个，在多项式时间图推理任务上显著超越现有最佳方法。

**[Scaling and Transferability of Annealing Strategies in Large Language Model Training](llm_nlp/scaling_and_transferability_of_annealing_strategies_in_large_language_model_trai.md)**

:   提出模型无关的预测框架，分解训练损失为前向效应项（学习率积分S）、退火动量项（Adam-style动量积分M）和模型尺寸项N，证明退火策略可从小模型/小batch迁移到大模型/大batch，预测误差MAPE<2%。

**[Scaling Equitable Reflection Assessment in Education via Large Language Models](llm_nlp/scaling_equitable_reflection_assessment_in_education_via_large_language_models_a.md)**

:   研究用 LLM 自动评估教育场景中学生的反思写作质量——在保持与人类评分者高一致性的同时，系统分析了 LLM 评估在种族、性别、社会经济背景等维度上的公平性，发现 LLM 评分可以达到甚至超越人类评分者间的一致性，但在某些人口统计维度上仍存在偏差。

**[TransMamba: A Sequence-Level Hybrid Transformer-Mamba Language Model](llm_nlp/transmamba_a_sequence-level_hybrid_transformer-mamba_language_model.md)**

:   提出 TransMamba，一种序列级别的 Transformer-Mamba 混合架构，通过共享 QKV/CBx 参数和 Memory Converter 在不同 token 长度时动态切换 Attention 和 SSM，兼顾长短序列的效率。

**[Uncertainty Under the Curve: A Sequence-Level Entropy Area Metric for Reasoning LLMs](llm_nlp/uncertainty_under_the_curve_a_sequence-level_entropy_area_metric_for_reasoning_l.md)**

:   提出 Entropy Area Score (EAS)——通过单次前向传播积分 token 级预测熵来量化推理 LLM 的不确定性。EAS 无需外部模型或重复采样，与答案熵强相关（Pearson r=0.82），用于训练数据选择时比 Pass Rate 过滤多提升 1.2-2.3% Pass@1，是高效可解释的 LLM 不确定性工具。

**[Uncovering Pretraining Code in LLMs: A Syntax-Aware Attribution Approach](llm_nlp/uncovering_pretraining_code_in_llms_a_syntax-aware_attribution_approach.md)**

:   提出SynPrune——首个语法感知的代码成员推断攻击方法，通过识别47种Python语法约定并在计算成员推断分数时剪除语法决定的token（仅保留反映作者特征的token），平均AUROC提升15.4%，可有效检测代码LLM的预训练数据归属。

**[Vision Transformers are Circulant Attention Learners](llm_nlp/vision_transformers_are_circulant_attention_learners.md)**

:   发现 ViT 的自注意力内禁学习了 BCCB 模式，据此提出 Circulant Attention，通过 2D FFT 实现 $O(N\log N)$ 复杂度，在 ImageNet 分类、COCO 检测、ADE20K 分割上一致提升。

**[VSPO: Validating Semantic Pitfalls in Ontology via LLM-Based CQ Generation](llm_nlp/vspo_validating_semantic_pitfalls_in_ontology_via_llm-based_cq_generation.md)**

:   提出 VSPO 框架，通过构造"定义-公理"错位数据集并微调 LLaMA-3.1-8B-Instruct，生成能够验证本体语义陷阱（如 allValuesFrom 误用）的能力问题（CQ），精度和召回率分别超过 GPT-4.1 达 26% 和 28.2%。

**[Where Norms and References Collide: Evaluating LLMs on Normative Reasoning](llm_nlp/where_norms_and_references_collide_evaluating_llms_on_normative_reasoning.md)**

:   提出 SNIC 诊断测试台（9,000 实例/51 场景），评估 LLM 能否利用隐式社会规范来解决歧义参考消解（如"递给我杯子"时存在多个杯子）。发现 LLM 在仅看场景描述时平均准确率仅 44%，加上 Prolog 形式逻辑无显著改善（44.2%），但显式提供规范列表后猛升到 70.5%（GPT-4.1 达 99.6%），证明 LLM 缺乏隐式物理规范知识但能有效利用显式规范。

**[X-MuTest: A Multilingual Benchmark for Explainable Hate Speech Detection](llm_nlp/x-mutest_a_multilingual_benchmark_for_explainable_hate_speech_detection_and_a_no.md)**

:   提出 X-MuTest，一个多语言可解释仇恨言论检测基准，覆盖多种语言和文化背景，评估 LLM 不仅检测仇恨言论的能力，更关注其提供可解释性理由的能力，发现当前模型在多语言和跨文化场景中存在显著性能差异。

---

## 🦾 LLM Agent { #llm_agent }

**[A2Flow: Automating Agentic Workflow Generation via Self-Adaptive Abstraction Operators](llm_agent/a2flow_automating_agentic_workflow_generation_via_self-adaptive_abstraction_oper.md)**

:   提出 A2Flow 框架，通过三阶段流水线（案例生成→功能聚类→深度提取）从专家数据中全自动提取可复用的抽象执行算子，替代人工预定义算子，并引入算子记忆机制累积中间输出辅助节点决策，在 8 个基准上整体超越 AFLOW 等 SOTA，资源消耗降低 37%。

**[A Multi-Agent Conversational Bandit Approach to Online Evaluation and Selection of User-Aligned LLM Responses](llm_agent/a_multi-agent_conversational_bandit_approach_to_online_evaluation_and_selection_.md)**

:   提出 MACO（Multi-Agent Conversational Online Learning），将 LLM 回复选择建模为多 Agent 对话式赌博机问题，通过本地 Agent 淘汰低质量回复 + 云端自适应关键词对话收集偏好，实现近似最优的在线回复评估和用户偏好对齐。

**[KDR-Agent: A Multi-Agent LLM Framework for Multi-Domain Low-Resource In-Context NER via Knowledge Retrieval](llm_agent/a_multi-agent_llm_framework_for_multi-domain_low-resource_in-context_ner_via_kno.md)**

:   提出 KDR-Agent 多 Agent 框架，通过中央规划器协调知识检索、上下文消歧和反思纠错三个专用 Agent，结合自然语言类型定义和实体级正负对比示例，无需微调即可在 5 个领域 10 个低资源 NER 数据集上全面超越 zero-shot 和 few-shot 基线（GPT-4o 上 BC5CDR F1=82.47，WNUT-17 F1=80.78）。

**[A Multi-Agent LLM Framework for Multi-Domain Low-Resource In-Context NER via Knowledge Retrieval, Disambiguation and Reflective Analysis](llm_agent/a_multi-agent_llm_framework_for_multi-domain_low-resource_in.md)**

:   提出 KDR-Agent 多智能体框架，通过知识检索（Wikipedia）、歧义消解和反思式自我纠错三个专业智能体协同工作，在仅使用少量静态标注示例的条件下，在5个领域10个NER数据集上显著超越现有零样本和少样本ICL NER方法。

**[AgentSense: Virtual Sensor Data Generation Using LLM Agents in Simulated Home Environments](llm_agent/agentsense_virtual_sensor_data_generation_using_llm_agents_i.md)**

:   利用LLM驱动的具身智能体在模拟智能家居中"生活"，生成虚拟环境传感器数据用于预训练HAR模型，在低资源场景下显著提升活动识别性能。

**[AgentSwift: Efficient LLM Agent Design via Value-guided Hierarchical Search](llm_agent/agentswift_efficient_llm_agent_design_via_value-guided_hierarchical_search.md)**

:   提出AgentSwift框架，通过层次化搜索空间（同时优化agentic workflow和功能组件）、轻量级value model预测agent性能、以及不确定性引导的MCTS搜索策略，自动发现高性能LLM agent设计，在7个基准上平均提升8.34%。

**[AquaSentinel: Next-Generation AI System Integrating Sensor Networks for Urban Underground Water Pipeline Anomaly Detection via Collaborative MoE-LLM Agent Architecture](llm_agent/aquasentinel_next-generation_ai_system_integrating_sensor_ne.md)**

:   提出AquaSentinel，一个物理信息驱动的AI系统，通过稀疏传感器部署+物理增强虚拟传感器+MoE时空GNN集成+双阈值RTCA检测算法+因果流定位+LLM报告生成，仅用20-30%节点覆盖即可实现全网管道泄漏检测，在110个泄漏场景中达到100%检测率。

**[ARCANE: A Multi-Agent Framework for Interpretable and Configurable Alignment](llm_agent/arcane_a_multi-agent_framework_for_interpretable_and_configurable_alignment.md)**

:   提出ARCANE框架，将对齐建模为多智能体协作问题——manager agent通过与stakeholder对话学习生成自然语言rubric（加权可验证准则集），作为worker agent的可解释代理奖励函数，通过SFT+GSPO两阶段训练实现测试时可配置的对齐，在GDPVal基准上GSPO版本的mean return从0.58提升至0.74（N=8）。

**[Automating Complex Document Workflows Via Stepwise And Rollback-Enabled Operatio](llm_agent/automating_complex_document_workflows_via_stepwise_and_rollback-enabled_operatio.md)**

:   提出AutoDW框架，通过逐步规划（每次生成一个API调用）+自适应回滚（参数级+API级两层回滚）实现复杂文档工作流自动化，在250会话/1708指令的DWBench上达到90%指令级和62%会话级完成率，分别超越最强基线40%和76%。

**[AutoTool: Efficient Tool Selection for Large Language Model Agents](llm_agent/autotool_efficient_tool_selection_for_large_language_model_a.md)**

:   提出AutoTool，一个基于图的免训练工具选择框架，通过发现并利用"工具使用惯性"（tool usage inertia）——即工具调用遵循可预测的顺序模式这一经验现象——构建工具惯性图（TIG），用统计方法代替部分LLM推理来高效选择工具和填充参数，在保持任务完成率的同时减少15-40%的推理成本。

**[AutoTool: Efficient Tool Selection for Large Language Model Agents](llm_agent/autotool_efficient_tool_selection_for_large_language_model_agents.md)**

:   提出 AutoTool，一种基于图的工具选择框架，利用工具使用惯性（tool usage inertia）构建工具惯性图（TIG），通过统计结构绕过重复的 LLM 推理来选择工具和填充参数，在保持任务完成率的同时减少最多 30% 的推理开销。

**[BayesAgent: Bayesian Agentic Reasoning Under Uncertainty via Verbalized Probabilistic Graphical Modeling](llm_agent/bayesagent_bayesian_agentic_reasoning_under_uncertainty_via_.md)**

:   提出 vPGM 框架，通过自然语言引导 LLM Agent 模拟概率图模型（PGM）的贝叶斯推理过程，发现隐变量并推断后验分布，再用 Dirichlet 先验做数值贝叶斯校准（BayesVPGM），在多个推理任务上同时提升准确率和置信度校准。

**[Beyond ReAct: A Planner-Centric Framework for Complex Tool-Augmented LLM Reasoning](llm_agent/beyond_react_a_planner-centric_framework_for_complex_tool-au.md)**

:   提出以Planner为核心的Plan-Execute框架，将复杂查询转化为DAG执行计划，通过SFT+GRPO两阶段训练专门的Planner模型，在ComplexTool-Plan和StableToolBench上超越ReAct等反应式方法，用更少推理步骤实现更高成功率。

**[CausalTrace: A Neurosymbolic Causal Analysis Agent for Smart Manufacturing](llm_agent/causaltrace_a_neurosymbolic_causal_analysis_agent_for_smart_manufacturing.md)**

:   提出 CausalTrace——一个集成于工业 CoPilot（SmartPilot）中的神经符号因果分析智能体，融合数据驱动因果发现与工业本体/知识图谱，实现了实时的根因分析、反事实推理和可解释决策支持。

**[Co-EPG: A Framework for Co-Evolution of Planning and Grounding in Autonomous GUI Agents](llm_agent/co-epg_a_framework_for_co-evolution_of_planning_and_groundin.md)**

:   提出Co-EPG框架，将GUI Agent解耦为Planning和Grounding两个模型，通过GRPO协同训练和基于置信度的动态奖励集成机制（C-DREM）建立正反馈循环，使两个模型自迭代协同进化，仅用基准数据集（无需外部数据）即在Multimodal-Mind2Web（58.4%）和AndroidControl（83.1%）上达到SOTA。

**[COACH: Collaborative Agents for Contextual Highlighting -- A Multi-Agent Framework for Sports Video Analysis](llm_agent/coach_collaborative_agents_for_contextual_highlighting_--_a_multi-agent_framewor.md)**

:   提出 COACH 框架——一个基于共享骨干模型的可重配置多智能体系统，通过意图驱动的策略编排和结构化 CoT 微调实现角色专业化，在羽毛球视频分析的 QA 和摘要两个任务上显著超越 Gemini 2.5 Pro 等通才模型。

**[Cook and Clean Together: Teaching Embodied Agents for Parallel Task Execution](llm_agent/cook_and_clean_together_teaching_embodied_agents_for_paralle.md)**

:   提出ORS3D任务——将运筹学(OR)知识引入具身AI的任务调度，要求智能体利用可并行子任务的等待时间执行其他任务以最小化总完成时间，同时在3D场景中定位目标物体；构建60K级数据集ORS3D-60K，并提出GRANT模型通过调度token机制连接外部动态规划求解器，在时间效率上比baseline提升30.53%。

**[COVR: Collaborative Optimization of VLMs and RL Agent for Visual-Based Control](llm_agent/covrcollaborative_optimization_of_vlms_and_rl_agent_for_visu.md)**

:   提出 VLM 与 RL 双向协同优化框架 COVR：RL 生成的高质量交互数据用于微调 VLM，增强后的 VLM 反过来通过 action prior 指导 RL 策略学习，在 CARLA 和 DMControl 上取得 SOTA。

**[D-GARA: A Dynamic Benchmarking Framework for GUI Agent Robustness in Real-World Anomalies](llm_agent/d-gara_a_dynamic_benchmarking_framework_for_gui_agent_robust.md)**

:   提出 D-GARA，一个面向 Android GUI Agent 的动态鲁棒性评估框架，通过在实时交互过程中注入权限弹窗、电量警告、应用崩溃等真实世界异常，揭示现有 SOTA Agent（包括 UI-TARS-72B、GPT-4o）在中断场景下平均成功率下降超过 17.5%，最高达 33% 的严重脆弱性。

**[DEPO: Dual-Efficiency Preference Optimization for LLM Agents](llm_agent/depo_dual-efficiency_preference_optimization_for_llm_agents.md)**

:   提出双重效率（dual-efficiency）的概念，将 LLM Agent 的效率分解为 step 级（减少每步 token 数）和 trajectory 级（减少总步数），并基于 KTO 设计了 DEPO 方法，通过在 desirable 样本的 reward 中加入效率 bonus 来联合优化效率与性能。

**[EcoAgent: An Efficient Device-Cloud Collaborative Multi-Agent Framework for Mobile Automation](llm_agent/ecoagent_an_efficient_device-cloud_collaborative_multi-agent.md)**

:   提出 EcoAgent，一个闭环设备-云端协作的多 Agent 移动自动化框架，通过 Dual-ReACT 双层推理规划 + 设备端轻量验证反馈 + Pre-Understanding 文本压缩模块，在 AndroidWorld 上达到与全云端 Agent 相当的成功率，同时大幅降低延迟（3.9s vs 15.3s）、云端调用（降89%）和上行数据量（降48.6倍）。

**[Extracting Events Like Code: A Multi-Agent Programming Framework for Zero-Shot Event Extraction](llm_agent/extracting_events_like_code_a_multi-agent_programming_framework_for_zero-shot_ev.md)**

:   提出 Agent-Event-Coder (AEC)，将零样本事件抽取类比为软件工程流程，用4个专职Agent（Retrieval→Planning→Coding→Verification）协作完成抽取，并将事件schema编码为可执行Python类实现编译器式确定性验证与双循环迭代修正，在5个领域、6个LLM上全面超越零样本基线。

**[Fact2Fiction: Targeted Poisoning Attack to Agentic Fact-checking System](llm_agent/fact2fiction_targeted_poisoning_attack_to_agentic_fact-check.md)**

:   提出 Fact2Fiction，首个针对 Agent 化事实核查系统（如 DEFAME、InFact）的投毒攻击框架：通过 Planner Agent 模拟声明分解生成子问题，利用系统的 justification 反向工程关键推理点来制作定向恶意证据，并按重要性分配投毒预算，在仅 1% 投毒率下比 SOTA PoisonedRAG 高 8.9%-21.2% 的攻击成功率。

**[FinRpt: Dataset, Evaluation System and LLM-based Multi-agent Framework for Equity Research Report Generation](llm_agent/finrpt_dataset_evaluation_system_and_llm-based_multi-agent_framework_for_equity_.md)**

:   首次系统化地定义股票研究报告（ERR）自动生成任务——构建 FinRpt 数据集（6,825篇中英文高质量研报，整合7类金融数据），提出11指标评估体系和9 Agent协作的FinRpt-Gen生成框架（含评级修正/专家审查/润色三阶段增强），人类评估显示生成报告质量接近专家撰写。

**[From Biased Chatbots to Biased Agents: Examining Role Assignment Effects on LLM Agent Robustness](llm_agent/from_biased_chatbots_to_biased_agents_examining_role_assignment_effects_on_llm_a.md)**

:   首个系统性案例研究，揭示基于人口统计学的 persona 分配会导致 LLM Agent 在 5 个操作领域的任务执行中出现最高 26.2% 的性能下降，证明 persona 诱导的偏见从文本生成延伸到了行动决策层面。

**[History-Aware Reasoning for GUI Agents](llm_agent/history-aware_reasoning_for_gui_agents.md)**

:   提出 HAR 框架，通过构建反思学习场景、合成纠错指南、设计混合 RL 奖励函数（含 Memory-Augmented Reward），将 GUI Agent 的推理模式从"历史无感知"转变为"历史感知"，3B 模型在 AITW/Mind2Web/GUI-Odyssey 等多个 benchmark 上超越更大模型。

**[iMAD: Intelligent Multi-Agent Debate for Efficient and Accurate LLM Inference](llm_agent/imad_intelligent_multi-agent_debate_for_efficient_and_accura.md)**

:   iMAD 提出选择性触发多Agent辩论的框架：先让单Agent生成带自我批判的结构化响应，从中提取 41 个可解释的语言/语义特征，用轻量 MLP 分类器（FocusCal 损失训练）判断是否需要触发 MAD，在 6 个 QA/VQA 数据集上减少高达 92% 的 Token 开销，同时提升准确率高达 13.5%。

**[LieCraft: A Multi-Agent Framework for Evaluating Deceptive Capabilities in Language Models](llm_agent/liecraft_a_multi-agent_framework_for_evaluating_deceptive_capabilities_in_langua.md)**

:   设计LieCraft多人隐藏角色博弈框架（约束满足问题确保平衡），评估12个LLM的战略欺骗能力，发现所有测试的前沿LLM（含GPT-4）在激励下都展现90%+的欺骗率——安全训练未消除策略性撒谎能力。

**[Llandmark A Multi-Agent Framework For Landmark-Aware Multimodal Interactive Vide](llm_agent/llandmark_a_multi-agent_framework_for_landmark-aware_multimodal_interactive_vide.md)**

:   提出 LLandMark 模块化多 Agent 框架，通过地标知识增强、LLM 辅助图像检索和 OCR 精炼模块，在越南大规模视频检索挑战赛（HCMAIC 2025）中实现地标感知的多模态交互式视频检索，总分 77.40/88。

**[LLMTM: Benchmarking and Optimizing LLMs for Temporal Motif Analysis in Dynamic Graphs](llm_agent/llmtm_benchmarking_and_optimizing_llms_for_temporal_motif_analysis_in_dynamic_gr.md)**

:   提出 LLMTM——首个评估 LLM 处理动态图中时序 motif 分析能力的综合基准，包含 6 类任务覆盖 9 种时序 motif 类型，评估 9 个模型后发现 LLM 对时序 motif 的识别能力随 motif 复杂度快速下降。提出结构感知分派器（Structure-Aware Dispatcher），根据图的结构属性和认知负荷智能路由查询到标准 LLM 提示或工具增强 Agent，在维持高准确率的同时降低计算成本。

**[Loss-Guided Auxiliary Agents for Overcoming Mode Collapse in GFlowNets](llm_agent/loss-guided_auxiliary_agents_for_overcoming_mode_collapse_in_gflownets.md)**

:   提出 LGGFN（Loss-Guided GFlowNets），用辅助 GFlowNet 的探索直接由主 GFlowNet 的训练损失驱动——辅助 Agent 的奖励 = 原始奖励 + λ·主模型损失，优先采样主模型理解不足的区域，在网格/序列/贝叶斯结构学习任务上分别发现 40× 更多唯一模式、99% 探索误差降低。

**[MedLA: A Logic-Driven Multi-Agent Framework for Complex Medical Reasoning with Large Language Models](llm_agent/medla_a_logic-driven_multi-agent_framework_for_complex_medic.md)**

:   提出 MedLA，首个基于三段论逻辑树的医学多 Agent 推理框架：每个 Agent 将推理组织为显式的逻辑树（大前提-小前提-结论三段论节点），多个 Agent 通过图引导的多轮讨论在前提级别对齐和修正逻辑树，在 MedDDx 上超越所有基线 7.4%（8B 模型），在医学 QA 上以 8B 模型达到 69.9% 平均准确率（超 70B RAG 模型）。

**[MoralReason: Generalizable Moral Decision Alignment For LLM Agents Using Reasoning-Level Reinforcement Learning](llm_agent/moralreason_generalizable_moral_decision_alignment_for_llm_agents_using_reasonin.md)**

:   使用Group Relative Policy Optimization (GRPO)在推理层面训练LLM进行道德框架对齐，在Moral-Reason-QA数据集（680个高歧义场景）上实现功利主义对齐分数从0.207提升到0.964的分布外泛化。

**[Parallelism Meets Adaptiveness Scalable Documents Understanding In Multi-Agent L](llm_agent/parallelism_meets_adaptiveness_scalable_documents_understanding_in_multi-agent_l.md)**

:   提出自适应协调的多 Agent LLM 框架，通过并行竞争评估、动态任务路由和双向反馈机制，在高复杂度金融文档分析任务中实现 27% 的合规准确率提升和 74% 的修订率降低。

**[Pertouch Vlm-Driven Agent For Personalized And Semantic Image Retouching](llm_agent/pertouch_vlm-driven_agent_for_personalized_and_semantic_image_retouching.md)**

:   提出 PerTouch 框架，结合基于 Stable Diffusion + ControlNet 的语义区域级修图模型和 VLM 驱动的 Agent（含反馈重思考机制和场景感知记忆），实现精细化、个性化的图像修图。

**[Physics-Informed Autonomous LLM Agents for Explainable Power Electronics Modulation Design](llm_agent/physics-informed_autonomous_llm_agents_for_explainable_power_electronics_modulat.md)**

:   提出PHIA系统：LLM规划器通过聊天接口收集设计需求，协调物理信息神经网络代理模型（层次化PINN）和优化算法自主迭代生成电力转换器调制设计方案，MAE降低63.2%、设计速度提升33倍、20位专家验证可用性。

**[ProBench: Benchmarking GUI Agents with Accurate Process Information](llm_agent/probench_benchmarking_gui_agents_with_accurate_process_infor.md)**

:   提出 ProBench，首个同时评估"最终状态"和"操作过程"的移动端 GUI Agent benchmark：200+ 挑战性任务覆盖 34 个中英文主流 App，通过 Process Provider（Structure Description Converter + MLLM Summarizer）自动捕获精确的中间过程信息，评估发现最强模型 Gemini 2.5 Pro 也仅完成 40.1% 任务，暴露了 grounding 不足、历史操作感知差、任务规划过于简化三大普遍问题。

**[Promoting Sustainable Web Agents: Benchmarking and Estimating Energy Consumption Through Empirical and Theoretical Analysis](llm_agent/promoting_sustainable_web_agents_benchmarking_and_estimating_energy_consumption_.md)**

:   首次系统性地从实证基准测试和理论估算两个角度量化了 Web Agent 的能耗与碳排放，发现更高能耗并不等于更好性能，并倡导在评测中引入能效指标。

**[Prune4Web: DOM Tree Pruning Programming for Web Agent](llm_agent/prune4web_dom_tree_pruning_programming_for_web_agent.md)**

:   提出 Prune4Web，通过"LLM 生成评分函数参数 + 固定启发式模板执行"的编程式 DOM 剪枝方法实现 25-50 倍候选元素缩减：三阶段 pipeline（Planner 分解子任务 → Programmatic Filter 生成评分函数剪枝 DOM → Grounder 执行操作），3B 模型在 Multimodal-Mind2Web 上达到 52.4% Step SR（超越所有同参数量基线甚至部分 9.6B/32B 模型），低级 grounding 准确率从 46.8% 提升至 88.28%。

**[Real-Time Trust Verification For Safe Agentic Actions Using Trustbench](llm_agent/real-time_trust_verification_for_safe_agentic_actions_using_trustbench.md)**

:   提出TrustBench双模式框架：(1) 基准模式——结合传统指标和LLM-as-a-Judge评估8个信任维度，学习Agent置信度与实际正确率的校准映射；(2) 验证模式——在Agent制定行动后、执行前实时计算信任分数，阻止87%的有害行动，延迟低于200ms，通过领域插件（医疗/金融/QA）实现专业化验证。

**[Reflection-Driven Control for Trustworthy Code Agents](llm_agent/reflection-driven_control_for_trustworthy_code_agents.md)**

:   提出 Reflection-Driven Control 模块，将"自我反思"从事后补丁提升为 Agent 推理过程中的一等控制回路，通过轻量自检、证据驱动修复和反思记忆库三个组件，在安全代码生成任务上显著提升代码安全率。

**[SoMe: A Realistic Benchmark for LLM-based Social Media Agents](llm_agent/some_a_realistic_benchmark_for_llm-based_social_media_agents.md)**

:   提出 SoMe，首个全面评估 LLM 社交媒体 Agent 的 benchmark：8 个任务覆盖帖子分析、用户理解、综合推理，917 万帖子 + 6591 用户 + 1.7 万标注查询，配套 8 个 MCP 兼容工具，评估 13 个主流 LLM 发现最强模型仅 54.33 分（满分 100），揭示了推理能力≠Agent能力、工具调用幻觉普遍存在等关键发现。

**[SPAN: Benchmarking and Improving Cross-Calendar Temporal Reasoning of Large Language Models](llm_agent/span_benchmarking_and_improving_cross-calendar_temporal_reasoning_of_large_langu.md)**

:   提出SPAN跨日历时间推理基准（6种日历×10推理方向×100年范围×37380实例），发现基础LLM平均仅34.5%准确率（无一超过80%），揭示Future-Date Degradation和Calendar Asymmetry Bias两种系统性失败模式，工具增强的Time Agent达95.31%——证明跨日历推理需要外部工具而非参数化知识。

**[Structured Personalization: Modeling Constraints as Matroids for Data-Minimal LLM Agents](llm_agent/structured_personalization_modeling_constraints_as_matroids_for_data-minimal_llm.md)**

:   将 LLM Agent 个性化中的结构化约束（逻辑依赖 + 层级配额）形式化为层叠拟阵（laminar matroid），证明贪心算法在此约束下仍具有常数因子近似保证，解决了有依赖关系和层级限制的数据最小化选择问题。

**[Thucy: An LLM-based Multi-Agent System for Claim Verification across Relational Databases](llm_agent/thucy_an_llm-based_multi-agent_system_for_claim_verification_across_relational_d.md)**

:   提出首个跨数据库、跨表的多 Agent 声明验证系统 Thucy，由 Verifier 领导三个专家 Agent（Data/Schema/SQL Expert），对数据源完全无先验知识，能自主发现、推理并生成 SQL 证据，在 TabFact 上超越 SOTA 5.6 个百分点（94.3%）。

**[TongUI: Internet-Scale Trajectories from Multimodal Web Tutorials for Generalized GUI Agents](llm_agent/tongui_internet-scale_trajectories_from_multimodal_web_tutor.md)**

:   TongUI 提出从互联网上的多模态教程（视频+图文）自动转化为 GUI 操作轨迹数据的框架，构建了百万级的 GUI-Net-1M 数据集，用于微调 Qwen2.5-VL 模型，在多个 grounding 和 navigation 基准上超越或接近 UI-TARS 等 SOTA。

**[Towards Trustworthy Multi-Turn Llm Agents Via Behavioral Guidance](llm_agent/towards_trustworthy_multi-turn_llm_agents_via_behavioral_guidance.md)**

:   提出任务完成框架，通过任务分析器（Task Profiler）、推理模块（Reasoning Module）和生成模块（Generation Module）三组件协同进化，使 LLM Agent 在多轮交互环境中实现可验证和可靠的行为引导。

**[When Refusals Fail: Unstable Safety Mechanisms in Long-Context LLM Agents](llm_agent/when_refusals_fail_unstable_safety_mechanisms_in_long-context_llm_agents.md)**

:   系统研究 LLM Agent 在长上下文填充下的安全行为变化：发现声称支持 1M-2M token 的模型在 100K token 时已出现 >50% 的性能崩溃，拒绝率以不可预测的方式波动（GPT-4.1-nano 从 5% 升至 40%，Grok 4 Fast 从 80% 降至 10%），揭示了长上下文 Agent 系统的严重安全隐患。

**[With Great Capabilities Come Great Responsibilities: Introducing the Agentic Risk & Capability Framework for Governing Agentic AI Systems](llm_agent/with_great_capabilities_come_great_responsibilities_introducing_the_agentic_risk.md)**

:   提出 Agentic Risk & Capability (ARC) 框架，从能力（Capability）视角系统化地识别、评估和缓解智能体 AI 系统的安全与安全风险，为组织级治理提供可操作的结构化方法论。

---

## 💡 LLM 推理 { #llm_reasoning }

**[A Reasoning Paradigm for Named Entity Recognition](llm_reasoning/a_reasoning_paradigm_for_named_entity_recognition.md)**

:   提出 ReasoningNER，将命名实体识别从"隐式模式匹配"转变为"显式推理"范式，通过三阶段流程（CoT数据构建→CoT微调→GRPO强化增强）让模型先推理再抽取实体，在零样本设定下F1超GPT-4达12.3个百分点，8B模型在CrossNER上达72.4平均F1。

**[Answering the Unanswerable Is to Err Knowingly: Analyzing and Mitigating Abstention Failures in Large Reasoning Models](llm_reasoning/answering_the_unanswerable_is_to_err_knowingly_analyzing_and.md)**

:   系统分析大推理模型(LRM)面对不可回答数学题时的弃权失败现象，发现LRM内部有足够认知能力识别问题不可解（探针分类准确率>80%）但外部行为仍偏向强答，提出认知监控+推理时干预的两阶段方法，将弃权率从16-54%提升至60-92%且不损害可回答题的推理性能。

**[ARCHE: A Novel Task to Evaluate LLMs on Latent Reasoning Chain Extraction](llm_reasoning/arche_a_novel_task_to_evaluate_llms_on_latent_reasoning_chai.md)**

:   提出潜在推理链提取 (ARCHE) 任务，要求 LLM 将科学论文中的论证分解为基于 Peirce 三种推理范式的推理逻辑树 (RLT)，并通过 Entity Coverage 和 Reasoning Edge Accuracy 两个指标揭示了 10 个主流 LLM 在内容完整性与逻辑正确性之间的本质权衡。

**[BadThink: Triggered Overthinking Attacks on Chain-of-Thought Reasoning in Large Language Models](llm_reasoning/badthink_triggered_overthinking_attacks_on_chain-of-thought_reasoning_in_large_l.md)**

:   提出 BadThink——首个针对 CoT 推理效率的训练时后门攻击，通过 LLM 迭代优化生成自然的冗长推理模板进行数据投毒，触发后模型生成膨胀 17× 以上的推理链（MATH-500），同时保持最终答案正确和良好隐蔽性。

**[BLM-Guard: Explainable Multimodal Ad Moderation with Chain-of-Thought and Policy-Aligned Rewards](llm_reasoning/blm-guard_explainable_multimodal_ad_moderation_with_chain-of.md)**

:   提出 BLM-Guard，一个面向短视频商业广告的可解释多模态审核框架：先通过 Rule-driven ICoT 数据合成 + SFT 冷启动建立结构化推理能力，再用 Self-Adaptive GRPO 强化学习（结合规则正确性奖励 + 自适应一致性奖励 SCA-R）优化策略对齐，在真实广告 benchmark 上达到 91.4% 严格准确率和 0.845 推理一致性分数。

**[Chain-of-Thought Driven Adversarial Scenario Extrapolation for Robust Language Models](llm_reasoning/chain-of-thought_driven_adversarial_scenario_extrapolation_for_robust_language_m.md)**

:   提出 ASE（Adversarial Scenario Extrapolation），一种推理时 CoT 防御框架，让 LLM 在回答前自主模拟对抗场景并制定防御策略，在四类安全威胁（越狱、毒性、幻觉、偏见）上实现近零攻击成功率，同时将直接拒绝率降至≤4%，兼顾鲁棒性和用户体验。

**[CMMCoT: Enhancing Complex Multi-Image Comprehension via Multi-Modal Chain-of-Thought and Memory Augmentation](llm_reasoning/cmmcot_enhancing_complex_multi-image_comprehension_via_multi.md)**

:   提出 CMMCoT 框架，通过构建交错的多模态多步推理链（含视觉区域 token 监督）和测试时检索式记忆增强模块（RIFREM），在不增加参数的前提下提升多图场景下的慢思考推理能力，基于 Qwen2.5-VL-7B 在多图基准上平均提升 1.4 分。

**[Deep Hidden Cognition Facilitates Reliable Chain-of-Thought Reasoning](llm_reasoning/deep_hidden_cognition_facilitates_reliable_chain-of-thought_.md)**

:   本文发现 LLM 在 CoT 推理过程中，中间层的注意力头激活值隐式编码了推理步骤的真实性信息（最高 85% 探测准确率），据此训练置信度预测器引导 Beam Search 动态选择高置信度推理路径，在数学/符号/常识推理任务上超越 Self-Consistency 和 PRM Guided Search。

**[Dropouts in Confidence: Moral Uncertainty in Human-LLM Alignment](llm_reasoning/dropouts_in_confidence_moral_uncertainty_in_human-llm_alignment.md)**

:   系统研究 32 个开源 LLM 在道德困境（电车问题）中的决策不确定性，发现不确定性主要受模型架构而非道德维度驱动；在推理时引入 attention dropout 增加随机性后，模型的互信息显著上升，human-LLM 道德对齐度也随之改善——表明降低 LLM 在道德场景中的过度自信可以改善与人类偏好的一致性。

**[ESG-Bench: Benchmarking Long-Context ESG Reports for Hallucination Mitigation](llm_reasoning/esg-bench_benchmarking_long-context_esg_reports_for_hallucination_mitigation.md)**

:   构建 ESG-Bench——270 个人工标注 QA 对来自 94 份真实 ESG 报告（2020-2024），提出三阶段幻觉缓解：SFT（有基础答案+「不提供」弃权标签）→ CoT Prompting（2/4步提示模板）→ CoT 微调（人工推理链），其中 4 步 CoT 微调的 Llama-3 达到 92.52% 有答案准确率 + 99.37% 无答案准确率（平衡 96%），且迁移到 HaluEval/BioASQ 也有提升。

**[Evaluating, Synthesizing, and Enhancing for Customer Support Conversation](llm_reasoning/evaluating_synthesizing_and_enhancing_for_customer_support_conversation.md)**

:   基于COPC行业标准定义客服对话的5个阶段和12种策略，通过5个LLM Agent角色扮演生成11232条策略丰富的合成对话（RoleCS），并构建1855条真实对话改写的评估集（CSConv），微调后显著提升策略对齐的回复质量和问题解决率。

**[Exposing the Cracks: Vulnerabilities of Retrieval-Augmented LLM-Based Machine Translation](llm_reasoning/exposing_the_cracks_vulnerabilities_of_retrieval-augmented_llm-based_machine_tra.md)**

:   开发受控噪声注入框架系统评估检索增强翻译（REAL-MT），引入Fidelity和CAR两个新指标，在10语言对×4种噪声类型上揭示模型即使面对矛盾上下文仍盲目采纳（CAR保持65-78%），大推理模型（LRM）反而更脆弱（会"合理化"错误上下文），且噪声鲁棒性与干净上下文利用率存在根本性trade-off。

**[ExtendAttack: Attacking Servers of LRMs via Extending Reasoning](llm_reasoning/extendattack_attacking_servers_of_lrms_via_extending_reasoning.md)**

:   提出 ExtendAttack，一种针对大推理模型（LRM）的资源耗尽攻击：通过将 prompt 中的字符随机转换为多进制 ASCII 编码，迫使模型在回答问题前先执行大量逐字符解码推理，使 o3 的响应长度增加 2.7 倍以上、延迟翻倍，同时保持答案准确率基本不变。

**[Graph of Verification: Structured Verification of LLM Reasoning with Directed Acyclic Graphs](llm_reasoning/graph_of_verification_structured_verification_of_llm_reasoning_with_directed_acy.md)**

:   提出 Graph of Verification (GoV)，一种将 LLM 推理过程建模为有向无环图 (DAG) 的结构化验证框架，通过灵活的节点块(Node Block)架构实现多粒度验证——从形式化任务的原子步骤到自然语言叙述的段落级验证——在结构化和松散结构化推理基准上均显著优于整体验证和其他分解验证方法。

**[Improving Value-based Process Verifier via Low-Cost Variance Reduction](llm_reasoning/improving_value-based_process_verifier_via_low-cost_variance_reduction.md)**

:   针对基于值的过程验证器(PRM)训练中蒙特卡罗(MC)估计因采样数有限导致的高方差问题，提出Compound Monte Carlo Sampling (ComMCS)方法，通过线性组合当前步和后续步的MC估计量来无偏地降低方差，无需额外LLM推理开销，在MATH-500上Best-of-32实验中提升2.2个点。

**[Incorporating Self-Rewriting into Large Language Model Reasoning Reinforcement](llm_reasoning/incorporating_self-rewriting_into_large_language_model_reasoning_reinforcement.md)**

:   提出Self-Rewriting框架，让LRM在RL训练中对"简单"样本（全部回答正确的query）重写自身推理文本并从中学习，仅增加约10%训练开销即可在保持准确率的同时将推理长度减少46%，内部推理质量（LLM-as-Judge）提升7.2分，有效缓解过度思考、冗余思考等问题。

**[Intention Chain-of-Thought Prompting with Dynamic Routing for Code Generation](llm_reasoning/intention_chain-of-thought_prompting_with_dynamic_routing_for_code_generation.md)**

:   提出 RoutingGen——基于认知经济原则的难度感知代码生成框架：用 Qwen3-8B 分类器动态路由任务到简单路径（few-shot 直接生成）或复杂路径（Intention CoT = 规格约束 + 算法意图 + 复杂度分析），在 McEval 上提升 +45.15% 同时平均减少 46.37% token 消耗。

**[Jupiter: Enhancing LLM Data Analysis Capabilities via Notebook and Inference-Time Value-Guided Search](llm_reasoning/jupiter_enhancing_llm_data_analysis_capabilities_via_notebook_and_inference-time.md)**

:   构建NbQA数据集（从真实Jupyter Notebook提取3.8万task-solution对）+ 提出Jupiter框架（将数据分析建模为状态级搜索问题，用值模型引导PUCT搜索），使Qwen2.5-14B在InfiAgent-DABench上达86.38%超越GPT-4o(85.99%)，Qwen2.5-7B在DSBench上从63.51%提升至89.19%。

**[L2V-CoT: Cross-Modal Transfer of Chain-of-Thought Reasoning via Latent Intervention](llm_reasoning/l2v-cot_cross-modal_transfer_of_chain-of-thought_reasoning_v.md)**

:   通过 LAT 分析发现 LLM 和 VLM 的低频 CoT 方向表示具有相似分布，提出 L2V-CoT：从 LLM 提取 CoT 方向表示 → 低通滤波 → 频域重采样匹配维度 → 注入 VLM 隐藏层，training-free 地将 LLM 的推理能力迁移到 VLM，平均提升 3.7%，最高 8.6%。

**[N2N-GQA: Noise-to-Narrative for Graph-Based Table-Text Question Answering Using LLMs](llm_reasoning/n2n-gqa_noise-to-narrative_for_graph-based_table-text_question_answering_using_l.md)**

:   提出 N2N-GQA——首个用于开放域混合表格-文本问答的零样本框架，核心思路是将检索到的嘈杂文档构建为动态证据图（文档为节点、TF-IDF共享词为边），通过图中心性剪枝识别"桥接文档"连接多跳推理链，在 OTT-QA 上比 Vanilla RAG 提升 +39.6 EM（从 8.0 到 48.8），零样本即接近微调系统 CORE (49.0 EM)。

**[PRIME: Planning and Retrieval-Integrated Memory for Enhanced Reasoning](llm_reasoning/prime_planning_and_retrieval-integrated_memory_for_enhanced_reasoning.md)**

:   受双系统认知理论启发，提出PRIME多Agent推理框架——Quick Thinking Agent（System 1）快速生成直觉答案，Reflection Agent评估可信度，不确定时触发System 2的6个专门化Agent（规划/搜索/阅读/假设/整合/决策）进行深度知识检索推理，使开源LLaMA 3在医学/多跳QA上接近GPT-4o性能。

**[ReCode: Updating Code API Knowledge with Reinforcement Learning](llm_reasoning/recode_updating_code_api_knowledge_with_reinforcement_learning.md)**

:   提出 ReCode 框架，通过基于规则的强化学习（而非 SFT）训练 LLM 在 prompt 中正确利用 API 更新文档完成代码版本迁移，使 7B 模型在 CodeUpdateArena 上超越 32B 模型。

**[Relation-R1: Progressively Cognitive Chain-of-Thought Guided Reinforcement Learning for Unified Relation Comprehension](llm_reasoning/relation-r1_progressively_cognitive_chain-of-thought_guided_reinforcement_learni.md)**

:   提出 Relation-R1，首个统一二元和 N 元关系理解的框架，通过渐进式认知 CoT 引导的 SFT（模板 CoT → MLLM 生成 CoT）+ GRPO 多奖励优化，在 PSG 数据集上提升 6.84~6.90%，在 SWiG 上也取得 SOTA。

**[RPM-MCTS: Knowledge-Retrieval as Process Reward Model with Monte Carlo Tree Search for Code Generation](llm_reasoning/rpm-mcts_knowledge-retrieval_as_process_reward_model_with_monte_carlo_tree_searc.md)**

:   提出 RPM-MCTS——用知识库检索替代训练的过程奖励模型（PRM）来指导代码生成的 MCTS 搜索。利用同类算法实现的同质性，从知识库中检索正确算法步骤作为评估信号，配合相似度过滤去除冗余扩展节点和沙箱执行定位错误，实现 ~15% token 减少同时超越 SOTA。

**[SAPO: Self-Adaptive Process Optimization Makes Small Reasoners Stronger](llm_reasoning/sapo_self-adaptive_process_optimization_makes_small_reasoners_stronger.md)**

:   受神经科学中Error-Related Negativity启发，提出自适应过程优化方法SAPO，通过首错检测+局部后验估计替代低效的逐步蒙特卡洛rollout，在降低2-3倍计算成本的同时实现推理器-验证器协同优化，使小语言模型（≤2B）在数学和代码推理任务上超越多数自演化方法。

**[SCALE: Selective Resource Allocation for Overcoming Performance Bottlenecks in Mathematical Test-time Scaling](llm_reasoning/scale_selective_resource_allocation_for_overcoming_performance_bottlenecks_in_ma.md)**

:   基于认知科学的双过程理论，提出SCALE框架将数学问题分解为子问题后按难度分配不同计算资源（System 1快速计算 vs System 2深度推理），在AIME25上将Qwen3-32B从57.50%提升至71.25%，同时比InftyThink节省33-53%的token。

**[SERL: Self-Examining Reinforcement Learning on Open-Domain](llm_reasoning/serl_self-examining_reinforcement_learning_on_open-domain.md)**

:   提出SERL自我改进框架，LLM同时作为Actor（生成者）和Judge（评估者），用Copeland成对比较方法从自身判断中推导奖励信号，无需外部奖励模型或人工标注，使Qwen3-8B在AlpacaEval 2.0上从52.37%提升到59.90%（+7.53%），接近Qwen3-32B水平。

**[SPARE: Single-Pass Annotation with Reference-Guided Evaluation for Automatic Process Supervision](llm_reasoning/spare_single-pass_annotation_with_reference-guided_evaluation_for_automatic_proc.md)**

:   提出 SPARE 框架，通过单次结构化生成同时完成解题步骤与参考解的对齐和准确性判断（含显式推理），无需额外训练数据，比 MCTS 方法快 2.3 倍且仅需 16% 训练样本即可实现 OOD 泛化。

**[Stable Voting and the Splitting of Cycles](llm_reasoning/stable_voting_and_the_splitting_of_cycles.md)**

:   研究Simple Stable Voting (SSV)——已在数百次实际选举中使用的递归投票规则——是否总是精化(refine)Split Cycle (SC)方法的猜想，通过数学证明（≤5候选人）和SAT求解（6-7候选人）确定：猜想在≤6候选人时成立，≥7候选人时被反驳，并通过构造性证明推广到任意多候选人。

**[Text-To-Scene With Large Reasoning Models](llm_reasoning/text-to-scene_with_large_reasoning_models.md)**

:   提出Reason-3D，利用大推理模型（LRM）的多步空间推理能力，通过语义投票式物体检索+双阶段布局（自回归放置+碰撞感知优化）实现从文本到3D场景的零样本生成，在人工评价中Elo评分达2248（远超Holodeck的1500和LayoutVLM的1650）。

**[The Curious Case of Analogies: Investigating Analogical Reasoning in Large Language Models](llm_reasoning/the_curious_case_of_analogies_investigating_analogical_reasoning_in_large_langua.md)**

:   通过 Patchscopes、注意力屏蔽和线性探针等机制可解释性工具，系统揭示了 LLM 类比推理的内部机制：模型能在中上层有效编码关系信息，但**应用**关系信息到新实体是比**提取**更大的瓶颈；成功的类比推理与故事间强结构对齐相关联，失败则反映弱化或错位的对齐。

**[ToC: Tree-of-Claims Search with Multi-Agent Language Models](llm_reasoning/toc_tree-of-claims_search_with_multi-agent_language_models.md)**

:   提出 Tree-of-Claims (ToC) 框架，将专利权利要求编辑建模为结构化搜索问题，通过 MCTS 与 EditorAgent/ExaminerAgent 多智能体协作，在新颖性、范围保持和语义一致性之间联合优化，比零/少样本 LLM 基线平均提升约 8% 综合分。

**[Trade-offs in Large Reasoning Models: An Empirical Analysis of Deliberative and Adaptive Reasoning over Foundational Capabilities](llm_reasoning/trade-offs_in_large_reasoning_models_an_empirical_analysis_of_deliberative_and_a.md)**

:   系统评估了LRM（如DeepSeek-R1、QwQ、OpenThinker等）在获取深度推理能力后对基础能力（helpfulness和harmlessness）的负面影响，发现deliberative reasoning显著降低指令遵循和安全性能力，并提出Zero-Thinking、Less-Thinking、Summary-Thinking等自适应推理模式可有效缓解这些缺陷。

---

## 🧩 多模态 VLM { #multimodal_vlm }

**[Aligning the True Semantics: Constrained Decoupling and Distribution Sampling for Cross-Modal Alignment](multimodal_vlm/aligning_the_true_semantics_constrained_decoupling_and_distr.md)**

:   提出 CDDS 算法，通过双路径 UNet 将嵌入解耦为语义和模态分量，并利用分布采样方法间接实现跨模态语义对齐，避免直接调整嵌入导致的分布扭曲，在 Flickr30K 和 MS-COCO 上超越 SOTA 6.6%~14.2%。

**[anyECG-chat: A Generalist ECG-MLLM for Flexible ECG Input and Multi-Task Understanding](multimodal_vlm/anyecg-chat_a_generalist_ecg-mllm_for_flexible_ecg_input_and.md)**

:   构建anyECG数据集（含报告生成、波形定位、多ECG比较三大任务）并提出anyECG-chat模型，通过动态ECG输入机制支持变长/少导联/多ECG输入，采用三阶段课程学习训练，在报告生成的OOD泛化、秒级异常波形定位和多ECG对比分析上全面超越现有ECG-MLLM。

**[AStar: Boosting Multimodal Reasoning with Automated Structured Thinking](multimodal_vlm/astar_boosting_multimodal_reasoning_with_automated_structure.md)**

:   提出AStar，一种training-free的多模态推理范式，通过从500个种子样本中构建高层"thought cards"推理模板库，在推理时自适应检索最优模板引导MLLM结构化推理，7B模型在MathVerse上达53.9%准确率（超越GPT-4o的50.2%），仅需50分钟预处理时间且无需训练。

**[BOFA: Bridge-Layer Orthogonal Low-Rank Fusion for CLIP-Based Class-Incremental Learning](multimodal_vlm/bofa_bridge-layer_orthogonal_low-rank_fusion_for_clip-based_.md)**

:   提出BOFA框架，仅微调CLIP已有的跨模态投影层（bridge-layer），通过正交低秩融合（Orthogonal Low-Rank Fusion）将参数更新约束在与旧任务特征正交的低秩"安全子空间"中，配合跨模态混合原型分类器，在不增加任何额外参数和推理开销的前提下实现了SOTA的无样本存储类增量学习。

**[Branch, or Layer? Zeroth-Order Optimization for Continual Learning of Vision-Language Models](multimodal_vlm/branch_or_layer_zeroth-order_optimization_for_continual_lear.md)**

:   本文系统探索了零阶（ZO）优化在基于PEFT的视觉-语言持续学习（VLCL）中的应用，发现全ZO替换会导致训练不稳定，提出从分支级（branch-wise）到层级（layer-wise）的渐进式ZO-FO混合策略，并基于视觉模态方差更大的理论发现提出MoZO策略（梯度符号归一化+视觉扰动约束），在四个benchmark上达到SOTA。

**[Bridging Modalities via Progressive Re-alignment for Multimodal Test-Time Adaptation (BriMPR)](multimodal_vlm/bridging_modalities_via_progressive_re-alignment_for_multimo.md)**

:   提出 BriMPR 框架，通过"分而治之"策略将多模态测试时自适应(MMTTA)分解为多个单模态特征对齐子问题，先用 prompt tuning 校准各模态全局特征分布实现初始跨模态语义对齐，再通过跨模态掩码嵌入重组和实例级对比学习精细化对齐。

**[Bridging the Copyright Gap: Do Large Vision-Language Models Recognize and Respect Copyrighted Content?](multimodal_vlm/bridging_the_copyright_gap_do_large_vision-language_models_r.md)**

:   首次系统评估 LVLM 在多模态上下文中对版权内容的识别和遵守能力，构建了 50,000 对多模态查询-内容的大规模 benchmark，发现 11/12 个 SOTA LVLM 即使面对明确版权声明也无法有效拒绝侵权请求，并提出 CopyGuard 工具增强框架将侵权拒绝率从 ~3% 提升至 ~62%。

**[Concept-RuleNet: Grounded Multi-Agent Neurosymbolic Reasoning in Vision Language Models](multimodal_vlm/concept-rulenet_grounded_multi-agent_neurosymbolic_reasoning.md)**

:   提出Concept-RuleNet——一个三智能体协作的神经符号推理框架，通过从训练图像中提取视觉概念来条件化符号生成和规则构建，解决了现有方法（如Symbol-LLM）仅依赖标签导致的符号幻觉和不代表性问题，在5个OOD基准上平均提升~5%准确率，幻觉符号减少达50%。

**[Cross-modal Proxy Evolving for OOD Detection with Vision-Language Models](multimodal_vlm/cross-modal_proxy_evolving_for_ood_detection_with_vision-lan.md)**

:   提出 CoEvo，一个 training-free 和 annotation-free 的 test-time 框架，通过双向 sample-conditioned 的文本/视觉 proxy 协同演化机制动态更新正负代理缓存，在 ImageNet-1K 上比最强负标签基线 AUROC 提升 1.33%、FPR95 降低 45.98%（从 18.92% 降至 10.22%），实现 SOTA 的 zero-shot OOD 检测。

**[Cross-Modal Unlearning via Influential Neuron Path Editing in Multimodal Large Language Models](multimodal_vlm/cross-modal_unlearning_via_influential_neuron_path_editing_i.md)**

:   提出 MIP-Editor，通过跨层梯度积分（文本）和 Fisher 积分（视觉）定位多模态大语言模型中编码待遗忘知识的**影响力神经元路径**，再用基于路径的表示误导（RMisU）编辑这些神经元，在 MLLMU-Bench 上实现最高 87.75% 的遗忘率和 54.26% 的通用知识保留提升。

**[CrossCheck-Bench: Diagnosing Compositional Failures in Multimodal Conflict Resolution](multimodal_vlm/crosscheck-bench_diagnosing_compositional_failures_in_multim.md)**

:   构建CrossCheck-Bench——首个专注于多模态矛盾检测与解决的诊断基准，包含15K QA对、3层推理复杂度和7种原子能力，发现13个SOTA VLM从感知匹配到逻辑矛盾检测性能一致下降，CoT/SoM等提示策略收效甚微，仅交错符号推理+视觉grounding的方法才有稳定提升。

**[CrossVid: A Comprehensive Benchmark for Evaluating Cross-Video Reasoning in Multimodal Large Language Models](multimodal_vlm/crossvid_a_comprehensive_benchmark_for_evaluating_cross-vide.md)**

:   提出首个系统评估多模态大语言模型（MLLM）跨视频推理（Cross-Video Reasoning, CVR）能力的综合基准CrossVid，涵盖4个维度10个任务、5,331个视频和9,015个QA对，实验揭示当前最佳模型Gemini-2.5-Pro仅达50.4%准确率，远低于人类89.2%。

**[Difference Vector Equalization for Robust Fine-tuning of Vision-Language Models](multimodal_vlm/difference_vector_equalization_for_robust_fine-tuning_of_vis.md)**

:   提出DiVE方法，通过约束预训练和微调模型嵌入之间的"差异向量"在各样本间保持相等，从而在CLIP微调过程中保持嵌入空间的几何结构，同时在ID、OOD、零样本三个指标上取得全面优于现有方法的结果（零样本平均提升8+点）。

**[EM-KD: Distilling Efficient Multimodal Large Language Model with Unbalanced Vision Tokens](multimodal_vlm/em-kd_distilling_efficient_multimodal_large_language_model_w.md)**

:   提出EM-KD框架，通过Hungarian算法解决teacher-student间视觉token数量不平衡问题，结合视觉语义蒸馏(VSD)和视觉-语言亲和力蒸馏(VLAD)将vanilla teacher的知识迁移到高效student MLLM，在11个benchmark上以144 token/patch达到50.4均分，超越576 token的LLaVA-NeXT(49.4)同时推理速度提升近2倍。

**[Exo2Ego: Exocentric Knowledge Guided MLLM for Egocentric Video Understanding](multimodal_vlm/exo2ego_exocentric_knowledge_guided_mllm_for_egocentric_vide.md)**

:   提出 Exo2Ego 框架，通过学习外中心(第三人称)与自中心(第一人称)域之间的映射关系，将 MLLM 中丰富的外中心知识迁移到自中心视频理解，结合新构建的 110万同步 ego-exo clip-text 对数据集 Ego-ExoClip 和 60万指令微调数据集 EgoIT，在 8 个自中心视频基准上取得了领先的开源模型性能。

**[Filter, Correlate, Compress: Training-Free Token Reduction for MLLM Acceleration](multimodal_vlm/filter_correlate_compress_training-free_token_reduction_for_.md)**

:   提出FiCoCo三阶段框架（Filter-Correlate-Compress），通过集成视觉感知+语义感知冗余度量筛选丢弃token，利用token间相关性自适应回收信息，实现training-free的MLLM加速。在LLaVA-NeXT上达14.7×FLOPs压缩同时保留93.6%性能，在5种MLLM架构上全面超越FastV、SparseVLM等SOTA。

**[Global Compression Commander: Plug-and-Play Inference Acceleration for High-Resolution Large Vision-Language Models](multimodal_vlm/global_compression_commander_plug-and-play_inference_acceler.md)**

:   提出GlobalCom²，一个**即插即用、无需训练**的token压缩框架，专为动态裁剪（dynamic cropping）结构的高分辨率VLM设计：利用全局缩略图（thumbnail）作为"指挥官"引导局部裁剪区域（crop）的差异化压缩，在压缩90%视觉token的同时保持>90%原始性能。

**[Graph-of-Mark: Promote Spatial Reasoning in Multimodal Language Models with Graph-Based Visual Prompting](multimodal_vlm/graph-of-mark_promote_spatial_reasoning_in_multimodal_langua.md)**

:   提出 Graph-of-Mark (GoM)，一种无需训练的像素级视觉提示方法，通过在输入图像上直接叠加深度感知的场景图（包含节点和有向边），显式编码物体间的空间关系，使多模态语言模型在 VQA 和定位任务中的零样本空间推理准确率最高提升 11 个百分点。

**[HeadHunt-VAD: Hunting Robust Anomaly-Sensitive Heads in MLLM for Tuning-Free Video Anomaly Detection](multimodal_vlm/headhunt-vad_hunting_robust_anomaly-sensitive_heads_in_mllm_.md)**

:   提出HeadHunt-VAD，不用MLLM的文本输出，而是直接从冻结MLLM中"猎取"一小批对异常敏感且跨prompt鲁棒的注意力头，配合轻量逻辑回归scorer，在仅用1%数据、零微调的条件下，在UCF-Crime(87.03% AUC)和XD-Violence(82.63% AP)上达到tuning-free方法SOTA。

**[HiMo-CLIP: Modeling Semantic Hierarchy and Monotonicity in Vision-Language Alignment](multimodal_vlm/himo-clip_modeling_semantic_hierarchy_and_monotonicity_in_vi.md)**

:   提出 HiMo-CLIP，通过对文本嵌入做 batch 内 PCA 分解（HiDe）提取多粒度语义成分，配合双分支单调性感知对比损失（MoLo），在不修改编码器的前提下让 CLIP 学会"文本越完整、对齐分数越高"的语义单调性，在长文本检索上显著超越现有方法。

**[InEx: Hallucination Mitigation via Introspection and Cross-Modal Multi-Agent Collaboration](multimodal_vlm/inex_hallucination_mitigation_via_introspection_and_cross-mo.md)**

:   提出 InEx 框架，通过内部自省推理（TVER 驱动的不确定性感知视觉增强）和外部跨模态多智能体协作（文本自反思 + 图像编辑验证 + 视觉自反思）迭代验证和修正 MLLM 输出，在 POPE 上提升 8.9%，在多个幻觉和通用 benchmark 上持续超越 OPERA/VCD/ICD。

**[LLM-CAS: Dynamic Neuron Perturbation for Real-Time Hallucination Correction](multimodal_vlm/llm-cas_dynamic_neuron_perturbation_for_real-time_hallucinat.md)**

:   LLM-CAS 首次将 LLM 实时幻觉纠正建模为层次强化学习（HRL）问题，训练 RL Agent 在推理时动态选择最优的神经元扰动策略（高层选择功能网络类别，低层选择扰动类型和幅度），结合自适应掩码+因果追踪精确定位目标神经元，在 StoryCloze 上提升 10.98%，超越 ITI/CAA/SADI 等静态/动态基线。

**[Multimodal DeepResearcher: Generating Text-Chart Interleaved Reports From Scratch with Agentic Framework](multimodal_vlm/multimodal_deepresearcher_generating_text-chart_interleaved_.md)**

:   提出 Multimodal DeepResearcher，一个四阶段 Agent 框架从零生成图文交替研究报告：通过形式化可视化描述（FDV）让 LLM 学习和生成多样化图表，结合 Actor-Critic 迭代精炼机制（LLM生成D3.js代码→浏览器渲染→多模态LLM评审），在自建 MultimodalReportBench 上达到 82% 整体胜率（Claude 3.7），人类评估 100% 胜率。

**[RMAdapter: Reconstruction-based Multi-Modal Adapter for Vision-Language Models (Oral)](multimodal_vlm/rmadapter_reconstructionbased_multimodal_adapter_for_visionlanguage.md)**

:   提出 RMAdapter，一种双分支适配器架构：在标准 adapter 的适应分支旁增加重建分支（类 AutoEncoder），通过共享下投影层和逐层本地重建损失，在 CLIP 少样本微调中实现任务特定适应与通用知识保持的最佳平衡，在 Base-to-Novel 泛化、跨数据集和领域泛化三个任务上全面超越 SOTA（含 Prompt-based 方法）。

**[SafeR-CLIP: Mitigating NSFW Content in Vision-Language Models While Preserving Pre-Trained Knowledge](multimodal_vlm/safer-clip_mitigating_nsfw_content_in_vision-language_models_while_preserving_pr.md)**

:   提出SafeR-CLIP框架，通过近邻感知重定向（将不安全嵌入重定向到语义最近的安全目标而非固定配对）和相对跨模态重定向损失（仅以不安全表示作为负样本而非随机批内负样本），在保持安全性的同时将零样本分类精度比Safe-CLIP恢复8.0%。

**[TOFA: Training-Free One-Shot Federated Adaptation for Vision-Language Models](multimodal_vlm/tofa_training-free_one-shot_federated_adaptation_for_vision-language_models.md)**

:   提出TOFA框架，在联邦学习场景下通过层次贝叶斯模型学习个性化视觉prototype分布 + 全局对齐的LLM文本增强 + 自适应模态融合，实现无需训练、仅一轮通信的CLIP高效适配，在9个数据集上超越one-shot基线甚至部分多轮训练方法。

**[URaG: Unified Retrieval and Generation in Multimodal LLMs for Efficient Long Document Understanding](multimodal_vlm/urag_unified_retrieval_and_generation_in_multimodal_llms_for.md)**

:   URaG 发现 MLLM 处理长文档时存在类人的"粗到细"推理模式（浅层注意力均匀分散、深层集中于证据页），基于此洞察在第 6 层插入轻量跨模态检索模块（仅占参数 0.05%），选取 Top-5 相关页面丢弃其余内容，实现 SOTA 性能的同时减少 44-56% 计算量。

**[VipAct: Visual-Perception Enhancement via Specialized VLM Agent Collaboration and Tool-use](multimodal_vlm/vipact_visual-perception_enhancement_via_specialized_vlm_age.md)**

:   VipAct 提出了一个多Agent协作框架，通过编排器Agent（任务分析+规划+协调）、专用Agent（描述/比较/视觉提示解读）和视觉专家模型（深度估计/目标检测/分割等）三层协作，显著提升 VLM 在细粒度视觉感知任务上的表现，在 Blink 上从 63.74% (zero-shot GPT-4o) 提升到 73.79%。

**[VP-Bench: A Comprehensive Benchmark for Visual Prompting in Multimodal Large Language Models](multimodal_vlm/vp-bench_a_comprehensive_benchmark_for_visual_prompting_in_m.md)**

:   VP-Bench 提出了首个系统评估 MLLM 视觉提示（Visual Prompt）理解能力的两阶段 Benchmark：Stage 1 用 30K+ 图像覆盖 8 种 VP 形状×355 种属性组合评测 VP 感知能力，Stage 2 评测 VP 对 6 个下游任务的实际效果。在 28 个 MLLM 上的评测揭示了 VP 形状选择对性能的关键影响。

---

## ⚖️ 对齐 / RLHF { #llm_alignment }

**[Align to Structure: Aligning Large Language Models with Structural Information](llm_alignment/align_to_structure_aligning_large_language_models_with_struc.md)**

:   提出 Structural Alignment 方法，通过将语言学篇章结构框架（表层文本结构评分 + 基于RST的篇章motif分类器）融入PPO强化学习训练，并设计基于篇章motif的密集奖励机制，使LLM生成更连贯、更具人类写作风格的长文本，在论文写作和长文档摘要任务上均优于标准RLHF模型。

**[AlignTree: Efficient Defense Against LLM Jailbreak Attacks](llm_alignment/aligntree_efficient_defense_against_llm_jailbreak_attacks.md)**

:   AlignTree 利用 LLM 内部激活特征（线性 refusal direction + 非线性 SVM 信号）训练轻量级随机森林分类器，在几乎不增加计算开销的情况下高效检测越狱攻击，实现了 SOTA 的攻击成功率（ASR）降低效果。

**[AMaPO: Adaptive Margin-attached Preference Optimization for Language Model Alignment](llm_alignment/amapo_adaptive_margin-attached_preference_optimization_for_l.md)**

:   提出AMaPO算法，通过实例级自适应margin（结合Z-normalization和指数缩放）动态调节梯度幅度，解决DPO等离线偏好优化方法中对已正确排序样本过拟合、对错误排序样本欠拟合的核心矛盾，显著提升排序准确率和下游对齐性能。

**[BiasJailbreak: Analyzing Ethical Biases and Jailbreak Vulnerabilities in Large Language Models](llm_alignment/biasjailbreakanalyzing_ethical_biases_and_jailbreak_vulnerabilities_in_large_lan.md)**

:   揭示LLM安全对齐中引入的伦理偏见可被反向利用作为越狱攻击向量——边缘化群体关键词的越狱成功率比优势群体高出20%，并提出基于提示词的轻量防御方法BiasDefense。

**[DeCoRL: Decoupling Reasoning Chains via Parallel Sub-Step Generation and Cascaded Reinforcement for Interpretable and Scalable RLHF](llm_alignment/decorl_decoupling_reasoning_chains_via_parallel_sub-step_gen.md)**

:   DeCoRL 将 CoT 推理从单体顺序处理转变为"交响乐团式"的模块化并行协作——9 个专用子模型（解析/语义/实体/事实核查/风格/质量/计算/验证/整合）并行生成推理子步骤，通过双重奖励归因（本地质量+贡献度）+ 级联 DRPO 优化协调，在 RM-Bench 上达到 80.8%（超越所有基线），同时实现 3.8 倍推理加速和 22.7% 的可解释性提升。

**[Differentiated Directional Intervention: A Framework for Evading LLM Safety Alignment](llm_alignment/differentiated_directional_intervention_a_framework_for_evading_llm_safety_align.md)**

:   将 LLM 安全对齐的内部表征从传统的"单一拒绝方向"解构为功能独立的"危害检测方向"和"拒绝执行方向"，在此基础上提出 DBDI 框架，分别用自适应投影消除和直接引导两种策略精准干预两个方向，在 Llama-2 上实现 97.88% 的越狱成功率。

**[EASE: Practical and Efficient Safety Alignment for Small Language Models](llm_alignment/ease_practical_and_efficient_safety_alignment_for_small_language_models.md)**

:   提出 EASE——面向边缘部署小语言模型（SLM）的安全对齐框架，通过两阶段设计解决"浅层拒绝不够安全 vs 深度推理太贵"的矛盾：第一阶段从大型推理模型蒸馏安全推理能力到 SLM，第二阶段用选择性推理激活（仅对脆弱语义区域的对抗查询启用推理，良性查询直接响应），越狱攻击成功率降低 17%（vs 浅层对齐）同时推理开销降低 90%（vs 全推理）。

**[Enhancing Uncertainty Estimation In Llms With Expectation Of Aggregated Internal](llm_alignment/enhancing_uncertainty_estimation_in_llms_with_expectation_of_aggregated_internal.md)**

:   提出EAGLE方法，通过聚合LLM多个中间层隐藏状态的logits并计算置信度分布的期望值来估计不确定性，无需训练额外参数，在多个数据集和模型上ECE从12.6%降至3.2%，AUROC从59.0%提升至61.6%。

**[EPO: Diverse and Realistic Protein Ensemble Generation via Energy Preference Optimization](llm_alignment/epo_diverse_and_realistic_protein_ensemble_generation_via_energy_preference_opti.md)**

:   提出EPO（Energy Preference Optimization），将反向SDE采样与listwise能量排序偏好优化结合，用能量信号对齐预训练蛋白质生成器与目标Boltzmann分布，在Tetrapeptides/ATLAS/Fast-Folding三个基准9个指标上达到SOTA，完全消除了昂贵的分子动力学（MD）模拟需求。

**[EquaCode: A Multi-Strategy Jailbreak Approach for Large Language Models via Equation Solving and Code Completion](llm_alignment/equacode_a_multi-strategy_jailbreak_approach_for_large_language_models_via_equat.md)**

:   提出EquaCode多策略越狱方法，将恶意查询分解为方程求解（B+C+x=A）和代码补全（补全Solver类的solve()方法）的跨域组合，在GPT系列上平均攻击成功率92.78%，在最新模型（Gemini/DeepSeek/Grok）上接近100%。

**[Exploring the Effects of Alignment on Numerical Bias in Large Language Models](llm_alignment/exploring_the_effects_of_alignment_on_numerical_bias_in_large_language_models.md)**

:   系统揭示了LLM对齐过程（指令调优+偏好调优）是LLM评估器产生数值偏差的根本原因，并验证分数范围调整是最有效的缓解策略。

**[From Classification to Ranking: Enhancing LLM Reasoning for MBTI Personality Detection](llm_alignment/from_classification_to_ranking_enhancing_llm_reasoning_capabilities_for_mbti_per.md)**

:   将MBTI人格检测从传统的四维二分类重构为listwise排序任务，通过SFT冷启动+GRPO强化学习（NDCG+维度相似度双奖励），在Kaggle和PANDORA数据集上以7B模型达到SOTA。

**[Importance-Aware Data Selection for Efficient LLM Instruction Tuning](llm_alignment/importance-aware_data_selection_for_efficient_llm_instruction_tuning.md)**

:   提出MIWV（Model Instruction Weakness Value）指标，通过比较LLM在有/无one-shot ICL示例下的损失差来衡量每条指令数据对模型能力提升的重要性，在Alpaca数据集上仅用1%（520条）数据即全面超越全量52002条的微调效果。

**[Margin-aware Preference Optimization for Aligning Diffusion Models without Reference](llm_alignment/margin-aware_preference_optimization_for_aligning_diffusion_models_without_refer.md)**

:   提出 MaPO（Margin-aware Preference Optimization），一种无需参考模型的偏好对齐方法，通过直接优化 Bradley-Terry 模型下偏好/非偏好输出的似然 margin 来对齐 T2I 扩散模型，在风格适配、安全生成、通用偏好对齐等 5 个领域均超越 DPO 和专用方法。

**[MetaGDPO: Alleviating Catastrophic Forgetting with Metacognitive Knowledge through Group Direct Preference Optimization](llm_alignment/metagdpo_alleviating_catastrophic_forgetting_with_metacognitive_knowledge_throug.md)**

:   提出MetaGDPO方法，从数据侧（基于元认知知识的5K数据构建MetaKL）和训练侧（GDPO——将GRPO的在线采样替换为大模型离线response group的DPO变体）两方面缓解小模型（<8B）在推理能力蒸馏中的灾难性遗忘问题。

**[On the Exponential Convergence for Offline RLHF with Pairwise Comparisons](llm_alignment/on_the_exponential_convergence_for_offline_rlhf_with_pairwise_comparisons.md)**

:   在离线RLHF的成对比较设定下，提出RL-LOW算法实现了simple regret的指数收敛 $\exp(-\Omega(n/H))$，并首次导出实例依赖下界证明该速率在指数意义上是最优的。

**[Probing Preference Representations: A Multi-Dimensional Evaluation and Analysis Method for Reward Models](llm_alignment/probing_preference_representations_a_multi-dimensional_evaluation_and_analysis_m.md)**

:   提出 MRMBench 基准，通过 6 个维度（无害性、有帮助性、正确性、连贯性、复杂性、冗长性）的探针任务评估奖励模型是否有效捕获多维偏好，发现探针性能与 PPO 对齐质量强相关（Pearson $r > 0.8$），并提出推理时探针方法将 AlpacaEval win rate 从 57.3% 提升至 62.5%。

**[Reducing the Scope of Language Models](llm_alignment/reducing_the_scope_of_language_models.md)**

:   系统评估 LLM "范围限制"（scoping）方法——让部署在特定用途的 LLM 只响应域内查询、拒绝所有域外请求。在 3 个模型家族×多种任务上比较 prompting / SFT / DPO / 探针 / Circuit Breakers (CB)，发现 SFT 在高数据多样性下最强、CB 在低多样性下最强、分层组合 (SFT→CB) 保留两者优势——关键发现是范围限制的可行性高度依赖训练数据多样性。

**[Rethinking Direct Preference Optimization in Diffusion Models](llm_alignment/rethinking_direct_preference_optimization_in_diffusion_models.md)**

:   提出两个正交改进增强扩散模型偏好优化：(1) 稳定参考模型更新策略放松冻结参考模型并通过正则化鼓励探索；(2) 时间步感知训练策略缓解跨时间步奖励尺度不平衡。二者可嵌入多种偏好优化算法，在人类偏好评估基准上提升SOTA。

**[SafeNlidb: A Privacy-Preserving Safety Alignment Framework for LLM-based Natural Language Database Interfaces](llm_alignment/safenlidb_a_privacy-preserving_safety_alignment_framework_for_llm-based_natural_.md)**

:   提出SafeNlidb框架，通过安全感知数据合成管线和交替偏好优化策略，实现LLM驱动的自然语言数据库接口（NLIDB）在安全推理与SQL生成之间的联合优化，有效防御隐式推理攻击下的隐私泄露。

**[SceneJailEval: A Scenario-Adaptive Multi-Dimensional Framework for Jailbreak Evaluation](llm_alignment/scenejaileval_a_scenario-adaptive_multi-dimensional_framework_for_jailbreak_eval.md)**

:   提出SceneJailEval，一个场景自适应的多维度越狱评估框架，定义14个越狱场景和10个评估维度，通过场景分类→维度动态选择→多维检测→加权危害评分的流程，在自建数据集上F1达0.917（超SOTA 6%），在JBB上达0.995（超SOTA 3%），同时支持危害程度量化而非仅二分类。

**[SOM Directions are Better than One: Multi-Directional Refusal Suppression in Language Models](llm_alignment/som_directions_are_better_than_one_multi-directional_refusal_suppression_in_lang.md)**

:   证明LLM的拒绝行为并非由单一方向编码，而是形成低维流形，利用自组织映射（SOM）提取多个拒绝方向并通过贝叶斯优化搜索最优消融组合，在多个模型上超越单方向基线和专用越狱算法。

**[Canoe: Teaching LLMs to Maintain Contextual Faithfulness via Synthetic Tasks and RL](llm_alignment/teaching_large_language_models_to_maintain_contextual_faithfulness_via_synthetic.md)**

:   提出 Canoe 框架，通过从 Wikidata 三元组合成四类可验证的短形式 QA 数据，配合 Dual-GRPO（含准确率奖励、长形式代理奖励和格式奖励）同时优化短/长形式生成的忠实度，使 Llama-3-8B 在 11 个下游任务上平均提升 22.6%，超越 GPT-4o。

**[TEMPLE: Incentivizing Temporal Understanding of Video LLMs via Progressive Pre-SFT Alignment](llm_alignment/temple_incentivizing_temporal_understanding_of_video_large_language_models_via_p.md)**

:   提出 TEMPLE，通过自动化的视频时间偏好数据生成管线（视频筛选→时间扰动→对比响应）和创新的 Progressive Pre-SFT Alignment 策略（课程学习 + DPO 先于 SFT），用少量自生成 DPO 数据显著提升 Video LLM 的时间推理能力，在 VideoMME、MLVU、Vinoground 等多个基准上一致改进。

**[Towards Inference-Time Scaling for Continuous Space Reasoning](llm_alignment/towards_inference-time_scaling_for_continuous_space_reasoning.md)**

:   首次系统研究离散文本推理中的inference-time scaling技术能否迁移到连续潜空间推理模型（COCONUT），发现dropout采样能生成多样推理路径（Pass@32达44.43%），但PRM/ORM仅带来不足2.3%提升，根因在于连续思维表示缺乏区分正误推理的几何归纳偏置。

**[When Human Preferences Flip: An Instance-Dependent Robust Loss for RLHF](llm_alignment/when_human_preferences_flip_an_instance-dependent_robust_loss_for_rlhf.md)**

:   针对人类偏好标注中普遍存在的"偏好翻转"问题，提出 FA-DPO（Flipping-Aware DPO），将标注过程建模为"真实意图 + 实例依赖翻转概率"两阶段，通过修正 BT 模型损失和迭代优化翻转估计模块，在多种噪声场景下显著提升对齐鲁棒性，实例依赖翻转率高时比 DPO 提升 16.7%。

---

## 🎨 图像生成 { #image_generation }

**[AbductiveMLLM: Boosting Visual Abductive Reasoning Within MLLMs](image_generation/abductivemllm_boosting_visual_abductive_reasoning_within_mll.md)**

:   模仿人类的"语言溯因+图像想象"双模式认知，提出AbductiveMLLM，通过Reasoner(因果感知假设生成+筛选)和Imaginer(扩散模型引导的图像想象)两个组件端到端联合训练，在VAR和YouCookII两个benchmark上显著超越传统方法和通用MLLM，设置新的SOTA。

**[AEDR: Training-Free AI-Generated Image Attribution via Autoencoder Double-Reconstruction](image_generation/aedr_training-free_ai-generated_image_attribution_via_autoen.md)**

:   提出一种基于自编码器双重重建损失比值的免训练图像归因方法，通过图像均匀度校准消除纹理复杂度偏差，在8个主流扩散模型上平均准确率达95.1%，比最强基线高24.7%，且速度快约100倍。

**[Aggregating Diverse Cue Experts for AI-Generated Image Detection](image_generation/aggregating_diverse_cue_experts_for_ai-generated_image_detec.md)**

:   提出Multi-Cue Aggregation Network (MCAN)，通过混合编码器适配器(MoEA)将原始图像、高频信息和新提出的色度不一致性(CI)三种互补线索统一融合，实现跨生成模型的鲁棒AI生成图像检测。

**[Annealed Relaxation of Speculative Decoding for Faster Autoregressive Image Generation](image_generation/annealed_relaxation_of_speculative_decoding_for_faster_autor.md)**

:   提出Cool-SD，一种有理论支撑的退火松弛speculative decoding框架：通过推导TV距离上界得到最优重采样分布，并证明接受概率递减调度比均匀调度产生更小的分布偏移，在LlamaGen和Lumina-mGPT上实现了比LANTERN++更优的速度-质量权衡。

**[AnoStyler: Text-Driven Localized Anomaly Generation via Lightweight Style Transfer](image_generation/anostyler_text-driven_localized_anomaly_generation_via_light.md)**

:   将零样本异常生成建模为文本引导的局部风格迁移问题，通过轻量级U-Net + CLIP损失将正常图像的掩码区域风格化为语义对齐的异常图像，在MVTec-AD和VisA上以263M参数（仅0.61M可训练）超越扩散模型基线，同时显著提升下游异常检测性能。

**[CAD-VAE: Leveraging Correlation-Aware Latents for Comprehensive Fair Disentanglement](image_generation/cad-vae_leveraging_correlation-aware_latents_for_comprehensive_fair_disentanglem.md)**

:   提出CAD-VAE，引入"相关隐变量" $z_R$ 显式建模目标属性和敏感属性之间的共享信息，通过最小化条件互信息 $I(z_Y;z_S|z_R)$ 实现公平解缠绕，无需领域知识即可产生公平表示和高质量反事实样本。

**[Conditional Diffusion Model for Multi-Agent Dynamic Task Decomposition](image_generation/conditional_diffusion_model_for_multi-agent_dynamic_task_dec.md)**

:   提出 CD3T，一个两层层次化 MARL 框架：用条件扩散模型学习动作语义表示（以观测和他人动作为条件，预测下一观测和奖励），通过 k-means 聚类得到子任务划分，高层选择子任务、低层在受限动作空间执行策略，在 SMAC 的 Super Hard 场景上显著超越所有基线。

**[DICE: Distilling Classifier-Free Guidance into Text Embeddings](image_generation/dice_distilling_classifier-free_guidance_into_text_embedding.md)**

:   提出 DICE，训练一个仅 2M 参数的轻量 sharpener 将 CFG 的引导效果蒸馏进 text embedding，使无引导采样达到与 CFG 同等的生成质量、推理计算量减半，在 SD1.5 多个变体、SDXL 和 PixArt-α 上全面验证有效，是 AAAI 2026 口头报告论文。

**[DiffBench Meets DiffAgent: End-to-End LLM-Driven Diffusion Acceleration Code Generation](image_generation/diffbench_meets_diffagent_end-to-end_llm-driven_diffusion_ac.md)**

:   提出DiffBench（604个扩散模型加速任务的评估基准，分5个难度等级）和DiffAgent（集成规划-编码-调试三Agent + 遗传算法选择器的闭环框架），在Claude Sonnet 4上将扩散加速代码生成通过率从54.30%提升到81.59%，复杂优化任务达成率68.27%。

**[Difficulty Controlled Diffusion Model for Synthesizing Effective Training Data](image_generation/difficulty_controlled_diffusion_model_for_synthesizing_effec.md)**

:   在Stable Diffusion中引入难度编码器（MLP，输入类别+难度分数），通过LoRA微调解耦"域对齐"和"难度控制"两个目标，使生成数据的学习难度可控——仅用10%额外合成数据即超过Real-Fake的最佳结果，节省63.4 GPU小时。

**[DOS: Directional Object Separation in Text Embeddings for Multi-Object Image Generation](image_generation/dos_directional_object_separation_in_text_embeddings_for_mul.md)**

:   识别出多物体生成失败的四种场景（相似形状/纹理、不同背景偏好、多物体），通过构建方向性分离向量修改CLIP的三类文本嵌入（语义token/EOT/pooled），在SDXL上将成功率提升16-25%并将融合率降低3-12%，推理速度接近baseline（约4×快于Attend-and-Excite）。

**[HACK: Head-Aware KV Cache Compression for Efficient Visual Autoregressive Modeling](image_generation/head-aware_kv_cache_compression_for_efficient_visual_autoreg.md)**

:   发现VAR模型中attention head天然分为Contextual Heads（语义一致性，垂直注意力模式）和Structural Heads（空间连贯性，多对角线模式），提出HACK框架通过非对称预算分配和模式特定压缩策略，在70%压缩率下实现无损生成质量，Infinity-8B上1.75×显存减少和1.57×加速。

**[Infinite-Story: A Training-Free Consistent Text-to-Image Generation](image_generation/infinite-story_a_training-free_consistent_text-to-image_gene.md)**

:   基于 scale-wise 自回归模型（Infinity），通过三个 training-free 技术——Identity Prompt Replacement（消除文本编码器的上下文偏差）、Adaptive Style Injection（参考图像特征注入）和 Synchronized Guidance Adaptation（同步 CFG 两个分支），实现了身份与风格一致的多图像生成，速度比扩散模型快 6 倍（1.72 秒/张）。

**[Laytrol: Preserving Pretrained Knowledge in Layout Control for Multimodal Diffusion Transformers](image_generation/laytrol_preserving_pretrained_knowledge_in_layout_control_fo.md)**

:   通过从 MM-DiT 复制参数初始化布局控制网络、设计专用初始化方案（布局编码器初始化为纯文本编码器 + 输出零初始化）、并用 FLUX 自己生成的图像构建 LaySyn 数据集来缓解分布偏移，实现了在 FLUX 上高质量的布局到图像生成。

**[T2I-RiskyPrompt: A Benchmark for Safety Evaluation, Attack, and Defense on Text-to-Image Model](image_generation/t2i-riskyprompt_a_benchmark_for_safety_evaluation_attack_and_defense_on_text-to-.md)**

:   构建T2I-RiskyPrompt——一个包含6,432条有效风险prompt的综合基准，涵盖6大类14细分风险类别，每条prompt带有层次化标注和详细风险原因，并提出reason-driven的MLLM风险检测方法（3B模型达91.8%准确率），系统评估了8个T2I模型、9种防御方法、5种安全过滤器和5种攻击策略。

---

## 🚗 自动驾驶 { #autonomous_driving }

**[A Data-Driven Model Predictive Control Framework for Multi-Aircraft TMA Routing Under Travel Time Uncertainty](autonomous_driving/a_data-driven_model_predictive_control_framework_for_multi-aircraft_tma_routing_.md)**

:   提出面向终端管制区（TMA）多机冲突解脱和着陆调度的闭环 MPC 框架——集成 XGBoost 到达时间预测、MILP 优化模型（路径选择+速度调整+等待约束）和交通仿真器，在樟宜机场 50 海里 STAR 网络上实现实时无冲突调度，高峰期计算时间比一次性优化降低 7 倍，Monte Carlo 仿真验证鲁棒性。

**[Beta Distribution Learning for Reliable Roadway Crash Risk Assessment](autonomous_driving/beta_distribution_learning_for_reliable_roadway_crash_risk_a.md)**

:   提出基于 Beta 分布学习的地理空间深度学习框架，利用多尺度卫星图像预测道路致命事故风险的完整概率分布（而非点估计），在 Recall 上提升 17-23%，并通过分布形状自然表达不确定性。

**[Bridging Day and Night: Target-Class Hallucination Suppression in Unpaired Image Translation](autonomous_driving/bridging_day_and_night_target-class_hallucination_suppressio.md)**

:   首次系统性解决无配对日→夜图像翻译中的"目标类幻觉"问题，通过双头判别器（风格头+SAM2伪标签分割头）检测幻觉 + 类原型对比学习抑制幻觉，在BDD100K日夜域适应检测上将mAP从15.08提升到17.40（+15.5%），交通灯AP提升31.7%。

**[CompTrack: 信息瓶颈引导的低秩动态Token压缩用于点云跟踪 (Oral)](autonomous_driving/comptrack_information_bottleneckguided_lowrank_dynamic_token_compres.md)**

:   针对LiDAR点云3D单目标跟踪中的"双重冗余"问题（空间冗余：大量背景噪声；信息冗余：前景中大量不具区分性的平面点），提出SFP前景预测器+IB-DTC信息瓶颈引导动态Token压缩两个模块，在KITTI/nuScenes/Waymo上达到SOTA，90 FPS实时运行（比P2P快1.4倍）。

**[FastDriveVLA: Efficient End-to-End Driving via Plug-and-Play Reconstruction-based Token Pruning](autonomous_driving/fastdrivevla_efficient_end-to-end_driving_via_plug-and-play_.md)**

:   提出 FastDriveVLA，通过 MAE 风格的前景像素重建训练轻量级 plug-and-play 的 ReconPruner 模块（仅 0.07B），利用对抗前景-背景重建策略优先保留驾驶决策所需的前景 token，在 nuScenes 开环规划基准上各剪枝率均达 SOTA，一次训练可迁移至同一视觉编码器的不同 VLA 模型。

**[FQ-PETR: Fully Quantized Position Embedding Transformation for Multi-View 3D Object Detection](autonomous_driving/fq-petr_fully_quantized_position_embedding_transformation_fo.md)**

:   首次实现PETR系列3D检测器的全INT8量化部署，通过量化友好的LiDAR-ray位置编码(QFPE)解决多模态特征幅度不匹配问题、双查找表(DULUT)高效逼近非线性算子、数值稳定后量化(QANS)避免softmax注意力失真，在PETR/StreamPETR/PETRv2/MV2D上W8A8精度损失<1%且延迟降低75%（3.9×加速）。

**[LiDARCrafter: Dynamic 4D World Modeling from LiDAR Sequences](autonomous_driving/lidarcrafter_dynamic_4d_world_modeling_from_lidar_sequences.md)**

:   提出LiDARCrafter，首个专用于LiDAR的4D生成世界模型，通过Text2Layout（LLM解析文本→场景图→三分支扩散生成4D布局）→Layout2Scene（Range-image扩散生成高保真单帧）→Scene2Seq（自回归warp+扩散生成时序一致的序列）三阶段流程，在nuScenes上取得SOTA。

**[MambaSeg: Harnessing Mamba for Accurate and Efficient Image-Event Semantic Segmentation](autonomous_driving/mambaseg_harnessing_mamba_for_accurate_and_efficient_image-e.md)**

:   提出 MambaSeg，用双分支并行 Mamba 编码器分别处理 RGB 图像和事件流，通过空间-时间双维度交互模块 (DDIM) 实现细粒度跨模态融合，在 DDD17 和 DSEC 数据集上以 25.44M 参数取得 77.56%/75.10% mIoU 的 SOTA，效率远优于 Transformer 方案。

**[SPARC: 用单一策略驾驶100辆未见车辆的OOD泛化](autonomous_driving/out-of-distribution_generalization_with_a_sparc_racing_100_u.md)**

:   提出 SPARC（Single-Phase Adaptation for Robust Control），将 RMA 的两阶段上下文编码与历史适应统一为单阶段训练，在 Gran Turismo 7 高保真赛车模拟器中用单一策略驾驶100+未见车辆实现SOTA OOD泛化性能。

**[PriorDrive: 用统一向量先验增强在线HD地图构建](autonomous_driving/priordrive_enhancing_online_hd_mapping_with_unified_vector_p.md)**

:   提出 PriorDrive 框架，通过 Unified Vector Encoder (UVE) 和 Hybrid Prior Representation (HPQuery) 将多种向量化先验地图（SD地图、旧HD地图、历史预测地图）统一编码并集成到各种在线建图模型中，在 nuScenes 上 mAP 提升 14.3，兼容 query-based 和 non-query-based 两类建图架构。

**[ReflexDiffusion: 反思增强的高侧向加速度自动驾驶轨迹规划](autonomous_driving/reflexdiffusion_reflection-enhanced_trajectory_planning_for_.md)**

:   提出 ReflexDiffusion，在扩散模型推理阶段引入物理感知的反思机制，通过梯度注入强化曲率-速度-加速度耦合约束（a_y = κv²），在 nuPlan 高侧向加速度长尾场景中驾驶分数提升 14.1%，架构无关可直接部署到现有扩散规划器。

**[Task Prototype-Based Knowledge Retrieval for Multi-Task Learning from Partially Annotated Data](autonomous_driving/task_prototype-based_knowledge_retrieval_for_multi-task_lear.md)**

:   提出基于任务原型的知识检索框架，通过可学习 Task Prototype 嵌入任务特性并量化任务关联、Knowledge Retrieval Transformer 基于 task-affinity score 自适应精炼特征表示，在部分标注多任务学习（MTPSL）中避免依赖未标注任务的预测，PASCAL-Context 和 NYUD-v2 上全面超越 SOTA。

**[VILTA: A VLM-in-the-Loop Adversary for Enhancing Driving Policy Robustness](autonomous_driving/vilta_a_vlm-in-the-loop_adversary_for_enhancing_driving_poli.md)**

:   VILTA 将 VLM（Gemini-2.5-Flash）直接嵌入自动驾驶 RL 训练循环中，通过"Vision-Language-Editing"（VLE）范式让 VLM 编辑周围车辆的未来轨迹来生成具有挑战性的危险场景，训练出的驾驶策略在 CARLA 挑战场景中路线完成率提升 13.3%、碰撞率降低 28.5%。

**[Vision-Only Gaussian Splatting for Collaborative Semantic Occupancy Prediction (Oral)](autonomous_driving/visiononly_gaussian_splatting_for_collaborative_semantic_occupancy_p.md)**

:   首次将 3D 高斯 Splatting 作为多智能体协同感知的通信媒介和中间表征，利用高斯基元的刚体变换可解析性和稀疏性，通过高斯打包（ROI 裁剪+刚体变换）和跨智能体邻域融合模块，实现了高效且可解释的视觉协同语义占用预测。

---

## ⚡ LLM 效率 { #llm_efficiency }

**[A Content-Preserving Secure Linguistic Steganography](llm_efficiency/a_content-preserving_secure_linguistic_steganography.md)**

:   提出首个内容保持型语言隐写术范式CLstega，通过微调掩码语言模型（MLM）来可控地变换预测分布，将秘密信息嵌入到不做任何修改的原始文本中，实现了100%提取成功率和近乎完美的安全性（隐写分析检测准确率接近随机猜测的0.5）。

**[Attention Retention for Continual Learning with Vision Transformers](llm_efficiency/attention_retention_for_continual_learning_with_vision_transformers.md)**

:   提出ARCL-ViT框架，通过注意力掩码生成和梯度掩码两步策略防止ViT在持续学习中的注意力漂移，在ImageNet-R和CIFAR-100上取得SOTA结果，证明保持注意力模式是解决灾难性遗忘的关键。

**[C3TG: Conflict-aware, Composite, and Collaborative Controlled Text Generation](llm_efficiency/c3tg_conflict-aware_composite_and_collaborative_controlled_text_generation.md)**

:   提出 C3TG 框架，通过两阶段方法实现多维度细粒度可控文本生成：生成阶段用加权 KL 散度融合属性分布调整 token 概率，优化阶段用能量函数（分类器分数 + 冲突惩罚项）结合 Feedback Agent 迭代重写，在 17 个属性子类上达到 90.4% 属性准确率且大幅降低毒性。

**[Collaborative LLM Numerical Reasoning with Local Data Protection](llm_efficiency/collaborative_llm_numerical_reasoning_with_local_data_protection.md)**

:   提出一种大小模型协作框架，通过对本地查询进行"主题迁移+数值替换"的两阶段匿名化来保护敏感数据，同时让远端 GPT-4 以可执行 Python 代码（即插即用工具）形式返回推理方案，本地仅需做数值回代即可获得答案，在 FinQA 和 MultiHiertt 上准确率提升 16-44% 且数据泄露降低 2-45%。

**[Do Retrieval Augmented Language Models Know When They Don't Know?](llm_efficiency/do_retrieval_augmented_language_models_know_when_they_dont_know.md)**

:   系统分析RAG模型的拒绝校准问题，发现RALM在检索文档全部不相关时过度拒绝率超过55%（即使模型内部知识足够回答），提出结合不确定性估计和拒绝感知微调的机制来平衡拒绝与回答质量。

**[ELSPR: Evaluator LLM Training Data Self-Purification on Non-Transitive Preferences](llm_efficiency/elspr_evaluator_llm_training_data_self-purification_on_non-transitive_preference.md)**

:   ELSPR 将 LLM 评估器的成对偏好建模为锦标赛图，通过强连通分量 (SCC) 识别非传递偏好，提出归一化有向图结构熵指标，并基于图重构过滤有问题的训练数据——过滤后的评估器非传递性降低 13.8%、结构熵降低 0.088，且丢弃数据的人类一致性仅 34.4%（vs 保留数据 52.6%）。

**[Factor(U,T): Controlling Untrusted AI by Monitoring their Plans](llm_efficiency/factorut_controlling_untrusted_ai_by_monitoring_their_plans.md)**

:   研究不可信 AI 做任务分解、可信 AI 做执行的 Factor(U,T) 协议安全性，发现监控分解计划的 AUROC 仅 0.52（接近随机），而监控具体代码实现可达 0.96——恶意意图在抽象计划中难以检测但在具体实现中暴露，结论是"结构性预防（可信分解器）优于事后监控"。

**[Harnessing the Unseen: The Hidden Influence of Intrinsic Knowledge in Long-Context Language Models](llm_efficiency/harnessing_the_unseen_the_hidden_influence_of_intrinsic_knowledge_in_long-contex.md)**

:   首次系统研究长上下文语言模型中参数知识(parametric knowledge)对生成的影响，发现其影响随上下文长度增长而增强，且现有方法提升外部检索能力会抑制参数召回能力，据此提出Hybrid Needle-in-a-Haystack测试来同时评估两种能力。

**[InterMoE: Individual-Specific 3D Human Interaction Generation via Dynamic Temporal-Selective MoE](llm_efficiency/intermoe_individual-specific_3d_human_interaction_generation_via_dynamic_tempora.md)**

:   提出 InterMoE，通过 Dynamic Temporal-Selective MoE 架构解决文本驱动的双人 3D 交互运动生成中的个体特征保持和语义忠实度问题：Synergistic Router 融合语义和运动学特征引导路由，Dynamic Temporal Selection 让专家动态选择关键时间帧，在 InterHuman 上 FID 降低 9%、InterX 上降低 22%。

**[Judge Q: Trainable Queries for Optimized Information Retention in KV Cache Eviction](llm_efficiency/judge_q_trainable_queries_for_optimized_information_retention_in_kv_cache_evicti.md)**

:   提出Judge Q，在模型词表中引入可训练的soft token，训练其注意力模式对齐实际解码token的注意力模式，使其在prefill阶段能替代局部窗口查询来评估KV cache重要性，从而更好地保留全局信息，在LongBench上提升~1分，RULER上提升3+分。

**[Learning from the Undesirable: Robust Adaptation of Language Models without Forgetting](llm_efficiency/learning_from_the_undesirable_robust_adaptation_of_language_models_without_forge.md)**

:   提出 Learning-from-the-Undesirable (LfU)，一种面向 SFT 的正则化方法，通过对辅助模型施加梯度上升模拟"不良行为"，再通过表示级一致性损失约束原模型与不良模型的内部表征保持一致，有效缓解有限数据微调中的过拟合、遗忘和对抗脆弱性问题。

**[MicroEvoEval: A Systematic Evaluation Framework for Image-Based Microstructure Evolution Prediction](llm_efficiency/microevoeval_a_systematic_evaluation_framework_for_image-based_microstructure_ev.md)**

:   提出 MicroEvoEval，首个面向图像级微观结构演化预测的标准化基准：涵盖 4 个代表性物理任务（平面波、晶粒生长、旋节分解、枝晶凝固）、14 个模型（5 个领域特定 + 9 个通用时空架构）、多维度评估（数值精度 + 物理保真度 + 计算效率），发现现代通用架构（如 VMamba）在长期稳定性和物理保真度上优于领域特定模型，且计算效率高一个数量级。

**[Model Editing as a Double-Edged Sword: Steering Agent Ethical Behavior](llm_efficiency/model_editing_as_a_double-edged_sword_steering_agent_ethical_behavior_toward_ben.md)**

:   将 Agent 伦理行为引导建模为模型编辑任务（Behavior Editing），提出基于心理学道德理论的三层 BehaviorBench 基准，在 9 个开源模型和 20 个闭源模型上验证了模型编辑可以精确地将 Agent 引导向善意或恶意方向，且单次编辑可导致全局道德对齐偏移。

**[Think How Your Teammates Think: Active Inference Can Benefit Decentralized Execution](llm_efficiency/think_how_your_teammates_think_active_inference_can_benefit_decentralized_execut.md)**

:   提出 AIM（Active Inference Modeling）框架，在去中心化多智能体强化学习中，不依赖通信机制，仅基于局部观测建模队友的主动推理过程（感知-信念-动作三重肖像），并通过准确性-相关性双重过滤机制选择性融合队友信念，在 SMAC、SMACv2、MPE 和 GRF 四大基准上取得最优或接近最优表现。

---

## 📈 时间序列 { #time_series }

**[A Unified Shape-Aware Foundation Model for Time Series Classification](time_series/a_unified_shape-aware_foundation_model_for_time_series_class.md)**

:   提出 UniShape——一个面向时间序列分类的基础模型，通过 shape-aware adapter 自适应聚合多尺度判别性子序列（shapelet），并结合原型对比预训练在实例和 shape 两个层面学习可迁移的 shapelet 表示，在 128 个 UCR 数据集上以 3.1M 参数达到 SOTA（平均准确率 87.08%），同时提供良好的分类可解释性。

**[Beyond Observations Reconstruction Error-Guided Irregularly Sampled Time Series ](time_series/beyond_observations_reconstruction_error-guided_irregularly_sampled_time_series_.md)**

:   提出 iTimER，利用模型自身的重建误差分布作为学习信号——从观测点估计误差分布后采样生成未观测时刻的伪观测值，通过 Wasserstein 距离对齐观测/伪观测区域的误差分布 + 对比学习，在不规则采样时序的分类、插值、预测任务上全面超越 SOTA。

**[C3Rl Rethinking The Combination Of Channel-Independence And Channel-Mixing From ](time_series/c3rl_rethinking_the_combination_of_channel-independence_and_channel-mixing_from_.md)**

:   提出 C3RL，基于 SimSiam 对比学习框架将通道独立（CI）和通道混合（CM）策略视为同一数据的两个转置视图构建正样本对，通过孪生网络联合表示学习和预测学习，将 CI 模型的最佳性能率从 43.6% 提升到 81.4%，CM 模型从 23.8% 提升到 76.3%。

**[Coherent Multi-Agent Trajectory Forecasting in Team Sports with CausalTraj](time_series/coherent_multi-agent_trajectory_forecasting_in_team_sports_with_causaltraj.md)**

:   提出 CausalTraj，一种时间因果的似然模型，用自回归逐步预测多智能体位移的高斯混合分布，强调联合预测指标（minJADE/minJFDE）而非独立评估每个个体，在 NBA、Basketball-U 和 Football-U 数据集上实现最佳联合指标。

**[Cometnet Contextual Motif-Guided Long-Term Time Series Forecasting](time_series/cometnet_contextual_motif-guided_long-term_time_series_forecasting.md)**

:   提出 CometNet，通过从完整历史序列中提取循环出现的"上下文 motif"构建 motif 库，再用 motif 引导的 MoE 架构动态关联当前窗口与相关motif进行预测，突破了有限回看窗口的感受野瓶颈，在8个数据集上显著超越 TimeMixer++、iTransformer 等 SOTA。

**[Counterfactual Explainable Ai Xai Method For Deep Learning-Based Multivariate Ti](time_series/counterfactual_explainable_ai_xai_method_for_deep_learning-based_multivariate_ti.md)**

:   提出 CONFETTI，一种面向多变量时序分类的多目标反事实解释方法，通过 CAM 引导的子序列提取 + NUN 值替换 + NSGA-III 多目标优化，同时优化预测置信度、近似性和稀疏性，在 7 个 UEA 数据集上置信度提升 ≥10%、稀疏性改善 ≥40%。

**[Deepboots Dual-Stream Residual Boosting For Drift-Resilient Time-Series Forecast](time_series/deepboots_dual-stream_residual_boosting_for_drift-resilient_time-series_forecast.md)**

:   提出 DeepBooTS，通过偏差-方差分解理论证明加权集成可降低方差从而缓解概念漂移，设计双流残差递减 boosting 架构，每个 block 的输出修正前一个 block 的残差，在多个数据集上平均提升 15.8%。

**[DEF: Detecting the Future — All-at-Once Event Sequence Forecasting with Horizon Matching](time_series/detecting_the_future_all-at-once_event_sequence_forecasting_with_horizon_matchin.md)**

:   提出 DEF (Detection-based Event Forecasting)，类比目标检测中的 DETR 思想，用多个并行预测头同时生成 K 个未来事件候选，通过匈牙利匹配损失将预测与真实事件最优对齐，解决自回归方法在长程预测中输出趋于重复/恒定的问题，在 5 个数据集上长程预测提升达 50%。

**[GBOC: Finding Time Series Anomalies using Granular-ball Vector Data Description](time_series/finding_time_series_anomalies_using_granular-ball_vector_data_description.md)**

:   提出 GBOC (Granular-Ball One-Class Network)，在潜空间中用密度引导的层次分裂构建粒球向量数据描述（GVDD），每个粒球代表局部正常行为原型，通过剪枝低质量粒球 + 最近粒球中心对齐训练 + 基于粒球距离的推理实现鲁棒的时序异常检测。

**[FreqCycle: A Multi-Scale Time-Frequency Analysis Method for Time Series Forecasting](time_series/freqcycle_a_multi-scale_time-frequency_analysis_method_for_time_series_forecasti.md)**

:   提出 FreqCycle，结合时域周期显式建模（FECF，学习共享日/周周期基底 + 自适应低通滤波）和频域中高频增强（SFPL，分段 STFT + 可学习加权融合），并扩展为 MFreqCycle 处理耦合多周期性，在 7 个数据集上达到 SOTA 精度且推理更快。

**[HN-MVTS: HyperNetwork-based Multivariate Time Series Forecasting](time_series/hn-mvts_hypernetwork-based_multivariate_time_series_forecasting.md)**

:   提出 HN-MVTS，用超网络从可学习的通道嵌入向量生成预测模型最后一层的权重，自适应地在通道独立（CI）和通道依赖（CD）之间插值，即插即用地提升 DLinear/PatchTST/TSMixer 等多种 SOTA 模型的性能，且不增加推理时间。

**[IdealTSF: Can Non-Ideal Data Contribute to Enhancing Time Series Forecasting?](time_series/idealtsf_can_non-ideal_data_contribute_to_enhancing_the_performance_of_time_seri.md)**

:   提出 IdealTSF 框架，通过三阶段渐进式设计——负样本预训练（用稳定分布+多尺度噪声+结构删除模拟非理想数据）、正样本训练（混合平滑插值修复数据）、ECOS 优化器（对抗扰动引导到平坦极值）——使基础 attention 模型在含噪声/缺失的时序数据上获得约 10% 的性能提升。

**[SELDON: Supernova Explosions Learned by Deep ODE Networks](time_series/seldon_supernova_explosions_learned_by_deep_ode_networks.md)**

:   提出SELDON，一种结合masked GRU-ODE编码器、隐式Neural ODE传播器和可解释高斯基函数解码器的连续时间VAE，用于稀疏、不规则采样的天文光变曲线预测，在仅观测20%数据时即可超越基线方法做出准确的多波段通量预测。

**[Urban Incident Prediction with Graph Neural Networks: Integrating Government Ratings and Crowdsourced Reports](time_series/urban_incident_prediction_with_graph_neural_networks_integrating_government_rati.md)**

:   提出 URBAN（多视图多输出GNN模型），联合利用稀疏但无偏的政府检查评级数据和密集但有偏的众包报告数据来预测城市事件的真实潜在状态，在纽约市960万+报告和100万+检查数据上验证，预测相关性比仅用报告数据高5.3倍。

---

## 🤖 机器人/具身智能 { #robotics }

**[A Computable Game-Theoretic Framework for Multi-Agent Theory of Mind](robotics/a_computable_game-theoretic_framework_for_multi-agent_theory_of_mind.md)**

:   提出基于 Poisson 认知层次（cognitive hierarchy）的博弈论框架，通过 Gamma-Poisson 共轭贝叶斯更新实现可计算的多智能体 Theory of Mind，在避免 POMDP 不可判定性的同时支持递归式有限理性决策与在线信念修正。

**[Adaptive Theory of Mind for LLM-based Multi-Agent Coordination](robotics/adaptive_theory_of_mind_for_llm-based_multi-agent_coordination.md)**

:   提出自适应心智理论智能体(A-ToM)，将ToM阶数对齐建模为在线专家建议问题，通过FTL或Hedge算法实时估计伙伴的ToM阶数并动态调整自身推理深度，在重复矩阵博弈、网格导航和Overcooked等4类任务上实现鲁棒的零样本多智能体协作。

**[Affordance-Guided Coarse-to-Fine Exploration for Base Placement in Open-Vocabulary Mobile Manipulation](robotics/affordance-guided_coarse-to-fine_exploration_for_base_placem.md)**

:   针对开放词汇移动操控中机器人基座选位问题，提出一种零样本框架，通过构建跨模态表征（Affordance RGB + Obstacle Map+）将语义affordance线索投射到障碍物地图上，再用粗到细迭代优化平衡语义和几何约束，在5个操控任务上达到85%成功率，大幅超越几何规划器和纯VLM方法。

**[Attention as Binding: A Vector-Symbolic Perspective on Transformer Reasoning](robotics/attention_as_binding_a_vector-symbolic_perspective_on_transformer_reasoning.md)**

:   本文提出将Transformer自注意力机制重新解释为向量符号架构(VSA)中的软绑定/解绑定算子——Query/Key定义角色空间、Value编码填充项、注意力权重实现可微解绑定、残差连接实现叠加——从而以代数视角统一解释LLM在符号推理中的能力与脆弱性，并提出显式绑定头、超维记忆层等VSA启发的架构改进方向。

**[Causal Inference Under Threshold Manipulation: Bayesian Mixture Modeling and Heterogeneous Treatment Effects](robotics/causal_inference_under_threshold_manipulation_bayesian_mixtu.md)**

:   提出 BMTM/HBMTM 贝叶斯混合模型框架，在消费者策略性操纵消费额以达到奖励阈值的场景下，通过将观测分布拆解为 bunching 与 non-bunching 两个子分布，准确估计阈值因果效应及跨子群的异质性处理效应。

**[EvoEmpirBench: Dynamic Spatial Reasoning with Agent-ExpVer](robotics/evoempirbench_dynamic_spatial_reasoning_with_agent-expver.md)**

:   提出 EvoEmpirBench（EEB），包含两个动态交互式 benchmark（局部可观测迷宫导航 + 消消乐），以及 Agent-ExpVer 三智能体在线学习框架（GeoLink 交互 + InsightForce 经验抽象 + TruthWeaver 知识管理），通过"经验→验证→真理归纳"的认知循环实现无参数更新的持续策略进化，使 GPT-4.1 成功率提升 5.6%、Qwen-32B 提升 29%。

**[iSeal: Encrypted Fingerprinting for Reliable LLM Ownership Verification](robotics/iseal_encrypted_fingerprinting_for_reliable_llm_ownership_verification.md)**

:   提出 iSeal——首个在模型窃取者完全控制推理过程的黑盒场景下仍能可靠验证 LLM 所有权的主动指纹方法，通过外部加密编码器 + RSC 纠错 + 相似度匹配三重机制，在 12 个 LLM、10+ 种攻击下均保持 100% 指纹成功率（FSR），而已有方法降至 0%。

**[Neural Graph Navigation for Intelligent Subgraph Matching](robotics/neural_graph_navigation_for_intelligent_subgraph_matching.md)**

:   提出 NeuGN（Neural Graph Navigation）框架，首次将生成式神经导航集成到子图匹配的核心枚举阶段，通过 QSExtractor 提取查询图结构信号 + GGNavigator 将暴力枚举转为结构感知的候选节点优先排序，在保证完备性的同时将 First Match Steps 最高减少 98.2%。

**[Robust Out-of-Order Retrieval for Grid-Based Storage at Maximum Capacity](robotics/robust_out-of-order_retrieval_for_grid-based_storage_at_maximum_capacity.md)**

:   针对满载 2D 网格存储系统中检索顺序不确定的问题，提出 k-bounded perturbation 不确定性模型，证明 Θ(k) 列宽是零重定位的充要条件，并给出高效鲁棒存储求解器与贪心检索策略，当 k ≤ 0.5c 时几乎消除重定位，k 到达 c 时仍减少 50%+ 重定位。

**[Shadows in the Code: Exploring the Risks and Defenses of LLM-based Multi-Agent Software Development Systems](robotics/shadows_in_the_code_exploring_the_risks_and_defenses_of_llm-.md)**

:   首次系统分析 LLM 多 Agent 软件开发系统（ChatDev/MetaGPT/AgentVerse）的安全风险：提出 IMBIA 攻击框架覆盖两种威胁场景（恶意用户+良性Agent / 良性用户+恶意Agent）和 12 种恶意行为（5 大恶意软件家族），攻击成功率高达 93%（ChatDev），并设计 Adv-IMBIA 对抗性防御将 ASR 降低 40-73%。

**[Towards Reinforcement Learning from Neural Feedback: Mapping fNIRS Signals to Agent Performance](robotics/towards_reinforcement_learning_from_neural_feedback_mapping_.md)**

:   提出 NEURO-LOOP 框架，利用 fNIRS（功能性近红外光谱）脑信号作为隐式神经反馈评估 RL agent 表现，发布 25 名被试 × 3 领域 × 6 条件的 fNIRS 数据集，分类 F1 达 67%（二分类）/ 46%（多分类），跨被试 fine-tuning 分别提升 17% 和 41%，奠定 Reinforcement Learning from Neural Feedback (RLNF) 基础。

**[Unintended Misalignment from Agentic Fine-Tuning: Risks and Mitigation](robotics/unintended_misalignment_from_agentic_fine-tuning_risks_and_m.md)**

:   本文揭示了在良性 Agent 数据上微调 LLM 会导致意外的安全对齐偏移（攻击成功率增加 32-38%），并提出 PING（Prefix Injection Guard）——通过迭代生成+评估自然语言前缀来引导微调后的 Agent 拒绝有害请求，平均提升拒绝率 66%（Web）和 44%（代码），同时保持任务性能（仅降 1.8%）。

---

## 🎮 强化学习 { #reinforcement_learning }

**[A Course Correction in Steerability Evaluation: Revealing Miscalibration and Side Effects in LLMs](reinforcement_learning/a_course_correction_in_steerability_evaluation_revealing_mis.md)**

:   本文提出了一个基于多维目标空间的 LLM 可操控性（steerability）评估框架，将 steering error 分解为校准偏差（miscalibration）和副作用（side effects/orthogonality），在文本改写任务上发现即使是最强的 LLM 也会产生严重副作用，prompt engineering 无效、best-of-N 采样代价高、RL 微调有改善但仍未彻底解决。

**[A Learning Framework For Cooperative Collision Avoidance of UAV Swarms Leveraging Domain Knowledge](reinforcement_learning/a_learning_framework_for_cooperative_collision_avoidance_of_.md)**

:   提出 reMARL 框架，将图像处理中的主动轮廓模型（active contour）作为领域知识引入多智能体强化学习的奖励设计，使无人机集群仅通过最大化个体奖励即可学会协作避撞，在大规模集群（≤10架）中性能显著优于 COMA/VDN/QMIX/MAPPO 等 SOTA MARL 方法，反应时间比元启发式方法快 98.75%，能耗降低 85.37%。

**[A Learning Framework For Cooperative Collision Avoidance of UAV Swarms Leveraging Domain Knowledge](reinforcement_learning/a_learning_framework_for_cooperative_collision_avoidance_of_uav_swarms_leveragin.md)**

:   提出 reMARL 框架，利用图像处理领域知识（active contour model）设计多智能体强化学习奖励函数，实现无人机集群的协作避碰，相比传统元启发式方法反应时间缩短 98.75%、能耗降低 85.37%。

**[A Multi-Agent Conversational Bandit Approach to Online Evaluation and Selection of User-Aligned LLM Responses](reinforcement_learning/a_multi-agent_conversational_bandit_approach_to_online_evaluation_and_selection_.md)**

:   提出 MACO 多智能体会话式 Bandit 框架，通过本地 agent 的在线淘汰和云服务器的自适应偏好查询机制，实现 LLM 响应的在线评估与用户偏好对齐，达到 $\tilde{O}(\sqrt{dMT})$ 的近优 regret 界。

**[Aligning Machiavellian Agents: Behavior Steering via Test-Time Policy Shaping](reinforcement_learning/aligning_machiavellian_agents_behavior_steering_via_test-tim.md)**

:   提出一种测试时策略塑形方法，通过轻量级伦理属性分类器在推理阶段插值修改预训练 RL 智能体的动作概率分布，无需重训练即可实现对多种伦理属性的细粒度行为引导。

**[BAMAS: Structuring Budget-Aware Multi-Agent Systems](reinforcement_learning/bamas_structuring_budget-aware_multi-agent_systems.md)**

:   提出 BAMAS 框架，通过整数线性规划（ILP）在预算约束下选择最优 LLM 组合，再用强化学习策略选择最佳协作拓扑（线性/星型/反馈/规划驱动），在 GSM8K/MBPP/MATH 上达到与 SOTA 多 Agent 系统相当的准确率，同时成本降低最高 86%。

**[Does Self-Evaluation Enable Wireheading in Language Models?](reinforcement_learning/does_self-evaluation_enable_wireheading_in_language_models.md)**

:   形式化证明在POMDP框架下自评分耦合奖励信号时wireheading（操纵评估而非优化任务）严格优于诚实行为（Lemma 1），并在Llama-3.1-8B和Mistral-7B上实验验证——Selfgrade条件下摘要任务奖励饱和至0.95但准确率仅0.05，解耦自评分与奖励可消除直接wireheading激励。

**[MARS: Multi-Agent Adaptive Reasoning with Socratic Guidance for Automated Prompt Optimization](reinforcement_learning/mars_multi-agent_adaptive_reasoning_with_socratic_guidance_f.md)**

:   提出 MARS 五智能体框架做自动提示优化（APO）：Planner 生成任务特定的优化轨迹，Teacher-Critic-Student 三体进行苏格拉底对话式迭代精炼 prompt（模拟文本空间中的伪梯度下降），Target 执行并反馈，整体建模为 POMDP，在 17 个数据集上平均超越前 SOTA（PE2）6.04%（通用任务）和 6.42%（领域任务），且仅需 1-shot 训练数据。

**[MMhops-R1: Multimodal Multi-hop Reasoning](reinforcement_learning/mmhops-r1_multimodal_multi-hop_reasoning.md)**

:   提出了 MMhops 基准（31K 样本、3-4 跳推理深度）和 MMhops-R1 框架，通过强化学习训练 MLLM 自主规划推理路径、动态调用图像/文本检索器，实现多模态多跳推理，7B 模型超越 72B 基线和现有 mRAG 方法。

**[One-Step Generative Policies with Q-Learning: A Reformulation of MeanFlow](reinforcement_learning/one-step_generative_policies_with_q-learning_a_reformulation_of_meanflow.md)**

:   将MeanFlow重新形式化为残差映射 $g(a_t,b,t) = a_t - u(a_t,b,t)$，实现一步噪声→动作的生成式策略，无需蒸馏或多步ODE积分，可直接与Q-learning联合训练，在OGBench和D4RL的73个任务上取得强性能。

**[Test-driven Reinforcement Learning in Continuous Control](reinforcement_learning/test-driven_reinforcement_learning_in_continuous_control.md)**

:   提出 Test-driven Reinforcement Learning (TdRL) 框架，用多个测试函数（pass-fail 测试定义最优目标 + indicative 测试引导学习）替代单一奖励函数表示任务目标，通过字典序启发式轨迹比较学习回报函数，在 DeepMind Control Suite 上匹配或超越手工奖励方法，天然支持多目标优化。

---

## 🛡️ AI 安全 { #ai_safety }

**[Alternative Fairness and Accuracy Optimization in Criminal Justice](ai_safety/alternative_fairness_and_accuracy_optimization_in_criminal_j.md)**

:   本文系统综述了算法公平性的三大维度（群体公平、个体公平、过程公平），提出了一种基于容差约束的改进群体公平性优化公式，并构建了面向公共决策系统的"公平三支柱"部署框架。

**[An Improved Privacy and Utility Analysis of Differentially Private SGD with Bounded Domain and Smooth Losses](ai_safety/an_improved_privacy_and_utility_analysis_of_differentially_p.md)**

:   在仅假设损失函数L-光滑（不需要凸性）的条件下，为DPSGD推导出了更紧的闭式RDP隐私界，并首次在有界域场景下给出了完整的收敛性/效用分析，揭示了较小的参数域直径可以同时改善隐私和效用。

**[An Information Theoretic Evaluation Metric for Strong Unlearning](ai_safety/an_information_theoretic_evaluation_metric_for_strong_unlear.md)**

:   提出 Information Difference Index (IDI)，一种基于信息论的白盒评估指标，通过度量中间层特征与遗忘标签之间的互信息来衡量机器遗忘的彻底程度，揭示了现有黑盒指标（MIA、JSD等）无法捕捉的中间层残留信息问题，并提出 COLA 方法在特征层面消除残余信息。

**[An Information Theoretic Evaluation Metric for Strong Unlearning](ai_safety/an_information_theoretic_evaluation_metric_for_strong_unlearning.md)**

:   揭示现有黑盒遗忘评估指标（MIA/JSD等）的根本缺陷——仅修改最后一层即可满足所有黑盒指标但中间层完整保留遗忘数据信息，提出IDI白盒指标通过InfoNCE估计各层与遗忘标签的互信息差异来量化遗忘效果，并提出COLA方法在CIFAR-10/100和ImageNet-1K上实现接近Retrain的IDI得分。

**[An LLM-Based Simulation Framework for Embodied Conversational Agents in Psychological Counseling](ai_safety/an_llm-based_simulation_framework_for_embodied_conversationa.md)**

:   提出 ECAs 框架，基于认知行为治疗(CBT)等心理学理论，利用 LLM 将真实咨询案例扩展为具身认知记忆空间，模拟心理咨询中来访者的完整认知过程，生成高保真度的咨询对话数据，在专家评估和自动评估中均显著优于基线。

**[Angular Gradient Sign Method: Uncovering Vulnerabilities in Hyperbolic Networks](ai_safety/angular_gradient_sign_method_uncovering_vulnerabilities_in_h.md)**

:   提出Angular Gradient Sign Method (AGSM)，将双曲空间中的梯度分解为径向（层次深度）和角度（语义）分量，仅沿角度方向施加扰动来生成对抗样本，在图像分类和跨模态检索任务上比标准FGSM/PGD多降低5-13%的准确率。

**[Argumentative Debates for Transparent Bias Detection (ABIDE)](ai_safety/argumentative_debates_for_transparent_bias_detection_technic.md)**

:   提出ABIDE框架，将偏见检测过程结构化为基于量化二极论辩框架（QBAF）的辩论：通过邻域级局部统计公平性（neighbourhood-based local statistical parity）生成偏见论据，利用批判性问题（critical questions）作为攻击机制挑战不可靠论据，在合成/真实/LLM模型上均优于IRB基线。

**[AUVIC: Adversarial Unlearning of Visual Concepts for Multi-modal Large Language Models](ai_safety/auvic_adversarial_unlearning_of_visual_concepts_for_multi-mo.md)**

:   提出AUVIC框架，通过对抗性扰动生成器 + 动态锚点保留机制，在MLLM中精确遗忘目标视觉概念（如特定人脸），同时避免对语义相似概念的附带遗忘，并构建了首个面向群体场景视觉概念遗忘的评测基准VCUBench。

**[DeepTracer: Tracing Stolen Model via Deep Coupled Watermarks](ai_safety/deeptracer_tracing_stolen_model_via_deep_coupled_watermarks.md)**

:   提出DeepTracer鲁棒水印框架，通过自适应源类选择（K-Means聚类覆盖特征空间）+ 同类耦合损失（拉近水印样本与目标类在输出空间的距离）+ 两阶段关键样本过滤，使水印任务与主任务深度耦合，在6种模型窃取攻击（含hard-label和data-free）下水印成功率平均达77-100%，远超现有方法。

**[Truth, Justice, and Secrecy: Cake Cutting Under Privacy Constraints](ai_safety/truth_justice_and_secrecy_cake_cutting_under_privacy_constraints.md)**

:   首个隐私保护蛋糕切割协议，在保持无嫉妒性和策略防谋性的同时，通过秘密共享和安全多方计算（MPC）技术确保参与者的估值函数不被泄露。

---

## 📦 模型压缩 { #model_compression }

**[A Closer Look at Knowledge Distillation in Spiking Neural Network Training](model_compression/a_closer_look_at_knowledge_distillation_in_spiking_neural_ne.md)**

:   针对ANN→SNN知识蒸馏中教师ANN连续特征/logits与学生SNN离散稀疏spike特征/logits之间分布差异被忽视的问题，提出基于显著性缩放激活图蒸馏（SAMD）和噪声平滑logits蒸馏（NLD）的CKDSNN框架，在CIFAR-10/100、ImageNet-1K和CIFAR10-DVS上均取得SNN训练的新SOTA。

**[AdaFuse: Accelerating Dynamic Adapter Inference via Token-Level Pre-Gating and Fused Kernel Optimization](model_compression/adafuse_accelerating_dynamic_adapter_inference_via_token-lev.md)**

:   针对动态MoE-LoRA适配器推理延迟暴增（250%-950%）的问题，提出了一种token级预门控架构，只在第一层做一次全局路由决策，配合自研的SGMM融合CUDA内核将所有激活的LoRA适配器一次性合并进骨干网络，在保持精度的同时将解码延迟降低2.4倍。

**[AgentODRL: A Large Language Model-based Multi-agent System for ODRL Generation](model_compression/agentodrl_a_large_language_model-based_multi-agent_system_fo.md)**

:   提出AgentODRL，一个基于Orchestrator-Workers架构的LLM多智能体系统，通过任务分解、语法验证循环和LoRA驱动的语义反思机制，将自然语言数据权限规则高质量地转换为ODRL格式。

**[ALTER: Asymmetric LoRA for Token-Entropy-Guided Unlearning of LLMs](model_compression/alter_asymmetric_lora_for_token-entropy-guided_unlearning_of.md)**

:   提出ALTER框架，利用非对称LoRA架构结合Token级别的Tsallis熵引导，实现LLM中目标知识的精准遗忘，同时通过参数隔离机制保留模型基础能力，在TOFU、WMDP和MUSE三个基准上达到SOTA。

**[CAMERA: Multi-Matrix Joint Compression for MoE Models via Micro-Expert Redundancy Analysis](model_compression/camera_multi-matrix_joint_compression_for_moe_models_via_mic.md)**

:   提出"micro-expert"概念将MoE层的输出分解为跨矩阵（up/gate/down_proj）的微专家线性组合，基于能量排序进行结构化剪枝(Camera-P)和混合精度量化(Camera-Q)，在Deepseek-MoE-16B/Qwen2-57B/Qwen3-30B上20%-60%剪枝率全面超越NAEE和D²-MoE，且分析Qwen2-57B仅需单卡A100不到5分钟。

**[Distilling Cross-Modal Knowledge via Feature Disentanglement](model_compression/distilling_cross-modal_knowledge_via_feature_disentanglement.md)**

:   提出频域解耦跨模态知识蒸馏（FD-CMKD），通过傅里叶变换将特征分解为低频（模态共享语义）和高频（模态特有细节）分量，分别施加强一致性 MSE 和弱一致性 logMSE 损失，并引入尺度标准化与共享分类器对齐特征空间，在音频-视觉、图像-文本、语义分割等多个跨模态场景全面超越现有蒸馏方法。

**[DynaQuant: Dynamic Mixed-Precision Quantization for Learned Image Compression](model_compression/dynaquant_dynamic_mixed-precision_quantization_for_learned_i.md)**

:   针对学习图像压缩（LIC）模型部署效率低的痛点，提出DynaQuant框架，在参数层面通过可学习scale/zero-point + Distance-Aware Gradient Modulator实现内容自适应量化，在架构层面通过轻量Bit-Width Selector动态为每层分配最优比特宽度，在Cheng2020/ELIC/Ballé三个基线上实现接近FP32的R-D性能，同时获得最高5.17×加速和模型大小降至原来的~1/4。

**[KVmix: Gradient-Based Layer Importance-Aware Mixed-Precision Quantization for KV Cache](model_compression/kvmix_gradient-based_layer_importance-aware_mixed-precision_.md)**

:   提出 KVmix，通过计算 Key/Value 投影权重梯度的 $L_2$ 范数来评估各层 KV Cache 的重要性，实现层级混合精度量化（Key 平均 2.19bit、Value 平均 2.38bit），并结合动态关键上下文选择（RPC）策略，在 Llama/Mistral 等模型上实现近无损推理、4.9× 内存压缩和 5.3× 吞吐加速。

**[SCoPe: Intrinsic Semantic Space Control for Mitigating Copyright Infringement in LLMs](model_compression/scope_intrinsic_semantic_space_control_for_mitigating_copyright_infringement_in_.md)**

:   将LLM版权侵权缓解问题重新定义为内在语义空间控制，利用稀疏自编码器(SAE)将隐状态映射到高维稀疏空间，识别版权敏感子空间并在解码时钳制其激活，无需外部过滤器或参数更新即可有效减少版权内容复制，同时保持模型通用能力。

**[SparK: Query-Aware Unstructured Sparsity with Recoverable KV Cache Channel Pruning](model_compression/spark_query-aware_unstructured_sparsity_with_recoverable_kv_cache_channel_prunin.md)**

:   提出SparK——一种training-free的KV cache通道级非结构化剪枝方法，通过query-aware的saliency评估选择关键通道+recovery机制恢复被剪枝通道的贡献，在80%剪枝率下性能损失<5%，与token eviction方法正交互补，可额外减少30%+ KV cache存储。

---

## 🧊 3D 视觉 { #3d_vision }

**[3D-ANC: Adaptive Neural Collapse for Robust 3D Point Cloud Recognition](3d_vision/3d-anc_adaptive_neural_collapse_for_robust_3d_point_cloud_re.md)**

:   将Neural Collapse(NC)机制引入3D点云对抗鲁棒性，用固定的ETF分类头+自适应训练框架(RBL+FDL)构建解耦的特征空间，在ModelNet40上将DGCNN的对抗准确率从27.2%提升到80.9%，超出最佳baseline 34个点。

**[3D-Free Meets 3D Priors: Novel View Synthesis from a Single Image with Pretrained Diffusion Guidance](3d_vision/3d-free_meets_3d_priors_novel_view_synthesis_from_a_single_image_with_pretrained.md)**

:   提出将 3D-free 方法（HawkI 风格的 test-time optimization）与 3D-based 先验（Zero123++ 的弱引导图）结合的框架，无需额外 3D 数据或训练即可从单张图片生成指定仰角/方位角的相机控制视图，在复杂场景下 LPIPS、CLIP-Score 等指标全面超越 Zero123++、HawkI 和 Stable Zero123。

**[4DSTR: Advancing Generative 4D Gaussians with Spatial-Temporal Rectification for High-Quality and Consistent 4D Generation](3d_vision/4dstr_advancing_generative_4d_gaussians_with_spatial-tempora.md)**

:   提出4DSTR框架，通过基于Mamba的时序关联校正（修正高斯点的尺度和旋转）以及逐帧自适应稠密化与裁剪策略，显著提升4D高斯生成的时空一致性和对快速时序变化的适应能力。

**[Adapt-As-You-Walk Through the Clouds: Training-Free Online Test-Time Adaptation of 3D Vision-Language Foundation Models](3d_vision/adapt-as-you-walk_through_the_clouds_training-free_online_te.md)**

:   提出 Uni-Adapter，一种面向3D视觉-语言基础模型(VLFM)的无训练在线测试时适应框架，通过基于聚类的动态原型缓存和图正则化标签平滑来应对分布偏移，在多个3D损坏基准上取得SOTA。

**[AnchorDS: Anchoring Dynamic Sources for Semantically Consistent Text-to-3D Generation](3d_vision/anchords_anchoring_dynamic_sources_for_semantically_consiste.md)**

:   揭示 SDS 中源分布是动态演化而非静态的关键问题，提出 AnchorDS，通过将当前渲染图像作为图像条件输入双条件扩散模型来锚定源分布，解决了 SDS 的语义过度平滑和多视角不一致问题，在 T3Bench 上全面超越 SDS/VSD/SDS-Bridge。

**[AnchorHOI: Zero-shot Generation of 4D Human-Object Interaction via Anchor-based Prior Distillation](3d_vision/anchorhoi_zero-shot_generation_of_4d_human-object_interactio.md)**

:   提出 AnchorHOI，通过锚点NeRF和锚点关键点两种中间桥梁，分别从图像/视频扩散模型中蒸馏交互先验和运动先验，实现零样本的文本驱动4D人物-物体交互生成，在静态3D和动态4D HOI生成上均超越已有方法。

**[Arbitrary-Scale 3D Gaussian Super-Resolution](3d_vision/arbitrary-scale_3d_gaussian_super-resolution.md)**

:   提出一个集成框架实现3D高斯溅射(3DGS)的任意倍率超分辨率渲染，通过尺度感知渲染、生成先验引导优化和渐进超分机制，用单个3D模型支持整数和非整数倍率的HR渲染，PSNR提升6.59dB同时保持85 FPS实时速度。

**[FoundationSLAM: 释放深度基础模型在端到端稠密视觉SLAM中的潜力](3d_vision/foundationslam_unleashing_the_power_of_depth_foundation_models_for.md)**

:   将深度基础模型的几何先验注入光流式SLAM系统，通过混合光流网络、双向一致BA层和可靠性感知精炼三个模块形成闭环，在TUM/EuRoC/7Scenes/ETH3D四大数据集取得SOTA轨迹精度和稠密重建质量，18 FPS实时运行。

**[Gaussian Blending: Rethinking Alpha Blending in 3D Gaussian Splatting](3d_vision/gaussian_blending_rethinking_alpha_blending_in_3d_gaussian_splatting.md)**

:   重新审视3DGS中的标量alpha blending，指出其忽略像素内空间变化是多尺度渲染伪影（放大erosion/缩小dilation）的根源，提出Gaussian Blending——将alpha和transmittance建模为像素内的空间分布（2D uniform window），实现实时抗锯齿且无需重训练，在多尺度Blender上PSNR从31.59→35.80。

---

## 🏥 医学图像 { #medical_imaging }

**[A Disease-Aware Dual-Stage Framework for Chest X-ray Report Generation](medical_imaging/a_disease-aware_dual-stage_framework_for_chest_x-ray_report_.md)**

:   提出一种两阶段疾病感知框架，通过学习14个与病理类别对应的疾病感知语义token（DASTs）实现显式的疾病表征，再利用疾病-视觉注意力融合（DVAF）和双模态相似性检索（DMSR）机制辅助LLM生成临床准确的胸部X光报告，在CheXpert Plus、IU X-Ray和MIMIC-CXR三个数据集上取得SOTA。

**[A Principle-Driven Adaptive Policy for Group Cognitive Stimulation Dialogue for Elderly with Cognitive Impairment](medical_imaging/a_principle-driven_adaptive_policy_for_group_cognitive_stimu.md)**

:   针对老年认知障碍患者的群体认知刺激治疗（CST）场景，提出GCSD系统：通过多说话人上下文控制、动态参与者状态建模（soft prompt）、认知刺激注意力损失和多维奖励策略优化四个模块，基于Qwen-2.5-3B微调，在500+小时真实粤语CST对话和1万+模拟对话上训练，BLEU-4达27.93超越GPT-4o等大模型，A/B测试胜率50% vs GPT-4o的39%。

**[Advancing Safe Mechanical Ventilation Using Offline RL With Hybrid Actions and Clinically Aligned Rewards](medical_imaging/advancing_safe_mechanical_ventilation_using_offline_rl_with_.md)**

:   针对ICU机械通气（MV）设置优化问题，提出混合动作空间的离线RL方法（HybridIQL/HybridEDAC），避免传统离散化导致的分布偏移，同时引入基于无通气天数（VFD）和生理参数安全范围的临床对齐奖励函数，通过多目标优化选择最优奖励，将可优化的通气参数从2-3个扩展到6个，HybridIQL在性能和策略覆盖率间取得最佳平衡。

**[Ambiguity-aware Truncated Flow Matching for Ambiguous Medical Image Segmentation](medical_imaging/ambiguity-aware_truncated_flow_matching_for_ambiguous_medica.md)**

:   提出 ATFM 框架，通过数据层级推理范式将预测精度和多样性解耦到分布级和样本级分别优化，结合高斯截断表示（GTR）和分割流匹配（SFM）两个模块，在模糊医学图像分割任务中同时提升预测的精度、保真度和多样性。

**[Apo2Mol: 3D Molecule Generation via Dynamic Pocket-Aware Diffusion Models](medical_imaging/apo2mol_3d_molecule_generation_via_dynamic_pocket-aware_diff.md)**

:   提出Apo2Mol，一个基于扩散的全原子框架，从蛋白质apo（未结合）构象出发，同时生成3D配体分子和对应的holo（结合态）口袋构象，使用24K实验解析的apo-holo结构对训练，在结合亲和力（Vina min -7.86）和药物类似性上达到SOTA。

**[EgoEMS: A High-Fidelity Multimodal Egocentric Dataset for Cognitive Assistance in Emergency Medical Services](medical_imaging/egoems_a_high-fidelity_multimodal_egocentric_dataset_for_cognitive_assistance_in.md)**

:   发布首个高保真多人多模态自我中心EMS数据集，包含233个试验20小时视频、9项干预67个关键步骤标注，提供三个基准任务（步骤分类/在线分割/CPR质量估计）推动EMS认知协助系统开发。

**[FourierPET: Deep Fourier-based Unrolled Network for Low-count PET Reconstruction](medical_imaging/fourierpet_deep_fourier-based_unrolled_network_for_low-count_pet_reconstruction.md)**

:   发现低剂量 PET 的三类退化在频域可分离——泊松噪声/光子不足导致高频相位扰动，衰减校正误差抑制低频幅度——据此提出 FourierPET：基于 ADMM 展开的频率感知重建框架，仅 0.44M 参数在三个数据集上全面 SOTA。

**[Learning Cell-Aware Hierarchical Multi-Modal Representations for Robust Molecular Modeling](medical_imaging/learning_cell-aware_hierarchical_multi-modal_representations.md)**

:   提出CHMR框架，将分子结构(1D/2D/3D)与细胞形态/基因表达等生物模态联合建模，通过结构感知的模态增强解决>90%的外部生物模态缺失问题，用树状向量量化(Tree-VQ)捕获分子-细胞-基因的层次化依赖关系，在9个benchmark的728个任务上超越SOTA，分类平均AUC提升3.6%，回归MAE降低17.2%。

**[ProtSAE: Disentangling and Interpreting Protein Language Models via Semantically-Guided Sparse Autoencoders](medical_imaging/protsae_disentangling_and_interpreting_protein_language_models_via_semantically-.md)**

:   提出 ProtSAE，在稀疏自编码器训练中引入语义标注和领域本体知识作为引导信号，解决传统 SAE 的语义纠缠问题，使蛋白质语言模型的隐层特征与生物学概念（分子功能、生物过程、离子结合位点等）精准对齐，同时保持高重建保真度并支持概念级别的生成控制。

---

## 🎯 目标检测 { #object_detection }

**[A Theoretical Analysis of Detecting Large Model-Generated Time Series](object_detection/a_theoretical_analysis_of_detecting_large_model-generated_time_series.md)**

:   首次研究时间序列大模型（TSLM）生成内容的检测问题——提出收缩假说（Contraction Hypothesis）：TSLM 生成的时间序列在递归预测下不确定性逐渐降低（分布越来越集中），而真实序列不会。基于此提出白盒检测器 UCE（Uncertainty Contraction Estimator），在 32 个数据集上超越 SOTA 基线。

**[Actor-Critic for Continuous Action Chunks: A Reinforcement Learning Framework for Long-Horizon Robotic Manipulation with Sparse Reward](object_detection/actor-critic_for_continuous_action_chunks_a_reinforcement_le.md)**

:   AC3 提出了一个直接学习连续动作序列（action chunk）的 actor-critic 框架，通过"仅从成功轨迹更新 actor"的非对称更新规则和基于自监督锚点的内在奖励来稳定稀疏奖励下的长时域机器人操作学习，在 BiGym 和 RLBench 的 25 个任务上取得优于现有方法的成功率。

**[AerialMind: Towards Referring Multi-Object Tracking in UAV Scenarios](object_detection/aerialmind_towards_referring_multi-object_tracking_in_uav_sc.md)**

:   构建了首个面向无人机场景的大规模 Referring Multi-Object Tracking（RMOT）基准数据集 AerialMind，并提出 HawkEyeTrack（HETrack）方法，通过视觉-语言共进化融合编码器和尺度自适应上下文精炼模块，在无人机航拍场景中实现语言引导的多目标跟踪。

**[Beyond Boundaries: Leveraging Vision Foundation Models for Source-Free Object Detection](object_detection/beyond_boundaries_leveraging_vision_foundation_models_for_so.md)**

:   提出利用VFM（DINOv2+Grounding DINO）增强无源域自适应目标检测（SFOD）的框架，通过全局特征对齐(PGFA)、实例级原型对比学习(PIFA)和双源伪标签融合(DEPF)三个模块，在6个跨域检测基准上取得SOTA，例如Cityscapes→Foggy Cityscapes达47.1% mAP（比DRU高3.5%），Sim10k→Cityscapes达67.4% AP（比DRU高8.7%）。

**[Connecting the Dots: Training-Free Visual Grounding via Agentic Reasoning](object_detection/connecting_the_dots_training-free_visual_grounding_via_agent.md)**

:   提出 GroundingAgent，一个完全不需要任务特定微调的视觉定位框架，通过组合预训练的开放词汇检测器（YOLO World）、MLLM（Llama-3.2-11B-Vision）和 LLM（DeepSeek-V3）进行结构化迭代推理，在 RefCOCO/+/g 上实现 65.1% 的零样本平均准确率，大幅超越之前的 zero-shot 方法。

**[Continuous Vision-Language-Action Co-Learning with Semantic-Physical Alignment for Behavioral Cloning](object_detection/continuous_vision-language-action_co-learning_with_semantic-.md)**

:   提出CCoL框架，通过NeuralODE驱动的多模态连续协同学习（MCC）和双向交叉注意力的语义-物理对齐（CSA），在Behavioral Cloning中同时解决动作序列的物理不连续性和语义-物理失配问题，在三个仿真平台上平均相对提升8.0%，双臂插入任务最高达19.2%。

**[Sketch-HARP: 分层自回归草图生成实现灵活笔画级绘制操控](object_detection/generating_sketches_in_a_hierarchical_auto-regressive_proces.md)**

:   提出 Sketch-HARP 分层自回归草图生成框架，通过三阶段层次化过程（预测笔画嵌入→确定画布位置→生成绘制动作序列），首次实现草图绘制过程中的灵活笔画级操控，在替换/擦除/扩展等任务上显著优于 SketchEdit。

**[SAGA: Learning Signal-Aligned Distributions for Improved Text-to-Image Generation](object_detection/saga_learning_signal-aligned_distributions_for_improved_text-to-image_generation.md)**

:   提出SAGA方法，通过学习与提示词对齐的高斯分布来改进文本到图像生成模型的语义对齐，无需重新训练且支持文本和空间双条件生成，在SD 1.4和SD 3上大幅提升对齐性能（TIAM-3从8.4%提升到50.7%）。

**[TTF-VLA: Temporal Token Fusion via Pixel-Attention Integration for Vision-Language-Action Models](object_detection/ttf-vla_temporal_token_fusion_via_pixel-attention_integratio.md)**

:   TTF-VLA 提出了一种免训练的时序 Token 融合方法，通过灰度像素差异+注意力语义检测的双维度机制选择性地复用历史帧的视觉 Token，提升 VLA 模型在机器人操作任务中的推理质量，在 LIBERO 上平均提升 4.0 个百分点。

---

## 🔄 自监督/表示学习 { #self_supervised }

**[BCE3S: Binary Cross-Entropy Based Tripartite Synergistic Learning for Long-tailed Recognition](self_supervised/bce3s_binary_cross-entropy_based_tripartite_synergistic_learning_for_long-tailed.md)**

:   提出 BCE3S，一种基于二元交叉熵（BCE）的三方协同学习框架，将 BCE 式联合学习、BCE 式对比学习和 BCE 式分类器均匀性学习集成在一起，通过 Sigmoid 解耦不同类别的度量来抑制长尾不平衡效应，在 CIFAR10/100-LT、ImageNet-LT 和 iNaturalist2018 上均取得 SOTA。

**[Explainable Melanoma Diagnosis with Contrastive Learning and LLM-based Report Generation](self_supervised/explainable_melanoma_diagnosis_with_contrastive_learning_and_llm-based_report_ge.md)**

:   提出 CEFM 框架，通过跨模态对比学习将 ViT 视觉特征与基于 ABCD 规则的临床特征（不对称性、边界、颜色）对齐，再由 CLIP + DeepSeek 生成结构化诊断报告，在 ISIC 数据集上达到 92.79% 准确率和 0.961 AUC，专家评分可解释性达 4.6/5。

**[Explanation-Preserving Augmentation for Semi-Supervised Graph Representation Learning](self_supervised/explanation-preserving_augmentation_for_semi-supervised_graph_representation_lea.md)**

:   提出EPA-GRL（Explanation-Preserving Augmentation），利用少量标签训练的GNN explainer识别图的语义子图（explanation subgraph），增强时只扰动非语义部分（marginal subgraph），实现语义保持的图增强，在6个benchmark上显著优于语义无关的随机增强方法。

**[GOAL: Geometrically Optimal Alignment for Continual Generalized Category Discovery](self_supervised/goal_geometrically_optimal_alignment_for_continual_generalized_category_discover.md)**

:   基于 Neural Collapse 理论，使用固定等角紧框架（ETF）分类器替代动态分类器，通过监督对齐和置信度引导的无监督对齐实现持续泛化类别发现，在四个基准上遗忘率降低 16.1%、新类发现提升 3.2%。

**[CAE: Hierarchical Semantic Alignment for Image Clustering](self_supervised/hierarchical_semantic_alignment_for_image_clustering.md)**

:   结合名词级（WordNet）和描述级（Flickr 图片描述）两种互补语义，通过最优传输对齐构建语义空间并自适应融合，实现 training-free 的图像聚类，在 ImageNet-1K 上准确率提升 4.2%。

**[Let The Void Be Void Robust Open-Set Semi-Supervised Learning Via Selective Non-](self_supervised/let_the_void_be_void_robust_open-set_semi-supervised_learning_via_selective_non-.md)**

:   提出 SkipAlign 框架，在对比学习的传统 pull/push 操作之外引入第三种 "skip" 操作，对低置信度样本选择性跳过对齐（只做温和排斥），使 ID 类形成紧凑"星系"、OOD 样本自然散布于"星际虚空"，在未见过的 OOD 检测中平均 AUC 提升 +3.1，最高 +7.1。

**[MovSemCL: Movement-Semantics Contrastive Learning for Trajectory Similarity](self_supervised/movsemcl_movement-semantics_contrastive_learning_for_trajectory_similarity_exten.md)**

:   提出 MovSemCL，将 GPS 轨迹转化为运动语义特征（位移向量+航向角+空间图嵌入），通过 patch 级双层注意力（intra-patch 局部 + inter-patch 全局）实现层次编码（复杂度从 $O(L^2)$ 降为近线性），并设计曲率引导增广保留转弯/路口等关键片段，在轨迹相似性搜索上精度提升达 72.6%、推理延迟降低 43.4%。

**[NeuroBridge: Bio-Inspired Self-Supervised EEG-to-Image Decoding](self_supervised/neurobridge_bio-inspired_self-supervised_eeg-to-image_decoding_via_cognitive_pri.md)**

:   提出 NeuroBridge，通过认知先验增强（CPA，对 EEG 和图像分别用非对称增广模拟感知变异性）+ 共享语义投影器（SSP，双向对齐到统一语义空间），在 200 类零样本 EEG-图像检索任务上达到 63.2% Top-1（+12.3%）和 89.9% Top-5（+10.2%），大幅超越现有 SOTA。

**[Robust Tabular Foundation Models](self_supervised/robust_tabular_foundation_models.md)**

:   提出 RTFM，一种模型无关的对抗训练框架，通过在参数化合成数据生成过程中寻找 TFM 表现不佳的"困难区域"（相比树模型基线的最优性差距最大），仅用 <10万额外合成数据集将 TabPFN V2 的平均归一化 AUC 提升最高 6%。

---

## 🎵 音频/语音 { #audio_speech }

**[DeepDebater: A Superpersuasive Autonomous Policy Debating System](audio_speech/a_superpersuasive_autonomous_policy_debating_system.md)**

:   提出 DeepDebater，首个能参与并赢得完整美式政策辩论赛的自主多 Agent 系统——层级式 Agent 工作流分工完成论证构建（正方 Advantage/反方 DA+CP+K），基于 OpenDebateEvidence 300 万张证据卡做检索增强，辅以 GPT-4o TTS 语音合成和 EchoMimic 数字人动画，在专家评估和模拟对局中全面超越人类编写的案例。

**[AHAMask: Reliable Task Specification for Large Audio Language Models without Instructions](audio_speech/ahamask_reliable_task_specification_for_large_audio_language.md)**

:   通过对大音频语言模型（LALM）Transformer 骨干中的注意力头进行二值掩码（AHAMask），无需文本指令即可可靠触发特定声学任务功能，同时揭示了 LALM 内部存在"声学功能通路"。

**[Aligning Generative Music AI with Human Preferences: Methods and Challenges](audio_speech/aligning_generative_music_ai_with_human_preferences_methods_and_challenges.md)**

:   综述论文，系统梳理了偏好对齐技术在音乐生成中的应用——包括 MusicRL（大规模 RLHF）、DiffRhythm+（扩散模型 DPO）和 Text2midi-InferAlign（推理时树搜索），讨论了音乐领域特有的对齐挑战（时间连贯性、和声一致性、主观性评估）和未来方向。

**[Cross-Space Synergy: A Unified Framework for Multimodal Emotion Recognition in Conversation](audio_speech/cross-space_synergy_a_unified_framework_for_multimodal_emotion_recognition_in_co.md)**

:   提出 Cross-Space Synergy（CSS）框架，通过表示空间的协同多项式融合（SPF）和梯度空间的 Pareto 梯度调节器（PGM）双管齐下，同时解决多模态对话情感识别中融合表达力不足和多目标梯度冲突两大难题。

**[DualSpeechLM: Towards Unified Speech Understanding and Generation via Dual Speech Token Modeling](audio_speech/dualspeechlm_towards_unified_speech_understanding_and_generation_via_dual_speech.md)**

:   提出 DualSpeechLM 框架，通过理解驱动语音分词器（USTokenizer）提取高层语义 token 作为 LLM 输入、声学 token 作为输出，在一个端到端框架中同时优化语音理解和生成能力。

**[Generalizing Analogical Inference from Boolean to Continuous Domains](audio_speech/generalizing_analogical_inference_from_boolean_to_continuous_domains.md)**

:   从基础理论层面重新审视类比推理：首先构造反例证明布尔域上经典泛化界失效，然后提出基于参数化广义均值的统一类比推理框架，将离散分类扩展到连续回归域。

**[Let the Model Learn to Feel: Mode-Guided Tonality Injection for Symbolic Music Emotion Recognition](audio_speech/let_the_model_learn_to_feel_mode-guided_tonality_injection_f.md)**

:   通过 MoGE 诊断策略系统发现 MIDIBERT 未有效编码调式-情感关联，提出 MoFi 注入框架通过 FiLM 机制将大调/小调先验注入 MIDIBERT 第 1 层（诊断确定的最弱情感信息层），在 EMOPIA 上准确率 75.2%（+11.8%），VGMIDI 上 59.1%（+11.8%），F1 提升 12.3%/15.5%。

**[Use A Unified Model For Universal Sound Separation And Extraction](audio_speech/use_a_unified_model_for_universal_sound_separation_and_extraction.md)**

:   提出 USE 统一框架，通过 EDA 网络推断声源数量和声学线索实现声音分离 (SS)，多模态融合网络解释用户提供的文本/视频/标签线索实现目标声音提取 (TSE)，联合训练+跨任务对齐使两项任务互相增强，SS +1.4dB SDR，TSE 匹配准确率 86%。

---

## 🕸️ 图学习 { #graph_learning }

**[Adaptive Initial Residual Connections for GNNs with Theoretical Guarantees](graph_learning/adaptive_initial_residual_connections_for_gnns_with_theoretical_guarantees.md)**

:   研究图神经网络中自适应初始残差连接（Adaptive IRC）——每个节点有个性化的残差强度——证明该方案防止过平滑（Dirichlet 能量有下界）、保持嵌入矩阵秩，在异质图上显著优于标准消息传递，并提出基于 PageRank 的非学习变体大幅降低复杂度。

**[Adaptive Riemannian Graph Neural Networks](graph_learning/adaptive_riemannian_graph_neural_networks.md)**

:   提出 ARGNN 框架，为图上每个节点学习一个连续的、各向异性的对角黎曼度量张量，从而自适应地捕获图中不同区域（层级结构 vs 密集社区）的局部几何特性，统一并超越了固定曲率和离散混合曲率的几何 GNN 方法。

**[Are Graph Transformers Necessary? Efficient Long-Range Message Passing with Fractal Nodes in MPNNs](graph_learning/are_graph_transformers_necessary_efficient_long-range_messag.md)**

:   提出分形节点（Fractal Nodes）增强 MPNN 的长距离消息传递：通过 METIS 图划分生成子图级聚合节点，结合低通+高通滤波器（LPF+HPF）与可学习频率参数 $\omega$，使用 MLP-Mixer 实跨子图通信，在保持 $O(L(|V|+|E|))$ 线性复杂度的同时达到甚至超越图 Transformer 的性能，获 AAAI Oral。

**[Assemble Your Crew: Automatic Multi-agent Communication Topology Design via Autoregressive Graph Generation](graph_learning/assemble_your_crew_automatic_multi-agent_communication_topol.md)**

:   提出 ARG-Designer，将多 Agent 系统的拓扑设计重新定义为条件自回归图生成任务，从零开始逐步生成 Agent 节点和通信边（而非从模板图剪枝），在6个基准上达到 SOTA（平均 92.78%），同时 Token 消耗比 G-Designer 降低约 50%，且支持无需重训练的角色扩展。

**[Posterior Label Smoothing for Node Classification](graph_learning/posterior_label_smoothing_for_node_classification.md)**

:   提出PosteL（Posterior Label Smoothing），通过贝叶斯后验分布从邻域标签中推导soft label用于节点分类，自然适应同质图和异质图，在8种backbone×10个数据集的80个组合中76个取得精度提升。

**[Relink: Constructing Query-Driven Evidence Graph On-the-Fly for GraphRAG](graph_learning/relink_constructing_query-driven_evidence_graph_on-the-fly_for_graphrag.md)**

:   提出从"先构建再推理"到"边推理边构建"的GraphRAG范式转变，通过Relink框架动态构建查询特定的证据图——结合高精度KG骨架和高召回潜在关系池，用查询驱动的排序器统一评估、按需补全缺失路径并过滤干扰事实——在5个多跳QA基准上平均提升EM 5.4%和F1 5.2%。

**[RFKG-CoT: Relation-Driven Adaptive Hop-count Selection and Few-Shot Path Guidance for Knowledge-Aware QA](graph_learning/rfkg-cot_relation-driven_adaptive_hop-count_selection_and_few-shot_path_guidance.md)**

:   提出RFKG-CoT，通过关系驱动的自适应跳数选择（利用KG关系激活掩码动态调整推理步数）和Few-Shot路径引导（Question-Paths-Answer格式的in-context示例），在4个KGQA基准上显著提升LLM的知识图谱推理能力，GPT-4在WebQSP上达91.5%（+6.6pp），Llama2-7B提升幅度最大达+14.7pp。

**[S-DAG: A Subject-Based Directed Acyclic Graph for Multi-Agent Heterogeneous Reasoning](graph_learning/s-dag_a_subject-based_directed_acyclic_graph_for_multi-agent.md)**

:   提出 S-DAG，通过 GNN 从问题中识别相关学科及其依赖关系构建有向无环图，将学科节点匹配到最擅长的专家 LLM（14 个 7-13B 领域模型），按 DAG 拓扑顺序协作推理（支撑学科→主导学科），用小模型池超越 GPT-4o-mini（59.73 vs 58.52）且接近 72B 模型。

---

## 🎬 视频理解 { #video_understanding }

**[3D4D: An Interactive Editable 4D World Model via 3D Video Generation](video_understanding/3d4d_an_interactive_editable_4d_world_model_via_3d_video_generation.md)**

:   提出 3D4D，一个集成 WebGL 和 Supersplat 渲染的交互式 4D 可视化框架，通过四个后端模块（3D重建、图像生视频、视频分帧、4D场景生成）将静态图片和文本转化为可实时交互的 4D 场景，并引入 VLM 引导的注视点渲染策略在保持语义一致性的同时实现 60fps 实时交互。

**[APVR: Hour-Level Long Video Understanding with Adaptive Pivot Visual Information Retrieval](video_understanding/apvr_hour-level_long_video_understanding_with_adaptive_pivot.md)**

:   提出APVR，一个训练免费的双粒度视觉信息检索框架：帧级别通过查询扩展+时空语义置信度打分迭代检索关键帧（最多1024帧），token级别通过查询感知的注意力驱动选择压缩视觉token，突破内存墙限制处理小时级长视频，在LongVideoBench/VideoMME/MLVU上分别提升最高9.5%/4.6%/9.7%。

**[Distillation Dynamics: Towards Understanding Feature-Based Distillation in Vision Transformers](video_understanding/distillation_dynamics_towards_understanding_feature-based_di.md)**

:   提出"蒸馏动力学"分析框架（频谱分析+信息熵+激活幅值），揭示ViT具有独特的U型信息处理模式（先压缩后扩展），证明feature-based蒸馏在ViT中失败的根本原因是teacher后层的分布式高维编码范式与student有限通道容量之间的表征范式不匹配，而非简单的容量差距。

**[DreamRunner: Fine-Grained Compositional Story-to-Video Generation with Retrieval-Augmented Motion Adaptation](video_understanding/dreamrunner_fine-grained_compositional_story-to-video_genera.md)**

:   提出 DreamRunner 框架，通过 LLM 双层规划 + 检索增强运动先验学习 + 时空区域3D注意力模块(SR3AI)，实现细粒度可控的多角色多事件故事视频生成。

**[MambaMia: State-Space Hierarchical Compression for Hour-Long Video Understanding in Large Multimodal Models](video_understanding/state-space_hierarchical_compression_with_gated_attention_an.md)**

:   MambaMia 提出了基于双向 Mamba 的两阶段层次化视频 Token 压缩框架：门控 Patch 聚合（GPA）做空间-时间局部压缩 + 时间轴聚合器（TAA）利用 Mamba 的自适应步长 $\Delta_t$ 做数据驱动的关键帧采样，将小时级视频压缩到仅 4.7K Token，在 LVBench 上达到 44.6 分超越 Qwen2-VL 和 mPLUG-Owl3。

**[VIR-Bench: Evaluating Geospatial and Temporal Understanding of MLLMs via Travel Video Itinerary Reconstruction](video_understanding/vir-bench_evaluating_geospatial_and_temporal_understanding_of_mllms_via_travel_v.md)**

:   提出VIR-Bench——一个基于200个日本旅行vlog视频的benchmark，通过行程重建任务（visiting order graph构建）评估MLLM的地理空间和时间理解能力，发现SOTA模型（包括GPT-4.1和Gemini-2.5）在POI识别和时间转移推理上仍困难重重。

---

## 🔗 因果推理 { #causal_inference }

**[CaDyT: Causal Structure Learning for Dynamical Systems with Theoretical Score Analysis](causal_inference/causal_structure_learning_for_dynamical_systems_with_theoretical_score_analysis.md)**

:   提出 CaDyT，结合高斯过程连续时间动力学建模（Adams-Bashforth 积分器实现精确推断）和 MDL 最小描述长度原则进行结构搜索，同时解决不规则采样和因果结构识别两个挑战，在双质点弹簧/菱形图/Rössler 振荡器上大幅超越所有基线（AUPRC 0.79 vs 次优 0.39）。

**[Causally-Grounded Dual-Path Attention Intervention for Object Hallucination Mitigation in LVLMs](causal_inference/causally-grounded_dual-path_attention_intervention_for_objec.md)**

:   提出 Owl 框架，通过结构因果模型将视觉/文本注意力建模为中介变量，引入 VTACR 指标量化跨模态注意力失衡，设计 VTACR 引导的自适应注意力调制 + 双路径对比解码策略，在 POPE 和 CHAIR 上实现 SOTA 的幻觉抑制效果。

**[Hallucinate Less by Thinking More: Aspect-Based Causal Abstention for Large Language Models](causal_inference/hallucinate_less_by_thinking_more_aspect-based_causal_absten.md)**

:   提出 ABCA（Aspect-Based Causal Abstention），一个生成前弃权框架：通过双 Agent 辩论发现"方面变量"（如学科、法律语境、时间框架）来激活 LLM 不同的知识分支，用 AIPW 双鲁棒估计器计算因果效应，基于质心角偏差（CAD）检测知识冲突（Type-1）或知识不足（Type-2），在 TruthfulQA 上达到 91.4% 准确率，不可回答问题识别率 96.4%（远超基线的 44%）。

**[Learning Subgroups with Maximum Treatment Effects without Causal Heuristics](causal_inference/learning_subgroups_with_maximum_treatment_effects_without_causal_heuristics.md)**

:   在 SCM 框架下证明最大处理效应子群必须具有同质点效应（定理1），在分区模型假设下证明最优子群发现可化简为标准监督学习（定理2），用 CART+Gini 指数即可实现——在 77 个 ACIC-2016 半合成数据集上均值处理效应 10.54（vs 次优 7.84），51.9% 排名第一。

**[MUG: Multi-agent Undercover Gaming — Hallucination Removal via Counterfactual Test for Multimodal Reasoning](causal_inference/multi-agent_undercover_gaming_hallucination_removal_via_coun.md)**

:   MUG 将多 Agent 辩论（MAD）重新定义为"谁是卧底"社交推理游戏——通过图像反事实编辑（修改参考图片）引入信息不对称，让一个 Agent 持有修改后的图片作为"卧底"，其他 Agent 通过推理和投票识别卧底（幻觉来源），在 HallusionBench 上 Qwen2.5VL-7B 从 46.4% 提升到 53.8%。

---

## 🧑 人体理解 { #human_understanding }

**[10 Open Challenges Steering the Future of Vision-Language-Action Models](human_understanding/10_open_challenges_steering_the_future_of_vision-language-ac.md)**

:   一篇针对Vision-Language-Action(VLA)模型的综述/展望论文，系统梳理了VLA领域的10大开放挑战（多模态感知、鲁棒推理、数据质量、评估、跨机器人泛化、效率、全身协调、安全、多智能体、人机协作）以及4大新兴趋势（层次化规划、空间理解、世界动力学建模、数据合成），为VLA研究指明方向。

**[AHAN: Asymmetric Hierarchical Attention Network for Identical Twin Face Verification](human_understanding/ahan_asymmetric_hierarchical_attention_network_for_identical.md)**

:   针对同卵双胞胎人脸验证这一极端细粒度识别挑战，提出 AHAN 多流架构，通过层次交叉注意力 (HCA) 对语义面部区域做多尺度分析、面部不对称注意力模块 (FAAM) 捕获左右脸差异签名、以及双胞胎感知配对交叉注意力 (TA-PWCA) 训练正则化，在 ND_TWIN 数据集上将双胞胎验证精度从 88.9% 提升至 92.3%（+3.4%）。

**[Anti-adversarial Learning: Desensitizing Prompts for Large Language Models](human_understanding/anti-adversarial_learning_desensitizing_prompts_for_large_la.md)**

:   提出 PromptObfus，通过"反对抗学习"思路将用户 prompt 中的敏感词替换为语义不同但不影响任务输出的词，从而在不降低远端 LLM 任务表现的前提下彻底消除显式隐私泄露，并将隐式隐私推理攻击成功率降低 62.70%。

**[Bias Association Discovery Framework for Open-Ended LLM Generations](human_understanding/bias_association_discovery_framework_for_open-ended_llm_generations.md)**

:   提出偏见关联发现框架 BADF，通过分析 LLM 开放式故事生成中的叙事内容，系统性地提取人口统计身份与描述性概念之间的已知和未知偏见关联，突破了以往依赖预定义偏见概念的局限。

**[RENEW: Risk- and Energy-Aware Navigation in Dynamic Waterways](human_understanding/renew_risk-_and_energy-aware_navigation_in_dynamic_waterways.md)**

:   提出 RENEW 全局路径规划器，为水面自主航行器 (ASV) 在动态水流 (洋流) 环境中引入统一的风险感知和能量感知策略，通过自适应不可导航区域识别、最佳努力应急策略和基于约束 Delaunay 三角化的分层架构实现安全高效导航，应急碰撞测试中实现零碰撞。

---

## ✍️ 文本生成 { #nlp_generation }

**[A Coherence-Based Measure of AGI](nlp_generation/a_coherence-based_measure_of_agi.md)**

:   指出现有 AGI 评分用算术平均隐含"可补偿"假设（强项弥补弱项），提出基于广义均值连续谱的一致性度量 $\text{AGI}_{\text{AUC}}$：在补偿性参数 $p \in [-1, 1]$ 上积分，惩罚能力不均衡，暴露被算术平均掩盖的瓶颈。

**[AutoMalDesc: Large-Scale Script Analysis for Cyber Threat Research](nlp_generation/automaldesc_large-scale_script_analysis_for_cyber_threat_research.md)**

:   提出 AutoMalDesc 自动化静态分析框架，通过迭代自步学习流水线——从 900 个专家标注种子样本出发，经 LoRA 微调 Llama-3.3-70B 生成伪标签，多阶段质量过滤后进行 V2 训练——实现 5 种脚本语言的恶意软件自动分类和行为描述，Batch 脚本检测准确率从 52.7% 提升到 82.4%。

**[Magnitude Matters: A Superior Class of Similarity Metrics for Holistic Semantic Understanding](nlp_generation/magnitude_matters_a_superior_class_of_similarity_metrics_for_holistic_semantic_u.md)**

:   提出两种无参数、幅度感知的向量相似度度量——Overlap Similarity (OS) 和 Hyperbolic Tangent Similarity (HTS)，在 4 个句子嵌入模型和 8 个 NLP 基准上，对分类任务（释义、推理）的 MSE 显著低于 Cosine Similarity 和 Dot Product，且无需任何额外训练开销。

**[PERSIST: Persistent Instability in LLM's Personality Measurements](nlp_generation/persistent_instability_in_llms_personality_measurements_effects_of_scale_reasoni.md)**

:   PERSIST 框架系统评估 25 个开源 LLM（1B-685B）在 200 万+响应上的人格测量稳定性，发现即使 400B+模型在 5 分制量表上仍有 SD>0.3 的不稳定性，且 CoT 推理悖论性地增加变异性同时降低困惑度，LLM 适配问卷与传统人类问卷表现出相似的不稳定性。

**[TAPA: Training-Free Adaptation of Programmatic Agents via LLM-Guided Program Synthesis in Dynamic Environments](nlp_generation/tapas_are_free_training-free_adaptation_of_programmatic_agen.md)**

:   TAPA 将 LLM 定位为符号动作空间的"智能调制器"而非直接决策者，通过 LLM 引导的程序合成动态适配程序化 Agent 的符号动作，无需重新训练即可适应动态环境，在网络安全 DDoS 防御（77.7% 网络正常运行率）和群体智能编队控制中表现优异。

---

## 🎁 推荐系统 { #recommender }

**[Align³GR: Unified Multi-Level Alignment for LLM-based Generative Recommendation](recommender/align3gr_unified_multi-level_alignment_for_llm-based_generat.md)**

:   提出统一三层对齐框架 Align³GR，在 token 级（双端 SCID）、行为建模级（多任务 SFT）和偏好级（渐进式 DPO）系统性弥合 LLM 与推荐系统之间的语义-行为鸿沟。

**[FreqRec: Exploiting Inter-Session Information with Frequency-enhanced Dual-Path Networks for Sequential Recommendation](recommender/exploiting_inter-session_information_with_frequency-enhanced_dual-path_networks_.md)**

:   提出FreqRec双路径架构，通过batch维和时间维两条频域路径分别捕获跨session群体节律和用户个体细粒度兴趣，并引入频域一致性损失显式对齐预测与真实频谱，在三个Amazon数据集上NDCG@10最高提升7.38%。

**[From Parameter to Representation: A Closed-Form Approach for Controllable Model Merging](recommender/from_parameter_to_representation_a_closed-form_approach_for_controllable_model_m.md)**

:   提出 ReACT，将可控模型合并从参数空间优化转移到表征空间校正，通过闭式解实现任意用户偏好下的 Pareto 最优模型即时生成，比现有方法快 36-208 倍且性能更优。

**[Inference-Aware Prompt Optimization for Aligning Black-Box Large Language Models](recommender/inference-aware_prompt_optimization_for_aligning_black-box_large_language_models.md)**

:   揭示 prompt 选择与推理策略（Best-of-N、Majority Voting）之间存在非平凡交互关系，提出 IAPO 框架将 prompt 设计与推理规模联合优化为上下文最优臂识别问题，并设计 PSST 固定预算训练算法，在 6 个任务上相比推理无关方法提升最高 50%。

**[Wavelet Enhanced Adaptive Frequency Filter for Sequential Recommendation](recommender/wavelet_enhanced_adaptive_frequency_filter_for_sequential_re.md)**

:   提出WEARec模型结合动态频域滤波(DFF)和小波特征增强(WFE)两个模块，分别捕获个性化全局频域信息和增强非平稳短期波动，在四个公开数据集上超越频域推荐SOTA基线，长序列场景提升可达11.4%。

---

## 🛰️ 遥感 { #remote_sensing }

**[Consistency-based Abductive Reasoning over Perceptual Errors of Multiple Pre-trained Models in Novel Environments](remote_sensing/consistency-based_abductive_reasoning_over_perceptual_errors_of_multiple_pre-tra.md)**

:   将多个预训练感知模型在新环境中的冲突预测建模为一致性溯因推理问题，通过逻辑程序编码各模型的错误检测规则和领域约束，寻找在保持不一致率低于阈值的同时最大化预测覆盖率的最优假设，在15个航拍测试集上平均F1提升13.6%。

**[Debiasing Machine Learning Predictions for Causal Inference Without Additional Ground Truth Data](remote_sensing/debiasing_machine_learning_predictions_for_causal_inference_without_additional_g.md)**

:   针对ML卫星贫困预测因均值回归导致因果处理效应衰减的问题，提出两种无需新标注数据的后处理校正方法——线性校准校正(LCC)和Tweedie局部去收缩——使同一预测地图可在多个下游因果试验中复用（"一图多试"范式），Tweedie校正在模拟和DHS真实数据上实现近无偏的处理效应估计。

**[M3SR: Multi-Scale Multi-Perceptual Mamba for Efficient Spectral Reconstruction](remote_sensing/m3sr_multi-scale_multi-perceptual_mamba_for_efficient_spectral_reconstruction.md)**

:   提出 M3SR，基于 Mamba 的 U-Net 架构，通过多感知融合 (MPF) 模块在空间、频率和光谱三个维度并行建模并自适应融合，以 2.17M 参数和 100.9G FLOPs 实现 4 个基准上的 SOTA 高光谱重建（NTIRE2022 PSNR 31.40）。

**[Machine Learning for Sustainable Rice Production: Region-Scale Monitoring of Water-Saving Practices in Punjab, India](remote_sensing/machine_learning_for_sustainable_rice_production_region-scale_monitoring_of_wate.md)**

:   提出维度分类方法将水稻节水实践识别解耦为播种维度(DSR vs PTR)和灌溉维度(AWD vs CF)两个独立二分类任务，仅使用Sentinel-1 SAR影像实现播种F1=0.80和灌溉F1=0.74，并在旁遮普邦300万+地块上进行大规模推理，地区级采纳率与政府统计高度相关（Spearman ρ=0.69）。

**[TDCNet: Spatio-Temporal Context Learning with Temporal Difference Convolution for Moving IRSTD](remote_sensing/spatio-temporal_context_learning_with_temporal_difference_convolution_for_moving.md)**

:   提出 TDCNet，将时间差分和 3D 卷积融合为统一的时间差分卷积 (TDC)，通过重参数化实现推理零额外开销，配合 TDC 引导的时空注意力，在自建 IRSTD-UAV 数据集上 F1 达 97.12%（AP50 93.83%），同时发布 15,106 帧真实红外无人机数据集。

---

## ✂️ 语义分割 { #segmentation }

**[3DTeethSAM: Taming SAM2 for 3D Teeth Segmentation](segmentation/3dteethsam_taming_sam2_for_3d_teeth_segmentation.md)**

:   将SAM2基础模型迁移到3D牙齿分割任务，通过多视角渲染将3D mesh转为2D图像、设计三个轻量适配器（Prompt生成器、Mask精化器、Mask分类器）和可变形全局注意力插件（DGAP）来解决自动提示、边界精化和语义分类问题，在Teeth3DS上以91.90% T-mIoU刷新SOTA。

**[A²LC: Active and Automated Label Correction for Semantic Segmentation](segmentation/a2lc_active_and_automated_label_correction_for_semantic_segm.md)**

:   提出 A²LC 框架，在传统主动标签校正（人工逐一纠错）的基础上增加一个自动校正阶段（Label Correction Module），利用标注员的反馈自动修正相似的错误mask，并设计自适应平衡采集函数缓解类别不平衡，在 Cityscapes 上仅用 20% 预算即超越前 SOTA，同等预算下 mIoU 提升 27.23%。

**[Adaptive Morph-Patch Transformer for Aortic Vessel Segmentation](segmentation/adaptive_morph-patch_transformer_for_aortic_vessel_segmentat.md)**

:   提出 Morph-Patch Transformer (MPT)，通过基于速度场的自适应 patch 划分策略生成形态感知 patch（保持血管拓扑完整性），并引入语义聚类注意力（SCA）动态聚合语义相似 patch 的特征，在 AVT、AortaSeg24 和 TBAD 三个主动脉分割数据集上均达 SOTA。

**[Causal-Tune: Mining Causal Factors from Vision Foundation Models for Domain Generalized Semantic Segmentation](segmentation/causal-tune_mining_causal_factors_from_vision_foundation_mod.md)**

:   提出Causal-Tune，从因果视角分析VFM特征中的artifacts，利用DCT频域分解+高斯带通滤波分离因果/非因果因素，结合因果感知可学习token在频域精化特征，在Cityscapes→ACDC跨域分割中平均提升+2.4% mIoU（Snow场景+4.8%），仅需单卡RTX3090/14GB训练。

**[InfoCLIP: Bridging Vision-Language Pretraining and Open-Vocabulary Semantic Segmentation via Information-Theoretic Alignment Transfer](segmentation/infoclip_bridging_vision-language_pretraining_and_open-vocab.md)**

:   提出InfoCLIP，基于信息论视角设计信息瓶颈压缩和互信息蒸馏两个目标，在CLIP微调过程中去除预训练pixel-text对齐中的噪声并保留语义对齐知识，在6个开放词汇语义分割测试集上全面超越SOTA（A-847: 16.6, A-150: 38.5, PC-59: 63.5 mIoU），且仅增加0.53M参数和极少计算开销。

---

## 📡 信号/通信 { #signal_comm }

**[Beyond Perplexity: Let the Reader Select Retrieval Summaries via Spectrum Projection Score](signal_comm/beyond_perplexity_let_the_reader_select_retrieval_summaries_via_spectrum_project.md)**

:   提出 Spectrum Projection Score (SPS) 这一无需训练的指标，通过衡量摘要 token 嵌入与 reader LLM 主子空间的对齐程度来评估检索摘要质量，替代传统困惑度指标。结合 xCompress 推理时控制器，在 5 个 QA 数据集上显著优于基于困惑度的方法（HotpotQA EM +3.6）。

**[GateRA: Token-Aware Modulation for Parameter-Efficient Fine-Tuning](signal_comm/gatera_token-aware_modulation_for_parameter-efficient_fine-tuning.md)**

:   提出 GateRA，在 PEFT 方法（LoRA/DoRA/HiRA）中引入轻量级 token 感知门控模块，通过 sigmoid 门控动态调整每个 token 的适配强度——对分布内/简单 token 抑制更新以保留预训练知识，对挑战性 token 放大适配。结合熵正则化促进近二值门控决策，在常识推理（+1.1%）、对话和数学推理上一致优于 HiRA。

**[Task Aware Modulation Using Representation Learning For Upsaling Of Terrestrial ](signal_comm/task_aware_modulation_using_representation_learning_for_upsaling_of_terrestrial_.md)**

:   提出 TAM-RL 框架，将陆地碳通量升尺度问题建模为零样本回归迁移学习任务，用 BiLSTM 任务编码器+FiLM 调制结合碳平衡方程知识引导损失，在 150+ 通量塔站点上将 GPP RMSE 降低 9.6%、NEE R² 提升 43.8%（相较 FLUXCOM-X-BASE）。

**[Text-Guided Channel Perturbation And Pretrained Knowledge Integration For Unifie](signal_comm/text-guided_channel_perturbation_and_pretrained_knowledge_integration_for_unifie.md)**

:   提出 UP-Fusion 统一多模态图像融合框架，通过语义感知通道剪枝 (SCPM)、几何仿射调制 (GAM) 和 CLIP 文本引导通道扰动 (TCPM) 三个模块，用单组权重（仅在红外-可见光数据上训练）同时处理 IVIF 和医学图像融合，在两类任务上均达到 SOTA。

**[Toward Gaze Target Detection in Young Autistic Children](signal_comm/toward_gaze_target_detection_of_young_autistic_children.md)**

:   针对自闭症儿童注视目标检测中面部注视（6.6%）严重不足的类别不平衡问题，提出 Socially Aware Coarse-to-Fine (SACF) 框架，用微调的 Qwen2.5-VL 作为社交上下文感知门控，将输入路由到社交感知/社交无关两个专家模型，在首创的 AGT 数据集上显著提升了面部注视检测性能（Face L2 在 Sharingan 上降低 13.9%, F1 从 0.753 提升至 0.761）。

---

## 🖼️ 图像恢复 { #image_restoration }

**[Clear Nights Ahead: Towards Multi-Weather Nighttime Image Restoration](image_restoration/clear_nights_ahead_towards_multi-weather_nighttime_image_res.md)**

:   首次定义并探索多天气夜间图像复原任务，构建 AllWeatherNight 数据集（8K 训练 + 1K 合成测试 + 1K 真实测试），提出 ClearNight 统一框架通过 Retinex 双先验引导和天气感知动态专一性-共性协作，一阶段同时移除雾/雨条/雨滴/雪/flare 复合退化，仅 2.84M 参数全面超越 SOTA。

**[ICLR: Inter-Chrominance and Luminance Interaction for Natural Color Restoration in Low-Light Image Enhancement](image_restoration/iclr_inter-chrominance_and_luminance_interaction_for_natural_color_restoration_i.md)**

:   针对HVI色彩空间中色度和亮度分支分布差异大导致互补特征提取不足、以及色度分支间弱相关导致梯度冲突的问题，提出ICLR框架，通过双流交互增强模块(DIEM)和协方差校正损失(CCL)分别从融合增强和统计分布优化两个角度解决，在LOL系列数据集上取得SOTA。

**[SD-PSFNet: Sequential and Dynamic Point Spread Function Network for Image Deraining](image_restoration/sd-psfnet_sequential_and_dynamic_point_spread_function_netwo.md)**

:   提出基于动态 PSF 机制的级联 CNN 去雨网络 SD-PSFNet，通过多尺度可学习 PSF 字典建模雨滴光学效应，配合自适应门控融合的序列化修复架构，在 Rain100H 达 33.12 dB、RealRain-1k-L 达 42.28 dB 均为 SOTA，对比基线 MPRNet 累计提升 5.04 dB（13.5%）。

**[Seeing the Unseen: Zooming in the Dark with Event Cameras](image_restoration/seeing_the_unseen_zooming_in_the_dark_with_event_cameras.md)**

:   提出首个事件驱动低光照视频超分辨率框架RetinexEVSR，通过Retinex启发的双向融合策略（光照引导事件增强+事件引导反射增强），在SDSD-indoor上较EvTexture提升2.95 dB且FLOPs减少86%、推理加速65%。

---

## 📖 NLP 理解 { #nlp_understanding }

**[Language Models and Logic Programs for Trustworthy Tax Reasoning](nlp_understanding/language_models_and_logic_programs_for_trustworthy_tax_reasoning.md)**

:   将税法推理重新定义为语义解析任务，让LLM将法规文本和纳税案例翻译为Prolog逻辑程序，由符号求解器执行计算，通过金标准法规+智能检索案例示例+自一致性检查，在SARA数据集上实现86/100的正确率，并将预计部署成本降至15.78美元/人（低于美国人均报税成本的6%）。

**[NeSTR: A Neuro-Symbolic Abductive Framework for Temporal Reasoning in Large Language Models](nlp_understanding/nestr_a_neuro-symbolic_abductive_framework_for_temporal_reasoning_in_large_langu.md)**

:   提出 NeSTR 神经符号提示策略，通过将自然语言时间事实转化为结构化符号谓词，结合一致性验证和溯因反思修正，在零样本设置下让 LLM 实现高质量时间推理，GPT-4o-mini 上平均 F1 达 89.7（vs vanilla 64.9，TISER 85.8）。

**[REAP: Enhancing RAG with Recursive Evaluation and Adaptive Planning for Multi-Hop Question Answering](nlp_understanding/reap_enhancing_rag_with_recursive_evaluation_and_adaptive_planning_for_multi-hop.md)**

:   提出 REAP 双模块迭代框架，通过子任务规划器 (SP) 维护全局视角动态指导推理轨迹，事实提取器 (FE) 从检索内容中提取结构化事实和潜在线索，两者递归协作解决多跳问答。在 4 个基准上以 Llama-3.1-8B 显著超越所有基线（HotpotQA F1 68.0 vs 次优 63.4）。

**[Understanding Syllogistic Reasoning in LLMs from Formal and Natural Language Perspectives](nlp_understanding/understanding_syllogistic_reasoning_in_llms_from_formal_and_natural_language_per.md)**

:   系统评估14个LLM在160个三段论上的推理表现，发现顶级模型在形式逻辑（句法有效性）上接近完美但在自然语言可信度判断上仅为随机水平——这与人类推理模式恰好相反；12/14模型存在信念偏差，且few-shot提示反而显著降低形式推理性能。

---

## 📐 优化/理论 { #optimization }

**[A Distributed Asynchronous Generalized Momentum Algorithm Without Delay Bounds](optimization/a_distributed_asynchronous_generalized_momentum_algorithm_wi.md)**

:   提出一种完全异步（totally asynchronous）的广义动量（Generalized Momentum）分布式优化算法，无需假设通信/计算延迟的上界即可保证线性收敛，在 Fashion-MNIST 分类任务上比梯度下降快 71%、比 Heavy Ball 快 41%、比 Nesterov 加速梯度法快 19%。

**[A Unified Convergence Analysis for Semi-Decentralized Learning: Sampled-to-Sampled vs. Sampled-to-All Communication](optimization/a_unified_convergence_analysis_for_semi-decentralized_learni.md)**

:   本文在统一的收敛分析框架下，首次系统比较了半去中心化联邦学习中两种服务器-设备通信原语（S2S仅返回被采样设备 vs. S2A广播给所有设备），揭示了S2S在高组间异质性下更优、S2A在低异质性下更优的不同regime，并给出了实用的系统配置指南。

**[Explore How to Inject Beneficial Noise in MLLMs](optimization/explore_how_to_inject_beneficial_noise_in_mllms.md)**

:   提出 Multimodal Noise Generator (MuNG)，通过变分推断框架从图文对中动态生成"有益噪声"注入冻结的MLLM视觉特征中，以抑制无关语义、增强跨模态表征对齐，仅需约1%额外参数即可超越全参数微调和LoRA等PEFT方法。

**[On the Learning Dynamics of Two-Layer Linear Networks with Label Noise SGD](optimization/on_the_learning_dynamics_of_two-layer_linear_networks_with_label_noise_sgd.md)**

:   在二层过参数化线性网络上理论分析 Label Noise SGD 的学习动力学，揭示了两阶段行为——Phase I 中权重范数逐渐缩小使模型从 lazy regime 逃逸到 rich regime，Phase II 中权重与真实插值器对齐并收敛——并将该理论扩展到 SAM 优化器。

---

## 🧮 科学计算 { #scientific_computing }

**[PhysicsCorrect: A Training-Free Approach for Stable Neural PDE Simulations](scientific_computing/physicscorrect_a_training-free_approach_for_stable_neural_pde_simulations.md)**

:   提出 PhysicsCorrect，一种无需训练的校正框架，通过将 PDE 残差校正建模为线性化逆问题并预计算伪逆缓存，在推理时以 <5% 计算开销实现最高 100× 误差降低，适用于 FNO/UNet/ViT 等任意预训练神经算子。

**[Pimrl Physics-Informed Multi-Scale Recurrent Learning For Burst-Sampled Spatiote](scientific_computing/pimrl_physics-informed_multi-scale_recurrent_learning_for_burst-sampled_spatiote.md)**

:   提出 PIMRL 框架，针对 burst 采样（短段高频+长间隔）的稀疏时空数据，结合宏观尺度潜空间推理和微观尺度物理校正的双模块架构，通过跨尺度消息传递融合信息，在 5 个 PDE 基准上将误差最多降低 80%。

**[SAOT: An Enhanced Locality-Aware Spectral Transformer for Solving PDEs](scientific_computing/saot_an_enhanced_locality-aware_spectral_transformer_for_solving_pdes.md)**

:   提出 SAOT（Spectral Attention Operator Transformer），通过线性复杂度的小波注意力 (WA) 捕获高频局部特征，与傅里叶注意力 (FA) 的全局感受野通过门控融合，在 6 个算子学习基准上达到 SOTA（Navier-Stokes 误差比 Transolver 降 22.3%）。

**[Scientific Knowledge-Guided Machine Learning for Vessel Power Prediction: A Comparative Study](scientific_computing/scientific_knowledge-guided_machine_learning_for_vessel_power_prediction_a_compa.md)**

:   提出物理基线+数据驱动残差的混合建模框架，将海试功率曲线（螺旋桨定律 $P=cV^n$）作为基线，用 XGBoost/NN/PINN 学习残差修正，在稀疏数据区域显著提升外推稳定性和物理一致性。

---

## 🔎 AIGC 检测 { #aigc_detection }

**[ActiShade: Activating Overshadowed Knowledge to Guide Multi-Hop Reasoning in Large Language Models](aigc_detection/actishade_activating_overshadowed_knowledge_to_guide_multi-h.md)**

:   提出ActiShade框架，通过高斯噪声扰动检测LLM在多跳推理中被"遮蔽"的关键短语，结合定制对比学习检索器获取补充文档，迭代重构查询以减少知识遮蔽导致的错误累积，在HotpotQA/2WikiMQA/MuSiQue上显著超越DRAGIN等SOTA。

**[BAID: A Benchmark for Bias Assessment of AI Detectors](aigc_detection/baid_a_benchmark_for_bias_assessment_of_ai_detectors.md)**

:   提出 BAID 基准数据集（20.8万样本对，覆盖7类偏见维度、41个子群体），系统评估4个开源 AI 文本检测器在不同人口统计和语言学子群体上的公平性表现，揭示检测器对方言、非正式英语和少数群体文本存在显著的召回率差异。

**[Optimized Algorithms for Text Clustering with LLM-Generated Constraints](aigc_detection/optimized_algorithms_for_text_clustering_with_llm-generated_constraints.md)**

:   提出 LSCK-HC 框架，利用 LLM 生成集合形式的 must-link/cannot-link 约束（而非传统成对约束），配合带惩罚项的局部搜索聚类算法，在5个短文本数据集上实现与 SOTA 可比的聚类精度，同时将 LLM 查询次数减少 20 倍以上。

---

## ⚛️ 物理学 { #physics }

**[Adaptive Fidelity Estimation for Quantum Programs with Graph-Guided Noise Awareness](physics/adaptive_fidelity_estimation_for_quantum_programs_with_graph.md)**

:   提出 QuFid 框架，将量子电路建模为有向无环图，通过控制流感知的随机游走刻画噪声传播，利用算子谱特征量化电路复杂度，实现自适应测量预算分配，在保持保真度精度的同时大幅减少测量次数。

**[Data Verification is the Future of Quantum Computing Copilots](physics/data_verification_is_the_future_of_quantum_computing_copilots.md)**

:   这是一篇 position paper，提出量子计算 AI 助手（Copilot）必须将数据验证从事后过滤提升为架构级基础——通过三个立场论证：(1) 验证数据是最低要求，(2) 先验约束优于后验过滤，(3) 受物理定律约束的科学领域需要验证感知架构。实验表明无验证数据的 LLM 在电路优化上最高仅达 79% 准确率。

**[STELLAR: Scene Text Editor for Low-Resource Languages and Real-World Data](physics/stellar_scene_text_editor_for_low-resource_languages_and_real-world_data.md)**

:   提出 STELLAR 框架，通过语言自适应字形编码器和多阶段训练策略（合成预训练+真实图像微调），实现韩语/阿拉伯语/日语等低资源语言的场景文本编辑，并提出 TAS 指标独立评估字体/颜色/背景的风格保持，韩语识别准确率从基线的 22.1% 提升至 80.4%。

---

## 🌍 地球科学 { #earth_science }

**[MdaIF: Robust One-Stop Multi-Degradation-Aware Image Fusion with Language-Driven Semantics](earth_science/mdaif_robust_one-stop_multi-degradation-aware_image_fusion_with_language-driven_.md)**

:   提出 MdaIF 框架，利用视觉语言模型（VLM）提取退化感知语义先验来引导混合专家（MoE）路由和通道注意力调制，实现无需退化类型标注的一站式多退化场景红外-可见光图像融合。

---

## 📂 其他 { #others }

**[A Fast Heuristic Search Approach for Energy-Optimal Profile Routing for Electric Vehicles](others/a_fast_heuristic_search_approach_for_energy-optimal_profile_.md)**

:   提出基于多目标A*搜索的label-setting方法（Pr-A*），在初始电量未知时高效求解电动车能耗最优路径（profile搜索），通过profile支配关系剪枝避免传统方法中复杂的profile合并操作，在大规模路网上性能接近已知初始电量的标准A*搜索。

**[A Graph-Theoretical Perspective on Law Design for Multiagent Systems](others/a_graph-theoretical_perspective_on_law_design_for_multiagent.md)**

:   将多智能体系统中的法律设计问题（包括"有用法律"和"无责任缺口法律"）形式化为超图上的顶点覆盖问题，证明了两类法律最小化问题都是NP-hard的，并给出了基于超图顶点覆盖近似算法的多项式时间近似方案。

**[A Graph-Theoretical Perspective on Law Design for Multiagent Systems](others/a_graph-theoretical_perspective_on_law_design_for_multiagent_systems.md)**

:   从图论角度研究多智能体系统中的法律设计问题，将 useful law 和 gap-free law 的最小化设计分别归约为超图的顶点覆盖问题，证明了 NP-hardness 并给出近似算法。

**[A Mind Cannot Be Smeared Across Time](others/a_mind_cannot_be_smeared_across_time.md)**

:   从 Stack Theory 出发形式化证明：在时间窗口内的存在性时序实现不保持合取——系统可以跨时间实现意识体验的每个成分而永远不在同一时刻实例化它们的合取，从而区分 Chord（共时性必要）和 Arpeggio（序列即可）两个意识假设，论证纯序列硬件上的软件意识在 Chord 假设下不可能。

**[A New Strategy for Verifying Reach-Avoid Specifications in Neural Feedback Systems](others/a_new_strategy_for_verifying_reach-avoid_specifications_in_neural_feedback_syste.md)**

:   提出 FaBRe（Forward and Backward Reachability）策略，开发神经反馈系统后向可达集的过近似和欠近似算法（Golden Section Search / Iterative Convex Hull / Largest Empty Box），并将其与现有前向可达性分析结合，构建统一的 reach-avoid 验证框架。

**[A Phase Transition for Opinion Dynamics with Competing Biases](others/a_phase_transition_for_opinion_dynamics_with_competing_biase.md)**

:   在有向随机图上建模两种对立力量（外部颠覆性偏差 vs 个体顽固性）对二元观点传播的影响，证明系统存在尖锐相变：偏差超过临界阈值 $p_c$ 时群体快速达成新共识，低于阈值则长期处于亚稳极化状态，且临界点仅由度序列的两个简单统计量决定。

**[A Switching Framework for Online Interval Scheduling with Predictions](others/a_switching_framework_for_online_interval_scheduling_with_pr.md)**

:   针对不可撤销的在线区间调度问题，提出 SemiTrust-and-Switch 框架和 SmoothMerge 随机算法，通过在信任预测和经典贪心算法之间切换/融合，在预测准确时趋近最优（一致性），预测错误时性能优雅退化（鲁棒性和平滑性），并证明了该框架在特定实例上的紧性。

**[A Topological Rewriting of Tarski's Mereogeometry](others/a_topological_rewriting_of_tarskis_mereogeometry.md)**

:   本文在Coq定理证明器中，基于λ-MM库（Leśniewski部分整体论的类型论实现），将Tarski的实体几何（geometry of solids）重新用拓扑学语言改写：先证明部分整体论的类（m-class）对应正则开集从而构成拓扑空间，再证明Tarski几何形成该拓扑的子空间并满足Hausdorff（T₂）分离性质，从而为定性空间推理提供了一个统一的、机器验证的部分整体论-几何-拓扑理论。

**[Adaptive Evidential Learning for Temporal-Semantic Robustness in Moment Retrieval](others/adaptive_evidential_learning_for_temporal-semantic_robustnes.md)**

:   提出 DEMR 框架，将深度证据回归（DER）引入视频时刻检索任务，通过 Reflective Flipped Fusion 模块缓解模态不平衡、通过 Geom-regularizer 修复原始 DER 中不确定性估计的反直觉偏差，在标准和去偏数据集上均取得了显著提升。

**[Agent-SAMA: State-Aware Mobile Assistant](others/agent-sama_state-aware_mobile_assistant.md)**

:   提出Agent-SAMA，首次将有限状态机（FSM）引入移动端GUI Agent，将UI屏幕建模为状态、用户操作建模为转移，通过四个专门化Agent协作实现状态感知的任务规划、执行验证和错误恢复，在跨App基准上成功率提升最高12%、恢复率提升13.8%。

**[Align When They Want, Complement When They Need! Human-Centered Ensembles for Adaptive Human-AI Collaboration](others/align_when_they_want_complement_when_they_need_human-centere.md)**

:   揭示了人机协作中"互补性"（complementarity）与"对齐性"（alignment）之间存在根本性权衡——单一模型无法同时优化二者，提出自适应AI集成框架，通过Rational Routing Shortcut（RRS）机制在对齐模型和互补模型之间动态切换，团队准确率较标准AI提升最高9%。

**[AMS-IO-Bench and AMS-IO-Agent: Benchmarking and Structured Reasoning for Analog and Mixed-Signal Integrated Circuit Input/Output Design](others/ams-io-bench_and_ams-io-agent_benchmarking_and_structured_re.md)**

:   提出AMS-IO-Agent，一个基于LLM的领域专用智能体，通过结构化意图图(Intent Graph)和领域知识库将自然语言设计意图转化为可生产的模拟混合信号IC I/O环设计，配套提出首个AMS I/O环自动化基准AMS-IO-Bench，在28nm CMOS流片中验证了智能体生成的I/O环可直接用于实际芯片制造。

**[An Epistemic Perspective on Agent Awareness](others/an_epistemic_perspective_on_agent_awareness.md)**

:   本文首次将 agent awareness（智能体感知/意识）视为一种知识形式，区分了 de re（关于物理对象的）和 de dicto（关于概念/描述的）两种感知模态，并基于 2D 语义学提出了一个可靠且完备的逻辑系统来刻画这两种模态与标准"事实知识"模态之间的相互作用。

**[Approximation Algorithm for Constrained k-Center Clustering: A Local Search Approach](others/approximation_algorithm_for_constrained_k-center_clustering_.md)**

:   研究带 cannot-link (CL) 和 must-link (ML) 实例级约束的 k-center 聚类问题，提出基于支配匹配集（dominating matching set, DMS）转化的局部搜索框架，在不相交 CL 集条件下首次通过局部搜索达到最优近似比 2，解决了该领域一个开放问题。

**[Area-Optimal Control Strategies for Heterogeneous Multi-Agent Pursuit](others/area-optimal_control_strategies_for_heterogeneous_multi-agen.md)**

:   研究异构速度下多追逐者-单逃避者的追逃博弈——定义逃避者安全可达集为所有追逐者-逃避者对的 Apollonius 圆的交集，将捕获策略建模为追逐者最小化/逃避者最大化该交集面积的零和博弈，推导出闭式瞬时最优航向控制律，仿真验证追逐者可系统性缩小安全区域实现保证捕获。

**[Bilevel MCTS for Amortized O(1) Node Selection in Classical Planning](others/bilevel_mcts_for_amortized_o1_node_selection_in_classical_planning.md)**

:   提出双层MCTS（Bilevel MCTS），在MCTS选中的叶节点处运行深度比例预算的最优优先搜索，将节点选择均摊复杂度从 $O(\log N)$ 降至 $O(1)$，辅以树崩塌（Tree Collapsing）减少动作选择步数，最终整合为 Nεbula 规划器，在IPC2018/2023基准上以192.2/230.6解题数（5min/30min）超越LAMA、DecStar、NOLAN、SM-Type-LAMA等全部SOTA。

**[Extreme Value Monte Carlo Tree Search for Classical Planning](others/extreme_value_monte_carlo_tree_search_for_classical_planning.md)**

:   利用 Peaks-Over-Threshold 极值理论（POT EVT）为经典规划中 MCTS 的 Full Bellman Backup 提供统计理论基础，提出 UCB1-Uniform bandit 算法，用均匀分布（Generalized Pareto 的特例）的 MLE 估计指导动作选择，在 Pyperplan 上以 $10^4$ 节点预算超越 GBFS 67.8 个实例、超越 Softmin-Type(h) 33.2 个实例。

**[MeshA*: Efficient Path Planning With Motion Primitives](others/mesha_efficient_path_planning_with_motion_primitives.md)**

:   提出 MeshA* 算法，将 lattice-based 路径规划从"在运动基元层面搜索"转变为"在网格单元层面搜索并同时拟合基元序列"，通过定义"扩展网格单元"（extended cell）新搜索空间，在保证完备性和最优性的同时，实现相比标准 LBA* 1.5x-2x 的运行时加速。

**[Symbolic Planning and Multi-Agent Path Finding in Extremely Dense Environments with Unassigned Agents](others/symbolic_planning_and_multi-agent_path_finding_in_extremely_dense_environments_w.md)**

:   提出 Block Rearrangement Problem (BRaP) 形式化定义，并设计五种基于配置空间搜索、PDDL 符号规划和 MAPF 的求解算法，其中 BR-LaCAM 在最大 80×80 的极端密集网格上达到 92% 成功率和毫秒级求解速度。

**[TaylorPODA: A Taylor Expansion-Based Method to Improve Post-Hoc Attributions for Opaque Models](others/taylorpoda_a_taylor_expansion-based_method_to_improve_post-hoc_attributions_for_.md)**

:   在Taylor展开框架下提出精确性(precision)、联合性(federation)、零偏差(zero-discrepancy)三个公设规范特征归因，并引入自适应属性(adaptation)通过AUP目标优化交互效应的分配权重，成为唯一同时满足所有公设和属性的事后模型无关归因方法。
