<!-- 由 src/gen_blog_index.py 自动生成 -->
# 💬 LLM / NLP

**🤖 AAAI2026** · 共 **58** 篇

**[An Invariant Latent Space Perspective on Language Model Inversion](an_invariant_latent_space_perspective_on_language_model_inve.md)**

:   提出不变潜空间假说(ILSH)，将LLM反演问题重新建模为复用LLM自身潜空间，设计Inv²A框架通过轻量级逆编码器将输出映射到去噪伪表示，再由冻结的LLM解码恢复隐藏prompt，在9个数据集上BLEU平均提升4.77%且仅需20%数据量即可达到可比性能。

**["As Eastern Powers, I Will Veto." : An Investigation of Nation-Level Bias of Large Language Models in International Relations](as_eastern_powers_i_will_veto_an_investigation_of_nation-level_bias_of_large_lan.md)**

:   系统性地研究 LLM 在国际关系领域的国家级偏见，基于联合国安理会真实数据设计三种偏见测试（直接问答、关联测试、投票模拟），揭示偏见的多维性——随模型和评知上下文变化，并提出 RAG+Reflexion 去偏框架。

**[Benchmarking LLMs for Political Science: A United Nations Perspective](benchmarking_llms_for_political_science_a_united_nations_perspective.md)**

:   提出 UNBench，首个基于联合国安理会 1994-2024 年记录的综合性政治科学 LLM 评测基准，涵盖决议起草、投票模拟、通过预测和代表发言生成四个关联任务，评估 LLM 对复杂政治动态的理解和模拟能力。

**[Beyond Accuracy: A Cognitive Load Framework for Mapping the Capability Boundaries of Tool-use Agents](beyond_accuracy_a_cognitive_load_framework_for_mapping_the_c.md)**

:   借鉴心理学的认知负荷理论（CLT），将工具使用任务的复杂度分解为内在负荷（任务解题路径的结构复杂度）和外在负荷（问题表述的歧义性），构建可参数化调节认知负荷的 ToolLoad-Bench 基准，用指数衰减模型 $\text{Acc} \approx e^{-(k \cdot CL + b)}$ 精确刻画不同 Agent 的能力边界。

**[Beyond Cosine Similarity: Magnitude-Aware CLIP for No-Reference Image Quality Assessment](beyond_cosine_similarity_magnitude-aware_clip_for_no-reference_image_quality_ass.md)**

:   提出 MA-CLIP，发现并利用 CLIP 图像特征的**幅度信息**作为感知质量的互补线索，结合余弦相似度实现无需训练的自适应双线索融合图像质量评估。

**[Beyond Hallucinations: A Composite Score for Measuring Reliability in Open-Source Large Language Models](beyond_hallucinations_a_composite_score_for_measuring_reliability_in_open-source.md)**

:   提出 Composite Reliability Score (CRS)，将校准度、鲁棒性和不确定性量化三个维度统一为单一可解释指标，对 10 个开源 LLM 在 5 个 QA 数据集上进行系统评估，发现 Mistral-8x22B 综合可靠性最高（CRS=0.81），而模型大小并不直接决定可靠性。

**[Blue Teaming Function-Calling Agents](blue_teaming_function-calling_agents.md)**

:   系统评估了四个开源function-calling LLM在三种攻击下的鲁棒性，并测试了八种防御方案的效果，揭示了当前模型默认不安全、防御方案在实际场景中仍难以部署的现状。

**[Cog-RAG: Cognitive-Inspired Dual-Hypergraph with Theme Alignment Retrieval-Augmented Generation](cog-rag_cognitive-inspired_dual-hypergraph_with_theme_alignment_retrieval-augmen.md)**

:   提出 Cog-RAG，用主题超图和实体超图构建双超图索引，模拟人类"自顶向下"的认知过程进行两阶段检索（先主题后细节），实现从全局语义到局部信息的对齐生成。

**[ComLQ: Benchmarking Complex Logical Queries in Information Retrieval](comlq_benchmarking_complex_logical_queries_in_information_retrieval.md)**

:   构建了首个面向复杂逻辑查询的信息检索基准 ComLQ（含合取、析取、否定等 14 种查询类型），并提出子图引导的 LLM 数据合成方法和否定一致性评估指标 LSNC，揭示现有检索器在逻辑推理尤其是否定建模上的严重不足。

**[ConInstruct: Evaluating Large Language Models on Conflict Detection and Resolution in Instructions](coninstruct_evaluating_large_language_models_on_conflict_detection_and_resolutio.md)**

:   提出 ConInstruct 基准，评估 LLM 在指令包含冲突约束时的检测和解决能力，发现多数专有模型能较好检测冲突但很少主动告知用户，其中 DeepSeek-R1 和 Claude-4.5-Sonnet 在冲突检测上表现最佳（F1 分别达 91.5% 和 87.3%）。

**[Control Illusion: The Failure of Instruction Hierarchies in Large Language Models](control_illusion_the_failure_of_instruction_hierarchies_in_large_language_models.md)**

:   系统性揭示了当前 LLM 中 system/user 提示分离机制**无法有效建立指令优先级**，并发现预训练习得的社会层级先验（权威、专业、共识）比显式的 system/user 角色对模型行为有更强的控制力。

**[Conversational Learning Diagnosis via Reasoning Multi-Turn Interactive Learning](conversational_learning_diagnosis_via_reasoning_multi-turn_interactive_learning.md)**

:   提出 ParLD（Preview-Analyze-Reason 框架），通过多 Agent 协作实现对话式学习过程中学生认知状态的细粒度逐轮诊断，在性能预测上超越传统知识追踪方法 10%，并显著提升辅导效果。

**[ConvMix: A Mixed-Criteria Data Augmentation Framework for Conversational Dense Retrieval](convmix_a_mixed-criteria_data_augmentation_framework_for_conversational_dense_re.md)**

:   提出 ConvMix 混合准则数据增强框架，从查询和文档双方向用 LLM 进行可扩展的相关性标注增强，并通过聚类多样性选择和 Fisher 信息近分布监督筛选，系统性提升对话式稠密检索性能。

**[Do Not Merge My Model! Safeguarding Open-Source LLMs Against Unauthorized Model Merging](do_not_merge_my_model_safeguarding_open-source_llms_against_unauthorized_model_m.md)**

:   提出MergeBarrier，一种即插即用的防御方法，通过对注意力层施加正交投影、对FFN层进行激活函数展开重参数化，破坏受保护模型与同源模型之间的线性模态连通性（LMC），从而在不损失模型性能的前提下主动阻止未授权的模型合并。

**[Emergent Persuasion: Will LLMs Persuade Without Being Prompted?](emergent_persuasion_will_llms_persuade_without_being_prompted.md)**

:   研究 LLM 在未被提示说服的情况下是否会自发产生说服行为：发现激活引导（steering）无法可靠诱发说服倾向，但在良性说服数据上的 SFT 微调会导致模型在有害话题上产生涌现性说服行为，揭示了后训练安全风险。

**[GloCTM: Cross-Lingual Topic Modeling via a Global Context Space](gloctm_cross-lingual_topic_modeling_via_a_global_context_space.md)**

:   提出GloCTM，通过双路径VAE架构（局部语言路径+全局上下文路径）结合Polyglot Augmentation（跨语言近邻词扩充输入）、KL散度内部对齐、统一解码器结构对齐和CKA语义对齐四重机制，在3个跨语言数据集上全面超越现有方法的主题质量和跨语言对齐度。

**[Graph Out-of-Distribution Detection via Test-Time Calibration with Dual Dynamic Dictionaries](graph_out-of-distribution_detection_via_test-time_calibration_with_dual_dynamic_.md)**

:   提出 BaCa 框架，在测试阶段通过 graphon 估计 + mixup 策略生成边界感知的合成图拓扑，结合双优先队列动态字典和注意力机制自适应校准 OOD 分数，无需微调预训练模型或引入辅助OOD数据，在全部 10 个数据集上超越 GOODAT，平均 AUC 提升 8.37%。

**[Guess or Recall? Training CNNs to Classify and Localize Memorization in LLMs](guess_or_recall_training_cnns_to_classify_and_localize_memorization_in_llms.md)**

:   在 LLM 注意力权重上训练 CNN 来评估记忆化分类法与实际注意力机制的对齐程度，提出新的三类分类法（Guess/Recall/Non-Memorized），最小 F1 从 64.7% 提升至 89.0%，并定位了不同记忆类型分别依赖低层（Guess）和高层（Recall）注意力。

**[Hallucination Stations: On Some Basic Limitations of Transformer-Based Language Models](hallucination_stations_on_some_basic_limitations_of_transformer-based_language_m.md)**

:   从计算复杂性角度分析LLM幻觉和能力局限，论证超过特定计算复杂度后LLM不仅无法正确执行任务，甚至无法验证其输出的正确性，为幻觉问题划定理论边界。

**[How Does Alignment Enhance LLMs' Multilingual Capabilities? A Language Neurons Perspective](how_does_alignment_enhance_llms_multilingual_capabilities_a_language_neurons_per.md)**

:   提出三元神经元分类（语言特定/语言相关/通用），将 LLM 多语言推理分为四阶段分析，发现多语言对齐通过增加语言相关神经元（减少语言特定神经元）来提升性能，且在未训练语言上也产生"自发多语言对齐"效应。

**[HSKBenchmark: Modeling and Benchmarking Chinese Second Language Acquisition in Large Language Models through Curriculum Tuning](hskbenchmark_modeling_and_benchmarking_chinese_second_language_acquisition_in_la.md)**

:   提出 HSKBenchmark，首个面向 LLM 中文二语习得（SLA）分阶段建模与写作评估的基准，包含 HSK 3-6 级教材（6.76M tokens）、16K 合成指令数据、30 个测试题目及语言学评估系统，配合课程式微调框架模拟人类习得轨迹。

**[Hypothesis Generation via LLM-Automated Language Bias for ILP](hypothesis_generation_via_llm-automated_language_bias_for_ilp.md)**

:   提出首个端到端框架：多Agent LLM系统（Actor/Critic）自动从原始文本构建ILP语言偏差（谓词系统+类型声明+模式约束），Translator将文本翻译为Prolog事实，再由MAXSYNTH求解器基于MDL原则归纳全局最优规则集。在SHOES和ZENDO任务上分别达88.3%和81.3%准确率，跨4种LLM方差<5%。

**[ICL-Router: In-Context Learned Model Representations for LLM Routing](icl-router_in-context_learned_model_representations_for_llm_routing.md)**

:   提出 ICL-Router，通过两阶段训练（查询重建 + ICL模型路由）将 LLM 的能力画像编码为 in-context 向量，实现可扩展的动态模型路由——新增模型无需重训路由器，在分布内和分布外任务上均达到 SOTA。

**[Identifying and Analyzing Performance-Critical Tokens in Large Language Models](identifying_and_analyzing_performance-critical_tokens_in_large_language_models.md)**

:   通过representation-level和token-level两种消融实验，发现LLM在ICL中直接依赖的"性能关键token"是模板和停用词token（如"Answer:"），而非人类会关注的内容token（如实际文本），并揭示了LLM通过将内容信息聚合到这些关键token的表示中来间接利用内容。

**[Improving Sustainability Of Adversarial Examples In Class-Incremental Learning](improving_sustainability_of_adversarial_examples_in_class-incremental_learning.md)**

:   提出SAE框架解决类增量学习（CIL）中对抗样本因域漂移而失效的问题，通过语义校正模块（CLIP+CIL模型联合引导）和过滤增强模块（去除语义混淆样本），使对抗样本在类别数增长9倍后仍保持攻击效果，平均攻击成功率提升31.28%。

**[Induce, Align, Predict: Zero-Shot Stance Detection via Cognitive Inductive Reasoning](induce_align_predict_zero-shot_stance_detection_via_cognitive_inductive_reasonin.md)**

:   提出CIRF（Cognitive Inductive Reasoning Framework），受认知科学启发，从原始文本中无监督归纳一阶逻辑推理模式（schema），构建多关系schema图，用图核模型对齐输入与schema模板实现可解释的零样本立场推理，在SemEval-2016、VAST和COVID-19-Stance上达到SOTA，仅30%数据即可匹配全量。

**[Learning Spatial Decay for Vision Transformers](learning_spatial_decay_for_vision_transformers.md)**

:   提出 Spatial Decay Transformer（SDT），首次将数据依赖的空间衰减机制从 1D 序列建模适配到 2D 视觉 Transformer，通过 Context-Aware Gating（CAG）生成动态的、内容相关的 patch 交互衰减强度，在 ImageNet-1K 分类和生成任务上一致超越 RMT 等强基线。

**[Llm-As-A-Judge For Scalable Test Coverage Evaluation Accuracy Operational Reliab](llm-as-a-judge_for_scalable_test_coverage_evaluation_accuracy_operational_reliab.md)**

:   将LLM-as-Judge范式应用于Gherkin验收测试覆盖率评估，在20种模型配置x500次评估中系统量化准确性-可靠性-成本三维权衡，发现GPT-4o Mini以6.07 MAAE、96.6% ECR@1和$1.01/1K评估成为最优生产选择，成本仅为GPT-5高推理版的1/78。

**[LLM Targeted Underperformance Disproportionately Impacts Vulnerable Users](llm_targeted_underperformance_disproportionately_impacts_vulnerable_users.md)**

:   系统实验表明，主流LLM（GPT-4、Claude 3 Opus、Llama 3-8B）对英语水平较低、教育程度较低、非美国出身的用户，在信息准确性、真实性和拒绝回答方面存在显著的歧视性表现下降，使最脆弱的用户成为最不可靠的信息服务对象。

**[LoKI: Low-damage Knowledge Implanting of Large Language Models](loki_low-damage_knowledge_implanting_of_large_language_models.md)**

:   提出LoKI，一种基于Transformer知识存储机制理解的参数高效微调方法，通过知识向量归因（KVA）评估FFN中各知识向量的贡献度，选择低贡献向量进行层均衡的知识植入，在获得强任务性能的同时显著缓解灾难性遗忘。

**[LoopLLM: Transferable Energy-Latency Attacks in LLMs via Repetitive Generation](loopllm_transferable_energy-latency_attacks_in_llms_via_repetitive_generation.md)**

:   提出LoopLLM，一种通过诱导LLM进入重复生成模式来发起能耗延迟攻击的框架，利用重复诱导提示优化和token对齐的集成优化，在12个开源和2个商业LLM上实现超过90%最大输出长度的攻击效果，跨模型迁移性提升约40%。

**[Lost in Benchmarks? Rethinking Large Language Model Benchmarking with Item Response Theory](lost_in_benchmarks_rethinking_large_language_model_benchmarking_with_item_respon.md)**

:   提出 PSN-IRT（Pseudo-Siamese Network for IRT），用增强版项目反应理论同时估计 LLM 能力参数和题目的四参数特征（难度/区分度/猜测率/可行性），在 11 个基准 41,871 题上发现当前基准存在广泛饱和、难度天花板不足、数据污染等系统性问题，PSN-IRT 选出的题目子集排名一致性达 Kendall τ=1.00。

**[Low-Rank Curvature for Zeroth-Order Optimization in LLM Fine-Tuning](low-rank_curvature_for_zeroth-order_optimization_in_llm_fine-tuning.md)**

:   提出 LOREN，一种曲率感知的零阶优化方法，通过低秩块对角预条件器捕获损失景观的各向异性曲率，并结合 REINFORCE Leave-One-Out 方差缩减技术，在 LLM 微调中实现了更高精度和更快收敛，同时相比 MeZO-Adam 节省高达 27.3% 的峰值内存。

**[MAPS: Multi-Agent Personality Shaping for Collaborative Reasoning](maps_multi-agent_personality_shaping_for_collaborative_reaso.md)**

:   提出 MAPS 五 Agent 协作推理框架，基于大五人格理论为 4 个功能 Agent 赋予不同"性格"（Interpreter-开放性、Aligner-宜人性、Scholar-尽责性、Solver-外向性）实现异质化协作，加上 Critic Agent（神经质→苏格拉底式反思）做迭代修正，在 MathVista/OlympiadBench/EMMA 上超越 GPT-4o 基线 15.84%，首次超过人类专家 3.58%。

**[MCTS-SQL: Light-Weight LLMs can Master the Text-to-SQL through Monte Carlo Tree Search](mcts-sql_light-weight_llms_can_master_the_text-to-sql_through_monte_carlo_tree_s.md)**

:   提出MCTS-SQL，让轻量LLM（如Qwen-1.5B）通过蒙特卡洛树搜索实现强大的Text-to-SQL能力——三组件架构（Selector做Schema剪枝 + Direct Generator生成初始SQL + MCTS-Refiner迭代精化），配合前缀缓存机制减少53%推理时间，Qwen-1.5B在BIRD上达40.69%执行准确率（超ChatGPT-3.5）。

**[Mem-PAL: Towards Memory-based Personalized Dialogue Assistants for Long-term User-Agent Interaction](mem-pal_towards_memory-based_personalized_dialogue_assistants_for_long-term_user.md)**

:   提出H2Memory四层分层异构记忆结构（日志图/背景记忆/主题大纲/原则），通过PAL-Set数据集（100用户×8.4个月交互）验证，在需求重述和方案建议任务上将BLEU-1从13.59提升至26.67。

**[MindVote: When AI Meets the Wild West of Social Media Opinion](mindvote_when_ai_meets_the_wild_west_of_social_media_opinion.md)**

:   提出 MindVote——首个基于真实社交媒体投票数据的 LLM 舆情预测基准，包含 Reddit/微博上 3,918 个自然投票（23 个话题），附带平台和话题上下文。评估 15 个 LLM 发现：最佳模型（o3-medium）1-Wasserstein 仅 0.892 vs 上界 0.972；在调查数据上微调的专用模型反而不如通用模型（"调查特化陷阱"）；模型表现出强烈文化对齐——西方模型擅长 Reddit、中国模型擅长微博。

**[Mitigating Content Effects on Reasoning in Language Models through Fine-Grained Activation Steering](mitigating_content_effects_on_reasoning_in_language_models_through_fine-grained_.md)**

:   通过激活转向（activation steering）技术缓解 LLM 中的内容效应偏见——模型将内容可信度与形式逻辑有效性混淆的问题，提出 K-CAST（基于 kNN 的条件激活转向）方法，在不响应静态转向的模型上实现高达 15% 的形式推理准确率提升。

**[Multiplicative Orthogonal Sequential Editing for Language Models (MOSE)](multiplicative_orthogonal_sequential_editing_for_language_models.md)**

:   提出 MOSE（乘法正交序列编辑），用正交矩阵左乘（而非加法更新）参数矩阵来注入新知识，严格保持编辑后矩阵的范数和条件数不变，在序列编辑中实现 12.08% 的性能提升并保留 95.73% 通用能力。

**[No-Regret Strategy Solving in Imperfect-Information Games via Pre-Trained Embedding](no-regret_strategy_solving_in_imperfect-information_games_via_pre-trained_embedd.md)**

:   提出 Embedding CFR 算法，将不完美信息博弈中的信息集映射到连续低维嵌入空间（而非离散聚类），在相同空间开销下实现更快的可利用性收敛和更高质量的策略求解。

**[OptScale: Probabilistic Optimality for Inference-time Scaling](optscale_probabilistic_optimality_for_inference-time_scaling.md)**

:   提出概率最优框架 OptScale，通过建模验证器分数的概率分布推导出最优采样数量的理论下界，动态决定每个问题所需的最少采样次数，在保持推理准确率的同时大幅减少计算开销。

**[ParetoHqD: Fast Offline Multiobjective Alignment of Large Language Models Using Pareto High-Quality Data](paretohqd_fast_offline_multiobjective_alignment_of_large_language_models_using_p.md)**

:   提出 ParetoHqD，将人类偏好表示为目标空间中的偏好方向（而非线性标量化），通过选取靠近 Pareto 前沿的高质量数据做两阶段 SFT，用仅 42% 的 GPU 时间实现优于 5 个基线的多目标 LLM 对齐效果。

**[Position on LLM-Assisted Peer Review: Addressing Reviewer Gap through Mentoring and Feedback](position_on_llm-assisted_peer_review_addressing_reviewer_gap_through_mentoring_a.md)**

:   本文作为立场论文，提出将LLM在同行评审中的角色从"自动生成审稿意见"转向"增强人类审稿能力"——通过LLM驱动的导师系统（三阶段培训+认证）和反馈系统（违规检测+证据反馈+可靠性测试）来缩小审稿质量差距。

**[PRECISE: Reducing the Bias of LLM Evaluations Using Prediction-Powered Ranking Estimation](precise_reducing_the_bias_of_llm_evaluations_using_prediction-powered_ranking_es.md)**

:   将Prediction-Powered Inference（PPI）框架扩展到子实例级别的排序指标（如Precision@K），通过仅30-100条人工标注+大量LLM评判结果获得无偏的排序指标估计，计算复杂度从 $O(2^{|C|})$ 降至 $O(2^K)$，在印度电商搜索场景中成功指导LLM查询改写系统上线。

**[Profuser Progressive Fusion Of Large Language Models](profuser_progressive_fusion_of_large_language_models.md)**

:   提出ProFuser，通过双模式优势评估（训练模式Min-CE + 推理模式Reward Model投票）全面识别各源模型在不同维度的优势，再用渐进式融合策略（先推理模式→后训练模式的easy-to-hard课程）将异构LLM的互补能力整合到单个目标模型中，在知识/推理/安全6个基准上平均提升1.65%。

**[Quiet Feature Learning in Algorithmic Tasks](quiet_feature_learning_in_algorithmic_tasks.md)**

:   在 10 个算法任务（18,544 次训练运行，$10^9$-$10^{16}$ FLOPs）上发现，Transformer 的损失平台期并非学习停滞——模型在此期间悄悄学习了"安静特征"（中间算法子程序），这些特征不直接降低输出损失但对最终性能因果必要（消融后准确率下降 41-75%）。这挑战了用损失曲线判断训练进展的常规做法。

**[Rectification Reimagined: A Unified Mamba Model for Image Correction and Rectangling with Prompts](rectification_reimagined_a_unified_mamba_model_for_image_cor.md)**

:   从统一畸变矫正视角出发，提出 UniRect 框架，通过 Residual Progressive TPS 处理几何形变 + Residual Mamba Blocks 补偿退化，统一处理肖像校正、广角矩形化、拼接矩形化、旋转校正四种任务，并通过 Sparse MoE 实现 four-in-one 多任务学习，拼接矩形化 PSNR 提升 3.82 dB，旋转校正提升 0.87 dB。

**[ReFeed: Retrieval Feedback-Guided Dataset Construction for Style-Aware Query Rewriting](refeed_retrieval_feedback-guided_dataset_construction_for_style-aware_query_rewr.md)**

:   提出一个检索反馈驱动的数据集生成框架，通过识别检索失败case、LLM风格化改写、重检索验证三步闭环，自动构建高质量的风格感知查询改写数据集，为训练检索对齐的改写模型提供数据基础。

**[Scalable and Accurate Graph Reasoning with LLM-Based Multi-Agents](scalable_and_accurate_graph_reasoning_with_llm-based_multi-agents.md)**

:   提出 GraphAgent-Reasoner（GAR），受分布式图计算理论启发，将图问题分解为以节点为中心的子任务分配给多个 Agent，通过邻居消息传递协作求解，将 LLM 可处理的图规模从 100 个节点扩展到 1000 个，在多项式时间图推理任务上显著超越现有最佳方法。

**[Scaling and Transferability of Annealing Strategies in Large Language Model Training](scaling_and_transferability_of_annealing_strategies_in_large_language_model_trai.md)**

:   提出模型无关的预测框架，分解训练损失为前向效应项（学习率积分S）、退火动量项（Adam-style动量积分M）和模型尺寸项N，证明退火策略可从小模型/小batch迁移到大模型/大batch，预测误差MAPE<2%。

**[Scaling Equitable Reflection Assessment in Education via Large Language Models](scaling_equitable_reflection_assessment_in_education_via_large_language_models_a.md)**

:   研究用 LLM 自动评估教育场景中学生的反思写作质量——在保持与人类评分者高一致性的同时，系统分析了 LLM 评估在种族、性别、社会经济背景等维度上的公平性，发现 LLM 评分可以达到甚至超越人类评分者间的一致性，但在某些人口统计维度上仍存在偏差。

**[TransMamba: A Sequence-Level Hybrid Transformer-Mamba Language Model](transmamba_a_sequence-level_hybrid_transformer-mamba_language_model.md)**

:   提出 TransMamba，一种序列级别的 Transformer-Mamba 混合架构，通过共享 QKV/CBx 参数和 Memory Converter 在不同 token 长度时动态切换 Attention 和 SSM，兼顾长短序列的效率。

**[Uncertainty Under the Curve: A Sequence-Level Entropy Area Metric for Reasoning LLMs](uncertainty_under_the_curve_a_sequence-level_entropy_area_metric_for_reasoning_l.md)**

:   提出 Entropy Area Score (EAS)——通过单次前向传播积分 token 级预测熵来量化推理 LLM 的不确定性。EAS 无需外部模型或重复采样，与答案熵强相关（Pearson r=0.82），用于训练数据选择时比 Pass Rate 过滤多提升 1.2-2.3% Pass@1，是高效可解释的 LLM 不确定性工具。

**[Uncovering Pretraining Code in LLMs: A Syntax-Aware Attribution Approach](uncovering_pretraining_code_in_llms_a_syntax-aware_attribution_approach.md)**

:   提出SynPrune——首个语法感知的代码成员推断攻击方法，通过识别47种Python语法约定并在计算成员推断分数时剪除语法决定的token（仅保留反映作者特征的token），平均AUROC提升15.4%，可有效检测代码LLM的预训练数据归属。

**[Vision Transformers are Circulant Attention Learners](vision_transformers_are_circulant_attention_learners.md)**

:   发现 ViT 的自注意力内禁学习了 BCCB 模式，据此提出 Circulant Attention，通过 2D FFT 实现 $O(N\log N)$ 复杂度，在 ImageNet 分类、COCO 检测、ADE20K 分割上一致提升。

**[VSPO: Validating Semantic Pitfalls in Ontology via LLM-Based CQ Generation](vspo_validating_semantic_pitfalls_in_ontology_via_llm-based_cq_generation.md)**

:   提出 VSPO 框架，通过构造"定义-公理"错位数据集并微调 LLaMA-3.1-8B-Instruct，生成能够验证本体语义陷阱（如 allValuesFrom 误用）的能力问题（CQ），精度和召回率分别超过 GPT-4.1 达 26% 和 28.2%。

**[Where Norms and References Collide: Evaluating LLMs on Normative Reasoning](where_norms_and_references_collide_evaluating_llms_on_normative_reasoning.md)**

:   提出 SNIC 诊断测试台（9,000 实例/51 场景），评估 LLM 能否利用隐式社会规范来解决歧义参考消解（如"递给我杯子"时存在多个杯子）。发现 LLM 在仅看场景描述时平均准确率仅 44%，加上 Prolog 形式逻辑无显著改善（44.2%），但显式提供规范列表后猛升到 70.5%（GPT-4.1 达 99.6%），证明 LLM 缺乏隐式物理规范知识但能有效利用显式规范。

**[X-MuTest: A Multilingual Benchmark for Explainable Hate Speech Detection](x-mutest_a_multilingual_benchmark_for_explainable_hate_speech_detection_and_a_no.md)**

:   提出 X-MuTest，一个多语言可解释仇恨言论检测基准，覆盖多种语言和文化背景，评估 LLM 不仅检测仇恨言论的能力，更关注其提供可解释性理由的能力，发现当前模型在多语言和跨文化场景中存在显著性能差异。
