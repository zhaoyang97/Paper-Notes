<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📂 其他

**💬 ACL2025** · 共 **125** 篇

**[Towards Robust ESG Analysis Against Greenwashing Risks: A3CG](a3cg_esg_greenwashing.md)**

:   提出 A3CG 数据集和方面-行动分析任务（从可持续性声明中提取方面及其行动类型：已实施/计划中/不确定），通过跨类别泛化设置评估 NLP 方法抵御漂绿风险的鲁棒性，发现监督学习（GRACE F1=47.51）优于 LLM（Claude 3.5 F1=42.03）但泛化效率更差。

**[A Conformal Risk Control Framework for Granular Word Assessment and Uncertainty Calibration of CLIPScore Quality Estimates](a_conformal_risk_control_framework_for_granular_word_assessment_and_uncertainty_.md)**

:   提出基于保形风险控制（Conformal Risk Control）框架校准 CLIPScore 的方法——通过对 CLIP 视觉/文本编码器的注意力掩码采样生成 CLIPScore 分布，然后利用保形风险控制 (1) 检测图像描述中的干扰词（foil words），(2) 生成校准置信区间，在 FOIL-it/FOIL-nocaps/Rich-HF 基准上以简单方法达到与复杂专用方法相当的干扰词检测性能，同时提供形式化风险保证。

**[A Dual-Perspective NLG Meta-Evaluation Framework with Automatic Benchmark and Better Interpretability](a_dual-perspective_nlg_meta-evaluation_framework_with_automatic_benchmark_and_be.md)**

:   揭示传统 NLG 元评估的局限（人工评分平均聚合不合理、相关系数选择模糊），提出双视角元评估框架：全局视角（序数分类，评估粗粒度评级能力）+ 局部视角（相邻成对比较，评估细粒度区分能力），并引入基于可控错误注入的自动基准构建方法，在 16 个 LLM 上验证不同模型在两个视角上的能力分布差异显著。

**[GKI-ICD: A General Knowledge Injection Framework for ICD Coding](a_general_knowledge_injection_framework_for_icd_coding.md)**

:   提出 GKI-ICD，首个无需额外专用网络模块即可同时注入三种 ICD 编码知识（描述、同义词、层级结构）的通用框架——通过合成知识引导文本（Guideline Synthesis）+ 多任务学习实现知识注入，在 MIMIC-III 和 MIMIC-III-50 上大多数指标达到 SOTA。

**[Barec: A Large and Balanced Corpus for Fine-grained Arabic Readability Assessment](a_large_and_balanced_corpus_for_fine-grained_arabic_readability_assessment.md)**

:   构建 Barec（Balanced Arabic Readability Evaluation Corpus）——首个大规模细粒度阿拉伯语可读性评估语料库，包含 69K+ 句子（100万+词），覆盖 19 个可读性等级（从幼儿园到研究生），由 6 名专业阿拉伯语教育者标注，并在多种分级粒度（19/7/5/3级）上基准测试自动可读性评估模型。

**[A Little Human Data Goes A Long Way](a_little_human_data_goes_a_long_way.md)**

:   首次系统研究合成数据在事实验证和问答中能否替代人工标注——替换 90% 仅轻微下降，但替换最后 10% 严重退化；仅 125 条人工数据可显著提升纯合成模型，等效增益需 10 倍以上合成数据。

**[MisMatched: A Benchmark for Scientific Natural Language Inference](a_mismatched_benchmark_for_scientific_natural_language_inference.md)**

:   引入 MisMatched——首个覆盖非 CS 领域（心理学、工程、公共卫生）的科学 NLI 评估基准，2700 对人工标注句子对，最佳基线 Macro F1 仅 78.17%，且发现训练时加入隐式关系句子对可提升性能。

**[A Practical Approach for Building Production-Grade Conversational Agents with Workflow Graphs](a_practical_approach_for_building_production-grade_conversational_agents_with_wo.md)**

:   提出基于有向无环图（DAG）工作流的混合对话 Agent 框架——每个图节点有独立系统提示/工具/执行规则以处理特定场景约束。结合原型 Agent 数据收集、状态感知的响应掩码微调策略，在 Kakao 移动电商场景中任务准确率提升 52%、格式遵从度提升 50%，超越 GPT-4o。

**[SSUF: A Semi-supervised Scalable Unified Framework for E-commerce Query Classification](a_semi-supervised_scalable_unified_framework_for_e-commerce_query_classification.md)**

:   提出电商查询分类的半监督可扩展统一框架 SSUF——三个可插拔模块：知识增强（LLM 世界知识+后验点击）解决短查询信息不足、标签增强（语义编码+半监督信号）打破对后验标签的依赖、结构增强（共现+语义+层级图 GCN）传播长尾标签梯度。已在 JD.COM 部署，离线和在线 A/B 实验均显著超越 SOTA。

**[A Spatio-Temporal Point Process for Fine-Grained Modeling of Reading Behavior](a_spatio-temporal_point_process_for_fine-grained_modeling_of_reading_behavior.md)**

:   提出基于标记时空点过程（Marked Spatio-Temporal Point Process）的阅读行为概率模型——用 Hawkes 过程建模跳视（何时何处注视），用对数正态分布+卷积建模注视时长（持续多久），避免传统聚合测量的信息丢失。实证发现：Hawkes 模型比基线更好拟合跳视数据，但上下文 surprisal 作为预测因子仅带来边际改善——surprisal 理论难以解释细粒度眼动。

**[IDR²: Accelerating Adaptive RAG via Instruction-Driven Representation Reduction of Retrieval Overlaps](accelerating_adaptive_retrieval_augmented_generation_via_instruction-driven_repr.md)**

:   首次识别自适应 RAG（A-RAG）中多轮检索结果重叠导致的冗余计算问题，提出 IDR²（Instruction-Driven Representation Reduction）框架：跨迭代 KV 缓存共享（CICS）加速预填充 2.79 倍、指令引导去重增强（IDGR）帮助 LLM 正确处理缓存vs新文档、信息引导并行生成（IGPG）加速解码 2.33 倍，总体 A-RAG 流程加速 2.0 倍且不损失生成质量。

**[Access Denied Inc: The First Benchmark Environment for Sensitivity Awareness](access_denied_inc_the_first_benchmark_environment_for_sensitivity_awareness.md)**

:   提出敏感性感知（Sensitivity Awareness, SA）概念——评估 LLM 是否能遵守基于角色的访问控制规则——并构建首个评估基准 Access Denied Inc：模拟企业数据库 + 多用户组权限 + 自动化问卷+半自动评分（99.9%自动），揭示模型在拒绝未授权请求和响应合法查询上的显著差异。

**[AceCoder: Acing Coder RL via Automated Test-Case Synthesis](acecoder_acing_coder_rl_via_automated.md)**

:   构建 AceCode-87K（87K 编码题 + 138 万自动合成测试用例），训练代码专用 Reward Model（7B 超越 340B Nemotron），Best-of-N 提升 Llama-3.1-8B 平均 8.9 分，R1 风格从 base 直接 RL 仅 80 步 HumanEval+ 提升 22.5%。

**[ACORD: An Expert-Annotated Retrieval Dataset for Legal Contract Clause Retrieval](acord_an_expert-annotated_retrieval_dataset_for_legal_contract_drafting.md)**

:   构建首个面向合同起草的专家标注条款检索基准 ACORD——114 个律师编写的查询、126,000+ 查询-条款对、1-5 星相关性评分，聚焦责任限制/赔偿/控制权变更/最惠国等复杂条款；bi-encoder 检索 + LLM 点式重排序表现有前景但距律师需求仍有显著差距。LLM 直接起草对比律师修改暴露了多种缺陷。

**[NTIL: Advancing Sequential Numerical Prediction in Autoregressive Models](advancing_sequential_numerical_prediction_in_autoregressive_models.md)**

:   提出 NTIL（Numerical Token Integrity Loss）解决自回归模型数值预测的两大缺陷——(1) token 级用 EMD 替代交叉熵保留数字间序数关系+指数位置加权，(2) 序列级通过可微数值构建+相对偏差度量评估整体数值误差。首次将 EMD 用于自回归模型优化，在目标检测/文本识别/数学推理上显著提升。

**[AIDE: Attribute-Guided Multi-Hop Data Expansion for Data Scarcity in Task-Specific Fine-tuning](aide_attribute-guided_multi-hop_data_expansion_for_data_scarcity_in_task-specifi.md)**

:   提出 AIDE——属性引导的多跳数据扩展框架，从仅 10 个种子数据点出发，通过提取主题/属性/关系三元组引导 LLM 多跳递归合成新数据，加入 Persona 增加多样性和残差连接防止偏离，在 Mistral-7B/Llama-3.1-8B/Llama-3.2-3B 上超越人工标注数据微调，比 Evol-Instruct 等 SOTA 提升 30%+。

**[AIR-Bench: Automated Heterogeneous Information Retrieval Benchmark](air-bench_automated_heterogeneous_information_retrieval_benchmark.md)**

:   提出 AIR-Bench——首个自动化、异构、动态的信息检索评测基准，通过 LLM 自动从真实语料生成高质量测试数据（三阶段管线：语料准备→候选生成→质量控制），覆盖 2 类任务、9 个领域、13 种语言共 69 个数据集，与人工标注数据高度一致，且持续动态更新避免数据泄露。

**[Aligned but Blind: Alignment Increases Implicit Bias by Reducing Awareness of Race](aligned_but_blind_implicit_bias.md)**

:   揭示对齐训练的"种族盲视"副作用：对齐使 LLM 在歧义上下文中不再将 black/white 表征为种族概念，安全护栏因此无法激活，导致隐式偏见从 64.1% 飙升至 91.4%；反直觉地，在早期层注入种族感知激活（而非遗忘）可将隐式偏见从 97.3% 降至 42.4%。

**[AmbiK: Dataset of Ambiguous Tasks in Kitchen Environment](ambik_dataset_of_ambiguous_tasks_in_kitchen_environment.md)**

:   提出 AmbiK，一个专门用于厨房环境中歧义指令检测的纯文本数据集，包含 1000 对歧义/非歧义指令，按三种歧义类型（用户偏好/常识/安全）分类标注，并评估了多种基于 conformal prediction 的歧义检测方法，发现现有方法在该基准上表现很差。

**[AnalyticKWS: Towards Exemplar-Free Analytic Class Incremental Learning for Small-footprint Keyword Spotting](analytickws_towards_exemplar-free_analytic_class_incremental_learning_for_small-.md)**

:   提出 AnalyticKWS，一种无需存储历史样本的关键词检测增量学习方法，通过冻结特征提取器 + 递归最小二乘解析解更新分类器，在 GSC 和 SC-100 数据集上超过了所有基于样本回放的方法，且训练时间和内存开销极低。

**[AntiLeakBench: Preventing Data Contamination by Automatically Constructing Benchmarks with Updated Real-World Knowledge](antileakbench_preventing_data_contamination_by_automatically_constructing_benchm.md)**

:   提出 AntiLeakBench——自动化反泄露基准框架，通过识别 LLM 知识截止后更新的真实世界新知识自动构建 QA 测试样本（而非简单收集新发布数据），确保测试知识严格不在训练集中，全自动流程无需人工标注，实验证实截止后性能普遍下降验证了数据污染的普遍存在。

**[Anything Goes? A Crosslinguistic Study of (Im)possible Language Learning in LMs](anything_goes_a_crosslinguistic_study_of_impossible_language_learning_in_lms.md)**

:   跨语言研究 LM 能否区分可能语言和不可能语言——在 12 种语言（4 个语系）上训练 GPT-2 Small 的可能/不可能/未见证变体，发现模型大体上能区分可能 vs 不可能语言（单语言内），但**跨语言**时区分能力减弱，且对类型学未见证语言（Greenberg Universal 20 的未见证词序）的区分仅在泛化测试中有效而在困惑度上无效——LM 有部分人类样的归纳偏置但弱于人类。

**[Are Bias Evaluation Methods Biased?](are_bias_evaluation_methods_biased.md)**

:   严格控制变量后比较三种主流偏见评估方法（结构化问答 BBQ、LLM-as-a-Judge、情感分析），发现不同方法对同一组 LLM 产生显著不同的偏见排名——偏见评估方法本身就是有偏的，企业不应依赖单一偏见基准来选择模型。

**[ARise: Towards Knowledge-Augmented Reasoning via Risk-Adaptive Search](arise_risk_adaptive_search.md)**

:   提出 ARise 框架，将贝叶斯风险评估与动态 RAG 集成到蒙特卡洛树搜索中，解决知识增强推理中的错误传播和验证瓶颈问题，在多跳QA任务上平均准确率超 SOTA KAR 方法 23.10%，超 RAG-equipped 推理模型（DeepSeek-R1）25.37%。

**[Attention Entropy is a Key Factor for Parallel Context Encoding](attention_entropy_parallel_encoding.md)**

:   发现并行上下文编码导致 query token 的注意力熵异常升高是性能下降的关键因素，并提出 Attention Sink 共享前缀和 Selective Attention 两种免微调方法有效缓解该问题。

**[AutoDS: Autonomous Data Selection with Zero-shot Generative Classifiers for Mathematical Texts](autonomous_data_selection_with_zero-shot_generative_classifiers_for_mathematical.md)**

:   提出 AutoDS——用基座 LLM 自身作为零样本"生成分类器"自动评估数学文本质量。通过两个 yes/no 问题的 logits 计算连续 LM-Score（而非二分类），筛选高质量数学文本做持续预训练，在 MATH/GSM8K/BBH 上大幅提升并实现约 2 倍 token 效率提升。发布 AutoMathText 数据集。

**[Behavioural vs. Representational Systematicity in End-to-End Models: An Opinionated Survey](behavioural_vs_representational_systematicity_in_end-to-end_models_an_opinionate.md)**

:   区分行为系统性（模型能否正确处理新组合）与表征系统性（模型内部表征是否结构化），指出当前基准和模型主要测试行为系统性却常声称解决了 Fodor-Pylyshyn 对表征系统性的挑战。基于 Hadley (1994) 的三级分类（弱/准/强系统性）分析语言和视觉关键基准的测试范围，最终呼吁用机械可解释性方法在行为评估之上补充表征分析。

**[Better Embeddings with Coupled Adam](better_embeddings_with_coupled_adam.md)**

:   从理论上证明 Adam 优化器的逐 token 二阶矩是导致 LLM 词嵌入各向异性（均值偏移）的根因，提出 Coupled Adam——对嵌入层的二阶矩取词汇平均——消除了各向异性问题，并在大规模实验中提升了嵌入质量和下游性能。

**[Beyond Frameworks: Unpacking Collaboration Strategies in Multi-Agent Systems](beyond_frameworks_multi_agent_collaboration.md)**

:   本文系统化地将多智能体协作分解为四个维度（治理模式、参与控制、交互模式、上下文管理），通过两个上下文依赖任务的大量实验证明：集中治理+指导者控制参与+有序交互+指导者摘要的组合最优，在保持甚至提升准确率的同时减少高达 93% 的 token 消耗。

**[Bone Soups: A Seek-and-Soup Model Merging Approach for Controllable Multi-Objective Generation](bone_soups_multi_objective_gen.md)**

:   提出 Bone Soup 模型合并方法，通过先构造"骨架奖励"（多目标奖励的组合）训练骨架模型、再用对称循环矩阵映射确定合并系数，解决了 Rewarded Soup 中单目标模型合并的次优性问题，在三个多目标生成任务上实现更好的 Pareto 前沿和可控性。

**[Bregman Conditional Random Fields: Sequence Labeling with Parallelizable Inference](bregman_conditional_random_fields_sequence_labeling_with_parallelizable_inferenc.md)**

:   提出 Bregman CRF (Bcrf)，一种基于均值正则化（mean regularization）的新型序列标注判别模型，使用迭代 Bregman 投影实现可并行化的推理算法，替代传统 CRF 中固有顺序的 Viterbi/Forward 算法，在 POS/NER/分词任务上性能与标准 CRF 持平但更快，且在有禁止标签转移约束的场景下优于 Mean Field 方法。

**[Byte Latent Transformer: Patches Scale Better Than Tokens](byte_latent_transformer.md)**

:   Meta FAIR提出BLT——首个在大规模（8B参数/4T字节）上匹配基于tokenizer的LLM性能的字节级架构，通过基于下一字节熵的动态分组（patching）将字节聚合为可变长度patch，在保持性能的同时实现最高50%推理FLOP节省，并开辟了"同时增大模型和patch尺寸"的全新scaling维度。

**[Causal Estimation of Tokenisation Bias](causal_tokenisation_bias.md)**

:   本文首次将 tokeniser 选择对语言模型输出的影响定义为"分词偏差"(tokenisation bias)，并利用因果推断中的断点回归设计(RDD)来量化这一效应——发现当一个 subword 被纳入词表时，其对应字符串的概率最高可提升 17 倍（小模型），揭示分词是语言建模中一个被低估的关键设计选择。

**[CoachMe: Decoding Sport Elements with a Reference-Based Coaching Instruction Generation Model](coachme_sport_instruction.md)**

:   提出 CoachMe，通过对比学习者动作与参考动作的差异（时间+物理两个维度），自动生成运动特异性的教练指导文本，在花样滑冰和拳击上分别超过 GPT-4o 31.6% 和 58.3%（G-Eval）。

**[Collapse of Dense Retrievers: Short, Early, and Literal Biases Outranking Factual Evidence](collapse_dense_retrievers.md)**

:   本文首次系统研究稠密检索器中多种启发式偏见（简短偏见、前置偏见、字面偏见、重复偏见）的个体和组合效应，发现当多种偏见叠加时，检索器选择包含答案的文档的概率低于10%，且这些偏见可被利用来操纵RAG系统，导致34%的性能下降。

**[Com2: A Causal-Guided Benchmark for Complex Commonsense Reasoning](com2_causal_commonsense.md)**

:   提出Com2基准，利用因果事件图和因果理论（干预/反事实）构建复杂常识推理任务，发现LLM在推理深度和广度上存在不足，后训练和慢思考可部分缓解。

**[Commonsense Reasoning in Arab Culture](commonsense_arab_culture.md)**

:   构建首个阿拉伯文化特定常识推理数据集 ArabCulture（3482 道由母语者原创的题目，覆盖 13 国×54 主题），评估多种 LLM 发现即使 32B 参数模型也在文化常识推理上表现不佳，且不同地区表现差异显著，地理/文化上下文线索的加入仅部分有效。

**[Completing A Systematic Review in Hours instead of Months with Interactive AI Agents](completing_a_systematic_review_in_hours.md)**

:   提出 InsightAgent，一个以人为中心的交互式多 Agent 系统，通过语义聚类分区、多 agent 并行阅读和实时用户交互，将医学系统综述的撰写时间从数月缩短到约 1.5 小时，达到人类撰写质量的 79.7%。

**[CORAL: Learning Consistent Representations across Multi-step Training with Lighter Speculative Drafter](coral_speculative_drafting.md)**

:   CORAL 通过跨步表示对齐（CSRA）改进多步训练中 draft 模型的特征一致性，并用权重分组机制压缩大词表 LM head 的推理延迟，在 LLaMA3/Qwen2.5 上实现 2.50-4.07× 加速，超越 EAGLE-2 和 HASS。

**[Cramming 1568 Tokens into a Single Vector and Back Again: Exploring the Limits of Embedding Space Capacity](cramming_tokens_embedding_capacity.md)**

:   通过逐样本优化方法将文本压缩到可训练的 [mem] 向量中，发现 Llama-3.1-8B 可以将 1568 个 token 无损压缩到单个输入向量中，揭示了现有方法（约 x10 压缩比）与实际可达极限（x1500+）之间存在两个数量级的差距。

**[Generating Plausible Distractors for Multiple-Choice Questions via Student Choice Prediction](distractor_gen_multiple_choice.md)**

:   提出选择题干扰项生成的三步流水线：(1) 训练配对排序器预测学生误选哪个干扰项；(2) 用排序器构造偏好数据集；(3) 用 DPO 训练生成器产生更具迷惑性的干扰项。在 CS 领域（Python/DB/ML）实验中，生成的干扰项更难以区分，且题目鉴别指数(DI)更高。

**[Divide-Then-Align: Honest Alignment based on the Knowledge Boundary of RAG](divide_then_align_rag_knowledge_boundary.md)**

:   DTA 提出将 RAG 查询按参数知识边界和检索知识边界划分为四个象限，对"两者都不知道"的查询构造偏好数据用 DPO 训练模型回答"我不知道"，解决了 RAFT 模型即使在检索完全噪声时也强行生成答案的问题，在准确率和适当弃权之间实现了有效平衡。

**[DoMIX: An Efficient Framework for Exploiting Domain Knowledge in Fine-Tuning](domix_an_efficient_framework_for_exploiting.md)**

:   提出 DoMIX，将各领域知识用独立 LoRA 模块存储后通过对角初始化的 bridge 矩阵在微调时灵活组合利用，在持续领域适应预训练场景下减少 58% 预训练时间和 87% GPU 内存，同时性能超越 SOTA。

**[Principled Content Selection to Generate Diverse and Personalized Multi-Document Summaries](dpp_diverse_multidoc_summary.md)**

:   用行列式点过程（DPP）替代 LLM 的隐式内容选择来生成多样化多文档摘要——将任务分解为"提取原子关键点→DPP 选择多样子集→LLM 重写为摘要"三步，解决了 LLM 的"lost in the middle"问题，在 DiverseSumm 基准上一致提升了源覆盖率。

**[DREsS: Dataset for Rubric-based Essay Scoring on EFL Writing](dress_dataset_rubric_based_essay_scoring_efl_writing.md)**

:   发布 DREsS——一个面向 EFL（英语作为外语）写作教育的大规模标准化评分量规数据集（48.9K 样本），并提出基于文本损坏的 CASE 数据增强策略，将基线性能提升 45.44%。

**[Divide-Then-Aggregate: An Efficient Tool Learning Method via Parallel Tool Invocation](dta_llama_parallel_tool_invocation.md)**

:   提出 DTA-Llama，将传统树搜索的串行工具调用路径转换为有向无环图（DAG）结构实现并行调用，设计 Process/Thread 推理框架使 LLM 在每轮中可分解任务并并行执行多个工具，在 StableToolBench 上使 Llama2-7B 达到 GPT-3.5 Parallel Function Calling 的水平。

**[An Effective Incorporating Heterogeneous Knowledge Curriculum Learning for Sequence Labeling](dual_stage_curriculum_learning_sequence_labeling.md)**

:   提出面向序列标注任务的双阶段课程学习（DCL）框架，通过数据级和模型级两阶段的由易到难训练策略以及基于贝叶斯不确定性的动态难度度量，在提升性能的同时加速训练超过 25%。

**[EcomScriptBench: A Multi-task Benchmark for E-commerce Script Planning via Step-wise Intention-Driven Product Association](ecomscriptbench.md)**

:   提出电商脚本规划（EcomScript）任务及其首个大规模benchmark EcomScriptBench（605K脚本、2.4M产品），通过购买意图（purchase intention）桥接用户行动步骤与产品检索的语义鸿沟，实验发现当前LLM在涉及产品的子任务上表现显著不足，注入意图知识可提升性能。

**[Efficient Knowledge Editing via Minimal Precomputation](efficient_knowledge_editing.md)**

:   证明了 MEMIT/ROME/EMMET 等知识编辑方法的预计算步骤（缓存 4400 万隐向量）可以减少到理论最小值的 2-10 倍（不到原来的 0.3%），将预计算时间从数十小时降到几分钟，且编辑性能基本无损。

**[Enhancing Transformers for Generalizable First-Order Logical Entailment](enhancing_fol_entailment.md)**

:   系统性研究 Transformer 在一阶逻辑蕴涵任务中的泛化推理能力，揭示了查询语法、token 嵌入和 Transformer 架构（特别是位置编码）的影响，并提出 TEGA（Transformer Encoder with Guided Attention）在相对位置编码设定下显著提升逻辑推理性能。

**[Model Extrapolation Expedites Alignment](expo_model_extrapolation.md)**

:   基于"对齐训练仅产生微小参数变化"的观察，提出ExPO方法——通过放大SFT→DPO的参数变化方向（$\theta_2 = \theta_1 + \alpha\Delta\theta$），在零额外训练开销下提升对齐性能，使仅训练20%步骤的DPO模型超越完整训练的版本。

**[FaithfulRAG: Fact-Level Conflict Modeling for Context-Faithful Retrieval-Augmented Generation](faithfulrag_fact_level_conflict.md)**

:   发现现有忠实 RAG 方法通过强制抑制参数知识来实现上下文忠实，但这增加了误解上下文的风险（不忠实错误减少 6.65% 的同时错误匹配增加 6.42%）。提出 FaithfulRAG，通过事实级冲突识别（自事实挖掘）和冲突推理（自思考模块）解决知识冲突，在 FaithEval/SQuAD/MuSiQue/RealtimeQA 上超越最强基线 8-9 个百分点。

**[Towards Robust and Efficient Federated Low-Rank Adaptation with Heterogeneous Clients](federated_lora_heterogeneous.md)**

:   提出 LoRA-A²（Low Rank Adaptation with Alternating freeze and Adaptive rank selection），通过交替冻结 A/B 矩阵解决联邦 LoRA 聚合不一致问题，并结合自适应秩选择机制在大幅压缩上传参数量（最高减少 99.8%）的同时保持鲁棒性，尤其在低秩+高数据异构场景下显著优于现有方法。

**[FinanceReasoning: Benchmarking Financial Numerical Reasoning More Credible, Comprehensive and Challenging](financereasoning_benchmarking_financial_numerical_reasoning_more.md)**

:   提出 FinanceReasoning benchmark，通过重标注公开数据集、构建 3,133 个 Python 金融函数库和新增 908 道专家标注难题，在可信度、全面性和挑战性三个维度提升金融数值推理评估能力。

**[Behind Closed Words: Creating and Investigating the forePLay Annotated Dataset for Polish Erotic Discourse](foreplay_polish_erotic_detection.md)**

:   构建了首个波兰语色情内容检测数据集 forePLay（24,768 句，5 类标签），提出涵盖模糊性、暴力和社会不可接受行为的多维标注体系，评估发现专用波兰语模型显著优于多语言模型，且 Transformer 编码器模型在不平衡类别处理上表现最强。

**[Frictional Agent Alignment Framework: Slow Down and Don't Break Things](frictional_agent_alignment.md)**

:   提出摩擦对齐框架 FAAF（Frictional Agent Alignment Framework），通过双策略（frictive state policy + intervention policy）目标函数，训练 LLM 在协作对话中识别信念冲突并生成促进反思与审议的"摩擦"干预，超越 DPO/IPO/PPO 等对齐方法。

**[GainRAG: Preference Alignment in Retrieval-Augmented Generation through Gain Signal Synthesis](gainrag_preference_alignment.md)**

:   发现 RAG 中检索器的"相关性"与 LLM 的"偏好"存在偏差——含正确答案的段落仍可能导致错误生成，而间接相关段落反而有用。提出 GainRAG，用基于对比解码困惑度的"增益"信号量化 LLM 偏好，训练轻量选择器在检索结果中选择真正有"增益"的段落，在 6 个 QA 数据集上显著超越 Standard RAG 和 Rerank 基线。

**[Controllable and Reliable Knowledge-Intensive Task-Oriented Conversational Agents with Declarative Genie Worksheets](genie_worksheets_tod_agent.md)**

:   Genie 提出了一个可编程的知识密集型任务导向对话框架，通过声明式 Worksheet 规范定义 Agent 策略，将 LLM 限制在语义解析和回复生成两个角色，由算法化运行时系统强制执行策略，实现从 21.8% 到 82.8% 的真实任务完成率提升。

**[Gumbel Reranking: Differentiable End-to-End Reranker Optimization](gumbel_reranking.md)**

:   将 RAG 系统中的重排序过程重新建模为文档级 Top-k 注意力掩码问题，利用 Gumbel 技巧和松弛 Top-k 采样实现端到端可微优化，直接最小化最终语言建模损失，在 HotpotQA 上 Recall@5 提升 10.4%。

**[HateDay: Insights from a Global Hate Speech Dataset Representative of a Day on Twitter](hateday_global_hate_speech.md)**

:   HateDay 构建了首个全球代表性仇恨言论数据集——24 万条随机采样的 Twitter 推文覆盖 8 种语言和 4 个英语国家，揭示了学术数据集大幅高估了检测模型在真实场景中的表现，尤其对非欧洲语言检测能力极差。

**[Hierarchical Bracketing Encodings for Dependency Parsing as Tagging](hierarchical_bracketing_dep_parsing.md)**

:   提出层次化括号编码家族用于依存句法分析的序列标注范式，证明现有4-bit编码是该家族的非最优特例，推导出仅需12个标签的最优编码，并将其推广到处理任意非投射性。

**[Hierarchical Memory Organization for Wikipedia Generation](hierarchical_memory_wikipedia_gen.md)**

:   提出 Memory Organization-based Generation（MOG）框架，从网页文档中提取细粒度记忆单元（factoid），通过递归聚类-摘要算法组织为层次化 Wikipedia 大纲结构，使每个章节都有直接的记忆支撑，在 FreshWiki 和 WikiStart 数据集上信息量、引用率和可验证性全面超越 RAG 和 STORM 基线。

**[Hierarchical Level-Wise News Article Clustering via Multilingual Matryoshka Embeddings](hierarchical_news_clustering.md)**

:   本文提出利用多语言 Matryoshka 嵌入的分层特性进行新闻文章聚类：低维捕捉主题级相似度、中维捕捉叙事级相似度、高维捕捉事件级相似度，结合改良的 RAC 层级聚类算法，在 SemEval 2022 Task 8 上达到 SOTA（Pearson ρ = 0.816）。

**[Counterspeech the Ultimate Shield! Multi-Conditioned Counterspeech Generation through Attributed Prefix Learning](hippro_counterspeech_gen.md)**

:   提出 HiPPrO 两阶段框架用于多条件反仇恨言论生成——第一阶段通过层次化前缀学习在多个属性（策略+情感）空间中优化反言论生成，第二阶段用无参考无奖励的偏好优化提升建设性，策略一致性提升 ~38%，ROUGE 指标提升 2-3%。

**[HybGRAG: Hybrid Retrieval-Augmented Generation on Textual and Relational Knowledge Bases](hybgrag_hybrid_rag_skb.md)**

:   提出 HybGRAG 方法，通过检索器库（Retriever Bank）同时利用文本和关系信息，配合 Critic 模块的自反思迭代纠正问题路由错误，在半结构化知识库上的混合问答任务中 Hit@1 平均提升 51%。

**[Enhancing Hyperbole and Metaphor Detection with Their Bidirectional Dynamic Interaction and Emotion Knowledge](hyperbole_metaphor_detection.md)**

:   提出 EmoBi 框架，通过情感分析→情感引导的域映射→双向动态交互三阶段 prompting 流程，利用 LLM 挖掘夸张和隐喻背后的情感线索及二者的互促关系，在四个数据集上大幅超越 SoTA（TroFi 上夸张检测 F1 提升 28.1%，HYPO-L 上隐喻检测 F1 提升 23.1%）。

**[If Attention Serves As A Cognitive](if_attention_serves_as_a_cognitive.md)**

:   通过将 Transformer Grammar（TG）的注意力机制与人类阅读时间数据关联，首次证明在句法结构上操作的注意力比在 token 序列上操作的普通 Transformer 注意力能更好地预测人类阅读行为，揭示人类句子处理涉及"句法结构+词序列"的双重记忆表征。

**[Predicting Implicit Arguments in Procedural Video Instructions](implicit_arguments_video_instructions.md)**

:   提出 Implicit-VidSRL 数据集与 iSRL-Qwen2-VL 模型，针对过程性视频指令中省略的隐含论元（食材成分）进行预测，通过 SRL 框架将多步指令分解为 {verb, what, where/with} 三元组，在银标数据上微调后在隐含论元 F1 上超越 GPT-4o 达 17%。

**[ImpliHateVid: Implicit Hate Speech Detection in Videos](implihatevid_video_hate.md)**

:   首次提出视频中隐性仇恨言论检测任务，构建2009个视频的ImpliHateVid数据集，并设计两阶段对比学习框架融合文本、图像、音频三模态特征。

**[Ewe: Improving Factuality with Explicit Working Memory](improving_factuality_with_explicit_working_memory.md)**

:   提出 Ewe（Explicit Working mEmory），在 LLM 解码过程中引入由多个 KV cache 单元组成的显式工作记忆，实时接收检索知识反馈和事实核查反馈，检测到错误时删除错误句子并用更新后的记忆重新生成，在 4 个事实性长文本生成基准上将 VeriScore F1 提升 2–6 分且不损失回答有用性。

**[Improving Language and Modality Transfer in Translation by Character-level Modeling](improving_language_and_modality_transfer_in.md)**

:   提出基于字符级编码器 charSONAR 的跨语言跨模态翻译方法，通过 teacher-student 训练获得字符级文本编码器，再用轻量适配器连接 1000+ 语言的 CTC ASR 模型（MMS），在 75 语言文本翻译和 33 语言语音翻译上实现 SOTA，零资源低资源场景表现尤其突出。

**[InSerter: Speech Instruction Following with Unsupervised Interleaved Pre-training](inserter_speech_instruction.md)**

:   提出 InSerter（交错语音-文本预训练）方法，通过 TTS 将大规模文本语料合成为交错的语音-文本序列进行预训练，大幅提升 SpeechLLM 的语音指令遵循能力，并构建首个全面的语音指令遵循基准 SpeechInstructBench。

**[InspireDebate: Multi-Dimensional Evaluation-Guided Reasoning for Debating](inspiredebate_multidim_evaluation_debating.md)**

:   提出双组件框架：InspireScore（融合4个主观维度+2个客观维度的辩论评估系统）和 InspireDebate（通过CoT-SFT + 多维DPO + Web-RAG 三阶段优化的辩论框架），评估系统与专家判断相关性提高 44%，辩论性能超越基线 57%。

**[Intuitive Fine-Tuning: Towards Simplifying Alignment into a Single Process](intuitive_fine_tuning.md)**

:   通过MDP框架将SFT和偏好优化统一建模为"偏好估计+转移优化"两个子过程，揭示SFT本质上是偏好优化的特殊退化形式（使用偏差先验），提出IFT方法通过时间残差连接在仅使用SFT格式数据的条件下实现接近或超越SFT+PO顺序训练的对齐效果。

**[KnowShiftQA: How Robust are RAG Systems when Textbook Knowledge Shifts in K-12 Education?](knowshiftqa_rag_knowledge_shifts.md)**

:   构建了 KnowShiftQA 数据集（3,005 道题，覆盖 5 个学科），通过假设性知识更新模拟教科书与 LLM 参数知识的差异，系统评估 RAG 系统面对知识偏移时的鲁棒性，发现现有 RAG 系统在知识偏移下性能下降 22-27%。

**[KokoroChat: A Japanese Psychological Counseling Dialogue Dataset Collected via Role-Playing by Trained Counselors](kokorochat_a_japanese_psychological_counseling_dialogue.md)**

:   提出 KokoroChat，一个通过训练有素的咨询师角色扮演收集的日语心理咨询对话数据集，包含 6,589 段长对话及详细的客户反馈评分，用于提升 LLM 的心理咨询回复生成和对话评估能力。

**[LAQuer: Localized Attribution Queries in Content-grounded Generation](laquer_localized_attribution.md)**

:   提出 Localized Attribution Queries (LAQuer) 任务——将生成文本中用户选定的片段精确定位到源文档的对应片段，实现比句子级归因更精细、比子句级归因更用户导向的溯源，在多文档摘要和长文本问答上显著减少了归因文本长度。

**[Length-Induced Embedding Collapse in PLM-based Models](length-induced_embedding_collapse_in_plm-based_models.md)**

:   发现并严格证明了 PLM 文本嵌入模型中的"长度坍缩"现象——长文本嵌入趋于聚集，源于 self-attention 作为低通滤波器随文本长度增加而滤波率增强，高频信息被过度抑制；提出 TempScale 方法通过降低 attention 温度来缓解长短文本嵌入分布差异，在 MTEB 上提升 0.94%、LongEmbed 上提升 1.10%。

**[Literature Meets Data: A Synergistic Approach to Hypothesis Generation](literature_meets_data_hypothesis.md)**

:   提出首个将文献驱动和数据驱动假设生成进行协同整合的方法，通过 Refinement 和 Union 两种策略让 LLM 从论文摘要和观测数据中联合生成更具泛化性的假设，在五个社会科学分类任务的 OOD 数据集上比纯数据驱动方法平均提升 3.37%，并首次通过人类实验证明 LLM 生成的假设能显著改善人类决策准确率（+7.44% / +14.19%）。

**[LoGU: Long-form Generation with Uncertainty Expressions](logu_longform_gen_uncertainty.md)**

:   定义"长文本不确定性生成"（LoGU）任务，识别不确定性抑制和不确定性错位两个子挑战，提出基于分解的数据构造框架和 SFT+DPO 两阶段训练流水线，使 LLM 在长文本生成中对不确定事实显式表达不确定性，在三个数据集上将 Llama3-8B 的事实准确率从 51.9% 提升到 71.6%，错误声明数从 20.4 降到 5.81。

**[Read it in Two Steps: Translating Extremely Low-Resource Languages with Code-Augmented Grammar Books](low_resource_translation.md)**

:   将语法书辅助的极低资源翻译（XLR MT）分解为语法规则检索和规则应用两步，并提出用代码格式表示语法规则以提升 LLM 在两步中的表现，在壮语翻译上实现了 13.1% BLEU 的提升。

**[Micro-Act: Mitigate Knowledge Conflict in QA via Actionable Self-Reasoning](micro_act_knowledge_conflict_reasoning.md)**

:   提出 Micro-Act 框架，通过层次化动作空间（导航/功能/桥接动作）和自适应粒度分解，让 LLM 自动感知上下文复杂度并逐层拆解知识对比，在 5 个知识冲突基准上全面超越 SOTA，同时在无冲突场景下也保持鲁棒。

**[MindRef: Mimicking Human Memory for Hierarchical Reference Retrieval with Fine-Grained Location Awareness](mindref_mimicking_human_memory_hierarchical_reference_retrieval.md)**

:   提出 MindRef 框架，模拟人类先回忆文档标题再定位具体段落的两阶段记忆模式，通过 Trie 和 FM-Index 约束解码让 LLM 独立召回参考段落，无需额外检索模型或预分段。

**[InterpoLL: Mitigating Shortcut Learning with InterpoLated Learning](mitigating_shortcut_learning_with_interpolated_learning.md)**

:   提出 InterpoLL，通过将多数样本的表征与同类少数样本的表征做插值 $z_i = (1-\lambda)f_{enc}(x_i) + \lambda f_{enc}(x_j)$ 来弱化捷径特征影响，在 MNLI/FEVER/QQP 等 NLU 任务上少数样本泛化显著超越 ERM 和 SOTA 捷径缓解方法，且不损失多数样本准确率。

**[Modular Sentence Encoders: Separating Language Specialization from Cross-Lingual Alignment](modular_sentence_encoders.md)**

:   本文提出模块化多语言句子编码器训练方案：先训练语言特定模块（embedding + 语言适配器 + 句子编码适配器）缓解多语言诅咒，再训练跨语言对齐适配器同时使用平行和释义数据解决不同跨语言任务间的性能权衡，在 4 个任务和 23 种语言上全面优于单体模型训练。

**[MOSAIC: Multiple Observers Spotting AI Content](mosaic_multiple_observers_spotting_ai_content.md)**

:   基于信息论中的通用压缩原理，提出 MOSAIC——多 LLM 集成的 AI 生成文本检测方法，通过 Blahut-Arimoto 算法为多个 detector LLM 计算最优组合权重，构建混合分布作为观察者，比较文本的实际 surprisal 与混合模型的期望交叉熵差异来判断是否为 AI 生成，在多个域/语言/生成器上鲁棒优于单模型和双模型（如 Binoculars）方法。

**[Data Quality Issues in Multilingual Speech Datasets: The Need for Sociolinguistic Awareness and Proactive Language Planning](multilingual_speech_data_quality.md)**

:   对三大公开多语言语音数据集（Common Voice、FLEURS、VoxPopuli）进行系统质量审计，发现低资源语言存在严重的微观和宏观质量问题，并提出基于社会语言学意识的数据集创建指南。

**[Unifying Continuous and Discrete Text Diffusion with Non-simultaneous Diffusion Processes](neodiff_unified_text_diffusion.md)**

:   提出 NeoDiff，通过引入"外在时间"（句子级扩散进度）和"内在时间"（token 级扩散进度）的双时间框架，利用 Poisson 过程为每个 token 独立分配细粒度噪声水平，并用上下文感知的时间预测器自适应调节去噪进度，统一了离散和连续文本扩散模型的理论框架，在机器翻译、复述、文本简化等多个任务上超越现有扩散基线。

**[Normalized AOPC: Fixing Misleading Faithfulness Metrics for Feature Attribution Explainability](normalized_aopc_faithfulness_metrics.md)**

:   本文揭示了广泛使用的 AOPC（扰动曲线下面积）忠实度指标在跨模型比较时会产生误导性结论（因为不同模型的 AOPC 上下界差异巨大），提出 Normalized AOPC (NAOPC) 通过 min-max 归一化消除模型间的不可比性，实验表明归一化可以根本性地改变模型忠实度排名。

**[Enhancing Automated Interpretability with Output-Centric Feature Descriptions](output_centric_interpretability.md)**

:   提出两种以输出为中心的特征描述方法（VocabProj 和 TokenChange），弥补现有基于输入激活的自动可解释性流程只能捕获"什么激活了特征"而忽略"特征如何影响输出"的缺陷，并证明输入+输出方法的集成能生成最忠实的特征描述。

**[Towards Better Evaluation for Generated Patent Claims](patclaimeval_patent_evaluation.md)**

:   提出首个专利权利要求评估基准 Patent-CE（1228 个专家标注的比较评估数据点）和专用评估方法 PatClaimEval（基于 Longformer + 对比学习变体），在特征完整性、概念清晰度、术语一致性、逻辑连接和整体质量五个维度上与人类专家评估的相关性全面超越 13 种现有指标（包括 G-Eval-4），整体质量维度的 Spearman 提升 58%。

**[Personalized Generation In Large Model Era A Survey](personalized_generation_in_large_model_era_a_survey.md)**

:   首篇跨模态个性化生成（PGen）综合综述，提出统一的用户中心视角将 NLP/CV/IR 社区的研究纳入一个框架，系统化梳理了文本/图像/视频/音频/3D/跨模态六大模态下的个性化技术、数据集和评估指标，涵盖 200+ 篇文献，并指出可扩展性、偏好演化、隐私公平等关键挑战。

**[All That Glitters is Not Novel: Plagiarism in AI Generated Research](plagiarism_ai_generated_research.md)**

:   在对自主科研 Agent（如 AI Scientist）生成的研究文档进行专家审查后发现，24% 的文档是"智能剽窃"——方法论与已有工作一一对应但不引用原始来源，且现有剽窃检测工具无法识别这种"改头换面"的抄袭。

**[Principled Understanding of Generalization for Generative Transformer Models in Arithmetic Reasoning Tasks](principled_generalization_arithmetic.md)**

:   建立首个统一理论框架来理解 Transformer 在算术任务（加法/乘法/模运算）上的泛化行为——从任务性质（平移不变性）和位置编码类型（APE/RPE）的交互出发，解释了之前困扰领域的多个泛化谜题（如加法能泛化但乘法不能，模100能泛化但模101不能），实验验证了理论预测。

**[PRISM: A Framework for Producing Interpretable Political Bias Embeddings](prism_political_bias_embeddings.md)**

:   提出PRISM框架——首个专为政治偏见嵌入设计的方法，通过争议性话题偏见指标挖掘和Cross-Encoder政治偏见评分两阶段，生成可解释的偏见感知嵌入，在政治偏见分类上超越通用文本嵌入模型。

**[Substance over Style: Evaluating Proactive Conversational Coaching Agents](proactive_conversational_coaching.md)**

:   通过健康教练领域的专家访谈和用户研究（31 名参与者、155 段对话），系统评估了五种不同对话风格（Directive、Interrogative、Facilitative）的 LLM 教练 Agent，发现用户高度重视核心功能性（substance）而对缺乏功能性时的风格修饰（style）持负面态度，同时揭示了用户第一人称评价与专家/LLM 第三方评价之间的显著不一致。

**[ProxAnn: Use-Oriented Evaluations of Topic Models and Document Clustering](proxann_topic_model_eval.md)**

:   ProxAnn 设计了一个面向实际使用场景的主题模型/文档聚类评估协议——标注者先从模型输出推断类别再将类别应用到新文档——并证明 LLM 代理标注者可以统计不可分辨地替代人类标注者，同时发现经典 LDA 在此评估下表现不弱于现代方法。

**[Uncovering Visual-Semantic Psycholinguistic Properties from the Distributional Structure of Text Embedding Space](psycholinguistic_visual_semantic.md)**

:   提出 Neighborhood Stability Measure (NSM)，仅通过文本嵌入空间中邻域的"尖锐度"来无监督估计词汇的可意象性(imageability)和具体性(concreteness)，无需视觉模态或生成模型即可超越现有方法。

**[PVP: An Image Dataset for Personalized Visual Persuasion with Persuasion Strategies, Viewer Characteristics, and Persuasiveness Ratings](pvp_an_image_dataset_for_personalized.md)**

:   构建了首个大规模个性化视觉说服数据集 PVP（28,454 张图像、596 条消息、9 种说服策略、2,521 位标注者的心理特征），并提出说服图像生成和评估两个任务，发现融入观看者的心理特征能显著提升说服图像的生成和评估效果。

**[Quantifying Lexical Semantic Shift via Unbalanced Optimal Transport](quantifying_lexical_semantic_shift_via_unbalanced_optimal_transport.md)**

:   将 Unbalanced OT 应用于上下文化词嵌入来量化词义变迁，提出 Sense Usage Shift (SUS) 指标在每个用法实例级别量化语义变化（义项使用频率增减），统一解决实例级变化检测、词级变化幅度量化、词义扩展/缩小判定。

**[ReflectDiffu: Reflect between Emotion-intent Contagion and Mimicry for Empathetic Response Generation via a RL-Diffusion Framework](reflectdiffu_empathetic_response.md)**

:   提出轻量级共情对话框架 ReflectDiffu，融合情感传染（捕捉情绪）、意图二次机制（Exploring-Sampling-Correcting将情绪映射为行动意图）和扩散模型生成，在相关性、可控性和信息量上全面超越现有基线和 Llama-3.1-8B。

**[Re-identification of De-identified Documents with Autoregressive Infilling](reidentification_deidentified.md)**

:   提出一种基于 RAG 的再识别攻击方法，通过稀疏检索+稠密检索+自回归填充的三阶段流程，对去标识化文档进行逆向攻击，在三个数据集上高达 80% 的被遮蔽文本片段可被成功恢复，揭示了当前去标识化方法的脆弱性。

**[Identifying Reliable Evaluation Metrics for Scientific Text Revision](reliable_eval_metrics_scientific.md)**

:   系统分析了传统相似度指标（ROUGE、BERTScore 等）在科学文本修订评估中的局限性，发现它们强相关于编辑距离且惩罚深度修改，并证明结合 LLM-as-Judge 和任务特定指标的混合方法最能对齐人类评判。

**[REP: Keys to Robust Edits — From Theoretical Insights to Practical Advances](rep_robust_knowledge_editing.md)**

:   揭示locate-and-edit知识编辑方法中语义键的根本缺陷——内部表示无法同时满足鲁棒性和特异性，提出REP模块通过对比学习解耦编辑键，在鲁棒性测试上提升最高66.4%。

**[Retrieve to Explain: Evidence-driven Predictions for Explainable Drug Target Identification](retrieve_to_explain_drug_target_identification.md)**

:   提出 R2E (Retrieve to Explain)，一种基于检索的架构，通过从文献语料库中检索证据来评分和排序所有候选答案，并利用 Shapley 值将预测忠实地归因到支撑证据，在药物靶点识别任务上超越了遗传学基线和 GPT-4 基线。

**[Revisiting Weak-to-Strong Generalization: Reverse KL vs. Forward KL](revisiting_weak-to-strong_generalization_in_theory_and_practice_reverse_kl_vs_fo.md)**

:   在 Weak-to-Strong Generalization (W2SG) 框架中，提出用 reverse KL 替代 forward KL 作为损失函数——理论证明 reverse KL 的 mode-seeking 特性可保证强模型超过弱监督者至少"分歧量"的幅度，实验在 GPT-2/Pythia/Qwen2.5 系列上验证 reverse KL/CE 在 12/12 设置中超越 forward KL 且噪声鲁棒性更好。

**[Rubrik's Cube: Testing a New Rubric for Evaluating Explanations on the CUBE Dataset](rubriks_cube_testing_a_new_rubric_for_evaluating_explanations_on_the_cube_datase.md)**

:   提出 Rubrik——受教育评估启发的解释质量评价量规，基于三层嵌套类型体系（Commentary⊆Justification⊆Argument）+ 8 维质量维度，配套 CUBE 数据集（26K 条由人类和 6 个 LLM 生成的解释），发现 LLM 解释低质主因是缺乏简洁性而非连贯性。

**[SCAR: Data Selection via Style Consistency-Aware Response Ranking for Efficient Instruction-Tuning](scar_style_consistency_data_selection.md)**

:   SCAR 识别出回复的"语言形式"和"指令惊奇度"是影响 LLM 指令微调效果的两个关键风格因素，并提出基于风格一致性的排序方法自动选择高质量训练数据，仅用 0.7% 的原始数据就能让微调后的 LLM 匹配甚至超越全数据集训练的性能。

**[SEOE: A Scalable and Reliable Semantic Evaluation Framework for Open Domain Event Detection](seoe_semantic_eval.md)**

:   针对开放域事件检测（ODED）评估的两大痛点——有限 benchmark 缺乏真实世界代表性、token 级匹配指标无法捕捉语义相似性——提出 SEOE 框架，构建包含 564 种事件类型覆盖 7 大领域的可扩展 benchmark，并引入基于 LLM 的语义 F1 评估指标。

**[Shifting from Ranking to Set Selection for Retrieval Augmented Generation](setr_set_selection_rag.md)**

:   提出从"逐个排序"到"集合选择"的 RAG 检索范式转换——SetR 通过 CoT 推理显式识别查询的信息需求，然后选择一组能共同满足这些需求的段落（而非逐个评分最相关的），在多跳 RAG 基准上超越 GPT-4o 级别的重排器。

**[Share Text To Sql Correction](share_text_to_sql_correction.md)**

:   提出 SHARE 框架，通过三个专用小语言模型（SLM）组成的顺序管道，将 SQL 查询转换为逐步动作轨迹并分别修正 schema 错误和逻辑错误，从而以低成本高效辅助 LLM 进行 Text-to-SQL 自纠正。

**[ShifCon: Enhancing Non-Dominant Language Capabilities with a Shift-based Multilingual Contrastive Framework](shifcon_nondominant_language.md)**

:   提出 ShifCon 框架，通过将非优势语言的表示 shift 到优势语言子空间以获取更丰富的模型知识，再 shift 回原语言子空间进行生成，结合多语言对比学习，显著提升低资源语言的表现。

**[Statistical Deficiency for Task Inclusion Estimation](statistical_deficiency_task_inclusion.md)**

:   基于统计缺陷性（statistical deficiency）理论，提出一种理论驱动的任务包含关系（task inclusion）定义与度量框架，以信息充分性（information sufficiency, IS）作为可计算代理指标，通过比较微调模型的中间层表征来估计任务间的包含程度，并在合成数据和真实NLP任务上成功重建了经典NLP pipeline的层次关系。

**[STRICTA: Structured Reasoning in Critical Text Assessment for Peer Review and Beyond](stricta_structured_reasoning_in_critical_text_assessment_for_peer_review_and_bey.md)**

:   提出 STRICTA 框架，基于结构因果模型（SCM）将文本评审建模为显式的逐步推理图（workflow），在生物医学论文评审中收集 40+ 位专家的 4000+ 推理步骤数据集，发现先验知识差异是专家分歧主因、写作风格对最终评审有因果影响，LLM 存在错误传播但人类监督可有效缓解。

**[SudoLM: Learning Access Control of Parametric Knowledge with Authorization Alignment](sudolm_authorization_alignment.md)**

:   SudoLM 提出了一种 LLM 参数化知识访问控制框架，通过"SUDO key"机制让授权用户解锁受限知识（如医学领域知识），未授权用户则只能访问公开知识，用 DPO 的 authorization alignment 在一个模型内实现了传统需要多版本模型才能完成的分级访问控制。

**[The Harmonic Structure of Information Contours](the_harmonic_structure_of_information_contours.md)**

:   提出 Harmonic Surprisal (HS) 假说——文本中 surprisal 曲线呈周期性波动且周期与语篇结构（EDU/句子/段落）对齐，用带时间缩放的谐波回归检验，在 6 种语言上发现一致的周期模式，精化了经典的 Uniform Information Density 假说。

**[Learning to Reason Over Time: Timeline Self-Reflection for Temporal Reasoning](tiser_timeline_self_reflection_temporal.md)**

:   提出 TISER 框架，通过"推理→时间线构建→自反思→答案生成"四阶段管道实现LLM时间推理的test-time scaling，配合合成推理轨迹数据微调，让 7B 开源模型在多个时间推理基准上超越 GPT-4，在TGQA等任务上达到 SOTA。

**[TokAlign: Efficient Vocabulary Adaptation via Token Alignment](tokalign_vocab_adaptation.md)**

:   提出 TokAlign，基于 Token 共现信息学习两个词表之间的一对一映射矩阵，高效替换 LLM 的词表，实现跨语言知识迁移和跨模型 token 级蒸馏。

**[Did Translation Models Get More Robust Without Anyone Even Noticing?](translation_robustness.md)**

:   通过合成噪声和社交媒体文本的系统性实验，证明现代大规模预训练翻译模型（LLM）在未经任何专门鲁棒性训练的情况下，对多种输入噪声的鲁棒性已远超传统 NMT 模型，鲁棒性随模型规模增长自然提升。

**[TROVE: A Challenge for Fine-Grained Text Provenance via Source Sentence Tracing and Relationship Classification](trove_a_challenge_for_finegrained_text.md)**

:   提出TROVE文本溯源挑战，将目标文本中每个句子追溯到源文档中的具体源句，并分类其细粒度关系（引用、压缩、推理等），覆盖多文档和长文档场景。

**[Tuna: Comprehensive Fine-grained Temporal Understanding Evaluation on Dense Dynamic Videos](tuna_temporal_understanding.md)**

:   本文提出 Tuna，一个面向密集动态视频的细粒度时序理解基准，包含 1000 个精心标注的视频和两个互补任务（Tuna-cap 字幕生成和 Tuna-mcq 视频问答），覆盖相机状态、场景、动作、属性四大动态要素，评估 21 个模型后发现即使 GPT-4o 也仅达 58.5% F1，揭示了当前模型在时序理解上的重大不足。

**[USDC: A Dataset of User Stance and Dogmatism in Long Conversations](usdc_a_dataset_of_underlineuser_underlinestance_and_underlinedogmatism_in_long_u.md)**

:   构建 USDC——首个用户级长对话立场和教条主义数据集，764 个多用户 Reddit 对话（22 子版块），用 {Mistral Large, GPT-4} × {zero/one/few-shot} 共 6 设置多数投票标注立场(5级)+教条程度(4级)，并用 7 个 SLM 微调/指令微调建立基线。

**[VITAL: A New Dataset for Benchmarking Pluralistic Alignment in Healthcare](vital_pluralistic_alignment_healthcare.md)**

:   本文构建了首个面向医疗健康领域的多元化对齐（pluralistic alignment）基准数据集 VITAL，包含 13.1K 价值观情境和 5.4K 多选题，并通过对 8 个 LLM 的广泛评估表明，现有多元化对齐技术（尤其是 ModPlural）在医疗场景下表现不佳，简单的 prompting 反而效果更好。

**[When to Speak, When to Abstain: Contrastive Decoding with Abstention](when_to_speak_when_to_abstain.md)**

:   提出 CDA（Contrastive Decoding with Abstention），一种免训练解码方法，通过熵校准的不确定性估计让 LLM 在参数/上下文知识可用时生成正确回答、在两者都不可靠时主动弃权，覆盖全部四种知识可用性场景。

**[X-Turing: Towards an Enhanced and Efficient Turing Test for Long-Term Dialogue Agents](xturing_enhanced_turing_test.md)**

:   提出 X-Turing 框架，通过引入 burst 对话模式和伪对话生成技术来增强和高效化图灵测试，能够评估 LLM 在长期对话中的人类模仿能力，发现 LLM 随着对话轮次增加表现显著下降。
