<!-- 由 src/gen_blog_index.py 自动生成 -->
# 💡 LLM 推理

**🤖 AAAI2026** · 共 **33** 篇

**[A Reasoning Paradigm for Named Entity Recognition](a_reasoning_paradigm_for_named_entity_recognition.md)**

:   提出 ReasoningNER，将命名实体识别从"隐式模式匹配"转变为"显式推理"范式，通过三阶段流程（CoT数据构建→CoT微调→GRPO强化增强）让模型先推理再抽取实体，在零样本设定下F1超GPT-4达12.3个百分点，8B模型在CrossNER上达72.4平均F1。

**[Answering the Unanswerable Is to Err Knowingly: Analyzing and Mitigating Abstention Failures in Large Reasoning Models](answering_the_unanswerable_is_to_err_knowingly_analyzing_and.md)**

:   系统分析大推理模型(LRM)面对不可回答数学题时的弃权失败现象，发现LRM内部有足够认知能力识别问题不可解（探针分类准确率>80%）但外部行为仍偏向强答，提出认知监控+推理时干预的两阶段方法，将弃权率从16-54%提升至60-92%且不损害可回答题的推理性能。

**[ARCHE: A Novel Task to Evaluate LLMs on Latent Reasoning Chain Extraction](arche_a_novel_task_to_evaluate_llms_on_latent_reasoning_chai.md)**

:   提出潜在推理链提取 (ARCHE) 任务，要求 LLM 将科学论文中的论证分解为基于 Peirce 三种推理范式的推理逻辑树 (RLT)，并通过 Entity Coverage 和 Reasoning Edge Accuracy 两个指标揭示了 10 个主流 LLM 在内容完整性与逻辑正确性之间的本质权衡。

**[BadThink: Triggered Overthinking Attacks on Chain-of-Thought Reasoning in Large Language Models](badthink_triggered_overthinking_attacks_on_chain-of-thought_reasoning_in_large_l.md)**

:   提出 BadThink——首个针对 CoT 推理效率的训练时后门攻击，通过 LLM 迭代优化生成自然的冗长推理模板进行数据投毒，触发后模型生成膨胀 17× 以上的推理链（MATH-500），同时保持最终答案正确和良好隐蔽性。

**[BLM-Guard: Explainable Multimodal Ad Moderation with Chain-of-Thought and Policy-Aligned Rewards](blm-guard_explainable_multimodal_ad_moderation_with_chain-of.md)**

:   提出 BLM-Guard，一个面向短视频商业广告的可解释多模态审核框架：先通过 Rule-driven ICoT 数据合成 + SFT 冷启动建立结构化推理能力，再用 Self-Adaptive GRPO 强化学习（结合规则正确性奖励 + 自适应一致性奖励 SCA-R）优化策略对齐，在真实广告 benchmark 上达到 91.4% 严格准确率和 0.845 推理一致性分数。

**[Chain-of-Thought Driven Adversarial Scenario Extrapolation for Robust Language Models](chain-of-thought_driven_adversarial_scenario_extrapolation_for_robust_language_m.md)**

:   提出 ASE（Adversarial Scenario Extrapolation），一种推理时 CoT 防御框架，让 LLM 在回答前自主模拟对抗场景并制定防御策略，在四类安全威胁（越狱、毒性、幻觉、偏见）上实现近零攻击成功率，同时将直接拒绝率降至≤4%，兼顾鲁棒性和用户体验。

**[CMMCoT: Enhancing Complex Multi-Image Comprehension via Multi-Modal Chain-of-Thought and Memory Augmentation](cmmcot_enhancing_complex_multi-image_comprehension_via_multi.md)**

:   提出 CMMCoT 框架，通过构建交错的多模态多步推理链（含视觉区域 token 监督）和测试时检索式记忆增强模块（RIFREM），在不增加参数的前提下提升多图场景下的慢思考推理能力，基于 Qwen2.5-VL-7B 在多图基准上平均提升 1.4 分。

**[Deep Hidden Cognition Facilitates Reliable Chain-of-Thought Reasoning](deep_hidden_cognition_facilitates_reliable_chain-of-thought_.md)**

:   本文发现 LLM 在 CoT 推理过程中，中间层的注意力头激活值隐式编码了推理步骤的真实性信息（最高 85% 探测准确率），据此训练置信度预测器引导 Beam Search 动态选择高置信度推理路径，在数学/符号/常识推理任务上超越 Self-Consistency 和 PRM Guided Search。

**[Dropouts in Confidence: Moral Uncertainty in Human-LLM Alignment](dropouts_in_confidence_moral_uncertainty_in_human-llm_alignment.md)**

:   系统研究 32 个开源 LLM 在道德困境（电车问题）中的决策不确定性，发现不确定性主要受模型架构而非道德维度驱动；在推理时引入 attention dropout 增加随机性后，模型的互信息显著上升，human-LLM 道德对齐度也随之改善——表明降低 LLM 在道德场景中的过度自信可以改善与人类偏好的一致性。

**[ESG-Bench: Benchmarking Long-Context ESG Reports for Hallucination Mitigation](esg-bench_benchmarking_long-context_esg_reports_for_hallucination_mitigation.md)**

:   构建 ESG-Bench——270 个人工标注 QA 对来自 94 份真实 ESG 报告（2020-2024），提出三阶段幻觉缓解：SFT（有基础答案+「不提供」弃权标签）→ CoT Prompting（2/4步提示模板）→ CoT 微调（人工推理链），其中 4 步 CoT 微调的 Llama-3 达到 92.52% 有答案准确率 + 99.37% 无答案准确率（平衡 96%），且迁移到 HaluEval/BioASQ 也有提升。

**[Evaluating, Synthesizing, and Enhancing for Customer Support Conversation](evaluating_synthesizing_and_enhancing_for_customer_support_conversation.md)**

:   基于COPC行业标准定义客服对话的5个阶段和12种策略，通过5个LLM Agent角色扮演生成11232条策略丰富的合成对话（RoleCS），并构建1855条真实对话改写的评估集（CSConv），微调后显著提升策略对齐的回复质量和问题解决率。

**[Exposing the Cracks: Vulnerabilities of Retrieval-Augmented LLM-Based Machine Translation](exposing_the_cracks_vulnerabilities_of_retrieval-augmented_llm-based_machine_tra.md)**

:   开发受控噪声注入框架系统评估检索增强翻译（REAL-MT），引入Fidelity和CAR两个新指标，在10语言对×4种噪声类型上揭示模型即使面对矛盾上下文仍盲目采纳（CAR保持65-78%），大推理模型（LRM）反而更脆弱（会"合理化"错误上下文），且噪声鲁棒性与干净上下文利用率存在根本性trade-off。

**[ExtendAttack: Attacking Servers of LRMs via Extending Reasoning](extendattack_attacking_servers_of_lrms_via_extending_reasoning.md)**

:   提出 ExtendAttack，一种针对大推理模型（LRM）的资源耗尽攻击：通过将 prompt 中的字符随机转换为多进制 ASCII 编码，迫使模型在回答问题前先执行大量逐字符解码推理，使 o3 的响应长度增加 2.7 倍以上、延迟翻倍，同时保持答案准确率基本不变。

**[Graph of Verification: Structured Verification of LLM Reasoning with Directed Acyclic Graphs](graph_of_verification_structured_verification_of_llm_reasoning_with_directed_acy.md)**

:   提出 Graph of Verification (GoV)，一种将 LLM 推理过程建模为有向无环图 (DAG) 的结构化验证框架，通过灵活的节点块(Node Block)架构实现多粒度验证——从形式化任务的原子步骤到自然语言叙述的段落级验证——在结构化和松散结构化推理基准上均显著优于整体验证和其他分解验证方法。

**[Improving Value-based Process Verifier via Low-Cost Variance Reduction](improving_value-based_process_verifier_via_low-cost_variance_reduction.md)**

:   针对基于值的过程验证器(PRM)训练中蒙特卡罗(MC)估计因采样数有限导致的高方差问题，提出Compound Monte Carlo Sampling (ComMCS)方法，通过线性组合当前步和后续步的MC估计量来无偏地降低方差，无需额外LLM推理开销，在MATH-500上Best-of-32实验中提升2.2个点。

**[Incorporating Self-Rewriting into Large Language Model Reasoning Reinforcement](incorporating_self-rewriting_into_large_language_model_reasoning_reinforcement.md)**

:   提出Self-Rewriting框架，让LRM在RL训练中对"简单"样本（全部回答正确的query）重写自身推理文本并从中学习，仅增加约10%训练开销即可在保持准确率的同时将推理长度减少46%，内部推理质量（LLM-as-Judge）提升7.2分，有效缓解过度思考、冗余思考等问题。

**[Intention Chain-of-Thought Prompting with Dynamic Routing for Code Generation](intention_chain-of-thought_prompting_with_dynamic_routing_for_code_generation.md)**

:   提出 RoutingGen——基于认知经济原则的难度感知代码生成框架：用 Qwen3-8B 分类器动态路由任务到简单路径（few-shot 直接生成）或复杂路径（Intention CoT = 规格约束 + 算法意图 + 复杂度分析），在 McEval 上提升 +45.15% 同时平均减少 46.37% token 消耗。

**[Jupiter: Enhancing LLM Data Analysis Capabilities via Notebook and Inference-Time Value-Guided Search](jupiter_enhancing_llm_data_analysis_capabilities_via_notebook_and_inference-time.md)**

:   构建NbQA数据集（从真实Jupyter Notebook提取3.8万task-solution对）+ 提出Jupiter框架（将数据分析建模为状态级搜索问题，用值模型引导PUCT搜索），使Qwen2.5-14B在InfiAgent-DABench上达86.38%超越GPT-4o(85.99%)，Qwen2.5-7B在DSBench上从63.51%提升至89.19%。

**[L2V-CoT: Cross-Modal Transfer of Chain-of-Thought Reasoning via Latent Intervention](l2v-cot_cross-modal_transfer_of_chain-of-thought_reasoning_v.md)**

:   通过 LAT 分析发现 LLM 和 VLM 的低频 CoT 方向表示具有相似分布，提出 L2V-CoT：从 LLM 提取 CoT 方向表示 → 低通滤波 → 频域重采样匹配维度 → 注入 VLM 隐藏层，training-free 地将 LLM 的推理能力迁移到 VLM，平均提升 3.7%，最高 8.6%。

**[N2N-GQA: Noise-to-Narrative for Graph-Based Table-Text Question Answering Using LLMs](n2n-gqa_noise-to-narrative_for_graph-based_table-text_question_answering_using_l.md)**

:   提出 N2N-GQA——首个用于开放域混合表格-文本问答的零样本框架，核心思路是将检索到的嘈杂文档构建为动态证据图（文档为节点、TF-IDF共享词为边），通过图中心性剪枝识别"桥接文档"连接多跳推理链，在 OTT-QA 上比 Vanilla RAG 提升 +39.6 EM（从 8.0 到 48.8），零样本即接近微调系统 CORE (49.0 EM)。

**[PRIME: Planning and Retrieval-Integrated Memory for Enhanced Reasoning](prime_planning_and_retrieval-integrated_memory_for_enhanced_reasoning.md)**

:   受双系统认知理论启发，提出PRIME多Agent推理框架——Quick Thinking Agent（System 1）快速生成直觉答案，Reflection Agent评估可信度，不确定时触发System 2的6个专门化Agent（规划/搜索/阅读/假设/整合/决策）进行深度知识检索推理，使开源LLaMA 3在医学/多跳QA上接近GPT-4o性能。

**[ReCode: Updating Code API Knowledge with Reinforcement Learning](recode_updating_code_api_knowledge_with_reinforcement_learning.md)**

:   提出 ReCode 框架，通过基于规则的强化学习（而非 SFT）训练 LLM 在 prompt 中正确利用 API 更新文档完成代码版本迁移，使 7B 模型在 CodeUpdateArena 上超越 32B 模型。

**[Relation-R1: Progressively Cognitive Chain-of-Thought Guided Reinforcement Learning for Unified Relation Comprehension](relation-r1_progressively_cognitive_chain-of-thought_guided_reinforcement_learni.md)**

:   提出 Relation-R1，首个统一二元和 N 元关系理解的框架，通过渐进式认知 CoT 引导的 SFT（模板 CoT → MLLM 生成 CoT）+ GRPO 多奖励优化，在 PSG 数据集上提升 6.84~6.90%，在 SWiG 上也取得 SOTA。

**[RPM-MCTS: Knowledge-Retrieval as Process Reward Model with Monte Carlo Tree Search for Code Generation](rpm-mcts_knowledge-retrieval_as_process_reward_model_with_monte_carlo_tree_searc.md)**

:   提出 RPM-MCTS——用知识库检索替代训练的过程奖励模型（PRM）来指导代码生成的 MCTS 搜索。利用同类算法实现的同质性，从知识库中检索正确算法步骤作为评估信号，配合相似度过滤去除冗余扩展节点和沙箱执行定位错误，实现 ~15% token 减少同时超越 SOTA。

**[SAPO: Self-Adaptive Process Optimization Makes Small Reasoners Stronger](sapo_self-adaptive_process_optimization_makes_small_reasoners_stronger.md)**

:   受神经科学中Error-Related Negativity启发，提出自适应过程优化方法SAPO，通过首错检测+局部后验估计替代低效的逐步蒙特卡洛rollout，在降低2-3倍计算成本的同时实现推理器-验证器协同优化，使小语言模型（≤2B）在数学和代码推理任务上超越多数自演化方法。

**[SCALE: Selective Resource Allocation for Overcoming Performance Bottlenecks in Mathematical Test-time Scaling](scale_selective_resource_allocation_for_overcoming_performance_bottlenecks_in_ma.md)**

:   基于认知科学的双过程理论，提出SCALE框架将数学问题分解为子问题后按难度分配不同计算资源（System 1快速计算 vs System 2深度推理），在AIME25上将Qwen3-32B从57.50%提升至71.25%，同时比InftyThink节省33-53%的token。

**[SERL: Self-Examining Reinforcement Learning on Open-Domain](serl_self-examining_reinforcement_learning_on_open-domain.md)**

:   提出SERL自我改进框架，LLM同时作为Actor（生成者）和Judge（评估者），用Copeland成对比较方法从自身判断中推导奖励信号，无需外部奖励模型或人工标注，使Qwen3-8B在AlpacaEval 2.0上从52.37%提升到59.90%（+7.53%），接近Qwen3-32B水平。

**[SPARE: Single-Pass Annotation with Reference-Guided Evaluation for Automatic Process Supervision](spare_single-pass_annotation_with_reference-guided_evaluation_for_automatic_proc.md)**

:   提出 SPARE 框架，通过单次结构化生成同时完成解题步骤与参考解的对齐和准确性判断（含显式推理），无需额外训练数据，比 MCTS 方法快 2.3 倍且仅需 16% 训练样本即可实现 OOD 泛化。

**[Stable Voting and the Splitting of Cycles](stable_voting_and_the_splitting_of_cycles.md)**

:   研究Simple Stable Voting (SSV)——已在数百次实际选举中使用的递归投票规则——是否总是精化(refine)Split Cycle (SC)方法的猜想，通过数学证明（≤5候选人）和SAT求解（6-7候选人）确定：猜想在≤6候选人时成立，≥7候选人时被反驳，并通过构造性证明推广到任意多候选人。

**[Text-To-Scene With Large Reasoning Models](text-to-scene_with_large_reasoning_models.md)**

:   提出Reason-3D，利用大推理模型（LRM）的多步空间推理能力，通过语义投票式物体检索+双阶段布局（自回归放置+碰撞感知优化）实现从文本到3D场景的零样本生成，在人工评价中Elo评分达2248（远超Holodeck的1500和LayoutVLM的1650）。

**[The Curious Case of Analogies: Investigating Analogical Reasoning in Large Language Models](the_curious_case_of_analogies_investigating_analogical_reasoning_in_large_langua.md)**

:   通过 Patchscopes、注意力屏蔽和线性探针等机制可解释性工具，系统揭示了 LLM 类比推理的内部机制：模型能在中上层有效编码关系信息，但**应用**关系信息到新实体是比**提取**更大的瓶颈；成功的类比推理与故事间强结构对齐相关联，失败则反映弱化或错位的对齐。

**[ToC: Tree-of-Claims Search with Multi-Agent Language Models](toc_tree-of-claims_search_with_multi-agent_language_models.md)**

:   提出 Tree-of-Claims (ToC) 框架，将专利权利要求编辑建模为结构化搜索问题，通过 MCTS 与 EditorAgent/ExaminerAgent 多智能体协作，在新颖性、范围保持和语义一致性之间联合优化，比零/少样本 LLM 基线平均提升约 8% 综合分。

**[Trade-offs in Large Reasoning Models: An Empirical Analysis of Deliberative and Adaptive Reasoning over Foundational Capabilities](trade-offs_in_large_reasoning_models_an_empirical_analysis_of_deliberative_and_a.md)**

:   系统评估了LRM（如DeepSeek-R1、QwQ、OpenThinker等）在获取深度推理能力后对基础能力（helpfulness和harmlessness）的负面影响，发现deliberative reasoning显著降低指令遵循和安全性能力，并提出Zero-Thinking、Less-Thinking、Summary-Thinking等自适应推理模式可有效缓解这些缺陷。
