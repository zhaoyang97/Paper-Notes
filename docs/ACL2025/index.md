<!-- 由 src/gen_blog_index.py 自动生成 -->
# 💬 ACL2025 论文笔记

共 **708** 篇笔记，覆盖 **31** 个领域。

## 领域概览

| 领域 | 篇数 |
|:-----|-----:|
| 💬 [LLM / NLP](#llm_nlp) | 218 |
| 🧩 [多模态 VLM](#multimodal_vlm) | 50 |
| ⚖️ [对齐 / RLHF](#llm_alignment) | 48 |
| 🦾 [LLM Agent](#llm_agent) | 40 |
| 📦 [模型压缩](#model_compression) | 38 |
| 💡 [LLM 推理](#llm_reasoning) | 34 |
| ⚡ [LLM 效率](#llm_efficiency) | 21 |
| ✍️ [文本生成](#nlp_generation) | 20 |
| 📖 [NLP 理解](#nlp_understanding) | 18 |
| 🛡️ [AI 安全](#ai_safety) | 17 |
| 🏥 [医学图像](#medical_imaging) | 12 |
| 🎵 [音频/语音](#audio_speech) | 10 |
| 🕸️ [图学习](#graph_learning) | 10 |
| 🔎 [AIGC 检测](#aigc_detection) | 8 |
| 🔗 [因果推理](#causal_inference) | 4 |
| 🎨 [图像生成](#image_generation) | 4 |
| 🎯 [目标检测](#object_detection) | 4 |
| 🎮 [强化学习](#reinforcement_learning) | 4 |
| 🤖 [机器人/具身智能](#robotics) | 3 |
| 🔄 [自监督/表示学习](#self_supervised) | 3 |
| 📈 [时间序列](#time_series) | 3 |
| 🧑 [人体理解](#human_understanding) | 2 |
| 🖼️ [图像恢复](#image_restoration) | 2 |
| 🎁 [推荐系统](#recommender) | 2 |
| ✂️ [语义分割](#segmentation) | 2 |
| 🎬 [视频理解](#video_understanding) | 2 |
| 🧊 [3D 视觉](#3d_vision) | 1 |
| 🚗 [自动驾驶](#autonomous_driving) | 1 |
| 📐 [优化/理论](#optimization) | 1 |
| 📡 [信号/通信](#signal_comm) | 1 |
| 📂 [其他](#others) | 125 |

---

## 💬 LLM / NLP { #llm_nlp }

**[A Case Study of Cross-Lingual Zero-Shot Generalization for Classical Languages in LLMs](llm_nlp/a_case_study_of_cross-lingual_zero-shot_generalization_for_classical_languages_i.md)**

:   系统评估 LLM（GPT-4o/Llama-3.1）在三种古典语言（梵语、拉丁语、古希腊语）上的零样本跨语言泛化能力——涵盖 NER、机器翻译、问答三个 NLU 任务，发现大模型在域外数据上可比肩甚至超越微调基线，模型规模是决定性因素，并贡献了一个 1501 对的梵语事实问答数据集。

**[A Large-Scale Real-World Evaluation of LLM-Based Virtual Teaching Assistant](llm_nlp/a_large-scale_real-world_evaluation_of_llm-based_virtual_teaching_assistant.md)**

:   在 KAIST 477 名研究生的 AI 编程课上部署基于 GPT-4o-mini + RAG 的虚拟教学助手（VTA），通过三轮大规模问卷调查和 3,869 条交互日志分析系统性评估 VTA 的有效性与接受度，发现 VTA 在编程和概念问题上有效但信任度随时间下降。

**[A Survey of Automatic Prompt Optimization with Instruction-focused Heuristic-based Search Algorithm](llm_nlp/a_survey_of_automatic_prompt_optimization_with_instruction-focused_heuristic-bas.md)**

:   系统综述基于启发式搜索算法的自动 Prompt 优化方法——提出五维分类体系（Where/What/What criteria/Which operators/Which algorithms），覆盖优化空间、目标、评价标准、候选生成算子和搜索算法，并综述支撑数据集和工具框架。

**[A Survey of Large Language Models in Psychotherapy: Current Landscape and Future Directions](llm_nlp/a_survey_of_large_language_models_in_psychotherapy_current_landscape_and_future_.md)**

:   系统综述 LLM 在心理治疗中的应用——提出评估（Assessment）→ 诊断（Diagnosis）→ 治疗（Treatment）三阶段概念分类法，覆盖症状检测、诊断推理、治疗对话策略等，揭示当前研究失衡（聚焦常见障碍、语言偏差、方法碎片化、理论整合不足），并提出连续多阶段建模和实时自适应系统等未来方向。

**[A Survey of LLM-based Agents in Medicine: How Far Are We from Baymax?](llm_nlp/a_survey_of_llm-based_agents_in_medicine_how_far_are_we_from_baymax.md)**

:   系统综述 LLM-based Agent 在医学中的应用——分析 Agent 架构（系统配置/临床规划/医学推理/外部能力增强）、应用场景（临床决策/文档/训练模拟/服务优化）和评估框架，覆盖 60 篇研究，识别幻觉管理、多模态整合、部署障碍和伦理问题等关键挑战。

**[A Survey on Efficient Large Language Model Training: From Data-centric Perspectives](llm_nlp/a_survey_on_efficient_large_language.md)**

:   首个系统性的数据高效 LLM 后训练综述，提出"数据价值飞轮"分类法，将方法分为五大类（数据选择、质量增强、合成生成、蒸馏压缩、自演进生态），覆盖 100+ 篇代表性工作并展望未来方向。

**[A Survey on Proactive Defense Strategies Against Misinformation in Large Language Models](llm_nlp/a_survey_on_proactive_defense_strategies_against_misinformation_in_large_languag.md)**

:   系统综述 LLM 主动防御错误信息的策略——提出"三支柱"框架：(1) 知识可信度（训练数据质量+知识编辑+RAG）, (2) 推理可靠性（自对齐+解码策略）, (3) 输入鲁棒性（对抗攻击防御+输入净化）。127 种技术的分类映射，48 项基准研究的元分析显示主动防御比传统检测方法提升 42-63%。

**[A Systematic Study of Compositional Syntactic Transformer Language Models](llm_nlp/a_systematic_study_of_compositional_syntactic_transformer_language_models.md)**

:   对组合型句法 Transformer 语言模型（Compositional SLMs）进行系统研究——识别四个关键设计维度（句法树二叉化、线性化方向、组合函数、子成分掩码），提出统一框架涵盖 16 种变体（含 13 种全新变体），在语言建模、句法泛化、摘要、对话和推理效率上全面评估，给出多条设计建议。

**[AAD-LLM: Neural Attention-Driven Auditory Scene Understanding](llm_nlp/aad-llm_neural_attention-driven_auditory_scene_understanding.md)**

:   提出意图感知听觉场景理解（II-ASU）范式和 AAD-LLM 原型系统——通过颅内脑电（iEEG）解码听者正在关注哪个说话人，将注意力状态注入听觉 LLM（Qwen2-Audio），使模型在多说话人场景中生成与听者感知对齐的回答，在描述/转录/提取/问答四个任务上主观和客观评估均优于无注意力感知的基线。

**[AbGen: Evaluating Large Language Models in Ablation Study Design and Evaluation for Scientific Research](llm_nlp/abgen_evaluating_large_language_models_in.md)**

:   提出 AbGen——首个评估 LLM 设计消融实验能力的基准（1500 条专家标注数据来自 807 篇 NLP 论文），发现最强 LLM (DeepSeek-R1) 与人类专家差距 14.4%，且 LLM-as-Judge 评分与人类评估严重不一致。

**[Ad-hoc Concept Forming in the Game Codenames as a Means for Evaluating Large Language Models](llm_nlp/ad-hoc_concept_forming_in_the_game_codenames_as_a_means_for_evaluating_large_lan.md)**

:   以桌游 Codenames 作为 LLM 评测工具——LLM 分别扮演线索给出者（Spymaster）和猜测者（Field Operative），通过控制词频/歧义性/具体性/风险等级/对手速度等变量系统评估 LLM 的临时概念形成、语义关联、合作推理和语用能力，发现 o3-mini 和 Claude-3.5 领先但所有模型在高风险和抽象词条件下均显著退化。

**[AD-LLM: Benchmarking Large Language Models for Anomaly Detection](llm_nlp/ad-llm_benchmarking_large_language_models_for_anomaly_detection.md)**

:   首个系统评估 LLM 在 NLP 异常检测中角色的基准 AD-LLM——覆盖三个关键任务：(1) 零样本检测（LLM 预训练知识直接做 AD），(2) 数据增强（生成合成数据/类别描述提升 AD 模型），(3) 模型选择（LLM 推荐无监督 AD 模型）。多数据集实验发现 LLM 零样本 AD 表现出色，精心设计的增强有用，但模型选择的可解释性仍是挑战。

**[Adaptive-VP: A Framework for LLM-Based Virtual Patients that Adapts to Trainees' Dialogue](llm_nlp/adaptive-vp_a_framework_for_llm-based_virtual_patients_that_adapts_to_trainees_d.md)**

:   提出 Adaptive-VP——基于 LLM 的虚拟病人对话生成框架，根据护理学员的沟通质量动态调整虚拟病人行为（沟通差→升级敌意，沟通好→缓和），包含案例开发管线+评估模块+动态适应模块+对话生成模块+安全监控模块五个组件，专家护士评估显示其交互自然度和真实感显著优于现有方法。

**[ToxEdit: Adaptive Detoxification Safeguarding General Capabilities of LLMs through Toxicity-Aware Knowledge Editing](llm_nlp/adaptive_detoxification_safeguarding_general_capabilities_of_llms_through_toxici.md)**

:   提出 ToxEdit——毒性感知的知识编辑方法，在前向传播中动态检测毒性激活模式（SVM 二分类器检测有害隐藏状态），将计算路由到原始 FFN 或编辑后的 FFN，实现自适应去毒而不过度编辑。增强 SafeEdit 基准加入指令遵从评估，在多个 LLM 上去毒能力和通用能力保留均 SOTA。

**[AfroBench: How Good Are Large Language Models on African Languages?](llm_nlp/afrobench_how_good_are_large_language_models_on_african_languages.md)**

:   构建 AfroBench——首个大规模非洲语言 LLM 多任务评测基准，覆盖 64 种非洲语言、15 个任务、22 个数据集（9 NLU + 6 生成 + 6 知识QA + 1 数学推理），系统对比提示式 LLM 与微调 BERT/T5 基线，发现非洲语言与英语之间存在巨大性能差距，且性能与单语资源可用性密切相关。

**[AIMSCheck: Leveraging LLMs for AI-Assisted Review of Modern Slavery Statements Across Jurisdictions](llm_nlp/aimscheck_modern_slavery.md)**

:   提出 AIMSCheck——使用 LLM 辅助审查企业现代奴隶制声明是否合规的端到端框架，构建英国和加拿大的新标注数据集（AIMS.uk/AIMS.ca），三层分解合规评估增强可解释性，在澳大利亚数据上训练的模型能有效跨司法管辖泛化到英国和加拿大。

**[Algorithmic Fidelity of Large Language Models in Generating Synthetic German Public Opinions: A Case Study](llm_nlp/algorithmic_fidelity_german_opinion.md)**

:   使用德国纵向选举研究（GLES）的开放式调查数据，评估 LLM 在生成反映德国亚群体公共舆论方面的"算法保真度"，发现 Llama2 在建模群体意见方面优于其他 LLM，但对左翼政党支持者的表征好于右翼（如 AfD），且提示中包含更多人口统计变量可改善表现。

**[NOVA: Aligning LLMs to Follow Instructions and Hallucinate Less via Effective Data Filtering](llm_nlp/aligning_large_language_models_to_follow_instructions_and_hallucinate_less_via_e.md)**

:   提出 NOVA 框架——通过过滤掉 LLM "不熟悉"的指令数据来同时减少幻觉和保持指令遵从能力——Internal Consistency Probing（ICP）通过多次自生成回复的隐藏状态一致性评估 LLM 对指令的熟悉度，Semantic Equivalence Identification（SEI）通过语义聚类+投票评估 LLM 对目标回复的熟悉度，再用质量奖励模型确保数据质量。大幅减少幻觉同时保持指令遵从。

**[Alignment Drift in CEFR-prompted LLMs for Interactive Spanish Tutoring](llm_nlp/alignment_drift_in_cefr-prompted_llms_for_interactive_spanish_tutoring.md)**

:   通过 LLM 模拟师生对话实验，发现基于 CEFR 等级的 system prompting 虽然能初步约束 LLM 输出的西班牙语难度，但随着对话轮次增加，这种约束效果逐渐衰减——作者将此现象命名为"alignment drift"，表明仅靠提示工程不足以支撑长期的自适应语言教学。

**[Answer When Needed, Forget When Not: Language Models Pretend to Forget via In-Context Knowledge Unlearning](llm_nlp/answer_when_needed_forget_when_not_language_models_pretend_to_forget_via_in-cont.md)**

:   提出"上下文知识遗忘"（In-Context Knowledge Unlearning）——通过引入遗忘 token（`<<UNL>>...<<UNL>>`）使 LLM 在推理时根据查询上下文选择性地"遗忘"特定知识，微调后 LLM 在 TOFU/AGE/RWKU 上达到 95% 遗忘准确率同时保留 80% 无关知识。更深入的内部分析发现 LLM 在中间层仍生成正确答案，仅在最后一层决定"假装遗忘"。

**[Revisiting Common Assumptions about Arabic Dialects in NLP](llm_nlp/arabic_dialects_assumptions_revisited.md)**

:   系统验证了阿拉伯语 NLP 中四个被广泛接受的假设，通过 978 个方言句子+33 名标注者的多标签标注数据集证明：56% 的方言句子在多个区域方言中有效，方言词表的区分度被高估，句子长度与方言歧义性的相关性远弱于方言化程度（ALDi），不同方言说话者对同一句子的 ALDi 评级差异显著。

**[ArithmAttack: Evaluating Robustness of LLMs to Noisy Context in Math Problem Solving](llm_nlp/arithmattack_evaluating_robustness_of_llms_to_noisy_context_in_math_problem_solv.md)**

:   提出 ArithmAttack，通过在数学题上下文中随机插入标点符号（不改变任何单词）来测试 LLM 的鲁棒性，发现八个主流 LLM（包括 Llama3、Mistral、DeepSeek）在面对这种简单噪声时性能都显著下降。

**[Astute RAG: Overcoming Imperfect Retrieval Augmentation and Knowledge Conflicts for Large Language Models](llm_nlp/astute_rag_knowledge_conflicts.md)**

:   Astute RAG 提出了一种对不完美检索具有鲁棒性的 RAG 方法，通过自适应生成 LLM 内部知识作为补充、带有来源标注的知识整合、以及基于可靠性的答案生成三个步骤，在 Gemini 和 Claude 上显著优于现有鲁棒 RAG 方法，且是唯一在最坏情况下（检索全部无用）不劣于无 RAG 基线的方法。

**[Atomic Calibration of LLMs in Long-Form Generations](llm_nlp/atomic_calibration_of_llms_in_long-form_generations.md)**

:   系统研究长文本生成中的原子级校准（Atomic Calibration）——将长回复分解为原子主张（atomic claims），为每个主张分配置信度分数，发现回复级校准良好的模型在原子级校准很差，将置信度获取方法分为判别式（内部状态）和生成式（外部评估）两类并发现它们互补，提出两种融合策略达到 SOTA 校准效果。

**[Automating Legal Interpretation with LLMs: Retrieval, Generation, and Evaluation](llm_nlp/atrie_legal_interpretation.md)**

:   提出 ATRIE 框架模拟法学教义研究流程——自动从判例法中检索概念相关信息、解释法律概念、并通过下游任务（法律概念蕴涵LCE）自动评估解释质量，生成的解释在全面性和可读性上与专家相当。

**[AutoGUI: Scaling GUI Grounding with Automatic Functionality Annotations from LLMs](llm_nlp/autogui_scaling_gui_grounding_with_automatic.md)**

:   提出 AutoGUI 自动标注管线——通过模拟交互比较 UI 状态变化 + LLM 推断元素功能 + LLM 验证过滤，构建 704K 高质量 UI 功能标注数据集，标注正确率 96.7% 可比人类，显著提升 VLM 的 UI grounding 能力且展现数据扩展效应。

**[LLM-AT: Automatic Transmission for LLM Tiers Optimizing Cost and Accuracy](llm_nlp/automatic_transmission_for_llm_tiers_optimizing_cost_and_accuracy_in_large_langu.md)**

:   提出 LLM-AT（LLM Automatic Transmission）框架——无需训练即可动态选择 LLM 层级，Starter 通过准确率估计器选择初始层级，Generator 生成回答，Judge 评估有效性，无效则自动升级到更高层级，在不同难度任务上平衡准确率和成本，比始终用顶级模型更高效。

**[Awes, Laws, and Flaws From Today's LLM Research](llm_nlp/awes_laws_and_flaws_from_todays_llm_research.md)**

:   对 2020-2024 年 2000+ 篇 LLM 相关论文进行科学方法论的系统批判——基于统计检验/可复现性/伦理声明等标准评估研究质量，发现多种趋势：伦理声明减少、LLM 作为评估器增多、无人工评估的推理能力声称增多、统计严谨性下降，但会议检查清单（如 ACL 强制 limitations 章节）确实有效缓解部分问题。

**[OPTS: Bandit-Based Prompt Design Strategy Selection Improves Prompt Optimizers](llm_nlp/bandit-based_prompt_design_strategy_selection_improves_prompt_optimizers.md)**

:   首次提出 Prompt 设计策略的显式选择机制 OPTS——将 CoT/角色提示/少样本等多种策略视为多臂老虎机的"臂"，用 Thompson 采样动态选择要应用的策略，集成到 EvoPrompt 后在 BIG-Bench Hard 上将 GPT-4o mini 性能提升最高 50%，超越隐式策略选择（APET）和均匀采样。

**[BanStereoSet: A Dataset to Measure Stereotypical Social Biases in LLMs for Bangla](llm_nlp/banstereoset_a_dataset_to_measure_stereotypical_social_biases_in_llms_for_bangla.md)**

:   构建 BanStereoSet，一个包含 1194 条填空式样本、覆盖 9 类偏见（种族/性别/宗教/职业/美貌/年龄/种姓/地区等）的孟加拉语刻板印象偏见数据集，用于评估多语言 LLM 在孟加拉语中的社会偏见，发现 GPT-4o 偏见最高，Mistral 最低。

**[Batayan: A Filipino NLP Benchmark for Evaluating Large Language Models](llm_nlp/batayan_a_filipino_nlp_benchmark_for_evaluating_large_language_models.md)**

:   提出 Batayan——首个全面的菲律宾语 LLM 评测基准，覆盖理解/推理/生成三大能力的 8 个任务（含 3 个全新菲律宾语任务），由母语者翻译和标注确保语言真实性，评测 50+ 开源和商用 LLM 后发现菲律宾语表现显著落后于英语，显式菲律宾语支持和模型规模的提升均能带来明显增益。

**[Binary Classifier Optimization for Large Language Model Alignment](llm_nlp/bco_binary_classifier_alignment.md)**

:   提出 BCO（Binary Classifier Optimization），从数学上证明二元交叉熵损失是 DPO 损失的上界，使 LLM 对齐仅需"点赞/踩"二元反馈而非成对偏好数据，并通过新颖的 reward shift 技术收紧上界，在配对偏好数据集上与 DPO 持平，在真实 Likert-5 标注数据上优于 DPO 和 KTO。

**[Between Circuits and Chomsky: Pre-pretraining on Formal Languages Imparts Linguistic Biases](llm_nlp/between_circuits_chomsky.md)**

:   提出在自然语言预训练前先在形式语言上进行"pre-pretraining"，发现具有层级依赖结构的形式语言（如 k-Shuffle Dyck）能为 Transformer 提供有效的归纳偏置，使 1B 参数模型以 33% 更少的 token 达到相同的语言建模损失。

**[Beyond In-Context Learning: Aligning Long-form Generation of LLMs via Task-Inherent Attribute Guidelines](llm_nlp/beyond_in-context_learning_aligning_long-form_generation_of_large_language_model.md)**

:   证明 ICL 示例不足以教会 LLM 任务的语言和格式分布，提出 LongGuide 自动生成质量指标和输出约束两种 guidelines 来增强长文本生成。

**[Beyond Profile: From Surface-Level Facts to Deep Persona Simulation in LLMs](llm_nlp/beyond_profile_from_surface-level_facts_to_deep_persona_simulation_in_llms.md)**

:   提出 CharacterBot，通过 4 个训练任务（视角重建预训练 + 选择题/生成式QA/风格迁移微调）和 CharLoRA 参数更新机制，从鲁迅 17 部杂文集中学习其语言风格和深层思想模式，在语言准确性和观点理解上显著超越各基线。

**[Beyond Prompt Engineering: Robust Behavior Control in LLMs via Steering Target Atoms](llm_nlp/beyond_prompt_engineering_robust_behavior_control_in_llms_via_steering_target_at.md)**

:   提出 STA（Steering Target Atoms），利用稀疏自编码器 (SAE) 将 LLM 的表示解耦为原子知识组件，通过激活幅度和频率筛选目标原子并操控，实现比提示工程更鲁棒、更精细的行为控制，在安全解毒和推理控制任务上效果优于现有 steering 方法。

**[Bias Attribution in Filipino Language Models: Extending a Bias Interpretability Metric for Application on Agglutinative Languages](llm_nlp/bias_attribution_in_filipino_language_models_extending_a_bias_interpretability_m.md)**

:   将信息论偏见归因分数指标扩展到黏着语（菲律宾语），通过对子词分数取均值来处理复杂词素结构，在 4 个多语言 PLM 上揭示菲律宾语模型的偏见由实体类主题词（人物/物品/关系）驱动，与英语中动作类主题词（犯罪/性行为）形成鲜明对比。

**[Bias in Language Models: Beyond Trick Tests and Towards RUTEd Evaluation](llm_nlp/bias_in_language_models_beyond_trick_tests_ruted_evaluation.md)**

:   通过对比标准偏见基准（"trick tests"）与基于真实使用场景的 RUTEd 评估，发现标准偏见基准与真实场景中的偏见表现无显著相关性，主张偏见评估应面向具体应用场景。

**[BiasGuard: A Reasoning-Enhanced Bias Detection Tool for Large Language Models](llm_nlp/biasguard_a_reasoning-enhanced_bias_detection_tool_for_large_language_models.md)**

:   提出 BiasGuard，通过显式推理公平性规范来检测 LLM 输出偏见：第一阶段用教师模型生成推理轨迹做 SFT 初始化，第二阶段用 DPO 强化推理质量，在 5 个数据集上超越分类器和 LLM-as-Judge 方法且降低过度公平误判。

**[Blessing of Multilinguality: A Systematic Analysis of Multilingual In-Context Learning](llm_nlp/blessing_of_multilinguality_a_systematic_analysis_of_multilingual_in-context_lea.md)**

:   系统分析多语言 ICL 策略，发现在 prompt 中混合多种高资源语言（HRL）的 demonstrations 一致性优于纯英文 demonstrations，尤其在低资源语言（LRL）上提升显著（Llama3.1 上 LRL 平均准确率提升 8.9~12.6%），甚至仅在 prompt 中加入不相关的非英语句子也能带来可测量的增益，揭示了"多语言暴露本身即有效"的现象。

**[BMIKE-53: Investigating Cross-Lingual Knowledge Editing with In-Context Learning](llm_nlp/bmike-53_investigating_cross-lingual_knowledge_editing_with_in-context_learning.md)**

:   提出 BMIKE-53，覆盖 53 种语言的跨语言上下文知识编辑 (IKE) 基准，统一了 zsRE/CounterFact/WikiFactDiff 三个知识编辑数据集，系统评估发现模型规模和示例对齐对跨语言 IKE 效果至关重要，文字系统类型是影响跨语言性能差异的关键因素。

**[Boosting LLM's Molecular Structure Elucidation with Knowledge Enhanced Tree Search Reasoning](llm_nlp/boosting_llms_molecular_structure_elucidation_with_knowledge_enhanced_tree_searc.md)**

:   提出 K-MSE，通过构建分子子结构知识库补充 LLM 化学知识 + 设计分子-光谱评分器作为奖励模型 + MCTS 树搜索推理框架，在分子结构解析任务上将 GPT-4o-mini 和 GPT-4o 的性能提升超过 20%。

**[Brevity is the soul of sustainability: Characterizing LLM response lengths](llm_nlp/brevity_is_the_soul_of_sustainability_characterizing_llm_response_lengths.md)**

:   系统研究 12 个 LLM 在 5 个数据集上的响应长度行为，发现 LLM 普遍生成远超必要的冗长回复（核心答案仅占 42%），并提出多种提示策略将响应长度缩短 25-88%、推理能耗降低 25-60%，同时保持甚至提升 ROUGE-L F1 质量。

**[Can Large Language Models Understand Internet Buzzwords Through User-Generated Content](llm_nlp/buzzword_understanding_ugc.md)**

:   研究 LLM 能否通过用户生成内容（UGC）理解中文网络流行语——构建首个中文网络流行语数据集 Cheer（含定义和相关UGC），提出 Ress 方法引导 LLM 模拟人类语言学习过程来生成流行语定义，揭示了 LLM 在流行语理解上的三大共性挑战。

**[Automated CAD Modeling Sequence Generation from Text Descriptions via Transformer-Based Large Language Models](llm_nlp/cadllm_cad_modeling_from_text.md)**

:   提出一个语言引导的 CAD 设计自动化框架——通过半自动数据标注流水线、Transformer CAD 生成器（TCADGen）和 LLM 增强模型（CADLLM）三个创新，从文本参数和外观描述自动生成 CAD 建模序列，在精度和效率上超越传统方法。

**[Can External Validation Tools Improve Annotation Quality for LLM-as-a-Judge?](llm_nlp/can_external_validation_tools_improve_annotation_quality_for_llm-as-a-judge.md)**

:   提出 Evaluation Agent，一个工具增强的 LLM-as-a-Judge 框架，通过集成网络搜索（事实核查）、代码执行和数学验证工具，在长文本事实验证上将与人类一致性从 63% 提升到 81%，在编程评估上从 31% 提升到 71%，且对无关领域几乎无退化。

**[Can Input Attributions Explain Inductive Reasoning in In-Context Learning?](llm_nlp/can_input_attributions_explain_inductive_reasoning_in_in-context_learning.md)**

:   设计受控的合成归纳推理任务评估 4 种输入归因方法解释 ICL 的能力，发现最简单的梯度范数常常最好，但所有方法在不同任务和模型规模上表现不一致且不稳定——ICL 的可解释性比预期更难。

**[CER: Confidence Enhanced Reasoning in LLMs](llm_nlp/cer_confidence_enhanced_reasoning.md)**

:   提出置信度增强推理框架 CER——在 CoT 推理的每个中间步骤中量化关键 token（数学任务的数值/开放域的专有名词）的置信度，用步间置信度乘积评估整条推理链的可靠性，用置信度加权聚合替代简单多数投票，在数学和开放域任务上比 self-consistency 分别提升最高 7.4% 和 5.8%。

**[ChainEdit: Propagating Ripple Effects in LLM Knowledge Editing through Logical Rule-Guided Chains](llm_nlp/chainedit_propagating_ripple_effects_in_llm.md)**

:   提出 ChainEdit 框架，通过将知识图谱中挖掘的逻辑规则与 LLM 内在逻辑推理能力对齐，实现知识编辑时的链式更新，将逻辑泛化准确率从约 20% 提升至 58-65%。

**[Enhancing Character-Level Understanding in LLMs through Token Internal Structure Learning](llm_nlp/character_level_understanding.md)**

:   提出 TIPA（Token Internal Position Awareness）方法，通过在 tokenizer 词汇表上进行逆序字符预测训练，增强 LLM 对 token 内部字符结构和位置的感知能力，显著提升中文拼写纠错等字符级任务的表现。

**[Circuit Compositions: Exploring Modular Structures in Transformer-Based Language Models](llm_nlp/circuit_compositions_modular_structures.md)**

:   通过在 PCFG SET 数据集上识别 10 个组合性字符串编辑操作的电路（circuits），研究 Transformer 中功能相关电路之间的模块化关系，发现功能相似的电路具有显著的节点重叠和跨任务忠实度，且电路可以通过集合运算（并集）组合以表示超出单个电路能力的更复杂功能。

**[CKnowEdit: A New Chinese Knowledge Editing Dataset for Linguistics, Facts, and Logic Error Correction in LLMs](llm_nlp/cknowedit_chinese_knowledge_editing_dataset_llms.md)**

:   提出首个面向中文语言特征的知识编辑数据集 CKnowEdit，涵盖语言学、事实性和逻辑性三大类共10个子类的1,854条样本，揭示了当前知识编辑方法在中文场景下的不足。

**[Classifying Unreliable Narrators with Large Language Models](llm_nlp/classifying_unreliable_narrators.md)**

:   借用叙事学理论定义三种不可靠叙事者类型，构建专家标注数据集 TUNa，系统评估 LLM 在零样本、少样本、微调和课程学习设定下的分类能力，发现该任务极具挑战性且课程学习对小模型有显著提升。

**[CodeReviewQA: The Code Review Comprehension Assessment for Large Language Models](llm_nlp/codereviewqa_the_code_review_comprehension_assessment_for_large_language_models.md)**

:   提出 CodeReviewQA 基准，将代码审查自动修正（ACR）任务分解为三个中间推理步骤——变更类型识别（CTR）、变更定位（CL）、解决方案识别（SI），各自设计为不同难度的多选题探测，在 900 个人工验证的高质量样例（9 种语言）上评测 72 个 LLM，揭示了模型在代码审查理解中的具体弱点。

**[CogniBench: A Legal-inspired Framework and Dataset for Assessing Cognitive Faithfulness of Large Language Models](llm_nlp/cognibench_cognitive_faithfulness.md)**

:   借鉴法律领域间接证据认定标准，提出分层评估框架和 CogniBench 数据集，首次系统性地定义和评估 LLM 在认知性陈述（推理、评价、解释）中的忠实度问题，并训练 CogniDet 检测器实现事实与认知幻觉的同时检测。

**[Enough Coin Flips Can Make LLMs Act Bayesian](llm_nlp/coin_flips_bayesian.md)**

:   通过受控的有偏硬币抛掷实验，证明 LLM 在获得足够的上下文示例后能以贝叶斯方式更新其先验，但初始先验通常存在系统性偏差（偏向正面），且注意力幅度对贝叶斯推理影响甚微。

**[Revisiting Compositional Generalization Capability of Large Language Models Considering Instruction Following Ability](llm_nlp/compositional_generalization_instruction.md)**

:   提出 Ordered CommonGen 基准，通过要求 LLM 按指定顺序生成包含所有概念的句子，同时评估组合泛化与指令遵循能力，在 36 个 LLM 上发现即使最强模型也仅能达到约 75% 的有序覆盖率。

**[Computation Mechanism Behind LLM Position Generalization](llm_nlp/computation_mechanism_behind_llm_position_generalization.md)**

:   揭示 LLM 注意力 logit 学习了位置相关性和语义重要性的近似算术加法解耦（$W_{i,j} \approx f(\mathbf{q}, i-j) + g(\mathbf{q}, \mathbf{k})$，线性相关 0.959），发现了使这种解耦成立的中间表示模式，并用此解释了 LLM 的位置排列容忍性和长度泛化能力。

**[How Humans and LLMs Organize Conceptual Knowledge: Exploring Subordinate Categories in Italian](llm_nlp/conceptual_knowledge_org.md)**

:   通过构建首个意大利语下位类别心理语言学数据集（187 个基本类别），系统对比了人类和 LLM 在下位概念层级上的类别组织结构，发现两者的对齐度较低但在不同语义领域存在显著差异。

**[ConsistencyChecker: Tree-based Evaluation of LLM Generalization Capabilities](llm_nlp/consistencychecker_tree_evaluation.md)**

:   ConsistencyChecker 提出基于自一致性树（self-consistency tree）的无参考 LLM 评估框架，通过构建可逆变换的树状多步路径（如多语言往返翻译、代码等价重写），量化模型在迭代变换中的语义/功能保持能力，动态生成 benchmark 从根源消除数据泄露，且与 WMT 2024 权威排名的相关性 r > 0.7，证明无需配对数据即可可靠评估 LLM 泛化能力。

**[Context-Robust Knowledge Editing for Language Models](llm_nlp/context-robust_knowledge_editing_for_language_models.md)**

:   发现现有知识编辑方法在前缀上下文存在时大幅失败（编辑成功率从 90.9% 降至 69.1%），提出 CHED 基准评估上下文鲁棒性，并设计 CoRE 方法通过多样化前缀上下文 + 跨前缀隐藏状态方差正则化来增强编辑的上下文鲁棒性，在保持模型通用能力的同时显著缩小有/无上下文的性能差距。

**[Contrastive Perplexity for Controlled Generation: An Application in Detoxifying Large Language Models](llm_nlp/contrastive_perplexity_controlled_gen.md)**

:   提出基于原型对比困惑度（Contrastive Perplexity, CP）的框架，通过构造语义相似但毒性属性不同的正负样本对，在困惑度空间中进行对比学习来微调 LLM，实现显著的毒性降低（Mistral-7b 毒性从 33.1% 降至 4.3%）且几乎不影响下游任务性能。

**[Contrastive Prompting Enhances Sentence Embeddings in LLMs through Inference-Time Steering](llm_nlp/contrastive_prompting_embeddings.md)**

:   提出对比提示（Contrastive Prompting, CP）方法，通过引入辅助提示（引导编码非核心信息）并在推理时与正常提示的激活值做对比减法，过滤掉停用词等无关语义，使 LLM 的句子嵌入更聚焦核心语义，在 STS 和分类任务上一致提升现有提示方法。

**[COSMIC: Generalized Refusal Direction Identification in LLM Activations](llm_nlp/cosmic_generalized_refusal_direction_identification_in_llm_activations.md)**

:   提出 COSMIC（Cosine Similarity Metrics for Inversion of Concepts），一种基于余弦相似度的自动化方向选择框架，无需依赖模型输出 token 或预定义的拒绝模板即可在 LLM 激活空间中识别拒绝方向，在对抗场景和弱对齐模型中仍能有效执行拒绝引导。

**[Cross-Lingual Transfer of Debiasing and Detoxification in Multilingual LLMs: An Extensive Investigation](llm_nlp/cross-lingual_transfer_of_debiasing_and_detoxification_in_multilingual_llms_an_e.md)**

:   在 7 个 LLM 和 20 种语言上系统研究了英语去偏见/去毒化微调的跨语言迁移效果，发现 SFT 有效去偏见、DPO 有效去毒化，但迁移到非英语语言时普遍伴随语言生成能力下降（语言一致性、流畅度、多样性均受损），迁移效果可由预训练数据中目标语言的数据量预测。

**[Cross-model Transferability among Large Language Models on the Platonic Representations of Concepts](llm_nlp/cross_model_transferability_sv.md)**

:   提出 L-Cross Modulation 方法，通过简单线性变换将一个 LLM 的概念方向向量（steering vectors）迁移到另一个 LLM 中实现行为控制，发现三个关键结论：(1) 跨模型 SV 迁移有效；(2) 不同概念共享同一变换矩阵；(3) 小模型的 SV 可以控制大模型（弱到强迁移）。

**[Cross-Lingual Pitfalls: Automatic Probing Cross-Lingual Weakness of Multilingual Large Language Models](llm_nlp/crosslingual_pitfalls.md)**

:   提出基于束搜索和 LLM 仿真的自动化方法来高效发现多语言 LLM 的跨语言弱点，构建了覆盖 16 种语言的 6000+ 双语问答对数据集，揭示即使 GPT-4o 也存在超过 30% 的跨语言性能下降。

**[Data Caricatures: On the Representation of African American Language in Pretraining Corpora](llm_nlp/data_caricatures_on_the_representation_of_african_american_language_in_pretraini.md)**

:   结合定量实验、人工判断和定性分析，系统评估了 12 个开源预训练语料库中非裔美国人语言（AAL）的数量与质量：发现 AAL 仅占 0.007%–0.18% 的文档（远低于人口比例），C4 中 28.9% 的 AAL 文本被判为不适合 LLM 生成、24.5% 强化有害刻板印象，且 16 种自动过滤器中有 13 种系统性地偏向保留白人主流英语（WME）而非 AAL。

**[Data Whisperer: Efficient Data Selection for Task-Specific LLM Fine-Tuning via Few-Shot In-Context Learning](llm_nlp/data_whisperer_data_selection.md)**

:   Data Whisperer 提出一种无需训练的注意力加权 few-shot ICL 数据选择方法，利用预训练模型自身的 ICL 能力和注意力分数为训练样本打分，仅用 10% 数据即可超越全量微调性能，同时比现有方法快 7-20 倍。

**[DeAL: Decoding-time Alignment for Large Language Models](llm_nlp/deal_decoding_time_alignment.md)**

:   DeAL 将 LLM 对齐问题重新形式化为解码时的启发式搜索问题，在推理阶段利用可定制的奖励函数（包括程序化约束和参数化 reward model）引导 token 选择，实现了灵活的多目标对齐且可与 RLHF 互补叠加。

**[Defense Against Prompt Injection Attack by Leveraging Attack Techniques](llm_nlp/defense_prompt_injection.md)**

:   本文提出一种"以攻为防"的 prompt injection 防御策略：将已有的攻击技术（ignore、escape、fake completion）反转用于防御，在被注入的数据内容后追加 shield prompt + 原始指令，使 LLM 忽略注入指令而执行原始指令，在多种攻击场景下将 ASR 降至接近零。

**[When People are Floods: Analyzing Dehumanizing Metaphors in Immigration Discourse with Large Language Models](llm_nlp/dehumanizing_metaphors_immigration.md)**

:   提出结合 LLM 词级隐喻检测与 SBERT 篇章级语义关联的计算框架，在 40 万条美国移民推文上揭示保守派更多使用去人化隐喻、但生物类隐喻对自由派的用户互动效应更强的复杂图景。

**[Deontological Keyword Bias: The Impact of Modal Expressions on Normative Judgments of Language Models](llm_nlp/deontological_keyword_bias.md)**

:   本文揭示LLM存在"义务论关键词偏见"(DKB)——当提示中包含"must"、"ought to"等模态义务表达时，模型会将超过90%的常识场景误判为义务，并提出基于少样本示例与推理提示的去偏策略。

**[Refuse Whenever You Feel Unsafe: Improving Safety in LLMs via Decoupled Refusal Training](llm_nlp/derta_decoupled_refusal.md)**

:   发现标准安全微调数据存在"拒绝位置偏差"——模型只学会在回答开头拒绝，中途发现不安全时无法中断。提出 DeRTa（Decoupled Refusal Training），通过"有害前缀+安全拒绝"的 MLE 训练和在每个位置模拟"从有害到安全"转换的 RTO 训练，让 LLM 能在回答的任何位置感知到不安全时拒绝，在六种攻击场景下超越 GPT-4 和 LLaMA3-Instruct。

**[Disentangling Language and Culture for Evaluating Multilingual Large Language Models](llm_nlp/disentangle_language_culture.md)**

:   提出 Dual Evaluation Framework，将多语言 LLM 评估沿"语言媒介"和"文化语境"两个维度解耦，发现"文化-语言协同"(Cultural-Linguistic Synergy) 现象——模型在文化语境与提问语言对齐时表现更好，并通过 FFN 神经元激活分析从可解释性角度给出解释。

**[Disentangling Memory and Reasoning Ability in Large Language Models](llm_nlp/disentangle_memory_reasoning.md)**

:   提出将 LLM 的推理过程显式分解为"记忆回忆"和"逻辑推理"两个步骤——引入 `<memory>` 和 `<reason>` 两个可学习特殊 token 标记每步是知识回忆还是逻辑推理，用双 LLM 框架生成训练数据后 LoRA 微调，在 StrategyQA/CommonsenseQA/TruthfulQA 上提升性能并增强可解释性，8B 模型在 TruthfulQA 上超越 GPT-4o。

**[DIVE into MoE: Diversity-Enhanced Reconstruction of Large Language Models from Dense into Mixture-of-Experts](llm_nlp/dive_moe_reconstruction.md)**

:   提出 DIVE，一种将 Dense LLM 重构为 MoE 架构的方法，核心洞察是不同领域的校准数据集会让结构化剪枝产生不同的剪枝结果，利用这种多样性构建领域特异的专家，配合高效的两阶段重训练（router dense训练 + expert LoRA稀疏训练），在仅调不到 1% 参数的情况下实现优于现有剪枝和 MoE 重构方法的效果。

**[Diversity-oriented Data Augmentation with Large Language Models](llm_nlp/diversity_data_augmentation.md)**

:   提出 DoAug 框架，通过 SFT+DPO 微调 LLM 释义器并结合核心集选择与多样性采样，在保持语义一致性的同时显著提升增强数据集的多样性，在 12 个数据集上平均性能提升 10.52%，超出次优基线 3.76 个百分点。

**[Diversity Explains Inference Scaling Laws: Through a Case Study of Minimum Bayes Risk Decoding](llm_nlp/diversity_explains_inference_scaling_laws_through_a_case_study_of_minimum_bayes_.md)**

:   从 bias-diversity 分解的理论视角重新解释 MBR 解码：质量估计误差 MSE = Bias - Diversity，增加 diversity（伪参考的多样性）是提升 MBR 性能的关键；进一步通过信息论扩展到一般推理方法，揭示 diversity 是推理 scaling law（增加采样提升性能但边际递减）的理论根源，并在机器翻译、摘要、图像描述任务上实证验证。

**[Do Large Language Models Perform Latent Multi-Hop Reasoning without Exploiting Shortcuts?](llm_nlp/do_large_language_models_perform_latent_multi-hop_reasoning_without_exploiting_s.md)**

:   构建无快捷方式的评估数据集 SOCRATES，系统评估 41 个 LLM 在潜在多跳推理中的真实能力，发现模型在国家类桥接实体上可达 80% 组合率，但年份类仅约 5%。

**[Does Time Have Its Place? Temporal Heads Where Language Models Recall Time-specific Information](llm_nlp/does_time_have_its_place_temporal_heads_where_language_models_recall_time-specif.md)**

:   通过 EAP-IG 电路分析在 Llama-2/Qwen/Phi-3 中发现了专门处理时间条件知识的"时间头"（Temporal Heads），消融这些头只降低时间知识准确率（降 3-9%）而不影响时间无关知识和通用 QA，并展示了通过注入时间头激活值实现选择性时间知识编辑的可能性。

**[ECLM: Entity Level Language Model for Spoken Language Understanding with Chain of Intent](llm_nlp/eclm_entity_level_language_model_spoken_language_understanding.md)**

:   提出 ECLM 框架，将 LLM 应用于多意图口语理解：通过将 token 级槽填充转化为实体识别任务解决序列对齐问题，引入"意图链"（Chain of Intent）实现逐步多意图识别，在 MixATIS 和 MixSNIPS 上大幅超越 SOTA 基线。

**[EdiText: Controllable Coarse-to-Fine Text Editing with Diffusion Language Models](llm_nlp/editext_diffusion_text_editing.md)**

:   提出 EdiText，基于嵌入扩散模型（LD4LG）的可控文本编辑框架，通过将 SDEdit 技术从图像域迁移到文本域实现粗粒度编辑（控制加噪时间步），并创新性地利用自条件化（self-conditioning）机制实现细粒度编辑（将参考文本嵌入注入为去噪条件），两者结合实现从粗到细的多粒度文本属性编辑。

**[ELABORATION: A Comprehensive Benchmark on Human-LLM Competitive Programming](llm_nlp/elaboration_competitive_programming.md)**

:   提出ELABORATION——首个全面评估人类-LLM协作竞赛编程的基准，包含覆盖编程全流程（理解→规划→编码→调试）的人类反馈分类体系和8320题精标注数据集，实验表明LLM在困难题上仅3.4% Pass@1，但人类反馈（特别是在编码阶段）可平均提升9.3%。

**[ELI-Why: Evaluating the Pedagogical Utility of Language Model Explanations](llm_nlp/eli-why_evaluating_the_pedagogical_utility_of_language_model_explanations.md)**

:   构建了包含 13.4K "Why" 问题的 ELI-Why 基准，通过两项人类研究发现 GPT-4 生成的面向不同教育水平的解释仅 50% 能匹配目标年级（人工策划达 79%），且对学习者信息需求的满足度比人类答案低 20%。

**[Emergent Abilities of Large Language Models under Continued Pretraining for Language Adaptation](llm_nlp/emergent_abilities_continued_pt.md)**

:   揭示了持续预训练（CPT）进行语言适应时，混入英文数据对保留模型上下文学习（ICL）能力和下游涌现能力至关重要——尽管不影响验证困惑度；并提出课程学习和 EMA 权重平均作为替代方案。

**[Growing Through Experience: Scaling Episodic Grounding in Language Models](llm_nlp/episodic_grounding_experience.md)**

:   提出一个 weak-to-strong episodic grounding 框架，利用 MCTS 收集结构化经验数据，通过行为比率蒸馏将小模型的 episodic grounding 能力迁移到大模型，结合 DPO 优化实现从成功和失败经验中学习，在物理规划任务上超越 GPT-4o 等 SOTA 模型 3.45%。

**[Efficient and Accurate Prompt Optimization: the Benefit of Memory in Exemplar-Guided Reflection](llm_nlp/erm_prompt_optimization_memory.md)**

:   提出 ERM 方法，通过指导性元提示生成带详细解题过程的 exemplar 来增强 feedback 质量，并引入 Feedback Memory 和 Exemplar Factory 两种长期记忆机制来高效存储和复用历史反馈与示例，在多个任务上以约一半的优化步数超越了 SOTA prompt 优化方法。

**[EscapeBench: Towards Advancing Creative Intelligence of Language Model Agents](llm_nlp/escapebench_creative_agent.md)**

:   提出EscapeBench密室逃脱游戏基准（36个场景、3种难度）评估LM agent的创造性智能，并设计EscapeAgent通过Foresight（工具使用假设生成）和Reflection（未解任务追踪）模块将提示依赖降低约50%，但仍远落后于人类。

**[Evaluating Language Models as Synthetic Data Generators](llm_nlp/evaluating_lms_synthetic_data_gen.md)**

:   提出 AgoraBench 基准，系统评估 6 个 LLM 在 3 个领域×3 种数据生成方式下的数据生成能力，通过训练 99 个学生模型发现：LLM 的数据生成能力与问题求解能力不直接相关，GPT-4o 在实例生成上最强而 Claude-3.5-Sonnet 在质量增强上最强。

**[LLMs Can Simulate Standardized Patients via Agent Coevolution](llm_nlp/evopatient_standardized_patient.md)**

:   EvoPatient 提出了一个多智能体协同进化框架，通过患者 Agent 和医生 Agent 之间的自主模拟对话，让 LLM 无需人工监督即可学会模拟标准化病人（SP），在需求对齐度上超过现有推理方法 10%+。

**[Exploring Explanations Improves the Robustness of In-Context Learning](llm_nlp/exploring_explanations_improves_the_robustness_of_in-context_learning.md)**

:   提出 X²-ICL 框架，通过在上下文学习的示例中为所有可能的标签（而非仅观测标签）生成解释推理路径，系统性地探索隐变量推理空间，从而显著提升 ICL 在分布外（OOD）数据上的鲁棒性——在 5 个 LLM 上的 8 个 OOD 数据集中，X²-ICL 在 6-8 个上超越 ICL 和 X-ICL。

**[Exploring Graph Representations of Logical Forms for Language Modeling](llm_nlp/exploring_graph_representations_of_logical_forms_for_language_modeling.md)**

:   提出 GFoLDS，一种在 DMRS 逻辑形式图表示上预训练的图 Transformer 语言模型，并提出"语言知识催化假说"(LKCH)：逻辑形式语言模型几乎立刻学会基础语言现象，进而加速复杂模式学习，在相同数据量下大幅超越 BERT。

**[HiCUPID: Exploring the Potential of LLMs as Personalized Assistants](llm_nlp/exploring_the_potential_of_llms_as.md)**

:   提出 HiCUPID，首个全面满足个性化 AI 助手五大需求（用户信息遵循、隐含信息理解、多信息推理、长上下文建模、主动性回复）的基准，含 1,250 用户 × 25 人格 × 10 日程 + Llama-3.2 自动评估模型。

**[FoodTaxo: Generating Food Taxonomies with Large Language Models](llm_nlp/foodtaxo_generating_food_taxonomies_with_large_language_models.md)**

:   提出 FoodTaxo，基于 Llama-3 的迭代自底向上分类法生成与补全算法，利用 CoT 提示 + RAG 检索 + NLI 验证三阶段流程，从已知叶节点概念出发逐步构建层次化 taxonomy；在五个基准数据集上与 TacoPrompt 等 SOTA 方法竞争，同时通过 reference-free 指标和消融实验揭示了非叶节点放置这一根本性瓶颈。

**[A Survey on Foundation Language Models for Single-cell Biology](llm_nlp/foundation_lm_single_cell_survey.md)**

:   首篇从语言建模视角系统综述单细胞生物学基础语言模型的工作，将现有模型划分为 PLM（从头预训练）和 LLM（利用已有大模型）两大类，全面分析了数据 tokenization 策略、预训练/微调范式以及下游任务。

**[FR-Spec: Accelerating Large-Vocabulary Language Models via Frequency-Ranked Speculative Sampling](llm_nlp/fr_spec_speculative_sampling.md)**

:   发现大词表LLM（如LLaMA-3的128k词表）中投机采样的瓶颈从Transformer层转移到LM Head，提出FR-Spec通过频率排序将草稿模型的词表压缩75%（128k→32k），在EAGLE-2基础上额外获得1.12×加速，且保证最终输出分布数学等价。

**[From Selection To Generation A Survey](llm_nlp/from_selection_to_generation_a_survey.md)**

:   首篇系统综述 LLM 时代的主动学习（Active Learning），提出以 Querying（选择/生成）和 Annotation（标注）为核心的分类体系，全面梳理 LLM 如何变革传统主动学习的选择-标注流程。

**[GALLa: Graph Aligned Large Language Models for Improved Source Code Understanding](llm_nlp/galla_graph_aligned_large_language_models.md)**

:   提出 GALLa，通过 GNN 编码代码的 AST/DFG 结构图并用跨模态适配器对齐到 LLM 嵌入空间，在微调时作为辅助任务注入代码结构信息，推理时丢弃 GNN 和 adapter 实现零额外开销，在 5 个代码任务 × 7 个基线 LLM（350M-14B）上持续提升。

**[Gradient-Adaptive Policy Optimization: Towards Multi-Objective Alignment of Large Language Models](llm_nlp/gapo_multi_objective_alignment.md)**

:   提出GAPO，一种基于梯度自适应缩放的多目标策略优化方法，利用多梯度下降算法(MGDA)结合梯度归一化，平衡LLM在帮助性和无害性等冲突目标间的权衡，并通过P-GAPO支持用户偏好驱动的Pareto前沿生成。

**[GAPO: Learning Preferential Prompt through Generative Adversarial Policy Optimization](llm_nlp/gapo_preferential_prompt.md)**

:   提出 GAPO（Generative Adversarial Policy Optimization）框架，将 GAN 的对抗训练机制与 PPO 结合，使用 encoder-only 奖励模型替代传统 decoder-only 架构，通过"Preferential Prompt"（修改 prompt 中的约束而非 response）的新范式来增强 LLM 对细粒度约束的理解和遵循能力，在 IFEval 和产品描述生成任务上大幅超越 DPO/KTO/SimPO 等基线。

**[Generative Psycho-Lexical Approach for Constructing Value Systems in Large Language Models](llm_nlp/generative_psycholexical_approach_for_constructing_value.md)**

:   提出生成式心理词汇方法（GPLA），自动化构建面向LLM的五因素价值体系（社会责任、冒险性、规则遵循、自我效能、理性），在结构效度、安全预测和价值对齐上优于经典Schwartz人类价值体系。

**[Towards Geo-Culturally Grounded LLM Generations](llm_nlp/geocultural_grounded_llm.md)**

:   研究 RAG/搜索增强技术对 LLM 文化意识的影响——搜索增强显著提升了文化命题知识的选择题表现，但也增加了刻板印象风险，且在开放式文化流畅性的人工评估中改进不显著，揭示了"文化知识"和"文化流畅性"的本质区别。

**[Geometric Signatures of Compositionality Across a Language Model's Lifetime](llm_nlp/geometric_compositionality_lifetime.md)**

:   通过将数据集的组合性程度与语言模型表示的非线性内在维度(I_d)和线性有效维度(d)联系起来，揭示了一个形式-意义二分：非线性 I_d 编码有意义的组合语义复杂度，而线性 d 编码表面词形复杂度；该对应关系在训练过程中随语言能力涌现而建立。

**[Efficient Universal Goal Hijacking with Semantics-guided Prompt Organization](llm_nlp/goal_hijacking_attack.md)**

:   本文提出POUGH方法，通过高效的渐进式优化算法和两种语义引导的提示组织策略（采样策略+排序策略），实现了对LLM的高效通用目标劫持攻击，在四个开源LLM和十种恶意目标响应上平均攻击成功率达93.41%。

**[What Makes a Good Natural Language Prompt?](llm_nlp/good_natural_language_prompt.md)**

:   通过元分析150+篇prompting相关论文和博客，提出一个以属性为中心、以人为中心的prompt质量评估框架，涵盖6个维度21个属性，并发现单属性增强往往比多属性组合更有效。

**[GORP: Continual Gradient Low-Rank Projection Fine-Tuning for LLMs](llm_nlp/gorp_continual_gradient_projection.md)**

:   GORP 提出将全秩参数和 LoRA 低秩参数的梯度统一投影到低秩梯度子空间中联合更新，利用 Adam 一阶矩隐式构建跨任务共享梯度空间来缓解灾难性遗忘，在 T5 和 LLaMA2 上持续学习性能接近多任务联合训练上界。

**[Can Graph Descriptive Order Affect Solving Graph Problems with LLMs?](llm_nlp/graph_descriptive_order_llm.md)**

:   首次系统研究了图描述顺序（BFS、DFS、PageRank、PPR）对LLM解决图推理问题的影响，发现有序描述显著优于随机描述，且不同任务偏好不同的排列策略。

**[Group then Scale: Dynamic Mixture-of-Experts Multilingual Language Model](llm_nlp/group_then_scale_dynamic_mixture-of-experts_multilingual_language_model.md)**

:   提出 DMoE——基于参数偏差的动态语言分组 + 选择性 MoE 层扩展方法，通过仅 10 步微调量化语言间相似性，将相似语言分组共享同一 expert，只在参数偏差大的层（语言特定层）扩展为 MoE 层，在 18~128 种语言上 PPL 比持续预训练降低 11.4%，用 3.6 倍少的参数超越 X-ELM 9.6%。

**[GuessArena: Guess Who I Am? A Self-Adaptive Framework for Evaluating LLMs in Domain-Specific Knowledge and Reasoning](llm_nlp/guessarena_guess_who_i_am_a.md)**

:   提出 GuessArena，一种基于"猜猜我是谁"博弈游戏的自适应 LLM 评估框架，通过领域知识建模和多轮交互推理，在五个垂直行业中有效区分模型的领域知识和推理能力。

**[Hallucination Detox: Sensitivity Dropout (SenD) for Large Language Model Training](llm_nlp/hallucination_detox_send.md)**

:   本文揭示了 LLM 训练过程中幻觉行为的振荡现象，提出 Sensitivity Dropout（SenD）训练协议——通过识别并确定性丢弃高变异敏感嵌入索引来降低训练中的幻觉方差，同时提出计算高效的 Efficient EigenScore（EES）近似方法，在 Pythia 和 Llama 模型上实现高达 17% 的测试时可靠性提升。

**[HALoGEN: Fantastic LLM Hallucinations and Where to Find Them](llm_nlp/halogen_hallucinations.md)**

:   提出 HALoGEN——覆盖 9 个领域（含编程、科学引用、摘要等）的 10,923 条 prompt 的大规模幻觉评测框架，配套原子级自动验证器，在 14 个 LLM 的约 150,000 条生成上系统性评估幻觉，发现即使最佳模型也可能有高达 86% 的原子事实存在幻觉，并提出 Type A/B/C 三类错误分类法。

**[HellaSwag-Pro: A Large-Scale Bilingual Benchmark for Evaluating the Robustness of LLMs](llm_nlp/hellaswag-pro_a_large-scale_bilingual_benchmark_for_evaluating_the_robustness_of.md)**

:   构建首个大规模双语（中英）LLM 常识推理鲁棒性评估基准 HellaSwag-Pro，包含 7 种问题变体共 11,200 道题，系统评估 41 个 LLM 发现所有模型在常识推理上远未达到鲁棒。

**[Help Me Write a Story: Evaluating LLMs' Ability to Generate Writing Feedback](llm_nlp/help_write_story_feedback.md)**

:   探索 LLM 能否为创意写作者提供有意义的写作反馈——构建包含 1300 个故意引入写作问题的故事测试集，评估常用 LLM 的写作反馈生成能力，发现模型虽能提供具体且多数准确的反馈，但常错过最重要的写作问题且不会恰当地在批评和鼓励之间切换。

**[How Do LLMs Acquire New Knowledge? A Knowledge Circuits Perspective on Continual Pre-Training](llm_nlp/how_do_llms_acquire_new_knowledge_a_knowledge_circuits_perspective_on_continual_.md)**

:   从知识电路（knowledge circuits）角度研究 LLM 在持续预训练中如何获取新知识：新知识的获取依赖于与已有知识的关联性，电路经历"形成→优化"的阶段转变，且呈现从深层到浅层的演化模式。

**[How does Misinformation Affect Large Language Model Behaviors and Preferences?](llm_nlp/how_does_misinformation_affect_large_language.md)**

:   构建了目前最大的误信息评估基准 MisBench（1034 万条误信息），从知识冲突类型和文本风格两个维度系统分析 LLM 对误信息的行为和偏好，并提出 RtD 方法结合外部知识源提升误信息检测能力。

**[How Numerical Precision Affects Arithmetical Reasoning Capabilities of LLMs](llm_nlp/how_numerical_precision_affects_arithmetical_reasoning_capabilities_of_llms.md)**

:   从电路复杂度理论出发，严格证明低精度（如 int4/int8）Transformer 在迭代加法和整数乘法上需要超多项式规模才能求解，而标准精度（float32）Transformer 仅需常数深度+多项式宽度即可高效求解三类算术任务，并在 LLaMA-3.1-8B 上实验验证了精度对算术能力的关键影响。

**[HPSS: Heuristic Prompting Strategy Search for LLM Evaluators](llm_nlp/hpss_heuristic_prompting_strategy_search_for_llm_evaluators.md)**

:   整合 8 个影响 LLM 评估提示效果的关键因子（评分尺度、ICL 示例、评估标准、参考答案、CoT、AutoCoT、度量指标、组件顺序），提出基于遗传算法的启发式提示策略搜索方法 HPSS，在 12,960 种组合空间中高效找到最优提示策略，仅用基线 5% 的生成成本即超越 G-Eval 和 CloserLook。

**[How to Enable Effective Cooperation Between Humans and NLP Models: A Survey of Principles, Formalizations, and Beyond](llm_nlp/human_nlp_cooperation_survey.md)**

:   首次系统性地综述了人与NLP模型之间的合作范式，提出基于"谁为最终决策负责"的三类合作形式化分类体系（顺序合作、分诊合作、联合合作），并借鉴Grice合作原则定义了人机合作的基本原则，为后续研究提供了统一视角。

**[HumT DumT: Measuring and Controlling Human-like Language in LLMs](llm_nlp/humt_dumt_measuring_and_controlling_human-like_language_in_llms.md)**

:   提出基于 GPT-2 对数概率比的文本人类化语气度量 HumT 及其社会感知泛化版 SocioT，在 40 万+偏好样本上发现用户普遍偏好更低人类化的 LLM 输出且人类化语气与社交亲近（r=0.87）、低地位（r=-0.80）、女性化（r=0.47）强相关，进而通过仅 500 对偏好数据的 DPO 微调（DumT）有效降低人类化程度而不损模型性能。

**[HyGenar: An LLM-Driven Hybrid Genetic Algorithm for Few-Shot Grammar Generation](llm_nlp/hygenar_an_llm-driven_hybrid_genetic_algorithm_for_few-shot_grammar_generation.md)**

:   构建 540 个挑战的文法生成数据集，设计 6 种评测指标，提出基于 LLM 驱动的混合遗传算法 HyGenar，显著提升 LLM 从少量示例生成 BNF 文法的能力。

**[Enhancing the Rule Learning Ability of Large Language Model Agent through Induction, Deduction, and Abduction](llm_nlp/idea_enhancing_the_rule_learning_ability_of_large_language_model_agent_through_i.md)**

:   提出 RULEARN benchmark（300 个手工交互式文字环境谜题，涵盖三类场景）和 IDEA 框架（溯因假设生成→演绎计划验证→归纳反馈修正的迭代循环），在 GPT-4o 上达到 50.33% 成功率（+7% vs ReAct baseline），但仍远低于人类 63.33%，细粒度人类评估揭示了 LLM 在假设修正阶段的根本瓶颈。

**[The Impossibility of Fair LLMs](llm_nlp/impossibility_fair_llms.md)**

:   系统分析了多种技术公平性框架（fairness through unawareness、group fairness、fair representations、multi-sided fairness等）在通用LLM上的适用性，论证了每种框架要么在逻辑上无法扩展到通用AI场景、要么在实践中不可行——主要源于非结构化训练数据的敏感属性不可剥离、用例/人群组合的组合爆炸、以及公平性不具备可组合性。

**[Can Indirect Prompt Injection Attacks Be Detected and Removed?](llm_nlp/indirect_prompt_injection_detection.md)**

:   本文系统研究间接 prompt injection 攻击的检测与移除：构建评估基准，发现现有检测模型对间接攻击表现不佳但专门训练的模型可达 99% 准确率，提出分割移除和抽取移除两种方法，并将检测+移除组合为过滤管道，有效降低间接 prompt injection 的攻击成功率。

**[Leveraging Self-Attention for Input-Dependent Soft Prompting in LLMs](llm_nlp/input_dependent_soft_prompting.md)**

:   提出 ID-SPAM，通过在输入 token 嵌入上施加可学习自注意力层并经瓶颈 MLP 生成**输入依赖**的软提示，仅在单层 Transformer 输入端拼接即可超越多种 Soft Prompt 基线，且具备优秀的零样本跨任务/跨领域迁移能力。

**[Beyond Facts: Evaluating Intent Hallucination in Large Language Models](llm_nlp/intent_hallucination_eval.md)**

:   本文提出"意图幻觉"（Intent Hallucination）概念——LLM 在处理复杂多条件查询时遗漏或误解部分意图约束导致的偏离用户意图的生成，构建 FaithQA 基准（20,068 题）和 Constraint Score 评估指标，实验表明意图幻觉在 SOTA 模型中普遍存在且随查询复杂度增加而加剧。

**[IRT-Router: Effective and Interpretable Multi-LLM Routing via Item Response Theory](llm_nlp/irt_router_multi_llm.md)**

:   IRT-Router 借鉴心理测量学的项目反应理论（IRT），将 LLM 视为"考生"、query 视为"考题"，学习多维能力向量和难度/区分度参数实现可解释的多 LLM 路由，在 OOD 场景下达 87%+ 准确率且成本仅为 GPT-4o 的 1/30。

**[Is LLM an Overconfident Judge? Unveiling the Capabilities of LLMs in Detecting Offensive Language with Annotation Disagreement](llm_nlp/is_llm_an_overconfident_judge_unveiling_the_capabilities_of_llms_in_detecting_of.md)**

:   系统评估了多个 LLM 在攻击性语言检测中面对标注分歧时的表现，发现 LLM 在标注者高度一致的样本上表现优异（GPT-4o F1 85.24%）但在低一致度样本上骤降至 57.06%，且模型对不确定样本表现出严重的过度自信；进一步通过 few-shot 和指令微调实验证明，在训练中引入分歧样本可同时提升检测准确率和人-AI 对齐度。

**[JuStRank: Benchmarking LLM Judges for System Ranking](llm_nlp/justrank_llm_judge_system_ranking.md)**

:   首次大规模研究LLM判官在系统排名任务中的表现，提出JuStRank基准，揭示实例级判断能力与系统级排名能力之间的差距，并发现判官的"果断性"和"偏见"两个新兴特征。

**[Analyzing LLMs' Knowledge Boundary Cognition Across Languages Through the Lens of Internal Representations](llm_nlp/knowledge_boundary_crosslingual.md)**

:   通过探测 LLM 内部表示，揭示知识边界认知在多语言间呈线性结构，提出 training-free 对齐方法实现跨语言知识边界感知迁移，并发现"弱到强泛化"现象。

**[Knowledge Boundary of Large Language Models: A Survey](llm_nlp/knowledge_boundary_survey.md)**

:   本文提出了 LLM 知识边界的形式化定义框架，将知识分为四种类型（PAK/PSK/MSU/MAU），并围绕"为什么研究知识边界""如何识别知识边界""如何缓解知识边界问题"三个核心问题系统综述了相关研究。

**[La Leaderboard: A Large Language Model Leaderboard for Spanish Varieties and Languages of Spain and Latin America](llm_nlp/la_leaderboard_spanish.md)**

:   构建首个面向西班牙和拉丁美洲语言的开源LLM排行榜，整合66个数据集覆盖西班牙语、加泰罗尼亚语、巴斯克语、加利西亚语，评估50个模型并分析训练策略、算力与性能的关系。

**[LangSAMP: Language-Script Aware Multilingual Pretraining](llm_nlp/langsamp_multilingual_pretraining.md)**

:   提出 LangSAMP 方法，在多语言预训练中将语言和文字系统 (script) embedding 添加到 Transformer 输出端（而非输入端），使模型主干学到更语言中立的表示，在 500+ 语言的零样本跨语言迁移中一致优于基线。

**[Language-Codec: Bridging Discrete Codec Representations and Speech Language Models](llm_nlp/language_codec_bridging_discrete_codec_speech_language_models.md)**

:   提出 Language-Codec，通过掩码通道残差向量量化（MCRVQ）机制和改进的傅里叶变换解码器，弥合离散编解码器表示与下游语音语言模型之间的鸿沟，仅用4个码本通道即实现高质量音频重建。

**[LazyReview: A Dataset for Uncovering Lazy Thinking in NLP Peer Reviews](llm_nlp/lazyreview_peer_review.md)**

:   构建首个标注"懒惰思维"细粒度类别的同行评审数据集 LazyReview——发现 LLM 在零样本下难以检测评审中的懒惰思维启发式，但在 LazyReview 上指令微调后性能提升 10-20 个点，且经懒惰思维反馈修改的评审显著更全面和可操作。

**[Less, but Better: Efficient Multilingual Expansion for LLMs via Layer-wise Mixture-of-Experts](llm_nlp/less_but_better_efficient_multilingual_expansion.md)**

:   分析 LLM 不同层间的跨语言表征相似度，提出 LayerMoE 按层分配不同数量的新语言专家（高相似层少分配、低相似层多分配），用 60% 更少的专家参数超越 SOTA，并通过在高相似层添加路由分类器进一步缓解灾难性遗忘。

**[On the Limit of Language Models as Planning Formalizers](llm_nlp/limit_llm_planning_formalizer.md)**

:   系统评估"LLM-as-Formalizer"方法论的极限——首次要求 LLM 生成完整 PDDL 表示（而非部分），从不同自然度的文本描述中形式化规划领域，发现最强模型（GPT-4o/o3-mini/DeepSeek-R1）可有效形式化超越直接规划，但描述越自然性能越低，弱模型卡在语法错误而强模型面临语义错误。

**[Llama See, Llama Do: A Mechanistic Perspective on Contextual Entrainment and Distraction in LLMs](llm_nlp/llama_see_llama_do_entrainment.md)**

:   本文发现并定义了"上下文夹带"(contextual entrainment)现象——LLM会对上下文中出现过的任意token赋予更高概率，并通过可微掩码方法定位了负责该现象的entrainment heads，关闭这些头后可显著抑制干扰效应。

**[LLM as Effective Streaming Processor: Bridging Streaming-Batch Mismatches with Group Position Encoding](llm_nlp/llm_as_effective_streaming_processor_bridging_streaming-batch_mismatches_with_gr.md)**

:   系统性地识别并量化了 batch-trained LLM 适配流式场景的三种不匹配（输入注意力 / 输出注意力 / 位置 ID），发现仅输入注意力不匹配才是关键瓶颈（+2.20 BLEU），据此提出组位置编码（Group Position Encoding）——源/目标各自维护连续位置 ID 即可，无需昂贵的 KV cache 重编码，在机器翻译和 ASR 两种跨模态任务上均超越专用流式架构。

**[LLM Braces: Straightening Out LLM Predictions with Relevant Sub-Updates](llm_nlp/llm_braces_straightening.md)**

:   LLMBraces 通过计算 FFN 层中各 value 向量与输入的相关性得分，动态调节子更新（sub-update）的贡献权重，用极少参数（比 LoRA 少 75%）同时提升模型预测精度和实现可控文本生成。

**[LLM as a Broken Telephone: Iterative Generation Distorts Information](llm_nlp/llm_broken_telephone.md)**

:   类比"传话游戏"研究 LLM 在迭代生成中的信息失真现象，通过多语言翻译链实验发现：信息失真随迭代累积，受中间语言选择和链复杂度影响，可通过温度控制和受限提示缓解但无法消除。

**[Evaluation of LLM Vulnerabilities to Being Misused for Personalized Disinformation Generation](llm_nlp/llm_personalized_disinformation.md)**

:   系统评估了 6 个主流 LLM 生成个性化虚假信息的能力，发现大多数 LLM 能生成高质量个性化虚假新闻，且个性化请求反而降低了安全过滤器的触发率（相当于一种 jailbreak），同时轻微降低了机器生成文本的可检测性。

**[Re-ranking Using Large Language Models for Mitigating Exposure to Harmful Content on Social Media Platforms](llm_nlp/llm_reranking_harmful_content.md)**

:   提出基于 LLM 的成对偏好重排序方法，在零样本和少样本设置下对社交媒体推荐序列中的有害内容进行降级排序，显著优于 Perspective API 和 OpenAI Moderation API 等工业级分类器，同时引入 PP-k 和 EWN 两个新评估指标。

**[Wait, that's not an option: LLMs Robustness with Incorrect Multiple-Choice Options](llm_nlp/llm_robustness_incorrect_mcq.md)**

:   提出"反思判断力"（Reflective Judgment）概念来衡量 LLM 在所有选项都错误的选择题中拒绝选择的能力，发现对齐后的模型（GPT-4o 等）往往盲目服从指令选择错误选项，而基座模型反而表现更好，且该能力随模型规模增大而涌现。

**[LLM-Powered Test Case Generation for Detecting Bugs in Plausible Programs](llm_nlp/llm_test_case_gen_bugs.md)**

:   提出TrickCatcher——一种LLM驱动的测试用例生成方法，通过PUT引导的程序变体生成、基于生成器的输入生成和多样性驱动的差异测试三阶段流程，专门检测"plausible programs"（能通过现有测试套件但仍含隐蔽bug的程序）中的tricky bugs，F1分数达到SOTA基线的1.66倍。

**[LLMs instead of Human Judges? A Large Scale Empirical Study across 20 NLP Evaluation Tasks](llm_nlp/llm_vs_human_judges_study.md)**

:   构建包含20个NLP数据集（7万+实例）的 Judge-Bench 基准，系统评估11个LLM作为评判者与人类标注的一致性，发现模型在不同任务/属性/标注者专业水平上表现差异巨大，建议部署前必须针对特定任务做人类标注验证。

**[LLMs can Perform Multi-Dimensional Analytic Writing Assessments](llm_nlp/llm_writing_assessment.md)**

:   利用 L2 研究生文献综述语料库，系统评估了 LLM 在多维分析写作评估（评分+评论）上的能力，并提出可解释的反馈质量评估框架 ProEval。

**[Language Models, Graph Searching, and Supervision Adulteration: When More Supervision is Less and How to Make More More](llm_nlp/lm_graph_search_supervision.md)**

:   本文证明了 path-star 图搜索任务在 decoder-only LM 上的失败并非 next-token prediction 范式的根本缺陷，而是由"监督污染"（supervision adulteration）导致的——过量的 teacher-forcing 监督信号诱导模型学到 Clever Hans Cheat 捷径，阻碍了子任务分解；通过 token masking、ranking-into-the-future、scratchpad、树形拓扑等六种正交方法均可使任务可学。

**[Locate-and-Focus: Enhancing Terminology Translation in Speech Language Models](llm_nlp/locateandfocus_enhancing_terminology_translation_in_speech.md)**

:   提出Locate-and-Focus方法用于语音LLM的术语翻译：先用滑动窗口检索定位语音中包含术语的片段，再通过音频替换和Tag Cue引导模型聚焦翻译知识，在英中/英德方向上术语翻译成功率大幅提升。

**[Mapping 1,000+ Language Models via the Log-Likelihood Vector](llm_nlp/mapping_1000_models_loglikelihood.md)**

:   提出用对数似然向量（log-likelihood vector）将 1000+ 语言模型映射到一个统一空间，证明向量间欧氏距离近似 KL 散度，可实现模型聚类可视化、基准性能预测（r=0.96）和数据泄漏检测。

**[MAPS: Motivation-Aware Personalized Search via LLM-Driven Consultation Alignment](llm_nlp/maps_personalized_search.md)**

:   首次建模电商搜索中的"搜索动机"——用户在搜索前的咨询行为蕴含的真实需求，提出MAPS框架融合LLM语义、MoAE池化和双重对齐机制，在真实商业数据上HR@10提升24.4%（从0.5685到0.7071）。

**[Marco-Bench-MIF: On Multilingual Instruction-Following Capability of Large Language Models](llm_nlp/marco_bench_multilingual_if.md)**

:   将英文IFEval基准扩展到30种语言并进行文化本地化，揭示LLM在多语言指令遵循中高/低资源语言间25-35%的准确率差距，以及机器翻译数据低估模型性能7-22%。

**[MathFusion: Enhancing Mathematical Problem-solving of LLM through Instruction Fusion](llm_nlp/mathfusion_instruction_fusion.md)**

:   MathFusion 提出了跨问题指令融合的数学数据增强框架，通过顺序融合、并行融合和条件融合三种策略将两个数学问题合成新问题，仅用 45K 额外合成指令就在 6 个 benchmark 上平均提升 18 分准确率。

**[Math Neurosurgery: Isolating Language Models' Math Reasoning Abilities Using Only Forward Passes](llm_nlp/mathneuro_math_reasoning_isolation.md)**

:   MathNeuro 提出了一种仅用前向传播就能隔离 LLM 中数学推理特定参数的方法，通过计算权重×激活的重要性分数并过滤掉通用语言任务也需要的参数，实现了精准的数学能力"手术"——剪除这些参数删除数学能力，缩放它们则提升 4-35% 数学性能。

**[Mechanistic Interpretability of Emotion Inference in Large Language Models](llm_nlp/mechanistic_interpretability_of_emotion_inference_in_large_language_models.md)**

:   通过 probing、activation patching 和 generation steering 三种机制可解释性技术，发现 LLM 的情感表征功能性地定位于中间层的 MHSA 单元，并基于认知评估理论（appraisal theory）证明这些表征具有心理学合理性，成功通过干预评估概念（如 self-agency、pleasantness）引导情感输出。

**[MEraser: An Effective Fingerprint Erasure Approach for Large Language Models](llm_nlp/meraser_fingerprint_erasure.md)**

:   提出 MEraser（Mismatched Eraser），通过两阶段微调策略（错配数据擦除 + 干净数据恢复）以不到 1000 条样本完全移除 LLM 中基于后门的指纹水印，同时保持模型性能，并首创可迁移的 LoRA 擦除适配器。

**[MergePrint: Merge-Resistant Fingerprints for Robust Black-box Ownership Verification of Large Language Models](llm_nlp/mergeprint_fingerprint_ownership.md)**

:   提出 MergePrint，首个针对模型合并（model merging）场景的 LLM 黑盒指纹验证方法，通过伪合并模型模拟合并行为并两阶段优化（输入优化 + 参数优化），使嵌入的指纹在合并后仍可被检测，实现高效、无害、抗篡改的所有权验证。

**[Meta-rater: A Multi-dimensional Data Selection Method for Pre-training Language Models](llm_nlp/metarater_a_multidimensional_data_selection_method.md)**

:   提出Meta-rater多维数据选择框架，定义PRRC四个质量维度（专业性/可读性/推理性/清洁度），通过proxy模型回归学习多个质量分数的最优加权组合，使1.3B模型训练收敛速度翻倍、下游任务提升3.23%。

**[Multi-Level Explanations for Generative Language Models](llm_nlp/mexgen_multi_level_explanations.md)**

:   提出 MExGen（Multi-Level Explanations for Generative Language Models），将 LIME/SHAP 等扰动式归因方法扩展到 LLM 的上下文生成任务上——为摘要/问答等任务中 LLM 输出的每个部分量化上下文各段落的影响程度，比 LLM 自解释更忠实。

**[MHA2MLA: Towards Economical Inference by Enabling DeepSeek's Multi-Head Latent Attention in Any Transformer-based LLMs](llm_nlp/mha2mla_deepseek_latent_attention.md)**

:   MHA2MLA 首次提出将已训练好的 MHA 模型高效迁移到 DeepSeek 的 MLA 架构的方法，通过贡献度感知的 partial-RoPE 移除和联合 SVD 低秩近似，仅用 0.6%-1% 的训练数据即可恢复性能，将 Llama2-7B 的 KV cache 压缩 92.19% 且 LongBench 性能仅下降 1%。

**[Middle-Layer Representation Alignment for Cross-Lingual Transfer in Fine-Tuned LLMs](llm_nlp/mid_layer_crosslingual_alignment.md)**

:   通过分析 1000+ 语言对发现 LLM 中间层具有最强的跨语言对齐潜力，提出在任务训练中集成中间层对齐目标（对比损失），在槽填充（F1 61.7%）、机器翻译（BLEU 32.3）和结构化文本生成上显著提升跨语言迁移，对未见语言也有效。

**[Comparing Moral Values in Western English-speaking Societies and LLMs with Word Associations](llm_nlp/moral_values_western.md)**

:   提出通过词语联想（word association）而非直接提问来比较 LLM 与西方英语社会的道德价值观，发现 LLM 在正面道德维度上与人类更一致，但在情感多样性和具体性上存在系统性差异。

**[Delving into Multilingual Ethical Bias: The MSQAD with Statistical Hypothesis Tests for Large Language Models](llm_nlp/msqad_multilingual_ethical_bias.md)**

:   提出多语言敏感问答数据集MSQAD（基于Human Rights Watch 17个人权话题），通过McNemar检验和PERMANOVA检验两种统计假设检验方法，系统验证了LLM在不同语言下对相同敏感问题的回答存在显著伦理偏差——中文和印地语拒绝率最高，西班牙语和德语最容易生成不当回答，且该偏差在7个不同LLM中普遍存在。

**[Multi-Prompting Decoder Helps Better Language Understanding](llm_nlp/multi-prompting_decoder_helps_better_language_understanding.md)**

:   提出 Multi-Prompting Decoder（MPD）框架，通过多提示查询 PLM 获取多组隐状态和类别分数，结合最优传输匹配和校准解码策略，在 MaaS（模型即服务）场景下的 few-shot 分类任务上显著超越现有方法。

**[Multi-Attribute Steering of Language Models via Targeted Intervention](llm_nlp/multi_attribute_steering.md)**

:   提出 MAT-Steer，通过属性感知的 token 级 gating 机制和正交性约束，实现推理时对 LLM 多属性（如真实性、毒性、偏见）的同时精准干预，在 QA 和生成任务上全面超越现有 ITI 和微调方法。

**[Do Large Language Models Have an English Accent? Evaluating and Improving the Naturalness of Multilingual LLMs](llm_nlp/multilingual_llm_english_accent.md)**

:   本文揭示多语言 LLM 在非英语语言生成中存在"英语口音"——词汇和句法上偏向英语模式，提出了基于 JSD（词汇分布）和 WL 图核+MMD（句法依赖树）的语料级自然度指标，并通过 DPO 对齐方法有效提升目标语言的自然度。

**[Which of These Best Describes Multiple Choice Evaluation with LLMs? A) Forced B) Flawed C) Fixable D) All of the Above](llm_nlp/multiple_choice_eval.md)**

:   系统性论证了 MCQA（多选题问答）作为 LLM 评测标准格式的三大缺陷——格式本身的局限性、数据集构建质量问题、以及 LLM 在 MCQA 上的特有错误——并从教育测试学中引入改进方案。

**[Natural Language Processing in Support of Evidence-based Medicine: A Scoping Review](llm_nlp/natural_language_processing_in_support_of_evidence-based_medicine_a_scoping_revi.md)**

:   对 129 篇研究进行范围综述，系统梳理了 NLP 技术在循证医学（EBM）五步流程（Ask-Acquire-Appraise-Apply-Assess）中的应用，覆盖证据检索、PICO 提取、质量评估、证据合成、摘要生成、临床试验匹配等任务，指出了 LLM 时代的新机遇与挑战。

**[Nemotron-CC: Transforming Common Crawl into a Refined Long-Horizon Pretraining Dataset](llm_nlp/nemotron_cc_pretraining_data.md)**

:   Nemotron-CC 通过分类器集成、合成数据改写和减少启发式过滤三种策略，从 Common Crawl 构建了 6.3T token 的长期预训练数据集，在 15T token 训练中超越 Llama 3.1 8B。

**[NewsInterview: a Dataset and a Playground to Evaluate LLMs' Grounding Gap via Informational Interviews](llm_nlp/newsinterview_a_dataset_and_a_playground_to_evaluate_llms_grounding_gap_via_info.md)**

:   构建了 4 万条新闻采访对话数据集，发现 LLM 在采访场景中缺乏 acknowledgement（少 50%）和话题转换能力（少 30%），并设计了含说服机制的模拟博弈环境（NewsInterview），证明最优 LLM（gpt-4o）也仅能提取 50.4% 的信息项。

**[Nudging: Inference-time Alignment of LLMs via Guided Decoding](llm_nlp/nudging_inference_time_alignment.md)**

:   提出 Nudging，一种免训练的推理时对齐算法，利用小型对齐模型在基础模型不确定时注入少量"nudging tokens"来引导输出，用 7-14 倍小的模型就能达到甚至超过大型对齐模型的性能。

**[On Entity Identification in Language Models](llm_nlp/on_entity_identification_in_language_models.md)**

:   提出基于聚类的评估框架（Purity/Inverse Purity）分析 LLM 内部表示中的实体区分能力，发现实体信息在早期层（~归一化位置 0.2）的 20 维子空间中达到线性可分（F1~0.9），且不同大模型收敛到结构同构的实体编码——为"LLM 从纯文本训练中涌现离散知识结构"提供了系统性证据。

**[Pandora's Box or Aladdin's Lamp: A Comprehensive Analysis Revealing the Role of RAG Noise in Large Language Models](llm_nlp/pandora_box_rag_noise.md)**

:   本文从语言学视角定义了 RAG 系统中的 7 种噪声类型，构建了 NoiserBench 综合评测框架，通过 8 个 LLM 的大规模实验发现噪声可分为有害噪声（反事实、支持性、拼写）和有益噪声（语义、数据类型、非法句子），其中有益噪声反而能提升模型准确率 1-3%。

**[Enhancing Open-Domain Task-Solving Capability of LLMs via Autonomous Tool Integration from GitHub](llm_nlp/paper_2312_17294.md)**

:   提出 OpenAgent 系统，通过自主从 GitHub 发现和集成专业工具来解决开放域任务，并构建 OpenAct 基准评测 LLM 在需要领域特定工具的开放域问题上的能力。

**[Parenting: Optimizing Knowledge Selection of Retrieval-Augmented Language Models with Parameter Decoupling and Tailored Tuning](llm_nlp/parenting_optimizing_knowledge_selection_of_retrievalaugmented.md)**

:   受人脑功能分区启发，提出 Parenting 框架，通过解耦并定位 LLM 参数空间中与"上下文遵循"(adherence)和"噪声鲁棒"(robustness)相关的子空间，并为不同子空间设计定制化微调策略，实现两种能力的平衡提升。

**[Perspective Transition of Large Language Models for Solving Subjective Tasks](llm_nlp/perspective_transition_of_large_language_models_for_solving_subjective_tasks.md)**

:   提出 RPT（Reasoning through Perspective Transition），通过在同一 prompt 中让 LLM 依次探索直接/角色扮演/第三人称三种视角、按置信度排序、选最优视角推理，在 12 个主观任务、4 个模型（GPT-4/GPT-3.5/Llama-3/Qwen-2）上均超越固定视角与集成基线，GPT-3.5 上平均提升 +4.56 点。

**[Pitfalls of Scale: Investigating the Inverse Task of Redefinition in Large Language Models](llm_nlp/pitfalls_of_scale_investigating_the_inverse_task_of_redefinition_in_large_langua.md)**

:   通过重定义任务（给著名物理常数或度量单位赋予新值并要求 LLM 据此推理），揭示大模型比小模型更容易锚定于先验知识，展现规模增长带来推理灵活性下降的逆向缩放现象。

**[PlanGenLLMs: A Modern Survey of LLM Planning Capabilities](llm_nlp/plangenllms_planning_survey.md)**

:   PlanGenLLMs 是一篇系统性综述，基于经典 AI 规划评估框架提出完整性、可执行性、最优性、表示、泛化性和效率六大评估准则，全面梳理了 LLM 作为规划器的方法、评估和未来方向。

**[Plug-in and Fine-tuning: Bridging the Gap between Small Language Models and Large Language Models](llm_nlp/plugin_finetuning_bridge.md)**

:   提出 PiFi 框架，将 LLM 的单个冻结层"插入"到 SLM 中再进行微调，以极低的额外计算成本将 LLM 的语言知识和泛化能力迁移到小模型，在 NLU 和 NLG 任务上均获得一致提升。

**[KoGEM: Polishing Every Facet of the GEM: Testing Linguistic Competence of LLMs and Humans in Korean](llm_nlp/polishing_every_facet_of_the_gem.md)**

:   提出 KoGEM（韩语语法评估基准），包含 1,524 道基于理论语言学分类的多选题，覆盖音韵/形态/句法/语义/规范 5 大类 16 子类，零样本评估 27 个 LLM 并与人类对比，揭示 LLM 在需要经验知识的语言子类（如发音规则、音韵变化）上远逊人类，而显式补充经验知识（发音文本、语素分解）后可大幅提升。

**[Only a Little to the Left: A Theory-grounded Measure of Political Bias in LLMs](llm_nlp/political_bias_theory_grounded.md)**

:   本文用政治科学中经过验证的 World Values Survey (WVS) 替代缺乏科学基础的 Political Compass Test (PCT)，设计 30 种提示变体在 11 个开源/商业 LLM 上收集 88,110 条开放式回复并训练立场分类器自动标注，发现指令微调模型普遍偏左但偏见度量对提示高度敏感，PCT 会夸大特定模型（如 GPT-3.5）的政治偏见。

**[Pragmatics in the Era of Large Language Models: A Survey on Datasets, Evaluation, Opportunities and Challenges](llm_nlp/pragmatics_survey.md)**

:   全面梳理用于评估 NLP 系统语用能力的资源——按语用现象（隐含义、指称、言语行为、会话含义、预设等）分类数据集，分析任务设计、数据收集方法和评估方式，揭示了现代 LLM 在处理语境相关语言使用上的趋势、挑战和空白。

**[Pre³: Enabling Deterministic Pushdown Automata for Faster Structured LLM Generation](llm_nlp/pre3_deterministic_pda_structured_gen.md)**

:   提出 Pre³，将 LR(1) 文法转化为确定性下推自动机（DPDA），通过预计算前缀条件边消除运行时非确定性探索，实现结构化 LLM 生成的显著加速——每 token 耗时降低最高 40%，吞吐提升最高 36%。

**[Aligning Large Language Models with Implicit Preferences from User-Generated Content](llm_nlp/pugc_align_implicit_pref_ugc.md)**

:   提出 PUGC 框架，利用非标注用户生成内容（UGC）中的隐式人类偏好来生成偏好数据——将 UGC 转化为查询+参考文本，以此评分模型生成的响应，用 DPO 实现可扩展的领域特定对齐，在 Alpaca Eval 2 上基于 Mistral-7B 达到 35.93% 长度控制胜率 SOTA。

**[RARE: Retrieval-Augmented Reasoning Enhancement for Large Language Models](llm_nlp/rare_retrieval_augmented_reasoning.md)**

:   提出 RARE，在 rStar 的 MCTS 推理框架中引入两个检索增强动作（A6: 基于原始问题生成搜索查询并检索，A7: 对子问题进行检索并重新回答），并用检索增强的事实性评分器（RAFS）替代原始判别器，使 LLaMA 3.1 在医学和常识推理任务上达到甚至超越 GPT-4o 的水平。

**[Reasoning Circuits in Language Models: A Mechanistic Interpretation of Syllogistic Inference](llm_nlp/reasoning_circuits_in_language_models_a_mechanistic_interpretation_of_syllogisti.md)**

:   用机械可解释性技术（激活补丁 + Logit Lens + 电路消融）发现语言模型中实现三段论推理的完整电路：三阶段机制——长归纳偏差→中间项抑制（h11.10）→传递项移动，该电路在符号输入上既充分又必要，可迁移到自然语言输入，且跨 GPT-2/Pythia/LLaMA/Qwen 四种架构存在兼容模式。

**[Recent Advances in Speech Language Models: A Survey](llm_nlp/recent_advances_in_speech_language_models_a_survey.md)**

:   首篇 Speech Language Models (SpeechLMs) 综合综述，系统梳理从"ASR+LLM+TTS"级联架构到端到端语音语言模型的演进，提出按三大组件（speech tokenizer / language model / vocoder）和训练方案分类的分类体系，覆盖下游能力、评估指标、挑战与未来方向。

**[Recurrent Knowledge Identification and Fusion for Language Model Continual Learning](llm_nlp/recurrent_kif_continual_learning.md)**

:   提出Recurrent-KIF持续学习框架，通过内外循环迭代机制动态估计参数重要性分布，利用基于重要性的二值掩码进行知识融合，有效缓解灾难性遗忘并促进知识迁移。

**[Representation Bending for Large Language Model Safety](llm_nlp/repbend_representation_bending_safety.md)**

:   提出 RepBend，将 activation steering 的核心思想（安全/不安全表示的向量差异）引入 LoRA 微调的损失函数设计，通过"弯曲"模型的表示空间使安全和不安全状态在潜在空间中远离彼此，在多种越狱攻击基准上实现高达 95% 的攻击成功率降低，且对模型通用能力影响极小。

**[Reversal of Thought: Enhancing Large Language Models with Preference-Guided Reverse Reasoning Warm-up](llm_nlp/reversal_of_thought_enhancing_large_language.md)**

:   提出 Reversal of Thought (RoT)，一个即插即用的推理框架，通过偏好引导的逆向推理预热策略，让 LLM 从示例中反向生成"LLM 口味"的最优 prompt，再通过认知偏好管理器自动区分已知/未知任务，在多种推理任务上超越 CoT/ToT/GoT 等基线。

**[REVS: Unlearning Sensitive Information in Language Models via Rank Editing in the Vocabulary Space](llm_nlp/revs_unlearning_sensitive_information_in_language_models_via_rank_editing_in_the.md)**

:   提出 REVS，一种无梯度的模型编辑方法，通过在 FF2 层中定位与敏感 token 关联最强的神经元，将其投影到词汇空间后迭代降低目标 token 排名，在 SSN/Email/URL 三类敏感数据上 Unlearning Score 显著超越 6 种基线（89.58 vs 36.98），同时通用能力几乎零损（MMLU 61.05→60.87），且对 Logit-Lens 和 Delta 提取攻击高度鲁棒。

**[RuleArena: A Benchmark for Rule-Guided Reasoning with LLMs in Real-World Scenarios](llm_nlp/rulearena_rule_guided_reasoning.md)**

:   提出 RuleArena——一个基于航空行李费、NBA交易规则、税务法规三个真实场景的benchmark，用于评估LLM遵循复杂自然语言规则进行推理的能力；实验发现即使最强模型（o1-preview）在最难任务上准确率也不足50%，暴露了LLM在规则召回、规则区分和数学计算三方面的系统性缺陷。

**[Salience Sparse Fine Tuning](llm_nlp/salience_sparse_fine_tuning.md)**

:   首次系统评估 8 种 salience 指标用于稀疏微调（SPEFT）的效果，发现简单的梯度指标 + 静态掩码即可提供最佳性价比，在 GSM8k 上比 LoRA 高出 22.6%，质疑了"复杂方法才能做好 PEFT"的假设。

**[Beware of Your Po! Measuring and Mitigating AI Safety Risks in Role-Play Fine-Tuning of LLMs](llm_nlp/sarft_roleplay_safety.md)**

:   首次系统评估了角色扮演微调（role-play fine-tuning）对 LLM 安全性的影响，发现安全退化程度与角色特质（特别是反派角色）正相关，并提出 SaRFT 框架，通过隐式奖励函数自适应识别对不同角色有害的训练数据子集，配合 KL 散度正则化实现角色表现力与安全性的 Pareto 最优平衡。

**[SConU: Selective Conformal Uncertainty in Large Language Models](llm_nlp/sconu_selective_conformal_uncertainty_in_large_language_models.md)**

:   提出选择性保形不确定性框架 SConU，通过构建保形 p-value 进行显著性检验，首次实现对违反可交换性假设的不确定性数据异常点的自动过滤，从而在单域和跨域 QA 场景中严格管理 LLM 的错误覆盖率。

**[Segment-Level Diffusion: A Framework for Controllable Long-Form Generation with Diffusion Language Models](llm_nlp/segment_level_diffusion.md)**

:   提出段落级扩散（Segment-Level Diffusion, SLD），将长文本输出切分为多个段落（如句子/对话轮次），对每个段落的潜在表示进行扩散建模，结合对比学习和对抗训练增强表示鲁棒性，在摘要、故事生成、对话生成等任务上实现了比现有扩散模型更好的长文本生成质量。

**[Self-Training Elicits Concise Reasoning in Large Language Models](llm_nlp/self-training_elicits_concise_reasoning_in_large_language_models.md)**

:   发现 LLM 输出分布中天然包含简洁推理路径，提出 FS-BoN（Few-shot 条件化 + Best-of-N 采样）自训练框架，从模型自身分布中筛选短且正确的推理样本进行微调，在 GSM8K 和 MATH 上跨 5 个模型族实现平均 30% token 缩减且不损准确率，效率为先前方法 Rational Metareasoning 的 2.4 倍。

**[Self-Tuning: Instructing LLMs to Effectively Acquire New Knowledge through Self-Teaching](llm_nlp/self-tuning_instructing_llms_to_effectively_acquire_new_knowledge_through_self-t.md)**

:   受费曼学习法启发，提出 Self-Tuning 框架，通过记忆-理解-自省三层自教学策略，显著提升 LLM 从新文档中有效获取和回忆知识的能力。

**[SelfElicit: Your Language Model Secretly Knows Where is the Relevant Evidence](llm_nlp/selfelicit_evidence_highlighting.md)**

:   SelfElicit 发现 LLM 深层注意力分数天然能识别上下文中的关键证据（即使回答错误时也是），据此提出推理时自动高亮关键证据句的上下文增强方法，无需训练即可显著提升基于证据的 QA 任务性能。

**[Establishing Trustworthy LLM Evaluation via Shortcut Neuron Analysis](llm_nlp/shortcut_neuron_eval.md)**

:   提出通过对比分析和因果分析定位污染模型中的"捷径神经元"（shortcut neurons），并通过 activation patching 抑制这些神经元，实现更可信的 LLM 评估，与 MixEval 的 Spearman 相关系数超过 0.95。

**[Embracing Imperfection: Simulating Students with Diverse Cognitive Levels Using LLM-based Agents](llm_nlp/simulating_diverse_students.md)**

:   提出一种基于知识图谱认知原型的免训练框架，使LLM Agent能够模拟不同认知水平学生的学习行为（包括错误），在GPT-4o上实现94%的行为预测准确率，相比基线提升100%。

**[SkillVerse: Assessing and Enhancing LLMs with Tree Evaluation](llm_nlp/skillverse_tree_eval.md)**

:   提出 SkillVerse，一种无监督的树结构 LLM 诊断框架——用 LLM-as-Judge 批评模型回答后组织为层次化技能树（dendrogram），可在任意粒度上分析模型能力，并用于改善 ICL（提升 25%）和预测新模型弱点（55% 成功率，高于基线 22pp）。

**[SocialEval: Evaluating Social Intelligence of Large Language Models](llm_nlp/socialeval_evaluating_social_intelligence_of_large_language_models.md)**

:   提出基于叙事脚本的双语社会智能基准 SocialEval，通过"世界树"结构整合结果导向的目标达成评估和过程导向的人际能力评估，全面评测 LLM 的社会智能。

**[SongComposer: A Large Language Model for Lyric and Melody Generation in Song Composition](llm_nlp/songcomposer_llm_lyric_melody_generation.md)**

:   提出 SongComposer，首个能同时生成歌词和旋律的大语言模型，通过元组格式对齐歌词与旋律、标量音高初始化和渐进式结构感知训练，在多个歌曲生成任务上超越 GPT-4。

**[Stochastic Chameleons: Irrelevant Context Hallucinations Reveal Class-Based (Mis)Generalization in LLMs](llm_nlp/stochastic_chameleons_irrelevant_context_hallucinations_reveal_class-based_misge.md)**

:   通过行为分析和机械可解释性实验揭示 LLM 无关上下文幻觉的内部机制：模型在底层构建抽象类别表示（如"语言"），然后两条竞争电路（query-based vs context-based）争夺特征选择权，相对激活强度决定正确泛化还是产生幻觉。

**[StrucText-Eval: Evaluating Large Language Model's Reasoning Ability in Structure-Rich Text](llm_nlp/structext_eval.md)**

:   提出StrucText-Eval——一个覆盖8种结构化语言（JSON/YAML/XML/Markdown/LaTeX/Org/CSV/Tree）和29个任务的自动生成评测基准，共5,800个样本，通过可控的嵌套深度和结构宽度调节难度。实验揭示开源LLM在标准集最高仅74.9%准确率，困难集降至45.8%，而人类在困难集达92.6%，暴露了LLM在复杂结构推理上的严重不足。

**[Can LLMs Generate High-Quality Test Cases for Algorithm Problems? TestCase-Eval](llm_nlp/testcase_eval_llm_test_gen.md)**

:   提出 TestCase-Eval 基准评估 LLM 生成算法题测试用例的能力，包含 500 道 Codeforces 算法题和 10 万条人工解答，聚焦两个任务——故障覆盖（测试集能覆盖多少潜在错误）和故障暴露（能否为特定错误代码生成暴露性测试），对 19 个 SOTA LLM 的评估揭示了当前模型在测试生成上的能力和局限。

**[A Theory of Response Sampling in LLMs: Part Descriptive and Part Prescriptive](llm_nlp/theory_of_llm_sampling.md)**

:   本文从认知科学视角揭示了LLM的采样启发式机制与人类决策类似：采样不仅反映概念的统计规范（描述性成分），还系统性地偏向隐含的理想值（规范性成分），这种偏移在500个概念、15个模型上均显著，并可能导致医疗等应用中的有偏决策。

**[Theory of Mind in Large Language Models: Assessment and Enhancement](llm_nlp/theory_of_mind_llm.md)**

:   系统综述了 LLM 的心智理论（ToM）能力的评估基准（10+ story-based benchmarks）和增强策略（prompt-only 和 fine-tuning 两类方法），指出当前 LLM 在 ToM 推理上仍有显著不足，并提出未来方向。

**[The Impact of Token Granularity on the Predictive Power of Language Model Surprisal](llm_nlp/token_granularity_impact.md)**

:   本文系统研究了子词 token 粒度（词表大小 256~128K）对语言模型 surprisal 预测人类阅读时间能力的影响，发现约 8K 词表大小的中等粒度 token 在自然阅读时间预测上最优，而更粗粒度（更接近词级）的 token 在花园路径句法效应上表现更敏感。

**[Token Prepending: A Training-Free Approach for Eliciting Better Sentence Embeddings from LLMs](llm_nlp/token_prepending_training_free.md)**

:   提出 Token Prepending (TP) 技术，通过在每层将解码得到的句子嵌入前置到句子开头，使因果注意力机制下的早期 token 也能感知完整句子信息，无需训练即可显著提升 LLM 的句子嵌入质量。

**[Turning Trash into Treasure: Accelerating Inference of Large Language Models with Token Recycling](llm_nlp/token_recycling.md)**

:   提出Token Recycling——一种无需额外训练的投机解码方法，将解码过程中产生的候选token存入邻接矩阵，通过BFS算法构建draft tree并用tree attention验证，仅需<2MB额外存储即在所有规模LLM上实现约2倍加速，超越现有无训练方法30%和有训练方法25%。

**[Training Dynamics Underlying Language Model Scaling Laws: Loss Deceleration and Zero-Sum Learning](llm_nlp/training_dynamics_underlying_language_model_scaling_laws_loss_deceleration_and_z.md)**

:   发现语言模型训练中存在 loss deceleration（损失减速）现象——损失曲线在 log-log 空间呈分段线性，根因是 zero-sum learning（ZSL）：per-token 梯度系统性对立导致破坏性干涉，将一部分样本的改善抵消另一部分的恶化；scale up 通过降低减速触发损失 $L_d$ 和提升减速后斜率 $r_d$ 来缓解 ZSL，为突破 scaling law 瓶颈提供了可直接干预的机制。

**[Training Language Model to Critique for Better Refinement](llm_nlp/training_language_model_to_critique_for_better_refinement.md)**

:   提出 Refinement-oriented Critique Optimization（RCO），以"批判效用"（Critique Utility, CU）——即批判导致的精炼改善比例——作为奖励信号训练 critic 模型，通过 DPO 变体的 MSE 目标函数优化，无需直接评估批判质量；在对话生成、摘要、问答、数学推理、代码生成 5 个任务上，RCO 训练的 7B/13B critic 模型在 CU 和 RQS 指标上显著超过 70B 基线模型和 DPCO 方法。

**[UniConv: Unifying Retrieval and Response Generation for Large Language Models in Conversations](llm_nlp/uniconv_retrieval_response_gen.md)**

:   探索如何将对话场景中的稠密检索和响应生成统一到单个 LLM 中，通过三个联合训练目标（对话检索 + 响应生成 + 上下文识别指令）和数据差异缓解机制，在五个对话搜索数据集上实现检索和生成的相互促进，超越分离式基线。

**[ScaleQuest: Unleashing LLM Reasoning Capability via Scalable Question Synthesis from Scratch](llm_nlp/unleashing_llm_reasoning_capability_via_scalable.md)**

:   提出 ScaleQuest，通过 Question Fine-Tuning (QFT) + Question Preference Optimization (QPO) 两阶段训练将 7B 解题模型变为出题模型，从零合成 100 万高质量数学问题-解答对，在四个基准上全面超越所有开源数据集，且数据量扩展至 1M 时性能持续提升未见饱和。

**[Veracity Bias and Beyond: Uncovering LLMs' Hidden Beliefs in Problem-Solving Reasoning](llm_nlp/veracity_bias_llm_hidden_beliefs.md)**

:   揭示了 LLM 在推理任务中存在"真实性偏见"（Veracity Bias）——尽管显式对齐反对刻板印象，LLM 仍系统性地将正确答案归因于特定种族群体（归因偏差），并对相同解答因"作者"种族不同给出不同评价（评估偏差），在数学、编程、常识推理和写作任务中普遍存在。

**[Many Heads Are Better Than One: Improved Scientific Idea Generation by A LLM-Based Multi-Agent System](llm_nlp/virsci_multi_agent_idea_gen.md)**

:   提出基于 LLM 的多智能体系统 Virtual Scientists（VirSci），模拟真实科研团队的协作过程——组织多个 agent 团队协作生成、评估和改进科研 idea，在生成新颖科学想法方面超越单智能体 SOTA。

---

## 🧩 多模态 VLM { #multimodal_vlm }

**[Adaptive Linguistic Prompting (ALP) Enhances Phishing Webpage Detection in Multimodal Large Language Models](multimodal_vlm/adaptive_linguistic_prompting_alp_enhances_phishing_webpage_detection_in_multimo.md)**

:   提出 Adaptive Linguistic Prompting (ALP)，一种 8-shot 结构化提示方法，引导多模态 LLM 从 HTML 文本、截图和 URL 三个维度联合推理，检测钓鱼网页，在 GPT-4o 上组合分析达到 F1=0.93，超过传统零样本基线。

**[Can LLMs Deceive CLIP? Benchmarking Adversarial Compositionality of Pre-trained Multimodal Representation via Text Updates](multimodal_vlm/adversarial_compositionality_clip.md)**

:   提出MAC基准和diversity-promoting自训练方法，通过让LLM生成欺骗性文本来系统暴露CLIP等预训练多模态表征的组合性漏洞，在图像/视频/音频三个模态上均显著超越已有方法。

**[Agent-RewardBench: Towards a Unified Benchmark for Reward Modeling across Perception, Planning, and Safety in Real-World Multimodal Agents](multimodal_vlm/agent_rewardbench.md)**

:   Agent-RewardBench 提出了首个面向多模态 Agent 的奖励建模评测基准，覆盖感知/规划/安全 3 个维度、7 个真实场景、1136 个高质量样本，实验发现即使 GPT-4o 也仅达 61.4% 准确率，揭示了 Agent 奖励建模的巨大挑战。

**[AkaCE: A Multimodal Multi-party Dataset for Emotion Recognition in Movie Dialogues](multimodal_vlm/akan_cinematic_emotions_ace_a_multimodal_multi-party_dataset_for_emotion_recogni.md)**

:   构建 AkaCE——首个非洲语言多模态对话情感识别数据集，覆盖阿坎语（加纳主要语言，约 2000 万使用者），含 385 段对话 6162 条发言（音频+视觉+文本三模态）、308 名说话人（性别平衡 155男/153女），并提供首个非洲语言词级韵律突出标注。

**[Aligning VLM Assistants with Personalized Situated Cognition](multimodal_vlm/aligning_vlm_assistants_with_personalized_situated.md)**

:   基于社会学"角色集合"(Role-Set) 概念刻画用户多样性，提出 PCogAlign 框架，通过认知感知的动作导向奖励模型来为 VLM 助手生成个性化回复，使不同角色的用户在相同视觉场景下获得最适合自身需求的建议。

**[AlignMMBench: Evaluating Chinese Multimodal Alignment in Large Vision-Language Models](multimodal_vlm/alignmmbench_evaluating_chinese_multimodal_alignment_in_large_vision-language_mo.md)**

:   提出 AlignMMBench，首个面向中文视觉上下文的多模态对齐评测基准，涵盖 3 大类 13 项任务、1054 张图像和 4978 个 QA 对（含单轮/多轮对话），并训练了基于 ChatGLM3-6B 的评估器 CritiqueVLM，其评估一致性超过 GPT-4。

**[Attacking Vision-Language Computer Agents via Pop-ups](multimodal_vlm/attacking_vl_agents_popups.md)**

:   系统性设计了一套对抗性弹窗攻击方法来攻击基于视觉语言模型的计算机操控 agent，在 OSWorld 和 VisualWebArena 上平均攻击成功率达 86%，任务成功率下降 47%，基础防御手段几乎无效。

**[AVG-LLaVA: An Efficient Large Multimodal Model with Adaptive Visual Granularity](multimodal_vlm/avg-llava_an_efficient_large_multimodal_model_with_adaptive_visual_granularity.md)**

:   提出 AVG-LLaVA——在 LLaVA-NeXT 基础上引入视觉粒度缩放器（多级池化获取不同粒度视觉 token）+ 视觉粒度路由器（基于图像和指令自适应选择最合适粒度），并提出 RGLF 训练范式将路由器预测与 LMM 偏好对齐。在 11 个基准上实现更好性能同时大幅减少视觉 token（如 AI2D 上减少 85.3%，推理加速 2.53×）。

**[Enhancing Multimodal Continual Instruction Tuning with BranchLoRA](multimodal_vlm/branchlora_continual_instruction.md)**

:   发现MoELoRA在多模态持续指令微调(MCIT)中存在参数低效——矩阵A跨任务趋同而B保持区分，提出BranchLoRA：共享一个A矩阵（树干）+ 多个专有B矩阵（树枝） + 灵活调参-冻结机制 + 任务特定路由器，在CoIN benchmark上显著超越MoELoRA，有效缓解灾难性遗忘。

**[MMSafeAware: Can't See the Forest for the Trees: Benchmarking Multimodal Safety Awareness for Multimodal LLMs](multimodal_vlm/cant_see_the_forest_for_the.md)**

:   提出 MMSafeAware，首个同时评估"不安全内容识别"和"过度敏感"的多模态安全意识基准，包含 1,500 个跨 29 种安全场景的图文对，评估 9 个 MLLM 发现所有模型都存在安全与有用性的严重权衡——GPT-4V 将 36.1% 的不安全输入误判为安全，同时将 59.9% 的安全输入误判为不安全；三种改进方法均无法根本解决问题。

**[Centurio: On Drivers of Multilingual Ability of Large Vision-Language Model](multimodal_vlm/centurio_multilingual_vlm.md)**

:   系统研究多语言LVLM训练策略，发现可以同时支持100种语言、只需25-50%非英文数据即可大幅提升多语言性能且不损英语性能，最终训练的Centurio在14个任务56种语言上达到SOTA。

**[ChartCoder: Advancing Multimodal Large Language Model for Chart-to-Code Generation](multimodal_vlm/chartcoder_chart_to_code.md)**

:   提出首个专用chart-to-code MLLM（ChartCoder），以Code LLM为语言骨干+160K大规模图表-代码数据集+Snippet-of-Thought逐步推理方法，7B模型在三个基准上超越所有开源MLLM，接近GPT-4o水平。

**[Scaling Text-Rich Image Understanding via Code-Guided Synthetic Multimodal Data Generation](multimodal_vlm/code_guided_text_rich_image.md)**

:   提出CoSyn框架，利用纯文本LLM的代码生成能力自动创建40万张文本丰富图像（图表、文档、图表等）+270万条指令微调数据，训练的7B VLM在7个基准上达到SOTA，超越GPT-4V和Gemini 1.5 Flash。

**[Insight Over Sight: Exploring the Vision-Knowledge Conflicts in Multimodal LLMs](multimodal_vlm/conflictvis_vision_knowledge_conflict.md)**

:   首次系统探索 MLLM 中常识级别的视觉-知识冲突问题，提出自动化框架构建 ConflictVis 基准（374 图 + 1122 QA），发现 MLLM 在约 20% 的冲突场景中过度依赖参数化知识（尤其是 Yes-No 和动作类问题），并提出 Focus-on-Vision 提示策略进行缓解。

**[CoRe-MMRAG: Cross-Source Knowledge Reconciliation for Multimodal RAG](multimodal_vlm/core_mmrag_knowledge_reconciliation.md)**

:   CoRe-MMRAG 提出了一个端到端多模态 RAG 框架，通过四阶段流水线（参数知识生成→视觉-文本联合重排序→外部知识生成→内外知识整合）解决参数知识-检索知识不一致(PRKI)和视觉-文本知识不一致(VTKI)两个问题，在 InfoSeek 和 Encyclopedic-VQA 上分别提升 5.6% 和 9.3%。

**[COSMMIC: Comment-Sensitive Multimodal Multilingual Indian Corpus for Summarization and Headline Generation](multimodal_vlm/cosmmic_commentsensitive_multimodal_multilingual_indian_corpus.md)**

:   构建首个面向印度语言的评论感知多模态多语言数据集 COSMMIC（9 种语言、4,959 篇文章-图像对、24,484 条读者评论），提出评论过滤（IndicBERT）和图像分类（CLIP）增强方案，用 GPT-4 和 LLama3 建立摘要和标题生成的基准。

**[Cosyn Code Guided Synthetic Data](multimodal_vlm/cosyn_code_guided_synthetic_data.md)**

:   提出 CoSyn 框架，利用纯文本 LLM 的代码生成能力自动合成多样化的文本丰富型图像及对应指令微调数据，构建 400K 图像 + 2.7M 指令数据集，在 7 个 benchmark 上达到开源 SOTA 并超越 GPT-4V。

**[Cracking the Code of Hallucination in LVLMs with Vision-aware Head Divergence](multimodal_vlm/cracking_hallucination_vhd.md)**

:   提出Vision-aware Head Divergence (VHD)指标量化注意力头对视觉信息的敏感度，发现幻觉与模型过度依赖语言先验紧密相关，并提出Vision-aware Head Reinforcement (VHR)无训练方法，通过放大视觉敏感注意力头来缓解幻觉，在CHAIR上最高降低CHAIRS 16.36个点。

**[Donate or Create? Comparing Data Collection Strategies for Emotion-labeled Multimodal Social Media Posts](multimodal_vlm/donate_or_create_comparing_data_collection.md)**

:   系统比较了两种情感标注数据的收集策略——"捐赠"真实社交媒体帖子 vs "创造"帖子——发现创造的帖子更长、更依赖文本、偏向原型化情感事件，但用创造数据训练的模型可以很好泛化到真实数据，只是需要真实数据来做靠谱的效果评估。

**[EffiVLM-Bench: A Comprehensive Benchmark for Evaluating Training-Free Acceleration in Large Vision-Language Models](multimodal_vlm/effivlm_bench_acceleration.md)**

:   提出 EffiVLM-Bench，首个系统评估大型视觉语言模型（LVLM）训练免加速方法的统一框架，覆盖 17 个 benchmark、3 个前沿模型，引入泛化性和忠诚度等新指标，揭示了 token 压缩与参数压缩在不同场景下的性能-效率权衡。

**[Effivlm Bench Vlm Acceleration](multimodal_vlm/effivlm_bench_vlm_acceleration.md)**

:   提出 EffiVLM-Bench，一个统一评估框架，系统性地评估大型视觉语言模型(LVLM)的免训练加速方法，涵盖 token 压缩和参数压缩两大类，从性能、泛化性、忠实度和效率四个维度进行全面对比分析。

**[Exploring How Generative MLLMs Perceive More Than CLIP with the Same Vision Encoder](multimodal_vlm/exploring_how_generative_mllms_perceive_more.md)**

:   系统探究为何生成式多模态LLM（如LLaVA）使用与CLIP相同的视觉编码器却能在视觉推理任务上大幅超越CLIP，发现patch token、位置编码和prompt加权是关键因素。

**[HiDe-LLaVA: Hierarchical Decoupling for Continual Instruction Tuning of Multimodal Large Language Model](multimodal_vlm/hidellava_hierarchical_decoupling_for_continual_instruction.md)**

:   通过 CKA 分析发现 MLLM 顶层学任务特异信息而其余层学通用知识，提出 HiDe-LLaVA：顶层 LoRA 做 MoE 式任务特异扩展（双模态锚点匹配）+ 其余层 LoRA 做均匀融合，在新构建的无信息泄露基准 UCIT 上比最佳基线提升 5.8%。

**[HotelMatch-LLM: Joint Multi-Task Training of Small and Large Language Models for Efficient Multimodal Hotel Retrieval](multimodal_vlm/hotelmatch_llm_retrieval.md)**

:   提出 HotelMatch-LLM，用 SLM 编码 query + LLM 编码酒店文档的非对称架构，配合三目标多任务优化（检索对齐 + MLM地理预测 + 视觉设施识别）和 patch 级 mean pooling 多图处理，在旅行领域多模态检索任务上显著超过 MARVEL/VISTA 等 SOTA。

**[Inference Compute-Optimal Video Vision Language Models](multimodal_vlm/inference_compute_optimal_video_vlm.md)**

:   首次系统研究视频VLM推理时的计算预算最优分配问题：在固定推理FLOPs下，如何在语言模型大小(x_N)、帧数(x_T)和每帧视觉token数(x_V)三个维度间做最优权衡，通过大规模训练扫描（~100k A100小时）和参数化建模得出实用的分配策略。

**[Synergizing LLMs with Global Label Propagation for Multimodal Fake News Detection](multimodal_vlm/llm_label_propagation.md)**

:   提出 GLPN-LLM 框架，通过 mask-based 全局标签传播机制有效整合 LLM 生成的伪标签，解决了 LLM 伪标签直接组合效果不佳的问题，在 Twitter/PHEME/Weibo 三个数据集上全面超越 SOTA。

**[LongDocURL: a Comprehensive Multimodal Long Document Benchmark Integrating Understanding, Reasoning, and Locating](multimodal_vlm/longdocurl_multimodal_long_doc.md)**

:   提出LongDocURL基准（396篇文档、2,325 QA对、33,000+页，含理解/推理/定位三类20个子任务），通过半自动构建流程生成高质量长文档评测数据，26种配置的实验结果显示最强GPT-4o仅64.5分，开源最高30.6分，远低于人类84.8分。

**[MAmmoTH-VL: Eliciting Multimodal Reasoning with Instruction Tuning at Scale](multimodal_vlm/mammoth_vl_multimodal_reasoning.md)**

:   MAmmoTH-VL 提出了一种仅用开源模型构建 12M 多模态 CoT 推理指令数据的可扩展方法，通过数据收集→改写→自过滤三步管线，训练的 8B 模型在 MathVerse (+8.1%)、MMMU-Pro (+7%)、MuirBench (+13.3%) 上达到 SOTA。

**[Modality-Aware Neuron Pruning for Unlearning in Multimodal Large Language Models](multimodal_vlm/manu_modality_aware_unlearning.md)**

:   提出MANU框架解决MLLM中的多模态遗忘不平衡问题：通过四种重要性函数（绝对/频率/方差/RMS）识别跨模态知识纠缠的神经元，选择性剪枝实现多模态输入和纯文本输入下的均衡知识遗忘，同时保持模型通用能力。

**[Evaluating Multimodal Large Language Models on Video Captioning via Monte Carlo Tree Search](multimodal_vlm/mcts_video_captioning_eval.md)**

:   提出AutoCaption框架，利用蒙特卡洛树搜索(MCTS)自动迭代生成细粒度视频描述关键点（平均122个/视频），构建MCTS-VCB基准评估20+个MLLM的视频描述能力，并证明生成的数据可用于微调显著提升模型性能。

**[Mmmu Pro Robust Benchmark](multimodal_vlm/mmmu_pro_robust_benchmark.md)**

:   提出 MMMU-Pro 基准，通过过滤纯文本可解题目、扩增选项至 10 个、引入 vision-only 输入设置三步法，构建更鲁棒的多学科多模态理解评测，所有模型性能显著下降 16.8%-26.9%。

**[MMMU-Pro: A More Robust Multi-discipline Multimodal Understanding Benchmark](multimodal_vlm/mmmupro_a_more_robust_multidiscipline_multimodal.md)**

:   MMMU-Pro 通过三步流程（过滤纯文本可解题、扩展选项至 10 个、引入纯视觉输入设置）构建更鲁棒的多学科多模态理解基准，模型性能比原 MMMU 下降 16.8%~26.9%，揭示当前模型依赖捷径而非真正多模态理解。

**[MultiMM: Cultural Bias Matters — Cross-Cultural Benchmark for Multimodal Metaphors](multimodal_vlm/multimm_cultural_metaphor.md)**

:   提出MultiMM——首个跨文化多模态隐喻数据集，包含8461个中英文广告图文对及细粒度标注，并设计SEMD模型融合情感特征增强隐喻检测。

**[A Survey on Patent Analysis: From NLP to Multimodal AI](multimodal_vlm/patent_analysis_survey.md)**

:   全面综述基于 NLP 和多模态 AI 的专利分析方法，按专利生命周期中的四大任务（分类、检索、质量分析、生成）提出新的分类体系，覆盖从传统神经网络到 PLM/LLM 的技术演进。

**[PunchBench: Benchmarking MLLMs in Multimodal Punchline Comprehension](multimodal_vlm/punchbench_mllm_punchline.md)**

:   提出PunchBench（6,000图文对、54,000 QA对），通过同义/反义标题替换消除文本捷径、多种问题格式与两层任务（感知+推理）全面评测MLLM的多模态梗图理解能力，并设计SC-CoQ由简到繁的提问策略显著提升表现，但GPT-4o仍远低于人类98%+水平。

**[Sharper and Faster mean Better: Towards More Efficient Vision-Language Model for Hour-scale Long Video Understanding](multimodal_vlm/sophia_efficient_long_video.md)**

:   提出Sophia模型处理小时级长视频：通过Shot-adaptive Frame Pruning（基于镜头分割的两阶段帧剪枝）精准选择查询相关帧，结合O(N)复杂度的Hierarchical Attention替代全注意力，在8/8个长视频benchmark中6个SOTA，且注意力FLOPs仅为InternVL2的1/17。

**[Can Multimodal Large Language Models Understand Spatial Relations?](multimodal_vlm/spatialmqa_mllm_spatial_relations.md)**

:   提出SpatialMQA基准，通过5,392个基于COCO2017的人工标注多选题样本（无bbox、含视角替换、排除纯知识可答题），评测MLLM的空间关系推理能力，发现最强模型SpaceLLaVA仅达48.14%远低于人类98.40%。

**[Speaking Beyond Language: A Large-Scale Multimodal Dataset for Learning Nonverbal Cues from Video-Grounded Dialogues](multimodal_vlm/speaking_beyond_language.md)**

:   提出 VENUS——首个大规模多模态对话数据集（89,459 段对话、14,910 小时），包含时间对齐的文本、3D 面部表情和肢体语言标注；基于该数据集开发 MARS 多模态语言模型，通过 VQ-VAE 将非语言线索离散化后与文本统一建模，实现对话中文本与非语言动作的联合理解和生成。

**[SPHERE: Unveiling Spatial Blind Spots in Vision-Language Models Through Hierarchical Evaluation](multimodal_vlm/sphere_unveiling_spatial_blind_spots_in.md)**

:   提出 SPHERE 三层级空间推理评估框架（单技能→多技能→推理），基于 MS COCO 人工标注 2285 个 QA 对，发现 GPT-4o（67.9%）与人类（93.0%）差距 25%，尤其在距离判断、视角切换和物理推理上表现严重不足。

**[Symmetrical Visual Contrastive Optimization: Aligning Vision-Language Models with Minimal Contrastive Images](multimodal_vlm/symmetrical_visual_contrastive_optimization_aligning_visionlanguage.md)**

:   提出 S-VCO（对称视觉对比优化），一种新的 VLM 微调目标，通过对称地对齐/拒绝匹配/矛盾的图像-文本对来增强视觉依赖，配合最小视觉对比数据集 MVC，在幻觉检测上减少 22%，视觉依赖任务上显著提升。

**[Teaching Vision-Language Models to Ask: Resolving Ambiguity in Visual Questions](multimodal_vlm/teaching_vlm_ask_ambiguity.md)**

:   提出ClearVQA benchmark评估VLM处理歧义视觉问题的能力（覆盖指代歧义/意图欠明确/拼写歧义三类），并通过自动化pipeline生成歧义-澄清问题对用于SFT+DPO训练，使VLM学会"反问用户"而非猜测作答，VQA准确率相对提升13.3%。

**[TheoremExplainAgent: Towards Video-based Multimodal Explanations for LLM Theorem Understanding](multimodal_vlm/theorem_explain_agent.md)**

:   提出 TheoremExplainAgent，一个基于多 Agent 协作的系统，能自动生成 5 分钟以上的定理讲解视频（Manim 动画+语音旁白），并构建了 TheoremExplainBench（240 个 STEM 定理，5 维评估指标）用于系统评估。

**[TrimLLM: Progressive Layer Dropping for Domain-Specific LLMs](multimodal_vlm/trimllm_layer_dropping.md)**

:   基于"层级特化"现象（不同层对不同领域重要性不同），提出TrimLLM在领域微调过程中渐进式丢弃最不重要的层，将LLaMA-7B压缩到50%大小时性能几乎无损，在消费级GPU上实现2.1-5.7×推理加速——无需任何特殊硬件或kernel支持。

**[Mitigating Visual Forgetting via Take-along Visual Conditioning for Multi-modal Long CoT Reasoning](multimodal_vlm/tvc_mitigating_visual_forgetting.md)**

:   发现MLLM长链推理中存在严重的"视觉遗忘"现象——推理进行到一半时移除图片仅导致2%精度下降，说明模型过度依赖文本输出而非视觉输入。提出Take-along Visual Conditioning (TVC)，在推理过程中周期性重新注入压缩后的图像特征，在5个数学推理基准上平均超越之前SOTA 3.4个点。

**[Unsolvable Problem Detection: Evaluating Trustworthiness of Large Multimodal Models](multimodal_vlm/unsolvable_problem_detection.md)**

:   本文提出不可解问题检测（UPD）任务评估大型多模态模型（LMM）的鲁棒理解能力，包含三种不可解场景（答案缺失AAD、答案集不兼容IASD、图像问题不兼容IVQD），构建 MM-UPD Bench 基准，实验揭示现有 LMM 在标准 MCQA 上表现良好但在 UPD 上显著挣扎，且现有基准性能与 UPD 能力之间缺乏相关性。

**[Value-Spectrum: Quantifying Preferences of Vision-Language Models via Value Decomposition](multimodal_vlm/value_spectrum_vlm_pref.md)**

:   提出 Value-Spectrum 基准，通过 50K+ 社交媒体短视频截图和 Schwartz 价值理论框架，系统评估 VLM 的内在价值偏好及角色扮演时的偏好适配能力。

**[VF-Eval: Evaluating Multimodal LLMs for Generating Feedback on AIGC Videos](multimodal_vlm/vf_eval_aigc_video_feedback.md)**

:   提出VF-Eval基准，通过一致性验证、错误感知、错误类型检测、推理评估四大任务系统评估13个MLLM为AIGC视频提供反馈的能力，发现即使GPT-4.1也难以在所有任务上表现一致，揭示了AIGC视频理解的挑战性。

**[Visual Evidence Prompting Mitigates Hallucinations in Large Vision-Language Models](multimodal_vlm/visual_evidence_prompting.md)**

:   提出Visual Evidence Prompting (VEP)，利用小型视觉专家模型（目标检测器、场景图生成器）的输出作为文本化"视觉证据"输入LVLM，无需训练即可在11个LVLM上显著降低幻觉——LLaVA-1.5在POPE上提升7.2%、Claude 3上提升12.1%。

**[VLMInferSlow: Evaluating the Efficiency Robustness of Large Vision-Language Models as a Service](multimodal_vlm/vlminferslow_evaluating_the_efficiency_robustness_of.md)**

:   首次在黑盒设置下研究 VLM 的效率鲁棒性，提出 VLMInferSlow 方法，通过零阶优化搜索对抗性图像扰动，迫使 VLM 生成更长序列，将计算成本最高增加 128.47%，揭示了 VLM 在 MLaaS 部署场景下的效率安全隐患。

**[VReST: Enhancing Reasoning in Large Vision-Language Models through Tree Search and Self-Reward Mechanism](multimodal_vlm/vrest_tree_search_vlm_reasoning.md)**

:   提出VReST，首次将蒙特卡洛树搜索（MCTS）应用于多模态CoT推理：每个节点是一个推理步骤，通过多模态自奖励机制（sub-question有用性+答案正确性+视觉-语言线索相关性）评估推理质量，无需训练即在MathVista上达到64.50%（超越CoT的54.60%和ToT的60.20%），并展示出多模态测试时缩放定律。

---

## ⚖️ 对齐 / RLHF { #llm_alignment }

**[Aligned but Blind: Alignment Increases Implicit Bias by Reducing Awareness of Race](llm_alignment/aligned_but_blind_implicit_bias.md)**

:   发现 LLM 对齐训练的矛盾效应：对齐成功消除了显式偏见（Llama 3 70B 降至 8.13%），但反而放大了隐式偏见（从 64.1% 升至 91.4%），机制是对齐使模型在歧义上下文中不再表征种族概念（"种族盲视"），导致安全护栏无法在隐性场景中激活。通过在早期层注入种族感知激活可将隐式偏见从 97.3% 降至 71.2%。

**[ASPO: Adaptive Sentence-Level Preference Optimization for Fine-Grained Multimodal Reasoning](llm_alignment/aspo_adaptive_sentence-level_preference_optimization_for_fine-grained_multimodal.md)**

:   提出 ASPO（Adaptive Sentence-level Preference Optimization）——将 DPO 的偏好单元从"整个回复"细化到"每个句子"，为每个句子动态计算自适应奖励值（基于模型自身预测评估正确性和重要性），在多模态推理任务上显著优于传统回复级 DPO，有效减少幻觉并提升细粒度推理能力。

**[Atyaephyra at SemEval-2025 Task 4: Low-Rank Negative Preference Optimization](llm_alignment/atyaephyra_at_semeval-2025_task_4_low-rank_negative_preference_optimization.md)**

:   在 SemEval 2025 LLM 遗忘共享任务中，将负偏好优化 (NPO) 与低秩适配 (LoRA) 结合，利用 LoRA 的结构特性零开销获取原始模型分布来计算 KL 散度正则化，显著稳定了遗忘过程并超越了任务基线。

**[AutoMixAlign: Adaptive Data Mixing for Multi-Task Preference Optimization in LLMs](llm_alignment/automixalign_adaptive_data_mixing.md)**

:   AutoMixAlign 提出了一种理论驱动的多任务偏好优化数据混合方法：先训练各任务的 specialist model 确定最优 loss 基线，再通过 minimax 优化自适应调整数据混合比例，优先处理 excess loss（与 specialist 的差距）最大的任务，在 helpfulness/harmlessness/reasoning 多任务 DPO 中平均提升 9.42%。

**[Beyond Surface-Level Patterns: An Essence-Driven Defense Framework Against Jailbreak Attacks in LLMs](llm_alignment/beyond_surface-level_patterns_an_essence-driven_defense_framework_against_jailbr.md)**

:   提出 EDDF，一种基于"攻击本质"而非表面模式的越狱防御框架：离线提取已知攻击的本质策略存入向量数据库，在线时对新查询做本质抽象+检索+细粒度判断，将攻击成功率降低至少 20% 且误报率仅 2.18%。

**[Beyond the Tip of Efficiency: Uncovering the Submerged Threats of Jailbreak Attacks in Small Language Models](llm_alignment/beyond_the_tip_of_efficiency_uncovering_the_submerged_threats_of_jailbreak_attac.md)**

:   系统评估 13 个 SOTA 小语言模型（<4B参数）在 5 种越狱攻击下的安全性，发现 SLM 虽能抵御直接攻击但在越狱攻击下显著比大模型脆弱，进一步分析了架构压缩、量化和知识蒸馏等 SLM 技术对安全性的影响。

**[Boosting Vulnerability Detection of LLMs via Curriculum Preference Optimization with Synthetic Reasoning Data](llm_alignment/boosting_vulnerability_detection_of_llms_via_curriculum_preference_optimization_.md)**

:   提出 ReVD 框架，通过双向漏洞推理数据合成 + 三元组 SFT（同时学习漏洞代码/修复代码/代码差异的推理）+ 课程化在线偏好优化（COPO），将 LLM 的漏洞检测准确率提升 12-23%，在 PrimeVul 和 SVEN 上达到 SOTA。

**[Breaking the Ceiling: Exploring the Potential of Jailbreak Attacks through Expanding Strategy Space](llm_alignment/breaking_the_ceiling_exploring_the_potential_of_jailbreak_attacks_through_expand.md)**

:   基于精细化可能性模型 (ELM) 将越狱策略分解为四类可独立进化的组件（角色/内容支撑/语境/沟通技巧），提出 CL-GSO 遗传算法在组件级进行交叉与变异，将策略空间从既有方法的 40 种扩展到 839 种，在 Claude-3.5 上实现 96% 攻击成功率（此前方法最高仅 4%），同时提出基于意图一致性的评估机制，准确率达 96.5% 超越专用安全模型。

**[Chain-of-Jailbreak Attack for Image Generation Models via Editing Step by Step](llm_alignment/chain-of-jailbreak_attack_for_image_generation_models_via_ed.md)**

:   提出 CoJ（Chain-of-Jailbreak）攻击，将生成恶意图像的单步请求分解为多步编辑指令链（从无害种子图像逐步编辑到目标），绕过图像生成模型的安全过滤器，在 GPT-4o 等模型上达到高攻击成功率。

**[Cheems: A Practical Guidance for Building and Evaluating Chinese Reward Models from Scratch](llm_alignment/cheems_chinese_reward_models.md)**

:   为弥补中文 Reward Model 资源的空白，本文构建了 CheemsBench（首个大规模中文 RM 评测基准）和 CheemsPreference（首个大规模中文偏好数据集），通过人机协作标注 + 远程监督过滤策略训练的 CheemsRM 在中文场景显著超越现有所有开源 RM。

**[CodeDPO: Aligning Code Models with Self Generated and Verified Source Code](llm_alignment/codedpo_code_alignment.md)**

:   提出 CodeDPO，通过 PageRank 启发的自验证评分机制从自生成代码中构造高质量偏好对（93K 正确性 + 21K 效率），DPO 训练后在 8 个代码模型上 HumanEval 平均提升 10+ 分，同时提升代码执行效率 1.25-1.45×。

**[Curiosity-Driven Reinforcement Learning from Human Feedback](llm_alignment/curiosity_driven_rlhf.md)**

:   CD-RLHF 将好奇心驱动探索（curiosity-driven RL）引入 RLHF，通过前向动力学模型的预测误差作为内在奖励，结合 top-k 门控过滤与 reward whitening，在不损失对齐质量的前提下大幅提升 LLM 输出多样性（Llama-3.2-1B 上 Diversity 提升 40.26%，EAD 提升 8.92%）。

**[Debate, Reflect, and Distill: Multi-Agent Feedback with Tree-Structured Preference Optimization for Efficient Language Model Enhancement](llm_alignment/debate_reflect_and_distill_multi-agent_feedback_with_tree-structured_preference_.md)**

:   提出 D&R 框架，让小模型（student）与多个大模型（teacher）进行多轮辩论并收集自我反思和教师反馈，然后将辩论日志组织为偏好树做 Tree-structured DPO (T-DPO) 蒸馏，在 MMLU Pro 和 MATH 上平均提升 14.18 分，且推理效率优于基线。

**[DiffPO: Diffusion Alignment with Direct Preference Optimization](llm_alignment/diffpo_diffusion_alignment.md)**

:   提出 DiffPO，将 LLM 对齐重新建模为句子级扩散去噪过程，通过 parallel decoding 实现高效推理时对齐，作为即插即用模块可增强任意底座模型的对齐质量。

**[Expectation Confirmation Preference Optimization for Multi-Turn Conversational Recommendation Agent](llm_alignment/expectation_confirmation_preference_optimization_for_multi-turn_conversational_r.md)**

:   基于期望确认理论(ECT)提出 ECPO 多轮对话偏好优化框架，模拟用户在推荐对话中的满意度演变，百向期望确认定位不满意根因，用 LLM 用户模拟器(AILO)生成反馈，显著提升推荐 agent 的交互效果和效率。

**[Federated Data-Efficient Instruction Tuning for Large Language Models](llm_alignment/federated_data-efficient_instruction_tuning_for_large_language_models.md)**

:   提出 FedHDS（Federated Hierarchical Data Selection），通过 intra-client 和 inter-client 两级层次化数据选择消除联邦学习中客户端内部和跨客户端的数据冗余，结合多层 Transformer 特征融合提升 coreset 质量；仅用不到 1.5% 的数据，在 Rouge-L 上相对 SOTA 全数据联邦基线平均提升 10.72%，训练效率提升最高达 48.8 倍。

**[Focused-DPO: Enhancing Code Generation Through Focused Preference Optimization on Error-Prone Points](llm_alignment/focused-dpo_enhancing_code_generation_through_focused_preference_optimization_on.md)**

:   发现代码生成错误集中在特定"错误易发点"（error-prone points）——前缀/后缀通常正确，错误集中在中间代码段，提出 Focused-DPO 通过 PageRank 排序定位关键代码段并在 DPO 损失中加权放大，HumanEval+ 提升 4.41%、MBPP+ 提升 6.71%。

**[Understanding Impact of Human Feedback via Influence Functions](llm_alignment/influence_functions_rlhf.md)**

:   首次将影响函数应用于 RLHF 奖励模型的反馈数据审计，结合 OPORP 向量压缩实现 2.5 倍加速，在偏差检测上超越 GPT-4o（AUC 0.8 vs 0.747），并从 Anthropic-HH 数据集中发现 47% 的错标样本。

**[Internal Value Alignment in Large Language Models through Controlled Value Vector Activation](llm_alignment/internal_value_alignment_in_large_language_models_through_controlled_value_vecto.md)**

:   提出 ConVA（Controlled Value Vector Activation）框架，通过上下文控制的数据集精准识别 LLM 隐空间中的价值向量，并用门控最小扰动机制在推理时激活目标价值，在 Schwartz 10 种基本价值上实现平均 29.6% 的控制成功率提升，同时保持 97%+ 的文本流畅度和通用能力。

**[IOPO: Empowering LLMs with Complex Instruction Following via Input-Output Preference Optimization](llm_alignment/iopo_input_output_preference.md)**

:   提出 IOPO（Input-Output Preference Optimization），在传统 DPO 仅优化输出偏好的基础上，引入输入偏好建模——让模型学习"给定回复 y，哪个指令 x 更匹配"，从而增强对复杂多约束指令的细粒度感知能力；同时构建了包含 120K 训练数据、1K 评测数据、覆盖 5 大类 26 个约束维度的 Trace 基准。

**[JailbreakRadar: Comprehensive Assessment of Jailbreak Attacks Against LLMs](llm_alignment/jailbreakradar_comprehensive_assessment_jailbreak_attacks.md)**

:   本文提出了一个全面的越狱攻击评估框架 JailbreakRadar，收集了17种代表性越狱攻击方法，建立了六类攻击分类体系，并在9个对齐LLM上进行了大规模系统性评测，揭示了不同类型攻击在实用性和防御鲁棒性上的关键差异。

**[JsonTuning: Towards Generalizable, Robust, and Controllable Instruction Tuning](llm_alignment/jsontuning_towards_generalizable_robust_and_controllable_instruction_tuning.md)**

:   提出 JsonTuning——将指令微调的输入输出从自然语言文本替换为 JSON 结构化格式，通过显式表示任务元素、关系和输出约束（JSON Schema），在 7 个预训练模型和 6 类任务上一致超越传统 TextTuning，平均性能从 26.78 提升到 30.88，同时显著增强鲁棒性和可控性。

**[Enhancing Safe and Controllable Protein Generation via Knowledge Preference Optimization](llm_alignment/kpo_protein_safety.md)**

:   提出知识引导偏好优化（KPO）框架，通过蛋白质安全知识图谱识别安全/危险序列作为偏好信号，用强化学习训练蛋白质语言模型减少有害蛋白质序列的生成概率，同时保持功能性——为蛋白质生成的生物安全提供保障框架。

**[LLMs Caught in the Crossfire: Malware Requests and Jailbreak Challenges](llm_alignment/llms_caught_in_the_crossfire_malware_requests_and_jailbreak_challenges.md)**

:   构建 MalwareBench 基准（320 个手工恶意代码需求 × 11 种黑盒越狱方法 = 3520 个 prompt），系统评测 29 个 LLM 在恶意代码生成场景下的安全性，发现越狱攻击将平均拒绝率从 60.93% 降至 39.92%，且模型参数量与防御能力并非正比关系。

**[LSSF: Safety Alignment via Low-Rank Safety Subspace Fusion](llm_alignment/lssf_safety_subspace.md)**

:   LSSF 提出 LLM 的安全信息存在于低秩子空间中的假设，通过 SVD 提取安全对齐模型的主成分，利用安全奇异值熵自适应确定每层的保留秩，最终将提取的安全主成分线性融合到微调后的模型中，无需额外训练即可恢复因微调而退化的安全对齐，同时保持下游任务性能。

**[M2S: Multi-turn to Single-turn jailbreak in Red Teaming for LLMs](llm_alignment/m2s_multiturn_to_singleturn_jailbreak_in.md)**

:   提出 M2S 框架，通过三种简单的格式转换方法（Hyphenize/Numberize/Pythonize）将多轮人类越狱对话压缩为单轮 prompt，不仅保持甚至超越原始多轮攻击效果（ASR 高达 95.9%，比多轮提升最多 17.5%），同时 token 使用量减半以上。

**[M-RewardBench: Evaluating Reward Models in Multilingual Settings](llm_alignment/m_rewardbench.md)**

:   构建首个多语言奖励模型评估基准M-RewardBench（23种语言、2.87K偏好实例，覆盖对话/安全/推理/翻译四类能力），系统评估多种RM后发现英语与非英语RM性能存在显著差距，且翻译质量和语言资源量对RM表现有重要影响。

**[Measuring Data Diversity for Instruction Tuning: A Systematic Analysis and A Reliable Metric](llm_alignment/measuring_data_diversity_for_instruction_tuning_a_systematic_analysis_and_a_reli.md)**

:   系统分析 11 种现有多样性度量方法的局限性，提出 NovelSum——一种同时考虑样本间差异和信息密度的数据多样性指标，与指令微调性能达到 0.97 相关性。

**[MPO: Multilingual Safety Alignment via Reward Gap Optimization](llm_alignment/mpo_multilingual_safety_alignment.md)**

:   MPO 发现 LLM 在主导语言（英文）和目标语言间的隐式 Reward Gap 与安全性能强相关，提出直接最小化两者 Reward Gap 差异来将主导语言的安全对齐能力迁移到多语言，在三个模型上显著降低了低资源语言的攻击成功率且不损害通用能力。

**[Mutual-Taught for Co-adapting Policy and Reward Models](llm_alignment/mutual_taught_policy_reward.md)**

:   Mutual-Taught 提出了一种基于 EM 算法的自训练框架，在偏好优化过程中同时迭代更新 policy model 和 reward model：E-step 用当前 RM 优化 PM，M-step 用 PM 更新前后的输出差异构建伪偏好对来更新 RM，解决了分布偏移导致的 reward hacking 问题，8B 模型在 AlpacaEval-2 达到 54.1% LC win rate。

**[Optimal Transport-Based Token Weighting for Enhanced Preference Optimization](llm_alignment/otpo_token_weighting.md)**

:   OTPO 利用无平衡最优传输（UOT）在 chosen/rejected 回复的 token 表示之间计算语义对齐权重，使偏好优化聚焦于关键差异 token 而非均等对待所有 token，在 AlpacaEval2 上将 DPO 的 LC WR 从 48.14% 提升至 55.84%，并将 DPO/SimPO/SamPO/LDDPO 统一为 token 加权的特例。

**[Whose Boat Does it Float? Improving Personalization in Preference Optimization](llm_alignment/personalized_preference_opt.md)**

:   提出"溯因推理"视角的偏好个性化方法：先用 LLM 推断偏好 chosen/rejected 回答背后的用户画像（Persona Inference），再用画像增强的偏好数据训练模型（Persona Tailoring），显著提升模型对不同用户需求的个性化适配能力。

**[PIG: Privacy Jailbreak Attack on LLMs via Gradient-based Iterative Prompts](llm_alignment/pig_privacy_jailbreak.md)**

:   提出 PIG 框架，通过识别隐私查询中的 PII 实体类型、构建隐私上下文示例、并利用三种基于梯度的迭代优化策略更新上下文，实现对 LLM 的高效隐私越狱攻击，在白盒和黑盒模型上均达到 SOTA。

**[Probability-Consistent Preference Optimization for Enhanced LLM Reasoning](llm_alignment/probability-consistent_preference_optimization_for_enhanced_llm_reasoning.md)**

:   提出 PCPO（Probability-Consistent Preference Optimization），在偏好对选择时同时考虑答案正确性和 token 级概率一致性（用 Levenshtein 距离过滤+概率一致性评分），并在 DPO 损失中按一致性加权，在 GSM8K/MATH-500/Olympiadbench 上一致超越标准 DPO 和 ScPO。

**[QueryAttack: Jailbreaking Aligned Large Language Models Using Structured Non-natural Query Language](llm_alignment/queryattack_jailbreaking_aligned_large_language_models_using_structured_non-natu.md)**

:   提出 QueryAttack，将恶意自然语言查询分解为三个语义组件（内容、修饰符、类别）并填入编程语言模板（SQL/URL/Python/Java/C++ 等 9 种），结合 ICL 引导目标 LLM 直接用自然语言回复有害内容，无需解密步骤，在 GPT-4o 上 Ensemble 配置达到 96.35% ASR，且提出的跨语言 CoT 防御可将 ASR 降低最多 64%。

**[Red Queen: Safeguarding Large Language Models against Concealed Multi-Turn Jailbreaking](llm_alignment/red_queen_safeguarding_large_language_models_against_concealed_multi-turn_jailbr.md)**

:   提出 Red Queen Attack——首个基于 Theory of Mind（ToM）构建多轮对话场景并隐藏恶意意图的越狱攻击方法，生成 56K 多轮隐蔽攻击数据，在 GPT-4o 上达到 87.6% ASR；同时提出 Red Queen Guard 防御策略，通过多轮 DPO 数据训练将 ASR 降至 <1%，同时不影响通用基准性能。

**[Rethinking Table Instruction Tuning](llm_alignment/rethinking_table_instruction_tuning.md)**

:   系统消融表格指令微调中被忽视的超参数选择（学习率、数据量、epoch），揭示现有表格 LLM 因学习率过大（2e-5）导致通用能力严重退化（MMLU 降 14 分、AI2ARC 降 21 分），提出仅需 13 个数据集各 200 条（共 2600 条）+ 学习率 1e-6 + 2 epoch 微调 LLaMA 3.1 8B Instruct 即可构建 TAMA，在 13 个表格任务上匹配/超越 GPT-3.5 和 GPT-4，同时完整保持通用能力。

**[Reverse Preference Optimization for Complex Instruction Following](llm_alignment/reverse_preference_optimization_for_complex_instruction_following.md)**

:   提出反向偏好优化（RPO），通过动态反转指令中未满足的约束将任意回复转化为"完美"chosen 样本，消除多约束偏好对中的噪声，在多轮复杂指令遵循任务上显著超越 DPO 基线。

**[Towards Reward Fairness in RLHF: From a Resource Allocation Perspective](llm_alignment/reward_fairness_rlhf.md)**

:   将 RLHF 中的各种奖励偏差（长度偏差、类别偏差、社会偏差）统一定义为"奖励不公平"问题，从资源分配视角提出两种偏差无关的缓解方法——公平正则化和公平系数——在不针对特定偏差设计的情况下有效缓解多种偏差，实现更公平的人类偏好对齐。

**[Reward Generalization in RLHF: A Topological Perspective](llm_alignment/reward_generalization_in_rlhf_a_topological_perspective.md)**

:   从信息拓扑的角度系统刻画 RLHF 中 reward 信息的流动——宏观层面将 RLHF 建模为自编码过程，微观层面提出 Induced Bayesian Network (IBN) 分析偏好数据拓扑对 reward 泛化的影响，进而提出树结构偏好数据方法，在 HH-RLHF/GSM-8K/DialogSum 三个任务上平均 65% win rate 超越链式 baseline。

**[Rewrite to Jailbreak: Discover Learnable and Transferable Implicit Harmfulness Instruction](llm_alignment/rewrite_to_jailbreak_discover_learnable_and_transferable_implicit_harmfulness_in.md)**

:   提出 R2J（Rewrite to Jailbreak），一种可学习、可迁移的黑盒越狱方法——通过迭代训练 attacker LLM 学习改写有害指令（仅改措辞不改意图），相比 GCG/AutoDAN 等方法攻击成功率提高 20%+，且无额外前缀/后缀，更隐蔽且跨模型可迁移。

**[RISE: Subtle Errors in Reasoning: Preference Learning via Error-injected Self-editing](llm_alignment/rise_error_inject_preference.md)**

:   RISE 发现 LLM 约 75% 的数学错误是微妙的步内错误（数字替换、操作数交换、步骤遗漏），通过让 LLM 自编辑向正确解注入预定义微妙错误来构造高质量难负样本，配合错误感知 DPO 训练，仅用 4.5K 样本在 GSM8K 提升 3.0%、MATH 提升 7.9%，并泛化到逻辑推理和代码生成。

**[SEA: Low-Resource Safety Alignment for Multimodal Large Language Models via Synthetic Embeddings](llm_alignment/sea_lowresource_safety_alignment_for_multimodal.md)**

:   提出 SEA 框架，通过梯度优化生成合成模态 embedding（不需要真实图像/视频/音频），仅用文本安全数据就能实现多模态 LLM 的安全对齐，在单张 RTX3090 上 24 秒即可合成高质量 embedding，同时发布了视频和音频安全基准 VA-SafetyBench。

**[SynthesizeMe! Inducing Persona-Guided Prompts for Personalized Reward Models in LLMs](llm_alignment/synthesizeme_persona_prompts.md)**

:   提出SynthesizeMe，通过bootstrap推理→合成用户画像→筛选信息性示例三步流程，无需微调即可为个性化奖励模型构建有效prompt，在Chatbot Arena上提升LLM-as-a-Judge个性化准确率4.4%。

**[TableDreamer: Progressive and Weakness-Guided Data Synthesis from Scratch for Table Instruction Tuning](llm_alignment/tabledreamer_progressive_and_weakness-guided_data_synthesis_from_scratch_for_tab.md)**

:   提出 TableDreamer 两阶段数据合成框架：第一阶段从零合成多样化表格及种子指令数据，第二阶段通过弱点引导的迭代输入空间探索（在三个方向上演化数据，并用 LLM-as-Judge 筛选模型表现差的样本作为下一轮种子），仅用 27K GPT-4o 合成数据即将 Llama3.1-8B 的平均准确率提升 11.62%，超越使用 80K-100K 数据的所有基线方法。

**[Teaching an Old LLM Secure Coding: Localized Preference Optimization on Distilled Preferences](llm_alignment/teaching_an_old_llm_secure_coding.md)**

:   提出 DiSCo（从前沿 LLM 蒸馏的安全代码偏好数据集，10K 实例覆盖 431 种 CWE）和 LPO（局部偏好优化算法，仅在安全相关 token 上传播损失），在四个安全编码基准上减少 19-40% 的安全问题，同时提升 3-10% 的代码质量。

**[Think&Cite: Improving Attributed Text Generation with Self-Guided Tree Search and Progress Reward Modeling](llm_alignment/think_cite_attributed_text_gen.md)**

:   将归因文本生成（带引用的文本生成）建模为多步推理问题，提出自引导蒙特卡洛树搜索（SG-MCTS）结合进度奖励建模（PRM），通过多路径搜索+中间状态反思+生成/归因双维度进度奖励，在 ALCE 基准三个数据集上显著超越所有基线。

**[A Troublemaker with Contagious Jailbreak Makes Chaos in Honest Towns](llm_alignment/tmcht_contagious_jailbreak_multiagent.md)**

:   提出 TMCHT 多智能体多拓扑攻击评估框架和 ARCJ（对抗性复制传染越狱）方法，解决了多智能体系统中单智能体攻击方法面临的"毒性消散"问题——通过优化检索后缀确保中毒样本易被检索，优化复制后缀使中毒信息具有传染性自我复制能力，在 line/star 拓扑和 100 智能体系统中分别提升 23.51%/18.95%/52.93% 的攻击成功率。

---

## 🦾 LLM Agent { #llm_agent }

**[A Multi-Agent Framework for Mitigating Dialect Biases in Privacy Policy Question-Answering Systems](llm_agent/a_multi-agent_framework_for_mitigating_dialect_biases_in_privacy_policy_question.md)**

:   提出多 Agent 协作框架缓解隐私政策 QA 中的方言偏差——Dialect Agent 将方言查询翻译为标准美式英语（SAE）并验证意图保留，Privacy Policy Agent 利用领域专长生成答案，两者迭代协商至达成一致。在 PrivacyQA 和 PolicyQA 上将 GPT-4o-mini 零样本准确率从 0.394 提升至 0.601，方言间最大 F1 差距降低 82%。

**[Agentic Reasoning: A Streamlined Framework for Enhancing LLM Reasoning with Agentic Tools](llm_agent/agentic_reasoning_tools.md)**

:   Agentic Reasoning 提出了一个将 Web 搜索、代码执行和知识图谱记忆（Mind-Map）三种 Agent 工具集成到 LLM 推理过程中的框架，在 DeepSeek-R1 上将 Humanity's Last Exam 准确率从 9.4% 提升到 23.8%（+14.4%），GPQA 从 71.5% 到 81.2%，接近 OpenAI Deep Research 水平。

**[AndroidGen: Building an Android Language Agent under Data Scarcity](llm_agent/androidgen_agent_data_scarcity.md)**

:   提出AndroidGen——一个在数据稀缺条件下增强Android Agent能力的框架（ExpSearch+ReflectPlan+AutoCheck+StepCritic四模块），在AndroidWorld上以GPT-4o达到46.8%成功率（vs M3A的27.7%），并能自动生成高质量轨迹数据训练开源模型达到竞争力水平。

**[AndroidLab: Training and Systematic Benchmarking of Android Autonomous Agents](llm_agent/androidlab_autonomous_agent.md)**

:   提出AndroidLab——首个统一训练和评估Android Agent的系统性框架，包含9个App上的138个可复现任务，同时支持纯文本（XML模式）和多模态（SoM模式）模型，并构建Android Instruct数据集（94.3k步骤），将开源LLM的成功率从4.59%提升至21.50%。

**[Assessing Agentic LLMs in Multilingual National Bias](llm_agent/assessing_agentic_large_language_models_in_multilingual_national_bias.md)**

:   首次研究 LLM 作为推理型 Agent 在多语言场景下的国籍偏见——在大学申请/旅行/搬迁三个决策场景中，让 GPT-3.5/GPT-4/Sonnet 对同一实体（大学/城市）用不同语言打分，发现普遍存在"本地语言偏向"（用中文问清华得 10 分，用英文问只得 7 分），GPT-4 在英语上偏见减少但非英语上偏见显著，CoT 不一定缓解反而可能放大偏差。

**[Bel Esprit: Multi-Agent Framework for Building AI Model Pipelines](llm_agent/bel_esprit_multi-agent_framework_for_building_ai_model_pipelines.md)**

:   提出 Bel Esprit——基于多 Agent 框架的对话式 AI 管线构建系统，通过子 Agent 协作（需求澄清→管线构建→验证→模型填充）将用户的模糊需求转化为由多个 AI 模型组成的可执行管线（如多语言视频配音→语音识别+翻译×3+TTS×3），在人工策划和合成数据上验证有效性。

**[Beyond Numeric Rewards: In-Context Dueling Bandits with LLM Agents](llm_agent/beyond_numeric_rewards_in-context_dueling_bandits_with_llm_agents.md)**

:   系统评估了 LLM 在 Dueling Bandits（偏好反馈强化学习）中的零样本上下文决策能力，发现 GPT-4 Turbo 在弱遗憾（weak regret）上表现出色但强遗憾（strong regret）存在差距，进而提出 LEAD 框架（LLM with Enhanced Algorithmic Dueling），通过将经典 DB 算法与 LLM 智能体细粒度自适应融合来同时获得理论保证和鲁棒性。

**[Caution for the Environment: Multimodal LLM Agents are Susceptible to Environmental Distractions](llm_agent/caution_environment_gui_agent_distractions.md)**

:   本文首次系统研究了多模态 GUI Agent 对环境干扰（弹窗广告、推荐内容等）的脆弱性，在无恶意攻击的自然场景下，即使最强的 MLLM（包括GPT-4o）也有 20-40% 的概率被环境中的无关内容分散注意力而执行偏离用户目标的操作。

**[Context-Aware Sentiment Forecasting via LLM-based Multi-Perspective Role-Playing Agents](llm_agent/context_aware_sentiment_forecasting_agents.md)**

:   提出一个基于 LLM 的多视角角色扮演框架（MPR），通过主观 Agent 模拟用户发帖、客观 Agent（微调的"心理学家"LLM）审查行为一致性，以迭代纠正的方式预测社交媒体用户对实时事件的未来情感反应，在宏观和微观层面均大幅超越传统方法。

**[Leveraging Dual Process Theory in Language Agent Framework for Real-time Simultaneous Human-AI Collaboration](llm_agent/dpt_agent_dual_process.md)**

:   提出 DPT-Agent，首个将双过程理论（Dual Process Theory）系统化地融入语言智能体框架的方法——用有限状态机(FSM)+code-as-policy 作为快速直觉的 System 1，用心智理论(ToM)+异步反思的 LLM 作为慢速深思的 System 2，首次实现了自主的实时同步人机协作（在 Overcooked 困难版中）。

**[EMULATE: A Multi-Agent Framework for Determining the Veracity of Atomic Claims by Emulating Human Actions](llm_agent/emulate_a_multi-agent_framework_for_determining_the_veracity_of_atomic_claims_by.md)**

:   提出 EMULATE 多智能体事实核查框架，通过 7 个专职 LLM agent 模拟人类验证声明的完整行为链（搜索→排序→内容评估→证据充分性判断→分类），在三个事实核查 benchmark 上的 Macro-F1 和 Weighted-F1 均超越现有方法。

**[Explorer: Scaling Exploration-Driven Web Trajectory Synthesis for Multimodal Web Agents](llm_agent/explorer_scaling_exploration-driven_web_trajectory_synthesis_for_multimodal_web_.md)**

:   提出 Explorer——一个可扩展的多智能体 pipeline，通过自主网页探索和逐步精炼来合成大规模多模态 web 轨迹数据集（94K 成功轨迹，49K+ URL，720K 截图），训练的 Explorer-7B 在 Mind2Web-Live、MiniWob++ 等 benchmark 上达到甚至超过 GPT-4 水平。

**[FACT-AUDIT: An Adaptive Multi-Agent Framework for Dynamic Fact-Checking Evaluation of Large Language Models](llm_agent/fact_audit_factcheck.md)**

:   提出FACT-AUDIT——一个基于重要性采样和多智能体协作的自适应动态事实核查评估框架，通过动态生成测试数据、迭代探测模型弱点、并同时评估verdict预测和justification质量，全面审计LLM的事实核查能力边界。

**[GUI-explorer: Autonomous Exploration and Mining of Transition-aware Knowledge for GUI Agent](llm_agent/gui_explorer_autonomous.md)**

:   提出 GUI-explorer，一个无需训练的 GUI agent，通过自主探索收集功能感知的交互轨迹，并以无监督方式从状态转换三元组中挖掘 transition-aware 知识，在 SPA-Bench 和 AndroidWorld 上分别达到 53.7% 和 47.4% 的任务成功率。

**[GuideBench: Benchmarking Domain-Oriented Guideline Following for LLM Agents](llm_agent/guidebench_guideline_following.md)**

:   提出 GuideBench 基准测试，系统评估 LLM 在领域导向指南遵循方面的能力，覆盖 7 个任务类别共 1272 个实例，从规则遵循、规则更新鲁棒性和人类偏好对齐三个维度评估 18 个 LLM，发现当前模型在复杂领域规则遵循上仍有较大提升空间。

**[Gödel Agent: A Self-Referential Agent Framework for Recursive Self-Improvement](llm_agent/gödel_agent_a_self-referential_agent_framework_for_recursive_self-improvement.md)**

:   提出 Gödel Agent，一种受 Gödel Machine 启发的自引用 Agent 框架，能通过 Python monkey patching 在运行时读取和修改自身代码（包括修改自身的修改逻辑），实现递归自改进，在 DROP/MGSM/MMLU/GPQA 上超越手工设计和元学习优化的 Agent。

**[iAgent: LLM Agent as a Shield between User and Recommender Systems](llm_agent/iagent_llm_agent_as_a_shield_between_user_and_recommender_systems.md)**

:   提出用户-Agent-平台三层范式，在用户和推荐系统之间插入 LLM Agent 作为保护层，通过指令解析、知识获取、重排序和动态用户画像实现个性化推荐，在四个数据集上平均提升 16.6%，同时有效缓解回音室效应和低活跃用户的不公平问题。

**[Enhancing Interpretable Image Classification Through LLM Agents and Conditional Concept Bottleneck Models](llm_agent/llm_agent_image_classification.md)**

:   提出 Conditional Concept Bottleneck Models (CoCoBMs) 和 LLM-driven Concept Agent 框架，通过类别条件化的概念评分机制和基于环境反馈的动态概念库优化，在 6 个数据集上提升分类准确率 6% 的同时将可解释性提升约 30%。

**[Adaptive Tool Use in Large Language Models with Meta-Cognition Trigger](llm_agent/meco_metacognition_tool_use.md)**

:   提出 MeCo（Meta-Cognition Trigger），通过表示工程从 LLM 内部提取"元认知信号"——模型对自身能力的自我评估——来自适应决定是否需要调用外部工具，无需微调且计算开销极小，在多个骨干模型和基准上显著改善工具使用决策的准确性。

**[Unveiling Privacy Risks in LLM Agent Memory](llm_agent/mextra_agent_memory_privacy.md)**

:   本文系统研究了 LLM Agent 记忆模块的隐私风险，提出 MEXTRA 黑盒记忆提取攻击，通过精心设计的定位-对齐攻击 prompt 和自动化多样 prompt 生成方法，在医疗和网购两种 Agent 上成功提取大量私人查询记录。

**[A Multi-Agent Framework for Mitigating Dialect Biases in Privacy Policy Question-Answering Systems](llm_agent/multi_agent_dialect_bias_privacy_qa.md)**

:   提出一个双 Agent 框架（Dialect Agent + Privacy Policy Agent），通过方言感知翻译和迭代协作来消除隐私政策QA系统在不同英语方言间的性能差距，无需重训练或方言特定微调，在 PrivacyQA 和 PolicyQA 上将方言间最大性能差距降低最高 82%。

**[Multiple LLM Agents Debate for Equitable Cultural Alignment](llm_agent/multiple_llm_agents_debate_for_equitable.md)**

:   提出 Multi-Agent Debate 框架，让两个 LLM agent 围绕文化场景进行辩论并由 judge LLM 仲裁，在 NormAd-eti 基准上显著提升文化适应准确率和跨文化群体公平性，使 7-9B 小模型达到 27B 模型的性能水平。

**[NexusSum: Hierarchical LLM Agents for Long-Form Narrative Summarization](llm_agent/nexussum_narrative_summarization.md)**

:   提出 NexusSum，一个三阶段多Agent LLM框架（对话转描述→层次摘要→迭代压缩），无需微调即可处理书籍/电影/电视剧等长叙事文本的摘要生成，在 BookSum 上 BERTScore 提升达 30%。

**[OS-Kairos: Adaptive Interaction for MLLM-Powered GUI Agents](llm_agent/os-kairos_adaptive_interaction_for_mllm-powered_gui_agents.md)**

:   提出 OS-Kairos，通过协作探测框架标注每步置信度分数并微调进基座模型，使 GUI Agent 能在每步预测置信度、自主决定执行或请求人类干预，在复杂场景下任务成功率 (TSR) 从 OS-Atlas-Pro-7B 的 14.29% 提升到 88.20%，在 AITZ 和 Meta-GUI 基准上也有 24~87% 的绝对提升。

**[OS Agents: A Survey on MLLM-based Agents for General Computing Devices Use](llm_agent/os_agents_a_survey_on_mllm-based_agents_for_general_computing_devices_use.md)**

:   首篇全面综述 MLLM 驱动的操作系统代理（OS Agents），系统梳理其基础组件、构建方法、评估基准和未来方向。

**[OS Agents: A Survey on MLLM-based Agents for Computer, Phone and Browser Use](llm_agent/os_agents_survey_mllm.md)**

:   首个系统性综述基于（多模态）大语言模型的操作系统智能体（OS Agents），覆盖基础概念、构建方法（基础模型+Agent框架）、评估基准和商业产品，全面梳理了从CogAgent到Anthropic Computer Use等50+工作的技术演进。

**[OS-Genesis: Automating GUI Agent Trajectory Construction via Reverse Task Synthesis](llm_agent/os_genesis_gui_agent_trajectory.md)**

:   提出OS-Genesis，一种"先交互探索再逆向生成任务"的GUI agent轨迹数据合成范式，通过无人监督的UI元素遍历收集状态转移三元组，逆向合成任务指令后用Trajectory Reward Model质量控制，在AndroidWorld上将Qwen2-VL-7B性能从9.82%提升至17.41%，接近GPT-4o的23.70%。

**[Play2Prompt: Zero-shot Tool Instruction Optimization for LLM Agents via Tool Play](llm_agent/play2prompt_zero-shot_tool_instruction_optimization_for_llm_agents_via_tool_play.md)**

:   提出 Play2Prompt，通过让 LLM 自主"玩"工具（试探输入输出行为）来零样本地生成工具使用示例和优化工具文档，无需任何标注数据即可显著提升 LLM agent 的工具调用能力。

**[R2D2: Remembering, Replaying and Dynamic Decision Making with a Reflective Agentic Memory](llm_agent/r2d2_reflective_agentic_memory.md)**

:   R2D2 提出了一个结合 Remember（经验回放缓冲区 + A* 搜索导航）和 Reflect（错误反思 + 反思记忆存储）两范式的 Web Agent 框架，将 Web 导航从 Unknown MDP 转化为 Known MDP，在 WebArena 上导航错误减少 50%，任务完成率提升 3 倍，超越 SOTA 17%。

**[Select, Read, and Write: A Multi-Agent Framework of Full-Text-based Related Work Generation](llm_agent/select_read_and_write_a_multi-agent_framework_of_full-text-based_related_work_ge.md)**

:   提出 Select-Read-Write 三 Agent 协同框架，通过图感知的阅读顺序决策和共享工作记忆机制，实现基于论文全文（而非摘要）的 Related Work 自动生成，在 Llama3-8B / Claude-3-Haiku / GPT-4o 三个基座模型上均取得一致提升，Citation Graph 策略效果最优。

**[Self-Taught Agentic Long-Context Understanding](llm_agent/self_taught_agentic_long_ctx.md)**

:   提出 AgenticLU 框架，通过 Chain-of-Clarifications (CoC) 工作流让 LLM 自主生成澄清问题并检索相关上下文，再通过 SFT+DPO 两阶段微调将树搜索路径蒸馏到模型中，使 8B 模型在 128K 长上下文 QA 任务上大幅超越基线。

**[SMART: Self-Aware Agent for Tool Overuse Mitigation](llm_agent/smart_self-aware_agent_for_tool_overuse_mitigation.md)**

:   揭示 LLM Agent 中的"工具过度使用"现象（≥30% 的工具调用是不必要的），提出 SMART 元认知范式，通过在推理步骤中显式标注"知识驱动 vs 工具依赖"来训练 Agent 的自感知能力，7B 模型匹配 GPT-4o 水平。

**[SUDO: Screen-based Universal Detox2tox Offense for Agentic Security](llm_agent/sudo_rm_-rf_agentic_security.md)**

:   提出 SUDO 两阶段攻击框架针对计算机使用 Agent：静态阶段用 Detox2tox 将恶意请求去毒化→生成执行计划→回毒化恢复恶意载荷；动态阶段用检查清单迭代优化攻击，在 MANUS 上达到 63.19% 攻击成功率。

**[SynWorld: Virtual Scenario Synthesis for Agentic Action Knowledge Refinement](llm_agent/synworld_agentic_action_knowledge.md)**

:   SynWorld 提出让 Agent 在合成的虚拟场景中通过蒙特卡洛树搜索（MCTS）来探索和优化动作知识（工具描述和工作流），使 Agent 能够自主适应新环境的工具使用，在 ToolBench 上比 ReAct 基线提升约 9 个百分点。

**[Table-Critic: A Multi-Agent Framework for Collaborative Criticism and Refinement in Table Reasoning](llm_agent/table_critic_multi_agent.md)**

:   提出 Table-Critic 多智能体框架，通过 Judge-Critic-Refiner-Curator 四个专门化 Agent 的协作批评与迭代精化，配合自进化模板树累积批评知识，在 WikiTQ 和 TabFact 上分别实现 73.7% 和 91.7% 的准确率，大幅超越现有方法。

**[The Behavior Gap: Evaluating Zero-shot LLM Agents in Complex Task-Oriented Dialogs](llm_agent/the_behavior_gap_evaluating_zero-shot_llm_agents_in_complex_task-oriented_dialog.md)**

:   提出一个综合评估框架来量化 LLM agent 与人类专家在任务导向对话中的"行为差距"（dialog acts、工具使用、知识利用三个维度），发现行为差距随任务复杂度增加显著扩大（相关系数 0.963），缩小行为差距可平均提升 24.3% 性能。

**[Theorem-of-Thought: A Multi-Agent Framework for Abductive, Deductive, and Inductive Reasoning in Language Models](llm_agent/theorem-of-thought_a_multi-agent_framework_for_abductive_deductive_and_inductive.md)**

:   将推理建模为三个并行 agent（溢因、演绎、归纳），各自生成推理链并转化为形式化推理图(FRG)，通过 NLI 引导的贝叶斯信念传播评估内部一致性，选择得分最高的推理图作为最终答案，在 WebOfLies 和 MultiArith 上一致超越 CoT/SC 基线。

**[ToolCoder: A Systematic Code-Empowered Tool Learning Framework for Large Language Models](llm_agent/toolcoder_code_empowered_tool_learning.md)**

:   提出 ToolCoder 框架，将工具学习重新定义为代码生成任务，借鉴软件工程原则（需求分析→模块化设计→实现执行→错误调试→代码复用）让 LLM 通过生成和执行 Python 代码来完成多步工具调用，在 RestBench 和 API-Bank 上全面超越 ReAct、CodeAct 等基线方法。

**[ToolHop: A Query-Driven Benchmark for Evaluating Large Language Models in Multi-Hop Tool Use](llm_agent/toolhop_multi_hop_tool_use.md)**

:   提出ToolHop——首个query-driven构建的多跳工具使用评估数据集（995个多跳查询+3912个本地可执行工具），评估14个LLM后发现最强的GPT-4o仅达49.04%准确率，揭示了不同模型家族在工具使用策略上的显著差异。

**[ToolHop: A Query-Driven Benchmark for Evaluating Large Language Models in Multi-Hop Tool Use](llm_agent/toolhop_multi_hop_tool_use_benchmark.md)**

:   提出 ToolHop——一个包含 995 个多跳查询和 3912 个本地可执行工具的基准数据集，通过"查询驱动"的数据构建方式（先有查询再造工具）确保工具间有真实依赖关系和可验证答案，评测 14 个 LLM 发现最强的 GPT-4o 准确率仅 49%，揭示了不同模型家族在工具使用上的显著策略差异。

---

## 📦 模型压缩 { #model_compression }

**[AlignDistil: Token-Level Language Model Alignment as Adaptive Policy Distillation](model_compression/aligndistil_token_level_alignment.md)**

:   AlignDistil 证明了 RLHF 目标函数与 token 级蒸馏过程的理论等价性，并据此设计了一种简单的蒸馏方法：用 DPO 模型和反向 DPO 模型的 logit 分布线性组合构造教师分布，配合 token 自适应外推机制实现 token 级奖励优化，在 AlpacaEval 2.0、MT-Bench 和 Arena-Hard 上优于现有方法且收敛更快。

**[APB: Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks across GPUs](model_compression/apb_distributed_long_context.md)**

:   APB 提出了一种分布式长上下文推理框架，通过在序列并行框架中引入本地 KV cache 压缩和跨 GPU 传递压缩上下文块的机制，在不损失任务性能的前提下实现了相比 FlashAttn/RingAttn/StarAttn 分别高达 9.2x/4.2x/1.6x 的 prefill 加速。

**[ODLRI: Assigning Distinct Roles to Quantized and Low-Rank Matrices Toward Optimal Weight Decomposition](model_compression/assigning_distinct_roles_to_quantized_and_low-rank_matrices_toward_optimal_weigh.md)**

:   提出 ODLRI（Outlier-Driven Low-Rank Initialization）——在量化+低秩联合权重分解 $\mathbf{W} \approx \mathbf{Q} + \mathbf{LR}$ 中，将低秩分量专门分配给捕获激活敏感权重（异常值相关），量化分量处理剩余权重。通过这种"角色分配"初始化，稳定量化、降低激活感知误差，在 Llama2/3 和 Mistral 的低比特设置中一致提升困惑度和零样本精度。

**[BeamLoRA: Beam-Constraint Low-Rank Adaptation](model_compression/beamlora_beam_constraint_lora.md)**

:   BeamLoRA 发现 LoRA 模块中不同 rank 的重要性存在显著差异且随训练动态演变，受 beam search 启发，提出在训练过程中动态评估 rank 重要性、剪枝不重要的 rank 并将参数空间扩展给重要 rank，在固定总 rank 下提升性能，在三个基座模型的 12 个数据集上持续优于 LoRA 及其变体。

**[Beyond Text Compression: Evaluating Tokenizers Across Scales](model_compression/beyond_text_compression_tokenizers.md)**

:   本文系统评估了 6 种 tokenizer 在 350M 和 2.7B 参数模型上的影响，发现 tokenizer 选择对英文任务影响极小但对多语言任务（如机器翻译）有显著且跨尺度一致的影响，并提出了基于 Zipf 定律的新型内在评估指标，比文本压缩率能更好地预测多语言场景下的下游性能。

**["Give Me BF16 or Give Me Death"? Accuracy-Performance Trade-Offs in LLM Quantization](model_compression/bf16_or_death_quantization_tradeoffs.md)**

:   这是迄今最全面的 LLM 量化实证研究，在 Llama-3.1 全系列（8B/70B/405B）上对 FP8/INT8/INT4 进行了超过 50 万次评估，发现 FP8 几乎无损、INT8 仅降 1-3%、INT4 出奇地有竞争力，并给出了不同部署场景的量化格式选择建议。

**[Capture the Key in Reasoning to Enhance CoT Distillation Generalization](model_compression/capture_key_cot_distillation.md)**

:   提出 EDIT（mistakE-Driven key reasonIng step distillaTion），通过构造正确/错误配对的 dual CoTs 数据，利用最小编辑距离算法定位关键推理步骤，并以 token 级细粒度损失函数引导小模型聚焦学习这些关键步骤，而非简单模仿教师的推理形式。

**[Cross-Lingual Generalization and Compression: From Language-Specific to Shared Neurons](model_compression/cross_lingual_neurons_compression.md)**

:   本文通过追踪多语言语言模型预训练过程中的检查点，发现模型从语言特定表示逐渐压缩为跨语言共享表示：中间层的语言识别能力下降、语义概念的"专家神经元"跨语言对齐，操控从西班牙语数据提取的概念神经元后模型反而生成语义相关的英语文本。

**[DAC: A Dynamic Attention-aware Approach for Task-Agnostic Prompt Compression](model_compression/dac_prompt_compression.md)**

:   DAC 提出动态注意力感知的 prompt 压缩方法，通过融合信息熵和注意力分数作为 token 重要性度量，并动态感知压缩过程中的熵偏移来进行细粒度压缩，在 LongBench 上比 SOTA 方法提升平均 1.33 分。

**[DRAG: Distilling RAG for SLMs from LLMs to Transfer Knowledge and Mitigate Hallucination](model_compression/drag_distilling_rag_slm.md)**

:   DRAG 提出了一种从大模型向小模型蒸馏 RAG 能力的框架：用大模型（如 GPT-4o）为给定问题生成证据和知识图谱三元组，经排序过滤后作为结构化上下文输入给小模型（2B-9B），无需微调即可将小模型在 ARC-C 上提升高达 27.7%，同时显著减少幻觉。

**[DRPruning: Efficient Large Language Model Pruning through Distributionally Robust Optimization](model_compression/drpruning_robust_pruning.md)**

:   DRPruning 将分布稳健优化（DRO）引入 LLM 结构化剪枝，通过 scaling law 预测各领域最终 loss 作为参考、动态调整训练数据分布来平衡剪枝后各领域性能，在单语和多语设置下分别以 -5.59% PPL 和 +2.95% 下游任务的提升超越 Sheared LLaMA。

**[EAC-MoE: Expert-Selection Aware Compressor for Mixture-of-Experts Large Language Models](model_compression/eac_moe_expert_aware_compression.md)**

:   EAC-MoE 从 MoE 模型的专家选择特性出发，提出量化时校准路由器缓解 expert-shift 问题（QESC）+ 推理时基于专家选择频率动态剪枝不重要专家（PESF），在 Mixtral-8x7B 上实现 4.92× 内存压缩和 1.68× 推理加速且精度损失不到 1%。

**[EfficientQAT: Efficient Quantization-Aware Training for Large Language Models](model_compression/efficientqat.md)**

:   EfficientQAT 提出两阶段 QAT 框架——先逐块训练所有参数（Block-AP）提供良好初始化，再端到端训练量化参数（E2E-QP）捕获跨块交互，在单张 A100 上 41 小时完成 Llama-2-70B 的 2-bit 量化，精度仅降 3 点。

**[FedEx-LoRA: Exact Aggregation for Federated and Efficient Fine-Tuning of Large Language Models](model_compression/fedex_lora_federated_exact_aggregation.md)**

:   FedEx-LoRA 发现联邦学习中独立平均 LoRA 的 A 和 B 矩阵会导致不精确的全局更新（"乘积的均值≠均值的乘积"），通过在冻结权重矩阵中加入残差误差项实现精确聚合，在多个推理和 NLU 任务上一致优于 FedIT 和 FFA-LoRA。

**[Language Fusion for Parameter-Efficient Cross-lingual Transfer (FLARE)](model_compression/flare_crosslingual_lora.md)**

:   FLARE 在 LoRA 适配器的低秩瓶颈中通过轻量线性/非线性变换融合源语言（英语）和目标语言的逐层表示，无需额外参数即可实现参数高效的跨语言迁移，在 Llama 3.1 上 QA 精确匹配提升 4.9%。

**[Flipping Knowledge Distillation: Leveraging Small Models' Expertise to Enhance LLMs in Text Matching](model_compression/flipping_kd_small_to_large.md)**

:   本文提出"反向知识蒸馏"范式——让 LLM 从微调过的小模型学习文本匹配的领域专家知识，通过将 decoder-only LLM 重新解释为 encoder-decoder 架构（用 LoRA 的压缩矩阵做 encoder）并设计 Margin-aware Contrastive Loss 来对齐表示相似度。

**[A Silver Bullet or a Compromise for Full Attention? A Comprehensive Study of Gist Token-based Context Compression](model_compression/gist_token_context_compression.md)**

:   系统研究Gist Token上下文压缩方法，提出统一框架分类现有架构（记忆位置×粒度），发现Fine-grained KV Cache在RAG/QA上近无损但在合成召回上有明显缺陷，识别出三种失败模式（边界丢失/意外丢失/中途丢失），并提出细粒度自编码和分段token重要性估计两种改进策略。

**[Graph Counselor: Adaptive Graph Exploration via Multi-Agent Synergy to Enhance LLM Reasoning](model_compression/graph_counselor_multiagent_graphrag.md)**

:   Graph Counselor 提出了一个多智能体协作的 GraphRAG 推理框架，通过 Planning/Thought/Execution 三个 Agent 自适应提取图结构信息，并引入多视角自反思机制纠正推理偏差，在多个图推理任务上超越现有方法。

**[L4Q: Parameter Efficient Quantization-Aware Fine-Tuning on Large Language Models](model_compression/l4q_parameter_efficient_quantization_aware_finetuning.md)**

:   提出 L4Q，将量化感知训练 (QAT) 与 LoRA 深度整合：先合并权重与LoRA参数再统一量化，通过定制反向传播路径消除权重梯度存储开销，实现联合优化量化与微调参数，在4-bit和3-bit量化下显著超越现有方法。

**[Language Models Resist Alignment: Evidence From Data Compression](model_compression/language_models_resist_alignment.md)**

:   本文从压缩理论视角揭示了LLM存在"弹性"(elasticity)现象——模型倾向于保持预训练分布而抵触对齐分布，且对齐后的模型在受到扰动时会以与数据量差距成反比的速率回弹到预训练状态，这解释了为什么对齐如此脆弱且容易被少量微调逆转。

**[Unveiling Language-Specific Features in Large Language Models via Sparse Autoencoders](model_compression/language_specific_features.md)**

:   利用 Sparse Autoencoders (SAEs) 分析多语言 LLM 的内部表示，发现存在强烈的语言特定 SAE features，这些 features 不仅与语言特有 token 相关还与语言上下文相关，消融它们只影响对应语言能力，且多个语言 features 之间存在协同效应；进一步利用这些 features 增强 steering vectors 实现对生成语言的精确控制。

**[Towards the Law of Capacity Gap in Distilling Language Models](model_compression/law_of_capacity_gap_distilling_language_models.md)**

:   揭示了语言模型蒸馏中的"容量差距定律"——最优教师模型的参数量与学生模型成线性关系（约 2.5 倍），将 LLM 蒸馏中的"不可能三角"转化为可解问题，并据此成功蒸馏出 3B 的 MiniMA 模型。

**[Memorization Inheritance in Sequence-Level Knowledge Distillation for Neural Machine Translation](model_compression/memorization_inheritance_seqkd.md)**

:   本文首次系统研究了序列级知识蒸馏（SeqKD）中教师模型的记忆行为如何传递给学生模型，发现学生模型虽未直接接触原始训练数据，但其提取式记忆率比基线模型高 57%，幻觉率也增加，并提出 Adaptive-SeqKD 通过在高质量子集上微调教师来缓解这些问题。

**[MoQAE: Mixed-Precision Quantization for Long-Context LLM Inference via Mixture of Quantization-Aware Experts](model_compression/moqae_mixed_precision_kv_cache.md)**

:   MoQAE 创造性地将不同量化比特宽度配置视为 MoE 中的"专家"，通过轻量路由器学习每个 chunk 的最优量化策略，结合路由冻结和路由共享机制，在几乎不损失精度的情况下大幅减少长上下文推理的 KV cache 内存。

**[mPLUG-DocOwl2: High-resolution Compressing for OCR-free Multi-page Document Understanding](model_compression/mplug_docowl2_doc_compress.md)**

:   提出High-resolution DocCompressor模块，利用低分辨率全局视觉特征作为query通过交叉注意力将高分辨率文档图像压缩为仅324个token（不到同类方法的20%），在多页文档理解benchmark上达到SOTA且首token延迟降低50%+。

**[Unraveling LoRA Interference: Orthogonal Subspaces for Robust Model Merging](model_compression/osrm_lora_merging_orthogonal.md)**

:   OSRM 发现 LoRA 模型合并失败的根因是参数与数据分布的交互干扰（而非仅仅是参数冲突），提出在微调前通过数据协方差矩阵的特征分解来初始化 LoRA 矩阵 A，使其子空间与其他任务的数据分布正交，从而在合并时最小化跨任务干扰，在 8 个数据集、5 个模型上显著提升合并性能。

**[Prompt Candidates, then Distill: A Teacher-Student Framework for LLM-driven Data Annotation](model_compression/prompt_distill_teacher_student.md)**

:   提出候选标注+蒸馏范式（CanDist）——当 LLM 对样本不确定时输出所有可能标签（而非强制给唯一标签），然后用小语言模型（SLM）从候选标注中蒸馏出唯一标签，理论证明候选标注蒸馏比直接使用单标签有更好的理论保证，在六个文本分类任务上验证有效。

**[PTQ1.61: Push the Real Limit of Extremely Low-Bit Post-Training Quantization Methods for Large Language Models](model_compression/ptq161_low_bit_quantization.md)**

:   首次将LLM权重真正量化到1.61-bit（此前号称sub-2bit的方法实际都超过2bit），通过一维结构化掩码（仅增加0.0002-bit/权重）保留显著通道、块级缩放因子优化和量化预处理三大创新，在LLaMA系列上以更低比特超越BiLLM和PB-LLM。

**[Quaff: Quantized Parameter-Efficient Fine-Tuning under Outlier Spatial Stability Hypothesis](model_compression/quaff_quantized_peft.md)**

:   本文提出 Outlier Spatial Stability Hypothesis (OSSH)——微调期间激活异常通道的空间位置保持稳定——并基于此设计了 Quaff 框架，通过目标动量缩放仅处理少量不变的异常通道，实现 1.73× 延迟降低和 30% 内存节省，同时在 GPQA 上精度还提升了 0.6%。

**[SEE: Strategic Exploration and Exploitation for Cohesive In-Context Prompt Optimization](model_compression/see_strategic_exploration_exploitation_prompt_optimization.md)**

:   本文提出 SEE 框架，首次将指令（instruction）和示例（examples）作为整体进行联合优化，采用元启发式优化原则设计四阶段探索-利用策略，在35个基准任务上实现平均13.94%的准确率提升并降低58.67%的计算成本。

**[Mitigating Selection Bias with Node Pruning and Auxiliary Options](model_compression/selection_bias_node_pruning.md)**

:   提出Bias Node Pruning (BNP)和Auxiliary Option Injection (AOI)两种互补方法，从模型内部和输入端同时缓解LLM在多选题中的选择偏差，仅剪除0.002%权重即可将Llama-3准确率从52.3%提升至65.3%（+24.9%组合提升）。

**[Semantic Exploration with Adaptive Gating for Efficient Problem Solving with Language Models](model_compression/semantic_exploration_adaptive_gating.md)**

:   针对 LLM 树搜索推理中"简单题也做复杂搜索"和"语义重复路径反复扩展"两大浪费问题，提出 SEAG 框架：先用 entropy 门控决定是否启动树搜索，再用语义聚类合并等价推理步骤，最终在准确率平均提升 4.3% 的同时仅需 RAP 31% 的推理开销。

**[State-offset Tuning: State-based Parameter-Efficient Fine-Tuning for State Space Models](model_compression/state_offset_tuning_ssm_peft.md)**

:   针对 SSM（如 Mamba）提出 State-offset Tuning，一种新的"状态基"PEFT 方法家族，通过在每个时间步直接注入可训练的状态偏移量 $h'$ 替代 Prefix-Tuning 的虚拟 token，解决了 prompt-based 方法在 SSM 上表达能力受限的问题，在更少参数量下持续优于 LoRA 和 Prefix-Tuning。

**[STUN: Structured-Then-Unstructured Pruning for Scalable MoE Pruning](model_compression/stun_moe_pruning.md)**

:   STUN 提出了结构化→非结构化的两阶段 MoE 剪枝方法，第一阶段用 O(1) 前向传播实现可扩展的专家级剪枝，第二阶段在剩余专家内做非结构化剪枝，在 480B 参数的 Snowflake Arctic 上以 40% 稀疏度几乎无性能损失。

**[TableLoRA: Low-rank Adaptation on Table Structure Understanding for Large Language Models](model_compression/table_lora_structure_understanding.md)**

:   TableLoRA 提出面向表格任务的专用 LoRA 模块，通过特殊 token 编码器改善表格序列化，并用 2D LoRA 编码单元格的行列位置信息，在参数高效微调设置下相比 vanilla LoRA 在 HiTab 上提升 5.9%，弥合了 LoRA 与全量微调之间 40.56% 的性能差距。

**[Trans-PEFT: Transferable Parameter-Efficient Fine-Tuning on Evolving Base Models](model_compression/trans_peft_transferable.md)**

:   Trans-PEFT 发现基座模型更新（如 Qwen2→Qwen2.5）主要改变 FFN 层的任务知识存储而较少影响 Attention 层的任务模式，据此提出层内知识掩码和跨层知识丢弃两种策略，使在旧版本上训练的 PEFT 模块可直接迁移到新版本而不需重新微调，性能提升可达 30%。

**[UniICL: An Efficient ICL Framework Unifying Compression, Selection, and Generation](model_compression/uniicl_icl_framework.md)**

:   提出 UniICL 框架，用**一个冻结的 LLM** 同时完成 demonstration 压缩（compress→virtual tokens）、demonstration 选择（基于压缩后的 virtual token 相似度排序）和最终响应生成三个任务，仅需 17M 可训练参数（projection layer + learnable embedding），配合 Demonstration Bank 缓存机制避免重复压缩，实现 12× 压缩率下从 4-shot 扩展到 64-shot ICL（24GB 显存内），在多个 out-of-domain 数据集上超越 AutoCompressor、ICAE、LLMLingua 等基线。

**[UniQuanF: Unifying Uniform and Binary-coding Quantization for Accurate Compression of Large Language Models](model_compression/uniquanf_unified_quantization.md)**

:   UniQuanF 统一了均匀量化（UQ,表现力弱但优化性强）和二进制编码量化（BCQ,表现力强但优化性差）的优势，通过统一初始化、局部周期映射和统一定理，实现无额外部署开销的高精度 LLM 量化，在 GSM8K 上提升最高 4.60%。

---

## 💡 LLM 推理 { #llm_reasoning }

**[Aristotle: Mastering Logical Reasoning with A Logic-Complete Decompose-Search-Resolve Framework](llm_reasoning/aristotle_logical_reasoning.md)**

:   提出 Aristotle 逻辑推理框架，将符号表达式和逻辑规则全面融入 Decompose-Search-Resolve 的每个阶段，通过逻辑分解器、搜索路由器和消解器三大组件实现逻辑完备的推理，在多个逻辑推理基准上以 GPT-4 平均提升 4.5%、GPT-4o 平均提升 5.4% 超越 SOTA。

**[BPP-Search: Enhancing Tree of Thought Reasoning for Mathematical Modeling Problem Solving](llm_reasoning/bpp-search_enhancing_tree_of_thought_reasoning_for_mathematical_modeling_problem.md)**

:   提出 BPP-Search 算法，将 Beam Search、过程奖励模型 (PRM) 和 Pairwise Preference 机制整合到 Tree-of-Thought 框架中，用于运筹学数学建模问题的自动求解，在 StructuredOR 等数据集上以更少的推理步骤显著超越 CoT/SC/ToT 基线。

**[Chain-of-Reasoning: Towards Unified Mathematical Reasoning in Large Language Models via a Multi-Paradigm Perspective](llm_reasoning/chain-of-reasoning_towards_unified_mathematical_reasoning_in_large_language_mode.md)**

:   提出 Chain-of-Reasoning (CoR) 框架，将自然语言推理 (NLR)、算法推理 (AR) 和符号推理 (SR) 三种范式统一整合，通过渐进范式训练 (PPT) 策略训练出 CoR-Math-7B，在定理证明任务上零样本超越 GPT-4o 41%，在 MATH 上超越 RL 方法 15%。

**[Chain-of-Reasoning: Towards Unified Mathematical Reasoning in Large Language Models via a Multi-Paradigm Perspective](llm_reasoning/chain_of_reasoning_unified_math.md)**

:   提出 Chain-of-Reasoning（CoR）框架，将自然语言推理（NLR）、算法推理（AR）和符号推理（SR）三种范式统一在一个推理链中，通过渐进范式训练（PPT）策略让 7B 模型（CoR-Math-7B）在零样本下超越 GPT-4o 41% 的定理证明准确率，在 MATH 基准上超过 RL 方法 15%。

**[ClozeMath: Improving Mathematical Reasoning in Language Models by Learning to Fill Equations](llm_reasoning/clozemath_improving_mathematical_reasoning_in_language_models_by_learning_to_fil.md)**

:   提出 ClozeMath，一种在微调时额外加入"方程填空"（text-infilling）目标的训练策略，让模型学会根据自然语言推理链预测被遮掩的数学方程，在 GSM8K/MATH 上显著超越 Masked Thought 基线，同时提高鲁棒性和 test-time scaling 效率。

**[CoT-UQ: Improving Response-wise Uncertainty Quantification in LLMs with Chain-of-Thought](llm_reasoning/cot-uq_improving_response-wise_uncertainty_quantification_in_llms_with_chain-of-.md)**

:   针对 LLM 在推理任务中过度自信的问题，提出 CoT-UQ 框架，将 CoT 推理步骤中的关键词提取和重要性评分整合到不确定性量化过程中，在逻辑和数学推理任务上 AUROC 平均提升 5.9%。

**[Critic-CoT: Boosting the Reasoning Abilities of Large Language Model via Chain-of-Thoughts Critic](llm_reasoning/critic-cot_boosting_the_reasoning_abilities_of_large_language_model_via_chain-of.md)**

:   提出 Critic-CoT 框架，通过逐步 Chain-of-Thought 批判范式和无需人工标注的弱监督数据自动构建，将 LLM 的自我批判从 System-1 式直觉判断推向 System-2 式慎重逐步分析；两阶段训练（GPT-4 蒸馏 + 自我批判）使 Llama-3-70B-Instruct 在 GSM8K 从 89.6% 提升至 95.4%，MATH500 从 50.4% 提升至 68.4%，并发现批判能力与任务求解能力可以相互增强。

**[Fine-Tuning on Diverse Reasoning Chains Drives Within-Inference CoT Refinement in LLMs](llm_reasoning/dcot_diverse_cot_refinement.md)**

:   提出 Diverse Chain of Thought (DCoT) 训练方法，通过在单次推理中生成多条串行推理链实现"推理内自修正"（within-inference refinement），在 1.3B–70B 模型上均超越标准 CoT 基线，尤其在大输出空间任务（数值/抽取型）上提升显著。

**[DeFine: Decision-Making with Analogical Reasoning over Factor Profiles](llm_reasoning/define_decision-making_with_analogical_reasoning_over_factor_profiles.md)**

:   提出 DeFine 框架，从财报电话会议等复杂场景的语音转录文本中构建概率因子画像(factor profile)，结合 Bradley-Terry 模型识别关键因子并通过因子画像间的 KL 散度做类比推理，用于辅助 LLM 在不确定性下做投资决策，准确率和 F1 均超越基线。

**[Dynamic and Generalizable Process Reward Modeling](llm_reasoning/dgprm_dynamic_process_reward.md)**

:   DG-PRM 提出了一种动态可泛化的过程奖励建模框架，通过奖励树存储多维度评估标准并动态选择步骤相关的奖励信号，用 Pareto 支配估计处理多面奖励，在 PRMBench 上达到 SOTA 且具有优异的跨领域泛化能力。

**[DRT: Deep Reasoning Translation via Long Chain-of-Thought](llm_reasoning/drt_deep_reasoning_translation_via_long_chain-of-thought.md)**

:   将长 CoT 推理引入机器翻译，构建多智能体框架（翻译器→顾问→评估器）迭代精炼含比喻/隐喻的文学翻译，合成 22K 长思维翻译训练样本，训练的 DRT-14B 在文学翻译上超越 QwQ-32B 和 DeepSeek-R1-Distill-32B 等大模型。

**[Unlocking General Long Chain-of-Thought Reasoning Capabilities of Large Language Models via Representation Engineering](llm_reasoning/glore_long_cot_representation.md)**

:   从表示空间角度发现 LLM 将长 CoT 推理编码为一种与普通 CoT 明确区分的通用能力，提出 GLoRE（General Long CoT Reasoning via Representation Engineering）——通过对比推理模式注入和领域特定表示调整来解锁长 CoT 能力，无需训练即可在域内和跨域场景下超越 SFT 方法。

**[Improve Vision Language Model Chain-of-thought Reasoning](llm_reasoning/improve_vlm_cot_reasoning.md)**

:   通过GPT-4o蒸馏193k CoT数据做SFT + 基于答案正确性构建偏好对做DPO，显著提升VLM的CoT推理能力（LLaVA-Reasoner在8个benchmark上CoT平均提升12.6%），且CoT训练还能反哺直接预测性能。

**[Local Look-Ahead Guidance via Verifier-in-the-Loop for Automated Theorem Proving](llm_reasoning/local_look-ahead_guidance_via_verifier-in-the-loop_for_automated_theorem_proving.md)**

:   提出 LeanListener，在自动定理证明(ATP)中引入 verifier-in-the-loop 设计，利用 Lean 验证器在每步提供中间反馈（子目标数变化）而非仅轨迹级奖励，通过在线 GRPO 训练使 ReProver 的 tactic 有效率和证明率均获提升，证明速度快 20%。

**[LogicPro: Improving Complex Logical Reasoning via Program-Guided Learning](llm_reasoning/logicpro_program_guided_reasoning.md)**

:   提出 LogicPro 数据合成方法，利用 LeetCode 算法题和 Python 代码解作为逻辑源，通过"问题生成→代码中间变量提取→程序引导推理生成"三步流水线，从 2360 道算法题合成 540K 高质量文本推理数据，在 BBH27、LogicBench、DROP 等多个 OOD 基准上显著超越现有推理数据集。

**[Linguistic Generalizability of Test-Time Scaling in Mathematical Reasoning](llm_reasoning/mclm_multilingual_test_time_scaling.md)**

:   提出 MCLM（55 语言的竞赛级数学基准），发现三种 test-time scaling 方法（ORM/PRM/Budget Forcing）在英语上提升显著（如 AIME +20 分），但在其他语言上平均仅提升 1.94 分，表明 test-time scaling 的多语言泛化能力严重不足。

**[MMBoundary: Advancing MLLM Knowledge Boundary Awareness through Reasoning Step Confidence Calibration](llm_reasoning/mmboundary_reasoning_step_confidence.md)**

:   提出 MMBoundary 框架，通过在推理链的每一步插入自然语言置信度表述（而非只在最终回答后给置信度），结合文本+跨模态的自奖励信号估计置信度，并用 SFT+RL 两阶段训练实现步级置信度校准，平均降低 7.5% 校准误差并提升 8.3% 任务准确率。

**[One Missing Piece for Open-Source Reasoning Models: A Dataset to Mitigate Cold-Starting Short CoT LLMs in RL](llm_reasoning/one_missing_piece_for_open-source_reasoning_models_a_dataset_to_mitigate_cold-st.md)**

:   构建 100K 长 CoT 数据集解决开源短 CoT LLM 在 RL 训练中的冷启动问题——用 1K 种子数据捕捉 o1 的推理流模式，然后用短 CoT LLM (GPT-4o) 扩展为长推理链，训练后的模型在 RLVR 初始化后获得 2-3 倍更大的收益。

**[PCoT: Persuasion-Augmented Chain of Thought for Detecting Fake News and Social Media Disinformation](llm_reasoning/pcot_persuasion-augmented_chain_of_thought_for_detecting_fake_news_and_social_me.md)**

:   提出 PCoT，一种零样本方法，利用说服知识增强虚假信息检测——受心理学研究启发（识别说服谬误可提高假新闻检测），两阶段处理：先识别分析说服信号，再将说服分析融入推理判断，平均提升 15%。

**[PCoT: Persuasion-Augmented Chain of Thought for Detecting Fake News and Social Media Disinformation](llm_reasoning/pcot_persuasion_chain_of_thought_fake_news.md)**

:   提出 PCoT（说服增强的思维链）方法，通过两阶段推理——先让 LLM 分析文本中的说服策略，再结合说服分析结果判断是否为虚假信息——在零样本设置下，5 个 LLM 和 5 个数据集上平均提升 15% 的检测 F1。

**[Ranked Voting based Self-Consistency of Large Language Models](llm_reasoning/ranked_voting_based_self-consistency_of_large_language_models.md)**

:   将 Self-Consistency 的多数投票升级为排序投票，让 LLM 每次推理生成多个候选答案的偏好排序而非单一答案，用三种排序投票方法（IRV/BCV/MRRV）聚合多次推理的排序信息，在 6 个数据集上一致超越传统 SC，最高提升 12.46%。

**[Rethinking the Role of Prompting Strategies in LLM Test-Time Scaling: A Perspective of Probability Theory](llm_reasoning/rethinking_the_role_of_prompting_strategies_in_llm_test-time_scaling_a_perspecti.md)**

:   本文在 6 个 LLM × 8 种 prompting 策略 × 6 个 benchmark 上系统实验发现，随着 majority voting 采样次数增加，简单的 CoT 始终超越复杂 prompting 策略；并从概率论角度给出理论证明，提出 $O(1)$ 复杂度的 scaling 性能预测方法和两种改进策略。

**[Revisiting Self-Consistency from Dynamic Distributional Alignment Perspective on Answer Aggregation](llm_reasoning/revisiting_self-consistency_from_dynamic_distributional_alignment_perspective_on.md)**

:   将 Self-Consistency 重新理解为采样分布与真实答案分布的动态对齐问题，揭示温度不仅控制采样随机性还直接塑造真实答案分布，据此提出置信度驱动的三阶段动态温度调节机制（FSD 阈值理论推导），在 10 个模型 × GSM8K/MATH 上零训练开销同时提升平均和最佳性能。

**[Safe: Enhancing Mathematical Reasoning in Large Language Models via Retrospective Step-aware Formal Verification](llm_reasoning/safe_math_reasoning.md)**

:   提出 Safe 框架，首次利用 Lean 4 形式化语言对 LLM 数学推理的每一步进行回顾性逐步验证，通过自动形式化+自动定理证明检测幻觉，并与前瞻性 PRM 分数融合，在多个数学数据集上取得 SOTA，同时发布包含 30,809 条形式化声明的 FormalStep 基准。

**[Stepwise Reasoning Disruption Attack of LLMs](llm_reasoning/seed_stepwise_reasoning_disruption_attack.md)**

:   提出 SEED（Stepwise rEasoning Error Disruption）攻击方法，通过在 LLM 的推理链前几步中巧妙注入细微错误（如微调计算数字），让模型在后续推理中自然传播错误得出错误答案，兼容零样本/少样本设置，GPT-4o 检测率低至 0.8%，揭示了 LLM 逐步推理过程的严重安全漏洞。

**[SoftCoT: Soft Chain-of-Thought for Efficient Reasoning with LLMs](llm_reasoning/softcot_soft_chain_of_thought.md)**

:   提出 SoftCoT，用一个冻结的小型辅助模型（如 LLaMA-3.2-1B）生成实例特定的"软思维 token"（连续隐状态），通过可训练的投影模块映射到主 LLM 的表示空间作为推理前缀，实现参数高效的连续空间 CoT 推理，避免了全模型微调导致的灾难性遗忘问题。

**[STRICTA: Structured Reasoning in Critical Text Assessment for Peer Review and Beyond](llm_reasoning/stricta_structured_reasoning_peer_review.md)**

:   提出 STRICTA 框架，将专家文本评估（如论文审稿）建模为基于结构因果模型（SCM）的逐步推理图，收集 40+ 生物医学专家对 22 篇论文的 4000+ 推理步骤数据，揭示先验知识差异是评审分歧的主因、写作风格对终审影响过大，并发现 LLM 在人工监督下可有效辅助结构化评估。

**[Is That Your Final Answer? Test-Time Scaling Improves Selective Question Answering](llm_reasoning/test_time_scaling_selective_qa.md)**

:   首次评估 test-time scaling 模型（DeepSeek R1、S1）在选择性问答（允许拒绝回答）场景下的表现，发现增加推理计算不仅提高正确率还提升对正确答案的置信度，提出基于置信度阈值的选择函数和 "Jeopardy Odds" 效用函数来评估非零错误惩罚下的 test-time scaling 性能。

**[ThinkGuard: Deliberative Slow Thinking Leads to Cautious Guardrails](llm_reasoning/thinkguard_deliberative_slow_thinking_leads_to_cautious_guardrails.md)**

:   提出 ThinkGuard，一种批判增强的安全护栏模型，通过从强 LLM 蒸馏结构化批判(安全标签+详细理由)来训练护栏模型，实现“慢思考”式安全判断，相比 LLaMA Guard 3 准确率提升 16.1%、宏 F1 提升 27.0%。

**[Towards Better Chain-of-Thought: A Reflection on Effectiveness and Faithfulness](llm_reasoning/towards_better_chain-of-thought_a_reflection_on_effectiveness_and_faithfulness.md)**

:   系统分析 CoT 有效性和忠实性的影响因素，发现 CoT 有效性取决于问题难度、信息增益和信息流向，忠实性是有效性的关键前提，提出 QUIRE 方法（先回忆再增强）提升有效性 2.4% 和忠实性 5.6%。

**[Towards Safety Reasoning in LLMs: AI-agentic Deliberation for Policy-embedded CoT Data Creation](llm_reasoning/towards_safety_reasoning_in_llms_ai-agentic_deliberation_for_policy-embedded_cot.md)**

:   提出 AIDsafe 多智能体审议框架，自动生成嵌入安全策略的 CoT 训练数据，通过多 agent 协作扩展安全推理链并过滤欺骗性/冗余思维，微调后的模型在安全性和越狱鲁棒性上显著提升且不影响实用性。

**[TRACT: Regression-Aware Fine-tuning Meets Chain-of-Thought Reasoning](llm_reasoning/tract_regression_cot.md)**

:   提出 TRACT，一种两阶段回归感知微调方法，将 CoT 推理与回归损失（squared error）结合，用于提升 LLM-as-a-Judge 场景中的数值评分精度，显著优于仅用交叉熵训练或仅用回归损失的现有方案。

**[Training Turn-By-Turn Verifiers For Dialogue Tutoring Agents The Curious Case Of](llm_reasoning/training_turn-by-turn_verifiers_for_dialogue_tutoring_agents_the_curious_case_of.md)**

:   提出 **Traver**（Trace-and-Verify）agent 工作流，通过**知识追踪**显式估计学生知识状态 + **逐轮验证器**（turn-by-turn verifier）对候选辅导话语打分选优，并设计 **Dict** 自动评估协议（模拟学生 + 代码生成测试），在编程辅导场景中将学生 Pass 率从 38.7% 提升至 43.7%（相对提升 106.5%），显著超越 Vanilla Instruct、Self-Refine 和 TreeInstruct。

**[Unveiling the Key Factors for Distilling Chain-of-Thought Reasoning](llm_reasoning/unveiling_the_key_factors_for_distilling_chain-of-thought_reasoning.md)**

:   系统研究影响 CoT 蒸馏的三大因素（粒度、格式、教师模型），发现 SLM 与粒度呈非单调关系、格式影响较小、强教师不总是更好。

---

## ⚡ LLM 效率 { #llm_efficiency }

**[Extending LLM Context Window with Adaptive Grouped Positional Encoding: A Training-Free Method](llm_efficiency/adaptive_grouped_pe_context_window.md)**

:   提出 AdaGroPE（Adaptive Grouped Positional Encoding），一种无需训练的即插即用方法，通过让位置复用次数随距离递增式增长、并根据输入序列长度动态调整位置编码映射，将 LLM 上下文窗口外推到远超预训练长度，在多个 benchmark 上达到 SOTA 甚至超过原生长上下文模型。

**[CLaSp: In-Context Layer Skip for Self-Speculative Decoding](llm_efficiency/clasp_self_speculative_decoding.md)**

:   CLaSp 提出一种无需训练的自推测解码方法，通过动态规划算法在每个验证步骤后根据上下文动态调整跳层策略，利用上一次验证的完整隐状态作为目标来选择最优跳层集合，在 LLaMA3 系列上实现 1.3-1.7× 加速且不改变生成分布。

**[CNNSum: Exploring Long-Context Summarization with Large Language Models in Chinese Novels](llm_efficiency/cnnsum_exploring_long-context_summarization_with_large_language_models_in_chines.md)**

:   构建了 CNNSum——基于中文小说的多尺度长文本摘要基准（695 样本，16k-128k tokens），通过人工标注确保质量，系统测评了 20+ 个 LLM，发现高级 LLM 倾向生成主观评述导致摘要模糊、小模型性价比更高、Base 版微调效果优于 Chat 版，且用短文本数据微调即可显著提升长文本摘要能力。

**[Dynamic Chunking and Selection for Reading Comprehension of Ultra-Long Context in Large Language Models](llm_efficiency/dynamic_chunking_and_selection_for_reading_comprehension_of_ultra-long_context_i.md)**

:   提出 Dynamic Chunking and Selection (DCS)，通过基于语义相似度的动态分块和问题感知分类器的块选择，解决长文本固定分块导致的语义断裂问题，在 12 个长文本 QA 数据集上以 Llama3 为基座实现 single-hop 平均 35.50（+28.6%）和 multi-hop 平均 29.07（+20.0%）的提升，且在 256k token 输入下保持鲁棒。

**[FlashBack: Efficient Retrieval-Augmented Language Modeling for Fast Inference](llm_efficiency/flashbackefficient_retrieval-augmented_language_modeling_for_long_context_infere.md)**

:   针对检索增强语言模型(RALM)中因检索内容前置(prepending)导致 KV cache 反复重算的推理效率问题，提出 FlashBack，将检索内容后置(appending)以保留输入的 KV cache，并用 Marking Token + LoRA 微调适配新的上下文模式，在 Llama 2-7B 上实现最高 4 倍推理加速且 perplexity 持平。

**[GigaChat Family: Efficient Russian Language Modeling Through Mixture of Experts Architecture](llm_efficiency/gigachat_family_efficient_russian_language_modeling_through_mixture_of_experts_a.md)**

:   介绍 GigaChat 系列——首个从头为俄语设计并预训练的 MoE 架构 LLM 家族，包含 20B 总参数/3.3B 激活参数的基座和指令微调模型，在俄语 benchmark 上达到同规模 SOTA，训练速度是同量级 dense 模型的 2 倍，推理延迟降低 40%。

**[Graph of Records: Boosting Retrieval Augmented Generation for Long-context Summarization with Graphs](llm_efficiency/gor_rag_long_context_summary.md)**

:   提出 Graph of Records（GoR），将 LLM 历史响应与检索文本块构建为图结构，用 GNN 学习节点间的语义和逻辑关联，配合 BERTScore 自监督训练目标，在四个长文本全局摘要数据集上比检索基线提升 8-19%（ROUGE 指标）。

**[GradOT: Training-free Gradient-preserving Offsite-tuning for Large Language Models](llm_efficiency/gradot_offsite_tuning.md)**

:   从优化理论角度首次系统分析 Offsite-tuning 问题，提出梯度保持压缩分数（GCS），并基于此设计了 GradOT 方法，对 MHA 使用动态秩分解（DRD）、对 MLP 使用选择性通道剪枝（SCP），在免训练条件下同时实现性能保持和隐私保护。

**[KV-Latent: Dimensional-level KV Cache Reduction with Frequency-aware Rotary Positional Embedding](llm_efficiency/kv_latent_cache_reduction.md)**

:   提出KV-Latent范式，通过对KV头的维度进行降采样将其映射到潜空间，仅需不到1%预训练量的额外训练即可恢复性能，在LLaMA-3-8B上KV Cache减少50%（dqk=dvo=64→50%缓存），同时TTFT延迟降低8%。

**[LADM: Long-context Training Data Selection with Attention-based Dependency Measurement](llm_efficiency/ladm_long_context_data.md)**

:   提出 LADM 框架，利用注意力机制的内在检索能力来度量长上下文数据中的跨 span 依赖关系，从大规模预训练语料中高效筛选高质量长上下文训练数据，仅用 1B token 持续预训练即可显著提升多种 LLM 的长上下文能力。

**[Literary Evidence Retrieval via Long-Context Language Models](llm_efficiency/literary_evidence_retrieval_via_long-context_language_models.md)**

:   构建文学证据检索 benchmark，要求模型给定完整小说文本和文学评论摘录后生成缺失的引用，Gemini Pro 2.5 达 62.5% 准确率超过人类专家(55%)，但最佳开源模型仅 29.1%，揭示了巨大能力差距。

**[What Really Matters in Many-Shot Attacks? An Empirical Study of Long-Context Vulnerabilities in LLMs](llm_efficiency/many_shot_attacks_long_context.md)**

:   系统分析 Many-Shot Jailbreaking（MSJ）攻击的关键因素，发现上下文长度是攻击成功的决定性因素，而内容的有害性、主题、格式几乎不重要——即使重复安全内容、随机无意义文本（Lorem Ipsum）都能在长上下文下突破模型安全对齐。

**[Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](llm_efficiency/native_sparse_attention.md)**

:   DeepSeek提出NSA——一种原生可训练的稀疏注意力机制，通过"压缩+选择+滑动窗口"的层次化稀疏策略和硬件对齐的Triton kernel设计，在27B参数模型上实现了超越Full Attention的性能，同时在64k序列上获得前向9倍、解码11.6倍的加速。

**[On Many-Shot In-Context Learning for Long-Context Evaluation](llm_efficiency/on_many-shot_in-context_learning_for_long-context_evaluation.md)**

:   深入研究 many-shot ICL 用于长上下文语言模型评估，提出 Sample Learning Ratio 指标区分 SSL 和 ASL 任务，构建 ManyICLBench 基准全面评测 12 个 LCLM。

**[Ref-Long: Benchmarking the Long-Context Referencing Capability of Long-Context Language Models](llm_efficiency/ref-long_benchmarking_the_long-context_referencing_capability_of_long-context_la.md)**

:   提出 Ref-Long benchmark，从"引用定位"（给定 key 识别哪些文档引用了它并返回索引）这一被忽视的维度评估长上下文模型，包含 3 个子集（合成→真实）共 4300 个任务；发现即使 GPT-4o 在 Multi-Hard-24K 上 ExAcc 仅 19%，远低于人类 92%，且 prompt 工程和专项微调均无法根本解决该问题。

**[MegaBeam: Scaling Context, Not Parameters](llm_efficiency/scaling_context_not_parameters_training_a_compact_7b_language_model_for_efficien.md)**

:   提出四阶段继续预训练策略将 Mistral-7B 的上下文长度扩展到 512K，7B 模型在 RULER-128K 上超越 GPT-4-1106 和 Llama-3.1-70B，是首个在 512K BABILong 上不用 RAG 就达到 35% 的开源模型。

**[SEAL: Scaling to Emphasize Attention for Long-Context Retrieval](llm_efficiency/seal_scaling_to_emphasize_attention_for_long-context_retrieval.md)**

:   通过剪枝分析发现特定注意力头与长上下文检索高度相关，提出 SEAL（可学习标量/向量缩放注意力头/通道），仅需 ~1024 个额外参数 + 50 个合成样本训练，即可将 LongChat-7B 在 31K 上的检索准确率从 0.32 提升到 0.88，且缩放参数可离线合并到权重中实现零推理开销。

**[Sliding Windows Are Not the End: Exploring Full Ranking with Long-Context Large Language Models](llm_efficiency/sliding_windows_full_ranking.md)**

:   本文系统研究了长上下文LLM在段落排序中的应用，提出用 full ranking（一次性排序所有段落）替代传统滑动窗口策略，并设计了多轮滑动窗口标签构造方法和重要性感知损失函数来微调 full ranking 模型，在效率提升约30-65%的同时实现了排序效果的全面超越。

**[SpindleKV: A Novel KV Cache Reduction Method Balancing Both Shallow and Deep Layers](llm_efficiency/spindlekv_layered_kv_cache.md)**

:   SpindleKV 提出分层处理 KV cache 压缩的策略——深层使用注意力驱动的 token eviction（利用稀疏注意力），浅层使用基于相似性学习的 codebook 替换（利用 token 间高相似度），并解决了 GQA 兼容性问题，实现 50% KV cache 缩减而不损失性能。

**[How to Train Long-Context Language Models (Effectively)](llm_efficiency/train_long_context_effectively.md)**

:   本文系统研究了如何通过持续预训练和监督微调（SFT）有效训练长上下文语言模型，提出了包括数据配比、训练长度缩放等一系列关键发现，最终训练出的 ProLong-8B 模型仅用 Llama-3.1 5% 的长上下文训练数据量即在 128K 长度上达到同规模最优性能。

**[What are the Essential Factors in Crafting Effective Long Context Multi-Hop Instruction Datasets? Insights and Best Practices](llm_efficiency/what_are_the_essential_factors_in_crafting_effective_long_context_multi-hop_inst.md)**

:   提出多智能体交互式多跳生成（MIMG）框架，通过质量验证、单跳问题生成、多问题采样和多跳合并四个模块，系统性地合成高质量长上下文多跳指令数据，训练后模型平均提升7.54%，甚至超越更大规模人工标注数据集。

---

## ✍️ 文本生成 { #nlp_generation }

**[A Representation Level Analysis of NMT Model Robustness to Grammatical Errors](nlp_generation/a_representation_level_analysis_of_nmt_model_robustness_to_grammatical_errors.md)**

:   从模型内部表示视角分析 NMT 对语法错误的鲁棒性机制——发现编码器先"检测"语法错误（GED 探测精度在前半层上升），再"纠正"它（不正确词的表示向正确形式靠拢），并识别出"鲁棒性头"（Robustness Heads）——特定注意力头关注可解释的语言单元以修正错误表示，微调后模型更多依赖这些头。

**[An Empirical Study of Many-to-Many Summarization with Large Language Models](nlp_generation/an_empirical_study_of_manytomany_summarization.md)**

:   首次系统研究LLM在多对多摘要（M2MS）任务上的表现，整合8个数据集构建涵盖5个领域6种语言的47.8K样本基准，评测18个LLM发现零样本LLM可媲美微调传统模型，指令微调后显著超越，但事实性问题仍是关键瓶颈。

**[ATGen: A Framework for Active Text Generation](nlp_generation/atgen_a_framework_for_active_text_generation.md)**

:   推出 ATGen——首个将主动学习（AL）与文本生成（NLG）任务桥接的综合框架，支持人类标注和 LLM 自动标注，集成 SOTA AL 策略和实验设计方法，提供 Web 标注界面和统一基准平台。实验证明 AL 显著减少人工标注时间和 LLM API 调用成本。

**[Beyond N-Grams: Rethinking Evaluation Metrics and Strategies for Multilingual Abstractive Summarization](nlp_generation/beyond_n-grams_rethinking_evaluation_metrics_and_strategies_for_multilingual_abs.md)**

:   系统评估了 n-gram 和神经指标在 8 种语言（4 种类型学家族）上与人工判断的相关性，发现 n-gram 指标在融合语言中可靠性差，而专门训练的神经指标 COMET 在所有语言上一致优于其他指标；还发现分词策略可以显著改善融合语言的评估效果。

**[Improving the Calibration of Confidence Scores in Text Generation Using the Output Distribution's Characteristics](nlp_generation/calibration_confidence_text_gen.md)**

:   针对文本生成中多个有效输出导致传统置信度指标失效的问题，提出两种任务无关的置信度度量——"比率"（头部vs中部概率比）和"尾部稀薄度"（分布尾部薄厚），仅依赖模型输出概率即可改善 BART/Flan-T5 在摘要、翻译、问答任务上的置信度校准。

**[CLEME2.0: Towards Interpretable Evaluation by Disentangling Edits for Grammatical Error Correction](nlp_generation/cleme2_gec_evaluation.md)**

:   本文提出 CLEME2.0，一种可解释的 GEC 参考评估指标，通过将编辑解耦为四类（正确纠正 TP、错误纠正 FPne、欠纠正 FN、过纠正 FPun）并结合编辑加权技术，在 GJG15 和 SEEDA 两个人工评判数据集上达到了与人工判断最高相关性的 SOTA 结果。

**[CoCoLex: Confidence-guided Copy-based Decoding for Grounded Legal Text Generation](nlp_generation/cocolex_legal_text_gen.md)**

:   提出 CoCoLex，一种无需训练的解码策略，通过置信度引导的动态插值将模型词表分布与上下文复制分布结合，鼓励从检索上下文中直接复制 token，在五个法律文本生成基准上显著提升生成文本对源文档的忠实性。

**[Dehumanizing Machines: Mitigating Anthropomorphic Behaviors in Text Generation Systems](nlp_generation/dehumanizing_machines_anthropomorphic.md)**

:   系统研究如何缓解文本生成系统的拟人化行为——编制基于文献和众包的干预手段清单，发展概念框架来表征干预空间、区分干预类型和评估干预效果，为减少用户对 AI 的过度依赖和情感依附提供理论和实证基础。

**[Document-Level Text Generation with Minimum Bayes Risk Decoding using Optimal Transport](nlp_generation/doc_level_mbr_optimal_transport.md)**

:   提出 MBR-OT，将最优传输（Wasserstein距离）引入最小贝叶斯风险（MBR）解码，实现用句子级效用函数评估文档级输出质量，在文档级机器翻译、文本简化和密集图像描述任务上显著优于标准 MBR 解码。

**[Multi-document Summarization through Event Relation Graph Reasoning for Framing Bias Mitigation](nlp_generation/event_graph_bias_mitigation_summarization.md)**

:   提出基于多文档事件关系图的中立化摘要方法，通过构建包含文内事件关系（时间/因果/子事件/共指）、跨文档事件共指和事件级道德观点的知识图，以图文本化（硬提示）和图提示调优（软提示）两种方式引导 LLM 生成去偏见的中立摘要，在内容保持和偏见消除上均优于基线。

**[FEA-Bench: A Benchmark for Evaluating Repository-Level Code Generation for Feature Implementation](nlp_generation/feabench_repo_code_gen.md)**

:   提出 FEA-Bench——首个评估 LLM 在仓库级代码库中实现新特性（Feature Implementation）能力的基准，包含来自 83 个 GitHub 仓库的 1401 个任务实例，每个实例配有单元测试。最强模型 DeepSeek-R1 仅解决约 10% 的任务，揭示了仓库级增量开发对当前 LLM 的巨大挑战。

**[GiFT: Gibbs Fine-Tuning for Code Generation](nlp_generation/gift_gibbs_fine_tuning_code_gen.md)**

:   提出 Gibbs Fine-Tuning（GiFT），受 Gibbs 采样启发，通过"代码→描述→代码"的迭代翻译从边际分布而非条件分布中采样自生成代码，结合困惑度引导的长尾数据选择，在 APPS+/MBPP+/CodeInsight 上比标准自训练提升最高 9.8%。

**[LEMONADE: A Large Multilingual Expert-Annotated Abstractive Event Dataset for the Real World](nlp_generation/lemonade_a_large_multilingual_expert-annotated_abstractive_event_dataset_for_the.md)**

:   发布 Lemonade——基于 ACLED 冲突数据的大规模多语言专家标注事件数据集（39,786 事件，20 种语言，171 个国家，10,707 实体），提出 Abstractive Event Extraction (AEE) 新任务范式，事件参数不限于文本 span 而是归一化为数值/类别/实体，配套 Zest 零样本实体链接系统在 AEL 子任务上 F1=45.7% 大幅超越 baseline 的 23.7%。

**[Has Machine Translation Evaluation Achieved Human Parity?](nlp_generation/mt_eval_human_parity.md)**

:   首次将人类基线引入 WMT Metrics Shared Task 的排名，发现最先进的自动指标经常与人类评估者排名持平甚至更高，但论证了现在声称"人类对等"为时尚早，并讨论了衡量 MT 评估进步的根本困难。

**[Odysseus Navigates the Sirens' Song: Dynamic Focus Decoding for Factual and Diverse Open-Ended Text Generation](nlp_generation/odysseus_dynamic_focus_decoding.md)**

:   提出动态聚焦解码（DFD），通过追踪 LLM 各层间分布差异（KL 散度）来识别知识密集型解码步骤，自适应调整温度——知识密集步用低温保事实性，非知识密集步用高温促多样性——在七个数据集上同时提升事实性和多样性。

**[Personality-Guided Code Generation Using Large Language Models](nlp_generation/personality_guided_code_gen.md)**

:   让 GPT-4o 为每个编程任务生成适配的 MBTI 人格类型和描述，再让 LLM 以该人格角色扮演程序员生成代码，在 28 个 LLM-数据集组合中 23 个取得 pass rate 提升，最高达 12.9%，且可与 CoT 等策略叠加使用。

**[Towards Better Open-Ended Text Generation: A Multicriteria Evaluation Framework](nlp_generation/towards_better_open-ended_text_generation_a_multicriteria_evaluation_framework.md)**

:   针对开放式文本生成中多指标（coherence/diversity/perplexity）之间的权衡问题，提出三种互补的多准则评估方法——Extended Bradley-Terry 模型（序数排名）、Union-Free Generic Depth（允许不可比性的偏序）和 Q*Text（基数评估综合指标），在6个 LLM × 59种解码策略 × 180万+生成文本上验证，发现中等超参配置普遍优于极端配置，小模型+合理解码策略可匹敌大模型。

**[Tree-of-Evolution: Tree-Structured Instruction Evolution for Code Generation in Large Language Models](nlp_generation/tree_of_evolution_code_gen.md)**

:   提出Tree-of-Evolution (ToE)——一种树结构的代码指令合成框架，通过多路径进化和质量驱动优化来克服现有方法（如Code Evol-Instruct/OSS-Instruct）的单向合成和随机生成限制，仅用75K合成数据微调base model即可达到或超越Qwen2.5-Coder-Instruct（数百万样本微调）的性能。

**[What Is That Talk About? A Video-to-Text Summarization Dataset for Scientific Presentations](nlp_generation/video_text_summarization.md)**

:   构建了 VISTA 数据集（18,599 个 AI 会议演讲视频-摘要对），首次系统性基准测试科学视频到文本摘要任务，并提出基于计划（plan-based）的框架，通过显式建模摘要结构来提升生成质量和事实一致性。

**[Writing Like the Best: Exemplar-Based Expository Text Generation](nlp_generation/writing_like_best_exemplar.md)**

:   定义"基于范例的说明文生成"新任务——给定一篇关于源主题的范例文本，生成关于目标主题的说明文，提出 Recurrent Plan-then-Adapt（RePA）框架，通过逐段模仿规划+检索增强自适应生成+双记忆机制，在 Wikipedia/RoleEE/USNews 三个数据集上显著优于 GPT-4 和 o1 基线。

---

## 📖 NLP 理解 { #nlp_understanding }

**[A Variational Approach for Mitigating Entity Bias in Relation Extraction](nlp_understanding/a_variational_approach_for_mitigating_entity_bias_in_relation_extraction.md)**

:   将变分信息瓶颈（VIB）应用于关系抽取的实体去偏——将实体映射到概率分布 $\mathcal{N}(\mu,\sigma)$，通过方差控制实体信息的压缩程度（高方差=更多依赖上下文），在 TACRED/REFinD/BioRED 三个领域（通用/金融/生物医学）的域内和域外设置上达到 SOTA，同时方差分析提供可解释性。

**[Adapting Psycholinguistic Research for LLMs: Gender-Inclusive Language in a Coreference Context](nlp_understanding/adapting_psycholinguistic_research_for_llms_gender-inclusive_language_in_a_coref.md)**

:   将心理语言学方法从法语适配到英语和德语，研究 LLM 如何处理性别包容性语言——发现英语 LLM 基本保持先行词性别一致但内含男性默认偏见（不愿用 they 单数），德语 LLM 男性偏见更强烈（压倒所有性别中性化策略），但德语性别包容形式确实增加了女性/中性性别的出现概率。

**[Analyzing Political Bias in LLMs via Target-Oriented Sentiment Classification](nlp_understanding/analyzing_political_bias_in_llms_via_target-oriented_sentiment_classification.md)**

:   提出基于目标导向情感分类（TSC）不一致性的 LLM 政治偏见分析新方法——在 450 个政治句子中插入 1319 名不同政治光谱/人口特征的政治家名字，用 7 个模型×6 种语言预测情感，定义熵基不一致性指标量化预测变异性，发现所有模型均存在显著偏见（左翼正面/极右翼负面），大模型偏见更强且更一致，用虚构名字替换可部分缓解。

**[ArgHiTZ at ArchEHR-QA 2025: A Two-Step Divide and Conquer Approach to Patient Question Answering for Top Factuality](nlp_understanding/arghitz_at_archehr-qa_2025_a_two-step_divide_and_conquer_approach_to_patient_que.md)**

:   在 ArchEHR-QA 2025 共享任务中提出两阶段"分治"方法：先用重排序模型从电子健康记录中提取关键句子，再用小型医学 LLM 生成回复，在不使用外部知识的情况下取得事实性排名第一、总分第 8/30 的成绩。

**[AskQE: Question Answering as Automatic Evaluation for Machine Translation](nlp_understanding/askqe_question_answering_as_automatic_evaluation_for_machine_translation.md)**

:   提出 AskQE——基于问答的机器翻译质量估计框架，通过对源文本生成问题、分别在源文本和回译输出上回答、对比答案差异来检测翻译错误，帮助不懂目标语言的用户判断翻译是否可接受，在 BioMQM 数据集上 Kendall's τ 相关和决策准确率均优于现有 QE 指标。

**[Automatic Generation of Inference Making Questions for Reading Comprehension Assessments](nlp_understanding/automatic_generation_of_inference_making_questions_for_reading_comprehension_ass.md)**

:   开发了一套阅读理解推理题分类法（代词桥接/文本连接/填补空白），用 GPT-4o few-shot 提示自动生成针对特定推理类型的多项选择题；93.8% 的题目质量合格，但仅 42.6% 准确匹配目标推理类型，说明 LLM 在精确推理能力控制上仍有不足。

**[BelarusianGLUE: Towards a Natural Language Understanding Benchmark for Belarusian](nlp_understanding/belarusian_glue.md)**

:   为白俄罗斯语（Belarusian，东斯拉夫语族）构建了首个NLU benchmark——BelarusianGLUE，包含5个任务约15K条实例，系统评估了BERT系列和LLM的表现，发现简单任务（情感分析）接近人类水平但难任务（Winograd）仍有显著差距，且最优模型类型因任务而异。

**[BESSTIE: A Benchmark for Sentiment and Sarcasm Classification for Varieties of English](nlp_understanding/besstie_a_benchmark_for_sentiment_and_sarcasm_classification_for_varieties_of_en.md)**

:   构建 BESSTIE，首个针对英语变体（澳大利亚/印度/英国英语）的情感分析和讽刺检测标注基准，通过 9 个微调 LLM 评估发现模型在印度英语（外圈变体）上表现显著差于内圈变体，跨变体泛化能力也有限。

**[BookCoref: Coreference Resolution at Book Scale](nlp_understanding/bookcoref_book_scale.md)**

:   提出首个书级别共指消解基准BookCoref，通过角色链接+LLM过滤+窗口扩展的自动标注管线，在50本完整小说上生成高质量银标注数据，平均文档长度超过20万tokens。

**[CaLMQA: Exploring Culturally Specific Long-Form Question Answering across 23 Languages](nlp_understanding/calmqa_cultural_multilingual_qa.md)**

:   构建了首个多语言长文本问答数据集 CaLMQA（51.7K 问题，23 种语言），通过无翻译方式收集文化特异性问题，发现 LLM 回答文化特异性问题的事实性（45-52%）显著低于文化无关问题（64-71%），低资源语言表现尤其差。

**[Dynamic Order Template Prediction for Generative Aspect-Based Sentiment Analysis](nlp_understanding/dot_absa_template.md)**

:   提出 Dynamic Order Template（DOT）方法用于生成式方面级情感分析——为每个实例动态创建最优的预测模板顺序（只含必要的视角），在 ASQP 和 ACOS 数据集上提升 F1 的同时显著减少推理时间。

**[Beyond Prompting: An Efficient Embedding Framework for Open-Domain Question Answering](nlp_understanding/embqa_embedding_odqa.md)**

:   EmbQA 提出嵌入级 ODQA 框架，用轻量线性层和无监督对比学习优化查询表示实现段落重排序，并引入基于序统计量的探索性嵌入扩展候选答案多样性，配合熵选择机制自动选答，在 4 个 ODQA 数据集上以更低计算成本超越 SuRe 等 prompt 级方法。

**[Towards a More Generalized Approach in Open Relation Extraction](nlp_understanding/generalized_open_relation_extract.md)**

:   提出 MixORE 框架，在更通用的 Open Relation Extraction 设定下（无标注数据同时包含已知和新颖关系，且不做长尾或预分割假设），通过 Semantic Autoencoder 检测新关系 + 开放世界半监督联合学习，在 FewRel/TACRED/Re-TACRED 上全面超越 SOTA。

**[LACA: Improving Cross-lingual Aspect-Based Sentiment Analysis with LLM Data Augmentation](nlp_understanding/laca_crosslingual_absa.md)**

:   提出 LACA 框架，利用 LLM 为目标语言生成高质量伪标注数据（而非依赖机器翻译），在六种语言上显著提升跨语言 ABSA 性能，在 mBERT 和 XLM-R 上分别平均超过前 SOTA 1.50% 和 2.62%。

**[NeuSym-RAG: Hybrid Neural Symbolic Retrieval with Multiview Structuring for PDF Question Answering](nlp_understanding/neusym_rag_pdf_qa.md)**

:   NeuSym-RAG 提出了一个混合神经-符号检索框架，将 PDF 文档通过多视角分块解析同时存入关系数据库和向量库，LLM Agent 通过可执行动作（SQL 查询 + 向量检索 + 查看图片等）迭代式交互检索，在学术论文 QA 上比经典 RAG 提升 17.3%。

**[Exploring Persona Sentiment Sensitivity in Personalized Dialogue Generation](nlp_understanding/persona_sentiment_dialogue.md)**

:   大规模分析 LLM 对人设情感极性的敏感性，发现负面人设导致过度强调人设属性和对话矛盾、弱/中性人设产生低质量对话，提出结合逐轮生成、人设排序和情感感知提示的对话生成框架来缓解这些问题。

**[ReSCORE: Label-free Iterative Retriever Training for Multi-hop Question Answering with Relevance-Consistency Supervision](nlp_understanding/rescore_multihop_qa.md)**

:   提出 ReSCORE，利用 LLM 生成的文档-问题相关性（relevance）和文档-答案一致性（consistency）的联合概率作为伪标签，在迭代 RAG 框架中无监督训练 dense retriever，在三个多跳 QA 数据集上达到 SOTA。

**[YESciEval: Robust LLM-as-a-Judge for Scientific Question Answering](nlp_understanding/yescieval_llm_judge_science.md)**

:   提出YESciEval框架，结合九维细粒度评估准则和SFT+RL对齐策略来缓解LLM评估者的乐观偏差(optimism bias)，在科学问答场景下构建鲁棒的开源LLM-as-a-Judge系统，无需人类标注和闭源模型。

---

## 🛡️ AI 安全 { #ai_safety }

**[CENTAUR: Bridging the Impossible Trinity of Privacy, Efficiency, and Performance in Privacy-Preserving Transformer Inference](ai_safety/centaur_bridging_the_impossible_trinity_of.md)**

:   提出 Centaur 框架，融合随机置换矩阵和安全多方计算（SMPC）来打破隐私保护 Transformer 推理（PPTI）中的"不可能三角"——同时实现强隐私保护、5-30x 加速和明文级别推理精度。

**[Assessing Dialect Fairness and Robustness of Large Language Models in Reasoning Tasks](ai_safety/dialect_fairness_robustness.md)**

:   本文提出首个系统评估LLM在非标准方言（AAVE）推理任务中公平性与鲁棒性的研究，构建了包含1.2K+平行查询对的ReDial基准，发现几乎所有主流LLM在AAVE输入上表现出显著的性能下降和不公平。

**[Ensemble Watermarks for Large Language Models](ai_safety/ensemble_watermarks_llm.md)**

:   提出集成水印方法，将文体特征（藏头词 acrostic + 感觉运动词 sensorimotor norms）与已有红绿水印组合，在 paraphrasing 攻击后三特征集成检测率达 95%，而单独红绿水印仅 49%。

**[FairI Tales: Evaluation of Fairness in Indian Contexts with a Focus on Bias and Stereotypes](ai_safety/fairi_tales_evaluation_of_fairness_in_indian_contexts_with_a_focus_on_bias_and_s.md)**

:   发布 Indic-Bias——首个印度社会文化的大规模 LLM 公平性 benchmark，覆盖 85 个身份群体（种姓/宗教/地区/部落），20,000 个场景模板+3 类评估任务，14 个 LLM 评测揭示模型对达利特等边缘群体系统性负面偏见且 70%+ 案例中强化刻板印象。

**[Fairness through Difference Awareness: Measuring Desired Group Discrimination in LLMs](ai_safety/fairness_difference_awareness.md)**

:   本文挑战了主流公平性研究中"对所有群体一视同仁即为公平"的假设，提出"差异意识"(Difference Awareness)概念，构建了包含8个基准共16k问题的评测套件，发现现有"最公平"的LLM在该维度上表现不佳，且现有去偏方法会适得其反。

**[From Trade-off to Synergy: A Versatile Symbiotic Watermarking Framework for Large Language Models](ai_safety/from_tradeoff_to_synergy_a_versatile.md)**

:   提出SymMark共生水印框架，融合logits-based和sampling-based两类水印方法（串行/并行/混合三种策略），通过token熵和语义熵自适应选择水印策略，在可检测性、鲁棒性、文本质量和安全性上实现SOTA。

**[Gender Inclusivity Fairness Index (GIFI): A Multilevel Framework](ai_safety/gifi_gender_fairness.md)**

:   提出 GIFI（Gender Inclusivity Fairness Index），一个多层次综合评估框架，涵盖代词识别、情感中立性、毒性、反事实公平性、刻板印象关联、职业公平性和推理性能一致性七个维度，在 22 个 LLM 上系统评估二元与非二元性别的公平性。

**[Improved Unbiased Watermark for Large Language Models](ai_safety/improved_unbiased_watermark_for_large_language.md)**

:   提出 MCmark，一族基于多通道（Multi-Channel）的无偏水印算法，通过将词表分割为 $l$ 个段并在选中段内提升 token 概率来嵌入统计信号，在保持 LLM 原始输出分布的同时，可检测性比现有无偏水印提升超 10%。

**[Can LLM Watermarks Robustly Prevent Unauthorized Knowledge Distillation?](ai_safety/llm_watermark_distillation_robustness.md)**

:   本文首次系统研究 LLM 水印在防止未授权知识蒸馏中的鲁棒性，提出三种水印去除攻击（无目标/有目标释义 + 推理时水印中和），发现有目标释义和水印中和可以彻底去除继承的水印，其中水印中和在保持知识迁移效率的同时实现零额外训练开销的水印去除。

**[MorphMark: Flexible Adaptive Watermarking for Large Language Models](ai_safety/morphmark_adaptive_watermarking.md)**

:   MorphMark 通过多目标权衡分析框架揭示了绿表概率 P_G 在水印效果与文本质量之间的关键作用，并据此提出自适应调整水印强度 r 的方法——当 P_G 高时增强水印、P_G 低时减弱水印，实现了在不依赖额外模型训练的前提下同时提升水印可检测性和文本质量。

**[PrivaCI-Bench: Evaluating Privacy with Contextual Integrity and Legal Compliance](ai_safety/privacibench_evaluating_privacy_with_contextual_integrity.md)**

:   提出 PrivaCI-Bench，基于 Contextual Integrity 理论构建了目前最大的上下文隐私评估基准（154K 实例），涵盖真实法院案例、隐私政策和 EU AI Act 合规检查器合成数据，评估 LLM 在 HIPAA/GDPR/AI Act 下的法律合规能力。

**[Sandcastles in the Storm: Revisiting Watermarking Impossibility](ai_safety/sandcastles_watermarking_impossibility.md)**

:   本文通过大规模实验和人类评估挑战了 "Watermarks in the Sand" (WITS) 的理论不可能性结论：证明随机游走攻击的两个关键假设在实践中不成立——混合(mixing)速度极慢（100% 的攻击文本仍可追溯原始来源）且质量预言机(quality oracle)不可靠（仅 77% 准确率），自动攻击仅 26% 成功率，人类质量审核后降至 10%。

**[SpeechFake: A Large-Scale Multilingual Speech Deepfake Dataset Incorporating Cutting-Edge Generation Methods](ai_safety/speechfake_a_largescale_multilingual_speech_deepfake.md)**

:   构建 SpeechFake，目前最大的语音深度伪造检测数据集——超 300 万样本、3000+ 小时、40 种生成工具（含最新 TTS/VC/Neural Vocoder）、46 种语言，在自有及未见测试集上展现强基线性能。

**[TIP of the Iceberg: Task-in-Prompt Adversarial Attacks on LLMs](ai_safety/tip_iceberg_adversarial_attacks.md)**

:   本文提出 Task-in-Prompt (TIP) 攻击——一类通过在 prompt 中嵌入序列到序列任务（如密码解码、谜语、代码执行）来间接生成违禁内容的新型越狱攻击类别，并构建 PHRYGE benchmark 系统评估，证明该攻击可成功绕过 GPT-4o、LLaMA 3.2 等六种 SOTA LLM 的安全防护。

**[The Tug of War Within: Mitigating the Fairness-Privacy Conflicts in Large Language Models](ai_safety/tug_of_war_fairness_privacy.md)**

:   发现 LLM 通过 SFT 增强隐私意识会显著降低公平性意识（trade-off），提出无训练方法 SPIN（抑制公平-隐私耦合神经元），基于信息论解耦两种意识，在 Qwen2-7B 上同时提升公平性 12.2% 和隐私意识 14.0%。

**[Efficiently Identifying Watermarked Segments in Mixed-Source Texts](ai_safety/watermark_segment_detection.md)**

:   提出两种高效方法（Geometric Cover Detector 和 Adaptive Online Locator）用于在长文混合来源文本中检测和精确定位水印片段，时间复杂度从 O(n²) 降至 O(n log n)，在三种主流水印技术上均显著优于 baseline。

**[WET: Overcoming Paraphrasing Vulnerabilities in Embeddings-as-a-Service with Linear Transformation Watermark](ai_safety/wet_eaas_watermark.md)**

:   揭示了现有 EaaS 嵌入水印（EmbMarker/WARDEN）可被改写攻击绕过，提出 WET（线性变换水印），通过秘密循环矩阵对嵌入做线性变换注入水印，理论和实验证明其对改写攻击具有鲁棒性，验证 AUC 接近 100%。

---

## 🏥 医学图像 { #medical_imaging }

**[A Retrieval-Based Approach to Medical Procedure Matching in Romanian](medical_imaging/a_retrieval-based_approach_to_medical_procedure_matching_in_romanian.md)**

:   提出基于检索的罗马尼亚语医疗程序名称匹配架构——将诊所的非标准程序描述匹配到保险公司标准化术语表，比较 BM25 稀疏检索、mE5/RoBERT/BioClinicalBERT 密集嵌入及 RRF 混合方法，在 14 万+映射对上评估，度量学习微调后 mE5 表现最佳。

**[AfriMed-QA: A Pan-African, Multi-Specialty, Medical Question-Answering Benchmark Dataset](medical_imaging/afrimed_qa_pan_african.md)**

:   构建了首个大规模泛非洲多专科医学问答基准 AfriMed-QA（15,275 题，来自 16 个国家 60+ 医学院，涵盖 32 个专科），评估 30 个 LLM 发现：大模型在非洲医疗问题上的准确率显著低于 USMLE，生物医学专用 LLM 反而不如通用模型，消费者盲评时更偏好 LLM 回答而非临床医生回答。

**[The Impact of Auxiliary Patient Data on Automated Chest X-Ray Report Generation and How to Incorporate It](medical_imaging/auxiliary_patient_data_xray.md)**

:   本文研究如何将急诊科患者数据（生命体征、药物、分诊信息等）整合到多模态语言模型中用于自动胸部X光报告生成，提出将异构表格数据、文本和图像转化为统一嵌入的方法，在MIMIC-CXR + MIMIC-IV-ED数据集上显著提升了报告的诊断准确性，超越了包括CXRMate-RRG24在内的多个基准模型。

**[Improving Automatic Evaluation of LLMs in Biomedical Relation Extraction via LLMs-as-the-Judge](medical_imaging/biore_llm_judge_evaluation.md)**

:   本文首次系统研究了 LLM-as-Judge 在生物医学关系抽取评估中的表现，发现其准确率通常低于 50%，并提出结构化输出格式（JSON）和域适应技术来提升约 15% 的评估准确率。

**[CheXalign: Preference Fine-tuning in Chest X-ray Interpretation Models without Human Feedback](medical_imaging/chexalign_preference_finetuning.md)**

:   CheXalign 提出了一种无需放射科医生反馈的自动化偏好数据生成管线，利用公开数据集中的参考报告和基于参考的评估指标（如 GREEN、BERTScore）构造偏好对，通过 DPO 等直接对齐算法对胸部X光报告生成模型进行偏好微调，在 MIMIC-CXR 上取得 SOTA CheXbert 分数。

**[Aligning AI Research with the Needs of Clinical Coding Workflows: Eight Recommendations Based on US Data Analysis and Critical Review](medical_imaging/clinical_coding_eight_recommendations.md)**

:   这篇 position paper 通过对 MIMIC 数据集和现有自动化临床编码研究的深入分析，指出当前评估方法（如仅关注前50个高频编码、使用不恰当指标）与真实临床场景严重脱节，并提出八条具体建议来改进评估方法和研究方向。

**[MultiMed: Multilingual Medical Speech Recognition via Attention Encoder Decoder](medical_imaging/multimed_multilingual_medical_speech_recognition_via_attention_encoder_decoder.md)**

:   发布 MultiMed——首个多语言医学 ASR 数据集（150小时，5种语言，10种录制场景，16种口音），配套小到大规模的端到端 Whisper 模型基线，首次系统研究医学领域的多语言 ASR：单语 vs 多语微调、AED vs Hybrid 架构对比，发现多语联合训练在小模型上有收益但大模型上可能退化。

**[Online Iterative Self-Alignment for Radiology Report Generation](medical_imaging/oisa_radiology_report_gen.md)**

:   提出在线迭代自对齐（OISA）方法用于放射学报告生成——四阶段循环（自生成多样数据→自评估多目标偏好→自对齐多目标优化→自迭代进一步提升），无需额外人工标注即可迭代提升报告质量，在多个评估指标上达到 SOTA。

**[Towards Omni-RAG: Comprehensive Retrieval-Augmented Generation for Large Language Models in Medical Applications](medical_imaging/omni_rag_medical.md)**

:   针对医疗 LLM 需要多类型多结构知识源（教科书/指南/论文/知识图谱等）的特殊需求，提出 MedOmniKB 多源知识库和 Source Planning Optimization 方法——让模型学会"该从哪个源检索什么信息"，优化后的小模型在多源医疗知识利用上达到 SOTA。

**[RADAR: Enhancing Radiology Report Generation with Supplementary Knowledge Injection](medical_imaging/radar_radiology_report_gen.md)**

:   提出 Radar 框架，通过"内部知识提取+外部补充知识检索+聚合注入"三步策略增强放射学报告生成——先提取 LLM 已有的与专家分类一致的知识，再检索缺失的补充知识，最终聚合两者生成更准确的放射学报告，在 MIMIC-CXR/CheXpert-Plus/IU X-ray 三个数据集上超越 SOTA。

**[ReflecTool: Towards Reflection-Aware Tool-Augmented Clinical Agents](medical_imaging/reflectool_clinical_agent.md)**

:   ReflecTool 提出了一个反思感知的工具增强临床 Agent 框架，通过优化阶段积累成功轨迹和工具级经验，推理阶段检索相似案例并用验证器改进工具使用，在涵盖 18 个任务的 ClinicalAgent Bench 上超越纯 LLM 10+ 分、超越已有 Agent 方法 3 分。

**[Query-driven Document-level Scientific Evidence Extraction from Biomedical Studies](medical_imaging/urca_biomedical_evidence_extraction.md)**

:   本文提出 URCA（Uniform Retrieval Clustered Augmentation）框架，通过均匀检索+聚类+知识提取的 RAG 流程，从 RCT 研究全文中自动提取与临床问题相关的科学证据结论，在新构建的 CochraneForest 数据集上比最佳基线提升了 8.81% F1。

---

## 🎵 音频/语音 { #audio_speech }

**[Finding A Voice: Exploring the Potential of African American Dialect and Voice Generation for Chatbots](audio_speech/aae_voice_chatbot.md)**

:   研究将非裔美式英语（AAE）整合到聊天机器人中的效果——开发文本和语音 AAE 聊天机器人并用 AAE 说话者评估，发现文本 AAE 聊天机器人常表现不佳（方言生成不够自然），但语音聊天机器人结合非裔声音和 AAE 元素时用户体验更好，揭示了语言个性化的复杂性。

**[AI4Reading: Chinese Audiobook Interpretation System Based on Multi-Agent Collaboration](audio_speech/ai4reading_chinese_audiobook_interpretation_system_based_on_multi-agent_collabor.md)**

:   提出 AI4Reading，一个基于 11 个专业化 LLM Agent 协作的中文有声书解读系统，通过主题分析、案例扩展、编辑润色、口语化改写和整合修订等阶段自动生成解读稿，并用 TTS 合成音频，在解读脚本质量（简洁性、完整性、准确性、连贯性）上超过专业人工解读平台樊登读书。

**[Benchmarking Open-ended Audio Dialogue Understanding for Large Audio-Language Models](audio_speech/audio_dialogue_benchmark.md)**

:   提出 ADU-Bench，一个包含 20,000+ 开放式音频对话的综合基准，覆盖 3 种通用场景、12 项技能、9 种语言和 4 类歧义处理，首次系统评估大型音频语言模型（LALM）的音频对话理解能力，在 16 个模型上的实验揭示了现有 LALM 在数学符号、角色扮演、多语言和语音歧义处理上的显著不足。

**[Autoregressive Speech Synthesis without Vector Quantization](audio_speech/autoregressive_speech_synthesis_without_vq.md)**

:   MELLE 提出了一种基于连续 mel-spectrogram 帧的自回归语言模型 TTS 方法，通过回归损失 + 变分推断采样模块 + spectrogram flux loss 直接预测连续频谱帧，避免了向量量化带来的保真度损失和采样鲁棒性问题，单阶段模型即可达到与人类水平相当的语音合成质量。

**[ControlSpeech: Towards Simultaneous and Independent Zero-shot Speaker Cloning and Zero-shot Language Style Control](audio_speech/controlspeech_zero_shot.md)**

:   ControlSpeech 是首个同时实现零样本音色克隆和零样本语言风格控制的TTS系统，通过离散编解码器空间中的解耦表示和风格混合语义密度（SMSD）模块解决了风格控制中的多对多问题。

**[GigaSpeech 2: An Evolving, Large-Scale and Multi-domain ASR Corpus for Low-Resource Languages](audio_speech/gigaspeech2_low_resource_asr.md)**

:   GigaSpeech 2 构建了一个约 30,000 小时的大规模低资源语言（泰语、印尼语、越南语）ASR 语料库，通过自动化爬取-转录-精炼管线从无标注 YouTube 视频生成高质量伪标签，训练的模型仅用 10% 参数量即可将 WER 比 Whisper large-v3 降低 25%-40%。

**[Sparsify: Learning Sparsity for Effective and Efficient Music Performance Question Answering](audio_speech/sparsify_music_avqa.md)**

:   Sparsify 提出三层稀疏化策略（稀疏掩码+自适应稀疏合并+关键子集选择）用于音乐表演视听问答（Music AVQA），在 MUSIC-AVQA 和 v2.0 两个 benchmark 上达到 SOTA（81.75%/81.30%），训练时间减少 28.32%，25% 数据即保持 74% 的全量性能。

**[SpeechIQ: Speech-Agentic Intelligence Quotient Across Cognitive Levels in Voice Understanding by Large Language Models](audio_speech/speechiq_speechagentic_intelligence_quotient_across_cognitive.md)**

:   提出 SpeechIQ，一个基于 Bloom 认知分类学的层次化语音理解评估框架，从 Remember（WER）、Understand（语义相似度）、Apply（QA 准确率）三个层次综合评估语音 LLM 的智能水平，发现级联 ASR+LLM 系统在同规模下优于端到端多模态模型。

**[T2A-Feedback: Improving Basic Capabilities of Text-to-Audio Generation via Fine-grained AI Feedback](audio_speech/t2a_feedback_audio_gen.md)**

:   提出三个细粒度 AI 音频评分管线（事件出现/事件顺序/声学和谐质量）替代人工标注构建大规模音频偏好数据集 T2A-FeedBack（41K提示+249K音频），用偏好调优增强 TTA 模型的基础能力，在简单（AudioCaps）和复杂（T2A-EpicBench）场景下都显著提升多事件音频生成质量。

**[In-the-wild Audio Spatialization with Flexible Text-guided Localization](audio_speech/tas_audio_spatialization.md)**

:   提出 TAS（Text-guided Audio Spatialization）框架，用灵活的文本提示（3D 空间位置描述或声源间相对位置描述）引导潜在扩散模型将单声道音频转换为双耳音频，构建了 376K 样本的 SpatialTAS 数据集，在模拟和真实录制数据上均超越现有方法，并基于 Llama-3.1-8B 开发了空间语义一致性评估模型。

---

## 🕸️ 图学习 { #graph_learning }

**[Beyond Completion: A Foundation Model for General Knowledge Graph Reasoning](graph_learning/beyond_completion_a_foundation_model_for_general_knowledge_graph_reasoning.md)**

:   提出 MERRY，一个统一处理 KG 内（零样本 KGC）和 KG 外（KGQA）推理任务的知识图谱基础模型，通过多视角条件消息传递 (CMP) 融合文本和结构信息，在 28 个数据集上超越现有方法。

**[Morpher: Can Graph Neural Networks Learn Language with Extremely Weak Text Supervision?](graph_learning/can_graph_neural_networks_learn_language.md)**

:   提出 Morpher，首个图-文多模态 prompt learning 范式——在冻结 GNN 和 LLM 参数的前提下，同时学习图 prompt 和文本 prompt + 跨模态投影器，用极弱文本监督（仅类别名几个词）将 GNN 表征对齐到 LLM 语义空间，首次实现 GNN 的 CLIP 式零样本图分类。

**[Croppable Knowledge Graph Embedding](graph_learning/croppable_knowledge_graph_embedding.md)**

:   提出 MED 框架训练"可裁剪"知识图谱嵌入——一次训练同时优化 64 个不同维度的子模型（共享嵌入前缀），通过互学习、进化改进和动态损失权重，各维度子模型直接裁剪使用即超越独立训练和蒸馏方法，训练速度快 10 倍。

**[Extending Complex Logical Queries on Uncertain Knowledge Graphs](graph_learning/extending_complex_logical_queries_uncertain_knowledge_graphs.md)**

:   提出在不确定知识图谱（Uncertain KG）上进行软查询（Soft Query）的新问题设定 SQUK，结合必要性（necessity）和重要性（importance）扩展一阶逻辑查询语义，并设计带校准的神经符号推理方法 SRC，避免前向推理中的级联错误。

**[GraphNarrator: Generating Textual Explanations for Graph Neural Networks](graph_learning/graphnarrator.md)**

:   提出GraphNarrator——首个为图神经网络生成自然语言解释的方法，通过将显著性图解释"语言化"为文本段落、用Expert Iteration迭代优化伪标签质量、最终蒸馏到端到端解释器模型，在三个数据集上生成的解释在忠实度、简洁性和人类偏好上均优于GPT-4o零样本解释。

**[Can Knowledge Graphs Make Large Language Models More Trustworthy? An Empirical Study Over Open-ended Question Answering](graph_learning/kg_llm_trustworthy_qa.md)**

:   提出开放域知识图谱问答基准 OKGQA 及其扰动变体 OKGQA-P，通过统一的图引导检索-生成框架系统性地验证了 KG 增强可以有效降低 LLM 幻觉率（FActScore 提升约 20 个百分点），子图检索在各类查询上表现最优且对 KG 噪声具有鲁棒性。

**[Knowledge Graph Retrieval-Augmented Generation for LLM-based Recommendation](graph_learning/kg_rag_recommendation.md)**

:   提出 K-RagRec 框架，将知识图谱（KG）中的结构化关系信息引入 LLM 推荐系统的 RAG 流程——从 KG 中检索高质量的结构化实体关系信息来增强推荐生成，解决纯文本 RAG 忽略结构关系和引入噪声的问题。

**[Can LLMs Evaluate Complex Attribution in QA? Automatic Benchmarking using Knowledge Graphs](graph_learning/paper_2401_14640.md)**

:   提出 CAQA 基准，利用知识图谱自动生成包含四类归因类别（支持、部分支持、矛盾、无关）和四种推理复杂度的大规模问答归因评估数据集（161K 样本），系统性地评测了 25 种自动归因评估器的能力。

**[Predicate-Conditional Conformalized Answer Sets for Knowledge Graph Embeddings](graph_learning/predicate-conditional_conformalized_answer_sets_for_knowledge_graph_embeddings.md)**

:   提出 CondKGCP——基于谓词条件的 conformal prediction 方法用于知识图谱嵌入的不确定性量化，通过合并相似谓词增大校准集+双重校准（score+rank）减小预测集大小，在保证谓词级条件覆盖率的同时输出更紧凑的答案集，在多个KGE基准上优于5个baseline。

**[RSCF: Relation-Semantics Consistent Filter for Entity Embedding of Knowledge Graph](graph_learning/rscf_relationsemantics_consistent_filter_for_entity.md)**

:   提出 RSCF 插件式 KGE 方法，通过共享仿射变换 + 根植实体变换 + 归一化三特征确保"语义相似的关系产生相似的实体变换"（关系语义一致性），在距离模型和张量分解模型上均显著超越 SOTA，并从理论和实验上验证了一致性保持率。

---

## 🔎 AIGC 检测 { #aigc_detection }

**[A Rose by Any Other Name: LLM-Generated Explanations Are Good Proxies for Human Explanations to Collect Label Distributions on NLI](aigc_detection/a_rose_by_any_other_name_llm-generated_explanations_are_good_proxies_for_human_e.md)**

:   研究 LLM 生成的解释能否替代昂贵的人工解释来近似 NLI 的人工判断分布（HJD）——发现在提供人工标签的条件下，LLM 生成的解释与人工解释在近似 HJD 方面效果相当（"名字不重要，玫瑰依然芬芳"），且方法可推广到无人工解释的数据集和域外测试集。

**[Are We in the AI-Generated Text World Already? Quantifying and Monitoring AIGT on Social Media](aigc_detection/aigt_social_media_monitoring.md)**

:   首次大规模量化社交媒体上 AI 生成文本(AIGT)的占比变化——收集 Medium/Quora/Reddit 上 240 万帖子，构建 AIGTBench 训练最佳检测器 OSM-Det，发现 2022-2024 年间 Medium 和 Quora 的 AIGT 占比从~2% 飙升至~37-39%，而 Reddit 仅从 1.3% 增至 2.5%。

**[People who frequently use ChatGPT for writing tasks are accurate and robust detectors of AI-generated text](aigc_detection/chatgpt_user_ai_text_detection.md)**

:   通过 1,740 条标注实验发现，经常使用 LLM 进行写作任务的人类标注者可以极高精度（5人投票仅错 1/300）检测 AI 生成文本，即使面对改写和人性化逃逸策略也显著优于大多数自动检测器。

**[Iron Sharpens Iron: Defending Against Attacks in Machine-Generated Text Detection with Adversarial Training](aigc_detection/greater_adversarial_mgt_detection.md)**

:   提出 GREATER 对抗训练框架，同步训练对抗攻击器（Greater-A）和 MGT 检测器（Greater-D），对抗器通过代理模型梯度识别关键 token 并在嵌入空间扰动生成对抗样本，检测器从课程式对抗样本中学习泛化防御，在 16 种攻击下 ASR 降至 5.53%（SOTA 为 6.20%），攻击效率比 SOTA 快 4 倍。

**[Comparing LLM-generated and human-authored news text using formal syntactic theory](aigc_detection/llm_vs_human_formal_syntax.md)**

:   首次使用形式句法理论（HPSG）系统比较六个 LLM 生成的纽约时报风格文本与真实人类撰写的 NYT 文本，发现 LLM 和人类写作在 HPSG 语法类型分布上存在系统性差异，揭示了 LLM 句法行为与人类的本质不同。

**[Low-Perplexity LLM-Generated Sequences and Where To Find Them](aigc_detection/low-perplexity_llm-generated_sequences_and_where_to_find_them.md)**

:   提出一个系统化的 pipeline，通过分析 LLM 生成的低困惑度序列（每个 token 预测概率 ≥0.9），将其追溯到训练数据来源，发现 30-60% 的低困惑度片段无法匹配训练数据，并将可匹配的片段分为四种行为类别。

**[MultiSocial: Multilingual Benchmark of Machine-Generated Text Detection of Social-Media Texts](aigc_detection/multisocial_mgt_detection.md)**

:   构建首个多语言(22种语言)、多平台(5个社交媒体)、多生成器(7个LLM)的社交媒体机器生成文本检测基准 MultiSocial（47万文本），填补了社交媒体短文本+非英语场景下 MGT 检测研究的空白，发现微调检测器可在社交媒体文本上有效训练且训练平台选择很重要。

**[Who Writes What: Unveiling the Impact of Author Roles on AI-generated Text Detection](aigc_detection/who_writes_what_ai_detection.md)**

:   揭示作者的社会语言学属性（性别、CEFR水平、学科领域、语言环境）会系统性地影响AI生成文本检测器的准确率，其中语言水平和语言环境的偏差最为显著且一致，提出了基于多因素WLS+ANOVA的偏差量化框架。

---

## 🔗 因果推理 { #causal_inference }

**[CausalRAG: Integrating Causal Graphs into Retrieval-Augmented Generation](causal_inference/causalrag_integrating_causal_graphs_into_retrieval-augmented_generation.md)**

:   提出 CausalRAG，将因果图集成到 RAG 的检索过程中——从文档构建文本图并识别因果关系，在查询时通过因果路径发现和因果摘要生成来检索上下文，在文档问答中显著提升上下文精度（92.86%）和检索召回率。

**[IRIS: An Iterative and Integrated Framework for Verifiable Causal Discovery](causal_inference/iris_an_iterative_and_integrated_framework.md)**

:   提出 IRIS 框架——仅需一组初始变量名作为输入，即可自动检索文档、提取变量值构建结构化数据、通过混合因果发现（GES 统计算法 + LLM 因果关系验证）构建因果图，并通过缺失变量提议组件迭代扩展变量集合，放松了传统方法的无环和因果充分性假设，在 Cancer、Diabetes、Obesity、ADNI、Insurance 等 6 个数据集上 F1 全面超越 0-shot/CoT/RAG 基线。

**[On the Reliability of Large Language Models for Causal Discovery](causal_inference/llm_causal_discovery_reliability.md)**

:   利用开源 LLM（OLMo、BLOOM）可访问的预训练语料库，实证验证了"因果鹦鹉"假说——LLM 识别因果关系的能力与预训练数据中该关系的出现频率高度相关（Spearman r=0.9），且错误因果关系的存在和上下文变化都会显著影响预测可靠性。

**[Reasoning is All You Need for Video Generalization: A Counterfactual Benchmark with Sub-question Evaluation](causal_inference/reasoning_is_all_you_need_for_video_generalization_a_counterfactual_benchmark_wi.md)**

:   提出 COVER（COunterfactual VidEo Reasoning），一个多维度视频反事实推理 benchmark，将评估任务按抽象-具体和感知-认知两个维度分为四象限共 13 类任务，并通过将复杂问题分解为子问题（必要条件）来揭示——子问题准确率与反事实推理能力强相关，提升推理能力是改善视频理解鲁棒性的关键。

---

## 🎨 图像生成 { #image_generation }

**[D-GEN: Automatic Distractor Generation and Evaluation for Reliable Assessment of Generative Models](image_generation/d-gen_automatic_distractor_generation_and_evaluation_for_reliable_assessment_of_.md)**

:   提出 D-GEN——首个开源干扰项生成模型（LLaMA微调，8B/70B），自动将开放式评测题转为多选题格式，配套排名对齐+熵分析两种评估方法验证干扰项质量，在 MMLU 上 Spearman's ρ=0.99 保持模型排名一致性。

**[Planning with Diffusion Models for Target-Oriented Dialogue Systems](image_generation/difftod_diffusion_dialogue_planning.md)**

:   DiffTOD 将对话规划建模为轨迹生成问题，利用掩码扩散语言模型实现非顺序对话规划，并设计三种引导机制（词级/语义级/搜索级）灵活控制对话朝目标推进，在谈判/推荐/闲聊三种场景上显著超越基线。

**[FlashAudio: Rectified Flows for Fast and High-Fidelity Text-to-Audio Generation](image_generation/flashaudio_rectified_flow_tta.md)**

:   将整流流（Rectified Flow）引入文本转音频生成，通过双焦采样器优化时间步分布、不混溶流减少数据-噪声总距离、锚定优化修正 CFG 引导误差，实现单步生成 FAD=1.49 超越百步扩散模型，生成速度达实时 400 倍。

**[R-VC: Rhythm Controllable and Efficient Zero-Shot Voice Conversion via Shortcut Flow Matching](image_generation/rvc_rhythm_voice_conversion.md)**

:   R-VC 是首个实现节奏可控的零样本语音转换系统，通过 Mask Transformer 时长模型建模目标说话人的节奏风格，结合 Shortcut Flow Matching 的 DiT 解码器实现仅 2 步采样的高效高质量语音生成，在 LibriSpeech 上 WER 3.51、说话人相似度 0.930。

---

## 🎯 目标检测 { #object_detection }

**[Anchored Answers: Unravelling Positional Bias in GPT-2's Multiple-Choice Questions](object_detection/anchored_answers_unravelling_positional_bias_in_gpt-2s_multiple-choice_questions.md)**

:   首次对 GPT-2 系列在选择题中的"锚定偏差"（始终偏好选项 A）进行全面机械可解释性分析——通过 Logit Lens 定位导致偏差的 MLP 值向量和注意力头，然后更新值向量+重校准注意力权重，以最小干预消除偏差并将 MCQ 准确率平均提升 70%+。

**[Dolphin: Document Image Parsing via Heterogeneous Anchor Prompting](object_detection/dolphin_document_image_parsing_via_heterogeneous_anchor_prompting.md)**

:   提出 Dolphin，一个轻量级（322M）的文档图像解析模型，采用"先分析后解析"（analyze-then-parse）两阶段范式——先进行页面级布局分析生成阅读顺序的元素序列，再利用异构锚点提示（heterogeneous anchor prompting）并行解析各元素内容，以仅 322M 参数在页面级和元素级解析任务上超越 7B+ 模型和商业系统。

**[Weed Out, Then Harvest: Dual Low-Rank Adaptation is an Effective Noisy Label Detector for Noise-Robust Learning](object_detection/weed_out_then_harvest_dual_low-rank_adaptation_is_an_effective_noisy_label_detec.md)**

:   提出Delora框架，通过引入clean LoRA和noisy LoRA双模块构建噪声标签检测器，将样本选择与模型训练解耦，打破传统"小损失"方法中样本选择与训练互相影响的恶性循环。

**[Why Safeguarded Ships Run Aground? Aligned Large Language Models' Safety Mechanisms Tend to Be Anchored in The Template Region](object_detection/why_safeguarded_ships_run_aground_aligned_large_language_models_safety_mechanism.md)**

:   揭示了安全对齐LLM的一个普遍现象：安全机制过度锚定在chat template区域（TASA），导致越狱攻击可通过干扰template区域的信息处理来绕过安全防线，并提出通过将安全探针从template区域迁移到生成阶段来缓解该漏洞。

---

## 🎮 强化学习 { #reinforcement_learning }

**[Align-SLM: Textless Spoken Language Models with Reinforcement Learning from AI Feedback](reinforcement_learning/align-slm_textless_spoken_language_models_with_reinforcement_learning_from_ai_fe.md)**

:   首次将偏好优化（DPO + RLAIF）应用于无文本口语语言模型（SLM）——从预训练 TWIST 模型生成多个语音续写候选，通过 ASR→LLM 评分自动创建偏好数据对，用 DPO 训练 SLM 一致性地生成语义更好的语音续写，结合课程学习进一步提升。在 ZeroSpeech/StoryCloze 基准上达到 SLM SOTA（sWUGGY 77.9%、S-StoryCloze 61.1%、T-StoryCloze 86.8%）。

**[An Efficient Task-Oriented Dialogue Policy: Evolutionary Reinforcement Learning Injected by Elite Individuals](reinforcement_learning/eierl_dialogue_policy.md)**

:   提出 EIERL 方法，将进化算法（EA）的全局搜索能力与深度强化学习（DRL）的局部优化能力结合用于任务导向对话策略学习，并设计精英个体注入（EII）机制自适应地将高性能个体注入 EA 种群以加速进化，在 4 个数据集上显著提升探索-利用平衡。

**[Prompt-based Personality Profiling: Reinforcement Learning for Relevance Filtering](reinforcement_learning/prompt-based_personality_profiling_reinforcement_learning_for_relevance_filterin.md)**

:   提出RL-Profiler方法，用强化学习训练一个帖子相关性过滤器（SelNet），从用户Profile的大量帖子中筛选出与人格特征相关的少量帖子，再交给LLM零样本预测人格，在大幅减少上下文长度的同时保持接近使用全部帖子的预测效果。

**[TreeRL: LLM Reinforcement Learning with On-Policy Tree Search](reinforcement_learning/treerl_tree_search_rl.md)**

:   提出 TreeRL，将基于熵引导的树搜索（EPTree）直接集成到 LLM 的 on-policy 强化学习训练中，通过在高不确定性 token 处分叉来扩展推理路径多样性，并利用树结构提供的全局+局部优势作为过程监督信号，在数学和代码推理任务上超过传统的多链采样 RL。

---

## 🤖 机器人/具身智能 { #robotics }

**[Rolling the DICE on Idiomaticity: How LLMs Fail to Grasp Context](robotics/dice_idiomaticity.md)**

:   提出 DICE 数据集，通过严格控制习语表面形式不变而仅改变上下文，揭示 LLM 在习语消歧任务上的高准确率很大程度来自对表面线索的依赖而非真正的上下文理解，并发现句子概率和搭配频率对模型表现有复杂影响。

**[Do Emotions Really Affect Argument Convincingness? A Dynamic Approach with LLM-based Manipulation Checks](robotics/do_emotions_really_affect_argument_convincingness_a_dynamic_approach_with_llm-ba.md)**

:   提出一种受心理学操控检验启发的动态框架，利用LLM调节论证的情感强度，系统考察情感对论证说服力的因果影响，发现超过半数情况下人类的说服力判断不受情感变化影响，而当情感有影响时更多是增强而非削弱说服力。

**[SELF-PERCEPT: Introspection Improves LLMs' Detection of Multi-Person Mental Manipulation in Conversations](robotics/self_percept_manipulation_detection.md)**

:   提出 SELF-PERCEPT 两阶段 prompting 框架，借鉴心理学自我知觉理论（Self-Perception Theory），引导 LLM 先观察对话参与者的行为线索再推断内在态度，显著提升多人多轮对话中心理操纵的检测效果。

---

## 🔄 自监督/表示学习 { #self_supervised }

**[Contrastive Learning on LLM Back Generation Treebank for Cross-domain Constituency Parsing](self_supervised/llm_back_gen_treebank.md)**

:   提出 LLM 反向生成方法自动构建跨领域成分句法树库——给定只有领域关键词叶节点的不完整句法树，用 LLM 填充缺失词汇生成完整的跨领域句法树库，结合 span 级对比学习预训练，在 MCTB 五个目标领域上达到跨领域成分句法分析 SOTA。

**[QAEncoder: Towards Aligned Representation Learning in Question Answering Systems](self_supervised/qaencoder_aligned_representation.md)**

:   提出 QAEncoder，一种免训练方法通过蒙特卡洛采样估计文档对应查询的期望嵌入作为文档表示的代理，配合文档指纹保持区分性，在 BEIR 上将 bge-large 从 58.5 提升到 61.8 NDCG@10，零额外存储和延迟开销。

**[SHuBERT: Self-Supervised Sign Language Representation Learning via Multi-Stream Cluster Prediction](self_supervised/shubert_self-supervised_sign_language_representation_learning_via_multi-stream_c.md)**

:   提出 SHuBERT（Sign Hidden-Unit BERT），将语音自监督学习模型 HuBERT 的 masked cluster prediction 范式迁移到手语视频——对手部、面部、身体姿态四个流分别聚类并同时预测 masked 帧的聚类标签，在约 984 小时 ASL 视频上预训练后，在翻译/孤立识别/指拼检测多任务上达到公开数据 SOTA。

---

## 📈 时间序列 { #time_series }

**[CTPD: Cross-Modal Temporal Pattern Discovery for Enhanced Multimodal Electronic Health Records Analysis](time_series/ctpd_cross-modal_temporal_pattern_discovery_for_enhanced_multimodal_electronic_h.md)**

:   提出 CTPD 框架，利用 Slot Attention 从多模态 EHR 数据（不规则时间序列+临床笔记）中发现跨模态共享的时序原型模式，通过 TP-NCE 对比损失对齐两模态的时序语义，在 MIMIC-III 的死亡率预测和表型分类任务上取得 SOTA。

**[LETS-C: Leveraging Text Embedding for Time Series Classification](time_series/lets-c_leveraging_text_embedding_for_time_series_classification.md)**

:   提出 LETS-C——将时间序列数字化为文本字符串后用 text embedding 模型编码，与原始时间序列元素级相加融合后送入轻量 CNN+MLP 分类头，在 UEA 10 个多变量时间序列数据集上以仅 14.5% 的可训练参数量超越 OneFitsAll（GPT-2 微调）等 27 个 baseline 达到 SOTA。

**[Time-MQA: Time Series Multi-Task Question Answering with Context Enhancement](time_series/time-mqa_time_series_multi-task_question_answering_with_context_enhancement.md)**

:   提出Time-MQA框架和TSQA数据集（~200k QA对），将时间序列的预测、填补、异常检测、分类和开放式推理问答统一到自然语言问答范式下，通过持续预训练LLM使其具备时间序列理解和推理能力。

---

## 🧑 人体理解 { #human_understanding }

**[I See What You Mean: Co-Speech Gestures for Reference Resolution in Multimodal Dialogue](human_understanding/i_see_what_you_mean_co-speech_gestures_for_reference_resolution_in_multimodal_di.md)**

:   提出自监督预训练方法学习表征性共语手势（co-speech iconic gestures）的嵌入表示，将骨骼动作 grounded 到语言中，在面对面对话的指称消解任务上证明手势与语音的互补性——手势+语音准确率 31% 远超单独语音 24% 或手势 19%。

**[TransBench: Breaking Barriers for Transferable Graphical User Interface Agents in Dynamic Digital Environments](human_understanding/transbench_breaking_barriers_for_transferable_graphical_user_interface_agents_in.md)**

:   提出首个系统评估 GUI Agent **迁移性**（跨版本/跨平台/跨应用）的 benchmark TransBench，涵盖 81 个中文 App、1459 张截图、22K+ 标注指令，实验表明在旧版本上微调可有效迁移到新版本和其他平台，而跨平台迁移中 Android 数据的泛化性最强。

---

## 🖼️ 图像恢复 { #image_restoration }

**[DiffuseDef: Improved Robustness to Adversarial Attacks via Iterative Denoising](image_restoration/diffusedef_adversarial_defense.md)**

:   DiffuseDef 提出了一种将扩散层作为去噪器插入编码器和分类器之间的对抗防御方法，通过扩散训练学会预测隐状态噪声，推理时对对抗隐状态加噪+迭代去噪+集成，在黑盒和白盒攻击下达 SOTA 鲁棒性。

**[PreP-OCR: A Complete Pipeline for Document Image Restoration and Enhanced OCR Accuracy](image_restoration/prep-ocr_a_complete_pipeline_for_document_image_restoration_and_enhanced_ocr_acc.md)**

:   提出 PreP-OCR 两阶段流水线：先用合成退化数据训练的 ResShift 模型修复历史文档图像（多方向 patch 提取+中值融合），再用 ByT5 做 OCR 后语义纠错，在 13,831 页真实历史文档上降低 CER 63.9-70.3%。

---

## 🎁 推荐系统 { #recommender }

**[GRAM: Generative Recommendation via Semantic-aware Multi-granular Late Fusion](recommender/gram_generative_recommendation.md)**

:   提出 GRAM 生成式推荐模型，通过语义到词汇的翻译（将隐式物品关系编码到 LLM 词汇空间）和多粒度迟融合（分别编码不同粒度提示后在解码时融合），在四个基准上比八个 SOTA 方法在 Recall@5 上提升 11.5-16.0%。

**[RecLM: Recommendation Instruction Tuning](recommender/reclm_recommendation_instruction_tuning.md)**

:   提出 RecLM，一个模型无关的推荐指令微调框架，通过两轮对话式指令微调将协同过滤信号注入 LLM 生成的用户/商品画像，再用 RLHF（PPO）精炼画像质量，在 MIND/Netflix/工业数据集上作为即插即用组件为 BiasMF/NCF/LightGCN/SGL/SimGCL 一致带来提升，尤其在冷启动场景效果显著。

---

## ✂️ 语义分割 { #segmentation }

**[DEF-DTS: Deductive Reasoning for Open-domain Dialogue Topic Segmentation](segmentation/def-dts_deductive_reasoning_for_open-domain_dialogue_topic_segmentation.md)**

:   提出 DEF-DTS，一种基于 LLM 多步演绎推理的对话话题分割方法——通过双向上下文摘要 → 话语意图分类（5 类） → 演绎话题转移判断三步 pipeline，在 TIAGE、SuperDialseg、Dialseg711 三个数据集上取得无监督/prompt 方法 SOTA，在 Dialseg711 上超越监督方法。

**[InstructPart: Task-Oriented Part Segmentation with Instruction Reasoning](segmentation/instructpart_task-oriented_part_segmentation_with_instruction_reasoning.md)**

:   提出 InstructPart，首个将任务导向指令与部件级分割结合的真实世界 benchmark——2400 张图像、48 类物体、44 类部件、9600 条人工标注的任务指令，评估发现当前 VLM 在指令驱动的部件分割上严重不足，基于 LISA+DINOv2 的 baseline 微调后性能提升约 100%。

---

## 🎬 视频理解 { #video_understanding }

**[Generative Frame Sampler for Long Video Understanding](video_understanding/generative_frame_sampler_for_long_video_understanding.md)**

:   提出 GenS，一个基于 VideoLLM 的生成式帧采样模块，用自然语言输出question-aware的相关帧时间段和置信度分数，作为即插即用模块在 LongVideoBench/MLVU/HourVideo 上为多种 VideoLLM 带来 2-4 个点的一致提升。

**[VidCapBench: A Comprehensive Benchmark of Video Captioning for Controllable Text-to-Video Generation](video_understanding/vidcapbench_a_comprehensive_benchmark_of_video_captioning_for_controllable_text-.md)**

:   提出 VidCapBench，首个专为可控文生视频（T2V）设计的视频描述评估 benchmark，从美学/内容/运动/物理规律四个维度评估 caption 质量，643 个视频+10,644 个 QA 对，实验证明 VidCapBench 分数与 T2V 生成质量高度正相关。

---

## 🧊 3D 视觉 { #3d_vision }

**[Slamming: Training a Speech Language Model on One GPU in a Day](3d_vision/slamming_training_a_speech_language_model_on_one_gpu_in_a_day.md)**

:   提出 Slam 训练配方，通过系统化的模型初始化、架构选择、合成数据、偏好优化等环节优化，在单张 A5000 GPU 上 24 小时内训练出性能媲美大规模 SLM 的语音语言模型。

---

## 🚗 自动驾驶 { #autonomous_driving }

**[Embracing Large Language Models in Traffic Flow Forecasting](autonomous_driving/embracing_large_language_models_in_traffic_flow_forecasting.md)**

:   提出 LEAF 框架，用图分支（pair-wise关系）和超图分支（non-pair-wise关系）的双分支预测器生成候选预测，再用冻结的 LLM 作为选择器（判别而非生成）挑选最优预测，通过 ranking loss 反馈优化预测器，在 PEMS 数据集上取得 SOTA。

---

## 📐 优化/理论 { #optimization }

**[ScaleBiO: Scalable Bilevel Optimization for LLM Data Reweighting](optimization/scalebio_bilevel_data_reweighting.md)**

:   ScaleBiO 提出基于罚函数重构的全一阶双层优化算法，首次将双层优化应用于 30B+ 参数 LLM 的数据源重加权，在 Qwen-2.5-32B 上实现 GSM8K +9%、MATH +5.8% 的提升。

---

## 📡 信号/通信 { #signal_comm }

**[ToolSpectrum: Towards Personalized Tool Utilization for Large Language Models](signal_comm/toolspectrum_towards_personalized_tool_utilization_for_large_language_models.md)**

:   提出 ToolSpectrum benchmark，首次评估 LLM 在用户画像和环境因素双维度下的个性化工具选择能力，发现现有 SOTA 模型在联合推理两个维度时表现显著下降。

---

## 📂 其他 { #others }

**[Towards Robust ESG Analysis Against Greenwashing Risks: A3CG](others/a3cg_esg_greenwashing.md)**

:   提出 A3CG 数据集和方面-行动分析任务（从可持续性声明中提取方面及其行动类型：已实施/计划中/不确定），通过跨类别泛化设置评估 NLP 方法抵御漂绿风险的鲁棒性，发现监督学习（GRACE F1=47.51）优于 LLM（Claude 3.5 F1=42.03）但泛化效率更差。

**[A Conformal Risk Control Framework for Granular Word Assessment and Uncertainty Calibration of CLIPScore Quality Estimates](others/a_conformal_risk_control_framework_for_granular_word_assessment_and_uncertainty_.md)**

:   提出基于保形风险控制（Conformal Risk Control）框架校准 CLIPScore 的方法——通过对 CLIP 视觉/文本编码器的注意力掩码采样生成 CLIPScore 分布，然后利用保形风险控制 (1) 检测图像描述中的干扰词（foil words），(2) 生成校准置信区间，在 FOIL-it/FOIL-nocaps/Rich-HF 基准上以简单方法达到与复杂专用方法相当的干扰词检测性能，同时提供形式化风险保证。

**[A Dual-Perspective NLG Meta-Evaluation Framework with Automatic Benchmark and Better Interpretability](others/a_dual-perspective_nlg_meta-evaluation_framework_with_automatic_benchmark_and_be.md)**

:   揭示传统 NLG 元评估的局限（人工评分平均聚合不合理、相关系数选择模糊），提出双视角元评估框架：全局视角（序数分类，评估粗粒度评级能力）+ 局部视角（相邻成对比较，评估细粒度区分能力），并引入基于可控错误注入的自动基准构建方法，在 16 个 LLM 上验证不同模型在两个视角上的能力分布差异显著。

**[GKI-ICD: A General Knowledge Injection Framework for ICD Coding](others/a_general_knowledge_injection_framework_for_icd_coding.md)**

:   提出 GKI-ICD，首个无需额外专用网络模块即可同时注入三种 ICD 编码知识（描述、同义词、层级结构）的通用框架——通过合成知识引导文本（Guideline Synthesis）+ 多任务学习实现知识注入，在 MIMIC-III 和 MIMIC-III-50 上大多数指标达到 SOTA。

**[Barec: A Large and Balanced Corpus for Fine-grained Arabic Readability Assessment](others/a_large_and_balanced_corpus_for_fine-grained_arabic_readability_assessment.md)**

:   构建 Barec（Balanced Arabic Readability Evaluation Corpus）——首个大规模细粒度阿拉伯语可读性评估语料库，包含 69K+ 句子（100万+词），覆盖 19 个可读性等级（从幼儿园到研究生），由 6 名专业阿拉伯语教育者标注，并在多种分级粒度（19/7/5/3级）上基准测试自动可读性评估模型。

**[A Little Human Data Goes A Long Way](others/a_little_human_data_goes_a_long_way.md)**

:   首次系统研究合成数据在事实验证和问答中能否替代人工标注——替换 90% 仅轻微下降，但替换最后 10% 严重退化；仅 125 条人工数据可显著提升纯合成模型，等效增益需 10 倍以上合成数据。

**[MisMatched: A Benchmark for Scientific Natural Language Inference](others/a_mismatched_benchmark_for_scientific_natural_language_inference.md)**

:   引入 MisMatched——首个覆盖非 CS 领域（心理学、工程、公共卫生）的科学 NLI 评估基准，2700 对人工标注句子对，最佳基线 Macro F1 仅 78.17%，且发现训练时加入隐式关系句子对可提升性能。

**[A Practical Approach for Building Production-Grade Conversational Agents with Workflow Graphs](others/a_practical_approach_for_building_production-grade_conversational_agents_with_wo.md)**

:   提出基于有向无环图（DAG）工作流的混合对话 Agent 框架——每个图节点有独立系统提示/工具/执行规则以处理特定场景约束。结合原型 Agent 数据收集、状态感知的响应掩码微调策略，在 Kakao 移动电商场景中任务准确率提升 52%、格式遵从度提升 50%，超越 GPT-4o。

**[SSUF: A Semi-supervised Scalable Unified Framework for E-commerce Query Classification](others/a_semi-supervised_scalable_unified_framework_for_e-commerce_query_classification.md)**

:   提出电商查询分类的半监督可扩展统一框架 SSUF——三个可插拔模块：知识增强（LLM 世界知识+后验点击）解决短查询信息不足、标签增强（语义编码+半监督信号）打破对后验标签的依赖、结构增强（共现+语义+层级图 GCN）传播长尾标签梯度。已在 JD.COM 部署，离线和在线 A/B 实验均显著超越 SOTA。

**[A Spatio-Temporal Point Process for Fine-Grained Modeling of Reading Behavior](others/a_spatio-temporal_point_process_for_fine-grained_modeling_of_reading_behavior.md)**

:   提出基于标记时空点过程（Marked Spatio-Temporal Point Process）的阅读行为概率模型——用 Hawkes 过程建模跳视（何时何处注视），用对数正态分布+卷积建模注视时长（持续多久），避免传统聚合测量的信息丢失。实证发现：Hawkes 模型比基线更好拟合跳视数据，但上下文 surprisal 作为预测因子仅带来边际改善——surprisal 理论难以解释细粒度眼动。

**[IDR²: Accelerating Adaptive RAG via Instruction-Driven Representation Reduction of Retrieval Overlaps](others/accelerating_adaptive_retrieval_augmented_generation_via_instruction-driven_repr.md)**

:   首次识别自适应 RAG（A-RAG）中多轮检索结果重叠导致的冗余计算问题，提出 IDR²（Instruction-Driven Representation Reduction）框架：跨迭代 KV 缓存共享（CICS）加速预填充 2.79 倍、指令引导去重增强（IDGR）帮助 LLM 正确处理缓存vs新文档、信息引导并行生成（IGPG）加速解码 2.33 倍，总体 A-RAG 流程加速 2.0 倍且不损失生成质量。

**[Access Denied Inc: The First Benchmark Environment for Sensitivity Awareness](others/access_denied_inc_the_first_benchmark_environment_for_sensitivity_awareness.md)**

:   提出敏感性感知（Sensitivity Awareness, SA）概念——评估 LLM 是否能遵守基于角色的访问控制规则——并构建首个评估基准 Access Denied Inc：模拟企业数据库 + 多用户组权限 + 自动化问卷+半自动评分（99.9%自动），揭示模型在拒绝未授权请求和响应合法查询上的显著差异。

**[AceCoder: Acing Coder RL via Automated Test-Case Synthesis](others/acecoder_acing_coder_rl_via_automated.md)**

:   构建 AceCode-87K（87K 编码题 + 138 万自动合成测试用例），训练代码专用 Reward Model（7B 超越 340B Nemotron），Best-of-N 提升 Llama-3.1-8B 平均 8.9 分，R1 风格从 base 直接 RL 仅 80 步 HumanEval+ 提升 22.5%。

**[ACORD: An Expert-Annotated Retrieval Dataset for Legal Contract Clause Retrieval](others/acord_an_expert-annotated_retrieval_dataset_for_legal_contract_drafting.md)**

:   构建首个面向合同起草的专家标注条款检索基准 ACORD——114 个律师编写的查询、126,000+ 查询-条款对、1-5 星相关性评分，聚焦责任限制/赔偿/控制权变更/最惠国等复杂条款；bi-encoder 检索 + LLM 点式重排序表现有前景但距律师需求仍有显著差距。LLM 直接起草对比律师修改暴露了多种缺陷。

**[NTIL: Advancing Sequential Numerical Prediction in Autoregressive Models](others/advancing_sequential_numerical_prediction_in_autoregressive_models.md)**

:   提出 NTIL（Numerical Token Integrity Loss）解决自回归模型数值预测的两大缺陷——(1) token 级用 EMD 替代交叉熵保留数字间序数关系+指数位置加权，(2) 序列级通过可微数值构建+相对偏差度量评估整体数值误差。首次将 EMD 用于自回归模型优化，在目标检测/文本识别/数学推理上显著提升。

**[AIDE: Attribute-Guided Multi-Hop Data Expansion for Data Scarcity in Task-Specific Fine-tuning](others/aide_attribute-guided_multi-hop_data_expansion_for_data_scarcity_in_task-specifi.md)**

:   提出 AIDE——属性引导的多跳数据扩展框架，从仅 10 个种子数据点出发，通过提取主题/属性/关系三元组引导 LLM 多跳递归合成新数据，加入 Persona 增加多样性和残差连接防止偏离，在 Mistral-7B/Llama-3.1-8B/Llama-3.2-3B 上超越人工标注数据微调，比 Evol-Instruct 等 SOTA 提升 30%+。

**[AIR-Bench: Automated Heterogeneous Information Retrieval Benchmark](others/air-bench_automated_heterogeneous_information_retrieval_benchmark.md)**

:   提出 AIR-Bench——首个自动化、异构、动态的信息检索评测基准，通过 LLM 自动从真实语料生成高质量测试数据（三阶段管线：语料准备→候选生成→质量控制），覆盖 2 类任务、9 个领域、13 种语言共 69 个数据集，与人工标注数据高度一致，且持续动态更新避免数据泄露。

**[Aligned but Blind: Alignment Increases Implicit Bias by Reducing Awareness of Race](others/aligned_but_blind_implicit_bias.md)**

:   揭示对齐训练的"种族盲视"副作用：对齐使 LLM 在歧义上下文中不再将 black/white 表征为种族概念，安全护栏因此无法激活，导致隐式偏见从 64.1% 飙升至 91.4%；反直觉地，在早期层注入种族感知激活（而非遗忘）可将隐式偏见从 97.3% 降至 42.4%。

**[AmbiK: Dataset of Ambiguous Tasks in Kitchen Environment](others/ambik_dataset_of_ambiguous_tasks_in_kitchen_environment.md)**

:   提出 AmbiK，一个专门用于厨房环境中歧义指令检测的纯文本数据集，包含 1000 对歧义/非歧义指令，按三种歧义类型（用户偏好/常识/安全）分类标注，并评估了多种基于 conformal prediction 的歧义检测方法，发现现有方法在该基准上表现很差。

**[AnalyticKWS: Towards Exemplar-Free Analytic Class Incremental Learning for Small-footprint Keyword Spotting](others/analytickws_towards_exemplar-free_analytic_class_incremental_learning_for_small-.md)**

:   提出 AnalyticKWS，一种无需存储历史样本的关键词检测增量学习方法，通过冻结特征提取器 + 递归最小二乘解析解更新分类器，在 GSC 和 SC-100 数据集上超过了所有基于样本回放的方法，且训练时间和内存开销极低。

**[AntiLeakBench: Preventing Data Contamination by Automatically Constructing Benchmarks with Updated Real-World Knowledge](others/antileakbench_preventing_data_contamination_by_automatically_constructing_benchm.md)**

:   提出 AntiLeakBench——自动化反泄露基准框架，通过识别 LLM 知识截止后更新的真实世界新知识自动构建 QA 测试样本（而非简单收集新发布数据），确保测试知识严格不在训练集中，全自动流程无需人工标注，实验证实截止后性能普遍下降验证了数据污染的普遍存在。

**[Anything Goes? A Crosslinguistic Study of (Im)possible Language Learning in LMs](others/anything_goes_a_crosslinguistic_study_of_impossible_language_learning_in_lms.md)**

:   跨语言研究 LM 能否区分可能语言和不可能语言——在 12 种语言（4 个语系）上训练 GPT-2 Small 的可能/不可能/未见证变体，发现模型大体上能区分可能 vs 不可能语言（单语言内），但**跨语言**时区分能力减弱，且对类型学未见证语言（Greenberg Universal 20 的未见证词序）的区分仅在泛化测试中有效而在困惑度上无效——LM 有部分人类样的归纳偏置但弱于人类。

**[Are Bias Evaluation Methods Biased?](others/are_bias_evaluation_methods_biased.md)**

:   严格控制变量后比较三种主流偏见评估方法（结构化问答 BBQ、LLM-as-a-Judge、情感分析），发现不同方法对同一组 LLM 产生显著不同的偏见排名——偏见评估方法本身就是有偏的，企业不应依赖单一偏见基准来选择模型。

**[ARise: Towards Knowledge-Augmented Reasoning via Risk-Adaptive Search](others/arise_risk_adaptive_search.md)**

:   提出 ARise 框架，将贝叶斯风险评估与动态 RAG 集成到蒙特卡洛树搜索中，解决知识增强推理中的错误传播和验证瓶颈问题，在多跳QA任务上平均准确率超 SOTA KAR 方法 23.10%，超 RAG-equipped 推理模型（DeepSeek-R1）25.37%。

**[Attention Entropy is a Key Factor for Parallel Context Encoding](others/attention_entropy_parallel_encoding.md)**

:   发现并行上下文编码导致 query token 的注意力熵异常升高是性能下降的关键因素，并提出 Attention Sink 共享前缀和 Selective Attention 两种免微调方法有效缓解该问题。

**[AutoDS: Autonomous Data Selection with Zero-shot Generative Classifiers for Mathematical Texts](others/autonomous_data_selection_with_zero-shot_generative_classifiers_for_mathematical.md)**

:   提出 AutoDS——用基座 LLM 自身作为零样本"生成分类器"自动评估数学文本质量。通过两个 yes/no 问题的 logits 计算连续 LM-Score（而非二分类），筛选高质量数学文本做持续预训练，在 MATH/GSM8K/BBH 上大幅提升并实现约 2 倍 token 效率提升。发布 AutoMathText 数据集。

**[Behavioural vs. Representational Systematicity in End-to-End Models: An Opinionated Survey](others/behavioural_vs_representational_systematicity_in_end-to-end_models_an_opinionate.md)**

:   区分行为系统性（模型能否正确处理新组合）与表征系统性（模型内部表征是否结构化），指出当前基准和模型主要测试行为系统性却常声称解决了 Fodor-Pylyshyn 对表征系统性的挑战。基于 Hadley (1994) 的三级分类（弱/准/强系统性）分析语言和视觉关键基准的测试范围，最终呼吁用机械可解释性方法在行为评估之上补充表征分析。

**[Better Embeddings with Coupled Adam](others/better_embeddings_with_coupled_adam.md)**

:   从理论上证明 Adam 优化器的逐 token 二阶矩是导致 LLM 词嵌入各向异性（均值偏移）的根因，提出 Coupled Adam——对嵌入层的二阶矩取词汇平均——消除了各向异性问题，并在大规模实验中提升了嵌入质量和下游性能。

**[Beyond Frameworks: Unpacking Collaboration Strategies in Multi-Agent Systems](others/beyond_frameworks_multi_agent_collaboration.md)**

:   本文系统化地将多智能体协作分解为四个维度（治理模式、参与控制、交互模式、上下文管理），通过两个上下文依赖任务的大量实验证明：集中治理+指导者控制参与+有序交互+指导者摘要的组合最优，在保持甚至提升准确率的同时减少高达 93% 的 token 消耗。

**[Bone Soups: A Seek-and-Soup Model Merging Approach for Controllable Multi-Objective Generation](others/bone_soups_multi_objective_gen.md)**

:   提出 Bone Soup 模型合并方法，通过先构造"骨架奖励"（多目标奖励的组合）训练骨架模型、再用对称循环矩阵映射确定合并系数，解决了 Rewarded Soup 中单目标模型合并的次优性问题，在三个多目标生成任务上实现更好的 Pareto 前沿和可控性。

**[Bregman Conditional Random Fields: Sequence Labeling with Parallelizable Inference](others/bregman_conditional_random_fields_sequence_labeling_with_parallelizable_inferenc.md)**

:   提出 Bregman CRF (Bcrf)，一种基于均值正则化（mean regularization）的新型序列标注判别模型，使用迭代 Bregman 投影实现可并行化的推理算法，替代传统 CRF 中固有顺序的 Viterbi/Forward 算法，在 POS/NER/分词任务上性能与标准 CRF 持平但更快，且在有禁止标签转移约束的场景下优于 Mean Field 方法。

**[Byte Latent Transformer: Patches Scale Better Than Tokens](others/byte_latent_transformer.md)**

:   Meta FAIR提出BLT——首个在大规模（8B参数/4T字节）上匹配基于tokenizer的LLM性能的字节级架构，通过基于下一字节熵的动态分组（patching）将字节聚合为可变长度patch，在保持性能的同时实现最高50%推理FLOP节省，并开辟了"同时增大模型和patch尺寸"的全新scaling维度。

**[Causal Estimation of Tokenisation Bias](others/causal_tokenisation_bias.md)**

:   本文首次将 tokeniser 选择对语言模型输出的影响定义为"分词偏差"(tokenisation bias)，并利用因果推断中的断点回归设计(RDD)来量化这一效应——发现当一个 subword 被纳入词表时，其对应字符串的概率最高可提升 17 倍（小模型），揭示分词是语言建模中一个被低估的关键设计选择。

**[CoachMe: Decoding Sport Elements with a Reference-Based Coaching Instruction Generation Model](others/coachme_sport_instruction.md)**

:   提出 CoachMe，通过对比学习者动作与参考动作的差异（时间+物理两个维度），自动生成运动特异性的教练指导文本，在花样滑冰和拳击上分别超过 GPT-4o 31.6% 和 58.3%（G-Eval）。

**[Collapse of Dense Retrievers: Short, Early, and Literal Biases Outranking Factual Evidence](others/collapse_dense_retrievers.md)**

:   本文首次系统研究稠密检索器中多种启发式偏见（简短偏见、前置偏见、字面偏见、重复偏见）的个体和组合效应，发现当多种偏见叠加时，检索器选择包含答案的文档的概率低于10%，且这些偏见可被利用来操纵RAG系统，导致34%的性能下降。

**[Com2: A Causal-Guided Benchmark for Complex Commonsense Reasoning](others/com2_causal_commonsense.md)**

:   提出Com2基准，利用因果事件图和因果理论（干预/反事实）构建复杂常识推理任务，发现LLM在推理深度和广度上存在不足，后训练和慢思考可部分缓解。

**[Commonsense Reasoning in Arab Culture](others/commonsense_arab_culture.md)**

:   构建首个阿拉伯文化特定常识推理数据集 ArabCulture（3482 道由母语者原创的题目，覆盖 13 国×54 主题），评估多种 LLM 发现即使 32B 参数模型也在文化常识推理上表现不佳，且不同地区表现差异显著，地理/文化上下文线索的加入仅部分有效。

**[Completing A Systematic Review in Hours instead of Months with Interactive AI Agents](others/completing_a_systematic_review_in_hours.md)**

:   提出 InsightAgent，一个以人为中心的交互式多 Agent 系统，通过语义聚类分区、多 agent 并行阅读和实时用户交互，将医学系统综述的撰写时间从数月缩短到约 1.5 小时，达到人类撰写质量的 79.7%。

**[CORAL: Learning Consistent Representations across Multi-step Training with Lighter Speculative Drafter](others/coral_speculative_drafting.md)**

:   CORAL 通过跨步表示对齐（CSRA）改进多步训练中 draft 模型的特征一致性，并用权重分组机制压缩大词表 LM head 的推理延迟，在 LLaMA3/Qwen2.5 上实现 2.50-4.07× 加速，超越 EAGLE-2 和 HASS。

**[Cramming 1568 Tokens into a Single Vector and Back Again: Exploring the Limits of Embedding Space Capacity](others/cramming_tokens_embedding_capacity.md)**

:   通过逐样本优化方法将文本压缩到可训练的 [mem] 向量中，发现 Llama-3.1-8B 可以将 1568 个 token 无损压缩到单个输入向量中，揭示了现有方法（约 x10 压缩比）与实际可达极限（x1500+）之间存在两个数量级的差距。

**[Generating Plausible Distractors for Multiple-Choice Questions via Student Choice Prediction](others/distractor_gen_multiple_choice.md)**

:   提出选择题干扰项生成的三步流水线：(1) 训练配对排序器预测学生误选哪个干扰项；(2) 用排序器构造偏好数据集；(3) 用 DPO 训练生成器产生更具迷惑性的干扰项。在 CS 领域（Python/DB/ML）实验中，生成的干扰项更难以区分，且题目鉴别指数(DI)更高。

**[Divide-Then-Align: Honest Alignment based on the Knowledge Boundary of RAG](others/divide_then_align_rag_knowledge_boundary.md)**

:   DTA 提出将 RAG 查询按参数知识边界和检索知识边界划分为四个象限，对"两者都不知道"的查询构造偏好数据用 DPO 训练模型回答"我不知道"，解决了 RAFT 模型即使在检索完全噪声时也强行生成答案的问题，在准确率和适当弃权之间实现了有效平衡。

**[DoMIX: An Efficient Framework for Exploiting Domain Knowledge in Fine-Tuning](others/domix_an_efficient_framework_for_exploiting.md)**

:   提出 DoMIX，将各领域知识用独立 LoRA 模块存储后通过对角初始化的 bridge 矩阵在微调时灵活组合利用，在持续领域适应预训练场景下减少 58% 预训练时间和 87% GPU 内存，同时性能超越 SOTA。

**[Principled Content Selection to Generate Diverse and Personalized Multi-Document Summaries](others/dpp_diverse_multidoc_summary.md)**

:   用行列式点过程（DPP）替代 LLM 的隐式内容选择来生成多样化多文档摘要——将任务分解为"提取原子关键点→DPP 选择多样子集→LLM 重写为摘要"三步，解决了 LLM 的"lost in the middle"问题，在 DiverseSumm 基准上一致提升了源覆盖率。

**[DREsS: Dataset for Rubric-based Essay Scoring on EFL Writing](others/dress_dataset_rubric_based_essay_scoring_efl_writing.md)**

:   发布 DREsS——一个面向 EFL（英语作为外语）写作教育的大规模标准化评分量规数据集（48.9K 样本），并提出基于文本损坏的 CASE 数据增强策略，将基线性能提升 45.44%。

**[Divide-Then-Aggregate: An Efficient Tool Learning Method via Parallel Tool Invocation](others/dta_llama_parallel_tool_invocation.md)**

:   提出 DTA-Llama，将传统树搜索的串行工具调用路径转换为有向无环图（DAG）结构实现并行调用，设计 Process/Thread 推理框架使 LLM 在每轮中可分解任务并并行执行多个工具，在 StableToolBench 上使 Llama2-7B 达到 GPT-3.5 Parallel Function Calling 的水平。

**[An Effective Incorporating Heterogeneous Knowledge Curriculum Learning for Sequence Labeling](others/dual_stage_curriculum_learning_sequence_labeling.md)**

:   提出面向序列标注任务的双阶段课程学习（DCL）框架，通过数据级和模型级两阶段的由易到难训练策略以及基于贝叶斯不确定性的动态难度度量，在提升性能的同时加速训练超过 25%。

**[EcomScriptBench: A Multi-task Benchmark for E-commerce Script Planning via Step-wise Intention-Driven Product Association](others/ecomscriptbench.md)**

:   提出电商脚本规划（EcomScript）任务及其首个大规模benchmark EcomScriptBench（605K脚本、2.4M产品），通过购买意图（purchase intention）桥接用户行动步骤与产品检索的语义鸿沟，实验发现当前LLM在涉及产品的子任务上表现显著不足，注入意图知识可提升性能。

**[Efficient Knowledge Editing via Minimal Precomputation](others/efficient_knowledge_editing.md)**

:   证明了 MEMIT/ROME/EMMET 等知识编辑方法的预计算步骤（缓存 4400 万隐向量）可以减少到理论最小值的 2-10 倍（不到原来的 0.3%），将预计算时间从数十小时降到几分钟，且编辑性能基本无损。

**[Enhancing Transformers for Generalizable First-Order Logical Entailment](others/enhancing_fol_entailment.md)**

:   系统性研究 Transformer 在一阶逻辑蕴涵任务中的泛化推理能力，揭示了查询语法、token 嵌入和 Transformer 架构（特别是位置编码）的影响，并提出 TEGA（Transformer Encoder with Guided Attention）在相对位置编码设定下显著提升逻辑推理性能。

**[Model Extrapolation Expedites Alignment](others/expo_model_extrapolation.md)**

:   基于"对齐训练仅产生微小参数变化"的观察，提出ExPO方法——通过放大SFT→DPO的参数变化方向（$\theta_2 = \theta_1 + \alpha\Delta\theta$），在零额外训练开销下提升对齐性能，使仅训练20%步骤的DPO模型超越完整训练的版本。

**[FaithfulRAG: Fact-Level Conflict Modeling for Context-Faithful Retrieval-Augmented Generation](others/faithfulrag_fact_level_conflict.md)**

:   发现现有忠实 RAG 方法通过强制抑制参数知识来实现上下文忠实，但这增加了误解上下文的风险（不忠实错误减少 6.65% 的同时错误匹配增加 6.42%）。提出 FaithfulRAG，通过事实级冲突识别（自事实挖掘）和冲突推理（自思考模块）解决知识冲突，在 FaithEval/SQuAD/MuSiQue/RealtimeQA 上超越最强基线 8-9 个百分点。

**[Towards Robust and Efficient Federated Low-Rank Adaptation with Heterogeneous Clients](others/federated_lora_heterogeneous.md)**

:   提出 LoRA-A²（Low Rank Adaptation with Alternating freeze and Adaptive rank selection），通过交替冻结 A/B 矩阵解决联邦 LoRA 聚合不一致问题，并结合自适应秩选择机制在大幅压缩上传参数量（最高减少 99.8%）的同时保持鲁棒性，尤其在低秩+高数据异构场景下显著优于现有方法。

**[FinanceReasoning: Benchmarking Financial Numerical Reasoning More Credible, Comprehensive and Challenging](others/financereasoning_benchmarking_financial_numerical_reasoning_more.md)**

:   提出 FinanceReasoning benchmark，通过重标注公开数据集、构建 3,133 个 Python 金融函数库和新增 908 道专家标注难题，在可信度、全面性和挑战性三个维度提升金融数值推理评估能力。

**[Behind Closed Words: Creating and Investigating the forePLay Annotated Dataset for Polish Erotic Discourse](others/foreplay_polish_erotic_detection.md)**

:   构建了首个波兰语色情内容检测数据集 forePLay（24,768 句，5 类标签），提出涵盖模糊性、暴力和社会不可接受行为的多维标注体系，评估发现专用波兰语模型显著优于多语言模型，且 Transformer 编码器模型在不平衡类别处理上表现最强。

**[Frictional Agent Alignment Framework: Slow Down and Don't Break Things](others/frictional_agent_alignment.md)**

:   提出摩擦对齐框架 FAAF（Frictional Agent Alignment Framework），通过双策略（frictive state policy + intervention policy）目标函数，训练 LLM 在协作对话中识别信念冲突并生成促进反思与审议的"摩擦"干预，超越 DPO/IPO/PPO 等对齐方法。

**[GainRAG: Preference Alignment in Retrieval-Augmented Generation through Gain Signal Synthesis](others/gainrag_preference_alignment.md)**

:   发现 RAG 中检索器的"相关性"与 LLM 的"偏好"存在偏差——含正确答案的段落仍可能导致错误生成，而间接相关段落反而有用。提出 GainRAG，用基于对比解码困惑度的"增益"信号量化 LLM 偏好，训练轻量选择器在检索结果中选择真正有"增益"的段落，在 6 个 QA 数据集上显著超越 Standard RAG 和 Rerank 基线。

**[Controllable and Reliable Knowledge-Intensive Task-Oriented Conversational Agents with Declarative Genie Worksheets](others/genie_worksheets_tod_agent.md)**

:   Genie 提出了一个可编程的知识密集型任务导向对话框架，通过声明式 Worksheet 规范定义 Agent 策略，将 LLM 限制在语义解析和回复生成两个角色，由算法化运行时系统强制执行策略，实现从 21.8% 到 82.8% 的真实任务完成率提升。

**[Gumbel Reranking: Differentiable End-to-End Reranker Optimization](others/gumbel_reranking.md)**

:   将 RAG 系统中的重排序过程重新建模为文档级 Top-k 注意力掩码问题，利用 Gumbel 技巧和松弛 Top-k 采样实现端到端可微优化，直接最小化最终语言建模损失，在 HotpotQA 上 Recall@5 提升 10.4%。

**[HateDay: Insights from a Global Hate Speech Dataset Representative of a Day on Twitter](others/hateday_global_hate_speech.md)**

:   HateDay 构建了首个全球代表性仇恨言论数据集——24 万条随机采样的 Twitter 推文覆盖 8 种语言和 4 个英语国家，揭示了学术数据集大幅高估了检测模型在真实场景中的表现，尤其对非欧洲语言检测能力极差。

**[Hierarchical Bracketing Encodings for Dependency Parsing as Tagging](others/hierarchical_bracketing_dep_parsing.md)**

:   提出层次化括号编码家族用于依存句法分析的序列标注范式，证明现有4-bit编码是该家族的非最优特例，推导出仅需12个标签的最优编码，并将其推广到处理任意非投射性。

**[Hierarchical Memory Organization for Wikipedia Generation](others/hierarchical_memory_wikipedia_gen.md)**

:   提出 Memory Organization-based Generation（MOG）框架，从网页文档中提取细粒度记忆单元（factoid），通过递归聚类-摘要算法组织为层次化 Wikipedia 大纲结构，使每个章节都有直接的记忆支撑，在 FreshWiki 和 WikiStart 数据集上信息量、引用率和可验证性全面超越 RAG 和 STORM 基线。

**[Hierarchical Level-Wise News Article Clustering via Multilingual Matryoshka Embeddings](others/hierarchical_news_clustering.md)**

:   本文提出利用多语言 Matryoshka 嵌入的分层特性进行新闻文章聚类：低维捕捉主题级相似度、中维捕捉叙事级相似度、高维捕捉事件级相似度，结合改良的 RAC 层级聚类算法，在 SemEval 2022 Task 8 上达到 SOTA（Pearson ρ = 0.816）。

**[Counterspeech the Ultimate Shield! Multi-Conditioned Counterspeech Generation through Attributed Prefix Learning](others/hippro_counterspeech_gen.md)**

:   提出 HiPPrO 两阶段框架用于多条件反仇恨言论生成——第一阶段通过层次化前缀学习在多个属性（策略+情感）空间中优化反言论生成，第二阶段用无参考无奖励的偏好优化提升建设性，策略一致性提升 ~38%，ROUGE 指标提升 2-3%。

**[HybGRAG: Hybrid Retrieval-Augmented Generation on Textual and Relational Knowledge Bases](others/hybgrag_hybrid_rag_skb.md)**

:   提出 HybGRAG 方法，通过检索器库（Retriever Bank）同时利用文本和关系信息，配合 Critic 模块的自反思迭代纠正问题路由错误，在半结构化知识库上的混合问答任务中 Hit@1 平均提升 51%。

**[Enhancing Hyperbole and Metaphor Detection with Their Bidirectional Dynamic Interaction and Emotion Knowledge](others/hyperbole_metaphor_detection.md)**

:   提出 EmoBi 框架，通过情感分析→情感引导的域映射→双向动态交互三阶段 prompting 流程，利用 LLM 挖掘夸张和隐喻背后的情感线索及二者的互促关系，在四个数据集上大幅超越 SoTA（TroFi 上夸张检测 F1 提升 28.1%，HYPO-L 上隐喻检测 F1 提升 23.1%）。

**[If Attention Serves As A Cognitive](others/if_attention_serves_as_a_cognitive.md)**

:   通过将 Transformer Grammar（TG）的注意力机制与人类阅读时间数据关联，首次证明在句法结构上操作的注意力比在 token 序列上操作的普通 Transformer 注意力能更好地预测人类阅读行为，揭示人类句子处理涉及"句法结构+词序列"的双重记忆表征。

**[Predicting Implicit Arguments in Procedural Video Instructions](others/implicit_arguments_video_instructions.md)**

:   提出 Implicit-VidSRL 数据集与 iSRL-Qwen2-VL 模型，针对过程性视频指令中省略的隐含论元（食材成分）进行预测，通过 SRL 框架将多步指令分解为 {verb, what, where/with} 三元组，在银标数据上微调后在隐含论元 F1 上超越 GPT-4o 达 17%。

**[ImpliHateVid: Implicit Hate Speech Detection in Videos](others/implihatevid_video_hate.md)**

:   首次提出视频中隐性仇恨言论检测任务，构建2009个视频的ImpliHateVid数据集，并设计两阶段对比学习框架融合文本、图像、音频三模态特征。

**[Ewe: Improving Factuality with Explicit Working Memory](others/improving_factuality_with_explicit_working_memory.md)**

:   提出 Ewe（Explicit Working mEmory），在 LLM 解码过程中引入由多个 KV cache 单元组成的显式工作记忆，实时接收检索知识反馈和事实核查反馈，检测到错误时删除错误句子并用更新后的记忆重新生成，在 4 个事实性长文本生成基准上将 VeriScore F1 提升 2–6 分且不损失回答有用性。

**[Improving Language and Modality Transfer in Translation by Character-level Modeling](others/improving_language_and_modality_transfer_in.md)**

:   提出基于字符级编码器 charSONAR 的跨语言跨模态翻译方法，通过 teacher-student 训练获得字符级文本编码器，再用轻量适配器连接 1000+ 语言的 CTC ASR 模型（MMS），在 75 语言文本翻译和 33 语言语音翻译上实现 SOTA，零资源低资源场景表现尤其突出。

**[InSerter: Speech Instruction Following with Unsupervised Interleaved Pre-training](others/inserter_speech_instruction.md)**

:   提出 InSerter（交错语音-文本预训练）方法，通过 TTS 将大规模文本语料合成为交错的语音-文本序列进行预训练，大幅提升 SpeechLLM 的语音指令遵循能力，并构建首个全面的语音指令遵循基准 SpeechInstructBench。

**[InspireDebate: Multi-Dimensional Evaluation-Guided Reasoning for Debating](others/inspiredebate_multidim_evaluation_debating.md)**

:   提出双组件框架：InspireScore（融合4个主观维度+2个客观维度的辩论评估系统）和 InspireDebate（通过CoT-SFT + 多维DPO + Web-RAG 三阶段优化的辩论框架），评估系统与专家判断相关性提高 44%，辩论性能超越基线 57%。

**[Intuitive Fine-Tuning: Towards Simplifying Alignment into a Single Process](others/intuitive_fine_tuning.md)**

:   通过MDP框架将SFT和偏好优化统一建模为"偏好估计+转移优化"两个子过程，揭示SFT本质上是偏好优化的特殊退化形式（使用偏差先验），提出IFT方法通过时间残差连接在仅使用SFT格式数据的条件下实现接近或超越SFT+PO顺序训练的对齐效果。

**[KnowShiftQA: How Robust are RAG Systems when Textbook Knowledge Shifts in K-12 Education?](others/knowshiftqa_rag_knowledge_shifts.md)**

:   构建了 KnowShiftQA 数据集（3,005 道题，覆盖 5 个学科），通过假设性知识更新模拟教科书与 LLM 参数知识的差异，系统评估 RAG 系统面对知识偏移时的鲁棒性，发现现有 RAG 系统在知识偏移下性能下降 22-27%。

**[KokoroChat: A Japanese Psychological Counseling Dialogue Dataset Collected via Role-Playing by Trained Counselors](others/kokorochat_a_japanese_psychological_counseling_dialogue.md)**

:   提出 KokoroChat，一个通过训练有素的咨询师角色扮演收集的日语心理咨询对话数据集，包含 6,589 段长对话及详细的客户反馈评分，用于提升 LLM 的心理咨询回复生成和对话评估能力。

**[LAQuer: Localized Attribution Queries in Content-grounded Generation](others/laquer_localized_attribution.md)**

:   提出 Localized Attribution Queries (LAQuer) 任务——将生成文本中用户选定的片段精确定位到源文档的对应片段，实现比句子级归因更精细、比子句级归因更用户导向的溯源，在多文档摘要和长文本问答上显著减少了归因文本长度。

**[Length-Induced Embedding Collapse in PLM-based Models](others/length-induced_embedding_collapse_in_plm-based_models.md)**

:   发现并严格证明了 PLM 文本嵌入模型中的"长度坍缩"现象——长文本嵌入趋于聚集，源于 self-attention 作为低通滤波器随文本长度增加而滤波率增强，高频信息被过度抑制；提出 TempScale 方法通过降低 attention 温度来缓解长短文本嵌入分布差异，在 MTEB 上提升 0.94%、LongEmbed 上提升 1.10%。

**[Literature Meets Data: A Synergistic Approach to Hypothesis Generation](others/literature_meets_data_hypothesis.md)**

:   提出首个将文献驱动和数据驱动假设生成进行协同整合的方法，通过 Refinement 和 Union 两种策略让 LLM 从论文摘要和观测数据中联合生成更具泛化性的假设，在五个社会科学分类任务的 OOD 数据集上比纯数据驱动方法平均提升 3.37%，并首次通过人类实验证明 LLM 生成的假设能显著改善人类决策准确率（+7.44% / +14.19%）。

**[LoGU: Long-form Generation with Uncertainty Expressions](others/logu_longform_gen_uncertainty.md)**

:   定义"长文本不确定性生成"（LoGU）任务，识别不确定性抑制和不确定性错位两个子挑战，提出基于分解的数据构造框架和 SFT+DPO 两阶段训练流水线，使 LLM 在长文本生成中对不确定事实显式表达不确定性，在三个数据集上将 Llama3-8B 的事实准确率从 51.9% 提升到 71.6%，错误声明数从 20.4 降到 5.81。

**[Read it in Two Steps: Translating Extremely Low-Resource Languages with Code-Augmented Grammar Books](others/low_resource_translation.md)**

:   将语法书辅助的极低资源翻译（XLR MT）分解为语法规则检索和规则应用两步，并提出用代码格式表示语法规则以提升 LLM 在两步中的表现，在壮语翻译上实现了 13.1% BLEU 的提升。

**[Micro-Act: Mitigate Knowledge Conflict in QA via Actionable Self-Reasoning](others/micro_act_knowledge_conflict_reasoning.md)**

:   提出 Micro-Act 框架，通过层次化动作空间（导航/功能/桥接动作）和自适应粒度分解，让 LLM 自动感知上下文复杂度并逐层拆解知识对比，在 5 个知识冲突基准上全面超越 SOTA，同时在无冲突场景下也保持鲁棒。

**[MindRef: Mimicking Human Memory for Hierarchical Reference Retrieval with Fine-Grained Location Awareness](others/mindref_mimicking_human_memory_hierarchical_reference_retrieval.md)**

:   提出 MindRef 框架，模拟人类先回忆文档标题再定位具体段落的两阶段记忆模式，通过 Trie 和 FM-Index 约束解码让 LLM 独立召回参考段落，无需额外检索模型或预分段。

**[InterpoLL: Mitigating Shortcut Learning with InterpoLated Learning](others/mitigating_shortcut_learning_with_interpolated_learning.md)**

:   提出 InterpoLL，通过将多数样本的表征与同类少数样本的表征做插值 $z_i = (1-\lambda)f_{enc}(x_i) + \lambda f_{enc}(x_j)$ 来弱化捷径特征影响，在 MNLI/FEVER/QQP 等 NLU 任务上少数样本泛化显著超越 ERM 和 SOTA 捷径缓解方法，且不损失多数样本准确率。

**[Modular Sentence Encoders: Separating Language Specialization from Cross-Lingual Alignment](others/modular_sentence_encoders.md)**

:   本文提出模块化多语言句子编码器训练方案：先训练语言特定模块（embedding + 语言适配器 + 句子编码适配器）缓解多语言诅咒，再训练跨语言对齐适配器同时使用平行和释义数据解决不同跨语言任务间的性能权衡，在 4 个任务和 23 种语言上全面优于单体模型训练。

**[MOSAIC: Multiple Observers Spotting AI Content](others/mosaic_multiple_observers_spotting_ai_content.md)**

:   基于信息论中的通用压缩原理，提出 MOSAIC——多 LLM 集成的 AI 生成文本检测方法，通过 Blahut-Arimoto 算法为多个 detector LLM 计算最优组合权重，构建混合分布作为观察者，比较文本的实际 surprisal 与混合模型的期望交叉熵差异来判断是否为 AI 生成，在多个域/语言/生成器上鲁棒优于单模型和双模型（如 Binoculars）方法。

**[Data Quality Issues in Multilingual Speech Datasets: The Need for Sociolinguistic Awareness and Proactive Language Planning](others/multilingual_speech_data_quality.md)**

:   对三大公开多语言语音数据集（Common Voice、FLEURS、VoxPopuli）进行系统质量审计，发现低资源语言存在严重的微观和宏观质量问题，并提出基于社会语言学意识的数据集创建指南。

**[Unifying Continuous and Discrete Text Diffusion with Non-simultaneous Diffusion Processes](others/neodiff_unified_text_diffusion.md)**

:   提出 NeoDiff，通过引入"外在时间"（句子级扩散进度）和"内在时间"（token 级扩散进度）的双时间框架，利用 Poisson 过程为每个 token 独立分配细粒度噪声水平，并用上下文感知的时间预测器自适应调节去噪进度，统一了离散和连续文本扩散模型的理论框架，在机器翻译、复述、文本简化等多个任务上超越现有扩散基线。

**[Normalized AOPC: Fixing Misleading Faithfulness Metrics for Feature Attribution Explainability](others/normalized_aopc_faithfulness_metrics.md)**

:   本文揭示了广泛使用的 AOPC（扰动曲线下面积）忠实度指标在跨模型比较时会产生误导性结论（因为不同模型的 AOPC 上下界差异巨大），提出 Normalized AOPC (NAOPC) 通过 min-max 归一化消除模型间的不可比性，实验表明归一化可以根本性地改变模型忠实度排名。

**[Enhancing Automated Interpretability with Output-Centric Feature Descriptions](others/output_centric_interpretability.md)**

:   提出两种以输出为中心的特征描述方法（VocabProj 和 TokenChange），弥补现有基于输入激活的自动可解释性流程只能捕获"什么激活了特征"而忽略"特征如何影响输出"的缺陷，并证明输入+输出方法的集成能生成最忠实的特征描述。

**[Towards Better Evaluation for Generated Patent Claims](others/patclaimeval_patent_evaluation.md)**

:   提出首个专利权利要求评估基准 Patent-CE（1228 个专家标注的比较评估数据点）和专用评估方法 PatClaimEval（基于 Longformer + 对比学习变体），在特征完整性、概念清晰度、术语一致性、逻辑连接和整体质量五个维度上与人类专家评估的相关性全面超越 13 种现有指标（包括 G-Eval-4），整体质量维度的 Spearman 提升 58%。

**[Personalized Generation In Large Model Era A Survey](others/personalized_generation_in_large_model_era_a_survey.md)**

:   首篇跨模态个性化生成（PGen）综合综述，提出统一的用户中心视角将 NLP/CV/IR 社区的研究纳入一个框架，系统化梳理了文本/图像/视频/音频/3D/跨模态六大模态下的个性化技术、数据集和评估指标，涵盖 200+ 篇文献，并指出可扩展性、偏好演化、隐私公平等关键挑战。

**[All That Glitters is Not Novel: Plagiarism in AI Generated Research](others/plagiarism_ai_generated_research.md)**

:   在对自主科研 Agent（如 AI Scientist）生成的研究文档进行专家审查后发现，24% 的文档是"智能剽窃"——方法论与已有工作一一对应但不引用原始来源，且现有剽窃检测工具无法识别这种"改头换面"的抄袭。

**[Principled Understanding of Generalization for Generative Transformer Models in Arithmetic Reasoning Tasks](others/principled_generalization_arithmetic.md)**

:   建立首个统一理论框架来理解 Transformer 在算术任务（加法/乘法/模运算）上的泛化行为——从任务性质（平移不变性）和位置编码类型（APE/RPE）的交互出发，解释了之前困扰领域的多个泛化谜题（如加法能泛化但乘法不能，模100能泛化但模101不能），实验验证了理论预测。

**[PRISM: A Framework for Producing Interpretable Political Bias Embeddings](others/prism_political_bias_embeddings.md)**

:   提出PRISM框架——首个专为政治偏见嵌入设计的方法，通过争议性话题偏见指标挖掘和Cross-Encoder政治偏见评分两阶段，生成可解释的偏见感知嵌入，在政治偏见分类上超越通用文本嵌入模型。

**[Substance over Style: Evaluating Proactive Conversational Coaching Agents](others/proactive_conversational_coaching.md)**

:   通过健康教练领域的专家访谈和用户研究（31 名参与者、155 段对话），系统评估了五种不同对话风格（Directive、Interrogative、Facilitative）的 LLM 教练 Agent，发现用户高度重视核心功能性（substance）而对缺乏功能性时的风格修饰（style）持负面态度，同时揭示了用户第一人称评价与专家/LLM 第三方评价之间的显著不一致。

**[ProxAnn: Use-Oriented Evaluations of Topic Models and Document Clustering](others/proxann_topic_model_eval.md)**

:   ProxAnn 设计了一个面向实际使用场景的主题模型/文档聚类评估协议——标注者先从模型输出推断类别再将类别应用到新文档——并证明 LLM 代理标注者可以统计不可分辨地替代人类标注者，同时发现经典 LDA 在此评估下表现不弱于现代方法。

**[Uncovering Visual-Semantic Psycholinguistic Properties from the Distributional Structure of Text Embedding Space](others/psycholinguistic_visual_semantic.md)**

:   提出 Neighborhood Stability Measure (NSM)，仅通过文本嵌入空间中邻域的"尖锐度"来无监督估计词汇的可意象性(imageability)和具体性(concreteness)，无需视觉模态或生成模型即可超越现有方法。

**[PVP: An Image Dataset for Personalized Visual Persuasion with Persuasion Strategies, Viewer Characteristics, and Persuasiveness Ratings](others/pvp_an_image_dataset_for_personalized.md)**

:   构建了首个大规模个性化视觉说服数据集 PVP（28,454 张图像、596 条消息、9 种说服策略、2,521 位标注者的心理特征），并提出说服图像生成和评估两个任务，发现融入观看者的心理特征能显著提升说服图像的生成和评估效果。

**[Quantifying Lexical Semantic Shift via Unbalanced Optimal Transport](others/quantifying_lexical_semantic_shift_via_unbalanced_optimal_transport.md)**

:   将 Unbalanced OT 应用于上下文化词嵌入来量化词义变迁，提出 Sense Usage Shift (SUS) 指标在每个用法实例级别量化语义变化（义项使用频率增减），统一解决实例级变化检测、词级变化幅度量化、词义扩展/缩小判定。

**[ReflectDiffu: Reflect between Emotion-intent Contagion and Mimicry for Empathetic Response Generation via a RL-Diffusion Framework](others/reflectdiffu_empathetic_response.md)**

:   提出轻量级共情对话框架 ReflectDiffu，融合情感传染（捕捉情绪）、意图二次机制（Exploring-Sampling-Correcting将情绪映射为行动意图）和扩散模型生成，在相关性、可控性和信息量上全面超越现有基线和 Llama-3.1-8B。

**[Re-identification of De-identified Documents with Autoregressive Infilling](others/reidentification_deidentified.md)**

:   提出一种基于 RAG 的再识别攻击方法，通过稀疏检索+稠密检索+自回归填充的三阶段流程，对去标识化文档进行逆向攻击，在三个数据集上高达 80% 的被遮蔽文本片段可被成功恢复，揭示了当前去标识化方法的脆弱性。

**[Identifying Reliable Evaluation Metrics for Scientific Text Revision](others/reliable_eval_metrics_scientific.md)**

:   系统分析了传统相似度指标（ROUGE、BERTScore 等）在科学文本修订评估中的局限性，发现它们强相关于编辑距离且惩罚深度修改，并证明结合 LLM-as-Judge 和任务特定指标的混合方法最能对齐人类评判。

**[REP: Keys to Robust Edits — From Theoretical Insights to Practical Advances](others/rep_robust_knowledge_editing.md)**

:   揭示locate-and-edit知识编辑方法中语义键的根本缺陷——内部表示无法同时满足鲁棒性和特异性，提出REP模块通过对比学习解耦编辑键，在鲁棒性测试上提升最高66.4%。

**[Retrieve to Explain: Evidence-driven Predictions for Explainable Drug Target Identification](others/retrieve_to_explain_drug_target_identification.md)**

:   提出 R2E (Retrieve to Explain)，一种基于检索的架构，通过从文献语料库中检索证据来评分和排序所有候选答案，并利用 Shapley 值将预测忠实地归因到支撑证据，在药物靶点识别任务上超越了遗传学基线和 GPT-4 基线。

**[Revisiting Weak-to-Strong Generalization: Reverse KL vs. Forward KL](others/revisiting_weak-to-strong_generalization_in_theory_and_practice_reverse_kl_vs_fo.md)**

:   在 Weak-to-Strong Generalization (W2SG) 框架中，提出用 reverse KL 替代 forward KL 作为损失函数——理论证明 reverse KL 的 mode-seeking 特性可保证强模型超过弱监督者至少"分歧量"的幅度，实验在 GPT-2/Pythia/Qwen2.5 系列上验证 reverse KL/CE 在 12/12 设置中超越 forward KL 且噪声鲁棒性更好。

**[Rubrik's Cube: Testing a New Rubric for Evaluating Explanations on the CUBE Dataset](others/rubriks_cube_testing_a_new_rubric_for_evaluating_explanations_on_the_cube_datase.md)**

:   提出 Rubrik——受教育评估启发的解释质量评价量规，基于三层嵌套类型体系（Commentary⊆Justification⊆Argument）+ 8 维质量维度，配套 CUBE 数据集（26K 条由人类和 6 个 LLM 生成的解释），发现 LLM 解释低质主因是缺乏简洁性而非连贯性。

**[SCAR: Data Selection via Style Consistency-Aware Response Ranking for Efficient Instruction-Tuning](others/scar_style_consistency_data_selection.md)**

:   SCAR 识别出回复的"语言形式"和"指令惊奇度"是影响 LLM 指令微调效果的两个关键风格因素，并提出基于风格一致性的排序方法自动选择高质量训练数据，仅用 0.7% 的原始数据就能让微调后的 LLM 匹配甚至超越全数据集训练的性能。

**[SEOE: A Scalable and Reliable Semantic Evaluation Framework for Open Domain Event Detection](others/seoe_semantic_eval.md)**

:   针对开放域事件检测（ODED）评估的两大痛点——有限 benchmark 缺乏真实世界代表性、token 级匹配指标无法捕捉语义相似性——提出 SEOE 框架，构建包含 564 种事件类型覆盖 7 大领域的可扩展 benchmark，并引入基于 LLM 的语义 F1 评估指标。

**[Shifting from Ranking to Set Selection for Retrieval Augmented Generation](others/setr_set_selection_rag.md)**

:   提出从"逐个排序"到"集合选择"的 RAG 检索范式转换——SetR 通过 CoT 推理显式识别查询的信息需求，然后选择一组能共同满足这些需求的段落（而非逐个评分最相关的），在多跳 RAG 基准上超越 GPT-4o 级别的重排器。

**[Share Text To Sql Correction](others/share_text_to_sql_correction.md)**

:   提出 SHARE 框架，通过三个专用小语言模型（SLM）组成的顺序管道，将 SQL 查询转换为逐步动作轨迹并分别修正 schema 错误和逻辑错误，从而以低成本高效辅助 LLM 进行 Text-to-SQL 自纠正。

**[ShifCon: Enhancing Non-Dominant Language Capabilities with a Shift-based Multilingual Contrastive Framework](others/shifcon_nondominant_language.md)**

:   提出 ShifCon 框架，通过将非优势语言的表示 shift 到优势语言子空间以获取更丰富的模型知识，再 shift 回原语言子空间进行生成，结合多语言对比学习，显著提升低资源语言的表现。

**[Statistical Deficiency for Task Inclusion Estimation](others/statistical_deficiency_task_inclusion.md)**

:   基于统计缺陷性（statistical deficiency）理论，提出一种理论驱动的任务包含关系（task inclusion）定义与度量框架，以信息充分性（information sufficiency, IS）作为可计算代理指标，通过比较微调模型的中间层表征来估计任务间的包含程度，并在合成数据和真实NLP任务上成功重建了经典NLP pipeline的层次关系。

**[STRICTA: Structured Reasoning in Critical Text Assessment for Peer Review and Beyond](others/stricta_structured_reasoning_in_critical_text_assessment_for_peer_review_and_bey.md)**

:   提出 STRICTA 框架，基于结构因果模型（SCM）将文本评审建模为显式的逐步推理图（workflow），在生物医学论文评审中收集 40+ 位专家的 4000+ 推理步骤数据集，发现先验知识差异是专家分歧主因、写作风格对最终评审有因果影响，LLM 存在错误传播但人类监督可有效缓解。

**[SudoLM: Learning Access Control of Parametric Knowledge with Authorization Alignment](others/sudolm_authorization_alignment.md)**

:   SudoLM 提出了一种 LLM 参数化知识访问控制框架，通过"SUDO key"机制让授权用户解锁受限知识（如医学领域知识），未授权用户则只能访问公开知识，用 DPO 的 authorization alignment 在一个模型内实现了传统需要多版本模型才能完成的分级访问控制。

**[The Harmonic Structure of Information Contours](others/the_harmonic_structure_of_information_contours.md)**

:   提出 Harmonic Surprisal (HS) 假说——文本中 surprisal 曲线呈周期性波动且周期与语篇结构（EDU/句子/段落）对齐，用带时间缩放的谐波回归检验，在 6 种语言上发现一致的周期模式，精化了经典的 Uniform Information Density 假说。

**[Learning to Reason Over Time: Timeline Self-Reflection for Temporal Reasoning](others/tiser_timeline_self_reflection_temporal.md)**

:   提出 TISER 框架，通过"推理→时间线构建→自反思→答案生成"四阶段管道实现LLM时间推理的test-time scaling，配合合成推理轨迹数据微调，让 7B 开源模型在多个时间推理基准上超越 GPT-4，在TGQA等任务上达到 SOTA。

**[TokAlign: Efficient Vocabulary Adaptation via Token Alignment](others/tokalign_vocab_adaptation.md)**

:   提出 TokAlign，基于 Token 共现信息学习两个词表之间的一对一映射矩阵，高效替换 LLM 的词表，实现跨语言知识迁移和跨模型 token 级蒸馏。

**[Did Translation Models Get More Robust Without Anyone Even Noticing?](others/translation_robustness.md)**

:   通过合成噪声和社交媒体文本的系统性实验，证明现代大规模预训练翻译模型（LLM）在未经任何专门鲁棒性训练的情况下，对多种输入噪声的鲁棒性已远超传统 NMT 模型，鲁棒性随模型规模增长自然提升。

**[TROVE: A Challenge for Fine-Grained Text Provenance via Source Sentence Tracing and Relationship Classification](others/trove_a_challenge_for_finegrained_text.md)**

:   提出TROVE文本溯源挑战，将目标文本中每个句子追溯到源文档中的具体源句，并分类其细粒度关系（引用、压缩、推理等），覆盖多文档和长文档场景。

**[Tuna: Comprehensive Fine-grained Temporal Understanding Evaluation on Dense Dynamic Videos](others/tuna_temporal_understanding.md)**

:   本文提出 Tuna，一个面向密集动态视频的细粒度时序理解基准，包含 1000 个精心标注的视频和两个互补任务（Tuna-cap 字幕生成和 Tuna-mcq 视频问答），覆盖相机状态、场景、动作、属性四大动态要素，评估 21 个模型后发现即使 GPT-4o 也仅达 58.5% F1，揭示了当前模型在时序理解上的重大不足。

**[USDC: A Dataset of User Stance and Dogmatism in Long Conversations](others/usdc_a_dataset_of_underlineuser_underlinestance_and_underlinedogmatism_in_long_u.md)**

:   构建 USDC——首个用户级长对话立场和教条主义数据集，764 个多用户 Reddit 对话（22 子版块），用 {Mistral Large, GPT-4} × {zero/one/few-shot} 共 6 设置多数投票标注立场(5级)+教条程度(4级)，并用 7 个 SLM 微调/指令微调建立基线。

**[VITAL: A New Dataset for Benchmarking Pluralistic Alignment in Healthcare](others/vital_pluralistic_alignment_healthcare.md)**

:   本文构建了首个面向医疗健康领域的多元化对齐（pluralistic alignment）基准数据集 VITAL，包含 13.1K 价值观情境和 5.4K 多选题，并通过对 8 个 LLM 的广泛评估表明，现有多元化对齐技术（尤其是 ModPlural）在医疗场景下表现不佳，简单的 prompting 反而效果更好。

**[When to Speak, When to Abstain: Contrastive Decoding with Abstention](others/when_to_speak_when_to_abstain.md)**

:   提出 CDA（Contrastive Decoding with Abstention），一种免训练解码方法，通过熵校准的不确定性估计让 LLM 在参数/上下文知识可用时生成正确回答、在两者都不可靠时主动弃权，覆盖全部四种知识可用性场景。

**[X-Turing: Towards an Enhanced and Efficient Turing Test for Long-Term Dialogue Agents](others/xturing_enhanced_turing_test.md)**

:   提出 X-Turing 框架，通过引入 burst 对话模式和伪对话生成技术来增强和高效化图灵测试，能够评估 LLM 在长期对话中的人类模仿能力，发现 LLM 随着对话轮次增加表现显著下降。
