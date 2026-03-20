---
search:
  exclude: true
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 💬 LLM / NLP

**💬 ACL2025** · 共 **141** 篇

**[A Survey on Efficient Large Language Model Training: From Data-centric Perspectives](a_survey_on_efficient_large_language.md)**

:   首个系统性的数据高效 LLM 后训练综述，提出"数据价值飞轮"分类法，将方法分为五大类（数据选择、质量增强、合成生成、蒸馏压缩、自演进生态），覆盖 100+ 篇代表性工作并展望未来方向。

**[AbGen: Evaluating Large Language Models in Ablation Study Design and Evaluation for Scientific Research](abgen_evaluating_large_language_models_in.md)**

:   提出 AbGen——首个评估 LLM 设计消融实验能力的基准（1500 条专家标注数据来自 807 篇 NLP 论文），发现最强 LLM (DeepSeek-R1) 与人类专家差距 14.4%，且 LLM-as-Judge 评分与人类评估严重不一致。

**[AIMSCheck: Leveraging LLMs for AI-Assisted Review of Modern Slavery Statements Across Jurisdictions](aimscheck_modern_slavery.md)**

:   提出 AIMSCheck——使用 LLM 辅助审查企业现代奴隶制声明是否合规的端到端框架，构建英国和加拿大的新标注数据集（AIMS.uk/AIMS.ca），三层分解合规评估增强可解释性，在澳大利亚数据上训练的模型能有效跨司法管辖泛化到英国和加拿大。

**[Algorithmic Fidelity of Large Language Models in Generating Synthetic German Public Opinions: A Case Study](algorithmic_fidelity_german_opinion.md)**

:   使用德国纵向选举研究（GLES）的开放式调查数据，评估 LLM 在生成反映德国亚群体公共舆论方面的"算法保真度"，发现 Llama2 在建模群体意见方面优于其他 LLM，但对左翼政党支持者的表征好于右翼（如 AfD），且提示中包含更多人口统计变量可改善表现。

**[Revisiting Common Assumptions about Arabic Dialects in NLP](arabic_dialects_assumptions_revisited.md)**

:   系统验证了阿拉伯语 NLP 中四个被广泛接受的假设，通过 978 个方言句子+33 名标注者的多标签标注数据集证明：56% 的方言句子在多个区域方言中有效，方言词表的区分度被高估，句子长度与方言歧义性的相关性远弱于方言化程度（ALDi），不同方言说话者对同一句子的 ALDi 评级差异显著。

**[Automating Legal Interpretation with LLMs: Retrieval, Generation, and Evaluation](atrie_legal_interpretation.md)**

:   提出 ATRIE 框架模拟法学教义研究流程——自动从判例法中检索概念相关信息、解释法律概念、并通过下游任务（法律概念蕴涵LCE）自动评估解释质量，生成的解释在全面性和可读性上与专家相当。

**[AutoGUI: Scaling GUI Grounding with Automatic Functionality Annotations from LLMs](autogui_scaling_gui_grounding_with_automatic.md)**

:   提出 AutoGUI 自动标注管线——通过模拟交互比较 UI 状态变化 + LLM 推断元素功能 + LLM 验证过滤，构建 704K 高质量 UI 功能标注数据集，标注正确率 96.7% 可比人类，显著提升 VLM 的 UI grounding 能力且展现数据扩展效应。

**[Binary Classifier Optimization for Large Language Model Alignment](bco_binary_classifier_alignment.md)**

:   提出 BCO（Binary Classifier Optimization），从数学上证明二元交叉熵损失是 DPO 损失的上界，使 LLM 对齐仅需"点赞/踩"二元反馈而非成对偏好数据，并通过新颖的 reward shift 技术收紧上界，在配对偏好数据集上与 DPO 持平，在真实 Likert-5 标注数据上优于 DPO 和 KTO。

**[Between Circuits and Chomsky: Pre-pretraining on Formal Languages Imparts Linguistic Biases](between_circuits_chomsky.md)**

:   提出在自然语言预训练前先在形式语言上进行"pre-pretraining"，发现具有层级依赖结构的形式语言（如 k-Shuffle Dyck）能为 Transformer 提供有效的归纳偏置，使 1B 参数模型以 33% 更少的 token 达到相同的语言建模损失。

**[Bias in Language Models: Beyond Trick Tests and Towards RUTEd Evaluation](bias_in_language_models_beyond_trick_tests_ruted_evaluation.md)**

:   通过对比标准偏见基准（"trick tests"）与基于真实使用场景的 RUTEd 评估，发现标准偏见基准与真实场景中的偏见表现无显著相关性，主张偏见评估应面向具体应用场景。

**[Can Large Language Models Understand Internet Buzzwords Through User-Generated Content](buzzword_understanding_ugc.md)**

:   研究 LLM 能否通过用户生成内容（UGC）理解中文网络流行语——构建首个中文网络流行语数据集 Cheer（含定义和相关UGC），提出 Ress 方法引导 LLM 模拟人类语言学习过程来生成流行语定义，揭示了 LLM 在流行语理解上的三大共性挑战。

**[Automated CAD Modeling Sequence Generation from Text Descriptions via Transformer-Based Large Language Models](cadllm_cad_modeling_from_text.md)**

:   提出一个语言引导的 CAD 设计自动化框架——通过半自动数据标注流水线、Transformer CAD 生成器（TCADGen）和 LLM 增强模型（CADLLM）三个创新，从文本参数和外观描述自动生成 CAD 建模序列，在精度和效率上超越传统方法。

**[CER: Confidence Enhanced Reasoning in LLMs](cer_confidence_enhanced_reasoning.md)**

:   提出置信度增强推理框架 CER——在 CoT 推理的每个中间步骤中量化关键 token（数学任务的数值/开放域的专有名词）的置信度，用步间置信度乘积评估整条推理链的可靠性，用置信度加权聚合替代简单多数投票，在数学和开放域任务上比 self-consistency 分别提升最高 7.4% 和 5.8%。

**[ChainEdit: Propagating Ripple Effects in LLM Knowledge Editing through Logical Rule-Guided Chains](chainedit_propagating_ripple_effects_in_llm.md)**

:   提出 ChainEdit 框架，通过将知识图谱中挖掘的逻辑规则与 LLM 内在逻辑推理能力对齐，实现知识编辑时的链式更新，将逻辑泛化准确率从约 20% 提升至 58-65%。

**[Enhancing Character-Level Understanding in LLMs through Token Internal Structure Learning](character_level_understanding.md)**

:   提出 TIPA（Token Internal Position Awareness）方法，通过在 tokenizer 词汇表上进行逆序字符预测训练，增强 LLM 对 token 内部字符结构和位置的感知能力，显著提升中文拼写纠错等字符级任务的表现。

**[Circuit Compositions: Exploring Modular Structures in Transformer-Based Language Models](circuit_compositions_modular_structures.md)**

:   通过在 PCFG SET 数据集上识别 10 个组合性字符串编辑操作的电路（circuits），研究 Transformer 中功能相关电路之间的模块化关系，发现功能相似的电路具有显著的节点重叠和跨任务忠实度，且电路可以通过集合运算（并集）组合以表示超出单个电路能力的更复杂功能。

**[CKnowEdit: A New Chinese Knowledge Editing Dataset for Linguistics, Facts, and Logic Error Correction in LLMs](cknowedit_chinese_knowledge_editing_dataset_llms.md)**

:   提出首个面向中文语言特征的知识编辑数据集 CKnowEdit，涵盖语言学、事实性和逻辑性三大类共10个子类的1,854条样本，揭示了当前知识编辑方法在中文场景下的不足。

**[Classifying Unreliable Narrators with Large Language Models](classifying_unreliable_narrators.md)**

:   借用叙事学理论定义三种不可靠叙事者类型，构建专家标注数据集 TUNa，系统评估 LLM 在零样本、少样本、微调和课程学习设定下的分类能力，发现该任务极具挑战性且课程学习对小模型有显著提升。

**[CogniBench: A Legal-inspired Framework and Dataset for Assessing Cognitive Faithfulness of Large Language Models](cognibench_cognitive_faithfulness.md)**

:   借鉴法律领域间接证据认定标准，提出分层评估框架和 CogniBench 数据集，首次系统性地定义和评估 LLM 在认知性陈述（推理、评价、解释）中的忠实度问题，并训练 CogniDet 检测器实现事实与认知幻觉的同时检测。

**[Enough Coin Flips Can Make LLMs Act Bayesian](coin_flips_bayesian.md)**

:   通过受控的有偏硬币抛掷实验，证明 LLM 在获得足够的上下文示例后能以贝叶斯方式更新其先验，但初始先验通常存在系统性偏差（偏向正面），且注意力幅度对贝叶斯推理影响甚微。

**[Revisiting Compositional Generalization Capability of Large Language Models Considering Instruction Following Ability](compositional_generalization_instruction.md)**

:   提出 Ordered CommonGen 基准，通过要求 LLM 按指定顺序生成包含所有概念的句子，同时评估组合泛化与指令遵循能力，在 36 个 LLM 上发现即使最强模型也仅能达到约 75% 的有序覆盖率。

**[How Humans and LLMs Organize Conceptual Knowledge: Exploring Subordinate Categories in Italian](conceptual_knowledge_org.md)**

:   通过构建首个意大利语下位类别心理语言学数据集（187 个基本类别），系统对比了人类和 LLM 在下位概念层级上的类别组织结构，发现两者的对齐度较低但在不同语义领域存在显著差异。

**[ConsistencyChecker: Tree-based Evaluation of LLM Generalization Capabilities](consistencychecker_tree_evaluation.md)**

:   ConsistencyChecker 提出基于树状结构的 LLM 泛化能力评估框架，通过可逆变换的往返一致性（如英语→西班牙语→英语）测量模型在多步变换中的语义/功能保持能力，动态生成 benchmark 避免数据泄露，与 WMT 2024 排名相关性 r > 0.7。

**[Contrastive Perplexity for Controlled Generation: An Application in Detoxifying Large Language Models](contrastive_perplexity_controlled_gen.md)**

:   提出基于原型对比困惑度（Contrastive Perplexity, CP）的框架，通过构造语义相似但毒性属性不同的正负样本对，在困惑度空间中进行对比学习来微调 LLM，实现显著的毒性降低（Mistral-7b 毒性从 33.1% 降至 4.3%）且几乎不影响下游任务性能。

**[Contrastive Prompting Enhances Sentence Embeddings in LLMs through Inference-Time Steering](contrastive_prompting_embeddings.md)**

:   提出对比提示（Contrastive Prompting, CP）方法，通过引入辅助提示（引导编码非核心信息）并在推理时与正常提示的激活值做对比减法，过滤掉停用词等无关语义，使 LLM 的句子嵌入更聚焦核心语义，在 STS 和分类任务上一致提升现有提示方法。

**[Cross-model Transferability among Large Language Models on the Platonic Representations of Concepts](cross_model_transferability_sv.md)**

:   提出 L-Cross Modulation 方法，通过简单线性变换将一个 LLM 的概念方向向量（steering vectors）迁移到另一个 LLM 中实现行为控制，发现三个关键结论：(1) 跨模型 SV 迁移有效；(2) 不同概念共享同一变换矩阵；(3) 小模型的 SV 可以控制大模型（弱到强迁移）。

**[Cross-Lingual Pitfalls: Automatic Probing Cross-Lingual Weakness of Multilingual Large Language Models](crosslingual_pitfalls.md)**

:   提出基于束搜索和 LLM 仿真的自动化方法来高效发现多语言 LLM 的跨语言弱点，构建了覆盖 16 种语言的 6000+ 双语问答对数据集，揭示即使 GPT-4o 也存在超过 30% 的跨语言性能下降。

**[Data Whisperer: Efficient Data Selection for Task-Specific LLM Fine-Tuning via Few-Shot In-Context Learning](data_whisperer_data_selection.md)**

:   Data Whisperer 提出一种无需训练的注意力加权 few-shot ICL 数据选择方法，利用预训练模型自身的 ICL 能力和注意力分数为训练样本打分，仅用 10% 数据即可超越全量微调性能，同时比现有方法快 7-20 倍。

**[DeAL: Decoding-time Alignment for Large Language Models](deal_decoding_time_alignment.md)**

:   DeAL 将 LLM 对齐问题重新形式化为解码时的启发式搜索问题，在推理阶段利用可定制的奖励函数（包括程序化约束和参数化 reward model）引导 token 选择，实现了灵活的多目标对齐且可与 RLHF 互补叠加。

**[Defense Against Prompt Injection Attack by Leveraging Attack Techniques](defense_prompt_injection.md)**

:   本文提出一种"以攻为防"的 prompt injection 防御策略：将已有的攻击技术（ignore、escape、fake completion）反转用于防御，在被注入的数据内容后追加 shield prompt + 原始指令，使 LLM 忽略注入指令而执行原始指令，在多种攻击场景下将 ASR 降至接近零。

**[When People are Floods: Analyzing Dehumanizing Metaphors in Immigration Discourse with Large Language Models](dehumanizing_metaphors_immigration.md)**

:   提出结合 LLM 词级隐喻检测与 SBERT 篇章级语义关联的计算框架，在 40 万条美国移民推文上揭示保守派更多使用去人化隐喻、但生物类隐喻对自由派的用户互动效应更强的复杂图景。

**[Deontological Keyword Bias: The Impact of Modal Expressions on Normative Judgments of Language Models](deontological_keyword_bias.md)**

:   本文揭示LLM存在"义务论关键词偏见"(DKB)——当提示中包含"must"、"ought to"等模态义务表达时，模型会将超过90%的常识场景误判为义务，并提出基于少样本示例与推理提示的去偏策略。

**[Refuse Whenever You Feel Unsafe: Improving Safety in LLMs via Decoupled Refusal Training](derta_decoupled_refusal.md)**

:   发现标准安全微调数据存在"拒绝位置偏差"——模型只学会在回答开头拒绝，中途发现不安全时无法中断。提出 DeRTa（Decoupled Refusal Training），通过"有害前缀+安全拒绝"的 MLE 训练和在每个位置模拟"从有害到安全"转换的 RTO 训练，让 LLM 能在回答的任何位置感知到不安全时拒绝，在六种攻击场景下超越 GPT-4 和 LLaMA3-Instruct。

**[Disentangling Language and Culture for Evaluating Multilingual Large Language Models](disentangle_language_culture.md)**

:   提出 Dual Evaluation Framework，将多语言 LLM 评估沿"语言媒介"和"文化语境"两个维度解耦，发现"文化-语言协同"(Cultural-Linguistic Synergy) 现象——模型在文化语境与提问语言对齐时表现更好，并通过 FFN 神经元激活分析从可解释性角度给出解释。

**[Disentangling Memory and Reasoning Ability in Large Language Models](disentangle_memory_reasoning.md)**

:   提出将 LLM 的推理过程显式分解为"记忆回忆"和"逻辑推理"两个步骤——引入 `<memory>` 和 `<reason>` 两个可学习特殊 token 标记每步是知识回忆还是逻辑推理，用双 LLM 框架生成训练数据后 LoRA 微调，在 StrategyQA/CommonsenseQA/TruthfulQA 上提升性能并增强可解释性，8B 模型在 TruthfulQA 上超越 GPT-4o。

**[DIVE into MoE: Diversity-Enhanced Reconstruction of Large Language Models from Dense into Mixture-of-Experts](dive_moe_reconstruction.md)**

:   提出 DIVE，一种将 Dense LLM 重构为 MoE 架构的方法，核心洞察是不同领域的校准数据集会让结构化剪枝产生不同的剪枝结果，利用这种多样性构建领域特异的专家，配合高效的两阶段重训练（router dense训练 + expert LoRA稀疏训练），在仅调不到 1% 参数的情况下实现优于现有剪枝和 MoE 重构方法的效果。

**[Diversity-oriented Data Augmentation with Large Language Models](diversity_data_augmentation.md)**

:   提出 DoAug 框架，通过 SFT+DPO 微调 LLM 释义器并结合核心集选择与多样性采样，在保持语义一致性的同时显著提升增强数据集的多样性，在 12 个数据集上平均性能提升 10.52%，超出次优基线 3.76 个百分点。

**[ECLM: Entity Level Language Model for Spoken Language Understanding with Chain of Intent](eclm_entity_level_language_model_spoken_language_understanding.md)**

:   提出 ECLM 框架，将 LLM 应用于多意图口语理解：通过将 token 级槽填充转化为实体识别任务解决序列对齐问题，引入"意图链"（Chain of Intent）实现逐步多意图识别，在 MixATIS 和 MixSNIPS 上大幅超越 SOTA 基线。

**[EdiText: Controllable Coarse-to-Fine Text Editing with Diffusion Language Models](editext_diffusion_text_editing.md)**

:   提出 EdiText，基于嵌入扩散模型（LD4LG）的可控文本编辑框架，通过将 SDEdit 技术从图像域迁移到文本域实现粗粒度编辑（控制加噪时间步），并创新性地利用自条件化（self-conditioning）机制实现细粒度编辑（将参考文本嵌入注入为去噪条件），两者结合实现从粗到细的多粒度文本属性编辑。

**[ELABORATION: A Comprehensive Benchmark on Human-LLM Competitive Programming](elaboration_competitive_programming.md)**

:   提出ELABORATION——首个全面评估人类-LLM协作竞赛编程的基准，包含覆盖编程全流程（理解→规划→编码→调试）的人类反馈分类体系和8320题精标注数据集，实验表明LLM在困难题上仅3.4% Pass@1，但人类反馈（特别是在编码阶段）可平均提升9.3%。

**[Emergent Abilities of Large Language Models under Continued Pretraining for Language Adaptation](emergent_abilities_continued_pt.md)**

:   揭示了持续预训练（CPT）进行语言适应时，混入英文数据对保留模型上下文学习（ICL）能力和下游涌现能力至关重要——尽管不影响验证困惑度；并提出课程学习和 EMA 权重平均作为替代方案。

**[Growing Through Experience: Scaling Episodic Grounding in Language Models](episodic_grounding_experience.md)**

:   提出一个 weak-to-strong episodic grounding 框架，利用 MCTS 收集结构化经验数据，通过行为比率蒸馏将小模型的 episodic grounding 能力迁移到大模型，结合 DPO 优化实现从成功和失败经验中学习，在物理规划任务上超越 GPT-4o 等 SOTA 模型 3.45%。

**[Efficient and Accurate Prompt Optimization: the Benefit of Memory in Exemplar-Guided Reflection](erm_prompt_optimization_memory.md)**

:   提出 ERM 方法，通过指导性元提示生成带详细解题过程的 exemplar 来增强 feedback 质量，并引入 Feedback Memory 和 Exemplar Factory 两种长期记忆机制来高效存储和复用历史反馈与示例，在多个任务上以约一半的优化步数超越了 SOTA prompt 优化方法。

**[EscapeBench: Towards Advancing Creative Intelligence of Language Model Agents](escapebench_creative_agent.md)**

:   提出EscapeBench密室逃脱游戏基准（36个场景、3种难度）评估LM agent的创造性智能，并设计EscapeAgent通过Foresight（工具使用假设生成）和Reflection（未解任务追踪）模块将提示依赖降低约50%，但仍远落后于人类。

**[Evaluating Language Models as Synthetic Data Generators](evaluating_lms_synthetic_data_gen.md)**

:   提出 AgoraBench 基准，系统评估 6 个 LLM 在 3 个领域×3 种数据生成方式下的数据生成能力，通过训练 99 个学生模型发现：LLM 的数据生成能力与问题求解能力不直接相关，GPT-4o 在实例生成上最强而 Claude-3.5-Sonnet 在质量增强上最强。

**[LLMs Can Simulate Standardized Patients via Agent Coevolution](evopatient_standardized_patient.md)**

:   EvoPatient 提出了一个多智能体协同进化框架，通过患者 Agent 和医生 Agent 之间的自主模拟对话，让 LLM 无需人工监督即可学会模拟标准化病人（SP），在需求对齐度上超过现有推理方法 10%+。

**[HiCUPID: Exploring the Potential of LLMs as Personalized Assistants](exploring_the_potential_of_llms_as.md)**

:   提出 HiCUPID，首个全面满足个性化 AI 助手五大需求（用户信息遵循、隐含信息理解、多信息推理、长上下文建模、主动性回复）的基准，含 1,250 用户 × 25 人格 × 10 日程 + Llama-3.2 自动评估模型。

**[A Survey on Foundation Language Models for Single-cell Biology](foundation_lm_single_cell_survey.md)**

:   首篇从语言建模视角系统综述单细胞生物学基础语言模型的工作，将现有模型划分为 PLM（从头预训练）和 LLM（利用已有大模型）两大类，全面分析了数据 tokenization 策略、预训练/微调范式以及下游任务。

**[FR-Spec: Accelerating Large-Vocabulary Language Models via Frequency-Ranked Speculative Sampling](fr_spec_speculative_sampling.md)**

:   发现大词表LLM（如LLaMA-3的128k词表）中投机采样的瓶颈从Transformer层转移到LM Head，提出FR-Spec通过频率排序将草稿模型的词表压缩75%（128k→32k），在EAGLE-2基础上额外获得1.12×加速，且保证最终输出分布数学等价。

**[From Selection To Generation A Survey](from_selection_to_generation_a_survey.md)**

:   首篇系统综述 LLM 时代的主动学习（Active Learning），提出以 Querying（选择/生成）和 Annotation（标注）为核心的分类体系，全面梳理 LLM 如何变革传统主动学习的选择-标注流程。

**[GALLa: Graph Aligned Large Language Models for Improved Source Code Understanding](galla_graph_aligned_large_language_models.md)**

:   提出 GALLa，通过 GNN 编码代码的 AST/DFG 结构图并用跨模态适配器对齐到 LLM 嵌入空间，在微调时作为辅助任务注入代码结构信息，推理时丢弃 GNN 和 adapter 实现零额外开销，在 5 个代码任务 × 7 个基线 LLM（350M-14B）上持续提升。

**[Gradient-Adaptive Policy Optimization: Towards Multi-Objective Alignment of Large Language Models](gapo_multi_objective_alignment.md)**

:   提出GAPO，一种基于梯度自适应缩放的多目标策略优化方法，利用多梯度下降算法(MGDA)结合梯度归一化，平衡LLM在帮助性和无害性等冲突目标间的权衡，并通过P-GAPO支持用户偏好驱动的Pareto前沿生成。

**[GAPO: Learning Preferential Prompt through Generative Adversarial Policy Optimization](gapo_preferential_prompt.md)**

:   提出 GAPO（Generative Adversarial Policy Optimization）框架，将 GAN 的对抗训练机制与 PPO 结合，使用 encoder-only 奖励模型替代传统 decoder-only 架构，通过"Preferential Prompt"（修改 prompt 中的约束而非 response）的新范式来增强 LLM 对细粒度约束的理解和遵循能力，在 IFEval 和产品描述生成任务上大幅超越 DPO/KTO/SimPO 等基线。

**[Generative Psycho-Lexical Approach for Constructing Value Systems in Large Language Models](generative_psycholexical_approach_for_constructing_value.md)**

:   提出生成式心理词汇方法（GPLA），自动化构建面向LLM的五因素价值体系（社会责任、冒险性、规则遵循、自我效能、理性），在结构效度、安全预测和价值对齐上优于经典Schwartz人类价值体系。

**[Towards Geo-Culturally Grounded LLM Generations](geocultural_grounded_llm.md)**

:   研究 RAG/搜索增强技术对 LLM 文化意识的影响——搜索增强显著提升了文化命题知识的选择题表现，但也增加了刻板印象风险，且在开放式文化流畅性的人工评估中改进不显著，揭示了"文化知识"和"文化流畅性"的本质区别。

**[Geometric Signatures of Compositionality Across a Language Model's Lifetime](geometric_compositionality_lifetime.md)**

:   通过将数据集的组合性程度与语言模型表示的非线性内在维度(I_d)和线性有效维度(d)联系起来，揭示了一个形式-意义二分：非线性 I_d 编码有意义的组合语义复杂度，而线性 d 编码表面词形复杂度；该对应关系在训练过程中随语言能力涌现而建立。

**[Efficient Universal Goal Hijacking with Semantics-guided Prompt Organization](goal_hijacking_attack.md)**

:   本文提出POUGH方法，通过高效的渐进式优化算法和两种语义引导的提示组织策略（采样策略+排序策略），实现了对LLM的高效通用目标劫持攻击，在四个开源LLM和十种恶意目标响应上平均攻击成功率达93.41%。

**[What Makes a Good Natural Language Prompt?](good_natural_language_prompt.md)**

:   通过元分析150+篇prompting相关论文和博客，提出一个以属性为中心、以人为中心的prompt质量评估框架，涵盖6个维度21个属性，并发现单属性增强往往比多属性组合更有效。

**[GORP: Continual Gradient Low-Rank Projection Fine-Tuning for LLMs](gorp_continual_gradient_projection.md)**

:   GORP 提出将全秩参数和 LoRA 低秩参数的梯度统一投影到低秩梯度子空间中联合更新，利用 Adam 一阶矩隐式构建跨任务共享梯度空间来缓解灾难性遗忘，在 T5 和 LLaMA2 上持续学习性能接近多任务联合训练上界。

**[Can Graph Descriptive Order Affect Solving Graph Problems with LLMs?](graph_descriptive_order_llm.md)**

:   首次系统研究了图描述顺序（BFS、DFS、PageRank、PPR）对LLM解决图推理问题的影响，发现有序描述显著优于随机描述，且不同任务偏好不同的排列策略。

**[GuessArena: Guess Who I Am? A Self-Adaptive Framework for Evaluating LLMs in Domain-Specific Knowledge and Reasoning](guessarena_guess_who_i_am_a.md)**

:   提出 GuessArena，一种基于"猜猜我是谁"博弈游戏的自适应 LLM 评估框架，通过领域知识建模和多轮交互推理，在五个垂直行业中有效区分模型的领域知识和推理能力。

**[Hallucination Detox: Sensitivity Dropout (SenD) for Large Language Model Training](hallucination_detox_send.md)**

:   本文揭示了 LLM 训练过程中幻觉行为的振荡现象，提出 Sensitivity Dropout（SenD）训练协议——通过识别并确定性丢弃高变异敏感嵌入索引来降低训练中的幻觉方差，同时提出计算高效的 Efficient EigenScore（EES）近似方法，在 Pythia 和 Llama 模型上实现高达 17% 的测试时可靠性提升。

**[HALoGEN: Fantastic LLM Hallucinations and Where to Find Them](halogen_hallucinations.md)**

:   提出 HALoGEN——覆盖 9 个领域（含编程、科学引用、摘要等）的 10,923 条 prompt 的大规模幻觉评测框架，配套原子级自动验证器，在 14 个 LLM 的约 150,000 条生成上系统性评估幻觉，发现即使最佳模型也可能有高达 86% 的原子事实存在幻觉，并提出 Type A/B/C 三类错误分类法。

**[Help Me Write a Story: Evaluating LLMs' Ability to Generate Writing Feedback](help_write_story_feedback.md)**

:   探索 LLM 能否为创意写作者提供有意义的写作反馈——构建包含 1300 个故意引入写作问题的故事测试集，评估常用 LLM 的写作反馈生成能力，发现模型虽能提供具体且多数准确的反馈，但常错过最重要的写作问题且不会恰当地在批评和鼓励之间切换。

**[How does Misinformation Affect Large Language Model Behaviors and Preferences?](how_does_misinformation_affect_large_language.md)**

:   构建了目前最大的误信息评估基准 MisBench（1034 万条误信息），从知识冲突类型和文本风格两个维度系统分析 LLM 对误信息的行为和偏好，并提出 RtD 方法结合外部知识源提升误信息检测能力。

**[How to Enable Effective Cooperation Between Humans and NLP Models: A Survey of Principles, Formalizations, and Beyond](human_nlp_cooperation_survey.md)**

:   首次系统性地综述了人与NLP模型之间的合作范式，提出基于"谁为最终决策负责"的三类合作形式化分类体系（顺序合作、分诊合作、联合合作），并借鉴Grice合作原则定义了人机合作的基本原则，为后续研究提供了统一视角。

**[The Impossibility of Fair LLMs](impossibility_fair_llms.md)**

:   系统分析了多种技术公平性框架（fairness through unawareness、group fairness、fair representations、multi-sided fairness等）在通用LLM上的适用性，论证了每种框架要么在逻辑上无法扩展到通用AI场景、要么在实践中不可行——主要源于非结构化训练数据的敏感属性不可剥离、用例/人群组合的组合爆炸、以及公平性不具备可组合性。

**[Can Indirect Prompt Injection Attacks Be Detected and Removed?](indirect_prompt_injection_detection.md)**

:   本文系统研究间接 prompt injection 攻击的检测与移除：构建评估基准，发现现有检测模型对间接攻击表现不佳但专门训练的模型可达 99% 准确率，提出分割移除和抽取移除两种方法，并将检测+移除组合为过滤管道，有效降低间接 prompt injection 的攻击成功率。

**[Leveraging Self-Attention for Input-Dependent Soft Prompting in LLMs](input_dependent_soft_prompting.md)**

:   提出 ID-SPAM，通过在输入 token 嵌入上施加可学习自注意力层并经瓶颈 MLP 生成**输入依赖**的软提示，仅在单层 Transformer 输入端拼接即可超越多种 Soft Prompt 基线，且具备优秀的零样本跨任务/跨领域迁移能力。

**[Beyond Facts: Evaluating Intent Hallucination in Large Language Models](intent_hallucination_eval.md)**

:   本文提出"意图幻觉"（Intent Hallucination）概念——LLM 在处理复杂多条件查询时遗漏或误解部分意图约束导致的偏离用户意图的生成，构建 FaithQA 基准（20,068 题）和 Constraint Score 评估指标，实验表明意图幻觉在 SOTA 模型中普遍存在且随查询复杂度增加而加剧。

**[IRT-Router: Effective and Interpretable Multi-LLM Routing via Item Response Theory](irt_router_multi_llm.md)**

:   IRT-Router 借鉴心理测量学的项目反应理论（IRT），将 LLM 视为"考生"、query 视为"考题"，学习多维能力向量和难度/区分度参数实现可解释的多 LLM 路由，在 OOD 场景下达 87%+ 准确率且成本仅为 GPT-4o 的 1/30。

**[JuStRank: Benchmarking LLM Judges for System Ranking](justrank_llm_judge_system_ranking.md)**

:   首次大规模研究LLM判官在系统排名任务中的表现，提出JuStRank基准，揭示实例级判断能力与系统级排名能力之间的差距，并发现判官的"果断性"和"偏见"两个新兴特征。

**[Analyzing LLMs' Knowledge Boundary Cognition Across Languages Through the Lens of Internal Representations](knowledge_boundary_crosslingual.md)**

:   通过探测 LLM 内部表示，揭示知识边界认知在多语言间呈线性结构，提出 training-free 对齐方法实现跨语言知识边界感知迁移，并发现"弱到强泛化"现象。

**[Knowledge Boundary of Large Language Models: A Survey](knowledge_boundary_survey.md)**

:   本文提出了 LLM 知识边界的形式化定义框架，将知识分为四种类型（PAK/PSK/MSU/MAU），并围绕"为什么研究知识边界""如何识别知识边界""如何缓解知识边界问题"三个核心问题系统综述了相关研究。

**[La Leaderboard: A Large Language Model Leaderboard for Spanish Varieties and Languages of Spain and Latin America](la_leaderboard_spanish.md)**

:   构建首个面向西班牙和拉丁美洲语言的开源LLM排行榜，整合66个数据集覆盖西班牙语、加泰罗尼亚语、巴斯克语、加利西亚语，评估50个模型并分析训练策略、算力与性能的关系。

**[LangSAMP: Language-Script Aware Multilingual Pretraining](langsamp_multilingual_pretraining.md)**

:   提出 LangSAMP 方法，在多语言预训练中将语言和文字系统 (script) embedding 添加到 Transformer 输出端（而非输入端），使模型主干学到更语言中立的表示，在 500+ 语言的零样本跨语言迁移中一致优于基线。

**[Language-Codec: Bridging Discrete Codec Representations and Speech Language Models](language_codec_bridging_discrete_codec_speech_language_models.md)**

:   提出 Language-Codec，通过掩码通道残差向量量化（MCRVQ）机制和改进的傅里叶变换解码器，弥合离散编解码器表示与下游语音语言模型之间的鸿沟，仅用4个码本通道即实现高质量音频重建。

**[LazyReview: A Dataset for Uncovering Lazy Thinking in NLP Peer Reviews](lazyreview_peer_review.md)**

:   构建首个标注"懒惰思维"细粒度类别的同行评审数据集 LazyReview——发现 LLM 在零样本下难以检测评审中的懒惰思维启发式，但在 LazyReview 上指令微调后性能提升 10-20 个点，且经懒惰思维反馈修改的评审显著更全面和可操作。

**[Less, but Better: Efficient Multilingual Expansion for LLMs via Layer-wise Mixture-of-Experts](less_but_better_efficient_multilingual_expansion.md)**

:   分析 LLM 不同层间的跨语言表征相似度，提出 LayerMoE 按层分配不同数量的新语言专家（高相似层少分配、低相似层多分配），用 60% 更少的专家参数超越 SOTA，并通过在高相似层添加路由分类器进一步缓解灾难性遗忘。

**[On the Limit of Language Models as Planning Formalizers](limit_llm_planning_formalizer.md)**

:   系统评估"LLM-as-Formalizer"方法论的极限——首次要求 LLM 生成完整 PDDL 表示（而非部分），从不同自然度的文本描述中形式化规划领域，发现最强模型（GPT-4o/o3-mini/DeepSeek-R1）可有效形式化超越直接规划，但描述越自然性能越低，弱模型卡在语法错误而强模型面临语义错误。

**[Llama See, Llama Do: A Mechanistic Perspective on Contextual Entrainment and Distraction in LLMs](llama_see_llama_do_entrainment.md)**

:   本文发现并定义了"上下文夹带"(contextual entrainment)现象——LLM会对上下文中出现过的任意token赋予更高概率，并通过可微掩码方法定位了负责该现象的entrainment heads，关闭这些头后可显著抑制干扰效应。

**[LLM Braces: Straightening Out LLM Predictions with Relevant Sub-Updates](llm_braces_straightening.md)**

:   LLMBraces 通过计算 FFN 层中各 value 向量与输入的相关性得分，动态调节子更新（sub-update）的贡献权重，用极少参数（比 LoRA 少 75%）同时提升模型预测精度和实现可控文本生成。

**[LLM as a Broken Telephone: Iterative Generation Distorts Information](llm_broken_telephone.md)**

:   类比"传话游戏"研究 LLM 在迭代生成中的信息失真现象，通过多语言翻译链实验发现：信息失真随迭代累积，受中间语言选择和链复杂度影响，可通过温度控制和受限提示缓解但无法消除。

**[Evaluation of LLM Vulnerabilities to Being Misused for Personalized Disinformation Generation](llm_personalized_disinformation.md)**

:   系统评估了 6 个主流 LLM 生成个性化虚假信息的能力，发现大多数 LLM 能生成高质量个性化虚假新闻，且个性化请求反而降低了安全过滤器的触发率（相当于一种 jailbreak），同时轻微降低了机器生成文本的可检测性。

**[Re-ranking Using Large Language Models for Mitigating Exposure to Harmful Content on Social Media Platforms](llm_reranking_harmful_content.md)**

:   提出基于 LLM 的成对偏好重排序方法，在零样本和少样本设置下对社交媒体推荐序列中的有害内容进行降级排序，显著优于 Perspective API 和 OpenAI Moderation API 等工业级分类器，同时引入 PP-k 和 EWN 两个新评估指标。

**[Wait, that's not an option: LLMs Robustness with Incorrect Multiple-Choice Options](llm_robustness_incorrect_mcq.md)**

:   提出"反思判断力"（Reflective Judgment）概念来衡量 LLM 在所有选项都错误的选择题中拒绝选择的能力，发现对齐后的模型（GPT-4o 等）往往盲目服从指令选择错误选项，而基座模型反而表现更好，且该能力随模型规模增大而涌现。

**[LLM-Powered Test Case Generation for Detecting Bugs in Plausible Programs](llm_test_case_gen_bugs.md)**

:   提出TrickCatcher——一种LLM驱动的测试用例生成方法，通过PUT引导的程序变体生成、基于生成器的输入生成和多样性驱动的差异测试三阶段流程，专门检测"plausible programs"（能通过现有测试套件但仍含隐蔽bug的程序）中的tricky bugs，F1分数达到SOTA基线的1.66倍。

**[LLMs instead of Human Judges? A Large Scale Empirical Study across 20 NLP Evaluation Tasks](llm_vs_human_judges_study.md)**

:   构建包含20个NLP数据集（7万+实例）的 Judge-Bench 基准，系统评估11个LLM作为评判者与人类标注的一致性，发现模型在不同任务/属性/标注者专业水平上表现差异巨大，建议部署前必须针对特定任务做人类标注验证。

**[LLMs can Perform Multi-Dimensional Analytic Writing Assessments](llm_writing_assessment.md)**

:   利用 L2 研究生文献综述语料库，系统评估了 LLM 在多维分析写作评估（评分+评论）上的能力，并提出可解释的反馈质量评估框架 ProEval。

**[Language Models, Graph Searching, and Supervision Adulteration: When More Supervision is Less and How to Make More More](lm_graph_search_supervision.md)**

:   本文证明了 path-star 图搜索任务在 decoder-only LM 上的失败并非 next-token prediction 范式的根本缺陷，而是由"监督污染"（supervision adulteration）导致的——过量的 teacher-forcing 监督信号诱导模型学到 Clever Hans Cheat 捷径，阻碍了子任务分解；通过 token masking、ranking-into-the-future、scratchpad、树形拓扑等六种正交方法均可使任务可学。

**[Locate-and-Focus: Enhancing Terminology Translation in Speech Language Models](locateandfocus_enhancing_terminology_translation_in_speech.md)**

:   提出Locate-and-Focus方法用于语音LLM的术语翻译：先用滑动窗口检索定位语音中包含术语的片段，再通过音频替换和Tag Cue引导模型聚焦翻译知识，在英中/英德方向上术语翻译成功率大幅提升。

**[Mapping 1,000+ Language Models via the Log-Likelihood Vector](mapping_1000_models_loglikelihood.md)**

:   提出用对数似然向量（log-likelihood vector）将 1000+ 语言模型映射到一个统一空间，证明向量间欧氏距离近似 KL 散度，可实现模型聚类可视化、基准性能预测（r=0.96）和数据泄漏检测。

**[MAPS: Motivation-Aware Personalized Search via LLM-Driven Consultation Alignment](maps_personalized_search.md)**

:   首次建模电商搜索中的"搜索动机"——用户在搜索前的咨询行为蕴含的真实需求，提出MAPS框架融合LLM语义、MoAE池化和双重对齐机制，在真实商业数据上HR@10提升24.4%（从0.5685到0.7071）。

**[Marco-Bench-MIF: On Multilingual Instruction-Following Capability of Large Language Models](marco_bench_multilingual_if.md)**

:   将英文IFEval基准扩展到30种语言并进行文化本地化，揭示LLM在多语言指令遵循中高/低资源语言间25-35%的准确率差距，以及机器翻译数据低估模型性能7-22%。

**[MathFusion: Enhancing Mathematical Problem-solving of LLM through Instruction Fusion](mathfusion_instruction_fusion.md)**

:   MathFusion 提出了跨问题指令融合的数学数据增强框架，通过顺序融合、并行融合和条件融合三种策略将两个数学问题合成新问题，仅用 45K 额外合成指令就在 6 个 benchmark 上平均提升 18 分准确率。

**[Math Neurosurgery: Isolating Language Models' Math Reasoning Abilities Using Only Forward Passes](mathneuro_math_reasoning_isolation.md)**

:   MathNeuro 提出了一种仅用前向传播就能隔离 LLM 中数学推理特定参数的方法，通过计算权重×激活的重要性分数并过滤掉通用语言任务也需要的参数，实现了精准的数学能力"手术"——剪除这些参数删除数学能力，缩放它们则提升 4-35% 数学性能。

**[MEraser: An Effective Fingerprint Erasure Approach for Large Language Models](meraser_fingerprint_erasure.md)**

:   提出 MEraser（Mismatched Eraser），通过两阶段微调策略（错配数据擦除 + 干净数据恢复）以不到 1000 条样本完全移除 LLM 中基于后门的指纹水印，同时保持模型性能，并首创可迁移的 LoRA 擦除适配器。

**[MergePrint: Merge-Resistant Fingerprints for Robust Black-box Ownership Verification of Large Language Models](mergeprint_fingerprint_ownership.md)**

:   提出 MergePrint，首个针对模型合并（model merging）场景的 LLM 黑盒指纹验证方法，通过伪合并模型模拟合并行为并两阶段优化（输入优化 + 参数优化），使嵌入的指纹在合并后仍可被检测，实现高效、无害、抗篡改的所有权验证。

**[Meta-rater: A Multi-dimensional Data Selection Method for Pre-training Language Models](metarater_a_multidimensional_data_selection_method.md)**

:   提出Meta-rater多维数据选择框架，定义PRRC四个质量维度（专业性/可读性/推理性/清洁度），通过proxy模型回归学习多个质量分数的最优加权组合，使1.3B模型训练收敛速度翻倍、下游任务提升3.23%。

**[Multi-Level Explanations for Generative Language Models](mexgen_multi_level_explanations.md)**

:   提出 MExGen（Multi-Level Explanations for Generative Language Models），将 LIME/SHAP 等扰动式归因方法扩展到 LLM 的上下文生成任务上——为摘要/问答等任务中 LLM 输出的每个部分量化上下文各段落的影响程度，比 LLM 自解释更忠实。

**[Middle-Layer Representation Alignment for Cross-Lingual Transfer in Fine-Tuned LLMs](mid_layer_crosslingual_alignment.md)**

:   通过分析 1000+ 语言对发现 LLM 中间层具有最强的跨语言对齐潜力，提出在任务训练中集成中间层对齐目标（对比损失），在槽填充（F1 61.7%）、机器翻译（BLEU 32.3）和结构化文本生成上显著提升跨语言迁移，对未见语言也有效。

**[Comparing Moral Values in Western English-speaking Societies and LLMs with Word Associations](moral_values_western.md)**

:   提出通过词语联想（word association）而非直接提问来比较 LLM 与西方英语社会的道德价值观，发现 LLM 在正面道德维度上与人类更一致，但在情感多样性和具体性上存在系统性差异。

**[Delving into Multilingual Ethical Bias: The MSQAD with Statistical Hypothesis Tests for Large Language Models](msqad_multilingual_ethical_bias.md)**

:   提出多语言敏感问答数据集MSQAD（基于Human Rights Watch 17个人权话题），通过McNemar检验和PERMANOVA检验两种统计假设检验方法，系统验证了LLM在不同语言下对相同敏感问题的回答存在显著伦理偏差——中文和印地语拒绝率最高，西班牙语和德语最容易生成不当回答，且该偏差在7个不同LLM中普遍存在。

**[Multi-Attribute Steering of Language Models via Targeted Intervention](multi_attribute_steering.md)**

:   提出 MAT-Steer，通过属性感知的 token 级 gating 机制和正交性约束，实现推理时对 LLM 多属性（如真实性、毒性、偏见）的同时精准干预，在 QA 和生成任务上全面超越现有 ITI 和微调方法。

**[Do Large Language Models Have an English Accent? Evaluating and Improving the Naturalness of Multilingual LLMs](multilingual_llm_english_accent.md)**

:   本文揭示多语言 LLM 在非英语语言生成中存在"英语口音"——词汇和句法上偏向英语模式，提出了基于 JSD（词汇分布）和 WL 图核+MMD（句法依赖树）的语料级自然度指标，并通过 DPO 对齐方法有效提升目标语言的自然度。

**[Which of These Best Describes Multiple Choice Evaluation with LLMs? A) Forced B) Flawed C) Fixable D) All of the Above](multiple_choice_eval.md)**

:   系统性论证了 MCQA（多选题问答）作为 LLM 评测标准格式的三大缺陷——格式本身的局限性、数据集构建质量问题、以及 LLM 在 MCQA 上的特有错误——并从教育测试学中引入改进方案。

**[Nemotron-CC: Transforming Common Crawl into a Refined Long-Horizon Pretraining Dataset](nemotron_cc_pretraining_data.md)**

:   Nemotron-CC 通过分类器集成、合成数据改写和减少启发式过滤三种策略，从 Common Crawl 构建了 6.3T token 的长期预训练数据集，在 15T token 训练中超越 Llama 3.1 8B。

**[Nudging: Inference-time Alignment of LLMs via Guided Decoding](nudging_inference_time_alignment.md)**

:   提出 Nudging，一种免训练的推理时对齐算法，利用小型对齐模型在基础模型不确定时注入少量"nudging tokens"来引导输出，用 7-14 倍小的模型就能达到甚至超过大型对齐模型的性能。

**[Pandora's Box or Aladdin's Lamp: A Comprehensive Analysis Revealing the Role of RAG Noise in Large Language Models](pandora_box_rag_noise.md)**

:   本文从语言学视角定义了 RAG 系统中的 7 种噪声类型，构建了 NoiserBench 综合评测框架，通过 8 个 LLM 的大规模实验发现噪声可分为有害噪声（反事实、支持性、拼写）和有益噪声（语义、数据类型、非法句子），其中有益噪声反而能提升模型准确率 1-3%。

**[Enhancing Open-Domain Task-Solving Capability of LLMs via Autonomous Tool Integration from GitHub](paper_2312_17294.md)**

:   提出 OpenAgent 系统，通过自主从 GitHub 发现和集成专业工具来解决开放域任务，并构建 OpenAct 基准评测 LLM 在需要领域特定工具的开放域问题上的能力。

**[Parenting: Optimizing Knowledge Selection of Retrieval-Augmented Language Models with Parameter Decoupling and Tailored Tuning](parenting_optimizing_knowledge_selection_of_retrievalaugmented.md)**

:   受人脑功能分区启发，提出 Parenting 框架，通过解耦并定位 LLM 参数空间中与"上下文遵循"(adherence)和"噪声鲁棒"(robustness)相关的子空间，并为不同子空间设计定制化微调策略，实现两种能力的平衡提升。

**[PlanGenLLMs: A Modern Survey of LLM Planning Capabilities](plangenllms_planning_survey.md)**

:   PlanGenLLMs 是一篇系统性综述，基于经典 AI 规划评估框架提出完整性、可执行性、最优性、表示、泛化性和效率六大评估准则，全面梳理了 LLM 作为规划器的方法、评估和未来方向。

**[Plug-in and Fine-tuning: Bridging the Gap between Small Language Models and Large Language Models](plugin_finetuning_bridge.md)**

:   提出 PiFi 框架，将 LLM 的单个冻结层"插入"到 SLM 中再进行微调，以极低的额外计算成本将 LLM 的语言知识和泛化能力迁移到小模型，在 NLU 和 NLG 任务上均获得一致提升。

**[KoGEM: Polishing Every Facet of the GEM: Testing Linguistic Competence of LLMs and Humans in Korean](polishing_every_facet_of_the_gem.md)**

:   提出 KoGEM（韩语语法评估基准），1,524 道多选题覆盖音韵/形态/句法/语义/规范 5 大类 16 子类，零样本评估 27 个 LLM 发现模型在需要经验知识的任务（如发音规则）上远逊人类，而补充经验知识后可显著提升。

**[Only a Little to the Left: A Theory-grounded Measure of Political Bias in LLMs](political_bias_theory_grounded.md)**

:   基于政治科学理论构建科学有效的LLM政治偏见测量方法，对11个模型用多种提示变体收集88,110条回复，发现指令微调模型普遍偏左，但偏见度量对提示高度敏感，且Political Compass Test会夸大某些模型的偏见。

**[Pragmatics in the Era of Large Language Models: A Survey on Datasets, Evaluation, Opportunities and Challenges](pragmatics_survey.md)**

:   全面梳理用于评估 NLP 系统语用能力的资源——按语用现象（隐含义、指称、言语行为、会话含义、预设等）分类数据集，分析任务设计、数据收集方法和评估方式，揭示了现代 LLM 在处理语境相关语言使用上的趋势、挑战和空白。

**[Pre³: Enabling Deterministic Pushdown Automata for Faster Structured LLM Generation](pre3_deterministic_pda_structured_gen.md)**

:   提出 Pre³，将 LR(1) 文法转化为确定性下推自动机（DPDA），通过预计算前缀条件边消除运行时非确定性探索，实现结构化 LLM 生成的显著加速——每 token 耗时降低最高 40%，吞吐提升最高 36%。

**[Aligning Large Language Models with Implicit Preferences from User-Generated Content](pugc_align_implicit_pref_ugc.md)**

:   提出 PUGC 框架，利用非标注用户生成内容（UGC）中的隐式人类偏好来生成偏好数据——将 UGC 转化为查询+参考文本，以此评分模型生成的响应，用 DPO 实现可扩展的领域特定对齐，在 Alpaca Eval 2 上基于 Mistral-7B 达到 35.93% 长度控制胜率 SOTA。

**[RARE: Retrieval-Augmented Reasoning Enhancement for Large Language Models](rare_retrieval_augmented_reasoning.md)**

:   提出 RARE，在 rStar 的 MCTS 推理框架中引入两个检索增强动作（A6: 基于原始问题生成搜索查询并检索，A7: 对子问题进行检索并重新回答），并用检索增强的事实性评分器（RAFS）替代原始判别器，使 LLaMA 3.1 在医学和常识推理任务上达到甚至超越 GPT-4o 的水平。

**[Recurrent Knowledge Identification and Fusion for Language Model Continual Learning](recurrent_kif_continual_learning.md)**

:   提出Recurrent-KIF持续学习框架，通过内外循环迭代机制动态估计参数重要性分布，利用基于重要性的二值掩码进行知识融合，有效缓解灾难性遗忘并促进知识迁移。

**[Representation Bending for Large Language Model Safety](repbend_representation_bending_safety.md)**

:   提出 RepBend，将 activation steering 的核心思想（安全/不安全表示的向量差异）引入 LoRA 微调的损失函数设计，通过"弯曲"模型的表示空间使安全和不安全状态在潜在空间中远离彼此，在多种越狱攻击基准上实现高达 95% 的攻击成功率降低，且对模型通用能力影响极小。

**[Reversal of Thought: Enhancing Large Language Models with Preference-Guided Reverse Reasoning Warm-up](reversal_of_thought_enhancing_large_language.md)**

:   提出 Reversal of Thought (RoT)，一个即插即用的推理框架，通过偏好引导的逆向推理预热策略，让 LLM 从示例中反向生成"LLM 口味"的最优 prompt，再通过认知偏好管理器自动区分已知/未知任务，在多种推理任务上超越 CoT/ToT/GoT 等基线。

**[RuleArena: A Benchmark for Rule-Guided Reasoning with LLMs in Real-World Scenarios](rulearena_rule_guided_reasoning.md)**

:   提出 RuleArena——一个基于航空行李费、NBA交易规则、税务法规三个真实场景的benchmark，用于评估LLM遵循复杂自然语言规则进行推理的能力；实验发现即使最强模型（o1-preview）在最难任务上准确率也不足50%，暴露了LLM在规则召回、规则区分和数学计算三方面的系统性缺陷。

**[Salience Sparse Fine Tuning](salience_sparse_fine_tuning.md)**

:   首次系统评估 8 种 salience 指标用于稀疏微调（SPEFT）的效果，发现简单的梯度指标 + 静态掩码即可提供最佳性价比，在 GSM8k 上比 LoRA 高出 22.6%，质疑了"复杂方法才能做好 PEFT"的假设。

**[Segment-Level Diffusion: A Framework for Controllable Long-Form Generation with Diffusion Language Models](segment_level_diffusion.md)**

:   提出段落级扩散（Segment-Level Diffusion, SLD），将长文本输出切分为多个段落（如句子/对话轮次），对每个段落的潜在表示进行扩散建模，结合对比学习和对抗训练增强表示鲁棒性，在摘要、故事生成、对话生成等任务上实现了比现有扩散模型更好的长文本生成质量。

**[SelfElicit: Your Language Model Secretly Knows Where is the Relevant Evidence](selfelicit_evidence_highlighting.md)**

:   SelfElicit 发现 LLM 深层注意力分数天然能识别上下文中的关键证据（即使回答错误时也是），据此提出推理时自动高亮关键证据句的上下文增强方法，无需训练即可显著提升基于证据的 QA 任务性能。

**[Establishing Trustworthy LLM Evaluation via Shortcut Neuron Analysis](shortcut_neuron_eval.md)**

:   提出通过对比分析和因果分析定位污染模型中的"捷径神经元"（shortcut neurons），并通过 activation patching 抑制这些神经元，实现更可信的 LLM 评估，与 MixEval 的 Spearman 相关系数超过 0.95。

**[Embracing Imperfection: Simulating Students with Diverse Cognitive Levels Using LLM-based Agents](simulating_diverse_students.md)**

:   提出一种基于知识图谱认知原型的免训练框架，使LLM Agent能够模拟不同认知水平学生的学习行为（包括错误），在GPT-4o上实现94%的行为预测准确率，相比基线提升100%。

**[SkillVerse: Assessing and Enhancing LLMs with Tree Evaluation](skillverse_tree_eval.md)**

:   提出 SkillVerse，一种无监督的树结构 LLM 诊断框架——用 LLM-as-Judge 批评模型回答后组织为层次化技能树（dendrogram），可在任意粒度上分析模型能力，并用于改善 ICL（提升 25%）和预测新模型弱点（55% 成功率，高于基线 22pp）。

**[SongComposer: A Large Language Model for Lyric and Melody Generation in Song Composition](songcomposer_llm_lyric_melody_generation.md)**

:   提出 SongComposer，首个能同时生成歌词和旋律的大语言模型，通过元组格式对齐歌词与旋律、标量音高初始化和渐进式结构感知训练，在多个歌曲生成任务上超越 GPT-4。

**[StrucText-Eval: Evaluating Large Language Model's Reasoning Ability in Structure-Rich Text](structext_eval.md)**

:   提出StrucText-Eval——一个覆盖8种结构化语言（JSON/YAML/XML/Markdown/LaTeX/Org/CSV/Tree）和29个任务的自动生成评测基准，共5,800个样本，通过可控的嵌套深度和结构宽度调节难度。实验揭示开源LLM在标准集最高仅74.9%准确率，困难集降至45.8%，而人类在困难集达92.6%，暴露了LLM在复杂结构推理上的严重不足。

**[Can LLMs Generate High-Quality Test Cases for Algorithm Problems? TestCase-Eval](testcase_eval_llm_test_gen.md)**

:   提出 TestCase-Eval 基准评估 LLM 生成算法题测试用例的能力，包含 500 道 Codeforces 算法题和 10 万条人工解答，聚焦两个任务——故障覆盖（测试集能覆盖多少潜在错误）和故障暴露（能否为特定错误代码生成暴露性测试），对 19 个 SOTA LLM 的评估揭示了当前模型在测试生成上的能力和局限。

**[A Theory of Response Sampling in LLMs: Part Descriptive and Part Prescriptive](theory_of_llm_sampling.md)**

:   本文从认知科学视角揭示了LLM的采样启发式机制与人类决策类似：采样不仅反映概念的统计规范（描述性成分），还系统性地偏向隐含的理想值（规范性成分），这种偏移在500个概念、15个模型上均显著，并可能导致医疗等应用中的有偏决策。

**[Theory of Mind in Large Language Models: Assessment and Enhancement](theory_of_mind_llm.md)**

:   系统综述了 LLM 的心智理论（ToM）能力的评估基准（10+ story-based benchmarks）和增强策略（prompt-only 和 fine-tuning 两类方法），指出当前 LLM 在 ToM 推理上仍有显著不足，并提出未来方向。

**[The Impact of Token Granularity on the Predictive Power of Language Model Surprisal](token_granularity_impact.md)**

:   本文系统研究了子词 token 粒度（词表大小 256~128K）对语言模型 surprisal 预测人类阅读时间能力的影响，发现约 8K 词表大小的中等粒度 token 在自然阅读时间预测上最优，而更粗粒度（更接近词级）的 token 在花园路径句法效应上表现更敏感。

**[Token Prepending: A Training-Free Approach for Eliciting Better Sentence Embeddings from LLMs](token_prepending_training_free.md)**

:   提出 Token Prepending (TP) 技术，通过在每层将解码得到的句子嵌入前置到句子开头，使因果注意力机制下的早期 token 也能感知完整句子信息，无需训练即可显著提升 LLM 的句子嵌入质量。

**[Turning Trash into Treasure: Accelerating Inference of Large Language Models with Token Recycling](token_recycling.md)**

:   提出Token Recycling——一种无需额外训练的投机解码方法，将解码过程中产生的候选token存入邻接矩阵，通过BFS算法构建draft tree并用tree attention验证，仅需<2MB额外存储即在所有规模LLM上实现约2倍加速，超越现有无训练方法30%和有训练方法25%。

**[UniConv: Unifying Retrieval and Response Generation for Large Language Models in Conversations](uniconv_retrieval_response_gen.md)**

:   探索如何将对话场景中的稠密检索和响应生成统一到单个 LLM 中，通过三个联合训练目标（对话检索 + 响应生成 + 上下文识别指令）和数据差异缓解机制，在五个对话搜索数据集上实现检索和生成的相互促进，超越分离式基线。

**[ScaleQuest: Unleashing LLM Reasoning Capability via Scalable Question Synthesis from Scratch](unleashing_llm_reasoning_capability_via_scalable.md)**

:   提出 ScaleQuest，通过 Question Fine-Tuning (QFT) + Question Preference Optimization (QPO) 两阶段解锁 7B 模型的数学题生成能力，从零合成 100 万高质量问题-解答对，训练效果超越所有开源数据集并展现持续的数据扩展潜力。

**[Veracity Bias and Beyond: Uncovering LLMs' Hidden Beliefs in Problem-Solving Reasoning](veracity_bias_llm_hidden_beliefs.md)**

:   揭示了 LLM 在推理任务中存在"真实性偏见"（Veracity Bias）——尽管显式对齐反对刻板印象，LLM 仍系统性地将正确答案归因于特定种族群体（归因偏差），并对相同解答因"作者"种族不同给出不同评价（评估偏差），在数学、编程、常识推理和写作任务中普遍存在。

**[Many Heads Are Better Than One: Improved Scientific Idea Generation by A LLM-Based Multi-Agent System](virsci_multi_agent_idea_gen.md)**

:   提出基于 LLM 的多智能体系统 Virtual Scientists（VirSci），模拟真实科研团队的协作过程——组织多个 agent 团队协作生成、评估和改进科研 idea，在生成新颖科学想法方面超越单智能体 SOTA。
