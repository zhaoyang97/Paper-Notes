<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚖️ 对齐 / RLHF

**💬 ACL2025** · 共 **49** 篇

**[Aligned but Blind: Alignment Increases Implicit Bias by Reducing Awareness of Race](aligned_but_blind_implicit_bias.md)**

:   发现 LLM 对齐训练的矛盾效应：对齐成功消除了显式偏见（Llama 3 70B 降至 8.13%），但反而放大了隐式偏见（从 64.1% 升至 91.4%），机制是对齐使模型在歧义上下文中不再表征种族概念（"种族盲视"），导致安全护栏无法在隐性场景中激活。通过在早期层注入种族感知激活可将隐式偏见从 97.3% 降至 71.2%。

**[ASPO: Adaptive Sentence-Level Preference Optimization for Fine-Grained Multimodal Reasoning](aspo_adaptive_sentence-level_preference_optimization_for_fine-grained_multimodal.md)**

:   提出 ASPO（Adaptive Sentence-level Preference Optimization）——将 DPO 的偏好单元从"整个回复"细化到"每个句子"，为每个句子动态计算自适应奖励值（基于模型自身预测评估正确性和重要性），在多模态推理任务上显著优于传统回复级 DPO，有效减少幻觉并提升细粒度推理能力。

**[Atyaephyra at SemEval-2025 Task 4: Low-Rank Negative Preference Optimization](atyaephyra_at_semeval-2025_task_4_low-rank_negative_preference_optimization.md)**

:   在 SemEval 2025 LLM 遗忘共享任务中，将负偏好优化 (NPO) 与低秩适配 (LoRA) 结合，利用 LoRA 的结构特性零开销获取原始模型分布来计算 KL 散度正则化，显著稳定了遗忘过程并超越了任务基线。

**[AutoMixAlign: Adaptive Data Mixing for Multi-Task Preference Optimization in LLMs](automixalign_adaptive_data_mixing.md)**

:   AutoMixAlign 提出了一种理论驱动的多任务偏好优化数据混合方法：先训练各任务的 specialist model 确定最优 loss 基线，再通过 minimax 优化自适应调整数据混合比例，优先处理 excess loss（与 specialist 的差距）最大的任务，在 helpfulness/harmlessness/reasoning 多任务 DPO 中平均提升 9.42%。

**[Beyond Surface-Level Patterns: An Essence-Driven Defense Framework Against Jailbreak Attacks in LLMs](beyond_surface-level_patterns_an_essence-driven_defense_framework_against_jailbr.md)**

:   提出 EDDF，一种基于"攻击本质"而非表面模式的越狱防御框架：离线提取已知攻击的本质策略存入向量数据库，在线时对新查询做本质抽象+检索+细粒度判断，将攻击成功率降低至少 20% 且误报率仅 2.18%。

**[Beyond the Tip of Efficiency: Uncovering the Submerged Threats of Jailbreak Attacks in Small Language Models](beyond_the_tip_of_efficiency_uncovering_the_submerged_threats_of_jailbreak_attac.md)**

:   系统评估 13 个 SOTA 小语言模型（<4B参数）在 5 种越狱攻击下的安全性，发现 SLM 虽能抵御直接攻击但在越狱攻击下显著比大模型脆弱，进一步分析了架构压缩、量化和知识蒸馏等 SLM 技术对安全性的影响。

**[Boosting Vulnerability Detection of LLMs via Curriculum Preference Optimization with Synthetic Reasoning Data](boosting_vulnerability_detection_of_llms_via_curriculum_preference_optimization_.md)**

:   提出 ReVD 框架，通过双向漏洞推理数据合成 + 三元组 SFT（同时学习漏洞代码/修复代码/代码差异的推理）+ 课程化在线偏好优化（COPO），将 LLM 的漏洞检测准确率提升 12-23%，在 PrimeVul 和 SVEN 上达到 SOTA。

**[Breaking the Ceiling: Exploring the Potential of Jailbreak Attacks through Expanding Strategy Space](breaking_the_ceiling_exploring_the_potential_of_jailbreak_attacks_through_expand.md)**

:   基于精细化可能性模型 (ELM) 将越狱策略分解为四类可独立进化的组件（角色/内容支撑/语境/沟通技巧），提出 CL-GSO 遗传算法在组件级进行交叉与变异，将策略空间从既有方法的 40 种扩展到 839 种，在 Claude-3.5 上实现 96% 攻击成功率（此前方法最高仅 4%），同时提出基于意图一致性的评估机制，准确率达 96.5% 超越专用安全模型。

**[Chain-of-Jailbreak Attack for Image Generation Models via Editing Step by Step](chain-of-jailbreak_attack_for_image_generation_models_via_ed.md)**

:   提出 CoJ（Chain-of-Jailbreak）攻击，将生成恶意图像的单步请求分解为多步编辑指令链（从无害种子图像逐步编辑到目标），绕过图像生成模型的安全过滤器，在 GPT-4o 等模型上达到高攻击成功率。

**[Chain-of-Jailbreak Attack for Image Generation Models via Editing Step by Step](chain-of-jailbreak_attack_for_image_generation_models_via_editing_step_by_step.md)**

:   提出 Chain-of-Jailbreak（CoJ）攻击，将无法直接绕过安全护栏的恶意 query 分解为多步编辑子 query（删然后插、插然后删、改然后改回），在 GPT-4V/4o/Gemini 上达到 60%+ 越狱成功率；同时提出 Think-Twice Prompting 防御，拦截 95%+ 的 CoJ 攻击。

**[Cheems: A Practical Guidance for Building and Evaluating Chinese Reward Models from Scratch](cheems_chinese_reward_models.md)**

:   为弥补中文 Reward Model 资源的空白，本文构建了 CheemsBench（首个大规模中文 RM 评测基准）和 CheemsPreference（首个大规模中文偏好数据集），通过人机协作标注 + 远程监督过滤策略训练的 CheemsRM 在中文场景显著超越现有所有开源 RM。

**[CodeDPO: Aligning Code Models with Self Generated and Verified Source Code](codedpo_code_alignment.md)**

:   提出 CodeDPO，通过 PageRank 启发的自验证评分机制从自生成代码中构造高质量偏好对（93K 正确性 + 21K 效率），DPO 训练后在 8 个代码模型上 HumanEval 平均提升 10+ 分，同时提升代码执行效率 1.25-1.45×。

**[Curiosity-Driven Reinforcement Learning from Human Feedback](curiosity_driven_rlhf.md)**

:   CD-RLHF 将好奇心驱动探索（curiosity-driven RL）引入 RLHF，通过前向动力学模型的预测误差作为内在奖励，结合 top-k 门控过滤与 reward whitening，在不损失对齐质量的前提下大幅提升 LLM 输出多样性（Llama-3.2-1B 上 Diversity 提升 40.26%，EAD 提升 8.92%）。

**[Debate, Reflect, and Distill: Multi-Agent Feedback with Tree-Structured Preference Optimization for Efficient Language Model Enhancement](debate_reflect_and_distill_multi-agent_feedback_with_tree-structured_preference_.md)**

:   提出 D&R 框架，让小模型（student）与多个大模型（teacher）进行多轮辩论并收集自我反思和教师反馈，然后将辩论日志组织为偏好树做 Tree-structured DPO (T-DPO) 蒸馏，在 MMLU Pro 和 MATH 上平均提升 14.18 分，且推理效率优于基线。

**[DiffPO: Diffusion Alignment with Direct Preference Optimization](diffpo_diffusion_alignment.md)**

:   提出 DiffPO，将 LLM 对齐重新建模为句子级扩散去噪过程，通过 parallel decoding 实现高效推理时对齐，作为即插即用模块可增强任意底座模型的对齐质量。

**[Expectation Confirmation Preference Optimization for Multi-Turn Conversational Recommendation Agent](expectation_confirmation_preference_optimization_for_multi-turn_conversational_r.md)**

:   基于期望确认理论(ECT)提出 ECPO 多轮对话偏好优化框架，模拟用户在推荐对话中的满意度演变，百向期望确认定位不满意根因，用 LLM 用户模拟器(AILO)生成反馈，显著提升推荐 agent 的交互效果和效率。

**[Federated Data-Efficient Instruction Tuning for Large Language Models](federated_data-efficient_instruction_tuning_for_large_language_models.md)**

:   提出 FedHDS（Federated Hierarchical Data Selection），通过 intra-client 和 inter-client 两级层次化数据选择消除联邦学习中客户端内部和跨客户端的数据冗余，结合多层 Transformer 特征融合提升 coreset 质量；仅用不到 1.5% 的数据，在 Rouge-L 上相对 SOTA 全数据联邦基线平均提升 10.72%，训练效率提升最高达 48.8 倍。

**[Focused-DPO: Enhancing Code Generation Through Focused Preference Optimization on Error-Prone Points](focused-dpo_enhancing_code_generation_through_focused_preference_optimization_on.md)**

:   发现代码生成错误集中在特定"错误易发点"（error-prone points）——前缀/后缀通常正确，错误集中在中间代码段，提出 Focused-DPO 通过 PageRank 排序定位关键代码段并在 DPO 损失中加权放大，HumanEval+ 提升 4.41%、MBPP+ 提升 6.71%。

**[Understanding Impact of Human Feedback via Influence Functions](influence_functions_rlhf.md)**

:   首次将影响函数应用于 RLHF 奖励模型的反馈数据审计，结合 OPORP 向量压缩实现 2.5 倍加速，在偏差检测上超越 GPT-4o（AUC 0.8 vs 0.747），并从 Anthropic-HH 数据集中发现 47% 的错标样本。

**[Internal Value Alignment in Large Language Models through Controlled Value Vector Activation](internal_value_alignment_in_large_language_models_through_controlled_value_vecto.md)**

:   提出 ConVA（Controlled Value Vector Activation）框架，通过上下文控制的数据集精准识别 LLM 隐空间中的价值向量，并用门控最小扰动机制在推理时激活目标价值，在 Schwartz 10 种基本价值上实现平均 29.6% 的控制成功率提升，同时保持 97%+ 的文本流畅度和通用能力。

**[IOPO: Empowering LLMs with Complex Instruction Following via Input-Output Preference Optimization](iopo_input_output_preference.md)**

:   提出 IOPO（Input-Output Preference Optimization），在传统 DPO 仅优化输出偏好的基础上，引入输入偏好建模——让模型学习"给定回复 y，哪个指令 x 更匹配"，从而增强对复杂多约束指令的细粒度感知能力；同时构建了包含 120K 训练数据、1K 评测数据、覆盖 5 大类 26 个约束维度的 Trace 基准。

**[JailbreakRadar: Comprehensive Assessment of Jailbreak Attacks Against LLMs](jailbreakradar_comprehensive_assessment_jailbreak_attacks.md)**

:   本文提出了一个全面的越狱攻击评估框架 JailbreakRadar，收集了17种代表性越狱攻击方法，建立了六类攻击分类体系，并在9个对齐LLM上进行了大规模系统性评测，揭示了不同类型攻击在实用性和防御鲁棒性上的关键差异。

**[JsonTuning: Towards Generalizable, Robust, and Controllable Instruction Tuning](jsontuning_towards_generalizable_robust_and_controllable_instruction_tuning.md)**

:   提出 JsonTuning——将指令微调的输入输出从自然语言文本替换为 JSON 结构化格式，通过显式表示任务元素、关系和输出约束（JSON Schema），在 7 个预训练模型和 6 类任务上一致超越传统 TextTuning，平均性能从 26.78 提升到 30.88，同时显著增强鲁棒性和可控性。

**[Enhancing Safe and Controllable Protein Generation via Knowledge Preference Optimization](kpo_protein_safety.md)**

:   提出知识引导偏好优化（KPO）框架，通过蛋白质安全知识图谱识别安全/危险序列作为偏好信号，用强化学习训练蛋白质语言模型减少有害蛋白质序列的生成概率，同时保持功能性——为蛋白质生成的生物安全提供保障框架。

**[LLMs Caught in the Crossfire: Malware Requests and Jailbreak Challenges](llms_caught_in_the_crossfire_malware_requests_and_jailbreak_challenges.md)**

:   构建 MalwareBench 基准（320 个手工恶意代码需求 × 11 种黑盒越狱方法 = 3520 个 prompt），系统评测 29 个 LLM 在恶意代码生成场景下的安全性，发现越狱攻击将平均拒绝率从 60.93% 降至 39.92%，且模型参数量与防御能力并非正比关系。

**[LSSF: Safety Alignment via Low-Rank Safety Subspace Fusion](lssf_safety_subspace.md)**

:   LSSF 提出 LLM 的安全信息存在于低秩子空间中的假设，通过 SVD 提取安全对齐模型的主成分，利用安全奇异值熵自适应确定每层的保留秩，最终将提取的安全主成分线性融合到微调后的模型中，无需额外训练即可恢复因微调而退化的安全对齐，同时保持下游任务性能。

**[M2S: Multi-turn to Single-turn jailbreak in Red Teaming for LLMs](m2s_multiturn_to_singleturn_jailbreak_in.md)**

:   提出 M2S 框架，通过三种简单的格式转换方法（Hyphenize/Numberize/Pythonize）将多轮人类越狱对话压缩为单轮 prompt，不仅保持甚至超越原始多轮攻击效果（ASR 高达 95.9%，比多轮提升最多 17.5%），同时 token 使用量减半以上。

**[M-RewardBench: Evaluating Reward Models in Multilingual Settings](m_rewardbench.md)**

:   构建首个多语言奖励模型评估基准M-RewardBench（23种语言、2.87K偏好实例，覆盖对话/安全/推理/翻译四类能力），系统评估多种RM后发现英语与非英语RM性能存在显著差距，且翻译质量和语言资源量对RM表现有重要影响。

**[Measuring Data Diversity for Instruction Tuning: A Systematic Analysis and A Reliable Metric](measuring_data_diversity_for_instruction_tuning_a_systematic_analysis_and_a_reli.md)**

:   系统分析 11 种现有多样性度量方法的局限性，提出 NovelSum——一种同时考虑样本间差异和信息密度的数据多样性指标，与指令微调性能达到 0.97 相关性。

**[MPO: Multilingual Safety Alignment via Reward Gap Optimization](mpo_multilingual_safety_alignment.md)**

:   MPO 发现 LLM 在主导语言（英文）和目标语言间的隐式 Reward Gap 与安全性能强相关，提出直接最小化两者 Reward Gap 差异来将主导语言的安全对齐能力迁移到多语言，在三个模型上显著降低了低资源语言的攻击成功率且不损害通用能力。

**[Mutual-Taught for Co-adapting Policy and Reward Models](mutual_taught_policy_reward.md)**

:   Mutual-Taught 提出了一种基于 EM 算法的自训练框架，在偏好优化过程中同时迭代更新 policy model 和 reward model：E-step 用当前 RM 优化 PM，M-step 用 PM 更新前后的输出差异构建伪偏好对来更新 RM，解决了分布偏移导致的 reward hacking 问题，8B 模型在 AlpacaEval-2 达到 54.1% LC win rate。

**[Optimal Transport-Based Token Weighting for Enhanced Preference Optimization](otpo_token_weighting.md)**

:   OTPO 利用无平衡最优传输（UOT）在 chosen/rejected 回复的 token 表示之间计算语义对齐权重，使偏好优化聚焦于关键差异 token 而非均等对待所有 token，在 AlpacaEval2 上将 DPO 的 LC WR 从 48.14% 提升至 55.84%，并将 DPO/SimPO/SamPO/LDDPO 统一为 token 加权的特例。

**[Whose Boat Does it Float? Improving Personalization in Preference Optimization](personalized_preference_opt.md)**

:   提出"溯因推理"视角的偏好个性化方法：先用 LLM 推断偏好 chosen/rejected 回答背后的用户画像（Persona Inference），再用画像增强的偏好数据训练模型（Persona Tailoring），显著提升模型对不同用户需求的个性化适配能力。

**[PIG: Privacy Jailbreak Attack on LLMs via Gradient-based Iterative Prompts](pig_privacy_jailbreak.md)**

:   提出 PIG 框架，通过识别隐私查询中的 PII 实体类型、构建隐私上下文示例、并利用三种基于梯度的迭代优化策略更新上下文，实现对 LLM 的高效隐私越狱攻击，在白盒和黑盒模型上均达到 SOTA。

**[Probability-Consistent Preference Optimization for Enhanced LLM Reasoning](probability-consistent_preference_optimization_for_enhanced_llm_reasoning.md)**

:   提出 PCPO（Probability-Consistent Preference Optimization），在偏好对选择时同时考虑答案正确性和 token 级概率一致性（用 Levenshtein 距离过滤+概率一致性评分），并在 DPO 损失中按一致性加权，在 GSM8K/MATH-500/Olympiadbench 上一致超越标准 DPO 和 ScPO。

**[QueryAttack: Jailbreaking Aligned Large Language Models Using Structured Non-natural Query Language](queryattack_jailbreaking_aligned_large_language_models_using_structured_non-natu.md)**

:   提出 QueryAttack，将恶意自然语言查询分解为三个语义组件（内容、修饰符、类别）并填入编程语言模板（SQL/URL/Python/Java/C++ 等 9 种），结合 ICL 引导目标 LLM 直接用自然语言回复有害内容，无需解密步骤，在 GPT-4o 上 Ensemble 配置达到 96.35% ASR，且提出的跨语言 CoT 防御可将 ASR 降低最多 64%。

**[Red Queen: Safeguarding Large Language Models against Concealed Multi-Turn Jailbreaking](red_queen_safeguarding_large_language_models_against_concealed_multi-turn_jailbr.md)**

:   提出 Red Queen Attack——首个基于 Theory of Mind（ToM）构建多轮对话场景并隐藏恶意意图的越狱攻击方法，生成 56K 多轮隐蔽攻击数据，在 GPT-4o 上达到 87.6% ASR；同时提出 Red Queen Guard 防御策略，通过多轮 DPO 数据训练将 ASR 降至 <1%，同时不影响通用基准性能。

**[Rethinking Table Instruction Tuning](rethinking_table_instruction_tuning.md)**

:   系统消融表格指令微调中被忽视的超参数选择（学习率、数据量、epoch），揭示现有表格 LLM 因学习率过大（2e-5）导致通用能力严重退化（MMLU 降 14 分、AI2ARC 降 21 分），提出仅需 13 个数据集各 200 条（共 2600 条）+ 学习率 1e-6 + 2 epoch 微调 LLaMA 3.1 8B Instruct 即可构建 TAMA，在 13 个表格任务上匹配/超越 GPT-3.5 和 GPT-4，同时完整保持通用能力。

**[Reverse Preference Optimization for Complex Instruction Following](reverse_preference_optimization_for_complex_instruction_following.md)**

:   提出反向偏好优化（RPO），通过动态反转指令中未满足的约束将任意回复转化为"完美"chosen 样本，消除多约束偏好对中的噪声，在多轮复杂指令遵循任务上显著超越 DPO 基线。

**[Towards Reward Fairness in RLHF: From a Resource Allocation Perspective](reward_fairness_rlhf.md)**

:   将 RLHF 中的各种奖励偏差（长度偏差、类别偏差、社会偏差）统一定义为"奖励不公平"问题，从资源分配视角提出两种偏差无关的缓解方法——公平正则化和公平系数——在不针对特定偏差设计的情况下有效缓解多种偏差，实现更公平的人类偏好对齐。

**[Reward Generalization in RLHF: A Topological Perspective](reward_generalization_in_rlhf_a_topological_perspective.md)**

:   从信息拓扑的角度系统刻画 RLHF 中 reward 信息的流动——宏观层面将 RLHF 建模为自编码过程，微观层面提出 Induced Bayesian Network (IBN) 分析偏好数据拓扑对 reward 泛化的影响，进而提出树结构偏好数据方法，在 HH-RLHF/GSM-8K/DialogSum 三个任务上平均 65% win rate 超越链式 baseline。

**[Rewrite to Jailbreak: Discover Learnable and Transferable Implicit Harmfulness Instruction](rewrite_to_jailbreak_discover_learnable_and_transferable_implicit_harmfulness_in.md)**

:   提出 R2J（Rewrite to Jailbreak），一种可学习、可迁移的黑盒越狱方法——通过迭代训练 attacker LLM 学习改写有害指令（仅改措辞不改意图），相比 GCG/AutoDAN 等方法攻击成功率提高 20%+，且无额外前缀/后缀，更隐蔽且跨模型可迁移。

**[RISE: Subtle Errors in Reasoning: Preference Learning via Error-injected Self-editing](rise_error_inject_preference.md)**

:   RISE 发现 LLM 约 75% 的数学错误是微妙的步内错误（数字替换、操作数交换、步骤遗漏），通过让 LLM 自编辑向正确解注入预定义微妙错误来构造高质量难负样本，配合错误感知 DPO 训练，仅用 4.5K 样本在 GSM8K 提升 3.0%、MATH 提升 7.9%，并泛化到逻辑推理和代码生成。

**[SEA: Low-Resource Safety Alignment for Multimodal Large Language Models via Synthetic Embeddings](sea_lowresource_safety_alignment_for_multimodal.md)**

:   提出 SEA 框架，通过梯度优化生成合成模态 embedding（不需要真实图像/视频/音频），仅用文本安全数据就能实现多模态 LLM 的安全对齐，在单张 RTX3090 上 24 秒即可合成高质量 embedding，同时发布了视频和音频安全基准 VA-SafetyBench。

**[SynthesizeMe! Inducing Persona-Guided Prompts for Personalized Reward Models in LLMs](synthesizeme_persona_prompts.md)**

:   提出SynthesizeMe，通过bootstrap推理→合成用户画像→筛选信息性示例三步流程，无需微调即可为个性化奖励模型构建有效prompt，在Chatbot Arena上提升LLM-as-a-Judge个性化准确率4.4%。

**[TableDreamer: Progressive and Weakness-Guided Data Synthesis from Scratch for Table Instruction Tuning](tabledreamer_progressive_and_weakness-guided_data_synthesis_from_scratch_for_tab.md)**

:   提出 TableDreamer 两阶段数据合成框架：第一阶段从零合成多样化表格及种子指令数据，第二阶段通过弱点引导的迭代输入空间探索（在三个方向上演化数据，并用 LLM-as-Judge 筛选模型表现差的样本作为下一轮种子），仅用 27K GPT-4o 合成数据即将 Llama3.1-8B 的平均准确率提升 11.62%，超越使用 80K-100K 数据的所有基线方法。

**[Teaching an Old LLM Secure Coding: Localized Preference Optimization on Distilled Preferences](teaching_an_old_llm_secure_coding.md)**

:   提出 DiSCo（从前沿 LLM 蒸馏的安全代码偏好数据集，10K 实例覆盖 431 种 CWE）和 LPO（局部偏好优化算法，仅在安全相关 token 上传播损失），在四个安全编码基准上减少 19-40% 的安全问题，同时提升 3-10% 的代码质量。

**[Think&Cite: Improving Attributed Text Generation with Self-Guided Tree Search and Progress Reward Modeling](think_cite_attributed_text_gen.md)**

:   将归因文本生成（带引用的文本生成）建模为多步推理问题，提出自引导蒙特卡洛树搜索（SG-MCTS）结合进度奖励建模（PRM），通过多路径搜索+中间状态反思+生成/归因双维度进度奖励，在 ALCE 基准三个数据集上显著超越所有基线。

**[A Troublemaker with Contagious Jailbreak Makes Chaos in Honest Towns](tmcht_contagious_jailbreak_multiagent.md)**

:   提出 TMCHT 多智能体多拓扑攻击评估框架和 ARCJ（对抗性复制传染越狱）方法，解决了多智能体系统中单智能体攻击方法面临的"毒性消散"问题——通过优化检索后缀确保中毒样本易被检索，优化复制后缀使中毒信息具有传染性自我复制能力，在 line/star 拓扑和 100 智能体系统中分别提升 23.51%/18.95%/52.93% 的攻击成功率。
