<!-- 由 src/gen_blog_index.py 自动生成 -->
# 💡 LLM 推理

**💬 ACL2025** · 共 **34** 篇

**[Aristotle: Mastering Logical Reasoning with A Logic-Complete Decompose-Search-Resolve Framework](aristotle_logical_reasoning.md)**

:   提出 Aristotle 逻辑推理框架，将符号表达式和逻辑规则全面融入 Decompose-Search-Resolve 的每个阶段，通过逻辑分解器、搜索路由器和消解器三大组件实现逻辑完备的推理，在多个逻辑推理基准上以 GPT-4 平均提升 4.5%、GPT-4o 平均提升 5.4% 超越 SOTA。

**[BPP-Search: Enhancing Tree of Thought Reasoning for Mathematical Modeling Problem Solving](bpp-search_enhancing_tree_of_thought_reasoning_for_mathematical_modeling_problem.md)**

:   提出 BPP-Search 算法，将 Beam Search、过程奖励模型 (PRM) 和 Pairwise Preference 机制整合到 Tree-of-Thought 框架中，用于运筹学数学建模问题的自动求解，在 StructuredOR 等数据集上以更少的推理步骤显著超越 CoT/SC/ToT 基线。

**[Chain-of-Reasoning: Towards Unified Mathematical Reasoning in Large Language Models via a Multi-Paradigm Perspective](chain-of-reasoning_towards_unified_mathematical_reasoning_in_large_language_mode.md)**

:   提出 Chain-of-Reasoning (CoR) 框架，将自然语言推理 (NLR)、算法推理 (AR) 和符号推理 (SR) 三种范式统一整合，通过渐进范式训练 (PPT) 策略训练出 CoR-Math-7B，在定理证明任务上零样本超越 GPT-4o 41%，在 MATH 上超越 RL 方法 15%。

**[Chain-of-Reasoning: Towards Unified Mathematical Reasoning in Large Language Models via a Multi-Paradigm Perspective](chain_of_reasoning_unified_math.md)**

:   提出 Chain-of-Reasoning（CoR）框架，将自然语言推理（NLR）、算法推理（AR）和符号推理（SR）三种范式统一在一个推理链中，通过渐进范式训练（PPT）策略让 7B 模型（CoR-Math-7B）在零样本下超越 GPT-4o 41% 的定理证明准确率，在 MATH 基准上超过 RL 方法 15%。

**[ClozeMath: Improving Mathematical Reasoning in Language Models by Learning to Fill Equations](clozemath_improving_mathematical_reasoning_in_language_models_by_learning_to_fil.md)**

:   提出 ClozeMath，一种在微调时额外加入"方程填空"（text-infilling）目标的训练策略，让模型学会根据自然语言推理链预测被遮掩的数学方程，在 GSM8K/MATH 上显著超越 Masked Thought 基线，同时提高鲁棒性和 test-time scaling 效率。

**[CoT-UQ: Improving Response-wise Uncertainty Quantification in LLMs with Chain-of-Thought](cot-uq_improving_response-wise_uncertainty_quantification_in_llms_with_chain-of-.md)**

:   针对 LLM 在推理任务中过度自信的问题，提出 CoT-UQ 框架，将 CoT 推理步骤中的关键词提取和重要性评分整合到不确定性量化过程中，在逻辑和数学推理任务上 AUROC 平均提升 5.9%。

**[Critic-CoT: Boosting the Reasoning Abilities of Large Language Model via Chain-of-Thoughts Critic](critic-cot_boosting_the_reasoning_abilities_of_large_language_model_via_chain-of.md)**

:   提出 Critic-CoT 框架，通过逐步 Chain-of-Thought 批判范式和无需人工标注的弱监督数据自动构建，将 LLM 的自我批判从 System-1 式直觉判断推向 System-2 式慎重逐步分析；两阶段训练（GPT-4 蒸馏 + 自我批判）使 Llama-3-70B-Instruct 在 GSM8K 从 89.6% 提升至 95.4%，MATH500 从 50.4% 提升至 68.4%，并发现批判能力与任务求解能力可以相互增强。

**[Fine-Tuning on Diverse Reasoning Chains Drives Within-Inference CoT Refinement in LLMs](dcot_diverse_cot_refinement.md)**

:   提出 Diverse Chain of Thought (DCoT) 训练方法，通过在单次推理中生成多条串行推理链实现"推理内自修正"（within-inference refinement），在 1.3B–70B 模型上均超越标准 CoT 基线，尤其在大输出空间任务（数值/抽取型）上提升显著。

**[DeFine: Decision-Making with Analogical Reasoning over Factor Profiles](define_decision-making_with_analogical_reasoning_over_factor_profiles.md)**

:   提出 DeFine 框架，从财报电话会议等复杂场景的语音转录文本中构建概率因子画像(factor profile)，结合 Bradley-Terry 模型识别关键因子并通过因子画像间的 KL 散度做类比推理，用于辅助 LLM 在不确定性下做投资决策，准确率和 F1 均超越基线。

**[Dynamic and Generalizable Process Reward Modeling](dgprm_dynamic_process_reward.md)**

:   DG-PRM 提出了一种动态可泛化的过程奖励建模框架，通过奖励树存储多维度评估标准并动态选择步骤相关的奖励信号，用 Pareto 支配估计处理多面奖励，在 PRMBench 上达到 SOTA 且具有优异的跨领域泛化能力。

**[DRT: Deep Reasoning Translation via Long Chain-of-Thought](drt_deep_reasoning_translation_via_long_chain-of-thought.md)**

:   将长 CoT 推理引入机器翻译，构建多智能体框架（翻译器→顾问→评估器）迭代精炼含比喻/隐喻的文学翻译，合成 22K 长思维翻译训练样本，训练的 DRT-14B 在文学翻译上超越 QwQ-32B 和 DeepSeek-R1-Distill-32B 等大模型。

**[Unlocking General Long Chain-of-Thought Reasoning Capabilities of Large Language Models via Representation Engineering](glore_long_cot_representation.md)**

:   从表示空间角度发现 LLM 将长 CoT 推理编码为一种与普通 CoT 明确区分的通用能力，提出 GLoRE（General Long CoT Reasoning via Representation Engineering）——通过对比推理模式注入和领域特定表示调整来解锁长 CoT 能力，无需训练即可在域内和跨域场景下超越 SFT 方法。

**[Improve Vision Language Model Chain-of-thought Reasoning](improve_vlm_cot_reasoning.md)**

:   通过GPT-4o蒸馏193k CoT数据做SFT + 基于答案正确性构建偏好对做DPO，显著提升VLM的CoT推理能力（LLaVA-Reasoner在8个benchmark上CoT平均提升12.6%），且CoT训练还能反哺直接预测性能。

**[Local Look-Ahead Guidance via Verifier-in-the-Loop for Automated Theorem Proving](local_look-ahead_guidance_via_verifier-in-the-loop_for_automated_theorem_proving.md)**

:   提出 LeanListener，在自动定理证明(ATP)中引入 verifier-in-the-loop 设计，利用 Lean 验证器在每步提供中间反馈（子目标数变化）而非仅轨迹级奖励，通过在线 GRPO 训练使 ReProver 的 tactic 有效率和证明率均获提升，证明速度快 20%。

**[LogicPro: Improving Complex Logical Reasoning via Program-Guided Learning](logicpro_program_guided_reasoning.md)**

:   提出 LogicPro 数据合成方法，利用 LeetCode 算法题和 Python 代码解作为逻辑源，通过"问题生成→代码中间变量提取→程序引导推理生成"三步流水线，从 2360 道算法题合成 540K 高质量文本推理数据，在 BBH27、LogicBench、DROP 等多个 OOD 基准上显著超越现有推理数据集。

**[Linguistic Generalizability of Test-Time Scaling in Mathematical Reasoning](mclm_multilingual_test_time_scaling.md)**

:   提出 MCLM（55 语言的竞赛级数学基准），发现三种 test-time scaling 方法（ORM/PRM/Budget Forcing）在英语上提升显著（如 AIME +20 分），但在其他语言上平均仅提升 1.94 分，表明 test-time scaling 的多语言泛化能力严重不足。

**[MMBoundary: Advancing MLLM Knowledge Boundary Awareness through Reasoning Step Confidence Calibration](mmboundary_reasoning_step_confidence.md)**

:   提出 MMBoundary 框架，通过在推理链的每一步插入自然语言置信度表述（而非只在最终回答后给置信度），结合文本+跨模态的自奖励信号估计置信度，并用 SFT+RL 两阶段训练实现步级置信度校准，平均降低 7.5% 校准误差并提升 8.3% 任务准确率。

**[One Missing Piece for Open-Source Reasoning Models: A Dataset to Mitigate Cold-Starting Short CoT LLMs in RL](one_missing_piece_for_open-source_reasoning_models_a_dataset_to_mitigate_cold-st.md)**

:   构建 100K 长 CoT 数据集解决开源短 CoT LLM 在 RL 训练中的冷启动问题——用 1K 种子数据捕捉 o1 的推理流模式，然后用短 CoT LLM (GPT-4o) 扩展为长推理链，训练后的模型在 RLVR 初始化后获得 2-3 倍更大的收益。

**[PCoT: Persuasion-Augmented Chain of Thought for Detecting Fake News and Social Media Disinformation](pcot_persuasion-augmented_chain_of_thought_for_detecting_fake_news_and_social_me.md)**

:   提出 PCoT，一种零样本方法，利用说服知识增强虚假信息检测——受心理学研究启发（识别说服谬误可提高假新闻检测），两阶段处理：先识别分析说服信号，再将说服分析融入推理判断，平均提升 15%。

**[PCoT: Persuasion-Augmented Chain of Thought for Detecting Fake News and Social Media Disinformation](pcot_persuasion_chain_of_thought_fake_news.md)**

:   提出 PCoT（说服增强的思维链）方法，通过两阶段推理——先让 LLM 分析文本中的说服策略，再结合说服分析结果判断是否为虚假信息——在零样本设置下，5 个 LLM 和 5 个数据集上平均提升 15% 的检测 F1。

**[Ranked Voting based Self-Consistency of Large Language Models](ranked_voting_based_self-consistency_of_large_language_models.md)**

:   将 Self-Consistency 的多数投票升级为排序投票，让 LLM 每次推理生成多个候选答案的偏好排序而非单一答案，用三种排序投票方法（IRV/BCV/MRRV）聚合多次推理的排序信息，在 6 个数据集上一致超越传统 SC，最高提升 12.46%。

**[Rethinking the Role of Prompting Strategies in LLM Test-Time Scaling: A Perspective of Probability Theory](rethinking_the_role_of_prompting_strategies_in_llm_test-time_scaling_a_perspecti.md)**

:   本文在 6 个 LLM × 8 种 prompting 策略 × 6 个 benchmark 上系统实验发现，随着 majority voting 采样次数增加，简单的 CoT 始终超越复杂 prompting 策略；并从概率论角度给出理论证明，提出 $O(1)$ 复杂度的 scaling 性能预测方法和两种改进策略。

**[Revisiting Self-Consistency from Dynamic Distributional Alignment Perspective on Answer Aggregation](revisiting_self-consistency_from_dynamic_distributional_alignment_perspective_on.md)**

:   将 Self-Consistency 重新理解为采样分布与真实答案分布的动态对齐问题，揭示温度不仅控制采样随机性还直接塑造真实答案分布，据此提出置信度驱动的三阶段动态温度调节机制（FSD 阈值理论推导），在 10 个模型 × GSM8K/MATH 上零训练开销同时提升平均和最佳性能。

**[Safe: Enhancing Mathematical Reasoning in Large Language Models via Retrospective Step-aware Formal Verification](safe_math_reasoning.md)**

:   提出 Safe 框架，首次利用 Lean 4 形式化语言对 LLM 数学推理的每一步进行回顾性逐步验证，通过自动形式化+自动定理证明检测幻觉，并与前瞻性 PRM 分数融合，在多个数学数据集上取得 SOTA，同时发布包含 30,809 条形式化声明的 FormalStep 基准。

**[Stepwise Reasoning Disruption Attack of LLMs](seed_stepwise_reasoning_disruption_attack.md)**

:   提出 SEED（Stepwise rEasoning Error Disruption）攻击方法，通过在 LLM 的推理链前几步中巧妙注入细微错误（如微调计算数字），让模型在后续推理中自然传播错误得出错误答案，兼容零样本/少样本设置，GPT-4o 检测率低至 0.8%，揭示了 LLM 逐步推理过程的严重安全漏洞。

**[SoftCoT: Soft Chain-of-Thought for Efficient Reasoning with LLMs](softcot_soft_chain_of_thought.md)**

:   提出 SoftCoT，用一个冻结的小型辅助模型（如 LLaMA-3.2-1B）生成实例特定的"软思维 token"（连续隐状态），通过可训练的投影模块映射到主 LLM 的表示空间作为推理前缀，实现参数高效的连续空间 CoT 推理，避免了全模型微调导致的灾难性遗忘问题。

**[STRICTA: Structured Reasoning in Critical Text Assessment for Peer Review and Beyond](stricta_structured_reasoning_peer_review.md)**

:   提出 STRICTA 框架，将专家文本评估（如论文审稿）建模为基于结构因果模型（SCM）的逐步推理图，收集 40+ 生物医学专家对 22 篇论文的 4000+ 推理步骤数据，揭示先验知识差异是评审分歧的主因、写作风格对终审影响过大，并发现 LLM 在人工监督下可有效辅助结构化评估。

**[Is That Your Final Answer? Test-Time Scaling Improves Selective Question Answering](test_time_scaling_selective_qa.md)**

:   首次评估 test-time scaling 模型（DeepSeek R1、S1）在选择性问答（允许拒绝回答）场景下的表现，发现增加推理计算不仅提高正确率还提升对正确答案的置信度，提出基于置信度阈值的选择函数和 "Jeopardy Odds" 效用函数来评估非零错误惩罚下的 test-time scaling 性能。

**[ThinkGuard: Deliberative Slow Thinking Leads to Cautious Guardrails](thinkguard_deliberative_slow_thinking_leads_to_cautious_guardrails.md)**

:   提出 ThinkGuard，一种批判增强的安全护栏模型，通过从强 LLM 蒸馏结构化批判(安全标签+详细理由)来训练护栏模型，实现“慢思考”式安全判断，相比 LLaMA Guard 3 准确率提升 16.1%、宏 F1 提升 27.0%。

**[Towards Better Chain-of-Thought: A Reflection on Effectiveness and Faithfulness](towards_better_chain-of-thought_a_reflection_on_effectiveness_and_faithfulness.md)**

:   系统分析 CoT 有效性和忠实性的影响因素，发现 CoT 有效性取决于问题难度、信息增益和信息流向，忠实性是有效性的关键前提，提出 QUIRE 方法（先回忆再增强）提升有效性 2.4% 和忠实性 5.6%。

**[Towards Safety Reasoning in LLMs: AI-agentic Deliberation for Policy-embedded CoT Data Creation](towards_safety_reasoning_in_llms_ai-agentic_deliberation_for_policy-embedded_cot.md)**

:   提出 AIDsafe 多智能体审议框架，自动生成嵌入安全策略的 CoT 训练数据，通过多 agent 协作扩展安全推理链并过滤欺骗性/冗余思维，微调后的模型在安全性和越狱鲁棒性上显著提升且不影响实用性。

**[TRACT: Regression-Aware Fine-tuning Meets Chain-of-Thought Reasoning](tract_regression_cot.md)**

:   提出 TRACT，一种两阶段回归感知微调方法，将 CoT 推理与回归损失（squared error）结合，用于提升 LLM-as-a-Judge 场景中的数值评分精度，显著优于仅用交叉熵训练或仅用回归损失的现有方案。

**[Training Turn-By-Turn Verifiers For Dialogue Tutoring Agents The Curious Case Of](training_turn-by-turn_verifiers_for_dialogue_tutoring_agents_the_curious_case_of.md)**

:   提出 **Traver**（Trace-and-Verify）agent 工作流，通过**知识追踪**显式估计学生知识状态 + **逐轮验证器**（turn-by-turn verifier）对候选辅导话语打分选优，并设计 **Dict** 自动评估协议（模拟学生 + 代码生成测试），在编程辅导场景中将学生 Pass 率从 38.7% 提升至 43.7%（相对提升 106.5%），显著超越 Vanilla Instruct、Self-Refine 和 TreeInstruct。

**[Unveiling the Key Factors for Distilling Chain-of-Thought Reasoning](unveiling_the_key_factors_for_distilling_chain-of-thought_reasoning.md)**

:   系统研究影响 CoT 蒸馏的三大因素（粒度、格式、教师模型），发现 SLM 与粒度呈非单调关系、格式影响较小、强教师不总是更好。
