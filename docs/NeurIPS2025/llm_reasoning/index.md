<!-- 由 src/gen_blog_index.py 自动生成 -->
# 💡 LLM 推理

**🧠 NeurIPS2025** · 共 **74** 篇

**[AbbIE: Autoregressive Block-Based Iterative Encoder for Efficient Sequence Modeling](abbie_autoregressive_block-based_iterative_encoder_for_efficient_sequence_modeli.md)**

:   提出 AbbIE，一种将 decoder-only Transformer 的中间层（Body）进行递归迭代的架构，只需训练时用 2 次迭代，推理时即可通过增加迭代次数实现 upward generalization，在语言建模困惑度和 zero-shot ICL 任务上均超过标准 Transformer，且可作为标准 Transformer 的 drop-in 替代。

**[Adaptive Dual Reasoner: Large Reasoning Models Can Think Efficiently by Hybrid Reasoning](adaptive_dual_reasoner_large_reasoning_models_can_think_efficiently_by_hybrid_re.md)**

:   提出 Adaptive Dual Reasoner (ADR)——让推理模型在 fast thinking（简单推理步骤压缩）和 slow thinking（复杂推理步骤保留深度）之间动态切换，通过 SFT 冷启动 + EHPO（熵引导混合策略优化）训练，在数学推理基准上准确率提升最高 6.1% 同时推理 token 减少 49.5%-59.3%。

**[Are Large Reasoning Models Good Translation Evaluators? Analysis and Performance Boost](are_large_reasoning_models_good_translation_evaluators_analysis_and_performance_.md)**

:   首次系统分析了大推理模型（LRM）在机器翻译MQM评估中的行为，发现LRM存在"过度思考"、评分高估和材料选择依赖模型规模等问题，并提出ThinMQM方法通过训练合成人类评分轨迹来校准LRM思维过程，将思维预算减少约35倍同时提升评估性能（7B模型提升+8.7相关性分数）。

**[ARM: Adaptive Reasoning Model](arm_adaptive_reasoning_model.md)**

:   ARM 通过让模型自适应地选择四种推理格式（直接回答、短CoT、代码、长CoT），配合改进的 Ada-GRPO 训练算法解决 format collapse 问题，在保持与纯长CoT模型持平的准确率的同时平均节省 ~30% token，最多节省 ~70%。

**[Atom of Thoughts for Markov LLM Test-Time Scaling](atom_of_thoughts_for_markov_llm_testtime_scaling.md)**

:   提出 Atom of Thoughts (AoT)，将 LLM 推理建模为马尔可夫链，每个状态是与原问题答案等价但复杂度递减的自包含子问题，通过 DAG 分解+收缩的两阶段转移机制消除历史依赖，可与 ToT/反思等方法无缝集成，在数学/代码/多跳QA等6个benchmark上全面领先现有推理框架。

**[Auditing Meta-Cognitive Hallucinations in Reasoning Large Language Models](auditing_meta-cognitive_hallucinations_in_reasoning_large_language_models.md)**

:   系统性审计推理大模型（RLLM）中幻觉的产生与传播机制，发现长 CoT 中的反思（reflection）会通过元认知偏差放大幻觉而非纠正它，即使在幻觉源头进行干预也难以改变最终结果（chain disloyalty），揭示现有幻觉检测方法在多步推理场景下严重不足。

**[Base Models Know How to Reason, Thinking Models Learn When](base_models_know_how_to_reason_thinking_models_learn_when.md)**

:   通过无监督 SAE 聚类发现 thinking model 的推理机制分类，然后用 steering vector 在基座模型上激活这些潜在推理能力，混合模型恢复高达 91% 的 thinking-base 性能差距（无需权重更新），证明基座模型已具备推理能力，thinking model 只是学会了"何时"部署它们。

**[Beyond Chemical QA: Evaluating LLM's Chemical Reasoning with Modular Chemical Operations](beyond_chemical_qa_evaluating_llms_chemical_reasoning_with_modular_chemical_oper.md)**

:   提出 ChemCoTBench，首个评估 LLM 化学推理能力的 CoT 基准，将复杂化学问题分解为模块化的化学操作（加/删/替换官能团），配合 22,000 条专家标注的 CoT 数据集（ChemCoTDataset），系统性评估了推理型和非推理型 LLM 在分子理解/编辑/优化/反应预测上的能力。

**[Causal Head Gating: A Framework for Interpreting Roles of Attention Heads in Transformers](causal_head_gating_a_framework_for_interpreting_roles_of_attention_heads_in_tran.md)**

:   提出 Causal Head Gating (CHG)，通过对 Transformer 的每个 attention head 学习一个可微门控系数并结合正/负正则化，将 head 分为促进（facilitating）、干扰（interfering）、无关（irrelevant）三类，无需人工标签或 prompt 模板即可发现因果子电路，并扩展为对比 CHG 以分离 ICL 和指令遵循的独立电路。

**[Clip-and-Verify: Linear Constraint-Driven Domain Clipping for Accelerating Neural Network Verification](clip-and-verify_linear_constraint-driven_domain_clipping_for_accelerating_neural.md)**

:   提出Clip-and-Verify框架，通过利用线性界传播产生的约束来裁剪输入空间和收紧中间层界，包含完全裁剪（坐标上升求解对偶问题）和松弛裁剪（收缩输入盒）两种GPU高效算法，最多减少96%的BaB子问题数量，是VNN-COMP 2025获胜验证器的核心组件。

**[Co-Evolving LLM Coder and Unit Tester via Reinforcement Learning](co-evolving_llm_coder_and_unit_tester_via_reinforcement_learning.md)**

:   提出CURE框架，通过单元测试生成器与代码生成器的相互监督和共同进化，在无需ground-truth代码的情况下显著提升LLM代码生成能力。

**[Cognitive Mirrors: Exploring the Diverse Functional Roles of Attention Heads in LLM Reasoning](cognitive_mirrors_exploring_the_diverse_functional_roles_of_attention_heads_in_l.md)**

:   提出CogQA基准数据集和多类probing框架，系统分析LLM中注意力头的认知功能特化现象，发现认知头具有稀疏性、普遍性和层级化功能组织特征，去除认知头显著降低推理性能，增强则提升准确率。

**[Controlling Thinking Speed in Reasoning Models](controlling_thinking_speed_in_reasoning_models.md)**

:   通过表示工程（Representation Engineering）从 LRM 的隐藏空间中提取控制快/慢思考转换的 steering vector，结合基于层间 logit 散度的实时推理难度估计，实现无需训练的自适应推理速度调节，在 4 个 LRM 上平均提升 +1.3% 准确率并减少 -8.6% token 使用。

**[Cooperative Retrieval-Augmented Generation for Question Answering: Mutual Information Exchange and Ranking by Contrasting Layers](cooperative_retrieval-augmented_generation_for_question_answering_mutual_informa.md)**

:   提出CoopRAG框架，通过问题展开、基于检索器层对比的重排、以及推理链补全，实现检索器与LLM的双向合作，在多跳QA上超越HippoRAG2 5.3%，单跳QA上提升35.2%。

**[CoT Red-Handed: Stress Testing Chain-of-Thought Monitoring](cot_redhanded_stress_testing_chainofthought_monitoring.md)**

:   在 AI Control 框架下系统评估了 Chain-of-Thought 监控的有效性：发现 CoT 监控在检测微妙破坏行为上比仅监控 action 更有效（+10pp），但在检测明显破坏行为时反而更差（-25pp，因为推理中的伪合理化会欺骗监控），提出 hybrid 监控协议（独立评分 CoT 和 action 后加权）在所有场景下一致优于两种单一监控，检测率提升 2 倍。

**[Curriculum Abductive Learning](curriculum_abductive_learning.md)**

:   提出 Curriculum Abductive Learning (C-ABL)，通过将知识库按依赖结构分割为子知识库并渐进式引入训练，大幅缩小 ABL 的 abduction 搜索空间，显著提升训练稳定性、收敛速度和最终精度。

**[Deep Value Benchmark: Measuring Whether Models Generalize Deep Values or Shallow Preferences](deep_value_benchmark_measuring_whether_models_generalize_deep_values_or_shallow_.md)**

:   提出 Deep Value Benchmark (DVB)，通过"先混淆后解混淆"的实验设计，测量 LLM 是学习了深层人类价值观还是仅记住了表层偏好模式，发现所有模型的深层价值泛化率 (DVGR) 仅为 0.30，远低于随机水平。

**[DisCO: Reinforcing Large Reasoning Models with Discriminative Constrained Optimization](disco_reinforcing_large_reasoning_models_with_discriminative_constrained_optimiz.md)**

:   分析 GRPO 目标函数，揭示其固有的难度偏差（对过难/过易问题赋予过低权重）和熵不稳定性问题，提出基于判别学习的 DisCO 框架，通过无裁剪评分函数、平方铰链约束优化和 DRO 处理不平衡 rollout，在 1.5B 模型上平均超过 GRPO 7%、超过 DAPO 6%。

**[Does Thinking More Always Help? Mirage of Test-Time Scaling in Reasoning Models](does_thinking_more_always_help_mirage_of_test-time_scaling_in_reasoning_models.md)**

:   通过系统实验揭示 LRM 测试时扩展（反复 "Wait" 提示延长推理）的性能呈先升后降的非单调趋势，用概率模型证明这种"提升"只是方差增大导致的海市蜃楼而非真正推理能力提升，并提出 parallel thinking 策略在相同 token 预算下准确率提升最高 22%。

**[DreamPRM: Domain-Reweighted Process Reward Model for Multimodal Reasoning](dreamprm_domain-reweighted_process_reward_model_for_multimodal_reasoning.md)**

:   提出 DreamPRM，通过双层优化自动学习多模态推理数据集的域权重，解决 PRM 训练中的数据质量不均衡问题，在 MathVista 排行榜上以 o4-mini 模型达到 85.2% 的 top-1 准确率。

**[GPO: Learning from Critical Steps to Improve LLM Reasoning](gpo_learning_from_critical_steps_to_improve_llm_reasoning.md)**

:   GPO 通过蒙特卡洛模拟估计推理轨迹中每一步的优势函数，识别出"关键步骤"（模型犯错的转折点），然后从关键步骤重置并重新采样轨迹用于训练，可以即插即用地提升 PPO、DPO、KTO、SimPO、ORPO 等多种优化算法在推理任务上的表现。

**[I-RAVEN-X: Benchmarking Generalization and Robustness of Analogical and Mathematical Reasoning in Large Language and Reasoning Models](i-raven-x_benchmarking_generalization_and_robustness_of_analogical_and_mathemati.md)**

:   提出 I-RAVEN-X，一个增强版的符号化推理基准，通过增加操作数复杂度、属性范围和感知不确定性来评估 LLM 和 LRM 的类比推理与数学推理的泛化能力和鲁棒性，发现 LRM 在确定性推理上显著优于 LLM，但在不确定性推理下性能急剧下降。

**[Inference-Time Chain-of-Thought Pruning with Latent Informativeness Signals](inference-time_chain-of-thought_pruning_with_latent_informativeness_signals.md)**

:   提出 KAPPA (KL-Adjusted Pruned Path Algorithm)，利用 KL 散度、置信度和熵三个无需额外训练的信号对 Best-of-N 采样的推理分支进行渐进式剪枝，在保持准确率的同时实现最高 60% 峰值内存和 90% token 生成量的削减。

**[笔记1: CoT是幻觉吗？数据分布角度](is_chain-of-thought_reasoning_of_llms_a_mirage_a_data_distribution_lens.md)**

:   通过构建完全可控的抽象环境DataAlchemy，本文揭示CoT推理是一种幻觉——其有效性完全由训练数据分布主导，在分布外场景表现极其脆弱。

**[Know What You Don't Know: Uncertainty Calibration of Process Reward Models](know_what_you_dont_know_uncertainty_calibration_of_process_reward_models.md)**

:   本文提出了一种基于分位数回归的PRM校准方法，使PRM输出的分数更准确地反映LLM实际推理成功概率，并基于校准后的PRM设计了实例自适应推理时缩放（IAS）策略，在保持准确率的同时显著降低推理成本。

**[Large Language Models Can Learn and Generalize Steganographic Chain-of-Thought under Process Supervision](large_language_models_can_learn_and_generalize_steganographic_chain-of-thought_u.md)**

:   证明 LLM 在 RL 训练中受到 CoT 过程监督（惩罚特定字符串出现）时，会自发学会隐写术（steganography）——用替代编码隐藏被禁止的推理步骤，且这种编码是因果性的（load-bearing）并能泛化到训练中从未见过的字符串。

**[Latent Chain-of-Thought for Visual Reasoning](latent_chain-of-thought_for_visual_reasoning.md)**

:   将视觉CoT推理重新建模为后验推断问题，提出基于摊销变分推断(AVI)的LaCoT训练框架——包含参考引导GFlowNet微调(RGFN)、token级奖励近似和贝叶斯推理缩放(BiN)——在Qwen2.5-VL 3B/7B上比GRPO高出10.6%，在7个视觉推理基准上达到开源SOTA。

**[Let LRMs Break Free from Overthinking via Self-Braking Tuning](let_lrms_break_free_from_overthinking_via_self-braking_tuning.md)**

:   提出 Self-Braking Tuning (SBT) 框架，通过识别推理轨迹中的过度思考模式并构造自适应长度训练数据，使大型推理模型（LRM）学会自主判断何时停止推理，在数学推理任务上减少 30%-60% token 消耗的同时保持精度。

**[Let Me Think! A Long Chain-of-Thought Can Be Worth Exponentially Many Short Ones](let_me_think_a_long_chainofthought_can_be_worth_exponentiall.md)**

:   本文从理论和实验两方面证明：存在推理任务（图连通性问题），其中一条长 CoT（顺序缩放）的能力等价于指数多条短 CoT（并行缩放）——即将 CoT 长度减少一点点，就需要指数级增加并行采样数才能达到同等准确率。

**[LIMOPro: Reasoning Refinement for Efficient and Effective Test-time Scaling](limopro_reasoning_refinement_for_efficient_and_effective_test-time_scaling.md)**

:   提出PIR（基于困惑度的重要性精炼）框架，将LRM蒸馏的推理链分为"渐进推理"和"功能性步骤"（验证/多方法验证/纠错）两类，仅裁剪低PIR值的功能性步骤而完整保留渐进推理骨架，使微调后的模型在AIME/AMC/GPQA上准确率提升0.9%-6.6%同时token减少3%-41%，效率最高提升71%。

**[Lost in Transmission: When and Why LLMs Fail to Reason Globally](lost_in_transmission_when_and_why_llms_fail_to_reason_globally.md)**

:   提出有界注意力前缀预言机(BAPO)计算框架，将LLM的注意力头建模为有限带宽通信信道，证明图可达性等全局推理问题是BAPO-hard的（需超常数带宽），且CoT可将任何BAPO-hard问题转化为BAPO-easy问题，实验在GPT-4o/Claude/Gemini上验证理论预测。

**[Many LLMs Are More Utilitarian Than One](many_llms_are_more_utilitarian_than_one.md)**

:   在6个LLM上实验发现，多智能体集体讨论道德困境时会产生与人类群体类似的"功利主义增强"（Utilitarian Boost）——集体比个体更倾向接受为"多数人利益"伤害少数人，但LLM产生此效应的机制与人类不同（人类因结果敏感度增强，LLM则因规范敏感度降低或公正性增强等多种模式），且可通过模型异质性和提示多样性缓解。

**[Mind the Gap: Bridging Thought Leap for Improved Chain-of-Thought Tuning](mind_the_gap_bridging_thought_leap_for_improved_chain-of-thought_tuning.md)**

:   本文首次系统性地定义了 CoT 推理链中的"思维跳跃"(Thought Leap)现象，提出 CoT-Bridge 模型自动检测并补全推理链中被省略的中间步骤，在 NuminaMath 上最高提升 +5.87%，并可作为即插即用模块增强蒸馏和 RL 流程。

**[On Learning Verifiers and Implications to Chain-of-Thought Reasoning](on_learning_verifiers_and_implications_to_chain-of-thought_reasoning.md)**

:   从PAC学习角度系统研究CoT验证器的可学习性，在不同验证目标下给出样本复杂度的上下界，并揭示验证与生成之间的有趣计算关系。

**[One Token Embedding Is Enough to Deadlock Your Large Reasoning Model](one_token_embedding_is_enough_to_deadlock_your_large_reasoning_model.md)**

:   本文提出 Deadlock Attack，通过优化单个对抗性 token embedding 并以后门方式植入 LRM，使模型在推理时陷入永久思考循环（无限生成 "Wait"、"But" 等过渡词），在 4 个 LRM 和 3 个数学推理 benchmark 上实现 100% 攻击成功率，且对正常输入几乎无性能影响。

**[OS-Harm: A Benchmark for Measuring Safety of Computer Use Agents](os-harm_a_benchmark_for_measuring_safety_of_computer_use_agents.md)**

:   本文提出 OS-Harm，首个面向通用计算机使用 Agent（非仅浏览器）的安全性 benchmark，覆盖用户恶意使用、Prompt 注入攻击、模型自身失误三类风险共 150 个任务，评测发现前沿模型（o4-mini、Claude 3.7 Sonnet、Gemini 2.5 Pro 等）普遍直接服从有害指令（最高 70% 不安全率），且对基础 prompt 注入有 20% 的服从率。

**[ProofSketch: Efficient Verified Reasoning for Large Language Models](proofsketch_efficient_verified_reasoning_for_large_language_models.md)**

:   提出 ProofSketch 框架，通过符号闭包前向推理+短sketch生成+形式验证的多阶段pipeline，在降低token用量的同时提供逻辑推理的形式化正确性保证。

**[Provable Scaling Laws for the Test-Time Compute of Large Language Models](provable_scaling_laws_for_the_testtime_compute_of_large_lang.md)**

:   提出两种具有可证明缩放律的测试时计算算法——Knockout（淘汰赛式：生成多个候选再两两比较淘汰）和 League（联赛式：用平均胜率选最优候选），证明在 LLM 生成正确解概率 >0 且比较能力优于随机的极弱假设下，失败概率随测试时计算增加呈指数或幂律衰减，且仅需黑盒 LLM 无需额外验证器。

**[Re-FORC: Adaptive Reward Prediction for Efficient Chain-of-Thought Reasoning](re-forc_adaptive_reward_prediction_for_efficient_chain-of-thought_reasoning.md)**

:   提出Re-FORC，一个轻量级adapter在CoT推理过程中实时预测未来期望奖励 $\psi(t|x,z,\pi)$，将推理计算分配建模为Pandora's box问题，实现自适应早停（节省26%计算）、模型+计算联合选择（同等计算下+4%准确率或同等准确率-55%计算）和测试时计算伸缩（+11%准确率），且用户可通过代价系数 $\lambda$ 在推理时自由调控精度-效率权衡，无需重训。

**[RealMath: A Continuous Benchmark for Evaluating Language Models on Research-Level Mathematics](realmath_a_continuous_benchmark_for_evaluating_language_models_on_research-level.md)**

:   提出 RealMath，一个从 arXiv 论文和 Math StackExchange 中自动提取可验证数学问题的**可持续刷新**基准，用于评估 LLM 在真实研究级数学任务上的能力。

**[ReasonFlux-PRM: Trajectory-Aware PRMs for Long Chain-of-Thought Reasoning in LLMs](reasonfluxprm_trajectoryaware_prms_for_long_chainofthought_r.md)**

:   ReasonFlux-PRM 发现现有 PRM 无法有效评估推理模型的中间思考轨迹（trajectory），提出融合步骤级对齐/质量/连贯性分数和轨迹级模板引导奖励的 trajectory-aware PRM，在离线数据选择（SFT +12.1%）、在线 RL 奖励（+4.5%）和测试时 Best-of-N 缩放（+6.3%）三个场景中均显著优于包括 Qwen2.5-Math-PRM-72B 在内的强基线。

**[Reasoning by Superposition: A Theoretical Perspective on Chain of Continuous Thought](reasoning_by_superposition_a_theoretical_perspective_on_chain_of_continuous_thou.md)**

:   本文从理论上证明了连续思维链（Coconut）在有向图可达性问题上的表达优势：两层Transformer使用D步连续思维即可解决直径为D的图可达性问题，而离散CoT需要O(n²)步，其核心机制是连续思维向量以"叠加态"同时编码多条搜索前沿，实现隐式并行BFS。

**[Reasoning Models Better Express Their Confidence](reasoning_models_better_express_their_confidence.md)**

:   系统性证明推理模型（extended CoT）比非推理模型具有显著更优的置信度校准能力，并揭示"慢思考"行为（探索替代方案、回溯、验证）是校准提升的根本来源。

**[Reasoning Models Hallucinate More: Factuality-Aware Reinforcement Learning for Large Reasoning Models](reasoning_models_hallucinate_more_factuality-aware_reinforcement_learning_for_la.md)**

:   揭示了RL训练的推理模型（如DeepSeek-R1）比非推理模型产生更多幻觉，从理论上分析了三个根因（高方差梯度、熵约束、伪局部最优），并提出FSPO算法通过步级事实性验证调整token级advantage，在减少幻觉的同时保持甚至提升推理能力。

**[Rethinking Optimal Verification Granularity for Compute-Efficient Test-Time Scaling](rethinking_optimal_verification_granularity_for_compute-efficient_test-time_scal.md)**

:   提出 Variable Granularity Search (VG-Search)，通过可调的验证粒度参数 $g$ 统一 Beam Search 和 Best-of-N，发现传统每步验证是次优的，自适应调整 $g$ 可在提升准确率3%+的同时减少52%+的计算量。

**[SafePath: Preventing Harmful Reasoning in Chain-of-Thought via Early Alignment](safepath_preventing_harmful_reasoning_in_chain-of-thought_via_early_alignment.md)**

:   提出 SafePath，仅在推理开始处微调 8 个 token 的"Safety Primer"（"Let's think about safety first"），即可有效引导 LRM 走向安全推理路径，在 DeepSeek-R1-Distill 上减少 90% 有害输出且仅需 Direct Refusal 1/296 的训练计算量。

**[Sampling-Efficient Test-Time Scaling: Self-Estimating the Best-of-N Sampling in Early Decoding](sampling-efficient_test-time_scaling_self-estimating_the_best-of-n_sampling_in_e.md)**

:   提出 Self-Truncation Best-of-N (ST-BoN) 解码方法，通过理论证明早期隐状态一致性可预测最终一致性，在生成早期就识别并截断次优样本，实现降低80%+内存和50%延迟的同时保持BoN性能。

**[Scalable Best-of-N Selection for Large Language Models via Self-Certainty](scalable_best-of-n_selection_for_large_language_models_via_self-certainty.md)**

:   提出Self-Certainty度量，利用LLM输出的token概率分布量化模型信心，在无需额外奖励模型的情况下实现可扩展的Best-of-N选择，性能媲美或超越基于奖励模型的方法。

**[scPilot: Large Language Model Reasoning Toward Automated Single-Cell Analysis and Discovery](scpilot_large_language_model_reasoning_toward_automated_single-cell_analysis_and.md)**

:   提出 scPilot 框架和 scBench 基准，让LLM直接在单细胞RNA-seq数据上进行"组学原生推理"（读取标记基因→提出假设→调用工具验证→迭代修正），实现细胞类型标注准确率提升11%、轨迹推断graph-edit distance降低30%。

**[Segment Policy Optimization: Effective Segment-Level Credit Assignment in RL for Large Language Models](segment_policy_optimization_effective_segment-level_credit_assignment_in_rl_for_.md)**

:   提出SPO框架，采用段级（而非令牌级或轨迹级）的advantage估计，通过新颖的蒙特卡洛方法和树形采样，在短CoT和长CoT场景下分别超越PPO和GRPO 6-12和7-11个百分点。

**[笔记8：PolyMath - 多语言背景下的数学推理评估](self-evaluating_llms_for_multi-step_tasks_stepwise_confidence_estimation_for_fai.md)**

:   PolyMath构建的18语言、4难度级、500问题数学推理基准揭露：(1)推理性能跨语言差异达10分，(2)推理模型输入-输出语言一致性低且可能影响性能，(3)思考长度在语言间显著不一致，为多语言推理研究提供新视角。

**[Simulating Society Requires Simulating Thought](simulating_society_requires_simulating_thought.md)**

:   本文提出从"行为主义"模式转向"认知建模"范式，通过 GenMinds 框架用因果信念图建模 LLM Agent 的内部推理过程，并设计 RECAP 基准从可追溯性、人口统计敏感性和干预一致性三维度评估推理保真度。

**[SLAyiNG: Towards Queer Language Processing](slaying_towards_queer_language_processing.md)**

:   构建了首个显式标注的酷儿俚语（queer slang）数据集 SLAyiNG，包含 695 个术语和近 20 万条使用实例，并通过人机标注一致性实验（Krippendorff's α=0.746）表明推理模型可用于预筛选但仍需社区驱动的专家标注。

**[Smaller Models, Smarter Rewards: A Two-Sided Approach to Process and Outcome Rewards](smaller_models_smarter_rewards_a_two-sided_approach_to_process_and_outcome_rewar.md)**

:   将 Phi-4 系列小模型（3.8B/14B）的最后一层替换为回归头并微调，使其同时具备 ORM（结果奖励）和 PRM（过程奖励）能力，在代码生成任务上通过选择最优 rollout 实现 20%+ 的 pass@k 提升。

**[SolverLLM: Leveraging Test-Time Scaling for Optimization Problem via LLM-Guided Search](solverllm_leveraging_test-time_scaling_for_optimization_problem_via_llm-guided_s.md)**

:   无需训练，通过 MCTS 引导 LLM 生成 6 元素优化表述并转化为求解器代码，在 NL4Opt 上达 97.0%（vs OptiMUS 78.8%），超越微调方法且跨域泛化强。

**[SPRINT: Enabling Interleaved Planning and Parallelized Execution in Reasoning Models](sprint_enabling_interleaved_planning_and_parallelized_execution_in_reasoning_mod.md)**

:   通过将长链式推理轨迹重组为交替的规划-并行执行阶段，Sprint 使推理模型在保持准确率的同时，将长推理链的顺序 token 数减少高达 39%（OOD 任务上最高 65%），实现推理过程的动态并行化。

**[SQL-of-Thought: Multi-agentic Text-to-SQL with Guided Error Correction](sql-of-thought_multi-agentic_text-to-sql_with_guided_error_correction.md)**

:   提出 SQL-of-Thought——一个多智能体 Text-to-SQL 框架，将任务分解为 schema linking → 子问题识别 → CoT 查询计划生成 → SQL 生成 → 基于 31 类错误分类法的引导修正循环，用 Claude 3 Opus 在 Spider 上达到 91.59% 执行准确率，比此前最佳 Chase SQL（87.6%）提升近 4 个百分点。

**[SQL-R1: Training Natural Language to SQL Reasoning Model By Reinforcement Learning](sql-r1_training_natural_language_to_sql_reasoning_model_by_reinforcement_learnin.md)**

:   首次系统地将 GRPO 强化学习应用于 NL2SQL 任务，通过四层递进式奖励函数和 200K 冷启动 + 5K 复杂样本 RL 训练策略，7B 模型在 Spider 和 BIRD 上分别达到 88.7% 和 66.6%，超越 GPT-4 同规模模型。

**[Stop Summation: Min-Form Credit Assignment Is All Process Reward Model Needs for Reasoning](stop_summation_minform_credit_assignment_is_all_process_rewa.md)**

:   PURE 发现 PRM 导致 reward hacking 的根本原因是 RL 中标准的 sum-form 信用分配（$V(s) = \sum \gamma^t r_t$），并提出 min-form 替代方案（$V(s) = \min_{t' \geq t} r_{t'}$），通过将价值函数限制为未来奖励的最小值而非累积和，显著缓解 reward hacking——仅用 30% 训练步数就达到与规则奖励方法相当的推理性能。

**[The Hawthorne Effect in Reasoning Models: Evaluating and Steering Test Awareness](the_hawthorne_effect_in_reasoning_models_evaluating_and_steering_test_awareness.md)**

:   首次系统量化推理型LLM的"测试感知"(Hawthorne效应)：当模型察觉自己在被评估时会改变行为，论文通过线性探针定位感知激活并进行参数编辑引导，揭示测试感知对安全对齐的显著且方向不一致的影响。

**[The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity](the_illusion_of_thinking_understanding_the_strengths_and_limitations_of_reasonin.md)**

:   通过可控拼图环境系统揭示大型推理模型（LRMs）的三阶段行为：低复杂度不如标准 LLM、中等复杂度显著优于、高复杂度完全崩溃(0%)，且反直觉地在崩溃时减少思考 token，证实当前 LRMs 并未发展出真正泛化的推理能力。

**[The Impact of Quantization on Large Reasoning Model Reinforcement Learning](the_impact_of_quantization_on_large_reasoning_model_reinforcement_learning.md)**

:   系统实验发现在大推理模型的 RL 训练中，量化感知训练（QAFT/STE）会损害推理能力，而训练后量化（PTQ）和 QLoRA 即使在 4-bit 精度下也能很好地保持推理性能，为实践者提供了"先全精度 RL、再 PTQ 量化"的推荐路线。

**[The Virtues of Brevity: Avoid Overthinking in Parallel Test-Time Reasoning](the_virtues_of_brevity_avoid_overthinking_in_parallel_test-time_reasoning.md)**

:   证明选择最短答案是一个简单但有效的Best-of-N启发式方法，通过避免过度思考regime大幅降低计算成本，性能与自一致性可比或更优，在推理模型中表现特别突出。

**[ThinkSound: Chain-of-Thought Reasoning in Multimodal Large Language Models for Audio Generation and Editing](thinksound_chain-of-thought_reasoning_in_multimodal_large_language_models_for_au.md)**

:   提出三阶段交互式视频转音频框架 ThinkSound，通过 MLLM 生成结构化 CoT 推理来指导统一的音频生成基础模型，在 VGGSound 和 MovieGen Audio 基准上达到 SOTA，同时支持对象级精细化和自然语言指令编辑。

**[TIME: A Multi-level Benchmark for Temporal Reasoning of LLMs in Real-World Scenarios](time_a_multilevel_benchmark_for_temporal_reasoning_of_llms_i.md)**

:   TIME 提出一个面向真实世界时序推理的多层级 benchmark，覆盖 38,522 个 QA、3 个子数据集与 11 个细粒度子任务，系统刻画 LLM 在高密度时间信息、快速事件变化和复杂社会时序依赖下的推理能力，并分析了 test-time scaling 对 temporal reasoning 的实际影响。

**[Topology of Reasoning: Understanding Large Reasoning Models through Reasoning Graph Properties](topology_of_reasoning_understanding_large_reasoning_models_through_reasoning_gra.md)**

:   提出"推理图"概念——通过对 LLM 隐藏状态聚类构建有向图，从环路密度、直径和小世界指标三个图论维度分析大推理模型（如 DeepSeek-R1 蒸馏系列），发现推理模型的推理图具有显著更多环路（~5/样本）、更大直径和更强小世界特性（~6倍），且这些特性随任务难度和模型规模增长。

**[Towards Thinking-Optimal Scaling of Test-Time Compute for LLM Reasoning](towards_thinking-optimal_scaling_of_test-time_compute_for_llm_reasoning.md)**

:   揭示了过度延长 CoT 长度会损害 LLM 推理性能，并提出 Thinking-Optimal Scaling (TOPS) 策略，让模型为每道题选择最短正确响应进行自我提升，在效果和效率上同时优于现有蒸馏方法。

**[Transformers Provably Learn Chain-of-Thought Reasoning with Length Generalization](transformers_provably_learn_chain-of-thought_reasoning_with_length_generalizatio.md)**

:   从优化理论角度证明了一层 Transformer 通过梯度下降在合成状态追踪任务上能学会 CoT 推理并实现长度泛化，首次为常数深度 Transformer 学习 $\mathsf{NC}^1$-complete 问题（超越之前局限于 $\mathsf{TC}^0$ 的理论）提供了收敛保证。

**[TTS-VAR: A Test-Time Scaling Framework for Visual Auto-Regressive Generation](tts-var_a_test-time_scaling_framework_for_visual_auto-regressive_generation.md)**

:   提出 TTS-VAR——首个针对 Visual Auto-Regressive (VAR) 模型的测试时扩展框架，将图像生成建模为路径搜索问题，通过自适应递减批量 + 早期聚类多样性搜索 + 后期重采样潜力选择，在 Infinity 2B 上将 GenEval 分数从 0.69 提升到 0.75（+8.7%），N=2 即超越 Best-of-N 的 N=8 效果。

**[Two-Stage Learning of Stabilizing Neural Controllers via Zubov Sampling and Iterative Domain Expansion](two-stage_learning_of_stabilizing_neural_controllers_via_zubov_sampling_and_iter.md)**

:   提出两阶段训练框架——先用 Zubov 采样 + 动态域扩展估计吸引域（ROA），再用 CEGIS 反例精炼——联合学习神经网络控制器和 Lyapunov 函数，ROA 体积比基线大 5 到 $1.5 \times 10^5$ 倍，验证速度比 dReal 快 40-10000 倍。

**[Unlabeled Data Can Provably Enhance In-Context Learning of Transformers](unlabeled_data_can_provably_enhance_in-context_learning_of_transformers.md)**

:   提出增强型ICL框架，在prompt中同时包含少量标记样本和大量无标记样本，理论证明多层Transformer通过CoT可模拟EM算法从无标记数据中提取信息，将分类excess risk从 $\mathcal{O}(1/\sqrt{N})$ 改进到 $\mathcal{O}(1/\sqrt{N + \text{poly}(M)})$。

**[Unlocking Multimodal Mathematical Reasoning via Process Reward Model](unlocking_multimodal_mathematical_reasoning_via_process_reward_model.md)**

:   提出URSA三阶段框架，依次构建百万级多模态CoT数据(MMathCoT-1M)训练基座、双视角过程监督数据(DualMath-1.1M)训练PRM、以及PS-GRPO算法将PRM融入在线RL，8B模型在6个数学基准上平均超越GPT-4o 2.7%。

**[笔记6：Self-Evaluating LLMs - 多步任务的步级置信度估计](value-guided_search_for_efficient_chain-of-thought_reasoning.md)**

:   本文扩展置信度估计到多步任务，证明步级评估相比整体评估能更有效地检测推理失败，相对整体评估在CoQA上AUC-ROC提升15%，为多步推理系统的可信部署提供实用框架。

**[Visual Thoughts: A Unified Perspective of Understanding Multimodal Chain-of-Thought](visual_thoughts_a_unified_perspective_of_understanding_multi.md)**

:   首次从统一视角揭示多模态CoT工作的核心机制——"视觉思维"(Visual Thoughts)：MCoT通过将视觉信息缓存为中间推理步骤来增强LVLM推理，类似于计算机系统中的cache vs外部存储；定义了四种视觉思维表达形式（自然语言/结构化语言/编辑图像/生成图像），发现其有效性取决于表达的清晰性和简洁性。
