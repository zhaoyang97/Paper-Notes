---
search:
  exclude: true
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 💡 LLM 推理

**🧠 NeurIPS2025** · 共 **39** 篇

**[AbbIE: Autoregressive Block-Based Iterative Encoder for Efficient Sequence Modeling](abbie_autoregressive_block-based_iterative_encoder_for_efficient_sequence_modeli.md)**

:   提出 AbbIE，一种将 decoder-only Transformer 的中间层（Body）进行递归迭代的架构，只需训练时用 2 次迭代，推理时即可通过增加迭代次数实现 upward generalization，在语言建模困惑度和 zero-shot ICL 任务上均超过标准 Transformer，且可作为标准 Transformer 的 drop-in 替代。

**[Adaptive Dual Reasoner: Large Reasoning Models Can Think Efficiently by Hybrid Reasoning](adaptive_dual_reasoner_large_reasoning_models_can_think_efficiently_by_hybrid_re.md)**

:   提出 Adaptive Dual Reasoner (ADR)——让推理模型在 fast thinking（简单推理步骤压缩）和 slow thinking（复杂推理步骤保留深度）之间动态切换，通过 SFT 冷启动 + EHPO（熵引导混合策略优化）训练，在数学推理基准上准确率提升最高 6.1% 同时推理 token 减少 49.5%-59.3%。

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

**[Co-Evolving LLM Coder and Unit Tester via Reinforcement Learning](co-evolving_llm_coder_and_unit_tester_via_reinforcement_learning.md)**

:   提出CURE框架，通过单元测试生成器与代码生成器的相互监督和共同进化，在无需ground-truth代码的情况下显著提升LLM代码生成能力。

**[Controlling Thinking Speed in Reasoning Models](controlling_thinking_speed_in_reasoning_models.md)**

:   通过表示工程（Representation Engineering）从 LRM 的隐藏空间中提取控制快/慢思考转换的 steering vector，结合基于层间 logit 散度的实时推理难度估计，实现无需训练的自适应推理速度调节，在 4 个 LRM 上平均提升 +1.3% 准确率并减少 -8.6% token 使用。

**[Cooperative Retrieval-Augmented Generation for Question Answering: Mutual Information Exchange and Ranking by Contrasting Layers](cooperative_retrieval-augmented_generation_for_question_answering_mutual_informa.md)**

:   提出CoopRAG框架，通过问题展开、基于检索器层对比的重排、以及推理链补全，实现检索器与LLM的双向合作，在多跳QA上超越HippoRAG2 5.3%，单跳QA上提升35.2%。

**[CoT Red-Handed: Stress Testing Chain-of-Thought Monitoring](cot_redhanded_stress_testing_chainofthought_monitoring.md)**

:   在 AI Control 框架下系统评估了 Chain-of-Thought 监控的有效性：发现 CoT 监控在检测微妙破坏行为上比仅监控 action 更有效（+10pp），但在检测明显破坏行为时反而更差（-25pp，因为推理中的伪合理化会欺骗监控），提出 hybrid 监控协议（独立评分 CoT 和 action 后加权）在所有场景下一致优于两种单一监控，检测率提升 2 倍。

**[Curriculum Abductive Learning](curriculum_abductive_learning.md)**

:   提出 Curriculum Abductive Learning (C-ABL)，通过将知识库按依赖结构分割为子知识库并渐进式引入训练，大幅缩小 ABL 的 abduction 搜索空间，显著提升训练稳定性、收敛速度和最终精度。

**[DisCO: Reinforcing Large Reasoning Models with Discriminative Constrained Optimization](disco_reinforcing_large_reasoning_models_with_discriminative_constrained_optimiz.md)**

:   分析 GRPO 目标函数，揭示其固有的难度偏差（对过难/过易问题赋予过低权重）和熵不稳定性问题，提出基于判别学习的 DisCO 框架，通过无裁剪评分函数、平方铰链约束优化和 DRO 处理不平衡 rollout，在 1.5B 模型上平均超过 GRPO 7%、超过 DAPO 6%。

**[Does Thinking More Always Help? Mirage of Test-Time Scaling in Reasoning Models](does_thinking_more_always_help_mirage_of_test-time_scaling_in_reasoning_models.md)**

:   通过系统实验揭示 LRM 测试时扩展（反复 "Wait" 提示延长推理）的性能呈先升后降的非单调趋势，用概率模型证明这种"提升"只是方差增大导致的海市蜃楼而非真正推理能力提升，并提出 parallel thinking 策略在相同 token 预算下准确率提升最高 22%。

**[GPO: Learning from Critical Steps to Improve LLM Reasoning](gpo_learning_from_critical_steps_to_improve_llm_reasoning.md)**

:   GPO 通过蒙特卡洛模拟估计推理轨迹中每一步的优势函数，识别出"关键步骤"（模型犯错的转折点），然后从关键步骤重置并重新采样轨迹用于训练，可以即插即用地提升 PPO、DPO、KTO、SimPO、ORPO 等多种优化算法在推理任务上的表现。

**[笔记1: CoT是幻觉吗？数据分布角度](is_chain-of-thought_reasoning_of_llms_a_mirage_a_data_distribution_lens.md)**

:   通过构建完全可控的抽象环境DataAlchemy，本文揭示CoT推理是一种幻觉——其有效性完全由训练数据分布主导，在分布外场景表现极其脆弱。

**[Know What You Don't Know: Uncertainty Calibration of Process Reward Models](know_what_you_dont_know_uncertainty_calibration_of_process_reward_models.md)**

:   本文提出了一种基于分位数回归的PRM校准方法，使PRM输出的分数更准确地反映LLM实际推理成功概率，并基于校准后的PRM设计了实例自适应推理时缩放（IAS）策略，在保持准确率的同时显著降低推理成本。

**[Let LRMs Break Free from Overthinking via Self-Braking Tuning](let_lrms_break_free_from_overthinking_via_self-braking_tuning.md)**

:   提出 Self-Braking Tuning (SBT) 框架，通过识别推理轨迹中的过度思考模式并构造自适应长度训练数据，使大型推理模型（LRM）学会自主判断何时停止推理，在数学推理任务上减少 30%-60% token 消耗的同时保持精度。

**[Let Me Think! A Long Chain-of-Thought Can Be Worth Exponentially Many Short Ones](let_me_think_a_long_chainofthought_can_be_worth_exponentiall.md)**

:   本文从理论和实验两方面证明：存在推理任务（图连通性问题），其中一条长 CoT（顺序缩放）的能力等价于指数多条短 CoT（并行缩放）——即将 CoT 长度减少一点点，就需要指数级增加并行采样数才能达到同等准确率。

**[On Learning Verifiers and Implications to Chain-of-Thought Reasoning](on_learning_verifiers_and_implications_to_chain-of-thought_reasoning.md)**

:   从PAC学习角度系统研究CoT验证器的可学习性，在不同验证目标下给出样本复杂度的上下界，并揭示验证与生成之间的有趣计算关系。

**[ProofSketch: Efficient Verified Reasoning for Large Language Models](proofsketch_efficient_verified_reasoning_for_large_language_models.md)**

:   提出 ProofSketch 框架，通过符号闭包前向推理+短sketch生成+形式验证的多阶段pipeline，在降低token用量的同时提供逻辑推理的形式化正确性保证。

**[Provable Scaling Laws for the Test-Time Compute of Large Language Models](provable_scaling_laws_for_the_testtime_compute_of_large_lang.md)**

:   提出两种具有可证明缩放律的测试时计算算法——Knockout（淘汰赛式：生成多个候选再两两比较淘汰）和 League（联赛式：用平均胜率选最优候选），证明在 LLM 生成正确解概率 >0 且比较能力优于随机的极弱假设下，失败概率随测试时计算增加呈指数或幂律衰减，且仅需黑盒 LLM 无需额外验证器。

**[RealMath: A Continuous Benchmark for Evaluating Language Models on Research-Level Mathematics](realmath_a_continuous_benchmark_for_evaluating_language_models_on_research-level.md)**

:   提出 RealMath，一个从 arXiv 论文和 Math StackExchange 中自动提取可验证数学问题的**可持续刷新**基准，用于评估 LLM 在真实研究级数学任务上的能力。

**[ReasonFlux-PRM: Trajectory-Aware PRMs for Long Chain-of-Thought Reasoning in LLMs](reasonfluxprm_trajectoryaware_prms_for_long_chainofthought_r.md)**

:   ReasonFlux-PRM 发现现有 PRM 无法有效评估推理模型的中间思考轨迹（trajectory），提出融合步骤级对齐/质量/连贯性分数和轨迹级模板引导奖励的 trajectory-aware PRM，在离线数据选择（SFT +12.1%）、在线 RL 奖励（+4.5%）和测试时 Best-of-N 缩放（+6.3%）三个场景中均显著优于包括 Qwen2.5-Math-PRM-72B 在内的强基线。

**[Reasoning by Superposition: A Theoretical Perspective on Chain of Continuous Thought](reasoning_by_superposition_a_theoretical_perspective_on_chain_of_continuous_thou.md)**

:   本文从理论上证明了连续思维链（Coconut）在有向图可达性问题上的表达优势：两层Transformer使用D步连续思维即可解决直径为D的图可达性问题，而离散CoT需要O(n²)步，其核心机制是连续思维向量以"叠加态"同时编码多条搜索前沿，实现隐式并行BFS。

**[Rethinking Optimal Verification Granularity for Compute-Efficient Test-Time Scaling](rethinking_optimal_verification_granularity_for_compute-efficient_test-time_scal.md)**

:   提出 Variable Granularity Search (VG-Search)，通过可调的验证粒度参数 $g$ 统一 Beam Search 和 Best-of-N，发现传统每步验证是次优的，自适应调整 $g$ 可在提升准确率3%+的同时减少52%+的计算量。

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

**[Stop Summation: Min-Form Credit Assignment Is All Process Reward Model Needs for Reasoning](stop_summation_minform_credit_assignment_is_all_process_rewa.md)**

:   PURE 发现 PRM 导致 reward hacking 的根本原因是 RL 中标准的 sum-form 信用分配（$V(s) = \sum \gamma^t r_t$），并提出 min-form 替代方案（$V(s) = \min_{t' \geq t} r_{t'}$），通过将价值函数限制为未来奖励的最小值而非累积和，显著缓解 reward hacking——仅用 30% 训练步数就达到与规则奖励方法相当的推理性能。

**[The Virtues of Brevity: Avoid Overthinking in Parallel Test-Time Reasoning](the_virtues_of_brevity_avoid_overthinking_in_parallel_test-time_reasoning.md)**

:   证明选择最短答案是一个简单但有效的Best-of-N启发式方法，通过避免过度思考regime大幅降低计算成本，性能与自一致性可比或更优，在推理模型中表现特别突出。

**[TIME: A Multi-level Benchmark for Temporal Reasoning of LLMs in Real-World Scenarios](time_a_multilevel_benchmark_for_temporal_reasoning_of_llms_i.md)**

:   TIME 提出一个面向真实世界时序推理的多层级 benchmark，覆盖 38,522 个 QA、3 个子数据集与 11 个细粒度子任务，系统刻画 LLM 在高密度时间信息、快速事件变化和复杂社会时序依赖下的推理能力，并分析了 test-time scaling 对 temporal reasoning 的实际影响。

**[Towards Thinking-Optimal Scaling of Test-Time Compute for LLM Reasoning](towards_thinking-optimal_scaling_of_test-time_compute_for_llm_reasoning.md)**

:   揭示了过度延长 CoT 长度会损害 LLM 推理性能，并提出 Thinking-Optimal Scaling (TOPS) 策略，让模型为每道题选择最短正确响应进行自我提升，在效果和效率上同时优于现有蒸馏方法。

**[Unlabeled Data Can Provably Enhance In-Context Learning of Transformers](unlabeled_data_can_provably_enhance_in-context_learning_of_transformers.md)**

:   提出增强型ICL框架，在prompt中同时包含少量标记样本和大量无标记样本，理论证明多层Transformer通过CoT可模拟EM算法从无标记数据中提取信息，将分类excess risk从 $\mathcal{O}(1/\sqrt{N})$ 改进到 $\mathcal{O}(1/\sqrt{N + \text{poly}(M)})$。

**[笔记6：Self-Evaluating LLMs - 多步任务的步级置信度估计](value-guided_search_for_efficient_chain-of-thought_reasoning.md)**

:   本文扩展置信度估计到多步任务，证明步级评估相比整体评估能更有效地检测推理失败，相对整体评估在CoQA上AUC-ROC提升15%，为多步推理系统的可信部署提供实用框架。

**[Visual Thoughts: A Unified Perspective of Understanding Multimodal Chain-of-Thought](visual_thoughts_a_unified_perspective_of_understanding_multi.md)**

:   首次从统一视角揭示多模态CoT工作的核心机制——"视觉思维"(Visual Thoughts)：MCoT通过将视觉信息缓存为中间推理步骤来增强LVLM推理，类似于计算机系统中的cache vs外部存储；定义了四种视觉思维表达形式（自然语言/结构化语言/编辑图像/生成图像），发现其有效性取决于表达的清晰性和简洁性。
