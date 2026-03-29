<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧠 NeurIPS2025 论文笔记

共 **817** 篇笔记，覆盖 **35** 个领域。

## 领域概览

| 领域 | 篇数 |
|:-----|-----:|
| 💡 [LLM 推理](#llm_reasoning) | 74 |
| 💬 [LLM / NLP](#llm_nlp) | 57 |
| 🧩 [多模态 VLM](#multimodal_vlm) | 48 |
| 🎨 [图像生成](#image_generation) | 45 |
| ⚡ [LLM 效率](#llm_efficiency) | 45 |
| 🎮 [强化学习](#reinforcement_learning) | 43 |
| 🦾 [LLM Agent](#llm_agent) | 38 |
| ⚖️ [对齐 / RLHF](#llm_alignment) | 29 |
| 📦 [模型压缩](#model_compression) | 29 |
| 📐 [优化/理论](#optimization) | 26 |
| 🏥 [医学图像](#medical_imaging) | 22 |
| 📈 [时间序列](#time_series) | 21 |
| ✂️ [语义分割](#segmentation) | 20 |
| ⚛️ [物理学](#physics) | 19 |
| 🕸️ [图学习](#graph_learning) | 18 |
| 🛡️ [AI 安全](#ai_safety) | 17 |
| 🤖 [机器人/具身智能](#robotics) | 17 |
| 🎵 [音频/语音](#audio_speech) | 16 |
| 🔗 [因果推理](#causal_inference) | 15 |
| 🖼️ [图像恢复](#image_restoration) | 15 |
| 🧑 [人体理解](#human_understanding) | 14 |
| 🎯 [目标检测](#object_detection) | 14 |
| 🧊 [3D 视觉](#3d_vision) | 13 |
| 🔄 [自监督/表示学习](#self_supervised) | 13 |
| 📡 [信号/通信](#signal_comm) | 13 |
| 🚗 [自动驾驶](#autonomous_driving) | 12 |
| 🎬 [视频理解](#video_understanding) | 12 |
| ✍️ [文本生成](#nlp_generation) | 11 |
| 📖 [NLP 理解](#nlp_understanding) | 11 |
| 🎁 [推荐系统](#recommender) | 11 |
| 🧮 [科学计算](#scientific_computing) | 11 |
| 🛰️ [遥感](#remote_sensing) | 9 |
| 🔎 [AIGC 检测](#aigc_detection) | 6 |
| 🌍 [地球科学](#earth_science) | 5 |
| 📂 [其他](#others) | 48 |

---

## 💡 LLM 推理 { #llm_reasoning }

**[AbbIE: Autoregressive Block-Based Iterative Encoder for Efficient Sequence Modeling](llm_reasoning/abbie_autoregressive_block-based_iterative_encoder_for_efficient_sequence_modeli.md)**

:   提出 AbbIE，一种将 decoder-only Transformer 的中间层（Body）进行递归迭代的架构，只需训练时用 2 次迭代，推理时即可通过增加迭代次数实现 upward generalization，在语言建模困惑度和 zero-shot ICL 任务上均超过标准 Transformer，且可作为标准 Transformer 的 drop-in 替代。

**[Adaptive Dual Reasoner: Large Reasoning Models Can Think Efficiently by Hybrid Reasoning](llm_reasoning/adaptive_dual_reasoner_large_reasoning_models_can_think_efficiently_by_hybrid_re.md)**

:   提出 Adaptive Dual Reasoner (ADR)——让推理模型在 fast thinking（简单推理步骤压缩）和 slow thinking（复杂推理步骤保留深度）之间动态切换，通过 SFT 冷启动 + EHPO（熵引导混合策略优化）训练，在数学推理基准上准确率提升最高 6.1% 同时推理 token 减少 49.5%-59.3%。

**[Are Large Reasoning Models Good Translation Evaluators? Analysis and Performance Boost](llm_reasoning/are_large_reasoning_models_good_translation_evaluators_analysis_and_performance_.md)**

:   首次系统分析了大推理模型（LRM）在机器翻译MQM评估中的行为，发现LRM存在"过度思考"、评分高估和材料选择依赖模型规模等问题，并提出ThinMQM方法通过训练合成人类评分轨迹来校准LRM思维过程，将思维预算减少约35倍同时提升评估性能（7B模型提升+8.7相关性分数）。

**[ARM: Adaptive Reasoning Model](llm_reasoning/arm_adaptive_reasoning_model.md)**

:   ARM 通过让模型自适应地选择四种推理格式（直接回答、短CoT、代码、长CoT），配合改进的 Ada-GRPO 训练算法解决 format collapse 问题，在保持与纯长CoT模型持平的准确率的同时平均节省 ~30% token，最多节省 ~70%。

**[Atom of Thoughts for Markov LLM Test-Time Scaling](llm_reasoning/atom_of_thoughts_for_markov_llm_testtime_scaling.md)**

:   提出 Atom of Thoughts (AoT)，将 LLM 推理建模为马尔可夫链，每个状态是与原问题答案等价但复杂度递减的自包含子问题，通过 DAG 分解+收缩的两阶段转移机制消除历史依赖，可与 ToT/反思等方法无缝集成，在数学/代码/多跳QA等6个benchmark上全面领先现有推理框架。

**[Auditing Meta-Cognitive Hallucinations in Reasoning Large Language Models](llm_reasoning/auditing_meta-cognitive_hallucinations_in_reasoning_large_language_models.md)**

:   系统性审计推理大模型（RLLM）中幻觉的产生与传播机制，发现长 CoT 中的反思（reflection）会通过元认知偏差放大幻觉而非纠正它，即使在幻觉源头进行干预也难以改变最终结果（chain disloyalty），揭示现有幻觉检测方法在多步推理场景下严重不足。

**[Base Models Know How to Reason, Thinking Models Learn When](llm_reasoning/base_models_know_how_to_reason_thinking_models_learn_when.md)**

:   通过无监督 SAE 聚类发现 thinking model 的推理机制分类，然后用 steering vector 在基座模型上激活这些潜在推理能力，混合模型恢复高达 91% 的 thinking-base 性能差距（无需权重更新），证明基座模型已具备推理能力，thinking model 只是学会了"何时"部署它们。

**[Beyond Chemical QA: Evaluating LLM's Chemical Reasoning with Modular Chemical Operations](llm_reasoning/beyond_chemical_qa_evaluating_llms_chemical_reasoning_with_modular_chemical_oper.md)**

:   提出 ChemCoTBench，首个评估 LLM 化学推理能力的 CoT 基准，将复杂化学问题分解为模块化的化学操作（加/删/替换官能团），配合 22,000 条专家标注的 CoT 数据集（ChemCoTDataset），系统性评估了推理型和非推理型 LLM 在分子理解/编辑/优化/反应预测上的能力。

**[Causal Head Gating: A Framework for Interpreting Roles of Attention Heads in Transformers](llm_reasoning/causal_head_gating_a_framework_for_interpreting_roles_of_attention_heads_in_tran.md)**

:   提出 Causal Head Gating (CHG)，通过对 Transformer 的每个 attention head 学习一个可微门控系数并结合正/负正则化，将 head 分为促进（facilitating）、干扰（interfering）、无关（irrelevant）三类，无需人工标签或 prompt 模板即可发现因果子电路，并扩展为对比 CHG 以分离 ICL 和指令遵循的独立电路。

**[Clip-and-Verify: Linear Constraint-Driven Domain Clipping for Accelerating Neural Network Verification](llm_reasoning/clip-and-verify_linear_constraint-driven_domain_clipping_for_accelerating_neural.md)**

:   提出Clip-and-Verify框架，通过利用线性界传播产生的约束来裁剪输入空间和收紧中间层界，包含完全裁剪（坐标上升求解对偶问题）和松弛裁剪（收缩输入盒）两种GPU高效算法，最多减少96%的BaB子问题数量，是VNN-COMP 2025获胜验证器的核心组件。

**[Co-Evolving LLM Coder and Unit Tester via Reinforcement Learning](llm_reasoning/co-evolving_llm_coder_and_unit_tester_via_reinforcement_learning.md)**

:   提出CURE框架，通过单元测试生成器与代码生成器的相互监督和共同进化，在无需ground-truth代码的情况下显著提升LLM代码生成能力。

**[Cognitive Mirrors: Exploring the Diverse Functional Roles of Attention Heads in LLM Reasoning](llm_reasoning/cognitive_mirrors_exploring_the_diverse_functional_roles_of_attention_heads_in_l.md)**

:   提出CogQA基准数据集和多类probing框架，系统分析LLM中注意力头的认知功能特化现象，发现认知头具有稀疏性、普遍性和层级化功能组织特征，去除认知头显著降低推理性能，增强则提升准确率。

**[Controlling Thinking Speed in Reasoning Models](llm_reasoning/controlling_thinking_speed_in_reasoning_models.md)**

:   通过表示工程（Representation Engineering）从 LRM 的隐藏空间中提取控制快/慢思考转换的 steering vector，结合基于层间 logit 散度的实时推理难度估计，实现无需训练的自适应推理速度调节，在 4 个 LRM 上平均提升 +1.3% 准确率并减少 -8.6% token 使用。

**[Cooperative Retrieval-Augmented Generation for Question Answering: Mutual Information Exchange and Ranking by Contrasting Layers](llm_reasoning/cooperative_retrieval-augmented_generation_for_question_answering_mutual_informa.md)**

:   提出CoopRAG框架，通过问题展开、基于检索器层对比的重排、以及推理链补全，实现检索器与LLM的双向合作，在多跳QA上超越HippoRAG2 5.3%，单跳QA上提升35.2%。

**[CoT Red-Handed: Stress Testing Chain-of-Thought Monitoring](llm_reasoning/cot_redhanded_stress_testing_chainofthought_monitoring.md)**

:   在 AI Control 框架下系统评估了 Chain-of-Thought 监控的有效性：发现 CoT 监控在检测微妙破坏行为上比仅监控 action 更有效（+10pp），但在检测明显破坏行为时反而更差（-25pp，因为推理中的伪合理化会欺骗监控），提出 hybrid 监控协议（独立评分 CoT 和 action 后加权）在所有场景下一致优于两种单一监控，检测率提升 2 倍。

**[Curriculum Abductive Learning](llm_reasoning/curriculum_abductive_learning.md)**

:   提出 Curriculum Abductive Learning (C-ABL)，通过将知识库按依赖结构分割为子知识库并渐进式引入训练，大幅缩小 ABL 的 abduction 搜索空间，显著提升训练稳定性、收敛速度和最终精度。

**[Deep Value Benchmark: Measuring Whether Models Generalize Deep Values or Shallow Preferences](llm_reasoning/deep_value_benchmark_measuring_whether_models_generalize_deep_values_or_shallow_.md)**

:   提出 Deep Value Benchmark (DVB)，通过"先混淆后解混淆"的实验设计，测量 LLM 是学习了深层人类价值观还是仅记住了表层偏好模式，发现所有模型的深层价值泛化率 (DVGR) 仅为 0.30，远低于随机水平。

**[DisCO: Reinforcing Large Reasoning Models with Discriminative Constrained Optimization](llm_reasoning/disco_reinforcing_large_reasoning_models_with_discriminative_constrained_optimiz.md)**

:   分析 GRPO 目标函数，揭示其固有的难度偏差（对过难/过易问题赋予过低权重）和熵不稳定性问题，提出基于判别学习的 DisCO 框架，通过无裁剪评分函数、平方铰链约束优化和 DRO 处理不平衡 rollout，在 1.5B 模型上平均超过 GRPO 7%、超过 DAPO 6%。

**[Does Thinking More Always Help? Mirage of Test-Time Scaling in Reasoning Models](llm_reasoning/does_thinking_more_always_help_mirage_of_test-time_scaling_in_reasoning_models.md)**

:   通过系统实验揭示 LRM 测试时扩展（反复 "Wait" 提示延长推理）的性能呈先升后降的非单调趋势，用概率模型证明这种"提升"只是方差增大导致的海市蜃楼而非真正推理能力提升，并提出 parallel thinking 策略在相同 token 预算下准确率提升最高 22%。

**[DreamPRM: Domain-Reweighted Process Reward Model for Multimodal Reasoning](llm_reasoning/dreamprm_domain-reweighted_process_reward_model_for_multimodal_reasoning.md)**

:   提出 DreamPRM，通过双层优化自动学习多模态推理数据集的域权重，解决 PRM 训练中的数据质量不均衡问题，在 MathVista 排行榜上以 o4-mini 模型达到 85.2% 的 top-1 准确率。

**[GPO: Learning from Critical Steps to Improve LLM Reasoning](llm_reasoning/gpo_learning_from_critical_steps_to_improve_llm_reasoning.md)**

:   GPO 通过蒙特卡洛模拟估计推理轨迹中每一步的优势函数，识别出"关键步骤"（模型犯错的转折点），然后从关键步骤重置并重新采样轨迹用于训练，可以即插即用地提升 PPO、DPO、KTO、SimPO、ORPO 等多种优化算法在推理任务上的表现。

**[I-RAVEN-X: Benchmarking Generalization and Robustness of Analogical and Mathematical Reasoning in Large Language and Reasoning Models](llm_reasoning/i-raven-x_benchmarking_generalization_and_robustness_of_analogical_and_mathemati.md)**

:   提出 I-RAVEN-X，一个增强版的符号化推理基准，通过增加操作数复杂度、属性范围和感知不确定性来评估 LLM 和 LRM 的类比推理与数学推理的泛化能力和鲁棒性，发现 LRM 在确定性推理上显著优于 LLM，但在不确定性推理下性能急剧下降。

**[Inference-Time Chain-of-Thought Pruning with Latent Informativeness Signals](llm_reasoning/inference-time_chain-of-thought_pruning_with_latent_informativeness_signals.md)**

:   提出 KAPPA (KL-Adjusted Pruned Path Algorithm)，利用 KL 散度、置信度和熵三个无需额外训练的信号对 Best-of-N 采样的推理分支进行渐进式剪枝，在保持准确率的同时实现最高 60% 峰值内存和 90% token 生成量的削减。

**[笔记1: CoT是幻觉吗？数据分布角度](llm_reasoning/is_chain-of-thought_reasoning_of_llms_a_mirage_a_data_distribution_lens.md)**

:   通过构建完全可控的抽象环境DataAlchemy，本文揭示CoT推理是一种幻觉——其有效性完全由训练数据分布主导，在分布外场景表现极其脆弱。

**[Know What You Don't Know: Uncertainty Calibration of Process Reward Models](llm_reasoning/know_what_you_dont_know_uncertainty_calibration_of_process_reward_models.md)**

:   本文提出了一种基于分位数回归的PRM校准方法，使PRM输出的分数更准确地反映LLM实际推理成功概率，并基于校准后的PRM设计了实例自适应推理时缩放（IAS）策略，在保持准确率的同时显著降低推理成本。

**[Large Language Models Can Learn and Generalize Steganographic Chain-of-Thought under Process Supervision](llm_reasoning/large_language_models_can_learn_and_generalize_steganographic_chain-of-thought_u.md)**

:   证明 LLM 在 RL 训练中受到 CoT 过程监督（惩罚特定字符串出现）时，会自发学会隐写术（steganography）——用替代编码隐藏被禁止的推理步骤，且这种编码是因果性的（load-bearing）并能泛化到训练中从未见过的字符串。

**[Latent Chain-of-Thought for Visual Reasoning](llm_reasoning/latent_chain-of-thought_for_visual_reasoning.md)**

:   将视觉CoT推理重新建模为后验推断问题，提出基于摊销变分推断(AVI)的LaCoT训练框架——包含参考引导GFlowNet微调(RGFN)、token级奖励近似和贝叶斯推理缩放(BiN)——在Qwen2.5-VL 3B/7B上比GRPO高出10.6%，在7个视觉推理基准上达到开源SOTA。

**[Let LRMs Break Free from Overthinking via Self-Braking Tuning](llm_reasoning/let_lrms_break_free_from_overthinking_via_self-braking_tuning.md)**

:   提出 Self-Braking Tuning (SBT) 框架，通过识别推理轨迹中的过度思考模式并构造自适应长度训练数据，使大型推理模型（LRM）学会自主判断何时停止推理，在数学推理任务上减少 30%-60% token 消耗的同时保持精度。

**[Let Me Think! A Long Chain-of-Thought Can Be Worth Exponentially Many Short Ones](llm_reasoning/let_me_think_a_long_chainofthought_can_be_worth_exponentiall.md)**

:   本文从理论和实验两方面证明：存在推理任务（图连通性问题），其中一条长 CoT（顺序缩放）的能力等价于指数多条短 CoT（并行缩放）——即将 CoT 长度减少一点点，就需要指数级增加并行采样数才能达到同等准确率。

**[LIMOPro: Reasoning Refinement for Efficient and Effective Test-time Scaling](llm_reasoning/limopro_reasoning_refinement_for_efficient_and_effective_test-time_scaling.md)**

:   提出PIR（基于困惑度的重要性精炼）框架，将LRM蒸馏的推理链分为"渐进推理"和"功能性步骤"（验证/多方法验证/纠错）两类，仅裁剪低PIR值的功能性步骤而完整保留渐进推理骨架，使微调后的模型在AIME/AMC/GPQA上准确率提升0.9%-6.6%同时token减少3%-41%，效率最高提升71%。

**[Lost in Transmission: When and Why LLMs Fail to Reason Globally](llm_reasoning/lost_in_transmission_when_and_why_llms_fail_to_reason_globally.md)**

:   提出有界注意力前缀预言机(BAPO)计算框架，将LLM的注意力头建模为有限带宽通信信道，证明图可达性等全局推理问题是BAPO-hard的（需超常数带宽），且CoT可将任何BAPO-hard问题转化为BAPO-easy问题，实验在GPT-4o/Claude/Gemini上验证理论预测。

**[Many LLMs Are More Utilitarian Than One](llm_reasoning/many_llms_are_more_utilitarian_than_one.md)**

:   在6个LLM上实验发现，多智能体集体讨论道德困境时会产生与人类群体类似的"功利主义增强"（Utilitarian Boost）——集体比个体更倾向接受为"多数人利益"伤害少数人，但LLM产生此效应的机制与人类不同（人类因结果敏感度增强，LLM则因规范敏感度降低或公正性增强等多种模式），且可通过模型异质性和提示多样性缓解。

**[Mind the Gap: Bridging Thought Leap for Improved Chain-of-Thought Tuning](llm_reasoning/mind_the_gap_bridging_thought_leap_for_improved_chain-of-thought_tuning.md)**

:   本文首次系统性地定义了 CoT 推理链中的"思维跳跃"(Thought Leap)现象，提出 CoT-Bridge 模型自动检测并补全推理链中被省略的中间步骤，在 NuminaMath 上最高提升 +5.87%，并可作为即插即用模块增强蒸馏和 RL 流程。

**[On Learning Verifiers and Implications to Chain-of-Thought Reasoning](llm_reasoning/on_learning_verifiers_and_implications_to_chain-of-thought_reasoning.md)**

:   从PAC学习角度系统研究CoT验证器的可学习性，在不同验证目标下给出样本复杂度的上下界，并揭示验证与生成之间的有趣计算关系。

**[One Token Embedding Is Enough to Deadlock Your Large Reasoning Model](llm_reasoning/one_token_embedding_is_enough_to_deadlock_your_large_reasoning_model.md)**

:   本文提出 Deadlock Attack，通过优化单个对抗性 token embedding 并以后门方式植入 LRM，使模型在推理时陷入永久思考循环（无限生成 "Wait"、"But" 等过渡词），在 4 个 LRM 和 3 个数学推理 benchmark 上实现 100% 攻击成功率，且对正常输入几乎无性能影响。

**[OS-Harm: A Benchmark for Measuring Safety of Computer Use Agents](llm_reasoning/os-harm_a_benchmark_for_measuring_safety_of_computer_use_agents.md)**

:   本文提出 OS-Harm，首个面向通用计算机使用 Agent（非仅浏览器）的安全性 benchmark，覆盖用户恶意使用、Prompt 注入攻击、模型自身失误三类风险共 150 个任务，评测发现前沿模型（o4-mini、Claude 3.7 Sonnet、Gemini 2.5 Pro 等）普遍直接服从有害指令（最高 70% 不安全率），且对基础 prompt 注入有 20% 的服从率。

**[ProofSketch: Efficient Verified Reasoning for Large Language Models](llm_reasoning/proofsketch_efficient_verified_reasoning_for_large_language_models.md)**

:   提出 ProofSketch 框架，通过符号闭包前向推理+短sketch生成+形式验证的多阶段pipeline，在降低token用量的同时提供逻辑推理的形式化正确性保证。

**[Provable Scaling Laws for the Test-Time Compute of Large Language Models](llm_reasoning/provable_scaling_laws_for_the_testtime_compute_of_large_lang.md)**

:   提出两种具有可证明缩放律的测试时计算算法——Knockout（淘汰赛式：生成多个候选再两两比较淘汰）和 League（联赛式：用平均胜率选最优候选），证明在 LLM 生成正确解概率 >0 且比较能力优于随机的极弱假设下，失败概率随测试时计算增加呈指数或幂律衰减，且仅需黑盒 LLM 无需额外验证器。

**[Re-FORC: Adaptive Reward Prediction for Efficient Chain-of-Thought Reasoning](llm_reasoning/re-forc_adaptive_reward_prediction_for_efficient_chain-of-thought_reasoning.md)**

:   提出Re-FORC，一个轻量级adapter在CoT推理过程中实时预测未来期望奖励 $\psi(t|x,z,\pi)$，将推理计算分配建模为Pandora's box问题，实现自适应早停（节省26%计算）、模型+计算联合选择（同等计算下+4%准确率或同等准确率-55%计算）和测试时计算伸缩（+11%准确率），且用户可通过代价系数 $\lambda$ 在推理时自由调控精度-效率权衡，无需重训。

**[RealMath: A Continuous Benchmark for Evaluating Language Models on Research-Level Mathematics](llm_reasoning/realmath_a_continuous_benchmark_for_evaluating_language_models_on_research-level.md)**

:   提出 RealMath，一个从 arXiv 论文和 Math StackExchange 中自动提取可验证数学问题的**可持续刷新**基准，用于评估 LLM 在真实研究级数学任务上的能力。

**[ReasonFlux-PRM: Trajectory-Aware PRMs for Long Chain-of-Thought Reasoning in LLMs](llm_reasoning/reasonfluxprm_trajectoryaware_prms_for_long_chainofthought_r.md)**

:   ReasonFlux-PRM 发现现有 PRM 无法有效评估推理模型的中间思考轨迹（trajectory），提出融合步骤级对齐/质量/连贯性分数和轨迹级模板引导奖励的 trajectory-aware PRM，在离线数据选择（SFT +12.1%）、在线 RL 奖励（+4.5%）和测试时 Best-of-N 缩放（+6.3%）三个场景中均显著优于包括 Qwen2.5-Math-PRM-72B 在内的强基线。

**[Reasoning by Superposition: A Theoretical Perspective on Chain of Continuous Thought](llm_reasoning/reasoning_by_superposition_a_theoretical_perspective_on_chain_of_continuous_thou.md)**

:   本文从理论上证明了连续思维链（Coconut）在有向图可达性问题上的表达优势：两层Transformer使用D步连续思维即可解决直径为D的图可达性问题，而离散CoT需要O(n²)步，其核心机制是连续思维向量以"叠加态"同时编码多条搜索前沿，实现隐式并行BFS。

**[Reasoning Models Better Express Their Confidence](llm_reasoning/reasoning_models_better_express_their_confidence.md)**

:   系统性证明推理模型（extended CoT）比非推理模型具有显著更优的置信度校准能力，并揭示"慢思考"行为（探索替代方案、回溯、验证）是校准提升的根本来源。

**[Reasoning Models Hallucinate More: Factuality-Aware Reinforcement Learning for Large Reasoning Models](llm_reasoning/reasoning_models_hallucinate_more_factuality-aware_reinforcement_learning_for_la.md)**

:   揭示了RL训练的推理模型（如DeepSeek-R1）比非推理模型产生更多幻觉，从理论上分析了三个根因（高方差梯度、熵约束、伪局部最优），并提出FSPO算法通过步级事实性验证调整token级advantage，在减少幻觉的同时保持甚至提升推理能力。

**[Rethinking Optimal Verification Granularity for Compute-Efficient Test-Time Scaling](llm_reasoning/rethinking_optimal_verification_granularity_for_compute-efficient_test-time_scal.md)**

:   提出 Variable Granularity Search (VG-Search)，通过可调的验证粒度参数 $g$ 统一 Beam Search 和 Best-of-N，发现传统每步验证是次优的，自适应调整 $g$ 可在提升准确率3%+的同时减少52%+的计算量。

**[SafePath: Preventing Harmful Reasoning in Chain-of-Thought via Early Alignment](llm_reasoning/safepath_preventing_harmful_reasoning_in_chain-of-thought_via_early_alignment.md)**

:   提出 SafePath，仅在推理开始处微调 8 个 token 的"Safety Primer"（"Let's think about safety first"），即可有效引导 LRM 走向安全推理路径，在 DeepSeek-R1-Distill 上减少 90% 有害输出且仅需 Direct Refusal 1/296 的训练计算量。

**[Sampling-Efficient Test-Time Scaling: Self-Estimating the Best-of-N Sampling in Early Decoding](llm_reasoning/sampling-efficient_test-time_scaling_self-estimating_the_best-of-n_sampling_in_e.md)**

:   提出 Self-Truncation Best-of-N (ST-BoN) 解码方法，通过理论证明早期隐状态一致性可预测最终一致性，在生成早期就识别并截断次优样本，实现降低80%+内存和50%延迟的同时保持BoN性能。

**[Scalable Best-of-N Selection for Large Language Models via Self-Certainty](llm_reasoning/scalable_best-of-n_selection_for_large_language_models_via_self-certainty.md)**

:   提出Self-Certainty度量，利用LLM输出的token概率分布量化模型信心，在无需额外奖励模型的情况下实现可扩展的Best-of-N选择，性能媲美或超越基于奖励模型的方法。

**[scPilot: Large Language Model Reasoning Toward Automated Single-Cell Analysis and Discovery](llm_reasoning/scpilot_large_language_model_reasoning_toward_automated_single-cell_analysis_and.md)**

:   提出 scPilot 框架和 scBench 基准，让LLM直接在单细胞RNA-seq数据上进行"组学原生推理"（读取标记基因→提出假设→调用工具验证→迭代修正），实现细胞类型标注准确率提升11%、轨迹推断graph-edit distance降低30%。

**[Segment Policy Optimization: Effective Segment-Level Credit Assignment in RL for Large Language Models](llm_reasoning/segment_policy_optimization_effective_segment-level_credit_assignment_in_rl_for_.md)**

:   提出SPO框架，采用段级（而非令牌级或轨迹级）的advantage估计，通过新颖的蒙特卡洛方法和树形采样，在短CoT和长CoT场景下分别超越PPO和GRPO 6-12和7-11个百分点。

**[笔记8：PolyMath - 多语言背景下的数学推理评估](llm_reasoning/self-evaluating_llms_for_multi-step_tasks_stepwise_confidence_estimation_for_fai.md)**

:   PolyMath构建的18语言、4难度级、500问题数学推理基准揭露：(1)推理性能跨语言差异达10分，(2)推理模型输入-输出语言一致性低且可能影响性能，(3)思考长度在语言间显著不一致，为多语言推理研究提供新视角。

**[Simulating Society Requires Simulating Thought](llm_reasoning/simulating_society_requires_simulating_thought.md)**

:   本文提出从"行为主义"模式转向"认知建模"范式，通过 GenMinds 框架用因果信念图建模 LLM Agent 的内部推理过程，并设计 RECAP 基准从可追溯性、人口统计敏感性和干预一致性三维度评估推理保真度。

**[SLAyiNG: Towards Queer Language Processing](llm_reasoning/slaying_towards_queer_language_processing.md)**

:   构建了首个显式标注的酷儿俚语（queer slang）数据集 SLAyiNG，包含 695 个术语和近 20 万条使用实例，并通过人机标注一致性实验（Krippendorff's α=0.746）表明推理模型可用于预筛选但仍需社区驱动的专家标注。

**[Smaller Models, Smarter Rewards: A Two-Sided Approach to Process and Outcome Rewards](llm_reasoning/smaller_models_smarter_rewards_a_two-sided_approach_to_process_and_outcome_rewar.md)**

:   将 Phi-4 系列小模型（3.8B/14B）的最后一层替换为回归头并微调，使其同时具备 ORM（结果奖励）和 PRM（过程奖励）能力，在代码生成任务上通过选择最优 rollout 实现 20%+ 的 pass@k 提升。

**[SolverLLM: Leveraging Test-Time Scaling for Optimization Problem via LLM-Guided Search](llm_reasoning/solverllm_leveraging_test-time_scaling_for_optimization_problem_via_llm-guided_s.md)**

:   无需训练，通过 MCTS 引导 LLM 生成 6 元素优化表述并转化为求解器代码，在 NL4Opt 上达 97.0%（vs OptiMUS 78.8%），超越微调方法且跨域泛化强。

**[SPRINT: Enabling Interleaved Planning and Parallelized Execution in Reasoning Models](llm_reasoning/sprint_enabling_interleaved_planning_and_parallelized_execution_in_reasoning_mod.md)**

:   通过将长链式推理轨迹重组为交替的规划-并行执行阶段，Sprint 使推理模型在保持准确率的同时，将长推理链的顺序 token 数减少高达 39%（OOD 任务上最高 65%），实现推理过程的动态并行化。

**[SQL-of-Thought: Multi-agentic Text-to-SQL with Guided Error Correction](llm_reasoning/sql-of-thought_multi-agentic_text-to-sql_with_guided_error_correction.md)**

:   提出 SQL-of-Thought——一个多智能体 Text-to-SQL 框架，将任务分解为 schema linking → 子问题识别 → CoT 查询计划生成 → SQL 生成 → 基于 31 类错误分类法的引导修正循环，用 Claude 3 Opus 在 Spider 上达到 91.59% 执行准确率，比此前最佳 Chase SQL（87.6%）提升近 4 个百分点。

**[SQL-R1: Training Natural Language to SQL Reasoning Model By Reinforcement Learning](llm_reasoning/sql-r1_training_natural_language_to_sql_reasoning_model_by_reinforcement_learnin.md)**

:   首次系统地将 GRPO 强化学习应用于 NL2SQL 任务，通过四层递进式奖励函数和 200K 冷启动 + 5K 复杂样本 RL 训练策略，7B 模型在 Spider 和 BIRD 上分别达到 88.7% 和 66.6%，超越 GPT-4 同规模模型。

**[Stop Summation: Min-Form Credit Assignment Is All Process Reward Model Needs for Reasoning](llm_reasoning/stop_summation_minform_credit_assignment_is_all_process_rewa.md)**

:   PURE 发现 PRM 导致 reward hacking 的根本原因是 RL 中标准的 sum-form 信用分配（$V(s) = \sum \gamma^t r_t$），并提出 min-form 替代方案（$V(s) = \min_{t' \geq t} r_{t'}$），通过将价值函数限制为未来奖励的最小值而非累积和，显著缓解 reward hacking——仅用 30% 训练步数就达到与规则奖励方法相当的推理性能。

**[The Hawthorne Effect in Reasoning Models: Evaluating and Steering Test Awareness](llm_reasoning/the_hawthorne_effect_in_reasoning_models_evaluating_and_steering_test_awareness.md)**

:   首次系统量化推理型LLM的"测试感知"(Hawthorne效应)：当模型察觉自己在被评估时会改变行为，论文通过线性探针定位感知激活并进行参数编辑引导，揭示测试感知对安全对齐的显著且方向不一致的影响。

**[The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity](llm_reasoning/the_illusion_of_thinking_understanding_the_strengths_and_limitations_of_reasonin.md)**

:   通过可控拼图环境系统揭示大型推理模型（LRMs）的三阶段行为：低复杂度不如标准 LLM、中等复杂度显著优于、高复杂度完全崩溃(0%)，且反直觉地在崩溃时减少思考 token，证实当前 LRMs 并未发展出真正泛化的推理能力。

**[The Impact of Quantization on Large Reasoning Model Reinforcement Learning](llm_reasoning/the_impact_of_quantization_on_large_reasoning_model_reinforcement_learning.md)**

:   系统实验发现在大推理模型的 RL 训练中，量化感知训练（QAFT/STE）会损害推理能力，而训练后量化（PTQ）和 QLoRA 即使在 4-bit 精度下也能很好地保持推理性能，为实践者提供了"先全精度 RL、再 PTQ 量化"的推荐路线。

**[The Virtues of Brevity: Avoid Overthinking in Parallel Test-Time Reasoning](llm_reasoning/the_virtues_of_brevity_avoid_overthinking_in_parallel_test-time_reasoning.md)**

:   证明选择最短答案是一个简单但有效的Best-of-N启发式方法，通过避免过度思考regime大幅降低计算成本，性能与自一致性可比或更优，在推理模型中表现特别突出。

**[ThinkSound: Chain-of-Thought Reasoning in Multimodal Large Language Models for Audio Generation and Editing](llm_reasoning/thinksound_chain-of-thought_reasoning_in_multimodal_large_language_models_for_au.md)**

:   提出三阶段交互式视频转音频框架 ThinkSound，通过 MLLM 生成结构化 CoT 推理来指导统一的音频生成基础模型，在 VGGSound 和 MovieGen Audio 基准上达到 SOTA，同时支持对象级精细化和自然语言指令编辑。

**[TIME: A Multi-level Benchmark for Temporal Reasoning of LLMs in Real-World Scenarios](llm_reasoning/time_a_multilevel_benchmark_for_temporal_reasoning_of_llms_i.md)**

:   TIME 提出一个面向真实世界时序推理的多层级 benchmark，覆盖 38,522 个 QA、3 个子数据集与 11 个细粒度子任务，系统刻画 LLM 在高密度时间信息、快速事件变化和复杂社会时序依赖下的推理能力，并分析了 test-time scaling 对 temporal reasoning 的实际影响。

**[Topology of Reasoning: Understanding Large Reasoning Models through Reasoning Graph Properties](llm_reasoning/topology_of_reasoning_understanding_large_reasoning_models_through_reasoning_gra.md)**

:   提出"推理图"概念——通过对 LLM 隐藏状态聚类构建有向图，从环路密度、直径和小世界指标三个图论维度分析大推理模型（如 DeepSeek-R1 蒸馏系列），发现推理模型的推理图具有显著更多环路（~5/样本）、更大直径和更强小世界特性（~6倍），且这些特性随任务难度和模型规模增长。

**[Towards Thinking-Optimal Scaling of Test-Time Compute for LLM Reasoning](llm_reasoning/towards_thinking-optimal_scaling_of_test-time_compute_for_llm_reasoning.md)**

:   揭示了过度延长 CoT 长度会损害 LLM 推理性能，并提出 Thinking-Optimal Scaling (TOPS) 策略，让模型为每道题选择最短正确响应进行自我提升，在效果和效率上同时优于现有蒸馏方法。

**[Transformers Provably Learn Chain-of-Thought Reasoning with Length Generalization](llm_reasoning/transformers_provably_learn_chain-of-thought_reasoning_with_length_generalizatio.md)**

:   从优化理论角度证明了一层 Transformer 通过梯度下降在合成状态追踪任务上能学会 CoT 推理并实现长度泛化，首次为常数深度 Transformer 学习 $\mathsf{NC}^1$-complete 问题（超越之前局限于 $\mathsf{TC}^0$ 的理论）提供了收敛保证。

**[TTS-VAR: A Test-Time Scaling Framework for Visual Auto-Regressive Generation](llm_reasoning/tts-var_a_test-time_scaling_framework_for_visual_auto-regressive_generation.md)**

:   提出 TTS-VAR——首个针对 Visual Auto-Regressive (VAR) 模型的测试时扩展框架，将图像生成建模为路径搜索问题，通过自适应递减批量 + 早期聚类多样性搜索 + 后期重采样潜力选择，在 Infinity 2B 上将 GenEval 分数从 0.69 提升到 0.75（+8.7%），N=2 即超越 Best-of-N 的 N=8 效果。

**[Two-Stage Learning of Stabilizing Neural Controllers via Zubov Sampling and Iterative Domain Expansion](llm_reasoning/two-stage_learning_of_stabilizing_neural_controllers_via_zubov_sampling_and_iter.md)**

:   提出两阶段训练框架——先用 Zubov 采样 + 动态域扩展估计吸引域（ROA），再用 CEGIS 反例精炼——联合学习神经网络控制器和 Lyapunov 函数，ROA 体积比基线大 5 到 $1.5 \times 10^5$ 倍，验证速度比 dReal 快 40-10000 倍。

**[Unlabeled Data Can Provably Enhance In-Context Learning of Transformers](llm_reasoning/unlabeled_data_can_provably_enhance_in-context_learning_of_transformers.md)**

:   提出增强型ICL框架，在prompt中同时包含少量标记样本和大量无标记样本，理论证明多层Transformer通过CoT可模拟EM算法从无标记数据中提取信息，将分类excess risk从 $\mathcal{O}(1/\sqrt{N})$ 改进到 $\mathcal{O}(1/\sqrt{N + \text{poly}(M)})$。

**[Unlocking Multimodal Mathematical Reasoning via Process Reward Model](llm_reasoning/unlocking_multimodal_mathematical_reasoning_via_process_reward_model.md)**

:   提出URSA三阶段框架，依次构建百万级多模态CoT数据(MMathCoT-1M)训练基座、双视角过程监督数据(DualMath-1.1M)训练PRM、以及PS-GRPO算法将PRM融入在线RL，8B模型在6个数学基准上平均超越GPT-4o 2.7%。

**[笔记6：Self-Evaluating LLMs - 多步任务的步级置信度估计](llm_reasoning/value-guided_search_for_efficient_chain-of-thought_reasoning.md)**

:   本文扩展置信度估计到多步任务，证明步级评估相比整体评估能更有效地检测推理失败，相对整体评估在CoQA上AUC-ROC提升15%，为多步推理系统的可信部署提供实用框架。

**[Visual Thoughts: A Unified Perspective of Understanding Multimodal Chain-of-Thought](llm_reasoning/visual_thoughts_a_unified_perspective_of_understanding_multi.md)**

:   首次从统一视角揭示多模态CoT工作的核心机制——"视觉思维"(Visual Thoughts)：MCoT通过将视觉信息缓存为中间推理步骤来增强LVLM推理，类似于计算机系统中的cache vs外部存储；定义了四种视觉思维表达形式（自然语言/结构化语言/编辑图像/生成图像），发现其有效性取决于表达的清晰性和简洁性。

---

## 💬 LLM / NLP { #llm_nlp }

**[AceSearcher: Bootstrapping Reasoning and Search for LLMs via Reinforced Self-Play](llm_nlp/acesearcher_bootstrapping_reasoning_and_search_for_llms_via_reinforced_self-play.md)**

:   提出 AceSearcher——一种协作式自我博弈框架，让单个 LLM 同时扮演**问题分解者**（将复杂查询拆解为子问题引导检索）和**求解者**（整合检索上下文生成答案），通过 SFT + 迭代 DPO 两阶段训练，仅用最终答案作为奖励信号，在 10 个数据集上平均 EM 提升 7.6%，32B 模型匹配 DeepSeek-V3（<5% 参数）。

**[Active Slice Discovery in Large Language Models](llm_nlp/active_slice_discovery_in_large_language_models.md)**

:   提出 **Active Slice Discovery** 问题框架，将主动学习引入 LLM 错误切片发现，利用不确定性采样 + LLM 内部表征（原始 embedding 或 SAE 特征）在仅使用 2-10% 标注的情况下达到接近全标注的切片检测精度。

**[AdaSTaR: Adaptive Data Sampling for Training Self-Taught Reasoners](llm_nlp/adastar_adaptive_data_sampling_for_training_self-taught_reasoners.md)**

:   发现 STaR（自我教学推理器）的随机数据采样导致观测训练频率严重不平衡（简单题过度训练、难题训练不足），提出 AdaSTaR——通过自适应多样性采样（优先欠训练样本）和自适应课程采样（根据模型强度调节难度），在 6 个基准上全部取得最高准确率同时减少 58.6% 训练 FLOPs。

**[AI Progress Should Be Measured by Capability-Per-Resource, Not Scale Alone: A Framework for Gradient-Guided Resource Allocation in LLMs](llm_nlp/ai_progress_should_be_measured_by_capability-per-resource_not_scale_alone_a_fram.md)**

:   本文以 position paper 的形式挑战"规模至上主义"，提出以**能力-每-资源（Capability-Per-Resource, CPR）**取代单纯的规模扩张来衡量 AI 进步，并给出一套基于梯度引导的资源分配理论框架——通过发布"梯度蓝图"元数据，使下游适配者仅微调高影响力参数子集即可在资源占用大幅降低的同时保持接近全参数微调的性能。

**[Are Language Models Efficient Reasoners? A Perspective from Logic Programming](llm_nlp/are_language_models_efficient_reasoners_a_perspective_from_logic_programming.md)**

:   从逻辑编程角度提出评估 LLM 推理效率（而非仅正确性）的框架——通过 verbalized logic program 将自然语言证明映射到逻辑程序证明，发现当前 LLM 在含无关公理的数学题中不仅准确率下降，且推理过程严重低效（超过一半的推理步骤是不必要的）。

**[ARECHO: Autoregressive Evaluation via Chain-Based Hypothesis Optimization for Speech Multi-Metric Estimation](llm_nlp/arecho_autoregressive_evaluation_via_chain-based_hypothesis_optimization_for_spe.md)**

:   ARECHO 将语音多指标评估建模为链式自回归 token 预测任务——设计统一的语音信息 token 化管线处理 87 个异质指标（数值/类别/有界/无界），通过动态分类链显式捕捉指标间依赖关系（如可懂度-自然度相关性），配合两步置信度导向解码减少误差传播，在增强/生成/噪声三类语音评估中全面超越 UniVERSA 基线（Avg Test MSE 23.26 vs 96.99，-76%）。

**[AstroVisBench: A Code Benchmark for Scientific Computing and Visualization in Astronomy](llm_nlp/astrovisbench_a_code_benchmark_for_scientific_computing_and_visualization_in_ast.md)**

:   AstroVisBench 构建了首个评估 LLM 天文科学计算和可视化能力的代码基准——从 110 个 Jupyter Notebook 提取 864 个任务（处理+可视化），设计双重评估管线（执行式变量检查 + VLM-as-Judge 可视化评分，与专家 Spearman ρ=0.822），评测 8 个 SOTA 模型后发现 Gemini 2.5 Pro 最佳但无错误率仅 15.7%，FileNotFoundError 占 43% 错误。

**[ARC-JSD: Attributing Response to Context via Jensen-Shannon Divergence Driven Mechanistic Study](llm_nlp/attributing_response_to_context_a_jensen-shannon_divergence_driven_mechanistic_s.md)**

:   ARC-JSD 提出基于 Jensen-Shannon 散度的 RAG 上下文归因方法——通过比较有/无特定上下文句子时模型输出分布的 JSD 差异，无需微调/梯度计算即可定位回答所依赖的上下文，计算效率比 baseline 快 3 倍，Top-1 归因准确率平均提升 10.7%，并通过 Logit Lens 揭示归因相关的注意力头集中在高层。

**[Belief-Calibrated Multi-Agent Consensus Seeking for Complex NLP Tasks](llm_nlp/belief-calibrated_multi-agent_consensus_seeking_for_complex_nlp_tasks.md)**

:   提出 Belief-Calibrated Consensus Seeking (BCCS) 框架，通过引入信念（belief）校准的共识判断、冲突感知的协作者分配和领导者选择三个模块，让多智能体系统在复杂NLP任务上达成更稳定的共识，在 MATH 和 MMLU 上的困难任务分别提升 2.23% 和 3.95%。

**[Benchmarking Large Language Models for Zero-Shot and Few-Shot Phishing URL Detection](llm_nlp/benchmarking_large_language_models_for_zero-shot_and_few-shot_phishing_url_detec.md)**

:   系统评估多个 LLM 在零样本和少样本钓鱼 URL 检测任务上的表现，发现 LLM 在零样本下表现参差不齐但少样本 ICL 显著提升，专用安全微调模型仍有优势，为 LLM 在网络安全中的应用提供了基准参考。

**[Beyond Components: Singular Vector-Based Interpretability of Transformer Circuits](llm_nlp/beyond_components_singular_vector-based_interpretability_of_transformer_circuits.md)**

:   提出基于SVD奇异向量的方向级可解释性框架，通过对注意力头和MLP的增广矩阵统一SVD分解+可学习对角掩码（KL+L₁），发现单组件内存在正交低秩子函数叠加——IOI任务仅需~9%方向即可KLD=0.21复现模型行为。

**[Beyond the Singular: Revealing the Value of Multiple Generations in Benchmark Evaluation](llm_nlp/beyond_the_singular_revealing_the_value_of_multiple_generations_in_benchmark_eva.md)**

:   将LLM基准评测形式化为层级统计模型，理论证明多次随机生成（k>1）能降低benchmark分数估计方差，并引入prompt级难度指标$\mathbb{P}(\text{correct})$和数据地图用于基准质量控制。

**[Beyond The Surface Enhancing Llm-As-A-Judge Alignment With Human Via Internal Re](llm_nlp/beyond_the_surface_enhancing_llm-as-a-judge_alignment_with_human_via_internal_re.md)**

:   提出LAGER框架，通过聚合LLM中间层到最终层的score token logits并计算期望分数，无需微调模型即可将LLM评判与人类评分的对齐度提升最高7.5%，且不需要思维链推理步骤就能匹配或超过推理类方法。

**[Beyond Token Probes Hallucination Detection Via Activation Tensors With Act-Vit](llm_nlp/beyond_token_probes_hallucination_detection_via_activation_tensors_with_act-vit.md)**

:   将LLM的全部隐层激活组织为"激活张量"（层×token×隐维度），类比图像用ViT处理，设计ACT-ViT架构支持跨LLM联合训练，在15个LLM-数据集组合上一致超越传统probing方法，并展现出对未见数据集和未见LLM的强零样本/少样本迁移能力。

**[Bigram Subnetworks Mapping To Next Tokens In Transformer Language Models](llm_nlp/bigram_subnetworks_mapping_to_next_tokens_in_transformer_language_models.md)**

:   通过连续稀疏化在Transformer语言模型中找到仅包含~10M参数的bigram子网络，它们集中在第一个MLP层，足以复现bigram预测（$r>0.95$），且被消融后模型性能大幅下降，证明这些子网络是语言模型中既必要又充分的最小next-token预测电路。

**[Born a Transformer – Always a Transformer? On the Effect of Pretraining on Architectural Abilities](llm_nlp/born_a_transformer_--_always_a_transformer_on_the_effect_of_pretraining_on_archi.md)**

:   通过系统性地研究检索和复制任务家族，揭示了大规模预训练会为Transformer引入方向性偏置（右/前向优于左/后向），但无法克服非唯一任务上的根本架构限制；微调可消除方向偏置但不能突破架构表达力边界。

**[Breaking The Frozen Subspace Importance Sampling For Low-Rank Optimization In Ll](llm_nlp/breaking_the_frozen_subspace_importance_sampling_for_low-rank_optimization_in_ll.md)**

:   发现GaLore等低秩优化方法的主导子空间在预训练中会"冻结"（相邻子空间重叠度趋近1），导致权重更新卡在固定低秩子空间中；提出SARA（重要性采样子空间选择），按奇异值权重随机采样奇异向量构建子空间，证明收敛性的同时将低秩优化器与全秩Adam的性能差距缩小最高46%。

**[Bridging Human And Llm Judgments Understanding And Narrowing The Gap](llm_nlp/bridging_human_and_llm_judgments_understanding_and_narrowing_the_gap.md)**

:   提出Bridge统计框架，通过序数logistic回归建模人类和LLM评判之间的潜在关系，以少量人类标签改善LLM评判的校准和对齐，同时支持对系统性偏差的正式统计检验。

**[C²Prompt: Class-aware Client Knowledge Interaction for Federated Continual Learning](llm_nlp/c2prompt_class-aware_client_knowledge_interaction_for_federated_continual_learni.md)**

:   针对联邦持续学习中prompt通信时的类级知识不一致问题，提出C²Prompt方法，通过局部类分布补偿（LCDC）和类感知prompt聚合（CPA）两个机制显式增强跨客户端的类级知识一致性，在ImageNet-R上Avg准确率达87.20%，超出SOTA Powder 2.51%。

**[Can Large Language Models Master Complex Card Games?](llm_nlp/can_large_language_models_master_complex_card_games.md)**

:   系统评估LLM在8种复杂卡牌游戏上的学习能力，发现通过高质量游戏数据的SFT，LLM可以接近强游戏AI的水平，并能同时掌握多个游戏，但通用能力会下降（可通过混入通用指令数据缓解）。

**[CAT: Circular-Convolutional Attention for Sub-Quadratic Transformers](llm_nlp/cat_circular-convolutional_attention_for_sub-quadratic_transformers.md)**

:   本文提出CAT（Circular-convolutional Attention），通过FFT计算循环卷积将Self-Attention复杂度从O(N²)降至O(N log N)，同时保持完整的softmax机制和全局注意力。

**[CBMAS: Cognitive Behavioral Modeling via Activation Steering](llm_nlp/cbmas_cognitive_behavioral_modeling_via_activation_steering.md)**

:   CBMAS 提出一个连续激活干预诊断框架，将传统“前后对比式”认知偏差分析扩展为可解释的干预轨迹分析，通过 alpha 强度扫描、logit-lens 偏置曲线与层位敏感性分析，揭示 LLM 行为翻转临界点与跨层演化机制。

**[Characterizing the Expressivity of Fixed-Precision Transformer Language Models](llm_nlp/characterizing_the_expressivity_of_fixed-precision_transformer_language_models.md)**

:   精确刻画了固定精度、严格未来掩码、软注意力、无位置编码的 Transformer 的表达能力——恰好等价于仅含过去算子的线性时态逻辑 LTL[P]，并将其与偏序确定有限自动机 (PODFA)、$\mathcal{R}$-trivial 幺半群统一起来。

**[CodeAssistBench (CAB): Dataset & Benchmarking for Multi-turn Chat-Based Code Assistance](llm_nlp/codeassistbench_cab_dataset_benchmarking_for_multi-turn_chat-based_code_assistan.md)**

:   提出 CodeAssistBench (CAB)，第一个评估多轮、项目级编程辅助的全自动 Benchmark，从 GitHub issues 自动构建 3,286 个真实编程求助场景，涵盖 7 种编程语言 214 个仓库，发现 SOTA 模型在 StackOverflow 式问题上达 70-83% 准确率，但在 post-training-cutoff 仓库上仅 7-16%。

**[ComPO: Preference Alignment via Comparison Oracles](llm_nlp/compo_preference_alignment_via_comparison_oracles.md)**

:   针对DPO中噪声偏好对（preferred和dispreferred响应相似）导致的似然位移和冗长问题，提出基于比较oracle的零阶偏好对齐方法ComPO，将数据分为干净/噪声子集，用DPO处理干净数据、用ComPO提取噪声数据中的信号，在AlpacaEval 2等benchmark上持续提升LC win rate。

**[Composing Linear Layers from Irreducibles](llm_nlp/composing_linear_layers_from_irreducibles.md)**

:   利用Clifford代数，将线性层表示为二向量（bivector）的组合——即旋量（rotor）的三明治乘积——仅需 $O(\log^2 d)$ 参数即可替代 $d \times d$ 密集矩阵，应用于LLM注意力层的Q/K/V投影时性能接近原始模型和强基线。

**[ConfTuner: Training Large Language Models to Express Their Confidence Verbally](llm_nlp/conftuner_training_large_language_models_to_express_their_confidence_verbally.md)**

:   ConfTuner 提出 tokenized Brier score 损失函数（理论证明为 proper scoring rule），仅需 2000 个样本 + 4 分钟 LoRA 微调即可让 LLM 输出校准的语言化置信度（如"我80%确定"），ECE 最大降低 60.9%，支持自我纠错和模型级联等下游应用。

**[CoRe: Benchmarking LLMs' Code Reasoning Capabilities through Static Analysis Tasks](llm_nlp/core_benchmarking_llms_code_reasoning_capabilities_through_static_analysis_tasks.md)**

:   提出 CoRe，一个包含 12,553 个人工验证任务实例的高质量 benchmark，通过数据依赖、控制依赖和信息流三类静态分析基础任务，直接评估 LLM 的代码语义推理能力，揭示模型在 trace 生成和源枚举等需要多步推理的任务上仍严重不足。

**[DATE-LM: Benchmarking Data Attribution Evaluation for Large Language Models](llm_nlp/date-lm_benchmarking_data_attribution_evaluation_for_large_language_models.md)**

:   DATE-LM是首个统一、应用驱动的LLM数据归因基准，涵盖数据选择、毒性过滤、事实归因三大应用，通过公开排行榜促进可复现和公平的方法比较。

**[Decoupled Entropy Minimization](llm_nlp/decoupled_entropy_minimization.md)**

:   将经典熵最小化（EM）解耦为两个对立部分——Cluster Aggregation Driving Factor (CADF，奖励主导类别)和 Gradient Mitigation Calibrator (GMC，惩罚高置信类别)，揭示了经典 EM 的两个固有缺陷（reward collapse 和 easy-class bias），提出 AdaDEM 通过归一化奖励和边际熵校准来修复这些问题，在半监督学习、域适应、强化学习等多任务上显著提升。

**[Demystifying Language Model Forgetting With Low-Rank Example Associations](llm_nlp/demystifying_language_model_forgetting_with_low-rank_example_associations.md)**

:   发现LLM微调后上游样本遗忘与新学任务之间的关联矩阵具有低秩结构（rank-3即$R^2>0.69$），利用矩阵补全预测未见任务导致的遗忘，指导选择性回放以减轻遗忘。

**[Detecting High-Stakes Interactions with Activation Probes](llm_nlp/detecting_high-stakes_interactions_with_activation_probes.md)**

:   用线性激活探针（在 LLM 内部表示上训练的轻量分类器）检测用户的"高风险交互"，在合成数据上训练后跨 6 个真实数据集 AUROC 达 0.88-0.92，匹敌 8-12B 微调 LLM但计算成本低 6 个数量级，级联架构（探针初筛+LLM 精判）进一步超越单独使用任一方法。

**[Do Different Prompting Methods Yield a Common Task Representation in Language Models?](llm_nlp/do_different_prompting_methods_yield_a_common_task_representation_in_language_mo.md)**

:   本文扩展函数向量方法至指令提示，发现演示和指令诱发的任务表示主要不同，仅部分重叠，解释了为何结合两者效果更优。

**[Do Language Models Use Their Depth Efficiently?](llm_nlp/do_language_models_use_their_depth_efficiently.md)**

:   通过因果干预、残差流分析和跨模型线性映射，证明当前 LLM 后半部分层不参与组合式计算，仅迭代细化输出概率分布，深层模型只是把浅层模型的计算"展延"到更多层。

**[Don't Be Lazy: CompleteP Enables Compute-Efficient Deep Transformers](llm_nlp/dont_be_lazy_completep_enables_compute-efficient_deep_transformers.md)**

:   CompleteP 参数化（α=1）是唯一同时实现深度方向超参转移和完全特征学习的方案，在深模型上相比 μP 节省 12-34% FLOPs。

**[EnCompass: Enhancing Agent Programming with Search Over Program Execution Paths](llm_nlp/encompass_enhancing_agent_programming_with_search_over_program_execution_paths.md)**

:   提出 Probabilistic Angelic Nondeterminism (PAN) 编程模型及 EnCompass Python 框架，将 agent 的核心工作流逻辑与推理时搜索策略解耦，程序员只需在 LLM 调用处加 `branchpoint()` 标记，即可用几行参数切换 best-of-N、beam search、tree search 等策略，代码修改量减少 3-6x。

**[Gemstones: A Model Suite for Multi-Faceted Scaling Laws](llm_nlp/gemstones_a_model_suite_for_multi-faceted_scaling_laws.md)**

:   Gemstones开源4000+检查点数据集（至2B参数），系统研究宽度-深度-训练代币在缩放律中的影响，揭示缩放律对设计选择的高度敏感性。

**[How Do Transformers Learn Implicit Reasoning?](llm_nlp/how_do_transformers_learn_implicit_reasoning.md)**

:   通过符号环境的精细控制研究，本文发现多跳隐式推理会经历记忆→分布内泛化→跨分布泛化三阶段，关键机制是中间实体表示在余弦空间的聚类。

**[HybridNorm: Towards Stable and Efficient Transformer Training via Hybrid Normalization](llm_nlp/hybridnorm_towards_stable_and_efficient_transformer_training_via_hybrid_normaliz.md)**

:   提出 HybridNorm 混合归一化策略——注意力模块用 QKV 归一化解耦梯度、FFN 用 Post-Norm 增强正则化，在 550M-7B 规模上同时获得 Pre-Norm 的训练稳定性和 Post-Norm 的泛化性能，7B 模型下游任务平均提升 2.45%。

**[Hyperparameter Transfer Enables Consistent Gains Of Matrix-Preconditioned Optimi](llm_nlp/hyperparameter_transfer_enables_consistent_gains_of_matrix-preconditioned_optimi.md)**

:   研究矩阵预条件优化器（Shampoo/SOAP/Muon）的超参数随模型宽度和深度的缩放规则（基于 μP），发现正确的超参缩放是实现一致加速的关键：使用 μP + 1/width weight decay，三者在 190M 到 1.4B 参数的 Llama 模型上一致实现约 1.4× 加速。

**[In-Context Learning of Linear Dynamical Systems with Transformers: Approximation Bounds and Depth-Separation](llm_nlp/in-context_learning_of_linear_dynamical_systems_with_transformers_approximation_.md)**

:   分析了线性 Transformer 在噪声线性动力系统上的 ICL 近似能力：$O(\log T)$ 深度可达到 $O(\log T / T)$ 测试误差（接近最小二乘估计器），而单层线性 Transformer 存在不可消除的下界——揭示了非 IID 数据下的深度分离现象。

**[Language Model Behavioral Phases are Consistent Across Architecture, Training Data, and Scale](llm_nlp/language_model_behavioral_phases_are_consistent_across_archi.md)**

:   论文在 Transformer、Mamba、RWKV，不同数据集与参数规模（14M 到 12B）上系统分析 1400+ checkpoints，发现语言模型预训练中存在高度一致的行为阶段；词级行为变化最多可由 unigram 频率、n-gram 概率、语义相似度三类简单启发式解释（最高约 98% 方差）。

**[Large Language Models Miss the Multi-Agent Mark](llm_nlp/large_language_models_miss_the_multi-agent_mark.md)**

:   Position paper 指出当前 MAS LLMs 在四个方面违背了传统多智能体系统（MAS）的基本原则：LLM 缺乏原生社会行为、环境设计以 LLM 为中心、缺少异步协调和标准通信协议、涌现行为缺乏量化评估，并为每个问题提出研究方向。

**[LooGLE v2: LLM在真实世界长依赖挑战上的准备情况评估](llm_nlp/loogle_v2_are_llms_ready_for_real_world_long_dependency_challenges.md)**

:   通过自动化流程从法律、金融、游戏、代码等领域采集16k-2M token的真实文本,设计10类长依赖复杂任务,共1934个QA实例,评估表明最优模型仅59.2%准确率,揭示LLM长序列理解能力的根本不足。

**[Memory Mosaics at Scale](llm_nlp/memory_mosaics_at_scale.md)**

:   Memory Mosaics v2 将关联存储网络扩展至 10B 参数、1T token 训练规模，在新任务学习和上下文学习上显著超越同规模甚至 8T token 训练的 Transformer。

**[MOOSE-Chem2: Exploring LLM Limits in Fine-Grained Scientific Hypothesis Discovery](llm_nlp/moose-chem2_exploring_llm_limits_in_fine-grained_scientific_hypothesis_discovery.md)**

:   将细粒度科学假设生成形式化为组合优化问题，提出层次启发式搜索（HHS）——利用 LLM 的成对比较作为梯度信号在假设空间中导航，层次化抽象平滑奖励景观减少局部最优陷阱，在 2024 年后化学论文 51 篇的专家标注 benchmark 上 Soft Recall 从 19.99% 提升到 40.35%。

**[Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models](llm_nlp/nemotron-flash_towards_latency-optimal_hybrid_small_language_models.md)**

:   Nemotron-Flash 通过系统优化深宽比、进化搜索混合算子组合（DeltaNet+Mamba2+Attention）以及权重归一化训练，构建延迟最优的小语言模型家族，相比 Qwen3-1.7B/0.6B 分别实现 1.3×/1.9× 延迟下降与 +5.5% 平均准确率提升。

**[Probabilistic Token Alignment for Large Language Model Fusion](llm_nlp/probabilistic_token_alignment_for_large_language_model_fusion.md)**

:   PTA-LLM 通过将 token 对齐建模为最优传输（OT）问题，用 Sinkhorn 算法计算两个 LLM 间最优概率匹配，融合 logit 分布而非硬映射，在 78 个任务上平均提升 +1.72%。

**[Retrospective In-Context Learning for Temporal Credit Assignment with Large Language Models](llm_nlp/retrospective_incontext_learning_for_temporal_credit_assignm.md)**

:   论文提出 RICL（Retrospective In-Context Learning），利用 LLM 的预训练知识把环境中的稀疏奖励回溯性转化为稠密 advantage supervision，再结合在线策略迭代框架 RICOL，在 BabyAI 四个场景中以更高样本效率达到与传统在线 RL 相当的收敛表现，展示了 LLM 在 temporal credit assignment 上的潜力。

**[Scaling Embedding Layers in Language Models](llm_nlp/scaling_embedding_layers_in_language_models.md)**

:   提出Scone方法，通过为高频n-gram学习上下文化的嵌入（用独立Transformer模型训练），在推理时将这些嵌入卸载到主存/SSD，实现"训练时用更多计算但推理时不增加加速器资源"的新缩放范式，1B参数模型超越1.9B基线。

**[Sloth: Scaling Laws for LLM Skills to Predict Multi-Benchmark Performance Across Families](llm_nlp/sloth_scaling_laws_for_llm_skills_to_predict_multi-benchmark_performance_across_.md)**

:   提出Skills Scaling Laws (Sloth)，通过假设LLM性能由低维潜在技能（如推理、指令遵循）驱动，利用benchmark间的相关性构建跨模型家族的缩放定律，用少量家族数据即可预测大模型在多个benchmark上的表现。

**[The Curse of Depth in Large Language Models](llm_nlp/the_curse_of_depth_in_large_language_models.md)**

:   揭示 Pre-LN Transformer 中输出方差指数增长导致深层退化为恒等映射的根本原因，提出无参数的 LayerNorm Scaling（LNS）策略——仅在 LayerNorm 后乘以 $1/\sqrt{\ell}$，将方差从指数增长压缩为多项式增长，在 130M-7B 全规模上稳定改进困惑度 5-8%。

**[The Trilemma of Truth in Large Language Models](llm_nlp/the_trilemma_of_truth_in_large_language_models.md)**

:   提出 sAwMIL（稀疏感知多实例学习）三类探测框架，结合 MIL 和保形预测，将 LLM 内部激活分类为 true/false/neither，揭示真假信号并非简单的双向对称编码，而是跨越多维子空间的分布式表征。

**[Understanding And Enhancing Mask-Based Pretraining Towards Universal Representat](llm_nlp/understanding_and_enhancing_mask-based_pretraining_towards_universal_representat.md)**

:   用高维线性回归理论精确刻画了 mask-based pretraining 中掩码率对测试风险的影响（偏差-方差分解），揭示了最优掩码率依赖于任务和模型大小，并据此提出 R2MAE（随机随机掩码），在视觉、语言、DNA、单细胞模型上一致超越固定掩码率。

**[Unifying Attention Heads and Task Vectors via Hidden State Geometry in In-Context Learning](llm_nlp/unifying_attention_heads_and_task_vectors_via_hidden_state_geometry_in_in-contex.md)**

:   本文提出基于隐状态几何（可分离性+对齐性）的统一框架，将ICL的两大解释路线——注意力头（PTH/IH）和任务向量——联系起来，揭示ICL在分类任务中的两阶段机制：早期层通过PTH建立可分离性，后期层通过IH改善与标签unembedding方向的对齐性。

**[What One Cannot, Two Can: Two-Layer Transformers Provably Represent Induction Heads on Any-Order Markov Chains](llm_nlp/what_one_cannot_two_can_two-layer_transformers_provably_represent_induction_head.md)**

:   理论证明两层单头 Transformer 足以表示任意 $k$ 阶马尔可夫过程的条件 $k$-gram 模型（即 $k$ 阶 induction head），给出了 Transformer 深度与马尔可夫阶数关系的最紧已知刻画，关键在于利用 MLP 中的 ReLU 和 LayerNorm 非线性来补偿减少的层数。

**[Yggdrasil: 桥接动态投机和静态运行时的延迟最优树型LLM解码](llm_nlp/yggdrasil_bridging_dynamic_speculation_and_static_runtime_for_latency-optimal_tr.md)**

:   通过等增长树(EGT)草稿算法和延迟感知目标，实现动态投机与静态图编译的兼容，配合前向执行阶段重叠，在A100上达3.98×加速。

---

## 🧩 多模态 VLM { #multimodal_vlm }

**[A Frustratingly Simple Yet Highly Effective Attack Baseline: Over 90% Success Rate Against the Strong Black-box Models of GPT-4.5/4o/o1](multimodal_vlm/a_frustratingly_simple_yet_highly_effective_attack_baseline.md)**

:   提出 M-Attack，通过对对抗图像做随机裁剪后与目标图像在嵌入空间做局部对齐（而非传统的全局对齐），配合多模型集成，使得生成的对抗扰动具有丰富的局部语义细节，在 GPT-4.5/4o/o1 等商业黑盒 LVLM 上实现超过 90% 的目标攻击成功率，大幅超越所有已有方法。

**[A Multimodal Benchmark for Framing of Oil & Gas Advertising and Potential Greenwashing Detection](multimodal_vlm/a_multimodal_benchmark_for_framing_of_oil_gas_advertising_an.md)**

:   构建了首个面向石油天然气行业视频广告的多模态框架分析基准数据集（706个视频，覆盖Facebook和YouTube两个平台，13种框架类型），用于评估VLM在检测企业"洗绿"宣传中的能力，发现GPT-4.1在环境信息检测上可达79% F1但在绿色创新识别上仅46% F1。

**[AC-LoRA: (Almost) Training-Free Access Control-Aware Multi-Modal LLMs](multimodal_vlm/aclora_almost_trainingfree_access_controlaware_multimodal_ll.md)**

:   设计AC-LoRA系统，通过为不同权限数据集维护独立的LoRA适配器，并基于查询相似度和用户权限进行检索+无训练合并，实现企业级LLM聊天机器人的强信息隔离保证。

**[ACT as Human: Multimodal Large Language Model Data Annotation with Critical Thinking](multimodal_vlm/act_as_human_multimodal_large_language_model_data_annotation.md)**

:   提出ACT（Annotation with Critical Thinking）流水线，先用MLLM批量标注数据，再用另一个MLLM作为"批评者"识别可能的错误标注，仅让人类审核被标记的样本，在减少70-90%人工标注成本的同时将性能差距控制在<2%。

**[AdaLRS: Loss-Guided Adaptive Learning Rate Search for Efficient Foundation Model Pretraining](multimodal_vlm/adalrs_lossguided_adaptive_learning_rate_search_for_efficien.md)**

:   提出AdaLRS，一种即插即用的在线学习率搜索算法，通过监控损失下降速度（loss velocity）来自适应调整学习率，将学习率超参搜索的成本从多次独立训练降低到单次训练，实现~50%的训练成本节省。

**[Adapting Vision-Language Models for Evaluating World Models](multimodal_vlm/adapting_visionlanguage_models_for_evaluating_world_models.md)**

:   提出UNIVERSE框架，通过仅微调PaliGemma 2的投影头（0.07%参数）和优化数据混合策略，实现对游戏世界模型rollout的高效视觉语言评估，在动作/角色识别任务上以极低成本接近完整微调的性能。

**[ADMN: A Layer-Wise Adaptive Multimodal Network for Dynamic Input Noise and Compute Resources](multimodal_vlm/admn_a_layerwise_adaptive_multimodal_network_for_dynamic_inp.md)**

:   提出 ADMN（Adaptive Depth Multimodal Network），通过两阶段训练——(1) Multimodal LayerDrop 微调使 backbone 适应任意层配置，(2) QoI感知控制器动态分配层预算给各模态——在严格计算约束下根据每个模态的信息质量(QoI)自适应分配层数，匹配全量模型精度同时减少 75% FLOPs 和 60% 延迟。

**[Advancing Compositional Awareness in CLIP with Efficient Fine-Tuning](multimodal_vlm/advancing_compositional_awareness_in_clip_with_efficient_fin.md)**

:   提出 CLIC（Compositionally-aware Learning in CLIP），通过拼接图像对 + 跨图词汇交换生成 hard negatives + 多正样本训练的策略，在仅微调文本编码器的情况下同时提升 CLIP 的组合推理能力和检索性能，在 SugarCrepe++ 上取得 CLIP 类模型 SOTA。

**[AffordBot: 3D Fine-grained Embodied Reasoning via Multimodal Large Language Models](multimodal_vlm/affordbot_3d_fine-grained_embodied_reasoning_via_multimodal_large_language_model.md)**

:   提出细粒度 3D 具身推理任务（预测可操作元素的空间位置+运动类型+运动轴），通过将 3D 点云渲染为环视图并投影 affordance 候选，结合定制的 CoT 推理范式指导 MLLM 实现 SOTA，AP25 达 23.3%。

**[Aligning by Misaligning: Boundary-aware Curriculum Learning for Multimodal Alignment](multimodal_vlm/aligning_by_misaligning_boundaryaware_curriculum_learning_fo.md)**

:   提出 BACL（Boundary-Aware Curriculum with Local Attention），通过可学习的边界感知负样本采样器（由易到难课程学习）+ 对比局部注意力损失（定位 token 级 mismatch），在 LAION-400M 上为 CLIP 带来 +32% R@1 提升，并在四个大规模基准上取得 SOTA。

**[AntiGrounding: Lifting Robotic Actions into VLM Representation Space for Decision Making](multimodal_vlm/antigrounding_lifting_robotic_actions_into_vlm_representatio.md)**

:   提出 AntiGrounding，逆转传统指令 grounding 过程——不是将语言映射到动作空间，而是将候选机器人动作"提升"到 VLM 表示空间（通过多视角轨迹渲染 + 结构化 VQA），实现零样本闭环机器人轨迹合成。

**[Approximate Domain Unlearning for Vision-Language Models](multimodal_vlm/approximate_domain_unlearning_for_visionlanguage_models.md)**

:   提出 Approximate Domain Unlearning (ADU) 新任务，通过 Domain Disentangling Loss (DDL) 和 Instance-wise Prompt Generator (InstaPG) 两个模块，让预训练 VLM 选择性遗忘指定域（如插画、素描）的识别能力，同时保持其他域（如真实照片）的分类精度，在四个多域数据集上大幅超越所有基线。

**[AQuaMaM: An Autoregressive Quaternion Manifold Model for Rapidly Estimating Complex Protein Structures](multimodal_vlm/aquamam_an_autoregressive_quaternion_manifold_model_for_rapidly_estimating_compl.md)**

:   AQuaMaM 提出基于四元数流形的自回归蛋白质结构预测模型——将蛋白质骨架的旋转表示为四元数（在 $S^3$ 流形上），用自回归方式沿序列逐步预测每个残基的局部坐标系旋转，实现比 AlphaFold 快数个量级的结构估计。

**[Balanced Token Pruning: Accelerating Vision Language Models Beyond Local Optimization](multimodal_vlm/balanced_token_pruning_accelerating_vision_language_models_b.md)**

:   提出 Balanced Token Pruning (BTP)，通过在浅层优先多样性剪枝、深层优先注意力剪枝的分阶段策略，联合优化局部输出一致性和全局表示质量，在仅保留 22% 视觉 token 的情况下保持原模型 98% 的性能。

**[Benchmarking Retrieval-Augmented Multimodal Generation for Document Question Answering](multimodal_vlm/benchmarking_retrievalaugmented_multimodal_generation_for_do.md)**

:   提出 MMDocRAG 基准（4055 个专家标注的 QA 对），系统评估了 60 个 VLM/LLM 和 14 个检索器在多模态文档检索增强生成中的引用选择和交错图文回答能力，揭示当前最强模型 GPT-4.1 的 Quote Selection F1 仅 70.2%，微调可显著提升性能。

**[Better Tokens for Better 3D: Advancing Vision-Language Modeling in 3D Medical Imaging](multimodal_vlm/better_tokens_for_better_3d_advancing_vision-language_modeling_in_3d_medical_ima.md)**

:   提出 BTB3D，一种基于因果卷积编解码器 + 3D Haar 小波压缩 + 三阶段渐进训练的 3D CT tokenizer，在放射报告生成和文本条件 CT 合成两大下游任务上大幅刷新 SOTA，证明"更好的 token 比更大的语言模型更重要"。

**[Beyond Greedy Exits: Improved Early Exit Decisions for Risk Control and Reliability](multimodal_vlm/beyond_greedy_exits_improved_early_exit_decisions_for_risk_control_and_reliabili.md)**

:   UAT（Unsupervised Adaptive Thresholding）为早退 DNN 设计了可靠性函数来评估中间层输出质量，并用多臂赌博机（MAB）算法在推理时动态学习最优退出阈值，实现 1.7-2.1× 加速且性能损失 <2%，同时对分布偏移鲁棒。

**[BioCLIP 2: Emergent Properties from Scaling Hierarchical Contrastive Learning](multimodal_vlm/bioclip_2_emergent_properties_from_scaling_hierarchical_contrastive_learning.md)**

:   BioCLIP 2 在 TreeOfLife-200M（2.14 亿图像/95.2 万物种）上用层级对比学习训练 ViT-L，零样本物种识别比 BioCLIP 提升 18%，并发现规模化带来的涌现性质——嵌入自动编码生态关系（如达尔文雀喙大小排列）且种内变异与种间差异正交。

**[BLINK-Twice: You See But Do You Observe? A Reasoning Benchmark on Visual Perception](multimodal_vlm/blink-twice_you_see_but_do_you_observe_a_reasoning_benchmark_on_visual_perceptio.md)**

:   提出视觉中心推理 benchmark BLINK-Twice（345 张视觉挑战图 + 103 个对抗样本 + 896 个 VQA + 1725 个推理步骤标注），通过 7 类视觉错觉场景评估 MLLM "看到但未观察到"的推理能力，发现最强模型 Gemini-2.5 Pro 的 G-Acc 仅 26.9%，多轮图像观察和主动视觉交互是提升方向。

**[CAPability: A Comprehensive Visual Caption Benchmark for Evaluation](multimodal_vlm/capability_a_comprehensive_visual_caption_benchmark_for_eval.md)**

:   构建 CAPability——11K 标注的图片/视频描述评估基准，从 6 个视角 12 个维度评估 VLM 的描述能力，引入 KT（know-but-cannot-tell）指标衡量 VLM 在 QA 中已知但描述中遗漏的信息差距。

**[Causal-LLaVA: Causal Disentanglement for Mitigating Hallucination in Multimodal Large Language Models](multimodal_vlm/causalllava_causal_disentanglement_for_mitigating_hallucinat.md)**

:   揭示 MLLM 中物体幻觉的表示层根因——数据集共现偏差导致的语义纠缠，提出双路因果解纠缠框架（Causal-Driven Projector + Causal Intervention Module），通过后门调整在 projector 和最终 Transformer 层分离共现物体表示，使 MME-Perception 提升 22.6%。

**[ChartMuseum: Testing Visual Reasoning Capabilities of Large Vision-Language Models](multimodal_vlm/chartmuseum_testing_visual_reasoning_capabilities_of_large_v.md)**

:   构建ChartMuseum——一个包含1,162个专家标注问题的图表QA benchmark，专门评估LVLM的复杂视觉和文本推理能力。与现有图表benchmark（前沿模型接近饱和）不同，ChartMuseum揭示了巨大的模型-人类性能差距：人类93%准确率 vs Gemini-2.5-Pro仅63.0% vs 最佳开源Qwen2.5-VL-72B仅38.5%，且所有模型在视觉推理重的问题上掉点35-55%。

**[Continual Multimodal Contrastive Learning](multimodal_vlm/continual_multimodal_contrastive_learning.md)**

:   首次形式化定义持续多模态对比学习(CMCL)问题——按顺序在不同模态对数据上训练而不忘记之前的对齐，提出Dual-sided Null Space (DNS)方法将新梯度投影到不影响旧知识的子空间，在7个数据集11个训练步骤上一致优于现有持续学习基线。

**[CovMatch: Cross-Covariance Guided Multimodal Dataset Distillation with Trainable Text Encoder](multimodal_vlm/covmatch_crosscovariance_guided_multimodal_dataset_distillat.md)**

:   提出 CovMatch，通过将多模态对比学习的双层优化简化为跨协方差矩阵对齐的闭式解，首次实现图文双编码器的联合优化进行多模态数据集蒸馏，仅用 500 个合成图文对在 Flickr30K 上获得 38.4 平均检索精度（+6.8% 超越 SOTA LoRS），在极端数据高效场景下大幅超越冻结文本编码器的方法。

**[DanmakuTPPBench: A Multi-modal Benchmark for Temporal Point Process Modeling and Understanding](multimodal_vlm/danmakutppbench_a_multimodal_benchmark_for_temporal_point_pr.md)**

:   论文提出首个面向多模态 Temporal Point Process 的系统 benchmark：一方面构建来自 Bilibili 弹幕视频的时间戳-文本-视频联合事件数据集 DanmakuTPP-Events，另一方面通过多智能体 LLM/MLLM pipeline 构建复杂时序推理问答集 DanmakuTPP-QA，系统揭示当前 TPP 模型与 MLLM 在多模态事件动态理解上的明显短板。

**[DOTA: Distributional Test-Time Adaptation of Vision-Language Models](multimodal_vlm/dota_distributional_testtime_adaptation_of_visionlanguage_mo.md)**

:   提出 DOTA（DistributiOnal Test-time Adaptation），不再简单缓存测试样本，而是**持续估计测试数据流的底层分布**，通过贝叶斯定理计算后验概率实现自适应，解决了缓存容量有限导致的灾难性遗忘问题，在多个分布偏移基准上达到 SOTA。

**[READ: Enhancing Compositional Reasoning in CLIP via Reconstruction and Alignment of Text Descriptions](multimodal_vlm/enhancing_compositional_reasoning_in_clip_via_reconstruction.md)**

:   提出 READ 微调方法，通过两个辅助目标——(1) token-level 重建（冻结解码器从文本嵌入重建替代描述）和 (2) sentence-level 对齐（强制改述的嵌入一致）——增强 CLIP 文本编码器的组合推理能力，在 5 个组合推理基准上达到 SOTA（超 NegCLIP 4.5%，超 FSC-CLIP 4.1%）。

**[Enhancing Vision-Language Model Reliability with Uncertainty-Guided Dropout Decoding](multimodal_vlm/enhancing_visionlanguage_model_reliability_with_uncertaintyg.md)**

:   提出Dropout Decoding——量化视觉token的认知不确定性(epistemic uncertainty)，选择性遮掩高不确定性token，通过集成多个遮掩后的解码结果做多数投票，无需训练即在InstructBLIP上CHAIR_I降低16%、CHAIR_S降低12%。

**[First SFT, Second RL, Third UPT: Continual Improving Multi-Modal LLM Reasoning via Unsupervised Post-Training](multimodal_vlm/first_sft_second_rl_third_upt_continual_improving_multi-modal_llm_reasoning_via_.md)**

:   提出 MM-UPT 框架，在 SFT 和 RL 之后引入第三阶段"无监督后训练"，通过多数投票作为伪奖励信号结合 GRPO 实现 MLLM 的自我改进，在 MathVista 上将 Qwen2.5-VL-7B 从 66.3% 提升至 72.9%。

**[FlowCut: Rethinking Redundancy via Information Flow for Efficient Vision-Language Models](multimodal_vlm/flowcut_rethinking_redundancy_via_information_flow_for_effic.md)**

:   从信息流（Information Flow）视角重新理解VLM中视觉token的冗余性：发现CLS token是信息中继站、冗余渐进式涌现、单层单标准评分不够可靠，提出FlowCut——基于信息流感知的多标准累积重要性剪枝框架，在LLaVA-1.5-7B上以88.9%的token减少率超越SOTA 1.6%，在LLaVA-NeXT-7B上超越4.3%。

**[Generalized Contrastive Learning for Universal Multimodal Retrieval](multimodal_vlm/generalized_contrastive_learning_for_universal_multimodal_re.md)**

:   提出 Generalized Contrastive Learning (GCL)——在 mini-batch 内对所有 6 种模态对组合（image↔text, image↔image+text, text↔image+text）执行对比学习，无需构建新的三元组数据集，仅用现有图文对即可在 M-BEIR 上将 VISTA 的平均检索精度从 21.18 提升到 34.06（+60.8%），在 MMEB 的 text→image+text 任务上从 10.1% 提升到 31.1%。

**[Generate, but Verify: Reducing Hallucination in Vision-Language Models with Retrospective Resampling](multimodal_vlm/generate_but_verify_reducing_hallucination_in_visionlanguage.md)**

:   提出REVERSE框架——首次在单一VLM内统一了生成、验证和纠正三个阶段：通过引入<SPAN>、</CN>（置信）、</UN>（不置信）三个特殊token训练幻觉感知模型，推理时当</UN>概率超过阈值就回溯到上一个</CN>重新生成，在CHAIR-MSCOCO上降低12%、HaloQuest上降低34%的幻觉率。

**[GLSim: Detecting Object Hallucinations in LVLMs via Global-Local Similarity](multimodal_vlm/glsim_detecting_object_hallucinations_in_lvlms_via_globalloc.md)**

:   提出 GLSim，一种无训练的物体幻觉检测框架，结合图像-文本间的全局和局部嵌入相似度信号来判断 LVLM 生成的物体是否为幻觉，显著超越仅使用全局或局部信号的方法。

**[HoPE: Hybrid of Position Embedding for Long Context Vision-Language Models](multimodal_vlm/hope_hybrid_of_position_embedding_for_long_context_visionlan.md)**

:   提出 HoPE（Hybrid of Position Embedding），通过混合频率分配策略和动态时间缩放机制改进 VLM 中的位置编码，解决 RoPE 在长视频等长上下文多模态场景中无法可靠捕捉时空语义相似性的问题，在四个长视频基准上一致超越现有方法。

**[Learning to Instruct for Visual Instruction Tuning](multimodal_vlm/learning_to_instruct_for_visual_instruction_tuning.md)**

:   提出 L2T（Learning to Instruct），仅通过将训练损失扩展到指令序列（不再只在回答上计算 loss）来改善视觉指令调优——无额外数据和几乎零计算开销，在 16 个多模态基准上获得高达 9% 的相对提升，captioning 提升 18%，同时缓解幻觉。

**[Mint: A Simple Test-Time Adaptation of Vision-Language Models against Common Corruptions](multimodal_vlm/mint_a_simple_testtime_adaptation_of_visionlanguage_models_a.md)**

:   发现 CLIP 在图像损坏下的性能退化根源在于**嵌入方差坍缩**——类内与类间方差同步缩小导致嵌入空间判别性丧失；提出 Mint，通过最大化伪标签类间方差（PL-inter）在线修复嵌入几何，仅凭均值累加器和梯度累加器两个极简组件即可在 BS=1 的在线场景下稳定提升 CLIP 在多种损坏基准上的分类精度，同时比最强 baseline 快 45 倍。

**[MMLongBench: Benchmarking Long-Context Vision-Language Models Effectively and Thoroughly](multimodal_vlm/mmlongbench_benchmarking_longcontext_visionlanguage_models_e.md)**

:   构建首个全面的长上下文视觉语言模型（LCVLM）评估基准 MMLongBench——13,331 个样本覆盖 5 类下游任务、混合图像类型、5 级标准化输入长度（8K-128K tokens），评估 46 个模型后发现单任务性能是整体能力的弱代理，且强推理能力与长上下文性能正相关。

**[Praxis-VLM: Vision-Grounded Decision Making via Text-Driven Reinforcement Learning](multimodal_vlm/praxisvlm_visiongrounded_decision_making_via_textdriven_rein.md)**

:   发现VLM的决策推理能力可以与视觉感知解耦——用文本描述替代图像时决策性能不降反升，据此提出Praxis-VLM：在纯文本场景上用GRPO训练决策推理能力，然后零样本迁移到视觉输入推理，在VIVA/PCA-Bench/EgoNormia三个决策benchmark上超越SFT基线且泛化性更强。

**[PrefixKV: Adaptive Prefix KV Cache is What Vision Instruction-Following Models Need for Efficient Generation](multimodal_vlm/prefixkv_adaptive_prefix_kv_cache_is_what_vision_instruction.md)**

:   提出 PrefixKV，将 LVLM 各层 KV 缓存大小的确定转化为搜索最优全局前缀配置的问题，通过二分搜索找到信息保留阈值实现自适应逐层 KV 保留，在 20% 压缩率下仍保持接近原模型性能，提供 1.8× 推理加速。

**[Rethinking Multimodal Learning from the Perspective of Mitigating Classification Ability Disproportion](multimodal_vlm/rethinking_multimodal_learning_from_the_perspective_of_mitig.md)**

:   提出"**分类能力不均衡**"视角理解多模态学习中的模态不平衡，设计 Sustained Boosting 算法（共享编码器 + 多可配置分类器，同时优化分类和残差误差）配合自适应分类器分配（ACA），理论证明跨模态 gap loss 以 $\mathcal{O}(1/T)$ 收敛，在 CREMAD 等 6 个数据集上大幅超越 SOTA。

**[Sherlock: Self-Correcting Reasoning in Vision-Language Models](multimodal_vlm/sherlock_selfcorrecting_reasoning_in_visionlanguage_models.md)**

:   首个系统研究VLM推理自纠正能力的框架：发现现有推理VLM几乎不能自纠正（<10%出现aha moment），提出Sherlock三阶段训练框架（SFT冷启动→离线轨迹级偏好学习→在线自我迭代）仅用20K标注数据超越使用100K-260K数据的LLaVA-CoT/Mulberry/LlamaV-o1。

**[Sparse Autoencoders Learn Monosemantic Features in Vision-Language Models](multimodal_vlm/sparse_autoencoders_learn_monosemantic_features_in_visionlan.md)**

:   将Sparse Autoencoder (SAE)从LLM可解释性扩展到VLM领域，提出MonoSemanticity Score (MS)量化视觉神经元的单义性，发现SAE能将VLM中多义的神经元分解为单义特征，且可直接通过操控单个SAE神经元来steering LLaVA的输出（插入或抑制概念），无需修改LLM。

**[SRPO: Enhancing Multimodal LLM Reasoning via Reflection-Aware Reinforcement Learning](multimodal_vlm/srpo_enhancing_multimodal_llm_reasoning_via_reflection-aware_reinforcement_learn.md)**

:   提出 SRPO（Self-Reflection enhanced reasoning with Group Relative Policy Optimization），一个两阶段反思感知 RL 框架：第一阶段用大模型生成反思数据做 SFT cold-start，第二阶段设计反思感知奖励函数在 GRPO 中强化简洁有效的自我反思能力，在 MathVista/MathVision/MMMU-Pro 等多模态推理基准上以 7B/32B 模型显著超越同规模 SOTA。

**[The Illusion of Progress? A Critical Look at Test-Time Adaptation for Vision-Language Models](multimodal_vlm/the_illusion_of_progress_a_critical_look_at_testtime_adaptat.md)**

:   提出TTA-VLM benchmark，在统一实验条件下评估8种episodic和7种online测试时适应(TTA)方法在15个数据集上的表现，发现三个令人意外的结论：(1) 现有TTA方法相比早期TPT基线提升有限；(2) TTA与训练时微调方法协作效果差；(3) 准确率提升以牺牲校准、OOD检测和鲁棒性为代价。

**[The Narrow Gate: Localized Image-Text Communication in Native Multimodal Models](multimodal_vlm/the_narrow_gate_localized_imagetext_communication_in_native.md)**

:   发现原生多模态VLM（如Chameleon、Emu3）中图像到文本的跨模态信息传递竟然集中在单一的end-of-image [EOI] token上（"narrow gate"机制），而非原生VLM（如LLaVA）则通过多个图像token分布式传递信息；删除[EOI]的attention可导致native模型性能崩溃，而修改[EOI]表示可精确控制模型的语义输出。

**[TRoVe: Discovering Error-Inducing Static Feature Biases in Temporal Vision-Language Models](multimodal_vlm/trove_discovering_errorinducing_static_feature_biases_in_tem.md)**

:   TRoVe 提出一个自动化诊断框架，用于发现 temporal VLM 在时序理解任务中错误依赖的静态特征偏置；它通过从验证集提取候选静态特征，并同时评估这些特征对错误率的影响与模型对其依赖程度，在 101 个带偏置真值标注的 temporal VLM 上较最强基线提升 28.6%，还能进一步辅助 test-time 改善模型表现。

**[Unveiling Chain of Step Reasoning for Vision-Language Models with Fine-grained Rewards](multimodal_vlm/unveiling_chain_of_step_reasoning_for_visionlanguage_models.md)**

:   提出Chain-of-Step (CoS)推理框架：将VLM的推理链分解为结构化步骤（Name+Thought+Reflection），训练Process Reward Model (PRM)提供步骤级精细奖励，通过迭代DPO和step-level beam search显著提升VLM推理能力——在InternVL-2.5-MPO-8B上平均提升4.0%达到73.4%，并揭示"对VLM而言推理质量比长度更重要"。

**[VL-SAE: Interpreting and Enhancing Vision-Language Alignment with a Unified Concept Set](multimodal_vlm/vlsae_interpreting_and_enhancing_visionlanguage_alignment_wi.md)**

:   提出VL-SAE，一种带有距离编码器和模态特定解码器的稀疏自编码器，将视觉和语言表示的语义映射到统一概念集，从而解释和增强VLM的视觉-语言对齐机制，在零样本分类平均提升0.6-0.9%，在POPE幻觉消除上超越专用方法VCD。

---

## 🎨 图像生成 { #image_generation }

**[70% Size, 100% Accuracy: Lossless LLM Compression for Efficient GPU Inference via Dynamic-Length Float (DFloat11)](image_generation/70_size_100_accuracy_lossless_llm_compression_for_efficient.md)**

:   DFloat11 利用 BFloat16 权重中指数位（exponent）的低熵特性，通过 Huffman 编码将 LLM/扩散模型无损压缩至原始大小的约 70%（等效 ~11 bit），并设计了层次化查找表和两阶段 GPU kernel 实现高效在线解压，使 Llama 3.1 405B 可在单节点 8×80GB GPU 上无损推理。

**[A Closer Look at Model Collapse: From a Generalization-to-Memorization Perspective](image_generation/a_closer_look_at_model_collapse_from_a_generalization-to-memorization_perspectiv.md)**

:   发现扩散模型在自消耗循环（用生成数据训练下一代模型）中存在从"泛化"到"记忆"的转变过程，揭示训练集熵与模型泛化能力的强线性相关性（Pearson r=0.91），并提出基于熵的数据选择策略（Greedy Selection / Threshold Decay Filter）有效减缓该转变，在 CIFAR-10 accumulate 范式下第 8 轮 FID 从 75.7 降至 44.7。

**[A Connection Between Score Matching and Local Intrinsic Dimension](image_generation/a_connection_between_score_matching_and_local_intrinsic_dimension.md)**

:   证明去噪得分匹配损失（denoising score matching loss）的下界恰好是数据流形的局部固有维度（LID），从而将 DSM loss 本身作为一个高效的 LID 估计器——无需梯度计算或多次前向传播，在 Stable Diffusion 3.5 上内存占用仅为 FLIPD 的 60%，且量化后估计更稳定。

**[A Data-Driven Prism: Multi-View Source Separation with Diffusion Model Priors](image_generation/a_data-driven_prism_multi-view_source_separation_with_diffusion_model_priors.md)**

:   提出 DDPRISM 方法，利用多视图观测中不同线性变换的结构性差异，在 EM 框架下为每个未知源学习独立的扩散模型先验，无需预先获得任何单独的源样本即可完成源分离和后验采样，在合成问题和真实星系观测上超越现有方法。

**[A Diffusion Model for Regular Time Series Generation from Irregular Data with Completion and Masking](image_generation/a_diffusion_model_for_regular_time_series_generation_from_irregular_data_with_co.md)**

:   提出两步框架从不规则采样时序数据生成规则时序：先用 TST 自编码器补全缺失值构造"自然邻域"，再在视觉扩散模型中用 masking 策略仅在观测像素上计算损失，避免对补全值的过度依赖，在判别分数上平均改善 70%，训练速度提升 6.5 倍。

**[A Gradient Flow Approach to Solving Inverse Problems with Latent Diffusion Models](image_generation/a_gradient_flow_approach_to_solving_inverse_problems_with_latent_diffusion_model.md)**

:   提出 DWGF（Diffusion-regularized Wasserstein Gradient Flow），将隐空间扩散模型的后验采样问题严格形式化为 KL 散度在 Wasserstein-2 空间上的正则化梯度流，推导出隐空间中的 ODE 系统用于求解图像逆问题，在 FFHQ-512 上的修复和超分辨率任务中 PSNR 大幅超越基线。

**[Accelerating Parallel Diffusion Model Serving with Residual Compression](image_generation/accelerating_parallel_diffusion_model_serving_with_residual_compression.md)**

:   提出 CompactFusion 框架，通过残差压缩（仅传输相邻去噪步骤间的激活差异而非完整激活）来消除并行扩散推理中的通信冗余，在 4×L20 上实现 3.0× 加速且生成质量远优于 DistriFusion，在模拟以太网带宽下实现 6.7× 加速，甚至在 100× 压缩下仍优于 DistriFusion。

**[AccuQuant: Simulating Multiple Denoising Steps for Quantizing Diffusion Models](image_generation/accuquant_simulating_multiple_denoising_steps_for_quantizing.md)**

:   提出AccuQuant，一种用于扩散模型的训练后量化（PTQ）方法，通过在校准过程中显式模拟多个去噪步骤来最小化量化误差的累积效应，并通过新型目标函数将内存复杂度从O(n)降至O(1)。

**[Adapting Speech Language Model to Singing Voice Synthesis](image_generation/adapting_speech_language_model_to_singing_voice_synthesis.md)**

:   将 1.7B 参数的 TTS 预训练 Speech Language Model 适配到歌声合成（SVS）任务，通过乐谱 tokenization + multi-stream LM 预测 + conditional flow matching 精修 + vocoder，仅用 135 小时合成歌声数据达到与专用 SVS 系统可比的性能。

**[ALE-Bench: A Benchmark for Long-Horizon Objective-Driven Algorithm Engineering](image_generation/alebench_a_benchmark_for_longhorizon_objectivedriven_algorit.md)**

:   提出 ALE-Bench，首个面向分数制算法工程竞赛（AtCoder Heuristic Contest）的 AI 基准，评估 LLM 和 Agent 在 NP-hard 优化问题上的长时间迭代改进能力，发现当前最强模型（o3-high）仅达人类平均水平，且在问题一致性和长时间改进方面与人类专家差距显著。

**[Aligning Compound AI Systems via System-level DPO](image_generation/aligning_compound_ai_systems_via_system-level_dpo.md)**

:   将复合 AI 系统建模为 DAG，提出 SysDPO 框架将 DPO 扩展到多组件联合对齐，通过 DAG 分解将系统级偏好转化为可端到端优化的损失函数，理论证明了 β-完美对齐保证，在 LLM+扩散模型和 LLM+LLM 系统上显著提升协作质量。

**[Aligning Text to Image in Diffusion Models is Easier Than You Think](image_generation/aligning_text_to_image_in_diffusion_models_is_easier_than_you_think.md)**

:   提出 SoftREPA——一种轻量级对比微调策略，通过引入可学习 soft text token（不到 1M 参数）在冻结的预训练 T2I 扩散模型上进行对比学习，显式提高文本和图像表征的互信息，在 SD1.5/SDXL/SD3 上显著提升文本-图像对齐质量，且适用于图像生成和图像编辑任务。

**[Amortized Sampling with Transferable Normalizing Flows](image_generation/amortized_sampling_with_transferable_normalizing_flows.md)**

:   提出 Prose——一个 285M 参数的全原子可迁移归一化流，基于 TarFlow 架构训练在 21,700 个短肽 MD 轨迹上（总计 4.3ms 模拟时长），实现对任意短肽系统的零样本无相关性提议采样，在能量评估预算相同时超越 MD 基线，生成速度比之前的可迁移玻尔兹曼生成器 (TBG) 快 4000 倍。

**[AugGen: Synthetic Augmentation using Diffusion Models Can Improve Recognition](image_generation/auggen_synthetic_augmentation_using_diffusion_models_can_imp.md)**

:   提出AugGen——一种自包含（self-contained）的合成数据增强方法：利用扩散模型的条件向量插值（$c^* = \alpha c_i + \beta c_j$）实现类间混合生成，无需外部数据或模型即可为人脸识别提供1-12%的性能提升，等效于1.7倍真实数据量，IR50+AugGen甚至超越IR101 real-only。

**[Autoregressive Adversarial Post-Training for Real-Time Interactive Video Generation](image_generation/autoregressive_adversarial_posttraining_for_realtime_interac.md)**

:   提出 AAPT（Autoregressive Adversarial Post-Training），将预训练的潜在视频扩散模型转化为实时交互式视频生成器——每帧仅需单次神经网络前向传播（1NFE），自回归逐帧生成，8B 模型在单张 H100 上以 24fps 流式生成 736×416 视频，最长可达一分钟（1440帧）。

**[BADiff: Bandwidth Adaptive Diffusion Model](image_generation/badiff_bandwidth_adaptive_diffusion_model.md)**

:   提出 BADiff——首个带宽自适应扩散模型，通过将目标熵约束作为条件嵌入扩散反向过程，配合可微熵正则化损失和自适应停止策略，使模型根据实时带宽动态调整生成质量并自适应提前终止采样，在保持感知质量的同时减少计算开销，从根本上避免了传统"高质量生成→后压缩"流程中的压缩伪影和计算浪费。

**[Balanced Conic Rectified Flow](image_generation/balanced_conic_rectified_flow.md)**

:   针对 k-rectified flow 中 reflow 步骤导致的分布漂移问题，提出 conic reflow：利用真实图像的反演噪声及其 Slerp 扰动构成锥形监督轨迹，大幅减少所需 fake pair 数量的同时获得更优的生成质量和更直的 ODE 路径。

**[Beyond Masked and Unmasked: Discrete Diffusion Models via Partial Masking](image_generation/beyond_masked_and_unmasked_discrete_diffusion_models_via_par.md)**

:   提出 Prime（Partial masking scheme），突破 Masked Diffusion Model 的二元状态（mask/unmask）限制，引入中间态（部分观测的 token 信息），减少冗余计算并实现更细粒度的去噪过程，在文本生成上 PPL 15.36 超越自回归模型（17.54）和标准 MDM（21.52），在图像生成上取得 CIFAR-10 FID 3.26。

**[BitMark: Watermarking Bitwise Autoregressive Image Generative Models](image_generation/bitmark_watermarking_bitwise_autoregressive_image_generative_models.md)**

:   提出 BitMark——首个针对比特级自回归图像生成模型（Infinity、Instella）的水印方案，在生成过程中通过对 logit 加偏置将 bit 序列引向"绿色列表"，实现可靠检测（z-test）、高图像保真度（FID 几乎不变）、对多种攻击的鲁棒性和放射性（训练在水印图上的下游模型也带有水印），为防止模型坍缩提供了关键工具。

**[Blameless Users in a Clean Room: Defining Copyright Protection for Generative Models](image_generation/blameless_users_in_a_clean_room_defining_copyright_protection_for_generative_mod.md)**

:   重建生成模型可证明版权保护的理论基础——证明现有的 Near Access-Freeness (NAF) 定义不能防止逐字复制（"被污染"），提出"无辜用户"(blameless) 框架和净室版权保护 ($(\kappa,\beta)$-clean) 定义，其中用户在反事实"净室设置"中不会复制则在真实世界中也不太可能复制，并证明差分隐私训练在"黄金数据集"假设下蕴含净室版权保护。

**[Blind Strong Gravitational Lensing Inversion: Joint Inference of Source and Lens Mass with Score-Based Models](image_generation/blind_strong_gravitational_lensing_inversion_joint_inference_of_source_and_lens_.md)**

:   首次将 score-based 生成模型先验应用于强引力透镜的盲反演——联合推断背景源天体形态和透镜质量分布参数，通过将 GibbsDDRM 扩展到连续时间域实现采样，重建残差与观测噪声一致，透镜参数边际后验无系统偏差。

**[BlurDM: A Blur Diffusion Model for Image Deblurring](image_generation/blurdm_a_blur_diffusion_model_for_image_deblurring.md)**

:   提出 BlurDM，将运动模糊的物理形成过程（连续曝光导致渐进模糊累积）集成到扩散模型——双扩散前向（同时加噪声+模糊）+ 双去噪去模糊反向，作为隐空间先验生成器一致性增强 4 种去模糊方法在 4 个数据集上的效果，GoPro 平均 +0.31 dB，RealBlur-J 平均 +0.78 dB，仅增加 ~4 GFLOPs 和 ~9ms。

**[BlurGuard: A Simple Approach for Robustifying Image Protection Against AI-Powered Edit](image_generation/blurguard_a_simple_approach_for_robustifying_image_protection_against_ai-powered.md)**

:   提出 BlurGuard——在生成对抗扰动之前先对图像做轻度模糊预处理，使扰动更鲁棒地抵御 JPEG 压缩、高斯噪声等后处理操作，从而更有效地保护图像不被 Stable Diffusion 等 AI 编辑工具篡改，在保护成功率上比不模糊基线提升 20%+。

**[BoltzNCE: Learning Likelihoods for Boltzmann Generation with Stochastic Interpolants](image_generation/boltznce_learning_likelihoods_for_boltzmann_generation_with_stochastic_interpola.md)**

:   BoltzNCE 用 Score Matching + InfoNCE 混合训练 Energy-Based Model 来近似 Boltzmann Generator 的似然，避免了昂贵的 Jacobian trace 计算，在丙氨酸二肽构象生成上实现 100× 推理加速且自由能误差仅 0.02 $k_BT$。

**[Boosting Generative Image Modeling via Joint Image-Feature Synthesis](image_generation/boosting_generative_image_modeling_via_joint_imagefeature_sy.md)**

:   提出 Latent-Semantic Diffusion，让扩散模型联合生成 VAE 低级图像 latent 和 DINO 高级语义特征，通过最小修改标准 DiT 实现生成质量和训练效率的显著提升，并解锁 Representation Guidance 推理策略。

**[Breaking AR's Sampling Bottleneck: Provable Acceleration via Diffusion Language Models](image_generation/breaking_ars_sampling_bottleneck_provable_acceleration_via_d.md)**

:   从信息论角度为扩散语言模型建立收敛保证，证明采样误差（KL散度）随迭代次数T成反比衰减且与token间互信息线性相关，关键证明了T<L（迭代次数可少于序列长度L）时仍可生成高质量样本，从理论上打破了自回归模型需要L步的基本采样瓶颈，并建立了匹配的上下界证明分析的紧致性。

**[CAMILA: Context-Aware Masking for Image Editing with Language Alignment](image_generation/camila_contextaware_masking_for_image_editing_with_language.md)**

:   提出 CAMILA，一种上下文感知的图像编辑方法，能够判断用户指令是否在当前图像上下文中可行，仅执行可行的编辑指令而忽略不可执行的指令，在单指令和多指令编辑场景中均优于现有方法。

**[Composition and Alignment of Diffusion Models using Constrained Learning](image_generation/composition_and_alignment_of_diffusion_models_using_constrai.md)**

:   提出统一的约束学习框架来处理扩散模型的对齐（alignment）和组合（composition），将多奖励对齐形式化为 KL 散度最小化+奖励约束，将模型组合形式化为 minimax KL 散度问题，通过拉格朗日对偶的原-对偶训练算法求解，相比传统加权方法更可解释且避免了手动调权。

**[Conditional Panoramic Image Generation via Masked Autoregressive Modeling](image_generation/conditional_panoramic_image_generation_via_masked_autoregres.md)**

:   提出PAR（Panoramic AutoRegressive model），首次用掩码自回归建模统一文本到全景图和全景图外延两大任务，通过循环平移一致性损失和双空间循环填充解决ERP全景图的边界不连续问题，在Matterport3D上取得37.37 FID，同时展示出良好的可扩展性和零样本泛化能力。

**[Continuous Diffusion Model for Language Modeling](image_generation/continuous_diffusion_model_for_language_modeling.md)**

:   提出一种面向离散语言建模的连续扩散框架，将离散扩散过程与统计流形上的连续流联系起来，并通过径向对称的 simulation-free 训练机制与降维技巧，显著提升扩散语言模型性能，接近自回归模型。

**[CORAL: Disentangling Latent Representations in Long-Tailed Diffusion](image_generation/coral_disentangling_latent_representations_in_longtailed_dif.md)**

:   论文系统分析长尾数据下扩散模型尾部类别生成质量下降的根因，指出 U-Net 瓶颈潜表示发生“头类-尾类子空间重叠”导致特征借用，并提出 CORAL 对比式潜空间对齐正则，显著提升尾类样本的多样性与视觉质量。

**[Co-Reinforcement Learning for Unified Multimodal Understanding and Generation](image_generation/coreinforcement_learning_for_unified_multimodal_understandin.md)**

:   提出CoRL框架——通过"统一RL→精细RL"两阶段GRPO训练策略，在不依赖额外监督数据的情况下，让统一多模态模型(ULM)的理解和生成能力协同进化，在Janus-Pro-1.5B上取得生成+7%、理解+23%的平均提升。

**[DEFT: Decompositional Efficient Fine-Tuning for Text-to-Image Models](image_generation/deft_decompositional_efficient_finetuning_for_texttoimage_mo.md)**

:   提出DEFT——将权重更新分解为两个可训练矩阵的组合：(1)低秩子空间的正交投影和(2)子空间内的低秩调整，相比LoRA在T2I个性化中CLIP-T从0.341提升到0.361（DreamBench+），在统一模型上实现风格迁移和条件生成的SOTA。

**[DiCo: Revitalizing ConvNets for Scalable and Efficient Diffusion Modeling](image_generation/dico_revitalizing_convnets_for_scalable_and_efficient_diffus.md)**

:   重新发掘卷积网络在扩散模型中的潜力——发现预训练DiT的全局自注意力主要捕获局部模式（冗余），提出用标准ConvNet模块+紧凑通道注意力构建纯卷积扩散模型DiCo，在ImageNet-256上以2.05 FID超越DiT-XL/2且速度快2.7倍。

**[Diffusion Adaptive Text Embedding for Text-to-Image Diffusion Models](image_generation/diffusion_adaptive_text_embedding_for_texttoimage_diffusion.md)**

:   发现T2I扩散模型中固定的text embedding在不同时间步是次优的，提出DATE——在推理时动态更新text embedding以最大化mean predicted image与文本的对齐评分（如CLIP Score/ImageReward），无需训练，可即插即用到任何扩散模型和采样器中，在多概念生成和图像编辑中一致提升text-image对齐。

**[Diffusion Classifiers Understand Compositionality, but Conditions Apply](image_generation/diffusion_classifiers_understand_compositionality_but_condit.md)**

:   全面研究零样本扩散分类器在组合理解任务上的判别能力：覆盖3个扩散模型(SD 1.5/2.0/3-m)×10个数据集×30+任务，引入Self-Bench诊断基准（用扩散模型自己生成的图像消除域差异），发现扩散分类器确实理解组合性但受域差距和时间步加权影响——"条件适用"。

**[Distilled Decoding 2: One-step Sampling of Image Auto-regressive Models with Conditional Score Distillation](image_generation/distilled_decoding_2_onestep_sampling_of_image_autoregressiv.md)**

:   提出 Distilled Decoding 2 (DD2)，通过条件分数蒸馏损失将图像自回归模型压缩为单步生成器，在 ImageNet-256 上 FID 仅从 3.40 增至 5.43，比 DD1 的 one-step 差距缩小 67%，训练加速 12.3×。

**[Emergence and Evolution of Interpretable Concepts in Diffusion Models](image_generation/emergence_and_evolution_of_interpretable_concepts_in_diffusi.md)**

:   首次将 Sparse Autoencoders (SAEs) 系统性地应用于多步扩散模型 (Stable Diffusion v1.4)，揭示了图像构图在第一步反向扩散就已涌现、风格概念在中期阶段形成的时间演化规律，并据此设计了时间自适应的因果干预技术。

**[Head Pursuit: Probing Attention Specialization in Multimodal Transformers](image_generation/head_pursuit_probing_attention_specialization_in_multimodal.md)**

:   用信号处理中的Simultaneous Orthogonal Matching Pursuit (SOMP)算法分解注意力头在unembedding矩阵上的稀疏表示，揭示注意力头的语义特化现象（如政治/国籍/月份/数字等），仅编辑1%的头即可可靠地抑制或增强特定概念——在语言和视觉-语言模型上均验证有效。

**[InfinityStar: Unified Spacetime AutoRegressive Modeling for Visual Generation](image_generation/infinitystar_unified_spacetime_autoregressive_modeling_for_v.md)**

:   提出 InfinityStar，首个能生成工业级 720p 视频的纯离散自回归模型，通过时空金字塔建模统一 T2I/T2V/I2V/交互式长视频生成，VBench 83.74 超越 HunyuanVideo，推理速度比扩散模型快 10-32×。

**[MagCache: Fast Video Generation with Magnitude-Aware Cache](image_generation/magcache_fast_video_generation_with_magnitudeaware_cache.md)**

:   发现视频扩散模型中连续时间步残差输出的幅度比(magnitude ratio)遵循统一的单调递减规律（跨模型、跨prompt稳定），提出MagCache基于此规律自适应跳过冗余时间步并复用缓存，仅需1个样本校准即可在Open-Sora/CogVideoX/Wan 2.1/HunyuanVideo上实现2.1-2.68×加速，视觉保真度全面超越现有方法。

**[OmniSync: Towards Universal Lip Synchronization via Diffusion Transformers](image_generation/omnisync_towards_universal_lip_synchronization_via_diffusion.md)**

:   OmniSync提出了一种基于Diffusion Transformer的通用唇形同步框架，通过无掩码训练范式、基于Flow Matching的渐进噪声初始化和动态时空CFG三大创新，在真实视频和AI生成视频上都大幅超越先前方法，尤其在风格化角色的唇形同步上达到87.78%成功率（之前最佳67.78%）。

**[PhysCtrl: Generative Physics for Controllable and Physics-Grounded Video Generation](image_generation/physctrl_generative_physics_for_controllable_and_physicsgrou.md)**

:   提出 PhysCtrl，通过生成式物理网络学习 4 种材质（弹性体、沙子、橡皮泥、刚体）的物理动力学分布，以 3D 点轨迹表示物理运动，结合 I2V 模型实现物理参数和力可控的视频生成。

**[Safe-Sora: Safe Text-to-Video Generation via Graphical Watermarking](image_generation/safesora_safe_texttovideo_generation_via_graphical_watermark.md)**

:   Safe-Sora 首次将**图形水印**（如logo图像）直接嵌入到视频生成管线中，通过分层粗到细自适应匹配将水印patch分配到视觉最相似的帧和区域，并设计3D小波变换增强Mamba架构实现时空融合，在视频质量（FVD 3.77 vs 次优154.35）和水印保真度上大幅超越所有基线。

**[Scalable, Explainable and Provably Robust Anomaly Detection with One-Step Flow Matching](image_generation/scalable_explainable_and_provably_robust_anomaly_detection_with_one-step_flow_ma.md)**

:   提出 TCCM（Time-Conditioned Contraction Matching），一种受 flow matching 启发的表格数据半监督异常检测方法，通过学习将正常数据收缩到原点的时间条件速度场，仅需单步前向推理即可计算异常分数，在 ADBench 47 个数据集上取得 AUROC 和 AUPRC 双第一，推理速度比 DTE 快 1573 倍。

---

## ⚡ LLM 效率 { #llm_efficiency }

**[3-Model Speculative Decoding (PyramidSD)](llm_efficiency/3model_speculative_decoding.md)**

:   在标准的draft-target两模型推测解码的中间插入一个"qualifier"模型，构成三层金字塔式解码架构（PyramidSD），利用模型家族天然的熵梯度来分级过滤token，以模糊接受准则放宽匹配阈值，实现最高1.91×的速度提升（在RTX 4090上达到124 tok/s）。

**[A Multi-Task Benchmark for Abusive Language Detection in Low-Resource Settings](llm_efficiency/a_multitask_benchmark_for_abusive_language_detection_in_lowr.md)**

:   针对低资源语言 Tigrinya，构建了首个大规模多任务基准数据集 TiALD（13,717条YouTube评论，涵盖滥用检测、情感分析、主题分类三任务），并证明小型微调模型在低资源场景下显著优于GPT-4o等前沿LLM（F1: 86.67% vs 79.31%）。

**[A Stochastic Differential Equation Framework for Multi-Objective LLM Interactions](llm_efficiency/a_stochastic_differential_equation_framework_for_multi-objective_llm_interaction.md)**

:   将 LLM 迭代交互中的多目标优化建模为 SDE（漂移-扩散过程），通过干扰矩阵量化目标间的耦合模式，通过特征值谱分析策略收敛行为，在代码生成（安全性、效率、功能性三目标）上验证了不同策略的收敛率（0.33-1.29）和可预测性（$R^2$ 达 0.74）。

**[A Unified Framework for Establishing the Universal Approximation of Transformer-Type Architectures](llm_efficiency/a_unified_framework_for_establishing_the_universal_approxima.md)**

:   本文建立了一个统一的理论框架来证明各类Transformer架构的万能逼近性(UAP)，将UAP归结为两个可验证条件——前馈层的非线性仿射不变性和注意力层的token可区分性——并利用解析性假设将后者简化为仅需检验两样本情形。

**[Advancing Expert Specialization for Better MoE](llm_efficiency/advancing_expert_specialization_for_better_moe.md)**

:   通过正交性损失（减少专家间投影重叠）和方差损失（增大路由分数差异）双目标优化，在不修改 MoE 架构的前提下将专家重叠减少 45%、路由方差提升 150%，11 个基准任务平均提升 23.79%，同时完全保持负载均衡。

**[Approximately Aligned Decoding](llm_efficiency/approximately_aligned_decoding.md)**

:   提出 Approximately Aligned Decoding (AprAD)，一种利用投机解码（speculative decoding）中的前缀选择算法来实现LLM受约束生成的方法——在遇到约束违反时，既不像约束生成那样仅回退一步（导致极端概率放大），也不像ASAp那样完全重新采样（计算成本过高），而是通过投机采样智能选择回退位置，在输出分布失真和计算效率之间取得良好平衡。

**[Auto-Search and Refinement: An Automated Framework for Gender Bias Mitigation in LLMs](llm_efficiency/auto-search_and_refinement_an_automated_framework_for_gender_bias_mitigation_in_.md)**

:   提出 FaIRMaker 框架，通过"自动搜索+精化"范式先用梯度优化找到去偏见触发词（Fairwords），再训练 seq2seq 模型将其转化为可读指令，在开源和闭源 LLM 上有效缓解性别偏见同时保持甚至提升任务性能。

**[Constant Bit-Size Transformers Are Turing Complete](llm_efficiency/constant_bit-size_transformers_are_turing_complete.md)**

:   首次证明常数 bit 精度、固定参数数量的 Transformer（仅允许上下文窗口增长）是图灵完备的，并建立了精确的复杂度等价关系 WINDOW[s(n)] = SPACE[s(n)]，表明扩展上下文窗口——而非模型尺寸——已足以实现通用计算。

**[Critical Batch Size Revisited: A Simple Empirical Approach to Large-Batch Language Model Training](llm_efficiency/critical_batch_size_revisited_a_simple_empirical_approach_to_large-batch_languag.md)**

:   提出 branched training 方法直接实证测量临界 batch size (CBS)，发现 CBS 在训练早期快速增长后趋于平稳且不依赖模型规模，据此设计 batch size warmup 策略以 43% 更少的梯度步数达到同等甚至更优的训练 loss。

**[Deep Compositional Phase Diffusion for Long Motion Sequence Generation](llm_efficiency/deep_compositional_phase_diffusion_for_long_motion_sequence_generation.md)**

:   提出 Compositional Phase Diffusion 框架，在 ACT-PAE 建立的频域相位空间中用 SPDM 和 TPDM 分别处理语义对齐和过渡连续性，实现长程组合式动作序列生成，在 BABEL-TEACH 上达到 SOTA。

**[Dense Associative Memory with Epanechnikov Energy](llm_efficiency/dense_associative_memory_with_epanechnikov_energy.md)**

:   提出基于 Epanechnikov 核的 log-sum-ReLU（LSR）能量函数替代传统的 log-sum-exp（LSE），在 Dense Associative Memory 中首次实现了"精确记忆所有模式 + 同时涌现新的创造性局部极小"的共存，且保持指数级记忆容量。

**[DICE: Discrete Interpretable Comparative Evaluation with Probabilistic Scoring for RAG](llm_efficiency/dice_discrete_interpretable_comparative_evaluation_with_probabilistic_scoring_fo.md)**

:   提出 DICE 框架，通过两阶段评估（证据耦合深度分析 + 概率化 {A,B,Tie} 打分）和瑞士赛制锦标赛实现 RAG 系统的可解释、鲁棒、高效评估，在中文金融 QA 数据集上达到 85.7% 人类专家一致率，远超 RAGAS（45.7%）。

**[DISC: Dynamic Decomposition Improves LLM Inference Scaling](llm_efficiency/disc_dynamic_decomposition_improves_llm_inference_scaling.md)**

:   DISC 提出了一种动态分解算法，在推理时根据每一步的 z-score（采样奖励的标准化最大值）自动、递归地调整推理步骤的粒度——困难步骤分更细、简单步骤一步跨过——可以即插即用地与贪心搜索、Beam Search、MCTS 结合，在 APPS、MATH、LiveCodeBench 上以更少的 token 预算达到更高的 pass@k。

**[Document Summarization with Conformal Importance Guarantees](llm_efficiency/document_summarization_with_conformal_importance_guarantees.md)**

:   首次将Conformal Prediction应用于文档摘要，通过校准句子重要性分数的阈值，为抽取式摘要提供用户可控的覆盖率($1-\alpha$)和召回率($\beta$)的严格统计保证，方法模型无关且仅需小规模校准集。

**[Dynamics of Spontaneous Topic Changes in Next Token Prediction with Self-Attention](llm_efficiency/dynamics_of_spontaneous_topic_changes_in_next_token_prediction_with_self-attenti.md)**

:   从理论和实验两方面研究自注意力模型中"自发主题切换"的动力学机制，证明在单层 self-attention 模型中：(1) 混合主题训练保持原主题的 token 优先级顺序；(2) 主题切换仅在低优先级 token 数量超过高优先级 token 时发生；(3) 更长输入和更模糊主题不会增加切换概率——与人类认知相反。

**[Edit Less Achieve More Dynamic Sparse Neuron Masking For Lifelong Knowledge Edit](llm_efficiency/edit_less_achieve_more_dynamic_sparse_neuron_masking_for_lifelong_knowledge_edit.md)**

:   提出 NMKE 框架，通过神经元级归因发现 knowledge-general 和 knowledge-specific 两类知识神经元，并结合熵引导的动态稀疏 mask，实现精准神经元级知识编辑，在 5000 步连续编辑后仍保持高编辑成功率和模型通用能力。

**[Efficient Training-Free Online Routing for High-Volume Multi-LLM Serving](llm_efficiency/efficient_training-free_online_routing_for_high-volume_multi-llm_serving.md)**

:   提出首个无需训练的在线 LLM 路由算法 PORT，通过近似最近邻搜索估计查询特征，并在少量初始查询上一次性优化对偶变量作为路由权重，在有限 token 预算下实现接近离线最优 ($1-o(1)$ 竞争比) 的路由性能，平均较基线提升 3.55× 性能、1.85× 成本效率和 4.25× 吞吐量。

**[Exploring the Translation Mechanism of Large Language Models](llm_efficiency/exploring_the_translation_mechanism_of_large_language_models.md)**

:   提出 subspace-intervened path patching 方法对 LLM 翻译机制进行精细因果分析，发现翻译由不到 5% 的稀疏 attention head 驱动——分为 source head、indicator head、positional head 三类功能角色，MLP 将其特征整合为以英语为中心的中间表示，仅微调 64 个关键 head 即可匹配全参数微调性能。

**[Frequency-Aware Token Reduction for Efficient Vision Transformer](llm_efficiency/frequency-aware_token_reduction_for_efficient_vision_transformer.md)**

:   从频域视角提出 frequency-aware token reduction，将 token 分为高频（HF）和低频（LF）两组，选择性保留 HF token 并将 LF token 聚合为 DC token，在缓解 rank collapse 的同时减少 ViT 的计算量，在 30% token 减少率下多个模型上超越现有 SOTA。

**[From Shortcut to Induction Head: How Data Diversity Shapes Algorithm Selection in Transformers](llm_efficiency/from_shortcut_to_induction_head_how_data_diversity_shapes_algorithm_selection_in.md)**

:   通过严格的理论分析证明了预训练数据的多样性（由"max-sum ratio"刻画）决定了单层Transformer学到的是可泛化的induction head还是无法OOD泛化的位置捷径，并给出了使模型学会induction head的最优预训练分布。

**[Hardware-Aligned Hierarchical Sparse Attention For Efficient Long-Term Memory Ac](llm_efficiency/hardware-aligned_hierarchical_sparse_attention_for_efficient_long-term_memory_ac.md)**

:   提出层次化稀疏注意力（HSA）及 RAMba 架构，通过两阶段 token-to-chunk 相关性学习与硬件对齐 kernel 设计，让 Mamba 获得高效长程随机访问能力，仅在 4K 上下文预训练即可在 64M passkey retrieval 上达到 100% 准确率。

**[Hierarchical Balance Packing: Towards Efficient Supervised Fine-tuning for Long-Context LLM](llm_efficiency/hierarchical_balance_packing_towards_efficient_supervised_fine-tuning_for_long-c.md)**

:   提出层次均衡打包（HBP）方法，通过多级打包分组、均衡批处理、自适应序列并行和稳定损失归一化，解决长短上下文混合 SFT 中的注意力计算不均衡和通信浪费问题，在 DeepSeek-V2 (236B) 上实现 2.4× 训练加速且性能无损。

**[HiFi-RAG: Hierarchical Content Filtering and Two-Pass Generation for Open-Domain RAG](llm_efficiency/hifi-rag_hierarchical_content_filtering_and_two-pass_generation_for_open-domain_.md)**

:   通过分离轻量级 Flash 模型的过滤能力与 Pro 模型的推理能力，构建多阶段管道（查询优化→分层过滤→两阶段生成→引文验证），在 MMU-RAGent 竞赛中实现 SOTA 性能。

**[HyGen: Efficient LLM Serving via Elastic Online-Offline Request Co-location](llm_efficiency/hygen_efficient_llm_serving_via_elastic_online-offline_request_co-location.md)**

:   HyGen是干扰感知LLM推理系统，通过延迟预测和虚拟队列调度实现在线离线工作负载的弹性共置，保证SLO同时获得3.87-5.84倍吞吐改进。

**[L-MTP: Leap Multi-Token Prediction Beyond Adjacent Context for Large Language Models](llm_efficiency/l-mtp_leap_multi-token_prediction_beyond_adjacent_context_for_large_language_mod.md)**

:   L-MTP 在多token预测（MTP）基础上引入跳跃机制，预测非相邻位置的token（如位置1,3,5,7而非1,2,3,4），通过"后向查找"解码策略复用先前预测填补空隙，在3B-12B模型上实现22%推理加速的同时保持或提升任务性能。

**[Learning in Compact Spaces with Approximately Normalized Transformer](llm_efficiency/learning_in_compact_spaces_with_approximately_normalized_transformer.md)**

:   提出 anGPT（近似归一化 Transformer），利用高维空间中向量范数的集中现象，用简单标量乘法替代逐层精确归一化，在消除权重衰减和学习率预热的同时实现了相比 GPT+（含 QK-norm）40% 的收敛加速，仅增加 3% 运行时开销。

**[Long-Context Modeling with Dynamic Hierarchical Sparse Attention for On-Device LLMs](llm_efficiency/long-context_modeling_with_dynamic_hierarchical_sparse_attention_for_on-device_l.md)**

:   提出动态分层稀疏注意力 (DHSA)，通过自适应 chunk 分割 + chunk 级相似度预测 + 上采样到 token 级的分层框架，在不重训基座模型的前提下将密集注意力替换为稀疏注意力，在 Gemma2/3 上实现与密集注意力同等精度、20-60% prefill 延迟降低和 35% 峰值内存节省。

**[MEMOIR: Lifelong Model Editing with Minimal Overwrite and Informed Retention for LLMs](llm_efficiency/memoir_lifelong_model_editing_with_minimal_overwrite_and_informed_retention_for_.md)**

:   提出MEMOIR框架，通过在FFN层引入零初始化的残差记忆矩阵，利用基于TopHash的稀疏掩码将每次编辑限制在记忆参数的不同子集上，推理时通过掩码重叠率识别相关编辑并条件性激活知识，在15000次连续编辑下仍保持可靠性、泛化性和局部性的最优平衡。

**[Mozart: Modularized and Efficient MoE Training on 3.5D Wafer-Scale Chiplet Architectures](llm_efficiency/mozart_modularized_and_efficient_moe_training_on_35d_wafer-scale_chiplet_archite.md)**

:   提出 Mozart 算法-硬件协同设计框架，通过专家聚类分配、细粒度流式调度和 3.5D 晶粒架构（NoP-Tree + 分层存储），在三个 MoE-LLM 上实现 1.9× 以上的训练加速。

**[On the Entropy Calibration of Language Models](llm_efficiency/on_the_entropy_calibration_of_language_models.md)**

:   系统研究语言模型的熵校准问题（生成文本的熵是否匹配在人类文本上的 log loss），发现由于数据分布的幂律特性（$\alpha \approx 1$），误差积累随模型规模的改善极为缓慢（scaling exponent $\approx -0.05$），并从理论上证明了在多项式时间内可以在不牺牲多样性的前提下校准熵。

**[On the Expressive Power of Mixture-of-Experts for Structured Complex Tasks](llm_efficiency/on_the_expressive_power_of_mixture-of-experts_for_structured_complex_tasks.md)**

:   首次系统分析 MoE 在结构化复杂任务上的表达能力：证明浅层 MoE 可在低维流形上克服维度诅咒（近似速率由内在维度 $d$ 而非环境维度 $D$ 决定），深层 MoE 通过 $E$ 专家 × $L$ 层的分层组合可高效近似有 $E^L$ 段的分段函数，远超朴素上界 $LE$。

**[One Prompt Fits All: Universal Graph Adaptation for Pretrained Models](llm_efficiency/one_prompt_fits_all_universal_graph_adaptation_for_pretrained_models.md)**

:   理论证明表示级图提示（representation-level prompt）本质等价于线性探针，据此提出 UniPrompt——基于可学习 kNN 拓扑提示图的输入级方法，通过 bootstrapping 策略融合提示图和原图，在同域和跨域 few-shot 节点分类中一致超越现有图提示学习方法。

**[ParallelPrompt: Extracting Parallelism from Large Language Model Queries](llm_efficiency/parallelprompt_extracting_parallelism_from_large_language_model_queries.md)**

:   构建了首个查询内并行（intra-query parallelism）基准数据集ParallelPrompt，包含37000+条真实用户提示的结构化分解标注，证明约10%的用户查询包含可并行的潜在结构，并行执行可实现最高5.7×的延迟加速且质量损失有限。

**[Scale-invariant Attention](llm_efficiency/scale-invariant_attention.md)**

:   借鉴自然图像的尺度不变性，提出对 attention logits 做位置相关的乘性缩放和加性偏移变换，使注意力在不同 token 范围上的总权重和稀疏度满足尺度不变性，从而实现从短序列训练到长序列推理的零样本泛化（4k→64k 仅需一个超参数 $\tau$）。

**[Silent Tokens, Loud Effects: Padding in LLMs](llm_efficiency/silent_tokens_loud_effects_padding_in_llms.md)**

:   系统性研究了padding token在未被正确掩码时对LLM的影响，发现即使少量padding也会漂移隐层表示、降低生成质量、不可预测地改变偏见，而128个padding token可将Llama-3.1-8B的有害提示攻击成功率从8%飙升到77.5%，本质上实现了jailbreak。

**[SkyLadder: Better and Faster Pretraining via Context Window Scheduling](llm_efficiency/skyladder_better_and_faster_pretraining_via_context_window_scheduling.md)**

:   通过上下文窗口短到长的渐进式调度策略 SkyLadder，在固定计算量下实现更优的预训练效率（节省 22% 训练时间）和更好的模型性能（+3.7%），反驳了"长上下文=好性能"的业界信念。

**[SPARTA Alignment: Collectively Aligning Multiple Language Models through Combat](llm_efficiency/sparta_alignment_collectively_aligning_multiple_language_models_through_combat.md)**

:   让多个LLM组成"斯巴达部落"互相竞技和互评，通过声誉加权的判断聚合生成偏好对，再用DPO迭代训练所有模型，在12个任务中的10个上超越Self-Rewarding等自对齐基线，平均提升7%。

**[上下文学习中的技术债务：长序列中的递减效率](llm_efficiency/technical_debt_in_in-context_learning_diminishing_efficiency_in_long_context.md)**

:   揭示ICL作为学习算法在少射大样本制度下存在本质低效：少射ICL样本复杂度接近贝叶斯最优(1.1×)，而多射时恶化至1.45×，信息论分析证明此低效来自非递减过剩风险。

**[Tensor Product Attention Is All You Need](llm_efficiency/tensor_product_attention_is_all_you_need.md)**

:   通过上下文张量分解将 Q/K/V 表示为低秩因子的加权和，将 KV 缓存压缩至原来的 1/10~1/16，同时在验证损失和下游任务精度上超越标准 MHA/MQA/GQA/MLA。

**[The Emergence of Sparse Attention: Impact of Data Distribution and Benefits of Repetition](llm_efficiency/the_emergence_of_sparse_attention_impact_of_data_distribution_and_benefits_of_re.md)**

:   通过理论分析和受控实验研究 sparse attention 的涌现机制，揭示涌现时间遵循关于序列长度和维度的幂律关系 $T_\epsilon \propto \sqrt{d} \cdot T$，并发现 in-context 和 cross-sample 两种数据重复策略都能加速涌现，为理解 LLM 能力涌现提供了统一的 sparse attention 视角。

**[Tiled Flash Linear Attention: More Efficient Linear RNN and xLSTM Kernels](llm_efficiency/tiled_flash_linear_attention_more_efficient_linear_rnn_and_xlstm_kernels.md)**

:   提出 TFLA（Tiled Flash Linear Attention）算法，通过二层序列并行化和 tiling 优化，实现高效的线性 RNN/mLSTM 内核，相比 FlashAttention 3 和 Mamba 2 获得显著墙钟加速（训练 >2x vs Mamba 2），同时保持等价的模型精度。

**[Towards Interpretability Without Sacrifice: Faithful Dense Layer Decomposition with Mixture of Decoders](llm_efficiency/towards_interpretability_without_sacrifice_faithful_dense_layer_decomposition_wi.md)**

:   提出 Mixture of Decoders (MxD)，将 LLM 的 MLP 层分解为数万个稀疏激活的专家子层（layer-level sparsity），每个专家通过 Hadamard 乘积张量分解实现满秩线性变换，在稀疏性-准确性权衡上显著优于 Transcoders，同时保持可解释性。

**[UMoE: Unifying Attention and FFN with Shared Experts](llm_efficiency/umoe_unifying_attention_and_ffn_with_shared_experts.md)**

:   通过重新表述多头注意力机制，揭示其与 FFN 共有的"两层矩阵乘法"结构，据此提出 UMoE 统一架构——在注意力和 FFN 层使用相同设计的专家并支持参数共享，在 Base(134M) 和 Large(1.1B) 模型上均优于现有 FFN-MoE 和 Attention-MoE 基线。

**[Vocabulary Customization For Efficient Domain-Specific Llm Deployment](llm_efficiency/vocabulary_customization_for_efficient_domain-specific_llm_deployment.md)**

:   提出一种保证不增加任何输入 token 数的词表扩展算法，通过向预训练 LLM 的 tokenizer 添加领域特定 token，在电商场景实现输入序列缩短 20%、推理吞吐量提升 20-30%，且不损失模型质量。

**[ZeroS: Zero-Sum Linear Attention for Efficient Transformers](llm_efficiency/zeros_zero-sum_linear_attention_for_efficient_transformers.md)**

:   通过移除 softmax 的零阶均匀项 $1/t$，构建零和权重的线性注意力机制 ZeroS，突破凸组合只能做加法混合的限制，支持单层内的差分/对比操作，在保持 $O(Nd^2)$ 线性复杂度的同时，在多个序列建模基准上匹配甚至超越标准 softmax 注意力。

---

## 🎮 强化学习 { #reinforcement_learning }

**[A Generalized Bisimulation Metric of State Similarity between Markov Decision Processes: From Theoretical Propositions to Applications](reinforcement_learning/a_generalized_bisimulation_metric_of_state_similarity_betwee.md)**

:   将传统只能在单个MDP内度量状态相似性的bisimulation metric (BSM)推广到跨MDP场景，提出广义双模拟度量(GBSM)，严格证明了对称性、跨MDP三角不等式和同状态距离上界三个基本度量性质，并在策略迁移、状态聚合和基于采样的估计三个应用中推导出比标准BSM更紧的误差界和闭式样本复杂度。

**[A Near-optimal, Scalable and Parallelizable Framework for Stochastic Bandits Robust to Adversarial Corruptions and Beyond](reinforcement_learning/a_nearoptimal_scalable_and_parallelizable_framework_for_stoc.md)**

:   提出 BARBAT 框架，改进了经典的 BARBAR 算法，通过固定 epoch 长度和逐 epoch 调整失败概率，将对抗腐蚀下随机多臂老虎机的 regret 从 $O(\sqrt{K}C)$ 降至近最优的 $O(C)$（消除了 $\sqrt{K}$ 因子），并成功扩展到多智能体、图老虎机、组合半老虎机和批量老虎机等多种场景。

**[A Theory of Multi-Agent Generative Flow Networks](reinforcement_learning/a_theory_of_multi-agent_generative_flow_networks.md)**

:   提出多智能体生成流网络（MA-GFlowNets）的理论框架，证明了"局部-全局原理"——联合流函数可分解为各智能体独立流的乘积形式，设计了四种算法（CFN/IFN/JFN/CJFN），其中 JFN 和 CJFN 实现中心化训练+去中心化执行（CTDE），在 Hyper-Grid 和 StarCraft 环境中超越 RL 和 MCMC 方法。

**[A Unifying View of Linear Function Approximation in Off-Policy RL Through Matrix Splitting and Preconditioning](reinforcement_learning/a_unifying_view_of_linear_function_approximation_in_offpolic.md)**

:   将线性函数逼近下的TD、FQI和PFQI统一为求解同一线性系统的迭代方法（仅预条件子不同），首次引入矩阵分裂理论来分析它们的收敛性，给出了各算法收敛的充要条件，并揭示了TD收敛不一定意味着FQI收敛（反之亦然）。

**[Act to See, See to Act: Diffusion-Driven Perception-Action Interplay for Adaptive Policies](reinforcement_learning/act_to_see_see_to_act_diffusion-driven_perception-action_interplay_for_adaptive_.md)**

:   提出 DP-AG（Action-Guided Diffusion Policy），通过将扩散策略的噪声预测的 Vector-Jacobian Product (VJP) 作为结构化随机力驱动隐观测特征在扩散步骤间动态演化，并用循环一致对比损失闭合感知-动作环路，在 Push-T 上提升 6%、Dynamic Push-T 上提升 13%、真实 UR5 机器人上成功率提升 23%+。

**[Actor-Free Continuous Control via Structurally Maximizable Q-Functions](reinforcement_learning/actorfree_continuous_control_via_structurally_maximizable_qf.md)**

:   提出Q3C（Q-learning for Continuous Control with Control-points），一种无actor的纯基于值函数的连续控制方法，通过控制点插值逼近任意形状的Q函数，在复杂（非凸、受限）Q函数景观中显著优于actor-critic方法。

**[Adaptive Cooperative Transmission Design For Ultra-Reliable Low-Latency Communic](reinforcement_learning/adaptive_cooperative_transmission_design_for_ultra-reliable_low-latency_communic.md)**

:   提出 DRL-CoLA 算法，用双 Agent DQN 分别在源节点和中继节点上自适应配置 5G NR 传输参数（numerology、mini-slot、MCS），在两跳中继系统中仅用本地 CSI 即可达到接近全局 CSI 最优的 URLLC 可靠性。

**[Adaptive Neighborhood-Constrained Q Learning for Offline Reinforcement Learning](reinforcement_learning/adaptive_neighborhoodconstrained_q_learning_for_offline_rein.md)**

:   提出 ANQ（Adaptive Neighborhood-constrained Q learning），在离线 RL 中引入基于优势函数的自适应邻域约束，在密度约束（过于保守）和支持约束（需精确建模行为策略）之间找到灵活的中间方案，通过双层优化框架实现高效 Q 学习，在 D4RL 基准上达到 SOTA。

**[Adaptively Coordinating with Novel Partners via Learned Latent Strategies](reinforcement_learning/adaptively_coordinating_with_novel_partners_via_learned_latent_strategies.md)**

:   提出 TALENTS 框架，通过 VAE 学习潜在策略空间 + K-Means 聚类发现策略类型 + Fixed-Share 遗憾最小化算法在线推断队友类型，实现对未知人类/智能体队友的零样本实时适应协作。

**[ALINE: Joint Amortization for Bayesian Inference and Active Data Acquisition](reinforcement_learning/aline_joint_amortization_for_bayesian_inference_and_active_data_acquisition.md)**

:   ALINE 提出统一的分摊贝叶斯推断和主动数据获取框架，用 Transformer 架构 + RL 训练，使模型能同时策略性地选择最有信息量的数据点并即时完成后验推断，还支持灵活地针对特定参数子集或预测目标进行数据获取。

**[FastSVERL: Approximating Shapley Explanations in Reinforcement Learning](reinforcement_learning/approximating_shapley_explanations_in_reinforcement_learning.md)**

:   提出 FastSVERL——首个针对 RL 的可扩展 Shapley 值近似方法，用参数化模型分摊计算成本，解决 RL 特有的时序依赖、off-policy 数据和策略演化等挑战，为 RL 决策提供原则性的特征归因解释。

**[Automaton Constrained Q-Learning](reinforcement_learning/automaton_constrained_q-learning.md)**

:   ACQL 将安全 RL 和目标条件 RL 提升到 LTL（线性时序逻辑）任务类——用自动机编码时序目标进展和非平稳安全约束，结合目标条件值学习（+HER密集化奖励）和基于 Hamilton-Jacobi 可达性的安全约束，在连续控制任务上显著超越现有 LTL RL 方法，并在 6-DOF 机械臂上成功部署。

**[BEAST: Efficient Tokenization of B-Splines Encoded Action Sequences for Imitation Learning](reinforcement_learning/beast_efficient_tokenization_of_b-splines_encoded_action_sequences_for_imitation.md)**

:   BEAST 用 B 样条曲线参数化动作序列——通过岭回归估计控制点并均匀量化为固定长度 token，实现 20× token 压缩（100 步→5 token）、数学保证的动作块间 $C^0$ 连续过渡，在 LIBERO-Long 上成功率排名第 1（86.4%），推理吞吐量 617 Hz（比 π₀ 快 2.14×、比 OpenVLA 快 101×）。

**[Behavior Injection: Preparing Language Models for Reinforcement Learning](reinforcement_learning/behavior_injection_preparing_language_models_for_reinforcement_learning.md)**

:   揭示 LLM 对 RL 微调响应不一致的根本原因——通过 per-step influence 分析发现 RL 效果取决于（1）rollout 准确率分布（中等最优）和（2）数据 co-influence 强度，提出 BRIDGE 在 SFT 阶段注入探索/利用行为，使后续 RL 增益从 6% 提升到 46.6%。

**[Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning](reinforcement_learning/beyond_the_8020_rule_highentropy_minority_tokens_drive_effec.md)**

:   从 token 熵模式的全新视角分析 RLVR，发现 CoT 推理中仅约 20% 的高熵"分叉 token"决定推理方向，仅在这些 token 上做梯度更新即可匹配甚至大幅超越全量更新（Qwen3-32B 上 AIME'25 +11.04），揭示 RLVR 本质是优化推理决策点。

**[Blending Complementary Memory Systems in Hybrid Quadratic-Linear Transformers](reinforcement_learning/blending_complementary_memory_systems_in_hybrid_quadratic-linear_transformers.md)**

:   提出混合二次-线性 Transformer（HQLT），将 KV-memory（softmax attention，精确检索但二次复杂度）与 FW-memory（DeltaNet/线性 attention，线性复杂度但检索粗糙）融合为互补记忆系统，比较三种混合策略（延迟流式/延迟分块/同步），在 340M 和 1.3B 参数规模的语言建模、检索、算法推理和 RL 任务上验证同步混合最优。

**[Checklists Are Better Than Reward Models For Aligning Language Models](reinforcement_learning/checklists_are_better_than_reward_models_for_aligning_langua.md)**

:   提出 Reinforcement Learning from Checklist Feedback (RLCF)，将指令分解为动态生成的 yes/no checklist，结合 AI judge 和代码验证器逐项评分后做 DPO 训练，在 5 个 benchmark 上一致性提升 Qwen2.5-7B-Instruct，是唯一在所有 benchmark 上都有正收益的方法（FollowBench +4pt, InFoBench +6pt, Arena-Hard +3pt）。

**[Confounding Robust Deep Reinforcement Learning: A Causal Approach](reinforcement_learning/confounding_robust_deep_reinforcement_learning_a_causal_approach.md)**

:   基于部分辨识（partial identification）理论扩展 DQN，提出 Causal DQN 从含有未观测混淆因子的离线数据中学习鲁棒策略——通过优化最坏情况下的价值函数下界来获得安全策略，在 12 个混淆 Atari 游戏中一致性地超越标准 DQN。

**[Continual Knowledge Adaptation for Reinforcement Learning](reinforcement_learning/continual_knowledge_adaptation_for_reinforcement_learning.md)**

:   提出 CKA-RL，为每个任务维护知识向量（task-specific knowledge vector），通过 softmax 加权的动态知识适配和自适应知识合并机制，在三个持续 RL 基准上实现 4.20% 的整体性能提升和 8.02% 的前向迁移提升。

**[Decoder-Hybrid-Decoder Architecture for Efficient Reasoning with Long Generation](reinforcement_learning/decoderhybriddecoder_architecture_for_efficient_reasoning_wi.md)**

:   SambaY 提出 Gated Memory Unit（GMU）用于跨层共享 SSM 的 token 混合表示，将 YOCO 的 cross-decoder 中一半的 cross-attention 层替换为轻量级 GMU，在保持线性预填充复杂度和长上下文检索能力的同时，大幅提升解码效率——最终产品 Phi4-mini-Flash-Reasoning (3.8B) 在推理任务上超越 Phi4-mini-Reasoning，且在 2K 提示 + 32K 生成场景下实现高达 10× 的解码吞吐提升。

**[Hybrid Latent Reasoning via Reinforcement Learning](reinforcement_learning/hybrid_latent_reasoning_via_reinforcement_learning.md)**

:   HRPO 提出混合潜在推理策略优化：通过可学习的门控机制将前一步的隐藏状态表示逐步融入到采样的 token embedding 中，使 LLM 在推理阶段同时利用离散 token 和连续潜在表示，无需 CoT 标注即可通过 RL 训练，在知识密集型和 STEM 推理任务上均超越 PPO/GRPO 等基线。

**[Interactive and Hybrid Imitation Learning: Provably Beating Behavior Cloning](reinforcement_learning/interactive_and_hybrid_imitation_learning_provably_beating_behavior_cloning.md)**

:   当标注成本按**状态**而非轨迹计量时，证明交互式方法 Stagger 在 $\mu$-可恢复条件下可证明地超越 Behavior Cloning（次优性 $O(\mu H \log B / N)$ vs $O(RH \log B / CN)$，$\mu \ll R$ 时优势显著）；进一步提出混合 IL 算法 Warm-Stagger，结合离线数据和交互标注，在特定 MDP 上实现两种数据源的严格互补优势。

**[Inverse Optimization Latent Variable Models for Learning Costs Applied to Route Problems](reinforcement_learning/inverse_optimization_latent_variable_models_for_learning_costs_applied_to_route_.md)**

:   提出 IO-LVM（Inverse Optimization Latent Variable Model），用 VAE 式编码器映射观测的 COP 解到潜在成本空间，通过 Fenchel-Young 损失和黑盒求解器（Dijkstra/TSP solver）在解码端保证可行性，无需 agent 标签即可从路径数据中学到成本函数的分布，成功不可监督地分离不同 agent 的导航偏好。

**[Last Iterate Convergence In Monotone Mean Field Games](reinforcement_learning/last_iterate_convergence_in_monotone_mean_field_games.md)**

:   在非严格单调平均场博弈(MFG)中，提出基于 KL 散度的近端点(PP)方法实现渐近最后迭代收敛(LIC)，并证明正则化镜像下降(RMD)以指数速率收敛到正则化均衡，两者结合的 APP 算法在标准基准上可靠收敛到非正则化均衡。

**[Learning Interactive World Model for Object-Centric Reinforcement Learning](reinforcement_learning/learning_interactive_world_model_for_object-centric_reinforcement_learning.md)**

:   提出因子化交互式对象中心世界模型（FIOC-WM），通过显式学习对象间交互图和静态/动态属性分解，结合层级策略实现长程任务的有效分解，在属性/组合泛化任务上显著超越 Dreamer-V3 等基线。

**[Learning Memory-Enhanced Improvement Heuristics For Flexible Job Shop Scheduling](reinforcement_learning/learning_memory-enhanced_improvement_heuristics_for_flexible_job_shop_scheduling.md)**

:   用异构图表示+记忆增强GNN解决柔性作业车间调度(FJSP)，通过提升启发式策略超越SOTA构造型DRL方法，并具备规模无关的泛化能力。

**[Massively Parallel Imitation Learning of Mouse Forelimb Musculoskeletal Reaching Dynamics](reinforcement_learning/massively_parallel_imitation_learning_of_mouse_forelimb_musculoskeletal_reaching.md)**

:   基于 MIMIC-MJX 平台构建小鼠前肢肌肉骨骼模拟学习流水线，通过 JAX 加速的大规模并行 PPO（120 万步/秒）训练物理感知模仿学习策略，证明控制成本正则化能使模拟肌肉活动更好地预测真实 EMG 信号，并用基于 Takens 定理的非线性动力学方法从关节运动学预测肌肉激活。

**[Mean-Field Sampling for Cooperative Multi-Agent Reinforcement Learning](reinforcement_learning/mean-field_sampling_for_cooperative_multi-agent_reinforcement_learning.md)**

:   提出 SUBSAMPLE-MFQ 算法，通过从 $n$ 个智能体中随机采样 $k$ 个进行均场 Q 学习，将多智能体强化学习的样本复杂度从 $\text{poly}(n)$ 降低到 $\text{poly}(k)$，且性能差距仅为 $\tilde{O}(1/\sqrt{k})$（与 $n$ 无关），当 $k = O(\log n)$ 时实现相对均场 MARL 的指数加速。

**[Multi-Agent Collaboration via Evolving Orchestration](reinforcement_learning/multi-agent_collaboration_via_evolving_orchestration.md)**

:   提出"木偶师"(Puppeteer)式多 Agent 协作范式——一个中心化编排器通过 RL 学习在每个推理步骤动态选择激活哪个 Agent，在封闭域和开放域任务上同时提升性能和效率，并发现演化后的拓扑趋向更紧凑的环形结构。

**[NoisyRollout: Reinforcing Visual Reasoning with Data Augmentation](reinforcement_learning/noisyrollout_reinforcing_visual_reasoning_with_data_augmenta.md)**

:   提出NoisyRollout，一种简单有效的数据增强方法——在VLM的RL训练中混合使用干净图像和适度扭曲图像的生成轨迹，通过注入感知多样性促进策略探索和鲁棒推理，配合噪声退火调度，零额外计算成本实现5个域外推理benchmark上的开源RL模型SOTA。

**[Qimeng-Salv Signal-Aware Learning For Verilog Code Generation](reinforcement_learning/qimeng-salv_signal-aware_learning_for_verilog_code_generation.md)**

:   从部分正确的Verilog模块中提取信号级正确实现用于信号感知DPO训练，使7B模型在RTLLM v1.1上达到671B DeepSeek-v3的水平（62.6% pass@1）。

**[Real-World Reinforcement Learning of Active Perception Behaviors](reinforcement_learning/real-world_reinforcement_learning_of_active_perception_behaviors.md)**

:   提出 AAWR（Asymmetric Advantage Weighted Regression），通过训练时利用 privileged 传感器信息（如物体检测mask、真实位置）训练价值函数、部署时仅用部分观测，在8个真实机器人任务上大幅超越对称RL和模仿学习基线，实现高效主动感知行为学习。

**[Reasoning Gym: Reasoning Environments for Reinforcement Learning with Verifiable Rewards](reinforcement_learning/reasoning_gym_reasoning_environments_for_reinforcement_learning_with_verifiable_.md)**

:   发布Reasoning Gym库，包含100+可验证推理任务的过程生成环境，支持动态难度调整和无限数据生成，可用于RLVR训练和推理评估。

**[Reinforcement Learning for Long-Horizon Multi-Turn Search Agents](reinforcement_learning/reinforcement_learning_for_long-horizon_multi-turn_search_agents.md)**

:   展示 RL 训练的 14B 参数搜索 agent 在法律文档检索任务上通过多轮交互可以超越 frontier 模型（85% vs GPT o3 的 81%），关键在于精心设计的分段奖励结构和允许长 horizon 多轮交互。

**[Retrosynthesis Planning Via Worst-Path Policy Optimisation In Tree-Structured Md](reinforcement_learning/retrosynthesis_planning_via_worst-path_policy_optimisation_in_tree-structured_md.md)**

:   将逆合成规划重新建模为树结构MDP中的最差路径优化问题，用自模仿学习确保所有合成路线都能终止于可购买的起始材料。

**[RL Tango: Reinforcing Generator and Verifier Together for Language Reasoning](reinforcement_learning/rl_tango_reinforcing_generator_and_verifier_together_for_lan.md)**

:   Tango 提出一种交替 RL 训练生成器和验证器的框架——验证器是生成式过程级 LLM（用自然语言逐步评判），仅用结果级正确性奖励训练（无需步骤标注），通过与生成器的共进化相互增强——在 7B/8B 级别模型上达到SOTA，AIME 2025 准确率相对 vanilla GRPO 提升 100%。

**[Sample-Efficient Tabular Self-Play for Offline Robust Reinforcement Learning](reinforcement_learning/sample-efficient_tabular_self-play_for_offline_robust_reinforcement_learning.md)**

:   提出 RTZ-VI-LCB 算法用于离线鲁棒两人零和 Markov 博弈（RTZM G），通过乐观鲁棒值迭代 + Bernstein 风格惩罚，实现近最优样本复杂度 $O(C_r^* \cdot H^4 \cdot S \cdot (A+B) / \varepsilon^2)$，较此前最优结果 $O(H^5 \cdot S^2 \cdot AB / \varepsilon^2)$ 在状态空间和动作空间依赖上均有显著改善。

**[Structured Reinforcement Learning for Combinatorial Decision-Making](reinforcement_learning/structured_reinforcement_learning_for_combinatorial_decision-making.md)**

:   提出 Structured Reinforcement Learning (SRL)，将组合优化求解器作为可微层嵌入 actor-critic 的 actor 中，通过 Fenchel-Young 损失 + 高斯扰动实现端到端梯度传播，纯在线学习、无需专家数据，在6个工业级组合决策问题上匹配模仿学习、超越无结构 RL 最高 92%。

**[SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution](reinforcement_learning/swe-rl_advancing_llm_reasoning_via_reinforcement_learning_on_open_software_evolu.md)**

:   首次将强化学习 (RL) 应用于真实世界软件工程任务（GitHub PR/Issue 修复），仅用基于规则的序列相似度奖励训练 Llama-3.3-70B，在 SWE-bench Verified 上达到 41.0% 解决率（中等规模模型 SOTA），且 RL 训练仅在 issue-solving 数据上进行，却涌现出在代码推理、数学、通用语言理解等域外任务上的泛化推理能力。

**[The Burden Of Interactive Alignment With Inconsistent Preferences](reinforcement_learning/the_burden_of_interactive_alignment_with_inconsistent_preferences.md)**

:   研究用户与参与度驱动算法的交互对齐问题，证明存在关键视野阈值，充分的前瞻性或低成本信号可显著降低对齐负担。

**[Training Language Models to Reason Efficiently](reinforcement_learning/training_language_models_to_reason_efficiently.md)**

:   通过在 RL 奖励中加入长度惩罚项——正确回答的奖励乘以 $(1 - \alpha \cdot \sigma(\text{norm\_len}))$，用单一超参数 $\alpha$ 控制 token-准确率权衡曲线，仅 100 步 RL 训练即可让 7B 推理模型减少 50% token 使用量而准确率仅下降 <5%。

**[VolleyBots: A Testbed for Multi-Drone Volleyball Game Combining Motion Control and Strategic Play](reinforcement_learning/volleybots_a_testbed_for_multi-drone_volleyball_game_combining_motion_control_an.md)**

:   提出 VolleyBots——首个整合低层运动控制与高层策略规划的多无人机排球对抗平台，包含从单机训练到3v3对抗的9级任务课程，揭示了现有多智能体RL算法在需要联合控制+决策的复杂运动任务上的瓶颈。

**[Zero-Shot Context Generalization In Reinforcement Learning From Few Training Con](reinforcement_learning/zero-shot_context_generalization_in_reinforcement_learning_from_few_training_con.md)**

:   提出基于context-enhanced Bellman方程的数据增强方法，使RL agent在仅见过少量训练context的情况下零样本泛化到新context。

---

## 🦾 LLM Agent { #llm_agent }

**[A-MEM: Agentic Memory for LLM Agents](llm_agent/a-mem_agentic_memory_for_llm_agents.md)**

:   提出 A-Mem，一种受 Zettelkasten 启发的 LLM Agent 智能记忆系统，每条记忆自动生成结构化笔记（关键词/标签/上下文描述），动态建立记忆间链接，并在新记忆加入时触发旧记忆的演化更新，在 LoCoMo 长对话 QA 上显著超越 MemGPT 等基线。

**[A Differentiable Model of Supply-Chain Shocks](llm_agent/a_differentiable_model_of_supply-chain_shocks.md)**

:   本文用 JAX 实现了一个可微分的供应链 Agent-Based Model（ABM），通过 GPU 并行化和自动微分实现了比传统无梯度方法快 3 个数量级的贝叶斯参数校准，为大规模供应网络建模打开了可能性。

**[A Self-Improving Coding Agent](llm_agent/a_selfimproving_coding_agent.md)**

:   提出SICA（Self-Improving Coding Agent），一个能自主编辑自身代码库来提升性能的编程Agent——消除了meta-agent和target-agent的区分，通过迭代式自我改进在SWE-Bench Verified子集上从17%提升到53%。

**[Adaptive Cooperative Transmission Design for URLLC via Deep Reinforcement Learning](llm_agent/adaptive_cooperative_transmission_design_for_ultra-reliable_low-latency_communic.md)**

:   针对两跳协作中继通信中的URLLC难题，提出DRL-CoLA算法：将每跳传输参数配置建模为MDP，用双agent DQN在仅观测本地CSI和ARQ反馈下学习分布式时延感知传输策略，接近全局最优可靠性。

**[AgentAuditor: Human-Level Safety and Security Evaluation for LLM Agents](llm_agent/agentauditor_humanlevel_safety_and_security_evaluation_for_l.md)**

:   提出 AgentAuditor，一个通用的无训练记忆增强推理框架，使 LLM 评估者能模拟人类专家评估 agent 的安全与安全性——通过自适应提取结构化语义特征并生成CoT推理轨迹构建经验记忆，多阶段上下文感知 RAG 检索相关经验指导新案例评估，在自建的 ASSEBench（2293条记录×15类風险×29场景）上达到人类水平准确率。

**[AgentChangeBench: A Multi-Dimensional Evaluation Framework for Goal-Shift Robustness](llm_agent/agentchangebench_a_multi-dimensional_evaluation_framework_for_goal-shift_robustn.md)**

:   AgentChangeBench 是首个系统评估 LLM agent 在对话中途目标切换时适应能力的 benchmark：315 基础任务 × 9 变体 = 2835 序列，覆盖 3 个企业领域（银行/零售/航空）和 5 种 user persona，引入 GSRT（目标切换恢复时间）等 4 个互补指标，揭示高 pass@k 掩盖的效率和鲁棒性差距——如 GPT-4o 航空恢复率 92.2% 但零售冗余率达 89.1%。

**[AgentDAM: Privacy Leakage Evaluation for Autonomous Web Agents](llm_agent/agentdam_privacy_leakage_evaluation_for_autonomous_web_agent.md)**

:   提出 AgentDAM，首个在真实 Web 环境中端到端评估 AI Agent 数据最小化能力的基准，包含 246 个跨 Reddit/GitLab/Shopping 的任务，发现 GPT-4o 等主流模型在无缓解措施时隐私泄露率高达 36-46%，而 CoT 隐私提示可将泄露率降至 6-8%。

**[Agentic NL2SQL to Reduce Computational Costs](llm_agent/agentic_nl2sql_to_reduce_computational_costs.md)**

:   提出 Datalake Agent，一个基于交互循环的 agentic NL2SQL 系统，通过分层的信息获取策略（GetDBDescription -> GetTables -> GetColumns -> DBQueryFinalSQL）让 LLM 按需请求数据库 schema 信息而非一次性接收全部，在 319 张表的场景下将 token 使用量减少 87%、成本降低 8 倍，同时在复杂查询上保持更好的性能。

**[Agentic Plan Caching: Test-Time Memory for Fast and Cost-Efficient LLM Agents](llm_agent/agentic_plan_caching_test-time_memory_for_fast_and_cost-efficient_llm_agents.md)**

:   提出 Agentic Plan Caching (APC)——从 agent 执行日志中提取结构化计划模板，通过关键词匹配缓存命中后用小模型适配复用，平均降低 50.31% 成本和 27.28% 延迟，同时保持 96.61% 的最优准确率。

**[AgentMisalignment: Measuring the Propensity for Misaligned Behaviour in LLM-Based Agents](llm_agent/agentmisalignment_measuring_the_propensity_for_misaligned_behaviour_in_llm-based.md)**

:   提出 AgentMisalignment 基准套件，包含 9 个现实场景评估任务，测量 LLM Agent 在非恶意指令下 **自发偏离** 部署者意图的倾向（而非能力），发现更强的模型倾向于更高的错误对齐，且人格提示（persona prompt）有时比模型选择本身对错误对齐行为的影响更大。

**[AgentTTS: Large Language Model Agent for Test-time Compute-optimal Scaling Strategy in Complex Tasks](llm_agent/agenttts_large_language_model_agent_for_testtime_computeopti.md)**

:   提出 AgentTTS，一个用 LLM agent 自动搜索多阶段复杂任务中**测试时计算最优缩放策略**（模型选择+预算分配）的框架，通过迭代反馈驱动的交互显著提升搜索效率和性能。

**[Are Large Language Models Sensitive to the Motives Behind Communication?](llm_agent/are_large_language_models_sensitive_to_the_motives_behind_communication.md)**

:   系统评估 LLM 对通信动机的敏感性（motivational vigilance）——在控制实验中 LLM 能像人类一样折扣有偏见信源的建议（与理性模型相关系数 r>0.78），但在真实场景（YouTube 赞助广告）中表现大幅下降（r<0.2），通过简单的 prompt steering 可部分恢复。

**[Attractive Metadata Attack: Inducing LLM Agents to Invoke Malicious Tools](llm_agent/attractive_metadata_attack_inducing_llm_agents_to_invoke_malicious_tools.md)**

:   AMA（Attractive Metadata Attack）证明仅通过精心设计恶意工具的元数据（名称、描述、参数模式），不需要提示注入或模型内部访问，就能诱导 LLM Agent 以 81-95% 的成功率调用攻击者工具并泄露隐私，同时几乎不影响原始任务完成（98%+），且现有防御（审计器、提示重写）效果有限。

**[Automated Composition of Agents: A Knapsack Approach for Agentic Component Selection](llm_agent/automated_composition_of_agents_a_knapsack_approach_for_agentic_component_select.md)**

:   将 Agent 组件选择问题形式化为在线背包问题，提出 Composer Agent 框架：通过沙盒实测（而非静态语义检索）评估组件真实能力，结合 ZCL 在线算法在预算约束下动态选取最优组件组合，单 Agent 工具选择成功率提升最高 31.6%，多 Agent 子代理选择成功率从 37% 跃升至 87%。

**[VeriMaAS: Automated Multi-Agent Workflows for RTL Design](llm_agent/automated_multi-agent_workflows_for_rtl_design.md)**

:   VeriMaAS 提出自动组合 agent 工作流的框架用于 RTL 代码生成——关键创新是将 HDL 工具的形式化验证反馈直接整合到工作流生成中，无需梯度更新或长推理链，在 pass@k 上超过微调基线 5-7%，且训练样本需求降低一个量级。

**[Benchmarking Agentic Systems in Automated Scientific Information Extraction](llm_agent/benchmarking_agentic_systems_in_automated_scientific_information_extraction_with.md)**

:   构建 ChemX——10 个人工标注的多模态化学数据提取数据集，系统评估 SOTA Agent 系统（ChatGPT Agent、SLM-Matrix、FutureHouse、nanoMINER）和 LLM（GPT-5 等），发现单 Agent 方法（GPT-5）在纳米酶数据集上达到 F1=0.58，超越专用多 Agent 系统。

**[BTL-UI: Blink-Think-Link Reasoning Model for GUI Agent](llm_agent/btlui_blinkthinklink_reasoning_model_for_gui_agent.md)**

:   提出"Blink-Think-Link"（BTL）脑启发框架模拟人类与GUI交互的认知过程——分解为Blink（快速注意力检测，类似眼跳）、Think（高级推理决策，类似认知规划）、Link（生成可执行命令，类似动作选择）三个生物合理阶段，配合自动化Blink数据标注和首个基于规则的过程+结果复合奖励机制，BTL-UI在静态GUI理解和动态交互任务上均达competitive性能。

**[CAM: A Constructivist View of Agentic Memory for LLM-Based Reading Comprehension](llm_agent/cam_a_constructivist_view_of_agentic_memory_for_llm-based_reading_comprehension.md)**

:   受皮亚杰建构主义理论启发，提出CAM——一种具有结构性（层次化schema）、灵活性（重叠聚类的同化）和动态性（增量适应）三大特征的智能体记忆系统，在6个长文本阅读理解任务上全面超越RAPTOR、GraphRAG等基线。

**[ContextAgent: Context-Aware Proactive LLM Agents with Open-World Sensory Perceptions](llm_agent/contextagent_context-aware_proactive_llm_agents_with_open-world_sensory_percepti.md)**

:   提出 ContextAgent，首个利用可穿戴设备多模态感知（视频+音频+通知）来理解用户意图并主动提供工具增强服务的 LLM Agent 框架，同时构建了包含 1000 个样本的 ContextAgentBench 基准，在主动预测准确率和工具调用上分别提升 8.5% 和 6.0%。

**[CORE: Full-Path Evaluation of LLM Agents Beyond Final State](llm_agent/core_full-path_evaluation_of_llm_agents_beyond_final_state.md)**

:   提出CORE框架：用确定有限自动机（DFA）编码Agent任务的合法工具调用路径，引入5个互补指标（路径正确性、顺序正确性、前缀危险性、有害调用率、效率）从全路径而非仅终态评估Agent行为，揭示了传统终态评估中不可见的安全和效率差异。

**[Crucible: Quantifying the Potential of Control Algorithms through LLM Agents](llm_agent/crucible_quantifying_the_potential_of_control_algorithms_through_llm_agents.md)**

:   首次提出"调优潜能"（Tuning Potential）概念并给出形式化度量，通过 LLM Agent 模拟不同能力水平的开发者对控制算法进行参数调优和逻辑级改进，在 ABR 任务上相比贝叶斯优化提升 44.1%，CartPole 上 Bang-bang 从 34→500 达到 DQN 水平。

**[Debate or Vote: Which Yields Better Decisions in Multi-Agent Large Language Models?](llm_agent/debate_or_vote_which_yields_better_decisions_in_multi-agent_large_language_model.md)**

:   通过理论和实验证明，多智能体辩论（MAD）的性能提升主要来自多数投票（ensembling）而非辩论本身——辩论过程构成 martingale（期望不变），即辩论不系统性地提升正确率，并基于此理论提出通过偏向正确信号来改进 MAD。

**[Deep Video Discovery: Agentic Search with Tool Use for Long-form Video Understanding](llm_agent/deep_video_discovery_agentic_search_with_tool_use_for_longfo.md)**

:   提出 DVD（Deep Video Discovery）agent，将长视频理解建模为多步信息搜索问题：先将长视频构建为多粒度结构化数据库（全局摘要 + clip 级字幕嵌入 + 帧级像素），再提供三种搜索工具（Global Browse / Clip Search / Frame Inspect），由 reasoning LLM 通过 observe-reason-act 循环自主编排搜索轨迹，在 LVBench 达 74.2%（超先前 SOTA MR.Video 13.4 pp），加字幕 76.0%。

**[Distilling LLM Agent into Small Models with Retrieval and Code Tools](llm_agent/distilling_llm_agent_into_small_models_with_retrieval_and_co.md)**

:   提出 Agent Distillation 框架，将 LLM agent 的完整 reason-act-observe 交互行为（而非静态 CoT）蒸馏到 0.5B-7B 小模型中，配合 first-thought prefix 提升教师轨迹质量和 self-consistent action generation 提升推理鲁棒性，使小模型达到比其大 2-4× 的 CoT 蒸馏模型的性能。

**[DRIFT: Dynamic Rule-Based Defense with Injection Isolation for Securing LLM Agents](llm_agent/drift_dynamic_rulebased_defense_with_injection_isolation_for.md)**

:   提出 DRIFT 系统级 Agent 安全框架，通过 Secure Planner（预规划函数轨迹+参数检查表）、Dynamic Validator（基于 Read/Write/Execute 权限的动态策略更新）和 Injection Isolator（从 memory stream 中检测并屏蔽注入指令）三层防御，在 AgentDojo 上将 ASR 从 30.7% 降至 1.3%，同时比 CaMeL 提升 20.1% utility。

**[Evaluating LLMs in Open-Source Games](llm_agent/evaluating_llms_in_open-source_games.md)**

:   通过开源游戏（智能体提交程序而非原始行动）这一新范式，系统评估 LLM 在战略推理、互相学习和合作博弈中的能力，发现 LLM 可自动发现近似程序平衡。

**[Ground-Compose-Reinforce: Grounding Language in Agentic Behaviours using Limited Data](llm_agent/ground-compose-reinforce_grounding_language_in_agentic_behaviours_using_limited_.md)**

:   提出 Ground-Compose-Reinforce (GCR)，一个端到端的神经符号框架，通过少量标注轨迹（仅350条）学习原子命题的接地语义（Ground），将其通过 Reward Machine 组合成复杂任务规范（Compose），然后用自生成的稠密奖励训练 RL 智能体（Reinforce），无需手工奖励函数即可引出分布外的复杂行为。

**[Group-in-Group Policy Optimization for LLM Agent Training](llm_agent/groupingroup_policy_optimization_for_llm_agent_training.md)**

:   GiGPO 通过在 GRPO 的 episode 级分组内嵌套 step 级分组（利用跨轨迹的重复环境状态作为 anchor state），实现了无需额外 rollout 和 critic 模型的细粒度 credit assignment，在 ALFWorld 上比 GRPO 提升 >12%，WebShop 上提升 >9%。

**[It's LIT! Reliability-Optimized LLMs with Inspectable Tools](llm_agent/its_lit_reliability-optimized_llms_with_inspectable_tools.md)**

:   通过为工具分配可靠性和可调试性成本，引导 LLM 主动选择更透明、更易理解的工具，在 61/65 测试场景中提高可解释性，同时保持性能。

**[Lessons Learned: A Multi-Agent Framework for Code LLMs to Learn and Improve](llm_agent/lessons_learned_a_multi-agent_framework_for_code_llms_to_learn_and_improve.md)**

:   提出 LessonL 框架，使多个小 LLM 智能体通过相互学习的"课程"(lesson)对成功和失败案例进行反思，协同优化代码性能，3 个 7B-14B 模型组合达到 GPT-4o 甚至接近 o3 的代码优化效果。

**[MAT-Agent: Adaptive Multi-Agent Training Optimization](llm_agent/mat-agent_adaptive_multi-agent_training_optimization.md)**

:   提出 MAT-Agent，一个由四个自主 agent（分别负责数据增强、优化器、学习率调度、损失函数）组成的多智能体框架，在训练过程中动态调整训练配置，用 DQN 学习策略以替代传统静态超参配置，在多标签图像分类任务上实现了 SOTA。

**[MLRC-Bench: Can Language Agents Solve Machine Learning Research Challenges?](llm_agent/mlrc-bench_can_language_agents_solve_machine_learning_research_challenges.md)**

:   基于真实 ML 会议竞赛构建动态基准 MLRC-Bench，评估 LLM Agent 提出和实现新颖研究方法的能力，发现最强 Agent（Gemini）仅达人类顶级方案 9.3% 的相对改进，且提供 AI/人类想法并不能一致改善实现质量。

**[Out of Control -- Why Alignment Needs Formal Control Theory (and an Alignment Control Stack)](llm_agent/out_of_control_--_why_alignment_needs_formal_control_theory_and_an_alignment_con.md)**

:   本文是一篇 position paper，主张将形式化最优控制理论作为 AI 对齐研究的核心工具，并提出"对齐控制栈"(Alignment Control Stack, ACS)——一个从物理硬件层到社会治理层的十层分层框架，用于系统地组织和分析不同对齐方法的测量、控制与互操作性。

**[SuffixDecoding: Extreme Speculative Decoding for Emerging AI Applications](llm_agent/suffixdecoding_extreme_speculative_decoding_for_emerging_ai_applications.md)**

:   利用后缀树缓存长序列，通过自适应推测长度实现 5.3 倍加速，特别针对 Agent 场景中高度可预测的重复推理任务。

**[T1: A Tool-Oriented Conversational Dataset for Multi-Turn Agentic Planning](llm_agent/t1_a_tool-oriented_conversational_dataset_for_multi-turn_agentic_planning.md)**

:   构建 T1 数据集——13.5K 多轮对话覆盖 9 个领域（4 单领域 + 5 跨领域）、14 个工具，聚焦工具间依赖和动态重规划，并提出 T1-Agent（代码生成 + 缓存机制）作为基线系统；实验发现 SFT 后的 Llama 8B 在 Tool Call F1 上达 87.17%，超越未微调的 70B 模型，但仍落后于 GPT-5/o3 等闭源模型。

**[The Lighthouse of Language: Enhancing LLM Agents via Critique-Guided Improvement](llm_agent/the_lighthouse_of_language_enhancing_llm_agents_via_critique-guided_improvement.md)**

:   提出 CGI（Critique-Guided Improvement）双角色框架，训练专门的 Critic 模型为 Actor Agent 提供结构化自然语言反馈（判别+修正建议），并通过迭代动作精炼让 Actor 学会利用这些反馈，在 WebShop/ScienceWorld/TextCraft 三个环境中平均得分 74.20%，超越 GPT-4o（45.46%）和 Iterative SFT（58.21%）。

**[TrajAgent: An LLM-Agent Framework for Trajectory Modeling via Large-and-Small Model Collaboration](llm_agent/trajagent_an_llm-agent_framework_for_trajectory_modeling_via_large-and-small_mod.md)**

:   首个 LLM 代理框架自动处理轨迹建模全流程，通过 UniEnv 统一接口和协作学习双层优化（LLM 推理 + 小模型训练），性能相比基线最高提升 69.91%。

**[Web-Shepherd: Advancing PRMs for Reinforcing Web Agents](llm_agent/web-shepherd_advancing_prms_for_reinforcing_web_agents.md)**

:   提出首个针对网页导航的过程奖励模型 Web-Shepherd，通过检查清单分解任务目标为可评估的子目标，3B/8B 模型在轨迹准确率上碾压 GPT-4o（85% vs 10%），同时成本仅为 1/10，使网页 Agent 的强化学习和推理时搜索变得实际可行。

---

## ⚖️ 对齐 / RLHF { #llm_alignment }

**[A Systematic Evaluation of Preference Aggregation in Federated RLHF for Pluralistic Alignment of LLMs](llm_alignment/a_systematic_evaluation_of_preference_aggregation_in_federated_rlhf_for_pluralis.md)**

:   提出一种自适应 Alpha 聚合策略，在联邦 RLHF 框架中根据各用户群体的历史对齐表现动态调整奖励权重，从而在多元偏好对齐中同时实现高公平性和强对齐性能。

**[Alignment of Large Language Models with Constrained Learning](llm_alignment/alignment_of_large_language_models_with_constrained_learning.md)**

:   将LLM对齐形式化为约束优化问题（最大化主要奖励同时满足次要效用约束如安全性），提出基于拉格朗日对偶的迭代方法交替更新LLM策略和对偶变量，理论上刻画了分布空间与LLM参数空间之间的原对偶间隙和最优性间隙，证明方法可以找到近最优约束LLM策略。

**[Ask a Strong LLM Judge when Your Reward Model is Uncertain](llm_alignment/ask_a_strong_llm_judge_when_your_reward_model_is_uncertain.md)**

:   提出基于不确定性的路由框架，用SNGP对pairwise reward model做不确定性量化，将高认知不确定性的样本路由到强LLM judge（DeepSeek-R1），在仅调用9.2%~42.5% judge的成本下显著超越随机路由的准确率，且有效改善下游在线RLHF对齐效果。

**[Attack via Overfitting: 10-shot Benign Fine-tuning to Jailbreak LLMs](llm_alignment/attack_via_overfitting_10-shot_benign_fine-tuning_to_jailbreak_llms.md)**

:   提出两阶段微调攻击：第一阶段用10个问题配相同拒绝答案使LLM过拟合到窄最优解（尖锐loss landscape），第二阶段用相同10个问题配正常答案触发灾难性遗忘——安全对齐被"忘掉"，仅用完全良性数据即达94.84%越狱成功率，与恶意微调（97.25%）相当且完全绕过审核模型。

**[Can DPO Learn Diverse Human Values? A Theoretical Scaling Law](llm_alignment/can_dpo_learn_diverse_human_values_a_theoretical_scaling_law.md)**

:   建立了 DPO 在多元人类价值设定下的理论泛化框架——通过分析有限梯度步后 reward margin 的动态轨迹，证明了每种价值所需样本量必须随价值类别数 $K$ 对数增长（$Q = \Theta(\log K)$）才能维持泛化性能，揭示了对齐多元化社会价值的统计代价。

**[Capturing Individual Human Preferences with Reward Features](llm_alignment/capturing_individual_human_preferences_with_reward_features.md)**

:   提出奖励特征模型（RFM）：学习共享奖励特征 $\phi_\theta(x,y)$，每个用户通过线性权重 $\mathbf{w}_h$ 组合这些特征得到个性化奖励 $r_h = \langle \phi_\theta, \mathbf{w}_h \rangle$，并首次给出多评价者偏好学习的PAC泛化界，证明增加评价者数 $m$ 比增加每人样本数 $n$ 更有效，仅30个样本即可快速适应新用户。

**[Concept-Level Explainability for Auditing & Steering LLM Responses](llm_alignment/concept-level_explainability_for_auditing_steering_llm_responses.md)**

:   提出 ConceptX，一种基于概念级（而非 token 级）Shapley 归因的 LLM 可解释性方法，通过语义相似度而非 token 重合度来衡量输入概念对输出的影响，可用于审计偏见和通过 prompt 编辑引导 LLM 输出，在越狱防御中将攻击成功率从 0.463 降至 0.242。

**[Deep Research Brings Deeper Harm](llm_alignment/deep_research_brings_deeper_harm.md)**

:   揭示 Deep Research (DR) 智能体的严重安全隐患——即使底层 LLM 能正确拒绝有害请求，部署为 DR 智能体后仍能生成详细专业的危险报告；提出 Plan Injection 和 Intent Hijack 两种针对性越狱方法，以及 DeepREJECT 评估指标，在 6 个 LLM 上验证了 DR 智能体系统性地削弱了对齐机制。

**[DeepVideo-R1: Video Reinforcement Fine-Tuning via Difficulty-aware Regressive GRPO](llm_alignment/deepvideor1_video_reinforcement_finetuning_via_difficultyawa.md)**

:   探索GRPO在VideoLLM中的应用，发现"安全门依赖"和"优势消失"两个阻碍有效学习的问题，提出Reg-GRPO（将GRPO loss重建为直接回归优势值的任务，消除clipping/min等安全门操作）和难度感知数据增强策略，在多个视频推理benchmark上显著提升性能。

**[DenseDPO: Fine-Grained Temporal Preference Optimization for Video Diffusion Models](llm_alignment/densedpo_finegrained_temporal_preference_optimization_for_vi.md)**

:   提出 DenseDPO，通过三个创新解决视频扩散模型 DPO 训练的根本缺陷：(1) 从 GT 视频加噪去噪构造对齐的视频对消除运动偏差，(2) 在短时间片段而非整个视频上标注偏好提供更密集的学习信号，(3) 用 GPT 等 VLM 自动标注片段级偏好取代人工标注。仅用 1/3 标注数据即大幅提升运动生成质量。

**[Diffusion Model as a Noise-Aware Latent Reward Model for Step-Level Preference Optimization](llm_alignment/diffusion_model_as_a_noiseaware_latent_reward_model_for_step.md)**

:   提出 Latent Reward Model (LRM) 和 Latent Preference Optimization (LPO)，将预训练扩散模型本身复用为噪声感知的潜空间奖励模型，在噪声潜在空间直接进行步级偏好优化，相比 Diffusion-DPO 实现 10-28× 训练加速，相比 SPO 实现 2.5-3.5× 加速。

**[EvoRefuse: Evolutionary Prompt Optimization for Evaluation and Mitigation of LLM Over-Refusal](llm_alignment/evorefuse_evolutionary_prompt_optimization_for_evaluation_and_mitigation_of_llm_.md)**

:   提出 EvoRefuse——用进化搜索（变异/重组 + ELBO 适应度 + 模拟退火）生成语义无害但能可靠触发 LLM 拒绝的"伪恶意"指令，比最强基线的拒绝触发率高 85.34%，并用生成的数据进行 SFT/DPO 微调，将过度拒绝降低 29.85%-45.96%。

**[From Judgment to Interference: Early Stopping LLM Harmful Outputs via Streaming Content Monitoring](llm_alignment/from_judgment_to_interference_early_stopping_llm_harmful_outputs_via_streaming_c.md)**

:   提出 Streaming Content Monitor (SCM)——首个原生支持部分检测的流式有害内容监控器，通过 FineHarm 数据集（29K 样本含 token 级标注）和层次一致性感知学习，平均仅需看到 18% 的 response tokens 即可达到 0.95+ macro F1，实现对 LLM 有害输出的实时早停。

**[g-DPO: Scalable Preference Optimization for Protein Language Models](llm_alignment/g-dpo_scalable_preference_optimization_for_protein_language_models.md)**

:   针对蛋白质语言模型的偏好优化提出 g-DPO，通过序列空间聚类剪枝冗余对和 union masking 摊销似然计算，在保持标准 DPO 性能的同时实现 1.7-5.4× 训练加速。

**[Gasp Efficient Black-Box Generation Of Adversarial Suffixes For Jailbreaking Llm](llm_alignment/gasp_efficient_black-box_generation_of_adversarial_suffixes_for_jailbreaking_llm.md)**

:   提出GASP框架，通过训练专用的SuffixLLM生成可读的对抗后缀，利用潜在贝叶斯优化（LBO）在连续嵌入空间中高效搜索并用ORPO迭代微调生成器，在完全黑盒设置下实现高攻击成功率且生成的后缀保持人类可读性。

**[Greedy Sampling Is Provably Efficient For Rlhf](llm_alignment/greedy_sampling_is_provably_efficient_for_rlhf.md)**

:   证明了在KL正则化的RLHF设置下，直接使用经验估计的贪心采样（无需构建乐观/悲观估计）就能在在线和离线两种设置中实现$O(\log T)$遗憾界和$O(\varepsilon^{-1})$样本复杂度，这是首次在一般偏好模型下达到这些阶数。

**[GVPO: Group Variance Policy Optimization for Large Language Model Post-Training](llm_alignment/gvpo_group_variance_policy_optimization_for_large_language_model_post-training.md)**

:   通过将 KL 约束奖励最大化的解析解融入梯度权重（零和权重消除配分函数），设计了比 GRPO 更稳定的 LLM 后训练方法 GVPO，在 AIME 上达到 20.72%（GRPO 14.79%），并证明具有唯一全局最优解。

**[HelpSteer3-Preference: Open Human-Annotated Preference Data across Diverse Tasks and Languages](llm_alignment/helpsteer3-preference_open_human-annotated_preference_data_across_diverse_tasks_.md)**

:   NVIDIA 发布的 40K+ 开源人工标注偏好数据集，覆盖通用/STEM/代码/多语言（13 种语言），训练的奖励模型在 RM-Bench 上达 82.4%（+10%），CC-BY-4.0 许可对商业友好。

**[LASeR: Learning to Adaptively Select Reward Models with Multi-Armed Bandits](llm_alignment/laser_learning_to_adaptively_select_reward_models_with_multi-armed_bandits.md)**

:   将多个奖励模型（RM）的选择建模为上下文多臂老虎机（LinUCB）问题，在迭代 LLM 训练中自适应地为每个 batch 选择最合适的 RM，在推理、指令跟随和长上下文任务上以 2-3 倍效率优势全面超越 RM 集成和单 RM 基线。

**[LLM Safety Alignment is Divergence Estimation in Disguise](llm_alignment/llm_safety_alignment_is_divergence_estimation_in_disguise.md)**

:   建立统一理论框架证明 RLHF/DPO/KTO/BCO 等对齐方法本质上是在估计安全分布 $\mathcal{D}^+$ 与不安全分布 $\mathcal{D}^-$ 之间的散度，由此解释了对齐后隐空间分离现象，并提出基于 KL 散度的 KLDO 对齐方法，在 5 个模型上实现最佳鲁棒性。

**[LongVPO: From Anchored Cues to Self-Reasoning for Long-Form Video Preference Optimization](llm_alignment/longvpo_from_anchored_cues_to_selfreasoning_for_longform_vid.md)**

:   提出 LongVPO，一个两阶段 DPO 框架使短上下文 VLM 无需长视频标注即可理解超长视频——阶段1通过锚定短片段构造偏好数据解决位置偏差问题，阶段2通过递归描述+多段推理任务培养跨片段推理能力，仅用 16K 合成样本即超越 SOTA 开源模型。

**[Mechanism Design for LLM Fine-tuning with Multiple Reward Models](llm_alignment/mechanism_design_for_llm_fine-tuning_with_multiple_reward_models.md)**

:   将多方偏好聚合的 RLHF 微调建模为机制设计问题，证明了在社会福利最大化训练规则下各方有动机虚报偏好，并通过扩展 VCG 支付机制实现了占优策略激励相容（DSIC），确保各方如实报告偏好。

**[On Extending Direct Preference Optimization to Accommodate Ties](llm_alignment/on_extending_direct_preference_optimization_to_accommodate_ties.md)**

:   将 DPO 中的 Bradley-Terry 偏好模型替换为 Rao-Kupper 和 Davidson 扩展，使偏好优化能够显式建模"平局"数据，避免丢弃模糊偏好对，在翻译和数学推理上获得更好的正则化和性能。

**[Preference Optimization by Estimating the Ratio of the Data Distribution](llm_alignment/preference_optimization_by_estimating_the_ratio_of_the_data_distribution.md)**

:   将 DPO 重新解释为似然比估计（ratio matching）问题，基于 Bregman 散度框架提出 BPO（Bregman Preference Optimization），包含 DPO 为特例的广义损失函数族，并设计了 SBA（Scaled Basu's Power Divergence）实例，在 Llama-3-8B 上实现 55.9% AlpacaEval2 length-controlled win rate 的 SOTA。

**[Robust LLM Alignment via Distributionally Robust Direct Preference Optimization](llm_alignment/robust_llm_alignment_via_distributionally_robust_direct_preference_optimization.md)**

:   通过分布鲁棒优化（DRO）框架提出 WDPO（Wasserstein）和 KLDPO（KL散度）两种鲁棒 DPO 变体，解决用户偏好分布转移导致的对齐失败问题，提供 $O(n^{-1/4})$ 收敛保证，在多维对齐任务和 OpenLLM 榜单上显著优于标准 DPO。

**[Short-length Adversarial Training Helps LLMs Defend Long-length Jailbreak Attacks](llm_alignment/short-length_adversarial_training_helps_llms_defend_long-length_jailbreak_attack.md)**

:   理论证明并实验验证：防御长度 $\Theta(M)$ 的后缀越狱攻击，只需要在长度 $\Theta(\sqrt{M})$ 的对抗后缀上做对抗训练即可，即"短对抗训练防长越狱"——在5个主流LLM上，20 token 对抗训练可将 120 token 越狱成功率降低至少 30%。

**[Simplicity Prevails: Rethinking Negative Preference Optimization for LLM Unlearning](llm_alignment/simplicity_prevails_rethinking_negative_preference_optimization_for_llm_unlearni.md)**

:   提出 SimNPO，通过移除参考模型依赖并采用长度归一化奖励替代 NPO 的参考模型比较，简化设计但更有效适配数据难度差异，TOFU FQ 从 0.79 提升至 0.91+。

**[Trajectory Bellman Residual Minimization: A Simple Value-Based Method for LLM Reasoning](llm_alignment/trajectory_bellman_residual_minimization_a_simple_value-based_method_for_llm_rea.md)**

:   TBRM 通过最小化轨迹级贝尔曼残差，将 LLM 输出 logits 视为隐式 Q 值，仅需每个 prompt 一次前向采样即可训练，复杂度远低于 PPO/GRPO 但数学推理性能相当或更优。

**[What Makes a Reward Model a Good Teacher? An Optimization Perspective](llm_alignment/what_makes_a_reward_model_a_good_teacher_an_optimization_perspective.md)**

:   从优化理论角度证明：奖励模型的准确率（accuracy）不足以衡量其作为 RLHF "教师"的质量——即使完美准确的奖励模型，如果诱导的奖励方差（reward variance）过低，也会导致 RLHF 目标函数景观平坦，使 policy gradient 优化极慢；不同的语言模型需要不同的奖励模型。

---

## 📦 模型压缩 { #model_compression }

**[3DID: Direct 3D Inverse Design for Aerodynamics with Physics-Aware Optimization](model_compression/3did_direct_3d_inverse_design_for_aerodynamics_with_physics-aware_optimization.md)**

:   提出 3DID 框架，通过学习物理-几何统一的三平面隐空间表示 + 目标梯度引导扩散采样 + 拓扑保持精炼的两阶段策略，从随机噪声开始直接在完整 3D 空间中进行逆向设计，在车辆气动外形优化上，模拟阻力（Sim-Drag）相比最优基线降低 13.6%。

**[4DGCPro: Efficient Hierarchical 4D Gaussian Compression for Progressive Volumetric Video Streaming](model_compression/4dgcpro_efficient_hierarchical_4d_gaussian_compression_for_p.md)**

:   提出层级化的4D高斯压缩框架4DGCPro，通过感知加权的层级高斯表示、运动感知自适应分组和端到端熵优化训练，在单一模型内实现多码率渐进式体积视频流媒体，可在移动设备上实时解码和渲染，RD性能超越现有SOTA。

**[A*-Thought: Efficient Reasoning via Bidirectional Compression for Low-Resource Settings](model_compression/a-thought_efficient_reasoning_via_bidirectional_compression_for_low-resource_set.md)**

:   提出 A*-Thought——基于 A* 搜索算法的 CoT 压缩框架，通过双向重要性评分（BIS）衡量每个推理步骤对问题和答案的相关性，结合路径级 A* 搜索在指数级搜索空间中高效找到最紧凑的推理路径，在 512 token 预算下将 QwQ-32B 准确率提升 2.39 倍，在 4096 token 预算下减少约 50% 输出 token 且几乎不损失准确率。

**[A Granular Study of Safety Pretraining under Model Abliteration](model_compression/a_granular_study_of_safety_pretraining_under_model_abliteration.md)**

:   本文系统地研究了 model abliteration（一种推理时激活空间编辑攻击）对不同数据驱动安全预训练阶段的影响，发现仅依赖 refusal 训练的安全机制极易被攻破，而 **组合多种安全信号**（safe-only 过滤 + 改写 + metatag + refusal）可使安全行为分散到更广泛的表征空间、从而更难被单一方向投影移除。

**[A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders](model_compression/a_is_for_absorption_studying_feature_splitting_and_absorption_in_sparse_autoenco.md)**

:   发现并系统研究了 SAE 中的"特征吸收"现象：看似单义的 SAE latent 会在特定 token 上不激活，其特征方向被更具体的子 latent "吸收"，这是层级特征+稀疏性损失的必然结果，对 SAE 用于可靠解释 LLM 构成根本挑战。

**[A Partition Cover Approach for Tokenization](model_compression/a_partition_cover_approach_to_tokenization.md)**

:   将分词（tokenization）问题重新建模为**分区覆盖（partition cover）**优化问题，证明其为NP-hard，并提出多项式时间的贪心算法GreedTok，在压缩率和1B参数LLM预训练下游任务上均优于BPE。

**[A Token is Worth over 1,000 Tokens: Efficient Knowledge Distillation through Low-Rank Clone](model_compression/a_token_is_worth_over_1000_tokens_efficient_knowledge_distillation_through_low-r.md)**

:   提出 Low-Rank Clone (LRC)，通过可学习低秩投影矩阵将 teacher 权重压缩为 student 权重（软剪枝），同时对齐 attention 和 FFN 的中间激活（激活克隆），仅用 20B tokens 训练的 1.7B 模型即超过用 36T tokens 训练的 Qwen3-1.7B（64.98 vs 63.17），实现 **1000 倍训练效率提升**。

**[Accurate and Efficient Low-Rank Model Merging in Core Space](model_compression/accurate_and_efficient_low-rank_model_merging_in_core_space.md)**

:   提出 Core Space Merging 框架——通过在低秩 LoRA 矩阵的公共参考基空间中进行模型合并，**无信息损失**地将合并操作从 $m \times n$ 全尺寸空间压缩到 $Tr \times Tr$ 紧凑空间（$T$ 为任务数，$r$ 为 LoRA 秩），在 Llama 3 8B 上达到 SOTA 合并精度同时计算成本降低数个数量级。

**[Adaptive Kernel Design for Bayesian Optimization Is a Piece of CAKE with LLMs](model_compression/adaptive_kernel_design_for_bayesian_optimization_is_a_piece_of_cake_with_llms.md)**

:   提出 CAKE (Context-Aware Kernel Evolution)，利用 LLM 作为遗传算法的交叉和变异算子，在贝叶斯优化过程中自适应地生成和进化 GP 核函数表达式，结合 BAKER 排序机制平衡模型拟合（BIC）与期望改进（EI），在超参数优化、控制器调参和光子芯片设计等任务上持续超越固定核和自适应核基线。

**[Adaptive Originality Filtering: Rejection-Based Prompting and RiddleScore for Culturally Grounded Multilingual Riddle Generation](model_compression/adaptive_originality_filtering_rejection_based_prompting_and_riddlescore_for_cul.md)**

:   提出 Adaptive Originality Filtering (AOF)——一种基于语义拒绝采样的提示策略，通过 MiniLM 嵌入的余弦相似度过滤重复/模板化输出，强制 LLM 生成更新颖、多样且文化匹配的多语言谜语；同时提出 RiddleScore 复合评估指标（Novelty + Diversity + Fluency + Alignment），与人类评分相关性达 $\rho=0.83$。

**[Adaptive Prediction-Powered AutoEval with Reliability and Efficiency Guarantees](model_compression/adaptive_predictionpowered_autoeval_with_reliability_and_eff.md)**

:   提出R-AutoEval+，通过e-value赌注算法自适应调整对合成数据（LLM评判器）的依赖权重，首次同时提供有限样本可靠性保证和可证明的采样效率改善，在GSM8K上比纯真实数据方法节省87个token。

**[Adaptive Stochastic Coefficients for Accelerating Diffusion Sampling](model_compression/adaptive_stochastic_coefficients_for_accelerating_diffusion_sampling.md)**

:   通过理论分析 ODE 和 SDE 求解器的互补弱点（ODE 积累不可消除的梯度误差，SDE 在少步时离散化误差放大），提出 AdaSDE——在每个去噪步引入可学习随机系数 $\gamma_i$ 控制噪声注入强度，通过轻量蒸馏优化，在 5 NFE 下实现 CIFAR-10 FID 4.18、FFHQ FID 8.05 的 SOTA。

**[AdmTree: Compressing Lengthy Context with Adaptive Semantic Trees](model_compression/admtree_compressing_lengthy_context_with_adaptive_semantic_trees.md)**

:   提出 AdmTree——一种自适应层次化上下文压缩框架,通过信息密度驱动的动态分段构建叶 gist token，再用二叉语义树底向上聚合实现多粒度语义保留，解决了显式方法丢失局部细节和隐式方法位置偏差的双重问题,在 LongBench 上比 SOTA 基线 Activation Beacon 高 10%+。

**[AI-Generated Video Detection via Perceptual Straightening](model_compression/ai-generated_video_detection_via_perceptual_straightening.md)**

:   提出 ReStraV 方法，基于"感知拉直"假说（真实视频在神经表示空间形成更直的轨迹），利用 DINOv2 特征空间中的时间曲率和步距统计量训练轻量分类器检测 AI 生成视频，在 VidProM 上达到 97.17% 准确率和 98.63% AUROC，推理仅需 ~48ms。

**[AutoDiscovery: Open-ended Scientific Discovery via Bayesian Surprise](model_compression/autodiscovery_open-ended_scientific_discovery_via_bayesian_surprise.md)**

:   AutoDiscovery 提出用贝叶斯惊奇度（Bayesian Surprise）作为开放式科学发现的客观奖励信号——通过 LLM 采样估计先验/后验信念分布的 KL 散度，配合 MCTS+渐进展宽在假设空间中探索，在 21 个真实数据集上比贪心/束搜索产生 5-29% 更多的惊奇发现，人类评估确认贝叶斯惊奇度与专家"惊讶感"的一致性（0.67）远超 LLM 自身评估的"新颖性"和"有用性"。

**[AutoJudge: Judge Decoding Without Manual Annotation](model_compression/autojudge_judge_decoding_without_manual_annotation.md)**

:   AutoJudge 自动化了 Judge Decoding 中"重要 token"的标注——通过半贪心搜索替换不匹配 token 并检查答案是否改变来标注重要性，训练逻辑回归分类器预测 token 重要性，使投机解码每轮接受 40+ token（vs 标准 ~20），在 GSM8K 上加速 1.5× 且准确率损失 <1%。

**[BaRISTA: Brain-Scale Informed Spatiotemporal Representation of Human Intracranial EEG](model_compression/barista_brain_scale_informed_spatiotemporal_representation_of_human_intracranial.md)**

:   BaRISTA 系统探索 iEEG Transformer 的空间编码尺度（电极/脑区/脑叶），发现脑区级编码 + 空间掩码重建在语言任务解码上达 86.2% AUC（vs PopT 79.5%），编码尺度选择的影响 > 掩码策略选择，且跨被试泛化性好。

**[Benford's Curse: Tracing Digit Bias to Numerical Hallucination in LLMs](model_compression/benfords_curse_tracing_digit_bias_to_numerical_hallucination_in_llms.md)**

:   本文发现 LLM 的数值幻觉根源于预训练语料中符合 Benford 定律的数字频率分布——数字 1 出现概率 ~30% 而数字 9 仅 ~5%，这种偏差被 FFN 后期层的特定"数字选择性神经元"内化，提出数字选择性分数（DSC）定位偏差神经元并通过剪枝 0.01% 的神经元修正 1.36-3.49% 的错误预测。

**[Better Estimation of the Kullback-Leibler Divergence Between Language Models](model_compression/better_estimation_of_the_kullback--leibler_divergence_between_language_models.md)**

:   提出 KL 散度的 Rao-Blackwell 化 Monte Carlo 估计器——在每个位置对下一个 token 的分布求精确 KL（而非只用采样的 token），理论证明无偏且方差严格不超过标准 MC 估计器，零额外计算开销，在 RLHF 情感控制任务中使训练更稳定、模型更频繁出现在 Pareto 前沿（78%）。

**[Beyond Accuracy: Dissecting Mathematical Reasoning for LLMs Under Reinforcement Learning](model_compression/beyond_accuracy_dissecting_mathematical_reasoning_for_llms_u.md)**

:   提出 SPARKLE 三轴分析框架（计划执行、知识整合、子问题分解）细粒度剖析 RL 如何改变 LLM 推理行为，发现 RL 主要增强了知识整合能力和计划灵活性而非计划执行能力，并提出 SparkleRL-PSS 多阶段 RL 训练 pipeline 通过 partial step scaffolding 有效利用难题数据。

**[Beyond Higher Rank: Token-wise Input-Output Projections for Efficient Low-Rank Adaptation](model_compression/beyond_higher_rank_token-wise_input-output_projections_for_efficient_low-rank_ad.md)**

:   TopLoRA 从输入-输出投影角度分析 LoRA 的表达能力，发现所有 token 共享同一投影矩阵是关键瓶颈，提出通过可学习的 token 级对角矩阵 $\Sigma_X$ 动态调整 LoRA 权重（$\Delta W_X = B\Sigma_X A$），在不增加秩的前提下实现细粒度适配，跨任务一致优于 LoRA 2-3%。

**[Beyond Random: Automatic Inner-Loop Optimization in Dataset Distillation](model_compression/beyond_random_automatic_inner-loop_optimization_in_dataset_distillation.md)**

:   提出 AT-BPTT（自适应截断 BPTT），将 DNN 训练分为早/中/晚三阶段并自适应调整截断策略和窗口大小，在 CIFAR-10/100/Tiny-ImageNet/ImageNet-1K 上平均提升 3-17%，同时实现 3.9× 加速和 63% 内存节省。

**[Bézier Splatting for Fast and Differentiable Vector Graphics Rendering](model_compression/bezier_splatting_for_fast_and_differentiable_vector_graphics_rendering.md)**

:   Bézier Splatting 将 Gaussian Splatting 框架与 Bézier 曲线结合，沿曲线均匀采样 2D Gaussian 点，通过 α-blending 渲染实现可微矢量图形，前向 30× / 反向 150× 加速（相比 DiffVG），同时保持或超越 LIVE 等方法的图像质量。

**[Binary Quadratic Quantization: Beyond First-Order Quantization for Real-Valued Matrix Compression](model_compression/binary_quadratic_quantization_beyond_first-order_quantization_for_real-valued_ma.md)**

:   BQQ 提出二次二值量化——用二值矩阵的乘积（而非线性组合）表示权重矩阵，突破传统一阶量化的表达能力限制，通过扩展 AMFD（退火均场下降）到 PUBO 问题求解混合整数优化，在 2-bit 无数据 ViT 量化上实现从 10.83% 到 58.25% 的准确率飞跃。

**[BioBench: A Blueprint to Move Beyond ImageNet for Scientific ML Benchmarks](model_compression/biobench_a_blueprint_to_move_beyond_imagenet_for_scientific_ml_benchmarks.md)**

:   提出 BioBench——一个统一 9 个生态视觉任务、4 个分类界、6 种图像模态、310 万张图像的基准，证明 ImageNet top-1 准确率仅解释 34% 的生态任务方差，在 >75% 精度的前沿模型中 30% 的排名是错误的。

**[CAS-Spec: Cascade Adaptive Self-Speculative Decoding for On-the-Fly Lossless Inference Acceleration of LLMs](model_compression/casspec_cascade_adaptive_selfspeculative_decoding_for_onthef.md)**

:   CAS-Spec 通过 Dynamically Switchable Inference Acceleration (DSIA) 策略（如不同程度的 layer sparsity）从目标模型自身构建多级 draft 模型层级，配合 Dynamic Tree Cascade (DyTC) 算法基于在线 acceptance rate 和延迟预测自适应路由 draft 模型和分配 draft 长度，在完全 training-free 的条件下实现 1.1×-2.3× 的无损推理加速，DyTC 比 cascade 和 tree baseline 分别提升 47% 和 48%。

**[ChunkKV: Semantic-Preserving KV Cache Compression for Efficient Long-Context LLM Inference](model_compression/chunkkv_semanticpreserving_kv_cache_compression_for_efficien.md)**

:   ChunkKV 将 KV cache 压缩的基本单元从离散 token 提升为语义 chunk（连续 token 组），通过 chunk 级 attention score 聚合来选择保留哪些语义完整的片段，并利用 chunk 带来的高跨层索引相似性实现 layer-wise index reuse，在 10% 压缩率下比 SnapKV/PyramidKV 提升最高 8.7%，吞吐量提升 26.5%。

**[DisMo: Disentangled Motion Representations for Open-World Motion Transfer](model_compression/dismo_disentangled_motion_representations_for_openworld_moti.md)**

:   DisMo 通过双流架构（运动提取器 + 帧生成器）和图像空间重建目标，从原始视频中学习与外观、姿态、类别无关的抽象运动表征，实现跨类别/跨视角的开放世界运动迁移，并在零样本动作分类上大幅超越 V-JEPA 等视频表征模型。

**[Eyes Wide Open: Ego Proactive Video-LLM for Streaming Video](model_compression/eyes_wide_open_ego_proactive_videollm_for_streaming_video.md)**

:   定义"第一视角流式视频主动理解"新任务——给定ego-streaming视频，AI助手在恰当时机主动回答多样化、随事件演变的问题，同时保持感知与推理的同步。提出ESTP-Bench评估框架、ESTP-F1指标，以及含数据引擎、多阶段训练和主动动态压缩的完整技术pipeline（VideoLLM-EyeWO），在ESTP-Bench上比最强baseline MiniCPM-V高11.8%。

---

## 📐 优化/理论 { #optimization }

**[A Single-Loop First-Order Algorithm for Linearly Constrained Bilevel Optimization](optimization/a_single-loop_first-order_algorithm_for_linearly_constrained_bilevel_optimizatio.md)**

:   针对下层问题带耦合线性约束的双层优化问题，提出单循环一阶算法 SFLCB，通过罚函数 + 增广拉格朗日重构消除 Hessian 依赖，将迭代复杂度从 $O(\epsilon^{-3}\log(\epsilon^{-1}))$ 改进至 $O(\epsilon^{-3})$。

**[A Theoretical Study on Bridging Internal Probability and Self-Consistency for LLM Reasoning](optimization/a_theoretical_study_on_bridging_internal_probability_and_sel.md)**

:   提出首个针对基于采样的测试时缩放方法的理论框架，将推理误差分解为估计误差和模型误差，揭示了Self-Consistency收敛慢、Perplexity模型误差大的局限，并提出RPC方法融合两者优势，在7个基准上以50%的采样成本达到同等推理性能。

**[A Unified Approach to Submodular Maximization Under Noise](optimization/a_unified_approach_to_submodular_maximization_under_noise.md)**

:   本文提出一个统一的元算法框架，可以将任何满足"鲁棒性"条件的精确子模最大化算法作为黑盒，自动转换为在持久噪声值预言机下保持近似比的算法，首次覆盖了非单调子模函数的拟阵约束和无约束情形。

**[A Unified Stability Analysis of SAM vs SGD: Role of Data Coherence and Emergence of Simplicity Bias](optimization/a_unified_stability_analysis_of_sam_vs_sgd_role_of_data_cohe.md)**

:   通过线性稳定性分析框架，证明了"平坦极小值⇒好泛化"和"SGD偏好简单函数"是同一枚硬币的两面——数据一致性(coherence)同时控制着两者，且SAM通过更严格的稳定性条件进一步放大了简单性偏好。

**[Adaptive Algorithms with Sharp Convergence Rates for Stochastic Hierarchical Optimization](optimization/adaptive_algorithms_with_sharp_convergence_rates_for_stochas.md)**

:   首次为随机层次化优化（极小极大和双层优化）提供自适应且sharp的收敛保证，通过动量归一化技术和新型自适应参数选择，在无需事先知道噪声大小的情况下实现最优收敛率Õ(1/√T + √σ̄/T^{1/4})。

**[An Adaptive Algorithm for Bilevel Optimization on Riemannian Manifolds](optimization/an_adaptive_algorithm_for_bilevel_optimization_on_riemannian_manifolds.md)**

:   AdaRHD 是首个无需预知问题参数（强凸常数、Lipschitz 界、流形曲率）的黎曼双层优化自适应算法——通过逆累计梯度范数策略自适应选择步长，在三阶段框架中逐步求解下层问题/线性系统/上层更新，收敛速率 $O(1/\epsilon)$ 匹配非自适应方法，对初始步长选择鲁棒性远超 RHGD。

**[Asymptotically Stable Quaternionic Hopfield Structured Neural Network with Supervised Projection-based Manifold Learning](optimization/asymptotically_stable_quaternion-valued_hopfield-structured_neural_network_with_.md)**

:   提出四元数值监督学习 Hopfield 结构神经网络 (QSHNN)，通过周期性投影策略保持权重矩阵的四元数结构一致性，并基于 Lyapunov 理论证明了不动点的存在唯一性和渐近稳定性，轨迹曲率有界保证机器人路径规划的平滑性。

**[Auto-Compressing Networks](optimization/auto-compressing_networks.md)**

:   Auto-Compressing Networks（ACN）用长程前向连接（所有层输出直接汇聚到最终输出）替代短残差连接，使得梯度的 Direct Gradient 成分远强于 Forward Gradient，隐式地将信息压缩到早期层——ViT 仅需 6 层达到标准 12 层性能，BERT 节省 75% 层数，还额外获得噪声鲁棒性（+6.4%）和持续学习抗遗忘（-18%）。

**[AutoOpt: A Dataset and a Unified Framework for Automating Optimization Problem Solving](optimization/autoopt_a_dataset_and_a_unified_framework_for_automating_optimization_problem_so.md)**

:   AutoOpt 构建了首个优化问题图像到代码的端到端框架——11554 张优化公式图像（手写+印刷）的 AutoOpt-11k 数据集 + M1 混合编码器（ResNet+Swin→mBART）图像转 LaTeX（BLEU 96.70）+ M2 DeepSeek-Coder LaTeX 转 PYOMO + M3 双层分解求解器，框架级成功率 94.20%。

**[Better NTK Conditioning: A Free Lunch from ReLU Nonlinear Activation in Wide Neural Networks](optimization/better_ntk_conditioning_a_free_lunch_from_relu_nonlinear_activation_in_wide_neur.md)**

:   证明 ReLU 激活函数对宽神经网络有一个此前未被注意的"免费"益处：(a) 在模型梯度特征空间中产生更好的数据分离（相似输入的角度在梯度空间中被放大），(b) 由此导致 NTK 矩阵条件数严格减小（相比线性网络）。深度进一步放大此效应——在无限宽然后无限深的极限下，所有数据对在梯度空间中等角分离（~75.5°），NTK 条件数收敛到仅依赖数据量 $n$ 的固定值 $(n+4)/3$。

**[Brain-like Variational Inference](optimization/brain-like_variational_inference.md)**

:   提出 FOND 框架（Free energy Online Natural-gradient Dynamics），从自由能最小化的第一原理推导出脉冲神经网络推断动力学，并实现 iPVAE（迭代泊松 VAE），在重建-稀疏性权衡、生物合理性和 OOD 泛化上优于标准 VAE 和预测编码模型。

**[Contribution of Task-Irrelevant Stimuli to Drift of Neural Representations](optimization/contribution_of_task-irrelevant_stimuli_to_drift_of_neural_representations.md)**

:   理论证明在线学习中任务无关刺激的统计特性（方差和维度）是表示漂移的重要驱动因素，在 Oja 规则、Similarity Matching、自编码器和监督两层网络中均观察到漂移率 $D \propto \lambda_\perp^2 (n-m)$，且学习噪声诱导的漂移具有各向异性几何特征，与高斯突触噪声的各向同性漂移定性不同。

**[Do Neural Networks Need Gradient Descent to Generalize? A Theoretical Study](optimization/do_neural_networks_need_gradient_descent_to_generalize_a_theoretical_study.md)**

:   本文在矩阵分解（神经网络理论的经典测试平台）上证明了 Guess & Check（随机抽参数直到拟合训练集）的泛化能力随宽度增加而退化（首次证明存在 G&C 可证明劣于梯度下降的典范情况），但随深度增加而改善，揭示了宽度和深度对泛化的截然不同作用。

**[Effective Policy Learning for Multi-Agent Online Coordination Beyond Submodular Objectives](optimization/effective_policy_learning_for_multi-agent_online_coordination_beyond_submodular_.md)**

:   提出 MA-SPL 和 MA-MPL 两个多智能体在线协调算法，通过"基于策略的连续扩展"技术突破次模性限制，首次在次模和弱次模目标函数上均实现最优 $(1 - c/e)$ 近似比，支持时变目标和仅局部反馈的实际约束。

**[Emergence and Scaling Laws in SGD Learning of Shallow Neural Networks](optimization/emergence_and_scaling_laws_in_sgd_learning_of_shallow_neural_networks.md)**

:   本文对浅层神经网络在线 SGD 学习加法模型（多个单指标函数叠加）的过程进行了精确分析，证明了每个教师神经元的学习呈现尖锐相变（emergence），而大量相变曲线的叠加自然产生平滑的幂律 scaling law。

**[From Linear to Nonlinear: Provable Weak-to-Strong Generalization through Feature Learning](optimization/from_linear_to_nonlinear_provable_weak-to-strong_generalization_through_feature_.md)**

:   本文首次在非线性特征学习设定（线性 CNN → 两层 ReLU CNN）下严格分析了 weak-to-strong 泛化现象，揭示了数据匮乏和数据丰富两种机制下的不同行为：前者通过良性过拟合实现泛化（或因有害过拟合失败），后者通过早停的标签纠正实现泛化（但过训练会退化）。

**[Gradient Descent As Loss Landscape Navigation A Normative Framework For Deriving](optimization/gradient_descent_as_loss_landscape_navigation_a_normative_framework_for_deriving.md)**

:   提出统一框架将各种学习规则（momentum、Adam、自然梯度等）推导为损失景观上的最优导航策略，不同度量和目标自然导出不同的优化器。

**[Large Language Bayes](optimization/large_language_bayes.md)**

:   将 LLM 和概率编程语言（PPL/Stan）数学地"胶合"成联合分布 $p(z,x,m|t) = p(m|t)_{\text{LLM}} \cdot p(z,x|m)_{\text{PPL}}$，用户只需提供非形式化的问题描述和数据，系统自动从 LLM 采样候选形式模型、做贝叶斯推断、通过边际似然加权平均，无需用户编写概率模型。

**[Learning Orthogonal Multi-Index Models A Fine-Grained Information Exponent Analy](optimization/learning_orthogonal_multi-index_models_a_fine-grained_information_exponent_analy.md)**

:   证明正交多索引模型 $f_*(\mathbf{x}) = \sum_{k=1}^P \phi(\mathbf{v}_k^* \cdot \mathbf{x})$ 可通过两阶段在线 SGD 以 $\tilde{O}(dP^{L-1})$ 样本复杂度学习（$L$ 为链接函数最低高阶 Hermite 阶），远优于仅用最低阶信息的 $\tilde{O}(Pd^{L-1})$——关键在于先用 2 阶项恢复子空间，再用 $L$ 阶项恢复方向，联合利用不同阶的 Hermite 分量。

**[Memory-Augmented Potential Field Theory: A Framework for Adaptive Control in Non-Convex Domains](optimization/memory-augmented_potential_field_theory_a_framework_for_adaptive_control_in_non-.md)**

:   提出记忆增强势场理论（MAPFT），在随机最优控制中维护一个动态记忆模块来检测并编码状态空间的拓扑特征（局部最小值、低梯度区等），通过动态修改价值函数景观实现非凸环境下的自适应控制，在 Humanoid-v4 等任务上比最优 RL 方法（SAC）提升 27% 累积奖励，且局部最优逃逸率从 ~30% 提升到 ~72%。

**[MESS+: Dynamically Learned Inference-Time LLM Routing in Model Zoos with Service Level Guarantees](optimization/mess_dynamically_learned_inference-time_llm_routing_in_model_zoos_with_service_l.md)**

:   MESS+是首个成本最优的LLM路由框架，通过在线学习请求满足度预测和虚拟队列约束，动态选择模型同时保证SLA合规，相比现有方法实现平均2倍成本节省。

**[Online Two-Stage Submodular Maximization](optimization/online_two-stage_submodular_maximization.md)**

:   首次提出在线两阶段子模最大化（O2SSM）问题，针对加权阈值势函数（WTP）设计了 RAOCO 算法，通过分数松弛+随机管道舍入实现多项式时间运行下的次线性 $(1-1/e)^2$-regret 保证，同时改进了离线问题的近似比。

**[Optimistic Online-to-Batch Conversions for Accelerated Convergence and Universality](optimization/optimistic_online-to-batch_conversions_for_accelerated_convergence_and_universal.md)**

:   提出乐观在线到批量（O2B）转换框架，将乐观性从在线算法中释放到转换机制本身，使简单的在线梯度下降就能实现 $O(T^{-2})$ 加速收敛率，并首次通过 O2B 转换实现强凸光滑目标的最优收敛，同时达到对光滑性的通用性。

**[Understanding the Generalization of Stochastic Gradient Adam in Learning Neural Networks](optimization/understanding_the_generalization_of_stochastic_gradient_adam_in_learning_neural_.md)**

:   首次理论分析 mini-batch Adam 的泛化行为，证明大 batch Adam/AdamW 即使带 weight decay 也收敛到高测试误差的解，而小 batch 版本通过随机梯度的隐式正则化 + weight decay 的显式正则化可实现近零测试误差，且 Adam 的有效 weight decay 上界严格小于 AdamW。

**[Unveiling the Power of Multiple Gossip Steps: A Stability-Based Generalization Analysis in Decentralized Training](optimization/unveiling_the_power_of_multiple_gossip_steps_a_stability-based_generalization_an.md)**

:   本文首次从算法稳定性角度分析去中心化 SGD（DSGD）中多步 Gossip 通信（MGS）的泛化效果，证明 MGS 以指数速率减少优化误差从而收紧泛化界，但即使 Gossip 步数趋于无穷也无法完全弥合与中心化训练的泛化差距。

**[VERA: Variational Inference Framework for Jailbreaking Large Language Models](optimization/vera_variational_inference_framework_for_jailbreaking_large_language_models.md)**

:   将越狱建模为后验推断问题，通过变分推断训练小型攻击 LLM 生成多样化黑盒越狱提示，固定时间内比 GPTFuzzer/AutoDAN 多 5× 成功越狱，且提示多样性显著更高。

---

## 🏥 医学图像 { #medical_imaging }

**[3D-RAD: A Comprehensive 3D Radiology Med-VQA Dataset with Multi-Temporal Analysis and Diverse Diagnostic Tasks](medical_imaging/3drad_a_comprehensive_3d_radiology_medvqa_dataset_with_multi.md)**

:   提出 3D-RAD——首个大规模3D医学VQA基准，包含170K条CT影像问答数据，覆盖六类临床任务（含创新性的多时相诊断任务），并配套136K训练集，揭示了现有VLM在3D时序推理上的严重不足。

**[A Novel Approach to Classification of ECG Arrhythmia Types with Latent ODEs](medical_imaging/a_novel_approach_to_classification_of_ecg_arrhythmia_types_with_latent_odes.md)**

:   将 Latent ODE 编码器与梯度提升决策树结合，构建端到端 ECG 心律失常分类流水线，在 360Hz→45Hz 降采样下 AUC-ROC 仅从 0.984 降至 0.976，展示了对低采样率的鲁棒性。

**[A Unified Solution to Video Fusion: From Multi-Frame Learning to Benchmarking](medical_imaging/a_unified_solution_to_video_fusion_from_multi-frame_learning_to_benchmarking.md)**

:   提出首个统一视频融合框架 UniVF（基于多帧学习 + 光流特征 warping + 时序一致性损失），并构建首个覆盖四大融合任务（多曝光、多焦点、红外-可见光、医学）的视频融合基准 VF-Bench，在全部子任务上取得 SOTA。

**[A Variational Manifold Embedding Framework for Nonlinear Dimensionality Reduction](medical_imaging/a_variational_manifold_embedding_framework_for_nonlinear_dimensionality_reductio.md)**

:   提出一种变分流形嵌入框架，将降维问题形式化为最优嵌入映射的优化问题（最小化先验分布与数据分布pullback之间的KL散度），在理论上统一了PCA与非线性降维方法，并利用变分法（Euler-Lagrange方程）和Noether定理为最优嵌入提供了可解释性约束。

**[AANet: Virtual Screening under Structural Uncertainty via Alignment and Aggregation](medical_imaging/aanet_virtual_screening_under_structural_uncertainty_via_alignment_and_aggregati.md)**

:   针对现实药物发现中蛋白质 holo 结构不可用的问题，提出 AANet——通过三模态对比学习（配体-holo pocket-检测cavity）对齐表征并用交叉注意力聚合多个候选结合位点，在 apo/predicted 蛋白质结构上的盲筛性能远超 SOTA（DUD-E 上 EF1% 从 11.75 提升至 37.19）。

**[Active Target Discovery under Uninformative Prior: The Power of Permanent and Transient Memory](medical_imaging/active_target_discovery_under_uninformative_prior_the_power_of_permanent_and_tra.md)**

:   提出 EM-PTDM 框架，受神经科学双记忆系统启发，利用预训练扩散模型作为"永久记忆"并结合基于 Doob's h-transform 的轻量"瞬时记忆"模块，在**无领域先验数据**的条件下实现高效的主动目标发现，理论保证先验单调改进。

**[Amortized Active Generation of Pareto Sets](medical_imaging/amortized_active_generation_of_pareto_sets.md)**

:   提出 A-GPS 框架，通过学习 Pareto 集的条件生成模型实现在线离散黑箱多目标优化——用非支配类概率估计器（CPE）作为 PHVI 的隐式估计替代显式超体积计算，并通过偏好方向向量实现摊还式后验偏好条件化（无需重新训练），在合成基准和蛋白质设计任务上展示了优越的样本效率。

**[Atomic Diffusion Models for Small Molecule Structure Elucidation from NMR Spectra](medical_imaging/atomic_diffusion_models_for_small_molecule_structure_elucidation_from_nmr_spectr.md)**

:   提出 ChefNMR，首个基于 3D 原子扩散模型的端到端框架，仅从 1D NMR 光谱和化学式直接预测未知小分子（尤其是复杂天然产物）的分子结构，在合成和实验数据集上均达到 SOTA。

**[GraphFLA: Augmenting Biological Fitness Prediction Benchmarks with Landscape Features](medical_imaging/augmenting_biological_fitness_prediction_benchmarks_with_landscapes_features_fro.md)**

:   GraphFLA 是一个高效的适应度景观分析框架——计算 20 个生物学意义的景观特征（粗糙度/上位性/可导航性/中性），在 5300+ 真实景观（ProteinGym/RNAGym/CIS-BP）上揭示模型性能高度依赖景观拓扑，如 VenusREM 在高可导航性景观上优于 ProSST 但在高上位性景观上弱于后者，处理百万突变体仅需 20 秒（vs MAGELLAN 5 小时）。

**[Autoencoding Random Forests](medical_imaging/autoencoding_random_forests.md)**

:   RFAE 首次为随机森林构建了原则性的编码-解码框架——利用 RF 核的正定性和普适性进行扩散映射谱分解得到低维编码，通过 k-NN 回归在叶节点空间中解码回原始特征，在 20 个表格数据集上重建质量排名 1.80（大幅优于 TVAE 3.38、AE 3.27），并成功应用于 MNIST 重建和 scRNA-seq 批次效应去除。

**[BarcodeMamba+: Advancing State-Space Models for Fungal Biodiversity Research](medical_imaging/barcodemamba_advancing_state-space_models_for_fungal_biodiversity_research.md)**

:   BarcodeMamba+ 是用于真菌 DNA 条形码分类的基础模型——基于状态空间模型架构，采用预训练+微调范式利用部分标注数据，结合层次标签平滑、加权损失和多头输出增强真菌分类（93%样本种级未标注），在所有分类层级上超越现有方法。

**[CrossNovo: Bidirectional Representations Augmented Autoregressive Biological Sequence Generation](medical_imaging/bidirectional_representations_augmented_autoregressive_biological_sequence_gener.md)**

:   CrossNovo 融合自回归（AR）和非自回归（NAR）解码器，通过共享谱编码器 + 重要性退火 + 梯度阻断知识蒸馏，让 NAR 的双向全局理解增强 AR 的序列生成能力，在 9-Species 基准上氨基酸精度达 0.811（+2.6%）、肽段召回 0.654（+5.3%）。

**[Brain Harmony: A Multimodal Foundation Model Unifying Morphology and Function into 1D Tokens](medical_imaging/brain_harmony_a_multimodal_foundation_model_unifying_morphology_and_function_int.md)**

:   首个统一脑结构形态（T1 sMRI）与功能动态（fMRI）的多模态脑基础模型，通过几何谐波预对齐和时序自适应 Patch Embedding（TAPE）将高维神经影像压缩为紧凑的 1D token 表示，在神经发育/退行性疾病诊断和认知预测任务上全面超越先前方法。

**[DyG-Mamba: Continuous State Space Modeling on Dynamic Graphs](medical_imaging/dyg-mamba_continuous_state_space_modeling_on_dynamic_graphs.md)**

:   DyG-Mamba 将连续状态空间模型（SSM）引入动态图学习，设计时间跨度感知的连续 SSM——用 Ebbinghaus 遗忘曲线启发的指数衰减函数建模不规则时间间隔，配合谱范数约束的输入依赖参数实现 Lipschitz 鲁棒性，在 12 个动态图基准上平均排名 2.42（vs DyGFormer 2.92），且保持 $O(bdL)$ 线性复杂度。

**[FGBench: A Dataset and Benchmark for Molecular Property Reasoning at Functional Group-Level](medical_imaging/fgbench_a_dataset_and_benchmark_for_molecular_property_reasoning_at_functional_g.md)**

:   FGBench 构建了首个官能团级分子属性推理基准（625K QA 对，覆盖 245 个官能团），通过相似分子配对 + AccFG 标注 + 重建验证确保数据质量，揭示即使 o3-mini 在交互任务上也仅 69.3%，化学专用模型（ChemLLM）甚至仅 23.3%。

**[H-DDx: A Hierarchical Evaluation Framework for Differential Diagnosis](medical_imaging/h-ddx_a_hierarchical_evaluation_framework_for_differential_diagnosis.md)**

:   H-DDx 提出基于 ICD-10 分类层级的鉴别诊断评估框架——将预测和真实诊断扩展到祖先节点后计算层级 F1（HDF1），奖励"临床相关的近似正确"而非仅精确匹配，评估 22 个 LLM 后发现领域特化模型（MediPhi）在 HDF1 上从第 20 名升至第 2 名（Top-5 指标完全遮蔽其优势）。

**[LoMix: Learnable Weighted Multi-Scale Logits Mixing for Medical Image Segmentation](medical_imaging/lomix_learnable_weighted_multi-scale_logits_mixing_for_medical_image_segmentatio.md)**

:   LoMix 提出通过组合突变模块（CMM）生成多尺度 logits 的"突变体"——4 种融合算子（加法/乘法/拼接/注意力加权）× 所有子集组合——配合 NAS 风格的 Softplus 可学习权重自动平衡各 logits 的贡献，在 Synapse 8 器官分割上 DICE 从 80.9% 提升到 85.1%（+4.2%），5% 训练数据下提升 +9.23%。

**[Mol-Llama Towards General Understanding Of Molecules In Large Molecular Language](medical_imaging/mol-llama_towards_general_understanding_of_molecules_in_large_molecular_language.md)**

:   构建分子语言模型Mol-LLaMA，通过multi-modal encoder（2D图、3D构象、SMILES文本）实现对分子特征和性质的通用理解。

**[QoQ-Med: Building Multimodal Clinical Foundation Models with Domain-Aware GRPO Training](medical_imaging/qoq-med_building_multimodal_clinical_foundation_models_with_domain-aware_grpo_tr.md)**

:   QoQ-Med 构建了覆盖 9 个临床模态（1D ECG + 6 类 2D 影像 + 2 类 3D 扫描）的多模态临床基础模型，提出域感知相对策略优化（DRPO）——通过层级温度缩放（域间 × 域内 K-means 聚类）解决模态/难度不平衡问题，在 261 万指令调优对上训练后平均 F1 达 0.295（vs GRPO 0.193，+52.8%），8 个模态中 6 个最优。

**[SpecMER: Fast Protein Generation with K-mer Guided Speculative Decoding](medical_imaging/specmer_fast_protein_generation_with_k-mer_guided_speculative_decoding.md)**

:   SpecMER 将投机解码引入蛋白质序列生成，用 K-mer 引导的批量选择策略从 draft 模型的多个候选中选取最符合进化保守性的序列供 target 模型验证，在保持分布一致性的同时实现 24-32% 加速，且生成序列的 NLL 和 pLDDT 结构置信度显著优于无引导的 baseline。

**[STAMP: Spatial-Temporal Adapter with Multi-Head Pooling](medical_imaging/stamp_spatial-temporal_adapter_with_multi-head_pooling.md)**

:   STAMP 为时间序列基础模型（TSFM）设计了仅 750K 参数的轻量空间-时间适配器，通过三组位置编码（token/空间/时间）+ 交叉 GMLP 混合 + 多头注意力池化，使冻结的 TSFM（如 MOMENT 385M）在 8 个 EEG 数据集上与 29M 参数的 EEG 专用模型（CBraMod）竞争或超越，在 BCIC-IV-2a 上 Kappa 比 CBraMod 高 193%。

**[The Biased Oracle: Assessing LLMs' Understandability and Empathy in Medical Diagnoses](medical_imaging/the_biased_oracle_assessing_llms_understandability_and_empathy_in_medical_diagno.md)**

:   系统评估 GPT-4o 和 Claude-3.7 在医疗诊断沟通中的可读性和共情能力，发现两者均产生超标的阅读难度（9-13 年级 vs 推荐的 6-8 年级），情感共情随诊断类型和患者教育水平显著变化，且 LLM-as-Judge 存在严重自我偏见（GPT 对自身共情评分膨胀 ~0.3 分）。

---

## 📈 时间序列 { #time_series }

**[A Graph Neural Network Approach for Localized and High-Resolution Temperature Forecasting](time_series/a_graph_neural_network_approach_for_localized_and_high-resolution_temperature_fo.md)**

:   提出一种 GCN-GRU 混合框架用于社区尺度（2.5km）高分辨率温度预报（1-48小时），在加拿大西南安大略三个区域上验证，最大区域平均 MAE 1.93°C、48h MAE 2.93°C，探索了 ClimateBERT 语言模型嵌入作为标准化输入的方案，为数据稀缺的全球南方地区提供可迁移的轻量级预报框架。

**[Abstain Mask Retain Core: Time Series Prediction by Adaptive Masking Loss with Representation Consistency](time_series/abstain_mask_retain_core_time_series_prediction_by_adaptive.md)**

:   揭示了时间序列预测中"适当截断历史数据反而提升精度"的反直觉现象（冗余特征学习问题），基于信息瓶颈理论提出AMRC方法，通过自适应掩码损失和表征一致性约束来抑制冗余特征学习，作为模型无关的训练框架在多种架构上显著提升性能。

**[AERO: A Redirection-Based Optimization Framework Inspired by Judo for Robust Probabilistic Forecasting](time_series/aero_a_redirection-based_optimization_framework_inspired_by_judo_for_robust_prob.md)**

:   AERO 提出受柔道"借力重定向"启发的优化框架，通过梯度投影、能量守恒和干扰预测将对抗性扰动重定向为有利优化方向，在概率太阳能价格预测上展示更稳定的收敛。

**[AttentionPredictor: Temporal Patterns Matter for KV Cache Compression](time_series/attentionpredictor_temporal_patterns_matter_for_kv_cache_com.md)**

:   首个基于学习的 KV Cache 压缩方法，通过轻量级时空卷积模型预测下一 token 的注意力分数来动态识别关键 token，实现 13× KV cache 压缩和 5.6× cache offloading 加速，显著优于静态方法。

**[Benchmarking Probabilistic Time Series Forecasting Models on Neural Activity](time_series/benchmarking_probabilistic_time_series_forecasting_models_on_neural_activity.md)**

:   首次系统评测 12 个概率时间序列预测模型在小鼠皮层钙成像数据上的表现，发现 PatchTST 一致最优（信息性预测窗口达 1.5 秒），零样本基础模型（Chronos）完全失败但微调后竞争力强，揭示神经活动的内在可预测性上限约 1.5 秒。

**[Causal Masking on Spatial Data: An Information-Theoretic Case for Learning Spatial Datasets with Unimodal Language Models](time_series/causal_masking_on_spatial_data_an_information-theoretic_case_for_learning_spatia.md)**

:   证明在空间数据（国际象棋棋盘FEN状态）上直接应用因果掩蔽训练单模态LLM，其表现优于先将数据线性化为序列（PGN棋步）后再应用因果掩蔽——FEN+因果掩蔽的Llama 1.3B达到~2630 Elo，而PGN+因果仅~2130 Elo。

**[Channel Matters: Estimating Channel Influence for Multivariate Time Series](time_series/channel_matters_estimating_channel_influence_for_multivariate_time_series.md)**

:   提出 Channel-wise Influence (ChInf)——首个能量化多变量时间序列中不同通道对模型性能影响的影响函数方法，将 TracIn 从整体样本级分解到通道级，衍生出通道级异常检测和通道剪枝两个应用，在 5 个异常检测基准上排名第一。

**[Connecting the Dots: A ML Ready Dataset for Ionospheric Forecasting](time_series/connecting_the_dots_a_machine_learning_ready_dataset_for_ionospheric_forecasting.md)**

:   构建了首个ML-ready电离层预测数据集，整合SDO、太阳风、地磁指数和TEC观测等多源异构数据为统一的时间-空间结构，并基准测试了多种时空ML架构用于TEC预测。

**[Demandcast Global Hourly Electricity Demand Forecasting](time_series/demandcast_global_hourly_electricity_demand_forecasting.md)**

:   构建DemandCast——覆盖56个国家(2000-2025)的XGBoost全球小时电力需求预测框架，融合ERA5温度/GDP/人口等特征，归一化目标（年度分数）+时间分割评估，MAPE 9.2%。

**[EcoCast: Spatio-Temporal Model for Continual Biodiversity Forecasting](time_series/ecocast_a_spatio-temporal_model_for_continual_biodiversity_and_climate_risk_fore.md)**

:   提出EcoCast，基于Transformer的时空模型，整合Sentinel-2、ERA5和GBIF数据进行近期物种分布预测，配合EWC持续学习机制，在非洲鸟类分布预测上F1从0.31提升至0.65。

**[How Foundational Are Foundation Models For Time Series Forecasting](time_series/how_foundational_are_foundation_models_for_time_series_forecasting.md)**

:   系统评估三个时间序列基础模型（TimesFM、TimeGPT、TiReX）在合成和真实数据集上的表现，发现其零样本能力与预训练域强相关、在真实分布偏移数据上微调后的 TSFM 并不一致优于从头训练的小模型（如 SAMFormer），挑战了 TSFM 的"one-size-fits-all"假设。

**[Improving Time Series Forecasting via Instance-aware Post-hoc Revision (PIR)](time_series/improving_time_series_forecasting_via_instance-aware_post-hoc_revision.md)**

:   PIR 提出实例感知的事后修正框架——通过不确定性估计识别预测失败实例，用局部修正（协变量+外生变量 Transformer）和全局修正（检索相似训练实例加权平均）的残差组合，作为即插即用模块使 SparseTSF MSE 降低 25.87%，PatchTST 降低 8.99%。

**[RiverMamba: A State Space Model for Global River Discharge and Flood Forecasting](time_series/rivermamba_a_state_space_model_for_global_river_discharge_and_flood_forecasting.md)**

:   首个能在 0.05°（~5.5km）全球网格上做 7 天河流流量预报的深度学习模型——用空间填充曲线将 3D 时空点序列化后输入双向 Mamba block，结合 ECMWF HRES 气象预报，在 1.5-500 年重现期洪水检测上 F1 =0.459 超越 LSTM（0.358）和物理模型 GloFAS。

**[ScatterAD: Temporal-Topological Scattering Mechanism for Time Series Anomaly Detection](time_series/scatterad_temporal-topological_scattering_mechanism_for_time_series_anomaly_dete.md)**

:   提出"散射性"（scattering）作为异常检测的新归纳偏置——异常样本在高维表示空间中比正常样本分布更分散，通过双编码器（时间+拓扑）+ 超球面散射中心约束 + 对比融合学习时拓扑联合表示，在 6 个工业 IoT 数据集上 15/24 设置取得最佳。

**[Sempo Lightweight Foundation Models For Time Series Forecasting](time_series/sempo_lightweight_foundation_models_for_time_series_forecasting.md)**

:   提出 SEMPO，一个仅 6.5M 参数、在 83M 时间点上预训练的轻量级时间序列基础模型，通过能量感知谱分解（EASD）充分利用低能量频率信号 + Mixture-of-Prompts Transformer（MoPFormer）学习异构时序模式，在零样本和少样本预测中分别将误差降低 12% 和 22%，超越数亿参数的 SOTA 模型。

**[STaRFormer: Semi-Supervised Task-Informed Representation Learning via Dynamic Attention-Based Regional Masking](time_series/starformer_semi-supervised_task-informed_representation_learning_via_dynamic_att.md)**

:   提出 STaRFormer，通过动态注意力区域掩码（DAReM）识别任务关键区域并施加掩码扰动，配合批内+类内半监督对比学习将任务信息嵌入潜在表示，在 56 个数据集（含非平稳、不规则采样、分类/异常检测/回归）上全面超越 SOTA。

**[Strap Spatio-Temporal Pattern Retrieval For Out-Of-Distribution Generalization](time_series/strap_spatio-temporal_pattern_retrieval_for_out-of-distribution_generalization.md)**

:   提出 StRap，一个检索增强的时空模式学习框架，通过构建空间/时间/时空三维模式库并在推理时检索相似模式注入模型，在流式时空图 OOD 任务上平均提升 7.17%。

**[Synthetic Series-Symbol Data Generation For Time Series Foundation Models](time_series/synthetic_series-symbol_data_generation_for_time_series_foundation_models.md)**

:   提出 Series-Symbol (S²) 数据生成机制和 SymTime 基础模型，通过符号表达式与时序数据的双模态对比学习预训练，在纯合成数据上训练即可在 5 大时序分析任务上与真实数据预训练的基础模型竞争。

**[Syntsbench Rethinking Temporal Pattern Learning In Deep Learning Models For Time](time_series/syntsbench_rethinking_temporal_pattern_learning_in_deep_learning_models_for_time.md)**

:   提出 SynTSBench，一个基于合成数据的时序预测模型评估框架，通过可编程特征配置（趋势/周期/噪声/依赖/多变量）和理论最优基准，系统揭示当前深度学习模型在各类时序模式上的能力边界。

**[Time-IMM: A Dataset and Benchmark for Irregular Multimodal Multivariate Time Series](time_series/time-imm_a_dataset_and_benchmark_for_irregular_multimodal_multivariate_time_seri.md)**

:   构建 Time-IMM 数据集——首个按因果机制分类不规则性的多模态多变量时序 benchmark（9 种不规则类型分为触发/约束/伪影三大类，9 个数据集），配套 IMM-TSF 预测库支持异步多模态融合，实验表明显式建模多模态在不规则时序上平均降低 MSE 6.71%，最高达 38.38%。

**[Time-O1 Time-Series Forecasting Needs Transformed Label Alignment](time_series/time-o1_time-series_forecasting_needs_transformed_label_alignment.md)**

:   提出 Time-o1，一种为时序预测设计的变换增强学习目标：用 SVD 将标签序列变换为去相关且按重要性排序的成分，然后只对齐最重要的成分，从而同时消除标签自相关带来的偏差和减少多任务优化的复杂度，在多种预测模型上一致提升 SOTA 性能。

---

## ✂️ 语义分割 { #segmentation }

**[Alligat0R: Pre-Training through Covisibility Segmentation for Relative Camera Pose Regression](segmentation/alligat0r_pre-training_through_co-visibility_segmentation_for_relative_camera_po.md)**

:   用共视性分割（covisibility segmentation）替代 CroCo 的跨视图补全作为双目视觉预训练任务，对每个像素预测"共视/遮挡/视野外"三类标签，在低重叠场景下显著超越 CroCo，RUBIK 基准总体成功率 60.3% 排第一。

**[ARGenSeg: Image Segmentation with Autoregressive Image Generation Model](segmentation/argenseg_image_segmentation_with_autoregressive_image_generation_model.md)**

:   提出ARGenSeg——首个利用自回归图像生成范式实现图像分割的统一MLLM框架，让模型直接输出visual tokens并通过VQ-VAE解码为分割mask，无需额外分割头，搭配next-scale prediction并行生成策略实现4×加速，在RefCOCO/+/g上以更少训练数据超越SOTA。

**[Attention (as Discrete-Time Markov) Chains](segmentation/attention_as_discrete-time_markov_chains.md)**

:   将 softmax 归一化后的注意力矩阵重新解读为离散时间 Markov 链（DTMC）的转移概率矩阵，提出多跳注意力（Multi-Bounce）和 TokenRank（稳态分布，类似 PageRank）来捕获间接注意力路径和全局 token 重要性，在 ImageNet 分割上达 94.29% mAP，并增强 Self-Attention Guidance 的图像生成质量。

**[Broken Tokens: Your Language Model Can Secretly Handle Non-Canonical Tokenization](segmentation/broken_tokens_your_language_model_can_secretly_handle_non-canonical_tokenization.md)**

:   揭示 LLM 能秘密处理非标准分词（如将"Hello"拆为"He"+"llo"而非标准的"Hello"整词token）——即使输入的 token 序列与训练时不同，模型表现出惊人的鲁棒性，且这种能力来自嵌入空间中子词嵌入的线性组合近似整词嵌入的特性。

**[ConnectomeBench: Can LLMs Proofread the Connectome?](segmentation/connectomebench_can_llms_proofread_the_connectome.md)**

:   构建ConnectomeBench评估LLM在3D神经元网格理解上的能力——包括片段识别、分裂错误修正和合并错误检测三个任务，Claude 3.7在片段识别达82%和分裂修正达85%（远超baseline 20-50%），但合并检测仍落后于人类专家。

**[COS3D: Collaborative Open-Vocabulary 3D Segmentation](segmentation/cos3d_collaborative_open-vocabulary_3d_segmentation.md)**

:   提出COS3D协作式开放词汇3D分割框架，在3D Gaussian Splatting中同时维护instance field（学习清晰边界）和language field（学习语义），通过两阶段训练实现Ins2Lang映射，推理时Language→Instance prompt精化实现互补协作，在LeRF数据集上mIoU达50.76%，大幅超越Dr.Splat（43.58%）。

**[Fast and Fluent Diffusion Language Models via Convolutional Decoding and Rejective Fine-tuning](segmentation/fast_and_fluent_diffusion_language_models_via_convolutional_decoding_and_rejecti.md)**

:   解决扩散语言模型（DLM）的长解码窗口问题——通过卷积归一化（替代硬半自回归分块）和基于规则的拒绝微调（R2FT, DPO风格下采权高先验/重复token），在128步（而非512+步）下实现DLM的SOTA生成质量。

**[Fast Foreground-Aware Diffusion With Accelerated Sampling Trajectory For Segment](segmentation/fast_foreground-aware_diffusion_with_accelerated_sampling_trajectory_for_segment.md)**

:   提出 FAST，一个面向分割的工业异常合成框架，通过前景感知重建模块（FARM）和异常感知加速采样（AIAS）在仅 10 步去噪下生成高质量合成异常，在 MVTec-AD 上 mIoU 达 76.72%，超越所有先前方法。

**[FineRS: Fine-grained Reasoning and Segmentation of Small Objects with Reinforcement Learning](segmentation/finers_fine-grained_reasoning_and_segmentation_of_small_objects_with_reinforceme.md)**

:   提出FineRS两阶段MLLM强化学习框架（全局语义探索GSE→局部感知精化LPR），通过locate-informed retrospective reward耦合两阶段，在自建FineRS-4k UAV高分辨率数据集上实现超小目标的推理与分割，gIoU达55.1%（超Seg-Zero† 8.5%），同时支持VQA（MVQA 83.3%）。

**[HAODiff: Human-Aware One-Step Diffusion via Dual-Prompt Guidance](segmentation/haodiff_human-aware_one-step_diffusion_via_dual-prompt_guidance.md)**

:   提出HAODiff——人体感知的单步扩散复原模型，设计包含人体运动模糊(HMB)的退化流水线、三分支双提示引导(DPG)网络产生正/负prompt对，通过CFG指导SD2.1单步去噪，在合成和真实场景均大幅领先（FID 8.36 vs 14.41，HMB残留率0.09 vs 0.19）。

**[LangHOPS: Language Grounded Hierarchical Open-Vocabulary Part Segmentation](segmentation/langhops_language_grounded_hierarchical_open-vocabulary_part_segmentation.md)**

:   提出LangHOPS——首个基于MLLM的开放词汇物体-部件实例分割框架，在语言空间中建立object-part层次结构，利用MLLM的知识和推理能力进行多粒度概念链接，在PartImageNet上in-domain达56.9 AP（超SOTA 6.5），cross-dataset设置下超越5.7 AP。

**[OmniSegmentor: A Flexible Multi-Modal Learning Framework for Semantic Segmentation](segmentation/omnisegmentor_a_flexible_multi-modal_learning_framework_for_semantic_segmentatio.md)**

:   OmniSegmentor 构建了含 5 种视觉模态的大规模 ImageNeXt 数据集（1.2M 样本），提出随机选择补充模态与 RGB 对齐的高效预训练策略，首次实现灵活的多模态预训练-微调流水线，在 6 个多模态语义分割基准上刷新 SOTA。

**[Panoptic Captioning An Equivalence Bridge For Image And Text](segmentation/panoptic_captioning_an_equivalence_bridge_for_image_and_text.md)**

:   提出 Panoptic Captioning 新任务，追求图像的"最小文本等价"——生成包含所有实体、位置、属性、关系和全局状态的全面描述，13B 模型配合解耦学习即超越 78B 开源和 GPT-4o 等商业模型。

**[PARTONOMY: Large Multimodal Models with Part-Level Visual Understanding](segmentation/partonomy_large_multimodal_models_with_part-level_visual_understanding.md)**

:   提出 Partonomy 部件级分割 benchmark（862 部件标签/534 物体标签）和 Plum 模型（用 span 标记替代 [SEG] token + mask 反馈循环），发现 SOTA 分割 LMM 在部件理解上仅 5.9% gIoU，Plum 通过避免分布偏移和利用历史预测显著提升。

**[HCLFuse: Revisiting Generative Infrared and Visible Image Fusion Based on Human Cognitive Laws](segmentation/revisiting_generative_infrared_and_visible_image_fusion_based_on_human_cognitive.md)**

:   HCLFuse 基于信息瓶颈原理和最优传输理论进行模态对齐，设计变分瓶颈编码器（VBE）+ 物理引导条件扩散模型，融合热传导/结构保持/物理一致性三种约束到扩散过程中，在 MSRS 数据集上梯度指标 AG 提升 69.87%，空间频率 SF 提升 39.41%。

**[Robust Ego-Exo Correspondence with Long-Term Memory](segmentation/robust_ego-exo_correspondence_with_long-term_memory.md)**

:   提出LM-EEC，基于SAM 2的自中心-外中心(ego-exo)视频跨视角目标分割框架，通过Memory-View MoE自适应融合记忆特征与跨视角特征，配合双记忆库压缩策略保持长期信息，在EgoExo4D基准上大幅超越现有方法（Ego2Exo IoU 54.98 vs 38.26）。

**[Robust Egocentric Referring Video Object Segmentation Via Dual-Modal Causal Inte](segmentation/robust_egocentric_referring_video_object_segmentation_via_dual-modal_causal_inte.md)**

:   提出CERES框架，通过双模态因果干预解决自中心指代视频分割(Ego-RVOS)中的鲁棒性问题：对语言偏见用后门调整（消除目标-动作频率偏差），对视觉混淆用前门调整（以深度信息引导视觉中介变量聚合），在VISOR/VOST/VSCOS上达到SOTA。

**[Sam-R1 Leveraging Sam For Reward Feedback In Multimodal Segmentation Via Reinfor](segmentation/sam-r1_leveraging_sam_for_reward_feedback_in_multimodal_segmentation_via_reinfor.md)**

:   提出 SAM-R1，将 SAM 作为 RL 训练循环中的奖励提供者（而非仅下游模块），设计分层 IoU 分割精度奖励 + 推理/分割格式奖励，配合改进的非对称裁剪 GRPO 算法，仅用 3k 训练样本在 ReasonSeg 零样本基准上超越 Seg-Zero 等方法，证明了细粒度分割奖励对 MLLM 推理-分割对齐的有效性。

**[Sansa Unleashing The Hidden Semantics In Sam2 For Few-Shot Segmentation](segmentation/sansa_unleashing_the_hidden_semantics_in_sam2_for_few-shot_segmentation.md)**

:   发现 SAM2 虽然是类无关预训练，但其特征中已隐含编码了丰富的语义结构（只是与跟踪特征纠缠在一起），通过在 Image Encoder 最后两层加入轻量 AdaptFormer 即可将语义结构显式化，将 SAM2 的"对象跟踪"重新诠释为"语义跟踪"，在少样本分割基准上达到 SOTA 且比竞争方法快 3× 小 4-5×。

**[Towards Robust Pseudo-Label Learning In Semantic Segmentation An Encoding Perspe](segmentation/towards_robust_pseudo-label_learning_in_semantic_segmentation_an_encoding_perspe.md)**

:   提出 ECOCSeg，用纠错输出码（ECOC）替代 one-hot 编码来表示伪标签，将 N 类分类分解为 K 个二分类子任务，通过 bit 级去噪和可靠位挖掘生成更鲁棒的伪标签，在 UDA 和 SSL 分割任务上一致提升。

---

## ⚛️ 物理学 { #physics }

**[Astroco Self-Supervised Conformer-Style Transformers For Light-Curve Embeddings](physics/astroco_self-supervised_conformer-style_transformers_for_light-curve_embeddings.md)**

:   提出 AstroCo，一种将 Conformer（注意力 + 深度可分离卷积 + 门控）引入天文不规则光变曲线的自监督编码器，在 MACHO 数据集上重建误差比 Astromer v1/v2 降低 61-70%，少样本分类 macro-F1 提升约 7%。

**[Encoding and Understanding Astrophysical Information in Large Language Model-Generated Summaries](physics/encoding_and_understanding_astrophysical_information_in_large_language_model-gen.md)**

:   探究LLM嵌入是否能编码从X射线天文观测导出的物理量（硬度比、幂律指数、变异性），发现结构化prompt设计可将物理属性聚类纯度提升5.9%-57.5%，稀疏自编码器揭示LLM通过识别天体类型来推断未显式给出的物理参数。

**[Exoplanet Formation Inference Using Conditional Invertible Neural Networks](physics/exoplanet_formation_inference_using_conditional_invertible_neural_networks.md)**

:   用条件可逆神经网络（cINN）训练于15,777颗合成行星数据，从观测量（行星质量、轨道距离）快速推断行星形成参数（盘质量、湍流α、尘气比），实现比物理模型快~10⁶倍的概率性参数回溯，并证明多行星系统数据比单行星数据更鲁棒。

**[FAIR Universe HiggsML Uncertainty Dataset and Competition](physics/fair_universe_higgsml_uncertainty_dataset_and_competition.md)**

:   提供2.8亿模拟LHC碰撞事件的标准化数据集和竞赛平台，包含6种参数化系统偏差（探测器校准+背景成分）及不确定性感知评估指标，要求参赛者为Higgs信号强度μ估计鲁棒的68.27%置信区间，优胜方案通过无聚焦替代建模实现比传统方法窄20%的置信区间。

**[Feat Free Energy Estimators With Adaptive Transport](physics/feat_free_energy_estimators_with_adaptive_transport.md)**

:   提出 FEAT 框架，基于随机插值学习自适应传输，通过 escorted Jarzynski 等式和 Crooks 定理提供一致、最小方差的自由能差估计器，统一了平衡与非平衡方法。

**[From Simulations to Surveys: Domain Adaptation for Galaxy Observations](physics/from_simulations_to_surveys_domain_adaptation_for_galaxy_observations.md)**

:   构建从模拟星系（TNG50）到真实巡天观测（SDSS）的域适应 pipeline，通过特征级对齐（欧几里得距离 + 最优传输 + top-$k$ 软匹配损失）和可训练权重调度，将星系形态分类的目标域准确率从 46.8%（无适应）提升到 87.3%，Macro F1 从 0.298 提升到 0.626。

**[Knowledge is Overrated: A Zero-Knowledge ML and Cryptographic Hashing-Based Framework for Verifiable, Low Latency Inference at the LHC](physics/knowledge_is_overrated_a_zero-knowledge_machine_learning_and_cryptographic_hashi.md)**

:   提出PHAZE框架，利用密码学哈希（Rabin指纹）和零知识机器学习（zkML）实现LHC触发器级别的可验证早退出推理，理论延迟降至~152-253ns量级，同时内建异常检测能力。

**[Latent Representation Learning In Heavy-Ion Collisions With Maskpoint Transforme](physics/latent_representation_learning_in_heavy-ion_collisions_with_maskpoint_transforme.md)**

:   将掩码点云 Transformer 自编码器引入重离子碰撞分析，通过自监督预训练+监督微调的两阶段范式，学习到比 PointNet 更强的非线性潜在表征（PC1 分布重叠从 2.42% 降至 0.27%），为 QGP 性质研究提供了通用特征学习框架。

**[Multi-Modal Masked Autoencoders for Galaxy Evolution and Cosmology](physics/multi-modal_masked_autoencoders_for_learning_image-spectrum_associations_for_gal.md)**

:   将多模态掩码自编码器 (MMAE) 应用于星系图像和光谱的联合重建，构建了 134,533 个星系的图像+光谱数据集，实现了光谱和图像的交叉重建以及仅从图像的红移回归，$\sigma_{\text{NMAD}} = 0.016$ 优于 AstroCLIP。

**[Neural Deprojection of Galaxy Stellar Mass Profiles](physics/neural_deprojection_of_galaxy_stellar_mass_profiles.md)**

:   用神经网络解决星系恒星质量轮廓的去投影问题——从 2D 投影光度轮廓恢复 3D 空间质量分布，替代传统的 Abel 反演解析方法，在处理噪声数据和复杂轮廓时更鲁棒且更快。

**[POLARIS: A High-contrast Polarimetric Imaging Benchmark Dataset for Exoplanetary Disk Representation Learning](physics/polaris_a_high-contrast_polarimetric_imaging_benchmark_dataset_for_exoplanetary_.md)**

:   构建首个系外行星偏振成像ML基准数据集POLARIS（921张VLT/SPHERE/IRDIS偏振图像+75,910张预处理曝光），提出Diff-SimCLR框架（扩散模型增强对比学习），在参考星vs目标星分类任务上达到93%准确率，仅需<10%手动标注。

**[Quantum Doubly Stochastic Transformers](physics/quantum_doubly_stochastic_transformers.md)**

:   提出 QDSFormer，用变分量子电路（QontOT）替换 ViT 中的 softmax 生成双随机注意力矩阵，在多个小规模图像识别任务上超越标准 ViT 和 Sinkformer，并显著稳定训练。

**[Simulation-Based Inference for Neutrino Interaction Model Parameter Tuning](physics/simulation-based_inference_for_neutrino_interaction_model_parameter_tuning.md)**

:   首次将基于仿真的推断（SBI）应用于中微子相互作用模型参数调优，使用神经后验估计（NPE）从200K个GENIE模拟的58-bin直方图中学习4个物理参数的后验分布，在MicroBooNE Tune的mock数据上准确恢复了真实参数值。

**[The Pareto Frontier of Resilient Jet Tagging](physics/the_pareto_frontier_of_resilient_jet_tagging.md)**

:   系统评估LHC射流标记任务中多种架构（DNN/PFN/EFN/ParT）的AUC-鲁棒性权衡，揭示更复杂模型虽AUC更高但对蒙特卡洛模型依赖性更强，构建Pareto前沿并通过案例研究证明低鲁棒性分类器即使校准后仍在下游参数估计中产生偏差。

**[The Platonic Universe: Do Foundation Models See the Same Sky?](physics/the_platonic_universe_do_foundation_models_see_the_same_sky.md)**

:   在天文学场景下验证柏拉图表征假说（PRH）：使用JWST、HSC、Legacy Survey和DESI光谱数据，测量6种基础模型（ViT/ConvNeXt/DINOv2/IJEPA/AstroPT/Specformer）的表征对齐度，发现模态内和跨模态MKNN分数随模型规模一致增加（p=3.31×10⁻⁵），支持不同架构和模态向共享表征收敛的假说。

**[Toward Complete Merger Identification at Cosmic Noon with Deep Learning](physics/toward_complete_merger_identification_at_cosmic_noon_with_deep_learning.md)**

:   在 IllustrisTNG50 模拟生成的模拟 HST CANDELS 图像上训练 ResNet18，首次证明深度学习可以在高红移 $1<z<1.5$ 下成功识别包括小质量比合并（minor merger, $\mu \geq 1/10$）和低质量星系（$M_\star > 10^8 M_\odot$）在内的星系合并，总体准确率约 73%，并通过 Grad-CAM 和 UMAP 深入分析了模型行为。

**[Tropical Attention Neural Algorithmic Reasoning For Combinatorial Algorithms](physics/tropical_attention_neural_algorithmic_reasoning_for_combinatorial_algorithms.md)**

:   提出 Tropical Attention，将注意力机制提升到热带射影空间中进行分段线性推理，在组合算法的 OOD 泛化上大幅超越 softmax 基线，同时推理速度快 3-9 倍、参数少 ~20%。

**[Unsupervised Discovery Of High-Redshift Galaxy Populations With Variational Auto](physics/unsupervised_discovery_of_high-redshift_galaxy_populations_with_variational_auto.md)**

:   用变分自编码器(VAE)对 2743 条 JWST 高红移($z>4$)星系光谱进行无监督聚类，发现 12 个不同的天体物理类别，使已知的后星暴星系、Lyman-α 发射星系、极端发射线星系、Little Red Dots 等稀有种群数量翻倍。

**[Why Is Attention Sparse In Particle Transformer](physics/why_is_attention_sparse_in_particle_transformer.md)**

:   分析 Particle Transformer (ParT) 在jet tagging中出现的二值化稀疏attention现象：稀疏性来自attention机制本身而非物理启发的interaction矩阵，但两者对性能都不可或缺。

---

## 🕸️ 图学习 { #graph_learning }

**[BLISS: Bandit Layer Importance Sampling Strategy for Efficient Training of Graph Neural Networks](graph_learning/bliss_bandit_layer_importance_sampling_strategy_for_efficient_training_of_graph_.md)**

:   提出 BLISS，将 GNN 的层级邻居采样建模为多臂老虎机问题，用 EXP3 算法动态调整每条边的采样概率，根据邻居对节点表示的方差贡献作为奖励信号，在 GCN 和 GAT 上维持或超越全批次训练精度。

**[Deliberation on Priors: Trustworthy Reasoning of LLMs on Knowledge Graphs](graph_learning/deliberation_on_priors_trustworthy_reasoning_of_large_language_models_on_knowled.md)**

:   提出Deliberation over Priors（DP）框架，通过渐进式知识蒸馏（SFT+KTO偏好优化）提升关系路径生成的忠实度，并通过约束引导的内省-回溯机制保障可靠性，在ComplexWebQuestions上H@1提升13%且LLM调用次数减少77%。

**[Diagnosing and Addressing Pitfalls in KG-RAG Datasets: Toward More Reliable Benchmarking](graph_learning/diagnosing_and_addressing_pitfalls_in_kg-rag_datasets_toward_more_reliable_bench.md)**

:   系统审计16个KGQA数据集发现平均事实正确率仅57%（WebQSP 52%，MetaQA 20%），提出KGQAGen框架——通过LLM引导的子图扩展+SPARQL自动验证构建高质量多跳QA数据集KGQAGen-10k（96.3%准确率），揭示KG-RAG的主要瓶颈在检索而非推理。

**[Disentangling Hyperedges through the Lens of Category Theory](graph_learning/disentangling_hyperedges_through_the_lens_of_category_theory.md)**

:   首次用范畴论分析超边解耦，导出"因子表示一致性"自然性标准（聚合后解耦 vs 解耦后聚合应一致），提出Natural-HNN模型在癌症分型任务上比最佳baseline提升4.7%（BRCA F1从75.7%到80.4%），并100%正确捕获功能通路上下文。

**[Geometric Imbalance In Semi-Supervised Node Classification](graph_learning/geometric_imbalance_in_semi-supervised_node_classification.md)**

:   在黎曼流形上形式化半监督节点分类中的几何不平衡问题，提出伪标签对齐框架，在9个基准上一致超越基线，特别在严重类别不平衡时效果显著。

**[GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation](graph_learning/gfm-rag_graph_foundation_model_for_retrieval_augmented_generation.md)**

:   提出首个图基础模型驱动的检索增强生成框架 GFM-RAG，通过 query-dependent GNN 在知识图谱上进行单步多跳推理，仅 8M 参数即可在未见数据集上零样本泛化，在多跳QA检索任务上大幅超越 SOTA。

**[Graph Persistence goes Spectral](graph_learning/graph_persistence_goes_spectral.md)**

:   提出 SpectRe——将图拉普拉斯谱信息融入持续同调（PH）图的新拓扑描述符，证明其表达力严格强于 PH 和谱信息单独使用，建立了局部稳定性理论，在合成和真实数据集上提升 GNN 的图分类能力。

**[Graphfaas Serverless Gnn Inference For Burst-Resilient Real-Time Intrusion Detec](graph_learning/graphfaas_serverless_gnn_inference_for_burst-resilient_real-time_intrusion_detec.md)**

:   提出GraphFaaS，基于Serverless的GNN推理架构用于突发负载下的实时入侵检测：时间局部性图构建+频率过滤+贪心图分区实现延迟降低85%、变异系数降低64%同时保持准确率。

**[GraphTOP: Graph Topology-Oriented Prompting for Graph Neural Networks](graph_learning/graphtop_graph_topology-oriented_prompting_for_graph_neural_networks.md)**

:   提出首个图拓扑导向的 prompting 框架 GraphTOP，通过将 topology-oriented prompting 建模为边重连问题并用 Gumbel-Softmax 松弛到连续空间，在 5 个数据集 4 种预训练策略下超越 6 个基线方法。

**[Making Classic GNNs Strong Baselines Across Varying Homophily: A Smoothness-Generalization Perspective](graph_learning/making_classic_gnns_strong_baselines_across_varying_homophily_a_smoothness-gener.md)**

:   从理论上揭示了 GNN 消息传递中平滑性（smoothness）与泛化性（generalization）之间的两难困境，提出 IGNN 框架通过三个简约设计原则（分离邻域变换、感知聚合、邻域关系学习）缓解该困境，在 30 个基线中表现最优且具备跨同质/异质图的通用性。

**[Moscat: Mixture of Scope Experts at Test for Generalizing Deeper GNNs](graph_learning/mixture_of_scope_experts_at_test_generalizing_deeper_graph_neural_networks_with_.md)**

:   通过 PAC-Bayes 界证明 GNN 深度变化导致不同同质性子群间的泛化偏好漂移，提出 Moscat——后处理注意力门控模型，在测试时自适应组合不同深度的独立训练 GNN 专家。

**[OCN: Effectively Utilizing Higher-Order Common Neighbors for Better Link Prediction](graph_learning/ocn_effectively_utilizing_higher-order_common_neighbors_for_better_link_predicti.md)**

:   揭示高阶公共邻居（CN）在链接预测中的冗余和过平滑问题，提出正交化（Gram-Schmidt 去除阶间线性相关）+ 归一化（除以路径数，广义资源分配启发式）解决方案，在 7 个数据集上平均提升 HR@100 7.7%，DDI 数据集上提升 13.3%。

**[Over-squashing in Spatiotemporal Graph Neural Networks](graph_learning/over-squashing_in_spatiotemporal_graph_neural_networks.md)**

:   首次形式化时空图神经网络(STGNN)中的 over-squashing 问题，揭示了因果卷积中反直觉的"时间远处偏好"现象（最早时间步对最终表示影响最大），并证明 time-and-space 和 time-then-space 架构在信息瓶颈上等价，为使用计算高效的 TTS 架构提供理论支持。

**[PKD: Preference-driven Knowledge Distillation for Few-shot Node Classification](graph_learning/preference-driven_knowledge_distillation_for_few-shot_node_classification.md)**

:   PKD 框架协同 LLM 和多 GNN 教师做文本属性图少样本节点分类——GNN 偏好节点选择器（GNS）用 KL 散度不确定性选择需要 LLM 标注的节点，节点偏好 GNN 选择器（NGS）用 RL 为每个节点匹配最优 GNN 教师，在 9 个数据集上一致 SOTA（Cornell 87% vs 基线 59-82%）。

**[Relieving the Over-Aggregating Effect in Graph Transformers](graph_learning/relieving_the_over-aggregating_effect_in_graph_transformers.md)**

:   发现了 Graph Transformer 中的 over-aggregating 现象——大量节点以近均匀注意力分数被聚合导致关键信息被稀释，提出 Wideformer 通过分割聚合+引导注意力来缓解，作为即插即用模块在 13 个数据集上一致提升骨干模型性能。

**[Uniedit A Unified Knowledge Editing Benchmark For Large Language Models](graph_learning/uniedit_a_unified_knowledge_editing_benchmark_for_large_language_models.md)**

:   构建UniEdit——基于25个开放域知识的统一LLM知识编辑基准，提出邻域多跳链采样(NMCS)算法评估编辑的波纹效应。

**[Unifying Text Semantics and Graph Structures for Temporal Text-attributed Graphs with LLMs](graph_learning/unifying_text_semantics_and_graph_structures_for_temporal_text-attributed_graphs.md)**

:   提出 Cross 框架——用 LLM 在策略采样的时间点上动态总结节点邻域的语义演变（Temporal Reasoning Chain），然后通过语义-结构协同编码器双向融合文本语义和图结构时序信息，在时序链接预测上平均 MRR 提升 24.7%，工业数据（微信）上 AUC 提升 3.7%。

**[What Expressivity Theory Misses: Message Passing Complexity for GNNs](graph_learning/what_expressivity_theory_misses_message_passing_complexity_for_gnns.md)**

:   批判 GNN 的二值表达力理论无法解释实际性能差异，提出 MPC——基于概率性 lossyWL 的连续、任务特定复杂度度量，与准确率的 Spearman 相关性达 -1（传统 WLC 恒为零），成功解释了 GCN+虚拟节点为何在长程任务上优于更高表达力的高阶模型。

---

## 🛡️ AI 安全 { #ai_safety }

**[A Set of Generalized Components to Achieve Effective Poison-only Clean-label Backdoor Attacks with Collaborative Sample Selection and Triggers](ai_safety/a_set_of_generalized_components_to_achieve_effective_poison-only_clean-label_bac.md)**

:   提出一组通用化组件（Component A/B/C），通过充分挖掘样本选择与触发器之间的双向协作关系，同时提升 Poison-only Clean-label 后门攻击的攻击成功率（ASR）和隐蔽性，并在多种攻击类型上展现了良好的泛化能力。

**[Adaptive LoRA Experts Allocation and Selection for Federated Fine-Tuning](ai_safety/adaptive_lora_experts_allocation_and_selection_for_federated_fine-tuning.md)**

:   提出 FedLEASE——解决联邦 LoRA 微调中两个关键问题：(1) 用 LoRA B 矩阵相似度聚类自动确定最优专家数量和分配，(2) 用扩展路由空间（$2M-1$ 维）实现自适应 top-M 专家选择（每个客户端自动决定用几个专家），在 GLUE 上比最强基线平均提升 5.53%。

**[Adversarial Paraphrasing: A Universal Attack for Humanizing AI-Generated Text](ai_safety/adversarial_paraphrasing_a_universal_attack_for_humanizing_ai-generated_text.md)**

:   提出 Adversarial Paraphrasing——一种无需训练的通用攻击框架，在逐 token 改写时利用 AI 文本检测器的反馈信号选择"最像人写"的 token，使改写后的 AI 文本在 8 种检测器上平均 T@1%F 下降 87.88%，且具有跨检测器的强迁移性。

**[AI Should Sense Better, Not Just Scale Bigger: Adaptive Sensing as a Paradigm Shift](ai_safety/ai_should_sense_better_not_just_scale_bigger_adaptive_sensin.md)**

:   提出"自适应感知"作为AI发展的范式级转变——受生物感觉系统启发，主张在传感器层面动态调整输入参数（如曝光、增益、多模态配置），而非仅靠扩大模型规模来应对分布偏移，实证表明5M参数的EfficientNet-B0通过自适应感知可超越632M参数的OpenCLIP-H。

**[ALMGuard: Safety Shortcuts and Where to Find Them as Guardrails for Audio-Language Models](ai_safety/almguard_safety_shortcuts_and_where_to_find_them_as_guardrails_for_audio-languag.md)**

:   首个针对音频语言模型（ALM）越狱攻击的防御框架——发现对齐过的 ALM 存在可被激活的潜在安全快捷路径（safety shortcuts），通过 Mel 梯度稀疏掩码（M-GSM）定位关键频率段，施加快捷路径激活扰动（SAP），将平均攻击成功率从 41.6% 降至 4.6%，同时几乎不影响正常任务性能。

**[Benchmarking is Broken — Don't Let AI be its Own Judge](ai_safety/benchmarking_is_broken_--_dont_let_ai_be_its_own_judge.md)**

:   系统性批评当前 AI 基准评估的根本缺陷——数据污染（MMLU 45%+ 重叠）、选择性报告、缺乏监考——并提出 PeerBench 方案：借鉴高考/GRE 的监考范式，用滚动更新的保密题库 + 同行评审质量控制 + 声誉加权评分 + 加密承诺机制构建下一代 AI 评估基础设施。

**[Beyond Last-Click: An Optimal Mechanism for Ad Attribution](ai_safety/beyond_last-click_an_optimal_mechanism_for_ad_attribution.md)**

:   从博弈论角度分析广告归因中 Last-Click 机制的策略操纵漏洞——平台可以通过篡改时间戳获取不公正的归因信用，提出 Peer-Validated Mechanism（PVM）——每个平台的信用仅取决于其他平台的报告（类比同行评审），理论证明 PVM 是占优策略激励兼容（DSIC）且在同质设置下最优，准确率从 34% 提升到 75%（2 平台）。

**[Bias in the Picture: Benchmarking VLMs with Social-Cue News Images and LLM-as-Judge](ai_safety/bias_in_the_picture_benchmarking_vlms_with_social-cue_news_images_and_llm-as-jud.md)**

:   构建 1,343 个新闻图片-问答对的偏见评估基准，标注年龄/性别/种族/职业等人口统计属性，用 GPT-4o 作为评判员（LLM-as-judge）评估 15 个 VLM 在开放式问答中的偏见表现，发现高忠实度不等于低偏见，且性别和职业偏见尤为严重。

**[Bits Leaked per Query: Information-Theoretic Bounds on Adversarial Attacks Against LLMs](ai_safety/bits_leaked_per_query_information-theoretic_bounds_on_adversarial_attacks_agains.md)**

:   将 LLM 对抗攻击建模为信息通道问题——定义每次查询的"泄漏比特数" $I(Z;T)$ 为攻击目标属性 $T$ 与可观测信号 $Z$ 的互信息，证明攻击达到误差 $\varepsilon$ 所需最少查询数为 $\log(1/\varepsilon)/I(Z;T)$，在 7 个 LLM 上验证：暴露 answer tokens 需 ~1000 次查询，加 logits 降到 ~100 次，加思维链降到 ~几十次，为透明性-安全性权衡提供首个原则性标尺。

**[Enhancing CLIP Robustness via Cross-Modality Alignment](ai_safety/enhancing_clip_robustness_via_crossmodality_alignment.md)**

:   提出COLA——一个training-free的框架，通过将对抗扰动后的图像特征投影到文本特征张成的子空间来消除非语义噪声，再用最优传输(OT)在分布层面细粒度对齐图文特征，在14个零样本分类基准上平均提升6.7%的对抗鲁棒准确率，同时维持干净样本性能。

**[Fair Representation Learning With Controllable High Confidence Guarantees Via Ad](ai_safety/fair_representation_learning_with_controllable_high_confidence_guarantees_via_ad.md)**

:   提出 FRG（Fair Representation learning with high-confidence Guarantees），首个允许用户指定公平性阈值 $\varepsilon$ 和置信水平 $1-\delta$ 的公平表征学习框架：通过 VAE 候选选择 + 对抗推断最大化协方差 + Student's t-检验构造高置信上界，保证对**任意**下游模型和任务，$\Delta_{DP} \leq \varepsilon$ 以至少 $1-\delta$ 概率成立。

**[Flux Efficient Descriptor-Driven Clustered Federated Learning Under Arbitrary Di](ai_safety/flux_efficient_descriptor-driven_clustered_federated_learning_under_arbitrary_di.md)**

:   提出Flux——基于描述符驱动聚类的联邦学习框架，通过提取隐私保护的客户端数据描述符（分布统计量的矩近似）和无监督密度聚类，自动处理四种分布偏移（特征/标签/P(Y|X)/P(X|Y)），在CheXpert医疗数据集上测试时精度比最佳基线高14.6pp。

**[LLM Strategic Reasoning: Agentic Study through Behavioral Game Theory](ai_safety/llm_strategic_reasoning_agentic_study_through_behavioral_gam.md)**

:   论文不再把大模型战略推理简单等同于“是否接近纳什均衡”，而是基于 behavioral game theory 构建评测框架，区分真实推理能力与上下文因素，系统测评 22 个 LLM 的互动决策行为，发现模型规模并不决定战略水平，CoT 提升也并非普遍有效，同时暴露出显著的人口属性偏置。

**[Matchings Under Biased and Correlated Evaluations](ai_safety/matchings_under_biased_and_correlated_evaluations.md)**

:   在两机构稳定匹配模型中引入评估相关性参数 $\gamma$（机构间评分的对齐程度），分析偏差 $\beta$ 和相关性 $\gamma$ 如何联合影响弱势群体的代表性比率，证明即使轻微的相关性损失也可导致代表性急剧下降，并提出公平性干预策略的 Pareto 前沿。

**[OmniFC: Rethinking Federated Clustering via Lossless and Secure Distance Reconstruction](ai_safety/omnifc_rethinking_federated_clustering_via_lossless_and_secure_distance_reconstr.md)**

:   提出 OmniFC，一个模型无关的联邦聚类框架：通过 Lagrange 编码计算在有限域上精确重建全局成对距离矩阵，任意集中式聚类方法（K-Means/谱聚类/DBSCAN/层次聚类等）可直接在其上运行，仅需一轮通信，天然抵抗 Non-IID，在 7 个数据集上全面超越 k-FED/MUFC/FedSC 等专用方法。

**[SAEMark: Steering Personalized Multilingual LLM Watermarks with Sparse Autoencoders](ai_safety/saemark_steering_personalized_multilingual_llm_watermarks_with_sparse_autoencode.md)**

:   利用稀疏自编码器（SAE）作为特征提取器，通过特征引导的拒绝采样实现多语言、多比特 LLM 水印，无需白盒访问或 logit 操纵，英文/中文 Acc 99%+。

**[The Unseen Threat Residual Knowledge In Machine Unlearning Under Perturbed Sampl](ai_safety/the_unseen_threat_residual_knowledge_in_machine_unlearning_under_perturbed_sampl.md)**

:   发现机器遗忘的关键安全漏洞：即使遗忘后的模型在统计意义上与重训练模型不可区分，对遗忘样本施加微小对抗扰动后，遗忘模型仍能正确识别而重训练模型则失败——揭示了"残余知识"这一新型隐私风险。提出 RURK 微调策略，通过惩罚对扰动遗忘样本的正确预测来消除残余知识，在 CIFAR-10 和 ImageNet-100 上有效抑制 11 种遗忘方法的残余知识。

---

## 🤖 机器人/具身智能 { #robotics }

**[A Snapshot of Influence: A Local Data Attribution Framework for Online Reinforcement Learning](robotics/a_snapshot_of_influence_a_local_data_attribution_framework_f.md)**

:   首次将数据归因（data attribution）引入在线强化学习，提出局部归因框架量化每条训练记录对策略更新的贡献，并基于此设计了迭代影响力过滤算法（IIF），在经典RL基准和LLM的RLHF上均显著提升了样本效率和最终性能。

**[Adaptive Frontier Exploration on Graphs with Applications to Network-Based Disease Testing](robotics/adaptive_frontier_exploration_on_graphs_with_applications_to_network-based_disea.md)**

:   提出 Adaptive Frontier Exploration on Graphs (AFEG) 问题框架，设计基于 Gittins index 的策略，在图是森林时可证明最优，在实际性传播疾病检测网络上仅测试一半人口即可检出几乎全部 HIV 感染者，大幅超越贪心和 DQN 等基线。

**[AutoToM: Scaling Model-based Mental Inference via Automated Agent Modeling](robotics/autotom_scaling_model-based_mental_inference_via_automated_agent_modeling.md)**

:   AutoToM 实现完全自动化的基于模型的心智理论推理——自动提出 agent 模型（贝叶斯网络结构）并进行贝叶斯逆规划，通过推理不确定性迭代调整模型（添加心智变量/扩展时间步），在5个 ToM benchmark 上超越 SOTA LLM 和推理模型，且产生类人的置信度估计。

**[Beyond Parallelism: Synergistic Computational Graph Effects in Multi-Head Attention](robotics/beyond_parallelism_synergistic_computational_graph_effects_in_multi-head_attenti.md)**

:   将多头注意力重新建模为共享汇节点的多个前馈 DAG 系统，理论证明多头可通过跨头路径实现协同效应——降低混合时间(mixing time)并放大 minimax 保真度(fidelity)，在序列操作任务上实验验证了该效应。

**[Bridging Embodiment Gaps: Deploying Vision-Language-Action Models on Soft Robots](robotics/bridging_embodiment_gaps_deploying_vision-language-action_models_on_soft_robots.md)**

:   首次在柔性连续体机械臂上部署 VLA 模型（OpenVLA-OFT 和 π₀），发现开箱即用的策略因构型不匹配完全失败，但通过针对性微调可弥合刚性-柔性的 embodiment gap，使柔性机器人在操作任务上达到与刚性 UR5 相当的成功率——证明 VLA + 柔性机器人可实现安全的人机交互。

**[Can Agents Fix Agent Issues](robotics/can_agents_fix_agent_issues.md)**

:   AgentIssue-Bench(50个bug任务)评估SE代理解决LLM代理bug的能力，仅0.67%-4.67%解决率。

**[CogVLA: Cognition-Aligned Vision-Language-Action Model via Instruction-Driven Routing & Sparsification](robotics/cogvla_cognition-aligned_vision-language-action_model_via_instruction-driven_rou.md)**

:   提出 CogVLA——模仿人类多模态认知的三阶段 VLA 架构：(1) EFA-Routing 将视觉 token 压缩至 25%；(2) LFP-Routing 裁剪 50% 的 LLM 无关 token；(3) V-L-A 耦合注意力保持语义一致性——在 LIBERO 上达 97.4% 成功率，训练成本降 2.5×，推理延迟降 2.8×。

**[EfficientNav: Towards On-Device Object-Goal Navigation with Navigation Map Caching and Retrieval](robotics/efficientnav_towards_on-device_object-goal_navigation_with_navigation_map_cachin.md)**

:   通过离散内存缓存（KV cache分组独立计算+选择性加载）、注意力驱动聚类（LLM浅层attention指导分组）和语义感知检索（CLIP+背包问题适配不同内存预算），首次在Jetson Orin上用LLaMA-3.2-11b实现零样本ObjNav，比GPT-4基线提升11.1% SR且实时延迟降低6.7×。

**[EgoThinker: Unveiling Egocentric Reasoning with Spatio-Temporal CoT](robotics/egothinker_unveiling_egocentric_reasoning_with_spatiotempora.md)**

:   针对第一人称视频推理中“主体不可见、意图隐含、交互细粒度”的挑战，EgoThinker 提出时空 CoT 监督与两阶段训练（SFT + RFT），并构建 EgoRe-5M 大规模 egocentric QA 数据，显著提升 MLLM 在自我中心视频推理与时空定位任务上的表现。

**[Generalizable Domain Adaptation for Sim-and-Real Policy Co-Training](robotics/generalizable_domain_adaptation_for_sim-and-real_policy_co-training.md)**

:   提出基于不平衡最优运输（UOT）的模拟-真实策略联合训练框架，通过对观察-动作联合分布进行对齐（而非仅对齐观察边际分布），结合时间对齐采样策略处理数据不平衡，在机器人操纵任务上实现30%的OOD泛化提升。

**[LabUtopia: High-Fidelity Simulation and Hierarchical Benchmark for Scientific Embodied Agents](robotics/labutopia_high-fidelity_simulation_and_hierarchical_benchmark_for_scientific_emb.md)**

:   提出 LabUtopia——面向科学实验室的高保真仿真与层级基准套件，包含支持化学反应建模的 LabSim 仿真器、可程序化生成实验室场景的 LabScene、以及从原子操作到长程移动操纵的五级 LabBench 基准，揭示现有模仿学习方法在长程实验流程和物体泛化方面的显著瓶颈。

**[Manipulating Feature Visualizations With Gradient Slingshots](robotics/manipulating_feature_visualizations_with_gradient_slingshots.md)**

:   提出梯度弹弓攻击，通过利用分布外梯度轨迹操纵神经网络特征可视化结果，无需修改模型参数，揭示特征可视化作为解释性工具的脆弱性。

**[MindForge: Empowering Embodied Agents with Theory of Mind for Lifelong Cultural Learning](robotics/mindforge_empowering_embodied_agents_with_theory_of_mind_for_lifelong_cultural_l.md)**

:   MindForge 为 LLM 驱动的具身智能体引入显式的心智理论（ToM）表征、自然语言通信和多组件记忆系统，使开源 LLM 智能体通过与专家协作对话（无需梯度更新）大幅提升任务完成率，在 Minecraft 中比 Voyager 多获得 3× 科技树里程碑和 2.3× 独特物品。

**[MineAnyBuild: Benchmarking Spatial Planning for Open-world AI Agents](robotics/mineanybuild_benchmarking_spatial_planning_for_openworld_ai.md)**

:   基于 Minecraft 构建空间规划基准 MineAnyBuild，要求 AI Agent 根据多模态指令生成可执行的建筑蓝图矩阵，包含 4000 个任务和 500+ 建筑/装饰资产，从空间理解、空间推理、创造力和空间常识四个维度系统评估 MLLM 的空间规划能力，揭示即便 GPT-4o 整体得分仅 41.02/100，开源模型更差。

**[MIP against Agent: Malicious Image Patches Hijacking Multimodal OS Agents](robotics/mip_against_agent_malicious_image_patches_hijacking_multimod.md)**

:   揭示针对多模态OS Agent的新型攻击向量——Malicious Image Patches (MIPs)：在屏幕截图中嵌入人类不可察觉的对抗性扰动图像块，当OS Agent截屏时自动触发恶意行为（如数据泄露、内存溢出），且可跨用户指令、屏幕布局和屏幕解析器泛化，甚至具备"计算机蠕虫"般的自传播潜力。

**[mmWalk: Towards Multi-modal Multi-view Walking Assistance](robotics/mmwalk_towards_multi-modal_multi-view_walking_assistance.md)**

:   mmWalk 构建了首个面向视障人群步行辅助的多模态多视角数据集（CARLA 仿真器生成 62K 帧/559K 全景图 + 69K VQA 对），基准测试发现 SOTA VLM 在风险评估和导航地标识别等安全关键任务上表现不足（最优仅 55.21%），微调后在真实数据集上泛化提升 16.7%。

**[SegMASt3R: Geometry Grounded Segment Matching](robotics/segmast3r_geometry_grounded_segment_matching.md)**

:   SegMASt3R 在预训练 MASt3R 3D 基础模型上添加轻量分割特征头和可微 Sinkhorn 匹配层，利用 3D 几何先验实现极端视角变化（达 180°）下的鲁棒语义段匹配，AUPRC 在 135-180° 基线上达 83.6%（vs SAM2 的 17%）。

---

## 🎵 音频/语音 { #audio_speech }

**[A Controllable Examination for Long-Context Language Models](audio_speech/a_controllable_examination_for_longcontext_language_models.md)**

:   提出LongBioBench，通过生成虚构传记作为可控的needle和haystack，构建满足"无缝上下文、可控设置、可靠评估"三大原则的长上下文LLM评估框架，测试18个模型后揭示当前LCLM在检索能力尚可的情况下推理和可信性仍有显著短板。

**[A TRIANGLE Enables Multimodal Alignment Beyond Cosine Similarity](audio_speech/a_triangle_enables_multimodal_alignment_beyond_cosine_simila.md)**

:   TRIANGLE提出用三模态嵌入向量端点构成的三角形面积作为相似度度量，替代传统的两两余弦相似度，实现视频-音频-文本的联合对齐，在视频检索任务上比VAST提升最高9个R@1点。

**[Accelerate Creation of Product Claims Using Generative AI](audio_speech/accelerate_creation_of_product_claims_using_generative_ai.md)**

:   开发 Claim Advisor 平台，利用 LLM 的 in-context learning 和 LoRA 微调加速消费品产品宣称的搜索、生成、优化和排序，通过模仿 MaxDiff 研究方法论让微调的 Phi-3 14B 模型在宣称排序上超越 GPT-4o（仅用 1 个示例 vs GPT 的 100 个示例），三轮迭代后 100% 的生成宣称达到"高吸引力"级别。

**[AdaptDel: Adaptable Deletion Rate Randomized Smoothing for Certified Robustness](audio_speech/adaptdel_adaptable_deletion_rate_randomized_smoothing_for_ce.md)**

:   提出AdaptDel方法，将随机平滑(randomized smoothing)中的固定删除率扩展为**自适应删除率**，根据输入长度等属性动态调整删除概率，在编辑距离攻击下实现认证鲁棒性的巨大提升（认证区域基数提升最高30个数量级）。

**[Associative Syntax and Maximal Repetitions Reveal Context-Dependent Complexity in Animal Vocalizations](audio_speech/associative_syntax_and_maximal_repetitions_reveal_context-dependent_complexity_i.md)**

:   提出基于"关联句法"和"最大重复"的信息论框架分析动物发声序列的结构复杂度，发现动物发声（如鲸鱼歌声）展现出上下文依赖的复杂句法结构，超越了简单的马尔可夫假设。

**[AudSemThinker: Enhancing Audio-Language Models through Reasoning over Semantics of Sound](audio_speech/audsemthinker_enhancing_audio-language_models_through_reasoning_over_semantics_o.md)**

:   AudSemThinker 为音频语言模型引入结构化语义推理框架——定义 9 类声音语义描述符（谁/什么/如何/何时/何地等），在 Qwen2.5-Omni-7B 上通过 SFT + GRPO（含可验证奖励和长度约束）训练产生 \<think\>\<semantic_elements\>\<answer\> 三阶段输出，MMAU 基准达 66.70%（超越 Audio-Reasoner 61.71% 和 Qwen2.5-Omni 65.60%）。

**[Benchmarking Egocentric Multimodal Goal Inference for Assistive Wearable Agents](audio_speech/benchmarking_egocentric_multimodal_goal_inference_for_assist.md)**

:   Meta 提出 WAGIBench，一个针对可穿戴辅助智能体的多模态目标推断基准，包含 348 名参与者的 3,477 条第一视角录制（29小时），涵盖视觉/音频/数字/纵向四种模态，人类准确率 93% vs 最佳 VLM 84%（MCQ），生成式评估中模型仅 55% 时间产生相关目标，揭示了当前 VLM 在实际可穿戴场景中的显著差距。

**[BNMusic: Blending Environmental Noises into Personalized Music](audio_speech/bnmusic_blending_environmental_noises_into_personalized_music.md)**

:   提出 BNMusic，一个两阶段框架将环境噪声融合到个性化生成音乐中：第一阶段通过 mel-spectrogram 的 outpainting + inpainting 生成与噪声节奏对齐的音乐，第二阶段利用听觉掩蔽理论自适应放大音乐信号以降低噪声感知，无需额外训练，在 EPIC-SOUNDS 和 ESC-50 上显著优于 baseline。

**[Can LLMs Outshine Conventional Recommenders? A Comparative Evaluation](audio_speech/can_llms_outshine_conventional_recommenders_a_comparative_evaluation.md)**

:   提出RecBench全面评估框架，在5个数据集上对比17个LLM和10个传统DLRM，发现LLM推荐器准确率提升5-170%但推理速度慢10-100×，传统DLRM+LLM特征组合以20×更快速度达到LLM~95%的性能，揭示了LLM-as-RS的实际部署不可行性。

**[Embedding Alignment In Code Generation For Audio](audio_speech/embedding_alignment_in_code_generation_for_audio.md)**

:   通过双MLP+对比学习将代码嵌入和音频嵌入映射到公共空间，使LLM生成的音频代码能隐式捕捉音乐相似性（CKA从0.090提升到0.590）。

**[Eurospeech A Multilingual Speech Corpus](audio_speech/eurospeech_a_multilingual_speech_corpus.md)**

:   构建EuroSpeech多语言语音数据集：从欧洲议会录音中自动化提取61K小时、覆盖22种语言的高质量语音数据，ASR WER降低41.8%。

**[Generating Physically Sound Designs From Text And A Set Of Physical Constraints](audio_speech/generating_physically_sound_designs_from_text_and_a_set_of_physical_constraints.md)**

:   联合优化视觉目标（CLIP文本对齐）和物理目标（可微分FEM结构约束），生成满足工程要求且包含文本指定特征的结构设计。

**[Instance-Specific Test-Time Training for Speech Editing in the Wild](audio_speech/instance-specific_test-time_training_for_speech_editing_in_the_wild.md)**

:   提出面向野外语音编辑的实例特定测试时训练方法：在推理前利用未编辑区域的真实声学特征做直接监督、编辑区域通过时长约束和音素预测辅助损失做间接监督，对模型进行实例级自适应微调，有效缓解编辑边界的带宽不连续问题，并支持通过 mask 长度调整精确控制语速，在野外 benchmark 上主客观评估均超越现有系统。

**[Multi-head Temporal Latent Attention](audio_speech/multi-head_temporal_latent_attention.md)**

:   MTLA 在 MLA 低秩潜在维度压缩基础上，用超网络动态融合时序相邻的 KV 向量，实现 KV 缓存在特征维度和时序维度的双重压缩，配合 stride-aware 因果 mask 保证训练-推理一致性，在语音翻译等任务上达到 4.29× 加速和 6.58× 内存降低，质量持平甚至略优于标准 MHA。

**[Perceptually Aligning Representations of Music via Noise-Augmented Autoencoders](audio_speech/perceptually_aligning_representations_of_music_via_noise-augmented_autoencoders.md)**

:   证明在自编码器训练中对潜变量加噪（noise-augmented latent training）配合感知损失，能使编码空间形成"感知层次结构"——感知最显著的音乐特征（如音高）编码在最粗粒度的潜在结构中，而次要特征（如音色细节）编码在细粒度结构中。这种对齐改善了潜在扩散解码下的音乐惊奇感估计和 EEG 脑响应预测。

**[Seeing Sound, Hearing Sight: Uncovering Modality Bias and Conflict of AI Models in Sound Localization](audio_speech/seeing_sound_hearing_sight_uncovering_modality_bias_and_conflict_of_ai_models_in.md)**

:   系统性地揭示了AI声源定位(SSL)模型存在严重视觉偏见——在视听冲突时降到随机水平，提出神经科学启发的EchoPin模型（HRTF滤波+耳蜗图+立体声），在AudioCOCO数据集上大幅超越现有方法并展现出类人的水平>垂直定位精度偏差。

---

## 🔗 因果推理 { #causal_inference }

**[A Principle of Targeted Intervention for Multi-Agent Reinforcement Learning](causal_inference/a_principle_of_targeted_intervention_for_multi-agent_reinforcement_learning.md)**

:   提出基于多智能体影响图（MAIDs）的**目标干预范式（Targeted Intervention）**，通过仅对单个目标智能体施加**预策略干预（Pre-Strategy Intervention, PSI）**，引导整个多智能体系统收敛到满足额外期望结果的优选Nash均衡，无需对所有智能体进行全局干预。

**[An Analysis of Causal Effect Estimation Using Outcome Invariant Data Augmentation](causal_inference/an_analysis_of_causal_effect_estimation_using_outcome_invariant_data_augmentatio.md)**

:   分析"结果不变数据增强"在因果效应估计中的作用——当增强操作不改变结果变量的条件分布时，可以在不引入偏差的条件下有效减少选择偏差，且在特定条件下可证明提升估计精度。

**[Bi-Level Decision-Focused Causal Learning for Large-Scale Marketing Optimization](causal_inference/bi-level_decision-focused_causal_learning_for_large-scale_marketing_optimization.md)**

:   提出 Bi-DFCL，通过双层优化框架联合利用观测数据和 RCT 实验数据来训练营销资源分配模型：上层用 RCT 数据的无偏决策损失端到端训练 Bridge Network 来动态纠正下层在观测数据上的偏差，同时设计了基于原始问题的可微代理决策损失（PPL/PIFD）和隐式微分算法，解决了传统两阶段方法的预测-决策不一致和偏差-方差困境。已在美团大规模在线部署。

**[Causality-Induced Positional Encoding for Transformer-Based Representation Learning of Non-Sequential Features](causal_inference/causality-induced_positional_encoding_for_transformer-based_representation_learn.md)**

:   CAPE 通过从表格数据中学习特征间的因果DAG结构，将其嵌入双曲空间生成因果感知的旋转位置编码（RoPE），使 Transformer 能处理非序列但因果相关的特征数据，在多组学数据的下游任务上显著提升性能。

**[Characterization and Learning of Causal Graphs from Hard Interventions](causal_inference/characterization_and_learning_of_causal_graphs_from_hard_interventions.md)**

:   首次系统分析硬干预（hard interventions）与软干预在因果发现中的差异，证明硬干预通过非局部打破d-分离关系提供更强的图区分能力，提出广义do-演算和基于孪生增强MAG的因果发现算法，实验表明硬干预将Markov等价类缩小37-57%。

**[Conformal Prediction for Causal Effects of Continuous Treatments](causal_inference/conformal_prediction_for_causal_effects_of_continuous_treatments.md)**

:   首次为连续处理（如药物剂量）的因果效应开发共形预测（CP）区间，通过倾向性散度标准化处理干预诱导的分布偏移，在已知/未知倾向性两种场景下提供有限样本 $1-\alpha$ 覆盖保证，在MIMIC-III临床数据上验证了实用性。

**[Cyclic Counterfactuals under Shift–Scale Interventions](causal_inference/cyclic_counterfactuals_under_shift-scale_interventions.md)**

:   为含有反馈循环的循环结构因果模型(cyclic SCM)建立了移位-缩放(shift-scale)干预下的反事实推断理论框架，证明了全局收缩条件下唯一可解性、干预复合封闭性，以及反事实泛函的sub-Gaussian集中不等式。

**[Demystifying Spectral Feature Learning For Instrumental Variable Regression](causal_inference/demystifying_spectral_feature_learning_for_instrumental_variable_regression.md)**

:   推导了谱特征学习在工具变量(IV)回归中的泛化界，根据谱对齐和特征值衰减率将性能分为"好/坏/丑"三类，并提出数据驱动的诊断方法。

**[Do-Pfn In-Context Learning For Causal Effect Estimation](causal_inference/do-pfn_in-context_learning_for_causal_effect_estimation.md)**

:   提出 Do-PFN，将 Prior-data Fitted Networks (PFN) 扩展到因果效应估计，在大量合成 SCM 数据上预训练 Transformer 进行 in-context 因果推理，仅需观测数据即可预测干预分布（CID）和 CATE，无需因果图知识或不混杂假设，在合成和半合成实验中表现出色。

**[Dynamic Causal Discovery In Alzheimers Disease Through Latent Pseudotime Modelli](causal_inference/dynamic_causal_discovery_in_alzheimers_disease_through_latent_pseudotime_modelli.md)**

:   将 BN-LTE（贝叶斯网络+潜在时间嵌入）应用于 ADNI 真实 AD 数据，推断随疾病伪时间演变的动态因果图，伪时间预测诊断 AUC 0.82 远超年龄 0.59，并揭示了新型生物标志物 NfL/GFAP 与传统 AD 标志物之间的动态因果关系。

**[Few-Shot Knowledge Distillation Of Llms With Counterfactual Explanations](causal_inference/few-shot_knowledge_distillation_of_llms_with_counterfactual_explanations.md)**

:   提出 CoD（Counterfactual-explanation-infused Distillation），通过系统性地将反事实解释（CFE）注入少样本训练集——CFE 位于 teacher 决策边界附近，作为"boundary pegs"将 student 的决策面钉在 teacher 附近——在 6 个数据集上用仅 8-512 样本的超低数据量显著超越标准蒸馏方法。

**[From Black-box to Causal-box: Towards Building More Interpretable Models](causal_inference/from_black-box_to_causal-box_towards_building_more_interpretable_models.md)**

:   提出"因果可解释性"（causal interpretability）的形式化定义，证明黑盒模型和概念瓶颈模型均不满足该性质，给出完整的图判据确定哪些模型架构能一致地回答反事实问题，揭示了因果可解释性与预测精度之间的根本性权衡。

**[It's Hard to Be Normal: The Impact of Noise on Structure-agnostic Estimation](causal_inference/its_hard_to_be_normal_the_impact_of_noise_on_structure-agnostic_estimation.md)**

:   证明 Double Machine Learning (DML) 在高斯处理噪声下是极小极大最优的（$O(\epsilon^2 + n^{-1/2})$），但在非高斯噪声下变得次优；提出 Agnostic Cumulant-based Estimation (ACE) 利用高阶累积量达到 $r$ 阶不敏感性 $O(\epsilon^r + n^{-1/2})$。

**[Practical do-Shapley Explanations with Estimand-Agnostic Causal Inference](causal_inference/practical_do-shapley_explanations_with_estimand-agnostic_causal_inference.md)**

:   提出 Estimand-Agnostic（EA）方法和 Frontier-Reducibility Algorithm（FRA）来高效计算因果 Shapley 值（do-SV），通过训练单个 SCM 学习观测分布即可回答任意可辨识的因果查询，并通过联盟约减将计算量降低约 90%。

**[Revealing Multimodal Causality With Large Language Models](causal_inference/revealing_multimodal_causality_with_large_language_models.md)**

:   提出 MLLM-CD，首个面向多模态非结构化数据因果发现的框架，包含三个模块：(1) 对比因子发现利用 MLLM 从模态内/间交互中识别因果变量；(2) 统计因果结构发现；(3) 迭代多模态反事实推理利用 MLLM 的世界知识生成反事实样本来消除结构歧义。在合成和真实数据集上显著优于现有方法。

---

## 🖼️ 图像恢复 { #image_restoration }

**[Adaptive Discretization for Consistency Models](image_restoration/adaptive_discretization_for_consistency_models.md)**

:   提出ADCM框架，将一致性模型(CM)的离散化步长选择形式化为约束优化问题，通过Gauss-Newton方法得到解析解，在局部一致性（可训练性）和全局一致性（稳定性）之间自适应平衡，以仅4%的额外计算开销实现显著的训练效率提升和FID改善。

**[Audio Super-Resolution With Latent Bridge Models](image_restoration/audio_super-resolution_with_latent_bridge_models.md)**

:   提出 AudioLBM，在波形隐空间中用桥模型实现 LR-to-HR latent-to-latent 音频超分，配合频率感知训练和级联设计，LSD 平均改善 21.5%，首次实现 any-to-192kHz 音频超分。

**[DenoiseRotator: Enhance Pruning Robustness for LLMs via Importance Concentration](image_restoration/denoiserotator_enhance_pruning_robustness_for_llms_via_importance_concentration.md)**

:   提出DenoiseRotator，在剪枝前插入可学习正交矩阵并通过熵最小化将参数重要性集中到子集上，使LLaMA3-70B在2:4稀疏下困惑度差距缩小58%（8.1→3.4），可与任何现有剪枝方法即插即用组合。

**[DynaGuide: Steering Diffusion Policies with Active Dynamic Guidance](image_restoration/dynaguide_steering_diffusion_polices_with_active_dynamic_guidance.md)**

:   提出DynaGuide，通过外部潜在动力学模型在DinoV2嵌入空间中预测未来视觉观测，利用分类器引导机制引导预训练扩散策略朝向目标条件动作，无需修改策略权重，在CALVIN上成功率70%、真实机器人80%。

**[Enhancing Infrared Vision: Progressive Prompt Fusion Network and Benchmark](image_restoration/enhancing_infrared_vision_progressive_prompt_fusion_network_and_benchmark.md)**

:   针对热红外(TIR)图像中低对比度、模糊、噪声等多种退化耦合的问题，提出基于双提示融合的渐进式网络PPFN和选择性渐进训练策略SPT，并构建首个大规模多场景TIR基准数据集HM-TIR，在复合退化场景下PSNR提升8.76%。

**[FIPER: Factorized Features for Robust Image Super-Resolution and Compression](image_restoration/fiper_factorized_features_for_robust_image_super-resolution_and_compression.md)**

:   提出统一的基-系数分解框架（学习非均匀基+空间变化系数+坐标变换），在超分辨率上PSNR相对提升204.4%，在图像压缩上BD-rate降低9.35%，通过多频调制同时建模高/低频内容。

**[GC4NC: A Benchmark Framework for Graph Condensation on Node Classification with New Insights](image_restoration/gc4nc_a_benchmark_framework_for_graph_condensation_on_node_classification_with_n.md)**

:   提出 GC4NC——首个系统化的图凝缩（Graph Condensation）评估基准框架，跨 8 个维度（性能/效率/隐私保护/去噪/NAS有效性/可迁移性等）统一评估多种图凝缩方法，发现轨迹匹配方法最优、无结构方法效率最高，并在 1000x 压缩下图凝缩显著优于图像凝缩。

**[Improving Diffusion-based Inverse Algorithms under Few-Step Constraint via Learnable Linear Extrapolation](image_restoration/improving_diffusion-based_inverse_algorithms_under_few-step_constraint_via_learn.md)**

:   提出 Learnable Linear Extrapolation (LLE)——用可学习的线性组合系数将当前和历史 clean data estimate 组合，以增强任何符合 Sampler-Corrector-Noiser 范式的扩散逆问题算法在少步（3-5步）下的表现，仅需 50 个样本、几分钟训练，跨 9+ 算法 × 5 个任务一致提升。

**[Learning Cocoercive Conservative Denoisers via Helmholtz Decomposition for Poisson Inverse Problems](image_restoration/learning_cocoercive_conservative_denoisers_via_helmholtz_decomposition_for_poiss.md)**

:   提出共循环保守(CoCo)去噪器概念，通过广义Helmholtz分解设计新的训练策略——Hamiltonian正则化促进保守性 + 谱正则化促进共循环性——使去噪器成为隐式弱凸先验的近端算子，从而在Poisson逆问题（光子受限去卷积、低剂量CT等）中实现有收敛保证且性能优越的PnP方法。

**[Mro Enhancing Reasoning In Diffusion Language Models Via Multi-Reward Optimizati](image_restoration/mro_enhancing_reasoning_in_diffusion_language_models_via_multi-reward_optimizati.md)**

:   MRO通过多奖励优化捕获扩散语言模型内/间序列token相关性，加速DLM推理同时保持性能。

**[Rethinking Circuit Completeness in Language Models: AND, OR, and ADDER Gates](image_restoration/rethinking_circuit_completeness_in_language_models_and_or_and_adder_gates.md)**

:   系统引入AND、OR、ADDER三种逻辑门来分解语言模型电路，揭示电路不完整性主要源于OR门的遗漏，提出结合noising和denoising干预的框架来完整恢复三种逻辑门，同时保证忠实度和完整性。

**[Rethinking Nighttime Image Deraining Via Learnable Color Space Transformation](image_restoration/rethinking_nighttime_image_deraining_via_learnable_color_space_transformation.md)**

:   提出CST-Net用于夜间图像去雨：基于夜间雨在Y通道（亮度）上比RGB更显著的观察，设计可学习颜色空间转换器(CSC)在YCbCr空间去雨，配合隐式光照引导模块(IIG)和新构建的光照感知合成数据集HQ-NightRain，在多个基准上达到SOTA。

**[Spiking Meets Attention Efficient Remote Sensing Image Super-Resolution With Att](image_restoration/spiking_meets_attention_efficient_remote_sensing_image_super-resolution_with_att.md)**

:   提出 SpikeSR，首个基于注意力脉冲神经网络(SNN)的遥感图像超分辨率框架，通过脉冲注意力块(SAB)结合混合维度注意力(HDA)和可变形相似度注意力(DSA)，在 AID/DOTA/DIOR 上达到 SOTA 性能同时保持高计算效率。

**[The Effect Of Optimal Self-Distillation In Noisy Gaussian Mixture Model](image_restoration/the_effect_of_optimal_self-distillation_in_noisy_gaussian_mixture_model.md)**

:   用统计物理replica方法分析噪声高斯混合模型上的自蒸馏，证明硬伪标签的去噪是性能提升主因，CIFAR-10实验验证。

**[Video Killed the Energy Budget: Characterizing the Latency and Power Regimes of Open T2V Models](image_restoration/video_killed_the_energy_budget_characterizing_the_latency_and_power_regimes_of_o.md)**

:   对开源T2V模型进行系统性延迟与能耗分析：建立了基于FLOP的compute-bound理论模型，验证了WAN2.1-T2V的二次空间/时间缩放和线性去噪步数缩放规律，并横向对比7个T2V模型发现能耗差异达3000倍（AnimateDiff 0.14Wh vs WAN2.1-14B 415Wh）。

---

## 🧑 人体理解 { #human_understanding }

**[A Differential and Pointwise Control Approach to Reinforcement Learning](human_understanding/a_differential_and_pointwise_control_approach_to_reinforceme.md)**

:   将RL问题通过连续时间控制的微分对偶形式重新表述，利用哈密顿结构嵌入物理先验，提出dfPO算法实现逐点策略优化，在科学计算任务（曲面建模、网格控制、分子动力学）上以更少样本超越12个RL基线。

**[A Practical Guide for Incorporating Symmetry in Diffusion Policy](human_understanding/a_practical_guide_for_incorporating_symmetry_in_diffusion_policy.md)**

:   本文提出了一套将对称性融入扩散策略的实用指南——通过不变性表征（相对轨迹动作 + 手眼感知）、等变视觉编码器和 Frame Averaging 三种简单方法，在 MimicGen 12 个任务上达到了接近甚至超越完全等变扩散策略的性能，同时实现复杂度大幅降低。

**[A Regularized Newton Method for Nonconvex Optimization with Global and Local Complexity Guarantees](human_understanding/a_regularized_newton_method_for_nonconvex_optimization_with.md)**

:   提出一类基于当前与历史梯度构造的新型正则化器，结合带负曲率监测的共轭梯度法求解正则化Newton方程，在不需要Hessian Lipschitz常数先验知识的自适应框架下，首次同时实现了$O(\epsilon^{-3/2})$最优全局迭代复杂度和二次局部收敛速率。

**[A Simple Linear Patch Revives Layer-Pruned Large Language Models](human_understanding/a_simple_linear_patch_revives_layerpruned_large_language_mod.md)**

:   提出 LinearPatch，一种即插即用的轻量修补技术，通过在剪枝界面插入一个融合了 Hadamard 变换（压制 token 级outlier）和通道缩放（对齐通道幅度）的对称矩阵，有效弥合层剪枝后的激活幅度失配问题，在 LLaMA-3-8B 上剪掉 5/32 层后仍保留 94.15% 性能（无训练），加上 30 分钟蒸馏可达 95.16%。

**[Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation for Efficient LLM Inference](human_understanding/ada-kv_optimizing_kv_cache_eviction_by_adaptive_budget_allocation_for_efficient_.md)**

:   发现现有 KV cache 驱逐方法对所有注意力头均匀分配预算忽略了头间注意力集中度的巨大差异,提出 Ada-KV——首个 head-wise 自适应预算分配策略,将稀疏头的预算重新分配给分散头,理论证明最小化驱逐损失上界,在 29 个数据集上即插即用地提升现有方法。

**[Agint: Agentic Graph Compilation for Software Engineering Agents](human_understanding/agint_agentic_graph_compilation_for_software_engineering_age.md)**

:   提出 Agint，一个将自然语言意图编译为类型化、效果感知的DAG（有向无环图）的 agentic 图编译器，通过六层类型地板（TEXT→TYPED→SPEC→STUB→SHIM→PURE）渐进式精化自然语言为可执行代码，支持中间表示可执行、混合JIT运行时和Unix风格的可组合工具链。

**[BEDLAM 2.0: Synthetic Humans and Cameras in Motion](human_understanding/bedlam20_synthetic_humans_and_cameras_in_motion.md)**

:   BEDLAM 数据集的重大升级版，新增多样化相机运动（合成+手持+头戴设备捕获）、更广的焦距范围（14-400mm）、更多样化体型/发型/鞋子/服装，总计 27K 序列 8M+ 帧，显著提升世界坐标 3D 人体估计的精度。

**[BubbleFormer: Forecasting Boiling with Transformers](human_understanding/bubbleformer_forecasting_boiling_with_transformers.md)**

:   提出 BubbleFormer，基于分解时空轴注意力的 Transformer 架构用于预测沸腾动力学——包括难以预测的自主气泡成核事件，配合 BubbleML 2.0 数据集（160+ 高保真仿真），在多种流体、几何和壁面条件下实现准确的沸腾时空过程预测。

**[Counteractive RL: Rethinking Core Principles for Efficient and Scalable Deep Reinforcement Learning](human_understanding/counteractive_rl_rethinking_core_principles_for_efficient_and_scalable_deep_rein.md)**

:   CoAct TD Learning 颠覆 ε-greedy 的随机探索范式——以概率 ε 选择最小化 $Q(s,a)$ 的动作（而非随机动作）来获取高时间差分信号，理论证明其产生更大 TD 误差，在 Atari 100K 上实现 248% 性能提升，仅需改动 2 行代码且零额外计算。

**[FACE: A General Framework for Mapping Collaborative Filtering Embeddings into LLM Tokens](human_understanding/face_a_general_framework_for_mapping_collaborative_filtering_embeddings_into_llm.md)**

:   FACE 提出将协同过滤（CF）嵌入通过解纠缠投影 + 残差量化映射为 LLM 预训练 token（描述符），再用对比学习对齐语义，无需微调 LLM 即可实现 CF 嵌入的语义解读和推荐性能增强。

**[Human-Machine Ritual: Synergic Performance through Real-Time Motion Recognition](human_understanding/human-machine_ritual_synergic_performance_through_real-time_motion_recognition.md)**

:   提出一种轻量级实时动作识别系统，利用可穿戴 IMU 传感器 + MiniRocket 时序分类器实现 <50ms 延迟的舞者特定动作识别（96.05% 准确率），通过"具身记忆映射"将舞者的个人动作-声音关联编码到系统中，构建了一种尊重人体表达深度的人机协作表演范式。

**[Incentivizing Reasoning For Advanced Instruction-Following Of Large Language Mod](human_understanding/incentivizing_reasoning_for_advanced_instruction-following_of_large_language_mod.md)**

:   提出 RAIF，通过 RL+规则中心奖励培养 LLM 在复杂指令（含 And/Chain/Selection/Nested 组合约束）下的深度推理能力：发现 vanilla CoT 对指令跟随有负面影响（因 LLM 只会浅层复述指令），设计 superior CoT enforcement（样本级对比过滤无效推理）+ 行为克隆控制分布偏移，1.5B 模型匹配 8B 性能，7 个 benchmark 平均提升 11.74%。

**[MOSPA: Human Motion Generation Driven by Spatial Audio](human_understanding/mospa_human_motion_generation_driven_by_spatial_audio.md)**

:   首次提出空间音频驱动的人体运动生成：构建 SAM 数据集（9+ 小时 Ambisonics 空间音频-运动配对数据），设计 MOSPA 扩散模型框架融合空间位置信息 + 语义音频特征，在 VR/游戏/辅助技术等方面有应用前景。

**[SPROD: Spurious-Aware Prototype Refinement for Reliable Out-of-Distribution Detection](human_understanding/spurious-aware_prototype_refinement_for_reliable_out-of-distribution_detection.md)**

:   SPROD 是一种后置（post-hoc）OOD 检测方法，专门应对训练数据中的虚假相关——通过将每个类别的原型细分为"正确分类"和"误分类"子组（后者共享虚假特征），配合 K-means 式精炼和距离式（生成式）评分，在 5 个虚假相关 OOD 基准上平均 AUROC 85.1%（+4.8% vs 次优 KNN），FPR@95 49.0%（-9.3% vs 次优）。

---

## 🎯 目标检测 { #object_detection }

**[All You Need is One: Capsule Prompt Tuning with a Single Vector](object_detection/all_you_need_is_one_capsule_prompt_tuning_with_a_single_vector.md)**

:   提出 Capsule Prompt-Tuning (CaPT)，发现现有 task-aware soft prompts 实际上与输入 tokens 缺乏交互（"attention 孤岛"），而将 instance-aware 信息融入单个 capsule prompt 可以作为"attention anchor"激活对关键结构信息的注意力，以极低参数量（如 Llama3.2-1B 上仅 0.003% 参数）实现超越多 prompt 方法的性能。

**[Angular Constraint Embedding via SpherePair Loss for Constrained Clustering](object_detection/angular_constraint_embedding_via_spherepair_loss_for_constrained_clustering.md)**

:   提出 SpherePair loss，在角度空间（而非欧氏空间）中学习约束聚类的表示，通过余弦相似度编码 pairwise 约束，避免了端到端 DCC 方法对 anchor 的依赖和欧氏嵌入中正负对距离平衡的困难，无需预知聚类数目即可实现 SOTA 的约束聚类性能。

**[Any Large Language Model Can Be a Reliable Judge: Debiasing with a Reasoning-based Bias Detector](object_detection/any_large_language_model_can_be_a_reliable_judge_debiasing_w.md)**

:   提出 Reasoning-based Bias Detector（RBD）作为 LLM 评判器的即插即用去偏模块——通过外部检测 4 种评估偏见（冗长/位置/从众/情感），生成带推理链的结构化反馈引导评判器自我纠正，RBD-8B 在 8 个 LLM 评判器上平均提升准确率 18.5%、一致性 10.9%。

**[Ascent Fails to Forget](object_detection/ascent_fails_to_forget.md)**

:   挑战了机器遗忘领域的常见信念，证明梯度上升（gradient ascent）基于的无约束优化方法在遗忘/保留集之间存在统计依赖时会系统性失败——遗忘集指标的降低不可避免地损害整体测试性能，logistic 回归示例甚至展示了遗忘过程使模型比原始模型更远离 oracle 的灾难性情况。

**[Automated Detection of Visual Attribute Reliance with a Self-Reflective Agent](object_detection/automated_detection_of_visual_attribute_reliance_with_a_self-reflective_agent.md)**

:   提出一个自反思 agent 框架，通过迭代的假设生成-测试-验证-反思循环来自动检测视觉模型中的属性依赖（如 CLIP 识别 teacher 依赖教室背景、YOLOv8 检测行人依赖人行横道），在 130 个注入已知属性依赖的模型 benchmark 上显示自反思显著提升检测准确性。

**[CQ-DINO: Mitigating Gradient Dilution via Category Queries for Vast Vocabulary Object Detection](object_detection/cq-dino_mitigating_gradient_dilution_via_category_queries_for_vast_vocabulary_ob.md)**

:   针对大规模类别（>10K）目标检测中分类头的正梯度稀释和难负样本梯度稀释问题，提出 CQ-DINO：用可学习类别查询替代分类头，通过图像引导的 Top-K 类别选择将负空间缩小 100 倍，在 V3Det（13204 类）上超越前 SOTA 2.1% AP，同时保持 COCO 竞争力。

**[DetectiumFire: A Comprehensive Multi-modal Dataset Bridging Vision and Language for Fire Understanding](object_detection/detectiumfire_a_comprehensive_multi-modal_dataset_bridging_vision_and_language_f.md)**

:   DetectiumFire 构建了最大的多模态火灾理解数据集——14.5K 真实图像 + 2.5K 视频 + 8K 合成图像 + 12K RLHF 偏好对，低重复率（0.03 PHash vs D-Fire 0.15），配合 4 级严重性分类标准和详细场景描述，微调 YOLOv11m 达 mAP 43.74，微调 LLaMA-3.2-11B 火灾严重性分类 83.84%。

**[DitHub: A Modular Framework for Incremental Open-Vocabulary Object Detection](object_detection/dithub_a_modular_framework_for_incremental_openvocabulary_ob.md)**

:   提出 DitHub，借鉴版本控制系统（Git）思想构建开放词汇目标检测的模块化适配框架——将不同领域的高效适配模块（LoRA）作为"分支"管理，支持按需获取（fetch）和合并（merge），在 ODinW-13 上达到 SOTA，首次系统性研究目标检测中适配模块的组合特性。

**[Generalizable Insights for Graph Transformers in Theory and Practice](object_detection/generalizable_insights_for_graph_transformers_in_theory_and_practice.md)**

:   提出 Generalized-Distance Transformer (GDT)，一种基于标准注意力（无需修改注意力机制）的图 Transformer 架构，理论证明其表达力等价于 GD-WL 算法，并通过覆盖 800 万图/2.7 亿 token 的大规模实验首次建立了 PE 表达力的细粒度经验层次，在 few-shot 迁移设置下无需微调即可超越 SOTA。

**[InstanceAssemble: Layout-Aware Image Generation via Instance Assembling Attention](object_detection/instanceassemble_layoutaware_image_generation_via_instance_a.md)**

:   提出InstanceAssemble，通过实例组装注意力机制（instance-assembling attention）实现layout条件的精确控制——支持bbox位置控制和多模态内容控制（文本+视觉内容），作为轻量LoRA模块适配到现有DiT模型，同时提出DenseLayout benchmark（5K图像90K实例）和Layout Grounding Score评估指标。

**[OverLayBench: A Benchmark for Layout-to-Image Generation with Dense Overlaps](object_detection/overlaybench_a_benchmark_for_layout-to-image_generation_with_dense_overlaps.md)**

:   OverLayBench 构建了首个聚焦密集重叠场景的 Layout-to-Image 基准（4052 样本 + OverLayScore 难度指标），揭示 SOTA 方法在复杂重叠下 mIoU 从 71%→54% 急剧退化，提出 Amodal Mask 监督在重叠 IoU 上提升 15.9%。

**[SAFE: Multitask Failure Detection for Vision-Language-Action Models](object_detection/safe_multitask_failure_detection_for_vision-language-action_models.md)**

:   SAFE 发现 VLA 模型的内部特征空间存在跨任务一致的"失败区域"，据此训练轻量 MLP/LSTM 失败检测器，配合功能保形预测（FCP）做阈值校准，在未见任务上达 78% ROC-AUC，计算开销 <1%，大幅优于 token 不确定性和一致性检测方法。

**[Test-Time Adaptive Object Detection with Foundation Model](object_detection/test-time_adaptive_object_detection_with_foundation_model.md)**

:   提出无需源域数据的开放词汇测试时自适应目标检测框架（TTAOD），通过多模态 Prompt Tuning + Mean-Teacher + 实例动态记忆（IDM）+ 记忆增强/幻觉策略，在 Pascal-C 上 AP50 达 56.2%（+11.0 vs SOTA），在 13 个跨域数据集上一致有效。

**[The Complexity of Finding Local Optima in Contrastive Learning](object_detection/the_complexity_of_finding_local_optima_in_contrastive_learning.md)**

:   证明对比学习中寻找局部最优是计算困难的：离散三元组最大化问题是 PLS-hard（即使 $d=1$），连续三元组损失最小化是 CLS-hard，意味着（在标准假设下）不存在多项式时间算法找到局部最优。

---

## 🧊 3D 视觉 { #3d_vision }

**[3D-Agent: Tri-Modal Multi-Agent Collaboration for Scalable 3D Object Annotation](3d_vision/3d-agenttri-modal_multi-agent_collaboration_for_scalable_3d_object_annotation.md)**

:   提出 Tri-MARF 三模态多智能体框架，通过 VLM 标注 Agent（多视角多候选描述）+ 信息聚合 Agent（BERT 聚类 + CLIP 加权 + UCB1 多臂赌博机选择）+ 点云门控 Agent（Uni3D 文本-点云对齐过滤幻觉），实现 CLIPScore 88.7（超越人类标注 82.4）、吞吐量 12k 物体/小时，已标注约 200 万 3D 模型。

**[3D Visual Illusion Depth Estimation](3d_vision/3d_visual_illusion_depth_estimation.md)**

:   揭示了3D视觉错觉（如墙面彩绘、屏幕重播、镜面反射等）会严重欺骗现有SOTA单目和双目深度估计方法，构建了包含约3k场景/200k图像的大规模数据集，并提出基于VLM常识推理的单目-双目自适应融合框架，在各类错觉场景下达到SOTA。

**[Anti-Aliased 2D Gaussian Splatting](3d_vision/anti-aliased_2d_gaussian_splatting.md)**

:   提出 AA-2DGS，通过世界空间平坦平滑核和物体空间 Mip 滤波器两个互补机制，解决 2D Gaussian Splatting 在不同采样率下渲染时的严重锯齿问题，在保持 2DGS 几何精度优势的同时显著提升多尺度渲染质量。

**[ARMesh: Autoregressive Mesh Generation via Next-Level-of-Detail Prediction](3d_vision/armesh_autoregressive_mesh_generation_via_next-level-of-detail_prediction.md)**

:   提出将 3D mesh 生成建模为"由粗到精"的逐级细化过程（next-level-of-detail prediction），通过反转广义网格简化算法（GSlim）获得渐进式细化序列，再用 Transformer 自回归学习，从单个点开始逐步增加几何与拓扑细节生成完整网格。

**[Atlasgs Atlanta-World Guided Surface Reconstruction With Implicit Structured Gau](3d_vision/atlasgs_atlanta-world_guided_surface_reconstruction_with_implicit_structured_gau.md)**

:   提出 AtlasGS，通过将 Atlanta-world 结构先验引入隐式结构化高斯表示（implicit-structured Gaussians），在室内和城市场景中实现平滑且保留高频细节的高质量表面重建，全面超越已有隐式和显式方法。

**[BecomingLit: Relightable Gaussian Avatars with Hybrid Neural Shading](3d_vision/becominglit_relightable_gaussian_avatars_with_hybrid_neural_shading.md)**

:   提出 BecomingLit，基于 3D Gaussian 原语和混合神经着色（neural diffuse BRDF + 解析 Cook-Torrance specular）从低成本 light stage 多视角序列重建可重光照、实时渲染的高保真头部 avatar，并发布了新的公开 OLAT 人脸数据集。

**[Can LLMs Write Faithfully? An Agent-Based Evaluation of LLM-generated Islamic Content](3d_vision/can_llms_write_faithfully_an_agent-based_evaluation_of_llm-generated_islamic_con.md)**

:   提出双Agent（定量+定性）评估框架，从神学准确性、引用完整性和文体恰当性三个维度系统评估 GPT-4o、Ansari AI 和 Fanar 在伊斯兰内容生成任务上的忠实度，发现即使最优模型也在引用可靠性上存在显著不足。

**[Concerto: Joint 2D-3D Self-Supervised Learning Emerges Spatial Representations](3d_vision/concerto_joint_2d-3d_self-supervised_learning_emerges_spatial_representations.md)**

:   Concerto 将 3D 点云模态内自蒸馏与 2D-3D 跨模态联合嵌入预测相结合，以极简设计让单一点云编码器（PTv3）涌现出超越 2D/3D 单模态甚至两者拼接的空间表征，在多个 3D 场景理解基准上刷新 SOTA（ScanNet 语义分割 80.7% mIoU）。

**[Cue3D: Quantifying the Role of Image Cues in Single-Image 3D Generation](3d_vision/cue3d_quantifying_the_role_of_image_cues_in_single-image_3d_generation.md)**

:   提出 Cue3D——首个模型无关的框架，通过系统性扰动 6 种图像线索（光照/纹理/轮廓/透视/边缘/局部连续性）量化其对单图 3D 生成的影响，在 7 个 SOTA 方法上揭示：形状意义而非纹理决定泛化性，光照比纹理更重要，模型过度依赖轮廓——为更透明、鲁棒的 3D 生成指明方向。

**[From Pixels To Views Learning Angular-Aware And Physics-Consistent Representatio](3d_vision/from_pixels_to_views_learning_angular-aware_and_physics-consistent_representatio.md)**

:   提出XLFM-Former用于扩展光场显微镜(XLFM)的3D重建：构建首个XLFM-Zebrafish标准化基准，设计Masked View Modeling (MVM-LF)自监督预训练学习角度先验，引入光学渲染一致性损失(ORC Loss)确保物理可信性，PSNR较SOTA提升7.7%（54.04 vs 50.16 dB）。

**[From Programs to Poses: Factored Real-World Scene Generation via Learned Program Libraries](3d_vision/from_programs_to_poses_factored_real-world_scene_generation_via_learned_program_.md)**

:   提出 FactoredScenes，将真实世界 3D 场景生成分解为五步因式分解——从合成数据学布局程序库、LLM 生成场景程序、执行程序获得轴对齐布局、程序条件化层次姿态预测、物体检索放置，在卧室上 FID 改善 38.3%、KID 改善 80.4%，人类仅 67% 能区分生成与真实 ScanNet。

**[Jasmine Harnessing Diffusion Prior For Self-Supervised Depth Estimation](3d_vision/jasmine_harnessing_diffusion_prior_for_self-supervised_depth_estimation.md)**

:   首次将Stable Diffusion视觉先验引入自监督单目深度估计：提出Mix-Batch Image Reconstruction避免自监督噪声损坏SD先验，设计Scale-Shift GRU桥接SD的尺度偏移不变性(SSI)与自监督的尺度不变性(SI)深度，在KITTI上AbsRel达0.102且泛化性强。

**[Object-Centric Representation Learning For Enhanced 3D Semantic Scene Graph Pred](3d_vision/object-centric_representation_learning_for_enhanced_3d_semantic_scene_graph_pred.md)**

:   通过实证分析揭示物体特征可区分性是 3D 场景图谓词预测的关键瓶颈（物体分类错误导致 92%+ 的谓词错误），提出独立对比预训练的物体编码器（3D-2D-Text 三模态对齐）+ 几何正则化关系编码器 + 双向边门控 GNN，在 3DSSG 上 Object R@1 59.53%、Predicate R@50 91.40% 均达新 SOTA。

---

## 🔄 自监督/表示学习 { #self_supervised }

**[A Unified Reasoning Framework for Holistic Zero-Shot Video Anomaly Analysis](self_supervised/a_unified_reasoning_framework_for_holistic_zeroshot_video_an.md)**

:   提出一个完全零样本、无需训练的视频异常分析框架，通过Intra-Task Reasoning（置信度门控的自我精化）和Inter-Task Chaining（从时序检测到空间定位到语义理解的级联prompt传递），在4个benchmark上全面超越先前零样本方法4-6% AUC。

**[Adv-SSL: Adversarial Self-Supervised Representation Learning with Theoretical Guarantees](self_supervised/adv-ssl_adversarial_self-supervised_representation_learning_with_theoretical_gua.md)**

:   提出 Adv-SSL，通过将协方差正则项的 Frobenius 范数重写为 minimax 对偶形式，消除了 Barlow Twins 等方法中样本级风险的有偏估计问题，在不增加额外计算成本的前提下显著提升下游分类性能，并给出端到端的理论收敛保证。

**[BrainOmni: A Brain Foundation Model for Unified EEG and MEG Signals](self_supervised/brainomni_a_brain_foundation_model_for_unified_eeg_and_meg_signals.md)**

:   提出BrainOmni——首个统一EEG和MEG的脑信号基础模型，通过BrainTokenizer（含物理传感器编码）实现设备不可知的信号离散化，使用Criss-Cross Transformer进行自监督预训练，在阿尔茨海默病检测等任务上提升11.7个百分点并实现未见设备的零样本泛化。

**[Chain-of-Retrieval Augmented Generation (CoRAG)](self_supervised/chain-of-retrieval_augmented_generation.md)**

:   提出CoRAG框架，通过拒绝采样自动生成中间检索链（子查询→子答案），微调LLM学习迭代检索和推理，并支持多种测试时解码策略（贪心/Best-of-N/树搜索）灵活扩展计算，在多跳QA上提升26+ EM点，KILT基准9/10任务达SOTA。

**[Connecting Jensen-Shannon and Kullback-Leibler Divergences: A New Bound for Representation Learning](self_supervised/connecting_jensenshannon_and_kullbackleibler_divergences_a_n.md)**

:   推导了一般情况下 KL 散度关于 JS 散度的新的紧致可计算下界，证明最大化 JSD 目标等价于最大化互信息的一个下界，为判别式学习在 MI 基础表示学习中的使用提供了理论基础，并在 MI 估计和 Information Bottleneck 中验证了其紧致性和实用性。

**[Continuous Subspace Optimization for Continual Learning (CoSO)](self_supervised/continuous_subspace_optimization_for_continual_learning.md)**

:   提出CoSO框架，通过从每步梯度的SVD动态导出连续子空间（而非LoRA的固定子空间），结合历史任务正交投影防止干扰和Frequent Directions高效聚合梯度信息，在ImageNet-R 20任务上比最佳baseline提升2.77个百分点。

**[Contrastive Representations for Temporal Reasoning](self_supervised/contrastive_representations_for_temporal_reasoning.md)**

:   论文研究能否用纯表示学习替代显式搜索来承担部分时序推理，指出标准 temporal contrastive learning 容易抓住伪特征而失去时序结构，进一步提出 CRTR（Combinatorial Representations for Temporal Reasoning），通过特制负采样从理论上去除伪特征，学到同时编码感知与时序结构的表示，在 Sokoban 和 Rubik's Cube 上取得强结果，甚至可在不依赖外部搜索算法的情况下求解任意初始魔方状态。

**[Know Thyself by Knowing Others: Learning Neuron Identity from Population Context](self_supervised/know_thyself_by_knowing_others_learning_neuron_identity_from_population_context.md)**

:   提出NuCLR自监督框架，通过对比学习对群体神经活动中同一神经元的不同时间窗口拉近、不同神经元推远，学习包含群体上下文的神经元级表征，在细胞类型和脑区解码上达到新SOTA，并首次展示了跨动物零样本泛化和数据缩放规律。

**[Minimal Semantic Sufficiency Meets Unsupervised Domain Generalization](self_supervised/minimal_semantic_sufficiency_meets_unsupervised_domain_generalization.md)**

:   MS-UDG 在无类别标签和域标签的条件下，通过信息解纠缠模块（IDM）将表征分解为语义和变异成分，配合最小语义充分性优化模块（SROM）最大化语义信息同时最小化变异干扰，在 PACS 上达 72.89% 准确率（+1.5% vs CycleMAE），理论证明最小充分语义表征最小化下游贝叶斯错误率。

**[Sciarena An Open Evaluation Platform For Non-Verifiable Scientific Literature-Gr](self_supervised/sciarena_an_open_evaluation_platform_for_non-verifiable_scientific_literature-gr.md)**

:   构建SciArena——社区驱动的科学文献基础模型开放评估平台，支持47个模型和20K+偏好投票，同时发布SciArena-Eval元基准评估自动评估系统判断能力。

**[Self-Supervised Contrastive Learning is Approximately Supervised Contrastive Learning](self_supervised/self-supervised_contrastive_learning_is_approximately_supervised_contrastive_lea.md)**

:   从理论上证明自监督对比学习（DCL）近似等价于一种有监督对比损失（NSCL），两者差距以 $O(1/C)$ 速度随类别数增加而消失；进一步证明 NSCL 全局最优解满足 Neural Collapse（增强坍缩 + 类内坍缩 + Simplex ETF），并提出基于方向性 CDNV 的更紧的 few-shot 误差界。

**[T-REGS: Minimum Spanning Tree Regularization for Self-Supervised Learning](self_supervised/t-regs_minimum_spanning_tree_regularization_for_self-supervised_learning.md)**

:   提出 T-REGS——一种基于最小生成树(MST)长度最大化的自监督学习正则化框架，理论证明可同时防止维度坍缩并促进表示分布均匀性，在紧致黎曼流形上成立，实验在标准 JE-SSL 基准上验证了有效性。

**[Tabstar A Tabular Foundation Model For Tabular Data With Text Fields](self_supervised/tabstar_a_tabular_foundation_model_for_tabular_data_with_text_fields.md)**

:   提出 TabSTAR，一个专为含文本字段的表格数据设计的基础模型：通过解冻文本编码器（e5-small-v2）端到端优化文本表征 + 目标感知 token 注入分类目标语义信息 + 无数据集特定参数的架构实现跨数据集迁移学习，在 350 个数据集上预训练后，分类任务上 14 个数据集中 12 个超越 CatBoost-Tuned（4h 调参），8/11 超越 TabPFN-v2。

---

## 📡 信号/通信 { #signal_comm }

**[Angular Steering: Behavior Control via Rotation in Activation Space](signal_comm/angular_steering_behavior_control_via_rotation_in_activation_space.md)**

:   提出 Angular Steering，将 LLM 激活引导统一建模为固定 2D 子空间中的旋转操作，提供连续、细粒度、范数保持的行为控制，统一了现有的激活加法和方向消融方法，在多个 LLM 家族（3B-14B）上实现鲁棒的行为控制。

**[Artificial Hivemind: The Open-Ended Homogeneity of Language Models (and Beyond)](signal_comm/artificial_hivemind_the_open-ended_homogeneity_of_language_models_and_beyond.md)**

:   构建了 Infinity-Chat 数据集（26K 开放式真实用户查询 + 31,250 条人类标注），揭示了 LM 在开放式生成中的"Artificial Hivemind"效应——模型内重复和模型间同质化严重，并发现 Reward Model 和 LM Judge 在个体偏好差异大的样本上校准失败。

**[Bispectral OT: Dataset Comparison using Symmetry-Aware Optimal Transport](signal_comm/bispectral_ot_dataset_comparison_using_symmetry-aware_optimal_transport.md)**

:   提出 Bispectral Optimal Transport (BOT)，将离散最优传输中的代价矩阵从原始像素距离替换为 bispectrum（群 Fourier 不变量）距离，使得传输计划在保持信号结构的同时精确消除群作用（如旋转）带来的变异，在旋转变换的 MNIST 等数据集上将类别保持准确率从 33% 提升至 84%。

**[ConTextTab: A Semantics-Aware Tabular In-Context Learner](signal_comm/contexttab_a_semantics-aware_tabular_in-context_learner.md)**

:   提出 ConTextTab，将语义理解融入 table-native ICL 框架，用数据类型特定嵌入并在大规模真实世界表格数据上训练，在语义丰富的 CARTE benchmark 上设立新 SOTA。

**[Contrastive Consolidation of Top-Down Modulations Achieves Sparsely Supervised Continual Learning](signal_comm/contrastive_consolidation_of_top-down_modulations_achieves_sparsely_supervised_c.md)**

:   提出 Task-Modulated Contrastive Learning (TMCL)，受大脑新皮层自顶向下调制启发，在持续学习中通过 affine modulation 集成稀疏标签信息（仅需 1% 标签），再利用对比学习将调制信息固化到前馈权重中，在 class-incremental 和迁移学习上超越无监督和有监督基线。

**[Don't Let It Fade: Preserving Edits in Diffusion Language Models via Token Timestep Allocation](signal_comm/dont_let_it_fade_preserving_edits_in_diffusion_language_mode.md)**

:   提出 Token Timestep Allocation (TTA-Diffusion)，通过为每个 token 分配独立的去噪时间步来解决扩散语言模型中 classifier guidance 导致的 update-forgetting 问题，实现可控文本生成的稳定性和效率大幅提升。

**[Estimation of Stochastic Optimal Transport Maps](signal_comm/estimation_of_stochastic_optimal_transport_maps.md)**

:   提出随机最优传输映射的新评价指标 $\mathcal{E}_p$（优化间隙+可行性间隙），发展了高效估计器，达到近优有限样本风险界 $\tilde{O}(n^{-1/(d+2p)})$，且仅需最小假设，是首个通用的（可能随机的）OT 映射估计理论。

**[Feature-aware Modulation for Learning from Temporal Tabular Data](signal_comm/feature-aware_modulation_for_learning_from_temporal_tabular_data.md)**

:   提出特征感知时间调制机制，通过基于时间上下文的可学习 Yeo-Johnson 变换动态调整特征分布（均值、标准差、偏度），实现跨时间语义对齐。

**[Masked Symbol Modeling for Demodulation of Oversampled Baseband Communication Signals](signal_comm/masked_symbol_modeling_for_demodulation_of_oversampled_baseband_communication_si.md)**

:   提出 Masked Symbol Modeling，将 BERT 的掩码预测范式应用于通信物理层，将脉冲成形引起的符号间贡献视为上下文信息，训练 Transformer 在干净信号上学习波形结构，推理时通过上下文恢复被冲激噪声破坏的符号。

**[Memory-Integrated Reconfigurable Adapters (MIRA)](signal_comm/memory-integrated_reconfigurable_adapters_a_unified_framework_for_settings_with_.md)**

:   提出 MIRA，将 Hopfield 联想记忆与 LoRA adapter 结合，在共享 backbone 的每个 ViT 层上存储 adapter 权重更新为 value、事后学习的 key 检索，统一处理域泛化、类增量学习和域增量学习，在多个设置下达到 SoTA。

**[Multi-Modal Masked Autoencoders for Galaxy Evolution and Cosmology](signal_comm/multi-modal_masked_autoencoders_for_learning_image-spectrum_associations_for_gal.md)**

:   将多模态掩码自编码器 (MMAE) 应用于星系图像和光谱的联合重建，构建了 134,533 个星系的图像+光谱数据集，实现了光谱和图像的交叉重建以及仅从图像的红移回归，$\sigma_{\text{NMAD}} = 0.016$ 优于 AstroCLIP。

**[Perturbation Bounds for Low-Rank Inverse Approximations under Noise](signal_comm/perturbation_bounds_for_low-rank_inverse_approximations_under_noise.md)**

:   首次给出在加性噪声下低秩逆近似 $\|(\tilde{A}^{-1})_p - A_p^{-1}\|$ 的非渐近谱范数扰动界，利用轮廓积分技术得到依赖特征间隙、谱衰减和噪声对齐的锐界，比经典全逆界改进高达 $\sqrt{n}$ 倍。

**[The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning](signal_comm/the_surprising_effectiveness_of_negative_reinforcement_in_llm_reasoning.md)**

:   揭示RLVR中负强化（仅惩罚错误）的効果超出预期，通过梯度分析说明其保持输出多样性和推理能力的机制，并提出改进的加权REINFORCE算法。

---

## 🚗 自动驾驶 { #autonomous_driving }

**[3EED: Ground Everything Everywhere in 3D](autonomous_driving/3eed_ground_everything_everywhere_in_3d.md)**

:   提出 3EED——首个大规模多平台（车、无人机、四足机器人）、多模态（LiDAR+RGB）室外 3D 视觉定位基准，包含超 12.8 万目标和 2.2 万语言描述，规模是现有室外数据集的 10 倍；同时设计了跨平台对齐、多尺度采样和尺度自适应融合的基线方法，揭示了跨平台 3D grounding 的巨大性能差距。

**[AHA -- Predicting What Matters Next: Online Highlight Detection Without Looking Ahead](autonomous_driving/aha_predicting_what_matters_next_online_highlight_detection.md)**

:   提出 AHA，一个自回归高光检测框架，在**不访问未来帧**的情况下根据自然语言任务描述实时预测每帧视频的相关性——利用多模态视觉语言模型+轻量解耦头+Dynamic SinkCache实现无限长度流媒体的恒定内存推理，在TVSum上超越离线全上下文方法+5.9% mAP、在Mr. Hisum上+8.3% mAP。

**[AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning](autonomous_driving/autovla_a_vision-language-action_model_for_end-to-end_autonomous_driving_with_ad.md)**

:   提出AutoVLA——基于Qwen2.5-VL-3B的端到端自动驾驶VLA模型，将连续轨迹离散化为物理action tokens嵌入语言模型词表，支持fast/slow thinking双模式推理，通过GRPO强化微调同时提升10.6%性能和66.8%推理效率，在NAVSIM和Bench2Drive上达SOTA。

**[Availability-aware Sensor Fusion via Unified Canonical Space](autonomous_driving/availability-aware_sensor_fusion_via_unified_canonical_space.md)**

:   提出 ASF（Availability-aware Sensor Fusion），通过统一规范投影（UCP）将 Camera/LiDAR/4D Radar 特征映射到共享空间 + 跨传感器沿 patch 交叉注意力（CASAP，复杂度 $O(N_qN_s)$ 而非 $O(N_qN_sN_p)$）自动适配可用传感器 + 传感器组合损失（SCL）覆盖所有 7 种组合，在 K-Radar 上 AP_3D 73.6%（超 SOTA 20.1%），传感器故障时性能仅降 1.7%。

**[BayesG: Bayesian Ego-Graph Inference for Networked Multi-Agent Reinforcement Learning](autonomous_driving/bayesian_ego-graph_inference_for_networked_multi-agent_reinforcement_learning.md)**

:   BayesG 让网络化 MARL 中的每个 agent 通过贝叶斯变分推断学习其局部通信图的动态结构——用 Gumbel-Softmax 采样边掩码、ELBO 目标联合优化策略和图结构，在 167 agent 的纽约交通场景中奖励比最佳 baseline 高 50%+。

**[Causality Meets Locality: Provably Generalizable and Scalable Policy Learning for Networked Systems](autonomous_driving/causality_meets_locality_provably_generalizable_and_scalable_policy_learning_for.md)**

:   提出 GSAC 框架，将因果表示学习与元 Actor-Critic 结合，通过从网络 MARL 中学习稀疏因果掩码构建近似紧凑表示 (ACR) 实现可扩展性，通过域因子条件化策略实现跨域泛化，给出了因果恢复、收敛和自适应间隙的有限样本保证。

**[Chronograph A Real-World Graph-Based Multivariate Time Series Dataset](autonomous_driving/chronograph_a_real-world_graph-based_multivariate_time_series_dataset.md)**

:   提出 ChronoGraph——首个同时包含多元时间序列、显式服务依赖图和事件标签的真实世界微服务数据集（6个月 / ~700服务 / 5维指标 / 8005时间步），基准测试表明现有预测和异常检测方法在长期预测和拓扑感知方面均存在较大提升空间。

**[DINO-Foresight: Looking into the Future with DINO](autonomous_driving/dino-foresight_looking_into_the_future_with_dino.md)**

:   提出DINO-Foresight——在VFM（视觉基础模型）的语义特征空间中预测未来帧，通过自监督Masked Feature Transformer预测DINOv2特征的时间演化，搭配即插即用的task-specific heads实现单一模型同时处理4种场景理解任务（语义分割/实例分割/深度/表面法线），大幅超越VISTA世界模型且快100×。

**[DriveDPO: Policy Learning via Safety DPO For End-to-End Autonomous Driving](autonomous_driving/drivedpo_policy_learning_via_safety_dpo_for_end-to-end_autonomous_driving.md)**

:   提出DriveDPO两阶段框架——先通过统一策略蒸馏将人类模仿相似度与规则安全分数融合为单一监督分布，再用Safety DPO构建"看似human-like但不安全 vs 既human-like又安全"的轨迹偏好对进行策略微调——在NAVSIM上达PDMS 90.0新SOTA。

**[FutureSightDrive: Thinking Visually with Spatio-Temporal CoT for Autonomous Driving](autonomous_driving/futuresightdrive_thinking_visually_with_spatiotemporal_cot_f.md)**

:   FutureSightDrive 认为自动驾驶 VLA 的文本 CoT 会把关键视觉时空信息压缩丢失，提出“视觉时空 CoT”范式：先让模型以 world model 方式生成融合未来背景、车道线和 3D 目标框的统一未来帧，再将该 imagined scene 作为推理中介供 inverse-dynamics 规划器生成轨迹，从而显著提升轨迹精度、降低碰撞并改善场景理解。

**[SDTagNet: Leveraging Text-Annotated Navigation Maps for Online HD Map Construction](autonomous_driving/sdtagnet_leveraging_text-annotated_navigation_maps_for_online_hd_map_constructio.md)**

:   提出 SDTagNet，首次通过 BERT 编码 OpenStreetMap 文本标注（路名/车道数/单行道等）并用点级图 Transformer 编码所有 SD 地图元素（点/线/关系），在远距离 HD 地图构建上相比无先验方法提升 +5.9 mAP（+45%），超越已有 SD 地图先验方法 +3.2 mAP（+20%）。

**[Unifying Appearance Codes and Bilateral Grids for Driving Scene Gaussian Splatting](autonomous_driving/unifying_appearance_codes_and_bilateral_grids_for_driving_scene_gaussian_splatti.md)**

:   提出多尺度双边网格金字塔统一全局外观编码和像素级双边网格——3 级层级（粗→中→细）分别捕捉全局/区域/像素级光度变化，通过亮度引导的切片-融合管线和自适应正则化解决驾驶场景 3DGS 的光度不一致问题，Waymo 上 Chamfer Distance 比 OmniRe 改善 28.2%。

---

## 🎬 视频理解 { #video_understanding }

**[A Little Depth Goes a Long Way: The Expressive Power of Log-Depth Transformers](video_understanding/a_little_depth_goes_a_long_way_the_expressive_power_of_logde.md)**

:   本文证明了将 Transformer 的深度从常数增长到 Θ(log n) 就能解锁识别正则语言和图连通性这两类固定深度 Transformer 无法表达的问题，且深度扩展比宽度（需超多项式增长）和 CoT 步数（需超对数增长）都更高效。

**[AdaVideoRAG: Omni-Contextual Adaptive Retrieval-Augmented Efficient Long Video Understanding](video_understanding/adavideorag_omnicontextual_adaptive_retrievalaugmented_effic.md)**

:   提出 AdaVideoRAG，通过轻量级意图分类器将查询按难度路由到三级检索路径（无检索/朴素检索/图检索），结合全知识索引模块（caption+ASR+OCR+视觉+知识图谱）实现长视频理解的效率-精度最优平衡，在 MLVU 上为 Qwen2.5-VL-7B 带来 39.8% 提升。

**[Adversarial Locomotion and Motion Imitation for Humanoid Policy Learning](video_understanding/adversarial_locomotion_and_motion_imitation_for_humanoid_policy_learning.md)**

:   ALMI提出上下半身对抗训练框架：下半身策略在上半身动作干扰下学习鲁棒运动，上半身策略在下半身运动干扰下学习精确动作模仿，通过迭代对抗训练收敛到Nash均衡，实现Unitree H1-2真实机器人的稳定全身协调控制。

**[Agentic Persona Control and Task State Tracking for Realistic User Simulation](video_understanding/agentic_persona_control_and_task_state_tracking_for_realistic_user_simulation_in.md)**

:   提出三 agent 协作框架用于逼真的用户模拟——User Agent（协调）+ State Tracking Agent（结构化任务状态）+ Message Attributes Generation Agent（基于 persona 和状态的行为属性控制），在餐厅点餐场景中综合仿真质量（CRRS）提升 102.6%，persona 保持度 +19.9%，行为自然度 +284.5%，且核心发现：无状态感知的行为控制导致 BVS=0（完全刚性）。

**[CleverBirds: A Multiple-Choice Benchmark for Fine-grained Human Knowledge Tracing](video_understanding/cleverbirds_a_multiple-choice_benchmark_for_fine-grained_human_knowledge_tracing.md)**

:   发布 CleverBirds——超大规模细粒度视觉知识追踪基准，包含 4万+用户的 1700万+多选题交互（覆盖 10000+鸟类物种），展示了追踪细粒度视觉专家技能发展的挑战性，为 KT 方法提供了前所未有的视觉领域评测平台。

**[Empower Words: DualGround for Structured Phrase and Sentence-Level Temporal Grounding](video_understanding/empower_words_dualground_for_structured_phrase_and_sentencel.md)**

:   论文指出现有视频时序定位模型在跨模态注意力中往往过度依赖句末 [EOS] token 的全局语义、忽视词级局部信号，提出 DualGround 双分支架构，将句子级全局语义与短语级局部语义显式解耦建模，在 QVHighlights 和 Charades-STA 上实现 Moment Retrieval 与 Highlight Detection 的 SOTA。

**[Enhancing Temporal Understanding in Video-LLMs through Stacked Temporal Attention in Vision Encoders](video_understanding/enhancing_temporal_understanding_in_videollms_through_stacke.md)**

:   提出 STAVEQ2，在 Vision Encoder 中堆叠参数高效的时序注意力模块（STA），解决现有 Video-LLM 在细粒度时序理解（如区分"从左到右拉"和"从右到左拉"）上的根本性架构缺陷，在 VITATECS/MVBench/Video-MME 上提升最高 5.5%。

**[FastVID: Dynamic Density Pruning for Fast Video Large Language Models](video_understanding/fastvid_dynamic_density_pruning_for_fast_video_large_languag.md)**

:   提出 FastVID，通过动态时序分割 (DySeg) + 密度空时剪枝 (STPrune) 从时间和视觉两个维度系统性消除视频 token 冗余，在 LLaVA-OneVision-7B 上剪掉 90.3% 视频 token 后仍保留 98% 精度，LLM prefill 阶段加速 7.1×。

**[Foresight: Adaptive Layer Reuse for Accelerated and High-Quality Text-to-Video Generation](video_understanding/foresight_adaptive_layer_reuse_for_accelerated_and_highquali.md)**

:   提出 Foresight，一种训练无关的自适应层复用框架，通过动态 MSE 阈值决策在 DiT 去噪过程中哪些层可复用缓存、哪些需重新计算，在 OpenSora/Latte/CogVideoX 上实现最高 1.63× 端到端加速且保持视频质量。

**[TempSamp-R1: Effective Temporal Sampling with Reinforcement Fine-Tuning for Video LLMs](video_understanding/tempsampr1_effective_temporal_sampling_with_reinforcement_fi.md)**

:   提出 TempSamp-R1，针对视频时序定位任务改进 GRPO 强化微调框架，通过 off-policy 时间精确引导 + 非线性软优势计算 + 混合 CoT 训练，在 Charades-STA/ActivityNet/QVHighlights 上分别提升 +2.7%/+5.3%/+3.0%。

**[Tool-Augmented Spatiotemporal Reasoning for Streamlining Video Question Answering Task](video_understanding/toolaugmented_spatiotemporal_reasoning_for_streamlining_vide.md)**

:   论文为复杂 VideoQA 提出一套轻量但可扩展的 Video Toolkit，并设计 STAR（Spatiotemporal Reasoning Framework）来调度时间工具与空间工具的调用顺序，逐步定位视频关键区域，显著增强 GPT-4o 的时空推理能力，在 VideoMME 上提升 8.2%，在 LongVideoBench 上提升 4.6%。

**[Two Causally Related Needles in a Video Haystack](video_understanding/two_causally_related_needles_in_a_video_haystack.md)**

:   提出CAUSAL2NEEDLES benchmark评估VLM的长视频双针(2-needle)因果推理能力：需要从视频两个不同位置提取因果关联的事件信息并联合推理，利用"桥接实体"迫使模型先理解结果再追溯原因，揭示即使GPT-4o在2-needle因果问题上仅达13.4%的Both准确率（vs人类79.3%）。

---

## ✍️ 文本生成 { #nlp_generation }

**[Bayesian Evaluation of Large Language Model Behavior](nlp_generation/bayesian_evaluation_of_large_language_model_behavior.md)**

:   提出基于 Beta-Binomial 贝叶斯模型的 LLM 行为评估框架，通过对每个 prompt 的随机生成结果建模 $\theta_m$ 后验分布，量化评估指标的统计不确定性，并引入 Thompson sampling 等序贯采样策略以更少的 API 调用获得更窄的置信区间。

**[Efficient Pre-Training of LLMs via Topology-Aware Communication Alignment on More Than 9600 GPUs](nlp_generation/efficient_pre-training_of_llms_via_topology-aware_communication_alignment_on_mor.md)**

:   提出 Arnold 调度系统，通过将 LLM 训练的通信模式（DP/PP group）与数据中心物理网络拓扑对齐，在模拟中将通信组最大跨度减少 1.67x，在 9600+ GPU 生产级训练中端到端性能提升 10.6%。

**[How Does Sequence Modeling Architecture Influence Base Capabilities of Pre-trained Language Models?](nlp_generation/how_does_sequence_modeling_architecture_influence_base_capabilities_of_pre-train.md)**

:   通过"限定领域预训练 + OOD 测试"的评估框架揭示 Mamba/RWKV 等 stateful 架构存在基础能力退化，并归纳出关键设计原则——"全序列任意选择能力"（full-sequence visibility + real relation calculation + non-uniform distribution），用极简的 Top-1 Element/Chunk Selection 架构验证该原则可恢复至接近 Transformer 的基础能力。

**[KScope: A Framework for Characterizing the Knowledge Status of Language Models](nlp_generation/kscope_a_framework_for_characterizing_the_knowledge_status_of_language_models.md)**

:   提出LLM知识状态的五分类法（一致正确/冲突正确/缺失/冲突错误/一致错误）和KScope层次化统计检验框架，通过重复采样+多步假设检验精确刻画LLM对给定问题的知识模式结构，并系统研究上下文如何更新各状态，发现受约束的上下文摘要+增强可信度平均提升4.3%的知识更新成功率。

**[Learning to Solve Complex Problems via Dataset Decomposition](nlp_generation/learning_to_solve_complex_problems_via_dataset_decomposition.md)**

:   提出Decomp方法，利用教师模型将复杂数学题按推理步骤递归分解为更简单的子问题，构建概念依赖图量化难度，再按从易到难的课程顺序训练学生模型——Qwen2.5-1.5B在MATH-500上达51.6%（超MuggleMath用147K数据的50.4%），Qwen3-4B在AIME2025仅用385样本达16.7%（超Qwen2.5-72B的15%）。

**[MaintainCoder: Maintainable Code Generation Under Dynamic Requirements](nlp_generation/maintaincoder_maintainable_code_generation_under_dynamic_requirements.md)**

:   首次系统定义并解决 LLM 代码生成的**可维护性**问题，同时贡献基准和方法：MaintainBench 通过 4 种需求变化模式 + 动态指标评测代码在需求演化下的可维护性；MaintainCoder 将 Waterfall 模型、设计模式与 6 个专业化 Agent 结合，动态可维护性指标提升 60%+，且初始代码正确性也一并提高。

**[Precise Information Control in Long-Form Text Generation](nlp_generation/precise_information_control_in_long-form_text_generation.md)**

:   提出Precise Information Control (PIC)任务——要求LLM生成的长文严格基于给定声明集合（不遗漏不添加），构建PIC-Bench评测8个任务发现SOTA模型70%以上生成包含忠实性幻觉，通过弱监督偏好数据构建+DPO训练的PIC-LM将8B模型F1从69.1%提升至91.0%。

**[Program Synthesis via Test-Time Transduction](nlp_generation/program_synthesis_via_test-time_transduction.md)**

:   提出 SYNTRA 框架，将程序合成重新定义为转导式学习——在测试时利用可见的 test inputs 和 LLM 的判断来迭代消除不一致的候选程序假设，通过 greedy maximin 算法最小化 LLM 查询次数，在 4 个 benchmark 上准确率提升最高达 196%。

**[SWE-rebench: An Automated Pipeline for Task Collection and Decontaminated Evaluation of Software Engineering Agents](nlp_generation/swe-rebench_an_automated_pipeline_for_task_collection_and_decontaminated_evaluat.md)**

:   构建全自动化流水线从 GitHub 持续挖掘真实软件工程交互任务，生成 21,000+ 可执行 Python 任务的 SWE-rebench 数据集和去污染 benchmark，揭示部分模型在 SWE-bench Verified 上的性能存在污染膨胀问题（如 DeepSeek-V3 在 SWE-bench 上 39.7% vs SWE-rebench 上 21.3%）。

**[Time Travel is Cheating: Going Live with DeepFund for Real-Time Fund Investment Benchmarking](nlp_generation/time_travel_is_cheating_going_live_with_deepfund_for_real-time_fund_investment_b.md)**

:   提出 DeepFund——首个实时基金投资 benchmark 工具，通过多智能体架构（Financial Planner + Analyst Team + Portfolio Manager）连接实时股市数据，避免传统回测中 LLM "时间旅行"导致的信息泄露问题。在 24 个交易日的实盘测试中，9 个旗舰 LLM 只有 Grok 3 实现盈利，揭示了当前 LLM 在主动基金管理中的重大局限。

**[URLs Help, Topics Guide: Understanding Metadata Utility in LLM Training](nlp_generation/urls_help_topics_guide_understanding_metadata_utility_in_llm_training.md)**

:   系统评估了三类元数据（URL、质量分数、主题/格式域信息）作为预训练上下文的效果：发现只有 URL 能加速训练（100B token 用 60B 即达到相同下游性能），且仅在长 prompt（5-shot）下有效；质量分数和主题域信息不加速训练但可用于 classifier-free guidance 实现可控生成。

---

## 📖 NLP 理解 { #nlp_understanding }

**[AgentiQL: An Agent-Inspired Multi-Expert Framework for Text-to-SQL Generation](nlp_understanding/agentiql_an_agent-inspired_multi-expert_framework_for_text-to-sql_generation.md)**

:   提出 AgentiQL，一个多专家 agent 框架用于 Text-to-SQL：reasoning agent 分解问题为子问题，coding agent 生成子查询，refinement 步骤校正列选择，adaptive router 在基线解析器和模块化 pipeline 之间智能路由，使用 14B 开源模型达到 86.07% EX（Spider），接近 GPT-4 SOTA(89.65%)。

**[Creativity or Brute Force? Using Brainteasers as a Window into the Problem-Solving Abilities of Large Language Models](nlp_understanding/creativity_or_brute_force_using_brainteasers_as_a_window_into_the_problem-solvin.md)**

:   构建Braingle Brainteaser基准（242数学+236逻辑谜题），系统评估LLM在脑筋急转弯上的推理策略——发现模型有时能产生创造性洞察式解法，但也常在有巧妙解法可用时退回暴力穷举，且自纠错能力有限、将叙事→数学格式翻译可小幅提升性能。

**[Efficient Semantic Uncertainty Quantification in Language Models via Diversity-Steered Sampling](nlp_understanding/efficient_semantic_uncertainty_quantification_in_language_models_via_diversity-s.md)**

:   提出 diversity-steered sampling 框架：在解码时注入基于 NLI 的语义相似度惩罚来驱动生成语义多样化的样本，并用重要性加权+控制变量纠正偏差降低方差，在仅 16 个样本下即可准确估计 LLM 的语义熵（偶然不确定性）和互信息（认知不确定性）。

**[Generalization Error Analysis for Selective State-Space Models Through the Lens of Attention](nlp_understanding/generalization_error_analysis_for_selective_state-space_models_through_the_lens_.md)**

:   将选择性SSM（Mamba）展开为注意力形式，利用覆盖数技术推导出受连续时间状态矩阵谱横断面$s_{\mathbf{A}}$控制的泛化界——$s_{\mathbf{A}}<0$时泛化界与序列长度无关，$s_{\mathbf{A}}\geq0$时指数增长，并证明这种依赖不可消除。

**[How Data Mixing Shapes In-Context Learning: Asymptotic Equivalence for Transformers with MLPs](nlp_understanding/how_data_mixing_shapes_in-context_learning_asymptotic_equivalence_for_transforme.md)**

:   在高维渐近框架下证明了带非线性MLP头的Transformer在ICL误差上等价于结构化多项式预测器，揭示了非线性MLP对非线性任务的增益机制，以及多源数据混合中低噪声和结构化协方差是高质量数据源的关键特征。

**[Planning without Search: Refining Frontier LLMs with Offline Goal-Conditioned RL](nlp_understanding/planning_without_search_refining_frontier_llms_with_offline_goal-conditioned_rl.md)**

:   提出PNLC方法，通过训练轻量级目标条件价值函数作为"自然语言评论家"，在推理步骤层面引导LLM智能体进行多轮规划和自我精化，无需直接微调或推理时搜索，在Web导航、社交推理、劝服等复杂交互任务上显著超越现有方法且推理速度快8-10倍。

**[Retrieval-Augmented Generation for Reliable Interpretation of Radio Regulations](nlp_understanding/retrieval-augmented_generation_for_reliable_interpretation_of_radio_regulations.md)**

:   针对无线电法规这一法律敏感的高风险领域，设计了专用 RAG 管道并构建了首个 ITU 无线电法规多选题评估集，检索准确率达 97%，在 GPT-4o 上实现 +11.9% 的问答准确率提升，远超直接将文档塞入 prompt 的方式。

**[SeCon-RAG: A Two-Stage Semantic Filtering and Conflict-Free Framework for Trustworthy RAG](nlp_understanding/secon-rag_a_two-stage_semantic_filtering_and_conflict-free_framework_for_trustwo.md)**

:   提出 SeCon-RAG 两阶段防御框架，第一阶段用聚类+语义图联合过滤毒化文档，第二阶段在推理时做冲突感知过滤，在5个LLM和3个QA数据集上全面超越现有RAG防御方法，在100%投毒率下仍保持高准确率和极低攻击成功率。

**[Text-to-Code Generation for Modular Building Layouts in Building Information Modeling](nlp_understanding/text-to-code_generation_for_modular_building_layouts_in_building_information_mod.md)**

:   提出 Text2MBL 框架，将自然语言描述转化为可执行的 BIM 代码（而非坐标序列），通过面向对象的代码架构和 LLM 微调实现模块化建筑布局的自动生成，在几何一致性上比坐标驱动方法提升 10%+ IoU。

**[The Atlas of In-Context Learning: How Attention Heads Shape In-Context Retrieval Augmentation](nlp_understanding/the_atlas_of_in-context_learning_how_attention_heads_shape_in-context_retrieval_.md)**

:   通过 AttnLRP 归因方法系统解剖 LLM 在 in-context retrieval augmented QA 中的内部机制，发现三类功能特化的注意力头——Task heads（中间层，解析指令/问题）、Retrieval heads（后层，逐字复制上下文答案）、Parametric heads（编码参数化知识），并通过 Function Vector 注入和来源追踪探针验证其功能，在 Llama-3.1/Mistral/Gemma 上 ROC AUC ≥94%。

**[Weak-to-Strong Generalization under Distribution Shifts](nlp_understanding/weak-to-strong_generalization_under_distribution_shifts.md)**

:   发现朴素的弱到强泛化 (weak-to-strong generalization) 在分布偏移下会失败（强模型表现甚至不如弱监督者），提出 RAVEN 框架通过动态学习弱模型的最优组合权重来实现鲁棒的弱到强泛化，在 OOD 任务上超越基线 30%+。

---

## 🎁 推荐系统 { #recommender }

**[ASAP: An Agentic Solution to Auto-Optimize Performance of Large-Scale LLM Training](recommender/asap_an_agentic_solution_to_auto-optimize_performance_of_large-scale_llm_trainin.md)**

:   ASAP 是一个多 Agent 系统（Coordinator + Analyzer + Proposal），自动化诊断大规模 LLM 分布式训练的瓶颈类型（计算/内存/通信）并提出 sharding 配置方案，在 3 个实验场景中均匹配人类专家方案，实现最高 2.58× 吞吐量提升。

**[Balancing Performance and Costs in Best Arm Identification](recommender/balancing_performance_and_costs_in_best_arm_identification.md)**

:   提出将最优臂识别（BAI）从固定预算/固定置信度框架重新定义为"误识别概率/简单遗憾 + 采样成本"的风险泛函最小化问题，推导出含相变现象的下界（差距过小时最优策略是直接猜），设计 DBCARE 算法在动态预算下达到对数因子内最优。

**[EMPATHIA: Multi-Faceted Human-AI Collaboration for Refugee Integration](recommender/empathia_multi-faceted_human-ai_collaboration_for_refugee_integration.md)**

:   提出EMPATHIA多Agent框架，基于Kegan建构性发展理论，通过情感/文化/伦理三个专业化Agent的选择器-验证器协商评估难民安置建议，在6,359名难民的真实数据上达到87.4%收敛率和92.1%文化专家同意率。

**[Estimating Hitting Times Locally At Scale](recommender/estimating_hitting_times_locally_at_scale.md)**

:   提出两种局部（亚线性）算法估计图上的命中时间——基于相遇时间的 Algorithm 1 和基于谱截断的 Algorithm 3，无需全图访问仅通过以 $u,v$ 为中心的短随机游走完成估计，在合成和真实图上相对误差 <1.4%，并证明了游走采样的最优样本复杂度下界。

**[MMPB: It's Time for Multi-Modal Personalization](recommender/mmpb_its_time_for_multi-modal_personalization.md)**

:   提出首个 VLM 个性化评测基准 MMPB，包含 111 个可个性化概念、10k+ 图文问答对和 15 种任务类型，评测了 23 个 VLM 后发现即使最强的 GPT-4o 在个性化任务上也表现不佳，揭示了 VLM 在偏好推理、视觉线索利用和安全对齐与个性化的冲突等方面的重大局限。

**[Overcoming Sparsity Artifacts In Crosscoders To Interpret Chat-Tuning](recommender/overcoming_sparsity_artifacts_in_crosscoders_to_interpret_chat-tuning.md)**

:   识别Crosscoder L1训练中的稀疏性伪影导致虚假模型特定潜变量归因，提出BatchTopK损失+Latent Scaling揭示真正的chat特定概念。

**[Radial Neighborhood Smoothing Recommender System](recommender/radial_neighborhood_smoothing_recommender_system.md)**

:   提出 Radial Neighborhood Estimator (RNE)，通过将隐空间距离用观测矩阵的行/列 L2 范数近似估计，构建同时包含重叠和部分重叠用户-物品对的径向邻域，用局部核回归做平滑插补，在理论保证和实验中均优于传统协同过滤和矩阵分解方法，并天然缓解冷启动问题。

**[Think before Recommendation: Autonomous Reasoning-enhanced Recommender](recommender/think_before_recommendation_autonomous_reasoning-enhanced_recommender.md)**

:   提出 RecZero（纯 RL 范式）和 RecOne（SFT+RL 混合范式），抛弃传统的 teacher-student 蒸馏方法，用 GRPO 强化学习直接训练单个 LLM 自主发展推理能力进行评分预测，通过结构化 "Think-before-Recommendation" 模板引导分步推理（分析用户→分析物品→匹配→评分），在 4 个数据集上显著超越现有基线。

**[Transformer Copilot: Learning from The Mistake Log in LLM Fine-tuning](recommender/transformer_copilot_learning_from_the_mistake_log_in_llm_fine-tuning.md)**

:   提出 Transformer Copilot 框架，在 LLM 微调过程中系统记录"错误日志"(Mistake Log)，训练一个辅助 Copilot 模型学习 Pilot 的错误模式，推理时通过 logits 修正提升生成质量，在 12 个基准上最高提升 34.5%。

**[Who You Are Matters: Bridging Topics and Social Roles via LLM-Enhanced Logical Recommendation](recommender/who_you_are_matters_bridging_topics_and_social_roles_via_llm-enhanced_logical_re.md)**

:   提出 TagCF 框架，通过 MLLM 提取用户角色标签和物品话题标签，再用 LLM 推理构建 U2I/I2U 逻辑图（用户角色与物品类型的因果关联），辅以标签编码器、对比学习增强和逻辑推理评分三种集成策略增强推荐，在亿级用户的工业在线A/B测试中互动指标提升0.946%、多样性提升0.102%，离线实验NDCG@10提升8.06%。

**[Wide-Horizon Thinking and Simulation-Based Evaluation for Real-World LLM Planning with Multifaceted Constraints](recommender/wide-horizon_thinking_and_simulation-based_evaluation_for_real-world_llm_plannin.md)**

:   提出 MAoP（Multiple Aspects of Planning）框架赋予 LLM "宽视野思维"能力，通过策略师预规划与路由机制并行整合多方面约束，配合 Travel-Sim 因果模拟评估基准，在旅行规划任务上大幅超越 CoT/分解方法，蒸馏后 3B 模型 PER 达 66.9%。

---

## 🧮 科学计算 { #scientific_computing }

**[Bayesian Surrogates for Risk-Aware Pre-Assessment of Aging Bridge Portfolios](scientific_computing/bayesian_surrogates_for_risk-aware_pre-assessment_of_aging_bridge_portfolios.md)**

:   提出基于贝叶斯神经网络（BNN）的代理模型，用于替代昂贵的非线性有限元分析（NLFEA），实现对老化桥梁组合的快速、不确定性感知的结构安全预评估，在真实铁路案例中为单座桥梁节省约37万美元。

**[Collapsing Taylor Mode Automatic Differentiation](scientific_computing/collapsing_taylor_mode_automatic_differentiation.md)**

:   提出 Taylor mode 自动微分的"折叠"(collapsing)优化技术，通过重写计算图将导数求和操作向上传播，大幅加速 PDE 算子（如 Laplacian、一般线性 PDE 算子）的计算，实现速度优于嵌套反向传播同时保持前向模式的低内存优势。

**[DeltaPhi: Physical States Residual Learning for Neural Operators in Data-Limited PDE Solving](scientific_computing/deltaphi_physical_states_residual_learning_for_neural_operators_in_data-limited_.md)**

:   提出 DeltaPhi 框架：不直接学习 PDE 的输入→输出映射，而是学习**相似物理状态之间的残差**，利用物理系统稳定性实现隐式数据增强，在数据稀缺场景下显著提升各类神经算子的性能。

**[Eddyformer Accelerated Neural Simulations Of Three-Dimensional Turbulence At Sca](scientific_computing/eddyformer_accelerated_neural_simulations_of_three-dimensional_turbulence_at_sca.md)**

:   提出 EddyFormer，一种基于谱元法 (SEM) 的 Transformer 架构，将流场分解为 LES（大尺度）和 SGS（小尺度）两路并行流，在 256³ 分辨率 3D 湍流上达到 DNS 级精度且加速 30 倍，并在未见的 4× 更大域上泛化良好。

**[Enforcing Governing Equation Constraints in Neural PDE Solvers via Training-free Projections](scientific_computing/enforcing_governing_equation_constraints_in_neural_pde_solvers_via_training-free.md)**

:   提出两种无需训练的后处理投影方法（非线性LBFGS优化和局部线性化投影），将神经PDE求解器的输出投影到满足控制方程约束的可行流形上，在Lorenz/KS/Navier-Stokes上大幅降低约束违反并提升精度，且效果显著优于physics-informed训练。

**[GyroSwin: 5D Surrogates for Gyrokinetic Plasma Turbulence Simulations](scientific_computing/gyroswin_5d_surrogates_for_gyrokinetic_plasma_turbulence_simulations.md)**

:   首次提出可扩展的5D神经网络代理模型 GyroSwin，将 Swin Transformer 扩展至5维回旋动力学相空间，通过交叉注意力实现3D↔5D交互、通道式模态分离捕获带状流，在等离子体湍流模拟中实现比传统准线性方法更高的精度，且比数值求解器（GKW）快3个数量级。

**[Inc An Indirect Neural Corrector For Auto-Regressive Hybrid Pde Solvers](scientific_computing/inc_an_indirect_neural_corrector_for_auto-regressive_hybrid_pde_solvers.md)**

:   提出间接神经校正器(INC)，将学习到的校正项嵌入PDE的右端项（而非直接修改状态），理论证明误差放大降低$\mathcal{O}(\Delta t^{-1}+L)$倍，在6个PDE系统（1D混沌到3D湍流）上大幅改善长期轨迹性能（R²提升达158.7%），实现最高330×加速。

**[Integration Matters for Learning PDEs with Backward SDEs](scientific_computing/integration_matters_for_learning_pdes_with_backward_sdes.md)**

:   揭示了标准 BSDE 方法性能不如 PINNs 的根本原因是 Euler-Maruyama 积分引入的不可消除离散化偏差，提出基于 Stratonovich 形式的 Heun-BSDE 方法彻底消除该偏差，在高维 PDE 上与 PINNs 竞争。

**[Neural Emulator Superiority: When Machine Learning for PDEs Surpasses its Training Data](scientific_computing/neural_emulator_superiority_when_machine_learning_for_pdes_surpasses_its_trainin.md)**

:   挑战了"神经 PDE 模拟器精度受限于训练数据（数值求解器）精度"的传统认知，发现并严格定义了 **emulator superiority** 现象——仅在低精度求解器数据上训练的神经网络，在以高精度参考解评估时竟能超越其训练求解器本身。

**[Neuro-Spectral Architectures for Causal Physics-Informed Networks](scientific_computing/neuro-spectral_architectures_for_causal_physics-informed_networks.md)**

:   NeuSA 将经典谱方法与 Neural ODE 结合，先将 PDE 投影到谱基（Fourier）上得到 ODE 系统，再用 NODE 学习动力学演化，从架构层面解决了传统 PINN 的谱偏差和因果性问题，在波动方程/Burgers方程/sine-Gordon方程上误差比 baseline 低 1-2 个数量级且训练更快。

**[Towards Universal Neural Operators Through Multiphysics Pretraining](scientific_computing/towards_universal_neural_operators_through_multiphysics_pretraining.md)**

:   研究基于 Transformer/SSM 的神经算子在多物理场 PDE 上的迁移学习能力，通过 adapter-based 架构（lifting/projection 作为问题特定适配器，算子层共享）实现跨 PDE 类型的预训练和微调，在多个场景中显著提升模型质量并减少微调成本。

---

## 🛰️ 遥感 { #remote_sensing }

**[C3PO: Cross-View Cross-Modality Correspondence by Pointmap Prediction](remote_sensing/c3po_cross-view_cross-modality_correspondence_by_pointmap_prediction.md)**

:   构建了包含 90K 地面照片-平面图对（597 个场景、153M 像素级对应和 85K 相机位姿）的 C3 数据集，揭示现有对应模型在跨视角跨模态（如地面照片 vs. 平面图）场景下的局限性，通过在该数据上训练可将最佳方法的 RMSE 降低 34%。

**[ChA-MAEViT: Unifying Channel-Aware Masked Autoencoders and Multi-Channel Vision Transformers for Improved Cross-Channel Learning](remote_sensing/chamaevit_unifying_channelaware_masked_autoencoders_and_mult.md)**

:   提出 ChA-MAEViT，通过动态通道-patch 联合掩码、记忆 token、混合 token 融合和通道感知解码器四个策略增强多通道成像（MCI）中的跨通道交互学习，在卫星和显微镜数据集上超越 SOTA MCI-ViT 方法 3.0-21.5%。

**[GeoLink: Empowering Remote Sensing FM with OpenStreetMap Data](remote_sensing/geolink_empowering_remote_sensing_foundation_model_with_openstreetmap_data.md)**

:   提出GeoLink，首个将OSM矢量数据直接融入遥感基础模型的框架，通过异构GNN编码OSM数据+多粒度跨模态对比/一致性学习+掩码高效预训练，在127万样本对上预训练后显著提升遥感和地理任务。

**[GreenHyperSpectra: Multi-Source Hyperspectral Dataset for Vegetation Traits](remote_sensing/greenhyperspectra_a_multi-source_hyperspectral_dataset_for_global_vegetation_tra.md)**

:   构建GreenHyperSpectra——14万+样本的多源高光谱植被数据集（跨地面/机载/星载传感器），框架化半/自监督方法用于多输出植被性状回归，在标签稀缺场景下显著超越监督基线。

**[Mass Conservation on Rails: PIML of Ice Flow Vector Fields](remote_sensing/mass_conservation_on_rails_--_rethinking_physics-informed_learning_of_ice_flow_v.md)**

:   对比硬约束（dfNN）、软约束（PINN）和无约束NN在南极冰流通量插值上的表现，通过流函数的辛梯度精确保证无散度的dfNN最优，结合方向引导进一步提升。

**[OrbitZoo: Real Orbital Systems Challenges for RL](remote_sensing/orbitzoo_real_orbital_systems_challenges_for_reinforcement_learning.md)**

:   构建OrbitZoo，基于工业标准库Orekit的多智能体RL环境，支持碰撞规避和协同机动，经Starlink真实数据验证MAPE仅0.16%。

**[OrthoLoC: UAV 6-DoF Localization Using Orthographic Geodata](remote_sensing/ortholoc_uav_6-dof_localization_and_calibration_using_orthographic_geodata.md)**

:   提出OrthoLoC——首个大规模UAV-正射影像配对数据集（16,425张，47地点，19城市），用于6-DoF定位和标定评估，AdHoP技术匹配精度提升95%、平移误差降低63%。

**[RSCC: Large-Scale Remote Sensing Change Caption Dataset for Disasters](remote_sensing/rscc_a_large-scale_remote_sensing_change_caption_dataset_for_disaster_events.md)**

:   提出RSCC数据集——62,351对灾前/灾后遥感图像配以丰富变化描述文本，覆盖地震/洪水/野火等，填补灾害相关双时相图像-文本缺口。

**[Scaling Image Geo-Localization to Continent Level](remote_sensing/scaling_image_geo-localization_to_continent_level.md)**

:   混合方法结合分类学习的原型和航拍图像嵌入，在覆盖西欧43.3万平方公里上实现200m内68%+、100m内59.2%的定位率，首次在大陆规模实现此精度。

---

## 🔎 AIGC 检测 { #aigc_detection }

**[ASCIIBench: Evaluating Language-Model-Based Understanding of Visually-Oriented Text](aigc_detection/asciibench_evaluating_language-model-based_understanding_of_visually-oriented_te.md)**

:   提出 ASCIIBench，首个用于评估 LLM 对 ASCII 艺术的生成和分类能力的基准数据集（5,315 张 ASCII 图像，752 类），发现当前 LLM 在需要空间/位置推理的 ASCII 任务上仍有显著局限，且 CLIP 嵌入在大多数 ASCII 类别上的区分能力接近随机水平。

**[Classical Planning with LLM-Generated Heuristics: Challenging the State of the Art with Python Code](aigc_detection/classical_planning_with_llm-generated_heuristics_challenging_the_state_of_the_ar.md)**

:   让 LLM 为经典规划问题生成 Python 启发式函数代码，从 n 个候选中选最优，在 IPC 2023 基准上用纯 Python 规划器超越了 C++ 实现的 SOTA 启发式（如 hFF），且保证所有计划正确。

**[DuoLens: A Framework for Robust Detection of Machine-Generated Multilingual Text and Code](aigc_detection/duolens_a_framework_for_robust_detection_of_machine-generated_multilingual_text_.md)**

:   提出 DuoLens，一种基于 CodeBERT + CodeBERTa 双编码器融合的 AI 生成内容检测框架，在多语言文本（8 种语言）和源代码（7 种编程语言）检测上以极低计算成本（延迟降低 8-12×，VRAM 降低 3-5×）实现 AUROC 0.97-0.99，远超 GPT-4o 等大模型。

**["Jutters"](aigc_detection/jutters.md)**

:   通过荷兰传统"jutters"（海岸拾荒者）的隐喻，构建了一个融合真实海滩碎片与AI生成图像/视频的沉浸式装置艺术，引导参观者以拾荒者心态反思如何对待AI生成内容。

**[Reasoning Compiler: LLM-Guided Optimizations for Efficient Model Serving](aigc_detection/reasoning_compiler_llm-guided_optimizations_for_efficient_model_serving.md)**

:   提出 Reasoning Compiler，将编译器优化建模为序列决策过程，用 LLM 作为上下文感知提案引擎 + MCTS 平衡探索/利用，在 5 个代表性 benchmark 和 5 个硬件平台上实现平均 5.0× 加速且采样效率比 TVM 进化搜索提升 10.8×。

**[Synthesizing Performance Constraints for Evaluating and Improving Code Efficiency](aigc_detection/synthesizing_performance_constraints_for_evaluating_and_improving_code_efficienc.md)**

:   提出Wedge框架——通过LLM合成性能刻画约束（performance-characterizing constraints）指导约束感知模糊测试，生成能暴露代码性能瓶颈的压力测试输入，构建PerfForge基准，使LLM代码优化器（如Effi-Learner）多减24% CPU指令。

---

## 🌍 地球科学 { #earth_science }

**[A Probabilistic U-Net Approach to Downscaling Climate Simulations](earth_science/a_probabilistic_unet_approach_to_downscaling_climate_simulat.md)**

:   将医学图像分割中的概率U-Net迁移到气候降尺度任务，通过变分隐空间建模不确定性，并系统比较了四种训练目标函数在捕捉极端事件与细尺度空间变异性方面的权衡。

**[Adaptive Online Emulation for Accelerating Complex Physical Simulations](earth_science/adaptive_online_emulation_for_accelerating_complex_physical_simulations.md)**

:   提出 Adaptive Online Emulation (AOE)，在物理模拟执行过程中动态训练 ELM 神经网络代理模型替代昂贵计算组件，无需离线预训练，在系外行星大气模拟上实现 11.1× 加速（91% 时间节省）且精度损失仅 ~0.01%。

**[ControlFusion: A Controllable Image Fusion Framework with Language-Vision Degradation Prompts](earth_science/controlfusion_a_controllable_image_fusion_framework_with_language-vision_degrada.md)**

:   提出 ControlFusion，一种基于语言-视觉退化提示的可控红外-可见光图像融合框架，通过物理驱动的退化成像模型模拟复合退化，并用 prompt-modulated 网络动态恢复+融合，在真实世界和复合退化场景下全面超越 SOTA。

**[Predicting Public Health Impacts of Electricity Usage](earth_science/predicting_public_health_impacts_of_electricity_usage.md)**

:   提出 HealthPredictor，一个将电力消费端到端映射到公共健康损害（以 $/MWh 计量）的 AI 流水线，包含燃料组合预测、空气质量转换和健康影响评估三个模块，健康驱动优化比燃料组合驱动基线显著降低健康影响预测误差，并在电动汽车充电调度案例中实现 24-42% 的健康损害减少。

**[Reasoning With a Star: A Heliophysics Dataset and Benchmark for Agentic Scientific Reasoning](earth_science/reasoning_with_a_star_a_heliophysics_dataset_and_benchmark_for_agentic_scientifi.md)**

:   提出 Reasoning With a Star (RWS)，一个源自 NASA 太阳物理暑期学校问题集的 158 道科学推理 benchmark（含数值/符号/文本三类答案），配合 unit-aware 评分器，比较了四种多 agent 协调模式（HMAW/PACE/PHASE/SCHEMA），发现没有单一模式在所有任务上占优——系统工程启发的 SCHEMA 在需要严格约束验证的任务上最强。

---

## 📂 其他 { #others }

**[4DGT: Learning a 4D Gaussian Transformer Using Real-World Monocular Videos](others/4dgt_learning_a_4d_gaussian_transformer_using_realworld_mono.md)**

:   提出4DGT——一种基于4D高斯的Transformer模型，完全在真实世界单目带位姿视频上训练，以前馈方式在几秒内完成动态场景重建，显著优于同类前馈网络，并达到与优化类方法可比的精度。

**[A Cramér–von Mises Approach to Incentivizing Truthful Data Sharing](others/a_cramrvon_mises_approach_to_incentivizing_truthful_data_sha.md)**

:   提出一种基于 Cramér-von Mises 两样本检验统计量的激励机制，在贝叶斯和无先验两种设定下均能证明"如实提交数据"构成（近似）Nash 均衡，同时鼓励参与者提交更多真实数据，且不依赖对数据分布的强假设（如高斯、伯努利）。

**[A Differentiable Model Of Supply-Chain Shocks](others/a_differentiable_model_of_supply-chain_shocks.md)**

:   用 JAX 实现可微分的供应链 Agent-Based Model（~1000 家企业），通过 GPU 并行化 + 自动微分实现比传统 ABC 快 3 个数量级的贝叶斯参数校准，为全球供应链网络的冲击传播建模铺平道路。

**[A Generalized Label Shift Perspective for Cross-Domain Gaze Estimation](others/a_generalized_label_shift_perspective_for_crossdomain_gaze_e.md)**

:   本文将跨域视线估计(CDGE)问题建模为广义标签偏移(GLS)问题，指出现有域不变表示学习方法在标签偏移存在时理论上不充分，提出基于截断高斯分布的连续重要性重加权和概率感知条件算子差异(PCOD)来联合纠正标签偏移和条件偏移，在多个backbone上平均降低误差12%~27%。

**[A High-Dimensional Statistical Method for Optimizing Transfer Quantities in Multi-Source Transfer Learning](others/a_highdimensional_statistical_method_for_optimizing_transfer.md)**

:   提出基于K-L散度和高维统计分析的理论框架，用于确定多源迁移学习中每个源任务的最优样本迁移数量，避免"用所有源数据"带来的负迁移问题，在DomainNet和Office-Home上超过SOTA 1.0-1.5%的同时减少47.85%的样本使用量和35.19%的训练时间。

**[A Reliable Cryptographic Framework for Empirical Machine Unlearning Evaluation](others/a_reliable_cryptographic_framework_for_empirical_machine_unl.md)**

:   将机器遗忘的评估问题建模为密码学博弈（unlearning sample inference game），通过定义adversary的"advantage"来衡量遗忘质量，克服了传统MIA准确率作为评估指标的多种缺陷（不以retrain为零基准、对数据划分敏感、对MIA选择敏感），并提出SWAP test作为高效的实用近似方案。

**[A Standardized Benchmark for Multilabel Antimicrobial Peptide Classification](others/a_standardized_benchmark_for_multilabel_antimicrobial_peptide_classification.md)**

:   提出 **ESCAPE**——首个标准化的多标签抗菌肽分类基准，整合 27 个公开数据库共 80,000+ 肽段，并设计基于双分支 Transformer + 双向交叉注意力的 Baseline 模型，在 mAP 上相对第二名提升 2.56%。

**[A Sustainable AI Economy Needs Data Deals That Work for Generators](others/a_sustainable_ai_economy_needs_data_deals_that_work_for_gene.md)**

:   本文通过分析73个公开数据交易案例，揭示了ML价值链中的"经济数据处理不等式"——从原始数据到模型权重再到合成输出，每一步都提炼了技术信号但剥夺了数据生成者的经济权益，并提出EDVEX框架来构建更公平的数据交换市场。

**[A Theoretical Framework for Grokking: Interpolation followed by Riemannian Norm Minimisation](others/a_theoretical_framework_for_grokking_interpolation_followed_by_riemannian_norm_m.md)**

:   本文从纯优化角度严格证明了 grokking 现象的成因：带小 weight decay 的梯度流在 $\lambda\to 0$ 极限下呈现两阶段动力学——先快速收敛到训练损失的临界流形 $\mathcal{M}$，再在 $t\approx 1/\lambda$ 时沿流形做黎曼梯度流以最小化 $\ell_2$ 范数，从而延迟实现泛化。

**[A Unified Framework for Provably Efficient Algorithms to Estimate Shapley Values](others/a_unified_framework_for_provably_efficient_algorithms_to_estimate_shapley_values.md)**

:   提出统一框架将 KernelSHAP、LeverageSHAP 等 Shapley 值估计器纳入随机草图（sketching）视角，首次为 KernelSHAP 提供非渐近理论保证，并通过算法改进（Poisson 近似等）将方法扩展到 CIFAR-10 等高维数据集。

**[A Unified Framework for Variable Selection in Model-Based Clustering with Missing Not at Random](others/a_unified_framework_for_variable_selection_in_modelbased_clu.md)**

:   提出了一个统一框架（SelvarMNARz），在高斯混合模型聚类中同时完成变量选择和MNAR（Missing Not At Random）缺失数据建模，通过两阶段策略（LASSO排序 + BIC角色分配）实现高维场景下的高效推理，并给出了可辨识性和选择一致性的理论保证。

**[Active Measurement: Efficient Estimation at Scale](others/active_measurement_efficient_estimation_at_scale.md)**

:   提出Active Measurement框架，结合AI检测器的自适应重要性采样和迭代人工标注，实现大规模科学测量（如鸟类计数、疟疾检测）的无偏估计，将原始检测器3.78的误差率降至0.06，同时提供理论保证的置信区间。

**[AcuRank: 不确定性感知的自适应计算重排序](others/acurank_uncertainty-aware_adaptive_computation_for_listwise_reranking.md)**

:   通过基于TrueSkill模型的不确定性估计，动态调整重排序子集大小和验证范围，在实现更优精度效率权衡的同时避免过度计算。

**[AdaptGrad: Adaptive Sampling to Reduce Noise](others/adaptgrad_adaptive_sampling_to_reduce_noise.md)**

:   通过卷积公式视角首次理论分析了SmoothGrad的噪声来源（越界采样），提出AdaptGrad方法通过概率界约束采样范围来抑制噪声，在不增加计算开销的前提下提升梯度显著性图的质量。

**[Adaptive Data Analysis for Growing Data](others/adaptive_data_analysis_for_growing_data.md)**

:   首次为动态/增长数据场景下的自适应数据分析提供泛化界，允许分析者根据当前数据规模和历史查询结果自适应地调度统计查询，在数据不断积累时获得更紧的准确性保证。

**[Additive Models Explained: A Computational Complexity Approach](others/additive_models_explained_a_computational_complexity_approach.md)**

:   对广义可加模型（GAM）的多种解释类型（充分理由、对比解释、Shapley值等）进行系统的计算复杂度分析，揭示了GAM的可解释性代价高度依赖于输入域类型、组件模型类型和任务类型（回归vs分类），某些看似"可解释"的设定实际上是NP-Hard甚至#P-Hard。

**[Addressing Mark Imbalance In Integrationfree Neural Marked T](others/addressing_mark_imbalance_in_integrationfree_neural_marked_t.md)**

:   论文针对现实事件流中常见的 mark 类别长尾失衡问题，提出基于先验归一化概率的阈值学习策略，并设计 integration-free 的神经 MTPP 架构，先预测 mark 再预测 time，在避免昂贵数值积分的同时显著提升稀有事件的 mark 与到达时间预测性能。

**[Adjoint Schrödinger Bridge Sampler](others/adjoint_schrödinger_bridge_sampler.md)**

:   提出 Adjoint Schrödinger Bridge Sampler (ASBS)，通过将 Schrödinger Bridge 问题重新解释为随机最优控制问题，消除了先前扩散采样器的 memoryless 条件限制，支持任意源分布（如高斯、谐波先验），使用可扩展的 matching 目标无需重要性权重估计，在多粒子能量函数和分子构象生成上全面超越先前方法。

**[ADPretrain: Advancing Industrial Anomaly Detection via Anomaly Representation Pretraining](others/adpretrain_advancing_industrial_anomaly_detection_via_anomaly_representation_pre.md)**

:   首次提出面向工业异常检测的专用表示预训练框架 ADPretrain，通过角度和范数导向的对比损失在大规模异常检测数据集 RealIAD 上学习残差特征表示，替换五种主流嵌入式 AD 方法的原始特征后在五个数据集、五个骨干网络上取得一致性提升。

**[Aggregation Hides OOD Generalization Failures from Spurious Correlations](others/aggregation_hides_out-of-distribution_generalization_failures_from_spurious_corr.md)**

:   揭示 OOD 泛化 benchmark 中"聚合掩蔽"现象——aggregate 评估显示 accuracy-on-the-line（ID 与 OOD 准确率正相关），但 OODSelect 方法可从同一 OOD 数据中找到大规模语义连贯子集（最高达 75%），这些子集上 ID 越高 OOD 反而越低（Pearson R 低至 -0.92），证明虚假相关的危害被聚合评估系统性隐藏。

**[Alias-Free ViT: Fractional Shift Invariance via Linear Attention](others/alias-free_vit_fractional_shift_invariance_via_linear_attention.md)**

:   提出Alias-Free ViT，通过两个关键组件实现Vision Transformer对整数和亚像素平移的鲁棒性：(1) 抗混叠下采样和非线性层设计，(2) 基于交叉协方差的线性注意力（shift-equivariant），在图像分类中保持竞争力的同时显著提升对抗性平移鲁棒性。

**[Alternating Gradient Flows: A Theory of Feature Learning in Two-layer Neural Networks](others/alternating_gradient_flows_a_theory_of_feature_learning_in_two-layer_neural_netw.md)**

:   提出交替梯度流（AGF）理论框架解释神经网络的逐步"鞍到鞍"特征学习动力学——将训练建模为休眠神经元的效用最大化和活跃神经元的代价最小化的交替过程，统一了对角线性网络、注意力模型和模块加法的特征选择分析，预测与实际梯度流高度一致。

**[An Analysis of Concept Bottleneck Models: Measuring, Understanding, and Mitigating Noisy Annotations](others/an_analysis_of_concept_bottleneck_models_measuring_understanding_and_mitigating_.md)**

:   首次系统研究噪声概念标注对 CBM 的影响——发现即使中等噪声也同时损害预测性能、可解释性和干预效果，识别出"脆弱概念"子集是性能下降的主因，提出训练阶段用 SAM 稳定脆弱概念学习 + 推断阶段用预测熵排序仅校正最不确定概念的两阶段缓解框架。

**[An Empirical Investigation of Neural ODEs and Symbolic Regression for Dynamical Systems](others/an_empirical_investigation_of_neural_odes_and_symbolic_regression_for_dynamical_.md)**

:   系统实证研究 Neural ODE 和符号回归（SR）在动力系统建模中的组合使用：NODE 可以在动态相似条件下外推到新边界条件，SR 可以从有噪声数据中恢复控制方程，且用 NODE 训练数据（仅 10% 原始数据）生成的数据也能让 SR 恢复大部分方程。

**[EPHAD: An Evidence-Based Post-Hoc Adjustment Framework for Anomaly Detection Under Data Contamination](others/an_evidence-based_post-hoc_adjustment_framework_for_anomaly_detection_under_data.md)**

:   EPHAD 提出一种测试时后处理框架来修正在被污染数据上训练的异常检测模型——在不接触训练流程/数据的前提下，用多模态基础模型（CLIP）或经典方法（LOF）等"证据"在测试时调整模型输出，在 8 个视觉+26 个表格 AD 数据集上有效提升性能。

**[Are Pixel-Wise Metrics Reliable For Sparse-View Computed Tomography Reconstructi](others/are_pixel-wise_metrics_reliable_for_sparse-view_computed_tomography_reconstructi.md)**

:   揭示 PSNR/SSIM 等像素级指标无法反映稀疏视图 CT 重建中解剖结构完整性（相关性仅 0.16-0.30），提出基于自动分割的解剖感知指标（NSD/clDice）和 CARE 框架——在扩散模型训练中加入分割引导损失，大器官结构完整性提升 32%、血管提升 36%。

**[Asymmetric Duos: Sidekicks Improve Uncertainty](others/asymmetric_duos_sidekicks_improve_uncertainty.md)**

:   Asymmetric Duos（AD）将一个大模型与一个小"sidekick"配对——通过温度加权的 logit 平均融合两者预测，在仅增加 10-20% FLOPs 的条件下达到接近 5× 深度集成的不确定性估计质量，RN50 AD（5% FLOPs 额外开销）在 AUROC/AURC/SAC@98 上接近 m=5 深度集成（400% 额外 FLOPs）。

**[AutoSciDACT: Automated Scientific Discovery through Contrastive Embedding and Hypothesis Testing](others/autoscidact_automated_scientific_discovery_through_contrastive_embedding_and_hyp.md)**

:   提出 AutoSciDACT 管线：先用有监督对比学习将高维科学数据压缩到 4 维嵌入空间，再用 NPLM（New Physics Learning Machine）似然比检验对嵌入空间中的分布偏差进行统计量化，在天文、粒子物理、病理、图像和合成数据集上以 ≤1% 的信号注入比例实现 ≥3σ 发现。

**[AVerImaTeC: A Dataset for Automatic Verification of Image-Text Claims with Evidence from the Web](others/averimatec_a_dataset_for_automatic_verification_of_image-text_claims_with_eviden.md)**

:   AVerImaTeC 构建了首个带完整证据标注的图文事实核查数据集——1297 条真实图文声明 + 5 阶段标注流水线（提取→QA 推理→充分性检查→迭代精炼→二次检查）+ 时间约束证据（防止时间泄露），基线系统在有 ground truth 证据时准确率 82%，但自动检索证据后降至 15-25%，揭示了图文核查的巨大挑战。

**[Beyond Benign Overfitting in Nadaraya-Watson Interpolators](others/beyond_benign_overfitting_in_nadaraya-watson_interpolators.md)**

:   通过调节 Nadaraya-Watson 插值器中的单一带宽参数 $\beta$，精确刻画了从灾难性过拟合（$\beta < d$）→ 良性过拟合（$\beta = d$）→ 温和过拟合（$\beta > d$）的完整相变谱，证明高估数据内禀维度比低估更安全。

**[Depth-Supervised Fusion Network For Seamless-Free Image Stitching](others/depth-supervised_fusion_network_for_seamless-free_image_stitching.md)**

:   利用深度监督进行多视角对齐和软缝合融合，解决大视差场景下的图像拼接问题，在大视差对齐和无缝融合上超越现有SOTA。

**[Exact Learning of Arithmetic with Differentiable Agents](others/exact_learning_of_arithmetic_with_differentiable_agents.md)**

:   提出可微有限状态转换器（DFST），一种图灵完备且端到端可微的模型族，在 2D 符号网格上通过观察专家算术计算的中间步骤（Policy-Trajectory Observations）训练，仅用 20 个样本（最长 3 位数加法）即可完美泛化到 3850 位二进制加法、2450 位十进制加法，未发现任何错误。

**[FlowMoE: 分布式MoE训练的可扩展流水线调度框架](others/flowmoe_a_scalable_pipeline_scheduling_framework_for_distributed_mixture-of-expe.md)**

:   通过统一的流水线调度和优先级驱动的all-reduce张量分块，实现MHA、门控、专家计算和A2A/all-reduce通信的完全重叠，训练时间减少13-57%。

**[Generalized Linear Mode Connectivity For Transformers](others/generalized_linear_mode_connectivity_for_transformers.md)**

:   提出统一框架捕捉Transformer中的置换、半置换、正交和可逆对称性，首次在ViT和GPT-2上实现线性模式连接性(LMC)，支持多模型融合。

**[Gradient-Weight Alignment As A Train-Time Proxy For Generalization In Classifica](others/gradient-weight_alignment_as_a_train-time_proxy_for_generalization_in_classifica.md)**

:   提出梯度-权重对齐(GWA)作为训练时的泛化代理指标，无需验证集即可准确确定早停时机和模型比较。

**[Graph Alignment Via Birkhoff Relaxation](others/graph_alignment_via_birkhoff_relaxation.md)**

:   分析图对齐的二次分配问题的凸松弛（Birkhoff松弛），证明在 $\sigma = o(n^{-1})$ 时可恢复~100%的对齐，超越了先前的simplex松弛结果。

**[笔记2：PRM必要吗？RL隐式诱导PRM能力](others/is_prm_necessary_problem-solving_rl_implicitly_induces_prm_capability_in_llms.md)**

:   令人惊讶地，纯RL训练无需显式PRM监督即可诱发出强大的过程理解能力，且现有PRMs在SOTA模型上甚至不如简单多数投票有效。

**[Learning Generalizable Shape Completion with SIM(3) Equivariance](others/learning_generalizable_shape_completion_with_sim3_equivariance.md)**

:   提出首个 SIM(3) 等变形状补全网络 SIMECO，通过特征规范化→相似不变几何推理→变换恢复的三阶段模块设计，在去偏评估协议下超越所有增广和等变基线，KITTI 上 MMD 降低 17%、OmniObject3D 上 CD-$\ell_1$ 降低 14%，且在更严格协议下仍优于竞争者在其偏向性设置下的表现。

**[Look-Ahead Reasoning on Learning Platforms](others/look-ahead_reasoning_on_learning_platforms.md)**

:   在学习平台的用户-算法交互中形式化 level-$k$ 前瞻推理，证明个体自私的高阶推理只加速收敛但不改变均衡（无长期收益），而集体协调的收益由学习者-用户效用函数的对齐程度决定，提供了刻画协调收益上界的理论框架。

**[MAS-ZERO: Designing Multi-Agent Systems with Zero Supervision](others/maszero_designing_multiagent_systems_with_zero_supervision.md)**

:   MAS-ZERO 是首个推理时自动 MAS 设计框架，通过 meta-agent 迭代设计、批评和改进 MAS 配置（包括任务分解和 sub-MAS 分配），无需验证集和训练，在推理（+16.69%）、编程（+16.66%）和搜索代理（+5.45%）任务上均超越手动和自动 MAS baseline，同时保持 Pareto 最优的准确率-成本权衡。

**[Model Context Protocol for Vision Systems: Audit, Security, and Protocol Extensions](others/model_context_protocol_for_vision_systems_audit_security_and_protocol_extensions.md)**

:   首次大规模审计 MCP 在 91 个视觉系统中的部署，揭示 78% 存在 schema 偏差、89% 缺乏运行时验证等协议级脆弱性，提出扩展方案增强编排可靠性。

**[MoESD: 揭示稀疏MoE推理中投机解码的潜力](others/moesd_unveil_speculative_decodings_potential_for_accelerating_sparse_moe.md)**

:   揭示投机解码在中等批大小下对MoE比对稠密模型更有效，通过目标效率指标捕捉系统级瓶颈，建立可靠的性能建模，达到2.29×加速。

**[OrbitZoo: Real Orbital Systems Challenges for RL](others/orbitzoo_real_orbital_systems_challenges_for_reinforcement_learning.md)**

:   构建OrbitZoo，基于工业标准库Orekit的多智能体RL环境，支持碰撞规避和协同机动，经Starlink真实数据验证MAPE仅0.16%。

**[OrthoLoC: UAV 6-DoF Localization Using Orthographic Geodata](others/ortholoc_uav_6-dof_localization_and_calibration_using_orthographic_geodata.md)**

:   提出OrthoLoC——首个大规模UAV-正射影像配对数据集（16,425张，47地点，19城市），用于6-DoF定位和标定评估，AdHoP技术匹配精度提升95%、平移误差降低63%。

**[笔记7：价值引导搜索 - 高效链式思考推理](others/polymath_evaluating_mathematical_reasoning_in_multilingual_contexts.md)**

:   提出Value-Guided Search(VGS)——通过token级价值模型指导块级束搜索，无需预定义"步骤"，相对多数投票在竞赛数学上准确度提升+14.5%，同时推理计算效率提升30%，超越现有PRM方案。

**[笔记5：ReSearch - 学习通过搜索推理](others/research_learning_to_reason_with_search_for_llms_via_reinforcement_learning.md)**

:   ReSearch框架将搜索操作嵌入推理链中作为第一类原语，通过GRPO强化学习自动学习何时何如搜索，无需任何推理步骤的监督标注，在多跳QA任务上相对基线平均提升15.81%。

**[Training The Untrainable Introducing Inductive Bias Via Representational Alignme](others/training_the_untrainable_introducing_inductive_bias_via_representational_alignme.md)**

:   通过层级CKA对齐将架构先验从引导模型转移到目标模型，使FCN停止过拟合、RNN-Transformer差距缩小、ResNet匹配基线性能。

**[笔记4：WebThinker - 赋予推理模型深度研究能力](others/webthinker_empowering_large_reasoning_models_with_deep_research_capability.md)**

:   WebThinker赋予大型推理模型(LRM)自主的网络搜索与导航能力，通过Think-Search-Draft策略实现推理、信息采集与报告生成的无缝交织，经RL优化后在复杂推理与科学报告生成任务上超越o1与Gemini。
