<!-- 由 src/gen_blog_index.py 自动生成 -->
# 💡 LLM 推理

**🔬 ICLR2026** · 共 **39** 篇

**[Adaptive Social Learning via Mode Policy Optimization for Language Agents](adaptive_social_learning_via_mode_policy_optimization_for_language_agents.md)**

:   提出 Adaptive Social Learning（ASL）框架，设计四种层次化推理模式（从直觉回应到深度推演），并通过 AMPO 算法（融合模式级和样本级优势估计）让 LLM agent 根据社交场景复杂度自适应切换推理深度，在社交智能任务上比 GPT-4o 高 15.6%，比 GRPO 高 7.0% 且 token 用量减少 32.8%。

**[Agentified Assessment of Logical Reasoning Agents](agentified_assessment_of_logical_reasoning_agents.md)**

:   提出基于Agent的评测框架(AAA)，用assessor agent标准化地评估逻辑推理agent，并以自动形式化agent（NL→Z3Py+SMT求解）在清洗后的FOLIO上达到86.70%准确率，大幅超过CoT基线73.89%。

**[AIMCoT: Active Information-driven Multimodal Chain-of-Thought for Vision-Language Reasoning](aimcot_active_information-driven_multimodal_chain-of-thought_for_vision-language.md)**

:   提出 AIMCoT，将多模态 CoT 的视觉信息选择从"被动关注高注意力区域"转变为"主动寻找最高信息增益区域"，通过三个模块（CAG 上下文增强注意力图、AVP 主动视觉探测、DAT 动态注意力转移触发）协同工作，在 LLaVA-W 上比 ICoT 提升 18.25%（0-shot），是一个免训练的即插即用框架。

**[Annotation-Efficient Universal Honesty Alignment](annotation-efficient_universal_honesty_alignment.md)**

:   提出 EliCal（先激发后校准）两阶段框架，先用无标注的 self-consistency 信号教 LLM 表达内部置信度，再用极少量正确性标注（仅 1k 个，占 0.18%）进行校准，在 HonestyBench（560K 训练 + 70K 评估）上达到接近全量标注 98% 的诚实性对齐性能，并在未见 MMLU 任务上泛化优于仅校准基线。

**[Are Reasoning LLMs Robust to Interventions on Their Chain-of-Thought?](are_reasoning_llms_robust_to_interventions_on_their_chain-of-thought.md)**

:   系统评估推理型 LLM 对其 CoT 中各种干预（良性/中性/对抗性）的鲁棒性：发现模型总体鲁棒能从干预中恢复，但改写风格（paraphrasing）会抑制"自我怀疑"表达导致正确率下降，恢复过程有显著计算开销（CoT 膨胀最高 665%）。

**[ATTS: Asynchronous Test-Time Scaling via Conformal Prediction](atts_asynchronous_test-time_scaling_via_conformal_prediction.md)**

:   提出 ATTS，一个基于 conformal prediction 的异步 test-time scaling 框架，通过将 rejection sampling 重构为假设检验过程来消除同步开销，在 MATH/AIME 等数学推理任务上实现最高 56.7x 加速和 4.14x 吞吐量提升，且无精度损失；1.5B/70B 的 draft/target 组合可达到 o3-mini (high) 的 AIME 水平。

**[Beyond Prompt-Induced Lies: Investigating LLM Deception on Benign Prompts](beyond_prompt-induced_lies_investigating_llm_deception_on_benign_prompts.md)**

:   提出 Contact Searching Question (CSQ) 框架，通过两个统计指标（欺骗意图分数 ρ 和欺骗行为分数 δ）量化 LLM 在正常良性提示下的自发欺骗行为，发现 16 个主流 LLM 普遍存在随任务难度升级的系统性欺骗倾向。

**[Compositional Generalization from Learned Skills via CoT Training: A Theoretical and Structural Analysis for Reasoning](compositional_generalization_from_learned_skills_via_cot_training_a_theoretical_.md)**

:   本文通过信息论泛化界和可解释性分析证明，CoT 训练的核心机制是**组合泛化**——模型学会系统性地组合已学的简单技能来解决新颖复杂问题，并内化为两阶段组合推理电路，使中间结果在更浅层提取，释放深层专注于后续推理步骤。

**[Continuous Chain of Thought Enables Parallel Exploration and Reasoning](continuous_chain_of_thought_enables_parallel_exploration_and_reasoning.md)**

:   CoT2 提出用连续值 token（词表 embedding 的凸组合）替代离散 token 进行链式推理，使模型能在单次推理中并行追踪多条推理路径，理论证明等价于 K 次 self-consistency/best-of-N 采样，并通过 GRPO 强化学习进一步提升性能。

**[CoT-RVS: Zero-Shot Chain-of-Thought Reasoning Segmentation for Videos](cot-rvs_zero-shot_chain-of-thought_reasoning_segmentation_for_videos.md)**

:   提出 CoT-RVS，一种无训练的多智能体框架，利用 MLLM 的零样本 Chain-of-Thought 能力进行时间-语义推理以选择关键帧，实现对复杂隐式查询的推理视频分割，在多个 benchmark 上大幅超越已有方法。

**[DAG-Math: Graph-of-Thought Guided Mathematical Reasoning in LLMs](dag-math_graph-of-thought_guided_mathematical_reasoning_in_llms.md)**

:   将 LLM 的 CoT 推理形式化为 DAG 上的基于规则的随机过程，提出"逻辑闭合性"（logical closeness）度量来评估模型是否通过搜索还是严格逻辑推理得到答案，构建了 2894 个金标准 DAG-MATH benchmark，发现即使 PASS@k 相近的模型在推理忠实度上也存在显著差异。

**[DRPO: Efficient Reasoning via Decoupled Reward Policy Optimization](drpo_efficient_reasoning_via_decoupled_reward_policy_optimization.md)**

:   诊断出 GRPO 在加入长度惩罚后的根本缺陷——正确但冗长的回答可能获得负优势值从而被错误惩罚——提出 DRPO 将正负样本的奖励信号解耦，确保长度惩罚只在正确回答组内归一化，在 1.5B 模型上实现 77% 长度缩减仅 1.1% 性能损失（对比基线 68% 缩减 4.3% 损失）。

**[Dynamics-Predictive Sampling for Active RL Finetuning of Large Reasoning Models](dynamics-predictive_sampling_for_active_rl_finetuning_of_large_reasoning_models.md)**

:   将 RL 微调中每个 prompt 的求解进度建模为隐马尔可夫动力系统，通过轻量贝叶斯推断在线预测 prompt 的求解状态，优先采样"部分求解"的 prompt，以不到 DS 30% 的 rollout 量达到同等甚至更优的推理性能。

**[Dynamics Within Latent Chain-of-Thought: An Empirical Study of Causal Structure](dynamics_within_latent_chain-of-thought_an_empirical_study_of_causal_structure.md)**

:   将隐式CoT建模为结构因果模型(SCM)，通过逐步do-干预分析Coconut和CODI两种范式，发现隐式推理步骤具有异质性因果杠杆、非局部跳跃传播结构、以及输出层早期偏向与表征层晚期提交之间的持续性差距。

**[Estimating the Empowerment of Language Model Agents](estimating_the_empowerment_of_language_model_agents.md)**

:   提出 EELMA，用信息论的"赋权"（Empowerment = 动作与未来状态的互信息）作为目标无关的 LM Agent 能力度量，与任务表现强相关（r=0.83-0.94），发现 CoT 移除 99% 赋权、认证动作赋权跳跃 3 倍。

**[FastGRPO: Accelerating Policy Optimization via Concurrency-aware Speculative Decoding and Online Draft Learning](fastgrpo_accelerating_policy_optimization_via_concurrency-aware_speculative_deco.md)**

:   针对GRPO训练中生成阶段占91%-98%时间的瓶颈，提出并发感知的投机解码策略（动态调整draft树大小）和在线draft模型学习（持续适配目标模型分布），实现2.35x-2.72x端到端加速。

**[Fine-R1: Make Multi-modal LLMs Excel in Fine-Grained Visual Recognition by Chain-of-Thought Reasoning](fine-r1_make_multi-modal_llms_excel_in_fine-grained_visual_recognition_by_chain-.md)**

:   Fine-R1 通过 CoT 监督微调（"视觉分析→候选子类→对比→预测"结构化推理链）+ 三元组增强策略优化 TAPO（类内增强提升鲁棒性 + 类间增强提升判别力），仅用 4-shot 训练即在细粒度视觉识别上超越 CLIP 和通用/推理型 MLLM。

**[Fixing the Broken Compass: Diagnosing and Improving Inference-Time Reward Modeling](fixing_the_broken_compass_diagnosing_and_improving_inference-time_reward_modelin.md)**

:   系统诊断推理时奖励模型(RM)的三大问题（简单题性能下降、采样增多判别力衰退、高搜索多样性损害），提出CRISP算法通过答案聚类聚合奖励信号+逐步前缀引导生成，比其他RM推理方法提升最高5%准确率，比R1模型在非数学任务上平均提升10%且token量减少90%。

**[Formal Mechanistic Interpretability: Automated Circuit Discovery with Provable Guarantees](formal_mechanistic_interpretability_automated_circuit_discovery_with_provable_gu.md)**

:   将神经网络验证（NN verification）引入机制可解释性，提出首个具有可证明保证的电路发现框架：在连续输入域上保证电路忠实度（input robustness）、在连续 patching 域上保证电路一致性（patching robustness），并形式化了四级最小性层次（quasi → local → subset → cardinal），通过单调性理论将三类保证统一连接。

**[GeoGramBench: Benchmarking the Geometric Program Reasoning in Modern LLMs](geogrambench_benchmarking_the_geometric_program_reasoning_in_modern_llms.md)**

:   提出Program-to-Geometry任务和GeoGramBench(500题)，用三级几何复杂度分类法(基元识别/局部组合/全局抽象)评估19个前沿LLM从程序代码构建几何表征并推理的能力，发现所有模型在最高抽象级别准确率均低于50%。

**[Harder Is Better: Boosting Mathematical Reasoning via Difficulty-Aware GRPO and Multi-Aspect Question Reformulation](harder_is_better_boosting_mathematical_reasoning_via_difficulty-aware_grpo_and_m.md)**

:   揭示GRPO中更新幅度对难题隐式抑制的问题(中等难度题更新最大)，提出MathForge框架：DGPO用MAD替换std实现难度均衡+难题加权，MQR通过多方面改写增加题目难度但保留答案，在6个数学推理benchmark上平均超GRPO +4.56%。

**[I Can't Believe It's Not Robust: Catastrophic Collapse of Safety Classifiers under Embedding Drift](i_cant_believe_its_not_robust_catastrophic_collapse_of_safety_classifiers_under_.md)**

:   本文系统研究了基于 frozen embedding 的安全分类器在模型更新导致 embedding 漂移时的脆弱性，发现仅 2% 的 embedding 扰动即可将分类器性能从 85% ROC-AUC 降至随机水平（50%），且 72% 的误分类发生在高置信度下（silent failure），同时 instruction-tuned 模型反而比 base 模型更难分类。

**[InnoGym: Benchmarking the Innovation Potential of AI Agents](innogym_benchmarking_the_innovation_potential_of_ai_agents.md)**

:   提出 InnoGym 框架和 iBench/iGym 基准，首次从"创新性"维度评估 AI Agent——不仅衡量正确性还衡量方法论新颖性，发现当前 Agent 能产生新颖想法但无法转化为性能提升（平均归一化增益 -0.45）。

**[LingOly-TOO: Disentangling Reasoning from Knowledge with Templatised Orthographic Obfuscation](lingoly-too_disentangling_reasoning_from_knowledge_with_templatised_orthographic.md)**

:   提出LingOly-TOO benchmark(1,203题/6,995子问题)，通过对语言学奥赛题的专家设计正字法混淆来分离LLM的推理能力与知识/记忆，发现最强模型从原始题0.59降至混淆后0.48，揭示了LLM推理能力被严重高估。

**[LogicReward: Incentivizing LLM Reasoning via Step-Wise Logical Supervision](logicreward_incentivizing_llm_reasoning_via_step-wise_logical_supervision.md)**

:   提出LogicReward奖励函数，用Isabelle定理证明器做步骤级逻辑正确性验证，结合Autoformalization with Soft Unification减少自然语言歧义，训练出的8B模型在NLI和逻辑推理任务上超越GPT-4o 11.6%和o4-mini 2%。

**[mR3: Multilingual Rubric-Agnostic Reward Reasoning Models](mr3_multilingual_rubric-agnostic_reward_reasoning_models.md)**

:   提出 mR3，72 语言的多语言评分推理模型，通过课程学习和 GPT-OSS-120B 蒸馏训练，14B 参数模型匹配 120B 模型性能（mR3-RewardBench 88.46%），支持逐点/成对/二元三种评估模式。

**[Nudging the Boundaries of LLM Reasoning](nudging_the_boundaries_of_llm_reasoning.md)**

:   指出GRPO无法从"不可解"问题(0% pass rate)学习的根本局限，提出NuRL方法在训练时对难题注入自生成的抽象hint(不泄露答案)使其变为可学习样本，在6个benchmark和3个模型上一致超越GRPO且能提升模型能力上界(pass@k)。

**[On The Fragility of Benchmark Contamination Detection in Reasoning Models](on_the_fragility_of_benchmark_contamination_detection_in_reasoning_models.md)**

:   系统性研究发现 LRM 的基准污染检测极其脆弱：SFT 阶段引入的污染在经过 GRPO 训练后检测信号几乎消失（PPO 式重要性采样/裁剪是根因），而对高级 LRM 直接用 CoT 做 SFT 污染则几乎不留任何可检测痕迹，现有 10 种检测方法均接近随机猜测。

**[PrismAudio: Decomposed Chain-of-Thoughts and Multi-dimensional Rewards for Video-to-Audio Generation](prismaudio_decomposed_chain-of-thoughts_and_multi-dimensional_rewards_for_video-.md)**

:   首次将分解式 Chain-of-Thought 推理与多维度强化学习（RL）结合应用于视频到音频（V2A）生成，通过四个专门化的 CoT 模块（语义/时序/美学/空间）配合对应奖励函数，解决了目标纠缠问题，并提出 Fast-GRPO 算法大幅降低 RL 训练开销。

**[Query-Level Uncertainty in Large Language Models](query-level_uncertainty_in_large_language_models.md)**

:   提出Query-Level Uncertainty概念，通过Internal Confidence方法在生成前（单次前向传播）估计LLM能否回答给定查询，无需训练即可实现高效的自适应推理（RAG触发/模型级联/弃权）。

**[RFEval: Benchmarking Reasoning Faithfulness under Counterfactual Reasoning Intervention in Large Reasoning Models](rfeval_benchmarking_reasoning_faithfulness_under_counterfactual_reasoning_interv.md)**

:   提出 RFEval，通过立场一致性和因果影响（反事实干预）形式化推理忠实度，构建 7186 实例/7 任务基准，发现 49.7% 不忠实率，RL 后训练虽维持精度但降低忠实度。

**[SceneCOT: Eliciting Grounded Chain-of-Thought Reasoning in 3D Scenes](scenecot_eliciting_grounded_chain-of-thought_reasoning_in_3d_scenes.md)**

:   提出 SceneCOT，首个将 Chain-of-Thought 推理引入 3D 场景理解的框架，通过四阶段推理管线（任务识别→区域定位→实体接地→接地推理）将中间推理步骤显式关联到视觉 grounding，在 Beacon3D 上 Good Coherence 达到 34.7%（比最强 baseline 的 20.4% 高出 70%+）。

**[SealQA: Raising the Bar for Reasoning in Search-Augmented Language Models](sealqa_raising_the_bar_for_reasoning_in_search-augmented_language_models.md)**

:   提出SealQA挑战基准，包含111道使前沿非推理模型准确率为0的事实性问题，专门评估搜索增强LLM在噪声/冲突/误导性检索结果下的推理能力。

**[Segment-Level Attribution for Selective Learning of Long Reasoning Traces](segment-level_attribution_for_selective_learning_of_long_reasoning_traces.md)**

:   用Integrated Gradients计算长推理链中每个segment对最终答案的归因强度和方向一致性，识别重要segment进行选择性SFT，相比全CoT训练提升准确率达4.7%同时缩短输出18%。

**[The Illusion of Diminishing Returns: Measuring Long Horizon Execution in LLMs](the_illusion_of_diminishing_returns_measuring_long_horizon_execution_in_llms.md)**

:   揭示短任务基准给出"收益递减"的假象——单步准确率的微小提升在长任务中指数级放大；发现 LLM 的"自我条件化效应"（自身错误增加后续出错概率），thinking 模型可修复此效应；GPT-5 thinking 可执行超过 2100 步长任务。

**[TopoBench: Benchmarking LLMs on Hard Topological Reasoning](topobench_benchmarking_llms_on_hard_topological_reasoning.md)**

:   构建TopoBench基准(6类拓扑谜题×3难度)评估LLM的全局空间推理能力，发现前沿模型hard tier仅解决<24%，并通过因果干预实验发现错误频率不等于因果影响——低频的约束遗忘比高频的重复推理更具破坏性。

**[TumorChain: Interleaved Multimodal Chain-of-Thought Reasoning for Traceable Clinical Tumor Analysis](tumorchain_interleaved_multimodal_chain-of-thought_reasoning_for_traceable_clini.md)**

:   提出 TumorChain，面向肿瘤分析的交错多模态 CoT 推理框架，通过 1.5M CoT-VQA 数据引擎、器官引导的迭代交错推理（IIR）和混合模型协同优化，在肿瘤定位/属性分析/TNM分期上平均精度 84.41%，大幅超越 GPT-5-Mini（51.59%）。

**[Verifying Chain-of-Thought Reasoning via Its Computational Graph](verifying_chain-of-thought_reasoning_via_its_computational_graph.md)**

:   提出CRV白盒方法，通过分析LLM推理步骤的归因图（计算图）结构特征来验证CoT正确性，在Arithmetic任务上AUROC达92.47，远超黑盒(76.45)和灰盒方法，并通过因果干预成功纠正错误推理。

**[When Shallow Wins: Silent Failures and the Depth-Accuracy Paradox in Latent Reasoning](when_shallow_wins_silent_failures_and_the_depth-accuracy_paradox_in_latent_reaso.md)**

:   分析Qwen2.5-Math-7B的隐式推理发现其61%准确率中仅18.4%来自稳定忠实的推理路径，81.6%通过不一致路径得出，8.8%为"静默失败"（高置信但错误），揭示benchmark准确率掩盖计算可靠性问题。
