<!-- 由 src/gen_blog_index.py 自动生成 -->
# 💡 LLM 推理

**🤖 AAAI2026** · 共 **15** 篇

**[A Reasoning Paradigm for Named Entity Recognition](a_reasoning_paradigm_for_named_entity_recognition.md)**

:   提出 ReasoningNER，将命名实体识别从"隐式模式匹配"转变为"显式推理"范式，通过三阶段流程（CoT数据构建→CoT微调→GRPO强化增强）让模型先推理再抽取实体，在零样本设定下F1超GPT-4达12.3个百分点，8B模型在CrossNER上达72.4平均F1。

**[Answering the Unanswerable Is to Err Knowingly: Analyzing and Mitigating Abstention Failures in Large Reasoning Models](answering_the_unanswerable_is_to_err_knowingly_analyzing_and.md)**

:   系统分析大推理模型(LRM)面对不可回答数学题时的弃权失败现象，发现LRM内部有足够认知能力识别问题不可解（探针分类准确率>80%）但外部行为仍偏向强答，提出认知监控+推理时干预的两阶段方法，将弃权率从16-54%提升至60-92%且不损害可回答题的推理性能。

**[ARCHE: A Novel Task to Evaluate LLMs on Latent Reasoning Chain Extraction](arche_a_novel_task_to_evaluate_llms_on_latent_reasoning_chai.md)**

:   提出潜在推理链提取 (ARCHE) 任务，要求 LLM 将科学论文中的论证分解为基于 Peirce 三种推理范式的推理逻辑树 (RLT)，并通过 Entity Coverage 和 Reasoning Edge Accuracy 两个指标揭示了 10 个主流 LLM 在内容完整性与逻辑正确性之间的本质权衡。

**[BLM-Guard: Explainable Multimodal Ad Moderation with Chain-of-Thought and Policy-Aligned Rewards](blm-guard_explainable_multimodal_ad_moderation_with_chain-of.md)**

:   提出 BLM-Guard，一个面向短视频商业广告的可解释多模态审核框架：先通过 Rule-driven ICoT 数据合成 + SFT 冷启动建立结构化推理能力，再用 Self-Adaptive GRPO 强化学习（结合规则正确性奖励 + 自适应一致性奖励 SCA-R）优化策略对齐，在真实广告 benchmark 上达到 91.4% 严格准确率和 0.845 推理一致性分数。

**[CMMCoT: Enhancing Complex Multi-Image Comprehension via Multi-Modal Chain-of-Thought and Memory Augmentation](cmmcot_enhancing_complex_multi-image_comprehension_via_multi.md)**

:   提出 CMMCoT 框架，通过构建交错的多模态多步推理链（含视觉区域 token 监督）和测试时检索式记忆增强模块（RIFREM），在不增加参数的前提下提升多图场景下的慢思考推理能力，基于 Qwen2.5-VL-7B 在多图基准上平均提升 1.4 分。

**[Deep Hidden Cognition Facilitates Reliable Chain-of-Thought Reasoning](deep_hidden_cognition_facilitates_reliable_chain-of-thought_.md)**

:   本文发现 LLM 在 CoT 推理过程中，中间层的注意力头激活值隐式编码了推理步骤的真实性信息（最高 85% 探测准确率），据此训练置信度预测器引导 Beam Search 动态选择高置信度推理路径，在数学/符号/常识推理任务上超越 Self-Consistency 和 PRM Guided Search。

**[Improving Value-based Process Verifier via Low-Cost Variance Reduction](improving_value-based_process_verifier_via_low-cost_variance_reduction.md)**

:   针对基于值的过程验证器(PRM)训练中蒙特卡罗(MC)估计因采样数有限导致的高方差问题，提出Compound Monte Carlo Sampling (ComMCS)方法，通过线性组合当前步和后续步的MC估计量来无偏地降低方差，无需额外LLM推理开销，在MATH-500上Best-of-32实验中提升2.2个点。

**[Incorporating Self-Rewriting into Large Language Model Reasoning Reinforcement](incorporating_self-rewriting_into_large_language_model_reasoning_reinforcement.md)**

:   提出Self-Rewriting框架，让LRM在RL训练中对"简单"样本（全部回答正确的query）重写自身推理文本并从中学习，仅增加约10%训练开销即可在保持准确率的同时将推理长度减少46%，内部推理质量（LLM-as-Judge）提升7.2分，有效缓解过度思考、冗余思考等问题。

**[Jupiter: Enhancing LLM Data Analysis Capabilities via Notebook and Inference-Time Value-Guided Search](jupiter_enhancing_llm_data_analysis_capabilities_via_notebook_and_inference-time.md)**

:   构建NbQA数据集（从真实Jupyter Notebook提取3.8万task-solution对）+ 提出Jupiter框架（将数据分析建模为状态级搜索问题，用值模型引导PUCT搜索），使Qwen2.5-14B在InfiAgent-DABench上达86.38%超越GPT-4o(85.99%)，Qwen2.5-7B在DSBench上从63.51%提升至89.19%。

**[L2V-CoT: Cross-Modal Transfer of Chain-of-Thought Reasoning via Latent Intervention](l2v-cot_cross-modal_transfer_of_chain-of-thought_reasoning_v.md)**

:   通过 LAT 分析发现 LLM 和 VLM 的低频 CoT 方向表示具有相似分布，提出 L2V-CoT：从 LLM 提取 CoT 方向表示 → 低通滤波 → 频域重采样匹配维度 → 注入 VLM 隐藏层，training-free 地将 LLM 的推理能力迁移到 VLM，平均提升 3.7%，最高 8.6%。

**[SAPO: Self-Adaptive Process Optimization Makes Small Reasoners Stronger](sapo_self-adaptive_process_optimization_makes_small_reasoners_stronger.md)**

:   受神经科学中Error-Related Negativity启发，提出自适应过程优化方法SAPO，通过首错检测+局部后验估计替代低效的逐步蒙特卡洛rollout，在降低2-3倍计算成本的同时实现推理器-验证器协同优化，使小语言模型（≤2B）在数学和代码推理任务上超越多数自演化方法。

**[SCALE: Selective Resource Allocation for Overcoming Performance Bottlenecks in Mathematical Test-time Scaling](scale_selective_resource_allocation_for_overcoming_performance_bottlenecks_in_ma.md)**

:   基于认知科学的双过程理论，提出SCALE框架将数学问题分解为子问题后按难度分配不同计算资源（System 1快速计算 vs System 2深度推理），在AIME25上将Qwen3-32B从57.50%提升至71.25%，同时比InftyThink节省33-53%的token。

**[Stable Voting and the Splitting of Cycles](stable_voting_and_the_splitting_of_cycles.md)**

:   研究Simple Stable Voting (SSV)——已在数百次实际选举中使用的递归投票规则——是否总是精化(refine)Split Cycle (SC)方法的猜想，通过数学证明（≤5候选人）和SAT求解（6-7候选人）确定：猜想在≤6候选人时成立，≥7候选人时被反驳，并通过构造性证明推广到任意多候选人。

**[ToC: Tree-of-Claims Search with Multi-Agent Language Models](toc_tree-of-claims_search_with_multi-agent_language_models.md)**

:   提出 Tree-of-Claims (ToC) 框架，将专利权利要求编辑建模为结构化搜索问题，通过 MCTS 与 EditorAgent/ExaminerAgent 多智能体协作，在新颖性、范围保持和语义一致性之间联合优化，比零/少样本 LLM 基线平均提升约 8% 综合分。

**[Trade-offs in Large Reasoning Models: An Empirical Analysis of Deliberative and Adaptive Reasoning over Foundational Capabilities](trade-offs_in_large_reasoning_models_an_empirical_analysis_of_deliberative_and_a.md)**

:   系统评估了LRM（如DeepSeek-R1、QwQ、OpenThinker等）在获取深度推理能力后对基础能力（helpfulness和harmlessness）的负面影响，发现deliberative reasoning显著降低指令遵循和安全性能力，并提出Zero-Thinking、Less-Thinking、Summary-Thinking等自适应推理模式可有效缓解这些缺陷。
