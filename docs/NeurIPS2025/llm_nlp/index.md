<!-- 由 src/gen_blog_index.py 自动生成 -->
# 💬 LLM / NLP

**🧠 NeurIPS2025** · 共 **32** 篇

**[AceSearcher: Bootstrapping Reasoning and Search for LLMs via Reinforced Self-Play](acesearcher_bootstrapping_reasoning_and_search_for_llms_via_reinforced_self-play.md)**

:   提出 AceSearcher——一种协作式自我博弈框架，让单个 LLM 同时扮演**问题分解者**（将复杂查询拆解为子问题引导检索）和**求解者**（整合检索上下文生成答案），通过 SFT + 迭代 DPO 两阶段训练，仅用最终答案作为奖励信号，在 10 个数据集上平均 EM 提升 7.6%，32B 模型匹配 DeepSeek-V3（<5% 参数）。

**[Active Slice Discovery in Large Language Models](active_slice_discovery_in_large_language_models.md)**

:   提出 **Active Slice Discovery** 问题框架，将主动学习引入 LLM 错误切片发现，利用不确定性采样 + LLM 内部表征（原始 embedding 或 SAE 特征）在仅使用 2-10% 标注的情况下达到接近全标注的切片检测精度。

**[AdaSTaR: Adaptive Data Sampling for Training Self-Taught Reasoners](adastar_adaptive_data_sampling_for_training_self-taught_reasoners.md)**

:   发现 STaR（自我教学推理器）的随机数据采样导致观测训练频率严重不平衡（简单题过度训练、难题训练不足），提出 AdaSTaR——通过自适应多样性采样（优先欠训练样本）和自适应课程采样（根据模型强度调节难度），在 6 个基准上全部取得最高准确率同时减少 58.6% 训练 FLOPs。

**[AI Progress Should Be Measured by Capability-Per-Resource, Not Scale Alone: A Framework for Gradient-Guided Resource Allocation in LLMs](ai_progress_should_be_measured_by_capability-per-resource_not_scale_alone_a_fram.md)**

:   本文以 position paper 的形式挑战"规模至上主义"，提出以**能力-每-资源（Capability-Per-Resource, CPR）**取代单纯的规模扩张来衡量 AI 进步，并给出一套基于梯度引导的资源分配理论框架——通过发布"梯度蓝图"元数据，使下游适配者仅微调高影响力参数子集即可在资源占用大幅降低的同时保持接近全参数微调的性能。

**[Are Language Models Efficient Reasoners? A Perspective from Logic Programming](are_language_models_efficient_reasoners_a_perspective_from_logic_programming.md)**

:   本文通过逻辑规划框架评估LLM的推理效率，发现当存在不相关公理时，LLMs会产生内容庞杂的证明而非最优证明，反映出模型存在显著的推理效率问题。

**[Belief-Calibrated Multi-Agent Consensus Seeking for Complex NLP Tasks](belief-calibrated_multi-agent_consensus_seeking_for_complex_nlp_tasks.md)**

:   提出 Belief-Calibrated Consensus Seeking (BCCS) 框架，通过引入信念（belief）校准的共识判断、冲突感知的协作者分配和领导者选择三个模块，让多智能体系统在复杂NLP任务上达成更稳定的共识，在 MATH 和 MMLU 上的困难任务分别提升 2.23% 和 3.95%。

**[Beyond Components: Singular Vector-Based Interpretability of Transformer Circuits](beyond_components_singular_vector-based_interpretability_of_transformer_circuits.md)**

:   提出基于SVD奇异向量的方向级可解释性框架，通过对注意力头和MLP的增广矩阵统一SVD分解+可学习对角掩码（KL+L₁），发现单组件内存在正交低秩子函数叠加——IOI任务仅需~9%方向即可KLD=0.21复现模型行为。

**[CAT: Circular-Convolutional Attention for Sub-Quadratic Transformers](cat_circular-convolutional_attention_for_sub-quadratic_transformers.md)**

:   本文提出CAT（Circular-convolutional Attention），通过FFT计算循环卷积将Self-Attention复杂度从O(N²)降至O(N log N)，同时保持完整的softmax机制和全局注意力。

**[CBMAS: Cognitive Behavioral Modeling via Activation Steering](cbmas_cognitive_behavioral_modeling_via_activation_steering.md)**

:   CBMAS 提出一个连续激活干预诊断框架，将传统“前后对比式”认知偏差分析扩展为可解释的干预轨迹分析，通过 alpha 强度扫描、logit-lens 偏置曲线与层位敏感性分析，揭示 LLM 行为翻转临界点与跨层演化机制。

**[CodeAssistBench (CAB): Dataset & Benchmarking for Multi-turn Chat-Based Code Assistance](codeassistbench_cab_dataset_benchmarking_for_multi-turn_chat-based_code_assistan.md)**

:   提出 CodeAssistBench (CAB)，第一个评估多轮、项目级编程辅助的全自动 Benchmark，从 GitHub issues 自动构建 3,286 个真实编程求助场景，涵盖 7 种编程语言 214 个仓库，发现 SOTA 模型在 StackOverflow 式问题上达 70-83% 准确率，但在 post-training-cutoff 仓库上仅 7-16%。

**[CoRe: Benchmarking LLMs' Code Reasoning Capabilities through Static Analysis Tasks](core_benchmarking_llms_code_reasoning_capabilities_through_static_analysis_tasks.md)**

:   提出 CoRe，一个包含 12,553 个人工验证任务实例的高质量 benchmark，通过数据依赖、控制依赖和信息流三类静态分析基础任务，直接评估 LLM 的代码语义推理能力，揭示模型在 trace 生成和源枚举等需要多步推理的任务上仍严重不足。

**[DATE-LM: Benchmarking Data Attribution Evaluation for Large Language Models](date-lm_benchmarking_data_attribution_evaluation_for_large_language_models.md)**

:   DATE-LM是首个统一、应用驱动的LLM数据归因基准，涵盖数据选择、毒性过滤、事实归因三大应用，通过公开排行榜促进可复现和公平的方法比较。

**[Decoupled Entropy Minimization](decoupled_entropy_minimization.md)**

:   将经典熵最小化（EM）解耦为两个对立部分——Cluster Aggregation Driving Factor (CADF，奖励主导类别)和 Gradient Mitigation Calibrator (GMC，惩罚高置信类别)，揭示了经典 EM 的两个固有缺陷（reward collapse 和 easy-class bias），提出 AdaDEM 通过归一化奖励和边际熵校准来修复这些问题，在半监督学习、域适应、强化学习等多任务上显著提升。

**[Detecting High-Stakes Interactions with Activation Probes](detecting_high-stakes_interactions_with_activation_probes.md)**

:   用线性激活探针检测 LLM 高风险交互，成本低于 LLM 监控 100 万倍，平均 AUROC>0.91 匹敌中等 LLM，级联架构实现效率与精度的最优平衡。

**[Do Different Prompting Methods Yield a Common Task Representation in Language Models?](do_different_prompting_methods_yield_a_common_task_representation_in_language_mo.md)**

:   本文扩展函数向量方法至指令提示，发现演示和指令诱发的任务表示主要不同，仅部分重叠，解释了为何结合两者效果更优。

**[Do Language Models Use Their Depth Efficiently?](do_language_models_use_their_depth_efficiently.md)**

:   通过因果干预、残差流分析和跨模型线性映射，证明当前 LLM 后半部分层不参与组合式计算，仅迭代细化输出概率分布，深层模型只是把浅层模型的计算"展延"到更多层。

**[Don't Be Lazy: CompleteP Enables Compute-Efficient Deep Transformers](dont_be_lazy_completep_enables_compute-efficient_deep_transformers.md)**

:   CompleteP 参数化（α=1）是唯一同时实现深度方向超参转移和完全特征学习的方案，在深模型上相比 μP 节省 12-34% FLOPs。

**[EnCompass: Enhancing Agent Programming with Search Over Program Execution Paths](encompass_enhancing_agent_programming_with_search_over_program_execution_paths.md)**

:   提出 Probabilistic Angelic Nondeterminism (PAN) 编程模型及 EnCompass Python 框架，将 agent 的核心工作流逻辑与推理时搜索策略解耦，程序员只需在 LLM 调用处加 `branchpoint()` 标记，即可用几行参数切换 best-of-N、beam search、tree search 等策略，代码修改量减少 3-6x。

**[Gemstones: A Model Suite for Multi-Faceted Scaling Laws](gemstones_a_model_suite_for_multi-faceted_scaling_laws.md)**

:   Gemstones开源4000+检查点数据集（至2B参数），系统研究宽度-深度-训练代币在缩放律中的影响，揭示缩放律对设计选择的高度敏感性。

**[How Do Transformers Learn Implicit Reasoning?](how_do_transformers_learn_implicit_reasoning.md)**

:   通过符号环境的精细控制研究，本文发现多跳隐式推理会经历记忆→分布内泛化→跨分布泛化三阶段，关键机制是中间实体表示在余弦空间的聚类。

**[Language Model Behavioral Phases are Consistent Across Architecture, Training Data, and Scale](language_model_behavioral_phases_are_consistent_across_archi.md)**

:   论文在 Transformer、Mamba、RWKV，不同数据集与参数规模（14M 到 12B）上系统分析 1400+ checkpoints，发现语言模型预训练中存在高度一致的行为阶段；词级行为变化最多可由 unigram 频率、n-gram 概率、语义相似度三类简单启发式解释（最高约 98% 方差）。

**[Large Language Models Miss the Multi-Agent Mark](large_language_models_miss_the_multi-agent_mark.md)**

:   Position paper 指出当前 MAS LLMs 在四个方面违背了传统多智能体系统（MAS）的基本原则：LLM 缺乏原生社会行为、环境设计以 LLM 为中心、缺少异步协调和标准通信协议、涌现行为缺乏量化评估，并为每个问题提出研究方向。

**[LooGLE v2: LLM在真实世界长依赖挑战上的准备情况评估](loogle_v2_are_llms_ready_for_real_world_long_dependency_challenges.md)**

:   通过自动化流程从法律、金融、游戏、代码等领域采集16k-2M token的真实文本,设计10类长依赖复杂任务,共1934个QA实例,评估表明最优模型仅59.2%准确率,揭示LLM长序列理解能力的根本不足。

**[Memory Mosaics at Scale](memory_mosaics_at_scale.md)**

:   Memory Mosaics v2 将关联存储网络扩展至 10B 参数、1T token 训练规模，在新任务学习和上下文学习上显著超越同规模甚至 8T token 训练的 Transformer。

**[Moose-Chem2 Exploring Llm Limits In Fine-Grained Scientific Hypothesis Discovery](moose-chem2_exploring_llm_limits_in_fine-grained_scientific_hypothesis_discovery.md)**

:   用层次化搜索平滑LLM判断的奖励景观，探索LLM在细粒度科学假设发现中的能力极限，在2024后专家标注的化学基准上一致超越基线。

**[Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models](nemotron-flash_towards_latency-optimal_hybrid_small_language_models.md)**

:   Nemotron-Flash 通过系统优化深宽比、进化搜索混合算子组合（DeltaNet+Mamba2+Attention）以及权重归一化训练，构建延迟最优的小语言模型家族，相比 Qwen3-1.7B/0.6B 分别实现 1.3×/1.9× 延迟下降与 +5.5% 平均准确率提升。

**[Probabilistic Token Alignment for Large Language Model Fusion](probabilistic_token_alignment_for_large_language_model_fusion.md)**

:   PTA-LLM 通过将 token 对齐建模为最优传输（OT）问题，用 Sinkhorn 算法计算两个 LLM 间最优概率匹配，融合 logit 分布而非硬映射，在 78 个任务上平均提升 +1.72%。

**[Retrospective In-Context Learning for Temporal Credit Assignment with Large Language Models](retrospective_incontext_learning_for_temporal_credit_assignm.md)**

:   论文提出 RICL（Retrospective In-Context Learning），利用 LLM 的预训练知识把环境中的稀疏奖励回溯性转化为稠密 advantage supervision，再结合在线策略迭代框架 RICOL，在 BabyAI 四个场景中以更高样本效率达到与传统在线 RL 相当的收敛表现，展示了 LLM 在 temporal credit assignment 上的潜力。

**[Scaling Embedding Layers in Language Models](scaling_embedding_layers_in_language_models.md)**

:   提出Scone方法，通过为高频n-gram学习上下文化的嵌入（用独立Transformer模型训练），在推理时将这些嵌入卸载到主存/SSD，实现"训练时用更多计算但推理时不增加加速器资源"的新缩放范式，1B参数模型超越1.9B基线。

**[Sloth: Scaling Laws for LLM Skills to Predict Multi-Benchmark Performance Across Families](sloth_scaling_laws_for_llm_skills_to_predict_multi-benchmark_performance_across_.md)**

:   提出Skills Scaling Laws (Sloth)，通过假设LLM性能由低维潜在技能（如推理、指令遵循）驱动，利用benchmark间的相关性构建跨模型家族的缩放定律，用少量家族数据即可预测大模型在多个benchmark上的表现。

**[Unifying Attention Heads and Task Vectors via Hidden State Geometry in In-Context Learning](unifying_attention_heads_and_task_vectors_via_hidden_state_geometry_in_in-contex.md)**

:   本文提出基于隐状态几何（可分离性+对齐性）的统一框架，将ICL的两大解释路线——注意力头（PTH/IH）和任务向量——联系起来，揭示ICL在分类任务中的两阶段机制：早期层通过PTH建立可分离性，后期层通过IH改善与标签unembedding方向的对齐性。

**[Yggdrasil: 桥接动态投机和静态运行时的延迟最优树型LLM解码](yggdrasil_bridging_dynamic_speculation_and_static_runtime_for_latency-optimal_tr.md)**

:   通过等增长树(EGT)草稿算法和延迟感知目标，实现动态投机与静态图编译的兼容，配合前向执行阶段重叠，在A100上达3.98×加速。
