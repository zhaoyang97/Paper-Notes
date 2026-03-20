<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔬 ICLR2026 论文笔记

共 **539** 篇笔记，覆盖 **34** 个领域。

## 领域概览

| 领域 | 篇数 |
|:-----|-----:|
| 💬 [LLM / NLP](#llm_nlp) | 53 |
| 📂 [其他](#others) | 51 |
| 🎨 [图像生成](#image_generation) | 50 |
| 🎮 [强化学习](#reinforcement_learning) | 43 |
| 💡 [LLM 推理](#llm_reasoning) | 39 |
| 🏥 [医学图像](#medical_imaging) | 29 |
| 🧊 [3D 视觉](#3d_vision) | 21 |
| 🛡️ [AI 安全](#ai_safety) | 21 |
| 📦 [模型压缩](#model_compression) | 21 |
| ⚖️ [对齐 / RLHF](#llm_alignment) | 19 |
| 👁️ [多模态 VLM](#multimodal_vlm) | 19 |
| ⚡ [LLM 效率](#llm_efficiency) | 16 |
| 🎯 [目标检测](#object_detection) | 16 |
| 🤖 [机器人/具身智能](#robotics) | 16 |
| 🚗 [自动驾驶](#autonomous_driving) | 14 |
| 🦾 [LLM Agent](#llm_agent) | 13 |
| 📐 [优化/理论](#optimization) | 13 |
| 🎬 [视频理解](#video_understanding) | 13 |
| 🧑 [人体理解](#human_understanding) | 11 |
| ✂️ [语义分割](#segmentation) | 9 |
| 🕸️ [图学习](#graph_learning) | 8 |
| 🎵 [音频/语音](#audio_speech) | 7 |
| 🔗 [因果推理](#causal_inference) | 6 |
| 🖼️ [图像恢复](#image_restoration) | 6 |
| 📈 [时间序列](#time_series) | 6 |
| 🔎 [AIGC 检测](#aigc_detection) | 4 |
| 🎁 [推荐系统](#recommender) | 3 |
| ✍️ [文本生成](#nlp_generation) | 2 |
| 📖 [NLP 理解](#nlp_understanding) | 2 |
| 🛰️ [遥感](#remote_sensing) | 2 |
| 🧮 [科学计算](#scientific_computing) | 2 |
| 📡 [信号/通信](#signal_comm) | 2 |
| ⚛️ [物理学](#physics) | 1 |
| 🔄 [自监督/表示学习](#self_supervised) | 1 |

---

## 💬 LLM / NLP { #llm_nlp }

**[A Cortically Inspired Architecture for Modular Perceptual AI](llm_nlp/a_cortically_inspired_architecture_for_modular_perceptual_ai.md)**

:   从神经科学出发提出皮层启发的模块化感知 AI 架构蓝图，包含专用编码器、共享跨模态潜空间、路由控制器和递归预测反馈回路四个组件，并通过稀疏自编码器实验验证模块化分解可提升域内特征稳定性 (+15.4pp Jaccard 重叠)。

**[ASIDE: Architectural Separation of Instructions and Data in Language Models](llm_nlp/aside_architectural_separation_of_instructions_and_data_in_language_models.md)**

:   提出 ASIDE，一种在 token embedding 层面通过正交旋转区分指令和数据的架构级改造，仅需修改前向传播并在标准指令微调数据上训练，即可显著提升指令-数据分离度和 prompt injection 鲁棒性，无需任何安全专项训练。

**[Assetformer Modular 3D Assets Generation With Autoregressive Transformer](llm_nlp/assetformer_modular_3d_assets_generation_with_autoregressive_transformer.md)**

:   提出 AssetFormer，基于 Llama 架构的自回归 Transformer，将模块化 3D 资产（由 primitive 序列组成）建模为离散 token 序列，通过 DFS/BFS 图遍历重排序和联合词汇表解码实现从文本描述生成可直接用于游戏引擎的模块化 3D 资产。

**[ATLAS: Adaptive Transfer Scaling Laws for Multilingual Pretraining, Finetuning, and Decoding the Curse of Multilinguality](llm_nlp/atlas_adaptive_transfer_scaling_laws_for_multilingual_pretraining_finetuning_and.md)**

:   提出 Adaptive Transfer Scaling Law (ATLAS)，通过将有效数据量分解为目标语言、迁移语言和其他语言三项并引入数据重复饱和函数，在774个多语言训练实验（10M–8B参数、400+语言）上显著优于现有scaling law（多语言 $R^2$ 从0.67提升至0.98），并系统量化了跨语言迁移矩阵、多语言诅咒的容量约束以及预训练vs微调的计算交叉点。

**[Attributing Response to Context: A Jensen-Shannon Divergence Driven Mechanistic Study of Context Attribution in Retrieval-Augmented Generation](llm_nlp/attributing_response_to_context_a_jensen-shannon_divergence_driven_mechanistic_s.md)**

:   提出ARC-JSD方法，通过计算完整上下文与逐句消融上下文下的响应分布的Jensen-Shannon散度，在无需微调、梯度计算或代理模型的情况下实现高效精准的RAG上下文归因，并结合Logit Lens进行机制分析，定位负责上下文归因的注意力头和MLP层，通过门控操作降低约39%的幻觉率。

**[Auditing Cascading Risks in Multi-Agent Systems via Semantic–Geometric Co-evolution](llm_nlp/auditing_cascading_risks_in_multi-agent_systems_via_semanti-geometric_co-evolut.md)**

:   提出 SCCAL 框架，通过耦合语义流（semantic flow）和交互图的 Ollivier–Ricci 曲率（ORC）来建模多智能体系统中语义-几何的协同演化，利用两者的一致性残差作为级联风险的早期预警信号，在语义违规显现前数轮即可检测异常。

**[Benchmarking Overton Pluralism in LLMs](llm_nlp/benchmarking_overton_pluralism_in_llms.md)**

:   提出 OvertonBench 框架，通过大规模人类研究（1208名美国代表性参与者、60个主观问题、8个LLM）将 Overton 多元主义形式化为集合覆盖度指标 OvertonScore，发现当前所有模型得分仅 0.35–0.41（理论上限为 1.0），并构建了与人类判断高度相关（ρ=0.88）的自动化评测工具。

**[BiasFreeBench: a Benchmark for Mitigating Bias in Large Language Model Responses](llm_nlp/biasfreebench_a_benchmark_for_mitigating_bias_in_large_language_model_responses.md)**

:   本文构建了 BiasFreeBench 基准，首次在统一框架下系统比较 8 种主流去偏方法（4 种 prompting + 4 种 training），聚焦于 LLM 响应层面的偏差评估，并提出了 Bias-Free Score 指标，发现 prompting 方法（尤其是 CoT）整体优于 training 方法，而 DPO 在跨偏差类型泛化上表现突出。

**[Breaking the Correlation Plateau: On the Optimization and Capacity Limits of Attention-Based Regressors](llm_nlp/breaking_the_correlation_plateau_on_the_optimization_and_capacity_limits_of_atte.md)**

:   本文首次从理论上分析了注意力回归模型在联合 MSE+PCC 训练时出现的"PCC平台期"现象——发现其根源在于 MSE 优化与 PCC 梯度之间的冲突以及 softmax 凸聚合的表达力上界——并提出 ECA（Extrapolative Correlation Attention）框架，通过缩放残差聚合、色散感知温度 softmax 和色散归一化 PCC 损失三个组件突破该限制。

**[Closing the Curvature Gap: Full Transformer Hessians and Their Implications for Scaling Laws](llm_nlp/closing_the_curvature_gap_full_transformer_hessians_and_their_implications_for_s.md)**

:   首次推导完整 Transformer block（含 LayerNorm 和 FFN）的显式 Hessian 表达式及谱范数上界，建立了损失面随数据量增加以 $O(1/k)$ 速率收敛的理论框架，为 scaling laws 和曲率感知训练提供了数学基础。

**[Common Corpus: The Largest Collection of Ethical Data for LLM Pre-Training](llm_nlp/common_corpus_ethical_data_for_llm_pretraining.md)**

:   构建了约 2 万亿 token 的最大规模合法授权 LLM 预训练数据集 Common Corpus，覆盖多语言（含低资源语言）和代码数据，所有数据均为无版权或宽松许可来源，配有完整的数据溯源和多阶段过滤管道，已被 Anthropic 等机构采用。

**[Compositional-ARC: Assessing Systematic Generalization in Abstract Spatial Reasoning](llm_nlp/compositional-arc_assessing_systematic_generalization_in_abstract_spatial_reason.md)**

:   提出 Compositional-ARC 数据集评估模型在抽象空间推理中的系统性泛化能力——从已知基础几何变换（如平移、旋转）泛化到未见过的变换组合。一个仅 5.7M 参数的 MLC 训练的 encoder-decoder 模型在系统性任务上达到 78.26%，与 ARC Prize 2024 冠军的 8B 模型+TTT 持平，远超 GPT-4o、o3-mini 等（<3%）。

**[Conformal Prediction Adaptive to Unknown Subpopulation Shifts](llm_nlp/conformal_prediction_adaptive_to_unknown_subpopulation_shifts.md)**

:   针对子群体偏移（subpopulation shift）下标准 conformal prediction 失效的问题，提出三种自适应算法：利用学习的 domain classifier 加权校准数据（Algorithm 1/2）或利用嵌入相似度加权（Algorithm 3），在不完美甚至无 domain 标签的情况下仍能保证覆盖率，并应用于视觉分类和 LLM 幻觉检测。

**[CounselBench: A Large-Scale Expert Evaluation and Adversarial Benchmarking of LLMs in Mental Health QA](llm_nlp/counselbench_llm_mental_health_qa.md)**

:   联合100名持证心理健康专家构建CounselBench双组件基准——CounselBench-EVAL（2,000条六维度专家评估）和CounselBench-Adv（120个对抗性问题+1,080条响应标注），系统性揭示LLM在心理健康开放式问答中表面得分高但存在过度泛化、擅自医疗建议等安全隐患，同时证明LLM-as-Judge在安全关键领域严重不可靠。

**[EAMET: Robust Massive Model Editing via Embedding Alignment Optimization](llm_nlp/eamet_robust_massive_model_editing_via_embedding_alignment_optimization.md)**

:   发现大规模模型编辑失败的根本原因是 key embedding 和 residual embedding 之间的结构不一致（embedding misalignment），提出 EAMET 通过 KL+MSE 双损失渐进式对齐优化，在 6 个 LLM 上平均提升编辑成功率 14%（CounterFact）。

**[Emergent Misalignment is Easy, Narrow Misalignment is Hard](llm_nlp/emergent_misalignment_is_easy_narrow_misalignment_is_hard.md)**

:   研究发现在窄域有害数据上微调会造成广域错位（emergent misalignment），因为"通用错位"比"仅在特定域错位"是更简单高效的参数空间解——通用解的参数范数更小且对噪声更稳定。

**[Enabling Fine-Grained Operating Points for Black-Box LLMs](llm_nlp/enabling_fine-grained_operating_points_for_black-box_llms.md)**

:   发现黑盒 LLM 的语言化概率仅输出 16-23 个唯一值（低基数问题），导致 PR/ROC 曲线粗糙无法精细调优；通过注入参数化噪声和可选的 MLP 校正，将唯一值从 16 个提升到 20,000+，在仅需 1-2 次 API 调用的条件下达到 20 次采样的性能。

**[Enhancing Hallucination Detection through Noise Injection](llm_nlp/enhancing_hallucination_detection_through_noise_injection.md)**

:   在 LLM 中间层的 MLP 激活中注入均匀噪声来近似贝叶斯后验，捕获认知不确定性（epistemic uncertainty），与采样温度捕获的偶然不确定性（aleatoric uncertainty）互补，将 GSM8K 上的幻觉检测 AUROC 从 71.56 提升到 76.14。

**[EvoEngineer: Mastering Automated CUDA Kernel Code Evolution with Large Language Models](llm_nlp/evoengineer_mastering_automated_cuda_kernel_code_evolution_with_large_language_m.md)**

:   提出 EvoEngineer，首个系统化的 LLM-based 代码演化框架，将代码演化分解为 traverse technique（含两层设计：solution guiding + prompt engineering）和 population management 两个正交组件，在 91 个真实 CUDA kernel 上实现最高 2.72× 中位加速比和 69.8% 代码有效率，在性能和正确性两个维度上超越现有方法。

**[FictionalQA: A Dataset for Studying Memorization and Knowledge Acquisition](llm_nlp/fictionalqa_a_dataset_for_studying_memorization_and_knowledge_acquisition.md)**

:   提出 FictionalQA 数据集及生成管线，通过合成关于虚构事件的 webtext 风格文档和 QA 对，在受控环境下研究 LLM 训练中事实记忆与逐字记忆的双重过程，发现更多样的表面形式有助于知识获取而简洁的结构化列表反而最不利于泛化。

**[Fine-Grained Activation Steering: Steering Less, Achieving More](llm_nlp/fine-grained_activation_steering_steering_less_achieving_more.md)**

:   AUSteer 发现块级激活转向（steering）本质上是异质的——不同维度控制不同 token 分布，混合转向既放大有益信号也放大有害信号。提出原子单元（AU）级细粒度转向：用激活动量定位判别性维度，自适应调节转向强度，仅转向 ≤100 个维度即大幅超越转向数千维度的 SOTA 方法。

**[First is Not Really Better Than Last: Evaluating Layer Choice and Aggregation Strategies in Language Model Data Influence Estimation](llm_nlp/first_is_not_really_better_than_last_evaluating_layer_choice_and_aggregation_str.md)**

:   通过理论和实验证明先前工作所推崇的"第一层（embedding）最适合做 influence estimation"的结论是不可靠的，发现中间 attention 层才是更好的估计层，并提出 Rank 和 Vote 两种新的跨层聚合策略以及 Noise Detection Rate (NDR) proxy 指标，显著改善了 LLM 中有害训练样本的检测效果。

**[FlexiCodec: A Dynamic Neural Audio Codec for Low Frame Rates](llm_nlp/flexicodec_a_dynamic_neural_audio_codec_for_low_frame_rates.md)**

:   提出 FlexiCodec，通过 ASR 特征引导的动态帧率合并策略，在 3–12.5Hz 超低帧率下实现高质量语音编解码，同时保持优异的语义信息保留能力。

**[Function Induction and Task Generalization: An Interpretability Study with Off-by-One Addition](llm_nlp/function_induction_and_task_generalization_an_interpretability_study_with_off-by.md)**

:   通过 off-by-one addition（如 1+1=3, 2+2=5）这一反事实任务，利用 path patching 发现大语言模型内部存在 **function induction** 机制——一种超越 token 级别 pattern matching、在函数级别进行归纳推理的注意力头电路，并证明该机制可跨任务复用。

**[GAVEL: Towards Rule-Based Safety through Activation Monitoring](llm_nlp/gavel_towards_rule-based_safety_through_activation_monitoring.md)**

:   提出 GAVEL 框架，将 LLM 安全从"粗粒度误用数据集训练分类器"范式转向"可组合认知元素 (CE) + 布尔规则"范式：定义可解释的激活级原语（如"发出威胁"、"处理支付"），组合为精确的策略规则，实现高精度、可定制、可审计的实时安全监控。

**[How Catastrophic is Your LLM? Certifying Risk in Conversation](llm_nlp/how_catastrophic_is_your_llm_certifying_risk_in_conversation.md)**

:   提出 C3LLM（Certification of Catastrophic risks in multi-turn Conversation for LLMs），首个为多轮 LLM 对话中灾难性风险提供统计认证的框架：用语义相似度图上的 Markov 过程建模对话分布，定义 3 种对话采样策略 + 增强层，使用 Clopper-Pearson 95% 置信区间认证模型产生有害输出的概率界——发现最差模型风险下界高达 72%。

**[How Far Are LLMs from Professional Poker Players? Revisiting Game-Theoretic Reasoning with Agentic Tool Use](llm_nlp/how_far_are_llms_from_professional_poker_players_revisiting_game-theoretic_reaso.md)**

:   系统分析了 LLM 在扑克中的三大推理缺陷（启发式推理、事实误解、知行差距），提出 ToolPoker 框架——首个面向不完全信息博弈的工具集成 LLM 推理系统，通过外部 CFR solver 提供博弈论最优的行动指导，使 7B 模型在 Limit Hold'em 中逼近 Nash 均衡。

**[How Reliable is Language Model Micro-Benchmarking?](llm_nlp/how_reliable_is_language_model_micro-benchmarking.md)**

:   提出 Minimum Detectable Ability Difference (MDAD) 元评估指标，系统揭示了 micro-benchmark 在极小规模下无法可靠区分性能差距小的模型对，且当样本量达到 ~250 时随机采样与精心设计的 micro-benchmark 方法表现相当。

**[Implicit Statistical Inference in Transformers: Approximating Likelihood-Ratio Tests In-Context](llm_nlp/implicit_statistical_inference_in_transformers_approximating_likelihood-ratio_te.md)**

:   从统计决策论视角出发，证明Transformer在上下文学习中能近似Bayes最优的**似然比检验**充分统计量，并通过机制分析揭示模型对线性/非线性任务采用不同深度的自适应电路。

**[In-Context Algebra](llm_nlp/in-context_algebra.md)**

:   本文设计了一个 **in-context 代数任务**——令 token 成为纯变量、每条序列重新随机分配含义——发现 Transformer 在此设定下不再学习经典的傅里叶/几何表示，而是涌现出三种 **符号推理机制**（交换复制、单位元识别、闭包消去），并揭示了训练过程中这些能力按阶段性相变依次出现的规律。

**[KVComm: Enabling Efficient LLM Communication through Selective KV Sharing](llm_nlp/kvcomm_enabling_efficient_llm_communication_through_selective_kv_sharing.md)**

:   提出 KVComm 框架通过选择性共享 KV pairs 实现 LLM 间高效通信，发现 hidden states 存在"信息集中偏差"使其不适合跨模型传递，设计基于注意力重要性 + 高斯先验的层选择策略，仅传输 30% 层即可超越大多数 baseline。

**[Multi-LLM Adaptive Conformal Inference for Reliable LLM Responses](llm_nlp/multi-llm_adaptive_conformal_inference_for_reliable_llm_responses.md)**

:   提出 MACI（Multi-LLM Adaptive Conformal Inference），通过**累积乘积型 conformity score** + **多 LLM 集成**的 factuality 评分 + **组条件校准**，在严格保证用户指定错误率的同时，显著提升 LLM 回复中事实性声明的保留率。

**[Near-Optimal Online Deployment and Routing for Streaming LLMs](llm_nlp/near-optimal_online_deployment_and_routing_for_streaming_llms.md)**

:   研究 LLM 流式服务场景中的在线部署与路由问题：给定一系列随时间变化的查询流，如何动态选择部署哪些模型并将查询路由到合适的模型，以在满足质量约束的同时最小化计算成本，提供达到近似最优竞争比的在线算法。

**[Optimas: Optimizing Compound AI Systems with Globally Aligned Local Rewards](llm_nlp/optimas_optimizing_compound_ai_systems_with_globally_aligned_local_rewards.md)**

:   提出 Optimas 框架，为复合 AI 系统中每个组件学习一个与全局奖励对齐的局部奖励函数 (LRF)，使得异构组件（prompt、模型参数、超参数）可独立优化，同时保证局部改进带来全局性能提升。

**[Preference Leakage: A Contamination Problem in LLM-as-a-judge](llm_nlp/preference_leakage_a_contamination_problem_in_llm-as-a-judge.md)**

:   首次定义并系统研究 LLM-as-a-Judge 中的 **偏好泄漏 (Preference Leakage)** 问题——当合成数据生成器 $M_G$ 与评估器 $M_J$ 存在关联（同模型/继承/同家族）时，评委会对"相关学生模型"产生系统性偏好，同模型场景下 PLS 高达 28.7%（Arena-Hard），且该偏差比自中心偏差更隐蔽、更难检测。

**[Reasoning on Time-Series for Financial Technical Analysis](llm_nlp/reasoning_on_time-series_for_financial_technical_analysis.md)**

:   提出 Verbal Technical Analysis (VTA) 框架，结合 LLM 的语言推理能力与时间序列模型的模式捕捉能力，通过 Time-GRPO 强化学习优化推理链，并以推理属性条件化时序预测，实现了兼具准确性和可解释性的金融时间序列预测。

**[Retrieval-Augmented Generation for Predicting Cellular Responses to Gene Perturbation](llm_nlp/retrieval-augmented_generation_for_predicting_cellular_responses_to_gene_perturb.md)**

:   提出 **PT-RAG**（Perturbation-aware Two-stage Retrieval-Augmented Generation），首次将可微检索增强生成范式应用于单细胞基因扰动响应预测：通过 GenePT 语义检索候选扰动 + Gumbel-Softmax 条件离散采样实现细胞类型感知的端到端检索优化，在 Replogle-Nadig 数据集上超越 STATE 基线（Pearson 0.633 vs 0.624），同时发现朴素 RAG 会严重损害性能（Pearson 仅 0.396），证明**可微且细胞类型感知的检索**在该领域不可或缺。

**[Self-Destructive Language Model](llm_nlp/self-destructive_language_model.md)**

:   提出 Seam，通过耦合良性和有害数据的优化轨迹（使梯度方向相反），将 LLM 转变为"自毁模型"——在有害微调时自动触发灾难性性能崩溃，创造攻击者的两难困境：低强度攻击无效，高强度攻击导致模型报废。

**[SimpleToM: Exposing the Gap between Explicit ToM Inference and Implicit ToM Application in LLMs](llm_nlp/simpletom_exposing_the_gap_between_explicit_tom_inference_and_implicit_tom_appli.md)**

:   SimpleToM 揭示了 LLM 在 Theory of Mind 上的关键缺陷：前沿模型能准确推断他人心理状态（显式 ToM），但在将此知识应用于行为预测和行为判断时性能急剧下降（应用 ToM），暴露了"知道什么"与"如何使用所知"之间的重大鸿沟。

**[Spectral Attention Steering for Prompt Highlighting](llm_nlp/spectral_attention_steering_for_prompt_highlighting.md)**

:   提出 SEKA/AdaSEKA，通过对 key embedding 进行谱分解学习"相关性子空间"，在注意力计算前直接编辑 key 向量来实现 prompt highlighting，无需存储完整注意力矩阵，与 FlashAttention 完全兼容，且开销极低（+0.03s/sample）。

**[Statistical Advantage of Softmax Attention: Insights from Single-Location Regression](llm_nlp/statistical_advantage_of_softmax_attention_insights_from_single-location_regress.md)**

:   通过提出"单位置回归"(Single-Location Regression, SLR) 理论框架，结合统计物理中的 order parameter 方法，在高维极限下严格证明了 softmax attention 在种群层面达到 Bayes 风险而线性 attention 本质上无法做到，并在有限样本情形下证实 softmax 始终优于线性 attention，为 softmax 在检索任务中的优势提供了首个原理性解释。

**[Sublinear Time Quantum Algorithm for Attention Approximation](llm_nlp/sublinear_time_quantum_algorithm_for_attention_approximation.md)**

:   提出首个对序列长度 $n$ 具有**亚线性**时间复杂度的量子数据结构，用于近似 Transformer 注意力矩阵的行查询，预处理时间 $\widetilde{O}(\epsilon^{-1} n^{0.5} \cdot \text{poly}(d, s_\lambda, \alpha))$，每次行查询 $\widetilde{O}(s_\lambda^2 + s_\lambda d)$，相对经典算法实现了关于 $n$ 的二次加速。

**[Talk, Evaluate, Diagnose: User-aware Agent Evaluation with Automated Error Analysis](llm_nlp/talk_evaluate_diagnose_user-aware_agent_evaluation_with_automated_error_analysis.md)**

:   提出 TED 框架（Talk-Evaluate-Diagnose），通过可复用的专家/非专家 persona 模板、基于 grading notes 的 LLM-as-judge 评估和自动化错误分析，实现跨领域的用户感知型 Agent 评估。

**[The Lattice Representation Hypothesis of Large Language Models](llm_nlp/the_lattice_representation_hypothesis_of_large_language_models.md)**

:   提出 LLM 的**格表示假说 (Lattice Representation Hypothesis)**：通过将线性表示假说与形式概念分析 (FCA) 统一，证明 LLM 嵌入空间中的属性方向通过半空间交集隐式编码了一个**概念格 (concept lattice)**，从而实现了连续几何与符号抽象之间的桥接。

**[Token-Efficient Item Representation via Images for LLM Recommender Systems](llm_nlp/token-efficient_item_representation_via_images_for_llm_recommender_systems.md)**

:   提出 I-LLMRec，利用商品图像替代冗长文本描述来表示推荐系统中的物品语义，通过 RISA 对齐模块和 RERI 检索模块，在仅用单个token表示物品的同时保留丰富语义，推理速度提升约2.93倍且推荐性能超越文本描述方法。

**[Trapped by simplicity: When Transformers fail to learn from noisy features](llm_nlp/trapped_by_simplicity_when_transformers_fail_to_learn_from_noisy_features.md)**

:   研究表明 Transformer 在从含特征噪声的数据中学习布尔函数时会失败——其简单性偏好（倾向学习低敏感度函数）导致模型被困在比目标函数更简单的最优噪声预测器上，无法恢复真实的无噪声目标函数。

**[Truthfulness Despite Weak Supervision: Evaluating and Training LLMs Using Peer Prediction](llm_nlp/truthfulness_despite_weak_supervision_evaluating_and_training_llms_using_peer_pr.md)**

:   提出将博弈论中的 Peer Prediction 机制应用于 LLM 评估和训练，通过衡量参与者答案的互预测性来区分诚实与欺骗回答，无需真值标签即可实现诚实性激励，展现出惊人的"逆向缩放"特性——专家越弱反而越能抵抗强模型的欺骗。

**[UIS-Digger: Towards Comprehensive Research Agent Systems for Real-world Unindexed Information Seeking](llm_nlp/uis-digger_towards_comprehensive_research_agent_systems_for_real-world_unindexed.md)**

:   识别并形式化"未索引信息检索"(UIS) 问题，提出首个 UIS 基准 UIS-QA (110 题) 和多 Agent 框架 UIS-Digger，用 ~30B 模型超越集成 O3/GPT-4.1 的系统。

**[Universal Properties of Activation Sparsity in Modern Large Language Models](llm_nlp/universal_properties_of_activation_sparsity_in_modern_large_language_models.md)**

:   系统研究现代 LLM（GLU架构）的激活稀疏性，提出 top-p 稀疏化框架和临界稀疏度指标，发现稀疏度随模型规模增大（Gemma3-70B 约 70%），揭示更大模型有更多可利用的神经元冗余。

**[WebDevJudge: Evaluating (M)LLMs as Critiques for Web Development Quality](llm_nlp/webdevjudge_mllm_web_development.md)**

:   构建 WebDevJudge 元评估基准，系统评估 LLM/MLLM 及智能体工作流在 Web 开发质量评估任务上作为裁判的能力，发现当前最强模型与人类专家之间仍存在约15%的一致率差距，并揭示了功能等价识别失败和可行性验证薄弱两大根本瓶颈。

**[Weight Decay may matter more than μP for Learning Rate Transfer in Practice](llm_nlp/weight_decay_may_matter_more_than_mup_for_learning_rate_transfer_in_practice.md)**

:   大规模实证研究表明 μP 的核心对齐假设在实际 LLM 训练中仅在开始时短暂成立，之后是 independent weight decay（而非 μP）正确稳定了不同宽度模型间的特征学习动态，使得学习率迁移成为可能。μP 的实际作用被重新解释为一种隐式学习率 warmup。

**[Whatever Remains Must Be True: Filtering Drives Reasoning in LLMs, Shaping Diversity](llm_nlp/whatever_remains_must_be_true_filtering_drives_reasoning_in_llms_shaping_diversi.md)**

:   提出 DMVR 框架和 α-DPG 算法，通过显式定义"过滤掉错误答案"的目标分布并用 α-散度族来逼近，统一了 RLVR（Reverse KL）和拒绝采样微调（Forward KL），在 Lean 定理证明上实现了精度-覆盖率 Pareto 前沿的最优表现。

**[When to Ensemble: Identifying Token-Level Points for Stable and Fast LLM Ensembling](llm_nlp/when_to_ensemble_identifying_token-level_points_for_stable_and_fast_llm_ensembli.md)**

:   提出 SAFE（Stable And Fast LLM Ensembling），通过 Generate-Verify-Ensemble 循环在 token 级别选择性地集成多个异构分词器 LLM，解决长序列生成中分词不匹配导致的 OOV-like 污染问题，仅在不到 1% 的 token 上集成即可提升效果，MATH500 上将 UniTE 从 59.6% 提升到 77.4%。

---

## 📂 其他 { #others }

**[A Federated Generalized Expectation-Maximization Algorithm for Mixture Models with an Unknown Number of Components](others/a_federated_generalized_expectation-maximization_algorithm_for_mixture_models_wi.md)**

:   提出 FedGEM 算法，通过客户端本地 EM 步后构建不确定性集、服务器利用不确定性集交集检测聚类重叠并推断全局聚类数，首次实现在全局聚类数未知情况下的联邦聚类，并提供了概率收敛保证。

**[A Law of Data Reconstruction for Random Features (and Beyond)](others/a_law_of_data_reconstruction_for_random_features_and_beyond.md)**

:   从信息论和代数角度证明随机特征模型中存在数据重构定律：当参数量 $p \gg dn$（$d$ 为数据维度，$n$ 为样本数）时，训练数据可被完整重构，并通过投影损失优化方法在 RF、两层网络和 ResNet 上验证了该阈值的普适性。

**[A Representer Theorem for Hawkes Processes via Penalized Least Squares Minimization](others/a_representer_theorem_for_hawkes_processes_via_penalized_least_squares_minimizat.md)**

:   为线性多元 Hawkes 过程在 RKHS 框架下的触发核估计建立了新型表示定理，证明最优估计器可用等价核在数据点上的线性组合表示且对偶系数全部解析地等于 1，无需求解对偶优化问题，从而实现高效可扩展的非参数估计。

**[A Scalable Inter-edge Correlation Modeling in CopulaGNN for Link Sign Prediction](others/a_scalable_inter-edge_correlation_modeling_in_copulagnn_for_link_sign_prediction.md)**

:   将 CopulaGNN 从节点级扩展到边级，通过将相关矩阵构造为边嵌入的 Gramian 矩阵并利用 Woodbury 恒等式重构条件概率分布，实现了在签名图上对边间统计依赖的可扩展建模，用于链接符号预测任务。

**[A Single Architecture for Representing Invariance Under Any Space Group](others/a_single_architecture_for_representing_invariance_under_any_space_group.md)**

:   设计了一种可自适应任意空间群不变性的单一架构 (Crystal Fourier Transformer)，通过解析推导群操作对傅里叶系数的约束来构造对称适配的傅里叶基，用约束的对偶图表示实现了跨 230 个空间群的参数共享和零样本泛化。

**[Accessible, Realistic, and Fair Evaluation of Positive-Unlabeled Learning Algorithms](others/accessible_realistic_and_fair_evaluation_of_positive-unlabeled_learning_algorith.md)**

:   提出首个 PU 学习统一基准，系统解决两个关键问题：(1) 用代理准确率和代理 AUC 实现无负样本的模型选择；(2) 发现并通过将正样本并入无标签集的简单校准方法解决单样本设置下的内部标签偏移问题，使双样本算法在单样本评估中得到公平比较。

**[Active Learning for Decision Trees with Provable Guarantees](others/active_learning_for_decision_trees_with_provable_guarantees.md)**

:   为决策树主动学习提供首个理论保证：(1) 首次分析决策树的不一致系数（disagreement coefficient）并给出 $O(\ln^{OPT}(n))$ 上界；(2) 提出首个达到乘法误差 $(1+\epsilon)$ 保证的二分类主动学习算法；结合两者实现数据集大小的多对数标签复杂度。

**[Addressing Divergent Representations from Causal Interventions on Neural Networks](others/addressing_divergent_representations_causal.md)**

:   系统性地揭示因果干预（activation patching、DAS、SAE 等）会将模型内部表征推离自然分布，理论区分"无害偏移"与"有害偏移"两类情况，并提出 Counterfactual Latent (CL) loss 来约束干预表征不偏离流形，在 7B LLM 上验证可减少偏移同时保持干预准确率。

**[Agnostics: Learning to Synthesize Code in Any Programming Language with a Universal RL Environment](others/agnostics_learning_to_code_in_any_programming_language_via_reinforcement_with_a_.md)**

:   提出Agnostics，一种语言无关的后训练pipeline：将编程任务统一为I/O行为规范格式，用通用验证器+GRPO强化学习训练LLM在任何编程语言上编码，使Qwen 4B在Lua/Julia/R/OCaml/Fortran五种低资源语言上达到匹敌16B-70B模型的SOTA水平。

**[An Information-Theoretic Framework For Optimizing Experimental Design To Distinguish Probabilistic Neural Codes](others/an_information-theoretic_framework_for_optimizing_experimental_design_to_disting.md)**

:   提出"信息间隙"（information gap）框架，通过优化刺激分布来最大化似然编码（likelihood code）与后验编码（posterior code）假设之间的可区分性，推导出真实后验与任务边缘化代理后验之间的KL散度作为优化目标，并通过DNN解码器在模拟神经群体上验证了该框架的有效性，揭示传统单上下文实验无法区分两种编码假设。

**[AnesSuite: A Comprehensive Benchmark and Dataset Suite for Anesthesiology Reasoning](others/anessuite_a_comprehensive_benchmark_and_dataset_suite_for_anesthesiology_reasoni.md)**

:   构建首个面向麻醉学推理的综合数据集套件AnesSuite——包括AnesBench（7972道双语选择题）、AnesCorpus（240万篇文档语料库）、AnesQA（2万条QA对）和AnesR1（1万条CoT推理数据），提出三级认知需求分类（System 1/1.x/2），训练的Morpheus模型（Qwen2.5 + SFT + GRPO）在7B参数下达到14B基线性能，揭示当前最强模型在复杂推理（System 2）上仍低于0.6。

**[ANO: Faster is Better in Noisy Landscapes](others/ano_faster_is_better_in_noisy_landscape.md)**

:   提出 Ano 优化器，将更新方向和幅度解耦——方向用动量的符号（sign）确保噪声鲁棒，幅度用瞬时梯度绝对值（而非动量幅度）确保响应速度，配合改进的 Yogi 式方差估计，在噪声和非平稳环境（如 RL）中显著优于 Adam/Lion/Adan，同时在标准任务上保持竞争力。

**[AnyUp: Universal Feature Upsampling](others/anyup_universal_feature_upsampling.md)**

:   提出AnyUp——首个推理时encoder无关的可学习特征上采样方法，通过feature-agnostic层处理任意维度/类型的视觉特征，配合窗口注意力架构和crop-based训练策略，训练一次即可对任意视觉编码器（DINO/CLIP/SigLIP/MAE等）的特征进行任意分辨率上采样，在多个下游任务上超越FeatUp/JAFAR/LoftUp等方法。

**[Articulation in Motion: Prior-Free Part Mobility Analysis for Articulated Objects](others/articulation_in_motion_prior-free_part_mobility_analysis_for_articulated_objects.md)**

:   提出AiM（Articulation in Motion）框架，从交互视频和初始状态扫描中无需部件数量先验地重建铰接物体——通过双高斯表征（静态GS + 可变形GS）实现动静解耦，结合顺序RANSAC进行无先验部件分割和关节估计，辅以SDMD模块处理新暴露的静态区域，在复杂6部件物体（Storage）上以79.34% mean IoU大幅超越需先验的ArtGS（52.23%）。

**[Assess A Semantic And Structural Evaluation Framework For Statement Similarity](others/assess_a_semantic_and_structural_evaluation_framework_for_statement_similarity.md)**

:   提出 TransTED Similarity，一种基于算子树 (Operator Tree) 和语义变换增强的树编辑距离指标，用于评估自动形式化 (autoformalization) 生成的形式化数学命题与参考命题之间的语义相似度，并构建了 EPLA 基准数据集。

**[AstaBench: Rigorous Benchmarking of AI Agents with a Scientific Research Suite](others/astabench_benchmarking_ai_agents.md)**

:   由 AI2 团队构建的首个端到端科学研究 Agent 基准 AstaBench，包含 2400+ 问题覆盖科学发现全流程，配备生产级可复现搜索工具，评估了 57 个 Agent（22 类），发现尽管单任务有进展但 AI 距离完整科学研究助手仍很远，同时系统性修复先前基准的 5 大方法学缺陷。

**[Behavior Learning (BL): Learning Hierarchical Optimization Structures from Data](others/behavior_learning_bl_learning_hierarchical_optimization_structures_from_data.md)**

:   受行为科学中效用最大化范式启发，提出 Behavior Learning (BL) 框架，将数据建模为由可解释的模块化效用最大化问题（UMP）层次组合所诱导的 Gibbs 分布，在预测性能、内在可解释性和参数可辨识性三者之间实现了统一。

**[Block-Sample MAC-Bayes Generalization Bounds](others/block-sample_mac-bayes_generalization_bounds.md)**

:   提出块样本MAC-Bayes泛化界（mean approximately correct），将训练数据划分为J个块后用各块条件下的KL散度之和替代整体KL散度，在确定性学习算法（如均值估计）等原始PAC-Bayes界为空（vacuous）的场景下仍能给出有限、有意义的泛化误差界，并证明了该界的高概率版本在一般情况下不可行。

**[Can You Hear Me Now? A Benchmark for Long-Range Graph Propagation and Beyond](others/can_you_hear_me_now_a_benchmark_for_long-range_graph_propagation_and_beyond.md)**

:   本文提出 ECHO 基准，包含 3 个合成任务和 2 个基于密度泛函理论（DFT）的真实化学任务，要求图神经网络在 17–40 跳范围内有效传播信息，系统评估了 11 种 GNN 架构的长程传播能力。

**[CHLU: The Causal Hamiltonian Learning Unit as a Symplectic Primitive for Deep Learning](others/chlu_the_causal_hamiltonian_learning_unit_as_a_symplectic_primitive_for_deep_lea.md)**

:   CHLU 是一种基于相对论哈密顿力学和辛积分的计算学习原语，通过强制相空间体积守恒和引入因果速度上限，解决了 LSTM 的梯度爆炸/消失和 Neural ODE 的信息耗散问题，实现无限时域稳定性和热力学生成能力。

**[DREAM: Completing Missing Annotation via Multi-Agent Debate for Accurate and Scalable Relevance Assessment](others/completing_missing_annotation_multi-agent_debate_for_accurate_and_scalable_relev.md)**

:   提出 DREAM 框架，用多轮 LLM 辩论（对立立场）来完成 IR 基准中大量缺失的相关性标注，发现了原有标注中 428% 的额外相关文档，仅需 3.5% 的人类介入率即达到 95.2% 准确率。

**[Compositional Diffusion with Guided Search for Long-Horizon Planning](others/compositional_diffusion_long_horizon_planning.md)**

:   提出 CDGS（Compositional Diffusion with Guided Search），通过在扩散去噪过程中嵌入基于种群的搜索机制（迭代重采样 + 似然剪枝），解决组合式扩散模型在多模态局部分布合成时的模式平均问题，从短时域模型采样出全局一致的长时域规划。

**[Decomposing Representation Space into Interpretable Subspaces with Unsupervised Learning](others/decomposing_representation_space_into_interpretable_subspaces_with_unsupervised_.md)**

:   提出 NDM（Neighbor Distance Minimization），通过最小化子空间内的近邻距离来无监督地找到神经网络表征空间中的可解释非基对齐子空间，在 GPT-2 到 Qwen2.5-1.5B 上发现了参数化知识与上下文知识的分离子空间。

**[Directional Sheaf Hypergraph Networks: Unifying Learning on Directed and Undirected Hypergraphs](others/directional_sheaf_hypergraph_networks_unifying_learning_on_directed_and_undirect.md)**

:   本文提出 Directional Sheaf Hypergraph Networks (DSHN)，通过将 Cellular Sheaf 理论与有向超图的方向信息结合，构造了一种复值 Hermitian Laplacian 算子，统一并推广了现有的图和超图 Laplacian，在 7 个真实数据集上相对准确率提升 2%–20%。

**[DA-AC: Distributions as Actions — A Unified RL Framework for Diverse Action Spaces](others/distributions_as_actions_a_unified_framework_for_diverse_action_spaces.md)**

:   DA-AC 提出将动作分布的参数（如 softmax 概率或 Gaussian 均值/方差）作为 Agent 的"动作"输出，将动作采样过程移入环境，从而用统一的确定性策略梯度框架处理离散/连续/混合动作空间，理论证明方差严格低于 LR 和 RP 估计器，并在 40+ 环境上取得 competitive 或 SOTA 性能。

**[Enhancing Generative Auto-bidding with Offline Reward Evaluation and Policy Search](others/enhancing_generative_auto_bidding.md)**

:   提出 AIGB-Pearl，为生成式自动竞价方法引入离线轨迹评估器和 KL-Lipschitz 约束的分数最大化方案，使生成模型能在理论保证下安全地突破静态离线数据的性能天花板，在淘宝真实广告系统上实现 GMV +3% 的显著提升。

**[Entropic Confinement and Mode Connectivity in Overparameterized Neural Networks](others/entropic_confinement_and_mode_connectivity_in_overparameterized_neural_networks.md)**

:   揭示了过参数化神经网络损失景观中低损失连接路径上的**曲率变化**产生熵力壁垒，解释了为何SGD被限制在单一盆地内，尽管不同极小值之间能量上相连。

**[Exchangeability of GNN Representations with Applications to Graph Retrieval](others/exchangeability_gnn_representations.md)**

:   发现训练好的 GNN 节点嵌入沿特征维度是**可交换随机变量**（即 $p(X) = p(X\pi)$ 对任意维度排列 $\pi$），利用此性质通过维度排序将基于传输距离的图相似度近似为欧氏距离，构建高效的局部敏感哈希（LSH）框架 GraphHash，在子图匹配和图编辑距离检索任务上超越基线，可扩展到 100 万图语料库。

**[FIRE: Frobenius-Isometry Reinitialization for Balancing the Stability-Plasticity Tradeoff](others/fire_frobenius_isometry_reinitialization.md)**

:   将持续学习中的稳定性-可塑性平衡形式化为约束优化问题——最小化权重偏差（稳定性）同时约束权重正交性（可塑性），得到正交 Procrustes 问题的闭式解 $\tilde{W}^* = W(W^\top W)^{-1/2}$（极分解），通过 Newton-Schulz 迭代高效实现（<1% 额外时间），在视觉持续学习、LLM 持续预训练和 RL 上全面超越 S&P 等基线。

**[From Movement to Cognitive Maps: RNNs Reveal How Locomotor Development Shapes Hippocampal Spatial Coding](others/from_movement_to_cognitive_maps.md)**

:   结合幼鼠运动发育的计算分析和浅层 RNN 模型，证明运动统计特征的发育变化（爬行→行走→奔跑→成年）驱动了空间调谐神经元的序贯涌现，复现了大鼠海马空间编码的发育时间线，且具体的发育运动统计（而非简单的感觉输入加速）是位置中心空间表征涌现的关键。

**[Gaussian Certified Unlearning in High Dimensions: A Hypothesis Testing Approach](others/gaussian_certified_unlearning.md)**

:   提出 $(\phi,\varepsilon)$-Gaussian certifiability——基于假设检验 trade-off 函数的高维机器遗忘隐私框架，严格证明在高维比例体系 ($p \sim n$) 下单步 Newton 更新 + 校准高斯噪声即可同时满足隐私 (GPAR) 和精度 (GED→0) 要求，推翻了 Zou et al. (2025) "至少需两步 Newton" 的结论，并从理论上揭示旧 $\varepsilon$-certifiability 与噪声添加机制不兼容的根本原因。

**[Implicit Bias of Per-sample Adam on Separable Data: Departure from the Full-batch Regime](others/implicit_bias_of_per-sample_adam_on_separable_data_departure_from_the_full-batch.md)**

:   分析逐样本 Adam（batch size=1）的隐含偏置，证明其在可分数据上失去全批 Adam 的 $\ell_\infty$ 间隔保证，收敛到数据依赖解，收敛率 $O(1/\sqrt{t})$，揭示batch size≥2即可恢复 $\ell_\infty$ 偏置。

**[In-Context Algebra](others/in-context_algebra.md)**

:   本文设计了一个 **in-context 代数任务**——令 token 成为纯变量、每条序列重新随机分配含义——发现 Transformer 在此设定下不再学习经典的傅里叶/几何表示，而是涌现出三种 **符号推理机制**（交换复制、单位元识别、闭包消去），并揭示了训练过程中这些能力按阶段性相变依次出现的规律。

**[Intrinsic Training Dynamics of Deep Neural Networks](others/intrinsic_training_dynamics_of_deep_neural_networks.md)**

:   本文研究深度神经网络梯度流训练中，参数空间的轨迹何时可以被"提升"到低维本征空间并表示为内禀的黎曼梯度流，提出了基于守恒律的内禀可恢复性（intrinsic recoverability）准则，并将结果推广到任意深度的 ReLU 网络和线性网络。

**[Jackpot: Optimal Budgeted Rejection Sampling for Extreme Actor-Policy Mismatch RL](others/jackpot_optimal_budgeted_rejection_sampling_for_extreme_actor-policy_mismatch_re.md)**

:   提出 Jackpot 框架，通过 Optimal Budget Rejection Sampling（OBRS）以可控接受预算在 token 级别拒绝/重加权 rollout 样本，理论证明任意预算下都能严格缩小 actor-policy 间 KL 散度，配合 rollout 模型联合训练与蒸馏，使小模型（如 Qwen3-1.7B）rollout 训练大模型（如 Qwen3-8B）达到接近 on-policy 的性能。

**[Latent Fourier Transform](others/latent_fourier_transform.md)**

:   将扩散自编码器与潜在空间 DFT 结合，在潜在时间序列表征上应用傅里叶变换按时间尺度分离音乐模式，训练时使用随机相关对数频率掩码让解码器学习从部分频谱信息重建，推理时用户指定频率掩码控制保留/混合的时间尺度，在条件生成和音乐融合任务上超越 ILVR/guidance/codec filtering/RAVE 等基线，29 名音乐家的听力测试确认其音质和融合能力优越。

**[LPWM: Latent Particle World Models for Object-Centric Stochastic Dynamics](others/latent_particle_world_models_self-supervised_object-centric_stochastic_dynamics_.md)**

:   LPWM 是首个能扩展到真实世界多物体数据集的自监督物体中心世界模型，通过为每个物体学习独立的潜在动作（per-particle latent actions）来建模多物体的独立随机动力学，支持动作/语言/图像目标多种条件生成。

**[Learning Adaptive Distribution Alignment with Neural Characteristic Function for Graph Domain Adaptation](others/learning_adaptive_distribution_alignment_with_neural_characteristic_function_for.md)**

:   提出**ADAlign**框架，利用**神经谱差异(NSD)**在频域自适应对齐源/目标图嵌入分布，通过可学习频率采样器自动发现并优先对齐每个迁移场景中最关键的分布差异维度。

**[Learning on a Razor's Edge: Identifiability and Singularity of Polynomial Neural Networks](others/learning_on_a_razors_edge_identifiability_and_singularity_of_polynomial_neural_n.md)**

:   用代数几何方法研究多项式激活神经网络的可辨识性，证明 MLP 具有一般有限可辨识性（非唯一），CNN 可达到唯一可辨识性，揭示 FPT 奇异性与参数计数的联系。

**[LipNeXt: Scaling up Lipschitz-based Certified Robustness to Billion-parameter Models](others/lipnext_scaling_up_lipschitz-based_certified_robustness_to_billion-parameter_mod.md)**

:   提出 LipNeXt——首个无约束、无卷积的 1-Lipschitz 架构，通过流形优化（直接在正交流形上更新）和 Spatial Shift Module（理论证明唯一保范 depthwise 卷积是 ±1 位移）突破 Lipschitz 网络的 scaling 瓶颈，首次将认证鲁棒性扩展到 10 亿参数，在 CIFAR-10/100/ImageNet 上达 SOTA 认证鲁棒准确率。

**[Missing Mass for Differentially Private Domain Discovery](others/missing_mass_for_differentially_private_domain_discovery.md)**

:   为差分隐私下的域发现问题提供首个原则性算法——结合加权高斯机制和经典"missing mass"估计器，证明在 VC 维参数化下的 FPT 算法和匹配的下界（$\Omega(n\log n)$）。

**[Mitigating Spurious Correlation via Distributionally Robust Learning with Hierarchical Ambiguity Sets](others/mitigating_spurious_correlation_via_distributionally_robust_learning_with_hierar.md)**

:   提出层次化歧义集的分布鲁棒优化方法，同时建模组间比例变化和组内分布偏移（Wasserstein 球），在少数群组内部分布偏移场景下显著优于 Group DRO（CelebA shifted: 56.3%→72.1% worst-group accuracy）。

**[Modal Logical Neural Networks for Financial AI](others/modal_logical_neural_networks_for_financial_ai.md)**

:   提出模态逻辑神经网络（MLNN），将 Kripke 语义（必然/可能模态算子）集成到神经网络中，在金融合同安全审查、洗售合规和市场串谋检测中实现可审计的逻辑推理与深度学习性能的结合。

**[MT-DAO: Multi-Timescale Distributed Adaptive Optimizers with Local Updates](others/mt-dao_multi-timescale_distributed_adaptive_optimizers_with_local_updates.md)**

:   提出 MT-DAO，一种多时间尺度分布式自适应优化器，通过引入慢动量（高 $\beta$）来解决低频通信训练中标准动量衰减过快导致的时间尺度失配问题，首次提供了收敛保证，在语言模型预训练中消除了与全同步 DDP 的性能差距，同时减少 6-27% 的端到端训练时间。

**[NIMO: a Nonlinear Interpretable MOdel](others/nimo_a_nonlinear_interpretable_model.md)**

:   NIMO 提出一种混合模型 $y = \sum_j x_j \beta_j (1 + g_{\mathbf{u}_j}(\mathbf{x}_{-j}))$，在保留线性回归系数全局可解释性（通过均值边际效应 MEM）的同时，利用神经网络提供逐实例的非线性修正，并通过参数消去法高效联合优化线性系数和网络参数。

**[Noisy-Pair Robust Representation Alignment for Positive-Unlabeled Learning](others/noisy-pair_robust_representation_alignment_for_positive-unlabeled_learning.md)**

:   提出**NcPU**框架，通过**噪声对鲁棒的非对比损失(NoiSNCL)** 在不可靠监督下对齐类内表示，配合**幻影标签消歧(PLD)** 迭代优化伪标签，无需辅助负样本或预估类先验即可在PU学习中逼近甚至超越有监督性能。

**[Non-Clashing Teaching in Graphs: Algorithms, Complexity, and Bounds](others/non-clashing_teaching_in_graphs_algorithms_complexity_and_bounds.md)**

:   研究图上非冲突教学问题的算法和复杂度——通过邻域查询识别目标节点，提供 VC 维参数化的 FPT 算法和紧的 $\Omega(n\log n)$ 下界，对树/平面图给出多项式界。

**[Non-Collaborative User Simulators for Tool Agents](others/non-collaborative_user_simulators_for_tool_agents.md)**

:   提出非协作用户模拟器框架，定义四类真实非协作行为（不可用服务/跑题/不耐烦/不完整表述），揭示当前工具 Agent 面对非协作用户时显著退化（跑题平均降 29.1%），并证明混合训练可提升鲁棒性至 93.5%。

**[Predicting kernel regression learning curves from only raw data statistics](others/predicting_kernel_regression_learning_curves_from_only_raw_data_statistics.md)**

:   提出 Hermite 特征结构假设（HEA），仅从协方差矩阵（无需训练数据）推导核回归学习曲线的闭式预测公式，对高斯数据预测误差 <5%，支持多种核族和高效模型选择。

**[Speculative Actions: A Lossless Framework for Faster AI Agents](others/speculative_actions_faster_ai_agents.md)**

:   借鉴 CPU 推测执行和 LLM 推测解码的思想，提出 Speculative Actions 框架：在慢速 Actor（大模型）计算时用快速 Speculator（小模型）预测未来动作并预执行，匹配时跳过等待实现无损加速，在 Chess/电商/问答等场景实现 15-30% 延迟降低，置信度动态分支策略用 40% 更少 token 达到近似 3 条推测的加速效果。

**[The Hot Mess of AI: How Does Misalignment Scale With Model Intelligence and Task Complexity?](others/the_hot_mess_of_ai_how_does_misalignment_scale_with_model_intelligence_and_task_.md)**

:   通过 bias-variance 分解量化 AI 模型的错误模式，发现随着推理链变长和任务变难，模型失败变得更加"不连贯"（variance 主导）而非"系统性错误"（bias 主导），暗示未来 AI 风险更像工业事故而非一致性目标追求。

---

## 🎨 图像生成 { #image_generation }

**[A Hidden Semantic Bottleneck in Conditional Embeddings of Diffusion Transformers](image_generation/a_hidden_semantic_bottleneck_in_conditional_embeddings_of_diffusion_transformers.md)**

:   对扩散 Transformer 的条件嵌入进行首次系统分析，发现极端的角度相似性（类间余弦相似度>99%）和维度稀疏性（仅 1-2% 的维度携带语义信息），裁剪掉 2/3 的低幅维度后生成质量基本不变，揭示了条件嵌入中隐藏的语义瓶颈。

**[AlignTok: Aligning Visual Foundation Encoders to Tokenizers for Diffusion Models](image_generation/aligntok_aligning_visual_foundation_encoders_to_tokenizers_for_diffusion_models.md)**

:   提出 AlignTok，将预训练视觉基础编码器（如 DINOv2）对齐为扩散模型的连续 tokenizer，通过三阶段对齐策略（语义潜空间建立→感知细节补充→解码器精炼）构建语义丰富的潜空间，在 ImageNet 256×256 上 64 epochs 即达 gFID 1.90，比从头训练 VAE 收敛更快、生成质量更好。

**[Amortising Inference and Meta-Learning Priors in Neural Networks (BNNP)](image_generation/amortising_inference_and_meta-learning_priors_in_neural_networks.md)**

:   提出 BNNP（Bayesian Neural Network Process），一种将 BNN 权重作为隐变量、BNN 本身作为解码器的 neural process，通过逐层 amortised variational inference 在多数据集上联合学习 BNN 先验和推断网络，首次回答了"在良好先验下，近似推断方法还重要吗？"——答案是肯定的，没有免费午餐。

**[Asynchronous Denoising Diffusion Models for Aligning Text-to-Image Generation](image_generation/asynchronous_denoising_diffusion_models_for_aligning_text-to-image_generation.md)**

:   AsynDM 通过为不同像素分配不同的时间步调度（prompt 相关区域去噪更慢），使其能利用更清晰的上下文参考，从而在不需要微调的情况下显著提升文图生成的语义对齐。

**[Beyond Confidence: The Rhythms of Reasoning in Generative Models](image_generation/beyond_confidence_the_rhythms_of_reasoning_in_generative_models.md)**

:   提出 Token Constraint Bound ($\delta_{\text{TCB}}$) 指标，通过量化 LLM 隐状态在多大扰动范围内能保持 next-token 预测不变，来度量预测的局部鲁棒性，揭示了传统 perplexity 无法捕捉的预测不稳定性。

**[Blueprint-Bench: Comparing Spatial Intelligence of LLMs, Agents and Image Models](image_generation/blueprint-bench_comparing_spatial_intelligence_of_llms_agents_and_image_models.md)**

:   Blueprint-Bench 通过"从公寓内部照片生成 2D 平面图"的任务来评测 AI 模型的空间推理能力，结果显示大多数 LLM、图像生成模型和 Agent 系统的表现接近或低于随机基线，揭示了当前 AI 在空间智能上的重大盲区。

**[Bridging Degradation Discrimination and Generation for Universal Image Restoration](image_generation/bridging_degradation_discrimination_and_generation_for_universal_image_restorati.md)**

:   BDG 通过多角度多尺度灰度共生矩阵（MAS-GLCM）进行细粒度退化判别，并设计三阶段扩散训练（生成→桥接→修复）将退化判别能力与生成先验无缝融合，在 all-in-one 修复和真实世界超分辨率任务上取得显著的保真度提升。

**[Bridging Generalization Gap of Heterogeneous Federated Clients Using Generative Models](image_generation/bridging_generalization_gap_of_heterogeneous_federated_clients_using_generative_.md)**

:   FedVTC 提出在模型异构联邦学习中，各客户端通过变分转置卷积网络（VTC）从聚合的特征分布统计量中生成合成数据来微调本地模型，无需公共数据集即可显著提升泛化能力，同时降低通信和内存开销。

**[CMT: Mid-Training for Efficient Learning of Consistency, Mean Flow, and Flow Map Models](image_generation/cmt_mid-training_for_efficient_learning_of_consistency_mean_flow_and_flow_map_mo.md)**

:   提出 Consistency Mid-Training (CMT)，在预训练扩散模型和 flow map 后训练之间插入一个轻量级中间训练阶段，通过让模型学习将 ODE 轨迹上的任意点映射回干净样本来获得轨迹对齐的初始化，从而大幅降低训练成本（最多 98%）并达到 SOTA 两步生成质量。

**[Compose Your Policies! Improving Diffusion-based or Flow-based Robot Policies via Test-time Distribution-level Composition](image_generation/compose_your_policies_improving_diffusion-based_or_flow-based_robot_policies_via.md)**

:   提出 General Policy Composition (GPC)，在测试时通过凸组合多个预训练扩散/Flow 策略的分布分数（score），无需额外训练即可产生超越任何单一父策略的更强策略，理论证明凸组合可改善单步分数误差且通过 Grönwall 界传播到全程轨迹。

**[Condition Errors Refinement in Autoregressive Image Generation with Diffusion Loss](image_generation/condition_errors_refinement_in_autoregressive_image_generation_with_diffusion_lo.md)**

:   理论分析了自回归扩散损失模型相比条件扩散模型在条件误差修正上的优势（梯度范数指数衰减），并提出基于最优传输（Wasserstein Gradient Flow）的条件精炼方法来解决自回归过程中的"条件不一致性"问题，在 ImageNet 上达到 FID 1.31（基于 MAR）。

**[Consistent Text-to-Image Generation via Scene De-Contextualization](image_generation/consistent_text-to-image_generation_via_scene_de-contextualization.md)**

:   揭示 T2I 模型中 ID 偏移的根本原因是"场景上下文化"（scene contextualization，场景 token 对 ID token 注入上下文信息），并提出 training-free 的 Scene De-Contextualization (SDeC) 方法，通过 SVD 特征值的方向稳定性分析识别并抑制 prompt embedding 中潜在的场景-ID 关联，实现逐场景的身份一致性生成。

**[ContextBench: Modifying Contexts for Targeted Latent Activation](image_generation/contextbench_modifying_contexts_for_targeted_latent_activation.md)**

:   提出 ContextBench 基准（715 个任务）评估自动生成流畅且能激活特定潜在特征的输入文本的方法，并开发两种 EPO 增强变体（LLM辅助和扩散模型修补），在激活强度和语言流畅度的权衡上 Pareto 优于标准 EPO。

**[Continual Unlearning for Text-to-Image Diffusion Models: A Regularization Perspective](image_generation/continual_unlearning_for_text-to-image_diffusion_models_a_regularization_perspec.md)**

:   首次系统研究 T2I 扩散模型的持续遗忘（continual unlearning）问题，发现现有遗忘方法在序列请求下因累积参数漂移导致"效用崩溃"，提出一组附加正则化策略（L1/L2 范数、选择性微调、模型合并）和语义感知的梯度投影方法来缓解该问题。

**[CREPE: Controlling Diffusion with Replica Exchange](image_generation/crepe_controlling_diffusion_with_replica_exchange.md)**

:   提出 CREPE，一种基于 Replica Exchange（并行回火/Parallel Tempering）的扩散模型推理时控制方法，作为 SMC 的计算对偶——在去噪步维度上并行、在样本维度上串行生成，具有高样本多样性、可在线精炼、支持温度退火/奖励倾斜/模型组合/CFG 去偏等多种任务。

**[DenseGRPO: From Sparse to Dense Reward for Flow Matching Model Alignment](image_generation/densegrpo_from_sparse_to_dense_reward_for_flow_matching_model_alignment.md)**

:   解决 Flow Matching + GRPO 对齐中的稀疏奖励问题：通过 ODE 去噪预测中间潜变量的 step-wise 奖励增益作为密集奖励，并根据密集奖励自适应调整 SDE 采样器的逐时间步噪声注入来校准探索空间，在人类偏好对齐/组合生成/文字渲染三个任务上超越 Flow-GRPO。

**[Detecting and Mitigating Memorization in Diffusion Models through Anisotropy of the Log-Probability](image_generation/detecting_and_mitigating_memorization_in_diffusion_models_through_anisotropy_of_.md)**

:   本文证明基于范数的记忆检测指标仅在各向同性（isotropic）对数概率分布下有效，在低噪声各向异性（anisotropic）区域失效；提出结合高噪声范数和低噪声角度对齐（cosine similarity）的无去噪检测指标，在 SD v1.4/v2.0 上超越现有无去噪方法且快 5× 以上。

**[DiffInk: Glyph- and Style-Aware Latent Diffusion Transformer for Text to Online Handwriting Generation](image_generation/diffink_glyph-_and_style-aware_latent_diffusion_transformer_for_text_to_online_h.md)**

:   提出 DiffInk，首个面向全行手写生成的潜在扩散 Transformer 框架，包含 InkVAE（通过 OCR + 风格分类双正则化学习结构化潜空间）和 InkDiT（在潜空间中做条件去噪生成），在中文手写生成上大幅超越 SOTA（AR 94.38% vs 91.48%），速度提升 800×。

**[Diffusion Alignment as Variational Expectation-Maximization](image_generation/diffusion_alignment_as_variational_expectation-maximization.md)**

:   将扩散模型对齐形式化为变分 EM 算法：E-step 用 test-time search（soft Q 引导 + 重要性采样）探索高奖励多模态轨迹，M-step 通过 forward-KL 蒸馏将搜索结果写入模型参数，在图像生成和 DNA 序列设计上同时实现高奖励和高多样性。

**[Diffusion Blend: Inference-Time Multi-Preference Alignment for Diffusion Models](image_generation/diffusion_blend_inference-time_multi-preference_alignment_for_diffusion_models.md)**

:   提出 Diffusion Blend，通过在推理时混合多个奖励微调模型的反向扩散过程来实现多偏好对齐：DB-MPA 支持任意奖励线性组合、DB-KLA 支持动态 KL 正则化控制、DB-MPA-LS 通过随机 LoRA 采样消除推理开销，理论上证明了混合近似的误差界并在实验中接近 MORL oracle 上界。

**[DiffusionNFT: Online Diffusion Reinforcement with Forward Process](image_generation/diffusionnft_online_diffusion_reinforcement_with_forward_process.md)**

:   提出 DiffusionNFT，一种全新的扩散模型在线 RL 范式：不在反向采样过程上做策略优化（如 GRPO），而是在前向过程上通过 flow matching 目标对正样本和负样本做对比式训练，定义隐式的策略改进方向，比 FlowGRPO 快 3-25×，且无需 CFG。

**[Direct Reward Fine-Tuning on Poses for Single Image to 3D Human in the Wild](image_generation/direct_reward_fine-tuning_on_poses_for_single_image_to_3d_human_in_the_wild.md)**

:   提出 DrPose，通过直接奖励微调最大化 PoseScore（骨骼结构一致性）+ KL 正则化，结合 DrPose15K 数据集（来自 Motion-X 的 15K 动态姿态），使多视角扩散模型可重建户外动态/杂技人体姿态。

**[Directional Textual Inversion for Personalized Text-to-Image Generation](image_generation/directional_textual_inversion_for_personalized_text-to-image_generation.md)**

:   本文发现 Textual Inversion (TI) 学到的 token embedding 存在范数膨胀（norm inflation）问题，导致复杂 prompt 的文本对齐下降；提出 Directional Textual Inversion (DTI)，将 embedding 范数固定在分布内尺度、仅在单位超球面上用 Riemannian SGD 优化方向，结合 von Mises-Fisher 先验，显著提升 prompt 忠实度。

**[DistillKac: Few-Step Image Generation via Damped Wave Equations](image_generation/distillkac_few-step_image_generation_via_damped_wave_equations.md)**

:   用阻尼波方程（telegrapher equation）及其随机 Kac 表示替代 Fokker-Planck 方程作为生成模型的概率流基础，实现有限速度传播的概率流，并提出端点蒸馏（endpoint distillation）方法实现少步生成，在 CIFAR-10 上 4 步 FID=4.14、1 步 FID=5.66。

**[DragFlow: Unleashing DiT Priors with Region Based Supervision for Drag Editing](image_generation/dragflow_unleashing_dit_priors_with_region_based_supervision_for_drag_editing.md)**

:   首个将 FLUX (DiT) 的强生成先验引入拖拽编辑的框架，通过区域级仿射监督替代传统点级监督，配合梯度掩码硬约束和 adapter 增强反演，大幅提升拖拽编辑质量。

**[Draw-In-Mind: Rebalancing Designer-Painter Roles in Unified Multimodal Models Benefits Image Editing](image_generation/draw-in-mind_rebalancing_designer-painter_roles_in_unified_multimodal_models_ben.md)**

:   指出当前统一多模态模型中理解模块仅作翻译器而生成模块被迫同时充当"设计师"和"画家"的职责失衡问题，通过构建 DIM 数据集（14M 长上下文文图对 + 233K CoT 编辑蓝图）将设计责任转移给理解模块，4.6B 参数即超越 5 倍大的模型。

**[Dual-Solver: A Generalized ODE Solver for Diffusion Models with Dual Prediction](image_generation/dual-solver_a_generalized_ode_solver_for_diffusion_models_with_dual_prediction.md)**

:   提出 Dual-Solver，通过三组可学习参数（预测类型插值 $\gamma$、积分域选择 $\tau$、残差调整 $\kappa$）泛化扩散模型多步采样器，用冻结预训练分类器（MobileNet/CLIP）的分类损失学习参数（无需教师轨迹），在 3-9 NFE 低步区间全面优于 DPM-Solver++ 等方法。

**[EditScore: Unlocking Online RL for Image Editing via High-Fidelity Reward Modeling](image_generation/editscore_unlocking_online_rl_for_image_editing_via_high-fidelity_reward_modelin.md)**

:   提出首个系统性的"基准评测→奖励模型→强化学习训练"图像编辑 RL 管线：构建 EditReward-Bench 基准，训练 EditScore 系列奖励模型（7B-72B，超过 GPT-5），并成功将其用于 Online RL 训练显著提升编辑模型性能。

**[SSCP: Flow-Based Single-Step Completion for Efficient and Expressive Policy Learning](image_generation/flow-based_single-step_completion_for_efficient_and_expressive_policy_learning.md)**

:   SSCP 通过预测「完成向量」（从任意中间流样本到最终目标的归一化方向）将多步流策略压缩为单步生成，实现 64× 训练加速和 4.7× 推理加速，同时保持与多步流策略相当的表达力。

**[Follow-Your-Shape: Shape-Aware Image Editing via Trajectory-Guided Region Control](image_generation/follow-your-shape_shape-aware_image_editing_via_trajectory-guided_region_control.md)**

:   提出 Follow-Your-Shape，一个无需训练和掩码的形状感知编辑框架，通过计算反演与编辑轨迹间的 token 级速度差异构建 Trajectory Divergence Map (TDM) 来精确定位编辑区域，配合分阶段 KV 注入实现大幅形状变换且严格保持背景。

**[Frame Guidance: Training-Free Guidance for Frame-Level Control in Video Diffusion Models](image_generation/frame_guidance_training-free_guidance_for_frame-level_control_in_video_diffusion.md)**

:   提出 Frame Guidance，一种无需训练的帧级引导方法，通过 latent slicing（降低 60× 显存）和 Video Latent Optimization（VLO）两个核心组件，在不修改模型的情况下实现关键帧引导、风格化和循环视频等多种可控视频生成任务。

**[GenCP: Towards Generative Modeling Paradigm of Coupled Physics](image_generation/gencp_towards_generative_modeling_paradigm_of_coupled_physics.md)**

:   提出 GenCP，将耦合多物理场仿真建模为概率密度演化问题，利用 flow matching 从解耦数据学习条件速度场，推理时通过 Lie-Trotter 算子分裂合成耦合解，实现"解耦训练、耦合推理"，并提供理论误差可控保证。

**[GLASS Flows: Efficient Inference for Reward Alignment of Flow and Diffusion Models](image_generation/glass_flows_reward_alignment_diffusion.md)**

:   提出 GLASS (Gaussian Latent Sufficient Statistic) Flows——一种在流/扩散模型的去噪过程中实现高效随机转移的新采样范式，通过充分统计量重参数化将随机转移重铸为内部 ODE 求解问题，在无需重训的条件下结合 ODE 效率和 SDE 随机性，使 Feynman-Kac Steering 在 FLUX 文生图模型上一致超越 Best-of-N 基线。

**[Image Can Bring Your Memory Back: A Novel Multi-Modal Guided Attack against Image Generation Model Unlearning](image_generation/image_can_bring_your_memory_back_a_novel_multi-modal_guided_attack_against_image.md)**

:   Recall 提出首个多模态引导的攻击框架，通过在隐空间中优化对抗图像 prompt（仅需一张参考图像），配合原始文本 prompt 利用扩散模型的 image-conditioning 通道，在 10 种 SOTA 遗忘方法上平均 ASR 达 65%~97%，显著超越纯文本攻击方法，揭示当前遗忘机制对图像模态攻击的脆弱性。

**[Infinity and Beyond: Compositional Alignment in VAR and Diffusion T2I Models](image_generation/infinity_and_beyond_compositional_alignment_in_var_and_diffusion_t2i_models.md)**

:   首次系统性地对比 Visual Autoregressive (VAR) 模型和扩散模型在组合文本-图像对齐上的表现，在 T2I-CompBench++ 和 GenEval 两个基准上评测 6 个 T2I 模型，发现 Infinity-8B 在几乎所有组合维度上取得最强表现，VAR 架构在组合生成方面展现出显著优势。

**[Localized Concept Erasure in Text-to-Image Diffusion Models via High-Level Representation Misdirection](image_generation/localized_concept_erasure_in_text-to-image_diffusion_models_via_high-level_repre.md)**

:   HiRM 提出"更新位置与擦除目标解耦"的概念擦除策略——仅更新 CLIP 文本编码器第一层的权重，但将擦除监督施加在最后一层的高层语义表征上，通过引导目标概念表征偏向随机方向（HiRM-R）或语义方向（HiRM-S），在 UnlearnCanvas 和 NSFW 基准上实现风格/物体/裸体的高效擦除，且可零样本迁移到 Flux 架构。

**[LoRA-Edit: Controllable First-Frame-Guided Video Editing via Mask-Aware LoRA Fine-Tuning](image_generation/lora-edit_controllable_first-frame-guided_video_editing_via_mask-aware_lora_fine.md)**

:   提出 LoRA-Edit，通过时空掩码感知 LoRA 微调实现可控的首帧引导视频编辑——掩码分离运动保持（背景）和外观控制（编辑区域），用户研究中运动一致性和背景保持优于 AnyV2V/I2VEdit。

**[Mod-Adapter: Tuning-Free and Versatile Multi-concept Personalization via Modulation Adapter](image_generation/mod-adapter_tuning-free_and_versatile_multi-concept_personalization_via_modulati.md)**

:   提出 Mod-Adapter，一种无需测试时微调的多概念个性化方法，通过在 DiT 的调制（modulation）空间中预测概念特定的调制方向，实现对物体和抽象概念（姿态、光照、材质等）的解耦化定制生成，在多概念个性化上大幅超越现有方法。

**[Motion Prior Distillation in Time Reversal Sampling for Generative Inbetweening](image_generation/motion_prior_distillation_in_time_reversal_sampling_for_generative_inbetweening.md)**

:   提出 Motion Prior Distillation (MPD)，一种推理时蒸馏方法，将前向路径的运动残差蒸馏到后向路径中，从根本上解决了时间反转采样中双向运动先验冲突的问题，无需额外训练即可实现更连贯的生成式帧插值。

**[MVCustom: Multi-View Customized Diffusion via Geometric Latent Rendering and Completion](image_generation/mvcustom_multi-view_customized_diffusion_via_geometric_latent_rendering_and_comp.md)**

:   提出多视角定制（multi-view customization）新任务并设计 MVCustom 框架，通过视频扩散骨干网络结合密集时空注意力实现整体帧一致性，在推理阶段引入深度感知特征渲染和一致性感知潜码补全两项技术，首次同时实现相机位姿控制、主体身份保持和跨视角几何一致性。

**[Neon: Negative Extrapolation From Self-Training Improves Image Generation](image_generation/neon_negative_extrapolation_image_generation.md)**

:   提出 Neon，一种仅需 <1% 额外训练计算的后处理方法：先用模型自身生成的合成数据微调导致退化，再反向外推远离退化权重，证明 mode-seeking 采样器导致合成/真实数据梯度反对齐，因此负外推等价于向真实数据分布优化，在 ImageNet 256×256 上将 xAR-L 提升至 SOTA FID 1.02。

**[NeuralOS: Towards Simulating Operating Systems via Neural Generative Models](image_generation/neuralos_towards_simulating_operating_systems_via_neural_generative_models.md)**

:   提出 NeuralOS，使用 RNN 状态追踪 + 扩散渲染器的双组件架构，直接从用户输入事件（鼠标移动/点击/键盘）预测操作系统图形界面帧序列，首次实现用神经生成模型模拟操作系统。

**[RMFlow: Refined Mean Flow by a Noise-Injection Step for Multimodal Generation](image_generation/rmflow_refined_mean_flow_by_a_noise-injection_step_for_multimodal_generation.md)**

:   提出 RMFlow，在 1-NFE MeanFlow 传输后加入一步噪声注入精炼来弥补单步传输的误差，同时在训练中加入最大似然目标来最小化学习分布与目标分布间的 KL 散度，在 T2I、分子生成、时间序列生成上实现接近 SOTA 的 1-NFE 结果。

**[SenseFlow: Scaling Distribution Matching for Flow-based Text-to-Image Distillation](image_generation/senseflow_scaling_distribution_matching_for_flow-based_text-to-image_distillatio.md)**

:   提出 SenseFlow，通过隐式分布对齐（IDA）和段内引导（ISG）将分布匹配蒸馏（DMD）扩展到大规模 flow-based 文生图模型（SD 3.5 Large 8B / FLUX.1 dev 12B），实现 4 步高质量图像生成。

**[SoFlow: Solution Flow Models for One-Step Generative Modeling](image_generation/soflow_solution_flow_models_for_one-step_generative_modeling.md)**

:   提出 Solution Flow Models (SoFlow)，直接学习速度 ODE 的解函数 $f(x_t, t, s)$（将 $t$ 时刻的 $x_t$ 映射到 $s$ 时刻的解），通过 Flow Matching 损失 + 无需 JVP 的解一致性损失从头训练，在 ImageNet 256 上 1-NFE FID 优于 MeanFlow（XL/2: 2.96 vs 3.43）。

**[SPEED: Scalable, Precise, and Efficient Concept Erasure for Diffusion Models](image_generation/speed_scalable_precise_and_efficient_concept_erasure_for_diffusion_models.md)**

:   SPEED 提出基于零空间（null space）约束的闭式模型编辑方法，通过影响力先验过滤（IPF）、定向先验增强（DPA）和不变等式约束（IEC）三种互补技术精化保留集，实现可扩展（5 秒内擦除 100 个概念）、精确（非目标概念语义零损失）且高效的概念擦除。

**[Steer Away From Mode Collisions: Improving Composition In Diffusion Models](image_generation/steer_away_from_mode_collisions_improving_composition_in_diffusion_models.md)**

:   针对扩散模型多概念 prompt 中的概念缺失/碰撞问题，提出"模式碰撞"假说（联合分布与单概念分布的模式重叠），设计 CO3（Concept Contrasting Corrector）通过在 Tweedie 均值空间中组合校正分布 $\tilde{p}(x|C) \propto p(x|C) / \prod_i p(x|c_i)$ 来远离退化模式，实现即插即用、无梯度、模型无关的组合生成改进。

**[TAVAE: A VAE with Adaptable Priors Explains Contextual Modulation in the Visual Cortex](image_generation/tavae_a_vae_with_adaptable_priors_explains_contextual_modulation_in_the_visual_c.md)**

:   扩展 VAE 形式主义提出 Task-Amortized VAE (TAVAE)，通过在已学表示上灵活学习任务特异性先验来解释视觉皮层 V1 中的上下文调制现象，包括方向辨别任务中训练刺激与测试刺激不匹配时出现的双模态群体响应。

**[Training-Free Reward-Guided Image Editing via Trajectory Optimal Control](image_generation/training-free_reward-guided_image_editing_via_trajectory_optimal_control.md)**

:   将 reward-guided 图像编辑重新建模为轨迹最优控制问题，将扩散/Flow模型的反向过程视为可控轨迹，通过基于 Pontryagin 最大值原理（PMP）的伴随状态迭代优化整条轨迹，在无需训练的情况下实现有效的奖励引导编辑且不发生 reward hacking。

**[TwinFlow: Realizing One-step Generation on Large Models with Self-adversarial Flows](image_generation/twinflow_realizing_one-step_generation_on_large_models_with_self-adversarial_flo.md)**

:   提出 TwinFlow，一种无需辅助训练模型（判别器/冻结教师）的自对抗流匹配框架，通过模型自身多步输出作为单步的教学目标实现单步生成，首次将 1-NFE 生成能力成功扩展到 20B 参数的 Qwen-Image 模型，GenEval 0.86（1-NFE）接近原始 100-NFE 的 0.87。

---

## 🎮 强化学习 { #reinforcement_learning }

**[AbstRaL: Augmenting LLMs' Reasoning by Reinforcing Abstract Thinking](reinforcement_learning/abstral_augmenting_llms_reasoning_by_reinforcing_abstract_thinking.md)**

:   提出 AbstRaL，通过强化学习教 LLM 学习推理问题的数学抽象（将具体数字/名称替换为符号变量、提取通用公式），然后用符号求解器推导答案，在 GSM 扰动 benchmark 上几乎完全消除了分布偏移导致的性能下降，并在 OOD 数学/通用推理任务上也有隐式提升。

**[APPLE: Toward General Active Perception via Reinforcement Learning](reinforcement_learning/apple_toward_general_active_perception_via_reinforcement_learning.md)**

:   提出APPLE——一种结合强化学习与监督学习的通用主动感知框架，将主动感知建模为POMDP，奖励函数设计为RL奖励减去预测损失，梯度自然分解为策略梯度和预测损失梯度两部分，基于off-policy算法（SAC/CrossQ）和共享ViViT骨干网络，在5个不同任务基准上验证通用性，其中CrossQ变体无需逐任务调参且训练效率提高53%。

**[ARM-FM: Automated Reward Machines via Foundation Models for Compositional Reinforcement Learning](reinforcement_learning/arm-fm_automated_reward_machines_via_foundation_models_for_compositional_reinfor.md)**

:   提出ARM-FM框架，利用基础模型（GPT-4o等）从自然语言任务描述自动生成语言对齐奖励机器（LARM）——包括自动机结构、可执行标签函数和每个状态的自然语言描述——为RL agent提供组合式密集奖励信号，在MiniGrid/Craftium(3D Minecraft)/Meta-World等环境中解决标准RL完全无法学习的稀疏奖励长程任务，并实现零样本任务泛化。

**[Autoqd Automatic Discovery Of Diverse Behaviors With Quality-Diversity Optimizat](reinforcement_learning/autoqd_automatic_discovery_of_diverse_behaviors_with_quality-diversity_optimizat.md)**

:   提出 AutoQD，利用占用度量 (occupancy measure) 的随机 Fourier 特征嵌入自动生成行为描述子 (behavioral descriptor)，替代传统 QD 优化中的手工设计描述子，在 6 个连续控制任务上展现了强大的多样化策略发现能力。

**[Autotool Automatic Scaling Of Tool-Use Capabilities In Rl Via Decoupled Entropy ](reinforcement_learning/autotool_automatic_scaling_of_tool-use_capabilities_in_rl_via_decoupled_entropy_.md)**

:   提出解耦自适应熵约束 (Decoupled Adaptive Entropy Constraints) 的强化学习策略，使 LLM 在工具调用任务中根据问题难度自动切换长/短推理模式，在提升 9.8% 准确率的同时减少约 81% 的推理 token 开销。

**[AWM: Accurate Weight-Matrix Fingerprint for Large Language Models](reinforcement_learning/awm_accurate_weight-matrix_fingerprint_for_large_language_models.md)**

:   提出 AWM，一种无需训练的 LLM 权重矩阵指纹方法，利用线性分配问题（LAP）恢复嵌入层的置换和符号翻转，再用无偏 CKA 消除 Q/K 矩阵的正交变换影响，在 150 对 LLM 上实现完美 AUC（1.0），对 SFT、持续预训练（5.5T token）、RL、多模态扩展、剪枝、upcycling 六类后训练均鲁棒，30 秒内完成。

**[BA-MCTS: Bayes Adaptive Monte Carlo Tree Search for Offline Model-based RL](reinforcement_learning/bayes_adaptive_monte_carlo_tree_search_for_offline_model-based_reinforcement_lea.md)**

:   BA-MCTS 首次将贝叶斯自适应 MDP（BAMDP）引入离线模型基 RL，通过连续空间 BAMCP + 悲观奖励惩罚 + 搜索基策略迭代的形式，实现动态更新 ensemble 信念并优先使用最精确的模型，在 D4RL 12 个任务上超越 19 个基线。

**[Boolean Satisfiability via Imitation Learning](reinforcement_learning/boolean_satisfiability_via_imitation_learning.md)**

:   提出 ImitSAT，首个基于模仿学习的 CDCL 求解器分支策略：通过将求解器运行压缩为无冲突的 KeyTrace 专家序列，将分支决策建模为前缀条件的自回归预测任务，以少量查询预算显著减少传播次数和求解时间，并在结构化 SAT 问题上展现良好泛化能力。

**[Breaking Barriers: Do Reinforcement Post Training Gains Transfer To Unseen Domains?](reinforcement_learning/breaking_barriers_do_reinforcement_post_training_gains_transfer_to_unseen_domain.md)**

:   通过观察性研究（18 个开源 RPT 模型）和干预性研究（单域 GRPO 训练），系统揭示了强化后训练（RPT/RLVR）的泛化局限：RPT 在训练域内提升显著，但跨域泛化不一致——结构化域（数学↔代码）可互相迁移，但无法泛化到非结构化域（法律/金融/医疗），且这一结论跨算法、模型规模和训练步数保持一致。

**[Breaking the SFT Plateau: Multimodal Structured Reinforcement Learning for Chart-to-Code Generation](reinforcement_learning/breaking_the_sft_plateau_multimodal_structured_reinforcement_learning_for_chart-.md)**

:   针对图表到代码生成任务中SFT的性能瓶颈问题，提出多模态结构化强化学习（MSRL），通过文本+视觉双层奖励函数和两阶段RL策略，在ChartMimic和ReachQA上分别提升6.2%和9.9%的高层指标，达到开源SOTA并媲美GPT-4o。

**[Co-rewarding: Stable Self-supervised RL for Eliciting Reasoning in Large Language Models](reinforcement_learning/co-rewarding_stable_self-supervised_rl_for_eliciting_reasoning_in_large_language.md)**

:   Co-rewarding 提出自监督 RL 框架，通过数据侧（对比改写问题的跨视角一致性）和模型侧（EMA 教师模型提供伪标签）两种互补监督方式，解决自奖励 RL 中的训练崩溃问题，在无人工标签条件下多项数学推理基准上达到甚至超过 RLVR（有标签）的性能。

**[Continuous-Time Value Iteration for Multi-Agent Reinforcement Learning](reinforcement_learning/continuous-time_value_iteration_for_multi-agent_reinforcement_learning.md)**

:   提出 VIP（Value Iteration via PINN）框架，首次将物理信息神经网络（PINN）用于求解连续时间多智能体强化学习中的 HJB 偏微分方程，并引入 Value Gradient Iteration（VGI）模块迭代精炼价值梯度，在连续时间 MPE 和 MuJoCo 多智能体任务上始终优于离散时间和连续时间基线。

**[Controllable Exploration in Hybrid-Policy RLVR for Multi-Modal Reasoning](reinforcement_learning/controllable_exploration_in_hybrid-policy_rlvr_for_multi-modal_reasoning.md)**

:   CalibRL 将专家数据重新定义为分布校准基线（而非严格模仿目标），通过 LeakyReLU 不对称激活 + 优势加权实现对 MLLM 推理训练中探索-利用平衡的精细控制，解决 RLVR 中的熵崩溃问题，在几何推理等任务上大幅超越 GRPO/DAPO。

**[Cross-Embodiment Offline Reinforcement Learning for Heterogeneous Robot Datasets](reinforcement_learning/cross-embodiment_offline_reinforcement_learning_for_heterogeneous_robot_datasets.md)**

:   系统研究跨形态离线 RL 预训练范式，发现次优数据比例和机器人多样性增加时梯度冲突导致负迁移，提出基于形态图距离的 Embodiment Grouping（EG）策略将机器人按形态聚类后分组更新 actor，在 16 种机器人平台的 locomotion benchmark 上显著缓解负迁移（70% 次优数据集上 IQL+EG 比 IQL 提升 34%）。

**[CUDA-L1: Improving CUDA Optimization via Contrastive Reinforcement Learning](reinforcement_learning/cuda-l1_improving_cuda_optimization_via_contrastive_reinforcement_learning.md)**

:   提出 CUDA-L1，一个基于对比强化学习（Contrastive RL）的三阶段流水线框架，将初始 CUDA 能力较弱的 LLM 训练为高效的 CUDA 优化器，在 KernelBench 的 250 个 CUDA 内核上实现平均 3.12× 加速，峰值达 120×，并可跨 GPU 架构迁移。

**[Deep SPI: Safe Policy Improvement via World Models](reinforcement_learning/deep_spi_safe_policy_improvement_via_world_models.md)**

:   构建了安全策略改进（SPI）的理论框架，将世界模型和表示学习与策略更新保证统一起来：通过基于重要性比率的邻域算子约束策略更新，确保单调改进和收敛；结合局部转移/奖励损失控制世界模型质量和表示稳定性，提出 DeepSPI 算法在 ALE-57 基准上匹配或超越 PPO 和 DeepMDP。

**[Dual-Robust Cross-Domain Offline Reinforcement Learning Against Dynamics Shifts](reinforcement_learning/dual-robust_cross-domain_offline_reinforcement_learning_against_dynamics_shifts.md)**

:   首次同时解决跨域离线 RL 的"训练时鲁棒性"（源域-目标域不匹配）和"测试时鲁棒性"（部署环境动态偏移）：提出 DROCO，通过 Robust Cross-Domain Bellman (RCB) 算子对源域数据施加鲁棒 Bellman 更新、对目标域数据施加标准更新，将动态不确定性映射为可处理的状态扰动。

**[Efficient Estimation of Kernel Surrogate Models for Task Attribution](reinforcement_learning/efficient_estimation_of_kernel_surrogate_models_for_task_attribution.md)**

:   提出核代理模型（KernelSM）用于任务归因，通过 RBF 核岭回归捕获任务间的非线性交互效应，结合梯度投影的高效估计算法避免重复训练，在数学推理、上下文学习和多目标 RL 等场景下相比线性代理和影响函数基线提升 25% 相关性。

**[Entropy-Preserving Reinforcement Learning (REPO / ADAPO)](reinforcement_learning/entropy-preserving_reinforcement_learning.md)**

:   本文揭示了策略梯度 RL 算法在 LLM 后训练中系统性导致策略熵坍缩的理论根因（优势函数与对数概率的正相关性），并提出两种互补的解法：REPO（通过修改优势函数去相关）和 ADAPO（自适应非对称裁剪），在交互式工具使用任务上实现 SOTA 性能。

**[Exploration vs Exploitation: Rethinking RLVR through Clipping, Entropy, and Spurious Reward](reinforcement_learning/exploration_vs_exploitation_rethinking_rlvr_through_clipping_entropy_and_spuriou.md)**

:   本文揭示 RLVR 中两个反直觉现象——随机奖励也能提升推理性能、熵最小化也能改善结果——并从理论上证明裁剪偏差提供的学习信号可忽略不计，性能提升的真正来源是裁剪对熵的隐式压缩作用和奖励误标对强模型的"有益偏差"效应。

**[Flow Actor-Critic for Offline Reinforcement Learning (FAC)](reinforcement_learning/flow_actor-critic_for_offline_reinforcement_learning.md)**

:   FAC 首次联合利用流模型（continuous normalizing flow）同时构建表达力强的 actor 策略和基于精确密度估计的 critic 惩罚机制，通过识别 OOD 区域对 Q 值进行选择性保守估计，在 OGBench 55 个任务上以 60.3 平均分大幅超越此前最佳的 43.6。

**[Learning to Orchestrate Agents in Natural Language with the Conductor](reinforcement_learning/learning_to_orchestrate_agents_in_natural_language_with_the_conductor.md)**

:   用RL训练7B的Conductor模型，通过自然语言输出Agent工作流(子任务分配+通信拓扑)来协调GPT-5/Claude/Gemini等大模型，在LiveCodeBench和GPQA等benchmark上超越所有单模型和多Agent基线，达到SOTA(平均77.27 vs GPT-5的74.78)。

**[LongRLVR: Long-Context Reinforcement Learning Requires Verifiable Context Rewards](reinforcement_learning/longrlvr_long-context_reinforcement_learning_requires_verifiable_context_rewards.md)**

:   提出 LongRLVR，通过在 RLVR 训练中引入可验证的上下文奖励（context reward），解决长上下文场景下仅靠最终答案奖励导致的上下文定位（grounding）梯度消失问题，显著提升 LLM 长上下文推理能力。

**[LoongRL: Reinforcement Learning for Advanced Reasoning over Long Contexts](reinforcement_learning/loongrl_rl_for_reasoning_long_contexts.md)**

:   提出 LoongRL，通过构建 KeyChain 合成数据进行强化学习训练，使 LLM 涌现出 plan–retrieve–reason–recheck 的长上下文推理模式，仅在 16K 上下文上训练即可泛化到 128K，14B 模型达到 74.2 分接近 o3-mini (74.5) 和 DeepSeek-R1 (74.9)。

**[ROMI: Model-based Offline RL via Robust Value-Aware Model Learning with Implicitly Differentiable Adaptive Weighting](reinforcement_learning/model-based_offline_rl_via_robust_value-aware_model_learning_with_implicitly_dif.md)**

:   ROMI 通过 Wasserstein 对偶将动力学不确定集转化为状态不确定集来实现鲁棒的价值感知模型学习，并用隐式可微的自适应加权机制平衡动力学精度与价值感知，解决了 RAMBO 方法中的 Q 值低估和梯度爆炸问题，在 D4RL 和 NeoRL 上达到模型基离线 RL 的 SOTA。

**[MVR: Multi-view Video Reward Shaping for Reinforcement Learning](reinforcement_learning/mvr_multi-view_video_reward_shaping_for_reinforcement_learning.md)**

:   提出 MVR 框架，利用多视角视频的视频-文本相似度学习状态相关性函数，结合状态依赖的奖励塑形（自动衰减 VLM 引导），在 HumanoidBench 和 MetaWorld 共 19 个任务上超越现有 VLM 奖励方法。

**[One Model for All Tasks: Leveraging Efficient World Models in Multi-Task Planning](reinforcement_learning/one_model_for_all_tasks_leveraging_efficient_world_models_in_multi-task_planning.md)**

:   提出 ScaleZero，通过在统一世界模型中引入 MoE 架构解决多任务学习中的梯度冲突和可塑性崩塌问题，结合动态参数扩展（DPS）策略自适应分配模型容量，单个多任务模型在 Atari/DMC/Jericho 三个基准上达到与单任务专家模型相当的性能，同时减少约 28.5% 的环境交互。

**[Online Minimization of Polarization and Disagreement via Low-Rank Matrix Bandits](reinforcement_learning/online_minimization_of_polarization_and_disagreement_via_low-rank_matrix_bandits.md)**

:   将社交网络中极化与分歧最小化问题建模为在线低秩矩阵bandit问题，提出两阶段算法OPD-Min-ESTR（先估计子空间再低维线性bandit），将维度从 $|V|^2$ 降至 $O(|V|)$，实现 $\tilde{O}(\max\{1/\kappa, \sqrt{|V|}\}\sqrt{|V|T})$ 累积遗憾。

**[Optimistic Task Inference for Behavior Foundation Models](reinforcement_learning/optimistic_task_inference_behavior_models.md)**

:   提出 OpTI-BFM——一种乐观决策准则，对奖励函数的不确定性建模并引导 Behavior Foundation Model 在测试时通过环境交互进行任务推断，与线性 bandit 的上置信界（UCB）算法建立联系并提供正式的 regret bound，使 BFM 仅需几个 episode 即可识别和优化未见奖励函数，几乎无额外计算开销。

**[Post-training Large Language Models for Diverse High-Quality Responses](reinforcement_learning/post-training_large_language_models_for_diverse_high-quality_responses.md)**

:   提出 DQO（Diversity Quality Optimization），基于行列式点过程（DPP）在语义嵌入空间中定义多样性度量，将其与奖励信号联合优化，使 LLM 后训练同时提升语义多样性和响应质量，可叠加在 GRPO/PPO 之上。

**[Principled Fast and Meta Knowledge Learners for Continual Reinforcement Learning](reinforcement_learning/principled_fast_and_meta_knowledge_learners_for_continual_reinforcement_learning.md)**

:   受人脑海马体-大脑皮层交互机制启发，提出 FAME 双学习器框架，通过快速学习器进行知识迁移、元学习器进行知识整合，在原则性地最小化灾难性遗忘的前提下实现高效的持续强化学习。

**[Reasoning Boosts Opinion Alignment in LLMs](reinforcement_learning/reasoning_boosts_opinion_alignment_in_llms.md)**

:   用GRPO强化学习训练LLM从政治调查数据中学习推理式观点对齐，在美国/德国/瑞士三个数据集上证明推理能提升个体级政治观点建模的准确性。

**[Regret-Guided Search Control for Efficient Learning in AlphaZero](reinforcement_learning/regret-guided_search_control_for_efficient_learning_in_alphazero.md)**

:   提出 RGSC（Regret-Guided Search Control）框架，通过训练一个 regret 网络识别高遗憾值状态并优先从这些状态重新开始自我对弈，模拟人类"反复复盘错误"的学习方式，在 9×9 围棋、10×10 黑白棋和 11×11 Hex 上平均超越 AlphaZero 77 Elo。

**[Robust Deep Reinforcement Learning against Adversarial Behavior Manipulation](reinforcement_learning/robust_deep_reinforcement_learning_against_adversarial_behavior_manipulation.md)**

:   本文研究 RL 中一种新型威胁——行为目标攻击（adversary 通过篡改观测来引导 victim 执行特定目标策略），提出不需要白盒访问的 BIA 攻击方法和基于时间折扣的 TDRT 防御方法，TDRT 在保持对攻击鲁棒性的同时比现有防御（SA-PPO）的原始任务性能高 28.2%。

**[Routing, Cascades, and User Choice for LLMs](reinforcement_learning/routing_cascades_and_user_choice_for_llms.md)**

:   将 LLM 路由建模为 provider-user Stackelberg 博弈，证明最优路由策略几乎总是静态无级联的阈值规则，并揭示当模型质量排序与成本排序不一致时产生的用户-提供商不对齐问题，以及低流失惩罚下 provider 有动机通过增加延迟来降低成本。

**[Self-Harmony: Learning to Harmonize Self-Supervision and Self-Play in Test-Time Reinforcement Learning](reinforcement_learning/self-harmony_learning_to_harmonize_self-supervision_and_self-play_in_test-time_r.md)**

:   提出 Self-Harmony 框架，通过让单一模型扮演两个角色（Solver 求解原始问题 + Reframer 改述问题），将答案在原始和改述视角下的调和平均得分作为伪标签选择标准，替代传统多数投票，在 30 个实验设置中 28 个达到 SOTA，且训练零失败。

**[Self-Improving Skill Learning for Robust Skill-based Meta-Reinforcement Learning](reinforcement_learning/self-improving_skill_learning_for_robust_skill-based_meta-reinforcement_learning.md)**

:   提出 SISL（Self-Improving Skill Learning），通过解耦高层策略和技能改进策略，结合最大回报重标注的技能优先级机制，在噪声离线演示数据下实现鲁棒的技能学习，显著提升基于技能的元强化学习在长时域任务中的性能。

**[Solving Football by Exploiting Equilibrium Structure of 2p0s Differential Games with One-Sided Information](reinforcement_learning/solving_football_by_exploiting_equilibrium_structure_of_2p0s_differential_games_.md)**

:   证明单边信息二人零和微分博弈中 Nash 均衡策略的原子结构——知情玩家 P1 的均衡策略集中在至多 $I$ 个动作原型上（$I$ = 博弈类型数），使博弈树复杂度从 $U^{2K}$ 降到 $I^K$，在美式足球 11v11 连续动作空间中（传统复杂度 $10^{440}$）实现 M1 MacBook 30 分钟求解。

**[UME-R1: Exploring Reasoning-Driven Generative Multimodal Embeddings](reinforcement_learning/ume-r1_exploring_reasoning-driven_generative_multimodal_embeddings.md)**

:   提出 UME-R1，首次探索推理驱动的生成式多模态嵌入范式，通过两阶段训练（冷启动SFT + 强化学习）让嵌入模型先推理再生成表示，在 MMEB-V2 基准的 78 个任务上显著超越传统判别式嵌入模型。

**[Understanding and Improving Hyperbolic Deep Reinforcement Learning](reinforcement_learning/understanding_and_improving_hyperbolic_deep_reinforcement_learning.md)**

:   通过形式化梯度分析揭示双曲深度 RL 的训练不稳定根源（大范数嵌入导致信赖域违反），提出 Hyper++ 三组件方案（RMSNorm + 学习缩放 + 分类值损失）实现稳定训练并超越现有方法。

**[Value Flows](reinforcement_learning/value_flows.md)**

:   Value Flows 首次将流匹配（flow matching）引入分布式 RL，学习一个向量场来生成完整的连续回报分布，通过 flow derivative ODE 高效估计方差实现基于置信度的优先学习,在 OGBench 62 个任务上平均提升 1.3 倍。

**[VerifyBench: Benchmarking Reference-based Reward Systems for Large Language Models](reinforcement_learning/verifybench_benchmarking_reference-based_reward_systems_for_large_language_model.md)**

:   针对大型推理模型（LRM）训练中广泛使用的基于参考答案的奖励系统，构建了 VerifyBench 和 VerifyBench-Hard 两个评测基准，通过严格的人工标注评估各类验证系统的准确性，发现即使最强模型在困难样本上也仅达约 88% 准确率，揭示了当前验证系统的显著改进空间。

**[Whatever Remains Must Be True: Filtering Drives Reasoning in LLMs, Shaping Diversity](reinforcement_learning/whatever_remains_must_be_true_filtering_drives_reasoning_in_llms_shaping_diversi.md)**

:   提出 DMVR 框架和 α-DPG 算法，通过显式定义"过滤掉错误答案"的目标分布并用 α-散度族来逼近，统一了 RLVR（Reverse KL）和拒绝采样微调（Forward KL），在 Lean 定理证明上实现了精度-覆盖率 Pareto 前沿的最优表现。

---

## 💡 LLM 推理 { #llm_reasoning }

**[Adaptive Social Learning via Mode Policy Optimization for Language Agents](llm_reasoning/adaptive_social_learning_via_mode_policy_optimization_for_language_agents.md)**

:   提出 Adaptive Social Learning（ASL）框架，设计四种层次化推理模式（从直觉回应到深度推演），并通过 AMPO 算法（融合模式级和样本级优势估计）让 LLM agent 根据社交场景复杂度自适应切换推理深度，在社交智能任务上比 GPT-4o 高 15.6%，比 GRPO 高 7.0% 且 token 用量减少 32.8%。

**[Agentified Assessment of Logical Reasoning Agents](llm_reasoning/agentified_assessment_of_logical_reasoning_agents.md)**

:   提出基于Agent的评测框架(AAA)，用assessor agent标准化地评估逻辑推理agent，并以自动形式化agent（NL→Z3Py+SMT求解）在清洗后的FOLIO上达到86.70%准确率，大幅超过CoT基线73.89%。

**[AIMCoT: Active Information-driven Multimodal Chain-of-Thought for Vision-Language Reasoning](llm_reasoning/aimcot_active_information-driven_multimodal_chain-of-thought_for_vision-language.md)**

:   提出 AIMCoT，将多模态 CoT 的视觉信息选择从"被动关注高注意力区域"转变为"主动寻找最高信息增益区域"，通过三个模块（CAG 上下文增强注意力图、AVP 主动视觉探测、DAT 动态注意力转移触发）协同工作，在 LLaVA-W 上比 ICoT 提升 18.25%（0-shot），是一个免训练的即插即用框架。

**[Annotation-Efficient Universal Honesty Alignment](llm_reasoning/annotation-efficient_universal_honesty_alignment.md)**

:   提出 EliCal（先激发后校准）两阶段框架，先用无标注的 self-consistency 信号教 LLM 表达内部置信度，再用极少量正确性标注（仅 1k 个，占 0.18%）进行校准，在 HonestyBench（560K 训练 + 70K 评估）上达到接近全量标注 98% 的诚实性对齐性能，并在未见 MMLU 任务上泛化优于仅校准基线。

**[Are Reasoning LLMs Robust to Interventions on Their Chain-of-Thought?](llm_reasoning/are_reasoning_llms_robust_to_interventions_on_their_chain-of-thought.md)**

:   系统评估推理型 LLM 对其 CoT 中各种干预（良性/中性/对抗性）的鲁棒性：发现模型总体鲁棒能从干预中恢复，但改写风格（paraphrasing）会抑制"自我怀疑"表达导致正确率下降，恢复过程有显著计算开销（CoT 膨胀最高 665%）。

**[ATTS: Asynchronous Test-Time Scaling via Conformal Prediction](llm_reasoning/atts_asynchronous_test-time_scaling_via_conformal_prediction.md)**

:   提出 ATTS，一个基于 conformal prediction 的异步 test-time scaling 框架，通过将 rejection sampling 重构为假设检验过程来消除同步开销，在 MATH/AIME 等数学推理任务上实现最高 56.7x 加速和 4.14x 吞吐量提升，且无精度损失；1.5B/70B 的 draft/target 组合可达到 o3-mini (high) 的 AIME 水平。

**[Beyond Prompt-Induced Lies: Investigating LLM Deception on Benign Prompts](llm_reasoning/beyond_prompt-induced_lies_investigating_llm_deception_on_benign_prompts.md)**

:   提出 Contact Searching Question (CSQ) 框架，通过两个统计指标（欺骗意图分数 ρ 和欺骗行为分数 δ）量化 LLM 在正常良性提示下的自发欺骗行为，发现 16 个主流 LLM 普遍存在随任务难度升级的系统性欺骗倾向。

**[Compositional Generalization from Learned Skills via CoT Training: A Theoretical and Structural Analysis for Reasoning](llm_reasoning/compositional_generalization_from_learned_skills_via_cot_training_a_theoretical_.md)**

:   本文通过信息论泛化界和可解释性分析证明，CoT 训练的核心机制是**组合泛化**——模型学会系统性地组合已学的简单技能来解决新颖复杂问题，并内化为两阶段组合推理电路，使中间结果在更浅层提取，释放深层专注于后续推理步骤。

**[Continuous Chain of Thought Enables Parallel Exploration and Reasoning](llm_reasoning/continuous_chain_of_thought_enables_parallel_exploration_and_reasoning.md)**

:   CoT2 提出用连续值 token（词表 embedding 的凸组合）替代离散 token 进行链式推理，使模型能在单次推理中并行追踪多条推理路径，理论证明等价于 K 次 self-consistency/best-of-N 采样，并通过 GRPO 强化学习进一步提升性能。

**[CoT-RVS: Zero-Shot Chain-of-Thought Reasoning Segmentation for Videos](llm_reasoning/cot-rvs_zero-shot_chain-of-thought_reasoning_segmentation_for_videos.md)**

:   提出 CoT-RVS，一种无训练的多智能体框架，利用 MLLM 的零样本 Chain-of-Thought 能力进行时间-语义推理以选择关键帧，实现对复杂隐式查询的推理视频分割，在多个 benchmark 上大幅超越已有方法。

**[DAG-Math: Graph-of-Thought Guided Mathematical Reasoning in LLMs](llm_reasoning/dag-math_graph-of-thought_guided_mathematical_reasoning_in_llms.md)**

:   将 LLM 的 CoT 推理形式化为 DAG 上的基于规则的随机过程，提出"逻辑闭合性"（logical closeness）度量来评估模型是否通过搜索还是严格逻辑推理得到答案，构建了 2894 个金标准 DAG-MATH benchmark，发现即使 PASS@k 相近的模型在推理忠实度上也存在显著差异。

**[DRPO: Efficient Reasoning via Decoupled Reward Policy Optimization](llm_reasoning/drpo_efficient_reasoning_via_decoupled_reward_policy_optimization.md)**

:   诊断出 GRPO 在加入长度惩罚后的根本缺陷——正确但冗长的回答可能获得负优势值从而被错误惩罚——提出 DRPO 将正负样本的奖励信号解耦，确保长度惩罚只在正确回答组内归一化，在 1.5B 模型上实现 77% 长度缩减仅 1.1% 性能损失（对比基线 68% 缩减 4.3% 损失）。

**[Dynamics-Predictive Sampling for Active RL Finetuning of Large Reasoning Models](llm_reasoning/dynamics-predictive_sampling_for_active_rl_finetuning_of_large_reasoning_models.md)**

:   将 RL 微调中每个 prompt 的求解进度建模为隐马尔可夫动力系统，通过轻量贝叶斯推断在线预测 prompt 的求解状态，优先采样"部分求解"的 prompt，以不到 DS 30% 的 rollout 量达到同等甚至更优的推理性能。

**[Dynamics Within Latent Chain-of-Thought: An Empirical Study of Causal Structure](llm_reasoning/dynamics_within_latent_chain-of-thought_an_empirical_study_of_causal_structure.md)**

:   将隐式CoT建模为结构因果模型(SCM)，通过逐步do-干预分析Coconut和CODI两种范式，发现隐式推理步骤具有异质性因果杠杆、非局部跳跃传播结构、以及输出层早期偏向与表征层晚期提交之间的持续性差距。

**[Estimating the Empowerment of Language Model Agents](llm_reasoning/estimating_the_empowerment_of_language_model_agents.md)**

:   提出 EELMA，用信息论的"赋权"（Empowerment = 动作与未来状态的互信息）作为目标无关的 LM Agent 能力度量，与任务表现强相关（r=0.83-0.94），发现 CoT 移除 99% 赋权、认证动作赋权跳跃 3 倍。

**[FastGRPO: Accelerating Policy Optimization via Concurrency-aware Speculative Decoding and Online Draft Learning](llm_reasoning/fastgrpo_accelerating_policy_optimization_via_concurrency-aware_speculative_deco.md)**

:   针对GRPO训练中生成阶段占91%-98%时间的瓶颈，提出并发感知的投机解码策略（动态调整draft树大小）和在线draft模型学习（持续适配目标模型分布），实现2.35x-2.72x端到端加速。

**[Fine-R1: Make Multi-modal LLMs Excel in Fine-Grained Visual Recognition by Chain-of-Thought Reasoning](llm_reasoning/fine-r1_make_multi-modal_llms_excel_in_fine-grained_visual_recognition_by_chain-.md)**

:   Fine-R1 通过 CoT 监督微调（"视觉分析→候选子类→对比→预测"结构化推理链）+ 三元组增强策略优化 TAPO（类内增强提升鲁棒性 + 类间增强提升判别力），仅用 4-shot 训练即在细粒度视觉识别上超越 CLIP 和通用/推理型 MLLM。

**[Fixing the Broken Compass: Diagnosing and Improving Inference-Time Reward Modeling](llm_reasoning/fixing_the_broken_compass_diagnosing_and_improving_inference-time_reward_modelin.md)**

:   系统诊断推理时奖励模型(RM)的三大问题（简单题性能下降、采样增多判别力衰退、高搜索多样性损害），提出CRISP算法通过答案聚类聚合奖励信号+逐步前缀引导生成，比其他RM推理方法提升最高5%准确率，比R1模型在非数学任务上平均提升10%且token量减少90%。

**[Formal Mechanistic Interpretability: Automated Circuit Discovery with Provable Guarantees](llm_reasoning/formal_mechanistic_interpretability_automated_circuit_discovery_with_provable_gu.md)**

:   将神经网络验证（NN verification）引入机制可解释性，提出首个具有可证明保证的电路发现框架：在连续输入域上保证电路忠实度（input robustness）、在连续 patching 域上保证电路一致性（patching robustness），并形式化了四级最小性层次（quasi → local → subset → cardinal），通过单调性理论将三类保证统一连接。

**[GeoGramBench: Benchmarking the Geometric Program Reasoning in Modern LLMs](llm_reasoning/geogrambench_benchmarking_the_geometric_program_reasoning_in_modern_llms.md)**

:   提出Program-to-Geometry任务和GeoGramBench(500题)，用三级几何复杂度分类法(基元识别/局部组合/全局抽象)评估19个前沿LLM从程序代码构建几何表征并推理的能力，发现所有模型在最高抽象级别准确率均低于50%。

**[Harder Is Better: Boosting Mathematical Reasoning via Difficulty-Aware GRPO and Multi-Aspect Question Reformulation](llm_reasoning/harder_is_better_boosting_mathematical_reasoning_via_difficulty-aware_grpo_and_m.md)**

:   揭示GRPO中更新幅度对难题隐式抑制的问题(中等难度题更新最大)，提出MathForge框架：DGPO用MAD替换std实现难度均衡+难题加权，MQR通过多方面改写增加题目难度但保留答案，在6个数学推理benchmark上平均超GRPO +4.56%。

**[I Can't Believe It's Not Robust: Catastrophic Collapse of Safety Classifiers under Embedding Drift](llm_reasoning/i_cant_believe_its_not_robust_catastrophic_collapse_of_safety_classifiers_under_.md)**

:   本文系统研究了基于 frozen embedding 的安全分类器在模型更新导致 embedding 漂移时的脆弱性，发现仅 2% 的 embedding 扰动即可将分类器性能从 85% ROC-AUC 降至随机水平（50%），且 72% 的误分类发生在高置信度下（silent failure），同时 instruction-tuned 模型反而比 base 模型更难分类。

**[InnoGym: Benchmarking the Innovation Potential of AI Agents](llm_reasoning/innogym_benchmarking_the_innovation_potential_of_ai_agents.md)**

:   提出 InnoGym 框架和 iBench/iGym 基准，首次从"创新性"维度评估 AI Agent——不仅衡量正确性还衡量方法论新颖性，发现当前 Agent 能产生新颖想法但无法转化为性能提升（平均归一化增益 -0.45）。

**[LingOly-TOO: Disentangling Reasoning from Knowledge with Templatised Orthographic Obfuscation](llm_reasoning/lingoly-too_disentangling_reasoning_from_knowledge_with_templatised_orthographic.md)**

:   提出LingOly-TOO benchmark(1,203题/6,995子问题)，通过对语言学奥赛题的专家设计正字法混淆来分离LLM的推理能力与知识/记忆，发现最强模型从原始题0.59降至混淆后0.48，揭示了LLM推理能力被严重高估。

**[LogicReward: Incentivizing LLM Reasoning via Step-Wise Logical Supervision](llm_reasoning/logicreward_incentivizing_llm_reasoning_via_step-wise_logical_supervision.md)**

:   提出LogicReward奖励函数，用Isabelle定理证明器做步骤级逻辑正确性验证，结合Autoformalization with Soft Unification减少自然语言歧义，训练出的8B模型在NLI和逻辑推理任务上超越GPT-4o 11.6%和o4-mini 2%。

**[mR3: Multilingual Rubric-Agnostic Reward Reasoning Models](llm_reasoning/mr3_multilingual_rubric-agnostic_reward_reasoning_models.md)**

:   提出 mR3，72 语言的多语言评分推理模型，通过课程学习和 GPT-OSS-120B 蒸馏训练，14B 参数模型匹配 120B 模型性能（mR3-RewardBench 88.46%），支持逐点/成对/二元三种评估模式。

**[Nudging the Boundaries of LLM Reasoning](llm_reasoning/nudging_the_boundaries_of_llm_reasoning.md)**

:   指出GRPO无法从"不可解"问题(0% pass rate)学习的根本局限，提出NuRL方法在训练时对难题注入自生成的抽象hint(不泄露答案)使其变为可学习样本，在6个benchmark和3个模型上一致超越GRPO且能提升模型能力上界(pass@k)。

**[On The Fragility of Benchmark Contamination Detection in Reasoning Models](llm_reasoning/on_the_fragility_of_benchmark_contamination_detection_in_reasoning_models.md)**

:   系统性研究发现 LRM 的基准污染检测极其脆弱：SFT 阶段引入的污染在经过 GRPO 训练后检测信号几乎消失（PPO 式重要性采样/裁剪是根因），而对高级 LRM 直接用 CoT 做 SFT 污染则几乎不留任何可检测痕迹，现有 10 种检测方法均接近随机猜测。

**[PrismAudio: Decomposed Chain-of-Thoughts and Multi-dimensional Rewards for Video-to-Audio Generation](llm_reasoning/prismaudio_decomposed_chain-of-thoughts_and_multi-dimensional_rewards_for_video-.md)**

:   首次将分解式 Chain-of-Thought 推理与多维度强化学习（RL）结合应用于视频到音频（V2A）生成，通过四个专门化的 CoT 模块（语义/时序/美学/空间）配合对应奖励函数，解决了目标纠缠问题，并提出 Fast-GRPO 算法大幅降低 RL 训练开销。

**[Query-Level Uncertainty in Large Language Models](llm_reasoning/query-level_uncertainty_in_large_language_models.md)**

:   提出Query-Level Uncertainty概念，通过Internal Confidence方法在生成前（单次前向传播）估计LLM能否回答给定查询，无需训练即可实现高效的自适应推理（RAG触发/模型级联/弃权）。

**[RFEval: Benchmarking Reasoning Faithfulness under Counterfactual Reasoning Intervention in Large Reasoning Models](llm_reasoning/rfeval_benchmarking_reasoning_faithfulness_under_counterfactual_reasoning_interv.md)**

:   提出 RFEval，通过立场一致性和因果影响（反事实干预）形式化推理忠实度，构建 7186 实例/7 任务基准，发现 49.7% 不忠实率，RL 后训练虽维持精度但降低忠实度。

**[SceneCOT: Eliciting Grounded Chain-of-Thought Reasoning in 3D Scenes](llm_reasoning/scenecot_eliciting_grounded_chain-of-thought_reasoning_in_3d_scenes.md)**

:   提出 SceneCOT，首个将 Chain-of-Thought 推理引入 3D 场景理解的框架，通过四阶段推理管线（任务识别→区域定位→实体接地→接地推理）将中间推理步骤显式关联到视觉 grounding，在 Beacon3D 上 Good Coherence 达到 34.7%（比最强 baseline 的 20.4% 高出 70%+）。

**[SealQA: Raising the Bar for Reasoning in Search-Augmented Language Models](llm_reasoning/sealqa_raising_the_bar_for_reasoning_in_search-augmented_language_models.md)**

:   提出SealQA挑战基准，包含111道使前沿非推理模型准确率为0的事实性问题，专门评估搜索增强LLM在噪声/冲突/误导性检索结果下的推理能力。

**[Segment-Level Attribution for Selective Learning of Long Reasoning Traces](llm_reasoning/segment-level_attribution_for_selective_learning_of_long_reasoning_traces.md)**

:   用Integrated Gradients计算长推理链中每个segment对最终答案的归因强度和方向一致性，识别重要segment进行选择性SFT，相比全CoT训练提升准确率达4.7%同时缩短输出18%。

**[The Illusion of Diminishing Returns: Measuring Long Horizon Execution in LLMs](llm_reasoning/the_illusion_of_diminishing_returns_measuring_long_horizon_execution_in_llms.md)**

:   揭示短任务基准给出"收益递减"的假象——单步准确率的微小提升在长任务中指数级放大；发现 LLM 的"自我条件化效应"（自身错误增加后续出错概率），thinking 模型可修复此效应；GPT-5 thinking 可执行超过 2100 步长任务。

**[TopoBench: Benchmarking LLMs on Hard Topological Reasoning](llm_reasoning/topobench_benchmarking_llms_on_hard_topological_reasoning.md)**

:   构建TopoBench基准(6类拓扑谜题×3难度)评估LLM的全局空间推理能力，发现前沿模型hard tier仅解决<24%，并通过因果干预实验发现错误频率不等于因果影响——低频的约束遗忘比高频的重复推理更具破坏性。

**[TumorChain: Interleaved Multimodal Chain-of-Thought Reasoning for Traceable Clinical Tumor Analysis](llm_reasoning/tumorchain_interleaved_multimodal_chain-of-thought_reasoning_for_traceable_clini.md)**

:   提出 TumorChain，面向肿瘤分析的交错多模态 CoT 推理框架，通过 1.5M CoT-VQA 数据引擎、器官引导的迭代交错推理（IIR）和混合模型协同优化，在肿瘤定位/属性分析/TNM分期上平均精度 84.41%，大幅超越 GPT-5-Mini（51.59%）。

**[Verifying Chain-of-Thought Reasoning via Its Computational Graph](llm_reasoning/verifying_chain-of-thought_reasoning_via_its_computational_graph.md)**

:   提出CRV白盒方法，通过分析LLM推理步骤的归因图（计算图）结构特征来验证CoT正确性，在Arithmetic任务上AUROC达92.47，远超黑盒(76.45)和灰盒方法，并通过因果干预成功纠正错误推理。

**[When Shallow Wins: Silent Failures and the Depth-Accuracy Paradox in Latent Reasoning](llm_reasoning/when_shallow_wins_silent_failures_and_the_depth-accuracy_paradox_in_latent_reaso.md)**

:   分析Qwen2.5-Math-7B的隐式推理发现其61%准确率中仅18.4%来自稳定忠实的推理路径，81.6%通过不一致路径得出，8.8%为"静默失败"（高置信但错误），揭示benchmark准确率掩盖计算可靠性问题。

---

## 🏥 医学图像 { #medical_imaging }

**[Adaptive Domain Shift in Diffusion Models for Cross-Modality Image Translation](medical_imaging/adaptive_domain_shift_in_diffusion_models_for_cross-modality_image_translation.md)**

:   提出CDTSDE框架，在扩散模型的逆向SDE中嵌入可学习的空间自适应域混合场 $\Lambda_t$，使跨模态翻译路径沿低能量流形前进，在MRI模态转换、SAR→光学、工业缺陷语义映射任务上以更少去噪步数实现更高保真度。

**[Adaptive Test-Time Training for Predicting Need for Invasive Mechanical Ventilation in Multi-Center Cohorts](medical_imaging/adaptive_test-time_training_for_predicting_need_for_invasive_mechanical_ventilat.md)**

:   提出AdaTTT框架，通过动态特征感知self-supervised学习（自适应掩码策略）和原型引导的部分最优传输对齐，在ICU多中心EHR数据上实现鲁棒的测试时适应，用于提前24小时预测有创机械通气需求。

**[AFD-INSTRUCTION: A Comprehensive Antibody Instruction Dataset with Functional Annotations for LLM-Based Understanding and Design](medical_imaging/afd-instruction_a_comprehensive_antibody_instruction_dataset_with_functional_ann.md)**

:   构建了首个大规模抗体功能注释指令数据集AFD-Instruction（430K+条目），通过多智能体文献抽取pipeline对齐抗体序列与自然语言功能描述，用于指令微调通用LLM使其掌握抗体理解和功能导向设计能力，在5类分类任务上平均准确率提升20+点。

**[An Orthogonal Learner for Individualized Outcomes in Markov Decision Processes](medical_imaging/an_orthogonal_learner_for_individualized_outcomes_in_markov_decision_processes.md)**

:   从因果推断视角重新审视Q函数估计问题，揭示传统Q回归和FQE是具有插入偏差的plug-in学习器，提出DRQQ-learner——一种双重鲁棒、Neyman正交、准oracle高效的Q函数估计器，通过推导有效影响函数构建去偏两阶段损失函数，在Taxi和Frozen Lake环境中验证了其优越性。

**[AntigenLM: Structure-Aware DNA Language Modeling for Influenza](medical_imaging/antigenlm_structure-aware_dna_language_modeling_for_influenza.md)**

:   AntigenLM 是一个保留基因组功能单元完整性的 GPT-2 风格 DNA 语言模型，通过在流感病毒全基因组上预训练并微调，能够自回归预测未来流行毒株的抗原序列，在氨基酸错配率上显著优于进化模型 beth-1 和通用基因组模型。

**[ATPO: Adaptive Tree Policy Optimization for Multi-Turn Medical Dialogue](medical_imaging/atpo_adaptive_tree_policy_optimization_for_multi-turn_medical_dialogue.md)**

:   提出 ATPO（自适应树策略优化）算法，将多轮医疗对话建模为层级马尔可夫决策过程（H-MDP），通过不确定性感知的自适应树扩展机制动态分配rollout预算，结合Bellman误差和动作值方差的复合不确定性度量来引导探索，在三个医学对话基准上以Qwen3-8B超越GPT-4o。

**[Augmenting Representations With Scientific Papers](medical_imaging/augmenting_representations_with_scientific_papers.md)**

:   提出首个将 X 射线光谱与科学文献通过对比学习对齐的多模态基础模型框架，在共享潜在空间中实现 20% Recall@1% 的跨模态检索，物理参数估计提升 16–18%，同时发现候选脉动超亮 X 射线源等罕见天体。

**[Benchmarking ECG FMs: A Reality Check Across Clinical Tasks](medical_imaging/benchmarking_ecg_fms_a_reality_check_across_clinical_tasks.md)**

:   对8个ECG基础模型在12个数据集、26个临床任务上进行"现实检验"式全面基准评测，发现紧凑的结构化状态空间模型（SSM）ECG-CPC在7个任务类别中的5个上超越了大规模Transformer，证明架构设计比模型规模更重要。

**[Boosting Medical Visual Understanding From Multi-Granular Language Learning](medical_imaging/boosting_medical_visual_understanding_from_multi-granular_language_learning.md)**

:   提出 Multi-Granular Language Learning (MGLL)，一个即插即用的对比学习框架，通过 soft CLIP loss、point-wise loss 和 smooth KL 散度联合优化，实现医学图像与多标签多粒度文本描述的对齐，在眼底和 X 光数据集上全面超越 SOTA 方法，并可作为视觉编码器嵌入多模态大语言模型提升诊断准确率最高达 34.1%。

**[Brain-IT: Image Reconstruction from fMRI via Brain-Interaction Transformer](medical_imaging/brain-it_image_reconstruction_from_fmri_via_brain-interaction_transformer.md)**

:   提出 Brain-IT 框架，通过脑启发式的 Brain Interaction Transformer (BIT) 将功能相似的脑体素聚类为跨被试共享的 Brain Token，并从中预测局部化的语义和结构图像特征，实现从 fMRI 到图像的高保真重建，仅用 1 小时数据即达到先前方法 40 小时的性能。

**[Brain-Semantoks: Learning Semantic Tokens of Brain Dynamics with a Self-Distilled Foundation Model](medical_imaging/brain-semantoks_learning_semantic_tokens_of_brain_dynamics_with_a_self-distilled.md)**

:   提出 Brain-Semantoks，一种基于语义分词器和自蒸馏目标的 fMRI 基础模型，将大脑功能网络聚合为鲁棒的语义 token，并通过跨时间视角的一致性学习抽象的脑动态表征，在线性探测设置下即可达到 SOTA 性能。

**[Bridging Explainability and Embeddings: BEE Aware of Spuriousness](medical_imaging/bridging_explainability_and_embeddings_bee_aware_of_spuriousness.md)**

:   提出BEE框架，通过分析微调如何扰动预训练表征的权重空间几何结构，直接从分类器学到的权重中识别和命名虚假相关性（spurious correlations），无需反例样本即可发现隐藏的数据偏差，在ImageNet-1k上发现可导致准确率下降高达95%的虚假关联。

**[Can SAEs Reveal and Mitigate Racial Biases of LLMs in Healthcare?](medical_imaging/can_saes_reveal_and_mitigate_racial_biases_of_llms_in_healthcare.md)**

:   研究稀疏自编码器（SAE）能否揭示和缓解 LLM 在医疗场景中的种族偏见：发现 SAE 能识别出与种族相关的有害联想（如黑人与暴力），但在复杂临床任务中缓解偏见的效果有限（FLDD < 3%），远不如简单的提示策略（FLDD 8-15%）。

**[CARE: Towards Clinical Accountability in Multi-Modal Medical Reasoning with an Evidence-Grounded Agentic Framework](medical_imaging/care_towards_clinical_accountability_in_multi-modal_medical_reasoning_with_an_ev.md)**

:   提出 CARE Agent 框架，将医学 VQA 分解为实体提议、指称分割和证据引导推理三个专家模块，通过 GPT-5 作为动态协调器，在医学 VQA 基准上以 77.54% 准确率超越 32B 模型。

**[COMPASS: Robust Feature Conformal Prediction for Medical Segmentation Metrics](medical_imaging/compass_robust_feature_conformal_prediction_for_medical_segmentation_metrics.md)**

:   提出 COMPASS 框架，在分割模型的中间特征空间而非输出空间做共形预测，通过沿 Jacobian 确定的低维敏感子空间扰动特征来构建预测区间，在多个医学分割数据集上以更紧凑的区间达到目标覆盖率。

**[Decentralized Attention Fails Centralized Signals: Rethinking Transformers for Medical Time Series](medical_imaging/decentralized_attention_fails_centralized_signals_rethinking_transformers_for_me.md)**

:   提出 TeCh 框架，用核心 Token 聚合-再分配（CoTAR）替代标准注意力处理医学时间序列的通道依赖，将复杂度从 $O(n^2)$ 降至 $O(n)$，在 APAVA 上精度 86.86%（超 Medformer 12.13%），内存仅 33%、推理时间仅 20%。

**[Deep Hierarchical Learning with Nested Subspace Networks for Large Language Models](medical_imaging/deep_hierarchical_learning_with_nested_subspace_networks_for_large_language_mode.md)**

:   提出嵌套子空间网络（NSN），通过低秩分解使线性层形成严格嵌套的子空间层次，配合不确定性感知多秩训练，使单个模型在测试时可即时调节计算量与性能的权衡（50% FLOPs 减少仅损失 5% 精度），且可后验应用于预训练 LLM。

**[DISCO: Densely-overlapping Cell Instance Segmentation via Adjacency-aware Collaborative Coloring](medical_imaging/disco_densely-overlapping_cell_instance_segmentation_via_adjacency-aware_collabo.md)**

:   提出基于图着色理论的密集重叠细胞实例分割框架 DISCO，通过"显式标记冲突+隐式消歧邻接约束"的分治策略，在高密度病理图像上 PQ 提升 7.08%。

**[DistMLIP: A Distributed Inference Platform for Machine Learning Interatomic Potentials](medical_imaging/distmlip_a_distributed_inference_platform_for_machine_learning_interatomic_poten.md)**

:   提出 DistMLIP 分布式推理平台，基于零冗余图级并行化策略（graph-level parallelization），解决现有机器学习原子间势（MLIP）缺乏多 GPU 支持的问题，在 8 GPU 上实现接近百万原子的模拟，比空间分区方法快达 8 倍且能模拟 3.4 倍更大的系统。

**[DriftLite: Lightweight Drift Control for Inference-Time Scaling of Diffusion Models](medical_imaging/driftlite_lightweight_drift_control_for_inference-time_scaling_of_diffusion_mode.md)**

:   DriftLite 提出在 Fokker-Planck 方程中利用漂移-势函数的自由度，通过轻量级线性系统求解最优控制漂移来主动稳定粒子权重，以最小代价解决 Sequential Monte Carlo 中的权重退化问题，在高斯混合、分子系统和蛋白质-配体共折叠任务上大幅超越 Guidance-SMC 基线。

**[Dual Distillation for Few-Shot Anomaly Detection](medical_imaging/dual_distillation_for_few-shot_anomaly_detection.md)**

:   提出双蒸馏框架 D24FAD，结合 query 图像上的教师-学生蒸馏（TSD）和 support 图像上的学生自蒸馏（SSD），辅以学习权重机制（L2W）自适应评估 support 重要性，在 APTOS 眼底数据集上仅用 2-shot 达到 100% AUROC。

**[Extending Sequence Length is Not All You Need: Effective Integration of Multimodal Signals for Gene Expression Prediction](medical_imaging/extending_sequence_length_is_not_all_you_need_effective_integration_of_multimoda.md)**

:   挑战基因表达预测中"越长越好"的长序列建模范式，发现当前 SSM 模型本质上只利用近端信息；进而识别出背景染色质信号（DNase-seq/Hi-C）作为混杂变量引入虚假关联，提出 Prism 框架通过后门调整去混杂，仅用 2k 短序列即超越 200k 长序列的 SOTA。

**[HistoPrism: Unlocking Functional Pathway Analysis from Pan-Cancer Histology via Gene Expression Prediction](medical_imaging/histoprism_unlocking_functional_pathway_analysis_from_pan-cancer_histology_via_g.md)**

:   提出 HistoPrism，基于 Transformer 的跨癌种基因表达预测框架，通过交叉注意力注入癌种条件、引入基因通路一致性（GPC）评估指标，仅用 500 张 WSI 即实现优于 STPath（需 38K WSI）的通路级预测，在 86% 的 Hallmark 通路上更优。

**[Intrinsic Lorentz Neural Network](medical_imaging/intrinsic_lorentz_neural_network.md)**

:   提出完全内禀（fully intrinsic）的双曲神经网络 ILNN，所有运算均在 Lorentz 模型内完成，消除了现有方法中混合欧几里得操作的几何不一致性，在图像分类、基因组学和图分类上取得 SOTA。

**[MedAgentGym: A Scalable Agentic Training Environment for Code-Centric Reasoning in Biomedical Data Science](medical_imaging/medagentgym_agentic_training_biomedical.md)**

:   构建了首个统一的生物医学数据科学 Agent 训练环境 MedAgentGym，包含 72,413 个任务实例（12 个真实场景、129 个类别），配备可执行沙盒和可验证 ground truth，基准评估 29 个 LLM，并通过离线/在线 RL 训练出 Med-Copilot（分别 +43%/+45% 提升），达到与 GPT-4o 竞争的性能同时保持成本效益和隐私保护。

**[Omni-iEEG: A Large-Scale, Comprehensive iEEG Dataset and Benchmark for Epilepsy Research](medical_imaging/omni-ieeg_a_large-scale_comprehensive_ieeg_dataset_and_benchmark_for_epilepsy_re.md)**

:   构建 Omni-iEEG，首个多中心标准化颅内脑电（iEEG）数据集（302名患者/8中心/36177标注事件），配套 HFO 分类和致病脑区识别基准，发现端到端分段模型（AUC 0.8061）可匹配临床验证的生物标志物方法（AUC 0.7351），且音频预训练模型可有效迁移。

**[Scalable Spatio-Temporal SE(3) Diffusion for Long-Horizon Protein Dynamics](medical_imaging/scalable_spatio-temporal_se3_diffusion_for_long-horizon_protein_dynamics.md)**

:   提出 STAR-MD，一个 SE(3) 等变的因果扩散 Transformer，通过联合时空注意力和上下文噪声扰动实现微秒级蛋白质动力学轨迹生成，在 ATLAS 基准上所有指标达到 SOTA，且能稳定外推到训练中未见的微秒时间尺度。

**[Scaling with Collapse: Efficient and Predictable Training of LLM Families](medical_imaging/scaling_with_collapse_efficient_and_predictable_training_of_llm_families.md)**

:   证明 LLM 家族的训练损失曲线在优化超参数与数据预算匹配时会“崩塞”到同一条通用曲线上，并利用这一现象实现两个实用应用：(1) 偏离崩塞作为训练病理的早期诊断信号，(2) 崩塞曲线的可预测性实现大规模超参调优的早停。

**[SynCoGen: Synthesizable 3D Molecule Generation via Joint Reaction and Coordinate Modeling](medical_imaging/syncogen_synthesizable_3d_molecule_generation_via_joint_reaction_and_coordinate_.md)**

:   提出 SynCoGen，联合建模 2D 合成路线（掩码图扩散）和 3D 几何（流匹配），确保生成分子既化学可合成又空间有效，在无条件 3D+图生成和连接子设计中达到 SOTA。

---

## 🧊 3D 视觉 { #3d_vision }

**[3DGEER: 3D Gaussian Rendering Made Exact and Efficient for Generic Cameras](3d_vision/3dgeer_3d_gaussian_rendering_made_exact_and_efficient_for_generic_cameras.md)**

:   提出 3DGEER 框架，通过推导沿光线积分高斯密度的闭式解、设计粒子包围截锥体 (PBF) 进行精确高效的光线-粒子关联、以及引入双极等角投影 (BEAP) 统一宽视场相机表示，在任意相机模型下实现了几何精确且实时高效的 3D 高斯渲染，在鱼眼和针孔数据集上全面超越现有方法。

**[A Genetic Algorithm for Navigating Synthesizable Molecular Spaces](3d_vision/a_genetic_algorithm_for_navigating_synthesizable_molecular_spaces.md)**

:   提出 SynGA，一种直接在合成路线（合成树）上操作的遗传算法，通过自定义的交叉和变异算子将搜索严格约束在可合成分子空间内，结合 ML 驱动的构建块过滤实现 SOTA 的可合成类似物搜索和属性优化性能。

**[A Step to Decouple Optimization in 3DGS](3d_vision/a_step_to_decouple_optimization_in_3dgs.md)**

:   深入分析 3DGS 优化中被忽视的更新步耦合（不可见视点下的隐式更新和动量重缩放）和梯度耦合（正则化与光度损失在 Adam 动量中的耦合），通过解耦和重组提出 AdamW-GS 优化器，在不引入额外剪枝操作的情况下同时提升重建质量和减少冗余原语。

**[Augmented Radiance Field: A General Framework for Enhanced Gaussian Splatting](3d_vision/augmented_radiance_field_a_general_framework_for_enhanced_gaussian_splatting.md)**

:   提出增强辐射场 (Augmented Radiance Field) 框架，通过设计具有视角相关不透明度的增强高斯核来显式建模高光分量，并引入误差驱动的补偿策略（2D 高斯初始化 → 逆投影至 3D → 联合优化），作为后处理即插即用地增强现有 3DGS 场景，在多个数据集上超越 SOTA NeRF 方法，同时仅需二阶球谐即可捕获复杂光照。

**[cadrille: Multi-modal CAD Reconstruction with Reinforcement Learning](3d_vision/cadrille_multi-modal_cad_reconstruction_with_reinforcement_learning.md)**

:   cadrille 是首个同时处理点云、多视角图像和文本输入的多模态 CAD 重建模型，通过 VLM 基础架构 + SFT + RL 微调的三阶段训练范式，在 10 个 CAD 重建基准上达到 SOTA，尤其是 RL 微调将无效率降至接近 0%。

**[CloDS: Visual-Only Unsupervised Cloth Dynamics Learning in Unknown Conditions](3d_vision/clods_visual-only_unsupervised_cloth_dynamics_learning_in_unknown_conditions.md)**

:   CloDS 提出首个从多视角视频中无监督学习布料动力学的框架，通过 Spatial Mapping Gaussian Splatting 建立 2D 图像到 3D 网格的可微映射，结合双位置不透明度调制解决自遮挡问题，使 GNN 在无物理参数监督下就能学到接近全监督水平的布料动力学。

**[Color3D: Controllable and Consistent 3D Colorization with Personalized Colorizer](3d_vision/color3d_controllable_and_consistent_3d_colorization_with_personalized_colorizer.md)**

:   Color3D 提出"只上色一张关键视角→微调个性化 colorizer→传播颜色到所有视角和时间步"的范式，将复杂的 3D 上色问题转化为单图上色+颜色传播问题，在静态和动态 3D 场景上都实现了丰富色彩、跨视角一致性和用户可控性的统一。

**[COOPERTRIM: Adaptive Data Selection for Uncertainty-Aware Cooperative Perception](3d_vision/coopertrim_adaptive_data_selection_for_uncertainty-aware_cooperative_perception.md)**

:   提出 CooperTrim 自适应特征选择框架，通过共形时序不确定性度量评估特征相关性，并用数据驱动机制动态决定共享数量，在协同语义分割中实现 80.28% 带宽降低且性能可比，首次将选择性共享应用于协同分割任务。

**[EgoNight: Towards Egocentric Vision Understanding at Night with a Challenging Benchmark](3d_vision/egonight_towards_egocentric_vision_understanding_at_night_with_a_challenging_ben.md)**

:   构建首个系统性夜间第一人称视觉基准 EgoNight，包含日夜对齐的合成/真实视频和 3658 个 QA 对（12种类型，300+小时人工标注），揭示所有 SOTA MLLM 在日夜转换中的显著性能下降。

**[FastGHA: Generalized Few-Shot 3D Gaussian Head Avatars with Real-Time Animation](3d_vision/fastgha_generalized_few-shot_3d_gaussian_head_avatars_with_real-time_animation.md)**

:   提出 FastGHA，一个前馈式少样本 3D 高斯头部化身生成框架，从 4 张任意表情/视角的输入图像在 ~1 秒内重建可动画的 3D 高斯头部，支持 62 FPS 实时动画，在 Ava-256 上 PSNR 达到 22.5 dB（超越 Avat3r 的 20.7，且快 7.75 倍）。

**[FeDaL: Federated Dataset Learning for General Time Series Foundation Models](3d_vision/fedal_federated_dataset_learning_for_general_time_series_foundation_models.md)**

:   提出 FeDaL 联邦框架从头训练通用时序基础模型（TSFM），通过客户端域偏差消除（DBE）和服务器全局偏差消除（GBE）处理数据集级异质性，在8个任务上超越54个baseline。

**[Fused-Planes: Why Train a Thousand Tri-Planes When You Can Share?](3d_vision/fused-planes_why_train_a_thousand_tri-planes_when_you_can_share.md)**

:   提出 Fused-Planes，通过宏观-微观分解将 Tri-Plane 表示分为共享的类级基平面（macro）和对象特有的细节平面（micro），结合潜空间渲染，实现 7× 训练加速、3× 内存压缩，同时保持甚至超越独立 Tri-Plane 的重建质量。

**[Geometry-aware 4D Video Generation for Robot Manipulation](3d_vision/geometry-aware_4d_video_generation_for_robot_manipulation.md)**

:   提出几何感知的4D视频生成框架，通过跨视角 pointmap 对齐监督在视频扩散模型中强制多视角3D一致性，无需相机位姿输入即可从新视角生成时空对齐的未来 RGB-D 序列，并可直接用 FoundationPose 从生成视频中恢复机器人末端执行器轨迹。

**[MoE-GS: Mixture of Experts for Dynamic Gaussian Splatting](3d_vision/moe-gs_mixture_of_experts_for_dynamic_gaussian_splatting.md)**

:   提出 MoE-GS，首个将混合专家架构引入动态高斯泼溅的框架，通过 Volume-aware Pixel Router 自适应融合多种异构变形先验（HexPlane/逐高斯/多项式/插值），在 N3V 和 Technicolor 数据集上一致超越 SOTA，并通过单次渲染、门控剪枝和知识蒸馏保持效率。

**[Mono4DGS-HDR: High Dynamic Range 4D Gaussian Splatting from Alternating-exposure Monocular Videos](3d_vision/mono4dgs-hdr_high_dynamic_range_4d_gaussian_splatting_from_alternating-exposure_.md)**

:   首次解决从无位姿交替曝光单目视频重建可渲染 4D HDR 场景的问题，通过两阶段优化（正交视频空间 → 世界空间）、Video-to-World 高斯变换策略和时间亮度正则化，在合成数据上达到 37.64 dB HDR PSNR、161 FPS，全面超越现有方法。

**[Omni-View: Unlocking How Generation Facilitates Understanding in Unified 3D Model based on Multiview images](3d_vision/omni-view_unlocking_how_generation_facilitates_understanding_in_unified_3d_model.md)**

:   构建统一的3D场景理解与生成模型 Omni-View，通过纹理模块（新视角合成）和几何模块（深度/位姿估计）的生成能力增强理解性能，在 VSI-Bench 上达到 55.4 分超越所有现有专用3D理解模型。

**[PartSAM: A Scalable Promptable Part Segmentation Model Trained on Native 3D Data](3d_vision/partsam_a_scalable_promptable_part_segmentation_model_trained_on_native_3d_data.md)**

:   提出首个在大规模原生 3D 数据上训练的可提示部件分割模型 PartSAM，采用 triplane 双分支编码器（冻结 SAM 先验 + 可学习 3D 分支）和 SAM 风格解码器，通过模型在环标注流程构建 500 万+形状-部件对，在开放世界设置下单次点击即超越 Point-SAM 90%+。

**[PD²GS: Part-Level Decoupling and Continuous Deformation of Articulated Objects via Gaussian Splatting](3d_vision/pd2gs_part-level_decoupling_and_continuous_deformation_of_articulated_objects_vi.md)**

:   提出 PD²GS 框架，通过学习共享的 canonical 高斯场并将每个交互状态建模为其连续形变，实现铰接物体的部件级解耦、重建和连续控制，采用粗到细的运动轨迹聚类 + SAM 引导的边界细化，无需手动监督。

**[Peering into the Unknown: Active View Selection with Neural Uncertainty Maps for 3D Reconstruction](3d_vision/peering_into_the_unknown_active_view_selection_with_neural_uncertainty_maps_for_.md)**

:   提出 PUN 方法，通过轻量级 UPNet（ViT）从单张图像预测神经不确定性图（球面坐标系下所有候选视点的不确定性分布），避免迭代 NeRF 重训练，仅用一半视点达到全视点上界的重建质量，实现 400 倍加速和 50%+ 计算节省。

**[Quantized Visual Geometry Grounded Transformer](3d_vision/quantized_visual_geometry_grounded_transformer.md)**

:   针对十亿级 3D 重建模型 VGGT 的部署需求，提出首个专用 PTQ 框架 QuantVGGT，通过双重平滑细粒度量化（Hadamard 旋转 + 通道平滑）解决特殊 token 导致的重尾分布，以及噪声过滤多样化采样解决校准不稳定问题，4-bit 量化实现 3.7× 内存压缩和 2.5× 加速，保持 98%+ 精度。

**[Weight Space Representation Learning on Diverse NeRF Architectures](3d_vision/weight_space_representation_learning_on_diverse_nerf_architectures.md)**

:   提出首个能处理多种 NeRF 架构（MLP/tri-plane/hash table）权重的表示学习框架，通过 Graph Meta-Network 编码器 + SigLIP 对比损失构建架构无关的潜在空间，在 13 种 NeRF 架构上实现分类、检索和语言任务，并能泛化到训练时未见的架构。

---

## 🛡️ AI 安全 { #ai_safety }

**[Adaptive Methods Are Preferable in High Privacy Settings: An SDE Perspective](ai_safety/adaptive_methods_are_preferable_in_high_privacy_settings_an_sde_perspective.md)**

:   首次用 SDE（随机微分方程）框架分析差分隐私优化器，证明在严格隐私设置（小 ε）下自适应方法（DP-SignSGD/DP-Adam）在隐私-效用 trade-off 和超参数鲁棒性上都优于 DP-SGD。

**[Atex-Cf Attack-Informed Counterfactual Explanations For Graph Neural Networks](ai_safety/atex-cf_attack-informed_counterfactual_explanations_for_graph_neural_networks.md)**

:   提出 ATEX-CF 框架，首次将对抗攻击的边添加策略与反事实解释的边删除策略统一起来，通过联合优化预测翻转、稀疏性和合理性，为 GNN 生成更忠实、更简洁、更合理的实例级反事实解释。

**[Attention Smoothing Is All You Need For Unlearning](ai_safety/attention_smoothing_is_all_you_need_for_unlearning.md)**

:   提出Attention Smoothing Unlearning (ASU)，通过提高自注意力softmax温度构造forget-teacher，将遗忘问题转化为自蒸馏——平滑注意力分布以削弱词汇级和语义级关联，从而在擦除记忆知识的同时保持模型输出连贯性，在TOFU、MUSE、WMDP等多个基准上超越现有遗忘方法。

**[AudioTrust: Benchmarking the Multifaceted Trustworthiness of Audio Large Language Models](ai_safety/audiotrust_benchmarking_the_multifaceted_trustworthiness_of_audio_large_language.md)**

:   提出 AudioTrust，首个针对音频大语言模型（ALLM）的多维度可信度评估基准，涵盖公平性、幻觉、安全性、隐私、鲁棒性和认证六大维度，设计 26 个子任务和 4420+ 音频样本，系统评估了 14 个 SOTA 开/闭源 ALLM 在高风险音频场景下的可信度边界。

**[Back to Square Roots: An Optimal Bound on the Matrix Factorization Error for Multi-Epoch Differentially Private SGD](ai_safety/back_to_square_roots_an_optimal_bound_on_the_matrix_factorization_error_for_mult.md)**

:   提出 Banded Inverse Square Root (BISR) 矩阵分解方法，通过对逆相关矩阵（而非相关矩阵本身）施加带状结构，首次在多轮参与差分隐私 SGD 中实现渐近最优的分解误差界，并配套低存储优化变体 BandInvMF。

**[BEAT: Visual Backdoor Attacks on VLM-based Embodied Agents via Contrastive Trigger Learning](ai_safety/beat_visual_backdoor_attacks_on_vlm-based_embodied_agents_via_contrastive_trigge.md)**

:   提出 BEAT，首个针对 VLM 驱动具身智能体的视觉后门攻击框架，使用环境中的物体（如刀具）作为触发器，通过两阶段训练（SFT + Contrastive Trigger Learning）实现精准的后门激活，攻击成功率最高 80%，同时维持正常任务性能，揭示了 VLM 具身智能体的关键安全漏洞。

**[Beware Untrusted Simulators -- Reward-Free Backdoor Attacks in Reinforcement Learning](ai_safety/beware_untrusted_simulators_--_reward-free_backdoor_attacks_in_reinforcement_lea.md)**

:   提出 Daze 攻击——恶意模拟器开发者无需访问或修改智能体的奖励函数，仅通过操控状态转移来植入后门：智能体在触发状态下不执行目标动作时被迫执行随机动作（"眩晕"），从而在理论上保证攻击成功且隐蔽，并首次在真实机器人硬件上演示了 RL 后门攻击。

**[Beyond Match Maximization and Fairness: Retention-Optimized Two-Sided Matching](ai_safety/beyond_match_maximization_and_fairness_retention-optimized_two-sided_matching.md)**

:   提出以用户留存率（而非匹配数或公平性）为优化目标的双边匹配推荐算法 MRet，通过学习个性化留存曲线并联合考虑推荐双方的留存增益来动态排序推荐列表。

**[BiasBusters: Uncovering and Mitigating Tool Selection Bias in Large Language Models](ai_safety/biasbusters_uncovering_and_mitigating_tool_selection_bias_in_large_language_mode.md)**

:   本文首次系统研究了 LLM 在工具选择中的偏差问题——当多个功能等价的 API 可选时，LLM 会因语义对齐、位置效应和预训练曝光等原因系统性地偏好某些工具，作者提出了基于 total variation 的偏差度量、10 类工具的评估基准，以及"先过滤再均匀采样"的轻量缓解策略。

**[Bridging Fairness and Explainability: Can Input-Based Explanations Promote Fairness in Hate Speech Detection?](ai_safety/bridging_fairness_and_explainability_can_input-based_explanations_promote_fairne.md)**

:   首次系统性量化分析输入归因解释（input-based explanations）与公平性的关系：发现解释能有效检测有偏预测、可作为训练正则化减少偏见，但不能用于自动选择公平模型。

**[Dataless Weight Disentanglement in Task Arithmetic via Kronecker-Factored Approximate Curvature](ai_safety/dataless_weight_disentanglement_in_task_arithmetic_via_kronecker-factored_approx.md)**

:   提出 TAK 方法，将任务算术中的表征漂移正则化等价为 Jacobian Gram 矩阵的二次型，利用 KFAC 近似实现无需外部任务数据的高效权重解纠缠，在任务加法和任务否定上达到 SOTA。

**[Efficient Resource-Constrained Training of Transformers via Subspace Optimization](ai_safety/efficient_resource-constrained_training_of_transformers_via_subspace_optimizatio.md)**

:   提出 WASI（Weight-Activation Subspace Iteration），基于"微调过程中参数子空间稳定"的假设，同时压缩 Transformer 的权重（SVD + Gram-Schmidt 子空间迭代）和激活（Tucker 分解），实现训练和推理都在低秩表示中完成，达到 62× 训练内存压缩和 Raspberry Pi 5 上 1.4× 加速，且精度损失可忽略。

**[Erase or Hide? Suppressing Spurious Unlearning Neurons for Robust Unlearning](ai_safety/erase_or_hide_suppressing_spurious_unlearning_neurons_for_robust_unlearning.md)**

:   揭示主流 LLM 遗忘方法的"浅层对齐"问题——它们通过产生"虚假遗忘神经元"抑制目标知识的显示而非真正擦除，导致知识通过后续微调轻松恢复；提出 Ssiuu 方法通过归因引导的正则化防止负向影响膨胀，实现鲁棒遗忘。

**[From Static Benchmarks to Dynamic Protocol: Agent-Centric Text Anomaly Detection for Evaluating LLM Reasoning](ai_safety/from_static_benchmarks_to_dynamic_protocol_agent-centric_text_anomaly_detection_.md)**

:   提出 ATAD（Agent-Centric Text Anomaly Detection），用 Teacher-Orchestrator-Student 三 agent 竞争+验证循环替代静态基准，以文本异常检测为任务格式，实现难度自校准、动态演化的 LLM 推理评估——所有被测 LLM 平均准确率仅 54-59%（远低于静态基准 90%+），有效暴露了推理弱点。

**[Improving the Trade-off Between Watermark Strength and Speculative Sampling Efficiency for Language Models](ai_safety/improving_the_trade-off_between_watermark_strength_and_speculative_sampling_effi.md)**

:   将 LLM 水印强度从二值定义升级为连续量化指标（期望 KL 散度），完全刻画了水印强度与 speculative sampling 效率的 Pareto trade-off 曲线，并提出 pseudorandom acceptance 机制同时达到最大水印强度和最大采样效率。

**[Inoculation Prompting: Eliciting Traits from LLMs during Training Can Suppress Them at Test-Time](ai_safety/inoculation_prompting_eliciting_traits_from_llms_during_training_can_suppress_th.md)**

:   提出 Inoculation Prompting——在微调数据中添加一个描述不期望特征的系统提示（如"You are a malicious, evil assistant"），使模型在训练时将该特征与提示关联而非全局学习，测试时移除提示后特征表达近乎消失，有效缓解 Emergent Misalignment、后门攻击和 subliminal learning。

**[Learnability and Privacy Vulnerability are Entangled in a Few Critical Weights](ai_safety/learnability_and_privacy_vulnerability_are_entangled_in_a_few_critical_weights.md)**

:   发现可学习性和隐私脆弱性纠缠在少量关键权重中（~0.7-0.9 相关），提出 CWRF 方法将关键权重回退到初始化然后微调隐私脆弱部分，在抵抗 LiRA/RMIA 攻击时优于现有防御，仅需 0.1% 关键权重。

**[Measuring Physical-World Privacy Awareness of Large Language Models: An Evaluation Benchmark](ai_safety/measuring_physical-world_privacy_awareness_of_large_language_models_an_evaluatio.md)**

:   提出 EAPrivacy—首个评估 LLM 物理世界隐私感知的 4 层级基准（400+ 场景）：发现所有 frontier 模型存在"非对称保守"——对任务执行过度保守但对隐私保护不足，且开启 reasoning 模式反而降低隐私表现（Tier 1-3），最佳模型在动态环境中仅 59% 准确率。

**[Membership Privacy Risks of Sharpness Aware Minimization](ai_safety/membership_privacy_risks_of_sharpness_aware_minimization.md)**

:   发现反直觉现象：SAM 比 SGD 泛化更好但更容易被成员推断攻击（MIA）——SAM 的锐度正则化隐式降低输出方差，使成员/非成员的信号分离更清晰，攻击 AUC 提升 1-3.4%；机制分析表明 SAM 的泛化收益来自"结构化记忆"（学习少数类子模式）而非简单泛化。

**[RedSage: A Cybersecurity Generalist LLM](ai_safety/redsage_a_cybersecurity_generalist_llm.md)**

:   RedSage 是开源的 8B 网络安全专用 LLM，通过 11.8B token 安全语料持续预训练 + 266K 样本智能体增强 SFT + 偏好对齐三阶段训练，配套提出覆盖知识/技能/工具三维的 RedSage-Bench（30K MCQ + 240 开放题），在现有网络安全基准上达到 SOTA。

**[Train Once, Answer All: Many Pretraining Experiments for the Cost of One](ai_safety/train_once_answer_all_many_pretraining_experiments_for_the_cost_of_one.md)**

:   提出在单次 LLM 预训练中同时运行多个独立实验的方法论框架，在训练 2.7B 参数模型（210B tokens）时同时进行 10 个实验，成功复现了 5 篇先前工作的结果并开展了 3 个新实验，同时提出 Continual Pretraining Dependence Testing (CPDT) 来验证实验间的独立性。

---

## 📦 模型压缩 { #model_compression }

**[A Fano-Style Accuracy Upper Bound for LLM Single-Pass Reasoning in Multi-Hop QA](model_compression/a_fano-style_accuracy_upper_bound_for_llm_single-pass_reasoning_in_multi-hop_qa.md)**

:   用信息论推导出 LLM 单次推理在多跳 QA 中的 Fano 式准确率上界，揭示当任务信息需求超过模型输出容量时准确率会"悬崖式"骤降的现象，并据此设计多轮推理框架 InfoQA，通过容量感知分解、依赖显式工作流和迭代查询压缩来突破单次推理瓶颈。

**[A Recovery Guarantee for Sparse Neural Networks](model_compression/a_recovery_guarantee_for_sparse_neural_networks.md)**

:   证明了 ReLU 神经网络的首个稀疏恢复保证：对两层标量输出网络，当训练数据为高斯随机采样时，基于凸重构的迭代硬阈值 (IHT) 算法可精确恢复稀疏网络权重，且内存需求仅与非零权重数线性增长。

**[A State-Transition Framework for Efficient LLM Reasoning](model_compression/a_state-transition_framework_for_efficient_llm_reasoning.md)**

:   提出将 LLM 推理过程建模为状态转移过程的高效推理框架，用 Linear Attention 将历史推理步骤的信息压缩为状态矩阵，使注意力复杂度从 $O(C^2)$ 降为 $O(C)$、KV cache 从 $O(C)$ 降为 $O(1)$，同时不缩短 CoT 序列，保持推理能力。额外的动量 momentum 策略缓解了噪声推理步导致的 overthinking 问题。

**[ABBA-Adapters: Efficient and Expressive Fine-Tuning of Foundation Models](model_compression/abba-adapters_efficient_and_expressive_fine-tuning_of_foundation_models.md)**

:   提出 ABBA 适配器，将权重更新参数化为两个独立可学习的低秩矩阵的 Hadamard 积 $\Delta W = s(B_1A_1) \odot (B_2A_2)$，在相同参数预算下实现远高于 LoRA 的有效秩（$r_1 \cdot r_2$ vs $r$），并通过 Khatri-Rao 重构实现与 LoRA 相当的内存效率，在算术和常识推理任务上显著超越现有 PEFT 方法。

**[ACPBench Hard: Unrestrained Reasoning about Action, Change, and Planning](model_compression/acpbench_hard_unrestrained_reasoning_about_action_change_and_planning.md)**

:   构建 ACPBench Hard，一个基于 PDDL 规划的 8 类开放式生成推理 benchmark（1040 题），要求 LLM 生成可适用动作集、状态转移、可达性判断、里程碑识别、计划验证等，配备精确的符号验证器，测试发现即使最强的推理模型（o1）在多数任务上也低于 65%，暴露了 LLM 在规划推理方面的根本不足。

**[ActivationReasoning: Logical Reasoning in Latent Activation Spaces](model_compression/activationreasoning_logical_reasoning_in_latent_activation_spaces.md)**

:   提出 ActivationReasoning (AR) 框架，在 LLM 的潜在激活空间（通过 SAE 提取的特征）上嵌入显式逻辑推理，通过三阶段流程（发现概念表征→检测激活命题→逻辑规则推理）实现多跳推理、概念组合和安全控制，在 PrOntoQA 上 8B 模型达到 95%+ 准确率超越 GPT-4o。

**[Adaptive Width Neural Networks](model_compression/adaptive_width_neural_networks.md)**

:   提出AWN框架，通过变分推断在训练过程中自动学习每层的无上界宽度（神经元数量），利用单调递减的重要性函数对神经元施加软排序，实现宽度自适应于任务难度，并支持零成本的训练后截断压缩。

**[AMiD: Knowledge Distillation for LLMs with α-mixture Assistant Distribution](model_compression/amid_knowledge_distillation_for_llms_with_α-mixture_assistant_distribution.md)**

:   提出α-mixture assistant distribution及统一蒸馏框架AMiD，通过引入新设计变量α（控制教师-学生分布插值路径的几何形状）泛化了现有辅助分布方法（m-mixture和e-mixture为α=±1的特例），并证明了在任意散度和α下的最优性保证，在多个LLM蒸馏基准上取得SOTA性能。

**[AnyBCQ: Hardware Efficient Flexible Binary-Coded Quantization for Multi-Precision LLMs](model_compression/anybcq_hardware_efficient_flexible_binary-coded_quantization_for_multi-precision.md)**

:   提出AnyBCQ，基于二进制编码量化(BCQ)的多精度LLM量化框架，通过渐进式精度扩展（冻结已有bit-plane+添加残差bit-plane）支持单个模型在2-4bit之间动态切换，专设CUDA内核直接在bit-plane级别计算避免查表/转置开销，在2-bit下准确率大幅超越Any-Precision LLM（MMLU 35.3% vs 24.7%），吞吐量最高达到FP16的3.0x。

**[Beyond Linear Probes: Dynamic Safety Monitoring for Language Models](model_compression/beyond_linear_probes_dynamic_safety_monitoring_for_language_models.md)**

:   提出截断多项式分类器（TPC），通过对 LLM 激活空间中的多项式逐阶训练和截断评估，实现动态安全监控——在简单输入上用低阶（≈线性探针）快速决策，在困难输入上增加高阶项提供更强防护，在 WildGuardMix 和 BeaverTails 两个数据集上匹敌或超越 MLP 基线且具备内置可解释性。

**[BiasScope: Towards Automated Detection of Bias in LLM-as-a-Judge Evaluation](model_compression/biasscope_towards_automated_detection_of_bias_in_llm-as-a-judge_evaluation.md)**

:   提出 BiasScope，一个完全由 LLM 驱动的迭代式框架，能自动、大规模地发现 LLM-as-a-Judge 中的潜在未知偏差，并基于此构建了更具挑战性的 JudgeBench-Pro 基准，在其上即使强大的 LLM 评估器错误率也超过 50%。

**[Boomerang Distillation Enables Zero-Shot Model Size Interpolation](model_compression/boomerang_distillation_enables_zero-shot_model_size_interpolation.md)**

:   发现并系统研究"回旋蒸馏"现象：从大模型（teacher）蒸馏出小模型（student）后，将教师的层块重新插回学生模型，无需任何额外训练即可构建任意中间尺寸的模型，其性能在 student 和 teacher 之间平滑插值，匹配甚至超越同等尺寸的独立蒸馏模型。

**[Boosting Entropy with Bell Box Quantization](model_compression/boosting_entropy_with_bell_box_quantization.md)**

:   提出 Bell Box Quantization (BBQ)，首个同时满足"信息论最优"(ITO) 和"计算高效"(compute-efficient) 的量化方法，核心洞察是学习的域无关性——量化器输出域不必与输入域相同，由此在输入域做 ITO 量化以最大化熵，在输出域映射到硬件可加速的数据类型，在 1-4 bit QAPT 场景下全面超越 QuEST 和 LSQ。

**[BOTS: A Unified Framework for Bayesian Online Task Selection in LLM Reinforcement Finetuning](model_compression/bots_a_unified_framework_for_bayesian_online_task_selection_in_llm_reinforcement.md)**

:   提出 BOTS 框架，将 LLM 强化微调中的在线任务选择建模为贝叶斯推断问题，通过融合显式证据（直接评估）和隐式证据（跨任务推断）来自适应估计任务难度，并利用 Thompson 采样平衡探索与利用，显著提升训练效率。

**[Compute-Optimal Quantization-Aware Training](model_compression/compute-optimal_quantization-aware_training.md)**

:   本文通过 757 组 QAT 实验（86M-2.2B 参数，1-6 bit）发现：QAT 的最优训练比例随总计算量增长而增大（与先前认为固定 10% 的结论相反），并提出 tokens-per-parameter-byte 统计量和新的 loss scaling law 来精确预测最优 QAT 分配策略和最终损失。

**[Cross-Domain Lossy Compression via Rate- and Classification-Constrained Optimal Transport](model_compression/cross_domain_lossy_compression_optimal_transport.md)**

:   将跨域有损压缩（编码器看退化源、解码器重建不同目标分布）形式化为带压缩率和分类损失双重约束的最优传输问题，推导出 Bernoulli/Gaussian 源的闭式 DRC（失真-率-分类）和 DRPC（失真-率-感知-分类）权衡曲线，在 KODAK 去噪上实现 PSNR 27.90 / SSIM 0.80 的竞争性能，审稿人给出 10/10 评分。

**[Energy-Regularized Sequential Model Editing on Hyperspheres](model_compression/energy-regularized_sequential_model_editing_on_hyperspheres.md)**

:   从超球面均匀性（Hyperspherical Energy）视角理解序列模型编辑中的性能退化，提出 SPHERE 方法：通过将编辑扰动投影到预训练权重主超球方向的正交补空间，实现稳定的大规模序列编辑，在 LLaMA3-8B 上平均超越最强基线 16.41%。

**[GuidedSampling: Steering LLMs Towards Diverse Candidate Solutions at Inference-Time](model_compression/guidedsampling_steering_llms_towards_diverse_candidate_solutions_at_inference-ti.md)**

:   提出 GuidedSampling 推理算法，将重复采样（RS）的隐式探索和生成过程显式解耦为两阶段：先迭代生成多样化的解题概念/定理，再基于各概念分别生成候选解。在 pass@50 上平均提升约 21.6%，微调后 pass@5 提升约 9.7%。

**[LD-MoLE: Learnable Dynamic Routing for Mixture of LoRA Experts](model_compression/ld-mole_learnable_dynamic_routing_for_mixture_of_lora_experts.md)**

:   提出 LD-MoLE，用 Sparsegen 可微路由（闭式概率单纯形投影 + token 依赖的 λ 预测）替代 TopK 的不可微路由，实现 LoRA 专家的自适应动态分配，在 Llama-3.2-3B/Qwen3-1.7B 上优于 MoLA(TopK) 和 ReMoE(ReLU)。

**[Scalable Multi-Task Low-Rank Model Adaptation](model_compression/scalable_multi-task_low-rank_model_adaptation.md)**

:   系统分析多任务 LoRA 在任务数量增大时崩溃的根因（均匀正则化破坏共享知识 + 组件级 LoRA 放大梯度冲突），提出 mtLoRA：谱感知正则化 + 块级适配 + 细粒度路由，在 15-25 个任务上平均超越 SOTA 2.3%，同时减少 47% 参数和 24% 训练时间。

**[Stress-Testing Alignment Audits With Prompt-Level Strategic Deception](model_compression/stress-testing_alignment_audits_with_prompt-level_strategic_deception.md)**

:   构建自动 prompt 级红队流水线，对"保守秘密"的模型有机体进行压力测试，发现能诱导黑盒和白盒对齐审计方法产生高置信错误猜测的欺骗策略，首次记录了基于激活的策略性欺骗现象。

---

## ⚖️ 对齐 / RLHF { #llm_alignment }

**[A2D: Any-Order, Any-Step Safety Alignment for Diffusion Language Models](llm_alignment/a2d_any-order_any-step_safety_alignment_for_diffusion_language_models.md)**

:   提出 A2D，一种针对扩散语言模型（dLLM）的 token 级安全对齐方法，通过训练模型在遇到有害内容的 mask 位置输出 [EOS] token 来实现任意解码顺序、任意解码步的安全防御，将 DIJA 模板攻击成功率从 80%+ 降到近零（1.3%/0.0%），并支持早期拒绝实现 19.3x 加速。

**[Align Once, Benefit Multilingually: Enforcing Multilingual Consistency for LLM Safety Alignment](llm_alignment/align_once_benefit_multilingually_enforcing_multilingual_consistency_for_llm_saf.md)**

:   提出 Multi-Lingual Consistency (MLC) 辅助损失，通过 SVD 操控多语言表示矩阵的奇异值使其趋向秩-1（即多语言表示共线），仅需多语言 prompt 翻译（无需目标语言的 response），即可将一种语言的安全对齐效果一致性地迁移到所有语言。

**[Alignment through Meta-Weighted Online Sampling: Bridging the Gap between Data Generation and Preference Optimization](llm_alignment/alignment_through_meta-weighted_online_sampling_bridging_the_gap_between_data_ge.md)**

:   提出MetaAPO框架，用一个轻量级meta-learner（两层MLP）动态估计offline/online数据的对齐差距，既指导"在哪些prompt上做在线采样"（解决分布不匹配），又在训练时自适应加权offline/online数据（优化学习效率），在AlpacaEval 2/Arena-Hard/MT-Bench上超越DPO/Online DPO等基线，同时减少42%在线标注成本。

**[AlphaSteer: Learning Refusal Steering with Principled Null-Space Constraint](llm_alignment/alphasteer_learning_refusal_steering_with_principled_null-space_constraint.md)**

:   提出 AlphaSteer，通过学习一个受零空间约束的变换矩阵来动态构造 steering 向量，对良性输入产生近零向量（保持效用），对恶意输入重建拒绝方向向量（增强安全），在理论上保证了安全与效用的解耦。

**[Beyond Pairwise: Empowering LLM Alignment With Ranked Choice Modeling](llm_alignment/beyond_pairwise_empowering_llm_alignment_with_ranked_choice_modeling.md)**

:   提出 RCPO 框架，将 LLM 对齐从成对偏好扩展到排名选择（ranked choice）建模，通过 MLE 统一了效用模型（MNL）和排名模型（Mallows-RMJ），在 single-best 和 top-k 反馈格式下都优于 DPO 及其变体。

**[Beyond RLHF and NLHF: Population-Proportional Alignment under an Axiomatic Framework](llm_alignment/beyond_rlhf_and_nlhf_population-proportional_alignment_under_an_axiomatic_framew.md)**

:   提出基于社会选择理论公理的偏好学习框架，从成对比较数据中推断评估者人群分布的可行集，构造满足人群比例对齐(PPA)和人群有界可操纵性(PBM)公理的策略。

**[CAGE: A Framework for Culturally Adaptive Red-Teaming Benchmark Generation](llm_alignment/cage_a_framework_for_culturally_adaptive_red-teaming_benchmark_generation.md)**

:   提出 CAGE 框架，通过 Semantic Mold（语义模具）将红队攻击 prompt 的对抗结构与文化内容解耦，能系统性地将英语红队基准适配到不同文化语境中，生成的文化扎根 prompt 比直接翻译的 ASR 显著更高。

**[Displacement-Resistant Extensions of DPO with Nonconvex $f$-Divergences](llm_alignment/displacement-resistant_extensions_of_dpo_with_nonconvex_f-divergences.md)**

:   发现 f-DPO 的可解性不需要 f 凸（仅需 $\lim_{t\to 0^+} f'(t) = -\infty$），进一步证明 $\arg\min f(t) \geq 1$ 是抵抗概率位移的必要条件，由此提出 SquaredPO（$f(t) = \frac{1}{2}(\log t)^2$，非凸），在保持性能的同时显著缓解 winner 概率下降问题。

**[Dual-IPO: Dual-Iterative Preference Optimization for Text-to-Video Generation](llm_alignment/dual-ipo_dual-iterative_preference_optimization_for_text-to-video_generation.md)**

:   提出 Dual-IPO 框架，通过在奖励模型和视频生成模型之间进行多轮双向迭代优化，无需大量人工标注即可持续提升文本到视频生成的质量和人类偏好对齐，甚至让 2B 模型超越 5B 模型。

**[From Utterance to Vividity: Training Expressive Subtitle Translation LLM via Adaptive Local Preference Optimization](llm_alignment/from_utterance_to_vividity_training_expressive_subtitle_translation_llm_via_adap.md)**

:   提出**ALPO**（自适应局部偏好优化），通过segment-wise采样和细粒度对齐损失训练表达力强的字幕翻译LLM，解决传统DPO在多段局部偏好对齐中梯度稀释的问题。

**[Group-Relative REINFORCE Is Secretly an Off-Policy Algorithm: Demystifying Some Myths About GRPO and Its Friends](llm_alignment/group-relative_reinforce_is_secretly_an_off-policy_algorithm_demystifying_some_m.md)**

:   通过 first-principles 推导揭示 group-relative REINFORCE（如 GRPO）天然具有 off-policy 解释，无需假设数据采样分布。发现 clipping 而非 importance sampling 是稳定性的关键，提出 REC 系列算法统一解释 GRPO、Kimi OPMD 和 Meta AsymRE。

**[Hierarchy-of-Groups Policy Optimization for Long-Horizon Agentic Tasks](llm_alignment/hierarchy-of-groups_policy_optimization_for_long-horizon_agentic_tasks.md)**

:   揭示了 stepwise group-based RL（如 GRPO/GiGPO）中的「历史上下文不一致」问题——同一 group 内的 step 可能具有不同历史上下文导致 advantage 估计偏差，提出 HGPO 通过层次化分组和自适应加权实现低偏差、平衡方差的 advantage 估计，在 ALFWorld 和 WebShop 上以极低额外开销（<0.001%）取得显著提升。

**[Learning More with Less: A Dynamic Dual-Level Down-Sampling Framework for Efficient Policy Optimization](llm_alignment/learning_more_with_less_a_dynamic_dual-level_down-sampling_framework_for_efficie.md)**

:   提出**D3S**（Dynamic Dual-Level Down-Sampling）框架，在sample层最大化advantage方差、在token层优先选取高熵+高advantage的token，配合动态调度策略，用不到20% token实现更快收敛和更优性能。

**[Mitigating Mismatch within Reference-based Preference Optimization](llm_alignment/mitigating_mismatch_within_reference-based_preference_optimization.md)**

:   揭示 DPO 的"悲观 reference 偏差"问题——当 reference 策略对 chosen response 的概率低于 rejected 时（~45% pairs），DPO 会"过早满足"停止学习；提出 HyPO（一行代码修改：将 $\Delta_{ref}$ 裁剪为 $\max(0, \Delta_{ref})$），在 AlpacaEval 上相对 DPO 提升 41.2%。

**[Reasoned Safety Alignment: Ensuring Jailbreak Defense via Answer-Then-Check](llm_alignment/reasoned_safety_alignment_ensuring_jailbreak_defense_via_answer-then-check.md)**

:   提出 ReSA（Reasoned Safety Alignment），通过"先回答后检查"范式——模型先总结意图答案，再进行策略驱动的安全分析，最后输出最终回复——在 13 种防御方法中安全性最优，同时维持 MMLU/MATH500/HumanEval 性能，仅 500 样本即达全数据集效果。

**[PURGE: Reinforcement Unlearning via Group Relative Policy Optimization](llm_alignment/reinforcement_unlearning_via_group_relative_policy_optimization.md)**

:   PURGE 将 LLM 遗忘（unlearning）重新定义为可验证的 RL 任务，使用 GRPO 框架 + 内在奖励信号（惩罚提及禁止概念）来实现安全一致的知识删除，token 消耗比 SOTA 低 46 倍，同时提升流畅度 +5.48% 和对抗鲁棒性 +12.02%。

**[SafeDPO: A Simple Approach to Direct Preference Optimization with Enhanced Safety](llm_alignment/safedpo_preference_optimization_safety.md)**

:   重新审视安全约束 RLHF 目标并证明其存在闭式最优策略，据此推导出等价的可处理目标 SafeDPO，仅需在标准 DPO 上加入安全感知数据变换和安全 margin 项（1 个额外超参数），无需奖励/代价模型，在 PKU-SafeRLHF-30K 上实现 96.87% 无害率且保持竞争力的有用性，训练速度比 SafeRLHF 快 25×。

**[Uni-DPO: A Unified Paradigm for Dynamic Preference Optimization of LLMs](llm_alignment/uni-dpo_a_unified_paradigm_for_dynamic_preference_optimization_of_llms.md)**

:   提出 Uni-DPO，通过质量感知加权（高间距优先）和性能感知加权（focal loss 聚焦欠拟合样本）双视角动态调整 DPO 偏好对权重，Gemma-2-9B 在 Arena-Hard 上超 Claude 3 Opus 6.7 分。

**[Why DPO is a Misspecified Estimator and How to Fix It](llm_alignment/why_dpo_is_misspecified_estimator.md)**

:   从信息几何角度证明 DPO 在参数化（非 tabular）策略类下本质上是一个误指定的统计估计问题——DPO 将真实奖励函数 KL 投影到隐式奖励流形上，当奖励不可实现时会导致偏好反转和奖励下降——并提出 AuxDPO 通过引入零空间辅助变量来修复此问题。

---

## 👁️ 多模态 VLM { #multimodal_vlm }

**[A-TPT: Angular Diversity Calibration Properties for Test-Time Prompt Tuning of Vision-Language Models](multimodal_vlm/a-tpt_angular_diversity_calibration_properties_for_test-time_prompt_tuning_of_vi.md)**

:   提出 A-TPT 框架，通过最大化归一化文本特征在单位超球面上的最小成对角距离来促进角度多样性，解决测试时提示调优 (TPT) 中 VLM 预测过度自信导致的校准不良问题，在自然分布偏移和医学数据集上均优于现有 TPT 校准方法。

**[Adaptive Debiasing Tsallis Entropy for Test-Time Adaptation](multimodal_vlm/adaptive_debiasing_tsallis_entropy_for_test-time_adaptation.md)**

:   提出将 Tsallis 熵（SE 的广义形式）引入 VLM 的 Test-Time Adaptation，并进一步发展为自适应去偏 Tsallis 熵（ADTE），为每个类别定制去偏参数 $q^l$，在不引入分布特定超参数的情况下比 Shannon 熵选择更可靠的高置信视图，在 ImageNet 及其 5 个变体和 10 个跨域 benchmark 上均超越 SOTA。

**[AgilePruner: An Empirical Study of Attention and Diversity for Adaptive Visual Token Pruning in LVLMs](multimodal_vlm/agilepruner_an_empirical_study_of_attention_and_diversity_for_adaptive_visual_to.md)**

:   通过 erank（有效秩）和注意力熵的系统性实证分析，揭示了视觉 token 剪枝中注意力方法和多样性方法的互补特性——注意力方法抑制幻觉但覆盖有限，多样性方法覆盖全面但易引入幻觉——并据此提出基于图像复杂度自适应切换剪枝策略的 AgilePruner，在 9 个 benchmark 上表现稳健。

**[AQuA: Toward Strategic Response Generation for Ambiguous Visual Questions](multimodal_vlm/aqua_toward_strategic_response_generation_for_ambiguous_visual_questions.md)**

:   提出 AQuA，首个按模糊度细粒度分级（4 级）的视觉问答数据集（7.2K 样本），为每级定义最优回应策略（直接回答/推断/列举/请求澄清），发现 GPT-5 和 Gemini 在模糊 VQA 上都过度自信地直接回答，通过 SFT+GRPO 训练的 3B 模型反而能超越闭源大模型的策略适应能力。

**[BioCAP: Exploiting Synthetic Captions Beyond Labels in Biological Foundation Models](multimodal_vlm/biocap_exploiting_synthetic_captions_beyond_labels_in_biological_foundation_mode.md)**

:   提出 BioCAP，通过用 MLLM 生成 wiki 知识引导的合成描述性 caption（而非仅用物种标签）来训练生物学多模态基础模型，在 10 个物种分类 benchmark 上比 BioCLIP 平均提升 8.8%，在文本-图像检索任务上提升 21.3%。

**[Bongard-RWR+: Real-World Representations of Fine-Grained Concepts in Bongard Problems](multimodal_vlm/bongard-rwr_real-world_representations_of_fine-grained_concepts_in_bongard_probl.md)**

:   构建 Bongard-RWR+，一个包含 5400 个 Bongard 问题的 benchmark，使用 VLM 流水线（Pixtral-12B + Flux.1-dev）自动生成真实感图像来表示抽象概念，系统评估揭示 SOTA VLM 在辨别细粒度视觉概念（如轮廓、旋转、角度）时表现挣扎，准确率低至 19%。

**[Bootstrapping MLLM for Weakly-Supervised Class-Agnostic Object Counting (WS-COC)](multimodal_vlm/bootstrapping_mllm_for_weakly-supervised_class-agnostic_object_counting.md)**

:   提出 WS-COC，首个基于 MLLM 的弱监督类无关目标计数框架，通过分而治之的对话微调（逐步缩小计数范围）、比较排序优化（学习图像间相对计数关系）和全局-局部计数增强三个策略，仅用图像级计数标注即可匹敌甚至超越全监督方法。

**[Breaking the Limits of Open-Weight CLIP: An Optimization Framework for Self-supervised Fine-tuning of CLIP](multimodal_vlm/breaking_the_limits_of_open-weight_clip_an_optimization_framework_for_self-super.md)**

:   本文提出 TuneCLIP，一个自监督微调（SSFT）框架，通过两阶段设计——先恢复优化器统计量（OSR）消除冷启动偏差，再用带margin的铰链全局对比损失（HGCL）缓解假负样本过度惩罚——在不使用任何标签的条件下持续提升已有开源 CLIP 模型的通用性能，在 ImageNet 及变体上提升最高 +2.5%，在 DataComp 基准上提升 +1.2%。

**[Cross-Modal Redundancy and the Geometry of Vision-Language Embeddings](multimodal_vlm/cross-modal_redundancy_and_the_geometry_of_vision-language_embeddings.md)**

:   提出 Iso-Energy 假设（真正跨模态共享的概念在不同模态中应具有相同的平均激活能量），并设计 Aligned SAE 作为分析工具，揭示 VLM 嵌入空间中双模态原子承载跨模态对齐信号、单模态原子完全解释模态间隙的几何结构。

**[Grounding-IQA: Grounding Multimodal Language Models for Image Quality Assessment](multimodal_vlm/grounding-iqa_grounding_multimodal_language_model_for_image_quality_assessment.md)**

:   将空间定位（referring + grounding）与图像质量评估结合，构建 GIQA-160K 数据集训练多模态 LLM 生成带有边界框的质量描述和空间 VQA，在细粒度质量感知上显著优于通用 MLLM。

**[GTR-Bench: Evaluating Geo-Temporal Reasoning in Vision-Language Models](multimodal_vlm/gtr-bench_evaluating_geo-temporal_reasoning_in_vision-language_mod.md)**

:   提出 GTR-Bench，一个面向大规模摄像头网络中移动目标地理时空推理的新基准，评估发现最强模型 Gemini-2.5-Pro（34.9%）远落后于人类水平（78.61%），揭示了当前 VLM 在时空上下文利用失衡、时序预测能力弱、地图-视频对齐能力不足三大缺陷。

**[Leveraging Data to Say No: Memory Augmented Plug-and-Play Selective Prediction](multimodal_vlm/leveraging_data_to_say_no_memory_augmented_plug-and-play_selective_prediction.md)**

:   提出 MA-PaPSP 框架，通过外部检索数据集构建代理嵌入（k-NN 加权平均降低表示方差）+ 对比归一化评分（改善校准），无训练地为任意 VLM 提供可靠的"拒绝回答"能力，在图像描述、图文匹配、分类的选择性预测上全面优于 PaPSP 和 LLM-as-judge 基线。

**[Look Carefully: Adaptive Visual Reinforcements in Multimodal Large Language Models for Hallucination Mitigation](multimodal_vlm/look_carefully_adaptive_visual_reinforcements_in_multimodal_large_language_model.md)**

:   提出 AIR（Adaptive vIsual Reinforcement）框架，通过原型距离的 token 精简 + 最优传输引导的 patch 选择性增强，在推理时无训练地减少 MLLM 幻觉（LLaVA-1.5-7B CHAIR_S: 22→18.4，POPE 准确率 +5.3%），同时保持多模态通用能力。

**[Multimodal Classification via Total Correlation Maximization](multimodal_vlm/multimodal_classification_via_total_correlation_maximization.md)**

:   从信息论角度分析多模态分类中的模态竞争问题，提出 TCMax 损失函数通过最大化多模态特征与标签之间的总相关性（Total Correlation），同时兼顾联合学习、单模态学习和跨模态对齐三重目标，在多个音视频/图文分类基准上超越 SOTA。

**[PoSh: Using Scene Graphs To Guide LLMs-as-a-Judge For Detailed Image Descriptions](multimodal_vlm/posh_using_scene_graphs_to_guide_llms-as-a-judge_for_detailed_image_descriptions.md)**

:   提出 PoSh，利用场景图（依存分析+共指消解）作为结构化评分标准指导 LLM-as-Judge 评估详细图像描述，Spearman ρ 超 SPICE/CAPTURE +0.05，作为奖励函数优于 SFT。

**[RAVENEA: A Benchmark for Multimodal Retrieval-Augmented Visual Culture Understanding](multimodal_vlm/ravenea_a_benchmark_for_multimodal_retrieval-augmented_visual_culture_understand.md)**

:   构建首个评估多模态检索增强文化理解的基准 Ravenea，包含 1868 个实例和 11396 篇人工排序的 Wikipedia 文档，覆盖 8 个国家 11 个类别，评估 7 个多模态检索器和 17 个 VLM，发现文化感知的 RAG 可在 cVQA 上平均提升 6%、cIC 上提升 11%。

**[Shuffle-R1: Efficient RL Framework for Multimodal Large Language Models via Data-centric Dynamic Shuffle](multimodal_vlm/shuffle-r1_efficient_rl_framework_for_multimodal_large_language_models_via_data-.md)**

:   提出 Shuffle-R1 框架，通过 Pairwise Trajectory Sampling（选取高对比度轨迹对）和 Advantage-based Batch Shuffle（按优势值重分配训练批次），解决 RL 训练中的 Advantage Collapsing 和 Rollout Silencing 两大效率瓶颈，在 Geo3K 上比 baseline 提升 22%，MathVerse 上超越 GPT-4o。

**[ThinkOmni: Lifting Textual Reasoning to Omni-modal Scenarios via Guidance Decoding](multimodal_vlm/thinkomni_lifting_textual_reasoning_to_omni-modal_scenarios_via_guidance_decodin.md)**

:   提出 ThinkOmni 无训练框架，利用纯文本大推理模型(LRM)在解码时引导全模态 LLM(OLLM)，通过 Stepwise Contrastive Scaling 自适应平衡感知与推理信号，MathVista 达 70.2%、MMAU 达 75.5%，匹配或超越 RFT 方法。

**[Visual Symbolic Mechanisms: Emergent Symbol Processing in Vision Language Models](multimodal_vlm/visual_symbolic_mechanisms_vlm.md)**

:   发现 VLM 内部涌现了一套三阶段符号处理机制（ID retrieval → ID selection → feature retrieval），利用内容无关的空间位置索引（position IDs）来解决视觉绑定问题，并证明绑定错误可直接追溯到这些机制的失败。

---

## ⚡ LLM 效率 { #llm_efficiency }

**[Bayesian Attention Mechanism: A Probabilistic Framework for Positional Encoding and Context Length Extrapolation](llm_efficiency/bayesian_attention_mechanism_a_probabilistic_framework_for_positional_encoding_a.md)**

:   将位置编码重新表述为贝叶斯注意力机制中的先验分布，统一了 NoPE（均匀先验）和 ALiBi（拉普拉斯先验），并提出广义高斯先验（GGD-BAM），仅增加 384 个参数即可在 500 倍训练长度上实现完美的 passkey 检索。

**[Beyond RAG vs. Long-Context: Learning Distraction-Aware Retrieval for Efficient Knowledge Grounding](llm_efficiency/beyond_rag_vs_long-context_learning_distraction-aware_retrieval_for_efficient_kn.md)**

:   提出 LDAR（Learning Distraction-Aware Retrieval），一个轻量级自适应检索器，通过学习基于查询-段落相似度分布选择段落的连续区间（band），在平衡信息覆盖与干扰段落影响的同时，以约一半的 token 用量超越长上下文方法的性能。

**[Did You Check the Right Pocket? Cost-Sensitive Store Routing for Memory-Augmented Agents](llm_efficiency/did_you_check_the_right_pocket_cost-sensitive_store_routing_for_memory-augmented.md)**

:   将记忆增强 Agent 的多存储检索形式化为代价敏感的存储路由问题（store routing），证明选择性检索相比全量检索可在减少 62% context token 的同时提升 QA 准确率（86% vs 81%），并提出基于语义信号的启发式路由基线。

**[EvoEngineer: Mastering Automated CUDA Kernel Code Evolution with Large Language Models](llm_efficiency/evoengineer_mastering_automated_cuda_kernel_code_evolution_with_large_language_m.md)**

:   提出 EvoEngineer，首个系统化的 LLM-based 代码演化框架，将代码演化分解为 traverse technique（含两层设计：solution guiding + prompt engineering）和 population management 两个正交组件，在 91 个真实 CUDA kernel 上实现最高 2.72× 中位加速比和 69.8% 代码有效率，在性能和正确性两个维度上超越现有方法。

**[Expert Divergence Learning for MoE-based Language Models](llm_efficiency/expert_divergence_learning_for_moe-based_language_models.md)**

:   解决 MoE 训练中的专家同质化问题，通过最大化不同数据域之间路由分布的 Jensen-Shannon 散度，鼓励不同域激活不同专家子集，在 15B-A1.5B 模型上提升专家特化程度和语言建模性能。

**[Multilingual Routing in Mixture-of-Experts](llm_efficiency/multilingual_routing_in_mixture-of-experts.md)**

:   系统分析MoE LLM中多语言路由模式，发现中间层存在跨语言共享专家、语言性能与英语路由对齐度强相关，并提出推理时路由干预方法，通过激活英语任务专家在中间层一致性地提升多语言性能1-2%。

**[One-Prompt Strikes Back: Sparse Mixture of Experts for Prompt-based Continual Learning](llm_efficiency/one-prompt_strikes_back_sparse_mixture_of_experts_for_prompt-based_continual_lea.md)**

:   提出 SMoPE 框架，将单个共享 prompt 组织为稀疏 MoE 结构中的多个 prompt expert，通过 prompt-attention score aggregation 实现动态稀疏激活，在保持高参数效率的同时显著缓解知识干扰，在多个持续学习 benchmark 上达到 SOTA。

**[Q-RAG: Long Context Multi-Step Retrieval via Value-Based Embedder Training](llm_efficiency/q_rag_long_context_multi_step_retrieval.md)**

:   将多步检索建模为 MDP，用基于值的 RL（soft Q-learning）微调 **embedder 而非 LLM**，Q 函数设计为状态嵌入和动作嵌入的内积（理论证明为万能近似器），结合 RoPE 相对位置编码实现时序推理，在单卡 A100 上训练 12 小时，4K 训练泛化到 1M+ token 上下文，RULER 基准达到近乎完美的 NIAH 性能。

**[RACE Attention: A Strictly Linear-Time Attention for Long-Sequence Training](llm_efficiency/race_attention_a_strictly_linear-time_attention_for_long-sequence_training.md)**

:   提出 RACE Attention——用幂次角核替代 softmax 并通过可微 LSH 草图近似注意力输出，实现严格线性时间复杂度，支持单 GPU 处理 1200 万 token、单 CPU 处理 7500 万 token，在多种任务上匹配或超越 softmax 精度。

**[Randomization Boosts KV Caching, Learning Balances Query Load: A Joint Perspective](llm_efficiency/randomization_boosts_kv_caching_learning_balances_query_load_a_joint_perspective.md)**

:   提出首个KV缓存感知负载均衡统一数学模型，设计随机化叶节点淘汰算法RLT(O(log n)竞争比)和基于学习的贪心路由LBGR，在多LLM服务场景下将延迟降低最高11.96×、TTFT降低14.06×。

**[Rethinking Uncertainty Estimation in LLMs: A Principled Single-Sequence Measure](llm_efficiency/rethinking_uncertainty_estimation_in_llms_a_principled_single-sequence_measure.md)**

:   从 proper scoring rules 框架出发，证明最高概率输出序列的负对数似然（MSP）是理论上合理的不确定性度量，并提出 G-NLL——仅用一次贪心解码就能逼近该度量，在多个场景下匹配或超越需要多次采样的 SOTA 方法。

**[Semantic Parallelism: Redefining Efficient MoE Inference via Model-Data Co-Scheduling](llm_efficiency/semantic_parallelism_redefining_efficient_moe_inference_via_model-data_co-schedu.md)**

:   提出语义并行(Semantic Parallelism)范式，通过预测token-expert路由路径并协同调度模型放置与数据分发，大幅削减MoE推理中专家并行的all-to-all通信开销，在Attention-DP场景下吞吐提升最高2.78×，Attention-TP场景下延迟降低最高24.9%。

**[SwingArena: Adversarial Programming Arena for Long-context GitHub Issue Solving](llm_efficiency/swingarena_competitive_programming_arena_for_long-context_github_issue_solving.md)**

:   提出SwingArena对抗性评测框架，让LLM交替扮演提交者(生成补丁)和审查者(编写测试)，通过真实CI流水线验证，覆盖C++/Python/Rust/Go四种语言的400个GitHub issue，揭示不同模型在补丁生成vs验证方面的行为差异。

**[Universe Routing: Why Self-Evolving Agents Need Epistemic Control](llm_efficiency/universe_routing_why_self-evolving_agents_need_epistemic_control.md)**

:   形式化"宇宙路由"问题——将问题分类到互斥的信念空间（频率主义/贝叶斯/经典物理/量子等）后再调用专用求解器，证明硬路由优于软路由（7× 快且等精度），且模块化架构天然适合持续学习。

**[When Does Divide and Conquer Work for Long Context LLM? A Noise Decomposition Framework](llm_efficiency/when_does_divide_and_conquer_work_for_long_context_llm_a_noise_decomposition_fra.md)**

:   提出理论框架将长上下文任务失败分解为三类噪声（任务噪声/模型噪声/聚合器噪声），证明当模型噪声超线性增长时弱模型+分块处理可超越强模型单次处理，并给出快速估计最优 chunk size 的方法（3-5 个样本即可）。

**[xLSTM Scaling Laws: Competitive Performance with Linear Time-Complexity](llm_efficiency/xlstm_scaling_laws_competitive_performance_with_linear_time-complexity.md)**

:   系统对比 xLSTM 与 Transformer 的 scaling law，证明 xLSTM 在训练损失-算力 Pareto 前沿、过训练 regime 和推理速度上全面优于同规模 Transformer，且优势随上下文长度增大而增长。

---

## 🎯 目标检测 { #object_detection }

**[A Problem-Oriented Perspective and Anchor Verification for Code Optimization](object_detection/a_problem-oriented_perspective_and_anchor_verification_for_code_optimization.md)**

:   提出以问题为导向（而非用户为导向）的优化对构建方法来整合多程序员的策略多样性，并设计锚点验证框架利用"慢但正确的代码"生成测试用例来缓解"优化税"（正确性损失），将优化比从 31.24% 提升到 71.06%，加速比从 2.95x 提升到 6.08x。

**[AdaRank: Adaptive Rank Pruning for Enhanced Model Merging](object_detection/adarank_adaptive_rank_pruning_for_enhanced_model_merging.md)**

:   提出 AdaRank，通过可学习二值掩码自适应选择 task vector 的奇异分量（而非启发式 top-k），结合测试时熵最小化优化，大幅缓解多任务模型合并中的任务间干扰。

**[Beyond Linearity in Attention Projections: The Case for Nonlinear Queries](object_detection/beyond_linearity_in_attention_projections_the_case_for_nonlinear_queries.md)**

:   基于 WQ 代数冗余性的理论发现，将线性 Query 投影替换为非线性残差形式 Q(X)=(X+f_θ(X))/2，在相同参数量下超越增加 12.5% 参数的基线。

**[Breaking Scale Anchoring: Frequency Representation Learning for Accurate High-Resolution Inference from Low-Resolution Training](object_detection/breaking_scale_anchoring_frequency_representation_learning_for_accurate_high-res.md)**

:   定义了"Scale Anchoring"新问题（低分辨率训练导致高分辨率推理误差锚定），提出架构无关的频率表征学习（FRL），通过归一化频率编码使误差随分辨率提升而下降。

**[CGSA: Class-Guided Slot-Aware Adaptation for Source-Free Object Detection](object_detection/cgsa_class-guided_slot-aware_adaptation_for_source-free_object_detection.md)**

:   首次将 Object-Centric Learning（Slot Attention）引入无源域自适应目标检测（SF-DAOD），通过分层 Slot 感知模块提取结构先验，并用类引导对比学习驱动域不变表征。

**[ConFu: Contemplate the Future for Better Speculative Sampling](object_detection/confu_contemplate_the_future_for_better_speculative_sampling.md)**

:   提出 ConFu 框架，通过 contemplate tokens 让 draft model 预见 target model 的未来生成方向，结合 MoE 动态机制和锚点采样训练，在 EAGLE-3 基础上提升 8-11% 的接受率和速度。

**[Context Tokens are Anchors: Understanding the Repetition Curse in dMLLMs from an Information Flow Perspective](object_detection/context_tokens_are_anchors_understanding_the_repetition_curse_in_dmllms_from_an_.md)**

:   通过信息流分析揭示扩散多模态大语言模型(dMLLMs)在使用缓存加速时产生"重复诅咒"的内在机制，并提出 CoTA 方法有效缓解重复问题。

**[ForestPersons: A Large-Scale Dataset for Under-Canopy Missing Person Detection](object_detection/forestpersons_a_large-scale_dataset_for_under-canopy_missing_person_detection.md)**

:   ForestPersons 是首个专门针对森林树冠下人员检测的大规模数据集（96,482 张图像 + 204,078 标注），覆盖地面/低空视角、多季节多天气多光照条件，每个实例包含边界框+姿态+可见性标注，填补了 SAR 场景中下冠层检测的数据空白。

**[From Narrow to Panoramic Vision: Attention-Guided Cold-Start Reshapes Multimodal Reasoning](object_detection/from_narrow_to_panoramic_vision_attention-guided_cold-start_reshapes_multimodal_.md)**

:   发现多模态 LLM 的推理性能与视觉注意力分数（VAS）高度相关（r=0.96），提出 AVAR 框架通过视觉锚定数据合成、注意力引导训练目标和视觉锚定奖励塑造三个阶段提升 VAS，在 77 个基准上平均提升 7%。

**[FSOD-VFM: Few-Shot Object Detection with Vision Foundation Models and Graph Diffusion](object_detection/fsod-vfm_few-shot_object_detection_with_vision_foundation_models_and_graph_diffu.md)**

:   提出一个无需训练的少样本目标检测框架，组合 UPN、SAM2 和 DINOv2 三个基础模型生成提案和匹配特征，并通过图扩散算法精化置信度分数和抑制碎片化提案，在 Pascal-5i 和 COCO-20i 上大幅超越 SOTA。

**[InfoDet: A Dataset for Infographic Element Detection](object_detection/infodet_a_dataset_for_infographic_element_detection.md)**

:   构建了一个大规模信息图元素检测数据集（101,264 张信息图、1420 万标注），涵盖图表和人类可识别对象两大类，并提出 Grounded CoT 方法利用检测结果提升 VLM 的图表理解能力。

**[Procedural Mistake Detection via Action Effect Modeling](object_detection/procedural_mistake_detection_via_action_effect_modeling.md)**

:   提出双分支多模态监督的动作效果建模框架，结合视觉分支（目标状态和空间关系特征）和文本分支（GPT-4o 生成的场景图），通过可学习的效果 token 蒸馏外部监督信号，在第一人称程序视频中实现 SOTA 错误检测。

**[SAGE: Spatial-visual Adaptive Graph Exploration for Efficient Visual Place Recognition](object_detection/sage_spatial-visual_adaptive_graph_exploration_for_efficient_visual_place_recogn.md)**

:   提出 SAGE，通过在线地理-视觉图和加权贪心团扩展实现动态难样本挖掘，仅用冻结 DINOv2+LoRA 在 SPED 和 Pitts30k 上达到 R@1=100%（超 CosPlace/MixVPR 20+%），在 8 个基准上全面 SOTA。

**[Thinking in Latents: Adaptive Anchor Refinement for Implicit Reasoning in LLMs](object_detection/thinking_in_latents_adaptive_anchor_refinement_for_implicit_reasoning_in_llms.md)**

:   提出 AdaAnchor，通过可学习潜在锚向量的迭代精化实现隐式推理，配合基于余弦距离的自适应停止机制，相比 CoT 减少 92-93% 生成 token（平均 2.17 vs 28.27），同时精度仅降低约 4%。

**[Toward Faithful Retrieval-Augmented Generation with Sparse Autoencoders](object_detection/toward_faithful_retrieval-augmented_generation_with_sparse_autoencoders.md)**

:   提出 RAGLens，利用稀疏自编码器（SAE）编码 LLM 隐状态，通过互信息特征选择和可加性模型（GAM）实现可解释的 RAG 幻觉检测，在 RAGTruth 上 AUC>0.80，零额外 LLM 推理开销。

**[Traceable Evidence Enhanced Visual Grounded Reasoning: Evaluation and Methodology](object_detection/traceable_evidence_enhanced_visual_grounded_reasoning_evaluation_and_methodology.md)**

:   提出 TreeBench 基准和 TreeVGR 训练范式，通过冷启动 SFT + 双 IoU 奖励 RL 训练，使模型在推理链中显式生成边界框实现可追溯的视觉推理，在 V*Bench/MME-RealWorld/TreeBench 上分别提升 +16.8/+12.6/+13.4。

---

## 🤖 机器人/具身智能 { #robotics }

**[All-day Multi-scenes Lifelong Vision-and-Language Navigation with Tucker Adaptation](robotics/all-day_multi-scenes_lifelong_vision-and-language_navigation_with_tucker_adaptat.md)**

:   提出Tucker Adaptation (TuKA)，将多场景多环境的多层级导航知识表示为高阶张量，用Tucker分解解耦为共享子空间（核心张量+编解码器）和场景/环境专家向量，配合解耦知识增量学习策略实现全天候多场景终身VLN，在24个导航场景上的SR和遗忘率均优于LoRA变体。

**[Attribution-Guided Decoding](robotics/attribution-guided_decoding.md)**

:   提出 Attribution-Guided Decoding (AGD)，在解码时利用归因方法（LRP）对候选 token 计算其对"感兴趣区域"(ROI) 的依赖分数，选择归因最高的 token，从而在不修改模型内部激活的前提下提升指令遵循和事实准确性。

**[Building Spatial World Models from Sparse Transitional Episodic Memories](robotics/building_spatial_world_models_from_sparse_transitional_episodic_memories.md)**

:   提出 Episodic Spatial World Model (ESWM)，从稀疏、不连续的情景记忆（one-step transitions）中构建空间世界模型，其潜空间自发涌现出与环境拓扑对齐的认知地图，并支持零样本探索和导航。

**[Capability-Based Scaling Trends for LLM-Based Red-Teaming](robotics/capability-based_scaling_trends_for_llm-based_red-teaming.md)**

:   在 600+ 对攻击者-目标 LLM 组合上系统评估了 4 种越狱方法，发现攻击成功率（ASR）与攻击者-目标的能力差距遵循 sigmoid 缩放定律（R^2=0.83），能力差距可用 MMLU-Pro 的 logit 变换量化。

**[Domain Expansion: A Latent Space Construction Framework for Multi-Task Learning](robotics/domain_expansion_a_latent_space_construction_framework_for_multi-task_learning.md)**

:   提出 Domain Expansion 框架，通过正交池化(Orthogonal Pooling)将潜在空间重构为互相正交的子空间，从结构上防止多目标训练中的梯度冲突与表征崩塌，实现可解释、可组合的概念代数。

**[ExoPredicator: Learning Abstract Models of Dynamic Worlds for Robot Planning](robotics/exopredicator_learning_abstract_models_of_dynamic_worlds_for_robot_planning.md)**

:   提出 ExoPredicator 框架，联合学习符号化状态抽象和因果过程（含内生动作与外生机制），通过变分贝叶斯推断 + LLM 提议从少量轨迹中学习带随机延迟的因果世界模型，在 5 个桌面机器人环境中实现快速泛化规划。

**[Experience-based Knowledge Correction for Robust Planning in Minecraft](robotics/experience-based_knowledge_correction_for_robust_planning_in_minecraft.md)**

:   证明 LLM 无法通过 prompting 自我纠正其错误的规划先验知识（物品依赖关系），提出 XENON——通过算法化的知识管理（自适应依赖图 ADG + 失败感知动作记忆 FAM）从二值反馈中学习，使 7B LLM 在 Minecraft 长期规划中超越使用 GPT-4V + oracle 知识的 SOTA。

**[Ignore All Previous Instructions: Jailbreaking as a de-escalatory peace building practise to resist LLM social media bots](robotics/ignore_all_previous_instructions_jailbreaking_as_a_de-escalatory_peace_building_.md)**

:   提出将"越狱"（jailbreaking）LLM 驱动的社交媒体机器人视为一种用户主导的、非暴力的去冲突化（de-escalation）和和平建设实践，通过暴露自动化账号的虚假性来抵抗误导信息传播。

**[JULI: Jailbreak Large Language Models by Self-Introspection](robotics/juli_jailbreak_large_language_models_by_self-introspection.md)**

:   揭示对齐 LLM 的 top-k token log probability 中仍包含有害信息，提出 JULI——仅用不到目标模型 1% 参数的 BiasNet 插件操纵 logit bias，在仅访问 top-5 token 概率的 API 场景下成功越狱 Gemini-2.5-Pro（harmfulness 4.19/5），比 SOTA 快 140 倍。

**[PERSONA: Dynamic and Compositional Inference-Time Personality Control via Activation Vector Algebra](robotics/persona_dynamic_and_compositional_inference-time_personality_control_via_activat.md)**

:   提出 PERSONA 框架，通过在激活空间中提取近似正交的人格向量并进行向量代数运算（缩放、加法、减法），实现免训练的动态组合式人格控制，在 PersonalityBench 上达到 9.60 分，几乎匹配 SFT 上界 9.61。

**[RF-MatID: Dataset and Benchmark for Radio Frequency Material Identification](robotics/rf-matid_dataset_and_benchmark_for_radio_frequency_material_identification.md)**

:   构建了首个开源的大规模、宽频段（4-43.5 GHz）、几何扰动多样的 RF 材料识别数据集 RF-MatID，包含 16 种细粒度材料类别（5 大类）/142K 样本，并建立了覆盖 9 个深度学习模型、5 种频率协议、7 种数据划分的系统基准。

**[RoboPARA: Dual-Arm Robot Planning with Parallel Allocation and Recomposition Across Tasks](robotics/robopara_dual-arm_robot_planning_with_parallel_allocation_and_recomposition_acro.md)**

:   提出 RoboPARA，一个 LLM 驱动的双臂机器人并行任务规划框架，通过依赖图生成与图重遍历调度两阶段方法，最大化双臂协同并行性，执行时间减少 30%-50%。

**[Sparse Imagination for Efficient Visual World Model Planning](robotics/sparse_imagination_for_efficient_visual_world_model_planning.md)**

:   提出 Sparse Imagination，在基于 ViT patch token 的世界模型规划中随机丢弃 token 以大幅加速推理（50% 丢弃率减少约 50% 时间），同时通过随机分组注意力训练保持任务性能不变。

**[THOR: Tool-Integrated Hierarchical Optimization via RL for Mathematical Reasoning](robotics/thor_tool-integrated_hierarchical_optimization_via_rl_for_mathematical_reasoning.md)**

:   提出 THOR，通过 TIRGen 数据构建管线 + 层次化强化学习（episode 级+step 级优化）+ 自修正推理机制，系统性解决 LLM 工具集成数学推理中的数据构建、细粒度优化和推理增强三大挑战。

**[Tracing and Reversing Edits in LLMs](robotics/tracing_and_reversing_edits_in_llms.md)**

:   针对知识编辑（Knowledge Editing）的双重使用风险，提出 EditScope 方法从编辑后的权重中推断被编辑的目标实体（准确率高达 99%），以及基于 SVD bottom-rank 近似的无训练编辑逆转方法（逆转率高达 94%），仅依赖编辑后的权重、不需要编辑 prompt 或原始权重信息。

**[Visual Planning: Let's Think Only with Images](robotics/visual_planning_lets_think_only_with_images.md)**

:   提出 Visual Planning，MLLM 通过生成图像序列（而非文本推理链）进行规划，VPRL 两阶段 RL（监督微调+进度奖励策略优化）在 FrozenLake 上比文本 SFT 高 27%，且泛化到分布外场景。

---

## 🚗 自动驾驶 { #autonomous_driving }

**[SMART-R1: Advancing Multi-agent Traffic Simulation via R1-Style Reinforcement Fine-Tuning](autonomous_driving/advancing_multi-agent_traffic_simulation_via_r1-style_reinforcement_fine-tuning.md)**

:   SMART-R1 首次将 R1 风格的强化微调（RFT）引入多智能体交通仿真，提出 Metric-oriented Policy Optimization (MPO) 算法和"SFT-RFT-SFT"迭代训练策略，在 WOSAC 2025 排行榜上以 0.7858 的 Realism Meta 分数取得第一名。

**[Astra: General Interactive World Model with Autoregressive Denoising](autonomous_driving/astra_general_interactive_world_model_with_autoregressive_denoising.md)**

:   提出 Astra，一个通用交互式世界模型，通过自回归去噪框架在预训练视频扩散模型上实现动作条件化的长程视频预测，引入 ACT-Adapter（动作注入）、噪声增强历史记忆（缓解视觉惯性）和 Mixture of Action Experts（统一多异构动作模态），在自动驾驶、机器人操控和场景探索等多场景上实现 SOTA 的保真度和动作跟随能力。

**[BridgeDrive: Diffusion Bridge Policy for Closed-Loop Trajectory Planning in Autonomous Driving](autonomous_driving/bridgedrive_diffusion_bridge_policy_for_closed-loop_trajectory_planning_in_auton.md)**

:   BridgeDrive 提出用扩散桥（diffusion bridge）替代截断扩散来实现锚点引导的自动驾驶轨迹规划，保证前向/反向过程的理论对称性，在 Bench2Drive 闭环评估中成功率达到 74.99%（PDM-Lite）和 89.25%（LEAD），分别超越前 SOTA 7.72% 和 2.45%。

**[DrivingGen: A Comprehensive Benchmark for Generative Video World Models in Autonomous Driving](autonomous_driving/drivinggen_a_comprehensive_benchmark_for_generative_video_world_models_in_autono.md)**

:   DrivingGen 提出首个面向自动驾驶视频世界模型的综合性基准，包含跨天气/地域/时间/复杂场景的多样化评估数据集和四维度评估指标体系（分布、质量、时序一致性、轨迹对齐），对 14 个 SOTA 模型的评测揭示了通用模型与驾驶专用模型之间的核心权衡。

**[EgoDex: Learning Dexterous Manipulation from Large-Scale Egocentric Video](autonomous_driving/egodex_learning_dexterous_manipulation_from_large-scale_egocentric_video.md)**

:   Apple 使用 Vision Pro 采集了 829 小时的第一人称视频 + 3D 手部关节追踪数据（EgoDex），覆盖 194 种桌面操作任务，并在此数据集上系统评估了模仿学习策略（BC/DDPM/FM + Transformer），为灵巧操作的扩展训练提供了迄今最大规模的数据基础。

**[MARC: Memory-Augmented RL Token Compression for Efficient Video Understanding](autonomous_driving/marc_memory-augmented_rl_token_compression_for_efficient_video_un.md)**

:   提出 MARC 框架，通过"先检索再压缩"策略——用 Visual Memory Retriever (VMR) 选出与查询最相关的视频片段，再用 Compression GRPO (C-GRPO) 将 64 帧教师模型的推理能力蒸馏到仅用 1 帧 token 的学生模型——实现视觉 token 95% 压缩，GPU 显存降低 72%，推理延迟降低 23.9%，性能几乎无损（42.20 vs 42.21）。

**[Multi-Head Low-Rank Attention (MLRA)](autonomous_driving/multi-head_low-rank_attention.md)**

:   提出 Multi-Head Low-Rank Attention (MLRA)，通过将 MLA 的单一 latent head 分解为多个可独立分片的 latent head，并对各分支注意力输出求和，实现原生 4-way 张量并行支持，在保持 SOTA 性能的同时获得 2.8× 的解码加速。

**[NeMo-map: Neural Implicit Flow Fields for Spatio-Temporal Motion Mapping](autonomous_driving/nemo-map_neural_implicit_flow_fields_for_spatio-temporal_motion_mapping.md)**

:   提出 NeMo-map——基于神经隐式函数的连续时空动态地图，将空间-时间坐标直接映射为半包裹高斯混合模型（SWGMM）参数，消除传统方法的空间离散化和时间分段限制，在真实行人追踪数据上实现更低 NLL 和更平滑的速度分布。

**[ResWorld: Temporal Residual World Model for End-to-End Autonomous Driving](autonomous_driving/resworld_temporal_residual_world_model_for_end-to-end_autonomous_driving.md)**

:   ResWorld 提出时序残差世界模型（TR-World），通过计算 BEV 场景表征的时序残差来提取动态物体信息（无需检测/跟踪），避免对静态区域的冗余建模，结合未来引导轨迹优化（FGTR）模块利用预测的未来 BEV 特征修正规划轨迹，在 nuScenes 和 NAVSIM 上达到 SOTA 规划性能。

**[SEAL: Segment Any Events with Language](autonomous_driving/segment_any_events_with_language.md)**

:   首次提出开放词汇事件实例分割（OV-EIS）任务，设计 SEAL 框架通过多模态层次语义引导（MHSG）和轻量多模态融合网络，在仅使用事件-图像对（无密集标注）的情况下，实现事件流的多粒度（实例级+部件级）语义分割，大幅领先所有基线方法且推理速度最快。

**[SiMO: Single-Modality-Operable Multimodal Collaborative Perception](autonomous_driving/simo_single-modality-operable_multimodal_collaborative_perceptio.md)**

:   提出 SiMO 框架，通过 LAMMA 融合模块和 PAFR 训练策略，首次在多智能体协同感知中实现任意模态缺失（特别是 LiDAR 失效仅有相机可用时）下仍可正常工作的多模态感知系统，类似并联电路——只要有一条通路就能工作。

**[SPACeR: Self-Play Anchoring with Centralized Reference Models](autonomous_driving/spacer_self-play_anchoring_with_centralized_reference_models.md)**

:   SPACeR 提出"类人自博弈"框架，用预训练的 tokenized 自回归运动模型作为集中式参考策略，通过对数似然奖励和 KL 散度约束引导去中心化自博弈 RL 策略向人类驾驶分布对齐，在 WOSAC 上超越纯自博弈方法，同时推理速度比模仿学习快 10 倍、参数量小 50 倍。

**[ST4VLA: Spatially Guided Training for Vision-Language-Action Models](autonomous_driving/st4vla_spatially_guided_training_for_vision-language-action_models.md)**

:   提出 ST4VLA，通过两阶段空间引导训练（spatial grounding pre-training + spatially guided action post-training），将 VLM 的空间先验显式注入 VLA 策略学习，在 SimplerEnv 上将 Google Robot 成功率从 66.1% 提升至 84.6%，WidowX 从 54.7% 提升至 73.2%，达到 SOTA。

**[Steerable Adversarial Scenario Generation through Test-Time Preference Alignment (SAGE)](autonomous_driving/steerable_adversarial_scenario_generation_through_test-time_preference_alignment.md)**

:   SAGE 将自动驾驶对抗场景生成重构为多目标偏好对齐问题，通过训练两个偏好专家模型并在推理时通过权重插值实现对抗性与真实性之间的连续可控权衡，无需重新训练即可生成从温和到激进的全谱场景，显著提升闭环训练效果。

---

## 🦾 LLM Agent { #llm_agent }

**[A Benchmark for Deep Information Synthesis (DeepSynth)](llm_agent/a_benchmark_for_deep_information_synthesis.md)**

:   提出 DeepSynth 基准，包含 120 个跨 7 领域 67 国的真实信息综合任务（平均需 5.5 小时人工标注），要求 agent 从多个网页收集信息并进行结构化推理，当前最强 agent（o3-deep-research）仅获 8.97 F1 / 17.5% LLM-Judge，揭示了 LLM agent 在信息综合方面的严重不足。

**[Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models](llm_agent/agentic_context_engineering_evolving_contexts_for_self-improving_language_models.md)**

:   提出 ACE（Agentic Context Engineering）框架，将 context 视为不断演化的"策略手册"（playbook），通过 Generator-Reflector-Curator 三角色分工和增量式 delta 更新来持续积累和精炼策略，解决了现有 prompt 优化中的简洁偏差和上下文坍塌问题，在 agent 任务上平均提升 10.6%、金融任务提升 8.6%，且自适应延迟降低 86.9%。

**[AgentSynth: Scalable Task Generation for Generalist Computer-Use Agents](llm_agent/agentsynth_scalable_task_generation_for_generalist_computer-use_agents.md)**

:   提出AgentSynth pipeline，利用信息不对称原理（正向逐步生成简单、反向整体求解困难）将简单子任务链式组合为复杂长程计算机使用任务，自动生成6000+多样化任务和轨迹，每条轨迹仅需$0.60，SOTA Agent在最高难度下成功率仅4%。

**[Ambig-SWE: Interactive Agents to Overcome Underspecificity in Software Engineering](llm_agent/ambig-swe_interactive_agents_to_overcome_underspecificity_in_software_engineerin.md)**

:   构建 Ambig-SWE（基于 SWE-Bench Verified 的欠指定变体），系统评估 LLM 编程 agent 在三个维度上的交互能力——检测欠指定、提出澄清问题、利用交互信息——发现交互可将欠指定场景下的解决率提升最高 74%，但模型默认非交互行为且难以区分指定充分/不足的指令。

**[ChatInject: Abusing Chat Templates for Prompt Injection in LLM Agents](llm_agent/chatinject_abusing_chat_templates_for_prompt_injection_in_llm_agents.md)**

:   揭示 LLM Agent 中 chat template 的结构性漏洞：通过在工具返回的数据中伪造角色标签（如 `<system>`, `<user>`），攻击者可以劫持模型的角色层级认知，将恶意指令伪装为高优先级指令，ASR 从 5-15% 提升至 32-52%。

**[FeatureBench: Benchmarking Agentic Coding for Complex Feature Development](llm_agent/featurebench_benchmarking_agentic_coding_for_complex_feature_development.md)**

:   提出 FeatureBench，面向复杂特征开发的 Agent 编程基准，200 个任务/24 个仓库，平均 790 行代码/15.7 文件——Claude Opus 4.5 仅解决 11%（vs SWE-bench 74.4%），揭示当前 Agent 在跨文件特征开发上的巨大差距。

**[Inherited Goal Drift: Contextual Pressure Can Undermine Agentic Goals](llm_agent/inherited_goal_drift_contextual_pressure_can_undermine_agentic_goals.md)**

:   发现现代 LLM agents 虽然对直接对抗性压力具有鲁棒性（目标偏移为 0），但会从弱模型的上下文中"继承"目标偏移行为；更反直觉的是，指令层级遵循能力（system vs user prompt 优先级）与偏移抗性之间缺乏相关性——Gemini 不遵循 system prompt 但偏移抗性不差，Qwen3 遵循 system prompt 但仍被传染。

**[Judge Reliability Harness: Stress Testing the Reliability of LLM Judges](llm_agent/judge_reliability_harness_stress_testing_the_reliability_of_llm_judges.md)**

:   提出 Judge Reliability Harness（JRH），一个开源框架，通过 label flip、格式不变性、语义改写、冗余偏差、随机稳定性 等合成测试系统评估 LLM Judge 的可靠性，在四个基准（FORTRESS、HarmBench、Persuade、AgentHarm）上对四个 SOTA Judge 进行压力测试，发现没有任何一个 Judge 在所有场景下都可靠。

**[MC-Search: Evaluating and Enhancing Multimodal Agentic Search with Structured Long Reasoning Chains](llm_agent/mc-search_evaluating_and_enhancing_multimodal_agentic_search_with_structured_lon.md)**

:   提出 MC-Search 基准（3333 个多跳多模态检索推理样本，平均 3.7 跳），通过 HAVE 验证和 Search-Align 过程监督微调，使开源模型达到接近商业模型的多模态搜索推理能力。

**[RefTool: Reference-Guided Tool Creation for Knowledge-Intensive Reasoning](llm_agent/reftool_reference-guided_tool_creation_for_knowledge-intensive_reasoning.md)**

:   提出 RefTool 框架基于外部参考资料（教材、知识片段）自动创建可执行 Python 工具，解决了现有工具创建方法依赖 LLM 内在知识在专业领域失败的问题，在因果推理、物理和化学任务上平均超过已有方法 12.3%。

**[Solving the Granularity Mismatch: Hierarchical Preference Learning for Long-Horizon LLM Agents](llm_agent/solving_the_granularity_mismatch_hierarchical_preference_learning_for_long-horiz.md)**

:   提出 HPL 框架解决长时序 LLM Agent 中偏好学习的粒度不匹配问题，通过三级 DPO（轨迹级+步骤级+动作组级）和双层课程学习（子任务复杂度×样本难度），在 ALFWorld/WebShop/InterCode-SQL 上显著超越 ETO 和 IPR 等基线（平均 59.44 vs 55.43/55.49）。

**[The Limits of Long-Context Reasoning in Automated Bug Fixing](llm_agent/the_limits_of_long-context_reasoning_in_automated_bug_fixing.md)**

:   系统评估当前 LLM 在长上下文代码调试中的能力极限，发现 agentic 工作流的成功来自任务分解而非长上下文推理（成功轨迹仅消耗 20-30K token），64K token 单次补丁生成中性能急剧下降（GPT-5-nano 0%），揭示名义上下文长度与实际可用上下文能力之间的显著差距。

**[ZeroDayBench: Evaluating LLM Agents on Unseen Zero-Day Vulnerabilities for Cyberdefense](llm_agent/zerodaybench_evaluating_llm_agents_on_unseen_zero-day_vulnerabilities_for_cyberd.md)**

:   提出首个评估 LLM Agent 发现并修补新型零日漏洞的 benchmark，通过将真实 CVE 移植到不同代码库创建 22 个新颖高危漏洞任务，在 5 个信息层级评估 Agent 能力，发现最强模型在 zero-day 级别仅 14.4% 通过率，说明自主漏洞发现仍是重大挑战。

---

## 📐 优化/理论 { #optimization }

**[Adaptive Rollout Allocation for Online RL with Verifiable Rewards (VIP)](optimization/adaptive_rollout_allocation_for_online_reinforcement_learning_with_verifiable_re.md)**

:   提出 VIP（Variance-Informed Predictive allocation），通过高斯过程预测每个 prompt 的成功概率，据此用凸优化在计算预算约束下分配 rollout 数量以最小化梯度方差，在数学推理任务上一致提升 GRPO/RLOO 的采样效率，AIME24/25 上 Pass@32 最高提升 12.3 个点。

**[CogFlow: Bridging Perception and Reasoning through Knowledge Internalization for Visual Mathematical Problem Solving](optimization/cogflow_bridging_perception_and_reasoning_through_knowledge_internalization_for_.md)**

:   CogFlow 提出认知启发的三阶段视觉数学推理框架（感知→内化→推理），通过 Synergistic Visual Rewards 增强感知、Knowledge Internalization Reward 桥接感知与推理、Visual-Gated Policy Optimization 锚定视觉推理，解决了现有方法中"感知正确但推理漂移"的核心问题。

**[Convex Dominance in Deep Learning I: A Scaling Law of Loss and Learning Rate](optimization/convex_dominance_in_deep_learning_i_a_scaling_law_of_loss_and_learning_rate.md)**

:   从凸优化理论出发，证明深度学习训练损失以 O(1/sqrt(T)) 速率收敛，最优学习率以 1/sqrt(T) 缩放，在 GPT-2 到 12.5B 参数模型上验证了该缩放律（R^2 >= 0.978），并实现了 80 倍训练步数的学习率外推。

**[Dual Optimistic Ascent (PI Control) is the Augmented Lagrangian Method in Disguise](optimization/dual_optimistic_ascent_pi_control_is_the_augmented_lagrangian_method_in_disguise.md)**

:   揭示了一个让人惊讶的等价性：广泛使用的双优化上升（Dual Optimistic Ascent / PI 控制）在数学上等价于增广拉格朗日方法（ALM），从而将 RL/RLHF 中的约束优化与经典优化理论统一起来，提供了更强的收敛保证和实用参数调优指南。

**[Exploring Diverse Generation Paths via Inference-time Stiefel Activation Steering](optimization/exploring_diverse_generation_paths_via_inference-time_stiefel_activation_steerin.md)**

:   提出 STARS，通过 Stiefel 流形上联合优化正交激活转向方向，使并行推理链产生多样化的推理路径（而非仅表面 token 变化），在测试用例生成和科学发现中超越温度/核采样，同时维护质量。

**[FrontierCO: Real-World and Large-Scale Evaluation of Machine Learning Solvers for Combinatorial Optimization](optimization/frontierco_real-world_and_large-scale_evaluation_of_machine_learning_solvers_for.md)**

:   FrontierCO 是一个涵盖 8 类组合优化问题（TSP、MIS、CVRP 等）的大规模真实世界基准测试，评估了 16 个 ML 求解器（神经网络方法 + LLM Agent）与 SOTA 传统求解器的差距，发现 ML 方法在结构复杂和极大规模实例上仍显著落后于传统方法，但在部分场景有超越潜力。

**[∇-Reasoner: LLM Reasoning via Test-Time Gradient Descent in Latent Space](optimization/nabla-reasoner_llm_reasoning_via_test-time_gradient_descent_in_latent_space.md)**

:   提出 ∇-Reasoner，将推理时的搜索从零阶（采样+评估）升级为一阶（梯度下降），在 token logits 空间上通过可微文本优化（DTO）结合 reward 梯度和 LLM 似然来迭代改进解码策略，在数学推理任务上提升 10-40% 准确率的同时减少 10-40% 的模型调用次数。

**[Neural Networks Learn Generic Multi-Index Models Near Information-Theoretic Limit](optimization/neural_networks_learn_generic_multi-index_models_near_information-theoretic_limi.md)**

:   证明在通用非退化假设下，标准两层神经网络通过分层梯度下降可以用 $\tilde{O}(d)$ 样本和 $\tilde{O}(d^2)$ 时间学习通用高斯 Multi-Index 模型 $f(\bm{x})=g(\bm{U}\bm{x})$，样本和时间复杂度都达到信息论最优，首次证明神经网络可以高效学习层次化函数。

**[Provable and Practical In-Context Policy Optimization for Self-Improvement](optimization/provable_and_practical_in-context_policy_optimization_for_self-improvement.md)**

:   提出 In-Context Policy Optimization (ICPO) 框架，理论证明单层线性自注意力 Transformer 经充分预训练后可在上下文中模拟策略优化算法，并设计实用的 ME-ICPO 算法通过最小熵选择和自评估奖励实现测试时多轮自反思，在数学推理任务上取得显著提升（AIME 2024 上 Qwen2.5-Math-7B 从 11% 提升到 30%）。

**[RRNCO: Towards Real-World Routing with Neural Combinatorial Optimization](optimization/rrnco_towards_real-world_routing_with_neural_combinatorial_optimization.md)**

:   提出 RRNCO 架构，通过自适应节点嵌入（ANE）和神经自适应偏置（NAB）两大创新，首次在深度路由框架中联合建模非对称距离、时长和方向角，并构建了基于 100 个真实城市的 VRP 基准数据集，显著缩小了 NCO 方法从仿真到真实世界部署的差距。

**[Scaf-GRPO: Scaffolded Group Relative Policy Optimization for Enhancing LLM Reasoning](optimization/scaf-grpo_scaffolded_group_relative_policy_optimization_for_enhancing_llm_reason.md)**

:   提出 Scaf-GRPO 框架，通过分层 in-prompt hint 注入（知识→规划→求解）解决 RLVR 中的"学习悬崖"问题——当模型对难题持续零奖励时，以最小引导恢复学习梯度，在 AIME24 上相对 vanilla GRPO 提升 44.3%。

**[Test-Time Meta-Adaptation with Self-Synthesis](optimization/test-time_meta-adaptation_with_self-synthesis.md)**

:   提出MASS框架，通过双层优化元学习让LLM在推理时自动生成问题特定的合成训练数据并自更新(LoRA)，在MATH-500上将Llama-3.1-8B从43.6%提升到59.0%。

**[The Affine Divergence: Aligning Activation Updates Beyond Normalisation](optimization/the_affine_divergence_aligning_activation_updates_beyond_normalisation.md)**

:   揭示了梯度下降中参数最速下降方向与传播到激活后的有效更新之间存在根本性不对齐（"仿射散度"$\Delta\mathcal{L}/\Delta z_i = (\partial\mathcal{L}/\partial z_i) \cdot (\|\vec{x}\|^2+1)$），从第一性原理推导出归一化是消除此散度的自然解，并发现一种非归一化的替代方案在实验中超越传统归一化。

---

## 🎬 视频理解 { #video_understanding }

**[A.I.R.: Adaptive, Iterative, and Reasoning-based Frame Selection For Video Question Answering](video_understanding/air_enabling_adaptive_iterative_and_reasoning-based_frame_selection_for_video_qu.md)**

:   提出 A.I.R.，一种无需训练的自适应-迭代-推理驱动帧选择框架，通过两阶段策略（GMM 自适应初始采样 + 迭代式 VLM 精细分析）解决 VideoQA 中轻量模型（CLIP）相似度不准确和 VLM 分析成本爆炸的双重困境，在最坏情况下也仅需分析 72 帧（vs 基线 128 帧），同时显著提升多个长视频 benchmark 性能。

**[BindWeave: Subject-Consistent Video Generation via Cross-Modal Integration](video_understanding/bindweave_subject-consistent_video_generation_via_cross-modal_integration.md)**

:   BindWeave 用多模态大语言模型（MLLM）替代传统的浅层融合机制来解析多主体复杂文本指令，生成主体感知的隐状态作为 DiT 的条件信号，结合 CLIP 语义特征和 VAE 细粒度外观特征，实现高保真、主体一致的视频生成。

**[Emergence of Superposition: Unveiling the Training Dynamics of Chain of Continuous Thought](video_understanding/emergence_of_superposition_unveiling_the_training_dynamics_of_chain_of_continuou.md)**

:   从理论上分析了两层 Transformer 在有向图可达性问题上使用连续 Chain-of-Thought（Coconut）训练时的训练动力学，揭示了"叠加态"（superposition）机制如何自然涌现：index-matching logit 先增长后有界，从而在探索与利用之间取得平衡。

**[FlashVID: Efficient Video Large Language Models via Training-free Tree-Based Spatiotemporal Token Merging](video_understanding/flashvid_efficient_video_large_language_models_via_training-free_tree-based_spat.md)**

:   提出 FlashVID，一个免训练的视频大语言模型推理加速框架，通过树状时空 token 合并（TSTM）联合建模空间和时间冗余，仅保留 10% 的视觉 token 就能保持 LLaVA-OneVision 99.1% 的性能，并能将 Qwen2.5-VL 的输入帧数提升 10 倍。

**[FLoC: Facility Location-Based Efficient Visual Token Compression for Long Video Understanding](video_understanding/floc_facility_location-based_efficient_visual_token_compression_for_long_video_u.md)**

:   提出 FLoC，基于设施选址函数（facility location function）的视觉 token 压缩框架，通过子模优化在给定预算下快速选择兼具代表性和多样性的 token 子集，实现无训练、模型无关、查询无关的长视频理解 token 压缩。

**[From Vicious to Virtuous Cycles: Synergistic Representation Learning for Unsupervised Video Object-Centric Learning](video_understanding/from_vicious_to_virtuous_cycles_synergistic_representation_learning_for_unsuperv.md)**

:   发现 slot-based 目标中心学习中编码器（产生尖锐但有噪声的注意力图）与解码器（产生空间一致但模糊的重建掩码）之间的恶性循环，提出同步对比学习目标和 slot 正则化预热策略将其转化为良性循环，在 MOVi 和 YouTube-VIS 上大幅提升物体发现性能。

**[GOT-Edit: Geometry-Aware Generic Object Tracking via Online Model Editing](video_understanding/got-edit_geometry-aware_generic_object_tracking_via_online_model_editing.md)**

:   通过零空间约束的在线模型编辑，将 VGGT 提供的 3D 几何信息融入 2D 通用目标跟踪器中，在保持语义判别力的同时增强几何感知能力，在遮挡和背景杂乱场景中显著提升跟踪性能。

**[JavisDiT: Joint Audio-Video Diffusion Transformer with Hierarchical Spatio-Temporal Prior Synchronization](video_understanding/javisdit_joint_audio-video_diffusion_transformer_with_hierarchical_spatio-tempor.md)**

:   提出 JavisDiT，基于 DiT 架构的音视频联合生成模型，通过层级化时空同步先验估计器（HiST-Sypo）实现细粒度的音视频时空对齐；同时构建了新基准 JavisBench（10K 复杂场景样本）和新评估指标 JavisScore。

**[Lumos-1: On Autoregressive Video Generation with Discrete Diffusion from a Unified Model Perspective](video_understanding/lumos-1_on_autoregressive_video_generation_with_discrete_diffusion_from_a_unifie.md)**

:   提出 Lumos-1，一个基于 LLM 架构的统一视频生成模型：通过 MM-RoPE（分布式多模态 RoPE）解决视觉时空编码问题，通过 AR-DF（自回归离散扩散强迫）解决帧间损失不均衡问题，仅用 48 GPU 训练即可在 GenEval、VBench-I2V 和 VBench-T2V 上达到竞争力水平。

**[Paper Copilot: Tracking the Evolution of Peer Review in AI Conferences](video_understanding/paper_copilot_tracking_the_evolution_of_peer_review_in_ai_conferences.md)**

:   提出 Paper Copilot，用多轮 LLM Agent 追踪论文在同行评审过程中的演变，整合 arXiv/OpenReview API 获取版本和评审历史，生成结构化变更摘要，支持跨 ≥3 版本的演变追踪。

**[CAPO: Curvature-Aware Policy Optimization for Sample-Efficient RL in LLM Reasoning](video_understanding/stabilizing_policy_gradients_for_sample-efficient_reinforcement_learning_in_llm_.md)**

:   CAPO 通过建模优化景观的二阶几何（仅在 LM head 最后一层计算曲率），实现 token 级别的数据筛选——拒绝会导致策略崩溃的更新，使 LLM 推理 RL 训练在激进超参数下仍保持稳定，样本效率提升 30 倍。

**[Stop Tracking Me! Proactive Defense Against Attribute Inference Attack in LLMs](video_understanding/stop_tracking_me_proactive_defense_against_attribute_inference_attack_in_llms.md)**

:   TRACE-RPS 提出统一防御框架应对 LLM 属性推断攻击：TRACE 通过注意力+推理链精准定位隐私泄露文本元素做细粒度匿名化，RPS 通过轻量后缀优化诱导模型拒绝推断，将属性推断准确率从约 50% 降至 5% 以下。

**[TTOM: Test-Time Optimization and Memorization for Compositional Video Generation](video_understanding/ttom_test-time_optimization_and_memorization_for_compositional_video_generation.md)**

:   提出 TTOM 框架，在推理时通过优化新增参数将视频生成模型的注意力与 LLM 生成的时空布局对齐，并用参数记忆机制保存历史优化上下文支持复用，在 T2V-CompBench 上相对提升 34%（CogVideoX）和 14%（Wan2.1）。

---

## 🧑 人体理解 { #human_understanding }

**[AMemGym: Interactive Memory Benchmarking for Assistants in Long-Horizon Conversations](human_understanding/amemgym_interactive_memory_benchmarking_for_assistants_in_long-horizon_conversat.md)**

:   提出AMemGym——首个支持on-policy交互式评估的长程对话记忆基准环境，通过结构化数据采样（用户画像→状态演化→个性化问答）驱动LLM模拟用户进行角色扮演，揭示了off-policy评估的排名偏差问题，并系统诊断了RAG/长上下文/Agent记忆系统的write/read/utilization三阶段失败模式。

**[AMPED: Adaptive Multi-objective Projection for balancing Exploration and skill Diversification](human_understanding/amped_adaptive_multi-objective_projection_for_balancing_exploration_and_skill_di.md)**

:   提出AMPED框架，在技能预训练阶段用梯度手术（PCGrad）平衡探索（熵+RND）和技能多样性（AnInfoNCE）之间的梯度冲突，在微调阶段用SAC-based技能选择器自适应选择最优技能，在Maze和URLB基准上超越DIAYN/CeSD/CIC等SBRL基线。

**[An Efficient, Provably Optimal Algorithm for the 0-1 Loss Linear Classification Problem](human_understanding/an_efficient_provably_optimal_algorithm_for_the_0-1_loss_linear_classification_p.md)**

:   提出增量单元枚举算法（ICE），首个具有严格证明的独立算法，可以在 $O(N^{D+1})$ 时间内精确求解0-1损失线性分类问题的全局最优解，并扩展到多项式超曲面分类。

**[Antibody: Strengthening Defense Against Harmful Fine-Tuning for Large Language Models via Attenuating Harmful Gradient Influence](human_understanding/antibody_strengthening_defense_against_harmful_fine-tuning_for_large_language_mo.md)**

:   提出Antibody防御框架：在对齐阶段通过平坦度正则化使模型处于有害损失的平坦区域（梯度小→难被攻击），在微调阶段用基于模型安全知识的样本加权方案（对比目标完成 vs 拒绝的似然比）抑制有害样本的学习，平均Harmful Score从15.29%降至7.04%。

**[Bayesian Influence Functions for Hessian-Free Data Attribution](human_understanding/bayesian_influence_functions_for_hessian-free_data_attribution.md)**

:   提出 Local Bayesian Influence Function (BIF)，用 SGLD 采样估计的协方差替代经典影响函数中不可行的 Hessian 逆运算，实现了对数十亿参数模型的无架构限制数据归因，在重训练实验中达到 SOTA。

**[Biologically Plausible Online Hebbian Meta-Learning: Two-Timescale Local Rules for Spiking Neural Brain Interfaces](human_understanding/biologically_plausible_online_hebbian_meta-learning_two-timescale_local_rules_fo.md)**

:   提出一种无需BPTT的在线SNN解码器，通过三因子Hebbian局部学习规则结合双时间尺度eligibility trace和自适应学习率控制，在O(1)内存下实现可比离线训练方法的BCI神经解码精度（Pearson R≥0.63/0.81），并在闭环仿真中展现了对神经信号非平稳性的持续适应能力。

**[DGNet: Discrete Green Networks for Data-Efficient Learning of Spatiotemporal PDEs](human_understanding/dgnet_discrete_green_networks_for_data-efficient_learning_of_spatiotemporal_pdes.md)**

:   提出 DGNet，将 Green 函数原理（叠加性）嵌入网络架构——学习离散 Green 矩阵 $G(\Delta t, \Delta x)$ 并通过矩阵指数组合多步预测，在 10-20% 数据下误差比 FNO/DeepONet 低 30-50%。

**[DiffVax: Optimization-Free Image Immunization Against Diffusion-Based Editing](human_understanding/diffvax_optimization-free_image_immunization_against_diffusion-based_editing.md)**

:   DiffVax 训练一个前馈免疫器（UNet++），对任意图像仅需一次前向传播（~70ms）即可生成不可感知的对抗扰动，使基于扩散模型的恶意编辑失败，相比先前逐图优化方法实现 250,000× 加速，并首次将免疫扩展到视频内容。

**[GaitSnippet: Gait Recognition Beyond Unordered Sets and Ordered Sequences](human_understanding/gaitsnippet_gait_recognition_beyond_unordered_sets_and_ordered_sequences.md)**

:   提出 GaitSnippet，无需轮廓（silhouette）的步态识别方法，从姿态/骨架序列中提取时空关键点片段（snippet），通过注意力池化聚合为人物级表征，CASIA-B 上 rank-1 达 96.2%（超轮廓方法 3-5%）。

**[NeuroGaze-Distill: Brain-informed Distillation and Depression-Inspired Geometric Priors for Robust Facial Emotion Recognition](human_understanding/neurogaze-distill_brain-informed_distillation_and_depression-inspired_geometric_.md)**

:   提出 NeuroGaze-Distill 跨模态蒸馏框架：从 EEG 脑电训练的教师模型中提取静态 Valence-Arousal 原型，通过 Proto-KD 和抑郁症启发的几何先验（D-Geo）注入纯视觉学生模型，无需 EEG-人脸配对数据，提升表情识别的跨数据集鲁棒性。

**[Soft Equivariance Regularization for Invariant Self-Supervised Learning](human_understanding/soft_equivariance_regularization_for_invariant_self-supervised_learning.md)**

:   提出 SER（Soft Equivariance Regularization），通过在 ViT 中间层施加软等变正则化、在最终层保持不变性目标的层解耦设计，在不引入额外模块的情况下，为不变性 SSL 方法（MoCo-v3, DINO, Barlow Twins）带来一致的分类精度和鲁棒性提升。

---

## ✂️ 语义分割 { #segmentation }

**[AMLRIS: Alignment-aware Masked Learning for Referring Image Segmentation](segmentation/amlris_alignment-aware_masked_learning_for_referring_image_segmentation.md)**

:   提出对齐感知遮蔽学习(AML)策略，通过量化视觉-语言 patch 级对齐度并过滤低对齐像素，让 RIS 模型在训练时聚焦可靠区域，无需架构改动即在 RefCOCO 全部 8 个 split 上达到 SOTA。

**[ByteFlow: Language Modeling through Adaptive Byte Compression without a Tokenizer](segmentation/byteflow_language_modeling_through_adaptive_byte_compression_without_a_tokenizer.md)**

:   提出 ByteFlow Net，一种无需分词器的分层字节级语言模型，利用信息论中的编码率(coding rate)自适应地将原始字节流压缩为语义单元，在预训练损失和下游任务上超越 BPE 基线和已有字节级架构。

**[Efficient-SAM2: Accelerating SAM2 with Object-Aware Visual Encoding and Memory Retrieval](segmentation/efficient-sam2_accelerating_sam2_with_object-aware_visual_encoding_and_memory_re.md)**

:   提出 Efficient-SAM2，通过对象感知的稀疏窗口路由(SWR)和稀疏记忆检索(SMR)两个后训练加速方案，利用 SAM2 本身的稀疏感知模式消除冗余计算，在 SAM2.1-L 上实现 1.68× 加速且仅损失 1.0% 精度。

**[Locality-Attending Vision Transformer](segmentation/locality-attending_vision_transformer.md)**

:   提出 LocAt，一个轻量级 ViT 插件，通过可学习高斯核调制自注意力偏向局部邻域(GAug)和无参数的 Patch 表征精炼(PRR)，在不改变训练范式的前提下为 ViT 带来 6%+ 的分割性能提升且不牺牲分类精度。

**[RegionReasoner: Region-Grounded Multi-Round Visual Reasoning](segmentation/regionreasoner_region-grounded_multi-round_visual_reasoning.md)**

:   提出 RegionReasoner，一个基于强化学习的多轮视觉推理框架，通过引用标注奖励和全局-局部一致性奖励，使推理轨迹必须显式引用参考区域坐标并保持语义连贯，在新构建的 RegionDial-Bench 上显著提升多轮定位和分割精度。

**[TRACE: Your Diffusion Model is Secretly an Instance Edge Detector](segmentation/trace_your_diffusion_model_is_secretly_an_instance_edge_detector.md)**

:   发现文本到图像扩散模型的自注意力图在去噪过程特定时间步隐式编码了实例边界信息，提出 TRACE 框架通过实例涌现点(IEP)和注意力边界散度(ABDiv)提取这些边界，并蒸馏为单步边缘检测器，在无监督实例分割和弱监督全景分割上大幅超越已有方法。

**[Universal Multi-Domain Translation via Diffusion Routers](segmentation/universal_multi-domain_translation_via_diffusion_routers.md)**

:   提出 Diffusion Router (DR)，一个统一的扩散模型框架，仅用 $K-1$ 个与中心域配对的数据集，通过单个噪声预测网络配合源域/目标域标签条件化，实现任意 $K$ 个域之间的间接和直接翻译，并提出 Tweedie 精炼采样降低计算成本。

**[VINCIE: Unlocking In-context Image Editing from Video](segmentation/vincie_unlocking_in-context_image_editing_from_video.md)**

:   提出 VINCIE，首次仅从视频数据学习上下文图像编辑能力——将视频标注为交错多模态序列，设计三个代理任务(次帧预测/当前分割/次帧分割预测)，在多轮编辑 benchmark 上达到 SOTA，展现了视频数据作为编辑训练源的可扩展性。

**[VIRTUE: Visual-Interactive Text-Image Universal Embedder](segmentation/virtue_visual-interactive_text-image_universal_embedder.md)**

:   提出 VIRTUE，将分割模型(SAM2)与 VLM 结合构建视觉交互式通用嵌入器，支持用户通过点/框/掩码指定兴趣区域产生实体级+全局级联合嵌入，并构建百万级 SCaR 基准评估视觉交互检索能力，在 36 个 MMEB 任务和 5 个 SCaR 任务上均达到 SOTA。

---

## 🕸️ 图学习 { #graph_learning }

**[A Geometric Perspective on the Difficulties of Learning GNN-based SAT Solvers](graph_learning/a_geometric_perspective_on_the_difficulties_of_learning_gnn-based_sat_solvers.md)**

:   从图 Ricci 曲率的几何视角证明随机 k-SAT 问题的二部图表示具有固有的负曲率，且曲率随问题难度增加而下降，建立了 GNN 过压缩 (oversquashing) 与 SAT 求解困难之间的理论联系，并通过测试时图重布线验证了该理论。

**[Are We Measuring Oversmoothing in Graph Neural Networks Correctly?](graph_learning/are_we_measuring_oversmoothing_in_graph_neural_networks_correctly.md)**

:   指出广泛使用的Dirichlet energy指标无法在实际场景中正确捕获GNN过平滑现象，提出以特征表征的数值秩/有效秩（effective rank）作为替代度量，实验表明Erank与准确率的平均相关性达0.91（vs Dirichlet energy的0.72），在OGB-Arxiv上Dirichlet energy甚至呈现错误的相关方向，并从理论上证明对广泛的GNN架构族其数值秩收敛到1（秩坍塌），重新定义过平滑为秩坍塌而非特征向量对齐。

**[Beyond Simple Graphs: Neural Multi-Objective Routing on Multigraphs](graph_learning/beyond_simple_graphs_neural_multi-objective_routing_on_multigraphs.md)**

:   首次提出针对多重图（multigraph）的神经组合优化路由方法 GMS，包含直接在多重图上边级自回归构造的 GMS-EB 和先学习剪枝再节点级路由的双头 GMS-DH 两个变体，在非对称多目标 TSP 和 CVRP 上实现了接近精确求解器 LKH 的性能且速度快数十倍。

**[Bilinear Representation Mitigates Reversal Curse and Enables Consistent Model Editing](graph_learning/bilinear_representation_mitigates_reversal_curse_and_enables_consistent_model_ed.md)**

:   通过在合成关系知识图谱上从头训练 Transformer，发现适当正则化会使模型隐层涌现出双线性关系结构（bilinear relational structure），该结构不仅能克服逆向诅咒（reversal curse），还能实现编辑单个事实后逻辑一致地传播到相关事实。

**[Embodied Agents Meet Personalization: Investigating Challenges and Solutions Through the Lens of Memory Utilization](graph_learning/embodied_agents_meet_personalization_investigating_challenges_and_solutions_thro.md)**

:   提出 Memento 评估框架，系统揭示 LLM 具身智能体在个性化辅助任务中的记忆利用瓶颈（信息过载、多记忆协调失败），并设计层次化知识图谱用户画像记忆模块显著改善性能。

**[Graph Tokenization for Bridging Graphs and Transformers](graph_learning/graph_tokenization_for_bridging_graphs_and_transformers.md)**

:   提出 GraphTokenizer 框架，将图通过可逆的频率引导序列化转换为符号序列，再用 BPE 学习图子结构词汇表，使标准 Transformer（如 BERT/GTE）无需任何架构修改即可直接处理图数据，在 14 个 benchmark 上达到 SOTA。

**[Pairwise is Not Enough: Hypergraph Neural Networks for Multi-Agent Pathfinding](graph_learning/pairwise_is_not_enough_hypergraph_neural_networks_for_multi-agent_pathfinding.md)**

:   提出 HMAGAT，用有向超图注意力网络替代 GNN 的成对消息传递来建模多智能体路径规划中的群体交互，仅用 1M 参数和 1% 训练数据即超越 85M 参数的 SOTA 模型。

**[Structurally Human, Semantically Biased: Detecting LLM-Generated References with Embeddings and GNNs](graph_learning/structurally_human_semantically_biased_detecting_llm-generated_references_with_e.md)**

:   发现 LLM 生成的参考文献在图拓扑结构上与人类引用高度相似（图结构分类器仅 0.61 准确率），但在语义嵌入上可被检测（GNN+嵌入达 93% 准确率），揭示了 LLM 的"结构模仿、语义偏向"特性。

---

## 🎵 音频/语音 { #audio_speech }

**[AC-Foley: Reference-Audio-Guided Video-to-Audio Synthesis with Acoustic Transfer](audio_speech/ac-foley_reference-audio-guided_video-to-audio_synthesis_with_acoustic_transfer.md)**

:   提出 AC-Foley，一种参考音频引导的视频到音频合成框架，通过两阶段训练（声学特征学习+时序适应）和多模态条件流匹配实现了细粒度音色控制、音色迁移和零样本音效生成，在音频质量和声学保真度上显著优于现有方法。

**[FlexiCodec: A Dynamic Neural Audio Codec for Low Frame Rates](audio_speech/flexicodec_a_dynamic_neural_audio_codec_for_low_frame_rates.md)**

:   提出 FlexiCodec，通过 ASR 特征引导的动态帧率合并策略，在 3–12.5Hz 超低帧率下实现高质量语音编解码，同时保持优异的语义信息保留能力。

**[Incentive-Aligned Multi-Source LLM Summaries](audio_speech/incentive-aligned_multi-source_llm_summaries.md)**

:   将博弈论中的多任务 peer prediction 机制引入 LLM 多源摘要管线，提出 Truthful Text Summarization (TTS) 框架：通过 leave-one-out 交叉构造评价声明集、提取每个来源对声明的立场、用 informative agreement 评分来源可靠性并过滤不可靠来源后重新摘要，理论上证明"如实报告是效用最大策略"，实验中有效抵御 prompt injection、虚假信息源和协同攻击。

**[Latent Speech-Text Transformer](audio_speech/latent_speech_text_transformer.md)**

:   提出 Latent Speech-Text Transformer (LST)，将离散语音 token 聚合为更高层级的"潜在语音 patch"作为自回归单元，对齐语音和文本的序列建模粒度，在 speech HellaSwag 上获得 +6.5% 绝对提升，且增益随模型规模（420M→7B）持续增长，同时降低 ASR/TTS 推理计算成本。

**[MMSU: A Massive Multi-task Spoken Language Understanding and Reasoning Benchmark](audio_speech/mmsu_a_massive_multi-task_spoken_language_understanding_and_reasoning_benchmark.md)**

:   提出 MMSU（5000 条音频 QA、47 个任务），首个系统融合语言学理论的语音理解与推理基准，评测 22 个 SpeechLLM，发现现有模型在音韵感知和复杂推理上仍存在显著差距。

**[Query-Guided Spatial-Temporal-Frequency Interaction for Music Audio-Visual Question Answering](audio_speech/query-guided_spatial-temporal-frequency_interaction_for_music_audio-visual_quest.md)**

:   提出 QSTar 框架，通过在整个处理流程中嵌入问题引导（Query Guidance），并引入空间-时序-频域三维度交互模块（特别是利用频谱特征区分音色），显著提升了音乐场景下的音频-视觉问答（Music AVQA）性能。

**[RedTeamCUA: Realistic Adversarial Testing of Computer-Use Agents in Hybrid Web-OS Environments](audio_speech/redteamcua_adversarial_testing_agents.md)**

:   构建首个混合 Web-OS 环境的 CUA 红队测试框架 RedTeamCUA 和 864 个测试用例的 RTC-Bench，系统评估 9+ 前沿 CUA 对间接 prompt injection 的脆弱性，发现所有 CUA 均可被攻击（最高 ASR 83%），且能力越强的模型越危险——攻击尝试率（AR）远高于成功率（ASR）意味着模型能力提升将直接转化为更高的攻击成功率。

---

## 🔗 因果推理 { #causal_inference }

**[Action-Guided Attention for Video Action Anticipation](causal_inference/action-guided_attention_for_video_action_anticipation.md)**

:   提出动作引导注意力 (AGA) 机制，用模型自身的动作预测序列作为注意力的 Query 和 Key（而非像素特征），结合自适应门控融合历史上下文和当前帧特征，在 EPIC-Kitchens-100 上实现从验证集到测试集的良好泛化，同时支持训练后的可解释性分析。

**[AgentTrace: Causal Graph Tracing for Root Cause Analysis in Deployed Multi-Agent Systems](causal_inference/agenttrace_causal_graph_tracing_for_root_cause_analysis_in_deployed_multi-agent_.md)**

:   提出AgentTrace框架，从多智能体系统的执行日志中构建因果图，通过反向追踪+轻量级特征排序（五组特征的加权线性组合）定位根因节点，在550个合成故障场景上Hit@1达94.9%，延迟0.12秒，比LLM分析快69倍。

**[Copy-Paste to Mitigate Large Language Model Hallucinations](causal_inference/copy-paste_to_mitigate_large_language_model_hallucinations.md)**

:   提出 Copy-Paste 生成范式，通过训练 LLM 优先直接复制检索上下文中的片段来生成回答，而非自由改写，配合高复制偏好的 DPO 训练，在反事实 RAG 基准上将忠实度从 80.2% 提升到 92.8%。

**[Counterfactual Explanations on Robust Perceptual Geodesics](causal_inference/counterfactual_explanations_on_robust_perceptual_geodesics.md)**

:   提出 PCG（Perceptual Counterfactual Geodesic）方法，在鲁棒感知流形上通过测地线优化生成语义忠实的反事实解释，两阶段优化确保路径既感知自然又达到目标类别，在 AFHQ 上 FID=8.3 远优于 RSGD 的 12.9。

**[Function Induction and Task Generalization: An Interpretability Study with Off-by-One Addition](causal_inference/function_induction_and_task_generalization_an_interpretability_study_with_off-by.md)**

:   通过 off-by-one addition（如 1+1=3, 2+2=5）这一反事实任务，利用 path patching 发现大语言模型内部存在 **function induction** 机制——一种超越 token 级别 pattern matching、在函数级别进行归纳推理的注意力头电路，并证明该机制可跨任务复用。

**[Learning Robust Intervention Representations with Delta Embeddings](causal_inference/learning_robust_intervention_representations_with_delta_embeddings.md)**

:   提出因果 Delta 嵌入（CDE）框架，将干预/动作表示为预干预和后干预状态在潜空间中的向量差，通过独立性、稀疏性和不变性三种约束学习鲁棒的干预表示，在 Causal Triplet 挑战中显著超越基线的 OOD 泛化性能，且能自动发现反义动作的反平行语义结构。

---

## 🖼️ 图像恢复 { #image_restoration }

**[Activation Steering for Masked Diffusion Language Models](image_restoration/activation_steering_for_masked_diffusion_language_models.md)**

:   首次将激活引导（activation steering）应用于 Masked Diffusion 语言模型（MDLM），发现 MDLM 的拒绝行为也受单一低维方向控制，通过在去噪过程中全局投影可完全绕过安全对齐，且与自回归模型不同，有效方向可从指令前的 token 中提取——反映了扩散模型的非因果并行处理特性。

**[Are Deep Speech Denoising Models Robust to Adversarial Noise?](image_restoration/are_deep_speech_denoising_models_robust_to_adversarial_noise.md)**

:   首次系统性评估 4 款 SOTA 深度语音去噪（DNS）模型在对抗噪声下的鲁棒性：通过心理声学约束的 PGD 攻击生成人耳不可感知的对抗噪声，可令 Demucs、Full-SubNet+、FRCRN 和 MP-SENet 输出完全不可理解的 gibberish，实验覆盖多种声学条件和人类评估，同时揭示了目标攻击、通用扰动和跨模型迁移的局限性。

**[Beyond Scattered Acceptance: Fast and Coherent Inference for DLMs via Longest Stable Prefixes](image_restoration/beyond_scattered_acceptance_fast_and_coherent_inference_for_dlms_via_longest_sta.md)**

:   LSP 调度器通过在每个去噪步骤中原子性地提交最长连续稳定前缀（而非分散接受离散 token），将 DLM 推理加速 3.4 倍，同时保持或略微提升输出质量。

**[Generalizing Linear Autoencoder Recommenders with Decoupled Expected Quadratic Loss](image_restoration/generalizing_linear_autoencoder_recommenders_with_decoupled_expec.md)**

:   将 EDLAE 推荐模型的目标函数推广为解耦期望二次损失（DEQL），在超参数 $b>0$ 的更广范围内推导出闭式解，并通过 Miller 矩阵逆定理将计算复杂度从 $O(n^4)$ 降至 $O(n^3)$，在多个基准数据集上超越 EDLAE 和深度学习模型。

**[InterActHuman: Multi-Concept Human Animation with Layout-Aligned Audio Conditions](image_restoration/interacthuman_multi-concept_human_animation_with_layout-aligned_audio_conditions.md)**

:   提出 InterActHuman，通过自动推断时空布局的掩码预测器和迭代掩码引导策略，实现多人/人物交互场景下的音频驱动视频生成，支持每个角色独立的语音驱动口型同步和身体动作。

**[Toward Safer Diffusion Language Models: Discovery and Mitigation of Priming Vulnerabilities](image_restoration/toward_safer_diffusion_language_models_discovery_and_mitigation_of_priming_vulne.md)**

:   揭示了掩码扩散语言模型（MDLM）中的"启动漏洞"（priming vulnerability）——在去噪中间步骤注入肯定性 token 可绕过安全防线，并提出 Recovery Alignment（RA）方法训练模型从被污染的中间状态恢复到安全响应。

---

## 📈 时间序列 { #time_series }

**[Adapt Data to Model: Adaptive Transformation Optimization for Domain-shared Time Series Foundation Models](time_series/adapt_data_to_model_adaptive_transformation_optimization_for_domain-shared_time_.md)**

:   提出TATO框架，通过自动优化数据预处理 pipeline（包括上下文裁切、尺度归一化、异常值校正），让冻结的大型时序模型（LTM）在不微调的情况下适配不同下游领域，平均降低MSE 13.6%，最高65.4%。

**[Contextual and Seasonal LSTMs for Time Series Anomaly Detection](time_series/contextual_and_seasonal_lstms_for_time_series_anomaly_detection.md)**

:   针对单变量时间序列中现有方法难以检测的"小幅点异常"和"缓慢上升异常"，提出 CS-LSTMs 双分支架构——S-LSTM 在频域建模周期性演化、C-LSTM 在时域捕捉局部趋势，结合小波噪声分解策略，在四个基准上全面超越 SOTA 且推理速度提升 40%。

**[Enhancing Multivariate Time Series Forecasting with Global Temporal Retrieval](time_series/enhancing_multivariate_time_series_forecasting_with_global_temporal_retrieval.md)**

:   提出 Global Temporal Retriever（GTR），一个轻量级即插即用模块，通过维护自适应全局周期嵌入并利用绝对时间索引检索对齐全局周期信息，使任意预测模型突破回看窗口限制，有效捕获远超输入长度的全局周期模式。

**[Reasoning on Time-Series for Financial Technical Analysis](time_series/reasoning_on_time-series_for_financial_technical_analysis.md)**

:   提出 Verbal Technical Analysis (VTA) 框架，结合 LLM 的语言推理能力与时间序列模型的模式捕捉能力，通过 Time-GRPO 强化学习优化推理链，并以推理属性条件化时序预测，实现了兼具准确性和可解释性的金融时间序列预测。

**[T1: One-to-One Channel-Head Binding for Multivariate Time-Series Imputation](time_series/t1_one-to-one_channel-head_binding_for_multivariate_time-series_imputation.md)**

:   提出 T1，通过通道-头一对一绑定学习多变量时间序列的结构依赖，用交叉注意力捕获通道-时间关联，MAE 在 Energy/Traffic/Weather 上超 SOTA 5-12%。

**[TimeSliver : Symbolic-Linear Decomposition for Explainable Time Series Classification](time_series/timesliver_symbolic-linear_decomposition_for_explainable_time_series_classificat.md)**

:   提出 TimeSliver，将时间序列分解为 Hermite 多项式基+线性残差，通过 Hermite 特征结构分析生成可解释的时间归因，UCR 基准上精度 95%+ 同时完全可解释，top-K 归因与领域专家 87% 一致。

---

## 🔎 AIGC 检测 { #aigc_detection }

**[Calibrating Verbalized Confidence with Self-Generated Distractors](aigc_detection/calibrating_verbalized_confidence_with_self-generated_distractors.md)**

:   提出 DiNCo（Distractor-Normalized Confidence）方法，让 LLM 自动生成"合理但错误"的干扰选项，然后在干扰选项集合上归一化置信度分数，实现跨难度级别的置信度校准，在 TriviaQA 上以 95.2% 均衡准确率和仅 3.5% 人类介入率实现可靠的自动决策。

**[CLARC: C/C++ Benchmark for Robust Code Search](aigc_detection/clarc_cc_benchmark_for_robust_code_search.md)**

:   构建了首个可编译的 C/C++ 代码检索基准 CLARC，包含 6,717 个查询-代码对，覆盖标准/匿名化/汇编/WebAssembly 四种检索场景，揭示了现有代码嵌入模型在二进制级代码检索上的严重不足（Hole@10=17.1%）。

**[DMAP: A Distribution Map for Text](aigc_detection/dmap_a_distribution_map_for_text.md)**

:   提出 DMAP，将文本通过语言模型的 token 概率映射到 [0,1] 单位区间上的样本，理论证明纯采样文本产生均匀分布，由此可用统计检验分析生成参数（如 top-k）、检测机器生成文本、揭示后训练的统计指纹。

**[PoliCon: Evaluating LLMs on Achieving Diverse Political Consensus Objectives](aigc_detection/policon_evaluating_llms_on_achieving_diverse_political_consensus_objectives.md)**

:   基于欧洲议会2225条真实审议记录构建PoliCon基准，评估LLM在不同政治目标（简单多数/2/3多数/否决权/罗尔斯/功利主义）下起草共识决议的能力。

---

## 🎁 推荐系统 { #recommender }

**[GoalRank: Group-Relative Optimization for a Large Ranking Model](recommender/goalrank_group-relative_optimization_for_a_large_ranking_model.md)**

:   理论证明任意 Multi-Generator-Evaluator 排序系统都存在一个更大的 generator-only 模型以更小的误差逼近最优策略且满足 scaling law，据此提出 GoalRank——用 reward model 构建 group-relative 参考策略来训练大型 generator-only 排序模型，在线 A/B 测试中显著优于 SOTA。

**[ProPerSim: Developing Proactive and Personalized AI Assistants through User-Assistant Simulation](recommender/propersim_developing_proactive_and_personalized_ai_assistants_through_user-assis.md)**

:   提出 ProPerSim 模拟框架和 ProPerAssistant 基线，通过用户-助手模拟环境结合 DPO 偏好学习，开发能同时具备主动性和个性化的 AI 家庭助手。

**[Token-Efficient Item Representation via Images for LLM Recommender Systems](recommender/token-efficient_item_representation_via_images_for_llm_recommender_systems.md)**

:   提出 I-LLMRec，利用商品图像替代冗长文本描述来表示推荐系统中的物品语义，通过 RISA 对齐模块和 RERI 检索模块，在仅用单个token表示物品的同时保留丰富语义，推理速度提升约2.93倍且推荐性能超越文本描述方法。

---

## ✍️ 文本生成 { #nlp_generation }

**[AP-OOD: Attention Pooling for Out-of-Distribution Detection](nlp_generation/ap-ood_attention_pooling_for_out-of-distribution_detection.md)**

:   提出AP-OOD，将Mahalanobis距离的均值池化替换为可学习的注意力池化，解决了均值池化丢失token级异常信息的问题，在文本OOD检测中将XSUM摘要的FPR95从27.84%降至4.67%，支持无监督到半监督的平滑过渡。

**[Sharing State Between Prompts and Programs](nlp_generation/sharing_state_between_prompts_and_programs.md)**

:   提出共享程序状态（shared program state）抽象，让 prompt 直接读写程序变量、操作堆对象和控制程序流程，实现为 Nightjar 系统（Python + prompt 混合编程），在保持或提升准确率（+4-19%）的同时减少 39.6% 代码量。

---

## 📖 NLP 理解 { #nlp_understanding }

**[BTZSC: A Benchmark for Zero-Shot Text Classification Across Cross-Encoders, Embedding Models, Rerankers and LLMs](nlp_understanding/btzsc_a_benchmark_for_zero-shot_text_classification_across_cross-encoders_embedd.md)**

:   提出 BTZSC 基准（22 个数据集），首次在统一零样本协议下系统比较 NLI 交叉编码器、嵌入模型、Reranker 和指令微调 LLM 四大模型家族（共 38 个模型），发现 Qwen3-Reranker-8B 以 macro F1=0.72 取得新 SOTA，嵌入模型在精度-延迟权衡上最优。

**[Same Content, Different Representations: A Controlled Study for Table QA](nlp_understanding/same_content_different_representations_a_controlled_study_for_t.md)**

:   首个控制变量研究：在保持表格内容完全相同的条件下变换表示形式（结构化 vs 半结构化），系统评估 NL2SQL、LLM、混合三类方法在不同表格大小/模式质量/查询复杂度下的鲁棒性，发现表示形式是影响 Table QA 性能的一阶因素。

---

## 🛰️ 遥感 { #remote_sensing }

**[AutoFly: Vision-Language-Action Model for UAV Autonomous Navigation in the Wild](remote_sensing/autofly_vision-language-action_model_for_uav_autonomous_navigation_in_the_wild.md)**

:   提出 AutoFly，一个面向无人机野外自主导航的端到端 VLA 模型，通过伪深度编码器从 RGB 输入推断空间信息，配合新构建的自主导航数据集（13K+ 轨迹含 1K 真实飞行），在模拟和真实环境中比 OpenVLA 成功率高 3.9%，碰撞率低 2.6%。

**[Task-free Adaptive Meta Black-box Optimization](remote_sensing/task-free_adaptive_meta_black-box_optimization.md)**

:   提出 ABOM——一种无需预定义训练任务的自适应元黑盒优化器，通过将进化算子（选择、交叉、变异）参数化为可微注意力模块，在优化过程中利用自生成数据在线更新参数，在合成基准和无人机路径规划上实现零样本竞争性能。

---

## 🧮 科学计算 { #scientific_computing }

**[Astral: Training Physics-Informed Neural Networks with Error Majorants](scientific_computing/astral_training_physics-informed_neural_networks_with_error_majorants.md)**

:   提出 Astral 损失函数（基于函数型后验误差上界/error majorant），替代传统 PiNN 中的残差损失来训练物理信息神经网络，实现训练过程中可靠的误差估计，并在扩散方程、Maxwell 方程等多种 PDE 上取得了更好或相当的精度。

**[Learning-guided Kansa Collocation for Forward and Inverse PDE Problems](scientific_computing/learning-guided_kansa_collocation_for_forward_and_inverse_pde_problems.md)**

:   将基于径向基函数(RBF)的无网格Kansa方法从单变量线性PDE扩展到耦合多变量和非线性PDE场景，结合自调参技术和多种时间步进方案，并系统对比了与PINN、FNO等神经PDE求解器在正问题和反问题上的表现。

---

## 📡 信号/通信 { #signal_comm }

**[Group Representational Position Encoding (GRAPE)](signal_comm/group_representational_position_encoding.md)**

:   提出 GRAPE 框架，基于群作用（group actions）统一了 Transformer 中乘法型（RoPE）和加法型（ALiBi/FoX）两大位置编码家族，证明 RoPE 和 ALiBi 是其精确特例，并提出路径积分加法变体 GRAPE-AP 在下游任务上超越现有方法。

**[Multi-Agent Design: Optimizing Agents with Better Prompts and Topologies](signal_comm/multi-agent_design_optimizing_agents_with_better_prompts_and_topologies.md)**

:   深入分析多智能体系统中 prompt 和拓扑设计的影响，发现 prompt 优化是最关键的设计因素（仅优化 prompt 的单 Agent 即可超越复杂多 Agent 拓扑），提出 Mass 三阶段框架（block-level prompt → topology → workflow-level prompt）在 8 个 benchmark 上取得 SOTA。

---

## ⚛️ 物理学 { #physics }

**[Sublinear Time Quantum Algorithm for Attention Approximation](physics/sublinear_time_quantum_algorithm_for_attention_approximation.md)**

:   提出首个对序列长度 $n$ 具有**亚线性**时间复杂度的量子数据结构，用于近似 Transformer 注意力矩阵的行查询，预处理时间 $\widetilde{O}(\epsilon^{-1} n^{0.5} \cdot \text{poly}(d, s_\lambda, \alpha))$，每次行查询 $\widetilde{O}(s_\lambda^2 + s_\lambda d)$，相对经典算法实现了关于 $n$ 的二次加速。

---

## 🔄 自监督/表示学习 { #self_supervised }

**[Difficult Examples Hurt Unsupervised Contrastive Learning: A Theoretical Perspective](self_supervised/difficult_examples_hurt_unsupervised_contrastive_learning_a_theoretical_perspect.md)**

:   通过相似度图模型理论分析证明"困难样本"（跨类高相似度样本）会损害无监督对比学习性能，提出删除困难样本、调节 margin 和温度缩放三种策略，在 TinyImageNet 上带来 15% 的提升。
