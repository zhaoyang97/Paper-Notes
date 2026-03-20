---
search:
  exclude: true
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📂 其他

**🔬 ICLR2026** · 共 **44** 篇

**[A Federated Generalized Expectation-Maximization Algorithm for Mixture Models with an Unknown Number of Components](a_federated_generalized_expectation-maximization_algorithm_for_mixture_models_wi.md)**

:   提出 FedGEM 算法，通过客户端本地 EM 步后构建不确定性集、服务器利用不确定性集交集检测聚类重叠并推断全局聚类数，首次实现在全局聚类数未知情况下的联邦聚类，并提供了概率收敛保证。

**[A Law of Data Reconstruction for Random Features (and Beyond)](a_law_of_data_reconstruction_for_random_features_and_beyond.md)**

:   从信息论和代数角度证明随机特征模型中存在数据重构定律：当参数量 $p \gg dn$（$d$ 为数据维度，$n$ 为样本数）时，训练数据可被完整重构，并通过投影损失优化方法在 RF、两层网络和 ResNet 上验证了该阈值的普适性。

**[A Representer Theorem for Hawkes Processes via Penalized Least Squares Minimization](a_representer_theorem_for_hawkes_processes_via_penalized_least_squares_minimizat.md)**

:   为线性多元 Hawkes 过程在 RKHS 框架下的触发核估计建立了新型表示定理，证明最优估计器可用等价核在数据点上的线性组合表示且对偶系数全部解析地等于 1，无需求解对偶优化问题，从而实现高效可扩展的非参数估计。

**[A Scalable Inter-edge Correlation Modeling in CopulaGNN for Link Sign Prediction](a_scalable_inter-edge_correlation_modeling_in_copulagnn_for_link_sign_prediction.md)**

:   将 CopulaGNN 从节点级扩展到边级，通过将相关矩阵构造为边嵌入的 Gramian 矩阵并利用 Woodbury 恒等式重构条件概率分布，实现了在签名图上对边间统计依赖的可扩展建模，用于链接符号预测任务。

**[A Single Architecture for Representing Invariance Under Any Space Group](a_single_architecture_for_representing_invariance_under_any_space_group.md)**

:   设计了一种可自适应任意空间群不变性的单一架构 (Crystal Fourier Transformer)，通过解析推导群操作对傅里叶系数的约束来构造对称适配的傅里叶基，用约束的对偶图表示实现了跨 230 个空间群的参数共享和零样本泛化。

**[Accessible, Realistic, and Fair Evaluation of Positive-Unlabeled Learning Algorithms](accessible_realistic_and_fair_evaluation_of_positive-unlabeled_learning_algorith.md)**

:   提出首个 PU 学习统一基准，系统解决两个关键问题：(1) 用代理准确率和代理 AUC 实现无负样本的模型选择；(2) 发现并通过将正样本并入无标签集的简单校准方法解决单样本设置下的内部标签偏移问题，使双样本算法在单样本评估中得到公平比较。

**[Active Learning for Decision Trees with Provable Guarantees](active_learning_for_decision_trees_with_provable_guarantees.md)**

:   为决策树主动学习提供首个理论保证：(1) 首次分析决策树的不一致系数（disagreement coefficient）并给出 $O(\ln^{OPT}(n))$ 上界；(2) 提出首个达到乘法误差 $(1+\epsilon)$ 保证的二分类主动学习算法；结合两者实现数据集大小的多对数标签复杂度。

**[Addressing Divergent Representations from Causal Interventions on Neural Networks](addressing_divergent_representations_causal.md)**

:   系统性地揭示因果干预（activation patching、DAS、SAE 等）会将模型内部表征推离自然分布，理论区分"无害偏移"与"有害偏移"两类情况，并提出 Counterfactual Latent (CL) loss 来约束干预表征不偏离流形，在 7B LLM 上验证可减少偏移同时保持干预准确率。

**[Agnostics: Learning to Synthesize Code in Any Programming Language with a Universal RL Environment](agnostics_learning_to_code_in_any_programming_language_via_reinforcement_with_a_.md)**

:   提出Agnostics，一种语言无关的后训练pipeline：将编程任务统一为I/O行为规范格式，用通用验证器+GRPO强化学习训练LLM在任何编程语言上编码，使Qwen 4B在Lua/Julia/R/OCaml/Fortran五种低资源语言上达到匹敌16B-70B模型的SOTA水平。

**[An Information-Theoretic Framework For Optimizing Experimental Design To Distinguish Probabilistic Neural Codes](an_information-theoretic_framework_for_optimizing_experimental_design_to_disting.md)**

:   提出"信息间隙"（information gap）框架，通过优化刺激分布来最大化似然编码（likelihood code）与后验编码（posterior code）假设之间的可区分性，推导出真实后验与任务边缘化代理后验之间的KL散度作为优化目标，并通过DNN解码器在模拟神经群体上验证了该框架的有效性，揭示传统单上下文实验无法区分两种编码假设。

**[AnesSuite: A Comprehensive Benchmark and Dataset Suite for Anesthesiology Reasoning](anessuite_a_comprehensive_benchmark_and_dataset_suite_for_anesthesiology_reasoni.md)**

:   构建首个面向麻醉学推理的综合数据集套件AnesSuite——包括AnesBench（7972道双语选择题）、AnesCorpus（240万篇文档语料库）、AnesQA（2万条QA对）和AnesR1（1万条CoT推理数据），提出三级认知需求分类（System 1/1.x/2），训练的Morpheus模型（Qwen2.5 + SFT + GRPO）在7B参数下达到14B基线性能，揭示当前最强模型在复杂推理（System 2）上仍低于0.6。

**[ANO: Faster is Better in Noisy Landscapes](ano_faster_is_better_in_noisy_landscape.md)**

:   提出 Ano 优化器，将更新方向和幅度解耦——方向用动量的符号（sign）确保噪声鲁棒，幅度用瞬时梯度绝对值（而非动量幅度）确保响应速度，配合改进的 Yogi 式方差估计，在噪声和非平稳环境（如 RL）中显著优于 Adam/Lion/Adan，同时在标准任务上保持竞争力。

**[AnyUp: Universal Feature Upsampling](anyup_universal_feature_upsampling.md)**

:   提出AnyUp——首个推理时encoder无关的可学习特征上采样方法，通过feature-agnostic层处理任意维度/类型的视觉特征，配合窗口注意力架构和crop-based训练策略，训练一次即可对任意视觉编码器（DINO/CLIP/SigLIP/MAE等）的特征进行任意分辨率上采样，在多个下游任务上超越FeatUp/JAFAR/LoftUp等方法。

**[Articulation in Motion: Prior-Free Part Mobility Analysis for Articulated Objects](articulation_in_motion_prior-free_part_mobility_analysis_for_articulated_objects.md)**

:   提出AiM（Articulation in Motion）框架，从交互视频和初始状态扫描中无需部件数量先验地重建铰接物体——通过双高斯表征（静态GS + 可变形GS）实现动静解耦，结合顺序RANSAC进行无先验部件分割和关节估计，辅以SDMD模块处理新暴露的静态区域，在复杂6部件物体（Storage）上以79.34% mean IoU大幅超越需先验的ArtGS（52.23%）。

**[Assess A Semantic And Structural Evaluation Framework For Statement Similarity](assess_a_semantic_and_structural_evaluation_framework_for_statement_similarity.md)**

:   提出 TransTED Similarity，一种基于算子树 (Operator Tree) 和语义变换增强的树编辑距离指标，用于评估自动形式化 (autoformalization) 生成的形式化数学命题与参考命题之间的语义相似度，并构建了 EPLA 基准数据集。

**[AstaBench: Rigorous Benchmarking of AI Agents with a Scientific Research Suite](astabench_benchmarking_ai_agents.md)**

:   由 AI2 团队构建的首个端到端科学研究 Agent 基准 AstaBench，包含 2400+ 问题覆盖科学发现全流程，配备生产级可复现搜索工具，评估了 57 个 Agent（22 类），发现尽管单任务有进展但 AI 距离完整科学研究助手仍很远，同时系统性修复先前基准的 5 大方法学缺陷。

**[Behavior Learning (BL): Learning Hierarchical Optimization Structures from Data](behavior_learning_bl_learning_hierarchical_optimization_structures_from_data.md)**

:   受行为科学中效用最大化范式启发，提出 Behavior Learning (BL) 框架，将数据建模为由可解释的模块化效用最大化问题（UMP）层次组合所诱导的 Gibbs 分布，在预测性能、内在可解释性和参数可辨识性三者之间实现了统一。

**[Block-Sample MAC-Bayes Generalization Bounds](block-sample_mac-bayes_generalization_bounds.md)**

:   提出块样本MAC-Bayes泛化界（mean approximately correct），将训练数据划分为J个块后用各块条件下的KL散度之和替代整体KL散度，在确定性学习算法（如均值估计）等原始PAC-Bayes界为空（vacuous）的场景下仍能给出有限、有意义的泛化误差界，并证明了该界的高概率版本在一般情况下不可行。

**[Can You Hear Me Now? A Benchmark for Long-Range Graph Propagation and Beyond](can_you_hear_me_now_a_benchmark_for_long-range_graph_propagation_and_beyond.md)**

:   本文提出 ECHO 基准，包含 3 个合成任务和 2 个基于密度泛函理论（DFT）的真实化学任务，要求图神经网络在 17–40 跳范围内有效传播信息，系统评估了 11 种 GNN 架构的长程传播能力。

**[CHLU: The Causal Hamiltonian Learning Unit as a Symplectic Primitive for Deep Learning](chlu_the_causal_hamiltonian_learning_unit_as_a_symplectic_primitive_for_deep_lea.md)**

:   CHLU 是一种基于相对论哈密顿力学和辛积分的计算学习原语，通过强制相空间体积守恒和引入因果速度上限，解决了 LSTM 的梯度爆炸/消失和 Neural ODE 的信息耗散问题，实现无限时域稳定性和热力学生成能力。

**[DREAM: Completing Missing Annotation via Multi-Agent Debate for Accurate and Scalable Relevance Assessment](completing_missing_annotation_multi-agent_debate_for_accurate_and_scalable_relev.md)**

:   提出 DREAM 框架，用多轮 LLM 辩论（对立立场）来完成 IR 基准中大量缺失的相关性标注，发现了原有标注中 428% 的额外相关文档，仅需 3.5% 的人类介入率即达到 95.2% 准确率。

**[Compositional Diffusion with Guided Search for Long-Horizon Planning](compositional_diffusion_long_horizon_planning.md)**

:   提出 CDGS（Compositional Diffusion with Guided Search），通过在扩散去噪过程中嵌入基于种群的搜索机制（迭代重采样 + 似然剪枝），解决组合式扩散模型在多模态局部分布合成时的模式平均问题，从短时域模型采样出全局一致的长时域规划。

**[Directional Sheaf Hypergraph Networks: Unifying Learning on Directed and Undirected Hypergraphs](directional_sheaf_hypergraph_networks_unifying_learning_on_directed_and_undirect.md)**

:   本文提出 Directional Sheaf Hypergraph Networks (DSHN)，通过将 Cellular Sheaf 理论与有向超图的方向信息结合，构造了一种复值 Hermitian Laplacian 算子，统一并推广了现有的图和超图 Laplacian，在 7 个真实数据集上相对准确率提升 2%–20%。

**[DA-AC: Distributions as Actions — A Unified RL Framework for Diverse Action Spaces](distributions_as_actions_a_unified_framework_for_diverse_action_spaces.md)**

:   DA-AC 提出将动作分布的参数（如 softmax 概率或 Gaussian 均值/方差）作为 Agent 的"动作"输出，将动作采样过程移入环境，从而用统一的确定性策略梯度框架处理离散/连续/混合动作空间，理论证明方差严格低于 LR 和 RP 估计器，并在 40+ 环境上取得 competitive 或 SOTA 性能。

**[Enhancing Generative Auto-bidding with Offline Reward Evaluation and Policy Search](enhancing_generative_auto_bidding.md)**

:   提出 AIGB-Pearl，为生成式自动竞价方法引入离线轨迹评估器和 KL-Lipschitz 约束的分数最大化方案，使生成模型能在理论保证下安全地突破静态离线数据的性能天花板，在淘宝真实广告系统上实现 GMV +3% 的显著提升。

**[Entropic Confinement and Mode Connectivity in Overparameterized Neural Networks](entropic_confinement_and_mode_connectivity_in_overparameterized_neural_networks.md)**

:   揭示了过参数化神经网络损失景观中低损失连接路径上的**曲率变化**产生熵力壁垒，解释了为何SGD被限制在单一盆地内，尽管不同极小值之间能量上相连。

**[Exchangeability of GNN Representations with Applications to Graph Retrieval](exchangeability_gnn_representations.md)**

:   发现训练好的 GNN 节点嵌入沿特征维度是**可交换随机变量**（即 $p(X) = p(X\pi)$ 对任意维度排列 $\pi$），利用此性质通过维度排序将基于传输距离的图相似度近似为欧氏距离，构建高效的局部敏感哈希（LSH）框架 GraphHash，在子图匹配和图编辑距离检索任务上超越基线，可扩展到 100 万图语料库。

**[FIRE: Frobenius-Isometry Reinitialization for Balancing the Stability-Plasticity Tradeoff](fire_frobenius_isometry_reinitialization.md)**

:   将持续学习中的稳定性-可塑性平衡形式化为约束优化问题——最小化权重偏差（稳定性）同时约束权重正交性（可塑性），得到正交 Procrustes 问题的闭式解 $\tilde{W}^* = W(W^\top W)^{-1/2}$（极分解），通过 Newton-Schulz 迭代高效实现（<1% 额外时间），在视觉持续学习、LLM 持续预训练和 RL 上全面超越 S&P 等基线。

**[From Movement to Cognitive Maps: RNNs Reveal How Locomotor Development Shapes Hippocampal Spatial Coding](from_movement_to_cognitive_maps.md)**

:   结合幼鼠运动发育的计算分析和浅层 RNN 模型，证明运动统计特征的发育变化（爬行→行走→奔跑→成年）驱动了空间调谐神经元的序贯涌现，复现了大鼠海马空间编码的发育时间线，且具体的发育运动统计（而非简单的感觉输入加速）是位置中心空间表征涌现的关键。

**[Gaussian Certified Unlearning in High Dimensions: A Hypothesis Testing Approach](gaussian_certified_unlearning.md)**

:   提出 $(\phi,\varepsilon)$-Gaussian certifiability——基于假设检验 trade-off 函数的高维机器遗忘隐私框架，严格证明在高维比例体系 ($p \sim n$) 下单步 Newton 更新 + 校准高斯噪声即可同时满足隐私 (GPAR) 和精度 (GED→0) 要求，推翻了 Zou et al. (2025) "至少需两步 Newton" 的结论，并从理论上揭示旧 $\varepsilon$-certifiability 与噪声添加机制不兼容的根本原因。

**[In-Context Algebra](in-context_algebra.md)**

:   本文设计了一个 **in-context 代数任务**——令 token 成为纯变量、每条序列重新随机分配含义——发现 Transformer 在此设定下不再学习经典的傅里叶/几何表示，而是涌现出三种 **符号推理机制**（交换复制、单位元识别、闭包消去），并揭示了训练过程中这些能力按阶段性相变依次出现的规律。

**[Intrinsic Training Dynamics of Deep Neural Networks](intrinsic_training_dynamics_of_deep_neural_networks.md)**

:   本文研究深度神经网络梯度流训练中，参数空间的轨迹何时可以被"提升"到低维本征空间并表示为内禀的黎曼梯度流，提出了基于守恒律的内禀可恢复性（intrinsic recoverability）准则，并将结果推广到任意深度的 ReLU 网络和线性网络。

**[Jackpot: Optimal Budgeted Rejection Sampling for Extreme Actor-Policy Mismatch RL](jackpot_optimal_budgeted_rejection_sampling_for_extreme_actor-policy_mismatch_re.md)**

:   提出 Jackpot 框架，通过 Optimal Budget Rejection Sampling（OBRS）以可控接受预算在 token 级别拒绝/重加权 rollout 样本，理论证明任意预算下都能严格缩小 actor-policy 间 KL 散度，配合 rollout 模型联合训练与蒸馏，使小模型（如 Qwen3-1.7B）rollout 训练大模型（如 Qwen3-8B）达到接近 on-policy 的性能。

**[Latent Fourier Transform](latent_fourier_transform.md)**

:   将扩散自编码器与潜在空间 DFT 结合，在潜在时间序列表征上应用傅里叶变换按时间尺度分离音乐模式，训练时使用随机相关对数频率掩码让解码器学习从部分频谱信息重建，推理时用户指定频率掩码控制保留/混合的时间尺度，在条件生成和音乐融合任务上超越 ILVR/guidance/codec filtering/RAVE 等基线，29 名音乐家的听力测试确认其音质和融合能力优越。

**[Learning Adaptive Distribution Alignment with Neural Characteristic Function for Graph Domain Adaptation](learning_adaptive_distribution_alignment_with_neural_characteristic_function_for.md)**

:   提出**ADAlign**框架，利用**神经谱差异(NSD)**在频域自适应对齐源/目标图嵌入分布，通过可学习频率采样器自动发现并优先对齐每个迁移场景中最关键的分布差异维度。

**[LipNeXt: Scaling up Lipschitz-based Certified Robustness to Billion-parameter Models](lipnext_scaling_up_lipschitz-based_certified_robustness_to_billion-parameter_mod.md)**

:   提出 LipNeXt——首个无约束、无卷积的 1-Lipschitz 架构，通过流形优化（直接在正交流形上更新）和 Spatial Shift Module（理论证明唯一保范 depthwise 卷积是 ±1 位移）突破 Lipschitz 网络的 scaling 瓶颈，首次将认证鲁棒性扩展到 10 亿参数，在 CIFAR-10/100/ImageNet 上达 SOTA 认证鲁棒准确率。

**[Mitigating Spurious Correlation via Distributionally Robust Learning with Hierarchical Ambiguity Sets](mitigating_spurious_correlation_via_distributionally_robust_learning_with_hierar.md)**

:   提出层次化歧义集的分布鲁棒优化方法，同时建模组间比例变化和组内分布偏移（Wasserstein 球），在少数群组内部分布偏移场景下显著优于 Group DRO（CelebA shifted: 56.3%→72.1% worst-group accuracy）。

**[Modal Logical Neural Networks for Financial AI](modal_logical_neural_networks_for_financial_ai.md)**

:   提出模态逻辑神经网络（MLNN），将 Kripke 语义（必然/可能模态算子）集成到神经网络中，在金融合同安全审查、洗售合规和市场串谋检测中实现可审计的逻辑推理与深度学习性能的结合。

**[MT-DAO: Multi-Timescale Distributed Adaptive Optimizers with Local Updates](mt-dao_multi-timescale_distributed_adaptive_optimizers_with_local_updates.md)**

:   提出 MT-DAO，一种多时间尺度分布式自适应优化器，通过引入慢动量（高 $\beta$）来解决低频通信训练中标准动量衰减过快导致的时间尺度失配问题，首次提供了收敛保证，在语言模型预训练中消除了与全同步 DDP 的性能差距，同时减少 6-27% 的端到端训练时间。

**[NIMO: a Nonlinear Interpretable MOdel](nimo_a_nonlinear_interpretable_model.md)**

:   NIMO 提出一种混合模型 $y = \sum_j x_j \beta_j (1 + g_{\mathbf{u}_j}(\mathbf{x}_{-j}))$，在保留线性回归系数全局可解释性（通过均值边际效应 MEM）的同时，利用神经网络提供逐实例的非线性修正，并通过参数消去法高效联合优化线性系数和网络参数。

**[Noisy-Pair Robust Representation Alignment for Positive-Unlabeled Learning](noisy-pair_robust_representation_alignment_for_positive-unlabeled_learning.md)**

:   提出**NcPU**框架，通过**噪声对鲁棒的非对比损失(NoiSNCL)** 在不可靠监督下对齐类内表示，配合**幻影标签消歧(PLD)** 迭代优化伪标签，无需辅助负样本或预估类先验即可在PU学习中逼近甚至超越有监督性能。

**[Non-Collaborative User Simulators for Tool Agents](non-collaborative_user_simulators_for_tool_agents.md)**

:   提出非协作用户模拟器框架，定义四类真实非协作行为（不可用服务/跑题/不耐烦/不完整表述），揭示当前工具 Agent 面对非协作用户时显著退化（跑题平均降 29.1%），并证明混合训练可提升鲁棒性至 93.5%。

**[Speculative Actions: A Lossless Framework for Faster AI Agents](speculative_actions_faster_ai_agents.md)**

:   借鉴 CPU 推测执行和 LLM 推测解码的思想，提出 Speculative Actions 框架：在慢速 Actor（大模型）计算时用快速 Speculator（小模型）预测未来动作并预执行，匹配时跳过等待实现无损加速，在 Chess/电商/问答等场景实现 15-30% 延迟降低，置信度动态分支策略用 40% 更少 token 达到近似 3 条推测的加速效果。

**[The Hot Mess of AI: How Does Misalignment Scale With Model Intelligence and Task Complexity?](the_hot_mess_of_ai_how_does_misalignment_scale_with_model_intelligence_and_task_.md)**

:   通过 bias-variance 分解量化 AI 模型的错误模式，发现随着推理链变长和任务变难，模型失败变得更加"不连贯"（variance 主导）而非"系统性错误"（bias 主导），暗示未来 AI 风险更像工业事故而非一致性目标追求。
