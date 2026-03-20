<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📂 其他

**🧠 NeurIPS2025** · 共 **36** 篇

**[4DGT: Learning a 4D Gaussian Transformer Using Real-World Monocular Videos](4dgt_learning_a_4d_gaussian_transformer_using_realworld_mono.md)**

:   提出4DGT——一种基于4D高斯的Transformer模型，完全在真实世界单目带位姿视频上训练，以前馈方式在几秒内完成动态场景重建，显著优于同类前馈网络，并达到与优化类方法可比的精度。

**[A Cramér–von Mises Approach to Incentivizing Truthful Data Sharing](a_cramrvon_mises_approach_to_incentivizing_truthful_data_sha.md)**

:   提出一种基于 Cramér-von Mises 两样本检验统计量的激励机制，在贝叶斯和无先验两种设定下均能证明"如实提交数据"构成（近似）Nash 均衡，同时鼓励参与者提交更多真实数据，且不依赖对数据分布的强假设（如高斯、伯努利）。

**[A Generalized Label Shift Perspective for Cross-Domain Gaze Estimation](a_generalized_label_shift_perspective_for_crossdomain_gaze_e.md)**

:   本文将跨域视线估计(CDGE)问题建模为广义标签偏移(GLS)问题，指出现有域不变表示学习方法在标签偏移存在时理论上不充分，提出基于截断高斯分布的连续重要性重加权和概率感知条件算子差异(PCOD)来联合纠正标签偏移和条件偏移，在多个backbone上平均降低误差12%~27%。

**[A High-Dimensional Statistical Method for Optimizing Transfer Quantities in Multi-Source Transfer Learning](a_highdimensional_statistical_method_for_optimizing_transfer.md)**

:   提出基于K-L散度和高维统计分析的理论框架，用于确定多源迁移学习中每个源任务的最优样本迁移数量，避免"用所有源数据"带来的负迁移问题，在DomainNet和Office-Home上超过SOTA 1.0-1.5%的同时减少47.85%的样本使用量和35.19%的训练时间。

**[A Reliable Cryptographic Framework for Empirical Machine Unlearning Evaluation](a_reliable_cryptographic_framework_for_empirical_machine_unl.md)**

:   将机器遗忘的评估问题建模为密码学博弈（unlearning sample inference game），通过定义adversary的"advantage"来衡量遗忘质量，克服了传统MIA准确率作为评估指标的多种缺陷（不以retrain为零基准、对数据划分敏感、对MIA选择敏感），并提出SWAP test作为高效的实用近似方案。

**[A Standardized Benchmark for Multilabel Antimicrobial Peptide Classification](a_standardized_benchmark_for_multilabel_antimicrobial_peptide_classification.md)**

:   提出 **ESCAPE**——首个标准化的多标签抗菌肽分类基准，整合 27 个公开数据库共 80,000+ 肽段，并设计基于双分支 Transformer + 双向交叉注意力的 Baseline 模型，在 mAP 上相对第二名提升 2.56%。

**[A Sustainable AI Economy Needs Data Deals That Work for Generators](a_sustainable_ai_economy_needs_data_deals_that_work_for_gene.md)**

:   本文通过分析73个公开数据交易案例，揭示了ML价值链中的"经济数据处理不等式"——从原始数据到模型权重再到合成输出，每一步都提炼了技术信号但剥夺了数据生成者的经济权益，并提出EDVEX框架来构建更公平的数据交换市场。

**[A Theoretical Framework for Grokking: Interpolation followed by Riemannian Norm Minimisation](a_theoretical_framework_for_grokking_interpolation_followed_by_riemannian_norm_m.md)**

:   本文从纯优化角度严格证明了 grokking 现象的成因：带小 weight decay 的梯度流在 $\lambda\to 0$ 极限下呈现两阶段动力学——先快速收敛到训练损失的临界流形 $\mathcal{M}$，再在 $t\approx 1/\lambda$ 时沿流形做黎曼梯度流以最小化 $\ell_2$ 范数，从而延迟实现泛化。

**[A Unified Framework for Provably Efficient Algorithms to Estimate Shapley Values](a_unified_framework_for_provably_efficient_algorithms_to_estimate_shapley_values.md)**

:   提出统一框架将 KernelSHAP、LeverageSHAP 等 Shapley 值估计器纳入随机草图（sketching）视角，首次为 KernelSHAP 提供非渐近理论保证，并通过算法改进（Poisson 近似等）将方法扩展到 CIFAR-10 等高维数据集。

**[A Unified Framework for Variable Selection in Model-Based Clustering with Missing Not at Random](a_unified_framework_for_variable_selection_in_modelbased_clu.md)**

:   提出了一个统一框架（SelvarMNARz），在高斯混合模型聚类中同时完成变量选择和MNAR（Missing Not At Random）缺失数据建模，通过两阶段策略（LASSO排序 + BIC角色分配）实现高维场景下的高效推理，并给出了可辨识性和选择一致性的理论保证。

**[Active Measurement: Efficient Estimation at Scale](active_measurement_efficient_estimation_at_scale.md)**

:   提出Active Measurement框架，结合AI检测器的自适应重要性采样和迭代人工标注，实现大规模科学测量（如鸟类计数、疟疾检测）的无偏估计，将原始检测器3.78的误差率降至0.06，同时提供理论保证的置信区间。

**[AcuRank: 不确定性感知的自适应计算重排序](acurank_uncertainty-aware_adaptive_computation_for_listwise_reranking.md)**

:   通过基于TrueSkill模型的不确定性估计，动态调整重排序子集大小和验证范围，在实现更优精度效率权衡的同时避免过度计算。

**[AdaptGrad: Adaptive Sampling to Reduce Noise](adaptgrad_adaptive_sampling_to_reduce_noise.md)**

:   通过卷积公式视角首次理论分析了SmoothGrad的噪声来源（越界采样），提出AdaptGrad方法通过概率界约束采样范围来抑制噪声，在不增加计算开销的前提下提升梯度显著性图的质量。

**[Adaptive Data Analysis for Growing Data](adaptive_data_analysis_for_growing_data.md)**

:   首次为动态/增长数据场景下的自适应数据分析提供泛化界，允许分析者根据当前数据规模和历史查询结果自适应地调度统计查询，在数据不断积累时获得更紧的准确性保证。

**[Addressing Mark Imbalance In Integrationfree Neural Marked T](addressing_mark_imbalance_in_integrationfree_neural_marked_t.md)**

:   论文针对现实事件流中常见的 mark 类别长尾失衡问题，提出基于先验归一化概率的阈值学习策略，并设计 integration-free 的神经 MTPP 架构，先预测 mark 再预测 time，在避免昂贵数值积分的同时显著提升稀有事件的 mark 与到达时间预测性能。

**[Adjoint Schrödinger Bridge Sampler](adjoint_schrödinger_bridge_sampler.md)**

:   提出 Adjoint Schrödinger Bridge Sampler (ASBS)，通过将 Schrödinger Bridge 问题重新解释为随机最优控制问题，消除了先前扩散采样器的 memoryless 条件限制，支持任意源分布（如高斯、谐波先验），使用可扩展的 matching 目标无需重要性权重估计，在多粒子能量函数和分子构象生成上全面超越先前方法。

**[ADPretrain: Advancing Industrial Anomaly Detection via Anomaly Representation Pretraining](adpretrain_advancing_industrial_anomaly_detection_via_anomaly_representation_pre.md)**

:   首次提出面向工业异常检测的专用表示预训练框架 ADPretrain，通过角度和范数导向的对比损失在大规模异常检测数据集 RealIAD 上学习残差特征表示，替换五种主流嵌入式 AD 方法的原始特征后在五个数据集、五个骨干网络上取得一致性提升。

**[Are Pixel-Wise Metrics Reliable For Sparse-View Computed Tomography Reconstructi](are_pixel-wise_metrics_reliable_for_sparse-view_computed_tomography_reconstructi.md)**

:   揭示 PSNR/SSIM 等像素级指标无法反映稀疏视图 CT 重建中解剖结构完整性（相关性仅 0.16-0.30），提出基于自动分割的解剖感知指标（NSD/clDice）和 CARE 框架——在扩散模型训练中加入分割引导损失，大器官结构完整性提升 32%、血管提升 36%。

**[AutoSciDACT: Automated Scientific Discovery through Contrastive Embedding and Hypothesis Testing](autoscidact_automated_scientific_discovery_through_contrastive_embedding_and_hyp.md)**

:   提出 AutoSciDACT 管线：先用有监督对比学习将高维科学数据压缩到 4 维嵌入空间，再用 NPLM（New Physics Learning Machine）似然比检验对嵌入空间中的分布偏差进行统计量化，在天文、粒子物理、病理、图像和合成数据集上以 ≤1% 的信号注入比例实现 ≥3σ 发现。

**[Beyond the Singular: Value of Multiple Generations in Benchmark Evaluation](beyond_the_singular_revealing_the_value_of_multiple_generations_in_benchmark_eva.md)**

:   通过多次生成(k=50)建立层级模型，证明能显著降低基准估计方差，量化个体prompt难度，检测标签错误。

**[Depth-Supervised Fusion Network For Seamless-Free Image Stitching](depth-supervised_fusion_network_for_seamless-free_image_stitching.md)**

:   利用深度监督进行多视角对齐和软缝合融合，解决大视差场景下的图像拼接问题，在大视差对齐和无缝融合上超越现有SOTA。

**[FlowMoE: 分布式MoE训练的可扩展流水线调度框架](flowmoe_a_scalable_pipeline_scheduling_framework_for_distributed_mixture-of-expe.md)**

:   通过统一的流水线调度和优先级驱动的all-reduce张量分块，实现MHA、门控、专家计算和A2A/all-reduce通信的完全重叠，训练时间减少13-57%。

**[Generalized Linear Mode Connectivity For Transformers](generalized_linear_mode_connectivity_for_transformers.md)**

:   提出统一框架捕捉Transformer中的置换、半置换、正交和可逆对称性，首次在ViT和GPT-2上实现线性模式连接性(LMC)，支持多模型融合。

**[Gradient-Weight Alignment As A Train-Time Proxy For Generalization In Classifica](gradient-weight_alignment_as_a_train-time_proxy_for_generalization_in_classifica.md)**

:   提出梯度-权重对齐(GWA)作为训练时的泛化代理指标，无需验证集即可准确确定早停时机和模型比较。

**[Graph Alignment Via Birkhoff Relaxation](graph_alignment_via_birkhoff_relaxation.md)**

:   分析图对齐的二次分配问题的凸松弛（Birkhoff松弛），证明在 $\sigma = o(n^{-1})$ 时可恢复~100%的对齐，超越了先前的simplex松弛结果。

**[笔记2：PRM必要吗？RL隐式诱导PRM能力](is_prm_necessary_problem-solving_rl_implicitly_induces_prm_capability_in_llms.md)**

:   令人惊讶地，纯RL训练无需显式PRM监督即可诱发出强大的过程理解能力，且现有PRMs在SOTA模型上甚至不如简单多数投票有效。

**[L-MTP: Leap Multi-Token Prediction Beyond Adjacent Context](l-mtp_leap_multi-token_prediction_beyond_adjacent_context_for_large_language_mod.md)**

:   L-MTP通过跳跃式预测非相邻token，相比MTP在3B-12B模型上实现22%推理加速，同时保持或提升任务性能。

**[Learning Generalizable Shape Completion With Sim3 Equivariance](learning_generalizable_shape_completion_with_sim3_equivariance.md)**

:   首个SIM(3)等变网络用于3D形状补全，通过规范化特征、几何推理和变换恢复三阶段，KITTI提升17%、OmniObject3D提升14%。

**[Look-Ahead Reasoning On Learning Platforms](look-ahead_reasoning_on_learning_platforms.md)**

:   在多领导Stackelberg博弈中形式化level-k前瞻性推理和协调战略行为，识别了对齐的关键优化视野，并证明基于努力的信号可减轻用户的对齐负担。

**[MAS-ZERO: Designing Multi-Agent Systems with Zero Supervision](maszero_designing_multiagent_systems_with_zero_supervision.md)**

:   MAS-ZERO 是首个推理时自动 MAS 设计框架，通过 meta-agent 迭代设计、批评和改进 MAS 配置（包括任务分解和 sub-MAS 分配），无需验证集和训练，在推理（+16.69%）、编程（+16.66%）和搜索代理（+5.45%）任务上均超越手动和自动 MAS baseline，同时保持 Pareto 最优的准确率-成本权衡。

**[Model Context Protocol for Vision Systems: Audit, Security, and Protocol Extensions](model_context_protocol_for_vision_systems_audit_security_and_protocol_extensions.md)**

:   首次大规模审计 MCP 在 91 个视觉系统中的部署，揭示 78% 存在 schema 偏差、89% 缺乏运行时验证等协议级脆弱性，提出扩展方案增强编排可靠性。

**[MoESD: 揭示稀疏MoE推理中投机解码的潜力](moesd_unveil_speculative_decodings_potential_for_accelerating_sparse_moe.md)**

:   揭示投机解码在中等批大小下对MoE比对稠密模型更有效，通过目标效率指标捕捉系统级瓶颈，建立可靠的性能建模，达到2.29×加速。

**[笔记7：价值引导搜索 - 高效链式思考推理](polymath_evaluating_mathematical_reasoning_in_multilingual_contexts.md)**

:   提出Value-Guided Search(VGS)——通过token级价值模型指导块级束搜索，无需预定义"步骤"，相对多数投票在竞赛数学上准确度提升+14.5%，同时推理计算效率提升30%，超越现有PRM方案。

**[笔记5：ReSearch - 学习通过搜索推理](research_learning_to_reason_with_search_for_llms_via_reinforcement_learning.md)**

:   ReSearch框架将搜索操作嵌入推理链中作为第一类原语，通过GRPO强化学习自动学习何时何如搜索，无需任何推理步骤的监督标注，在多跳QA任务上相对基线平均提升15.81%。

**[Training The Untrainable Introducing Inductive Bias Via Representational Alignme](training_the_untrainable_introducing_inductive_bias_via_representational_alignme.md)**

:   通过层级CKA对齐将架构先验从引导模型转移到目标模型，使FCN停止过拟合、RNN-Transformer差距缩小、ResNet匹配基线性能。

**[笔记4：WebThinker - 赋予推理模型深度研究能力](webthinker_empowering_large_reasoning_models_with_deep_research_capability.md)**

:   WebThinker赋予大型推理模型(LRM)自主的网络搜索与导航能力，通过Think-Search-Draft策略实现推理、信息采集与报告生成的无缝交织，经RL优化后在复杂推理与科学报告生成任务上超越o1与Gemini。
