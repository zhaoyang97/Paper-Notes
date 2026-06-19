---
title: >-
  ICLR2026 因果推理论文汇总 · 25篇论文解读
description: >-
  25篇ICLR2026的因果推理方向论文解读，涵盖对抗鲁棒、推理、多模态、布局/合成、RAG、自监督学习等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICLR2026"
  - "因果推理"
  - "论文解读"
  - "论文笔记"
  - "对抗鲁棒"
  - "推理"
  - "多模态"
  - "布局/合成"
  - "RAG"
  - "自监督学习"
item_list:
  - u: "action-guided_attention_for_video_action_anticipation/"
    t: "Action-Guided Attention for Video Action Anticipation"
  - u: "activecq_active_estimation_of_causal_quantities/"
    t: "ActiveCQ: Active Estimation of Causal Quantities"
  - u: "adjusting_prediction_model_through_wasserstein_geodesic_for_causal_inference/"
    t: "Adjusting Prediction Model Through Wasserstein Geodesic for Causal Inference"
  - u: "alm-mta_front-door_causal_multi-touch_attribution_method_for_creator-ecosystem_o/"
    t: "ALM-MTA: Front-Door Causal Multi-Touch Attribution Method for Creator-Ecosystem Optimization"
  - u: "an_orthogonal_learner_for_individualized_outcomes_in_markov_decision_processes/"
    t: "An Orthogonal Learner for Individualized Outcomes in Markov Decision Processes"
  - u: "beyond_dags_a_latent_partial_causal_model_for_multimodal_learning/"
    t: "Beyond DAGs: A Latent Partial Causal Model for Multimodal Learning"
  - u: "causal_discovery_in_the_wild_a_voting-theoretic_ensemble_approach/"
    t: "Causal Discovery in the Wild: A Voting-Theoretic Ensemble Approach"
  - u: "causal_imitation_learning_under_expert-observable_and_expert-unobservable_confou/"
    t: "Causal Imitation Learning under Expert-Observable and Expert-Unobservable Confounding"
  - u: "characterization_and_learning_of_causal_graphs_with_latent_confounders_and_post-/"
    t: "Characterization and Learning of Causal Graphs with Latent Confounders and Post-treatment Selection from Interventional Data"
  - u: "coarse-to-fine_learning_of_dynamic_causal_structures/"
    t: "Coarse-to-Fine Learning of Dynamic Causal Structures"
  - u: "counterfactual_explanations_on_robust_perceptual_geodesics/"
    t: "Counterfactual Explanations on Robust Perceptual Geodesics"
  - u: "direct_doubly_robust_estimation_of_conditional_quantile_contrasts/"
    t: "Direct Doubly Robust Estimation of Conditional Quantile Contrasts"
  - u: "distributional_equivalence_in_linear_non-gaussian_latent-variable_cyclic_causal_/"
    t: "Distributional Equivalence in Linear Non-Gaussian Latent-Variable Cyclic Causal Models"
  - u: "efficient_ensemble_conditional_independence_test_framework_for_causal_discovery/"
    t: "Efficient Ensemble Conditional Independence Test Framework for Causal Discovery"
  - u: "flattery_fluff_and_fog_diagnosing_and_mitigating_idiosyncratic_biases_in_prefere/"
    t: "Flattery, Fluff, and Fog: Diagnosing and Mitigating Idiosyncratic Biases in Preference Models"
  - u: "function_induction_and_task_generalization_an_interpretability_study_with_off-by/"
    t: "Function Induction and Task Generalization: An Interpretability Study with Off-by-One Addition"
  - u: "journey_to_the_centre_of_cluster_harnessing_interior_nodes_for_ab_testing_under_/"
    t: "Journey to the Centre of Cluster: Harnessing Interior Nodes for A/B Testing under Network Interference"
  - u: "learning_robust_intervention_representations_with_delta_embeddings/"
    t: "Learning Robust Intervention Representations with Delta Embeddings"
  - u: "on_the_eligibility_of_llms_for_counterfactual_reasoning_a_decompositional_study/"
    t: "On the Eligibility of LLMs for Counterfactual Reasoning: A Decompositional Study"
  - u: "resisting_contextual_interference_in_rag_via_parametric-knowledge_reinforcement/"
    t: "Resisting Contextual Interference in RAG via Parametric-Knowledge Reinforcement"
  - u: "rfeval_benchmarking_reasoning_faithfulness_under_counterfactual_perturbations/"
    t: "RFEval: Benchmarking Reasoning Faithfulness under Counterfactual Perturbations"
  - u: "self-supervised_learning_from_structural_invariance/"
    t: "Self-Supervised Learning from Structural Invariance"
  - u: "selfreflect_can_llms_communicate_their_internal_answer_distribution/"
    t: "SelfReflect: Can LLMs Communicate Their Internal Answer Distribution?"
  - u: "synthesising_counterfactual_explanations_via_label-conditional_gaussian_mixture_/"
    t: "Synthesising Counterfactual Explanations via Label-Conditional Gaussian Mixture Variational Autoencoders"
  - u: "validating_interpretability_in_sirna_efficacy_prediction_a_perturbation-based_da/"
    t: "Validating Interpretability in siRNA Efficacy Prediction: A Perturbation-Based, Dataset-Aware Protocol"
item_total: 25
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔗 因果推理

**🔬 ICLR2026** · **25** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (4)](../../CVPR2026/causal_inference/index.md) · [🧪 ICML2026 (19)](../../ICML2026/causal_inference/index.md) · [💬 ACL2026 (7)](../../ACL2026/causal_inference/index.md) · [🤖 AAAI2026 (7)](../../AAAI2026/causal_inference/index.md) · [🧠 NeurIPS2025 (20)](../../NeurIPS2025/causal_inference/index.md) · [📹 ICCV2025 (2)](../../ICCV2025/causal_inference/index.md)

🔥 **高频主题：** 对抗鲁棒 ×3 · 推理 ×2

**[Action-Guided Attention for Video Action Anticipation](action-guided_attention_for_video_action_anticipation.md)**

:   提出动作引导注意力 (AGA) 机制，用模型自身的动作预测序列作为注意力的 Query 和 Key（而非像素特征），结合自适应门控融合历史上下文和当前帧特征，在 EPIC-Kitchens-100 上实现从验证集到测试集的良好泛化，同时支持训练后的可解释性分析。

**[ActiveCQ: Active Estimation of Causal Quantities](activecq_active_estimation_of_causal_quantities.md)**

:   ActiveCQ 把"用尽量少的标注样本估准某个因果量（CATE/ATE/ATT/分布漂移下的 ATE）"这件事统一成一个主动学习问题：发现绝大多数因果量都可以写成"回归函数对某个分布求积分"的形式，于是用高斯过程（GP）建模回归函数、用 RKHS 里的条件均值嵌入（CME）建模那个积分用的分布，再从因果量后验不确定性里解析地推出采集函数（信息增益 / 全方差缩减），在多个模拟与半合成数据集上以更少标注显著超过随机、BALD、Coreset 等基线。

**[Adjusting Prediction Model Through Wasserstein Geodesic for Causal Inference](adjusting_prediction_model_through_wasserstein_geodesic_for_causal_inference.md)**

:   针对因果推断里"实验组和对照组分布失衡导致预测模型无法跨组泛化"的问题，本文提出 G-learner：不再像主流方法那样去对齐协变量（会丢掉预测信息、产生 over-balancing），而是沿着两组分布之间的 Wasserstein 测地线生成一串中间群，再用渐进自训练把预测模型从一组逐步搬到另一组，在 News/Twins/Jobs 和仿真数据上把 PEHE/ATE 误差压到 SOTA 或与之持平。

**[ALM-MTA: Front-Door Causal Multi-Touch Attribution Method for Creator-Ecosystem Optimization](alm-mta_front-door_causal_multi-touch_attribution_method_for_creator-ecosystem_o.md)**

:   针对短视频平台「消费驱动创作」场景中无真值标签、又存在系统级隐混淆的归因难题，本文用**前门准则 + 对抗式学习的代理中介**把每个消费触点对「用户是否上传」的因果 uplift 识别出来，并用对比学习保证大动作空间下的 overlap，在快手 4 亿 DAU 真实系统上把上传 AUC 提到 0.907（相对 SOTA +40%）、单位曝光效率提升 670%。

**[An Orthogonal Learner for Individualized Outcomes in Markov Decision Processes](an_orthogonal_learner_for_individualized_outcomes_in_markov_decision_processes.md)**

:   将因果推断中的半参数效率理论系统引入MDP的Q函数估计，证明经典的Q-regression和FQE本质上是有plug-in偏差的朴素学习器，并提出DRQQ-learner——一个同时具备双重鲁棒性、Neyman正交性和准oracle效率的元学习器，通过推导有效影响函数(EIF)构造去偏二阶段损失，在Taxi和Frozen Lake环境中全面超越基线方法。

**[Beyond DAGs: A Latent Partial Causal Model for Multimodal Learning](beyond_dags_a_latent_partial_causal_model_for_multimodal_learning.md)**

:   本文指出大规模多模态数据并不服从单一有向无环图（DAG）的生成假设，提出一个用"无向边连接两组潜在耦合变量"的潜在偏因果模型，并在球面和凸体两种潜在空间上证明：CLIP 这类多模态对比学习（MMCL）学到的表示与真实潜变量分别相差一个线性正交变换 / 置换变换，从而第一次给出 MMCL 的"逐分量解耦"理论保证，并把它落到 FastICA / PCA+FastICA 这种即插即用的解耦流程上，在少样本学习和域泛化上拿到提升。

**[Causal Discovery in the Wild: A Voting-Theoretic Ensemble Approach](causal_discovery_in_the_wild_a_voting-theoretic_ensemble_approach.md)**

:   把若干个因果发现算法当成"会犯错的投票专家"，用投票理论给结构集成建立一套有理论保证的加权贝叶斯投票框架——通过把图拆成边级子结构、再用最优传输估计每个专家的"能力矩阵"，最终在合成与真实数据上比现有启发式集成方法更稳更准，并给出了集成规模/能力/多样性该怎么选的明确指导。

**[Causal Imitation Learning under Expert-Observable and Expert-Unobservable Confounding](causal_imitation_learning_under_expert-observable_and_expert-unobservable_confou.md)**

:   本文提出一个统一的因果模仿学习框架，把"专家能看到但模仿者看不到"和"专家与模仿者都看不到"两类隐混杂同时建模，用 $k$ 步轨迹历史当工具变量把问题改写成条件矩约束（CMR）问题，并给出带模仿差距上界保证的 DML-IL 算法，在含混杂的 MuJoCo 等连续控制任务上超过现有因果 IL 基线。

**[Characterization and Learning of Causal Graphs with Latent Confounders and Post-treatment Selection from Interventional Data](characterization_and_learning_of_causal_graphs_with_latent_confounders_and_post-.md)**

:   本文指出干预因果发现中一个长期被忽视的难题——**后处理选择**（intervention 后才按质控标准筛样本，如单细胞实验只保留高活性细胞），它会伪装成因果响应使现有方法把"有无直接因果边"误判为同一等价类；作者用增广 DAG 显式建模选择变量，提出比传统等价类更细的 **FI-Markov 等价**与新图表示 **F-PAG**，并给出可证明 sound & complete 的 **F-FCI** 算法，能从观测+干预数据中同时辨认因果关系、潜在混杂与后处理选择。

**[Coarse-to-Fine Learning of Dynamic Causal Structures](coarse-to-fine_learning_of_dynamic_causal_structures.md)**

:   本文提出 **DyCausal**，用滑动卷积窗口先捕捉时间序列在「粗粒度」若干时间步上的因果结构，再用一阶 Taylor 线性插值把因果矩阵细化到每个时间步，并配上一个基于矩阵 1-范数缩放的「永远可微」无环约束 $h_\text{norm}$，从而第一次稳定高效地恢复出**全动态**（瞬时与滞后因果都随时间变化）的时变因果图，在合成与真实数据上全面超过现有方法。

**[Counterfactual Explanations on Robust Perceptual Geodesics](counterfactual_explanations_on_robust_perceptual_geodesics.md)**

:   提出 PCG（Perceptual Counterfactual Geodesic）方法，在鲁棒感知流形上通过测地线优化生成语义忠实的反事实解释，两阶段优化确保路径既感知自然又达到目标类别，在 AFHQ 上 FID=8.3 远优于 RSGD 的 12.9。

**[Direct Doubly Robust Estimation of Conditional Quantile Contrasts](direct_doubly_robust_estimation_of_conditional_quantile_contrasts.md)**

:   提出首个对条件分位数比较器 (CQC) 的**直接估计方法**，通过显式参数化 CQC 并结合双重鲁棒梯度下降，在理论上保持双重鲁棒性的同时，实验中在估计精度、可解释性和计算效率上全面优于现有的间接反演方法。

**[Distributional Equivalence in Linear Non-Gaussian Latent-Variable Cyclic Causal Models](distributional_equivalence_in_linear_non-gaussian_latent-variable_cyclic_causal_.md)**

:   首次在线性非高斯设定下、不依赖任何结构假设，给出了含潜变量和环的因果图之间分布等价性的完整图准则，核心工具是新提出的**边秩约束**（edge rank constraints），据此开发了遍历等价类和从数据恢复因果模型的算法——这是参数化因果模型中首个无结构假设的等价性刻画和发现方法。

**[Efficient Ensemble Conditional Independence Test Framework for Causal Discovery](efficient_ensemble_conditional_independence_test_framework_for_causal_discovery.md)**

:   提出 E-CIT（集成条件独立性检验）框架，通过将数据分割为子集后独立执行检验并基于**稳定分布**的 p 值聚合方法合并结果，将任意条件独立性检验的计算复杂度降至关于样本量线性，同时在重尾噪声和真实数据等复杂场景下保持甚至提升检验功效。

**[Flattery, Fluff, and Fog: Diagnosing and Mitigating Idiosyncratic Biases in Preference Models](flattery_fluff_and_fog_diagnosing_and_mitigating_idiosyncratic_biases_in_prefere.md)**

:   系统研究偏好模型对五种表面特征（冗长、结构化、术语、谄媚、模糊）的过度依赖——通过因果反事实对量化偏差来源于训练数据的分布不平衡，并提出基于**反事实数据增强 (CDA)** 的后训练方法，将模型与人类判断的平均失校准率从 39.4% 降至 32.5%。

**[Function Induction and Task Generalization: An Interpretability Study with Off-by-One Addition](function_induction_and_task_generalization_an_interpretability_study_with_off-by.md)**

:   通过 off-by-one addition（如 1+1=3, 2+2=5）这一反事实任务，利用 path patching 发现大语言模型内部存在 **function induction** 机制——一种超越 token 级别 pattern matching、在函数级别进行归纳推理的注意力头电路，并证明该机制可跨任务复用。

**[Journey to the Centre of Cluster: Harnessing Interior Nodes for A/B Testing under Network Interference](journey_to_the_centre_of_cluster_harnessing_interior_nodes_for_ab_testing_under_.md)**

:   针对网络干扰下 A/B 测试中 GATE 估计的高方差问题，提出 Mean-in-Interior (MII) 估计器——仅对 cluster 内部节点取均值，大幅降低方差；再通过反事实预测器进行协变量偏移校正，得到增广版 AMII 估计器，同时实现低偏差和低方差。

**[Learning Robust Intervention Representations with Delta Embeddings](learning_robust_intervention_representations_with_delta_embeddings.md)**

:   提出因果 Delta 嵌入（CDE）框架，将干预/动作表示为预干预和后干预状态在潜空间中的向量差，通过独立性、稀疏性和不变性三种约束学习鲁棒的干预表示，在 Causal Triplet 挑战中显著超越基线的 OOD 泛化性能，且能自动发现反义动作的反平行语义结构。

**[On the Eligibility of LLMs for Counterfactual Reasoning: A Decompositional Study](on_the_eligibility_of_llms_for_counterfactual_reasoning_a_decompositional_study.md)**

:   提出基于结构因果模型（SCM）的分解式评估框架，将 LLM 的反事实推理拆分为四个阶段（因果变量识别→因果图构建→干预识别→结果推理），在 11 个多模态数据集上系统诊断 LLM 在各阶段的能力瓶颈，并提出工具增强和高级 elicitation 策略来改善性能。

**[Resisting Contextual Interference in RAG via Parametric-Knowledge Reinforcement](resisting_contextual_interference_in_rag_via_parametric-knowledge_reinforcement.md)**

:   提出 Knowledgeable-R1，一个基于强化学习的框架，通过联合采样参数知识（PK）和上下文知识（CK）的轨迹，结合局部/全局优势计算和自适应不对称优势变换，使 LLM 在 RAG 场景中能够抵抗误导性检索上下文的干扰，同时保留对可靠上下文的利用能力。

**[RFEval: Benchmarking Reasoning Faithfulness under Counterfactual Perturbations](rfeval_benchmarking_reasoning_faithfulness_under_counterfactual_perturbations.md)**

:   本文提出推理忠实性的形式化框架（立场一致性 + 因果影响）和 RFEval 基准（7,186 实例 × 7 任务），通过输出层反事实干预评估 12 个开源 LRM，发现 49.7% 的输出不忠实，且准确率不是忠实性的可靠代理指标。

**[Self-Supervised Learning from Structural Invariance](self-supervised_learning_from_structural_invariance.md)**

:   提出 AdaSSL，通过引入潜变量建模正样本对之间的条件不确定性，推导出互信息的变分下界，使 SSL 能够处理自然配对数据中的复杂（多模态、异方差）条件分布，在因果表征学习、细粒度图像理解和视频世界模型上均优于基线。

**[SelfReflect: Can LLMs Communicate Their Internal Answer Distribution?](selfreflect_can_llms_communicate_their_internal_answer_distribution.md)**

:   提出SelfReflect度量指标——一个衡量LLM自述不确定性摘要与其真实内部答案分布之间差异的信息论距离，发现现代LLM普遍无法自主反映内部不确定性，但通过采样多个输出并反馈到上下文中可以生成忠实的不确定性摘要。

**[Synthesising Counterfactual Explanations via Label-Conditional Gaussian Mixture Variational Autoencoders](synthesising_counterfactual_explanations_via_label-conditional_gaussian_mixture_.md)**

:   提出 L-GMVAE（标签条件高斯混合 VAE）和 LAPACE 算法，通过在潜空间中学习每个类别的多个高斯聚类中心，然后从输入潜表征到目标类别中心进行线性插值，生成路径式反事实解释，同时保证有效性、似合性、多样性和对输入扰动的完美鲁棒性。

**[Validating Interpretability in siRNA Efficacy Prediction: A Perturbation-Based, Dataset-Aware Protocol](validating_interpretability_in_sirna_efficacy_prediction_a_perturbation-based_da.md)**

:   提出一个标准化的扰动式显著性忠实性验证协议用于 siRNA 效能预测，作为"合成前关卡"检验显著性图是否可信；同时提出 BioPrior 生物信息正则化提升解释忠实性，发现 19/20 折instances 通过验证，但跨数据集迁移暴露两种失败模式。
