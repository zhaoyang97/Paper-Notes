---
title: >-
  ICML2025 因果推理论文汇总 · 17篇论文解读
description: >-
  17篇ICML2025的因果推理方向论文解读，涵盖医学影像、对抗鲁棒、推理等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2025"
  - "因果推理"
  - "论文解读"
  - "论文笔记"
  - "医学影像"
  - "对抗鲁棒"
  - "推理"
item_list:
  - u: "causal_abstraction_inference_under_lossy_representations/"
    t: "Causal Abstraction Inference under Lossy Representations"
  - u: "causal_effect_identification_in_lvlingam_from_higher-order_cumulants/"
    t: "Causal Effect Identification in lvLiNGAM from Higher-Order Cumulants"
  - u: "causal_evidence_for_the_primordiality_of_colors_in_trans-neptunian_objects/"
    t: "Causal Evidence for the Primordiality of Colors in Trans-Neptunian Objects"
  - u: "classifier_reconstruction_through_counterfactual-aware_wasserstein_prototypes/"
    t: "Classifier Reconstruction Through Counterfactual-Aware Wasserstein Prototypes"
  - u: "doubly_protected_estimation_for_survival_outcomes_utilizing_external_controls_fo/"
    t: "Doubly Protected Estimation for Survival Outcomes Utilizing External Controls for Randomized Clinical Trials"
  - u: "e-lda_toward_interpretable_lda_topic_models_with_strong_guarantees_in_logarithmi/"
    t: "E-LDA: Toward Interpretable LDA Topic Models with Strong Guarantees in Logarithmic Parallel Time"
  - u: "estimating_causal_effects_in_gaussian_linear_scms_with_finite_data/"
    t: "Estimating Causal Effects in Gaussian Linear SCMs with Finite Data"
  - u: "exogenous_isomorphism_for_counterfactual_identifiability/"
    t: "Exogenous Isomorphism for Counterfactual Identifiability"
  - u: "internal_causal_mechanisms_robustly_predict_language_model_out-of-distribution_b/"
    t: "Internal Causal Mechanisms Robustly Predict Language Model Out-of-Distribution Behaviors"
  - u: "isolated_causal_effects_of_natural_language/"
    t: "Isolated Causal Effects of Natural Language"
  - u: "latent_variable_causal_discovery_under_selection_bias/"
    t: "Latent Variable Causal Discovery under Selection Bias"
  - u: "learning_time-aware_causal_representation_for_model_generalization_in_evolving_d/"
    t: "Learning Time-Aware Causal Representation for Model Generalization in Evolving Domains"
  - u: "mpf_aligning_and_debiasing_language_models_post_deployment_via_multi_perspective/"
    t: "MPF: Aligning and Debiasing Language Models post Deployment via Multi Perspective Fusion"
  - u: "position_causal_machine_learning_requires_rigorous_synthetic_experiments_for_bro/"
    t: "Position: Causal Machine Learning Requires Rigorous Synthetic Experiments for Broader Adoption"
  - u: "rate_causal_explainability_of_reward_models_with_imperfect_counterfactuals/"
    t: "RATE: Causal Explainability of Reward Models with Imperfect Counterfactuals"
  - u: "re-imagine_symbolic_benchmark_synthesis_for_reasoning_evaluation/"
    t: "RE-IMAGINE: Symbolic Benchmark Synthesis for Reasoning Evaluation"
  - u: "transformer-based_spatial-temporal_counterfactual_outcomes_estimation/"
    t: "Transformer-Based Spatial-Temporal Counterfactual Outcomes Estimation"
item_total: 17
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔗 因果推理

**🧪 ICML2025** · **17** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (4)](../../CVPR2026/causal_inference/index.md) · [🔬 ICLR2026 (63)](../../ICLR2026/causal_inference/index.md) · [💬 ACL2026 (7)](../../ACL2026/causal_inference/index.md) · [🧪 ICML2026 (19)](../../ICML2026/causal_inference/index.md) · [🤖 AAAI2026 (7)](../../AAAI2026/causal_inference/index.md) · [🧠 NeurIPS2025 (20)](../../NeurIPS2025/causal_inference/index.md)

**[Causal Abstraction Inference under Lossy Representations](causal_abstraction_inference_under_lossy_representations.md)**

:   提出 **投影抽象（Projected Abstraction）** 框架，突破现有因果抽象理论对"抽象不变性条件（AIC）"的依赖，使得在有损/降维表示下仍能进行数学一致的因果推断，并给出图模型层面的可识别性判据。

**[Causal Effect Identification in lvLiNGAM from Higher-Order Cumulants](causal_effect_identification_in_lvlingam_from_higher-order_cumulants.md)**

:   在存在潜在混淆的线性非高斯无环模型（lvLiNGAM）中，利用高阶累积量（而非仅协方差矩阵）识别因果效应，解决了两个挑战性设置：(1) 单个可能影响处理的代理变量; (2) 工具变量数少于处理变量数的欠定工具变量问题。两种情况下均证明了可识别性并提供了一致估计方法。

**[Causal Evidence for the Primordiality of Colors in Trans-Neptunian Objects](causal_evidence_for_the_primordiality_of_colors_in_trans-neptunian_objects.md)**

:   利用模型无关的因果发现方法（FCI算法），以 98.7% 的置信度证明海王星外天体（TNO）的颜色是其轨道倾角分布的根本原因，从而强有力地支持了 TNO 颜色的"原生性"假说——即颜色反映的是形成位置而非后期碰撞演化的结果。

**[Classifier Reconstruction Through Counterfactual-Aware Wasserstein Prototypes](classifier_reconstruction_through_counterfactual-aware_wasserstein_prototypes.md)**

:   提出利用 Wasserstein 重心将原始样本与反事实样本融合为类别原型，从而在有限查询预算下高保真地重建目标二分类器，有效缓解了朴素使用反事实样本导致的决策边界偏移问题。

**[Doubly Protected Estimation for Survival Outcomes Utilizing External Controls for Randomized Clinical Trials](doubly_protected_estimation_for_survival_outcomes_utilizing_external_controls_fo.md)**

:   提出一种双重保护（doubly protected）的生存结局估计框架，通过密度比加权校正协变量偏移、DR-Learner检测结局漂移并选择性借用可比外部对照，在保证一致性和效率提升的同时对外部数据异质性具有鲁棒性。

**[E-LDA: Toward Interpretable LDA Topic Models with Strong Guarantees in Logarithmic Parallel Time](e-lda_toward_interpretable_lda_topic_models_with_strong_guarantees_in_logarithmi.md)**

:   提出 E-LDA（Exemplar-LDA），通过将 LDA 的 MAP 主题-词分配问题重新形式化为单调子模函数最大化问题，首次获得了具有 $1-1/e$ 近似保证的实用算法，并且在对数并行时间内收敛，同时保证每个学到的主题都具有基于关键词的形式化可解释性。

**[Estimating Causal Effects in Gaussian Linear SCMs with Finite Data](estimating_causal_effects_in_gaussian_linear_scms_with_finite_data.md)**

:   提出 Centralized Gaussian Linear SCM (CGL-SCM)，通过将外生变量标准化为 $\mathcal{N}(0,1)$ 大幅减少参数量，并设计基于 EM 的估计算法，在有限观测数据下准确恢复可识别的因果效应。

**[Exogenous Isomorphism for Counterfactual Identifiability](exogenous_isomorphism_for_counterfactual_identifiability.md)**

:   提出**外生同构（Exogenous Isomorphism, EI）**概念，证明 $\sim_{\mathrm{EI}}$-identifiability 蕴含 $\sim_{\mathcal{L}_3}$-identifiability（完整反事实层可辨识性），并在双射SCM和三角单调SCM两类特殊模型上给出实现EI的充分条件，统一并推广了已有反事实可辨识性理论。

**[Internal Causal Mechanisms Robustly Predict Language Model Out-of-Distribution Behaviors](internal_causal_mechanisms_robustly_predict_language_model_out-of-distribution_b.md)**

:   利用LLM内部已识别的因果机制来预测模型在分布外输入上的输出正确性，提出反事实模拟和值探测两种方法，在OOD设置中比现有基线平均AUC-ROC提升13.84%。

**[Isolated Causal Effects of Natural Language](isolated_causal_effects_of_natural_language.md)**

:   提出"孤立因果效应"（Isolated Causal Effect）的形式化估计框架，通过双重稳健估计器和遗漏变量偏差（OVB）敏感性分析，将焦点语言属性的因果效应从相关的非焦点语言中隔离出来。

**[Latent Variable Causal Discovery under Selection Bias](latent_variable_causal_discovery_under_selection_bias.md)**

:   首次将秩约束推广到选择偏差场景，证明在线性选择机制下有偏协方差矩阵的秩仍保留因果结构和选择机制的信息，提出广义 t-separation 图准则，并在单因子模型上证明了可识别性，在合成和真实数据（World Value Survey、Big Five 人格）上验证了有效性。

**[Learning Time-Aware Causal Representation for Model Generalization in Evolving Domains](learning_time-aware_causal_representation_for_model_generalization_in_evolving_d.md)**

:   提出时间感知结构因果模型 (time-aware SCM) 和 SYNC 方法，通过同时学习静态与动态因果表示并建模因果机制漂移，在演化域泛化 (EDG) 任务中有效消除虚假相关，实现优越的时序泛化性能。

**[MPF: Aligning and Debiasing Language Models post Deployment via Multi Perspective Fusion](mpf_aligning_and_debiasing_language_models_post_deployment_via_multi_perspective.md)**

:   提出 Multiperspective Fusion (MPF)，一种无需微调的后部署对齐框架，通过将基线情感分布分解为可解释的视角成分，引导 LLM 生成与人类基线对齐的响应，有效缓解模型偏见。

**[Position: Causal Machine Learning Requires Rigorous Synthetic Experiments for Broader Adoption](position_causal_machine_learning_requires_rigorous_synthetic_experiments_for_bro.md)**

:   本文是一篇 Position Paper，主张合成实验对因果机器学习 (Causal ML) 方法的严格评估**不可或缺**，但当前的合成实验设计存在偏差和复杂度不足，需要遵循一套原则来提高实验质量，从而推动 Causal ML 的广泛采用。

**[RATE: Causal Explainability of Reward Models with Imperfect Counterfactuals](rate_causal_explainability_of_reward_models_with_imperfect_counterfactuals.md)**

:   提出 RATE（Rewrite-based Attribute Treatment Estimator），通过"双重重写"策略消除 LLM 不完美反事实重写引入的偏差，从而正确估计高层属性对奖励模型评分的因果效应。

**[RE-IMAGINE: Symbolic Benchmark Synthesis for Reasoning Evaluation](re-imagine_symbolic_benchmark_synthesis_for_reasoning_evaluation.md)**

:   受 Pearl 因果阶梯启发，提出 RE-IMAGINE 框架，通过将问题转化为中间符号表示（代码）并在计算图上执行多层级变异，生成不可通过记忆化解决的基准变体，系统评估 LLM 的真实推理能力。

**[Transformer-Based Spatial-Temporal Counterfactual Outcomes Estimation](transformer-based_spatial-temporal_counterfactual_outcomes_estimation.md)**

:   提出基于 Transformer 的时空反事实结果估计框架，利用 CNN 计算高维倾向性得分、Transformer 估计强度函数，在合成与真实数据上均优于传统因果推理方法。
