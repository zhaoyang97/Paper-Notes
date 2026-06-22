---
title: >-
  ICML2026 因果推理论文汇总 · 19篇论文解读
description: >-
  19篇ICML2026的因果推理方向论文解读，涵盖对抗鲁棒、推理等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2026"
  - "因果推理"
  - "论文解读"
  - "论文笔记"
  - "对抗鲁棒"
  - "推理"
item_list:
  - u: "an_odd_estimator_for_shapley_values/"
    t: "An Odd Estimator for Shapley Values"
  - u: "causal-jepa_learning_world_models_through_object-level_latent_masking/"
    t: "Causal-JEPA: Learning World Models through Object-Level Latent Masking"
  - u: "causal_modeling_of_selection_in_evolution/"
    t: "Causal Modeling of Selection in Evolution"
  - u: "controllable_generative_sandbox_for_causal_inference/"
    t: "Controllable Generative Sandbox for Causal Inference"
  - u: "density-guided_robust_counterfactual_explanations_on_tabular_data_under_model_mu/"
    t: "Density-Guided Robust Counterfactual Explanations on Tabular Data under Model Multiplicity"
  - u: "ecsel_explainable_classification_via_signomial_equation_learning/"
    t: "ECSEL: Explainable Classification via Signomial Equation Learning"
  - u: "evaluating_bivariate_causal_statements_based_on_mutual_compatibility/"
    t: "Evaluating Bivariate Causal Statements Based on Mutual Compatibility"
  - u: "finding_most_influential_sets/"
    t: "Finding Most Influential Sets"
  - u: "formalizing_and_falsifying_causal_pathways_of_rare_events/"
    t: "Formalizing and Falsifying Causal Pathways of Rare Events"
  - u: "from_observation_to_intervention_a_causal_audit_of_expert_importance_in_mixture-/"
    t: "From Observation to Intervention: A Causal Audit of Expert Importance in Mixture-of-Experts Models"
  - u: "investigating_memory_in_model-free_rl_with_popgym_arcade/"
    t: "Investigating Memory in Model-Free RL with POPGym Arcade"
  - u: "outcome-aware_spectral_feature_learning_for_instrumental_variable_regression/"
    t: "Outcome-Aware Spectral Feature Learning for Instrumental Variable Regression"
  - u: "rank-learner_orthogonal_ranking_of_treatment_effects/"
    t: "Rank-Learner: Orthogonal Ranking of Treatment Effects"
  - u: "tailoring_strictly_proper_scoring_rules_for_downstream_tasks_an_application_to_c/"
    t: "Tailoring Strictly Proper Scoring Rules for Downstream Tasks: An Application to Causal Inference"
  - u: "the_marginal_value_of_a_search_ad_an_online_causal_framework_for_repeated_second/"
    t: "The (Marginal) Value of a Search Ad: An Online Causal Framework for Repeated Second-price Auctions"
  - u: "the_synthetic_web_adversarially-curated_mini-internets_for_diagnosing_epistemic_/"
    t: "The Synthetic Web: Adversarially-Curated Mini-Internets for Diagnosing Epistemic Weaknesses of Language Agents"
  - u: "toward_scalable_and_valid_conditional_independence_testing_with_spectral_represe/"
    t: "Toward Scalable and Valid Conditional Independence Testing with Spectral Representations"
  - u: "towards_a_holistic_understanding_of_selection_bias_for_causal_effect_identificat/"
    t: "Towards a Holistic Understanding of Selection Bias for Causal Effect Identification"
  - u: "unveiling_the_structure_of_do-calculus_reasoning_via_derivation_graphs/"
    t: "Unveiling the Structure of Do-Calculus Reasoning via Derivation Graphs"
item_total: 19
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔗 因果推理

**🧪 ICML2026** · **19** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (4)](../../CVPR2026/causal_inference/index.md) · [🔬 ICLR2026 (63)](../../ICLR2026/causal_inference/index.md) · [💬 ACL2026 (7)](../../ACL2026/causal_inference/index.md) · [🤖 AAAI2026 (7)](../../AAAI2026/causal_inference/index.md) · [🧠 NeurIPS2025 (20)](../../NeurIPS2025/causal_inference/index.md) · [📹 ICCV2025 (2)](../../ICCV2025/causal_inference/index.md)

🔥 **高频主题：** 对抗鲁棒 ×2

**[An Odd Estimator for Shapley Values](an_odd_estimator_for_shapley_values.md)**

:   这篇论文证明 Shapley value 只依赖集合函数的 odd component，并据此提出 OddSHAP：用配对采样隔离 odd 信号、用 GBT 筛选高阶 odd Fourier 交互、再做稀疏 odd 回归，在中高维解释任务上显著优于灵活预算 Shapley 估计器。

**[Causal-JEPA: Learning World Models through Object-Level Latent Masking](causal-jepa_learning_world_models_through_object-level_latent_masking.md)**

:   提出 C-JEPA，将 JEPA 的掩码预测从图像 patch 级别扩展到对象级别潜在表示，通过对象级掩码作为潜在干预迫使模型学习交互依赖的动态，在反事实推理上比无掩码基线提升约 20%，在控制任务中仅用 1% 的 token 即达到可比性能且规划加速 8 倍以上。

**[Causal Modeling of Selection in Evolution](causal_modeling_of_selection_in_evolution.md)**

:   论文指出"选择"其实分**静态选择**（一次性过滤）和**演化选择**（多代差异繁殖累积）两种，现有图模型把二者混为一谈、在演化数据上会得出错误的因果发现；作者定义了显式刻画演化的因果图模型，并证明它的条件独立约束可以用一个"团扩张 DAG"无损表示，从而能直接套用标准 PC/GES/CDNOD 算法、只需重新解释输出。

**[Controllable Generative Sandbox for Causal Inference](controllable_generative_sandbox_for_causal_inference.md)**

:   本文提出 CausalMix：一个变分生成框架，把数据类型特定的 multi-head decoder + Bayesian Gaussian 混合潜在 prior 与三类可独立调控的因果"旋钮"（overlap $\alpha(X)$、CATE 函数 $\tau(X)$、未观测混杂 $\kappa(X,T)$）联合优化，从而在保持真实数据分布 fidelity 的前提下让用户自由设计 counterfactual benchmark，在 mCRPC（前列腺癌）真实病例上验证 CausalMix 既能高保真复现 mixed-type 表格，又能稳定地按需注入 overlap / confounding / 异质效应，用作 CATE 估计器的可控 stress test。

**[Density-Guided Robust Counterfactual Explanations on Tabular Data under Model Multiplicity](density-guided_robust_counterfactual_explanations_on_tabular_data_under_model_mu.md)**

:   DensityFlow 把"在模型多重性下生成鲁棒反事实解释 (RCE)"重新表述为带密度约束的最优传输问题，用 NCE 训练一个 (K+1) 类判别器同时学分类与类条件密度，再用 Neural ODE 把查询样本沿密度梯度运到目标类高密度流形上，并在黑盒场景下只对生成轨迹做局部蒸馏对齐，从而以远低于集成基线的查询量取得更高的跨模型 validity。

**[ECSEL: Explainable Classification via Signomial Equation Learning](ecsel_explainable_classification_via_signomial_equation_learning.md)**

:   ECSEL 把"每个类别一个 signomial（带实数指数的幂律和）函数 + softmax"作为分类器，配合 L1 稀疏正则与多阶段优化，既能在 AI Feynman 等符号回归 benchmark 上以远低于 SOTA 的算力恢复 95.86% 的目标方程，又能在 11 个分类数据集上与 XGBoost/MLP 打平，同时所有特征归因都由模型参数闭式给出。

**[Evaluating Bivariate Causal Statements Based on Mutual Compatibility](evaluating_bivariate_causal_statements_based_on_mutual_compatibility.md)**

:   本文针对"只有成对(bivariate)因果陈述、没有 ground truth"的场景，提出两个无需 faithfulness 的相容性评分（线性情形的 `comp` + 图结构情形的 `incomp`），通过判断这些两两陈述拼起来的多元模型是否需要"反常的额外混淆"来解释观测协方差，从而识别错误的因果论断，并用它给 LLM 的因果输出打分。

**[Finding Most Influential Sets](finding_most_influential_sets.md)**

:   要找"删掉后最大改变某个估计量的 size-$k$ 子集"（最具影响力集合 MIS）本来需要在 $\binom{n}{k}$ 个子集里穷举、根本算不动；本文证明只要留集删除效应能写成**线性分式**形式，MIS 选择就坍缩成一串"选 top-$k$"的子问题，用 Dinkelbach 方法做到每轮 $\mathcal{O}(n)$、有限步终止，并在偏线性模型里给出从"固定输入精确最优"到"统计意义上恢复 oracle 集合"的完整理论保证。

**[Formalizing and Falsifying Causal Pathways of Rare Events](formalizing_and_falsifying_causal_pathways_of_rare_events.md)**

:   本文把罕见事件的"口头因果解释"形式化为 **causal pathway**——一个由二值化事件构成的子图，并定义 **pathway explanation score** 来量化"根因 + 中介通路"对目标事件的解释力，得到一套可证伪的因果解释评价框架。

**[From Observation to Intervention: A Causal Audit of Expert Importance in Mixture-of-Experts Models](from_observation_to_intervention_a_causal_audit_of_expert_importance_in_mixture-.md)**

:   作者用「逐 token 消融」的干预性审计去检验 MoE 剪枝里「观测性路由统计量能预测哪些专家可删」这一隐含假设，在三个高冗余 MoE 模型上得到一个干净的「三模型零结果」：60 个指标-层组合里没有任何观测指标在多重比较校正后能预测专家的因果重要性，说明现有剪枝方法之所以有效，不是因为指标真挑出了「没用的专家」，而是因为早中层的冗余让几乎任何选择准则都同样安全。

**[Investigating Memory in Model-Free RL with POPGym Arcade](investigating_memory_in_model-free_rl_with_popgym_arcade.md)**

:   本文指出仅用回报来比较 RL 记忆模型并不可靠，作者构建了一个 GPU 加速的 MDP/POMDP "孪生"基准 POPGym Arcade，并提出 Observability Gap、Memory Bias、像素显著性和 Recall Density 四个工具，借此揭示了一种"价值涂抹（value smearing）"病理：记忆模型会把价值信用错误地分摊到无关的历史观测上，进而导致单个 OOD 观测就能通过 recurrent state 长期污染策略。

**[Outcome-Aware Spectral Feature Learning for Instrumental Variable Regression](outcome-aware_spectral_feature_learning_for_instrumental_variable_regression.md)**

:   针对非参数工具变量（NPIV）回归中 SpecIV 学到的谱特征"只看 X-Z 关系、不看结果 Y"的盲点，本文提出 Augmented Spectral Feature Learning：在 SpecIV 的对比损失里加上一项 Y 投影到 Z 特征上的回归损失，等价于对一个把 Y 信息拼进去的"增广算子" $\mathcal{T}_\delta = [\mathcal{T} \mid \delta r_0]$ 做截断 SVD，从而在结构函数 $h_0$ 与 $\mathcal{T}$ 顶端奇异函数对齐很差的"坏"情形下也能用极低秩特征恢复因果效应。

**[Rank-Learner: Orthogonal Ranking of Treatment Effects](rank-learner_orthogonal_ranking_of_treatment_effects.md)**

:   在观测数据上提出 Rank-Learner——第一个 Neyman-正交的两阶段处理效应**排序**学习器，用成对软标签 + 双重稳健修正项替代"先估 CATE 再排"的间接做法，在合成、半合成与 Criteo uplift 真实数据集上稳定优于 T/DR-learner 与非正交 plug-in ranker。

**[Tailoring Strictly Proper Scoring Rules for Downstream Tasks: An Application to Causal Inference](tailoring_strictly_proper_scoring_rules_for_downstream_tasks_an_application_to_c.md)**

:   本文提出一个通用框架：通过让训练损失的局部二阶曲率 $w_\ell(p)$ 匹配下游任务误差的曲率 $w_{\text{task}}(p)$，可派生出与下游任务"几何对齐"的严格 proper scoring rule；将其应用到 IPW 估计 ATE，得到闭式损失 + 闭式 canonical 激活函数（解一个四次方程），在 IHDP / Jobs / Kang-Schafer / ACIC 2017 上稳定优于 log-loss 与 covariate balancing 类基线。

**[The (Marginal) Value of a Search Ad: An Online Causal Framework for Repeated Second-price Auctions](the_marginal_value_of_a_search_ad_an_online_causal_framework_for_repeated_second.md)**

:   本文把搜索广告的真实价值建模为"赢拍 vs 输拍"的 treatment effect，在重复二价拍卖（SPA）binary 反馈下设计了一个利用支付规则的在线因果学习算法，得到 $\widetilde\Theta(\sqrt{dT})$ 的极小极大最优 regret，比同设定下的一价拍卖严格更易学。

**[The Synthetic Web: Adversarially-Curated Mini-Internets for Diagnosing Epistemic Weaknesses of Language Agents](the_synthetic_web_adversarially-curated_mini-internets_for_diagnosing_epistemic_.md)**

:   本文构造了一个程序化生成的"合成 Web"环境,通过在搜索 rank 0 注入单条高可信度蜜罐误信息,因果性地测出 GPT-5 等前沿 LLM agent 在 1/数千的对抗污染下准确率从 65% 暴跌到 18%,且模型不会增加搜索、依然高置信度作答,揭示了根深蒂固的"位置锚定"失败模式。

**[Toward Scalable and Valid Conditional Independence Testing with Spectral Representations](toward_scalable_and_valid_conditional_independence_testing_with_spectral_represe.md)**

:   SpectralCIT 把核方法里刻画条件独立的「偏协方差算子」用神经网络学到的低维谱特征来近似，再用一个形如 HSIC 的简单统计量做条件独立检验——它用一个双层对比算法学出算子的领先奇异特征，证明零假设下统计量渐近服从卡方分布、备择假设下有功效保证，从而把核方法的理论扎实性和现代表示学习的可扩展性接上。

**[Towards a Holistic Understanding of Selection Bias for Causal Effect Identification](towards_a_holistic_understanding_of_selection_bias_for_causal_effect_identificat.md)**

:   本文给出一个统一的"分布类"框架，刻画了在选择偏差下平均处理效应 (ATE) 全人群可识别的充要条件 (Condition 1)，并证明在 c-overlap 倾向得分 + 多项式指数族 / Gaussian / Laplace / Pareto / Log-normal 等常见分布下都满足该条件，配套提出 MLE 与 Score Matching 两种带选择函数 $\beta(x,y,t)$ 校正的估计器，在合成与 All of Us 半合成实验上显著优于 IPW 与多项式回归。

**[Unveiling the Structure of Do-Calculus Reasoning via Derivation Graphs](unveiling_the_structure_of_do-calculus_reasoning_via_derivation_graphs.md)**

:   通过引入推导图（derivation graphs）显式表示 do-演算规则的所有等价变换——揭示因果表达式空间的结构，并证明最多 4 步规则应用可达任意等价表达式。
