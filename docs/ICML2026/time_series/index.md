---
title: >-
  ICML2026 时间序列论文汇总 · 45篇论文解读
description: >-
  45篇ICML2026的时间序列方向论文解读，涵盖时序预测、对抗鲁棒、推理、异常检测、自监督学习、对齐/RLHF等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "ICML2026"
  - "时间序列"
  - "论文解读"
  - "论文笔记"
  - "时序预测"
  - "对抗鲁棒"
  - "推理"
  - "异常检测"
  - "自监督学习"
  - "对齐/RLHF"
item_list:
  - u: "adaptive_time_series_reasoning_via_segment_selection/"
    t: "Adaptive Time Series Reasoning via Segment Selection"
  - u: "anomseer_reinforcing_multimodal_llms_to_reason_for_time-series_anomaly_detection/"
    t: "AnomSeer: Reinforcing Multimodal LLMs to Reason for Time-Series Anomaly Detection"
  - u: "beyond_extrapolation_knowledge_utilization_paradigm_with_bidirectional_inspirati/"
    t: "Beyond Extrapolation: Knowledge Utilization Paradigm with Bidirectional Inspiration for Time Series Forecasting"
  - u: "building_social_world_models_with_large_language_models/"
    t: "Building Social World Models with Large Language Models"
  - u: "combinationts_a_modular_framework_for_understanding_time-series_forecasting_mode/"
    t: "CombinationTS: A Modular Framework for Understanding Time-Series Forecasting Models"
  - u: "dag_a_dual_correlation_network_for_time_series_forecasting_with_exogenous_variab/"
    t: "DAG: A Dual Correlation Network for Time Series Forecasting with Exogenous Variables"
  - u: "distmatch_adaptive_binning_via_distribution_matching_for_robust_sequential_confo/"
    t: "DistMatch: Adaptive Binning via Distribution Matching for Robust Sequential Conformal"
  - u: "divide_and_contrast_learning_robust_temporal_features_without_augmentation/"
    t: "Divide and Contrast: Learning Robust Temporal Features Without Augmentation"
  - u: "do_time_series_foundation_model_benchmarks_hide_regime-dependent_failures_eviden/"
    t: "Do Time Series Foundation Model Benchmarks Hide Regime-Dependent Failures? Evidence from Traffic Speed Forecasting"
  - u: "doubly_outlier-robust_online_infinite_hidden_markov_model/"
    t: "Doubly Outlier-Robust Online Infinite Hidden Markov Model"
  - u: "dynamic_tmoe_a_drift-aware_dynamic_mixture_of_experts_framework_for_non-stationa/"
    t: "Dynamic-TMoE: A Drift-Aware Dynamic Mixture of Experts Framework for Non-Stationary Time Series"
  - u: "ellipsoidal_time_series_forecasting/"
    t: "Ellipsoidal Time Series Forecasting"
  - u: "embedding_hybrid_systems_into_continuous_latent_vector_fields/"
    t: "Embedding Hybrid Systems into Continuous Latent Vector Fields"
  - u: "exposure_bias_as_epistemic_underidentification_in_recursive_forecasting/"
    t: "Exposure Bias as Epistemic Underidentification in Recursive Forecasting"
  - u: "factorynet_a_large-scale_dataset_toward_industrial_time-series_foundation_models/"
    t: "FactoryNet: A Large-Scale Dataset toward Industrial Time-Series Foundation Models"
  - u: "fractal_ssm_with_fractional_recurrent_architecture_for_computational_temporal_an/"
    t: "FRACTAL: State Space Model with Fractional Recurrent Architecture for Computational Temporal Analysis of Long Sequences"
  - u: "from_observations_to_states_latent_time_series_forecasting/"
    t: "From Observations to States: Latent Time Series Forecasting"
  - u: "generalizing_multi-scale_time-series_modeling_with_a_single_operator/"
    t: "Generalizing Multi-scale Time-Series Modeling with a Single Operator"
  - u: "helix_hybrid_encoding_with_learnable_identity_and_cross-dimensional_synthesis_fo/"
    t: "HELIX: Hybrid Encoding with Learnable Identity and Cross-dimensional Synthesis for Time Series Imputation"
  - u: "hepa_a_self-supervised_horizon-conditioned_event_predictive_architecture_for_tim/"
    t: "HEPA: A Self-Supervised Horizon-Conditioned Event Predictive Architecture for Time Series"
  - u: "hippo_zoo_explicit_memory_mechanisms_for_interpretable_state_space_models/"
    t: "HiPPO Zoo: Explicit Memory Mechanisms for Interpretable State Space Models"
  - u: "impact_influence_modeling_for_open-set_time_series_anomaly_detection/"
    t: "IMPACT: Influence Modeling for Open-Set Time Series Anomaly Detection"
  - u: "incremental_transformer_neural_processes/"
    t: "Incremental Transformer Neural Processes"
  - u: "interpretability_in_deep_time_series_models_demands_semantic_alignment/"
    t: "Interpretability in Deep Time Series Models Demands Semantic Alignment"
  - u: "its_time_towards_the_next_generation_of_time_series_forecasting_benchmarks/"
    t: "It's TIME: Towards the Next Generation of Time Series Forecasting Benchmarks"
  - u: "latent_laplace_diffusion_for_irregular_multivariate_time_series/"
    t: "Latent Laplace Diffusion for Irregular Multivariate Time Series"
  - u: "learning_long_range_spatio-temporal_representations_over_continuous_time_dynamic/"
    t: "Learning Long Range Spatio-Temporal Representations over Continuous Time Dynamic Graphs with State Space Models"
  - u: "learning_manifold_and_itô_dynamics_with_branched_neural_rough_differential_equat/"
    t: "Learning Manifold and Itô Dynamics with Branched Neural Rough Differential Equations"
  - u: "mix_dont_pick_why_synthetic_corpus_composition_matters_for_time_series_foundatio/"
    t: "Mix, Don't Pick: Why Synthetic Corpus Composition Matters for Time Series Foundation Model Pretraining"
  - u: "nested_spatio-temporal_time_series_forecasting/"
    t: "Nested Spatio-Temporal Time Series Forecasting"
item_total: 45
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📈 时间序列

**🧪 ICML2026** · **45** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (7)](../../CVPR2026/time_series/index.md) · [🔬 ICLR2026 (121)](../../ICLR2026/time_series/index.md) · [💬 ACL2026 (8)](../../ACL2026/time_series/index.md) · [🤖 AAAI2026 (31)](../../AAAI2026/time_series/index.md) · [🧠 NeurIPS2025 (54)](../../NeurIPS2025/time_series/index.md) · [📹 ICCV2025 (4)](../../ICCV2025/time_series/index.md)

🔥 **高频主题：** 时序预测 ×33 · 对抗鲁棒 ×3 · 推理 ×2 · 异常检测 ×2 · 自监督学习 ×2

**[Adaptive Time Series Reasoning via Segment Selection](adaptive_time_series_reasoning_via_segment_selection.md)**

:   这篇论文提出 ARTIST，把时间序列问答变成“边推理边选择片段”的序贯决策问题，通过 controller-reasoner 架构和层级自博弈 RL，让模型只读取与问题相关的时间片段并提升推理准确率。

**[AnomSeer: Reinforcing Multimodal LLMs to Reason for Time-Series Anomaly Detection](anomseer_reinforcing_multimodal_llms_to_reason_for_time-series_anomaly_detection.md)**

:   AnomSeer 将经典时间序列异常检测的统计证据写成专家推理轨迹，并用 TimerPO 强化多模态大模型，使其在折线图输入上同时完成异常类型判断、区间定位和细粒度解释。

**[Beyond Extrapolation: Knowledge Utilization Paradigm with Bidirectional Inspiration for Time Series Forecasting](beyond_extrapolation_knowledge_utilization_paradigm_with_bidirectional_inspirati.md)**

:   提出 KUP-BI 框架，从训练集中构建"后目标延续"知识库，通过比率式变换检索相似历史轨迹的延续模式，生成延续风格辅助流并与主干网络特征门控融合，在 6 个数据集、4 种骨干架构上一致提升长时预测性能。

**[Building Social World Models with Large Language Models](building_social_world_models_with_large_language_models.md)**

:   本文提出"社会世界模型"（SWM），把集体信念当作状态、把社会事件当作外生动作，用 LLM 作转移引擎学一个事件条件的状态转移分布 $P_\theta(\mathbf s_{t+1}\mid\mathbf s_t,e_t)$；通过一个冻结的"事后后验归因器"提供伪标签来绕开"事件→信念变化"标注缺失的难题，在用真实预测市场（Kalshi/Polymarket）构建的 SWM-Bench 上显著超过时间序列基础模型与 GPT-5.5 等强基线。

**[CombinationTS: A Modular Framework for Understanding Time-Series Forecasting Models](combinationts_a_modular_framework_for_understanding_time-series_forecasting_mode.md)**

:   CombinationTS 把时序预测模型解耦为 Input Transformation / Embedding / Encoder / Decoder / Output Transformation 五个正交模块，在共享的"评估条件空间"上做配对蒙特卡洛采样，用边际性能 $\mu$ 和稳定性 $\sigma$ 取代脆弱的单点 MSE，结论是：一旦数据视图（Embedding）设计得好，参数无关的 Identity Encoder 就能打平甚至超过复杂 Transformer，时序预测领域的"SOTA 增益"很大程度上来自看数据的方式而不是建模能力。

**[DAG: A Dual Correlation Network for Time Series Forecasting with Exogenous Variables](dag_a_dual_correlation_network_for_time_series_forecasting_with_exogenous_variab.md)**

:   针对"未来协变量已知"的时间序列预测 (TSF-X), DAG 设计了一个双通路网络: 一条沿时间维捕获"历史外生→未来外生"的注意力模式并注入到"历史内生→未来内生"的预测里, 另一条沿通道维捕获"历史外生→历史内生"的模式并注入到"未来外生→未来内生"的预测里, 在 12 个公开/新发布 TSF-X 数据集上 10/10 拿下 MSE 最佳, 显著超过 TimeXer、TFT、TiDE、CrossLinear、PatchTST 等。

**[DistMatch: Adaptive Binning via Distribution Matching for Robust Sequential Conformal](distmatch_adaptive_binning_via_distribution_matching_for_robust_sequential_confo.md)**

:   DistMatch 提出基于 **KS 统计量**的递归分箱方法——通过将残差分组到近似可交换的叶子节点中**摒弃权重重新分配**，在分布漂移下提供有效的保形预测间隔；5 个数据集上均实现最小的区间宽度，同时保持有效覆盖率。

**[Divide and Contrast: Learning Robust Temporal Features Without Augmentation](divide_and_contrast_learning_robust_temporal_features_without_augmentation.md)**

:   Di-COT 通过**随机划分序列为重叠子块**并对其进行对比学习——在不使用数据增强的情况下高效学习鲁棒的时间序列表示，相比现有方法速度快 2.5 倍、精度更高；6 大规模数据集 + 124 UCR + 28 UEA 上全面验证。

**[Do Time Series Foundation Model Benchmarks Hide Regime-Dependent Failures? Evidence from Traffic Speed Forecasting](do_time_series_foundation_model_benchmarks_hide_regime-dependent_failures_eviden.md)**

:   这篇论文指出时序基础模型（TSFM）在交通速度预测上"平均指标好看、关键时刻失灵"——它用按交通状态分层的评测揭穿了聚合指标掩盖的灾难性失败，并提出无需重训的后处理方法 BMA，把"过渡态"的预测区间覆盖率拉回接近历史基线的水平。

**[Doubly Outlier-Robust Online Infinite Hidden Markov Model](doubly_outlier-robust_online_infinite_hidden_markov_model.md)**

:   本文提出 BR-iHMM：把"鲁棒观测更新（WoLF）"与"批量化状态推断（degenerate sticky HDP prior）"结合起来，给在线无限隐马模型同时在观测空间和状态空间提供有界的 Posterior Influence Function（PIF），在金融订单簿、电力负荷、合成回归三类含异常值的流式数据上把一步预测 RMSE 最多降低 67%。

**[Dynamic-TMoE: A Drift-Aware Dynamic Mixture of Experts Framework for Non-Stationary Time Series](dynamic_tmoe_a_drift-aware_dynamic_mixture_of_experts_framework_for_non-stationa.md)**

:   通过 **MMD 检测分布漂移**并动态扩展异构专家池，结合**时间记忆路由器**保证选择一致性，Dynamic-TMoE 在九个时间序列基准上达到新的 SOTA——相比所有基线平均降低 MSE 10.4%、MAE 7.8%。

**[Ellipsoidal Time Series Forecasting](ellipsoidal_time_series_forecasting.md)**

:   Fern 把长期时间序列预测重新表述为「从固定高斯源到数据相关椭球的最优传输」，借助 Brenier 定理把搜索空间限制在 SPD（对称正半定）类 Jacobian 上，用 Householder 反射的低秩谱分解把代价从 $O(n^3)$ 压到 $O(Rn)$，并在非平稳冲击场景下相对 DLinear / Koopa 等基线取得最多 790× 的稳定性提升。

**[Embedding Hybrid Systems into Continuous Latent Vector Fields](embedding_hybrid_systems_into_continuous_latent_vector_fields.md)**

:   本文先证明一条存在性定理——只要隐空间维数 $m>2n$，一个本质上**不连续**的 $n$ 维混合系统就能被嵌入到 $m$ 维欧氏空间、并在其像上配出一个**连续**向量场——再据此设计隐空间 Neural ODE 框架 CHyLL++，仅凭时间序列就能高精度恢复各种几何拓扑的混合系统流。

**[Exposure Bias as Epistemic Underidentification in Recursive Forecasting](exposure_bias_as_epistemic_underidentification_in_recursive_forecasting.md)**

:   本文从理论上重新解释递归多步预测里的"暴露偏差"：它不只是训练（teacher forcing）与部署（自喂预测）之间的**分布偏移**，在部分可观测或状态截断下，它还是一个**认识论上的不可辨识（epistemic underidentification）**问题——一步监督只能在观测上下文上确定模型行为，无法确定 rollout 在自生成"诱导状态"上该输出什么，作者用"诱导状态 $Z$ + 来源变量 $P$"把这件事形式化，并给出误差分解和实验验证。

**[FactoryNet: A Large-Scale Dataset toward Industrial Time-Series Foundation Models](factorynet_a_large-scale_dataset_toward_industrial_time-series_foundation_models.md)**

:   FactoryNet 是首个统一控制环结构的工业时序大规模数据集——5100 万数据点 / 2.3 万端到端任务执行（1.33 万真实 + 9800 仿真）跨 6 个机器实体，按 Setpoint-Effort-Feedback-Context (S-E-F-C) 控制论分类对齐所有信号；27 种标注异常类型 + 健康基线 + 反事实对，使零样本跨实体迁移和参数高效异常检测成为可能。

**[FRACTAL: State Space Model with Fractional Recurrent Architecture for Computational Temporal Analysis of Long Sequences](fractal_ssm_with_fractional_recurrent_architecture_for_computational_temporal_an.md)**

:   本文把 HiPPO 框架背后的概率测度推广到带可调奇异指数 $\alpha$ 的分数阶幂律测度，从而首次同时拿到「全历史保留 + 近时敏感 + 尺度不变」，并将这一理论落地为 LTI 对角化 SSM——FRACTAL 在 Long Range Arena 上以 87.11% 平均分追平 S5，并在 ListOps 上拿到 61.85%。

**[From Observations to States: Latent Time Series Forecasting](from_observations_to_states_latent_time_series_forecasting.md)**

:   作者发现现有 TSF 模型即使预测精度高，其潜空间也常常是"时间错乱"的（Latent Chaos）；他们提出 LatentTSF——先用 AutoEncoder 把观察压到一个高维潜状态空间，然后让任何主流 backbone 在这个空间内做未来预测（Pred + Align 双损失），最后再解码回观察空间——在 6 个标准 benchmark 上稳定降 MSE/MAE，并恢复了潜表征的时间局部性和频谱结构。

**[Generalizing Multi-scale Time-Series Modeling with a Single Operator](generalizing_multi-scale_time-series_modeling_with_a_single_operator.md)**

:   Sigma 框架通过学习**离散高斯（LDG）核**实现**连续、距离感知的尺度参数**，统一了现有的离散多尺度算子——在长期和短期预测任务上达到 SOTA，同时大幅降低计算成本（训练快 5.3×、显存少 3.8×）。

**[HELIX: Hybrid Encoding with Learnable Identity and Cross-dimensional Synthesis for Time Series Imputation](helix_hybrid_encoding_with_learnable_identity_and_cross-dimensional_synthesis_fo.md)**

:   给每个特征学一个"身份嵌入"作为持久语义锚点，配合时间-特征双螺旋注意力，在 5 个公开多变量时序数据集 21 个缺失场景上全部拿下第一，比次优的 ImputeFormer 在 ETT-h1 等数据集上多 25% 以上的 MAE 降幅。

**[HEPA: A Self-Supervised Horizon-Conditioned Event Predictive Architecture for Time Series](hepa_a_self-supervised_horizon-conditioned_event_predictive_architecture_for_tim.md)**

:   HEPA 通过**地平线条件化的 JEPA 自监督预训练**学习时间序列中的可预测动态——冻结编码器只微调预测器，用单一架构和固定超参在 11 个领域 14 个基准上超越多个 SOTA 方法，仅用 2% 标签数据即可达到 92% 性能。

**[HiPPO Zoo: Explicit Memory Mechanisms for Interpretable State Space Models](hippo_zoo_explicit_memory_mechanisms_for_interpretable_state_space_models.md)**

:   将现代 SSM（如 Mamba）中隐式的内存机制**显式化**——通过扩展 HiPPO 框架提出"HiPPO Zoo"（5 个变体），每个变体用可解释的多项式表示来实现特定的现代 SSM 能力（非线性、自适应内存、关联记忆、多尺度、预测目标约束）；选择性复制和关联回忆任务上达到 100%。

**[IMPACT: Influence Modeling for Open-Set Time Series Anomaly Detection](impact_influence_modeling_for_open-set_time_series_anomaly_detection.md)**

:   IMPACT 把"影响函数"同时拿来当探照灯和手术刀——先用一个多通道偏差损失训出初始模型并算出每个训练样本对验证风险的影响分数，再在风险下降的理论保证下，把高影响的污染未标样本一键翻成有标异常、把对风险贡献最小的"边界正常样本"沿梯度方向扰动成"未见过的伪异常"，最后用双头网络分别学已见与未见两类异常，在 8 个真实时序基准上稳定超越十多个无监督与开放集基线。

**[Incremental Transformer Neural Processes](incremental_transformer_neural_processes.md)**

:   把大模型里的因果掩码 + KV 缓存搬进 Transformer 神经过程（TNP），让流式场景下每来一个新观测的更新代价从 $\mathcal{O}(N^2)$ 降到 $\mathcal{O}(N)$，配上一种「单次前向覆盖所有上下文长度」的稠密自回归训练，incTNP 不仅没掉点、反而常常超过标准 TNP，且预测规则的「隐式贝叶斯性」与排列不变的 TNP 相当。

**[Interpretability in Deep Time Series Models Demands Semantic Alignment](interpretability_in_deep_time_series_models_demands_semantic_alignment.md)**

:   本文是一篇**位置论文**——提出深度时间序列模型应该强制**语义对齐**：让模型的内部变量和机制对应领域专家的推理方式而非仅解释内部计算；核心创新是针对时间演化定义了语义对齐的持久性约束（这是时间序列特有问题）。

**[It's TIME: Towards the Next Generation of Time Series Forecasting Benchmarks](its_time_towards_the_next_generation_of_time_series_forecasting_benchmarks.md)**

:   TIME 是面向**时间序列基础模型（TSFM）**的下一代基准——通过**人工标注 + LLM 驱动的数据清洗**、**上下文对齐的任务设计**、**模式级别的评估视角**，克服现有基准的数据重用、质量问题、任务配置不当和评估粒度低等四大痛点；50 个全新数据集 × 98 任务 × 12 TSFM 评估。

**[Latent Laplace Diffusion for Irregular Multivariate Time Series](latent_laplace_diffusion_for_irregular_multivariate_time_series.md)**

:   LLapDiff 是在**隐空间中进行扩散**的生成框架——通过在拉普拉斯域用可学习的复共轭极点参数化**稳定的模态演化**，实现不规则时间序列的长期预测和缺失值补全，**无需逐步的物理时间积分**；7 个数据集上平均排名 2.1±1.7。

**[Learning Long Range Spatio-Temporal Representations over Continuous Time Dynamic Graphs with State Space Models](learning_long_range_spatio-temporal_representations_over_continuous_time_dynamic.md)**

:   CTDG-SSM 首次通过**拓扑感知 HiPPO 投影**和状态空间模型，同时捕捉动态图中的多跳长距离空间依赖（LRS）和长距离时间依赖（LRT），在链接预测 / 节点分类等任务上超越 SOTA 且参数量仅为竞争方法的 1/10。

**[Learning Manifold and Itô Dynamics with Branched Neural Rough Differential Equations](learning_manifold_and_itô_dynamics_with_branched_neural_rough_differential_equat.md)**

:   神经粗糙微分方程（NRDE）只能处理 Stratonovich 动力学（因为它依赖 shuffle 代数），本文把 NRDE 的 log-ODE 步骤换成 **Hopf 代数上的几何数值积分**——用 Grossman–Larson 有根树代数处理欧氏 Itô、用 Munthe–Kaas–Wright 平面有根树代数处理流形上的有序协变导数、shuffle 代数留给经典 Stratonovich，从而把签名方法首次推广到 Itô 与流形值动力学，并配一个分支签名核目标让二次变差项在训练中可见。

**[Mix, Don't Pick: Why Synthetic Corpus Composition Matters for Time Series Foundation Model Pretraining](mix_dont_pick_why_synthetic_corpus_composition_matters_for_time_series_foundatio.md)**

:   这篇论文用 11 个合成时序生成器 × 2 个从头训练的时序基础模型做系统对照，发现"选哪个生成器"在不同架构间排名都不稳定、最好与最差之间预测误差差到 2 倍，于是干脆不去解选择难题——把所有生成器等权一锅混（Mixed11）就能追平甚至超过最佳单一生成器，再掺入真实数据得到最强语料；结论是合成预训练是一个"语料组成"问题，而非"生成器选择"问题，且组成方案必须按模型架构逐一验证。

**[Nested Spatio-Temporal Time Series Forecasting](nested_spatio-temporal_time_series_forecasting.md)**

:   NeST 把"未来的宏观区域趋势"作为自顶向下引导，配合谱聚类构造的语义区域和双向跨尺度 cross-attention，让节点级时空预测在大规模交通网络上同时取得精度、长程稳定性与近线性复杂度的全面提升。

**[OLIVIA: Harmonizing Time Series Foundation Models with Power Spectral Density](olivia_harmonizing_time_series_foundation_models_with_power_spectral_density.md)**

:   OLIVIA 通过引入功率谱密度（PSD）驱动的协调机制——Harmonizer（基于 Householder 反射的正交二阶协调）和 HarmonicAttention（共鸣器低维交互）——显著改进了时间序列基础模型在异质数据上的预训练，在 TSLib 零样本 + GIFT-Eval + GluonTS 多基准上实现 SOTA。

**[Once-for-All: Scalable Simultaneous Forecasting via Equilibrium State Estimation](once-for-all_scalable_simultaneous_forecasting_via_equilibrium_state_estimation.md)**

:   针对「多个相互影响的系统要同时预测」这类场景（如 16 国汇率、几百个区县的疫情新增），本文提出 Equilibrium State Estimation（ESE）：先一次性估出所有系统的「均衡态比例」，再用当前状态偏离均衡的方向做单遍预测，从而把原本一个一个系统反复预测的 $O(n)$ 训练换成线性时间的一次推理，精度持平 SOTA、速度快 10–70×，还能即插即用地包住任意现成预测器。

**[Parametric Prior Mapping Framework for Non-stationary Probabilistic Time Series Forecasting](parametric_prior_mapping_framework_for_non-stationary_probabilistic_time_series_.md)**

:   PPM 用一个轻量编码器从历史序列里推断出 context-aware 高斯先验，再用一个两层 MLP 把这个先验"推前"成完整的预测分布，用 KDE-NLL + 均值 MSE 联合训练，在七个时序基准上同时打过 DeepAR 和 NsDiff 等扩散模型，推理还快 2× 到 100×。

**[PATRA: Pattern-Aware Alignment and Balanced Reasoning for Time Series Question Answering](patra_pattern-aware_alignment_and_balanced_reasoning_for_time_series_question_an.md)**

:   针对时间序列问答 (TSQA)，PATRA 在表征端把序列显式拆成 full / trend / season 三类模式，并通过三组可学习对齐 token 与文本做深度交叉对齐；在训练端用 SFT + GRPO 两阶段强化学习，把判别式与生成式任务的奖励统一映射到 $[0,2]$ 解决难度失衡，从而在四类 TSQA 任务上全面超越文本 LLM、ChatTS 等多模态时序 LLM。

**[Position: Current Benchmarking Hinders Real Progress in Deep Learning for Time Series](position_current_benchmarking_hinders_real_progress_in_deep_learning_for_time_se.md)**

:   这篇位置论文系统揭示了当前时间序列预测基准测试的核心问题——**设计选择的差异**（全局 / 局部参数、预处理、外源变量、时间和空间处理）往往被当作"实现细节"忽视，导致不同论文的对比不公平；通过 44 数据集 × 7 SOTA × 多个参考架构的受控实验，证明这些差异的影响（5-15%）常常**超过具体序列建模层的贡献**（1-3%）。

**[QuITE: Query-based Irregular Time Series Embedding](quite_query-based_irregular_time_series_embedding.md)**

:   QuITE 是一个**即插即用的嵌入模块**——使用可学习的查询令牌通过自注意力直接聚合不规则观测，将任意 MTS 模型适配到不规则多变量时间序列（IMTS），无需改动架构或生成人工值；在 iTransformer + QuITE 上预测平均相对提升 54.7%。

**[Self-Supervised Dynamical System Representations for Physiological Time-Series](self-supervised_dynamical_system_representations_for_physiological_time-series.md)**

:   PULSE 把生理时间序列看成由「可迁移的系统参数 + 不可迁移的样本特异噪声」共同生成，提出用一个交叉重建目标——让从一段窗口推断出的系统表示去重建同系统的另一段独立样本——逼着编码器只保留共享的动力学、丢掉初始条件和噪声，从而学到对临床语义更可迁移的表示。

**[Semantics-Enhanced Retrieval-Augmented Time Series Forecasting](semantics-enhanced_retrieval-augmented_time_series_forecasting.md)**

:   SERAF 给检索增强的时序预测加了一条"语义检索"通路：把每段历史时间序列自动翻译成一句结构化文字描述（季节/趋势/波动），既按数值相似度、又按文本语义相似度去历史库里捞两套"相似过去 + 对应未来"，再自适应融合，从而在非平稳序列上检索到那些"数值上不像、但本质同型"的历史模式，在七个真实数据集上整体优于纯数值检索的 SOTA。

**[Simulation-Augmented Multi-Step Split Conformal Prediction for Aggregated Forecasts](simulation-augmented_multi-step_split_conformal_prediction_for_aggregated_foreca.md)**

:   针对"年度总量""同比增长率"这类聚合预测目标，本文提出 SA-MSCP——用扩展窗口交叉验证收集残差、再用区块自助法（block bootstrap）模拟大量未来路径，最后从聚合轨迹的经验分位数构造预测区间，在 M4 和一份私有数据上把聚合目标的经验覆盖率显著拉高（但区间也明显变宽）。

**[Sonar-TS: Search-Then-Verify Natural Language Querying for Time Series Databases](sonar-ts_search-then-verify_natural_language_querying_for_time_series_databases.md)**

:   面向"在海量时序数据库（TSDB）上用自然语言查询形态化意图"这一新问题，提出 **Sonar-TS** 神经符号框架：像主动声纳一样先用 SQL 在多尺度特征索引上"发射声波（ping）"粗筛候选窗口、再用 LLM 生成的 Python 程序"锁定（lock-on）"原始信号做精确验证（Search-Then-Verify），配套首个面向库级长历史的基准 **NLQTSBench**，在传统 Text-to-SQL 与时序大模型都失效的复杂时序查询上取得大幅领先（平均 0.61 vs 最强基线 0.16）。

**[Spatiotemporal Imputation with Graph-Informed Flow Matching](spatiotemporal_imputation_with_graph-informed_flow_matching.md)**

:   针对时空数据缺失补全中"RNN/GNN 迭代传播误差累积、扩散模型靠问题无关高斯先验且采样慢"的问题，本文提出 GiFlow——用可观测信号的时空滤波构造一个"图先验"替代高斯先验，使流匹配的起点更贴近目标分布、传输路径更短，再配一个融合空间注意力/时间注意力/时空传播的混合向量场，在合成与真实数据集（空气质量、交通）上一致超过 SOTA。

**[The Cost of Learning Under Multiple Change Points](the_cost_of_learning_under_multiple_change_points.md)**

:   本文提出 Anytime Tracking CUSUM (ATC) 算法，通过时变自适应阈值 + 选择性检测原理，在**无任何可检测性假设**（最小间距 / 最小跳幅）下达到近似最小最优的动态遗憾 $O(\sigma^2 (S+1) \log T)$；并首次形式化量化了多变点场景中"漏检带来的内生混淆"的对数级退化界。

**[Time-series Forecasting Through the Lens of Dynamics](time-series_forecasting_through_the_lens_of_dynamics.md)**

:   作者用 Allen 时间区间代数提出 PRO-DYN 命名法，把任意时序预测模型拆成"前处理 PRO → 动力学 DYN → 后处理 PRO"三段，发现两条经验规律：(i) DYN 必须**可学习且完整**才能打过 LTSF-Linear，(ii) DYN 必须放在**整个流程末端**（PRE-DYN 配置）才能吃到长 lookback 的红利；并通过给 Informer/FEDformer/MICN/FiLM 加一个线性 DYN 层让性能稳定提升，给 iTransformer/PatchTST/Crossformer 把 DYN 挪到前端则性能下降，用实验验证两条规律。

**[TimeOmni-VL: Unified Models for Time Series Understanding and Generation](timeomni-vl_unified_models_for_time_series_understanding_and_generation.md)**

:   TimeOmni-VL 通过把时间序列转换为**高保真图像**（Bi-TSI）+ 引入**理解引导的生成机制**（CoT 作为扩散条件），首次实现在统一多模态框架中同时达成时间序列**理解与生成**任务，预测和插补均达业界最优。

**[U-Cast: A Surprisingly Simple and Efficient Frontier Probabilistic AI Weather Forecasting](u-cast_a_surprisingly_simple_and_efficient_frontier_probabilistic_ai_weather_for.md)**

:   U-Cast 用**简单的 U-Net 主干** + **两阶段训练课程**（MAE 预训练 → CRPS 微调） + **MC-Dropout** 实现了与复杂专业模型（GenCast）相当的概率性天气预报能力，同时减少 10× 训练计算和推理延迟——颠覆"前沿性能必须复杂"的行业刻板印象。
