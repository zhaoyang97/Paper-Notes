---
title: >-
  ICML2025 时间序列论文汇总 · 21篇论文解读
description: >-
  21篇ICML2025的时间序列方向论文解读，涵盖时序预测、个性化生成、LLM、情感分析、医学影像、图神经网络等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "ICML2025"
  - "时间序列"
  - "论文解读"
  - "论文笔记"
  - "时序预测"
  - "个性化生成"
  - "LLM"
  - "情感分析"
  - "医学影像"
  - "图神经网络"
item_list:
  - u: "a_generalizable_physics-enhanced_state_space_model_for_long-term_dynamics_foreca/"
    t: "A Generalizable Physics-Enhanced State Space Model for Long-Term Dynamics Forecasting in Complex Environments"
  - u: "are_llms_prescient_a_continuous_evaluation_using_daily_news_as_the_oracle/"
    t: "Are LLMs Prescient? A Continuous Evaluation using Daily News as the Oracle"
  - u: "causal_discovery_from_conditionally_stationary_time_series/"
    t: "Causal Discovery from Conditionally Stationary Time Series"
  - u: "channel_normalization_for_time_series_channel_identification/"
    t: "Channel Normalization for Time Series Channel Identification"
  - u: "customizing_the_inductive_biases_of_softmax_attention_using_structured_matrices/"
    t: "Customizing the Inductive Biases of Softmax Attention using Structured Matrices"
  - u: "event-aware_sentiment_factors_from_llm-augmented_financial_tweets_a_transparent_/"
    t: "Event-Aware Sentiment Factors from LLM-Augmented Financial Tweets: A Transparent Framework for Interpretable Quant Trading"
  - u: "foundation_models_for_clinical_records_at_health_system_scale/"
    t: "Foundation Models for Clinical Records at Health System Scale"
  - u: "hyperimts_hypergraph_neural_network_for_irregular_multivariate_time_series_forec/"
    t: "HyperIMTS: Hypergraph Neural Network for Irregular Multivariate Time Series Forecasting"
  - u: "imts_is_worth_time_times_channel_patches_visual_masked_autoencoders_for_irregula/"
    t: "IMTS is Worth Time × Channel Patches: Visual Masked Autoencoders for Irregular Multivariate Time Series Prediction"
  - u: "learning_soft_sparse_shapes_for_efficient_time-series_classification/"
    t: "Learning Soft Sparse Shapes for Efficient Time-Series Classification"
  - u: "lyapunov_learning_at_the_onset_of_chaos/"
    t: "Lyapunov Learning at the Onset of Chaos"
  - u: "risk_and_cross_validation_in_ridge_regression_with_correlated_samples/"
    t: "Risk and Cross Validation in Ridge Regression with Correlated Samples"
  - u: "tcp-diffusion_a_multi-modal_diffusion_model_for_global_tropical_cyclone_precipit/"
    t: "TCP-Diffusion: A Multi-modal Diffusion Model for Global Tropical Cyclone Precipitation Forecasting with Change Awareness"
  - u: "temporal_query_network_for_efficient_multivariate_time_series_forecasting/"
    t: "TQNet: Temporal Query Network for Efficient Multivariate Time Series Forecasting"
  - u: "timepoint_accelerated_time_series_alignment_via_self-supervised_keypoint_and_des/"
    t: "TimePoint: Accelerated Time Series Alignment via Self-Supervised Keypoint and Descriptor Learning"
  - u: "timepro_efficient_multivariate_long-term_time_series_forecasting_with_variable-_/"
    t: "TimePro: Efficient Multivariate Long-term Time Series Forecasting with Variable- and Time-Aware Hyper-state"
  - u: "transpl_vq-code_transition_matrices_for_pseudo-labeling_of_time_series_unsupervi/"
    t: "TransPL: VQ-Code Transition Matrices for Pseudo-Labeling of Time Series Unsupervised Domain Adaptation"
  - u: "understanding_the_limits_of_deep_tabular_methods_with_temporal_shift/"
    t: "Understanding the Limits of Deep Tabular Methods with Temporal Shift"
  - u: "visionts_visual_masked_autoencoders_are_free-lunch_zero-shot_time_series_forecas/"
    t: "VisionTS: Visual Masked Autoencoders Are Free-Lunch Zero-Shot Time Series Forecasters"
  - u: "wave_weighted_autoregressive_varying_gate_for_time_series_forecasting/"
    t: "WAVE: Weighted Autoregressive Varying Gate for Time Series Forecasting"
  - u: "winner-takes-all_for_multivariate_probabilistic_time_series_forecasting/"
    t: "Winner-takes-all for Multivariate Probabilistic Time Series Forecasting"
item_total: 21
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📈 时间序列

**🧪 ICML2025** · **21** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (7)](../../CVPR2026/time_series/index.md) · [🔬 ICLR2026 (121)](../../ICLR2026/time_series/index.md) · [💬 ACL2026 (8)](../../ACL2026/time_series/index.md) · [🧪 ICML2026 (45)](../../ICML2026/time_series/index.md) · [🤖 AAAI2026 (31)](../../AAAI2026/time_series/index.md) · [🧠 NeurIPS2025 (54)](../../NeurIPS2025/time_series/index.md)

🔥 **高频主题：** 时序预测 ×14

**[A Generalizable Physics-Enhanced State Space Model for Long-Term Dynamics Forecasting in Complex Environments](a_generalizable_physics-enhanced_state_space_model_for_long-term_dynamics_foreca.md)**

:   提出 Phy-SSM，将部分已知的物理知识融入深度状态空间模型（SSM），通过动力学分解（已知/未知矩阵）和物理状态正则化，实现对噪声大、不规则采样数据的长期动力学精准预测与外推。

**[Are LLMs Prescient? A Continuous Evaluation using Daily News as the Oracle](are_llms_prescient_a_continuous_evaluation_using_daily_news_as_the_oracle.md)**

:   提出 Daily Oracle——一个每日自动从新闻生成预测性 QA 对的持续评估基准，系统性揭示了 LLM 预测能力随预训练数据过时而平滑退化的规律，TF 题平均降 21.55%、MC 题降 11.33%，且 RAG 也无法完全挽救。

**[Causal Discovery from Conditionally Stationary Time Series](causal_discovery_from_conditionally_stationary_time_series.md)**

:   提出 SDCI（State-Dependent Causal Inference）——处理条件平稳时间序列的因果发现方法，通过离散潜状态变量建模非平稳行为，实现状态依赖的因果结构恢复，在粒子交互、基因调控网络和 NBA 球员运动预测中验证有效性。

**[Channel Normalization for Time Series Channel Identification](channel_normalization_for_time_series_channel_identification.md)**

:   提出通道归一化（Channel Normalization, CN），通过为每个通道分配独立的仿射变换参数来增强时间序列模型的通道可辨识性（CID），并扩展出自适应版本 ACN（动态调整参数）和原型版本 PCN（支持未知/可变通道数），在多种时间序列模型上实现显著性能提升。

**[Customizing the Inductive Biases of Softmax Attention using Structured Matrices](customizing_the_inductive_biases_of_softmax_attention_using_structured_matrices.md)**

:   提出用高效结构化矩阵（BTT 和 MLR）替换 softmax attention 中的低秩打分函数，既解决了标准 attention 的低秩瓶颈问题，又通过 MLR 引入了距离依赖的计算偏置，在上下文回归、语言建模和长程时间序列预测上均取得改进。

**[Event-Aware Sentiment Factors from LLM-Augmented Financial Tweets: A Transparent Framework for Interpretable Quant Trading](event-aware_sentiment_factors_from_llm-augmented_financial_tweets_a_transparent_.md)**

:   利用大语言模型对金融推文进行多标签事件分类标注，将非结构化社交媒体文本转化为结构化、可解释的事件驱动量化因子，发现特定事件类别（如谣言/投机）具有显著的负Alpha信号（Sharpe ratio低至-0.38）。

**[Foundation Models for Clinical Records at Health System Scale](foundation_models_for_clinical_records_at_health_system_scale.md)**

:   提出 GPT-EHR，一种基于下一次就诊事件预测的生成式预训练框架，在 NYU Langone 129 万患者的纵向 EHR 数据上训练 decoder-only Transformer，零样本即可预测痴呆和膝骨关节炎发病，性能媲美全量微调的 BERT 基线，同时揭示并解决了重复事件 token 造成评估指标虚高的关键陷阱。

**[HyperIMTS: Hypergraph Neural Network for Irregular Multivariate Time Series Forecasting](hyperimts_hypergraph_neural_network_for_irregular_multivariate_time_series_forec.md)**

:   提出 HyperIMTS，利用超图结构表示不规则多元时间序列（IMTS）中的观测值和其依赖关系，通过三种消息传递机制（节点→超边、超边→超边、超边→节点）实现不规则性感知的时间和变量依赖学习，在 5 个 IMTS 数据集上达到 SOTA 且计算效率优于 padding 方法。

**[IMTS is Worth Time × Channel Patches: Visual Masked Autoencoders for Irregular Multivariate Time Series Prediction](imts_is_worth_time_times_channel_patches_visual_masked_autoencoders_for_irregula.md)**

:   提出 VIMTS 框架，将不规则多变量时间序列（IMTS）转化为 time × channel 的类图像 patch 结构，借助在大规模 RGB 图像上预训练的视觉 MAE 的稀疏多通道建模能力，结合 GCN 跨通道补全与粗到细预测策略，在 IMTS 预测任务上实现 SOTA 性能和强 few-shot 能力。

**[Learning Soft Sparse Shapes for Efficient Time-Series Classification](learning_soft_sparse_shapes_for_efficient_time-series_classification.md)**

:   提出 SoftShape 模型，用基于贡献分数的软稀疏化替代传统硬筛选 shapelet 的方式，结合 MoE 驱动的 intra-shape 和 shared expert 的 inter-shape 双模式时序模式学习，在 128 个 UCR 数据集上取得 SOTA 分类精度。

**[Lyapunov Learning at the Onset of Chaos](lyapunov_learning_at_the_onset_of_chaos.md)**

:   提出 Lyapunov Learning 算法，通过将神经网络视为动力系统并在损失函数中加入 Lyapunov 指数正则项，将网络推向混沌边缘（edge of chaos），从而在非平稳时间序列发生 regime shift 时实现快速自适应，在 Lorenz 系统实验中将 post-shift MSE 降低约 96%。

**[Risk and Cross Validation in Ridge Regression with Correlated Samples](risk_and_cross_validation_in_ridge_regression_with_correlated_samples.md)**

:   利用随机矩阵理论和自由概率技术，为训练样本具有任意相关性的高维岭回归推导了精确的风险渐近公式，提出了修正的广义交叉验证估计器 CorrGCV，在样本相关条件下准确预测样本外风险。

**[TCP-Diffusion: A Multi-modal Diffusion Model for Global Tropical Cyclone Precipitation Forecasting with Change Awareness](tcp-diffusion_a_multi-modal_diffusion_model_for_global_tropical_cyclone_precipit.md)**

:   提出 TCP-Diffusion，融合历史降水、多模态气象变量和 NWP 预报的条件扩散模型，通过邻接残差预测（ARP）机制预测降水变化量而非绝对值，在全球热带气旋降水预报中超越 ECMWF 等权威 NWP 方法。

**[TQNet: Temporal Query Network for Efficient Multivariate Time Series Forecasting](temporal_query_network_for_efficient_multivariate_time_series_forecasting.md)**

:   提出Temporal Query（TQ）技术——使用周期性移位的可学习向量作为注意力机制的query来捕获全局变量间相关模式，同时keys/values来自原始数据以保留样本级局部信息，在此基础上构建的TQNet仅使用单层多头注意力和浅层MLP，即在12个真实数据集上达到整体SOTA，且效率接近线性方法DLinear。

**[TimePoint: Accelerated Time Series Alignment via Self-Supervised Keypoint and Descriptor Learning](timepoint_accelerated_time_series_alignment_via_self-supervised_keypoint_and_des.md)**

:   提出 TimePoint——受 2D 关键点检测启发但针对 1D 信号重新设计的自监督方法，通过学习时间序列的关键点和描述子实现稀疏表示，将 DTW 应用于稀疏关键点而非完整信号，在大幅加速对齐的同时通常提升对齐精度。

**[TimePro: Efficient Multivariate Long-term Time Series Forecasting with Variable- and Time-Aware Hyper-state](timepro_efficient_multivariate_long-term_time_series_forecasting_with_variable-_.md)**

:   提出基于 Mamba 的 TimePro 模型，通过构建变量感知和时间感知的超级状态（hyper-state），自适应选择关键时间点来调节变量维度的隐状态，以线性复杂度实现高效的多变量长期时间序列预测。

**[TransPL: VQ-Code Transition Matrices for Pseudo-Labeling of Time Series Unsupervised Domain Adaptation](transpl_vq-code_transition_matrices_for_pseudo-labeling_of_time_series_unsupervi.md)**

:   提出 TransPL，通过将时间序列 patch 离散化为 VQ 码并构建类别-通道级转移矩阵，利用贝叶斯定理在目标域生成可解释伪标签，实现时间序列无监督域适应中平均 6.1% 准确率和 4.9% F1 的提升。

**[Understanding the Limits of Deep Tabular Methods with Temporal Shift](understanding_the_limits_of_deep_tabular_methods_with_temporal_shift.md)**

:   揭示深度表格模型在时间分布偏移下失败的根因——训练滞后与验证偏差导致模型选择失效，以及模型表示丢失周期/趋势信息——并提出改进的时序划分策略和基于傅里叶级数的即插即用时间嵌入方法。

**[VisionTS: Visual Masked Autoencoders Are Free-Lunch Zero-Shot Time Series Forecasters](visionts_visual_masked_autoencoders_are_free-lunch_zero-shot_time_series_forecas.md)**

:   将时间序列重构为图像，利用 ImageNet 预训练的 MAE（Masked Autoencoder）在**零样本**设置下进行时序预测，无需任何时序数据训练即可匹敌甚至超越专门的时序基础模型。

**[WAVE: Weighted Autoregressive Varying Gate for Time Series Forecasting](wave_weighted_autoregressive_varying_gate_for_time_series_forecasting.md)**

:   将经典统计学中的ARMA（自回归移动平均）结构引入自回归Transformer注意力机制，通过间接MA权重生成方法在不增加时间复杂度和参数量的前提下，解耦长短期时序模式，显著提升时间序列预测性能。

**[Winner-takes-all for Multivariate Probabilistic Time Series Forecasting](winner-takes-all_for_multivariate_probabilistic_time_series_forecasting.md)**

:   提出 TimeMCL，将 Multiple Choice Learning 的 Winner-Takes-All (WTA) 损失引入多变量概率时序预测，通过多头网络单次前向传播即可生成多样且具代表性的未来轨迹，兼顾预测质量与计算效率。
