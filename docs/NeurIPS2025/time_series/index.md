<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📈 时间序列

**🧠 NeurIPS2025** · 共 **15** 篇

**[A Graph Neural Network Approach for Localized and High-Resolution Temperature Forecasting](a_graph_neural_network_approach_for_localized_and_high-resolution_temperature_fo.md)**

:   提出一种 GCN-GRU 混合框架用于社区尺度（2.5km）高分辨率温度预报（1-48小时），在加拿大西南安大略三个区域上验证，最大区域平均 MAE 1.93°C、48h MAE 2.93°C，探索了 ClimateBERT 语言模型嵌入作为标准化输入的方案，为数据稀缺的全球南方地区提供可迁移的轻量级预报框架。

**[Abstain Mask Retain Core: Time Series Prediction by Adaptive Masking Loss with Representation Consistency](abstain_mask_retain_core_time_series_prediction_by_adaptive.md)**

:   揭示了时间序列预测中"适当截断历史数据反而提升精度"的反直觉现象（冗余特征学习问题），基于信息瓶颈理论提出AMRC方法，通过自适应掩码损失和表征一致性约束来抑制冗余特征学习，作为模型无关的训练框架在多种架构上显著提升性能。

**[AERO: A Redirection-Based Optimization Framework Inspired by Judo for Robust Probabilistic Forecasting](aero_a_redirection-based_optimization_framework_inspired_by_judo_for_robust_prob.md)**

:   提出AERO物理启发优化框架——不抵抗而是"重定向"扰动（类比柔道的借力打力），通过15条公理定义能量守恒和自适应学习率的重定向力场 $\mathbf{F}_{red} = \mathbf{F}_{dist} \times \mathbf{R}(\theta)$，在太阳能概率预报中比标准优化器更鲁棒。

**[AttentionPredictor: Temporal Patterns Matter for KV Cache Compression](attentionpredictor_temporal_patterns_matter_for_kv_cache_com.md)**

:   首个基于学习的 KV Cache 压缩方法，通过轻量级时空卷积模型预测下一 token 的注意力分数来动态识别关键 token，实现 13× KV cache 压缩和 5.6× cache offloading 加速，显著优于静态方法。

**[Benchmarking Probabilistic Time Series Forecasting Models on Neural Activity](benchmarking_probabilistic_time_series_forecasting_models_on_neural_activity.md)**

:   对8个深度学习模型和4个统计方法在小鼠皮层自发钙成像活动上进行概率时序预测评估，PatchTST在0.5-1.5秒预测范围内显著优于所有baseline，但超过~50步（1.5秒）后性能不再优于预测均值+标准差。

**[Causal Masking on Spatial Data: An Information-Theoretic Case for Learning Spatial Datasets with Unimodal Language Models](causal_masking_on_spatial_data_an_information-theoretic_case_for_learning_spatia.md)**

:   证明在空间数据（国际象棋棋盘FEN状态）上直接应用因果掩蔽训练单模态LLM，其表现优于先将数据线性化为序列（PGN棋步）后再应用因果掩蔽——FEN+因果掩蔽的Llama 1.3B达到~2630 Elo，而PGN+因果仅~2130 Elo。

**[Channel Matters: Estimating Channel Influence for Multivariate Time Series](channel_matters_estimating_channel_influence_for_multivariate_time_series.md)**

:   提出 Channel-wise Influence (ChInf)——首个能量化多变量时间序列中不同通道对模型性能影响的影响函数方法，将 TracIn 从整体样本级分解到通道级，衍生出通道级异常检测和通道剪枝两个应用，在 5 个异常检测基准上排名第一。

**[Connecting the Dots: A ML Ready Dataset for Ionospheric Forecasting](connecting_the_dots_a_machine_learning_ready_dataset_for_ionospheric_forecasting.md)**

:   构建了首个ML-ready电离层预测数据集，整合SDO、太阳风、地磁指数和TEC观测等多源异构数据为统一的时间-空间结构，并基准测试了多种时空ML架构用于TEC预测。

**[EcoCast: Spatio-Temporal Model for Continual Biodiversity Forecasting](ecocast_a_spatio-temporal_model_for_continual_biodiversity_and_climate_risk_fore.md)**

:   提出EcoCast，基于Transformer的时空模型，整合Sentinel-2、ERA5和GBIF数据进行近期物种分布预测，配合EWC持续学习机制，在非洲鸟类分布预测上F1从0.31提升至0.65。

**[How Foundational Are Foundation Models For Time Series Forecasting](how_foundational_are_foundation_models_for_time_series_forecasting.md)**

:   系统评估三个时间序列基础模型（TimesFM、TimeGPT、TiReX）在合成和真实数据集上的表现，发现其零样本能力与预训练域强相关、在真实分布偏移数据上微调后的 TSFM 并不一致优于从头训练的小模型（如 SAMFormer），挑战了 TSFM 的"one-size-fits-all"假设。

**[Sempo Lightweight Foundation Models For Time Series Forecasting](sempo_lightweight_foundation_models_for_time_series_forecasting.md)**

:   提出 SEMPO，一个仅 6.5M 参数、在 83M 时间点上预训练的轻量级时间序列基础模型，通过能量感知谱分解（EASD）充分利用低能量频率信号 + Mixture-of-Prompts Transformer（MoPFormer）学习异构时序模式，在零样本和少样本预测中分别将误差降低 12% 和 22%，超越数亿参数的 SOTA 模型。

**[Strap Spatio-Temporal Pattern Retrieval For Out-Of-Distribution Generalization](strap_spatio-temporal_pattern_retrieval_for_out-of-distribution_generalization.md)**

:   提出 StRap，一个检索增强的时空模式学习框架，通过构建空间/时间/时空三维模式库并在推理时检索相似模式注入模型，在流式时空图 OOD 任务上平均提升 7.17%。

**[Synthetic Series-Symbol Data Generation For Time Series Foundation Models](synthetic_series-symbol_data_generation_for_time_series_foundation_models.md)**

:   提出 Series-Symbol (S²) 数据生成机制和 SymTime 基础模型，通过符号表达式与时序数据的双模态对比学习预训练，在纯合成数据上训练即可在 5 大时序分析任务上与真实数据预训练的基础模型竞争。

**[Syntsbench Rethinking Temporal Pattern Learning In Deep Learning Models For Time](syntsbench_rethinking_temporal_pattern_learning_in_deep_learning_models_for_time.md)**

:   提出 SynTSBench，一个基于合成数据的时序预测模型评估框架，通过可编程特征配置（趋势/周期/噪声/依赖/多变量）和理论最优基准，系统揭示当前深度学习模型在各类时序模式上的能力边界。

**[Time-O1 Time-Series Forecasting Needs Transformed Label Alignment](time-o1_time-series_forecasting_needs_transformed_label_alignment.md)**

:   提出 Time-o1，一种为时序预测设计的变换增强学习目标：用 SVD 将标签序列变换为去相关且按重要性排序的成分，然后只对齐最重要的成分，从而同时消除标签自相关带来的偏差和减少多任务优化的复杂度，在多种预测模型上一致提升 SOTA 性能。
