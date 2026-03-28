<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📈 时间序列

**🧠 NeurIPS2025** · 共 **21** 篇

**[A Graph Neural Network Approach for Localized and High-Resolution Temperature Forecasting](a_graph_neural_network_approach_for_localized_and_high-resolution_temperature_fo.md)**

:   提出一种 GCN-GRU 混合框架用于社区尺度（2.5km）高分辨率温度预报（1-48小时），在加拿大西南安大略三个区域上验证，最大区域平均 MAE 1.93°C、48h MAE 2.93°C，探索了 ClimateBERT 语言模型嵌入作为标准化输入的方案，为数据稀缺的全球南方地区提供可迁移的轻量级预报框架。

**[Abstain Mask Retain Core: Time Series Prediction by Adaptive Masking Loss with Representation Consistency](abstain_mask_retain_core_time_series_prediction_by_adaptive.md)**

:   揭示了时间序列预测中"适当截断历史数据反而提升精度"的反直觉现象（冗余特征学习问题），基于信息瓶颈理论提出AMRC方法，通过自适应掩码损失和表征一致性约束来抑制冗余特征学习，作为模型无关的训练框架在多种架构上显著提升性能。

**[AERO: A Redirection-Based Optimization Framework Inspired by Judo for Robust Probabilistic Forecasting](aero_a_redirection-based_optimization_framework_inspired_by_judo_for_robust_prob.md)**

:   AERO 提出受柔道"借力重定向"启发的优化框架，通过梯度投影、能量守恒和干扰预测将对抗性扰动重定向为有利优化方向，在概率太阳能价格预测上展示更稳定的收敛。

**[AttentionPredictor: Temporal Patterns Matter for KV Cache Compression](attentionpredictor_temporal_patterns_matter_for_kv_cache_com.md)**

:   首个基于学习的 KV Cache 压缩方法，通过轻量级时空卷积模型预测下一 token 的注意力分数来动态识别关键 token，实现 13× KV cache 压缩和 5.6× cache offloading 加速，显著优于静态方法。

**[Benchmarking Probabilistic Time Series Forecasting Models on Neural Activity](benchmarking_probabilistic_time_series_forecasting_models_on_neural_activity.md)**

:   首次系统评测 12 个概率时间序列预测模型在小鼠皮层钙成像数据上的表现，发现 PatchTST 一致最优（信息性预测窗口达 1.5 秒），零样本基础模型（Chronos）完全失败但微调后竞争力强，揭示神经活动的内在可预测性上限约 1.5 秒。

**[Causal Masking on Spatial Data: An Information-Theoretic Case for Learning Spatial Datasets with Unimodal Language Models](causal_masking_on_spatial_data_an_information-theoretic_case_for_learning_spatia.md)**

:   证明在空间数据（国际象棋棋盘FEN状态）上直接应用因果掩蔽训练单模态LLM，其表现优于先将数据线性化为序列（PGN棋步）后再应用因果掩蔽——FEN+因果掩蔽的Llama 1.3B达到~2630 Elo，而PGN+因果仅~2130 Elo。

**[Channel Matters: Estimating Channel Influence for Multivariate Time Series](channel_matters_estimating_channel_influence_for_multivariate_time_series.md)**

:   提出 Channel-wise Influence (ChInf)——首个能量化多变量时间序列中不同通道对模型性能影响的影响函数方法，将 TracIn 从整体样本级分解到通道级，衍生出通道级异常检测和通道剪枝两个应用，在 5 个异常检测基准上排名第一。

**[Connecting the Dots: A ML Ready Dataset for Ionospheric Forecasting](connecting_the_dots_a_machine_learning_ready_dataset_for_ionospheric_forecasting.md)**

:   构建了首个ML-ready电离层预测数据集，整合SDO、太阳风、地磁指数和TEC观测等多源异构数据为统一的时间-空间结构，并基准测试了多种时空ML架构用于TEC预测。

**[Demandcast Global Hourly Electricity Demand Forecasting](demandcast_global_hourly_electricity_demand_forecasting.md)**

:   构建DemandCast——覆盖56个国家(2000-2025)的XGBoost全球小时电力需求预测框架，融合ERA5温度/GDP/人口等特征，归一化目标（年度分数）+时间分割评估，MAPE 9.2%。

**[EcoCast: Spatio-Temporal Model for Continual Biodiversity Forecasting](ecocast_a_spatio-temporal_model_for_continual_biodiversity_and_climate_risk_fore.md)**

:   提出EcoCast，基于Transformer的时空模型，整合Sentinel-2、ERA5和GBIF数据进行近期物种分布预测，配合EWC持续学习机制，在非洲鸟类分布预测上F1从0.31提升至0.65。

**[How Foundational Are Foundation Models For Time Series Forecasting](how_foundational_are_foundation_models_for_time_series_forecasting.md)**

:   系统评估三个时间序列基础模型（TimesFM、TimeGPT、TiReX）在合成和真实数据集上的表现，发现其零样本能力与预训练域强相关、在真实分布偏移数据上微调后的 TSFM 并不一致优于从头训练的小模型（如 SAMFormer），挑战了 TSFM 的"one-size-fits-all"假设。

**[Improving Time Series Forecasting via Instance-aware Post-hoc Revision (PIR)](improving_time_series_forecasting_via_instance-aware_post-hoc_revision.md)**

:   PIR 提出实例感知的事后修正框架——通过不确定性估计识别预测失败实例，用局部修正（协变量+外生变量 Transformer）和全局修正（检索相似训练实例加权平均）的残差组合，作为即插即用模块使 SparseTSF MSE 降低 25.87%，PatchTST 降低 8.99%。

**[RiverMamba: A State Space Model for Global River Discharge and Flood Forecasting](rivermamba_a_state_space_model_for_global_river_discharge_and_flood_forecasting.md)**

:   首个能在 0.05°（~5.5km）全球网格上做 7 天河流流量预报的深度学习模型——用空间填充曲线将 3D 时空点序列化后输入双向 Mamba block，结合 ECMWF HRES 气象预报，在 1.5-500 年重现期洪水检测上 F1 =0.459 超越 LSTM（0.358）和物理模型 GloFAS。

**[ScatterAD: Temporal-Topological Scattering Mechanism for Time Series Anomaly Detection](scatterad_temporal-topological_scattering_mechanism_for_time_series_anomaly_dete.md)**

:   提出"散射性"（scattering）作为异常检测的新归纳偏置——异常样本在高维表示空间中比正常样本分布更分散，通过双编码器（时间+拓扑）+ 超球面散射中心约束 + 对比融合学习时拓扑联合表示，在 6 个工业 IoT 数据集上 15/24 设置取得最佳。

**[Sempo Lightweight Foundation Models For Time Series Forecasting](sempo_lightweight_foundation_models_for_time_series_forecasting.md)**

:   提出 SEMPO，一个仅 6.5M 参数、在 83M 时间点上预训练的轻量级时间序列基础模型，通过能量感知谱分解（EASD）充分利用低能量频率信号 + Mixture-of-Prompts Transformer（MoPFormer）学习异构时序模式，在零样本和少样本预测中分别将误差降低 12% 和 22%，超越数亿参数的 SOTA 模型。

**[STaRFormer: Semi-Supervised Task-Informed Representation Learning via Dynamic Attention-Based Regional Masking](starformer_semi-supervised_task-informed_representation_learning_via_dynamic_att.md)**

:   提出 STaRFormer，通过动态注意力区域掩码（DAReM）识别任务关键区域并施加掩码扰动，配合批内+类内半监督对比学习将任务信息嵌入潜在表示，在 56 个数据集（含非平稳、不规则采样、分类/异常检测/回归）上全面超越 SOTA。

**[Strap Spatio-Temporal Pattern Retrieval For Out-Of-Distribution Generalization](strap_spatio-temporal_pattern_retrieval_for_out-of-distribution_generalization.md)**

:   提出 StRap，一个检索增强的时空模式学习框架，通过构建空间/时间/时空三维模式库并在推理时检索相似模式注入模型，在流式时空图 OOD 任务上平均提升 7.17%。

**[Synthetic Series-Symbol Data Generation For Time Series Foundation Models](synthetic_series-symbol_data_generation_for_time_series_foundation_models.md)**

:   提出 Series-Symbol (S²) 数据生成机制和 SymTime 基础模型，通过符号表达式与时序数据的双模态对比学习预训练，在纯合成数据上训练即可在 5 大时序分析任务上与真实数据预训练的基础模型竞争。

**[Syntsbench Rethinking Temporal Pattern Learning In Deep Learning Models For Time](syntsbench_rethinking_temporal_pattern_learning_in_deep_learning_models_for_time.md)**

:   提出 SynTSBench，一个基于合成数据的时序预测模型评估框架，通过可编程特征配置（趋势/周期/噪声/依赖/多变量）和理论最优基准，系统揭示当前深度学习模型在各类时序模式上的能力边界。

**[Time-IMM: A Dataset and Benchmark for Irregular Multimodal Multivariate Time Series](time-imm_a_dataset_and_benchmark_for_irregular_multimodal_multivariate_time_seri.md)**

:   构建 Time-IMM 数据集——首个按因果机制分类不规则性的多模态多变量时序 benchmark（9 种不规则类型分为触发/约束/伪影三大类，9 个数据集），配套 IMM-TSF 预测库支持异步多模态融合，实验表明显式建模多模态在不规则时序上平均降低 MSE 6.71%，最高达 38.38%。

**[Time-O1 Time-Series Forecasting Needs Transformed Label Alignment](time-o1_time-series_forecasting_needs_transformed_label_alignment.md)**

:   提出 Time-o1，一种为时序预测设计的变换增强学习目标：用 SVD 将标签序列变换为去相关且按重要性排序的成分，然后只对齐最重要的成分，从而同时消除标签自相关带来的偏差和减少多任务优化的复杂度，在多种预测模型上一致提升 SOTA 性能。
