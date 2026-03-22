<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📈 时间序列

**🧠 NeurIPS2025** · 共 **8** 篇

**[A Graph Neural Network Approach for Localized and High-Resolution Temperature Forecasting](a_graph_neural_network_approach_for_localized_and_high-resolution_temperature_fo.md)**

:   提出一种 GCN-GRU 混合框架用于社区尺度（2.5km）高分辨率温度预报（1-48小时），在加拿大西南安大略三个区域上验证，最大区域平均 MAE 1.93°C、48h MAE 2.93°C，探索了 ClimateBERT 语言模型嵌入作为标准化输入的方案，为数据稀缺的全球南方地区提供可迁移的轻量级预报框架。

**[Abstain Mask Retain Core: Time Series Prediction by Adaptive Masking Loss with Representation Consistency](abstain_mask_retain_core_time_series_prediction_by_adaptive.md)**

:   揭示了时间序列预测中"适当截断历史数据反而提升精度"的反直觉现象（冗余特征学习问题），基于信息瓶颈理论提出AMRC方法，通过自适应掩码损失和表征一致性约束来抑制冗余特征学习，作为模型无关的训练框架在多种架构上显著提升性能。

**[AttentionPredictor: Temporal Patterns Matter for KV Cache Compression](attentionpredictor_temporal_patterns_matter_for_kv_cache_com.md)**

:   首个基于学习的 KV Cache 压缩方法，通过轻量级时空卷积模型预测下一 token 的注意力分数来动态识别关键 token，实现 13× KV cache 压缩和 5.6× cache offloading 加速，显著优于静态方法。

**[Connecting the Dots: A ML Ready Dataset for Ionospheric Forecasting](connecting_the_dots_a_machine_learning_ready_dataset_for_ionospheric_forecasting.md)**

:   构建了首个ML-ready电离层预测数据集，整合SDO、太阳风、地磁指数和TEC观测等多源异构数据为统一的时间-空间结构，并基准测试了多种时空ML架构用于TEC预测。

**[EcoCast: Spatio-Temporal Model for Continual Biodiversity Forecasting](ecocast_a_spatio-temporal_model_for_continual_biodiversity_and_climate_risk_fore.md)**

:   提出EcoCast，基于Transformer的时空模型，整合Sentinel-2、ERA5和GBIF数据进行近期物种分布预测，配合EWC持续学习机制，在非洲鸟类分布预测上F1从0.31提升至0.65。

**[How Foundational Are Foundation Models For Time Series Forecasting](how_foundational_are_foundation_models_for_time_series_forecasting.md)**

:   系统评估三个时间序列基础模型（TimesFM、TimeGPT、TiReX）在合成和真实数据集上的表现，发现其零样本能力与预训练域强相关、在真实分布偏移数据上微调后的 TSFM 并不一致优于从头训练的小模型（如 SAMFormer），挑战了 TSFM 的"one-size-fits-all"假设。

**[Sempo Lightweight Foundation Models For Time Series Forecasting](sempo_lightweight_foundation_models_for_time_series_forecasting.md)**

:   提出 SEMPO，一个仅 6.5M 参数、在 83M 时间点上预训练的轻量级时间序列基础模型，通过能量感知谱分解（EASD）充分利用低能量频率信号 + Mixture-of-Prompts Transformer（MoPFormer）学习异构时序模式，在零样本和少样本预测中分别将误差降低 12% 和 22%，超越数亿参数的 SOTA 模型。

**[Time-O1 Time-Series Forecasting Needs Transformed Label Alignment](time-o1_time-series_forecasting_needs_transformed_label_alignment.md)**

:   提出 Time-o1，一种为时序预测设计的变换增强学习目标：用 SVD 将标签序列变换为去相关且按重要性排序的成分，然后只对齐最重要的成分，从而同时消除标签自相关带来的偏差和减少多任务优化的复杂度，在多种预测模型上一致提升 SOTA 性能。
