<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📈 时间序列

**🤖 AAAI2026** · 共 **14** 篇

**[A Unified Shape-Aware Foundation Model for Time Series Classification](a_unified_shape-aware_foundation_model_for_time_series_class.md)**

:   提出 UniShape——一个面向时间序列分类的基础模型，通过 shape-aware adapter 自适应聚合多尺度判别性子序列（shapelet），并结合原型对比预训练在实例和 shape 两个层面学习可迁移的 shapelet 表示，在 128 个 UCR 数据集上以 3.1M 参数达到 SOTA（平均准确率 87.08%），同时提供良好的分类可解释性。

**[Beyond Observations Reconstruction Error-Guided Irregularly Sampled Time Series ](beyond_observations_reconstruction_error-guided_irregularly_sampled_time_series_.md)**

:   提出 iTimER，利用模型自身的重建误差分布作为学习信号——从观测点估计误差分布后采样生成未观测时刻的伪观测值，通过 Wasserstein 距离对齐观测/伪观测区域的误差分布 + 对比学习，在不规则采样时序的分类、插值、预测任务上全面超越 SOTA。

**[C3Rl Rethinking The Combination Of Channel-Independence And Channel-Mixing From ](c3rl_rethinking_the_combination_of_channel-independence_and_channel-mixing_from_.md)**

:   提出 C3RL，基于 SimSiam 对比学习框架将通道独立（CI）和通道混合（CM）策略视为同一数据的两个转置视图构建正样本对，通过孪生网络联合表示学习和预测学习，将 CI 模型的最佳性能率从 43.6% 提升到 81.4%，CM 模型从 23.8% 提升到 76.3%。

**[Coherent Multi-Agent Trajectory Forecasting in Team Sports with CausalTraj](coherent_multi-agent_trajectory_forecasting_in_team_sports_with_causaltraj.md)**

:   提出 CausalTraj，一种时间因果的似然模型，用自回归逐步预测多智能体位移的高斯混合分布，强调联合预测指标（minJADE/minJFDE）而非独立评估每个个体，在 NBA、Basketball-U 和 Football-U 数据集上实现最佳联合指标。

**[Cometnet Contextual Motif-Guided Long-Term Time Series Forecasting](cometnet_contextual_motif-guided_long-term_time_series_forecasting.md)**

:   提出 CometNet，通过从完整历史序列中提取循环出现的"上下文 motif"构建 motif 库，再用 motif 引导的 MoE 架构动态关联当前窗口与相关motif进行预测，突破了有限回看窗口的感受野瓶颈，在8个数据集上显著超越 TimeMixer++、iTransformer 等 SOTA。

**[Counterfactual Explainable Ai Xai Method For Deep Learning-Based Multivariate Ti](counterfactual_explainable_ai_xai_method_for_deep_learning-based_multivariate_ti.md)**

:   提出 CONFETTI，一种面向多变量时序分类的多目标反事实解释方法，通过 CAM 引导的子序列提取 + NUN 值替换 + NSGA-III 多目标优化，同时优化预测置信度、近似性和稀疏性，在 7 个 UEA 数据集上置信度提升 ≥10%、稀疏性改善 ≥40%。

**[Deepboots Dual-Stream Residual Boosting For Drift-Resilient Time-Series Forecast](deepboots_dual-stream_residual_boosting_for_drift-resilient_time-series_forecast.md)**

:   提出 DeepBooTS，通过偏差-方差分解理论证明加权集成可降低方差从而缓解概念漂移，设计双流残差递减 boosting 架构，每个 block 的输出修正前一个 block 的残差，在多个数据集上平均提升 15.8%。

**[DEF: Detecting the Future — All-at-Once Event Sequence Forecasting with Horizon Matching](detecting_the_future_all-at-once_event_sequence_forecasting_with_horizon_matchin.md)**

:   提出 DEF (Detection-based Event Forecasting)，类比目标检测中的 DETR 思想，用多个并行预测头同时生成 K 个未来事件候选，通过匈牙利匹配损失将预测与真实事件最优对齐，解决自回归方法在长程预测中输出趋于重复/恒定的问题，在 5 个数据集上长程预测提升达 50%。

**[GBOC: Finding Time Series Anomalies using Granular-ball Vector Data Description](finding_time_series_anomalies_using_granular-ball_vector_data_description.md)**

:   提出 GBOC (Granular-Ball One-Class Network)，在潜空间中用密度引导的层次分裂构建粒球向量数据描述（GVDD），每个粒球代表局部正常行为原型，通过剪枝低质量粒球 + 最近粒球中心对齐训练 + 基于粒球距离的推理实现鲁棒的时序异常检测。

**[FreqCycle: A Multi-Scale Time-Frequency Analysis Method for Time Series Forecasting](freqcycle_a_multi-scale_time-frequency_analysis_method_for_time_series_forecasti.md)**

:   提出 FreqCycle，结合时域周期显式建模（FECF，学习共享日/周周期基底 + 自适应低通滤波）和频域中高频增强（SFPL，分段 STFT + 可学习加权融合），并扩展为 MFreqCycle 处理耦合多周期性，在 7 个数据集上达到 SOTA 精度且推理更快。

**[HN-MVTS: HyperNetwork-based Multivariate Time Series Forecasting](hn-mvts_hypernetwork-based_multivariate_time_series_forecasting.md)**

:   提出 HN-MVTS，用超网络从可学习的通道嵌入向量生成预测模型最后一层的权重，自适应地在通道独立（CI）和通道依赖（CD）之间插值，即插即用地提升 DLinear/PatchTST/TSMixer 等多种 SOTA 模型的性能，且不增加推理时间。

**[IdealTSF: Can Non-Ideal Data Contribute to Enhancing Time Series Forecasting?](idealtsf_can_non-ideal_data_contribute_to_enhancing_the_performance_of_time_seri.md)**

:   提出 IdealTSF 框架，通过三阶段渐进式设计——负样本预训练（用稳定分布+多尺度噪声+结构删除模拟非理想数据）、正样本训练（混合平滑插值修复数据）、ECOS 优化器（对抗扰动引导到平坦极值）——使基础 attention 模型在含噪声/缺失的时序数据上获得约 10% 的性能提升。

**[SELDON: Supernova Explosions Learned by Deep ODE Networks](seldon_supernova_explosions_learned_by_deep_ode_networks.md)**

:   提出SELDON，一种结合masked GRU-ODE编码器、隐式Neural ODE传播器和可解释高斯基函数解码器的连续时间VAE，用于稀疏、不规则采样的天文光变曲线预测，在仅观测20%数据时即可超越基线方法做出准确的多波段通量预测。

**[Urban Incident Prediction with Graph Neural Networks: Integrating Government Ratings and Crowdsourced Reports](urban_incident_prediction_with_graph_neural_networks_integrating_government_rati.md)**

:   提出 URBAN（多视图多输出GNN模型），联合利用稀疏但无偏的政府检查评级数据和密集但有偏的众包报告数据来预测城市事件的真实潜在状态，在纽约市960万+报告和100万+检查数据上验证，预测相关性比仅用报告数据高5.3倍。
