<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📈 时间序列

**🔬 ICLR2026** · 共 **12** 篇

**[Adapt Data to Model: Adaptive Transformation Optimization for Domain-shared Time Series Foundation Models](adapt_data_to_model_adaptive_transformation_optimization_for_domain-shared_time_.md)**

:   提出TATO框架，通过自动优化数据预处理 pipeline（包括上下文裁切、尺度归一化、异常值校正），让冻结的大型时序模型（LTM）在不微调的情况下适配不同下游领域，平均降低MSE 13.6%，最高65.4%。

**[Contextual and Seasonal LSTMs for Time Series Anomaly Detection](contextual_and_seasonal_lstms_for_time_series_anomaly_detection.md)**

:   针对单变量时间序列中现有方法难以检测的"小幅点异常"和"缓慢上升异常"，提出 CS-LSTMs 双分支架构——S-LSTM 在频域建模周期性演化、C-LSTM 在时域捕捉局部趋势，结合小波噪声分解策略，在四个基准上全面超越 SOTA 且推理速度提升 40%。

**[CPiRi: Channel Permutation-Invariant Relational Interaction for Multivariate Time Series Forecasting](cpiri_channel_permutation-invariant_relational_interaction_for_multivariate_time_se.md)**

:   提出 CPiRi 框架，通过冻结预训练时序编码器 + 可训练置换等变空间模块 + 通道打乱训练策略，在不牺牲跨通道建模能力的前提下实现通道排序不变性（CPI），在多个交通基准上达到 SOTA。

**[EDINET-Bench: Evaluating LLMs on Complex Financial Tasks using Japanese Financial Statements](edinet-bench_evaluating_llms_on_complex_financial_tasks_using_japanese_financial.md)**

:   构建了基于日本 EDINET 十年年报的金融基准 EDINET-Bench，包含会计欺诈检测、盈利预测和行业分类三项专家级任务，发现即使是 SOTA LLM 也仅略优于逻辑回归。

**[Enhancing Multivariate Time Series Forecasting with Global Temporal Retrieval](enhancing_multivariate_time_series_forecasting_with_global_temporal_retrieval.md)**

:   提出 Global Temporal Retriever（GTR），一个轻量级即插即用模块，通过维护自适应全局周期嵌入并利用绝对时间索引检索对齐全局周期信息，使任意预测模型突破回看窗口限制，有效捕获远超输入长度的全局周期模式。

**[Free Energy Mixer](free_energy_mixer.md)**

:   提出 Free Energy Mixer (FEM)，通过将注意力的值读取重新定义为自由能（log-sum-exp）优化问题，实现了逐通道的值感知后验选择，克服了标准注意力"无损存储但有损读取"的固有瓶颈，可即插即用替换 softmax/线性注意力/RNN/SSM，在 NLP、视觉和时间序列任务上一致提升。

**[From Samples to Scenarios: A New Paradigm for Probabilistic Forecasting](from_samples_to_scenarios_a_new_paradigm_for_probabilistic_forecasting.md)**

:   提出 Probabilistic Scenarios 范式，用模型直接输出有限个 {场景, 概率} 对取代采样，并用仅含三层平行线性层的 TimePrism 在5个基准数据集上取得9/10 SOTA。

**[GTM: A General Time-series Model for Enhanced Representation Learning](gtm_a_general_time-series_model_for_enhanced_representation_learning_of_time-series.md)**

:   提出 GTM，一个通过频域注意力机制捕获时间粒度感知特征、并通过混合掩码统一重建与自回归预训练目标的通用时间序列基础模型，在预测、补全、异常检测、分类等多任务上均达到 SOTA。

**[HiVid: LLM-Guided Video Saliency For Content-Aware VOD And Live Streaming](hivid_llm-guided_video_saliency_for_content-aware_vod_and_live_streaming.md)**

:   提出 HiVid 框架，首次利用 LLM 作为人类代理为视频块生成内容重要性权重，通过感知模块（滑动窗口评分）、排序模块（LLM 引导归并排序去除评分偏差）和预测模块（多模态时间序列预测自适应延迟）实现内容感知流媒体传输，

**[Reasoning on Time-Series for Financial Technical Analysis](reasoning_on_time-series_for_financial_technical_analysis.md)**

:   提出 Verbal Technical Analysis (VTA) 框架，结合 LLM 的语言推理能力与时间序列模型的模式捕捉能力，通过 Time-GRPO 强化学习优化推理链，并以推理属性条件化时序预测，实现了兼具准确性和可解释性的金融时间序列预测。

**[T1: One-to-One Channel-Head Binding for Multivariate Time-Series Imputation](t1_one-to-one_channel-head_binding_for_multivariate_time-series_imputation.md)**

:   提出 T1，通过通道-头一对一绑定学习多变量时间序列的结构依赖，用交叉注意力捕获通道-时间关联，MAE 在 Energy/Traffic/Weather 上超 SOTA 5-12%。

**[TimeSliver : Symbolic-Linear Decomposition for Explainable Time Series Classification](timesliver_symbolic-linear_decomposition_for_explainable_time_series_classificat.md)**

:   提出 TimeSliver，将时间序列分解为 Hermite 多项式基+线性残差，通过 Hermite 特征结构分析生成可解释的时间归因，UCR 基准上精度 95%+ 同时完全可解释，top-K 归因与领域专家 87% 一致。
