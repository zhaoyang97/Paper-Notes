---
search:
  exclude: true
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📈 时间序列

**🔬 ICLR2026** · 共 **4** 篇

**[Adapt Data to Model: Adaptive Transformation Optimization for Domain-shared Time Series Foundation Models](adapt_data_to_model_adaptive_transformation_optimization_for_domain-shared_time_.md)**

:   提出TATO框架，通过自动优化数据预处理 pipeline（包括上下文裁切、尺度归一化、异常值校正），让冻结的大型时序模型（LTM）在不微调的情况下适配不同下游领域，平均降低MSE 13.6%，最高65.4%。

**[Contextual and Seasonal LSTMs for Time Series Anomaly Detection](contextual_and_seasonal_lstms_for_time_series_anomaly_detection.md)**

:   针对单变量时间序列中现有方法难以检测的"小幅点异常"和"缓慢上升异常"，提出 CS-LSTMs 双分支架构——S-LSTM 在频域建模周期性演化、C-LSTM 在时域捕捉局部趋势，结合小波噪声分解策略，在四个基准上全面超越 SOTA 且推理速度提升 40%。

**[Enhancing Multivariate Time Series Forecasting with Global Temporal Retrieval](enhancing_multivariate_time_series_forecasting_with_global_temporal_retrieval.md)**

:   提出 Global Temporal Retriever（GTR），一个轻量级即插即用模块，通过维护自适应全局周期嵌入并利用绝对时间索引检索对齐全局周期信息，使任意预测模型突破回看窗口限制，有效捕获远超输入长度的全局周期模式。

**[Reasoning on Time-Series for Financial Technical Analysis](reasoning_on_time-series_for_financial_technical_analysis.md)**

:   提出 Verbal Technical Analysis (VTA) 框架，结合 LLM 的语言推理能力与时间序列模型的模式捕捉能力，通过 Time-GRPO 强化学习优化推理链，并以推理属性条件化时序预测，实现了兼具准确性和可解释性的金融时间序列预测。
