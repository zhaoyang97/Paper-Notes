---
title: >-
  ACL2026 时间序列论文汇总 · 8篇论文解读
description: >-
  8篇ACL2026的时间序列方向论文解读，涵盖时序预测、推理、问答、强化学习、LLM等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ACL2026"
  - "时间序列"
  - "论文解读"
  - "论文笔记"
  - "时序预测"
  - "推理"
  - "问答"
  - "强化学习"
  - "LLM"
item_list:
  - u: "a_unified_framework_for_modeling_heterogeneous_financial_data_via_dual-granulari/"
    t: "A Unified Framework for Modeling Heterogeneous Financial Data via Dual-Granularity Prompting"
  - u: "odtqa-fore_an_open-domain_tabular_question_answering_dataset_for_future_data_for/"
    t: "ODTQA-FoRe: An Open-Domain Tabular Question Answering Dataset for Future Data Forecasting and Reasoning"
  - u: "stk-adapter_incorporating_evolving_graph_and_event_chain_for_temporal_knowledge_/"
    t: "STK-Adapter: Incorporating Evolving Graph and Event Chain for Temporal Knowledge Graph Extrapolation"
  - u: "streasoner_empowering_llms_for_spatio-temporal_reasoning_in_time_series_via_spat/"
    t: "STReasoner: Empowering LLMs for Spatio-Temporal Reasoning in Time Series via Spatial-Aware Reinforcement Learning"
  - u: "temporal_leakage_in_search-engine_date-filtered_web_retrieval_a_retrospective_fo/"
    t: "Temporal Leakage in Search-Engine Date-Filtered Web Retrieval: A Retrospective Forecasting Case Study"
  - u: "test_of_time_rethinking_temporal_signal_of_benchmark_contamination/"
    t: "Test of Time: Rethinking Temporal Signal of Benchmark Contamination"
  - u: "time-ra_towards_time_series_reasoning_for_anomaly_diagnosis_with_llm_feedback/"
    t: "Time-RA: Towards Time Series Reasoning for Anomaly Diagnosis with LLM Feedback"
  - u: "tsaqa_time_series_analysis_question_and_answering_benchmark/"
    t: "TSAQA: Time Series Analysis Question And Answering Benchmark"
item_total: 8
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📈 时间序列

**💬 ACL2026** · **8** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (7)](../../CVPR2026/time_series/index.md) · [🔬 ICLR2026 (121)](../../ICLR2026/time_series/index.md) · [🧪 ICML2026 (45)](../../ICML2026/time_series/index.md) · [🤖 AAAI2026 (31)](../../AAAI2026/time_series/index.md) · [🧠 NeurIPS2025 (54)](../../NeurIPS2025/time_series/index.md) · [📹 ICCV2025 (4)](../../ICCV2025/time_series/index.md)

🔥 **高频主题：** 时序预测 ×5 · 推理 ×3

**[A Unified Framework for Modeling Heterogeneous Financial Data via Dual-Granularity Prompting](a_unified_framework_for_modeling_heterogeneous_financial_data_via_dual-granulari.md)**

:   提出FinLangNet框架，通过双模块架构（DeepFM处理静态特征 + 双粒度提示机制的Transformer处理时序行为）实现多尺度信用风险预测，在滴滴金融平台部署后实现KS提升6.3pp和坏账率下降9.9%。

**[ODTQA-FoRe: An Open-Domain Tabular Question Answering Dataset for Future Data Forecasting and Reasoning](odtqa-fore_an_open-domain_tabular_question_answering_dataset_for_future_data_for.md)**

:   ODTQA-FoRe 提出面向未来数值预测和预测后推理的开放域表格问答任务，并用 TimeFore 三代理框架把表格检索、SQL 取数、专用时间序列预测和答案规范化串成一个可评测 baseline。

**[STK-Adapter: Incorporating Evolving Graph and Event Chain for Temporal Knowledge Graph Extrapolation](stk-adapter_incorporating_evolving_graph_and_event_chain_for_temporal_knowledge_.md)**

:   本文提出 STK-Adapter，通过在 LLM 每一层嵌入三个 MoE 模块（ST-MoE 捕捉时空结构、EA-MoE 建模事件链语义、CMA-MoE 深度跨模态对齐），解决现有方法将 TKG 嵌入与 LLM 浅层对齐导致的时空信息丢失和逐层稀释问题，在四个基准数据集上显著超越 SOTA。

**[STReasoner: Empowering LLMs for Spatio-Temporal Reasoning in Time Series via Spatial-Aware Reinforcement Learning](streasoner_empowering_llms_for_spatio-temporal_reasoning_in_time_series_via_spat.md)**

:   STReasoner 用网络 SDE 合成带图结构和文本语义的时空时间序列数据，再通过时间序列编码器、三阶段训练和空间感知 S-GRPO，让 LLM 学会基于时间动态与空间依赖做显式推理。

**[Temporal Leakage in Search-Engine Date-Filtered Web Retrieval: A Retrospective Forecasting Case Study](temporal_leakage_in_search-engine_date-filtered_web_retrieval_a_retrospective_fo.md)**

:   本文对 Google 和 DuckDuckGo 的日期过滤器进行系统审计，发现搜索引擎日期过滤在回顾性预测评估中严重失效——71%（Google）和 81%（DuckDuckGo）的问题至少有一个页面包含重大截止日期后信息泄漏，导致预测 Brier 分数从 0.24 虚降至 0.10。

**[Test of Time: Rethinking Temporal Signal of Benchmark Contamination](test_of_time_rethinking_temporal_signal_of_benchmark_contamination.md)**

:   这篇论文证明“cutoff 之后性能下降”并不是稳健的 benchmark contamination 证据：同一批源文档只要从原文填空题换成 LLM 改写题，时间衰减信号就会显著改变甚至消失。

**[Time-RA: Towards Time Series Reasoning for Anomaly Diagnosis with LLM Feedback](time-ra_towards_time_series_reasoning_for_anomaly_diagnosis_with_llm_feedback.md)**

:   定义 Time-RA 新任务将时间序列异常检测从二分类升级为生成式推理诊断（检测+分类+原因解释），构建首个包含约 4 万样本、10 个领域、20 种异常类型的多模态基准 RATs40K，并通过 AI 反馈标注流程和 LLM 微调验证了该范式的可行性。

**[TSAQA: Time Series Analysis Question And Answering Benchmark](tsaqa_time_series_analysis_question_and_answering_benchmark.md)**

:   TSAQA 是一个统一的时间序列问答基准：它把 6 类时序分析任务（异常检测、分类、表征、比较、数据变换、时间关系）全部铸造成 3 种封闭式题型（判断题 TF、选择题 MC、以及新提出的拼图题 PZ），跨 13 个领域共 210k 样本，用统一协议零样本评测 LLM 与时序基础模型——结果显示即便最强商用模型 Gemini-2.5-Flash 也只有 65.08 的平均准确率，基准仍有很大挑战空间。
