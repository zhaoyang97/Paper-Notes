---
title: >-
  ICLR2026 异常检测论文汇总 · 7篇论文解读
description: >-
  7篇ICLR2026的异常检测方向论文解读，涵盖异常检测、时序预测、少样本学习等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICLR2026"
  - "异常检测"
  - "论文解读"
  - "论文笔记"
  - "时序预测"
  - "少样本学习"
item_list:
  - u: "adaptive_conformal_anomaly_detection_with_time_series_foundation_models_for_sign/"
    t: "Adaptive Conformal Anomaly Detection with Time Series Foundation Models for Signal Monitoring"
  - u: "foundation_visual_encoders_are_secretly_few-shot_anomaly_detectors/"
    t: "Foundation Visual Encoders Are Secretly Few-Shot Anomaly Detectors"
  - u: "healthcare_insurance_fraud_detection_via_continual_fiedler_vector_graph_model/"
    t: "Healthcare Insurance Fraud Detection via Continual Fiedler Vector Graph Model"
  - u: "low_rank_transformer_for_multivariate_time_series_anomaly_detection_and_localiza/"
    t: "Low Rank Transformer for Multivariate Time Series Anomaly Detection and Localization"
  - u: "mrad_zero-shot_anomaly_detection_with_memory-driven_retrieval/"
    t: "MRAD: Zero-Shot Anomaly Detection with Memory-Driven Retrieval"
  - u: "retabad_a_benchmark_for_restoring_semantic_context_in_tabular_anomaly_detection/"
    t: "ReTabAD: A Benchmark for Restoring Semantic Context in Tabular Anomaly Detection"
  - u: "uniod_a_universal_model_for_outlier_detection_across_diverse_domains/"
    t: "UniOD: A Universal Model for Outlier Detection across Diverse Domains"
item_total: 7
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔍 异常检测

**🔬 ICLR2026** · **7** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (7)](../../CVPR2026/anomaly_detection/index.md)

🔥 **高频主题：** 异常检测 ×5 · 时序预测 ×2 · 少样本学习 ×2

**[Adaptive Conformal Anomaly Detection with Time Series Foundation Models for Signal Monitoring](adaptive_conformal_anomaly_detection_with_time_series_foundation_models_for_sign.md)**

:   提出 W1-ACAS：一种 post-hoc、免微调的自适应共形异常检测框架，把预训练时序基础模型（TSFM）的预测误差转成可直接解释为误报率（p-value）的异常分数，并通过最小化 Wasserstein 距离在线学习权重，在非平稳数据下稳定控制误报。

**[Foundation Visual Encoders Are Secretly Few-Shot Anomaly Detectors](foundation_visual_encoders_are_secretly_few-shot_anomaly_detectors.md)**

:   作者发现冻结的基础视觉编码器其实"悄悄"已经能区分异常——图像中异常区域的面积与其特征到自然图像流形的距离成正相关，于是只在编码器之上训练一个轻量非线性投影算子（FOUNDAD），把异常特征拉回正常流形、再用投影前后差异打分，就在少样本、多类别工业异常检测上达到 SOTA。

**[Healthcare Insurance Fraud Detection via Continual Fiedler Vector Graph Model](healthcare_insurance_fraud_detection_via_continual_fiedler_vector_graph_model.md)**

:   ConFVG 用图拉普拉斯的第二小特征向量（Fiedler 向量）指导图自编码器的掩码策略来在标签稀缺时学结构感知表征，再用子图注意力融合 + Mean Teacher 在无标签的在线流里持续适应不断变化的欺诈模式，实现医保欺诈的实时检测。

**[Low Rank Transformer for Multivariate Time Series Anomaly Detection and Localization](low_rank_transformer_for_multivariate_time_series_anomaly_detection_and_localiza.md)**

:   本文从理论上把 Transformer 编码器在多变量时间序列上的学习过程映射到经典 STAR 统计模型，进而提出对自注意力施加低秩正则的 ALoRa-T，用注意力矩阵的"秩"作为异常信号做检测，并借助可解释的贡献权重把异常回溯到具体变量做定位。

**[MRAD: Zero-Shot Anomaly Detection with Memory-Driven Retrieval](mrad_zero-shot_anomaly_detection_with_memory-driven_retrieval.md)**

:   MRAD 用「特征-标签记忆库的相似度检索」直接替代主流 ZSAD 的参数化拟合 $p(y|x)$，免训练版本就能打过 WinCLIP，再叠两层线性微调与区域先验注入的动态提示，便在 16 个工业/医疗数据集上刷到 SOTA。

**[ReTabAD: A Benchmark for Restoring Semantic Context in Tabular Anomaly Detection](retabad_a_benchmark_for_restoring_semantic_context_in_tabular_anomaly_detection.md)**

:   ReTabAD 是首个"上下文感知"的表格异常检测 benchmark：它把传统基准里被丢弃的文本语义（特征描述、领域知识、类别原文）重新还原回 20 个精选数据集，配齐 20 个跨经典/深度/LLM 的算法实现，并提出一个无需训练的零样本 LLM 框架，实验证明语义上下文能把检测 AUROC 平均提升 7.6 个百分点，让零样本 LLM 逼近 SOTA 训练方法。

**[UniOD: A Universal Model for Outlier Detection across Diverse Domains](uniod_a_universal_model_for_outlier_detection_across_diverse_domains.md)**

:   UniOD 用一批历史带标签数据集训练**一个**通用离群检测模型：先把任意维度/语义的表格数据集统一成"多尺度相似度图 + SVD 特征"，再用 GIN+GT 双路图网络把离群检测转成节点二分类，训练完成后对**任何未见过的新数据集免训练、免调参**直接打异常分数，在 30 个基准上平均 AUROC/AUPRC 超过 17 个基线且耗时更低。
