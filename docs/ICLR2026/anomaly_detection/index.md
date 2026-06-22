---
title: >-
  ICLR2026 异常检测论文汇总 · 10篇论文解读
description: >-
  10篇ICLR2026的异常检测方向论文解读，涵盖异常检测、时序预测、少样本学习等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
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
  - u: "let_ood_feature_exploring_vast_predefined_classifiers/"
    t: "Let OOD Feature Exploring Vast Predefined Classifiers"
  - u: "llm_as_an_algorithmist_enhancing_anomaly_detectors_via_programmatic_synthesis/"
    t: "LLM as an Algorithmist: Enhancing Anomaly Detectors via Programmatic Synthesis"
  - u: "low_rank_transformer_for_multivariate_time_series_anomaly_detection_and_localiza/"
    t: "Low Rank Transformer for Multivariate Time Series Anomaly Detection and Localization"
  - u: "mrad_zero-shot_anomaly_detection_with_memory-driven_retrieval/"
    t: "MRAD: Zero-Shot Anomaly Detection with Memory-Driven Retrieval"
  - u: "pirn_prototypical-based_intra-modal_reconstruction_with_normality_communication_/"
    t: "PIRN: Prototypical-based Intra-modal Reconstruction with Normality Communication for Multi-modal Anomaly Detection."
  - u: "retabad_a_benchmark_for_restoring_semantic_context_in_tabular_anomaly_detection/"
    t: "ReTabAD: A Benchmark for Restoring Semantic Context in Tabular Anomaly Detection"
  - u: "uniod_a_universal_model_for_outlier_detection_across_diverse_domains/"
    t: "UniOD: A Universal Model for Outlier Detection across Diverse Domains"
item_total: 10
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔍 异常检测

**🔬 ICLR2026** · **10** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (7)](../../CVPR2026/anomaly_detection/index.md)

🔥 **高频主题：** 异常检测 ×7 · 时序预测 ×2 · 少样本学习 ×2

**[Adaptive Conformal Anomaly Detection with Time Series Foundation Models for Signal Monitoring](adaptive_conformal_anomaly_detection_with_time_series_foundation_models_for_sign.md)**

:   提出 W1-ACAS：一种 post-hoc、免微调的自适应共形异常检测框架，把预训练时序基础模型（TSFM）的预测误差转成可直接解释为误报率（p-value）的异常分数，并通过最小化 Wasserstein 距离在线学习权重，在非平稳数据下稳定控制误报。

**[Foundation Visual Encoders Are Secretly Few-Shot Anomaly Detectors](foundation_visual_encoders_are_secretly_few-shot_anomaly_detectors.md)**

:   作者发现冻结的基础视觉编码器其实"悄悄"已经能区分异常——图像中异常区域的面积与其特征到自然图像流形的距离成正相关，于是只在编码器之上训练一个轻量非线性投影算子（FOUNDAD），把异常特征拉回正常流形、再用投影前后差异打分，就在少样本、多类别工业异常检测上达到 SOTA。

**[Healthcare Insurance Fraud Detection via Continual Fiedler Vector Graph Model](healthcare_insurance_fraud_detection_via_continual_fiedler_vector_graph_model.md)**

:   ConFVG 用图拉普拉斯的第二小特征向量（Fiedler 向量）指导图自编码器的掩码策略来在标签稀缺时学结构感知表征，再用子图注意力融合 + Mean Teacher 在无标签的在线流里持续适应不断变化的欺诈模式，实现医保欺诈的实时检测。

**[Let OOD Feature Exploring Vast Predefined Classifiers](let_ood_feature_exploring_vast_predefined_classifiers.md)**

:   这篇论文提出 VPC，用一组固定的等角原型把 ID 类别和 OOD 样本分别拉到两个预定义子空间，再用两个子空间上的 L2 激活强度差做 OOD 分数，在 CIFAR 和 ImageNet-1k 的 OE 训练场景中稳定降低 FPR95。

**[LLM as an Algorithmist: Enhancing Anomaly Detectors via Programmatic Synthesis](llm_as_an_algorithmist_enhancing_anomaly_detectors_via_programmatic_synthesis.md)**

:   把 LLM 从"数据处理器"重新定位为"算法策略师"——它只看检测器的算法描述、不碰任何真实数据，就推理出该检测器的逻辑盲点并生成一段可跨数据集复用的 Python 合成代码，用来造出专门骗过这个检测器的"困难异常"，从而把原本只有正常样本的单类问题升级成更可分的两类问题，在 36 个表格异常检测基准上稳定提升五种主流检测器。

**[Low Rank Transformer for Multivariate Time Series Anomaly Detection and Localization](low_rank_transformer_for_multivariate_time_series_anomaly_detection_and_localiza.md)**

:   本文从理论上把 Transformer 编码器在多变量时间序列上的学习过程映射到经典 STAR 统计模型，进而提出对自注意力施加低秩正则的 ALoRa-T，用注意力矩阵的"秩"作为异常信号做检测，并借助可解释的贡献权重把异常回溯到具体变量做定位。

**[MRAD: Zero-Shot Anomaly Detection with Memory-Driven Retrieval](mrad_zero-shot_anomaly_detection_with_memory-driven_retrieval.md)**

:   MRAD 用「特征-标签记忆库的相似度检索」直接替代主流 ZSAD 的参数化拟合 $p(y|x)$，免训练版本就能打过 WinCLIP，再叠两层线性微调与区域先验注入的动态提示，便在 16 个工业/医疗数据集上刷到 SOTA。

**[PIRN: Prototypical-based Intra-modal Reconstruction with Normality Communication for Multi-modal Anomaly Detection.](pirn_prototypical-based_intra-modal_reconstruction_with_normality_communication_.md)**

:   PIRN 面向 RGB 图像与 3D surface normal 的少样本多模态工业异常检测，用自适应原型码本重建每个模态的正常特征，再通过跨模态正常性通信互补纹理和几何线索，在 MVTec 3D-AD、Eyecandies 和 Real-IAD D3 上取得更强的检测与定位表现。

**[ReTabAD: A Benchmark for Restoring Semantic Context in Tabular Anomaly Detection](retabad_a_benchmark_for_restoring_semantic_context_in_tabular_anomaly_detection.md)**

:   ReTabAD 是首个"上下文感知"的表格异常检测 benchmark：它把传统基准里被丢弃的文本语义（特征描述、领域知识、类别原文）重新还原回 20 个精选数据集，配齐 20 个跨经典/深度/LLM 的算法实现，并提出一个无需训练的零样本 LLM 框架，实验证明语义上下文能把检测 AUROC 平均提升 7.6 个百分点，让零样本 LLM 逼近 SOTA 训练方法。

**[UniOD: A Universal Model for Outlier Detection across Diverse Domains](uniod_a_universal_model_for_outlier_detection_across_diverse_domains.md)**

:   UniOD 用一批历史带标签数据集训练**一个**通用离群检测模型：先把任意维度/语义的表格数据集统一成"多尺度相似度图 + SVD 特征"，再用 GIN+GT 双路图网络把离群检测转成节点二分类，训练完成后对**任何未见过的新数据集免训练、免调参**直接打异常分数，在 30 个基准上平均 AUROC/AUPRC 超过 17 个基线且耗时更低。
