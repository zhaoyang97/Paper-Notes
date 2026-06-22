---
title: >-
  AAAI2026 遥感论文汇总 · 7篇论文解读
description: >-
  7篇AAAI2026的遥感方向论文解读，涵盖推理、对抗鲁棒等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "AAAI2026"
  - "遥感"
  - "论文解读"
  - "论文笔记"
  - "推理"
  - "对抗鲁棒"
item_list:
  - u: "consistency-based_abductive_reasoning_over_perceptual_errors_of_multiple_pre-tra/"
    t: "Consistency-based Abductive Reasoning over Perceptual Errors of Multiple Pre-trained Models in Novel Environments"
  - u: "debiasing_machine_learning_predictions_for_causal_inference_without_additional_g/"
    t: "Debiasing Machine Learning Predictions for Causal Inference Without Additional Ground Truth Data"
  - u: "m3sr_multi-scale_multi-perceptual_mamba_for_efficient_spectral_reconstruction/"
    t: "M3SR: Multi-Scale Multi-Perceptual Mamba for Efficient Spectral Reconstruction"
  - u: "machine_learning_for_sustainable_rice_production_region-scale_monitoring_of_wate/"
    t: "Machine Learning for Sustainable Rice Production: Region-Scale Monitoring of Water-Saving Practices in Punjab, India"
  - u: "perceive_act_and_correct_confidence_is_not_enough_for_hyperspectral_classificati/"
    t: "Perceive, Act and Correct: Confidence Is Not Enough for Hyperspectral Classification"
  - u: "spatio-temporal_context_learning_with_temporal_difference_convolution_for_moving/"
    t: "TDCNet: Spatio-Temporal Context Learning with Temporal Difference Convolution for Moving IRSTD"
  - u: "uniabg_unified_adversarial_view_bridging_and_graph_correspondence_for_unsupervis/"
    t: "UniABG: Unified Adversarial View Bridging and Graph Correspondence for Unsupervised Cross-View Geo-Localization"
item_total: 7
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🛰️ 遥感

**🤖 AAAI2026** · **7** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (63)](../../CVPR2026/remote_sensing/index.md) · [🔬 ICLR2026 (11)](../../ICLR2026/remote_sensing/index.md) · [🧪 ICML2026 (3)](../../ICML2026/remote_sensing/index.md) · [🧠 NeurIPS2025 (12)](../../NeurIPS2025/remote_sensing/index.md) · [📹 ICCV2025 (11)](../../ICCV2025/remote_sensing/index.md) · [🧪 ICML2025 (7)](../../ICML2025/remote_sensing/index.md)

**[Consistency-based Abductive Reasoning over Perceptual Errors of Multiple Pre-trained Models in Novel Environments](consistency-based_abductive_reasoning_over_perceptual_errors_of_multiple_pre-tra.md)**

:   将多个预训练感知模型在新环境中的冲突预测建模为一致性溯因推理问题，通过逻辑程序编码各模型的错误检测规则和领域约束，寻找在保持不一致率低于阈值的同时最大化预测覆盖率的最优假设，在15个航拍测试集上平均F1提升13.6%。

**[Debiasing Machine Learning Predictions for Causal Inference Without Additional Ground Truth Data](debiasing_machine_learning_predictions_for_causal_inference_without_additional_g.md)**

:   针对ML卫星贫困预测因均值回归导致因果处理效应衰减的问题，提出两种无需新标注数据的后处理校正方法——线性校准校正(LCC)和Tweedie局部去收缩——使同一预测地图可在多个下游因果试验中复用（"一图多试"范式），Tweedie校正在模拟和DHS真实数据上实现近无偏的处理效应估计。

**[M3SR: Multi-Scale Multi-Perceptual Mamba for Efficient Spectral Reconstruction](m3sr_multi-scale_multi-perceptual_mamba_for_efficient_spectral_reconstruction.md)**

:   提出 M3SR，一种基于 Mamba 的多尺度多感知架构，通过空间-频率-光谱三分支并行融合结合 U-Net 多尺度结构，以 2.17M 参数和 100.9G FLOPs 的低计算代价在四个光谱重建基准上超越现有 SOTA 方法。

**[Machine Learning for Sustainable Rice Production: Region-Scale Monitoring of Water-Saving Practices in Punjab, India](machine_learning_for_sustainable_rice_production_region-scale_monitoring_of_wate.md)**

:   提出维度分类方法将水稻节水实践识别解耦为播种维度(DSR vs PTR)和灌溉维度(AWD vs CF)两个独立二分类任务，仅使用Sentinel-1 SAR影像实现播种F1=0.80和灌溉F1=0.74，并在旁遮普邦300万+地块上进行大规模推理，地区级采纳率与政府统计高度相关（Spearman ρ=0.69）。

**[Perceive, Act and Correct: Confidence Is Not Enough for Hyperspectral Classification](perceive_act_and_correct_confidence_is_not_enough_for_hyperspectral_classificati.md)**

:   提出 CABIN 框架，通过认知感知-行动-纠正的闭环学习机制，利用认识论不确定性（epistemic uncertainty）替代单纯的置信度来指导半监督高光谱图像分类中的样本选择与伪标签管理，在仅用 75% 标注的情况下显著超过全标注基线。

**[TDCNet: Spatio-Temporal Context Learning with Temporal Difference Convolution for Moving IRSTD](spatio-temporal_context_learning_with_temporal_difference_convolution_for_moving.md)**

:   提出 TDCNet，将时间差分和 3D 卷积融合为统一的时间差分卷积 (TDC)，通过重参数化实现推理零额外开销，配合 TDC 引导的时空注意力，在自建 IRSTD-UAV 数据集上 F1 达 97.12%（AP50 93.83%），同时发布 15,106 帧真实红外无人机数据集。

**[UniABG: Unified Adversarial View Bridging and Graph Correspondence for Unsupervised Cross-View Geo-Localization](uniabg_unified_adversarial_view_bridging_and_graph_correspondence_for_unsupervis.md)**

:   提出双阶段无监督跨视角地理定位框架 UniABG，通过对抗式视角桥接 (VAAB) 消除无人机/卫星视角域差距，再用异构图过滤校准 (HGFC) 净化跨视角关联，在 University-1652 上 Satellite→Drone AP 达 93.29%，超过多数有监督方法。
