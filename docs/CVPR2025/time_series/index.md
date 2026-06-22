---
title: >-
  CVPR2025 时间序列论文汇总 · 5篇论文解读
description: >-
  5篇CVPR2025的时间序列方向论文解读，涵盖时序预测、压缩/编码等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "CVPR2025"
  - "时间序列"
  - "论文解读"
  - "论文笔记"
  - "时序预测"
  - "压缩/编码"
item_list:
  - u: "competition-aware_cpc_forecasting_with_near-market_coverage/"
    t: "Competition-Aware CPC Forecasting with Near-Market Coverage"
  - u: "dejavid_encoder-agnostic_learned_temporal_matching_for_video_classification/"
    t: "DejaVid: Encoder-Agnostic Learned Temporal Matching for Video Classification"
  - u: "flavc_learned_video_compression_with_feature_level_attention/"
    t: "FLAVC: Learned Video Compression with Feature Level Attention"
  - u: "l2gtx_from_local_to_global_time_series_explanations/"
    t: "L2GTX: From Local to Global Time Series Explanations"
  - u: "learning_extremely_high_density_crowds_as_active_matters/"
    t: "Learning Extremely High Density Crowds as Active Matters"
item_total: 5
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📈 时间序列

**📷 CVPR2025** · **5** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (7)](../../CVPR2026/time_series/index.md) · [🔬 ICLR2026 (121)](../../ICLR2026/time_series/index.md) · [💬 ACL2026 (8)](../../ACL2026/time_series/index.md) · [🧪 ICML2026 (45)](../../ICML2026/time_series/index.md) · [🤖 AAAI2026 (31)](../../AAAI2026/time_series/index.md) · [🧠 NeurIPS2025 (54)](../../NeurIPS2025/time_series/index.md)

🔥 **高频主题：** 时序预测 ×2

**[Competition-Aware CPC Forecasting with Near-Market Coverage](competition-aware_cpc_forecasting_with_near-market_coverage.md)**

:   将付费搜索CPC预测重构为"部分可观测竞争下的预测"问题，通过语义邻域（Transformer嵌入）、行为邻域（DTW对齐）和地理意图三类竞争代理逼近不可观测的竞争状态，在1811个关键词×127周的Google Ads数据上显示竞争感知增强在中长期预测（6/12周）上显著优于单变量和弱上下文baseline。

**[DejaVid: Encoder-Agnostic Learned Temporal Matching for Video Classification](dejavid_encoder-agnostic_learned_temporal_matching_for_video_classification.md)**

:   提出 DejaVid，一种编码器无关的轻量级视频分类增强方法：将视频表示为变长时序嵌入序列 (TSE) 而非单个嵌入，通过学习每个时间步、每个特征维度的重要性权重，结合改进的可微分 DTW 算法做时序对齐分类，仅增加 <1.8% 参数就在 SSV2 达到 77.2%、K400 达到 89.1% 的 SOTA。

**[FLAVC: Learned Video Compression with Feature Level Attention](flavc_learned_video_compression_with_feature_level_attention.md)**

:   提出 FLAVC，在学习型视频压缩（LVC）框架中引入 Feature-level Attention（FLA）模块，通过将高层局部 patch embedding 转换为一维批次向量并替换传统注意力权重为全局上下文矩阵，实现全帧级全局感知，配合 Dense Overlapping Patcher 和 Transformer-CNN 混合编码器，在四个视频压缩数据集上取得 SOTA 率失真性能。

**[L2GTX: From Local to Global Time Series Explanations](l2gtx_from_local_to_global_time_series_explanations.md)**

:   L2GTX 提出一种完全模型无关的时间序列分类全局解释方法，通过聚合 LOMATCE 产生的参数化时间事件原语（PEPs）构建类级全局解释，在六个基准数据集上保持稳定的全局忠实度（R²）。

**[Learning Extremely High Density Crowds as Active Matters](learning_extremely_high_density_crowds_as_active_matters.md)**

:   本文将极端高密度人群（≥5人/m²）建模为主动物质（active matter），提出一种结合新型"人群材料"应力模型与Toner-Tu主动力的神经随机微分方程系统，通过混合欧拉-拉格朗日的CrowdMPM框架直接从野外视频光流中学习并预测人群动力学。
