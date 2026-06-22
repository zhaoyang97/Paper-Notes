---
title: >-
  ICLR2026 地球科学论文汇总 · 7篇论文解读
description: >-
  7篇ICLR2026的地球科学方向论文解读，涵盖超分辨率、多模态、对抗鲁棒等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICLR2026"
  - "地球科学"
  - "论文解读"
  - "论文笔记"
  - "超分辨率"
  - "多模态"
  - "对抗鲁棒"
item_list:
  - u: "geofar_geography-informed_frequency-aware_super-resolution_for_climate_data/"
    t: "GeoFAR: Geography-Informed Frequency-Aware Super-Resolution for Climate Data"
  - u: "omnifield_conditioned_neural_fields_for_robust_multimodal_spatiotemporal_learnin/"
    t: "OmniField: Conditioned Neural Fields for Robust Multimodal Spatiotemporal Learning"
  - u: "rainpro-8_an_efficient_deep_learning_model_to_estimate_rainfall_probabilities_ov/"
    t: "RainPro-8: An Efficient Deep Learning Model to Estimate Rainfall Probabilities Over 8 Hours"
  - u: "task-adaptive_parameter-efficient_fine-tuning_for_weather_foundation_models/"
    t: "Task-Adaptive Parameter-Efficient Fine-Tuning for Weather Foundation Models"
  - u: "the_seismic_wavefield_common_task_framework/"
    t: "The Seismic Wavefield Common Task Framework"
  - u: "tianquan-s2s_a_subseasonal-to-seasonal_global_weather_model_via_incorporate_clim/"
    t: "TianQuan-S2S：通过引入气候态构建次季节-季节全球天气预报模型"
  - u: "unveiling_the_mechanism_of_continuous_representation_full-waveform_inversion_a_w/"
    t: "揭示连续表示全波形反演的机制：一个基于波的神经正切核框架"
item_total: 7
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🌍 地球科学

**🔬 ICLR2026** · **7** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (2)](../../CVPR2026/earth_science/index.md) · [🧪 ICML2026 (2)](../../ICML2026/earth_science/index.md) · [🤖 AAAI2026 (2)](../../AAAI2026/earth_science/index.md) · [🧠 NeurIPS2025 (6)](../../NeurIPS2025/earth_science/index.md) · [📷 CVPR2025 (1)](../../CVPR2025/earth_science/index.md) · [🎞️ ECCV2024 (1)](../../ECCV2024/earth_science/index.md)

**[GeoFAR: Geography-Informed Frequency-Aware Super-Resolution for Climate Data](geofar_geography-informed_frequency-aware_super-resolution_for_climate_data.md)**

:   GeoFAR 将气候超分辨率中的低频偏置拆成“频率表达不足”和“地理条件缺失”两个问题，用 DCT 频率卷积核提取细粒度频带表示，再用经纬度与高程构成的地理隐式表示逐像素调制这些表示，从而在 ERA5、PRISM、CERRA 等多尺度气候降尺度任务上显著降低高频误差和复杂地形区域的预测偏差。

**[OmniField: Conditioned Neural Fields for Robust Multimodal Spatiotemporal Learning](omnifield_conditioned_neural_fields_for_robust_multimodal_spatiotemporal_learnin.md)**

:   OmniField 把"科学观测数据"（气候、空气污染）建模成一个**以可用模态为条件的连续神经场**，用多模态串扰块（MCT）+ 迭代跨模态精修（ICMR）在解码前对齐异构信号，无需打网格或插值预处理就能统一做重建/插值/预测/跨模态预测，相对 8 个强基线平均降低 22.4% 误差，且在重度传感器噪声下几乎不掉点。

**[RainPro-8: An Efficient Deep Learning Model to Estimate Rainfall Probabilities Over 8 Hours](rainpro-8_an_efficient_deep_learning_model_to_estimate_rainfall_probabilities_ov.md)**

:   RainPro-8 用一个仅 36.7M 参数的 MaxViT-U-Net，把雷达、卫星、数值天气预报（NWP）多源数据融合起来，通过「有序一致损失 + 单次前向预测全时刻」一次性输出欧洲 8 小时、高分辨率的概率性降水预报，精度超过现有 NWP、外推法和深度学习临近预报，同时推理比 MetNet 类方法快 48 倍。

**[Task-Adaptive Parameter-Efficient Fine-Tuning for Weather Foundation Models](task-adaptive_parameter-efficient_fine-tuning_for_weather_foundation_models.md)**

:   针对天气基础模型（WFM）微调，提出 WeatherPEFT：前向用 Task-Adaptive Dynamic Prompting（TADP）从编码器嵌入权重里抽出"变量×分辨率×时空"的任务特征生成软 prompt，反向用 Stochastic Fisher-Guided Adaptive Selection（SFAS）只更新 Fisher 信息最高的少量参数，在三个下游任务上用 ~0.3%–4% 的可训练参数追平甚至超过全量微调（Full-Tuning）。

**[The Seismic Wavefield Common Task Framework](the_seismic_wavefield_common_task_framework.md)**

:   这篇论文把 NLP/CV 里催生 ImageNet、AlphaZero 的"通用任务框架(Common Task Framework, CTF)"思路搬到地震学，提供三套多尺度地震波场数据集 + 一套 12 分制的隐藏测试集评测协议，并用它公平横评了 18 个主流科学机器学习模型——结果发现绝大多数复杂模型连"全预测 0"的朴素基线都打不过。

**[TianQuan-S2S：通过引入气候态构建次季节-季节全球天气预报模型](tianquan-s2s_a_subseasonal-to-seasonal_global_weather_model_via_incorporate_clim.md)**

:   TianQuan-S2S 把"长期气候平均态"通过注意力融合塞进 patch embedding、并在 ViT 每一层注入可学习的高斯噪声，专治数据驱动模型在 15–45 天次季节预报上"越预测越糊"的模型坍缩问题，在 ERA5 上同时超过数值模式 ECMWF-S2S 和数据驱动的 FuXi-S2S。

**[揭示连续表示全波形反演的机制：一个基于波的神经正切核框架](unveiling_the_mechanism_of_continuous_representation_full-waveform_inversion_a_w.md)**

:   本文把神经正切核（NTK）理论扩展到全波形反演（FWI），提出"基于波的 NTK"统一刻画传统 FWI 与连续表示 FWI（CR-FWI），用其特征值衰减速率解释了"为什么 INR 表示更鲁棒却高频收敛慢"，并据此设计出 INR 与多分辨率网格混合的 IG-FWI，在鲁棒性与收敛速度之间取得更优权衡。
