---
title: >-
  ECCV2024 信号/通信论文汇总 · 6篇论文解读
description: >-
  6篇ECCV2024的信号/通信方向论文解读，收录 Defect Spectrum、Optimizing Illuminant Estimati、PYRA等。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "ECCV2024"
  - "信号/通信"
  - "论文解读"
  - "论文笔记"
item_list:
  - u: "defect_spectrum_a_granular_look_of_large-scale_defect_datasets_with_rich_semanti/"
    t: "Defect Spectrum: A Granular Look of Large-Scale Defect Datasets with Rich Semantics"
  - u: "optimizing_illuminant_estimation_in_dual-exposure_hdr_imaging/"
    t: "Optimizing Illuminant Estimation in Dual-Exposure HDR Imaging"
  - u: "pyra_parallel_yielding_re-activation_for_training-inference_efficient_task_adapt/"
    t: "PYRA: Parallel Yielding Re-Activation for Training-Inference Efficient Task Adaptation"
  - u: "querycdr_query-based_controllable_distortion_rectification_network_for_fisheye_i/"
    t: "QueryCDR: Query-based Controllable Distortion Rectification Network for Fisheye Images"
  - u: "raw-adapter_adapting_pre-trained_visual_model_to_camera_raw_images/"
    t: "RAW-Adapter: Adapting Pre-trained Visual Model to Camera RAW Images"
  - u: "unsupervised_exposure_correction/"
    t: "Unsupervised Exposure Correction"
item_total: 6
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📡 信号/通信

**🎞️ ECCV2024** · **6** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (2)](../../CVPR2026/signal_comm/index.md) · [🔬 ICLR2026 (7)](../../ICLR2026/signal_comm/index.md) · [🧪 ICML2026 (2)](../../ICML2026/signal_comm/index.md) · [🤖 AAAI2026 (3)](../../AAAI2026/signal_comm/index.md) · [🧠 NeurIPS2025 (5)](../../NeurIPS2025/signal_comm/index.md) · [📹 ICCV2025 (3)](../../ICCV2025/signal_comm/index.md)

**[Defect Spectrum: A Granular Look of Large-Scale Defect Datasets with Rich Semantics](defect_spectrum_a_granular_look_of_large-scale_defect_datasets_with_rich_semanti.md)**

:   本文构建了 Defect Spectrum 数据集，在四个工业基准之上提供精细的、语义丰富的、大规模的多类缺陷标注（125种缺陷类别，3518+1920张），并提出两阶段扩散生成器 Defect-Gen 在少样本条件下合成高质量多样性缺陷图像，合成数据将缺陷分割 mIoU 最高提升 9.85。

**[Optimizing Illuminant Estimation in Dual-Exposure HDR Imaging](optimizing_illuminant_estimation_in_dual-exposure_hdr_imaging.md)**

:   本文提出从双曝光 HDR 图像对中提取一种简洁的双曝光特征（DEF），并基于此构建了两个超轻量级光源估计器 EMLP 和 ECCC，在仅使用几百到几千个参数的情况下即可达到或超越需要数十万参数的先前方法的性能。

**[PYRA: Parallel Yielding Re-Activation for Training-Inference Efficient Task Adaptation](pyra_parallel_yielding_re-activation_for_training-inference_efficient_task_adapt.md)**

:   本文提出 PYRA，通过并行生成解耦的自适应调制权重并以 re-activation 策略调节待合并 token 的特征，实现了 Vision Transformer 在下游任务适配时同时兼顾训练效率（仅调 0.4% 参数）和推理效率（约 1.7-3.2 倍加速），性能与不压缩的 PEFT 方法持平甚至更优。

**[QueryCDR: Query-based Controllable Distortion Rectification Network for Fisheye Images](querycdr_query-based_controllable_distortion_rectification_network_for_fisheye_i.md)**

:   提出QueryCDR网络，通过可学习查询机制（DLQM）和两种可控调制模块（CCMB/CAMB），首次实现不同畸变程度的鱼眼图像在**不重训**的情况下进行高质量可控矫正。

**[RAW-Adapter: Adapting Pre-trained Visual Model to Camera RAW Images](raw-adapter_adapting_pre-trained_visual_model_to_camera_raw_images.md)**

:   提出 RAW-Adapter，通过输入级适配器（可学习 ISP 阶段）和模型级适配器（ISP 中间特征注入骨干网络），以极小参数量（0.2-0.8M）将 sRGB 预训练模型高效适配到 Camera RAW 图像，在正常光/暗光/过曝等多种光照条件下的检测和分割任务上达到 SOTA。

**[Unsupervised Exposure Correction](unsupervised_exposure_correction.md)**

:   提出首个无监督曝光校正（UEC）方法，利用ISP管线自由生成的多曝光序列让图像互为ground truth进行训练，设计仅含19K参数的像素级变换函数保留图像细节，在曝光校正和下游边缘检测上超越有监督SOTA。
