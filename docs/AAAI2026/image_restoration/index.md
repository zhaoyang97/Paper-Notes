---
title: >-
  AAAI2026 图像恢复论文汇总 · 10篇论文解读
description: >-
  10篇AAAI2026的图像恢复方向论文解读，涵盖图像恢复、超分辨率、对抗鲁棒、扩散模型、多模态、情感分析等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "AAAI2026"
  - "图像恢复"
  - "论文解读"
  - "论文笔记"
  - "超分辨率"
  - "对抗鲁棒"
  - "扩散模型"
  - "多模态"
  - "情感分析"
item_list:
  - u: "blur-robust_detection_via_feature_restoration_an_end-to-end_framework_for_prior-/"
    t: "Blur-Robust Detection via Feature Restoration: An End-to-End Framework for Prior-Guided Infrared UAV Target Detection"
  - u: "clear_nights_ahead_towards_multi-weather_nighttime_image_res/"
    t: "Clear Nights Ahead: Towards Multi-Weather Nighttime Image Restoration"
  - u: "depth-synergized_mamba_meets_memory_experts_for_all-day_image_reflection_separat/"
    t: "Depth-Synergized Mamba Meets Memory Experts for All-Day Image Reflection Separation"
  - u: "iclr_inter-chrominance_and_luminance_interaction_for_natural_color_restoration_i/"
    t: "ICLR: Inter-Chrominance and Luminance Interaction for Natural Color Restoration in Low-Light Image Enhancement"
  - u: "mfmamba_a_multi-function_network_for_panchromatic_image_resolution_restoration_b/"
    t: "MFmamba: A Multi-function Network for Panchromatic Image Resolution Restoration Based on State-Space Model"
  - u: "refidiff_progressive_refinement_diffusion_for_efficient_missing_data_imputation/"
    t: "RefiDiff: Progressive Refinement Diffusion for Efficient Missing Data Imputation"
  - u: "sd-psfnet_sequential_and_dynamic_point_spread_function_netwo/"
    t: "SD-PSFNet: Sequential and Dynamic Point Spread Function Network for Image Deraining"
  - u: "spatiotemporal_difference_network_for_video_depth_super-resolution/"
    t: "SpatioTemporal Difference Network for Video Depth Super-Resolution"
  - u: "temporal_inconsistency_guidance_for_super-resolution_video_quality_assessment/"
    t: "Temporal Inconsistency Guidance for Super-resolution Video Quality Assessment"
  - u: "tmdc_a_two-stage_modality_denoising_and_complementation_framework_for_multimodal/"
    t: "TMDC: A Two-Stage Modality Denoising and Complementation Framework for Multimodal Sentiment Analysis"
item_total: 10
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🖼️ 图像恢复

**🤖 AAAI2026** · **10** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (135)](../../CVPR2026/image_restoration/index.md) · [🔬 ICLR2026 (61)](../../ICLR2026/image_restoration/index.md) · [🧪 ICML2026 (21)](../../ICML2026/image_restoration/index.md) · [🧠 NeurIPS2025 (26)](../../NeurIPS2025/image_restoration/index.md) · [📹 ICCV2025 (31)](../../ICCV2025/image_restoration/index.md) · [🧪 ICML2025 (5)](../../ICML2025/image_restoration/index.md)

🔥 **高频主题：** 图像恢复 ×3 · 超分辨率 ×2

**[Blur-Robust Detection via Feature Restoration: An End-to-End Framework for Prior-Guided Infrared UAV Target Detection](blur-robust_detection_via_feature_restoration_an_end-to-end_framework_for_prior-.md)**

:   提出 JFD3 端到端双分支框架，在特征域而非图像域进行去模糊，并利用频率结构先验引导检测网络，实现运动模糊条件下红外无人机目标的高精度实时检测。

**[Clear Nights Ahead: Towards Multi-Weather Nighttime Image Restoration](clear_nights_ahead_towards_multi-weather_nighttime_image_res.md)**

:   首次定义并探索多天气夜间图像复原任务，构建 AllWeatherNight 数据集（8K 训练 + 1K 合成测试 + 1K 真实测试），提出 ClearNight 统一框架通过 Retinex 双先验引导和天气感知动态专一性-共性协作，一阶段同时移除雾/雨条/雨滴/雪/flare 复合退化，仅 2.84M 参数全面超越 SOTA。

**[Depth-Synergized Mamba Meets Memory Experts for All-Day Image Reflection Separation](depth-synergized_mamba_meets_memory_experts_for_all-day_image_reflection_separat.md)**

:   提出 DMDNet，通过深度感知扫描策略（DAScan）引导 Mamba 关注显著结构，结合深度协同状态空间模型（DS-SSM）抑制模糊特征传播，并引入记忆专家补偿模块（MECM）利用跨图像历史知识，实现全天候（白天+夜间）的图像反射分离。

**[ICLR: Inter-Chrominance and Luminance Interaction for Natural Color Restoration in Low-Light Image Enhancement](iclr_inter-chrominance_and_luminance_interaction_for_natural_color_restoration_i.md)**

:   针对HVI色彩空间中色度和亮度分支分布差异大导致互补特征提取不足、以及色度分支间弱相关导致梯度冲突的问题，提出ICLR框架，通过双流交互增强模块(DIEM)和协方差校正损失(CCL)分别从融合增强和统计分布优化两个角度解决，在LOL系列数据集上取得SOTA。

**[MFmamba: A Multi-function Network for Panchromatic Image Resolution Restoration Based on State-Space Model](mfmamba_a_multi-function_network_for_panchromatic_image_resolution_restoration_b.md)**

:   提出MFmamba多功能网络，基于UNet++骨架结合Mamba上采样模块（MUB）、双池化注意力（DPA）和多尺度混合交叉块（MHCB），仅使用全色（PAN）图像输入即可同时实现超分辨率、光谱恢复及联合SR与着色三种任务。

**[RefiDiff: Progressive Refinement Diffusion for Efficient Missing Data Imputation](refidiff_progressive_refinement_diffusion_for_efficient_missing_data_imputation.md)**

:   提出 RefiDiff 四阶段框架（预处理→warm-up→扩散→polish），首次将 predictive 和 generative 缺失值填补范式渐进统一，结合 Mamba-based denoising 在 9 个数据集上取得 SOTA，速度比 DIFFPUTER 快 4 倍。

**[SD-PSFNet: Sequential and Dynamic Point Spread Function Network for Image Deraining](sd-psfnet_sequential_and_dynamic_point_spread_function_netwo.md)**

:   提出基于动态 PSF 机制的级联 CNN 去雨网络 SD-PSFNet，通过多尺度可学习 PSF 字典建模雨滴光学效应，配合自适应门控融合的序列化修复架构，在 Rain100H 达 33.12 dB、RealRain-1k-L 达 42.28 dB 均为 SOTA，对比基线 MPRNet 累计提升 5.04 dB（13.5%）。

**[SpatioTemporal Difference Network for Video Depth Super-Resolution](spatiotemporal_difference_network_for_video_depth_super-resolution.md)**

:   基于视频深度超分辨率（VDSR）中空间非光滑区域和时间变化区域呈长尾分布的统计发现，提出 STDNet，通过空间差异分支（学习空间差异表示进行帧内 RGB-D 自适应聚合）和时间差异分支（利用时间差异表示在变化区域进行运动补偿），在 TarTanAir 数据集上 ×16 超分 RMSE 从 112.04cm 降至 96.80cm，平均超越 SOTA 方法 27.6%-32.6%。

**[Temporal Inconsistency Guidance for Super-resolution Video Quality Assessment](temporal_inconsistency_guidance_for_super-resolution_video_quality_assessment.md)**

:   提出 TIG-SVQA 框架，首次将时间不一致性（temporal inconsistency）作为显式引导信号融入超分辨率视频质量评估，设计了不一致性高亮空间模块（IHSM）和不一致性引导时间模块（IGTM），在 SFD、MFD 和 Combined-VSR 三个数据集上 SRCC 分别达到 0.950、0.942、0.939，全面超越现有 IQA/VQA 方法。

**[TMDC: A Two-Stage Modality Denoising and Complementation Framework for Multimodal Sentiment Analysis](tmdc_a_two-stage_modality_denoising_and_complementation_framework_for_multimodal.md)**

:   提出 TMDC 两阶段框架，第一阶段在完整数据上学习去噪的 modality-specific 和 modality-common 表示，第二阶段利用可用模态的去噪表示补全缺失模态，首次同时处理 MSA 中的噪声和缺失问题。
