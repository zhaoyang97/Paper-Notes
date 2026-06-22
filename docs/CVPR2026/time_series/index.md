---
title: >-
  CVPR2026 时间序列论文汇总 · 7篇论文解读
description: >-
  7篇CVPR2026的时间序列方向论文解读，涵盖时序预测、对齐/RLHF、域适应等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "CVPR2026"
  - "时间序列"
  - "论文解读"
  - "论文笔记"
  - "时序预测"
  - "对齐/RLHF"
  - "域适应"
item_list:
  - u: "pfgnet_a_fully_convolutional_frequency-guided_peripheral_gating_network_for_effi/"
    t: "PFGNet: A Fully Convolutional Frequency-Guided Peripheral Gating Network for Efficient Spatiotemporal Predictive Learning"
  - u: "probabilistic_precipitation_nowcasting_with_rectified_flow_transformers/"
    t: "Probabilistic Precipitation Nowcasting with Rectified Flow Transformers"
  - u: "real-time_long_horizon_air_quality_forecasting_via_group-relative_policy_optimiz/"
    t: "Real-Time Long Horizon Air Quality Forecasting via Group-Relative Policy Optimization"
  - u: "sattc_structure-aware_label-free_test-time_calibration_for_cross-subject_eeg-to-/"
    t: "SATTC: Structure-Aware Label-Free Test-Time Calibration for Cross-Subject EEG-to-Image Retrieval"
  - u: "stable_spike_dual_consistency_optimization_via_bitwise_and_operations_for_spikin/"
    t: "Stable Spike: Dual Consistency Optimization via Bitwise AND Operations for Spiking Neural Networks"
  - u: "stcast_adaptive_boundary_alignment_for_global_and_regional_weather_forecasting/"
    t: "STCast: Adaptive Boundary Alignment for Global and Regional Weather Forecasting"
  - u: "towards_uncertainty-aware_unsupervised_domain_adaptation_for_videos_and_time-ser/"
    t: "Towards Uncertainty-aware Unsupervised Domain Adaptation for Videos and Time-Series with Causal Optimal Transport"
item_total: 7
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📈 时间序列

**📷 CVPR2026** · **7** 篇论文解读

📌 **同领域跨会议浏览：** [🔬 ICLR2026 (121)](../../ICLR2026/time_series/index.md) · [💬 ACL2026 (8)](../../ACL2026/time_series/index.md) · [🧪 ICML2026 (45)](../../ICML2026/time_series/index.md) · [🤖 AAAI2026 (31)](../../AAAI2026/time_series/index.md) · [🧠 NeurIPS2025 (54)](../../NeurIPS2025/time_series/index.md) · [📹 ICCV2025 (4)](../../ICCV2025/time_series/index.md)

🔥 **高频主题：** 时序预测 ×3

**[PFGNet: A Fully Convolutional Frequency-Guided Peripheral Gating Network for Efficient Spatiotemporal Predictive Learning](pfgnet_a_fully_convolutional_frequency-guided_peripheral_gating_network_for_effi.md)**

:   提出 PFGNet，一种纯卷积时空预测框架，通过像素级频率引导门控（PFG）动态调制多尺度大核外周响应并施加可学习中心抑制，模拟生物视觉的 center-surround 带通滤波机制，在 Moving MNIST、TaxiBJ、KTH、Human3.6M 四个基准上以极少参数和计算量达到 SOTA 或近 SOTA 性能。

**[Probabilistic Precipitation Nowcasting with Rectified Flow Transformers](probabilistic_precipitation_nowcasting_with_rectified_flow_transformers.md)**

:   本文提出 FREUD——一个用整流流（rectified flow）Transformer 充当"压缩第一阶段"的框架：帧级编码器独立编码每帧、联合视频解码器一次性重建所有帧，把确定性解码换成概率式解码，从而在压缩阶段就能量化不确定性；配合潜空间整流流临近预报模型，在 SEVIR 降水临近预报基准上取得 SOTA 的 CRPS（0.0190）和 SSIM。

**[Real-Time Long Horizon Air Quality Forecasting via Group-Relative Policy Optimization](real-time_long_horizon_air_quality_forecasting_via_group-relative_policy_optimiz.md)**

:   本文针对东亚长程（48–120 小时）PM 浓度预测，先发布一套观测对齐的区域数据集 CMAQ–OBS，再用「带时间累积损失的 SFT + 带类别 AQI 奖励的 GRPO」两阶段训练（FAKER-Air），把 MSE 训练固有的「过预报、误报多」问题对齐到真实的运营成本上，在保持 F1 的同时把误报率（FAR）相对 SFT 基线降低 47.3%。

**[SATTC: Structure-Aware Label-Free Test-Time Calibration for Cross-Subject EEG-to-Image Retrieval](sattc_structure-aware_label-free_test-time_calibration_for_cross-subject_eeg-to-.md)**

:   提出SATTC，一个无标签的测试时校准头，通过几何专家（被试自适应白化+自适应CSLS）和结构专家（互最近邻+双向top-k排名+类别流行度）的乘积专家融合，在冻结的EEG和图像编码器上直接操作相似度矩阵，显著改善跨被试EEG-to-image检索的Top-1精度并降低hubness效应。

**[Stable Spike: Dual Consistency Optimization via Bitwise AND Operations for Spiking Neural Networks](stable_spike_dual_consistency_optimization_via_bitwise_and_operations_for_spikin.md)**

:   提出 Stable Spike 双一致性优化框架，利用硬件友好的 AND 位运算从多时间步脉冲图中解耦稳定脉冲骨架，并注入振幅感知脉冲噪声增强泛化，在超低延迟(T=2)下将神经形态物体识别精度提升最高 8.33%。

**[STCast: Adaptive Boundary Alignment for Global and Regional Weather Forecasting](stcast_adaptive_boundary_alignment_for_global_and_regional_weather_forecasting.md)**

:   提出STCast框架，通过Spatial-Aligned Attention（SAA）用可学习的全球-区域分布替代静态边界来自适应融合全球大气信息到区域预报，并用Temporal Mixture-of-Experts（TMoE）按月动态路由专家增强时序建模，在全球预报、高分辨率区域预报、台风路径预测和集合预测四个任务上全面超越现有方法。

**[Towards Uncertainty-aware Unsupervised Domain Adaptation for Videos and Time-Series with Causal Optimal Transport](towards_uncertainty-aware_unsupervised_domain_adaptation_for_videos_and_time-ser.md)**

:   本文提出 Causal-OT：把通道间的 Granger 因果图嵌进最优传输（OT）的代价矩阵里做跨域对齐，同时用基于熵的不确定性筛选伪标签，让时序与视频的无监督域适应既保住时间-因果结构、又不被过自信的伪标签带偏，在 6 个时序基准上平均涨 4.5% 准确率、4 个视频基准上涨 2.5%。
