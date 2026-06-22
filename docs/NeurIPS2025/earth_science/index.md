---
title: >-
  NeurIPS2025 地球科学论文汇总 · 6篇论文解读
description: >-
  6篇NeurIPS2025的地球科学方向论文解读，涵盖推理等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "NeurIPS2025"
  - "地球科学"
  - "论文解读"
  - "论文笔记"
  - "推理"
item_list:
  - u: "a_probabilistic_unet_approach_to_downscaling_climate_simulat/"
    t: "A Probabilistic U-Net Approach to Downscaling Climate Simulations"
  - u: "adaptive_online_emulation_for_accelerating_complex_physical_simulations/"
    t: "Adaptive Online Emulation for Accelerating Complex Physical Simulations"
  - u: "controlfusion_a_controllable_image_fusion_framework_with_language-vision_degrada/"
    t: "ControlFusion: A Controllable Image Fusion Framework with Language-Vision Degradation Prompts"
  - u: "power_ensemble_aggregation_for_improved_extreme_event_ai_prediction/"
    t: "Power Ensemble Aggregation for Improved Extreme Event AI Prediction"
  - u: "predicting_public_health_impacts_of_electricity_usage/"
    t: "Predicting Public Health Impacts of Electricity Usage"
  - u: "reasoning_with_a_star_a_heliophysics_dataset_and_benchmark_for_agentic_scientifi/"
    t: "Reasoning With a Star: A Heliophysics Dataset and Benchmark for Agentic Scientific Reasoning"
item_total: 6
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🌍 地球科学

**🧠 NeurIPS2025** · **6** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (2)](../../CVPR2026/earth_science/index.md) · [🔬 ICLR2026 (7)](../../ICLR2026/earth_science/index.md) · [🧪 ICML2026 (2)](../../ICML2026/earth_science/index.md) · [🤖 AAAI2026 (2)](../../AAAI2026/earth_science/index.md) · [📷 CVPR2025 (1)](../../CVPR2025/earth_science/index.md) · [🎞️ ECCV2024 (1)](../../ECCV2024/earth_science/index.md)

**[A Probabilistic U-Net Approach to Downscaling Climate Simulations](a_probabilistic_unet_approach_to_downscaling_climate_simulat.md)**

:   首次将概率 U-Net 应用于气候统计降尺度（16× 超分辨率），通过变分隐空间采样生成集合预报来量化降尺度不确定性，并系统比较了 WMSE、MS-SSIM、WMSE-MS-SSIM 和 afCRPS 四种训练目标在捕捉极端事件与保留细尺度空间变异性方面的互补权衡。

**[Adaptive Online Emulation for Accelerating Complex Physical Simulations](adaptive_online_emulation_for_accelerating_complex_physical_simulations.md)**

:   提出 Adaptive Online Emulation (AOE)，在物理模拟执行过程中动态训练 ELM 神经网络代理模型替代昂贵计算组件，无需离线预训练，在系外行星大气模拟上实现 11.1× 加速（91% 时间节省）且精度损失仅 ~0.01%。

**[ControlFusion: A Controllable Image Fusion Framework with Language-Vision Degradation Prompts](controlfusion_a_controllable_image_fusion_framework_with_language-vision_degrada.md)**

:   提出 ControlFusion，一种基于语言-视觉退化提示的可控红外-可见光图像融合框架，通过物理驱动的退化成像模型模拟复合退化，并用 prompt-modulated 网络动态恢复+融合，在真实世界和复合退化场景下全面超越 SOTA。

**[Power Ensemble Aggregation for Improved Extreme Event AI Prediction](power_ensemble_aggregation_for_improved_extreme_event_ai_prediction.md)**

:   提出基于幂均值的自适应集成聚合方法，通过对生成式天气预测模型的集成成员得分施加非线性聚合（幂指数$p>1$），显著提升极端高温事件的分类性能，尤其在高分位数阈值下效果更佳。

**[Predicting Public Health Impacts of Electricity Usage](predicting_public_health_impacts_of_electricity_usage.md)**

:   提出 HealthPredictor，一个将电力消费端到端映射到公共健康损害（以 $/MWh 计量）的 AI 流水线，包含燃料组合预测、空气质量转换和健康影响评估三个模块，健康驱动优化比燃料组合驱动基线显著降低健康影响预测误差，并在电动汽车充电调度案例中实现 24-42% 的健康损害减少。

**[Reasoning With a Star: A Heliophysics Dataset and Benchmark for Agentic Scientific Reasoning](reasoning_with_a_star_a_heliophysics_dataset_and_benchmark_for_agentic_scientifi.md)**

:   提出 Reasoning With a Star (RWS)，一个源自 NASA 太阳物理暑期学校问题集的 158 道科学推理 benchmark（含数值/符号/文本三类答案），配合 unit-aware 评分器，比较了四种多 agent 协调模式（HMAW/PACE/PHASE/SCHEMA），发现没有单一模式在所有任务上占优——系统工程启发的 SCHEMA 在需要严格约束验证的任务上最强。
