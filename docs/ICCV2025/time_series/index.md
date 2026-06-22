---
title: >-
  ICCV2025 时间序列论文汇总 · 4篇论文解读
description: >-
  4篇ICCV2025的时间序列方向论文解读，涵盖时序预测、动态场景、Agent、多模态等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICCV2025"
  - "时间序列"
  - "论文解读"
  - "论文笔记"
  - "时序预测"
  - "动态场景"
  - "Agent"
  - "多模态"
item_list:
  - u: "i2-world_intra-inter_tokenization_for_efficient_dynamic_4d_scene_forecasting/"
    t: "I²-World: Intra-Inter Tokenization for Efficient Dynamic 4D Scene Forecasting"
  - u: "v2xpnp_vehicle-to-everything_spatio-temporal_fusion_for_multi-agent_perception_a/"
    t: "V2XPnP: Vehicle-to-Everything Spatio-Temporal Fusion for Multi-Agent Perception and Prediction"
  - u: "va-moe_variables-adaptive_mixture_of_experts_for_incremental_weather_forecasting/"
    t: "VA-MoE: Variables-Adaptive Mixture of Experts for Incremental Weather Forecasting"
  - u: "vlrmbench_a_comprehensive_and_challenging_benchmark_for_vision-language_reward_m/"
    t: "VLRMBench: A Comprehensive and Challenging Benchmark for Vision-Language Reward Models"
item_total: 4
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📈 时间序列

**📹 ICCV2025** · **4** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (7)](../../CVPR2026/time_series/index.md) · [🔬 ICLR2026 (121)](../../ICLR2026/time_series/index.md) · [💬 ACL2026 (8)](../../ACL2026/time_series/index.md) · [🧪 ICML2026 (45)](../../ICML2026/time_series/index.md) · [🤖 AAAI2026 (31)](../../AAAI2026/time_series/index.md) · [🧠 NeurIPS2025 (54)](../../NeurIPS2025/time_series/index.md)

🔥 **高频主题：** 时序预测 ×2

**[I²-World: Intra-Inter Tokenization for Efficient Dynamic 4D Scene Forecasting](i2-world_intra-inter_tokenization_for_efficient_dynamic_4d_scene_forecasting.md)**

:   提出 I²-World，通过将 3D 场景 tokenization 解耦为帧内（intra-scene）多尺度残差量化和帧间（inter-scene）时序量化两个互补过程，在保持 3D tokenizer 高压缩率的同时获得 4D tokenizer 的时序建模能力，实现高效且高质量的 4D occupancy 预测。

**[V2XPnP: Vehicle-to-Everything Spatio-Temporal Fusion for Multi-Agent Perception and Prediction](v2xpnp_vehicle-to-everything_spatio-temporal_fusion_for_multi-agent_perception_a.md)**

:   提出 V2XPnP，一个基于统一 Transformer 架构的 V2X 时空融合框架，在单步通信策略下实现多智能体端到端感知与预测，同时构建了首个支持所有 V2X 协作模式的大规模真实世界时序数据集，在感知和预测任务上达到 SOTA。

**[VA-MoE: Variables-Adaptive Mixture of Experts for Incremental Weather Forecasting](va-moe_variables-adaptive_mixture_of_experts_for_incremental_weather_forecasting.md)**

:   提出增量天气预报新范式和VA-MoE框架，通过变量自适应的MoE架构和索引嵌入机制，实现在仅25%可训练参数和50%初始训练数据的条件下达到与全量训练可比的预报精度。

**[VLRMBench: A Comprehensive and Challenging Benchmark for Vision-Language Reward Models](vlrmbench_a_comprehensive_and_challenging_benchmark_for_vision-language_reward_m.md)**

:   提出 VLRMBench，一个包含 12634 个问题、12 项任务的综合且具有挑战性的视觉语言奖励模型（VLRM）基准，覆盖过程理解、结果判断和批评生成三大方面，在 26 个模型上的广泛实验揭示了当前 VLRM 的显著不足。
