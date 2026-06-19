---
title: >-
  CVPR2026 科学计算论文汇总 · 3篇论文解读
description: >-
  3篇CVPR2026的科学计算方向论文解读，收录 Continuous Exposure-Time Model、EHETM、NESTOR等。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "CVPR2026"
  - "科学计算"
  - "论文解读"
  - "论文笔记"
item_list:
  - u: "continuous_exposure-time_modeling_for_realistic_atmospheric_turbulence_synthesis/"
    t: "Continuous Exposure-Time Modeling for Realistic Atmospheric Turbulence Synthesis"
  - u: "high-quality_and_efficient_turbulence_mitigation_with_events/"
    t: "EHETM: High-Quality and Efficient Turbulence Mitigation with Events"
  - u: "nestor_a_nested_moe-based_neural_operator_for_large-scale_pde_pre-training/"
    t: "NESTOR: A Nested MOE-based Neural Operator for Large-Scale PDE Pre-Training"
item_total: 3
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧮 科学计算

**📷 CVPR2026** · **3** 篇论文解读

**[Continuous Exposure-Time Modeling for Realistic Atmospheric Turbulence Synthesis](continuous_exposure-time_modeling_for_realistic_atmospheric_turbulence_synthesis.md)**

:   提出曝光时间依赖的调制传递函数（ET-MTF），将曝光时间建模为连续变量，构建了大规模合成湍流数据集 ET-Turb（5083视频、200万帧），显著提升湍流复原模型在真实数据上的泛化能力。

**[EHETM: High-Quality and Efficient Turbulence Mitigation with Events](high-quality_and_efficient_turbulence_mitigation_with_events.md)**

:   提出EHETM，首次利用事件相机的微秒时间分辨率突破传统多帧湍流缓解(TM)方法的精度-效率瓶颈，发现两个关键物理现象——湍流诱导事件的极性交替与清晰梯度相关、动态物体形成时空相干"事件管"——设计极性加权梯度和事件管约束两个互补模块，数据开销降低77.3%、系统延迟降低89.5%，尤其在动态物体场景显著超越SOTA。

**[NESTOR: A Nested MOE-based Neural Operator for Large-Scale PDE Pre-Training](nestor_a_nested_moe-based_neural_operator_for_large-scale_pde_pre-training.md)**

:   提出嵌套式 MoE 神经算子 NESTOR，通过 image-level MoE 捕获不同 PDE 类型的全局特征 + token-level Sub-MoE 捕获物理场内局部相关性，在 12 个 PDE 数据集上实现大规模预训练并有效迁移到下游任务。
