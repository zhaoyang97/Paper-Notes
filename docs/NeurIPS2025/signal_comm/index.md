---
title: >-
  NeurIPS2025 信号/通信论文汇总 · 5篇论文解读
description: >-
  5篇NeurIPS2025的信号/通信方向论文解读，涵盖持续学习等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "NeurIPS2025"
  - "信号/通信"
  - "论文解读"
  - "论文笔记"
  - "持续学习"
item_list:
  - u: "angular_steering_behavior_control_via_rotation_in_activation_space/"
    t: "Angular Steering: Behavior Control via Rotation in Activation Space"
  - u: "contrastive_consolidation_of_top-down_modulations_achieves_sparsely_supervised_c/"
    t: "Contrastive Consolidation of Top-Down Modulations Achieves Sparsely Supervised Continual Learning"
  - u: "feature-aware_modulation_for_learning_from_temporal_tabular_data/"
    t: "Feature-aware Modulation for Learning from Temporal Tabular Data"
  - u: "masked_symbol_modeling_for_demodulation_of_oversampled_baseband_communication_si/"
    t: "Masked Symbol Modeling for Demodulation of Oversampled Baseband Communication Signals"
  - u: "memory-integrated_reconfigurable_adapters_a_unified_framework_for_settings_with_/"
    t: "Memory-Integrated Reconfigurable Adapters: A Unified Framework for Settings with Multiple Tasks"
item_total: 5
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📡 信号/通信

**🧠 NeurIPS2025** · **5** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (2)](../../CVPR2026/signal_comm/index.md) · [🔬 ICLR2026 (7)](../../ICLR2026/signal_comm/index.md) · [🧪 ICML2026 (2)](../../ICML2026/signal_comm/index.md) · [🤖 AAAI2026 (3)](../../AAAI2026/signal_comm/index.md) · [📹 ICCV2025 (3)](../../ICCV2025/signal_comm/index.md) · [🧪 ICML2025 (3)](../../ICML2025/signal_comm/index.md)

**[Angular Steering: Behavior Control via Rotation in Activation Space](angular_steering_behavior_control_via_rotation_in_activation_space.md)**

:   提出Angular Steering，将LLM激活引导统一建模为固定2D子空间中的旋转操作——通过旋转角度提供0°-360°的连续、细粒度、范数保持的行为控制旋钮，统一了激活加法和方向消融为旋转的特例，在Llama 3/Qwen 2.5/Gemma 2（3B-14B）上实现鲁棒的行为调控。

**[Contrastive Consolidation of Top-Down Modulations Achieves Sparsely Supervised Continual Learning](contrastive_consolidation_of_top-down_modulations_achieves_sparsely_supervised_c.md)**

:   提出 Task-Modulated Contrastive Learning (TMCL)，受大脑新皮层自顶向下调制启发，在持续学习中通过 affine modulation 集成稀疏标签信息（仅需 1% 标签），再利用对比学习将调制信息固化到前馈权重中，在 class-incremental 和迁移学习上超越无监督和有监督基线。

**[Feature-aware Modulation for Learning from Temporal Tabular Data](feature-aware_modulation_for_learning_from_temporal_tabular_data.md)**

:   论文认为时间表格学习真正难的不是“再加一个时间 embedding”这么简单，而是很多特征的语义会随时间漂移，因此提出 feature-aware modulation，通过时间上下文动态生成每个特征的偏移、缩放与非线性形状参数，把跨时间的语义重新对齐，最终在 TabReD 上让深度模型第一次在平均排名上稳定压过 GBDT。

**[Masked Symbol Modeling for Demodulation of Oversampled Baseband Communication Signals](masked_symbol_modeling_for_demodulation_of_oversampled_baseband_communication_si.md)**

:   本文提出 Masked Symbol Modeling（MSM），将 BERT 的掩码预测范式应用于通信物理层——将脉冲成形产生的符号间贡献重新定义为"上下文信息"，训练 Transformer 在干净过采样基带信号上学习波形结构，推理时利用学到的上下文来恢复被冲激噪声破坏的符号。

**[Memory-Integrated Reconfigurable Adapters: A Unified Framework for Settings with Multiple Tasks](memory-integrated_reconfigurable_adapters_a_unified_framework_for_settings_with_.md)**

:   MIRA 将 Hopfield 式联想记忆模块嵌入 ViT 各层，以键值对方式存储和检索 LoRA 适配器权重，通过两阶段训练（适应+巩固），在一个统一架构下同时解决领域泛化（DG）、类增量学习（CIL）和域增量学习（DIL）三类任务，在多个基准上显著超过各任务的专用方法。
