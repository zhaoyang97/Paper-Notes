---
title: >-
  ICML2026 遥感论文汇总 · 3篇论文解读
description: >-
  3篇ICML2026的遥感方向论文解读，涵盖翻译、遥感、对齐/RLHF等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2026"
  - "遥感"
  - "论文解读"
  - "论文笔记"
  - "翻译"
  - "对齐/RLHF"
item_list:
  - u: "any2any_unified_arbitrary_modality_translation_for_remote_sensing/"
    t: "Any2Any: Unified Arbitrary Modality Translation for Remote Sensing"
  - u: "localized_high-resolution_geographic_representations_with_slepian_functions/"
    t: "Localized, High-resolution Geographic Representations with Slepian Functions"
  - u: "the_perception-physics_paradox_probing_scientific_alignment_with_tc-bench/"
    t: "The Perception-Physics Paradox: Probing Scientific Alignment with TC-Bench"
item_total: 3
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🛰️ 遥感

**🧪 ICML2026** · **3** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (63)](../../CVPR2026/remote_sensing/index.md) · [🔬 ICLR2026 (11)](../../ICLR2026/remote_sensing/index.md) · [🤖 AAAI2026 (7)](../../AAAI2026/remote_sensing/index.md) · [🧠 NeurIPS2025 (12)](../../NeurIPS2025/remote_sensing/index.md) · [📹 ICCV2025 (11)](../../ICCV2025/remote_sensing/index.md) · [🧪 ICML2025 (7)](../../ICML2025/remote_sensing/index.md)

**[Any2Any: Unified Arbitrary Modality Translation for Remote Sensing](any2any_unified_arbitrary_modality_translation_for_remote_sensing.md)**

:   Any2Any 把遥感中的 RGB、SAR、NIR、MS、PAN 等传感器互译从一堆成对模型改成一个共享潜空间里的统一潜扩散模型，并用百万级 RST-1M 数据集和目标模态残差适配器，在 14 个已见翻译方向和多个未见模态组合上取得更好的保真度与泛化能力。

**[Localized, High-resolution Geographic Representations with Slepian Functions](localized_high-resolution_geographic_representations_with_slepian_functions.md)**

:   本文用球面 Slepian 函数构造一种把表征容量集中在感兴趣区域 (ROI) 的地理位置编码器，并提出 Slepian-球面调和混合编码以同时兼顾局部高分辨率与全球粗粒度上下文，在五个分类、回归与图像增强预测任务上稳定超过 SH、Wavelet、RFF 等主流基线。

**[The Perception-Physics Paradox: Probing Scientific Alignment with TC-Bench](the_perception-physics_paradox_probing_scientific_alignment_with_tc-bench.md)**

:   作者指出视觉基础模型 (VFM) 在卫星图像上"看起来"很会预测，但在物理极端区段会沿物理坐标轴塌缩，于是用"结构同构"形式化"科学对齐"概念，并发布全球热带气旋基准 TC-Bench 与一套静态/动态/约束三层线性探针，系统揭露 DINO、CLIP、SigLIP、MAE 等冻结骨干在 $P_c<980$ hPa 强气旋段的表征崩溃。
