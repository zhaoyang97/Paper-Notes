---
title: >-
  CVPR2026 社会计算论文汇总 · 3篇论文解读
description: >-
  3篇CVPR2026的社会计算方向论文解读，涵盖多模态、目标跟踪、Agent等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "CVPR2026"
  - "社会计算"
  - "论文解读"
  - "论文笔记"
  - "多模态"
  - "目标跟踪"
  - "Agent"
item_list:
  - u: "bridging_pixels_and_words_mask-aware_local_semantic_fusion_for_multimodal_media_/"
    t: "Bridging Pixels and Words: Mask-Aware Local Semantic Fusion for Multimodal Media Verification"
  - u: "instance-level_visual_active_tracking_with_occlusion-aware_planning/"
    t: "Instance-level Visual Active Tracking with Occlusion-Aware Planning"
  - u: "revisiting_unknowns_towards_effective_and_efficient_open-set_active_learning/"
    t: "Revisiting Unknowns: Towards Effective and Efficient Open-Set Active Learning"
item_total: 3
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 👥 社会计算

**📷 CVPR2026** · **3** 篇论文解读

📌 **同领域跨会议浏览：** [🔬 ICLR2026 (17)](../../ICLR2026/social_computing/index.md) · [💬 ACL2026 (44)](../../ACL2026/social_computing/index.md) · [🧪 ICML2026 (9)](../../ICML2026/social_computing/index.md) · [🤖 AAAI2026 (10)](../../AAAI2026/social_computing/index.md) · [🧠 NeurIPS2025 (20)](../../NeurIPS2025/social_computing/index.md) · [📹 ICCV2025 (4)](../../ICCV2025/social_computing/index.md)

**[Bridging Pixels and Words: Mask-Aware Local Semantic Fusion for Multimodal Media Verification](bridging_pixels_and_words_mask-aware_local_semantic_fusion_for_multimodal_media_.md)**

:   提出 MaLSF 框架，利用掩码-标签对作为语义锚点，通过双向跨模态验证（BCV）和层级语义聚合（HSA）模块实现主动式局部语义冲突检测，在 DGM4 和假新闻检测任务上取得 SOTA。

**[Instance-level Visual Active Tracking with Occlusion-Aware Planning](instance-level_visual_active_tracking_with_occlusion-aware_planning.md)**

:   OA-VAT 用一张参考图离线构建判别性"实例原型"来对抗相似干扰物，在线 EMA 增强原型 + 置信度自适应卡尔曼滤波保持稳定跟踪，并训练一个以目标框为条件的扩散轨迹规划器在目标被遮挡时主动绕障找回——在 UnrealCV 上平均 SR 0.93、真实图像平均 CAR 90.8%、真机无人机 TSR 81.6%，且 RTX 3090 上 35 FPS 实时。

**[Revisiting Unknowns: Towards Effective and Efficient Open-Set Active Learning](revisiting_unknowns_towards_effective_and_efficient_open-set_active_learning.md)**

:   提出 E2OAL，一个无需额外检测器的开放集主动学习框架，通过标签引导聚类发现未知类潜在结构、Dirichlet 校准辅助头联合建模已知/未知类别，并设计两阶段自适应查询策略，在多个基准上同时实现高准确率、高查询纯度和高训练效率。
