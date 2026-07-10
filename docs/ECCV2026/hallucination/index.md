---
title: >-
  ECCV2026 幻觉检测论文汇总 · 2篇论文解读
description: >-
  2篇ECCV2026的幻觉检测方向论文解读，收录 From Hallucination to Grounding、No Place to Hide等。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "ECCV2026"
  - "幻觉检测"
  - "论文解读"
  - "论文笔记"
item_list:
  - u: "from_hallucination_to_grounding_diagnosing_visual_spatial_intelligence_via_crisp/"
    t: "From Hallucination to Grounding: Diagnosing Visual Spatial Intelligence via CRISP"
  - u: "no_place_to_hide_benchmarking_video_hallucination_with_background-controlled_pai/"
    t: "No Place to Hide: Benchmarking Video Hallucination with Background-Controlled Pairs"
item_total: 2
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 👻 幻觉检测

**🎞️ ECCV2026** · **2** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (33)](../../CVPR2026/hallucination/index.md) · [🔬 ICLR2026 (40)](../../ICLR2026/hallucination/index.md) · [💬 ACL2026 (28)](../../ACL2026/hallucination/index.md) · [🧪 ICML2026 (21)](../../ICML2026/hallucination/index.md) · [🤖 AAAI2026 (15)](../../AAAI2026/hallucination/index.md) · [🧠 NeurIPS2025 (17)](../../NeurIPS2025/hallucination/index.md)

**[From Hallucination to Grounding: Diagnosing Visual Spatial Intelligence via CRISP](from_hallucination_to_grounding_diagnosing_visual_spatial_intelligence_via_crisp.md)**

:   CRISP 是一个结构诊断式视觉空间智能评测基准，通过 Spatial QA + 3D Scene Graph 双任务范式与跨任务一致性协议，揭示 VLM 的空间推理究竟是真正的 3D 几何理解还是依靠语言先验的语义捷径。

**[No Place to Hide: Benchmarking Video Hallucination with Background-Controlled Pairs](no_place_to_hide_benchmarking_video_hallucination_with_background-controlled_pai.md)**

:   本文提出 VidPair-Halluc 视频幻觉 benchmark，用「背景高度相似、前景语义显著不同」的对抗视频对，把模型出错的原因从背景变化里剥离出来、干净地归因到前景幻觉；配套的 PairFlow 三阶段生成流水线借助 T2I / 视频生成模型自动造出 1K 高质量视频对与 11K 时空 QA，评测显示主流大视频模型在这种受控设置下普遍大幅掉点。
