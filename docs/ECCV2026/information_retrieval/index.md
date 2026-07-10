---
title: >-
  ECCV2026 信息检索/RAG论文汇总 · 2篇论文解读
description: >-
  2篇ECCV2026的信息检索/RAG 方向论文解读，涵盖对抗鲁棒等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ECCV2026"
  - "信息检索/RAG"
  - "论文解读"
  - "论文笔记"
  - "对抗鲁棒"
item_list:
  - u: "hyfl-clip_hyperbolic_fine-tuning_of_clip_for_robust_long-context_understanding/"
    t: "HyFL-CLIP: Hyperbolic Fine-Tuning of CLIP for Robust Long-Context Understanding"
  - u: "lightstar_efficient_visual_document_retrieval_via_lightweight_selection_with_vis/"
    t: "LightSTAR: Efficient Visual Document Retrieval via Lightweight Selection with Vision-Adaptive Refinement"
item_total: 2
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔍 信息检索/RAG

**🎞️ ECCV2026** · **2** 篇论文解读

📌 **同领域跨会议浏览：** [🔬 ICLR2026 (81)](../../ICLR2026/information_retrieval/index.md) · [💬 ACL2026 (73)](../../ACL2026/information_retrieval/index.md) · [🧪 ICML2026 (26)](../../ICML2026/information_retrieval/index.md) · [🤖 AAAI2026 (21)](../../AAAI2026/information_retrieval/index.md) · [🧠 NeurIPS2025 (25)](../../NeurIPS2025/information_retrieval/index.md) · [📹 ICCV2025 (5)](../../ICCV2025/information_retrieval/index.md)

**[HyFL-CLIP: Hyperbolic Fine-Tuning of CLIP for Robust Long-Context Understanding](hyfl-clip_hyperbolic_fine-tuning_of_clip_for_robust_long-context_understanding.md)**

:   HyFL-CLIP 将预训练 CLIP 的欧氏图文对齐蒸馏到双曲空间（Lorentz 模型），通过爱因斯坦中点聚合建模"全局描述--局部成分"的层次蕴涵关系，使模型在长文本被扰动（重排、删除、丢词）时仍能稳定检索正确图像，在扰动词下比最强基线提升最高 19.5%。

**[LightSTAR: Efficient Visual Document Retrieval via Lightweight Selection with Vision-Adaptive Refinement](lightstar_efficient_visual_document_retrieval_via_lightweight_selection_with_vis.md)**

:   LightSTAR 针对视觉文档检索中 MLLM 全量编码页面太慢的问题，先用 LLM-free 轻量视觉选择模块高召回筛出候选，再只对候选做视觉自适应语义精排，在 ViDoRe 上达到 89.1 NDCG@5，同时把 5000 页端到端延迟降到 123.9s。
