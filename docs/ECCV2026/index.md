---
title: >-
  ECCV2026 论文汇总 · 3篇论文解读，每篇5分钟读懂核心思想
description: >-
  3篇ECCV2026论文解读，涵盖 3D 视觉(1篇)、信息检索/RAG(1篇)、视频生成(1篇)等 3个方向。每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "ECCV2026"
  - "ECCV2026论文汇总"
  - "ECCV2026论文列表"
  - "AI顶会"
  - "论文解读"
  - "论文笔记"
  - "3D 视觉"
  - "信息检索/RAG"
  - "视频生成"
item_list:
  - u: "3d_vision/agentic_collaborative_cognition_for_zero-shot_3d_understanding/"
    t: "Agentic Collaborative Cognition for Zero-Shot 3D Understanding"
  - u: "information_retrieval/lightstar_efficient_visual_document_retrieval_via_lightweight_selection_with_vis/"
    t: "LightSTAR: Efficient Visual Document Retrieval via Lightweight Selection with Vision-Adaptive Refinement"
  - u: "video_generation/event-driven_video_generation/"
    t: "Event-Driven Video Generation"
item_total: 3
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎞️ ECCV2026 论文汇总

3篇ECCV2026论文解读，涵盖 3D 视觉(1篇)、信息检索/RAG(1篇)、视频生成(1篇)等 3个方向。每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。

<div class="conf-index" markdown>

---

## 🔍 信息检索/RAG (1) { #information_retrieval }

**[LightSTAR: Efficient Visual Document Retrieval via Lightweight Selection with Vision-Adaptive Refinement](information_retrieval/lightstar_efficient_visual_document_retrieval_via_lightweight_selection_with_vis.md)**

:   LightSTAR 针对视觉文档检索中 MLLM 全量编码页面太慢的问题，先用 LLM-free 轻量视觉选择模块高召回筛出候选，再只对候选做视觉自适应语义精排，在 ViDoRe 上达到 89.1 NDCG@5，同时把 5000 页端到端延迟降到 123.9s。

---

## 🎬 视频生成 (1) { #video_generation }

**[Event-Driven Video Generation](video_generation/event-driven_video_generation.md)**

:   EVD 给预训练视频 DiT 加了一个轻量 token-level event head，并在训练和采样时用事件门控约束 latent update 只发生在交互真正活跃的区域，从而减少物体提前动、接触缺失、支撑关系跳变和事件后漂移等视频生成动态错误。

---

## 🧊 3D 视觉 (1) { #3d_vision }

**[Agentic Collaborative Cognition for Zero-Shot 3D Understanding](3d_vision/agentic_collaborative_cognition_for_zero-shot_3d_understanding.md)**

:   这篇论文把零样本 3D 场景理解改造成 Planning Agent 与 Perception Agent 围绕显式 holistic cognitive map 反复协作的过程，通过主动补关键视角、跨视角记录物体属性和反馈式候选过滤，在 ScanRefer / Nr3D / SQA3D / ScanQA 等 6 个基准上明显优于已有零样本方法。

</div>