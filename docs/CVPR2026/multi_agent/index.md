---
title: >-
  CVPR2026 Multi-Agent论文汇总 · 2篇论文解读
description: >-
  2篇CVPR2026的 Multi-Agent 方向论文解读，涵盖 Agent、目标检测、少样本学习、推理等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "CVPR2026"
  - "Multi-Agent"
  - "论文解读"
  - "论文笔记"
  - "Agent"
  - "目标检测"
  - "少样本学习"
  - "推理"
item_list:
  - u: "agentdet_a_shared-blackboard_multi-agent_framework_for_zero-few-shot_object_dete/"
    t: "AgentDet: A Shared-Blackboard Multi-Agent Framework for Zero-/Few-Shot Object Detection"
  - u: "visual_document_understanding_and_reasoning_a_multi-agent_collaboration_framewor/"
    t: "Visual Document Understanding and Reasoning: A Multi-Agent Collaboration Framework with Agent-Wise Adaptive Test-Time Scaling"
item_total: 2
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 👥 Multi-Agent

**📷 CVPR2026** · **2** 篇论文解读

📌 **同领域跨会议浏览：** [🔬 ICLR2026 (47)](../../ICLR2026/multi_agent/index.md) · [💬 ACL2026 (39)](../../ACL2026/multi_agent/index.md) · [🧪 ICML2026 (24)](../../ICML2026/multi_agent/index.md) · [🤖 AAAI2026 (26)](../../AAAI2026/multi_agent/index.md) · [🧠 NeurIPS2025 (17)](../../NeurIPS2025/multi_agent/index.md) · [🧪 ICML2025 (7)](../../ICML2025/multi_agent/index.md)

🔥 **高频主题：** Agent ×2

**[AgentDet: A Shared-Blackboard Multi-Agent Framework for Zero-/Few-Shot Object Detection](agentdet_a_shared-blackboard_multi-agent_framework_for_zero-few-shot_object_dete.md)**

:   AgentDet 把零/少样本目标检测拆成 Scout / Pinner / Curator / Judge 四个 LLM 智能体，通过一块"共享黑板"+一个 patch 级"知识库"协作：把视觉证据碎片化存进知识库、组合成整体文本线索喂给 LLM 做框预测，并且只训练 Judge 一个智能体，就在 PASCAL VOC / COCO 的 ZSOD/FSOD 上做到了与 SOTA 强竞争的结果。

**[Visual Document Understanding and Reasoning: A Multi-Agent Collaboration Framework with Agent-Wise Adaptive Test-Time Scaling](visual_document_understanding_and_reasoning_a_multi-agent_collaboration_framewor.md)**

:   MACT 把"单模型一把梭"的视觉文档问答拆成规划、执行、判断、回答四个分工明确的智能体，并按每个智能体的认知负荷自适应分配测试时算力（而非统一堆参数），在 15 个基准上以 <30B 参数稳进前三、平均比基座模型提升 9.9–11.5%。
