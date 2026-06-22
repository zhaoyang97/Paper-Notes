---
title: >-
  ECCV2024 LLMAgent论文汇总 · 3篇论文解读
description: >-
  3篇ECCV2024的 LLM Agent 方向论文解读，涵盖 Agent、少样本学习、布局/合成、推理、多模态等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "ECCV2024"
  - "LLM Agent"
  - "论文解读"
  - "论文笔记"
  - "Agent"
  - "少样本学习"
  - "布局/合成"
  - "推理"
  - "多模态"
item_list:
  - u: "agent3d-zero_an_agent_for_zero-shot_3d_understanding/"
    t: "Agent3D-Zero: An Agent for Zero-shot 3D Understanding"
  - u: "hydra_a_hyper_agent_for_dynamic_compositional_visual_reasoning/"
    t: "HYDRA: A Hyper Agent for Dynamic Compositional Visual Reasoning"
  - u: "videoagent_a_memory-augmented_multimodal_agent_for_video_understanding/"
    t: "VideoAgent: A Memory-augmented Multimodal Agent for Video Understanding"
item_total: 3
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🦾 LLM Agent

**🎞️ ECCV2024** · **3** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (42)](../../CVPR2026/llm_agent/index.md) · [🔬 ICLR2026 (162)](../../ICLR2026/llm_agent/index.md) · [💬 ACL2026 (82)](../../ACL2026/llm_agent/index.md) · [🧪 ICML2026 (59)](../../ICML2026/llm_agent/index.md) · [🤖 AAAI2026 (33)](../../AAAI2026/llm_agent/index.md) · [🧠 NeurIPS2025 (39)](../../NeurIPS2025/llm_agent/index.md)

🔥 **高频主题：** Agent ×3

**[Agent3D-Zero: An Agent for Zero-shot 3D Understanding](agent3d-zero_an_agent_for_zero-shot_3d_understanding.md)**

:   Agent3D-Zero 提出一个基于 VLM 的零样本 3D 场景理解 Agent 框架，通过鸟瞰图上的 Set-of-Line 视觉提示引导 VLM 主动选择观察视角，并综合多视角图像进行 3D 推理，在 ScanQA 等任务上超越了需要微调的 3D-LLM 方法。

**[HYDRA: A Hyper Agent for Dynamic Compositional Visual Reasoning](hydra_a_hyper_agent_for_dynamic_compositional_visual_reasoning.md)**

:   （注：基于摘要的简要笔记）提出 HYDRA，一种多阶段动态组合式视觉推理框架，通过规划器（Planner）、强化学习认知控制器（RL Agent）和推理器（Reasoner）三模块协作，实现可靠且渐进式的视觉推理，在 RefCOCO/RefCOCO+、OK-VQA、GQA 等多个数据集上取得 SOTA。

**[VideoAgent: A Memory-augmented Multimodal Agent for Video Understanding](videoagent_a_memory-augmented_multimodal_agent_for_video_understanding.md)**

:   提出 VideoAgent，一个记忆增强的多模态 Agent，通过构建结构化记忆（temporal memory 存储事件描述 + object memory 存储物体跟踪状态）并利用 4 个工具与记忆交互，零样本完成长视频问答任务，在 NExT-QA 上平均 +6.6%、EgoSchema 上 +26.0%，接近 Gemini 1.5 Pro 的性能。
