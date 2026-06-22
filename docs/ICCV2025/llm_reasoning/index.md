---
title: >-
  ICCV2025 LLMReasoning论文汇总 · 3篇论文解读
description: >-
  3篇ICCV2025的 LLM Reasoning 方向论文解读，涵盖推理、LLM、多模态、对齐/RLHF、视频生成等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "ICCV2025"
  - "LLM Reasoning"
  - "论文解读"
  - "论文笔记"
  - "推理"
  - "LLM"
  - "多模态"
  - "对齐/RLHF"
  - "视频生成"
item_list:
  - u: "corvid_improving_multimodal_large_language_models_towards_chain-of-thought_reaso/"
    t: "CoRVid: Improving Multimodal Large Language Models Towards Chain-of-Thought Reasoning"
  - u: "unsupervised_visual_chain-of-thought_reasoning_via_preference_optimization/"
    t: "Unsupervised Visual Chain-of-Thought Reasoning via Preference Optimization"
  - u: "video-t1_test-time_scaling_for_video_generation/"
    t: "Video-T1: Test-Time Scaling for Video Generation"
item_total: 3
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 💡 LLM Reasoning

**📹 ICCV2025** · **3** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (16)](../../CVPR2026/llm_reasoning/index.md) · [🔬 ICLR2026 (241)](../../ICLR2026/llm_reasoning/index.md) · [💬 ACL2026 (82)](../../ACL2026/llm_reasoning/index.md) · [🧪 ICML2026 (78)](../../ICML2026/llm_reasoning/index.md) · [🤖 AAAI2026 (37)](../../AAAI2026/llm_reasoning/index.md) · [🧠 NeurIPS2025 (82)](../../NeurIPS2025/llm_reasoning/index.md)

🔥 **高频主题：** 推理 ×2

**[CoRVid: Improving Multimodal Large Language Models Towards Chain-of-Thought Reasoning](corvid_improving_multimodal_large_language_models_towards_chain-of-thought_reaso.md)**

:   提出 Corvid，通过混合视觉编码器 + GateMixer 连接器 + 高质量 CoT 数据集 + 推理时自验证策略，全面提升 MLLM 的链式推理能力，在数学推理和科学问题求解上超越同参数量级的开源模型。

**[Unsupervised Visual Chain-of-Thought Reasoning via Preference Optimization](unsupervised_visual_chain-of-thought_reasoning_via_preference_optimization.md)**

:   提出UV-CoT框架，通过自动生成偏好数据和改进的Score-DPO损失函数，在不需要人工标注bounding box的情况下实现图像级链式思维（Visual CoT）推理，在6个基准上超越有监督的Visual-CoT方法。

**[Video-T1: Test-Time Scaling for Video Generation](video-t1_test-time_scaling_for_video_generation.md)**

:   将LLM中的测试时缩放(TTS)思想迁移到视频生成领域，将TTS重新定义为从高斯噪声空间到目标视频分布的搜索问题，提出Tree-of-Frames (ToF)搜索算法实现高效的推理时计算扩展，在VBench上持续稳定提升各类视频生成模型的质量。
