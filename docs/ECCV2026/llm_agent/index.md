---
title: >-
  ECCV2026 LLMAgent论文汇总 · 4篇论文解读
description: >-
  4篇ECCV2026的 LLM Agent 方向论文解读，涵盖布局/合成、LLM、Agent、多模态等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ECCV2026"
  - "LLM Agent"
  - "论文解读"
  - "论文笔记"
  - "布局/合成"
  - "LLM"
  - "Agent"
  - "多模态"
item_list:
  - u: "guide_resolving_domain_bias_in_gui_agents_through_real-time_web_video_retrieval_/"
    t: "GUIDE: Resolving Domain Bias in GUI Agents through Real-Time Web Video Retrieval and Plug-and-Play Annotation"
  - u: "nala_a_3d_native_llm_layout_agent_for_high-quality_3d_scene_generation/"
    t: "NaLA: A 3D Native LLM Layout Agent for High-quality 3D Scene Generation"
  - u: "think_while_you_map_asynchronous_vision-language_agents_for_incremental_3d_scene/"
    t: "Think While You Map: Asynchronous Vision-Language Agents for Incremental 3D Scene Graphs"
  - u: "viscritic_visual_state_comparison_as_process_reward_for_gui_agents/"
    t: "VisCritic: Visual State Comparison as Process Reward for GUI Agents"
item_total: 4
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🦾 LLM Agent

**🎞️ ECCV2026** · **4** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (42)](../../CVPR2026/llm_agent/index.md) · [🔬 ICLR2026 (162)](../../ICLR2026/llm_agent/index.md) · [💬 ACL2026 (82)](../../ACL2026/llm_agent/index.md) · [🧪 ICML2026 (59)](../../ICML2026/llm_agent/index.md) · [🤖 AAAI2026 (33)](../../AAAI2026/llm_agent/index.md) · [🧠 NeurIPS2025 (39)](../../NeurIPS2025/llm_agent/index.md)

**[GUIDE: Resolving Domain Bias in GUI Agents through Real-Time Web Video Retrieval and Plug-and-Play Annotation](guide_resolving_domain_bias_in_gui_agents_through_real-time_web_video_retrieval_.md)**

:   GUIDE 是一个免训练、即插即用的框架，通过从 YouTube 教程视频中自动检索并提取领域专用的规划知识（Planning）和定位知识（Grounding），注入 GUI Agent 的对应模块来消除领域偏置（domain bias），在 OSWorld 上为三种不同架构的 agent 带来 +4.47 到 +7.48 个百分点的提升。

**[NaLA: A 3D Native LLM Layout Agent for High-quality 3D Scene Generation](nala_a_3d_native_llm_layout_agent_for_high-quality_3d_scene_generation.md)**

:   NaLA 提出了一种直接编码 3D 点云的大语言模型布局智能体，通过粗到细的位姿预测机制实现高质量、高效率的 3D 场景生成，在物理合理性、语义一致性和美观度上全面超越现有方法。

**[Think While You Map: Asynchronous Vision-Language Agents for Incremental 3D Scene Graphs](think_while_you_map_asynchronous_vision-language_agents_for_incremental_3d_scene.md)**

:   ThinkGraphs 提出异步架构，将轻量增量3D映射与重型VLM推理解耦，通过后台运行的 Critic Agent（语义闭环检测合并碎片化轨迹）和 Description Agent（多目标帧调度注入细粒度属性）在不阻塞在线映射的同时持续丰富场景图语义，在 Sr3D+/Nr3D/ScanRefer 三个视觉定位基准上超越此前最优 15.3-18.8 A@0.25。

**[VisCritic: Visual State Comparison as Process Reward for GUI Agents](viscritic_visual_state_comparison_as_process_reward_for_gui_agents.md)**

:   VisCritic 提出基于视觉状态对比的 GUI 智能体过程奖励框架，通过孪生 ViT 比较动作前后截图在语义特征空间中的差异，融合动作上下文联合预测动作成功概率、任务进度与错误类型，作为即插即用的推理时验证模块，在五个基准上普遍提升各类 GUI 智能体的任务成功率。
