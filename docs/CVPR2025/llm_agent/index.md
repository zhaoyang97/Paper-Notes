---
title: >-
  CVPR2025 LLMAgent论文汇总 · 9篇论文解读
description: >-
  9篇CVPR2025的 LLM Agent 方向论文解读，涵盖 Agent、动态场景、压缩/编码、多模态、推理等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "CVPR2025"
  - "LLM Agent"
  - "论文解读"
  - "论文笔记"
  - "Agent"
  - "动态场景"
  - "压缩/编码"
  - "多模态"
  - "推理"
item_list:
  - u: "ata_adaptive_transformation_agent_for_text-guided_subject-position_variable_back/"
    t: "ATA: Adaptive Transformation Agent for Text-Guided Subject-Position Variable Background Generation"
  - u: "feature4x_bridging_any_monocular_video_to_4d_agentic_ai_with_versatile_gaussian_/"
    t: "Feature4X: Bridging Any Monocular Video to 4D Agentic AI with Versatile Gaussian Feature Fields"
  - u: "gui-xplore_empowering_generalizable_gui_agents_with_one_exploration/"
    t: "GUI-Xplore: Empowering Generalizable GUI Agents with One Exploration"
  - u: "rl-rc-dot_a_block-level_rl_agent_for_task-aware_video_compression/"
    t: "RL-RC-DoT: A Block-level RL Agent for Task-Aware Video Compression"
  - u: "sceneassistant_a_visual_feedback_agent_for_open-vocabulary_3d_scene_generation/"
    t: "SceneAssistant: A Visual Feedback Agent for Open-Vocabulary 3D Scene Generation"
  - u: "sketchtopia_a_dataset_and_foundational_agents_for_benchmarking_asynchronous_mult/"
    t: "Sketchtopia: A Dataset and Foundational Agents for Benchmarking Asynchronous Multimodal Communication with Iconic Feedback"
  - u: "spiritsight_agent_advanced_gui_agent_with_one_look/"
    t: "SpiritSight Agent: Advanced GUI Agent with One Look"
  - u: "tango_training-free_embodied_ai_agents_for_open-world_tasks/"
    t: "TANGO: Training-free Embodied AI Agents for Open-world Tasks"
  - u: "visual_agentic_ai_for_spatial_reasoning_with_a_dynamic_api/"
    t: "Visual Agentic AI for Spatial Reasoning with a Dynamic API"
item_total: 9
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🦾 LLM Agent

**📷 CVPR2025** · **9** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (42)](../../CVPR2026/llm_agent/index.md) · [🔬 ICLR2026 (162)](../../ICLR2026/llm_agent/index.md) · [💬 ACL2026 (82)](../../ACL2026/llm_agent/index.md) · [🧪 ICML2026 (59)](../../ICML2026/llm_agent/index.md) · [🤖 AAAI2026 (33)](../../AAAI2026/llm_agent/index.md) · [🧠 NeurIPS2025 (39)](../../NeurIPS2025/llm_agent/index.md)

🔥 **高频主题：** Agent ×4

**[ATA: Adaptive Transformation Agent for Text-Guided Subject-Position Variable Background Generation](ata_adaptive_transformation_agent_for_text-guided_subject-position_variable_back.md)**

:   提出 ATA（Adaptive Transformation Agent）框架，在文本引导的背景生成中实现对主体位置和姿态的精确控制，通过自适应变换模块动态调整主体在背景中的放置，兼顾视觉一致性和语义合理性。

**[Feature4X: Bridging Any Monocular Video to 4D Agentic AI with Versatile Gaussian Feature Fields](feature4x_bridging_any_monocular_video_to_4d_agentic_ai_with_versatile_gaussian_.md)**

:   提出 Feature4X，一个通用框架，从任意单目视频通过动态优化策略将多种 2D 视觉基础模型（SAM2、InternVideo2 等）的功能蒸馏到统一的 4D 高斯特征场中，首次实现基于 Gaussian Splatting 的视频基础模型 4D 特征提升，支持新视角下的 segment anything、几何/外观编辑和自由形式 VQA。

**[GUI-Xplore: Empowering Generalizable GUI Agents with One Exploration](gui-xplore_empowering_generalizable_gui_agents_with_one_exploration.md)**

:   提出 GUI-Xplore 数据集（312 个应用、32K+ QA 对、五层级任务）和 Xplore-Agent 框架（Action-aware GUI 建模 + GUI Transition Graph 推理），通过模拟"先探索再推理"的人类策略，在陌生应用上比 SOTA GUI Agent 提升约 10% StepSR。

**[RL-RC-DoT: A Block-level RL Agent for Task-Aware Video Compression](rl-rc-dot_a_block-level_rl_agent_for_task-aware_video_compression.md)**

:   提出 RL-RC-DoT，一个基于强化学习的宏块级量化参数（QP）控制 agent，用于任务感知视频压缩。通过将 QP 选择建模为 RL 的顺序决策问题，agent 学习在给定码率约束下为任务相关区域分配更多码率，在车辆检测和 ROI 显著性编码两个任务上显著提升性能。关键优势在于推理时不需要运行下游任务模型，适合边缘设备部署。

**[SceneAssistant: A Visual Feedback Agent for Open-Vocabulary 3D Scene Generation](sceneassistant_a_visual_feedback_agent_for_open-vocabulary_3d_scene_generation.md)**

:   提出 SceneAssistant，一个基于视觉反馈的闭环 agentic 框架，通过为 VLM 设计一套功能完备的 Action API（13个原子操作覆盖物体增删、6DoF空间操作、相机控制），让 VLM 以 ReAct 范式迭代生成开放词汇的 3D 场景，在室内（偏好率61.25%）和开放域（偏好率65.00%）场景中均大幅优于 Holodeck 和 SceneWeaver。

**[Sketchtopia: A Dataset and Foundational Agents for Benchmarking Asynchronous Multimodal Communication with Iconic Feedback](sketchtopia_a_dataset_and_foundational_agents_for_benchmarking_asynchronous_mult.md)**

:   提出 Sketchtopia 大规模数据集（20K+ 游戏会话、263K 草图、916 名玩家）和三组件 Agent 框架（ActionDecider + DRAWBOT + GUESSBOT），在 Pictionary 场景下研究异步、目标驱动的多模态协作通信，引入 AAO/FRS/MATS 三个新评估指标。

**[SpiritSight Agent: Advanced GUI Agent with One Look](spiritsight_agent_advanced_gui_agent_with_one_look.md)**

:   提出 SpiritSight，一个基于视觉的端到端 GUI agent，通过 573 万样本的多层级数据集 GUI-Lasagne 和 Universal Block Parsing (UBP) 方法解决动态高分辨率输入的定位歧义，SpiritSight-8B 在 Multimodal-Mind2Web 上非候选元素设置下 Step SR 达 52.7%，全面超越所有视觉/语言/混合方法。

**[TANGO: Training-free Embodied AI Agents for Open-world Tasks](tango_training-free_embodied_ai_agents_for_open-world_tasks.md)**

:   提出 TANGO，通过 LLM 的程序组合能力编排两个最小化的导航基础原语（PointGoal Navigation + 记忆驱动探索策略），无需任何任务特定训练，仅用 few-shot 示例即可在 Open-Set ObjectGoal Navigation、Multi-Modal Lifelong Navigation 和 Open Embodied QA 三个不同的具身 AI 任务上达到 SOTA，体现了"最小原语集 + LLM 组合"的通用性。

**[Visual Agentic AI for Spatial Reasoning with a Dynamic API](visual_agentic_ai_for_spatial_reasoning_with_a_dynamic_api.md)**

:   提出 VADAR，一种 agentic 程序合成方法用于 3D 空间推理。多个 LLM agent 协作生成 Pythonic API 并在求解过程中动态扩展新函数来解决常见子问题，克服了 VisProg/ViperGPT 等先前方法依赖静态人工定义 API 的局限。同时引入涉及多步空间定位和推理的新 benchmark，在 3D 理解任务上超越现有零样本方法。
