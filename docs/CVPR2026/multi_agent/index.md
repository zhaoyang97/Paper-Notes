---
title: >-
  CVPR2026 Multi-Agent论文汇总 · 9篇论文解读
description: >-
  9篇CVPR2026的 Multi-Agent 方向论文解读，涵盖 Agent、少样本学习、推理等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "CVPR2026"
  - "Multi-Agent"
  - "论文解读"
  - "论文笔记"
  - "Agent"
  - "少样本学习"
  - "推理"
item_list:
  - u: "agent4faceforgery_multi-agent_llm_framework_for_realistic_face_forgery_detection/"
    t: "Agent4FaceForgery: Multi-Agent LLM Framework for Realistic Face Forgery Detection"
  - u: "agentdet_a_shared-blackboard_multi-agent_framework_for_zero-few-shot_object_dete/"
    t: "AgentDet: A Shared-Blackboard Multi-Agent Framework for Zero-/Few-Shot Object Detection"
  - u: "motor-bench_a_real-world_dataset_and_multi-agent_framework_for_zero-shot_human_m/"
    t: "MOTOR-Bench: A Real-world Dataset and Multi-agent Framework for Zero-shot Human Mental State Understanding"
  - u: "paper2figure_a_multi-agent_collaborative_system_for_figure_generation_towards_ac/"
    t: "Paper2Figure: A Multi-Agent Collaborative System for Figure Generation Towards Academic Research Paper"
  - u: "refer-agent_a_collaborative_multi-agent_system_with_reasoning_and_reflection_for/"
    t: "Refer-Agent: A Collaborative Multi-Agent System with Reasoning and Reflection for Referring Video Object Segmentation"
  - u: "scieducator_scientific_video_understanding_and_educating_via_deming-cycle_multi-/"
    t: "SciEducator: Scientific Video Understanding and Educating via Deming-Cycle Multi-Agent System"
  - u: "symphony_a_cognitively-inspired_multi-agent_system_for_long-video_understanding/"
    t: "Symphony: A Cognitively-Inspired Multi-Agent System for Long-Video Understanding"
  - u: "tackling_model_bias_via_game-theoretic_multi-agent_collaboration_framework_for_h/"
    t: "Tackling Model Bias via Game-theoretic Multi-agent Collaboration Framework for Hateful Meme Classification"
  - u: "visual_document_understanding_and_reasoning_a_multi-agent_collaboration_framewor/"
    t: "Visual Document Understanding and Reasoning: A Multi-Agent Collaboration Framework with Agent-Wise Adaptive Test-Time Scaling"
item_total: 9
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 👥 Multi-Agent

**📷 CVPR2026** · **9** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (15)](../../ICML2026/multi_agent/index.md) · [💬 ACL2026 (38)](../../ACL2026/multi_agent/index.md) · [🔬 ICLR2026 (15)](../../ICLR2026/multi_agent/index.md) · [🤖 AAAI2026 (26)](../../AAAI2026/multi_agent/index.md) · [🧠 NeurIPS2025 (17)](../../NeurIPS2025/multi_agent/index.md) · [🧪 ICML2025 (7)](../../ICML2025/multi_agent/index.md)

🔥 **高频主题：** Agent ×9 · 少样本学习 ×2 · 推理 ×2

**[Agent4FaceForgery: Multi-Agent LLM Framework for Realistic Face Forgery Detection](agent4faceforgery_multi-agent_llm_framework_for_realistic_face_forgery_detection.md)**

:   用一套 LLM 驱动的多智能体系统去"扮演"造假者和社交网络上的吃瓜群众，模拟人脸伪造从创作到传播的完整生命周期，合成出带文图一致性标注的训练数据，让 deepfake 检测器在跨域、跨伪造算法的真实场景下涨点显著（如 Celeb-DF AUC 从 70% 级提到 87.1%）。

**[AgentDet: A Shared-Blackboard Multi-Agent Framework for Zero-/Few-Shot Object Detection](agentdet_a_shared-blackboard_multi-agent_framework_for_zero-few-shot_object_dete.md)**

:   AgentDet 把零/少样本目标检测拆成 Scout / Pinner / Curator / Judge 四个 LLM 智能体，通过一块"共享黑板"+一个 patch 级"知识库"协作：把视觉证据碎片化存进知识库、组合成整体文本线索喂给 LLM 做框预测，并且只训练 Judge 一个智能体，就在 PASCAL VOC / COCO 的 ZSOD/FSOD 上做到了与 SOTA 强竞争的结果。

**[MOTOR-Bench: A Real-world Dataset and Multi-agent Framework for Zero-shot Human Mental State Understanding](motor-bench_a_real-world_dataset_and_multi-agent_framework_for_zero-shot_human_m.md)**

:   针对「从可观察行为推断深层心理状态」缺少结构化标注这一空白，本文构建了真实课堂协作学习场景的多模态数据集 MOTOR-dataset（1,440 段视频，行为/认知/情绪三维标注），并提出基于自我调节学习理论(SRL)的推理型多智能体框架 MOTOR-MAS——三个专职 agent 按「行为→认知→情绪」顺序级联推理，把前一阶段的预测当锚点喂给后一阶段，零样本下 Macro-F1 达 42.77，比最强单模型基线高 15.93 分。

**[Paper2Figure: A Multi-Agent Collaborative System for Figure Generation Towards Academic Research Paper](paper2figure_a_multi-agent_collaborative_system_for_figure_generation_towards_ac.md)**

:   Paper2Figure 用「生成智能体 + 精修智能体」双多智能体系统，先把论文文字描述翻译成自研的结构化中间语言 FigScript、渲染成图，再让一组批评-修订智能体闭环自纠，配上可交互的 Web 编辑器把人类控制权交还给作者，在自建的 Paper2Figure Bench 上准确性、美观度、完整度全面超过 SVG/Mermaid 代码生成和文生图基线（综合 +14.1%）。

**[Refer-Agent: A Collaborative Multi-Agent System with Reasoning and Reflection for Referring Video Object Segmentation](refer-agent_a_collaborative_multi-agent_system_with_reasoning_and_reflection_for.md)**

:   Refer-Agent 把指代视频目标分割（RVOS）拆成「帧选择→意图分析→目标定位→掩码生成」的分步推理流水线，再叠一层由提问者-回答者构成的双阶段 Chain-of-Reflection（存在性反思 + 一致性反思）在推理与反思之间交替自纠，从而在完全免训练、仅用 9B 开源 MLLM 的条件下，于 5 个 RVOS 基准上同时超过 SFT 方法和接入 GPT-4o 的零样本方法。

**[SciEducator: Scientific Video Understanding and Educating via Deming-Cycle Multi-Agent System](scieducator_scientific_video_understanding_and_educating_via_deming-cycle_multi-.md)**

:   SciEducator 把管理学里的戴明环（Plan–Do–Study–Act）改造成一个会自我进化的多智能体闭环，让系统反复"规划—执行—复盘—改进"地读懂科学实验视频，并进一步生成图文音并茂的儿童科普电子手册，在自建的 SciVBench 上大幅超过 GPT-4o、Gemini 等闭源 MLLM 和现有视频 Agent。

**[Symphony: A Cognitively-Inspired Multi-Agent System for Long-Video Understanding](symphony_a_cognitively-inspired_multi-agent_system_for_long-video_understanding.md)**

:   Symphony 模仿人类认知把长视频理解拆给"按能力维度分工"的多个专用智能体（规划、反思、grounding、字幕、视觉感知），用一个 Actor-Critic 式的反思增强动态协作机制反复纠偏推理，并为复杂问题设计了一个会"先扩写查询再用 VLM 打分"的 grounding 智能体，在 LVBench、LongVideoBench、Video-MME、MLVU 四个基准上达到 SOTA，LVBench 比前最优高 5.0%。

**[Tackling Model Bias via Game-theoretic Multi-agent Collaboration Framework for Hateful Meme Classification](tackling_model_bias_via_game-theoretic_multi-agent_collaboration_framework_for_h.md)**

:   GECO 把三个大型多模态模型加一个可学习智能体、一个主决策智能体组织成一场正则化博弈，用"混合奖励"驱动它们就正确标签达成共识，从而压制单模型与模型间的认知偏差，在五个仇恨表情包基准上刷新 SOTA。

**[Visual Document Understanding and Reasoning: A Multi-Agent Collaboration Framework with Agent-Wise Adaptive Test-Time Scaling](visual_document_understanding_and_reasoning_a_multi-agent_collaboration_framewor.md)**

:   MACT 把"单模型一把梭"的视觉文档问答拆成规划、执行、判断、回答四个分工明确的智能体，并按每个智能体的认知负荷自适应分配测试时算力（而非统一堆参数），在 15 个基准上以 <30B 参数稳进前三、平均比基座模型提升 9.9–11.5%。
