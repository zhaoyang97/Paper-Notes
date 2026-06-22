---
title: >-
  ACL2025 机器人/具身智能论文汇总 · 7篇论文解读
description: >-
  7篇ACL2025的机器人/具身智能方向论文解读，涵盖机器人、情感分析、LLM、RAG、持续学习、对话系统等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ACL2025"
  - "机器人/具身智能"
  - "论文解读"
  - "论文笔记"
  - "机器人"
  - "情感分析"
  - "LLM"
  - "RAG"
  - "持续学习"
  - "对话系统"
item_list:
  - u: "cheer-ekman_fine-grained_embodied_emotion_classification/"
    t: "CHEER-Ekman: Fine-grained Embodied Emotion Classification"
  - u: "dice_idiomaticity/"
    t: "Rolling the DICE on Idiomaticity: How LLMs Fail to Grasp Context"
  - u: "do_emotions_really_affect_argument_convincingness_a_dynamic_approach_with_llm-ba/"
    t: "Do Emotions Really Affect Argument Convincingness? A Dynamic Approach with LLM-based Manipulation Checks"
  - u: "drae_dynamic_retrieval-augmented_expert_networks_for_lifelong_learning_and_task_/"
    t: "DRAE: Dynamic Retrieval-Augmented Expert Networks for Lifelong Learning and Task Adaptation in Robotics"
  - u: "hierarchical-task-aware_multi-modal_mixture_of_incremental_lora_experts_for_embo/"
    t: "Task-aware MoILE: Hierarchical-Task-Aware Multi-modal Mixture of Incremental LoRA Experts for Embodied Continual Learning"
  - u: "self_percept_manipulation_detection/"
    t: "SELF-PERCEPT: Introspection Improves LLMs' Detection of Multi-Person Mental Manipulation in Conversations"
  - u: "vulnerability_of_llms_to_vertically_aligned_text_manipulations/"
    t: "Vulnerability of LLMs to Vertically Aligned Text Manipulations"
item_total: 7
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🤖 机器人/具身智能

**💬 ACL2025** · **7** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (146)](../../CVPR2026/robotics/index.md) · [🔬 ICLR2026 (162)](../../ICLR2026/robotics/index.md) · [💬 ACL2026 (11)](../../ACL2026/robotics/index.md) · [🧪 ICML2026 (53)](../../ICML2026/robotics/index.md) · [🤖 AAAI2026 (30)](../../AAAI2026/robotics/index.md) · [🧠 NeurIPS2025 (75)](../../NeurIPS2025/robotics/index.md)

🔥 **高频主题：** 机器人 ×5 · 情感分析 ×2

**[CHEER-Ekman: Fine-grained Embodied Emotion Classification](cheer-ekman_fine-grained_embodied_emotion_classification.md)**

:   本文提出CHEER-Ekman数据集，将CHEER数据集的二元具身情感标注扩展为Ekman六类基础情绪，并采用基于LLM的自动Best-Worst Scaling（BWS）技术实现无需任务特定训练的细粒度情感分类，性能超越有监督BERT。

**[Rolling the DICE on Idiomaticity: How LLMs Fail to Grasp Context](dice_idiomaticity.md)**

:   提出 DICE 数据集（2066 句，402 个习语），通过严格控制习语形式一致的对比评测，揭示 LLM 在需要上下文理解才能消歧习语（字面 vs 比喻义）时存在系统性缺陷。

**[Do Emotions Really Affect Argument Convincingness? A Dynamic Approach with LLM-based Manipulation Checks](do_emotions_really_affect_argument_convincingness_a_dynamic_approach_with_llm-ba.md)**

:   提出一种受心理学操控检验启发的动态框架，利用LLM调节论证的情感强度，系统考察情感对论证说服力的因果影响，发现超过半数情况下人类的说服力判断不受情感变化影响，而当情感有影响时更多是增强而非削弱说服力。

**[DRAE: Dynamic Retrieval-Augmented Expert Networks for Lifelong Learning and Task Adaptation in Robotics](drae_dynamic_retrieval-augmented_expert_networks_for_lifelong_learning_and_task_.md)**

:   提出 DRAE 框架，整合动态 MoE 路由、参数化 RAG（P-RAG）、三层认知控制架构（ReflexNet-SchemaPlanner-HyperOptima）和 DPMM 终身知识保留，在机器人操作和自动驾驶任务上平均成功率达 82.5%，有效缓解灾难性遗忘。

**[Task-aware MoILE: Hierarchical-Task-Aware Multi-modal Mixture of Incremental LoRA Experts for Embodied Continual Learning](hierarchical-task-aware_multi-modal_mixture_of_incremental_lora_experts_for_embo.md)**

:   提出层次化具身持续学习设置（HEC），将 agent 学习分为高层指令和低层动作两级，并设计 Task-aware MoILE 方法——通过跨模态聚类识别任务、双路由器选择 LoRA 专家、SVD 正交训练保留旧知识，在 5 种增量学习场景中遗忘率降至 3.37%（vs 前 SOTA 7.44%）。

**[SELF-PERCEPT: Introspection Improves LLMs' Detection of Multi-Person Mental Manipulation in Conversations](self_percept_manipulation_detection.md)**

:   提出 SELF-PERCEPT 两阶段 prompting 框架，借鉴心理学自我知觉理论（Self-Perception Theory），引导 LLM 先观察对话参与者的行为线索再推断内在态度，显著提升多人多轮对话中心理操纵的检测效果。

**[Vulnerability of LLMs to Vertically Aligned Text Manipulations](vulnerability_of_llms_to_vertically_aligned_text_manipulations.md)**

:   本文系统揭示了LLM对垂直排列文本输入的严重脆弱性：仅将少量关键词垂直排列即可导致文本分类准确率下降25-45个百分点，CoT推理无法缓解此问题，但精心设计的few-shot learning可有效恢复性能。
