---
title: >-
  AAAI2026 VLMReasoning论文汇总 · 10篇论文解读
description: >-
  10篇AAAI2026的 VLM Reasoning 方向论文解读，涵盖推理、多模态、Agent、LLM、布局/合成、对抗鲁棒等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "AAAI2026"
  - "VLM Reasoning"
  - "论文解读"
  - "论文笔记"
  - "推理"
  - "多模态"
  - "Agent"
  - "LLM"
  - "布局/合成"
  - "对抗鲁棒"
item_list:
  - u: "abductivemllm_boosting_visual_abductive_reasoning_within_mll/"
    t: "AbductiveMLLM: Boosting Visual Abductive Reasoning Within MLLMs"
  - u: "astar_boosting_multimodal_reasoning_with_automated_structure/"
    t: "AStar: Boosting Multimodal Reasoning with Automated Structured Thinking"
  - u: "concept-rulenet_grounded_multi-agent_neurosymbolic_reasoning/"
    t: "Concept-RuleNet: Grounded Multi-Agent Neurosymbolic Reasoning in Vision Language Models"
  - u: "crossvid_a_comprehensive_benchmark_for_evaluating_cross-vide/"
    t: "CrossVid: A Comprehensive Benchmark for Evaluating Cross-Video Reasoning in Multimodal Large Language Models"
  - u: "finmmdocr_benchmarking_financial_multimodal_reasoning_with_scenario_awareness_do/"
    t: "FinMMDocR: Benchmarking Financial Multimodal Reasoning with Scenario Awareness, Document Understanding, and Multi-Step Computation"
  - u: "graph-of-mark_promote_spatial_reasoning_in_multimodal_langua/"
    t: "Graph-of-Mark: Promote Spatial Reasoning in Multimodal Language Models with Graph-Based Visual Prompting"
  - u: "leveraging_textual_compositional_reasoning_for_robust_change_captioning/"
    t: "Leveraging Textual Compositional Reasoning for Robust Change Captioning"
  - u: "stola_self-adaptive_touch-language_framework_with_tactile_commonsense_reasoning_/"
    t: "SToLa: Self-Adaptive Touch-Language Framework with Tactile Commonsense Reasoning in Open-Ended Scenarios"
  - u: "tri-bench_stress-testing_vlm_reliability_on_spatial_reasoning_under_camera_tilt_/"
    t: "Tri-Bench: Stress-Testing VLM Reliability on Spatial Reasoning under Camera Tilt and Object Interference"
  - u: "yes_florence_i_will_do_better_next_time_agentic_feedback_reasoning_for_humorous_/"
    t: "Yes FLoReNce, I Will Do Better Next Time! Agentic Feedback Reasoning for Humorous Meme Detection"
item_total: 10
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧠 VLM Reasoning

**🤖 AAAI2026** · **10** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (150)](../../CVPR2026/vlm_reasoning/index.md) · [🔬 ICLR2026 (112)](../../ICLR2026/vlm_reasoning/index.md) · [💬 ACL2026 (32)](../../ACL2026/vlm_reasoning/index.md) · [🧪 ICML2026 (31)](../../ICML2026/vlm_reasoning/index.md) · [🧠 NeurIPS2025 (30)](../../NeurIPS2025/vlm_reasoning/index.md) · [📹 ICCV2025 (15)](../../ICCV2025/vlm_reasoning/index.md)

🔥 **高频主题：** 推理 ×10 · 多模态 ×6

**[AbductiveMLLM: Boosting Visual Abductive Reasoning Within MLLMs](abductivemllm_boosting_visual_abductive_reasoning_within_mll.md)**

:   受人类认知中"语言溯因+图像想象"双模式启发，提出 AbductiveMLLM，通过 Reasoner（因果对比学习筛选假设）和 Imaginer（扩散模型图像化推理）两个协同组件增强 MLLM 的视觉溯因推理能力，在 VAR 和 YouCookII 基准上取得 SOTA。

**[AStar: Boosting Multimodal Reasoning with Automated Structured Thinking](astar_boosting_multimodal_reasoning_with_automated_structure.md)**

:   提出AStar，一种training-free的多模态推理范式，通过从500个种子样本中构建高层"thought cards"推理模板库，在推理时自适应检索最优模板引导MLLM结构化推理，7B模型在MathVerse上达53.9%准确率（超越GPT-4o的50.2%），仅需50分钟预处理时间且无需训练。

**[Concept-RuleNet: Grounded Multi-Agent Neurosymbolic Reasoning in Vision Language Models](concept-rulenet_grounded_multi-agent_neurosymbolic_reasoning.md)**

:   提出Concept-RuleNet——一个三智能体协作的神经符号推理框架，通过从训练图像中提取视觉概念来条件化符号生成和规则构建，解决了现有方法（如Symbol-LLM）仅依赖标签导致的符号幻觉和不代表性问题，在5个OOD基准上平均提升~5%准确率，幻觉符号减少达50%。

**[CrossVid: A Comprehensive Benchmark for Evaluating Cross-Video Reasoning in Multimodal Large Language Models](crossvid_a_comprehensive_benchmark_for_evaluating_cross-vide.md)**

:   提出首个系统评估多模态大语言模型（MLLM）跨视频推理（Cross-Video Reasoning, CVR）能力的综合基准CrossVid，涵盖4个维度10个任务、5,331个视频和9,015个QA对，实验揭示当前最佳模型Gemini-2.5-Pro仅达50.4%准确率，远低于人类89.2%。

**[FinMMDocR: Benchmarking Financial Multimodal Reasoning with Scenario Awareness, Document Understanding, and Multi-Step Computation](finmmdocr_benchmarking_financial_multimodal_reasoning_with_scenario_awareness_do.md)**

:   本文提出FinMMDocR，一个面向真实金融场景的双语多模态推理基准，包含1200道专家标注的数值推理题目，涵盖12类隐式金融情景、9类长文档（平均50.8页）和平均11步推理链，最强MLLM (o4-mini-high) 仅达58%准确率，揭示现有模型在复杂金融推理中的严重不足。

**[Graph-of-Mark: Promote Spatial Reasoning in Multimodal Language Models with Graph-Based Visual Prompting](graph-of-mark_promote_spatial_reasoning_in_multimodal_langua.md)**

:   提出 Graph-of-Mark (GoM)，一种无需训练的像素级视觉提示方法，通过在输入图像上直接叠加深度感知的场景图（包含节点和有向边），显式编码物体间的空间关系，使多模态语言模型在 VQA 和定位任务中的零样本空间推理准确率最高提升 11 个百分点。

**[Leveraging Textual Compositional Reasoning for Robust Change Captioning](leveraging_textual_compositional_reasoning_for_robust_change_captioning.md)**

:   提出 CORTEX 框架，通过引入 VLM 生成的组合推理文本作为显式线索，结合图像-文本双重对齐模块（ITDA），增强纯视觉变化描述方法对物体关系和空间配置等结构化语义的理解能力。

**[SToLa: Self-Adaptive Touch-Language Framework with Tactile Commonsense Reasoning in Open-Ended Scenarios](stola_self-adaptive_touch-language_framework_with_tactile_commonsense_reasoning_.md)**

:   SToLa 提出首个基于混合专家（MoE）的触觉-语言框架，通过动态路由机制管理触觉和语言两种模态的差异，并构建了覆盖8种物理属性、4种交互特征的开放式触觉常识推理数据集 TactileBench，在 PhysiCLeAR 基准上以 7B 参数量超越 13B 的 Octopi 取得 SOTA。

**[Tri-Bench: Stress-Testing VLM Reliability on Spatial Reasoning under Camera Tilt and Object Interference](tri-bench_stress-testing_vlm_reliability_on_spatial_reasoning_under_camera_tilt_.md)**

:   Tri-Bench 是一个包含400张实拍三角形图像的紧凑基准，通过控制相机姿态（平面/倾斜）和物体干扰两个因素，系统测试了四个领先VLM的空间几何推理能力，发现模型默认依赖2D图像平面线索而非3D真实几何（即使提供了明确的参考框架提示），在非多数类形状上准确率降至接近0%。

**[Yes FLoReNce, I Will Do Better Next Time! Agentic Feedback Reasoning for Humorous Meme Detection](yes_florence_i_will_do_better_next_time_agentic_feedback_reasoning_for_humorous_.md)**

:   提出 FLoReNce 框架，将幽默 meme 理解建模为闭环控制系统，通过 Judge 反馈+PID 控制器+非参数知识库的闭环学习，在推理时通过检索相似经验调制 prompt，使冻结的 VLM 实现自适应推理，无需微调即可显著提升预测和解释质量。
