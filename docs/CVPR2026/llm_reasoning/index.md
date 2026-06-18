---
title: >-
  CVPR2026 LLMReasoning论文汇总 · 20篇论文解读
description: >-
  20篇CVPR2026的 LLM Reasoning 方向论文解读，涵盖推理、多模态、自动驾驶、强化学习、问答、医学影像等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "CVPR2026"
  - "LLM Reasoning"
  - "论文解读"
  - "论文笔记"
  - "推理"
  - "多模态"
  - "自动驾驶"
  - "强化学习"
  - "问答"
  - "医学影像"
item_list:
  - u: "agile_deliberation_concept_deliberation_for_subjective_visual_classification/"
    t: "Agile Deliberation: Concept Deliberation for Subjective Visual Classification"
  - u: "appo_attention-guided_perception_policy_optimization_for_video_reasoning/"
    t: "APPO: Attention-guided Perception Policy Optimization for Video Reasoning"
  - u: "dynamic_important_example_mining_for_reinforcement_finetuning/"
    t: "Dynamic Important Example Mining for Reinforcement Finetuning"
  - u: "e-comiq-zh_a_human-aligned_dataset_and_benchmark_for_fine-grained_evaluation_of_/"
    t: "E-comIQ-ZH: A Human-Aligned Dataset and Benchmark for Fine-Grained Evaluation of E-commerce Posters with Chain-of-Thought"
  - u: "eaglevision_a_dual-stage_framework_with_bev-grounding-based_chain-of-thought_for/"
    t: "EagleVision: A Dual-Stage Framework with BEV-grounding-based Chain-of-Thought for Spatial Intelligence"
  - u: "firescope_wildfire_risk_raster_prediction_with_a_chain-of-thought_oracle/"
    t: "FireScope: Wildfire Risk Raster Prediction with a Chain-of-Thought Oracle"
  - u: "hilbert-geo_solving_solid_geometric_problems_by_neural-symbolic_reasoning/"
    t: "Hilbert-Geo: Solving Solid Geometric Problems by Neural-Symbolic Reasoning"
  - u: "human-like_abstract_visual_reasoning_via_understanding_and_solving_reasoning_loo/"
    t: "Human-like Abstract Visual Reasoning via Understanding and Solving Reasoning Loop"
  - u: "improving_vision-language_models_with_perception-centric_process_reward_models/"
    t: "Improving Vision-language Models with Perception-centric Process Reward Models"
  - u: "latent_chain-of-thought_world_modeling_for_end-to-end_autonomous_driving/"
    t: "Latent Chain-of-Thought World Modeling for End-to-End Autonomous Driving"
  - u: "rationale-enhanced_decoding_for_multi-modal_chain-of-thought/"
    t: "Rationale-Enhanced Decoding for Multi-modal Chain-of-Thought"
  - u: "reasoning_palette_modulating_reasoning_via_latent_contextualization_for_controll/"
    t: "Reasoning Palette: Modulating Reasoning via Latent Contextualization for Controllable Exploration for (V)LMs"
  - u: "reinforcing_structured_chain-of-thought_for_video_understanding/"
    t: "Reinforcing Structured Chain-of-Thought for Video Understanding"
  - u: "relax_reasoning_with_latent_exploration_for_large_reasoning_models/"
    t: "ReLaX: Reasoning with Latent Exploration for Large Reasoning Models"
  - u: "revisiting_the_necessity_of_lengthy_chain-of-thought_in_vision-centric_reasoning/"
    t: "Revisiting the Necessity of Lengthy Chain-of-Thought in Vision-centric Reasoning Generalization"
  - u: "scaling_agentic_reinforcement_learning_for_tool-integrated_reasoning_in_vlms/"
    t: "Scaling Agentic Reinforcement Learning for Tool-Integrated Reasoning in VLMs"
  - u: "step-cot_stepwise_visual_chain-of-thought_for_medical_visual_question_answering/"
    t: "Step-CoT: Stepwise Visual Chain-of-Thought for Medical Visual Question Answering"
  - u: "think-as-you-see_streaming_chain-of-thought_reasoning_for_large_vision-language_/"
    t: "Think-as-You-See: Streaming Chain-of-Thought Reasoning for Large Vision-Language Models"
  - u: "think_360deg_beyond_depth_evaluating_the_width-centric_reasoning_capability_of_m/"
    t: "Think 360°: Beyond Depth — Evaluating the Width-centric Reasoning Capability of MLLMs"
  - u: "visref_visual_refocusing_test_time_scaling/"
    t: "VisRef: Visual Refocusing while Thinking Improves Test-Time Scaling in Multi-Modal Large Reasoning Models"
item_total: 20
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 💡 LLM Reasoning

**📷 CVPR2026** · **20** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (63)](../../ICML2026/llm_reasoning/index.md) · [💬 ACL2026 (81)](../../ACL2026/llm_reasoning/index.md) · [🔬 ICLR2026 (82)](../../ICLR2026/llm_reasoning/index.md) · [🤖 AAAI2026 (37)](../../AAAI2026/llm_reasoning/index.md) · [🧠 NeurIPS2025 (82)](../../NeurIPS2025/llm_reasoning/index.md) · [📹 ICCV2025 (3)](../../ICCV2025/llm_reasoning/index.md)

🔥 **高频主题：** 推理 ×17 · 多模态 ×2

**[Agile Deliberation: Concept Deliberation for Subjective Visual Classification](agile_deliberation_concept_deliberation_for_subjective_visual_classification.md)**

:   针对"健康食物""标题党"这类边界模糊的主观概念，提出 Agile Deliberation 人在回路框架：先把概念分解成正/负子概念层级，再迭代地检索"语义边界样本"让用户标注与反思、并自动把反馈编译成 VLM 提示，使图像分类器逐轮对齐用户不断演化的意图，18 场真人实验中 F1 比自动分解基线高 7.5%、比手动审议高 3%+。

**[APPO: Attention-guided Perception Policy Optimization for Video Reasoning](appo_attention-guided_perception_policy_optimization_for_video_reasoning.md)**

:   APPO 发现「视频推理瓶颈在感知而非推理」，于是用模型自身对视频帧的注意力把稀疏 outcome reward 转成 token 级稠密奖励——让不同回答里聚焦同一关键帧的「组内感知 token」按高/低奖励差异化加权学习，在 Qwen2.5-VL-3/7B 上稳定超过 GRPO 和 DAPO（0.5%∼4%）。

**[Dynamic Important Example Mining for Reinforcement Finetuning](dynamic_important_example_mining_for_reinforcement_finetuning.md)**

:   DIEM 在 RFT（GRPO/PPO 等）的每一步训练里，用「单样本梯度与 batch 总梯度的内积」实时估计每条样本对当前策略改进的边际贡献，再解一个保持梯度模长不变的约束优化问题给样本重加权，几乎零额外开销（+1.3% 时间）就让多模态推理 benchmark 平均提升 1–6 个点。

**[E-comIQ-ZH: A Human-Aligned Dataset and Benchmark for Fine-Grained Evaluation of E-commerce Posters with Chain-of-Thought](e-comiq-zh_a_human-aligned_dataset_and_benchmark_for_fine-grained_evaluation_of_.md)**

:   构建首个面向中文电商海报的多维度质量评估框架 E-comIQ-ZH，包含18K专家标注数据集（含CoT推理链）、专用评估模型 E-comIQ-M（SFT+GRPO训练）和标准化基准 E-comIQ-Bench。

**[EagleVision: A Dual-Stage Framework with BEV-grounding-based Chain-of-Thought for Spatial Intelligence](eaglevision_a_dual-stage_framework_with_bev-grounding-based_chain-of-thought_for.md)**

:   提出EagleVision双阶段框架，宏观感知阶段用语义-视角融合DPP(SPF-DPP)在SE(3)空间联合优化语义相关性和视角多样性选择关键帧，微观验证阶段让模型在BEV平面上主动查询新视角帧进行迭代空间CoT推理（假设→查看→验证闭环），查询策略纯RL训练无需人工标注，在VSI-Bench和SQA3D上达开源SOTA。

**[FireScope: Wildfire Risk Raster Prediction with a Chain-of-Thought Oracle](firescope_wildfire_risk_raster_prediction_with_a_chain-of-thought_oracle.md)**

:   用一个 GRPO 微调、会写思维链的 VLM（Oracle）先把卫星图+气候推理成一个标量野火风险分，再用 FiLM 把这个分喂给轻量视觉 Encoder-Decoder 去生成高分辨率连续风险栅格——在「美国训练、欧洲测试」的跨洲设定下，显式语言推理显著提升了分布外泛化，且推理痕迹可被野火专家复原、可解释。

**[Hilbert-Geo: Solving Solid Geometric Problems by Neural-Symbolic Reasoning](hilbert-geo_solving_solid_geometric_problems_by_neural-symbolic_reasoning.md)**

:   Hilbert-Geo 是首个面向**立体几何**的统一形式化语言框架（含谓词库 + 定理库），用"先解析后推理"的 Parse2Reason 方法——先让多模态大模型把文字题面和 3D 图示翻译成形式化的条件描述语言（CDL），再用专门的符号推理引擎做严格的定理搜索，从而把 MLLM 在立体几何上 50% 出头的准确率提到 77.3%，逼近人类水平。

**[Human-like Abstract Visual Reasoning via Understanding and Solving Reasoning Loop](human-like_abstract_visual_reasoning_via_understanding_and_solving_reasoning_loo.md)**

:   把人类"理解—求解—再理解"的迭代认知拆成可循环交互的理解模块（UM）与求解模块（SM），辅以表征同构约束和自适应停止机制，让一个仅 7M 参数的小模型在 ARC-AGI-1 上达到 47.2% 准确率，超过 TRM 与一众通用大模型。

**[Improving Vision-language Models with Perception-centric Process Reward Models](improving_vision-language_models_with_perception-centric_process_reward_models.md)**

:   针对 VLM 强化学习里"只有结果奖励、定位不到错在哪一步"的痛点，本文训练了一个感知中心的过程奖励模型 Perceval，逐条核查推理链里的图文一致性、标出幻觉 token，再把这个信号同时用于训练（token 级优势重分配改造 GRPO）和推理（截断-重生成），在多个视觉推理基准上稳定涨点，并意外地把"感知更准"泛化成了"整体推理更强"。

**[Latent Chain-of-Thought World Modeling for End-to-End Autonomous Driving](latent_chain-of-thought_world_modeling_for_end-to-end_autonomous_driving.md)**

:   LCDrive 提出潜在链式思考（Latent CoT）框架，用动作提议token和世界模型预测token替代自然语言CoT进行推理，通过冷启动+RL后训练实现更低延迟、更好轨迹质量的端到端自动驾驶。

**[Rationale-Enhanced Decoding for Multi-modal Chain-of-Thought](rationale-enhanced_decoding_for_multi-modal_chain-of-thought.md)**

:   发现现有LVLM在CoT推理时实际上忽略了中间rationale的内容，提出 RED (Rationale-Enhanced Decoding)——将图像条件和rationale条件的next-token分布在logit层面相乘，理论上等价于KL约束奖励最大化的最优解，无需训练即可显著提升多模态推理准确率。

**[Reasoning Palette: Modulating Reasoning via Latent Contextualization for Controllable Exploration for (V)LMs](reasoning_palette_modulating_reasoning_via_latent_contextualization_for_controll.md)**

:   这篇论文用一个 VAE 学到的隐空间给 (V)LM 注入一个"推理调色板"——每采一个隐变量就解码成一段可学习前缀贴到 prompt 前面，让模型在生成第一个 token 之前就选定某种推理风格，从而把 RL 里的"token 级随机采样"升级成"策略级结构化探索"，在多个数学推理 benchmark 上稳定超过标准 GRPO/RLOO。

**[Reinforcing Structured Chain-of-Thought for Video Understanding](reinforcing_structured_chain-of-thought_for_video_understanding.md)**

:   提出 SDRL（Summary-Driven Reinforcement Learning），一种无需 SFT 的单阶段 RL 框架，通过结构化 CoT（Summarize→Think→Answer）和两个自监督机制（CVK 和 DVR）增强视频时序推理，在 7 个 VideoQA 基准上达到 SOTA。

**[ReLaX: Reasoning with Latent Exploration for Large Reasoning Models](relax_reasoning_with_latent_exploration_for_large_reasoning_models.md)**

:   ReLaX 不再像现有方法那样在 token 层面强行抬高熵来对抗 RLVR 的熵坍缩，而是用 Koopman 算子把大推理模型的隐状态动力学线性化、提出"动态谱发散（DSD）"这个量化策略内部计算灵活度的指标，再把它整形进 GRPO 目标，在 7 个多模态 + 6 个纯文本推理基准上刷新同规模 SOTA。

**[Revisiting the Necessity of Lengthy Chain-of-Thought in Vision-centric Reasoning Generalization](revisiting_the_necessity_of_lengthy_chain-of-thought_in_vision-centric_reasoning.md)**

:   作者用可控的迷宫导航任务系统对比了语言 CoT、grounding CoT、视觉 CoT 三类「think with image」式监督格式，发现更长/更花哨的视觉 CoT 只能加快收敛、抬不高最终天花板，而**只保留最少 grounding 信息的极简 CoT（一条坐标路径）反而泛化最好**，提出「short is long」效应并给出构造可泛化视觉推理 SFT 数据的实操指南。

**[Scaling Agentic Reinforcement Learning for Tool-Integrated Reasoning in VLMs](scaling_agentic_reinforcement_learning_for_tool-integrated_reasoning_in_vlms.md)**

:   本文提出可规模化的视觉工具智能体训练环境 VISTA-Gym（7 类任务、13 个数据集、26 个标准化视觉工具），并在其中用「模仿学习预热 + 多轮 GRPO 在线 RL」训练出 VISTA-R1，让 8B 级 VLM 学会在推理过程中动态选择/调用/协同视觉工具，在 11 个推理密集型 VQA 基准上比同体量 SOTA 高出 9.51%–18.72%。

**[Step-CoT: Stepwise Visual Chain-of-Thought for Medical Visual Question Answering](step-cot_stepwise_visual_chain-of-thought_for_medical_visual_question_answering.md)**

:   构建首个对齐临床诊断工作流的结构化多步CoT医学推理数据集Step-CoT（10K+病例/70K QA对），并提出基于图注意力网络的教师-学生框架实现逐步推理监督，提升Med-VQA的准确性和可解释性。

**[Think-as-You-See: Streaming Chain-of-Thought Reasoning for Large Vision-Language Models](think-as-you-see_streaming_chain-of-thought_reasoning_for_large_vision-language_.md)**

:   TaYS 把大视觉语言模型（LVLM）的视频推理从「看完整段再想」的批处理范式，改造成「边看边想」的流式范式——通过流式注意力掩码、解耦位置编码和并行双 KV 缓存三件套，让推理与视频帧同步增量推进，在 VideoEspresso 上把首 token 延迟从 10.6s 压到近乎为零、推理-事件偏差降低 55%，同时推理准确率提升 2.9%。

**[Think 360°: Beyond Depth — Evaluating the Width-centric Reasoning Capability of MLLMs](think_360deg_beyond_depth_evaluating_the_width-centric_reasoning_capability_of_m.md)**

:   作者提出 Think360°，一个把"推理宽度"（广度搜索、多约束剪枝、回溯）作为与"推理深度"（长链顺序推理）正交维度的多模态基准——精选 1200+ 道跨域高质量题目，并设计一套 Tree-of-Thought 评测协议同时量化宽度与深度准确率，对 12 大系列 30+ MLLM 的测评显示：当前模型能做长链深推、却普遍不擅长"宽搜索 + 深链"结合的洞察式推理。

**[VisRef: Visual Refocusing while Thinking Improves Test-Time Scaling in Multi-Modal Large Reasoning Models](visref_visual_refocusing_test_time_scaling.md)**

:   提出 VisRef，一个免训练的视觉重聚焦框架——在多模态大推理模型（MLRM）的推理过程中，通过行列式点过程（DPP）在每步自适应选择与当前推理状态语义相关且视觉覆盖多样的 token 子集并重新注入，同时用基于熵的停止准则防止过度推理，在固定计算预算下将视觉推理准确率提升最高 6.4%。
