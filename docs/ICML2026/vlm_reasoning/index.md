---
title: >-
  ICML2026 VLMReasoning论文汇总 · 20篇论文解读
description: >-
  20篇ICML2026的 VLM Reasoning 方向论文解读，涵盖推理、多模态、LLM等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2026"
  - "VLM Reasoning"
  - "论文解读"
  - "论文笔记"
  - "推理"
  - "多模态"
  - "LLM"
item_list:
  - u: "3viewsense_spatial_and_mental_perspective_reasoning_from_orthographic_views_in_v/"
    t: "3ViewSense: Spatial and Mental Perspective Reasoning from Orthographic Views in Vision-Language Models"
  - u: "active_exploring_like_a_pigeon_reinforcing_spatial_reasoning_via_agentic_vision-/"
    t: "Active Exploring like a Pigeon: Reinforcing Spatial Reasoning via Agentic Vision-Language Models"
  - u: "bad_seeing_or_bad_thinking_rewarding_perception_for_vision-language_reasoning/"
    t: "Bad Seeing or Bad Thinking? Rewarding Perception for Vision-Language Reasoning"
  - u: "breaking_dual_bottlenecks_evolving_unified_multimodal_models_into_self-adaptive_/"
    t: "Breaking Dual Bottlenecks: Evolving Unified Multimodal Models into Self-Adaptive Interleaved Visual Reasoners"
  - u: "decomposed_on-policy_distillation_for_vision-language_reasoning_steering_gradien/"
    t: "Decomposed On-Policy Distillation for Vision-Language Reasoning: Steering Gradients for Visual Grounding"
  - u: "efficient_reasoning_with_hidden_thinking/"
    t: "Efficient Reasoning with Hidden Thinking"
  - u: "find_fix_reason_context_repair_for_video_reasoning/"
    t: "Find, Fix, Reason: Context Repair for Video Reasoning"
  - u: "from_seeing_to_thinking_decoupling_perception_and_reasoning_improves_post-traini/"
    t: "From Seeing to Thinking: Decoupling Perception and Reasoning Improves Post-Training of Vision-Language Models"
  - u: "ivgr_internalizing_visually_grounded_reasoning_for_mllms_with_reinforcement_lear/"
    t: "iVGR: Internalizing Visually Grounded Reasoning for MLLMs with Reinforcement Learning"
  - u: "learn_to_think_improving_multimodal_reasoning_through_vision-aware_self-improvem/"
    t: "Learn to Think: Improving Multimodal Reasoning through Vision-Aware Self-Improvement Training"
  - u: "learning_gui_grounding_with_spatial_reasoning_from_visual_feedback/"
    t: "Learning GUI Grounding with Spatial Reasoning from Visual Feedback"
  - u: "limssr_llm-driven_sequence-to-score_reasoning_under_training-time_incomplete_mul/"
    t: "LIMSSR: LLM-Driven Sequence-to-Score Reasoning under Training-Time Incomplete Multimodal Observations"
  - u: "look_on_demand_a_cognitive_scheduling_framework_for_visual_evidence_acquisition_/"
    t: "CSMR (Look on Demand): A Cognitive Scheduling Framework for Visual Evidence Acquisition in Multimodal Reasoning"
  - u: "r3l_reasoning_3d_layouts_from_relative_spatial_relations/"
    t: "R$^3$L: Reasoning 3D Layouts from Relative Spatial Relations"
  - u: "revsi_rebuilding_visual_spatial_intelligence_evaluation_for_accurate_assessment_/"
    t: "ReVSI: Rebuilding Visual Spatial Intelligence Evaluation for Accurate Assessment of VLM 3D Reasoning"
  - u: "spectral-progressive_thought_flow_for_lightweight_multimodal_reasoning/"
    t: "Spectral-Progressive Thought Flow for Lightweight Multimodal Reasoning"
  - u: "the_perceptual_bandwidth_bottleneck_in_vision-language_models_active_visual_reas/"
    t: "The Perceptual Bandwidth Bottleneck in Vision-Language Models: Active Visual Reasoning via Sequential Experimental Design"
  - u: "thinking_in_structures_evaluating_spatial_intelligence_in_constraint-governed_sp/"
    t: "Thinking in Structures: Evaluating Spatial Intelligence in Constraint-Governed Spaces"
  - u: "vision-aligned_latent_reasoning_for_multi-modal_large_language_model/"
    t: "Vision-aligned Latent Reasoning for Multi-modal Large Language Model"
  - u: "what_you_think_is_what_you_see_driving_exploration_in_vlm_agents_via_visual-ling/"
    t: "What You Think is What You See: Driving Exploration in VLM Agents via Visual-Linguistic Curiosity (GLANCE)"
item_total: 20
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧠 VLM Reasoning

**🧪 ICML2026** · **20** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (121)](../../CVPR2026/vlm_reasoning/index.md) · [💬 ACL2026 (31)](../../ACL2026/vlm_reasoning/index.md) · [🔬 ICLR2026 (23)](../../ICLR2026/vlm_reasoning/index.md) · [🤖 AAAI2026 (10)](../../AAAI2026/vlm_reasoning/index.md) · [🧠 NeurIPS2025 (30)](../../NeurIPS2025/vlm_reasoning/index.md) · [📹 ICCV2025 (15)](../../ICCV2025/vlm_reasoning/index.md)

🔥 **高频主题：** 推理 ×17 · 多模态 ×13 · LLM ×2

**[3ViewSense: Spatial and Mental Perspective Reasoning from Orthographic Views in Vision-Language Models](3viewsense_spatial_and_mental_perspective_reasoning_from_orthographic_views_in_v.md)**

:   3ViewSense 认为 VLM 空间推理的瓶颈不是视觉特征不够或语言推理太弱，而是缺少稳定的三维中间表示，因此让模型先从单张图像诱导前视图、左视图、俯视图，再基于这些正交视图推理，在遮挡计数和视角一致空间推理上显著优于同规模 VLM。

**[Active Exploring like a Pigeon: Reinforcing Spatial Reasoning via Agentic Vision-Language Models](active_exploring_like_a_pigeon_reinforcing_spatial_reasoning_via_agentic_vision-.md)**

:   本文把 VLM 空间推理从“被动看完所有视角再回答”改造成“按问题主动取景、更新认知地图、用可执行空间断言验证推理”的 agentic 流程，并用密集奖励微调 Qwen2.5-VL-3B，在 MindCube-Tiny 上取得 80.5% overall accuracy，尤其把 Rotation 子集提升到 85.0%。

**[Bad Seeing or Bad Thinking? Rewarding Perception for Vision-Language Reasoning](bad_seeing_or_bad_thinking_rewarding_perception_for_vision-language_reasoning.md)**

:   本文把 VLM 的输出强制拆成 `<recognition>` 感知块和 `<think>` 推理块，再用一个"蒙眼"文本推理代理（拿不到图，只看 VLM 写下的感知文字）能不能答对题作为感知奖励 $R_P$，配上结构化语言验证 SVV 作为结果奖励 $R_O$；MoCA 用 $R_P$ 当门控做模态级信用分配，让 7B 模型在 9 个 perception/reasoning/rich-modality benchmark 上同时提升，在多个指标上超过 GPT-4o。

**[Breaking Dual Bottlenecks: Evolving Unified Multimodal Models into Self-Adaptive Interleaved Visual Reasoners](breaking_dual_bottlenecks_evolving_unified_multimodal_models_into_self-adaptive_.md)**

:   针对统一多模态模型 (unified model) 在 anything-to-image (X2I) 任务上的"理解–生成 gap"（看得懂但生不出），本文提出 Self-Adaptive Interleaved Reasoner：用一个 hierarchical 数据合成 pipeline 在直接生成 / 自我反思 / 多步规划三种模式间分流 5 万条样本，再用 SFT + GRPO 训练并配上 step-wise 推理奖励和 intra-group 复杂度惩罚，让 Emu3.5 在 KRIS-Bench / OmniContext 上超越 GPT-4o、Gemini 2.5 Flash 等闭源模型。

**[Decomposed On-Policy Distillation for Vision-Language Reasoning: Steering Gradients for Visual Grounding](decomposed_on-policy_distillation_for_vision-language_reasoning_steering_gradien.md)**

:   作者把多模态在线蒸馏的 KL 损失沿贝叶斯链拆成"语言先验"和"视觉接地"两个子目标，发现两者梯度近乎正交、标准蒸馏只是被动取平分，提出 Visual Gradient Steering（VGS）主动把更新方向偏向视觉子空间，在 Qwen3-VL 8B→2B/4B 七个多模态推理基准上平均提升 +2.37%/+1.56%。

**[Efficient Reasoning with Hidden Thinking](efficient_reasoning_with_hidden_thinking.md)**

:   Heima 把多模态 LLM 的冗长 CoT 每个阶段（summary / caption / reasoning）蒸馏成**一个特殊 thinking token**，让模型在隐空间里"想"，token 数从 100-200 量级降到 13-16 个的同时 zero-shot 准确率反而比 LLaVA-CoT 更稳；配套训练一个 LLM "interpreter"用 thinking token 的 hidden state 重建出文字推理链，从而验证压缩损失的信息论上界。

**[Find, Fix, Reason: Context Repair for Video Reasoning](find_fix_reason_context_repair_for_video_reasoning.md)**

:   本文针对视频推理中"on-policy RL 在能力天花板停滞、off-policy 蒸馏又会熵塌缩"的两难，引入一个冻结的、工具集成的大教师模型在学生 rollout 失败时插入最小化的"证据补丁" (key-frame 区间、错误类型)，让学生在同一问题上重新作答，并把修复后的轨迹通过 chosen-rollout 机制纳入 GRPO 优化。

**[From Seeing to Thinking: Decoupling Perception and Reasoning Improves Post-Training of Vision-Language Models](from_seeing_to_thinking_decoupling_perception_and_reasoning_improves_post-traini.md)**

:   本文指出当前 VLM 后训练过度强调"长链推理"而忽视感知瓶颈，把后训练显式拆成"视觉感知 → 文本推理 → 视觉推理"三个独立阶段，并用 RLVR（而非 caption SFT）单独打磨感知，使 Qwen3-VL-8B 在视觉数学和感知 benchmark 上分别相对基线提升约 +5.9% 和 +1.2%，同时把推理 trace 缩短 20.8%。

**[iVGR: Internalizing Visually Grounded Reasoning for MLLMs with Reinforcement Learning](ivgr_internalizing_visually_grounded_reasoning_for_mllms_with_reinforcement_lear.md)**

:   针对“显式视觉 grounding 反而拖累 CoT 推理”这一反直觉现象，作者提出 iVGR——一个双流 GRPO 训练框架，让文本 CoT 和带框 grounded CoT 同时 rollout，并用一致性奖励把高质量 grounded 轨迹的视觉定位能力“内化”进纯文本 CoT，从而在推理时不用输出坐标就能拿到 grounded 推理的收益。

**[Learn to Think: Improving Multimodal Reasoning through Vision-Aware Self-Improvement Training](learn_to_think_improving_multimodal_reasoning_through_vision-aware_self-improvem.md)**

:   VISTA 把多模态大模型的自我改进训练改造成"难题靠 prefix 重采样补样本、伪正例靠视觉注意力分数 (VAS) 过滤"的两段式 pipeline，在 Qwen2.5-VL-3B 上把数学/医学多模态推理平均提升 +13.66%。

**[Learning GUI Grounding with Spatial Reasoning from Visual Feedback](learning_gui_grounding_with_spatial_reasoning_from_visual_feedback.md)**

:   把 GUI grounding 从「一次性预测坐标」改写成「在屏幕上挪鼠标找目标」的交互式搜索，用一个带轨迹惩罚的稠密奖励 + GRPO 训练 VLM，让模型从渲染出来的光标得到视觉反馈来对齐数字坐标与屏幕位置，仅用 8K 样本就在 ScreenSpot-Pro 上把 GTA1 的 50.1% 提到 58.1%。

**[LIMSSR: LLM-Driven Sequence-to-Score Reasoning under Training-Time Incomplete Multimodal Observations](limssr_llm-driven_sequence-to-score_reasoning_under_training-time_incomplete_mul.md)**

:   作者把"训练阶段就缺模态"的多模态动作质量评估重新建模成"基于 LLM 的条件序列到分数推理"问题，用 prompt + 特殊 token 让 LLM 在没有完整数据监督的情况下补全缺失语义，再配合掩码感知的双路融合抑制幻觉，在三个 AQA 数据集上全面超越依赖完整训练数据的 SOTA。

**[CSMR (Look on Demand): A Cognitive Scheduling Framework for Visual Evidence Acquisition in Multimodal Reasoning](look_on_demand_a_cognitive_scheduling_framework_for_visual_evidence_acquisition_.md)**

:   CSMR 受 Baddeley 工作记忆理论启发，把"视觉证据何时引入推理"做成动态决策——LLM 维护推理状态，按需调用独立感知模块（VLM）拉视觉证据，直到证据够再终止；解决两大现有范式的缺陷（pre-reasoning 文本化丢细节 / unified VL 空间被语言先验污染），在多个多模态推理基准上零样本超越基线。

**[R$^3$L: Reasoning 3D Layouts from Relative Spatial Relations](r3l_reasoning_3d_layouts_from_relative_spatial_relations.md)**

:   R³L 把 MLLM 多跳"相对空间关系"推理的两类系统性误差（语义漂移与度量漂移）归因于"反复发生的参考系变换"，并通过不变性空间分解（缩短关系链）、一致性空间想象（imagine-and-revise 循环消除冲突）与支持性空间优化（全局-局部位姿重参数化）三个模块，让 GPT-5 生成的开放词汇 3D 场景在 9 类场景下的碰撞率与越界率都接近 0、语义指标显著反超 LayoutVLM/Holodeck/LayoutGPT。

**[ReVSI: Rebuilding Visual Spatial Intelligence Evaluation for Accurate Assessment of VLM 3D Reasoning](revsi_rebuilding_visual_spatial_intelligence_evaluation_for_accurate_assessment_.md)**

:   本文系统揭示了被广泛使用的 VSI-Bench 因 3D 标注漂移与帧采样不一致而存在结构性失效，进而重新标注 381 个场景、5365 个对象，并设计帧预算自适应 QA 与"删除查询对象帧"的 dummy 视频压力测试，构建出名为 ReVSI 的高保真空间智能基准；评估显示开源 VLM 在 ReVSI 上掉点最多 40%，且在 dummy 视频上幻觉率仍高，暴露出现有空间推理能力被 VSI-Bench 系统性高估。

**[Spectral-Progressive Thought Flow for Lightweight Multimodal Reasoning](spectral-progressive_thought_flow_for_lightweight_multimodal_reasoning.md)**

:   SpecFlow 把多模态空间推理从"像素思维"切到"谱思维"——用块离散余弦变换 + 流匹配 + 渐进式频率激活在固定大小的谱工作空间里维护可视化中间思想，加上分类器无关引导（CFG）让文本制导视觉演化，在保持空间推理精度的同时把 KV 缓存削减 1.6–2.1×。

**[The Perceptual Bandwidth Bottleneck in Vision-Language Models: Active Visual Reasoning via Sequential Experimental Design](the_perceptual_bandwidth_bottleneck_in_vision-language_models_active_visual_reas.md)**

:   本文把"VLM 看不清细节"形式化为一个序贯贝叶斯最优实验设计 (S-BOED) 问题,并基于"覆盖率 × 分辨率"的可计算代理目标提出训练免费的 FOVEA 模块,在高分辨率/遥感等基准上稳定超过 Direct 与 ReAct-style 基线。

**[Thinking in Structures: Evaluating Spatial Intelligence in Constraint-Governed Spaces](thinking_in_structures_evaluating_spatial_intelligence_in_constraint-governed_sp.md)**

:   作者构造了 SSI-Bench，一个由 1,000 道排序型 VQA 组成、聚焦"受约束的结构化空间"（屋顶、桥梁、塔架等真实 3D 结构）的基准，要求 VLM 对 3-4 个候选构件按几何或拓扑准则给出完整排列；评测 31 个 VLM 后发现最强闭源模型 Gemini-3-Flash 仅 33.6%、最佳开源 GLM-4.6V 22.2%，而人类 91.6%，揭示当前 VLM 在受几何/连接/物理可行性共同约束的真实 3D 场景下缺乏一致的空间推理能力。

**[Vision-aligned Latent Reasoning for Multi-modal Large Language Model](vision-aligned_latent_reasoning_for_multi-modal_large_language_model.md)**

:   本文提出 VaLR：在 MLLM 的 CoT 推理每一步之前插入若干"潜在 token"，并用 DINOv3/SigLIP/π³ 等视觉编码器的 patch 特征对这些 token 做表征对齐（REPA），从而在长链推理中持续把视觉信息"喂回"模型，把 Qwen2.5-VL 在 VSI-Bench 上的准确率从 33.0% 拉到 52.9%，并首次让 MLLM 表现出"推理越长越准"的 test-time scaling 行为。

**[What You Think is What You See: Driving Exploration in VLM Agents via Visual-Linguistic Curiosity (GLANCE)](what_you_think_is_what_you_see_driving_exploration_in_vlm_agents_via_visual-ling.md)**

:   GLANCE 在 VLM agent 的强化学习里加了一个"想-看对齐"自监督头：让 LLM 在 CoT 里产出的"下一状态预测"通过一个轻量 projector 映射到由 EMA target 视觉编码器编码的真实下一帧表示，预测与实际之间的差距同时充当内在好奇心奖励、视觉编码器的训练信号、以及让 internalized world model "落地"的对齐损失；再配合周期性重置 projector 的课程探索机制对抗好奇心衰退，最终在 5 个 agentic 任务上稳定超过现有 exploitation-only 的 VLM-RL 方法。
