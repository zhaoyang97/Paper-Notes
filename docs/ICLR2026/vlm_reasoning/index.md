---
title: >-
  ICLR2026 VLMReasoning论文汇总 · 23篇论文解读
description: >-
  23篇ICLR2026的 VLM Reasoning 方向论文解读，涵盖推理、多模态、Agent、机器人等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICLR2026"
  - "VLM Reasoning"
  - "论文解读"
  - "论文笔记"
  - "推理"
  - "多模态"
  - "Agent"
  - "机器人"
item_list:
  - u: "diva-grpo_enhancing_multimodal_reasoning_through_difficulty-adaptive_variant_adv/"
    t: "DIVA-GRPO: Enhancing Multimodal Reasoning through Difficulty-Adaptive Variant Advantage"
  - u: "empowering_small_vlms_to_think_with_dynamic_memorization_and_exploration/"
    t: "Empowering Small VLMs to Think with Dynamic Memorization and Exploration"
  - u: "evaluating_vlms_spatial_reasoning_over_robot_motion_a_step_towards_robot_plannin/"
    t: "Evaluating VLMs' Spatial Reasoning Over Robot Motion: A Step Towards Robot Planning with Motion Preferences"
  - u: "frieda_benchmarking_multi-step_cartographic_reasoning_in_vision-language_models/"
    t: "FRIEDA: Benchmarking Multi-Step Cartographic Reasoning in Vision-Language Models"
  - u: "gtr-bench_evaluating_geo-temporal_reasoning_in_vision-language_mod/"
    t: "GTR-Bench: Evaluating Geo-Temporal Reasoning in Vision-Language Models"
  - u: "lets_think_in_two_steps_mitigating_agreement_bias_in_mllms_with_self-grounded_ve/"
    t: "Let's Think in Two Steps: Mitigating Agreement Bias in MLLMs with Self-Grounded Verification"
  - u: "mmr-life_piecing_together_real-life_scenes_for_multimodal_multi-image_reasoning/"
    t: "MMR-Life: Piecing Together Real-life Scenes for Multimodal Multi-image Reasoning"
  - u: "omnispatial_towards_comprehensive_spatial_reasoning_benchmark_for_vision_languag/"
    t: "OmniSpatial: Towards Comprehensive Spatial Reasoning Benchmark for Vision Language Models"
  - u: "reasoning-driven_multimodal_llm_for_domain_generalization/"
    t: "Reasoning-Driven Multimodal LLM for Domain Generalization"
  - u: "ref-adv_exploring_mllm_visual_reasoning_in_referring_expression_tasks/"
    t: "Ref-Adv: Exploring MLLM Visual Reasoning in Referring Expression Tasks"
  - u: "seeing_across_views_benchmarking_spatial_reasoning_of_vision-language_models_in_/"
    t: "Seeing Across Views: Benchmarking Spatial Reasoning of Vision-Language Models in Robotic Scenes"
  - u: "small_drafts_big_verdict_information-intensive_visual_reasoning_via_speculation/"
    t: "Small Drafts, Big Verdict: Information-Intensive Visual Reasoning via Speculation"
  - u: "sophiavl-r1_reinforcing_mllms_reasoning_with_thinking_reward/"
    t: "SophiaVL-R1: Reinforcing MLLMs Reasoning with Thinking Reward"
  - u: "spatial-dise_a_unified_benchmark_for_evaluating_spatial_reasoning_in_vision-lang/"
    t: "Spatial-DISE: A Unified Benchmark for Evaluating Spatial Reasoning in Vision-Language Models"
  - u: "spatial_captcha_generatively_benchmarking_spatial_reasoning_for_human-machine_di/"
    t: "Spatial CAPTCHA: Generatively Benchmarking Spatial Reasoning for Human-Machine Differentiation"
  - u: "spatial_reasoning_is_not_a_free_lunch_a_controlled_study_on_llava/"
    t: "Spatial Reasoning is Not a Free Lunch: A Controlled Study on LLaVA"
  - u: "spatialab_can_vision-language_models_perform_spatial_reasoning_in_the_wild/"
    t: "SpatiaLab: Can Vision-Language Models Perform Spatial Reasoning in the Wild?"
  - u: "spinbench_perspective_and_rotation_as_a_lens_on_spatial_reasoning_in_vlms/"
    t: "SpinBench: Perspective and Rotation as a Lens on Spatial Reasoning in VLMs"
  - u: "thinkomni_lifting_textual_reasoning_to_omni-modal_scenarios_via_guidance_decodin/"
    t: "ThinkOmni: Lifting Textual Reasoning to Omni-modal Scenarios via Guidance Decoding"
  - u: "through_the_lens_of_contrast_self-improving_visual_reasoning_in_vlms/"
    t: "Through the Lens of Contrast: Self-Improving Visual Reasoning in VLMs"
  - u: "vidguard-r1_ai-generated_video_detection_and_explanation_via_reasoning_mllms_and/"
    t: "VidGuard-R1: AI-Generated Video Detection and Explanation via Reasoning MLLMs and RL"
  - u: "vlm-subtlebench_how_far_are_vlms_from_human-level_subtle_comparative_reasoning/"
    t: "VLM-SubtleBench: How Far Are VLMs from Human-Level Subtle Comparative Reasoning?"
  - u: "vtool-r1_vlms_learn_to_think_with_images_via_reinforcement_learning_on_multimoda/"
    t: "VTool-R1: VLMs Learn to Think with Images via Reinforcement Learning on Multimodal Tool Use"
item_total: 23
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧠 VLM Reasoning

**🔬 ICLR2026** · **23** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (121)](../../CVPR2026/vlm_reasoning/index.md) · [🧪 ICML2026 (20)](../../ICML2026/vlm_reasoning/index.md) · [💬 ACL2026 (31)](../../ACL2026/vlm_reasoning/index.md) · [🤖 AAAI2026 (10)](../../AAAI2026/vlm_reasoning/index.md) · [🧠 NeurIPS2025 (30)](../../NeurIPS2025/vlm_reasoning/index.md) · [📹 ICCV2025 (15)](../../ICCV2025/vlm_reasoning/index.md)

🔥 **高频主题：** 推理 ×20 · 多模态 ×11 · Agent ×2 · 机器人 ×2

**[DIVA-GRPO: Enhancing Multimodal Reasoning through Difficulty-Adaptive Variant Advantage](diva-grpo_enhancing_multimodal_reasoning_through_difficulty-adaptive_variant_adv.md)**

:   提出 DIVA-GRPO，通过动态评估问题难度、自适应生成不同难度的语义一致变体、并结合难度加权的局部-全局 advantage 估计，解决 GRPO 训练中的 reward sparsity 和 advantage vanishing 问题，在 7B 规模模型上实现 SOTA 多模态推理性能。

**[Empowering Small VLMs to Think with Dynamic Memorization and Exploration](empowering_small_vlms_to_think_with_dynamic_memorization_and_exploration.md)**

:   提出 DyME（Dynamic Memorize-Explore），通过逐步动态切换 SFT 记忆模式与 GRPO 探索模式，首次赋予小规模视觉语言模型（<1B 参数）在特定任务上的思维推理能力。

**[Evaluating VLMs' Spatial Reasoning Over Robot Motion: A Step Towards Robot Planning with Motion Preferences](evaluating_vlms_spatial_reasoning_over_robot_motion_a_step_towards_robot_plannin.md)**

:   系统评估了 VLM 对机器人运动路径的空间推理能力，提出 4 种图像查询方法用于让 VLM 根据用户自然语言描述选择最佳运动路径，发现 Qwen2.5-VL 零样本准确率达 71.4%，且微调后小模型可获显著提升。

**[FRIEDA: Benchmarking Multi-Step Cartographic Reasoning in Vision-Language Models](frieda_benchmarking_multi-step_cartographic_reasoning_in_vision-language_models.md)**

:   提出 FRIEDA 基准，系统评估大型视觉语言模型在多步骤、跨地图的制图推理能力，发现最强模型 Gemini-2.5-Pro 准确率仅 38.20%，远低于人类 84.87%。

**[GTR-Bench: Evaluating Geo-Temporal Reasoning in Vision-Language Models](gtr-bench_evaluating_geo-temporal_reasoning_in_vision-language_mod.md)**

:   提出 GTR-Bench，一个面向大规模摄像头网络中移动目标地理时空推理的新基准，评估发现最强模型 Gemini-2.5-Pro（34.9%）远落后于人类水平（78.61%），揭示了当前 VLM 在时空上下文利用失衡、时序预测能力弱、地图-视频对齐能力不足三大缺陷。

**[Let's Think in Two Steps: Mitigating Agreement Bias in MLLMs with Self-Grounded Verification](lets_think_in_two_steps_mitigating_agreement_bias_in_mllms_with_self-grounded_ve.md)**

:   本文发现多模态大语言模型（MLLM）作为 agent 行为验证器时存在严重的"同意偏差"（agreement bias）——系统性地过度认可 agent 行为，并提出 Self-Grounded Verification（SGV）方法，通过两步生成（先提取行为先验、再条件化验证）缓解该偏差，在 web 导航、桌面操作和机器人操控任务中将失败检测率提升最高 25pp、准确率提升 14pp。

**[MMR-Life: Piecing Together Real-life Scenes for Multimodal Multi-image Reasoning](mmr-life_piecing_together_real-life_scenes_for_multimodal_multi-image_reasoning.md)**

:   提出 MMR-Life 基准（2646 道 5 选 1 多图选择题，基于 19108 张真实图像，覆盖 7 种推理类型和 21 个任务），首次系统评估 MLLM 在真实生活场景中的多图推理能力，发现最强模型 GPT-5 仅 58.69% 准确率，距人类水平差 14%，并揭示了推理增强方法在大模型上失效、RL 泛化弱于 BoN 等关键发现。

**[OmniSpatial: Towards Comprehensive Spatial Reasoning Benchmark for Vision Language Models](omnispatial_towards_comprehensive_spatial_reasoning_benchmark_for_vision_languag.md)**

:   基于认知心理学构建OmniSpatial——首个全面空间推理基准，系统覆盖动态推理、复杂空间逻辑、空间交互和透视转换4大维度50个子类别共8.4K人工标注QA对，让o3最强推理模型仅达56.33%而人类达92.63%→揭示复杂空间推理仍是VLM的核心瓶颈。

**[Reasoning-Driven Multimodal LLM for Domain Generalization](reasoning-driven_multimodal_llm_for_domain_generalization.md)**

:   提出 RD-MLDG——首个将 MLLM 推理链引入域泛化的框架。构建 DomainBed-Reasoning 数据集，系统分析推理监督的两大挑战（优化困难 + 推理模式不匹配），通过 MTCT（多任务交叉训练）与 SARR（自对齐推理正则化）协同解决，在 4 个标准 DG 基准上以 86.89% 的平均准确率大幅超越 GPT-4o（83.46%）和所有 CLIP/ViT 方法。

**[Ref-Adv: Exploring MLLM Visual Reasoning in Referring Expression Tasks](ref-adv_exploring_mllm_visual_reasoning_in_referring_expression_tasks.md)**

:   提出 Ref-Adv 基准数据集，通过 **硬干扰物配对 + LLM 辅助最小充分表达式生成 + 三人一致性人工验证** 的流水线，构建了一个消除"定位捷径"的现代 REC 基准，在该基准上 13 个当代 MLLM（包括 GPT-4o、Gemini 2.5、Qwen2.5-VL-72B 等）的准确率从 RefCOCO(+/g) 上的 90%+ 大幅下降至 50-68%，系统暴露了模型在复杂视觉推理和真实定位能力上的严重不足。

**[Seeing Across Views: Benchmarking Spatial Reasoning of Vision-Language Models in Robotic Scenes](seeing_across_views_benchmarking_spatial_reasoning_of_vision-language_models_in_.md)**

:   提出 MV-RoboBench，首个整合多视角空间推理与机器人操作执行评测的 benchmark，包含 1.7K 人工标注 QA，揭示当前最强 VLM（GPT-5 仅 56.4%）与人类（91.0%）之间存在巨大差距。

**[Small Drafts, Big Verdict: Information-Intensive Visual Reasoning via Speculation](small_drafts_big_verdict_information-intensive_visual_reasoning_via_speculation.md)**

:   借鉴 Speculative Decoding 的 draft-then-verify 范式提出 Speculative Verdict (SV)，用多个轻量 VLM 生成多样推理路径作为 draft，大模型作为 verdict 综合验证并纠错，在信息密集型 VQA 上无需训练即超过 GPT-4o 达 11.9%，且能修复 47-53% 的少数正确案例。

**[SophiaVL-R1: Reinforcing MLLMs Reasoning with Thinking Reward](sophiavl-r1_reinforcing_mllms_reasoning_with_thinking_reward.md)**

:   提出SophiaVL-R1——在规则基RL训练MLLM推理时引入整体级思维过程奖励：训练Thinking Reward Model从逻辑一致性/冗余度等五维度评估推理质量→提出Trust-GRPO基于正确/错误答案组的思维奖励对比计算可信度权重$\gamma$缓解reward hacking→退火策略$e^{-\text{steps}/T}$渐减思维奖励使后期更依赖准确的规则奖励→7B模型在MathVista(71.3%)和MMMU(61.3%)等多个基准全面超越LLaVA-OneVision-72B。

**[Spatial-DISE: A Unified Benchmark for Evaluating Spatial Reasoning in Vision-Language Models](spatial-dise_a_unified_benchmark_for_evaluating_spatial_reasoning_in_vision-lang.md)**

:   提出基于认知科学 2×2 分类法（内在/外在 × 静态/动态）的统一空间推理基准 Spatial-DISE，包含 559 个评估 VQA 对和 12K+ 训练数据，在 32 个 SOTA VLM 上的评测揭示了模型在动态空间推理（尤其是心理旋转和折叠）上与人类的巨大差距。

**[Spatial CAPTCHA: Generatively Benchmarking Spatial Reasoning for Human-Machine Differentiation](spatial_captcha_generatively_benchmarking_spatial_reasoning_for_human-machine_di.md)**

:   提出 Spatial CAPTCHA，一种基于 3D 空间推理的新型人类验证框架，利用人类与多模态大语言模型在几何推理、视角变换、遮挡处理和心理旋转等任务上的根本性能力差异来区分人与机器，最优 MLLM 仅达 31.0% Pass@1 准确率，远低于人类表现。

**[Spatial Reasoning is Not a Free Lunch: A Controlled Study on LLaVA](spatial_reasoning_is_not_a_free_lunch_a_controlled_study_on_llava.md)**

:   通过在 LLaVA 框架下控制实验，系统性地研究图像编码器训练目标和 2D 位置编码对 VLM 空间推理能力的影响，发现编码器选择主导空间性能、AIMv2 编码器一致性最好，但 2D-RoPE 的改进不稳定，空间推理的失败根植于当前 VLM 流水线的核心设计选择。

**[SpatiaLab: Can Vision-Language Models Perform Spatial Reasoning in the Wild?](spatialab_can_vision-language_models_perform_spatial_reasoning_in_the_wild.md)**

:   提出SpatiaLab，一个包含1400个视觉QA对的真实场景空间推理基准，涵盖6大类30子类空间任务，支持多选和开放式双格式评估，揭示当前最强VLM（InternVL3.5-72B MCQ 54.93%）与人类（87.57%）之间存在巨大空间推理鸿沟，且开放式设置下差距更大。

**[SpinBench: Perspective and Rotation as a Lens on Spatial Reasoning in VLMs](spinbench_perspective_and_rotation_as_a_lens_on_spatial_reasoning_in_vlms.md)**

:   提出 SpinBench，一个以认知科学为基础的诊断性基准测试，通过 7 类渐进式空间推理任务（从物体识别到视角转换）系统评估 37 个 VLMs 的空间理解能力，揭示了模型存在的自我中心偏差、旋转理解薄弱等系统性缺陷。

**[ThinkOmni: Lifting Textual Reasoning to Omni-modal Scenarios via Guidance Decoding](thinkomni_lifting_textual_reasoning_to_omni-modal_scenarios_via_guidance_decodin.md)**

:   提出 ThinkOmni 无训练框架，利用纯文本大推理模型(LRM)在解码时引导全模态 LLM(OLLM)，通过 Stepwise Contrastive Scaling 自适应平衡感知与推理信号，MathVista 达 70.2%、MMAU 达 75.5%，匹配或超越 RFT 方法。

**[Through the Lens of Contrast: Self-Improving Visual Reasoning in VLMs](through_the_lens_of_contrast_self-improving_visual_reasoning_in_vlms.md)**

:   提出 VC-STaR（Visual Contrastive Self-Taught Reasoner），基于"VLM 在对比两张相似图像时看得更准"的观察，设计了一套对比式自改进框架：通过构造对比 VQA 对让模型在对比中生成更忠实的视觉分析，再由 LLM 将对比分析融入推理路径，产出高质量视觉推理数据集 VisCoR-55K，微调后在 MMVP 上提升 5.7%、Hallusion 上提升 3.2%。

**[VidGuard-R1: AI-Generated Video Detection and Explanation via Reasoning MLLMs and RL](vidguard-r1_ai-generated_video_detection_and_explanation_via_reasoning_mllms_and.md)**

:   VidGuard-R1 是首个采用 GRPO（Group Relative Policy Optimization）强化学习微调 MLLM 的视频真伪检测器，通过构建 14 万无快捷方式的真/假视频对数据集，并设计时序伪影奖励和扩散步数质量奖励两种专用奖励机制，在自建数据集上达到 86.17% 准确率，在 GenVidBench 和 GenVideo 基准上实现 95%+ 的 SOTA 零样本检测性能，同时生成可解释的思维链推理。

**[VLM-SubtleBench: How Far Are VLMs from Human-Level Subtle Comparative Reasoning?](vlm-subtlebench_how_far_are_vlms_from_human-level_subtle_comparative_reasoning.md)**

:   提出 VLM-SubtleBench，一个评估视觉语言模型在细微差异比较推理能力的基准，覆盖 10 种差异类型和 6 个图像领域（自然、游戏、工业、航空、医学、合成），揭示了 VLM 与人类在空间/时间/视角推理上超过 30% 的性能差距。

**[VTool-R1: VLMs Learn to Think with Images via Reinforcement Learning on Multimodal Tool Use](vtool-r1_vlms_learn_to_think_with_images_via_reinforcement_learning_on_multimoda.md)**

:   提出 VTool-R1，首个通过强化学习微调训练 VLM 生成交错文本和视觉中间推理步骤的框架，使模型学会"用图像思考"。
