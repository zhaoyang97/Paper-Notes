---
title: >-
  ACL2025 图像生成论文汇总 · 9篇论文解读
description: >-
  9篇ACL2025的图像生成方向论文解读，涵盖语音、文生图、少样本学习等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ACL2025"
  - "图像生成"
  - "论文解读"
  - "论文笔记"
  - "语音"
  - "文生图"
  - "少样本学习"
item_list:
  - u: "a_unified_agentic_framework_for_evaluating_conditional_image_generation/"
    t: "A Unified Agentic Framework for Evaluating Conditional Image Generation"
  - u: "d-gen_automatic_distractor_generation_and_evaluation_for_reliable_assessment_of_/"
    t: "D-GEN: Automatic Distractor Generation and Evaluation for Reliable Assessment of Generative Models"
  - u: "difftod_diffusion_dialogue_planning/"
    t: "Planning with Diffusion Models for Target-Oriented Dialogue Systems"
  - u: "flashaudio_rectified_flow_tta/"
    t: "FlashAudio: Rectified Flows for Fast and High-Fidelity Text-to-Audio Generation"
  - u: "generating_pedagogically_meaningful_visuals_for_math_word_problems_a_new_benchma/"
    t: "Generating Pedagogically Meaningful Visuals for Math Word Problems: A New Benchmark and Analysis of Text-to-Image Models"
  - u: "multimodal_pragmatic_jailbreak_on_text-to-image_models/"
    t: "Multimodal Pragmatic Jailbreak on Text-to-image Models"
  - u: "ozspeech_one-step_zero-shot_speech_synthesis_with_learned-prior-conditioned_flow/"
    t: "OZSpeech: One-step Zero-shot Speech Synthesis with Learned-Prior-Conditioned Flow Matching"
  - u: "rvc_rhythm_voice_conversion/"
    t: "R-VC: Rhythm Controllable and Efficient Zero-Shot Voice Conversion via Shortcut Flow Matching"
  - u: "synthia_novel_concept_design_with_affordance_composition/"
    t: "Synthia: Novel Concept Design with Affordance Composition"
item_total: 9
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎨 图像生成

**💬 ACL2025** · **9** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (490)](../../CVPR2026/image_generation/index.md) · [🔬 ICLR2026 (352)](../../ICLR2026/image_generation/index.md) · [💬 ACL2026 (5)](../../ACL2026/image_generation/index.md) · [🧪 ICML2026 (141)](../../ICML2026/image_generation/index.md) · [🤖 AAAI2026 (79)](../../AAAI2026/image_generation/index.md) · [🧠 NeurIPS2025 (221)](../../NeurIPS2025/image_generation/index.md)

🔥 **高频主题：** 语音 ×3 · 文生图 ×2 · 少样本学习 ×2

**[A Unified Agentic Framework for Evaluating Conditional Image Generation](a_unified_agentic_framework_for_evaluating_conditional_image_generation.md)**

:   提出 CIGEval，一个基于大型多模态模型（LMM）的统一 Agent 评估框架，通过工具集成（Grounding、Highlight、Difference、Scene Graph）和分而治之的评估策略，在 7 种条件图像生成任务上达到与人类标注者相当的相关性（0.4625 vs 人类间 0.47），且仅用 2.3K 训练数据微调 7B 模型即超越 GPT-4o 版 SOTA。

**[D-GEN: Automatic Distractor Generation and Evaluation for Reliable Assessment of Generative Models](d-gen_automatic_distractor_generation_and_evaluation_for_reliable_assessment_of_.md)**

:   提出 D-GEN——首个开源干扰项生成模型（LLaMA微调，8B/70B），自动将开放式评测题转为多选题格式，配套排名对齐+熵分析两种评估方法验证干扰项质量，在 MMLU 上 Spearman's ρ=0.99 保持模型排名一致性。

**[Planning with Diffusion Models for Target-Oriented Dialogue Systems](difftod_diffusion_dialogue_planning.md)**

:   DiffTOD 将对话规划建模为轨迹生成问题，利用掩码扩散语言模型实现非顺序对话规划，并设计三种引导机制（词级/语义级/搜索级）灵活控制对话朝目标推进，在谈判/推荐/闲聊三种场景上显著超越基线。

**[FlashAudio: Rectified Flows for Fast and High-Fidelity Text-to-Audio Generation](flashaudio_rectified_flow_tta.md)**

:   将整流流（Rectified Flow）引入文本转音频生成，通过双焦采样器优化时间步分布、不混溶流减少数据-噪声总距离、锚定优化修正 CFG 引导误差，实现单步生成 FAD=1.49 超越百步扩散模型，生成速度达实时 400 倍。

**[Generating Pedagogically Meaningful Visuals for Math Word Problems: A New Benchmark and Analysis of Text-to-Image Models](generating_pedagogically_meaningful_visuals_for_math_word_problems_a_new_benchma.md)**

:   Math2Visual 提出了一个从数学应用题（MWP）文本描述自动生成教学可视化图像的框架，定义了基于教师访谈的视觉语言和设计空间，构建了 1,903 张标注数据集，并评估和微调了多个 TTI 模型，揭示了当前模型在数学关系表示上的关键不足。

**[Multimodal Pragmatic Jailbreak on Text-to-image Models](multimodal_pragmatic_jailbreak_on_text-to-image_models.md)**

:   提出"多模态语用越狱"（Multimodal Pragmatic Jailbreak）新型攻击方式，通过让T2I模型生成包含视觉文字的图像，使得图像内容和文字内容单独看都安全但组合后产生不安全内容，揭示了所有测试模型（包括DALL·E 3）均受此攻击影响。

**[OZSpeech: One-step Zero-shot Speech Synthesis with Learned-Prior-Conditioned Flow Matching](ozspeech_one-step_zero-shot_speech_synthesis_with_learned-prior-conditioned_flow.md)**

:   提出OZSpeech，首个将最优传输条件流匹配(OT-CFM)与学习先验分布相结合实现单步采样的零样本TTS系统，在内容准确性(WER)、推理速度和模型大小上均大幅领先现有方法。

**[R-VC: Rhythm Controllable and Efficient Zero-Shot Voice Conversion via Shortcut Flow Matching](rvc_rhythm_voice_conversion.md)**

:   R-VC 是首个实现节奏可控的零样本语音转换系统，通过 Mask Transformer 时长模型建模目标说话人的节奏风格，结合 Shortcut Flow Matching 的 DiT 解码器实现仅 2 步采样的高效高质量语音生成，在 LibriSpeech 上 WER 3.51、说话人相似度 0.930。

**[Synthia: Novel Concept Design with Affordance Composition](synthia_novel_concept_design_with_affordance_composition.md)**

:   Synthia 提出了一种基于 affordance（功能可供性）组合的新颖概念设计框架，通过层次化概念本体、affordance 采样策略和课程学习微调 T2I 模型，生成既视觉新颖又功能连贯的创新设计。
