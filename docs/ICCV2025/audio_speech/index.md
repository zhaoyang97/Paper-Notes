---
title: >-
  ICCV2025 音频/语音论文汇总 · 11篇论文解读
description: >-
  11篇ICCV2025的音频/语音方向论文解读，涵盖语音、多模态、扩散模型、少样本学习等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICCV2025"
  - "音频/语音"
  - "论文解读"
  - "论文笔记"
  - "语音"
  - "多模态"
  - "扩散模型"
  - "少样本学习"
item_list:
  - u: "25_years_in_class_a_multimodal_textbook_for_visionlanguage_p/"
    t: "2.5 Years in Class: A Multimodal Textbook for Vision-Language Pretraining"
  - u: "align_your_rhythm_generating_highly_aligned_dance_poses_with_gating-enhanced_rhy/"
    t: "Align Your Rhythm: Generating Highly Aligned Dance Poses with Gating-Enhanced Rhythm-Aware Feature Representation"
  - u: "everything_is_a_video_unifying_modalities_through_next-frame_prediction/"
    t: "Everything is a Video: Unifying Modalities through Next-Frame Prediction"
  - u: "how_would_it_sound_material-controlled_multimodal_acoustic_profile_generation_fo/"
    t: "How Would It Sound? Material-Controlled Multimodal Acoustic Profile Generation for Objects"
  - u: "latent_swap_joint_diffusion_for_2d_long-form_latent_generation/"
    t: "Latent Swap Joint Diffusion for 2D Long-Form Latent Generation"
  - u: "learning_to_see_inside_opaque_liquid_containers_using_speckle_vibrometry/"
    t: "Learning to See Inside Opaque Liquid Containers using Speckle Vibrometry"
  - u: "lyra_an_efficient_and_speechcentric_framework_for_omnicognit/"
    t: "Lyra: An Efficient and Speech-Centric Framework for Omni-Cognition"
  - u: "mug_pseudo_labeling_augmented_audio-visual_mamba_network_for_audio-visual_video_/"
    t: "MUG: Pseudo Labeling Augmented Audio-Visual Mamba Network for Audio-Visual Video Parsing"
  - u: "understanding_co-speech_gestures_in-the-wild/"
    t: "Understanding Co-speech Gestures in-the-wild"
  - u: "vggsounder_audio-visual_evaluations_for_foundation_models/"
    t: "VGGSounder: Audio-Visual Evaluations for Foundation Models"
  - u: "zero-avsr_zero-shot_audio-visual_speech_recognition_with_llms_by_learning_langua/"
    t: "Zero-AVSR: Zero-Shot Audio-Visual Speech Recognition with LLMs by Learning Language-Agnostic Speech Representations"
item_total: 11
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎵 音频/语音

**📹 ICCV2025** · **11** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (22)](../../CVPR2026/audio_speech/index.md) · [🔬 ICLR2026 (79)](../../ICLR2026/audio_speech/index.md) · [💬 ACL2026 (70)](../../ACL2026/audio_speech/index.md) · [🧪 ICML2026 (36)](../../ICML2026/audio_speech/index.md) · [🤖 AAAI2026 (31)](../../AAAI2026/audio_speech/index.md) · [🧠 NeurIPS2025 (47)](../../NeurIPS2025/audio_speech/index.md)

🔥 **高频主题：** 语音 ×5 · 多模态 ×2

**[2.5 Years in Class: A Multimodal Textbook for Vision-Language Pretraining](25_years_in_class_a_multimodal_textbook_for_visionlanguage_p.md)**

:   从YouTube收集2.5年(22,000课时)的教学视频，通过LLM驱动的多级抽取与过滤管线构建高质量交错图文"多模态教科书"语料(6.5M关键帧 + 0.75B文本token)，显著提升VLM在知识密集型和推理任务上的预训练效果，尤其在ScienceQA和MathVista上带来大幅提升。

**[Align Your Rhythm: Generating Highly Aligned Dance Poses with Gating-Enhanced Rhythm-Aware Feature Representation](align_your_rhythm_generating_highly_aligned_dance_poses_with_gating-enhanced_rhy.md)**

:   提出Danceba框架，通过基于相位的节奏提取（PRE）、时序门控因果注意力（TGCA）和并行Mamba运动建模（PMMM）三个核心模块，实现音乐驱动的高节奏对齐、高多样性舞蹈生成，在AIST++数据集上FIDk提升48.68%、BAS提升12%。

**[Everything is a Video: Unifying Modalities through Next-Frame Prediction](everything_is_a_video_unifying_modalities_through_next-frame_prediction.md)**

:   本文将多模态学习中的文本、图像、音频、视频等不同模态任务统一重构为下一帧预测问题（所有输入输出都渲染为 64×64 视频帧序列），用单一 Transformer 模型无需模态特定编码器即可处理跨模态任务，验证了"everything is a video"这一激进但可行的统一表征范式。

**[How Would It Sound? Material-Controlled Multimodal Acoustic Profile Generation for Objects](how_would_it_sound_material-controlled_multimodal_acoustic_profile_generation_fo.md)**

:   提出材质可控的声学特征生成任务（M-CAPA），给定室内场景的音视觉观测和用户定义的新材质配置，生成反映材质变化的目标房间脉冲响应（RIR），并构建了配套的 Acoustic Wonderland 数据集。

**[Latent Swap Joint Diffusion for 2D Long-Form Latent Generation](latent_swap_joint_diffusion_for_2d_long-form_latent_generation.md)**

:   提出SaFa（Swap Forward），一种模态无关的高效方法，通过两种潜空间交换算子（Self-Loop Latent Swap和Reference-Guided Latent Swap）替代传统联合扩散中的均值化操作，解决频谱混叠问题并保持跨视图一致性，在长音频和全景图生成中显著优于现有方法。

**[Learning to See Inside Opaque Liquid Containers using Speckle Vibrometry](learning_to_see_inside_opaque_liquid_containers_using_speckle_vibrometry.md)**

:   本文提出了一种基于激光散斑振动测量的非接触式系统，通过 2D 网格同时感知多个不透明容器表面的微小振动，再用 Vibration Transformer 从振动频谱中推断容器类型和隐藏液位，开创了"透视不透明容器内部液位"这一全新计算机视觉任务。

**[Lyra: An Efficient and Speech-Centric Framework for Omni-Cognition](lyra_an_efficient_and_speechcentric_framework_for_omnicognit.md)**

:   提出Lyra，一个以语音为中心的全模态MLLM框架，通过三大核心组件（DTW-based跨模态正则化器、多模态LoRA、Latent多模态提取器）和首个12K长语音SFT数据集，在仅用2.7M数据和少量训练的情况下，同时在视觉-语言、视觉-语音、语音-语言benchmark上达到SOTA，并能处理长达2小时的语音输入。

**[MUG: Pseudo Labeling Augmented Audio-Visual Mamba Network for Audio-Visual Video Parsing](mug_pseudo_labeling_augmented_audio-visual_mamba_network_for_audio-visual_video_.md)**

:   提出MUG框架，通过伪标签增强的跨模态随机组合数据增强策略和音视频Mamba网络，同时提升弱监督音视频解析任务中段级和事件级的预测性能。

**[Understanding Co-speech Gestures in-the-wild](understanding_co-speech_gestures_in-the-wild.md)**

:   本文提出 JEGAL——一个联合手势-语音-文本的三模态嵌入空间，通过全局短语对比损失和局部手势-词耦合损失在弱监督条件下学习共语手势表征，定义了三个新的手势理解任务和基准，超越了包括大型视觉语言模型在内的多种方法。

**[VGGSounder: Audio-Visual Evaluations for Foundation Models](vggsounder_audio-visual_evaluations_for_foundation_models.md)**

:   针对 VGGSound 数据集在多标签缺失、类别重叠和模态错位方面的局限性，构建了 VGGSounder——一个带有模态标注的多标签音视频分类基准，并提出"模态混淆"度量来揭示基础模型在多模态融合上的不足。

**[Zero-AVSR: Zero-Shot Audio-Visual Speech Recognition with LLMs by Learning Language-Agnostic Speech Representations](zero-avsr_zero-shot_audio-visual_speech_recognition_with_llms_by_learning_langua.md)**

:   提出 Zero-AVSR 框架，通过将语音转写为语言无关的罗马化文本（Roman text），再利用 LLM 将罗马文本转换为目标语言文字，实现无需目标语言语音数据的零样本视听语音识别，并构建了覆盖 82 种语言、2916 小时的 MARC 数据集。
