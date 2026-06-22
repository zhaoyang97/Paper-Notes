---
title: >-
  ECCV2024 音频/语音论文汇总 · 8篇论文解读
description: >-
  8篇ECCV2024的音频/语音方向论文解读，涵盖语音、人脸/视线等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ECCV2024"
  - "音频/语音"
  - "论文解读"
  - "论文笔记"
  - "语音"
  - "人脸/视线"
item_list:
  - u: "action2sound_ambientaware_generation_of_action_sounds_from_e/"
    t: "Action2Sound: Ambient-Aware Generation of Action Sounds from Egocentric Videos"
  - u: "beat-it_beat-synchronized_multi-condition_3d_dance_generation/"
    t: "Beat-It: Beat-Synchronized Multi-Condition 3D Dance Generation"
  - u: "coleaf_a_contrastive-collaborative_learning_framework_for_weakly_supervised_audi/"
    t: "CoLeaF: A Contrastive-Collaborative Learning Framework for Weakly Supervised Audio-Visual Video Parsing"
  - u: "controlllm_augment_language_models_with_tools_by_searching_on_graphs/"
    t: "ControlLLM: Augment Language Models with Tools by Searching on Graphs"
  - u: "label-anticipated_event_disentanglement_for_audio-visual_video_parsing/"
    t: "Label-Anticipated Event Disentanglement for Audio-Visual Video Parsing"
  - u: "latent-inr_a_flexible_framework_for_implicit_representations_of_videos_with_disc/"
    t: "Latent-INR: A Flexible Framework for Implicit Representations of Videos with Discriminative Semantics"
  - u: "listen_to_look_into_the_future_audio-visual_egocentric_gaze_anticipation/"
    t: "Listen to Look into the Future: Audio-Visual Egocentric Gaze Anticipation"
  - u: "siamese_vision_transformers_are_scalable_audio-visual_learners/"
    t: "Siamese Vision Transformers are Scalable Audio-Visual Learners"
item_total: 8
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎵 音频/语音

**🎞️ ECCV2024** · **8** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (22)](../../CVPR2026/audio_speech/index.md) · [🔬 ICLR2026 (79)](../../ICLR2026/audio_speech/index.md) · [💬 ACL2026 (70)](../../ACL2026/audio_speech/index.md) · [🧪 ICML2026 (36)](../../ICML2026/audio_speech/index.md) · [🤖 AAAI2026 (31)](../../AAAI2026/audio_speech/index.md) · [🧠 NeurIPS2025 (47)](../../NeurIPS2025/audio_speech/index.md)

🔥 **高频主题：** 语音 ×4

**[Action2Sound: Ambient-Aware Generation of Action Sounds from Egocentric Videos](action2sound_ambientaware_generation_of_action_sounds_from_e.md)**

:   提出 AV-LDM，通过在训练时引入同一视频不同时间段的音频作为环境音条件，隐式解耦前景动作声和背景环境音，结合检索增强生成(RAG)在推理时选择合适的环境音条件，在 Ego4D 和 EPIC-KITCHENS 上大幅超越已有方法。

**[Beat-It: Beat-Synchronized Multi-Condition 3D Dance Generation](beat-it_beat-synchronized_multi-condition_3d_dance_generation.md)**

:   提出 Beat-It 框架，通过将节拍条件从音乐中解耦并设计层次化多条件融合机制，实现了节拍同步且关键帧可控的 3D 舞蹈生成，在 AIST++ 上大幅领先现有方法。

**[CoLeaF: A Contrastive-Collaborative Learning Framework for Weakly Supervised Audio-Visual Video Parsing](coleaf_a_contrastive-collaborative_learning_framework_for_weakly_supervised_audi.md)**

:   提出 CoLeaF 双分支学习框架，通过事件感知对比学习显式优化跨模态上下文的整合，在弱监督音视频解析任务上平均提升 1.9% F-score。

**[ControlLLM: Augment Language Models with Tools by Searching on Graphs](controlllm_augment_language_models_with_tools_by_searching_on_graphs.md)**

:   提出 ControlLLM 框架，通过在预构建的工具图（Tool Graph）上进行图搜索（Thoughts-on-Graph）来规划多模态工具调用，显著提升了复杂任务中工具选择和参数赋值的准确性。

**[Label-Anticipated Event Disentanglement for Audio-Visual Video Parsing](label-anticipated_event_disentanglement_for_audio-visual_video_parsing.md)**

:   提出 LEAP（Label semantic-based Projection）解码范式，利用事件类别的标签文本嵌入作为语义锚点，通过跨模态注意力机制将音频/视觉隐特征中潜在重叠的事件语义解耦到独立的标签嵌入中，配合基于 EIoU 的音视觉语义相似度损失，在 AVVP 任务上取得 SOTA。

**[Latent-INR: A Flexible Framework for Implicit Representations of Videos with Discriminative Semantics](latent-inr_a_flexible_framework_for_implicit_representations_of_videos_with_disc.md)**

:   提出 Latent-INR 框架，通过为视频每帧学习一个隐式 latent code 并结合 hypernetwork 进行低秩权重调制，将视频 INR 的空间与时间建模解耦，在保持压缩性能的同时赋予表征语义判别能力，支持检索、视频插帧和任意分辨率推理等多种下游任务。

**[Listen to Look into the Future: Audio-Visual Egocentric Gaze Anticipation](listen_to_look_into_the_future_audio-visual_egocentric_gaze_anticipation.md)**

:   提出 CSTS（Contrastive Spatial-Temporal Separable）音视频融合方法，首次将音频信号引入第一人称注视预测任务，通过空间和时间分离融合模块分别建模音视频的空间共现和时序相关性，并用后融合对比学习增强表示，在 Ego4D 和 Aria 数据集上超越 SOTA。

**[Siamese Vision Transformers are Scalable Audio-Visual Learners](siamese_vision_transformers_are_scalable_audio-visual_learners.md)**

:   提出AVSiam框架，使用单个共享权重的ViT backbone同时处理音频和视觉输入，结合多比例随机掩码策略和对比+重建双目标预训练，以极低成本（比MAViL快28.9倍）在音视觉分类和检索上达到SOTA性能。
