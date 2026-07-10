---
title: >-
  ECCV2026 音频/语音论文汇总 · 7篇论文解读
description: >-
  7篇ECCV2026的音频/语音方向论文解读，涵盖语音、扩散模型等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ECCV2026"
  - "音频/语音"
  - "论文解读"
  - "论文笔记"
  - "语音"
  - "扩散模型"
item_list:
  - u: "a_first_exploration_of_neuromorphic_ot-cfm_for_multi-speaker_vsr/"
    t: "LipsFlow：神经形态 OT-CFM 在多说话人视觉语音识别中的首次探索"
  - u: "mg-rwkv_multi-grained_context-aware_rwkv_for_temporal_forgery_localization/"
    t: "MG-RWKV: Multi-Grained Context-Aware RWKV for Temporal Forgery Localization"
  - u: "olive_view-augmented_latent_prediction_with_waveform_reconstruction_for_speech_ssl/"
    t: "OLIVE: View-Augmented Latent Prediction with Waveform Reconstruction for Speech SSL"
  - u: "see_sniff_learning_visuo-olfactory_representations/"
    t: "See & Sniff: Learning Visuo-Olfactory Representations"
  - u: "sparsity-inducing_divergence_losses_for_biometric_verification/"
    t: "Sparsity-Inducing Divergence Losses for Biometric Verification"
  - u: "step-by-step_video-to-audio_synthesis_via_negative_audio_guidance/"
    t: "Step-by-Step Video-to-Audio Synthesis via Negative Audio Guidance"
  - u: "swiftaudio_data_efficient_caption_only_distillation/"
    t: "SwiftAudio: Data-Efficient Caption-Only Distillation for One-Step Text-to-Audio Diffusion-based Generation"
item_total: 7
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎵 音频/语音

**🎞️ ECCV2026** · **7** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (22)](../../CVPR2026/audio_speech/index.md) · [🔬 ICLR2026 (79)](../../ICLR2026/audio_speech/index.md) · [💬 ACL2026 (70)](../../ACL2026/audio_speech/index.md) · [🧪 ICML2026 (36)](../../ICML2026/audio_speech/index.md) · [🤖 AAAI2026 (31)](../../AAAI2026/audio_speech/index.md) · [🧠 NeurIPS2025 (47)](../../NeurIPS2025/audio_speech/index.md)

🔥 **高频主题：** 语音 ×3

**[LipsFlow：神经形态 OT-CFM 在多说话人视觉语音识别中的首次探索](a_first_exploration_of_neuromorphic_ot-cfm_for_multi-speaker_vsr.md)**

:   LipsFlow 首次将神经形态事件相机感知与最优传输条件流匹配（OT-CFM）引入多说话人视觉语音识别，通过可学习事件表示捕捉毫秒级唇部动态、OT-CFM 实现 2 步 ODE 确定性解码、以及双层语义监督解决同形异义词歧义，在 DVS-Lip 上以 22.3% WER 和 240ms 延迟取得 SOTA。

**[MG-RWKV: Multi-Grained Context-Aware RWKV for Temporal Forgery Localization](mg-rwkv_multi-grained_context-aware_rwkv_for_temporal_forgery_localization.md)**

:   MG-RWKV 提出基于 RWKV 线性复杂度递归架构的时序伪造定位（TFL）框架，通过双向 RWKV 捕获全局时序上下文、多粒度混合专家（MG-MoE）自适应选择显式时间感受野、以及跨粒度一致性约束（CGC）消除多尺度特征间的矛盾预测，在 Lav-DF、TVIL、Psynd、AV-Deepfake1M 四个基准上全面超越此前 SOTA，同时保持 O(T) 线性复杂度，推理仅需 73.4ms。

**[OLIVE: View-Augmented Latent Prediction with Waveform Reconstruction for Speech SSL](olive_view-augmented_latent_prediction_with_waveform_reconstruction_for_speech_ssl.md)**

:   OLIVE 提出一种同时优化「分析」（视图增强掩码蒸馏）与「合成」（波形重建）的语音自监督学习框架，通过在早期编码器特征上施加重建约束、后期上下文表征由不变性蒸馏主导，使单一预训练模型兼顾下游判别任务与高质量波形生成。

**[See & Sniff: Learning Visuo-Olfactory Representations](see_sniff_learning_visuo-olfactory_representations.md)**

:   本文利用"气味身份在语义类别内对视觉变换保持不变"这一关键洞见，将仅含气味传感数据的 SmellNet 数据集扩展为视觉-嗅觉配对数据集 SmellNet-V，并设计了基于密集局部对比对齐的自监督双流框架 See & Sniff，在气味分类、跨模态检索和新引入的像素级气味定位三个任务上均显著超越仅使用气味的基线方法。

**[Sparsity-Inducing Divergence Losses for Biometric Verification](sparsity-inducing_divergence_losses_for_biometric_verification.md)**

:   Q-Margin 将 margin 惩罚从几何化的 logit 修改迁移到 α-散度损失的概率化参考测度中，在保持后验稀疏性的同时在 IJB-B/C 和 VoxCeleb 低 FAR 场景下一致超越 ArcFace/CosFace 基线，并通过精确 top-K 截断将训练吞吐开销从 27% 降至 5%。

**[Step-by-Step Video-to-Audio Synthesis via Negative Audio Guidance](step-by-step_video-to-audio_synthesis_via_negative_audio_guidance.md)**

:   针对现有视频到音频（V2A）模型只能一次性生成整段声音、无法像 Foley 艺术家那样逐层补声的痛点，本文提出 Negative Audio Guidance（NAG）：训练一个以「已生成音轨」为条件的分支，采样时把它反向使用，把当前生成推离已有声音，从而只用普通单参考音视频数据集就实现了分步生成互补音轨、混合成高质量合成音。

**[SwiftAudio: Data-Efficient Caption-Only Distillation for One-Step Text-to-Audio Diffusion-based Generation](swiftaudio_data_efficient_caption_only_distillation.md)**

:   SwiftAudio 将 Variational Score Distillation（VSD）适配到音频域，仅使用约 45K 文本描述（无需配对音频数据）就将多步扩散教师蒸馏为一步文生音频生成器，配合时序全变分正则化约束潜空间的时间连贯性，在 AudioCaps 和 Clotho 上达到一步方法中的 SOTA，并大幅缩小了与多步扩散系统的差距。
