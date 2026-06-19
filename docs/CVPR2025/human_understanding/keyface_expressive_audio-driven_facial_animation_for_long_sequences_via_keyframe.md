---
title: >-
  [论文解读] KeyFace: Expressive Audio-Driven Facial Animation for Long Sequences via KeyFrame Interpolation
description: >-
  [CVPR 2025][人体理解][音频驱动面部动画] KeyFace 提出一个两阶段扩散框架——先以低帧率生成捕捉关键表情的锚帧，再通过插值模型填充中间帧——解决了现有音频驱动面部动画方法在长序列中身份漂移和质量退化的问题，同时首次支持连续情感（valence/arousal）建模和多种非语音发声 (NSV) 的动画生成。
tags:
  - "CVPR 2025"
  - "人体理解"
  - "音频驱动面部动画"
  - "扩散模型"
  - "关键帧生成"
  - "长序列生成"
  - "情感建模"
  - "非语音发声"
---

# KeyFace: Expressive Audio-Driven Facial Animation for Long Sequences via KeyFrame Interpolation

**会议**: CVPR 2025  
**arXiv**: [2503.01715](https://arxiv.org/abs/2503.01715)  
**代码**: 见项目页面  
**领域**: 人体理解  
**关键词**: 音频驱动面部动画, 扩散模型, 关键帧生成, 长序列生成, 情感建模, 非语音发声

## 一句话总结

KeyFace 提出一个两阶段扩散框架——先以低帧率生成捕捉关键表情的锚帧，再通过插值模型填充中间帧——解决了现有音频驱动面部动画方法在长序列中身份漂移和质量退化的问题，同时首次支持连续情感（valence/arousal）建模和多种非语音发声 (NSV) 的动画生成。

## 研究背景与动机

**领域现状**：音频驱动面部动画近年来随 GAN 和扩散模型的发展取得了显著进步，生成质量已接近真实视频。该技术在虚拟助手、教育、VR 等领域有广泛应用。

**现有痛点**：(1) **长序列质量退化**：大多数方法在超过几秒后出现身份漂移和整体质量下降——自回归方法误差累积，几秒后质量急剧恶化；(2) **额外空间控制的代价**：为解决长序列问题引入目标头部姿态/关键点作为输入，提高了时序一致性但限制了表情的自然度和灵活性；(3) **情感建模的局限**：现有情感驱动方法假设固定情感状态或使用离散情感标签（如"愤怒"、"悲伤"），无法表达情感的连续变化；(4) **忽略非语音发声**：笑声、叹息等 NSV 对自然交流至关重要，但几乎被现有方法完全忽略。

**核心矛盾**：长序列动画需要全局时序一致性，但自回归生成的局部视野导致误差累积——需要一种既有远程依赖又不引入刚性空间约束的生成策略。

**本文目标** 设计一个能生成长时序、高保真、情感丰富、支持 NSV 的音频驱动面部动画框架。

## 方法详解

### 整体框架

KeyFace 基于 Stable Video Diffusion (SVD) 架构的两阶段 pipeline：(1) **关键帧生成阶段**——以低帧率生成 T 个关键帧（间隔 S 帧），条件包括身份帧、音频嵌入和情感参数，覆盖较长时间跨度以捕捉关键表情变化；(2) **插值阶段**——相同架构但不同条件输入，以两个相邻关键帧为锚点，填充 S-2 个中间帧，确保平滑过渡和时序连贯。长视频可通过重复此过程生成，插值模型保证片段间无缝衔接。

### 关键设计

1. **关键帧生成模型**：
    - 功能：生成稀疏但信息丰富的锚帧，隐式分离运动和身份控制
    - 核心思路：输入为噪声序列，身份帧经 VAE 编码后重复 T 次拼接到噪声中。通过 U-Net 的 skip connection 保持身份信息。音频通过 cross-attention 和 timestep embedding 两种机制注入
    - 设计动机：低帧率生成使模型可以覆盖更长的时间跨度（数秒），从全局角度捕捉面部表情和运动模式，避免自回归方法的短视窗问题

2. **双音频编码器 (WavLM + BEATs)**：
    - 功能：同时捕捉语言内容和非语音声学特征
    - 核心思路：WavLM 擅长提取语言特征（唇同步），BEATs 擅长捕捉广泛的声学信号（包括 NSV）。两者的嵌入拼接后通过两种途径注入模型：(a) 作为 cross-attention 的 key/value；(b) 经 MLP 后加到扩散 timestep embedding 上
    - 实验验证：移除 BEATs 后 NSV 准确率从 42% 降至 10%（接近随机），移除 WavLM 则唇同步质量下降

3. **连续情感建模 (Valence & Arousal)**：
    - 功能：支持情感在视频生成过程中连续变化
    - 核心思路：每帧使用预训练情感识别模型提取 valence（效价）和 arousal（唤醒度），编码为正弦嵌入，加到扩散 timestep embedding 上
    - 仅在关键帧模型中使用情感条件，插值模型可自动传播情感表达
    - 推理时用户可提供任意 valence/arousal 值控制情感状态，并可在同一视频内插值产生情感变化

### 损失函数

$$L = \lambda_{tot}(L_2(z_0, z_{gt}) + L_2(x_0, x_{gt}) + L_p(x_0, x_{gt}))$$

其中 $\lambda_{tot} = \lambda(t) \cdot \lambda_{lower}$：

- **潜空间 L2 损失**：标准扩散训练损失，对所有帧计算
- **像素空间 L2 损失**：解码回 RGB 空间后与真实帧的重建损失，仅对单个随机帧计算（节省显存）
- **感知损失**：基于 VGG 特征匹配，增强感知质量
- **下半部权重 λ_lower=3**：对图像下半部分（嘴部区域）施加更高权重，增强唇同步质量

关键帧模型使用分离式 CFG（分别控制身份和音频的引导强度），插值模型使用 Autoguidance 避免 CFG 过度放大条件信号。

## 实验关键数据

| 方法 | FID↓ | FVD↓ | LipScore↑ | AQ↑ | Elo↑ |
|------|------|------|------|------|------|
| KeyFace (HDTF) | **16.76** | **137.25** | 0.36 | **0.59** | **1091.52** |
| Hallo | 19.22 | 236.97 | 0.27 | 0.55 | 1054.69 |
| V-Express | 34.68 | 200.67 | **0.37** | 0.55 | 985.35 |
| AniPortrait | 20.68 | 299.09 | 0.14 | 0.56 | 887.84 |
| EchoMimic | 20.35 | 213.30 | 0.17 | 0.55 | 1023.53 |
| SadTalker | 60.55 | 410.86 | 0.24 | 0.52 | 960.44 |

| 情感评估 (MEAD) | FID↓ | FVD↓ | Emo_acc↑ |
|------|------|------|------|
| KeyFace (V&A) | **44.43** | **447.74** | **0.67** |
| KeyFace (离散标签) | 50.34 | 509.13 | 0.43 |
| EDTalk | 101.19 | 619.90 | 0.72 |
| EAT | 75.69 | 560.61 | 0.54 |

## 亮点与洞察

1. **关键帧+插值的两阶段范式**：隐式分离了运动规划（关键帧捕捉 what happens）和运动细化（插值处理 how it transitions），相比自回归方法从根本上避免了误差累积——FID 随时间推移几乎不增长
2. **连续情感 vs 离散标签**：valence/arousal 的连续表示不仅比离散标签更细粒度（Emo_acc 从 0.43 提升到 0.67），更关键的是允许情感在视频内平滑变化，这对长序列叙述至关重要
3. **Autoguidance 替代 CFG 用于插值**：在插值阶段需要细腻的表情过渡，CFG 的条件放大可能破坏自然度——Autoguidance 用较小/少步模型引导，平衡了质量和多样性
4. **LipScore 新指标**：基于 lipreader 的感知评分比 SyncNet 更可靠，训练数据量是 SyncNet 的 6 倍，与人类感知更相关

## 局限性

1. 关键帧间隔 S 是固定超参数，不同说话速度或场景复杂度可能需要自适应调整
2. 训练需要 160 小时语音 + 30 小时 NSV 数据，数据收集成本较高
3. 情感标签来自伪标签（预训练模型提取），精度受限于情感识别模型的能力
4. 目前仅处理面部区域，不包含手势或身体运动的生成

## 相关工作

- **音频驱动面部动画**：Wav2Lip（唇同步专家判别器）、Hallo（自回归扩散）、AniPortrait（音频→关键点→视频两阶段）、EchoMimic（音频+关键点条件扩散）
- **情感驱动生成**：EAT（离散情感标签）、EDTalk（视频驱动情感迁移）
- **非语音发声**：Laughing Matters（笑声扩散模型）、LaughTalk（3D笑声+语音模型）
- **视频扩散模型**：SVD（Stable Video Diffusion 基础架构）、EDM（高效扩散框架）

## 评分

- 新颖性：⭐⭐⭐⭐（关键帧+插值范式 + 连续情感 + NSV 的组合在面部动画中是全新的）
- 实用性：⭐⭐⭐⭐⭐（直接解决长序列面部动画的核心痛点，有明确的应用场景）
- 技术深度：⭐⭐⭐⭐（双阶段扩散设计、分离式CFG/Autoguidance、多损失组合）
- 表达清晰度：⭐⭐⭐⭐⭐（结构清晰，消融实验充分，新指标定义合理）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] MoEE: Mixture of Emotion Experts for Audio-Driven Portrait Animation](moee_mixture_of_emotion_experts_for_audio-driven_portrait_animation.md)
- [\[CVPR 2025\] X-Dyna: Expressive Dynamic Human Image Animation](x-dyna_expressive_dynamic_human_image_animation.md)
- [\[CVPR 2025\] Sonic: Shifting Focus to Global Audio Perception in Portrait Animation](sonic_shifting_focus_to_global_audio_perception_in_portrait_animation.md)
- [\[CVPR 2026\] PC-Talk: Precise Facial Animation Control for Audio-Driven Talking Face Generation](../../CVPR2026/human_understanding/pc-talk_precise_facial_animation_control_for_audio-driven_talking_face_generatio.md)
- [\[NeurIPS 2025\] MOSPA: Human Motion Generation Driven by Spatial Audio](../../NeurIPS2025/human_understanding/mospa_human_motion_generation_driven_by_spatial_audio.md)

</div>

<!-- RELATED:END -->
