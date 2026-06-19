---
title: >-
  [论文解读] ImViD: Immersive Volumetric Videos for Enhanced VR Engagement
description: >-
  [CVPR 2025][音频/语音][体积视频] 构建首个沉浸式体积视频数据集——用 46 台同步 GoPro 的移动多视角系统拍摄 7 个场景（含室内/室外），提出 STG++ 增加可学习仿射颜色变换解决跨相机颜色不一致，实现 110.47 FPS 渲染/387MB 存储，并集成 HRTF 空间音频。
tags:
  - "CVPR 2025"
  - "音频/语音"
  - "体积视频"
  - "VR沉浸"
  - "多视角GoPro"
  - "3DGS时序"
  - "空间音频"
---

# ImViD: Immersive Volumetric Videos for Enhanced VR Engagement

**会议**: CVPR 2025  
**arXiv**: [2503.14359](https://arxiv.org/abs/2503.14359)  
**代码**: 待公开  
**领域**: 音频语音 / VR  
**关键词**: 体积视频, VR沉浸, 多视角GoPro, 3DGS时序, 空间音频

## 一句话总结

构建首个沉浸式体积视频数据集——用 46 台同步 GoPro 的移动多视角系统拍摄 7 个场景（含室内/室外），提出 STG++ 增加可学习仿射颜色变换解决跨相机颜色不一致，实现 110.47 FPS 渲染/387MB 存储，并集成 HRTF 空间音频。

## 研究背景与动机

### 领域现状

**领域现状**：VR 体验需要逼真的自由视角渲染。现有体积视频数据集要么基于固定相机阵列（空间受限），要么分辨率/帧率不足以支撑沉浸体验。

**现有痛点**：（1）缺少高分辨率（5K+）、高帧率（60FPS）、同步多视角的动态场景数据；（2）固定阵列覆盖角度有限，不支持自由移动；（3）现有方法不处理跨相机的颜色差异（光照遮挡导致各 GoPro 曝光不一致）。

**核心矛盾**：移动拍摄提供了更大的空间覆盖，但相机位姿标定困难（COLMAP 在视频序列上失败）。

**切入角度**：双策略采集——固定点拍摄（密集时序，可用 COLMAP 标定）+ 移动拍摄（大范围覆盖，位姿待解决）。STG++ 加入可学习颜色变换解决跨相机颜色不一致。

**核心 idea**：46 GoPro 移动阵列 + STG++ 颜色校正 + HRTF 空间音频 = 首个沉浸式体积视频数据集。

### 解决思路

**本文目标**：### 关键设计

1. **采集系统**：46 台同步 GoPro，5312×2988@60FPS。


## 方法详解

### 关键设计

1. **采集系统**：46 台同步 GoPro，5312×2988@60FPS。双策略：固定点（密集时序）+ 移动（大范围）

2. **STG++**：在标准 STG（Spacetime Gaussians）基础上增加可学习的逐相机仿射颜色变换 $C'_i = WC_i + T$——消除因光照遮挡导致的跨相机颜色不一致

3. **HRTF 空间音频**：基于 HRTF（头相关传输函数）将单声道音频转为双耳立体声，根据听者-声源方向 $\theta_s$ 和距离 $\lambda$ 动态调整

### 损失函数 / 训练策略

$\mathcal{L} = (1-\lambda_1)L_1 + \lambda_1 D_{SSIM}$。60 帧分段训练。

## 实验关键数据

| 场景 | STG++ PSNR | FPS | 内存 |
|------|-----------|-----|------|
| Opera | 31.24% | 110.47 | 387MB |
| Lab | 27.58% | — | — |
| 4DRotor (对比) | — | 46.22% | 5818MB |

用户研究（21 位专家）：空间感知 61.9% Excellent，整体沉浸 90.46% ≥ Good。

### 消融实验
- STG++ 颜色校正消除了分段间的颜色闪烁
- 方向+距离联合建模空间音频比单独建模更好
- 4DRotor 在高运动区域更好但内存高 15 倍

### 关键发现
- **颜色不一致是分段训练的核心挑战**——STG++ 的仿射变换简单有效
- **用户沉浸度很高**：90% 以上的专家评价 Good 或 Excellent
- **移动拍摄的位姿仍未解决**——这是留给未来的关键技术挑战

## 亮点与洞察
- **首个面向 VR 的高质量体积视频数据集**——7 场景，46 同步相机，60FPS 5K 分辨率
- **数据集 + 方法 + 评估三位一体**——不仅提供数据，还提供改进的渲染方法和主观评估

## 局限与展望
- **移动拍摄位姿未标定**——COLMAP 在视频序列上失败
- 局部闪烁仍存在（尽管全局颜色对齐）
- 声场模型假设单一静态全向声源
- 连续拍摄受限于散热（~30 分钟）

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个沉浸式体积视频+空间音频数据集
- 实验充分度: ⭐⭐⭐⭐ 渲染对比+用户研究+音频评估
- 写作质量: ⭐⭐⭐⭐ 数据集描述详尽
- 价值: ⭐⭐⭐⭐ 为 VR 内容创作和自由视角渲染提供了关键资源

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Action2Sound: Ambient-Aware Generation of Action Sounds from Egocentric Videos](../../ECCV2024/audio_speech/action2sound_ambientaware_generation_of_action_sounds_from_e.md)
- [\[ECCV 2024\] Latent-INR: A Flexible Framework for Implicit Representations of Videos with Discriminative Semantics](../../ECCV2024/audio_speech/latent-inr_a_flexible_framework_for_implicit_representations_of_videos_with_disc.md)
- [\[ICCV 2025\] Align Your Rhythm: Generating Highly Aligned Dance Poses with Gating-Enhanced Rhythm-Aware Feature Representation](../../ICCV2025/audio_speech/align_your_rhythm_generating_highly_aligned_dance_poses_with_gating-enhanced_rhy.md)
- [\[ACL 2026\] VAPO: End-to-end Slide-Enhanced Speech Recognition with Omni-modal Large Language Models](../../ACL2026/audio_speech/vapo_end-to-end_slide-enhanced_speech_recognition_with_omni-modal_large_language.md)
- [\[ICLR 2026\] Dynamic Parameter Memory: Temporary LoRA-Enhanced LLM for Long-Sequence Emotion Recognition in Conversation](../../ICLR2026/audio_speech/dynamic_parameter_memory_temporary_lora-enhanced_llm_for_long-sequence_emotion_r.md)

</div>

<!-- RELATED:END -->
