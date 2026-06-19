---
title: >-
  [论文解读] MobilePortrait: Real-Time One-Shot Neural Head Avatars on Mobile Devices
description: >-
  [CVPR 2025][图像生成][neural head avatar] 提出首个可在移动端实时运行的单张人脸头像动画方法 MobilePortrait，通过混合显隐式关键点 + 预计算外观知识，仅用 16 GFLOPs 即匹敌 SOTA（100–600+ GFLOPs）的效果。 领域现状： 现有神经头部头像（NHA）方法…
tags:
  - "CVPR 2025"
  - "图像生成"
  - "neural head avatar"
  - "face reenactment"
  - "mobile deployment"
  - "mixed keypoints"
  - "lightweight U-Net"
---

# MobilePortrait: Real-Time One-Shot Neural Head Avatars on Mobile Devices

**会议**: CVPR 2025  
**arXiv**: [2407.05712](https://arxiv.org/abs/2407.05712)  
**代码**: 待确认  
**领域**: 图像生成  
**关键词**: neural head avatar, face reenactment, mobile deployment, mixed keypoints, lightweight U-Net

## 一句话总结

提出首个可在移动端实时运行的单张人脸头像动画方法 MobilePortrait，通过混合显隐式关键点 + 预计算外观知识，仅用 16 GFLOPs 即匹敌 SOTA（100–600+ GFLOPs）的效果。

## 研究背景与动机

**领域现状**: 现有神经头部头像（NHA）方法（如 Real3D、MCNet、FaceV2V）在图像质量和运动范围上取得了显著进展，但计算成本普遍超过 100 GFLOPs，包含多尺度特征扭曲、动态卷积、注意力等复杂模块，无法在移动设备上运行。

**核心问题**: 随着 LLM 和智能手机普及，移动端头像将成为 AI 交互的关键界面，但业界对轻量级 NHA 几乎没有探索。如何在保持 SOTA 质量的同时将计算量降低一个数量级？

**关键观察**:
1. 显式建模（如 Real3D 用 3DMM）在面部之外区域（嘴内、脖子）效果差，因为这些区域没有定义
2. 隐式建模（如 MCNet 用 neural keypoints）在人物与背景边界处模糊，因为缺乏面部先验
3. 结合两者可以互补——面部知识强化运动网络，外观知识强化合成网络

**动机**: 把复杂任务简化为"开卷考试"——通过引入外部先验知识降低网络学习难度，从而使用简单 U-Net 作为骨干也能达到高质量。

## 方法详解

### 整体框架

MobilePortrait 包含两大模块：**运动生成（Motion Generation）**和**图像合成（Image Synthesis）**，两者均使用简单 U-Net（无多尺度特征扭曲、动态卷积、注意力）作为骨干。输入一张源图和驱动帧，通过混合关键点检测 → TPS 变换 → 密集运动网络生成光流 → 图像扭曲 → 合成网络生成最终图像。面部和外观知识在源图给定时仅需预计算一次，推理时几乎无额外开销。

### 关键设计 1：混合关键点表示（Mixed Keypoint Representation）

- **问题**: 计算量减少后，纯 neural keypoints（NK）无法区分面部和背景运动，产生"液化"伪影；纯 facial keypoints（FK）无法捕捉全局运动
- **方案**: 引入预训练面部关键点检测器提取 106 个 FK，与 50 个 NK 合并为 50 个 mixed keypoints，通过 MLP 实现融合
- **辅助设计**: 在密集运动网络最后一层添加残差光流（2 通道输出），增强光流表达能力
- **效果**: Mixed keypoints 相比 NK-Only（FID 48.3→29.2）和 FK-Only（FID 33.2→29.2）均有大幅提升

### 关键设计 2：面部感知运动生成（Face-Aware Motion Generation）

- **额外输入**: 将源图的前景 mask 和面部关键点 mask 加入密集运动网络输入（仅需计算一次）
- **训练辅助损失**: 在 DMN 最后特征层添加两个辅助预测器，分别预测驱动图的前景 mask（$\mathcal{L}_{mask}$）和关键点 mask（$\mathcal{L}_{landmark}$），用 L1 损失训练（仅训练时存在）
- **效果**: 帮助网络理解人像完整性，实现从面部级到视频级的运动捕捉

### 关键设计 3：外观知识增强合成（Appearance Knowledge Enhancement）

- **前景增强**: 从驱动视频中均匀采样 T 帧，与源图生成 T 张扭曲图像，提取 U-Net 最低分辨率层的特征作为伪多视角特征，通过额外卷积层与当前帧特征融合。实验表明 T=4 效果最佳
- **背景增强**: 使用离线修复模型（LaMa）对去除前景后的源图进行补全，得到完整背景图作为合成网络的额外输入。训练时对驱动图也执行修复以确保模型能利用背景信息
- **关键优势**: 所有外观知识均可预计算，推理时几乎零额外成本

### 损失函数

$$\mathcal{L} = \mathcal{L}_{percep} + \mathcal{L}_{L1} + \mathcal{L}_{kp} + \mathcal{L}_{eq} + \mathcal{L}_{landmark} + \mathcal{L}_{mask}$$

- $\mathcal{L}_{percep}$: 感知损失，优化特征距离
- $\mathcal{L}_{L1}$: 像素级 L1 损失
- $\mathcal{L}_{kp}$: 面部关键点距离损失
- $\mathcal{L}_{eq}$: 等变性损失，确保 neural keypoints 稳定性
- $\mathcal{L}_{landmark}$, $\mathcal{L}_{mask}$: 面部知识辅助损失

## 实验关键数据

### 主实验表

与 SOTA 方法在视频驱动同/跨身份重演上的对比（Table 1）:

| 方法 | FLOPs(G) | FID↓ | AKD↓ | HPD↓ | BCI↑ | CSIM↑(cross) |
|------|----------|------|------|------|------|---------------|
| PIRender | 131 | 39.1 | 2.14 | 0.99 | 96.9 | 45.7 |
| FaceV2V | 629 | 29.3 | 1.96 | 2.52 | 97.2 | 46.0 |
| TPS | 140 | 29.8 | 1.43 | 0.71 | 97.9 | 38.9 |
| MCNet | 200 | 27.2 | 1.33 | 0.81 | 97.8 | 27.6 |
| Real3D | 610 | 50.8 | 1.63 | 0.82 | 97.6 | 47.8 |
| **MobilePortrait** | **16** | **29.2** | **1.30** | **0.40** | **98.2** | 39.2 |

- 仅 16 GFLOPs（不到 MCNet 的 1/12），AKD 和 BCI **最优**，HPD 0.40 远超所有方法

### 消融表

**运动生成消融（Table 3）**:
- Mixed Keypoints vs NK-Only: FID 29.2 vs 48.3，AKD 1.30 vs 2.62
- 去除面部知识损失：AKD 从 1.30 降至 1.45
- 去除残差光流：AKD 从 1.30 降至 1.45

**图像合成消融（Table 4）**:
- 无任何增强: FID 30.1, AKD 1.54
- 仅 Inpainted BG: FID 29.2, AKD 1.30（最佳组合）
- 多视角数量：4 views 为最佳平衡点

### 关键发现

1. **移动端实测**: iPhone 14 Pro 上 16G 版本延迟 15.8ms（63 FPS），4G 精简版 5.9ms（169 FPS）；iPhone 12 上 16G 版本 25.5ms（39 FPS）
2. 即使压缩到 4 GFLOPs，借助外部知识仍能保持满意的 FID 和 AKD
3. 混合关键点融合方式（MLP merger）优于变换级联或卷积生成光流等替代方案
4. 背景训练时修复对利用背景信息至关重要

## 亮点与洞察

1. **轻量化思路独到**: 不是单纯压缩模型，而是通过引入外部知识降低任务难度——从"闭卷考试"变为"开卷考试"，让简单网络也能胜任
2. **显隐式互补**: 混合关键点设计简洁有效，用 MLP 合并即可，避免了复杂的多尺度或多阶段融合
3. **预计算策略精妙**: 面部 mask、多视角特征、背景图都仅需对源图计算一次，推理时无额外开销
4. **音频驱动扩展**: 通过 audio-to-mesh + mesh-to-NK 模块实现音频驱动，还可通过 3DMM 进行表情编辑

## 局限性

1. 跨身份场景的 CSIM（身份保持）指标未达最优，虽然可视化效果可接受
2. 强依赖预训练面部关键点检测器质量，非正脸或遮挡场景可能受限
3. 音频驱动模块仅提供 baseline 方案，更复杂的音频到运动设计可进一步提升
4. 背景修复依赖 LaMa 质量，复杂背景可能出现伪影

## 相关工作与启发

- **FOMM/TPS/MCNet**: 隐式运动建模的代表，本文在其基础上引入面部先验互补
- **Real3D/PIRenderer**: 显式 3DMM 驱动的代表，本文取其面部先验优势但避免了全局运动缺失
- **SadTalker/VividTalk**: 音频驱动方案的参考，MobilePortrait 与之兼容
- **启发**: 许多"困难"任务可以通过引入廉价的外部知识来大幅降低模型复杂度，这一思路值得在其他实时任务（如实时换脸、实时分割）中推广

## 评分

⭐⭐⭐⭐ — 首个移动端实时 NHA 方法，10x 计算量压缩下匹敌 SOTA，实用价值极高；核心思想（外部知识降低任务难度）简洁且具有启发性。扣 1 星因为跨身份保持和音频驱动部分相对初步。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Efficient Real-Time Raw-to-Raw Denoising for Extreme Low-Light Ultra HD Video on Mobile Devices](../../CVPR2026/image_generation/efficient_real-time_raw-to-raw_denoising_for_extreme_low-light_ultra_hd_video_on.md)
- [\[CVPR 2025\] SemanticDraw: Towards Real-Time Interactive Content Creation from Image Diffusion](semanticdraw_towards_real-time_interactive_content_creation_from_image_diffusion.md)
- [\[CVPR 2026\] StreamAvatar: Streaming Diffusion Models for Real-Time Interactive Human Avatars](../../CVPR2026/image_generation/streamavatar_streaming_diffusion_models_for_real-time_interactive_human_avatars.md)
- [\[CVPR 2025\] SnapGen-V: Generating a Five-Second Video within Five Seconds on a Mobile Device](snapgen-v_generating_a_five-second_video_within_five_seconds_on_a_mobile_device.md)
- [\[CVPR 2025\] DNF: Unconditional 4D Generation with Dictionary-Based Neural Fields](dnf_unconditional_4d_generation_with_dictionary-based_neural_fields.md)

</div>

<!-- RELATED:END -->
