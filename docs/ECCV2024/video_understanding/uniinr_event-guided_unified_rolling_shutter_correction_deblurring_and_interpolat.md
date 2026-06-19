---
title: >-
  [论文解读] UniINR: Event-guided Unified Rolling Shutter Correction, Deblurring, and Interpolation
description: >-
  [ECCV 2024][视频理解][事件相机] 提出 UniINR 框架，利用统一的时空隐式神经表征（INR）从单张卷帘快门模糊帧和配对事件流中，一次性同时完成卷帘快门校正、去模糊和任意帧率的视频帧插值。 基于 CMOS 传感器的消费级相机普遍采用卷帘快门（Rolling Shutter, RS）机制，在快速运动场景下…
tags:
  - "ECCV 2024"
  - "视频理解"
  - "事件相机"
  - "卷帘快门校正"
  - "去模糊"
  - "视频帧插值"
  - "隐式神经表征"
---

# UniINR: Event-guided Unified Rolling Shutter Correction, Deblurring, and Interpolation

**会议**: ECCV 2024  
**arXiv**: [2305.15078](https://arxiv.org/abs/2305.15078)  
**代码**: [有](https://github.com/yunfanLu/UniINR)  
**领域**: 视频理解  
**关键词**: 事件相机, 卷帘快门校正, 去模糊, 视频帧插值, 隐式神经表征

## 一句话总结

提出 UniINR 框架，利用统一的时空隐式神经表征（INR）从单张卷帘快门模糊帧和配对事件流中，一次性同时完成卷帘快门校正、去模糊和任意帧率的视频帧插值。

## 研究背景与动机

基于 CMOS 传感器的消费级相机普遍采用卷帘快门（Rolling Shutter, RS）机制，在快速运动场景下，拍摄的帧往往同时遭受 RS 畸变和运动模糊两种退化。恢复高帧率的全局快门（Global Shutter, GS）清晰帧需要同时考虑三个任务：

**RS 校正**：消除逐行曝光导致的空间畸变

**去模糊**：消除曝光时间内的运动模糊

**帧插值**：在时间维度上生成中间帧

传统做法是将三个任务拆分为独立子问题并级联现有网络，但这会导致累积误差和明显伪影。例如，将帧插值网络与 RS 校正网络级联会产生退化结果。事件相机凭借高时间分辨率的特性，为解决该问题提供了可能，但现有事件引导方法（如 EvUnRoll、TimeLens）也只能处理部分子任务，无法统一解决。

## 方法详解

### 整体框架

UniINR 将"从 RS 模糊帧和配对事件中恢复任意帧率 GS 清晰帧"的问题建模为一个函数 $F(\mathbf{x}, t, \theta)$，其中 $\mathbf{x}=(x,y)$ 是像素位置，$t$ 是曝光时间内的时间戳，$\theta$ 是函数参数。框架包含三个核心组件：

| 组件 | 功能 | 输入 | 输出 |
|------|------|------|------|
| STE（时空隐式编码） | 从 RS 模糊帧和事件中提取时空表征 | RS 模糊帧 + 事件流 | 时空表征 $\theta \in \mathbb{R}^{H \times W \times C}$ |
| ETE（曝光时间嵌入） | 将目标帧的曝光时间信息编码为时间张量 | GS/RS 时间戳 | 时间张量 $T \in \mathbb{R}^{H \times W \times C}$ |
| PPD（逐像素解码） | 从 STR 中查询并解码清晰帧 | $\theta + T$ | 清晰 GS/RS 帧 |

### 关键设计

**时空隐式编码（STE）**：受 eSL-Net 启发，采用基于稀疏学习的骨干网络从 RS 模糊帧和事件流中提取时空表征（STR）。STR 以 $H \times W \times C$ 的 3D 张量形式存储运动信息，可直接将时间和坐标映射到 RGB 值。这种方式避免了传统光流估计的高计算开销。

**曝光时间嵌入（ETE）**：由于 GS 帧的所有像素同时曝光、RS 帧逐行曝光，需要对两种模式分别构建时间戳图（Timestamp Map）：
- GS 时间戳图：$M_g[h][w] = t_g$（所有像素相同）
- RS 时间戳图：$M_r[h][w] = t_s + (t_e - t_s) \times h / H$（随行号线性变化）

通过单层 MLP 将 $H \times W \times 1$ 的时间戳图升维至 $H \times W \times C$，与 STR 维度对齐。

**逐像素解码（PPD）**：使用 5 层 MLP 解码器，将时间张量 $T$ 与 STR $\theta$ 通过逐元素相加融合后，逐像素解码输出清晰帧：$I = f_{mlp}^{\circlearrowright^5}(T \oplus \theta)$。核心优势在于编码器仅需调用一次，解码器可高效调用 N 次生成 N 帧。

### 损失函数 / 训练策略

总损失由两部分组成：

$$\mathcal{L} = \lambda_b \mathcal{L}_b + \lambda_{re} \mathcal{L}_{re}$$

- **模糊帧引导积分损失** $\mathcal{L}_b$：将一系列预测的 RS 清晰帧取平均重建 RS 模糊帧，与输入 RS 模糊帧做 Charbonnier 损失
- **重建损失** $\mathcal{L}_{re}$：直接监督预测的 GS 清晰帧与 GT 之间的 Charbonnier 损失

使用 Adam 优化器，学习率 $1 \times 10^{-4}$，在 2 块 NVIDIA RTX A5000 上训练 400 个 epoch，batch size 为 2，采用混合精度训练。

## 实验关键数据

### 主实验（表格）

**RS 校正性能对比（Fastec-Orig 数据集）**：

| 方法 | 输入帧数 | 事件 | 参数量(M) | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|---------|------|----------|-------|-------|--------|
| DSUN | 2 | ✗ | 3.91 | 26.52 | 0.79 | 0.122 |
| CVR | 2 | ✗ | 42.69 | 28.72 | 0.85 | 0.111 |
| EvUnroll | 1 | ✓ | 20.83 | 31.32 | 0.88 | 0.084 |
| EvShutter | 1 | ✓ | - | 32.41 | 0.91 | 0.061 |
| **UniINR** | **1** | **✓** | **0.38** | **33.91** | **0.92** | **0.049** |

### 消融实验（表格）

**RS 校正 + 去模糊性能（Gev-Orig 数据集）**：

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| DSUN | 23.10 | 0.70 | 0.166 |
| JCD | 24.90 | 0.82 | 0.105 |
| EvUnRoll | 30.14 | 0.91 | 0.061 |
| NIRE | 29.86 | 0.91 | - |
| **UniINR** | **31.47** | **0.93** | **0.038** |

### 关键发现

1. **参数量极少**：UniINR 仅 0.38M 参数，比 EvUnroll (20.83M) 小 55 倍，比 CVR (42.69M) 小 112 倍
2. **推理速度极快**：31× 帧插值时，每帧处理仅需 2.8ms，而级联方法 EvUnRoll+TimeLens 需要 177ms 以上
3. **全面优于先前方法**：在 RS 校正、RS+去模糊、RS+去模糊+帧插值三个设置上均显著超越现有方法
4. **效率随插值倍数增长非线性**：从 1× 到 31× 插值，实际时间从 31ms 增至 86ms，因为编码器只调用一次

## 亮点与洞察

- **统一建模的思想**：将三个紧密耦合的任务通过 INR 统一到一个连续函数中，避免了级联方法的累积误差，这一思路对其他多退化恢复问题有启发
- **轻量高效**：0.38M 参数 + 2.8ms/帧的推理速度使其具备实际应用潜力
- **事件相机的互补性**：事件流提供高时间分辨率的运动信息，完美弥补了 RS 模糊帧的信息缺失
- **灵活的输出模式**：通过改变 ETE 中的时间戳图，可以输出 RS 或 GS 帧，框架设计非常灵活

## 局限与展望

1. 依赖事件相机数据，限制了在普通相机场景的应用
2. 当前方法基于仿真数据训练，真实世界数据缺乏 GT 无法定量评估
3. 稀疏学习骨干网络的具体结构细节有待进一步分析
4. 对于极端运动或复杂遮挡场景的鲁棒性需进一步验证
5. 可以探索将该框架扩展到更高分辨率（如 4K）的场景

## 相关工作与启发

- 与 EvUnRoll（两阶段先去模糊再 RS 校正）和 TimeLens（事件引导帧插值）不同，UniINR 实现了真正的一体化处理
- INR 在低级视觉中的应用（LIIF 用于超分辨率、VideoINR 用于视频处理）为本文提供了灵感
- 可启发在其他多退化恢复任务中采用类似的统一隐式表征思路

## 评分

| 维度 | 评分 (1-5) | 说明 |
|------|-----------|------|
| 新颖性 | 4.5 | 首次统一 RS 校正+去模糊+帧插值为单一 INR 问题 |
| 技术深度 | 4 | 数学建模严谨，时空分解设计巧妙 |
| 实验充分性 | 4 | 多数据集、多任务设置，定量+定性对比全面 |
| 实用性 | 4 | 极轻量参数 + 高推理速度，实用性强 |
| 总体 | 4 | 方法简洁优雅，效果显著，是事件相机视频恢复领域的优秀工作 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] IAM-VFI: Interpolate Any Motion for Video Frame Interpolation with Motion Complexity Map](iam-vfi_interpolate_any_motion_for_video_frame_interpolation_with_motion_complex.md)
- [\[AAAI 2026\] VTinker: Guided Flow Upsampling and Texture Mapping for High-Resolution Video Frame Interpolation](../../AAAI2026/video_understanding/vtinker_guided_flow_upsampling_and_texture_mapping_for_high-resolution_video_fra.md)
- [\[ICCV 2025\] EMoTive: Event-Guided Trajectory Modeling for 3D Motion Estimation](../../ICCV2025/video_understanding/emotive_event-guided_trajectory_modeling_for_3d_motion_estimation.md)
- [\[CVPR 2026\] One-Shot Flow, Any-Time Frame: A Bidirectional Warping Framework for Event-Based Video Frame Interpolation](../../CVPR2026/video_understanding/one-shot_flow_any-time_frame_a_bidirectional_warping_framework_for_event-based_v.md)
- [\[ECCV 2024\] Text-Guided Video Masked Autoencoder](text-guided_video_masked_autoencoder.md)

</div>

<!-- RELATED:END -->
