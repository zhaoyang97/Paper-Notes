---
title: >-
  [论文解读] Spherical World-Locking for Audio-Visual Localization in Egocentric Videos
description: >-
  [ECCV 2024][视频理解][第一人称视频] 提出球面世界锁定（Spherical World-Locking, SWL）框架，通过将多模态感知流隐式变换到世界锁定的球面坐标系中，消除自身运动带来的挑战，实现更精准的第一人称视频中的音视觉定位。 第一人称视频提供了来自个人视角的全面上下文信息，但其最显著的特点——自身运…
tags:
  - "ECCV 2024"
  - "视频理解"
  - "第一人称视频"
  - "音视觉定位"
  - "球面世界锁定"
  - "多感官融合"
  - "Transformer"
---

# Spherical World-Locking for Audio-Visual Localization in Egocentric Videos

**会议**: ECCV 2024  
**arXiv**: [2408.05364](https://arxiv.org/abs/2408.05364)  
**代码**: [有](https://hs-yn.github.io/SWL/)  
**领域**: 音频语音  
**关键词**: 第一人称视频, 音视觉定位, 球面世界锁定, 多感官融合, Transformer

## 一句话总结

提出球面世界锁定（Spherical World-Locking, SWL）框架，通过将多模态感知流隐式变换到世界锁定的球面坐标系中，消除自身运动带来的挑战，实现更精准的第一人称视频中的音视觉定位。

## 研究背景与动机

第一人称视频提供了来自个人视角的全面上下文信息，但其最显著的特点——**自身运动**（self-motion）——带来了重大挑战：

**参考系变化**：头部运动导致目标在头部锁定坐标系中的相对位置不断变化

**视野限制**：有限的视场角加上频繁运动导致感兴趣目标频繁出入视野

**运动漂移**：累积运动使得时间对齐变得困难

然而，自身运动同时也是理解场景的重要线索——人类能够高效地稳定感知并利用头部运动来增强注意力。现有工作大多将自身运动视为单纯的挑战，而忽视了它作为行为代理的潜力。本文提出，以**球面世界锁定**的方式将音视觉流重新定位到全局参考系中，既能消除自身运动的干扰，又能利用运动信息增强场景理解。

## 方法详解

### 整体框架

框架围绕两个核心思想构建：

| 概念 | 头部锁定（Head-Locked） | 球面世界锁定（SWL） |
|------|------------------------|---------------------|
| 参考系 | 以头部为中心的 2D 平面 | 以世界坐标为中心的 3D 球面 |
| 自身运动处理 | 模型需从原始数据中学习补偿 | 通过 IMU 测量隐式补偿 |
| 空间同步 | 不同模态间空间对齐困难 | 天然的多模态空间对齐 |
| 计算开销 | 无额外开销 | 可忽略不计的额外开销 |

提出两种 SWL 实现方式：
- **显式 SWL**：将视频映射到 360° 全景图
- **隐式 SWL**（推荐）：保留原始输入，通过球面位置嵌入分离语义和位置信息

### 关键设计

**隐式球面世界锁定**：不直接变换输入数据，而是为每个 patch/token 分配球面位置坐标，保持语义嵌入和位置嵌入的分离。

**多分类 Token（Multi-CLS）**：部署多个分类 Token，每个都参数化为球面上的一个点 $c_i = \mathbf{W}_c p_i + \mathbf{b}_c$，用于捕获该位置周围的语义信息。训练时使用 $5 \times 10$ 的稀疏网格。

**MuST 编码器**（Multisensory Spherical World-Locked Transformer）包含两个关键创新：

1. **球面空间相似度**：基于旋转四元数计算多模态嵌入间的成对空间关系：
$$\mathbf{P}_{ij}^l = \text{Linear}(\text{GELU}(\text{Linear}([1+p_i \cdot p_j, p_i \times p_j])))$$
旋转四元数只需计算一次即可复用于所有层。

2. **模态感知操作（M-ops）**：对每种模态应用独立的 LayerNorm 和 q/k/v 投影（M-LN + M-Attn），促进跨模态交互的同时保持模态特性。

**MuST 解码器**：支持三种灵活的解码策略：
- 稀疏解码：对每个 CLS Token 用 MLP 预测
- 稠密解码：用轻量反卷积网络生成密集输出图
- 水平解码：仅使用赤道附近的 Token，利用 SWL 的重力对齐特性

### 损失函数 / 训练策略

- 使用二元交叉熵损失进行训练
- Adam 优化器，学习率 1e-4，无学习率调度
- 端到端训练 10 个 epoch
- 模型架构参考 DeiT-S（ViT 小变体），参数量略少于 ResNet-50

## 实验关键数据

### 主实验（表格）

**EasyCom 数据集：音视觉活跃说话人定位**：

| 方法 | mAP↑ |
|------|------|
| DOA（信号处理方法） | 52.62 |
| TalkNet | 69.13 |
| AVLN | 85.11 |
| MAVASL_C+E | 86.32 |
| **MuST** | **89.88** |
| Oracle（上界） | 91.03 |

MuST 超越先前最优 3.6 个百分点，接近使用近场麦克风的 Oracle 上界（差距仅 1.2%p）。

### 消融实验（表格）

**编码器组件消融（EasyCom）**：

| 变体 | mAP↑ |
|------|------|
| MuST w/o pose（无姿态信息） | 87.76 |
| MuST w/o rotation（无旋转信息） | 88.83 |
| MuST w/o M-ops（无模态感知操作） | 88.53 |
| MuST M-LN only | 89.67 |
| MuST M-LN+M-Attn（完整版） | 89.88 |

**模态贡献分析**：

| 输入模态组合 | mAP↑ |
|-------------|------|
| 仅姿态 | 47.95 |
| 姿态 + 单声道音频 | 68.57 |
| 姿态 + 视觉 | 68.78 |
| 姿态 + 多声道音频 + 视觉 | 89.88 |

### 关键发现

1. **SWL 的位置和旋转信息贡献显著**：+2.1%p 的性能提升
2. **多声道麦克风阵列是最大性能推动因素**：从单声道到多声道，mAP 从 73.47 跃升至 89.88
3. **模态感知操作至关重要**：去除后性能甚至低于不使用视觉的方案
4. **球面听觉定位表现优异**：在 RLR-CHAT 数据集上，MAE 从次优方法的 44.90° 降至 12.67°
5. **稀疏解码的效率**：仅使用赤道 Token（减少 5 倍数量），性能仅下降约 1°

## 亮点与洞察

- **范式转换**：将自身运动从"需要学习补偿的噪声"转变为"可直接利用的有用信号"，通过 IMU 以近乎零开销实现
- **统一多模态表征**：音频、视觉和行为信息在世界锁定球面上天然对齐，无需复杂的坐标变换
- **四元数旋转的巧妙应用**：用旋转四元数编码球面上两点间的空间关系，既数学自洽又计算高效
- **灵活的解码设计**：Multi-CLS + 多种解码策略（稀疏/稠密/水平）可灵活适配不同任务需求

## 局限与展望

1. 仅考虑旋转而忽略平移，对移动场景有局限
2. 时间窗口较短（<1秒），长时间依赖关系未被建模
3. 依赖 IMU 数据，限制了在无 IMU 设备上的应用
4. 显式 SWL 的全景图投影存在畸变问题
5. 可以探索将 SWL 扩展到其他第一人称视频任务（如动作识别、物体交互）

## 相关工作与启发

- 与传统的 360° 视频球面表征不同，SWL 处理的是平面视频在球面上的隐式表达
- Multi-CLS Token 的设计可启发其他需要空间化预测的视觉任务
- 模态感知操作（M-LN, M-Attn）为多模态 Transformer 设计提供了有效的工程经验

## 评分

| 维度 | 评分 (1-5) | 说明 |
|------|-----------|------|
| 新颖性 | 4.5 | 球面世界锁定的概念新颖，将自身运动转化为有用信号的思路出色 |
| 技术深度 | 4.5 | 四元数球面编码、模态感知操作等设计严谨优雅 |
| 实验充分性 | 4.5 | 三个不同任务/数据集 + 丰富消融，充分验证了方法的有效性和通用性 |
| 实用性 | 4 | 依赖 IMU 数据，但现代 AR/VR 设备普遍配备 |
| 总体 | 4.5 | 在第一人称视频理解领域具有范式意义的工作 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] AMEGO: Active Memory from Long EGOcentric Videos](amego_active_memory_from_long_egocentric_videos.md)
- [\[ECCV 2024\] Online Temporal Action Localization with Memory-Augmented Transformer](online_temporal_action_localization_with_memory-augmented_transformer.md)
- [\[ECCV 2024\] HAT: History-Augmented Anchor Transformer for Online Temporal Action Localization](hat_history-augmented_anchor_transformer_for_online_temporal_action_localization.md)
- [\[CVPR 2026\] Mistake Attribution: Fine-Grained Mistake Understanding in Egocentric Videos](../../CVPR2026/video_understanding/mistake_attribution_fine-grained_mistake_understanding_in_egocentric_videos.md)
- [\[ICML 2026\] AVTrack: Audio-Visual Tracking in Human-centric Complex Scenes](../../ICML2026/video_understanding/avtrack_audio-visual_tracking_in_human-centric_complex_scenes.md)

</div>

<!-- RELATED:END -->
