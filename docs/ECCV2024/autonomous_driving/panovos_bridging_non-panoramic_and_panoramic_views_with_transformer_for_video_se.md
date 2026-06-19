---
title: >-
  [论文解读] PanoVOS: Bridging Non-panoramic and Panoramic Views with Transformer for Video Segmentation
description: >-
  [ECCV 2024][自动驾驶][全景视频分割] 提出首个全景视频目标分割数据集 PanoVOS（150个视频、19K实例标注），揭示现有 VOS 模型无法处理全景视频的像素不连续和严重畸变问题，并设计 PSCFormer 利用全景空间一致性注意力解决左右边界连续性问题。 全景视频（360° × 180° FoV）在自动驾…
tags:
  - "ECCV 2024"
  - "自动驾驶"
  - "全景视频分割"
  - "视频目标分割"
  - "数据集"
  - "Transformer"
  - "空间一致性"
---

# PanoVOS: Bridging Non-panoramic and Panoramic Views with Transformer for Video Segmentation

**会议**: ECCV 2024  
**arXiv**: [2309.12303](https://arxiv.org/abs/2309.12303)  
**代码**: [https://github.com/shilinyan99/PanoVOS](https://github.com/shilinyan99/PanoVOS)  
**领域**: 自动驾驶  
**关键词**: 全景视频分割, 视频目标分割, 数据集, Transformer, 空间一致性

## 一句话总结

提出首个全景视频目标分割数据集 PanoVOS（150个视频、19K实例标注），揭示现有 VOS 模型无法处理全景视频的像素不连续和严重畸变问题，并设计 PSCFormer 利用全景空间一致性注意力解决左右边界连续性问题。

## 研究背景与动机

全景视频（360° × 180° FoV）在自动驾驶和 VR 中应用广泛，提供比普通视频更丰富的空间信息。然而现有 VOS 数据集（DAVIS、YouTube-VOS）仅关注传统针孔相机拍摄的平面视频，存在以下问题：

**数据集缺失**：没有针对全景视频的像素级实例标注数据集用于视频分割评估

**内容不连续**：全景视频等距柱状投影后，左右边界实际上是连续的，但在平面图像中被割裂。例如一个企鹅跨越图像左右边界时会被分成两部分

**畸变严重**：全景投影导致物体形状严重变形，现有方法的特征匹配机制在此场景下失效

已有的全景视频数据集（SHD360、SOD360、Wild360）要么仅用于显著性检测、要么无实例级标注、要么运动幅度小，无法用于 VOS 评估。

## 方法详解

### 整体框架

工作分两部分：(1) 构建 PanoVOS 数据集；(2) 提出 PSCFormer 模型。PSCFormer 基于 AOT 架构，引入全景空间一致性（PSC）模块替代标准注意力。给定查询帧和参考帧，通过 memory encoder 和 query encoder 提取特征，多个堆叠的 PSC block 进行时空匹配，最后解码器生成分割掩码。

### 关键设计

1. **PanoVOS 数据集构建**：

    - **规模**：150 个视频、13,995 帧、19,145 个实例标注、35 个类别
    - **特点**：平均视频长度 20s（是 DAVIS/YouTube-VOS 的 4 倍），近半视频为 4K 分辨率，运动幅度大
    - **类别**：覆盖人物（parkour、skateboard）、动物（elephant、monkey）、常见物体（basketball、hot balloon）
    - **划分**：训练 80、验证 35、测试 35 个视频，验证/测试集包含未见类别用于泛化评估
    - **标注流程**：半自动人机协作——先在 1fps 关键帧人工标注，用 AOT 模型传播到 6fps 的所有帧，再人工精修（特别是全景畸变和不连续区域）

2. **Panoramic Space Consistency Block (PSC Block)**：

    - **结构**：Self-Attention → Cross-Attention + PSC-Attention → FFN (GELU)
    - Self-Attention 聚合查询帧内目标的关联信息
    - Cross-Attention 从参考帧学习目标信息
    - PSC-Attention 专门处理左右边界一致性（核心创新）

3. **PSC-Attention（全景空间一致性注意力）**：

    - **核心思路**：通过特征拼接模拟全景图的左右边界连续性
    - 将参考帧特征 $\mathbf{f}(\mathbf{x}) \in \mathbb{R}^{H \times W \times C}$ 的右侧 $W/p$ 列移到最左边，左侧 $W/p$ 列移到最右边，中间不变，得到重排特征 $\mathbf{f}(\mathbf{x})'$
    - 在重排特征上施加窗口化注意力，每个 query token 只与 $(2s+1)^2$ 大小窗口内的 key 计算注意力：

    $\text{PSCAttn}(Q, K, V) = \text{softmax}\left(\frac{QK^T \mathbf{R}}{\sqrt{C}}\right) V$

   其中 $\mathbf{R}$ 是窗口掩码矩阵，位置 $(x,y)$ 的 query 只关注 $(i,j)$ 满足 $(x-i)^2 \leq s^2$ 且 $(y-j)^2 \leq s^2$ 的 key
    - 复杂度从 $(HW)^2$ 降至 $(2s+1)^2$，高效且有效
    - 设计动机：直接拼接全图长度翻倍计算量爆炸，只拼接边界区域并用窗口注意力就能在全景视频中连接原本割裂的物体
    - 超参：$p=2$（拼接比例），$s=7$（窗口大小），8 个注意力头

4. **两种模型变体**：

    - **Ours-Base**：仅用第一帧和前一帧作为参考（$\mathcal{R} = \{1, t-1\}$），推理快
    - **Ours-Large**：用多历史帧参考，性能更好

### 损失函数 / 训练策略

使用标准 VOS 损失（交叉熵 + dice loss）。先在静态图像数据集上预训练（COCO、ECSSD），再在 PanoVOS 训练集上主训练。训练和测试时的 $\delta$ 分别为 2 和 5（历史帧采样间隔）。

## 实验关键数据

### 主实验

**域迁移结果（在 PanoVOS 上测试现有方法，仅在传统数据集训练）**：

| 方法 | YouTube-VOS $\mathcal{J\&F}$ | PanoVOS Val $\mathcal{J\&F}$ | 下降 |
|------|-----|-----|-----|
| XMem | 85.7 | 66.1 | ↓19.6 |
| AOTL | 83.8 | 71.9 | ↓11.9 |
| AOTB | 83.5 | 70.5 | ↓13.0 |
| STCN | 83.0 | 61.8 | ↓21.2 |
| AFB-URR | 79.6 | 55.1 | ↓24.5 |

**在 PanoVOS 上训练后结果**：

| 方法 | Val $\mathcal{J\&F}$ | Test $\mathcal{J\&F}$ |
|------|-----|------|
| XMem | 55.7 | 53.5 |
| AOTL | 66.6 | 53.8 |
| AOTB | 67.6 | 55.4 |
| **Ours-Base** | **74.0** | **56.8** |
| **Ours-Large** | **77.9** | **59.9** |

### 消融实验

**PSC-Attention 效果**：

| 模型 | PSCAttn | Val $\mathcal{J\&F}$ | Test $\mathcal{J\&F}$ | 说明 |
|------|---------|------|------|------|
| Ours-Base | ✗ | 72.8 | 55.4 | 基线 |
| Ours-Base | ✓ | **74.0** | **56.8** | +1.2 / +1.4 |
| Ours-Large | ✗ | 74.8 | 59.5 | 基线 |
| Ours-Large | ✓ | **77.9** | **59.9** | +3.1 / +0.4 |

**PSCAttn vs. 标准 Cross-Attention**：

| 模型 | 注意力类型 | Val $\mathcal{J\&F}$ | Test $\mathcal{J\&F}$ |
|------|-----------|------|------|
| Ours-Base | CrossAttn | 72.5 | 54.8 |
| Ours-Base | PSCAttn | **74.0** | **56.8** |
| Ours-Large | CrossAttn | 76.8 | 59.1 |
| Ours-Large | PSCAttn | **77.9** | **59.9** |

**拼接比例 $p$ 分析**（Ours-Large）：$p=2$ 最优（$\mathcal{J\&F}$=77.9），不拼接时仅 73.7（↓4.2），$p=3$ 为 76.3，$p=5/10/15$ 都不如 $p=2$。

### 关键发现

- **巨大的域迁移差距**：所有 15 个现有 VOS 模型在 PanoVOS 上性能大幅下降（平均 ↓20+），证实全景视频是一个独特且未解决的挑战
- **SAM 系列表现差**：PerSAM 仅 19.1 $\mathcal{J\&F}$，SAM-PT 也只有 47.5，视觉基础模型在全景场景下仍有大量改进空间
- **PSCAttn 一致有效**：在 Base 和 Large 两种配置下都带来提升，且优于简单的额外 Cross-Attention
- 在更大规模数据（YouTube-VOS、BL30K）上训练可部分缓解域差距，但不能完全解决
- 剩余问题：严重畸变仍未处理（当前方法无 deformable 设计）

## 亮点与洞察

- **数据集贡献为核心**：PanoVOS 填补了全景 VOS 评估的空白，150 个高质量视频、20s 平均时长、4K 分辨率，比现有全景视频数据集更具挑战性
- **PSC-Attention 设计巧妙**：通过简单的边界区域交换 + 窗口注意力就能建模全景连续性，避免了全图拼接的计算爆炸，思路简洁有效
- 15 个模型的全面基准评估为后续全景视频理解研究提供了有价值的参照

## 局限与展望

- **畸变未处理**：PSCFormer 没有针对全景投影的严重畸变做特殊设计（如 deformable convolution），对极度变形物体仍会失败
- **方法通用性有限**：PSC-Attention 专为全景视频设计，对普通视频可能引入不必要的计算
- 数据集规模相对较小（150 个视频），可进一步扩展
- 可探索与全景深度估计、全景 3D 感知的结合

## 相关工作与启发

- 与 SHD360（小运动/人体场景）、SOD360/Wild360（无实例标注）不同，PanoVOS 提供了大运动+实例标注的全景 VOS 基准
- AOT 系列是主要基线，PSCFormer 在其基础上加入全景特有的空间一致性设计
- 类似的边界连续性问题也存在于全景语义分割（DensePASS）和全景 3D 检测中，可互相借鉴

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 数据集+方法双贡献，首次系统研究全景 VOS 问题，PSC-Attention 设计有创意
- **实验充分度**: ⭐⭐⭐⭐⭐ — 15 个模型的全面基准、SAM 评估、消融实验、超参分析、多种训练策略对比，非常详尽
- **写作质量**: ⭐⭐⭐⭐ — 问题动机清晰，数据集构建流程完整，5 个 RQ 结构化回答
- **价值**: ⭐⭐⭐⭐ — 开辟了全景 VOS 这一新方向，数据集对社区有持续价值，但方法本身可能适用范围较窄

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Panoramic Multimodal Semantic Occupancy Prediction for Quadruped Robots](../../CVPR2025/autonomous_driving/panoramic_multimodal_semantic_occupancy_prediction_for_quadruped_robots.md)
- [\[CVPR 2026\] OneOcc: Semantic Occupancy Prediction for Legged Robots with a Single Panoramic Camera](../../CVPR2026/autonomous_driving/oneocc_semantic_occupancy_prediction_for_legged_robots_with_a_single_panoramic_c.md)
- [\[ECCV 2024\] Reliability in Semantic Segmentation: Can We Use Synthetic Data?](reliability_in_semantic_segmentation_can_we_use_synthetic_data.md)
- [\[ECCV 2024\] Random Walk on Pixel Manifolds for Anomaly Segmentation of Complex Driving Scenes](random_walk_on_pixel_manifolds_for_anomaly_segmentation_of_complex_driving_scene.md)
- [\[ECCV 2024\] ItTakesTwo: Leveraging Peer Representations for Semi-supervised LiDAR Semantic Segmentation](ittakestwo_leveraging_peer_representations_for_semi-supervised_lidar_semantic_se.md)

</div>

<!-- RELATED:END -->
