---
title: >-
  [论文解读] PiTe: Pixel-Temporal Alignment for Large Video-Language Model
description: >-
  [ECCV 2024][视频理解][Large Video-Language Model] 提出 PiTe 模型，通过物体运动轨迹在像素级别实现视频与语言的时空对齐，构建 PiTe-143k 数据集，在零样本 QA、时序定位和密集描述任务上大幅超越现有方法。 领域现状：大语言模型（LLM）驱动了大视觉语言模型（LVLM）的发…
tags:
  - "ECCV 2024"
  - "视频理解"
  - "Large Video-Language Model"
  - "trajectory alignment"
  - "pixel-level"
  - "instruction tuning"
---

# PiTe: Pixel-Temporal Alignment for Large Video-Language Model

**会议**: ECCV 2024  
**arXiv**: [2409.07239](https://arxiv.org/abs/2409.07239)  
**代码**: [https://github.com/yliu-cs/PiTe](https://github.com/yliu-cs/PiTe)  
**领域**: 视频理解 / 视觉-语言  
**关键词**: Large Video-Language Model, trajectory alignment, pixel-level, instruction tuning, video understanding

## 一句话总结
提出 PiTe 模型，通过物体运动轨迹在像素级别实现视频与语言的时空对齐，构建 PiTe-143k 数据集，在零样本 QA、时序定位和密集描述任务上大幅超越现有方法。

## 研究背景与动机
**领域现状**：大语言模型（LLM）驱动了大视觉语言模型（LVLM）的发展，从图像扩展到视频理解成为热点。现有 LVidLM（如 VideoChat、Video-LLaMA、Video-ChatGPT）通过指令微调对齐视觉和语言特征。

**现有痛点**：传统的 QA 训练范式主要帮助 LLM 从空间角度理解视觉数据，难以有效捕捉时间动态和空间一致性关系。单纯依赖指令微调不足以实现全面的视频理解。

**核心矛盾**：视频包含复杂的时空数据结构，现有方案缺乏跨空间和时间维度的细粒度多模态对齐。

**本文目标**：如何在像素级别同时实现跨空间和时间维度的视频-语言细粒度对齐。

**切入角度**：利用物体运动轨迹（trajectory）作为视频和语言之间的桥梁，让模型预测文本中提及的物体在视频中的运动轨迹，从而学习细粒度的文本到像素的对齐。

**核心 idea**：通过轨迹引导的像素-时间对齐，让 LVidLM 在训练时预测每个物体的运动轨迹，实现空间和时间维度的细粒度对齐。

## 方法详解

### 整体框架
PiTe 由四个核心组件构成：(1) ViT 视觉编码器（CLIP ViT-L/14）提取帧特征；(2) 线性投影层（Visual Adapter）将视觉特征映射到 LLM 语义空间；(3) Vicuna v1.5 作为 LLM；(4) 定位投影器/轨迹投影器将 LLM 隐状态映射到坐标空间。模型采用三阶段训练策略逐步提升能力。

### 关键设计

1. **PiTe-143k 自动标注数据集**

    - 功能：构建包含物体运动轨迹的大规模视频-语言数据集
    - 核心思路：基于 InternVid-10M-FLT，通过两阶段自动标注管线生成。Stage 1 用 SuPar 提取名词短语，GLaMM 生成分割掩码；Stage 2 用 DOT 追踪点获得轨迹，k-means++ 聚类为 3 个关键点
    - 数据规模：143.64k 视频，343.93k 事件片段，1.02M 运动轨迹，总时长 2086.44 小时
    - 设计动机：现有视频指令数据集缺乏物体运动轨迹标注，无法支持像素级对齐研究

2. **三阶段训练策略**

    - 功能：逐步从图像定位 → 视频轨迹对齐 → 指令跟随
    - **Stage 1 — Referring Expression Localization**：
        - 使用 Localized Narratives 数据集训练视觉适配器
        - 在词汇映射层并行添加 MLP 定位投影器 $\varphi(\cdot)$，将语言特征映射为 2D 坐标：$p_i = \varphi(h_i)$
        - 损失：交叉熵 + L1 回归：$\mathcal{L}_1 = \frac{1}{\ell}\sum_{i=1}^{\ell}(\text{CE}(\text{LLM}(\mathbf{z}, \mathbf{w}_{1:i-1}), w_i) + \lambda|\hat{p}_i - p_i|)$
        - 使用 LoRA (r=64, α=128) 微调 LLM
    - **Stage 2 — Pixel-Temporal Alignment**：
        - 使用 PiTe-143k 数据集通过轨迹对齐视频和语言
        - 轨迹投影器 $\rho(\cdot)$ 输出 $P \times N$ 个 2D 坐标（P 个追踪点 × N 帧）：$\mathbf{p}_i = \rho(h_i)$
        - 损失：$\mathcal{L}_2 = \frac{1}{\ell}\sum_{i=1}^{\ell}(\text{CE} + \frac{\lambda}{P \cdot N}\sum_{j=1}^{P}\sum_{k=1}^{N}|\hat{p}_{ijk} - p_{ijk}|)$
        - 关键：用 Stage 1 的定位投影器权重初始化轨迹投影器，公式为 $\mathbf{m}_\varphi = \overbrace{\mathbf{m}_\rho \oplus \cdots \oplus \mathbf{m}_\rho}^{P \cdot N}$
    - **Stage 3 — Video QA Instruction Tuning**：
        - 使用 Valley + Video-ChatGPT 高质量对话数据微调
        - 仅用标准交叉熵自回归生成损失

3. **时间边界学习**

    - 功能：让模型学习事件的时间边界
    - 核心思路：在生成文本中结构化时间信息，以 "..., from s to e" 或 "From s to e, ..." 格式，s 和 e 为帧索引
    - 不存在轨迹的物体坐标统一设为 $(-1, -1)$ 表示缺失
    - 设计动机：增强模型对时间边界的感知能力

### 损失函数 / 训练策略
- 三个阶段分别使用不同损失：Stage 1 (CE + L1)、Stage 2 (CE + 轨迹 L1)、Stage 3 (仅 CE)
- 每阶段合并前一阶段 LoRA 权重并引入新 LoRA
- 训练配置：AdamW 优化器，lr=0.0001，cosine decay，BFloat16 精度
- 7B 模型单节点 8×A100 约 10 小时，13B 约 17 小时

## 实验关键数据

### 主实验 — 零样本视频问答

| 数据集 | 指标 | PiTe-7B | PiTe-13B | Video-ChatGPT | PG-Video-LLaVA | 提升(7B) |
|--------|------|---------|----------|---------------|-----------------|----------|
| MSVD-QA | Acc | 68.4 | 71.6 | 64.9 | 64.1 | +3.5 |
| MSRVTT-QA | Acc | 56.4 | 57.7 | 49.3 | 51.6 | +4.8 |
| ActivityNet-QA | Acc | 42.0 | 42.2 | 35.2 | 39.9 | +2.1 |

### 主实验 — 时序定位 & 密集描述 (ActivityNet)

| 任务 | 指标 | PiTe-7B | PiTe-13B | Video-ChatGPT |
|------|------|---------|----------|---------------|
| 时序定位 | R@0.3 | 30.4 | 37.2 | 26.4 |
| 时序定位 | R@0.5 | 17.8 | 23.7 | 13.6 |
| 时序定位 | mIoU | 22.0 | 26.0 | 18.9 |
| 密集描述 | CIDEr | 21.7 | 26.5 | 5.8 |
| 密集描述 | METEOR | 5.8 | 6.6 | 2.1 |

### 消融实验

| 配置 | MSVD Acc | R@0.3 | mIoU | CIDEr | 说明 |
|------|----------|-------|------|-------|------|
| PiTe (full) | 68.4 | 30.4 | 22.0 | 21.7 | 完整模型 |
| w/o 初始化策略 | 68.2 | 22.8 | 17.1 | 21.7 | 轨迹投影器不用定位投影器初始化 |
| w/o 轨迹对齐 | 68.1 | 23.9 | 17.4 | 21.4 | 去掉整个轨迹对齐阶段 |

### 关键发现
- 轨迹对齐对时序定位提升最大（mIoU 从 17.4 到 22.0），对 QA 提升有限
- 初始化策略至关重要，不使用初始化反而不如不做轨迹训练（不稳定参数阻碍时间感知）
- 追踪点数量 P=3 在多任务间表现最稳定
- 密集描述 CIDEr 从 5.8 飙升到 21.7，证明像素级对齐极大增强细粒度生成能力

## 亮点与洞察
- **数据集贡献突出**：PiTe-143k 填补了视频-语言数据集中缺乏物体轨迹标注的空白，自动标注管线可扩展。拥有 1.02M 个运动轨迹，规模很大。
- **投影器初始化是关键 trick**：用图像定位投影器权重重复拼接来初始化轨迹投影器，解决了从 2D 坐标到 $P \times N$ 维轨迹矩阵的维度扩展问题，这个设计直觉简单但效果显著。
- **时序定位和密集描述大幅领先**：相比同等 LLM 尺度的方法，时序定位 mIoU 提升 3.1，密集描述 CIDEr 提升 15.9，说明轨迹对齐对时序理解的增益远超对 QA 的增益。

## 局限与展望
- 仅采样 100 帧，对超长视频覆盖不足
- 轨迹标注管线依赖 GLaMM 和 DOT 的质量，小物体（如笔）检测困难时直接跳过
- 零样本评估，未与有监督方法对比时序定位
- 轨迹对齐对 QA 任务提升有限（~3.5%），说明 QA 可能更依赖高层语义而非像素级对齐

## 相关工作与启发
- **vs Video-ChatGPT**：PiTe 通过轨迹对齐在密集描述 CIDEr 上提升 15.9，说明细粒度对齐在生成任务上远优于纯指令微调
- **vs PixelLLM**：PixelLLM 在图像上用单词坐标连接模态，PiTe 将这一思路扩展到视频的时空维度

## 评分
- 新颖性: ⭐⭐⭐⭐ 轨迹对齐思想新颖，数据集构建管线完整
- 实验充分度: ⭐⭐⭐⭐ 三个任务六个数据集的零样本评估，消融分析到位
- 写作质量: ⭐⭐⭐ 符号体系略混乱，部分英文表达不通顺
- 价值: ⭐⭐⭐⭐ 数据集和对齐范式对后续工作有借鉴意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] VideoMamba: Spatio-Temporal Selective State Space Model](videomamba_spatio-temporal_selective_state_space_model.md)
- [\[CVPR 2026\] Towards Streaming Referring Video Segmentation via Large Language Model](../../CVPR2026/video_understanding/towards_streaming_referring_video_segmentation_via_large_language_model.md)
- [\[ECCV 2024\] Spatio-Temporal Proximity-Aware Dual-Path Model for Panoramic Activity Recognition](spatio-temporal_proximity-aware_dual-path_model_for_panoramic_activity_recogniti.md)
- [\[NeurIPS 2025\] Self-alignment of Large Video Language Models with Refined Regularized Preference Optimization](../../NeurIPS2025/video_understanding/self-alignment_of_large_video_language_models_with_refined_regularized_preferenc.md)
- [\[NeurIPS 2025\] Lattice Boltzmann Model for Learning Real-World Pixel Dynamicity](../../NeurIPS2025/video_understanding/lattice_boltzmann_model_for_learning_real-world_pixel_dynamicity.md)

</div>

<!-- RELATED:END -->
