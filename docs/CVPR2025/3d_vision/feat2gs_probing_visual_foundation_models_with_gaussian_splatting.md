---
title: >-
  [论文解读] Feat2GS: Probing Visual Foundation Models with Gaussian Splatting
description: >-
  [CVPR 2025][3D视觉][视觉基础模型] 本文提出 Feat2GS，一个统一框架，通过将视觉基础模型（VFM）的 2D 特征经轻量级 MLP 读出为 3D 高斯属性，在新视角合成任务上分别探测 VFM 的几何感知和纹理感知能力，无需 3D 真值数据即可在大规模多样数据集上全面评测 10+ 种 VFM 的 3D 意识。
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "视觉基础模型"
  - "3D感知"
  - "高斯溅射"
  - "新视角合成"
  - "探针评测"
---

# Feat2GS: Probing Visual Foundation Models with Gaussian Splatting

**会议**: CVPR 2025  
**arXiv**: [2412.09606](https://arxiv.org/abs/2412.09606)  
**代码**: [https://github.com/fanegg/Feat2GS](https://github.com/fanegg/Feat2GS)  
**领域**: 3D视觉  
**关键词**: 视觉基础模型, 3D感知, 高斯溅射, 新视角合成, 探针评测

## 一句话总结
本文提出 Feat2GS，一个统一框架，通过将视觉基础模型（VFM）的 2D 特征经轻量级 MLP 读出为 3D 高斯属性，在新视角合成任务上分别探测 VFM 的几何感知和纹理感知能力，无需 3D 真值数据即可在大规模多样数据集上全面评测 10+ 种 VFM 的 3D 意识。

## 研究背景与动机

**领域现状**：视觉基础模型（如 DINOv2、CLIP、DUSt3R、SAM、RADIO 等）在大量 2D 数据上训练，被广泛用于 3D 相关任务的特征提取。这些模型在架构（ViT、UNet）、训练策略（对比学习、自蒸馏、点图回归、去噪）和训练数据（2D vs 3D）上差异巨大。

**现有痛点**：现有 3D 探测方法主要有两类：(1) 单视图 2.5D 估计（深度/法线），(2) 双视图稀疏 2D 对应（匹配/追踪）。这两者都有严重限制：忽略了纹理感知能力的评测，且需要 3D 真值标注，严重限制了评测数据的规模和多样性。更关键的是，不同 VFM 之间缺乏统一公平的对比框架。

**核心矛盾**：要全面评测 VFM 的 3D 感知能力，需要同时覆盖几何和纹理、密集而非稀疏的评测方式、以及大规模多样化的数据——但现有方法无法同时满足这三个要求。

**本文目标**：设计一个统一的 3D 感知探测框架，能够 (1) 分别评测几何和纹理感知，(2) 使用密集像素级评测，(3) 仅需 2D 多视角图像，(4) 支持任意拍摄的稀疏视角图像。

**切入角度**：3D 高斯溅射（3DGS）的参数天然分为几何（位置 $\mathbf{x}$、不透明度 $\alpha$、协方差 $\Sigma$）和纹理（球谐系数 $\mathbf{c}$）两组，这一解耦特性可以直接用来分别探测 VFM 的几何和纹理感知。

**核心 idea**：用轻量 MLP 从 VFM 特征读出 3DGS 属性，通过新视角合成质量（PSNR/SSIM/LPIPS）作为 3D 感知的代理指标，在 7 个多样化数据集上全面评测 10 种 VFM。

## 方法详解

### 整体框架
Feat2GS 的流程：(1) 输入未标定的稀疏多视角图像；(2) 用预训练 VFM 提取每张图像的特征图，PCA 降维到 256 通道，双线性上采样到 512 分辨率；(3) 用 DUSt3R 初始化相机位姿和点云；(4) 2 层 MLP 读出层从 VFM 特征回归 3DGS 属性；(5) 可微分光栅化渲染，用光度损失优化读出层和相机参数；(6) 在未见测试视角上计算 NVS 质量指标。

### 关键设计

1. **GTA 三模式探测（Geometry/Texture/All Probing）**:

    - 功能：分离评测 VFM 的几何感知和纹理感知能力
    - 核心思路：三种模式——**Geometry 模式**：MLP 从 VFM 特征读出几何参数 $\{x_i, \alpha_i, \Sigma_i\} = g_\Theta^{(G)}(f_i)$，纹理参数自由优化；**Texture 模式**：MLP 读出纹理参数 $\{c_i\} = g_\Theta^{(T)}(f_i)$，几何参数自由优化；**All 模式**：MLP 同时读出所有参数。通过对比三种模式的 NVS 质量，可以精确定位 VFM 在几何和纹理上的优劣
    - 设计动机：这是本文最核心的贡献——利用 3DGS 参数的天然解耦来实现对两种 3D 感知能力的独立评测

2. **轻量级读出层设计**:

    - 功能：从 VFM 特征解码 3DGS 属性，同时防止过拟合
    - 核心思路：仅使用 2 层 MLP（256 维隐层，ReLU 激活）。故意限制网络容量，确保 3DGS 参数确实是从 VFM 特征中"读出"的，而非由网络自身学习。这与 InstantSplat 的自由优化形成鲜明对比——后者优化百万级参数容易在稀疏视角下过拟合
    - 设计动机：保证探测的公平性——不同 VFM 使用完全相同的读出架构和训练配置

3. **暖启动策略（Warm Start）**:

    - 功能：避免直接从 2D 特征解码 3D 结构时陷入局部最优
    - 核心思路：先用 DUSt3R 重建的点云作为目标，通过 $\min_\Theta \|g_\Theta(f) - G_{init}\|$ 预训练读出层 1K 轮，再切换到光度损失继续优化 7K 轮。这为所有 VFM 提供了一致的初始化条件
    - 设计动机：稀疏视角下直接优化极易失败，暖启动确保不同 VFM 特征都能收敛到合理解

### 损失函数 / 训练策略
优化目标为光度损失 $\min_{\Theta,T} \|\mathcal{R}(g_\Theta(f), T) - \mathcal{I}\|$，同时优化 MLP 参数、3DGS 参数和相机参数。使用 Adam 优化器，MLP 学习率从 $10^{-2}$ 衰减到 $10^{-4}$，不使用自适应密度控制。所有实验在单块 RTX 4090 上进行。

## 实验关键数据

### 主实验（跨 7 个数据集的平均 NVS 指标，Geometry 模式）

| VFM | 训练策略 | 训练数据 | Avg PSNR↑ | Avg SSIM↑ | Avg LPIPS↓ |
|-----|---------|---------|-----------|-----------|------------|
| RADIO | 多教师蒸馏 | DataComp-1B | **最高** | **最高** | **最低** |
| MASt3R | 点图回归 | 3D混合 | 第2 | 第2 | 第2 |
| DUSt3R | 点图回归 | 3D混合 | 第3 | 第3 | 第3 |
| DINO | 自蒸馏 | ImageNet-1K | 第4 | 第4 | 中等 |
| DINOv2 | 自蒸馏 | LVD-142M | 中等 | 中等 | 中等 |
| CLIP | 对比VLM | WIT-400M | 中等 | 中等 | 中等 |
| SD | 去噪VLM | LAION | 最低 | 最低 | 最高 |

### 消融/应用实验（NVS 基线对比，所有数据集平均）

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| InstantSplat (SOTA) | 18.87 | 0.6044 | 0.3039 |
| Feat2GS w/ RADIO | 19.73 | 0.6513 | 0.3143 |
| Feat2GS w/ concat all | 19.80 | 0.6545 | 0.3105 |
| Feat2GS w/ DUSt3R | 19.66 | 0.6469 | 0.3247 |
| Feat2GS w/ DUSt3R* (微调) | **19.75** | **0.6561** | **0.2928** |

### 关键发现
- **几何感知 Top4**：RADIO > MASt3R > DUSt3R > DINO。3D 数据训练（点图回归）对几何感知至关重要，深度回归（MiDaS）远不如点图回归（DUSt3R）
- **纹理感知 Top3**：MAE > SAM > MASt3R。掩码图像重建预训练有助于保留纹理信息，而 RADIO 虽然几何最强但纹理最差——蒸馏了 DINO/CLIP 的纹理不变性
- **All 模式被纹理拖累**：LPIPS 平均恶化 +0.05，All 模式的模糊渲染源自 VFM 纹理感知不足
- **特征拼接有效**：简单拼接 DINOv2+CLIP+SAM 的几何结果与蒸馏它们的 RADIO 相当；拼接最佳几何特征和最佳纹理特征（RADIO+MAE+IUVRGB）可超越最佳单一 VFM
- **2D 指标与 3D 指标高度相关**：在 DTU 数据集上，NVS 质量与点云精度/完整度呈强相关，验证了 NVS 作为 3D 评测代理的合理性

## 亮点与洞察
- GTA 三模式探测是极其巧妙的设计——利用 3DGS 参数的天然解耦，无需额外设计就能独立评测几何和纹理的 3D 感知。这种"用已有工具的特性来回答研究问题"的思路值得学习
- "VFM 纹理感知普遍较差"是重要的发现——大量 VFM 为了语义理解或几何鲁棒性而牺牲了纹理信息，这限制了它们在光度一致性任务中的应用
- 简单的特征拼接就能超越精心蒸馏的 RADIO，说明不同 VFM 的特征在 3D 任务中有互补性
- 暗示了未来 3D VFM 的设计方向：在规范空间中预测 3D 高斯 + 光度损失训练

## 局限与展望
- 作者承认的局限：(1) 依赖 DUSt3R 初始化位姿和点云，初始化失败时 Feat2GS 也会失败；(2) 假设短时间内拍摄的静态场景，无法处理长时间跨度的网络照片集；(3) 局限于静态场景，不支持动态视频
- 自己发现的局限：VFM 特征分辨率低（通常 1/16），限制了高频细节的恢复；暖启动使用 DUSt3R 点云可能引入偏差，有利于 DUSt3R 系列特征
- 改进思路：探索 VFM 特征上采样器、4D 高斯溅射支持动态场景、利用 VFM 特征估计初始位姿以消除对 DUSt3R 的依赖

## 相关工作与启发
- **vs Probe3D**: Probe3D 用 2.5D 估计（深度/法线）探测 VFM，但忽略纹理维度且需要 3D 真值。Feat2GS 通过 NVS 同时覆盖几何和纹理，且不需要 3D 标注
- **vs InstantSplat**: 两者都用 DUSt3R 初始化做稀疏视角 NVS，但 InstantSplat 自由优化 3DGS 容易过拟合，Feat2GS 通过 VFM 特征约束读出有效防止过拟合
- **vs 3D 特征场（LERF, N3F等）**: 这些工作假设 VFM 特征具有 3D 一致性，Feat2GS 则质疑并验证这一假设

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 利用 3DGS 参数解耦来探测 VFM 3D 感知是高度原创的问题设计
- 实验充分度: ⭐⭐⭐⭐⭐ 10 种 VFM、7 个数据集、3 种探测模式、DTU 上 3D 指标验证，极为全面
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑清晰，发现有深度，可视化精美
- 价值: ⭐⭐⭐⭐⭐ 对 VFM 社区和 3D 视觉社区都有重要指导意义，发现具有广泛影响力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] UniPre3D: Unified Pre-training of 3D Point Cloud Models with Cross-Modal Gaussian Splatting](unipre3d_unified_pre-training_of_3d_point_cloud_models_with_cross-modal_gaussian.md)
- [\[CVPR 2025\] Gaussian Splatting Feature Fields for Privacy-Preserving Visual Localization](gaussian_splatting_feature_fields_for_privacy-preserving_visual_localization.md)
- [\[CVPR 2025\] Perception Tokens Enhance Visual Reasoning in Multimodal Language Models](perception_tokens_enhance_visual_reasoning_in_multimodal_language_models.md)
- [\[CVPR 2026\] AVA-Bench: Atomic Visual Ability Benchmark for Vision Foundation Models](../../CVPR2026/3d_vision/ava-bench_atomic_visual_ability_benchmark_for_vision_foundation_models.md)
- [\[CVPR 2025\] Geometry Field Splatting with Gaussian Surfels](geometry_field_splatting_with_gaussian_surfels.md)

</div>

<!-- RELATED:END -->
