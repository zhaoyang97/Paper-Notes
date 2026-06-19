---
title: >-
  [论文解读] Protecting NeRFs' Copyright via Plug-And-Play Watermarking Base Model
description: >-
  [ECCV 2024][3D视觉][NeRF版权保护] 提出 NeRFProtector，利用预训练的水印基础模型（message extractor）以即插即用方式在 NeRF 创建过程中同步嵌入二进制水印，通过渐进式全局渲染（PGR）将水印知识蒸馏到 NeRF 表示中，无需修改 NeRF 架构即可实现高比特精度的版权保护。
tags:
  - "ECCV 2024"
  - "3D视觉"
  - "NeRF版权保护"
  - "数字水印"
  - "即插即用"
  - "渐进式全局渲染"
  - "知识蒸馏"
---

# Protecting NeRFs' Copyright via Plug-And-Play Watermarking Base Model

**会议**: ECCV 2024  
**arXiv**: [2407.07735](https://arxiv.org/abs/2407.07735)  
**代码**: [https://qsong2001.github.io/NeRFProtector](https://qsong2001.github.io/NeRFProtector)  
**领域**: 3D视觉  
**关键词**: NeRF版权保护, 数字水印, 即插即用, 渐进式全局渲染, 知识蒸馏

## 一句话总结

提出 NeRFProtector，利用预训练的水印基础模型（message extractor）以即插即用方式在 NeRF 创建过程中同步嵌入二进制水印，通过渐进式全局渲染（PGR）将水印知识蒸馏到 NeRF 表示中，无需修改 NeRF 架构即可实现高比特精度的版权保护。

## 研究背景与动机

**领域现状**：NeRF 已成为 3D 场景表示的关键技术，随着其影响力扩大，保护 NeRF 模型的知识产权变得日益重要。现有方法如 CopyRNeRF 通过在 NeRF 模型中嵌入二进制水印来保护版权。

**现有痛点**：CopyRNeRF 存在两个显著缺陷：一是水印嵌入发生在 NeRF 创建完成之后的模型微调阶段，创建与嵌入之间存在时间窗口，恶意用户可能在此期间获取未保护的模型；二是 NeRF 创建者需要在嵌入水印时联合训练消息提取器，整个过程极其耗时复杂（约30小时），可能导致创作者放弃使用水印保护。

**核心矛盾**：版权保护的实用性与易用性之间的矛盾——现有方法要么保护不及时（创建后才嵌入），要么使用门槛过高（需修改 NeRF 架构、联合训练额外模块），导致创作者不愿采用。

**本文目标** (1) 如何在 NeRF 创建过程中同步完成水印嵌入，消除保护时间窗口；(2) 如何让水印方案兼容多种 NeRF 变体而无需架构修改；(3) 如何在保持渲染质量的同时实现高比特精度的水印提取。

**切入角度**：作者观察到传统 2D 图像水印框架（如 HiDDeN）中已有训练好的消息提取器，这些提取器已经学会了水印模式的知识。如果能将这些知识"蒸馏"到 NeRF 中，就无需修改 NeRF 本身的架构。

**核心 idea**：利用预训练的 2D 水印提取器作为即插即用的基础模型，通过渐进式全局渲染将水印知识蒸馏到 NeRF 表示中，实现创建与保护的同步完成。

## 方法详解

### 整体框架

NeRFProtector 包含三个阶段：(1) 构建水印基础模型——从 HiDDeN 框架获取预训练的消息提取器 $\mathcal{F}$；(2) 在 NeRF 创建过程中，固定基础模型权重，通过渐进式全局渲染（PGR）将水印知识蒸馏到 NeRF 表示中；(3) 创建完成后，使用同一基础模型从渲染图像中提取二进制水印进行版权声明。输入为 3D 场景多视角图像和待嵌入的二进制消息，输出为带水印的 NeRF 模型。

### 关键设计

1. **水印基础模型（Watermarking Base Model）**:

    - 功能：提供即插即用的水印嵌入和提取能力
    - 核心思路：采用 HiDDeN 框架，联合训练编码器 $\mathcal{E}$ 和提取器 $\mathcal{F}$。编码器将 48-bit 二进制消息嵌入覆盖图像生成水印图像，提取器从（可能经过干扰的）水印图像中恢复消息。训练完成后丢弃编码器，仅保留提取器作为基础模型。训练过程中加入随机变换层 $T$ 以增强对常见图像失真的鲁棒性
    - 设计动机：利用已有成熟的 2D 水印框架避免重新设计，且提取器已经学会了消息模式的知识，便于后续蒸馏到 NeRF 中

2. **渐进式全局渲染（Progressive Global Rendering, PGR）**:

    - 功能：替代 NeRF 的随机局部渲染，实现全局水印嵌入
    - 核心思路：标准 NeRF 训练时每次只随机渲染一小部分像素（局部渲染），导致水印模式只能嵌入到随机位置，无法形成有效的全局模式。PGR 在多个分辨率尺度上渲染所有像素，生成 $N_k=3$ 层级联视图 $\hat{I}_{set}$，其中每层分辨率为 $\frac{W}{2^n} \times \frac{H}{2^n}$。由于使用降低分辨率的全局渲染，计算成本可控
    - 设计动机：全局渲染确保消息模式深度整合到场景表示中，多尺度渲染利用了 3D 信息在不同 2D 投影分辨率下的不同特性，有助于消息蒸馏

3. **消息蒸馏（Message Distillation）**:

    - 功能：将基础模型中的水印知识转移到 NeRF 表示中
    - 核心思路：对 PGR 生成的多尺度渲染图像，用基础模型提取消息 $\hat{m}_{set} = \mathcal{F}(\hat{I}_{set})$，通过最小化提取消息与目标消息之间的 BCE 损失进行蒸馏：$\mathcal{L}_{dis} = \sum_{i=1}^{N_k} \alpha_i \cdot BCE(m, \hat{m}_i)$。同时用不可见性损失 $\mathcal{L}_{inv}$ 约束渲染质量
    - 设计动机：不修改 NeRF 的基本表示结构，仅通过渲染方案的改变实现知识转移，保持了即插即用的特性

### 损失函数 / 训练策略

总损失为三部分加权和：$\mathcal{L} = \lambda_1 \mathcal{L}_{local} + \lambda_2 \mathcal{L}_{inv} + \lambda_3 \mathcal{L}_{dis}$，其中 $\lambda_1=0.01$，$\lambda_3=0.001$。$\mathcal{L}_{local}$ 为标准 NeRF 重建损失，$\mathcal{L}_{inv}$ 为最高分辨率渲染与真实值的 MSE 损失，$\mathcal{L}_{dis}$ 为多尺度蒸馏损失。基础模型权重固定不更新。

## 实验关键数据

### 主实验

| 数据集 | 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | Bit Acc.(None) | Bit Acc.(Crop) | Bit Acc.(Resize) | Bit Acc.(JPEG) |
|--------|------|-------|-------|--------|----------------|----------------|------------------|----------------|
| Blender | NeRF w/o wm | 30.62 | 0.9579 | 0.0343 | N/A | N/A | N/A | N/A |
| Blender | CopyRNeRF | 25.50 | 0.9073 | 0.0885 | 62.15% | 56.63% | 57.32% | 58.41% |
| Blender | **NeRFProtector** | **29.26** | **0.9393** | **0.0483** | **92.69%** | **92.95%** | **91.87%** | **78.62%** |
| LLFF | NeRF w/o wm | 26.37 | 0.8352 | 0.1013 | N/A | N/A | N/A | N/A |
| LLFF | CopyRNeRF | 25.80 | 0.8302 | 0.1035 | 63.72% | 60.45% | 55.34% | 54.11% |
| LLFF | **NeRFProtector** | **26.82** | **0.8569** | **0.0834** | **96.99%** | **93.57%** | **80.53%** | **76.26%** |

### 消融实验

| 渲染策略 | PSNR↑ | SSIM↑ | LPIPS↓ | Bit Accuracy |
|---------|-------|-------|--------|-------------|
| Local rendering only | 30.38 | 0.9521 | 0.0360 | 45.99% |
| Single-scale global | 29.57 | 0.9402 | 0.0449 | 87.27% |
| Progressive (完整) | 29.26 | 0.9394 | 0.0483 | **92.69%** |

| NeRF变体 + 基础模型 | PSNR↑ | Bit Accuracy |
|-------------------|-------|-------------|
| Instant-NGP + HiDDeN | 32.92 | 91.96% |
| TensorRF + HiDDeN | 32.73 | 89.35% |
| Plenoxels + HiDDeN | 34.19 | 97.92% |
| Instant-NGP + MBRS | 31.71 | 89.13% |

### 关键发现

- PGR 是最关键的设计：从局部渲染的 45.99% 跳升到渐进式全局渲染的 92.69% 比特精度
- 方法兼容多种 NeRF 变体（Instant-NGP、TensorRF、Plenoxels）和多种基础模型（HiDDeN、MBRS），验证了即插即用特性
- 训练时间仅约 50 分钟，而 CopyRNeRF 需要约 30 小时，效率提升 36 倍
- 在常见图像失真（裁剪、缩放）下比特精度保持在 80%+ 以上

## 亮点与洞察

- **即插即用设计理念**：将水印能力封装为独立的基础模型，与 NeRF 架构解耦，这种模块化思路可迁移到其他 3D 表示（如 3D Gaussian Splatting）的版权保护中
- **渲染策略与水印嵌入的关联发现**：揭示了 NeRF 的随机局部渲染无法有效嵌入全局水印模式，这一观察具有启发性——任何依赖全局模式的任务都可能受益于全局渲染策略
- **通过知识蒸馏实现跨维度迁移**：将 2D 水印提取知识迁移到 3D 场景表示中，无需设计专门的 3D 水印方案，体现了降维解决问题的思路

## 局限与展望

- 白盒攻击威胁：如果攻击者获取了基础模型，可通过 PGD 攻击以极小失真移除水印，基础模型的保密性是安全前提
- 如果攻击者获取了原始训练图像，可通过无水印损失微调去除水印
- 仅在 48-bit 消息长度上验证，更长消息的嵌入能力未探索
- 未考虑 3D Gaussian Splatting 等更新的 3D 表示方法
- 版权保护需要超越技术方案的综合策略，包括法律框架支持

## 相关工作与启发

- **vs CopyRNeRF**: CopyRNeRF 在 NeRF 创建后通过微调嵌入水印，需联合训练提取器，耗时约 30 小时且存在时间窗口；NeRFProtector 在创建时同步嵌入，仅需 50 分钟，消除了安全窗口
- **vs StegaNeRF**: StegaNeRF 在 NeRF 中隐藏数据但需要修改结构；NeRFProtector 保持 NeRF 架构不变，具有更好的兼容性
- **vs HiDDeN/MBRS**: 这些 2D 水印方法直接处理图像后再训练 NeRF，但水印信息无法在 3D 渲染中保持一致（比特精度~50%），NeRFProtector 通过蒸馏实现了跨视角一致性

## 评分

- 新颖性: ⭐⭐⭐⭐ 即插即用的水印基础模型思路新颖，但核心组件（HiDDeN、蒸馏）已有
- 实验充分度: ⭐⭐⭐⭐ 消融实验、跨变体验证、攻击分析全面，但仅使用两个数据集
- 写作质量: ⭐⭐⭐⭐ 问题动机清晰，方法描述详细，图表设计合理
- 价值: ⭐⭐⭐⭐ 解决了 NeRF 版权保护的实用性问题，但适用场景较窄

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] SCFlow2: Plug-and-Play Object Pose Refiner with Shape-Constraint Scene Flow](../../CVPR2025/3d_vision/scflow2_plug-and-play_object_pose_refiner_with_shape-constraint_scene_flow.md)
- [\[AAAI 2026\] Can Protective Watermarking Safeguard the Copyright of 3D Gaussian Splatting?](../../AAAI2026/3d_vision/can_protective_watermarking_safeguard_the_copyright_of_3d_gaussian_splatting.md)
- [\[ECCV 2024\] DATENeRF: Depth-Aware Text-based Editing of NeRFs](datenerf_depth-aware_text-based_editing_of_nerfs.md)
- [\[ECCV 2024\] ShapeFusion: A 3D Diffusion Model for Localized Shape Editing](shapefusion_a_3d_diffusion_model_for_localized_shape_editing.md)
- [\[CVPR 2026\] Plug-and-Play PDE Optimization for 3D Gaussian Splatting: Toward High-Quality Rendering and Reconstruction](../../CVPR2026/3d_vision/plug-and-play_pde_optimization_for_3d_gaussian_splatting_toward_high-quality_ren.md)

</div>

<!-- RELATED:END -->
