---
title: >-
  [论文解读] GeoRemover: Removing Objects and Their Causal Visual Artifacts
description: >-
  [NeurIPS2025 Spotlight][图像生成][目标移除] 提出几何感知的两阶段框架 GeoRemover，将目标移除解耦为几何移除（深度域）与外观渲染（RGB域），通过修改场景几何表示来隐式消除被移除物体的阴影和反射等因果视觉伪影。 目标移除（Object Removal）是图像编辑的核心任务之一…
tags:
  - "NeurIPS2025 Spotlight"
  - "图像生成"
  - "目标移除"
  - "因果视觉伪影"
  - "几何感知"
  - "深度图"
  - "扩散模型"
  - "DPO"
---

# GeoRemover: Removing Objects and Their Causal Visual Artifacts

**会议**: NeurIPS2025 Spotlight  
**arXiv**: [2509.18538](https://arxiv.org/abs/2509.18538)  
**代码**: [项目页面](https://buxiangzhiren.github.io/GeoRemover)  
**领域**: 图像生成  
**关键词**: 目标移除, 因果视觉伪影, 几何感知, 深度图, 扩散模型, DPO  

## 一句话总结
提出几何感知的两阶段框架 GeoRemover，将目标移除解耦为几何移除（深度域）与外观渲染（RGB域），通过修改场景几何表示来隐式消除被移除物体的阴影和反射等因果视觉伪影。

## 背景与动机

目标移除（Object Removal）是图像编辑的核心任务之一。现有方法面临一个关键难题：当移除一个物体时，该物体在场景中产生的**因果视觉伪影**（如阴影、反射）往往残留在图像中，导致编辑结果不自然。

现有方法可分为两类：

1. **严格掩码对齐方法**（Strictly Mask-Aligned）：只编辑用户标注的掩码区域，无法处理掩码外的阴影/反射伪影。用户若要消除这些伪影，需手动标注所有伪影区域，既不可扩展也不友好。
2. **宽松掩码对齐方法**（Loosely Mask-Aligned）：允许模型修改掩码外区域以推断和消除伪影，但由于缺乏明确的编辑边界指导，容易出现**过度擦除**问题——误删周围不相关的物体。

本文的关键观察：阴影和反射等视觉伪影是由物体的**几何存在**（Geometry Presence）在特定光照条件下产生的因果结果。如果从场景几何中移除物体的存在，其关联的光照效应自然不复存在。

## 核心问题

如何在移除目标物体的同时，自动消除其产生的阴影和反射等因果视觉伪影，且不对未标注区域造成不可控的修改？

## 方法详解

### 整体框架：几何-外观解耦

GeoRemover 将目标移除解耦为两个子任务：

$$x_0^- = \mathcal{D}(I^-), \quad x_0^+ = s_\theta(x_0^-, M) \quad \text{(Stage 1: 几何移除)}$$

$$I^+ = \mathcal{G}(I^-, x_0^-, x_0^+) \quad \text{(Stage 2: 外观渲染)}$$

其中 $\mathcal{D}$ 为深度估计器（Depth Anything），$s_\theta$ 为几何移除扩散模型，$\mathcal{G}$ 为外观渲染扩散模型。

### Stage 1：几何移除

**核心思路**：在深度域中移除物体。由于阴影和反射在深度图中不可见，深度域天然适合使用严格掩码对齐训练。

- 输入：RGB 图像的深度估计 $x_0$ + 物体掩码 $M$
- 目标：预测编辑后的深度图 $\hat{x}_0$，在掩码区域移除物体几何，掩码外保持不变
- 骨干网络：FLUX.1-Fill-dev + LoRA（rank=64）微调

**偏好引导的几何补全（DPO）**：

直接训练扩散模型做深度补全时，模型常在掩码区域**幻觉出新结构**（如凭空插入不存在的物体几何）。为解决此问题，引入 DPO 策略：

- **正样本** $x_0^+$：物体被成功移除，掩码区域深度平滑、深度流（depth flow）趋近零
- **负样本** $x_0^-$：物体仍然存在，掩码区域存在深度不连续性

深度流定义为空间梯度：

$$F_{ij}(x) = \{|d_{i+1,j} - d_{i,j}|, \; |d_{i,j+1} - d_{i,j}|\}$$

流损失衡量预测流与真实流的差异：

$$\mathcal{L}_{\text{flow}}(\hat{x}_0, x_0) = \frac{1}{|\Omega|}\sum_{(i,j) \in \Omega} \|F_{ij}(\hat{x}_0) - F_{ij}(x_0)\|_1$$

采用 Bradley-Terry 模型建模偏好概率，最终损失为：

$$\mathcal{L} = \mathcal{L}_{\text{DSM}} + \lambda \mathcal{L}_{\text{BT}}$$

### Stage 2：外观渲染

**核心思路**：根据几何变化条件，将带物体的图像翻译为无物体的图像。

- 输入：遮蔽的 RGB 图像 + 编辑前深度图 $x_0^-$ + 编辑后深度图 $x_0^+$
- 通过两张深度图的差异，模型定位被移除的物体位置，并分析配对图像中的视觉差异来学习物体与伪影的因果关系

**双向训练**：同时训练移除方向和插入方向，增强模型对几何-外观对应关系的理解：

$$I^+ = \mathcal{G}(I^- \mid x_0^-, x_0^+), \quad I^- = \mathcal{G}(I^+ \mid x_0^+, x_0^-)$$

实现方式是将 RGB 图像与两张彩色深度图沿宽度维度拼接为 $H \times 3W \times 3$ 的张量，直接输入扩散模型。

### 失败情况处理：Local Max Depth Fill-in

当运动模糊、透明/反射表面导致深度估计不可靠时，编辑前后深度差异不足以触发 Stage 2 的移除。此时采用**局部最大深度填充**：对掩码内缺乏可靠估计的像素，从 10×10 局部邻域传播最大深度值，恢复边界对比度。

## 实验关键数据

### 主实验（RemovalBench & RORD-Val）

| 方法 | FID↓ | CMMD↓ | LPIPS↓ | PSNR↑ |
|------|------|-------|--------|-------|
| LaMa | 99.88 | 0.351 | 0.156 | 18.72 |
| Attentive-Eraser | 55.49 | 0.232 | 0.146 | 20.60 |
| OmniEraser | 39.52 | 0.208 | 0.133 | 21.11 |
| **GeoRemover** | **29.88** | **0.089** | **0.124** | **25.52** |

在 RemovalBench 上，GeoRemover 的 FID 比 OmniEraser 降低 24.4%，PSNR 提升 4.41 dB。

### 消融实验（RORD-Val）

| 方法 | FID↓ | PSNR↑ | Insert.↓ |
|------|------|-------|----------|
| One-Stage | 56.24 | 17.52 | 2.81% |
| Two-Stage w/o DPO | 34.24 | 22.81 | 5.09% |
| Two-Stage w/ DPO | **31.15** | **23.70** | **1.48%** |

DPO 将结构幻觉插入率从 5.09% 降至 1.48%。

### 因果伪影移除（CausRem）

| 方法 | IoU% ↑ |
|------|--------|
| OmniEraser | 68.29 |
| **GeoRemover** | **73.76** |

### 几何移除精度（MAE）

| 方法 | MAE↓ |
|------|------|
| 输入深度 | 0.0827 |
| Two-Stage w/o DPO | 0.0490 |
| Two-Stage w/ DPO | **0.0387** |

## 亮点

1. **因果推理视角**：将目标移除重新思考为因果推理过程——几何存在是因，视觉伪影是果，通过因的移除自然消除果
2. **几何-外观解耦设计**：在深度域做严格掩码对齐的几何编辑（避免宽松对齐的不可控性），在 RGB 域做隐式伪影消除（避免严格对齐的能力不足）
3. **DPO 用于几何补全**：创新性地将 DPO 应用于深度图补全，利用深度流平滑度构造正负样本对，有效抑制了扩散模型的结构幻觉
4. **双向渲染训练**：通过同时训练移除和插入两个方向，增强了模型对几何变化与外观变化的对应关系理解
5. **实验全面领先**：在 RemovalBench、RORD-Val、CausRem 三个基准上全面超越 SOTA

## 局限与展望

1. **深度估计失效场景**：运动模糊、高透明度、镜面反射等场景导致深度估计不可靠，需要额外的 Fill-in 策略作为补救
2. **两阶段计算成本**：相比单阶段方法需要两次扩散推理，Stage 1 训练约 24h，Stage 2 约 60h（8×H100）
3. **自发光物体**：对彩色灯泡等自发光物体，Stage 2 可能生成弥漫辉光而非干净移除
4. **不完整掩码**：掩码未完全覆盖物体时，Stage 1 可能幻觉出完整物体而非移除
5. **反射残留**：半透明物体在反射表面上的颜色渗透仍可能残留
6. **训练数据依赖**：仅使用室内场景的 RORD 数据集训练，户外场景泛化能力有待验证

## 与相关工作的对比

| 方法 | 类型 | 伪影处理 | 可控性 | 核心差异 |
|------|------|---------|--------|---------|
| CLIPAway | 严格对齐 | ✗ | ✓ | CLIP 引导，但无法处理掩码外伪影 |
| Attentive-Eraser | 严格对齐 | 部分 | ✓ | 注意力重定向，但伪影处理有限 |
| OmniEraser | 宽松对齐 | ✓ | 部分 | 视频帧配对数据，但可控性受限 |
| ObjectDrop | 宽松对齐 | ✓ | 部分 | 反事实自举，代码未公开 |
| **GeoRemover** | 几何解耦 | ✓ | ✓ | 几何域严格对齐 + RGB 域隐式伪影消除 |

GeoRemover 的核心优势在于融合了两类方法的长处：在几何域享有严格对齐的精确可控性，在外观域通过几何引导实现隐式伪影消除。

## 启发与关联

1. **几何作为中间表示的普适价值**：深度图作为"不含伪影"的场景表示，为各种光照相关编辑任务（如重光照、场景合成）提供了有价值的中间表示思路
2. **DPO 在低级视觉中的应用**：将 RLHF/DPO 从语言模型和图像生成扩展到几何补全的低级视觉任务，启发更多传统 CV 任务可以引入偏好学习
3. **因果分析驱动的方法设计**：不直接学端到端映射，而是分析任务中的因果链（几何→光照→伪影），据此设计模块化管线，值得在其他编辑任务中借鉴
4. **Watermark 移除的意外扩展**：通过注入伪深度线索（Local Max Depth Fill-in），框架可泛化到非几何类的修复任务（如水印移除），暗示该方法论比预想的更通用

## 评分
- 新颖性: 8/10 — 因果推理视角和几何-外观解耦设计新颖，DPO 用于几何补全有创新性
- 实验充分度: 8/10 — 三个基准、多指标、完善的消融实验和失败案例分析
- 写作质量: 8/10 — 动机阐述清晰，方法逻辑链完整，图例直观
- 价值: 8/10 — 为目标移除提供了新范式，几何解耦思路对其他编辑任务有启发

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] DeCaFlow: A Deconfounding Causal Generative Model](decaflow_a_deconfounding_causal_generative_model.md)
- [\[CVPR 2025\] Temporal Score Analysis for Understanding and Correcting Diffusion Artifacts](../../CVPR2025/image_generation/temporal_score_analysis_for_understanding_and_correcting_diffusion_artifacts.md)
- [\[NeurIPS 2025\] Coupling Generative Modeling and an Autoencoder with the Causal Bridge](coupling_generative_modeling_and_an_autoencoder_with_the_causal_bridge.md)
- [\[NeurIPS 2025\] Non-Markovian Discrete Diffusion with Causal Language Models](non-markovian_discrete_diffusion_with_causal_language_models.md)
- [\[CVPR 2025\] Make It Count: Text-to-Image Generation with an Accurate Number of Objects](../../CVPR2025/image_generation/make_it_count_text-to-image_generation_with_an_accurate_number_of_objects.md)

</div>

<!-- RELATED:END -->
