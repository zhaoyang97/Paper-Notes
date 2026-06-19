---
title: >-
  [论文解读] FlexGen: Flexible Multi-View Generation from Text and Image Inputs
description: >-
  [ICCV 2025][3D视觉][多视角生成] 本文提出 FlexGen，一个灵活的多视角图像生成框架，通过 GPT-4V 生成 3D-aware 文本标注并设计自适应双控制模块，支持单图、文本或二者联合控制生成一致的多视角图像，实现未可见区域补全、材质编辑和纹理控制等多种可控能力。 多视角扩散模型（如 Zero123++…
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "多视角生成"
  - "3D-aware文本标注"
  - "可控生成"
  - "扩散模型"
  - "材质编辑"
---

# FlexGen: Flexible Multi-View Generation from Text and Image Inputs

**会议**: ICCV 2025  
**arXiv**: [2410.10745](https://arxiv.org/abs/2410.10745)  
**代码**: [https://xxu068.github.io/flexgen.github.io/](https://xxu068.github.io/flexgen.github.io/) (有项目页)  
**领域**: 3D视觉  
**关键词**: 多视角生成, 3D-aware文本标注, 可控生成, 扩散模型, 材质编辑

## 一句话总结

本文提出 FlexGen，一个灵活的多视角图像生成框架，通过 GPT-4V 生成 3D-aware 文本标注并设计自适应双控制模块，支持单图、文本或二者联合控制生成一致的多视角图像，实现未可见区域补全、材质编辑和纹理控制等多种可控能力。

## 研究背景与动机

多视角扩散模型（如 Zero123++、SyncDreamer、Wonder3D）已展现出利用预训练 2D 扩散模型生成 3D 一致多视角图像的潜力，为快速 3D 内容创建提供了可行路径。然而，**可控生成**在多视角扩散模型中仍然严重不足。

**现有方法的局限性**：
- *单视角条件不足*：大多数方法仅以单张图像为条件，对应物体未可见区域（如背面）只能简单复制正面信息，缺乏 3D-aware 的引导信号
- *3D 引导不友好*：Coin3D 使用基本形状作 3D 引导、Clay 使用稀疏点云和 3D 包围盒，这些对普通用户不友好
- *文本标注缺乏 3D 信息*：Cap3D 利用 BLIP-2 为每个渲染视角单独生成描述再由 GPT-4 汇总，但结果往往是高层概括，缺少局部细节和 3D 空间关系。原因有二：BLIP-2 仅生成全局描述；单视角信息冗余且不完整
- *单模态控制*：Instant3D 虽使用文本生成多视角但仅支持 text-to-3D，不够灵活

**核心矛盾**：文本是最自然的用户控制方式，能提供丰富的语义和空间关系信息，但如何为 3D 物体生成包含充分 3D-aware 信息的文本标注、如何在多视角扩散模型中有效融合图像和文本两种模态的控制信号，仍是未解问题。

**切入角度**：(1) 利用 GPT-4V 的强视觉推理能力，从四个正交视角的拼接图中生成全局-局部的 3D-aware 文本标注；(2) 设计自适应双控制模块实现图像和文本的联合控制，通过条件切换器支持单图、纯文本、或图文联合三种模式。

## 方法详解

### 整体框架

FlexGen 基于 Stable Diffusion 2.1 构建，接受单视角图像和/或文本提示作为输入，生成 2×2 布局的四个正交视角图像（前、左、后、右，512×512，固定仰角 5°）。核心包含三部分：3D-aware 文本标注生成、自适应双控制模块、灵活的训练与推理策略。

### 关键设计

1. **3D-Aware 文本标注生成（3D-Aware Caption Annotation）**

    - 功能：为 Objaverse 数据集中的 3D 物体生成富含 3D 空间关系信息的全局-局部文本描述
    - 核心思路：数据集构建分三步：
        - **渲染**：每个 3D 物体渲染为四个正交视角（前/左/后/右），512×512，拼成 2×2 的 tiled 图像
        - **标注**：将 tiled 图像输入 GPT-4V，利用其跨视角推理能力同时生成全局描述（整体属性 + 部件间 3D 空间关系）和局部描述（各部件颜色、姿态、纹理等）
        - **合并**：将全局和局部描述融合为"global-local text description"。训练时随机选取部分局部描述以模拟用户行为
    - 此外添加材质描述（metallic、roughness 等），使用 Blender 渲染时的实际材质参数作为标注
    - 设计动机：相比 Cap3D 的"每视角独立标注再汇总"，GPT-4V 同时观察四个正交视角能推理出 3D 空间关系（如"左侧有一个把手，而右侧没有"），标注质量显著提升

2. **自适应双控制模块（Adaptive Dual-Control Module）**

    - 功能：在扩散模型的去噪过程中同时融合图像和文本控制信号
    - 核心思路：
        - 基于 Reference Attention 机制——在额外参考图像上运行去噪 UNet，将其 self-attention 的 key/value 矩阵追加到目标分支的对应注意力层
        - 创新之处：在 reference attention 中注入文本信息。用户文本通过 CLIP 编码器获得 per-token 嵌入 $E \in \mathbb{R}^{L \times D}$，通过 cross-attention 与参考图像特征充分交互
        - 交互完成后，将双控制模块的 self-attention key/value 追加到去噪 UNet 的对应层
    - 设计动机：单模态控制（纯图像或纯文本）无法同时兼顾保真度和可控性。Reference attention 提供图像保真度，cross-attention 注入文本提供语义控制，二者在注意力层级融合

3. **条件切换器与灵活训练策略（Condition Switcher）**

    - 功能：支持图文联合、仅图像、仅文本三种推理模式
    - 核心思路：训练时以可配置概率随机丢弃输入条件：
        - 图文联合概率：0.3
        - 仅图像概率：0.3
        - 仅文本概率：0.3
        - 两者都无概率：0.1
        - 文本缺失时用空字符串，图像缺失时用黑色图像替代
    - 设计动机：通过 dropout 式的条件随机丢弃训练，使模型在推理时灵活适应不同输入场景（用户可能只有图片、只有文字、或两者兼有）

### 损失函数 / 训练策略

- 使用标准扩散模型去噪损失训练
- 基于 SD 2.1，8 张 A800 80GB GPU 训练 10 天，180K 迭代，batch size 32
- Adam 优化器，学习率 1e-5
- 推理使用 DDIM 75 步采样
- 训练数据：从 Objaverse 精选 147K 高质量（有纹理贴图、多边形数足够）3D 物体
- 每个物体渲染 24 张目标视角图（仰角 5°，方位均匀分布），输入视角随机采样（仰角 -30°~30°）

## 实验关键数据

### 主实验

GSO 数据集上的新视角合成与稀疏视角 3D 重建：

| 方法 | PSNR↑ | LPIPS↓ | CD↓ | FS@0.1↑ |
|------|-------|--------|-----|---------|
| SyncDreamer | 17.66 | 0.21 | 0.126 | 0.833 |
| Era3D | 18.52 | 0.19 | 0.245 | 0.713 |
| Zero123++ | 18.83 | 0.16 | 0.087 | 0.910 |
| Ours (w/o caption) | 21.12 | 0.14 | 0.078 | 0.921 |
| **FlexGen (Ours)** | **22.31** | **0.12** | **0.076** | **0.928** |

Text-to-multi-view 对比（GSO 300 样本）：

| 方法 | FID↓ | IS↑ | CLIP↑ |
|------|------|-----|-------|
| MVDream | 44.42 | 12.98±1.22 | 0.79 |
| **FlexGen (Ours)** | **35.56** | **13.41±0.87** | **0.83** |
| Ground truth | N/A | 13.81±1.40 | 0.89 |

### 消融实验

| 配置 | PSNR | LPIPS | 说明 |
|------|------|-------|------|
| Ours (w/o caption) | 21.12 | 0.14 | 无文本标注，仅图像控制 |
| Ours (Cap3D caption) | ~20.5 | ~0.15 | 使用 Cap3D 标注，缺乏 3D-aware 信息 |
| **Ours (full)** | **22.31** | **0.12** | GPT-4V 3D-aware 标注 + 双控制模块 |

### 关键发现

- 加入 3D-aware 文本标注后 PSNR 从 21.12 提升到 22.31（+1.19），证明文本控制对补全未可见区域有显著帮助
- FlexGen 的 FID 和 CLIP score 接近 ground truth 水平（FID 35.56 vs 参考 N/A，CLIP 0.83 vs 0.89），远优于 MVDream（FID 44.42, CLIP 0.79）
- 图文联合控制生成的多视角图像用于 3D 重建时，CD 和 FS 指标均优于仅图像条件的方法
- 通过修改文本提示中的材质描述（如"high metallic, low roughness"），可直接控制生成图像的材质属性

## 亮点与洞察

- GPT-4V 同时观察四个正交视角的拼接图来生成 3D-aware 标注，是一个巧妙的利用大模型视觉推理能力的方案
- 自适应双控制模块的设计允许图像和文本信息在注意力层充分交互，优于简单拼接的方式
- Condition switcher 的训练策略使单一模型同时支持三种推理模式，提升了实用性
- 材质可控生成（metallic/roughness）是一个有价值的创新点，对 3D 资产创建很有用

## 局限与展望

- 对复杂用户指令的解析能力有限，可能源于训练数据规模不足（147K）
- GPT-4V 标注需要 API 调用，数据集构建成本较高，且依赖闭源模型
- 仅生成 4 个正交视角（2×2 布局），对于需要更多视角或任意视角控制的场景不够灵活
- 固定 5° 仰角限制了视角多样性，对某些应用（如俯视/仰视）可能不适用
- 3D 重建质量依赖下游方法（InstantMesh），端到端质量有进一步提升空间

## 相关工作与启发

- Zero123++/SyncDreamer/Wonder3D：单图到多视角的基线方法，仅图像条件控制不足
- MVDream：文本到多视角方法，但缺乏 3D-aware 的精细文本控制
- Cap3D：3D 物体文本标注的先驱工作，但单视角独立标注+汇总缺乏 3D 空间关系
- ControlNet：2D 可控生成的代表方法，FlexGen 将类似思路扩展到多视角 3D 生成
- 启发：利用大模型的视觉推理能力为 3D 任务提供结构化先验（如 3D-aware 文本）是一个有前景的方向

## 评分

- 新颖性: ⭐⭐⭐⭐ （GPT-4V 标注方式和双控制模块有新意，但整体框架基于成熟组件）
- 实验充分度: ⭐⭐⭐⭐ （NVS+文本+3D重建多任务评估充分，但缺少与更多可控生成方法的对比）
- 写作质量: ⭐⭐⭐⭐ （方法描述清晰，可视化丰富，但部分对比实验可更详细）
- 价值: ⭐⭐⭐⭐ （多模态可控多视角生成方向有实际需求，对 3D 内容创建有推动作用）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] MV-Adapter: Multi-view Consistent Image Generation Made Easy](mv-adapter_multi-view_consistent_image_generation_made_easy.md)
- [\[ICCV 2025\] AR-1-to-3: Single Image to Consistent 3D Object Generation via Next-View Prediction](ar1to3_single_image_to_consistent_3d_object_via_nextview_pre.md)
- [\[ICCV 2025\] MaterialMVP: Illumination-Invariant Material Generation via Multi-view PBR Diffusion](materialmvp_illumination-invariant_material_generation_via_multi-view_pbr_diffus.md)
- [\[ICCV 2025\] SpinMeRound: Consistent Multi-View Identity Generation Using Diffusion Models](spinmeround_consistent_multi-view_identity_generation_using_diffusion_models.md)
- [\[ICCV 2025\] Benchmarking and Learning Multi-Dimensional Quality Evaluator for Text-to-3D Generation](benchmarking_and_learning_multi-dimensional_quality_evaluator_for_text-to-3d_gen.md)

</div>

<!-- RELATED:END -->
