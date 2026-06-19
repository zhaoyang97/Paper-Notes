---
title: >-
  [论文解读] WonderTurbo: Generating Interactive 3D World in 0.72 Seconds
description: >-
  [ICCV 2025][3D视觉][实时3D场景生成] WonderTurbo 提出首个实时交互式3D场景生成框架，通过 StepSplat（前馈式3DGS）、QuickDepth（轻量深度补全）和 FastPaint（2步扩散修复）三个模块协同加速，将单次场景扩展时间从 10+ 秒压缩到 0.72 秒，实现 15 倍加速的同时保持了与 WonderWorld 相当的生成质量。
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "实时3D场景生成"
  - "高斯溅射"
  - "深度补全"
  - "扩散蒸馏"
  - "交互式生成"
---

# WonderTurbo: Generating Interactive 3D World in 0.72 Seconds

**会议**: ICCV 2025  
**arXiv**: [2504.02261](https://arxiv.org/abs/2504.02261)  
**代码**: [https://github.com/GigaAI-research/WonderTurbo](https://github.com/GigaAI-research/WonderTurbo)  
**领域**: 3D视觉  
**关键词**: 实时3D场景生成、高斯溅射、深度补全、扩散蒸馏、交互式生成

## 一句话总结

WonderTurbo 提出首个实时交互式3D场景生成框架，通过 StepSplat（前馈式3DGS）、QuickDepth（轻量深度补全）和 FastPaint（2步扩散修复）三个模块协同加速，将单次场景扩展时间从 10+ 秒压缩到 0.72 秒，实现 15 倍加速的同时保持了与 WonderWorld 相当的生成质量。

## 研究背景与动机

**领域现状**：从单张图像交互式生成3D场景（online 3D scene generation）是沉浸式虚拟体验的核心技术。现有方法分为离线和在线两类：离线方法如 LucidDreamer、Text2Room 先生成多视角图像再优化3D表示，在线方法如 WonderJourney、WonderWorld 支持用户逐步交互式创建场景。

**现有痛点**：即便是当前最快的在线方法 WonderWorld，生成一个新视角仍需约 10 秒，远不能满足实时交互的需求。效率瓶颈主要来自两方面：(1) 几何建模依赖 3DGS 的逐场景迭代优化，需要数百次迭代；(2) 外观建模依赖扩散模型进行图像修复（inpainting），需要数十步推理。

**核心矛盾**：3D场景生成的质量要求与实时性要求之间存在严重矛盾——高质量的几何和外观建模都需要大量计算，而交互式场景要求亚秒级响应。

**本文目标**：设计一个能在 1 秒以内完成一次场景扩展的完整框架，同时几何和外观质量不能明显下降。具体分解为三个子问题——如何加速几何建模、如何提供一致的深度先验、如何加速外观修复。

**切入角度**：作者观察到前馈式3DGS方法（如 MVSplat、PixelSplat）可以跳过迭代优化直接推理高斯参数，但它们不支持视角逐步增加的交互场景；扩散模型的蒸馏技术可以将推理步数压缩到极少步。这两个方向结合起来可以同时解决几何和外观的效率瓶颈。

**核心 idea**：用前馈式3DGS + 特征记忆实现增量式几何建模，用轻量深度补全提供一致深度先验，用2步蒸馏扩散完成即时外观修复，三者组合实现 0.72 秒的实时3D交互。

## 方法详解

### 整体框架

WonderTurbo 的流水线工作如下：用户移动相机到新位置后，系统首先渲染当前3D场景得到图像 $I_{render}^i$ 和深度图 $D_{render}^i$。然后 FastPaint 接收渲染图像和用户文本，生成新区域的外观 $I_{target}^i$（0.22秒）；QuickDepth 接收渲染深度和新外观图像，补全新区域的深度 $D_{target}^i$（0.24秒）；最后 StepSplat 将新图像和深度转化为局部高斯表示，并增量融合到全局3D场景中（0.26秒）。总计 0.72 秒完成一次场景扩展。

### 关键设计

1. **StepSplat — 增量式前馈3DGS**:

    - 功能：在 0.26 秒内将新视角的图像和深度转化为3D高斯表示，并融合到全局场景
    - 核心思路：StepSplat 采用 RepVGG 作为骨干网络提取匹配特征 $F_m^i$ 和图像特征 $F_e^i$。引入特征记忆（Feature Memory）存储历史视角的匹配特征和位姿。对于新视角，根据位姿距离 $d(P_n, P_i) = \|P_n - P_i\|_2$ 选取最近的 $N_v$ 个邻近视角，利用 QuickDepth 提供的深度图在深度范围 $R = \{d \mid (1-a) \cdot D_{target}^i \leq d \leq (1+a) \cdot D_{target}^i\}$ 内均匀采样 $N_d$ 个深度候选值，通过平面扫描立体算法将邻近特征 warp 到当前视角，计算归一化点积相关性构建代价体 $S^i$。最终通过 softmax 加权平均得到深度预测 $\hat{d} = \text{softmax}(S^i) \cdot d$，深度值反投影为高斯中心。增量融合策略通过深度一致性约束 $|d_{local} - d_j^g| < \delta \cdot d_{local}$ 去除冲突高斯，将有效的局部高斯合并到全局表示
    - 设计动机：传统3DGS需要数百次迭代优化，而前馈方法直接推理高斯参数但不支持视角逐步增加。StepSplat 通过保持特征记忆并自适应构建代价体，将前馈范式扩展到交互式场景。深度引导的代价体确保了几何精度，增量融合减少了冗余高斯导致的浮点问题

2. **QuickDepth — 轻量深度补全**:

    - 功能：在 0.24 秒内为新生成区域补全完整深度图，提供一致的深度先验
    - 核心思路：以轻量深度估计模型（Depth Anything）初始化，输入目标帧的 RGB 图像、不完整深度图和二值掩码，预测完整深度。训练数据通过相邻帧的几何投影关系构造——将前一帧深度图 $D_{j-1}$ 通过相对位姿 $T_{j-1 \to j}$ 投影到当前帧坐标系，得到不完整深度 $D'_{j-1 \to j}$ 和有效性掩码 $M_{j-1 \to j}$，使用 $L_1$ 损失监督
    - 设计动机：现有深度补全方法主要针对稀疏深度补全（如LiDAR），难以处理3D场景生成中大面积无深度信息的情况。WonderWorld 的引导深度扩散方法需要 3 秒以上。QuickDepth 专门针对交互式3D生成的掩码模式训练，泛化能力强

3. **FastPaint — 2步扩散修复**:

    - 功能：在 0.22 秒内完成图像修复，为新区域生成外观
    - 核心思路：对预训练的 Stable Diffusion Inpainting 模型进行知识蒸馏，结合 ODE 轨迹保持和重构策略，将推理步数从数十步压缩到仅 2 步。同时在模拟交互式3D生成掩码模式的数据集上微调，使修复区域与3D场景生成的实际掩码分布对齐
    - 设计动机：原始扩散修复模型需要约 20-50 步推理，且微调时使用的掩码分布与3D生成场景不同，导致修复质量下降或需要额外VLM验证。FastPaint 通过蒸馏和针对性微调同时解决了速度和质量问题

### 训练数据构建

作者利用多种3D场景生成方法（WonderJourney、WonderWorld、Text2Room、LucidDreamer 等）构建了包含超过 600 万帧的训练数据集，涵盖室内环境（32%）、城市景观（28%）、自然地形（25%）和风格化艺术场景（15%），通过模拟交互式轨迹（旋转、线性、混合）生成训练数据，并使用 VLM 验证数据质量。

## 实验关键数据

### 主实验

| 方法 | 类型 | 几何建模(s) | 外观建模(s) | 总时间(s) |
|------|------|------------|------------|----------|
| LucidDreamer | Offline | 35.38 | 8.32 | 43.70 |
| Text2Room | Offline | 34.23 | 7.32 | 41.55 |
| Pano2Room | Offline | 27.91 | 1.47 | 29.38 |
| DreamScene360 | Offline | 44.29 | 1.45 | 45.74 |
| WonderJourney | Online | 78.12 | 1.45 | 79.57 |
| WonderWorld | Online | 6.62 | 4.43 | 11.05 |
| **WonderTurbo** | **Online** | **0.50** | **0.22** | **0.72** |

| 方法 | CS↑ | CC↑ | CIQA↑ | Q-Align↑ | CA↑ |
|------|-----|-----|-------|----------|-----|
| LucidDreamer | 27.72 | 0.9213 | 0.6023 | 3.5439 | 6.8231 |
| Text2Room | 24.50 | 0.9035 | 0.4910 | 2.6732 | 6.5324 |
| WonderJourney | 27.63 | 0.9652 | 0.4753 | 3.5272 | 7.0134 |
| WonderWorld | 28.14 | 0.9654 | 0.6764 | 3.7823 | 7.2121 |
| **WonderTurbo** | **28.65** | **0.9732** | **0.6812** | 3.7253 | **7.3243** |

### 消融实验

| 配置 | CS↑ | CC↑ | CIQA↑ | Q-Align↑ | CA↑ |
|------|-----|-----|-------|----------|-----|
| w/ FreeSplat | 27.65 | 0.9542 | 0.6460 | 3.1543 | 6.6235 |
| w/ DepthSplat | 27.32 | 0.9675 | 0.6620 | 3.2145 | 6.7432 |
| w/o depth guided | 27.72 | 0.9532 | 0.6359 | 3.4361 | 7.1734 |
| w/o incremental fusion | 27.87 | 0.9654 | 0.6459 | 3.5431 | 7.2734 |
| w/o FastPaint | 27.82 | 0.9683 | 0.6574 | 3.7146 | 7.2136 |
| **WonderTurbo (full)** | **28.65** | **0.9732** | **0.6812** | **3.7253** | **7.3243** |

### 关键发现

- StepSplat 相比 FreeSplat 和 DepthSplat 有显著优势，尤其在 Q-Align（+0.57/+0.51）和 CLIP aesthetic（+0.70/+0.58）上，说明深度引导的代价体对几何精度至关重要
- 去掉深度引导的代价体后掉点最严重（CS 下降 0.93，CC 下降 0.020），这是因为没有深度先验时代价体搜索范围过大，导致几何不准确
- 用户研究中 WonderTurbo 对比 WonderWorld 的胜率达 69.43%，对比其他方法胜率均超过 94%，说明在 15 倍加速下用户感知质量几乎没有下降
- FastPaint 的贡献主要体现在 CS（+0.83）和 CIQA（+0.124）上，说明针对性微调提升了修复区域与文本的语义一致性

## 亮点与洞察

- **前馈范式 + 特征记忆的结合**非常巧妙：通过保持历史视角的匹配特征而非原始图像，既利用了前馈推理的速度优势，又实现了类似迭代优化的多视角信息融合。这个设计可以迁移到任何需要增量式3D重建的场景
- **深度引导代价体的设计**：用 QuickDepth 的深度预测作为 prior 来缩小代价体的深度搜索范围，是 MVS 中经典的 coarse-to-fine 思想在交互式生成中的优雅应用。只需在深度预测 ±a 范围内搜索，大幅减少计算量同时提升精度
- **训练数据的构建策略值得借鉴**：利用多种现有3D生成方法的互补优势来构建训练数据，VLM 验证质量，这种 "用旧方法的输出训练新方法" 的bootstrap策略在数据稀缺时非常实用

## 局限与展望

- 代码尚未完全开源，GitHub 仓库为 placeholder，难以复现和验证
- 0.72 秒虽然接近实时，但距离真正的 30fps 实时交互还有差距（需要 33ms 以内），在 VR/AR 等高帧率场景中仍不够
- 训练数据依赖其他3D生成方法的输出，数据质量受限于这些方法的能力上限
- 用户研究虽然胜率高，但缺少对几何精度的定量评估（如点云精度、mesh质量）
- FastPaint 基于蒸馏，在复杂纹理和精细细节上可能不如完整推理步数的扩散模型

## 相关工作与启发

- **vs WonderWorld**: WonderWorld 用 FLAGS + 扩散引导深度估计，仍需 10 秒；WonderTurbo 用前馈 StepSplat + QuickDepth 替代迭代优化，速度快 15 倍，质量基本持平
- **vs MVSplat/DepthSplat**: 这些前馈3DGS方法针对固定双视角输入设计，不支持视角逐渐增加的交互场景；StepSplat 通过特征记忆和增量融合扩展了前馈范式的适用范围
- **vs Hyper-SD 等蒸馏方法**: FastPaint 采用类似的 ODE 轨迹蒸馏思路，但专门针对 inpainting 任务和3D生成的掩码分布进行了适配

## 评分

- 新颖性: ⭐⭐⭐⭐ 三个模块的组合创新扎实，但各模块本身的技术贡献相对增量
- 实验充分度: ⭐⭐⭐⭐ 定量对比、消融实验和用户研究齐全，但缺少几何精度的定量评估
- 写作质量: ⭐⭐⭐⭐ 结构清晰，pipeline 描述易于理解，公式推导规范
- 价值: ⭐⭐⭐⭐⭐ 实时交互式3D生成是强刚需，15 倍加速的工程价值极高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Bolt3D: Generating 3D Scenes in Seconds](bolt3d_generating_3d_scenes_in_seconds.md)
- [\[ICCV 2025\] A Recipe for Generating 3D Worlds from a Single Image](a_recipe_for_generating_3d_worlds_from_a_single_image.md)
- [\[CVPR 2025\] PhysGen3D: Crafting a Miniature Interactive World from a Single Image](../../CVPR2025/3d_vision/physgen3d_crafting_a_miniature_interactive_world_from_a_single_image.md)
- [\[ICCV 2025\] Text2VDM: Text to Vector Displacement Maps for Expressive and Interactive 3D Sculpting](text2vdm_text_to_vector_displacement_maps_for_expressive_and_interactive_3d_scul.md)
- [\[ICCV 2025\] Easy3D: A Simple Yet Effective Method for 3D Interactive Segmentation](easy3d_a_simple_yet_effective_method_for_3d_interactive_segmentation.md)

</div>

<!-- RELATED:END -->
