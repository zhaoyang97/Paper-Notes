---
title: >-
  [论文解读] MINIMA: Modality Invariant Image Matching
description: >-
  [CVPR 2025][图像生成][跨模态图像匹配] MINIMA 提出了一个统一的跨模态图像匹配框架，通过设计数据引擎从廉价的 RGB 图像对中生成多模态合成数据集 MD-syn（480M 对），使任何现有匹配管线仅需微调即可获得跨模态匹配能力，在 19 种跨模态场景下显著超越模态特定方法。 1. 领域现状：图像匹配是视觉…
tags:
  - "CVPR 2025"
  - "图像生成"
  - "跨模态图像匹配"
  - "数据引擎"
  - "合成数据"
  - "模态不变"
  - "特征匹配"
---

# MINIMA: Modality Invariant Image Matching

**会议**: CVPR 2025  
**arXiv**: [2412.19412](https://arxiv.org/abs/2412.19412)  
**代码**: [https://github.com/LSXI7/MINIMA](https://github.com/LSXI7/MINIMA)  
**领域**: 图像匹配 / 多模态感知  
**关键词**: 跨模态图像匹配, 数据引擎, 合成数据, 模态不变, 特征匹配

## 一句话总结
MINIMA 提出了一个统一的跨模态图像匹配框架，通过设计数据引擎从廉价的 RGB 图像对中生成多模态合成数据集 MD-syn（480M 对），使任何现有匹配管线仅需微调即可获得跨模态匹配能力，在 19 种跨模态场景下显著超越模态特定方法。

## 研究背景与动机

1. **领域现状**：图像匹配是视觉定位、目标检测等多种应用的基础。在 RGB 图像匹配上已有大量成熟的稀疏/半稠密/稠密匹配方法（LightGlue、LoFTR、RoMa 等），受益于 MegaDepth、ScanNet 等大规模数据集。
2. **现有痛点**：跨模态匹配（RGB-红外、RGB-深度、RGB-事件等）面临严重的域差异挑战。现有跨模态数据集规模极小（几千对 vs RGB 数据集几千万对），场景覆盖有限，且标注耗时昂贵。这导致模态特定方法在训练集外的场景泛化能力极差。
3. **核心矛盾**：问题本质是**数据缺口**——多模态图像采集需要多台不同成像设备同时拍摄同一场景，成本高且难以保证场景多样性；标注也不能直接用 COLMAP 等工具获得。
4. **本文目标** 如何用一个统一模型处理所有跨模态匹配？如何低成本获取大规模、高质量的多模态匹配训练数据？
5. **切入角度**：从便宜且丰富的 RGB 图像对出发，利用生成模型将 RGB 图像转换为其他模态的伪图像。由于转换是逐像素的，原始 RGB 对的深度、位姿等匹配标注可以直接继承到生成的多模态数据。
6. **核心 idea**：通过数据引擎将 MegaDepth 的 RGB 图像对扩展到 6 种模态（红外、深度、事件、法线、油画、素描），生成 480M 跨模态对用于训练任意匹配管线。

## 方法详解

### 整体框架
MINIMA 分为两个部分：(1) **数据引擎**：从 MegaDepth 的 RGB 图像对生成 6 种模态的合成图像，构建 MD-syn 数据集；(2) **模型训练**：在 RGB 上预训练匹配模型，然后在 MD-syn 上用随机选择的跨模态对微调。输出是能处理任意模态组合的统一匹配模型。

### 关键设计

1. **跨模态数据引擎 (Cross-Modal Data Engine)**:

    - 功能：从 RGB 图像自动生成多种模态的合成图像，同时继承原始匹配标注。
    - 核心思路：用 6 个生成模型分别将 RGB 转换为：红外（用 StyleBooth 在 LLVIP+M3FD 上微调的 LoRA），深度（Depth Anything V2），事件（基于物理模型模拟亮度变化），法线（DSINE），油画（Paint Transformer），素描（Anime2Sketch）。对于一对 RGB 图像 $(A_0, B_0)$，生成两组 K=6 种模态的图像集，总计可构造出 480M 跨模态图像对。
    - 设计动机：三大优势——Cheap（RGB 图像易获取）、Flexible（可自由控制生成规模和模态平衡）、High-quality（生成图像分辨率与 RGB 一致，标注直接继承）。

2. **两阶段训练策略 (Pre-train + Fine-tune)**:

    - 功能：让现有匹配管线高效获得跨模态能力。
    - 核心思路：Stage 1 在 RGB 数据上预训练匹配模型至收敛（或直接用官方预训练权重）；Stage 2 在 MD-syn 上用随机选择的跨模态对以小学习率微调。随机选择模态对使模型不偏向特定模态组合。
    - 设计动机：从零开始在多模态数据上训练难以收敛（模态间方差大），而 RGB 预训练提供了良好的匹配先验，微调可快速收敛并获得跨模态泛化。

3. **多管线适配 (Multi-Pipeline Adaptation)**:

    - 功能：证明数据引擎的通用性——不依赖特定网络架构。
    - 核心思路：选择三种代表性匹配管线——稀疏匹配 LightGlue、半稠密匹配 LoFTR、稠密匹配 RoMa，分别微调并发布 MINIMA_LG、MINIMA_LoFTR、MINIMA_RoMa。
    - 设计动机：表明 MINIMA 是一种与模型无关的数据层解决方案，任何先进匹配方法都能从中受益。

### 损失函数 / 训练策略
直接沿用各基线匹配管线的原始损失函数，微调时使用较小的学习率。红外模态生成器在 LLVIP+M3FD 上以 $lr=1\times10^{-4}$ 微调 210k 步，LoRA rank 256。

## 实验关键数据

### 主实验

在 6 个真实跨模态数据集上的总体匹配精度（AUC@10°或@10px）：

| 方法 | 类型 | RGB-IR | RGB-Depth | RGB-Normal | RGB-Event | RGB-Sketch | RGB-Paint |
|------|------|--------|-----------|------------|-----------|------------|-----------|
| LightGlue | 稀疏 | 17.73 | 2.87 | 24.93 | 22.40 | 44.47 | 27.99 |
| **MINIMA_LG** | 稀疏 | **30.24** | **32.53** | **37.33** | **36.27** | **45.71** | **32.85** |
| LoFTR | 半稠密 | 12.58 | 0.44 | 12.07 | 12.43 | 54.82 | 12.22 |
| **MINIMA_LoFTR** | 半稠密 | **32.36** | **28.81** | **44.26** | **32.74** | **53.54** | **15.45** |
| RoMa | 稠密 | 29.46 | 0.38 | 39.28 | 18.14 | 72.25 | 44.73 |
| **MINIMA_RoMa** | 稠密 | **46.77** | **42.17** | **50.87** | **44.32** | **73.10** | **50.34** |

在 RGB-Depth 上改进尤其惊人（LightGlue: 2.87→32.53）。

### 消融实验

| 数据策略 | AUC@10° (平均) | 说明 |
|---------|---------------|------|
| 仅 RGB 预训练 | ~20 | 基线，跨模态精度差 |
| + 单模态微调 | ~30 | 仅对训练模态有效 |
| + 随机多模态微调 (MD-syn) | ~38 | 全面提升所有模态 |

### 关键发现
- **数据引擎是核心贡献**：同一匹配架构，换用 MD-syn 数据训练后跨模态性能大幅提升，证明瓶颈在数据而非模型。
- **零样本跨模态泛化优异**：在从未见过的真实跨模态数据集上也能显著超越专门针对该模态训练的方法（如 ReDFeat、XoFTR）。
- **不同匹配管线均受益**：稀疏、半稠密、稠密三种类型都获得显著提升，说明数据引擎的通用性。
- 模态间数据平衡很重要——随机选择模态对防止了对简单模态的过拟合。

## 亮点与洞察
- **"数据即方法"的范式**极其优雅：不设计任何模态特定模块，仅通过扩充训练数据就让通用匹配器获得跨模态能力。这种思路可迁移到很多跨域问题——先构造大规模伪域数据，再微调现有最强模型。
- **合成数据继承标注**的方式很巧妙：逐像素的风格转换不改变几何关系，所以深度和位姿标注可以零成本继承，解决了多模态数据标注难的根本问题。
- 首次在统一框架下评估 19 种跨模态匹配场景，构建了全面的评测基准。

## 局限与展望
- 合成的伪模态与真实模态仍有域差异（如红外的合成质量受限于微调数据量），可能在某些挑战性场景下不够准确。
- 仅使用 MegaDepth（室外）作为基础数据集，室内场景的跨模态匹配性能可能受限。
- 当前仅支持 6 种模态，SAR、医学影像等更专业的模态未覆盖。
- 生成模型的计算开销较高（特别是基于扩散的红外生成器），数据集构建成本不可忽视。

## 相关工作与启发
- **vs ReDFeat**：为每种模态单独训练检测器+描述子，且在各数据集上分别训练/测试，泛化性差。MINIMA 用统一模型处理所有模态，性能更优。
- **vs XoFTR**：仅针对 RGB-IR 设计两阶段训练策略，需要模态特定匹配规则。MINIMA 在 RGB-IR 上也超越了它，且覆盖范围广得多。
- **vs GIM**：GIM 也尝试从视频扩展训练数据提升泛化性，但用了数倍规模的图像仅获得微小提升。MINIMA 选择从模态维度扩展更经济高效。

## 评分
- 新颖性: ⭐⭐⭐⭐ 数据引擎思路简单而有效，首次证明合成多模态数据可替代真实数据训练匹配器
- 实验充分度: ⭐⭐⭐⭐⭐ 19种跨模态场景、3种匹配范式、大量消融和零样本实验
- 写作质量: ⭐⭐⭐⭐ 动机清晰，实验组织系统化
- 价值: ⭐⭐⭐⭐⭐ 数据引擎和MD-syn数据集对社区价值极大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Not Just Text: Uncovering Vision Modality Typographic Threats in Image Generation Models](not_just_text_uncovering_vision_modality_typographic_threats_in_image_generation.md)
- [\[CVPR 2026\] Semantic Alignment for Pose-Invariant Identity Preserving Diffusion](../../CVPR2026/image_generation/semantic_alignment_for_pose-invariant_identity_preserving_diffusion.md)
- [\[ICCV 2025\] Bridging the Skeleton-Text Modality Gap: Diffusion-Powered Modality Alignment for Zero-shot Skeleton-based Action Recognition](../../ICCV2025/image_generation/bridging_the_skeleton_text_modality_gap_diffusion_powered_modality_alignment_for.md)
- [\[CVPR 2025\] Diff2Flow: Training Flow Matching Models via Diffusion Model Alignment](diff2flow_training_flow_matching_models_via_diffusion_model_alignment.md)
- [\[ICLR 2026\] When One Modality Rules Them All: Backdoor Modality Collapse in Multimodal Diffusion Models](../../ICLR2026/image_generation/when_one_modality_rules_them_all_backdoor_modality_collapse_in_multimodal_diffus.md)

</div>

<!-- RELATED:END -->
