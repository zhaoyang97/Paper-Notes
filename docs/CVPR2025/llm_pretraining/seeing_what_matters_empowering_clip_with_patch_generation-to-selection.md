---
title: >-
  [论文解读] Seeing What Matters: Empowering CLIP with Patch Generation-to-Selection
description: >-
  [CVPR 2025][预训练][CLIP训练效率] 提出 CLIP-PGS（Patch Generation-to-Selection），一种简洁有效的掩码策略，通过渐进式的"生成-选择"过程——先预选候选掩码patch、再用 Sobel 边缘检测保护关键语义区域、最后用最优传输归一化精细化选择——在提升 CLIP 训练效率（降至 0.5-0.6× 训练时间）的同时在零样本分类、检索等任务上取得 SOTA。
tags:
  - "CVPR 2025"
  - "预训练"
  - "CLIP训练效率"
  - "Patch掩码策略"
  - "边缘检测"
  - "最优传输正则化"
  - "语义保持"
---

# Seeing What Matters: Empowering CLIP with Patch Generation-to-Selection

**会议**: CVPR 2025  
**arXiv**: [2503.17080](https://arxiv.org/abs/2503.17080)  
**代码**: [GitHub](https://github.com/NUST-Machine-Intelligence-Laboratory/CLIP-PGS)  
**领域**: LLM评测  
**关键词**: CLIP训练效率, Patch掩码策略, 边缘检测, 最优传输正则化, 语义保持

## 一句话总结

提出 CLIP-PGS（Patch Generation-to-Selection），一种简洁有效的掩码策略，通过渐进式的"生成-选择"过程——先预选候选掩码patch、再用 Sobel 边缘检测保护关键语义区域、最后用最优传输归一化精细化选择——在提升 CLIP 训练效率（降至 0.5-0.6× 训练时间）的同时在零样本分类、检索等任务上取得 SOTA。

## 研究背景与动机

**领域现状**：CLIP 等视觉-语言预训练模型通过大规模图文对学习，展现了强大的零样本能力。但训练极其消耗计算资源。近年来掩码策略（如 FLIP、MaskCLIP、A-CLIP、E-CLIP）通过选择性移除图像 patch 来提升训练效率。

**现有痛点**：
- **随机掩码**（FLIP）：可能误删关键语义区域，破坏视觉-文本对齐
- **注意力掩码**（A-CLIP）：需要额外的注意力模块，增加计算复杂度
- **聚类掩码**（E-CLIP）：保持连贯视觉结构但缺乏细粒度语义保护，可能无意中遮挡与文本描述对应的区域

**核心矛盾**：掩码比例越高训练越快，但掩码越多越容易丢失关键语义信息，损害对齐质量。如何在高掩码比例下最大限度保持语义完整性？

**本文目标** 设计一种能在保持关键语义内容的前提下高效减少输入patch数量的掩码策略。

**切入角度**：渐进式的生成-选择过程——先粗选候选patch，再用边缘检测保护主要物体区域，最后用patch间相似度+最优传输做精细化选择。

**核心 idea**：通过边缘检测保护物体边界 + 最优传输归一化平衡patch相似度分布，实现"掩码多但不掩关键区域"的精准掩码。

## 方法详解

### 整体框架

CLIP-PGS 在标准 CLIP 训练流程中加入一个前处理步骤：在图像进入 ViT 图像编码器之前，通过 PGS 策略选择保留哪些 patch。文本端不做修改。最终使用标准的 InfoNCE 对比损失进行对齐。

### 关键设计

1. **渐进式动态掩码率**:
    - 功能：以较低的初始掩码率开始，逐步增加到目标掩码率
    - 核心思路：初始使用仅 5% 的随机掩码（相比 FLIP 的 50%），先预选少量候选 patch 作为潜在掩码区域。两个变体：CLIP-PGS_0.5（固定 0.5 掩码率）和 CLIP-PGS_0.3（在 [0.3, 0.5] 之间动态调整）
    - 设计动机：从小比例开始渐进式扩展掩码区域，比一步到位的随机掩码能更好地保护关键语义

2. **Sobel 边缘检测 (Edge Detection, ED)**:
    - 功能：生成边缘图以保护物体边界和高对比度区域
    - 核心思路：对整张图像应用 Sobel 算子生成边缘图。如果一个 patch 被初步标记为掩码区域但具有高边缘分数，则保留它；低边缘分数的候选 patch 更可能被掩码。额外计算开销仅约 1%
    - 设计动机：物体边界是语义最密集的区域，保护边缘区域等价于保护最关键的语义信息

3. **最优传输归一化 (Optimal Transport Normalization, OTN)**:
    - 功能：通过平衡patch间相似度分布来优化掩码选择
    - 核心思路：计算 patch 间余弦相似度矩阵 S，融合特征相似度和图像相似度（权重 α 随训练 epoch 调整）。使用 Sinkhorn 算法迭代归一化为双随机矩阵，得到平衡的相似度得分。保留与邻近 patch 相似度高的 patch（它们是冗余区域的代表，可安全掩码），掩码相似度低的 patch（它们包含独特信息）
    - 设计动机：仅靠边缘检测无法捕捉 patch 间的语义冗余关系，OTN 从特征级别补充了这一信息。额外开销约 1%

### 损失函数 / 训练策略

- **损失函数**：标准 InfoNCE 对比损失（与 CLIP 相同）
- **训练配置**：
    - 数据集：CC12M（约 1200 万图文对）
    - 架构：ViT-B/16 图像编码器 + 12层文本编码器（512维，8头）
    - 优化器：AdamW，lr=1e-3，β1=0.9，β2=0.98，权重衰减 0.2
    - 训练：32 epochs，batch size 4096，8 × V100 GPU
    - 额外开销：ED + OTN 总计 < 3%

## 实验关键数据

### 主实验

**零样本分类（17个数据集平均 Top-1 准确率）**：

| 方法 | 训练时间 | 平均准确率 |
|------|---------|-----------|
| CLIP | 1.0× | 35.1% |
| FLIP | 0.5× | 33.0% |
| A-CLIP | 1.1× | 35.9% |
| E-CLIP | 0.6× | 36.9% |
| CLIP-PGS_0.5 | **0.5×** | 37.6% |
| CLIP-PGS_0.3 | **0.6×** | **39.5%** |

**零样本检索（MS-COCO Text R@1 / Image R@1）**：
- CLIP-PGS_0.3: 36.0% / 25.1%（均为最佳）

**线性探测（ImageNet-1K）**：
- CLIP-PGS_0.3: 64.4%（vs. E-CLIP 62.7%, CLIP 62.3%）

**鲁棒性评估（ImageNet 变体平均）**：
- CLIP-PGS_0.3: 32.9% 总平均，31.8% OOD 平均（均为最佳）

### 消融实验

| 配置 | ZS (IN-1K) | LP (IN-1K) | TR (COCO) | IR (COCO) |
|------|-----------|-----------|-----------|-----------|
| CLIP 基线 | 36.1 | 62.3 | 34.6 | 23.5 |
| FLIP 随机掩码 | 34.4 | 61.3 | 32.6 | 22.6 |
| PGS_0.3 (无 ED, 无 OTN) | 35.9 | 61.7 | 33.5 | 23.0 |
| PGS_0.3 + ED | 36.8 | 63.2 | 34.3 | 24.0 |
| PGS_0.3 + OTN | 36.7 | 63.0 | 34.5 | 23.8 |
| PGS_0.3 + ED + OTN | **38.6** | **64.4** | **36.0** | **25.1** |

### 关键发现

- **ED 和 OTN 互补**：各自单独有效，联合使用效果最佳（38.6% vs 36.8%/36.7%）
- **Sobel 优于 Canny**：Sobel 边缘检测微弱优于 Canny（38.6% vs 38.5%）
- **初始掩码率 5% 最优**：过高的初始掩码率会降低性能
- **动态掩码率更好**：PGS_0.3（动态 0.3-0.5）优于 PGS_0.5（固定 0.5）
- **ViT-B/16 最佳**：ViT-B/16 > ViT-S/16 > ViT-B/32
- **计算开销极小**：ED + OTN 总共额外开销 < 3%

## 亮点与洞察

1. **方法简洁有效**：三步渐进式策略（预选→边缘保护→OTN精细化），无需额外训练模块
2. **全面超越前辈**：在同等训练时间下（0.5-0.6×），全面超越 CLIP、FLIP、A-CLIP、E-CLIP
3. **语义保持策略**：边缘检测 + 最优传输的组合巧妙地从两个角度保护了语义信息
4. **额外开销极小**：与 FLIP 相同训练时间下即可取得更好效果，ED+OTN 总共 < 3% 开销
5. **语言组合性也提升**：在 SugarCrepe 数据集上的提升表明更好的掩码策略确实改善了视觉-语言对齐质量

## 局限与展望

1. **数据集规模受限**：仅在 CC12M 上训练，更大数据集（如 LAION）上的表现未验证
2. **仅限 ViT 架构**：未来可扩展到 CNN 架构（如 ConvNeXt）
3. **仅限双编码器模型**：当前针对 CLIP 双编码器设计，可探索适配 MAE 等自监督方法
4. **掩码策略的可学习化**：当前策略是启发式的，可以探索端到端学习的掩码选择

## 相关工作与启发

- **CLIP**：基础对比学习框架
- **FLIP**：随机掩码提升 CLIP 训练效率的开创性工作
- **A-CLIP**：基于注意力的自适应掩码
- **E-CLIP**：基于聚类的掩码策略
- **MaskCLIP**：结合掩码图像建模与对比学习的自蒸馏
- **对后续研究的启发**：边缘检测和最优传输这类传统 CV 工具在深度学习预处理中仍然有用

## 评分

- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Improving Autoregressive Visual Generation with Cluster-Oriented Token Prediction](improving_autoregressive_visual_generation_with_cluster-oriented_token_predictio.md)
- [\[CVPR 2025\] ScaMo: Exploring the Scaling Law in Autoregressive Motion Generation Model](scamo_exploring_the_scaling_law_in_autoregressive_motion_generation_model.md)
- [\[ICML 2025\] The Double-Ellipsoid Geometry of CLIP](../../ICML2025/llm_pretraining/the_double-ellipsoid_geometry_of_clip.md)
- [\[ICML 2025\] Whitened CLIP as a Likelihood Surrogate of Images and Captions](../../ICML2025/llm_pretraining/whitened_clip_as_a_likelihood_surrogate_of_images_and_captions.md)
- [\[ICCV 2025\] SynCity: Training-Free Generation of 3D Worlds](../../ICCV2025/llm_pretraining/syncity_training-free_generation_of_3d_worlds.md)

</div>

<!-- RELATED:END -->
