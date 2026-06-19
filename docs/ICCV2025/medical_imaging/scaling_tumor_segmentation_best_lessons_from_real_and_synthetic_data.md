---
title: >-
  [论文解读] Scaling Tumor Segmentation: Best Lessons from Real and Synthetic Data
description: >-
  [ICCV 2025][医学图像][数据缩放定律] 通过在大规模私有数据集上系统研究数据缩放定律，发现合成肿瘤可大幅降低真实标注需求（从 1500 降至 500 例），并据此构建了 AbdomenAtlas 2.0——首个涵盖 6 种器官肿瘤的万级 CT 大规模人工标注数据集，在分布内和分布外测试上均取得显著提升。
tags:
  - "ICCV 2025"
  - "医学图像"
  - "数据缩放定律"
  - "肿瘤分割"
  - "合成数据"
  - "AbdomenAtlas"
  - "CT分割"
---

# Scaling Tumor Segmentation: Best Lessons from Real and Synthetic Data

**会议**: ICCV 2025  
**arXiv**: [2510.14831](https://arxiv.org/abs/2510.14831)  
**代码**: [https://github.com/BodyMaps/AbdomenAtlas2.0](https://github.com/BodyMaps/AbdomenAtlas2.0)  
**领域**: 医学影像 / 肿瘤分割  
**关键词**: 数据缩放定律, 肿瘤分割, 合成数据, AbdomenAtlas, CT分割

## 一句话总结

通过在大规模私有数据集上系统研究数据缩放定律，发现合成肿瘤可大幅降低真实标注需求（从 1500 降至 500 例），并据此构建了 AbdomenAtlas 2.0——首个涵盖 6 种器官肿瘤的万级 CT 大规模人工标注数据集，在分布内和分布外测试上均取得显著提升。

## 研究背景与动机

肿瘤分割 AI 受限于大规模体素级标注数据的匮乏。核心问题是：**训练有效的肿瘤分割 AI 到底需要多少标注数据？合成数据能否减少这一需求？**

作者在 JHH 私有数据集（3,000 例胰腺肿瘤标注 CT）上的关键发现：

**分布内性能在约 1,500 例后饱和**——继续增加同分布真实数据收益递减

**加入 3 倍合成数据后，仅需 500 例真实数据即可达到同等性能**——标注需求降低 70%

**分布外泛化性能持续提升**——即使到 3,000 例仍未饱和，数据多样性比数量更关键

基于这些发现，作者认为每种肿瘤类型 500-1,500 例标注即可构建有效 AI 模型，由此创建了 AbdomenAtlas 2.0。

## 方法详解

### 整体框架

两个核心贡献：
1. **AbdomenAtlas 2.0 数据集**：10,135 例 CT，6 种器官肿瘤，23 位放射科医师标注
2. **数据缩放定律研究**：系统揭示真实数据与合成数据对肿瘤分割的缩放效应

### 关键设计

1. **SMART-Annotator 标注流水线**：

    - **核心思想**：从零标注遗漏肿瘤远比移除 AI 生成的假阳性耗时，因此设计以最大化敏感度为优先
    - **Stage 1 - 模型准备**：针对每种肿瘤单独训练分割模型 $f(\cdot)$
    - **Stage 2 - FROC 曲线分析**：选择阈值 $\theta^*$ 使敏感度 >90%，同时保持可接受的假阳性率
    - **Stage 3 - 候选生成**：AI 生成候选分割，由资深放射科医师确认真阳性、剔除假阳性（平均 1.2-2.4 假阳性/扫描）
    - **Stage 4 - 标注修正**：初级放射科医师修正边界和遗漏，资深医师审核
    - **效率提升**：将每例标注时间从 5 分钟降至 5 秒，节省约 49,826 分钟（83 个工作日）

2. **数据集构建 (AbdomenAtlas 2.0)**：

    - **规模**：10,135 例 CT，470 万切片，15,130 个肿瘤实例
    - **覆盖**：肝脏、胰腺、肾脏、结肠、食管、子宫 6 种肿瘤
    - **来源**：89 家医院，17 个国家
    - **首创**：首个提供食管和子宫肿瘤体素级标注的公开数据集
    - **包含大量早期肿瘤**（<20mm）：肝脏 5,709 例、胰腺 850 例、肾脏 4,638 例

3. **合成肿瘤数据增强 (DiffTumor)**：

    - 使用 DiffTumor 生成合成肿瘤，小:中:大比例为 4:2:1
    - 合成数据量为真实数据的 3 倍
    - 合成肿瘤自动附带体素级标注（生成即标注）
    - 可叠加到任意来源的正常 CT 上，无需人工标注

### 损失函数 / 训练策略

- 使用 nnU-Net 框架，各向同性重采样至 1.5×1.5×1.5mm³
- 强度截断 [-175, 250]，线性归一化至 [0, 1]
- 随机裁剪 96×96×96 区域，SGD 优化器，学习率 0.01
- 训练 1000 epochs，每 epoch 250 次迭代，batch size = 2
- 推理时使用测试时增强和滑动窗口（50% 重叠）

## 实验关键数据

### 主实验 — MSD 排行榜

| 方法 | 肝脏肿瘤 DSC | 肝脏肿瘤 NSD | 胰腺肿瘤 DSC | 胰腺肿瘤 NSD |
|------|-------------|-------------|-------------|-------------|
| nnU-Net | 76.0 | 90.7 | 52.8 | 71.5 |
| Swin UNETR | 75.7 | 91.6 | 58.2 | 79.1 |
| Universal Model | 79.4 | 93.4 | 62.3 | 82.9 |
| **AbdomenAtlas 2.0** | **82.6** | **96.9** | **67.2** | **86.0** |
| Δ | +3.2 | +3.5 | +4.9 | +3.1 |

AbdomenAtlas 2.0 在 MSD 排行榜取得 **#1** 名次。

### 分布外泛化实验

| 外部数据集 | 最佳对比方法 DSC | AbdomenAtlas 2.0 DSC | Δ |
|-----------|----------------|---------------------|---|
| 3D-IRCADb (肝脏) | 67.1 (STU-Net) | **81.1** | **+14.0** |
| PANORAMA (胰腺) | 43.0 (SegResNet) | **55.3** | **+12.3** |
| Kipa (肾脏) | 76.4 (ResEncM) | **83.6** | **+7.2** |
| JHH (胰腺) | 39.5 (SegResNet) | **45.1** | **+5.6** |

分布外泛化性能全面大幅领先，3D-IRCADb 上 DSC 提升 14.0%、NSD 提升 17.0%。

### 消融实验 — 数据缩放

分布内饱和实验（JHH 私有数据集）：

| 真实 CT 数量 | DSC (仅真实) | DSC (真实+合成) |
|-------------|-------------|----------------|
| 60 | 40.2 | 48.2 |
| 278 | 52.7 | 58.1 |
| 500 | ~54 | ~59 (≈仅真实1500例) |
| 1500 | 59.3 | 59.2 |
| 3159 | 59.7 | 59.3 |

关键数据点：**500 例真实数据 + 3× 合成数据 ≈ 1500 例纯真实数据的效果**。

### 关键发现

- **三条核心规律**：
  1. 分布内性能约 1,500 例后饱和
  2. 合成肿瘤可将真实数据需求降低 70%（1500→500）
  3. 分布外泛化持续受益于数据多样性，不饱和
- 合成数据加速分布内收敛（40%-60% 真实数据即可达饱和）
- 每类肿瘤分别提升：肝脏+4.9%、胰腺+8.8%、肾脏+3.1%、结肠+3.6%、食管+7.3%、子宫+1.4%
- 即使在分布外测试中，合成数据也持续贡献性能提升

## 亮点与洞察

- **缩放定律视角**：首次系统研究肿瘤分割中的数据缩放定律，揭示饱和点和合成数据的加速效应
- **实用标注流水线**：SMART-Annotator 将标注时间降低 60 倍，是大规模医学标注的实用方案
- **数据集价值巨大**：10,135 例 CT 覆盖 6 种肿瘤，远超现有公开数据集之和
- **合成数据的深层价值**：不仅提升分布内效率，还通过注入到不同来源的正常 CT 提高分布外泛化
- **开源承诺**：代码、模型、数据全部开源

## 局限与展望

- 1,500 例饱和点仅在胰腺肿瘤上验证，其他器官是否相同未确认
- 合成肿瘤的解剖真实性（特别是浸润性、坏死性或早期病变）未经专家验证
- 仅使用 ResEncM 模型进行缩放实验，不同架构可能有不同饱和点
- 仅覆盖腹部 CT，其他模态和部位的可推广性待验证
- 标注流水线依赖初始 AI 模型质量，对罕见肿瘤类型可能需要额外适配

## 相关工作与启发

- 延续了 Kaplan 等人的缩放定律思想，从语言模型迁移到医学影像
- DiffTumor 的合成肿瘤生成为数据增强提供了新范式
- Universal Model 和 SuPreM 提供了强基线对比
- 启发：在数据受限的医学领域，合成数据 + 少量精标注的组合策略可能是最优解

## 评分

- **新颖性**: ⭐⭐⭐⭐ 缩放定律视角在医学影像中少见，合成数据加速训练的发现有洞察力
- **实验充分度**: ⭐⭐⭐⭐⭐ 大规模数据集、多基线对比、分布内外评估、详细缩放实验，极为充分
- **写作质量**: ⭐⭐⭐⭐ 论述逻辑性强，发现表述清晰，图表丰富
- **价值**: ⭐⭐⭐⭐⭐ 数据集对医学影像社区有巨大价值，缩放定律发现对未来数据集构建有指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] M-Net: MRI Brain Tumor Sequential Segmentation Network via Mesh-Cast](m-net_mri_brain_tumor_sequential_segmentation_network_via_mesh-cast.md)
- [\[ICCV 2025\] MRGen: Segmentation Data Engine for Underrepresented MRI Modalities](mrgen_segmentation_data_engine_for_underrepresented_mri_modalities.md)
- [\[ECCV 2024\] Brain Netflix: Scaling Data to Reconstruct Videos from Brain Signals](../../ECCV2024/medical_imaging/brain_netflix_scaling_data_to_reconstruct_videos_from_brain_signals.md)
- [\[ICCV 2025\] RadGPT: Constructing 3D Image-Text Tumor Datasets](radgpt_constructing_3d_image-text_tumor_datasets.md)
- [\[CVPR 2025\] Evidential learning driven Breast Tumor Segmentation with Stage-divided Vision-Language Interaction](../../CVPR2025/medical_imaging/evidential_learning_driven_breast_tumor_segmentation_with_stage-divided_vision-l.md)

</div>

<!-- RELATED:END -->
