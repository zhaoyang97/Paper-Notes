---
title: >-
  [论文解读] OpenScan: A Benchmark for Generalized Open-Vocabulary 3D Scene Understanding
description: >-
  [AAAI 2026][3D视觉][Open-Vocabulary 3D] 本文提出了广义开放词汇 3D 场景理解任务（GOV-3D）及对应的 OpenScan 基准，将 3D 场景理解从物体类别扩展到八种语言学属性维度，揭示了现有 OV-3D 方法在理解抽象物体属性方面的严重不足。 领域现状 开放词汇 3D 场景理解（OV…
tags:
  - "AAAI 2026"
  - "3D视觉"
  - "Open-Vocabulary 3D"
  - "属性理解"
  - "3D场景分割"
  - "benchmark"
  - "知识图谱"
---

# OpenScan: A Benchmark for Generalized Open-Vocabulary 3D Scene Understanding

**会议**: AAAI 2026  
**arXiv**: [2408.11030](https://arxiv.org/abs/2408.11030)  
**代码**: [https://youjunzhao.github.io/OpenScan/](https://youjunzhao.github.io/OpenScan/)  
**领域**: 3D场景理解/开放词汇  
**关键词**: Open-Vocabulary 3D, 属性理解, 3D场景分割, benchmark, 知识图谱

## 一句话总结

本文提出了广义开放词汇 3D 场景理解任务（GOV-3D）及对应的 OpenScan 基准，将 3D 场景理解从物体类别扩展到八种语言学属性维度，揭示了现有 OV-3D 方法在理解抽象物体属性方面的严重不足。

## 研究背景与动机

### 领域现状

开放词汇 3D 场景理解（OV-3D）旨在定位和分类训练集之外的新物体。近年来，借助 CLIP 等视觉-语言模型（VLMs），OV-3D 在物体类别级别的识别上取得了显著进展。代表性方法如 OpenMask3D、SAI3D、MaskClustering、Open3DIS 等在 ScanNet200 上表现优异。

### 现有痛点

现有方法和基准（ScanNet、ScanNet200）**仅关注物体类别**层面的开放词汇问题。然而，AI 系统对物体相关属性（如功能、材质、特性）的理解同样至关重要。例如机器人需要理解"可以坐的东西"（功能属性）而不仅仅是"椅子"（类别标签）。

### 核心矛盾

缺乏大规模的 3D 场景属性标注基准，导致无法系统性地评估 OV-3D 模型在物体属性理解方面的泛化能力。已有基准只包含物体类别标注，不包含属性标注。

### 本文目标

构建一个**超越物体类别**的综合评测基准，从多种语言学维度评估 OV-3D 模型对抽象物体属性的理解能力。

### 切入角度

引入 GOV-3D（Generalized Open-Vocabulary 3D Scene Understanding）任务，将查询从物体类别扩展到物体相关的抽象属性。基于 ScanNet200 构建 OpenScan 基准，通过知识图谱和人工标注相结合获取属性标注。

### 核心 idea

**物体类别识别只是 3D 场景理解的冰山一角**，真正的开放词汇理解应包含功能、材质、属性等多个语言学维度的抽象概念。

## 方法详解

### 整体框架

OpenScan 基准构建流程：
1. **知识图谱关联**：利用 ConceptNet 为 ScanNet200 中的 200 个物体类别建立与各类属性的关联
2. **人工标注**：对视觉属性（如材质）进行人工标注
3. **属性分类**：将属性归入八个语言学维度
4. **属性验证**：人工验证确保语义一致性
5. **查询生成**：生成隐藏物体名称的文本查询

### 关键设计一：八维语言学属性体系

**功能**：将物体属性划分为八个代表性语言学维度。

**具体维度**：
- **Affordance（功能）**：物体的功能或用途，如椅子的"sit"
- **Property（特性）**：物体特征，如枕头的"soft"
- **Type（类型）**：所属类别，如电话是"communication device"
- **Manner（行为方式）**：使用方式，如帽子"worn on a head"
- **Synonym（同义词）**：近义替换，如图片的"image"
- **Requirement（条件）**：必要条件，如自行车需要"balance to ride"
- **Element（组成部分）**：构成元素，如自行车有"two wheels"
- **Material（材质）**：材料类型，如瓶子的"plastic"

**设计动机**：这八个维度覆盖了从常识知识（功能、条件）到视觉知识（材质）的多层理解，能全面评估模型对物体的深度理解能力。

### 关键设计二：知识图谱驱动的标注生成

**功能**：利用 ConceptNet 自动生成物体与属性的关联标注。

**核心思路**：对 ScanNet200 中每个物体类别 $c_i$，从知识图谱 $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ 中查询相关边：
$$\{e\}_i = \{(v_m, r, w, v_n) \in \mathcal{E} | v_m = c_i\}$$

对同一关系 $r$ 保留权重 $w$ 最高的属性，确保每个物体在每个方面只保留最具代表性的属性。

**设计动机**：知识图谱提供了结构化、可扩展的常识知识来源，能以较低成本为大量物体生成属性标注。

### 关键设计三：查询模板设计

**功能**：生成隐藏物体名称的文本查询，用于评估 GOV-3D 任务。

**核心思路**：将查询中的物体类别 $v_m$ 替换为"this term"，然后拼接关系和属性：
$$q = \text{Concatenate}(t, r, v_n)$$
例如"this term is made of wood"。

**设计动机**：查询中不包含物体名称，强制模型**通过属性推理**来定位物体，而非简单的名称匹配。

### 损失函数

基准本身不涉及训练损失，评估指标采用标准 OV-3D 指标：实例分割用 AP/AP50/AP25，语义分割用 mIoU/mAcc。

## 实验关键数据

### 主实验：3D 实例分割

| 方法 | Affordance | Property | Synonym | Material | Mean | ScanNet200 |
|------|-----------|----------|---------|----------|------|------------|
| OpenMask3D | 7.2 | 7.5 | 16.9 | 18.8 | 9.9 | 15.4 |
| SAI3D | 5.3 | 5.8 | 10.0 | 11.3 | 7.7 | 12.7 |
| MaskClustering | 6.2 | 7.0 | 16.2 | 12.1 | 8.1 | 12.0 |
| **Open3DIS** | **11.9** | **12.8** | **26.7** | **28.3** | **15.8** | **23.7** |

（AP 指标，所有方法在 OpenScan 上的性能均**显著低于** ScanNet200）

### 3D 语义分割

| 方法 | OpenScan mIoU | OpenScan mAcc | ScanNet mIoU |
|------|--------------|---------------|--------------|
| OpenScene | 0.45 | 1.87 | 47.5 |
| PLA | 0.01 | 2.37 | 66.6 |
| RegionPLC | 0.07 | 2.36 | 68.7 |

语义分割方法在 OpenScan 上几乎完全失败（mIoU < 1%），表明**从类别到属性的泛化严重不足**。

### 消融实验：预训练词汇量影响

增大训练词汇量（$S = 10 \to 170$）对大多数属性维度**没有显著提升**，仅 material 维度有轻微改善。说明**简单扩大训练类别数量无法解决属性理解问题**。

### 关键发现

1. 所有 OV-3D 模型在 OpenScan 上的表现远逊于 ScanNet200，证实 GOV-3D 是更具挑战性的任务
2. **Synonym 和 Material** 表现相对较好：前者因与物体类别语义接近，后者因 CLIP 的视觉模式识别能力
3. **Affordance 和 Property** 最具挑战：需要常识推理能力，CLIP 的预训练目标未涵盖
4. 使用查询模板（含关系描述）比单纯属性词能提升 AP 约 0.2-6 个点

## 亮点与洞察

1. **问题定义有远见**：将 OV-3D 从类别扩展到属性，是一个自然但此前被忽视的研究方向
2. **基准设计系统化**：八维属性体系覆盖全面，知识图谱+人工标注的混合策略兼顾效率与质量
3. **实验发现揭示根本局限**：证明了简单扩大训练词汇量无法解决属性理解问题，指向了更深层的方法论变革需求
4. **规模可观**：153,644 条属性标注，341 个属性，平均每个物体 3.15 个属性标注

## 局限与展望

1. 常识属性标注依赖 ConceptNet，其覆盖面和质量可能受限
2. 视觉属性只标注了材质一个维度，颜色、形状等视觉属性未覆盖
3. 仅在 ScanNet200 上构建，场景类型局限于室内
4. 未提出针对 GOV-3D 任务的解决方法，仅暴露了问题
5. 八个属性维度的选择标准未充分论证，可能存在遗漏或重叠
6. 评估时假设查询的目标物体在场景中存在，但 GOV-3D 任务声称需要判断是否存在

## 相关工作与启发

1. **OpenScene** (Peng et al. 2023)：支持任意文本查询的零样本 3D 语义分割，但缺乏属性维度的定量评估
2. **SceneFun3D** (Delitzas et al. 2024)：机器人交互场景的功能性标注，但仅关注 affordance 维度
3. **MMScan** (Lyu et al. 2024)：视觉属性理解基准，缺少常识属性
4. **启发**：视觉-语言模型在常识推理方面的不足提示我们，仅靠图文对齐不够，可能需要引入结构化知识或多步推理能力

## 评分

⭐⭐⭐⭐ (4/5)

**优势**：问题定义有价值，基准构建扎实全面，实验发现具有重要指导意义。

**不足**：未提出解决方案，部分标注设计选择缺乏充分论证。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] OpenLex3D: A Tiered Evaluation Benchmark for Open-Vocabulary 3D Scene Representations](../../NeurIPS2025/3d_vision/openlex3d_a_tiered_evaluation_benchmark_for_open-vocabulary_3d_scene_representat.md)
- [\[ICCV 2025\] Open-Vocabulary Octree-Graph for 3D Scene Understanding](../../ICCV2025/3d_vision/open-vocabulary_octree-graph_for_3d_scene_understanding.md)
- [\[CVPR 2026\] LightSplat: Fast and Memory-Efficient Open-Vocabulary 3D Scene Understanding in Five Seconds](../../CVPR2026/3d_vision/lightsplat_fast_and_memory-efficient_open-vocabulary_3d_scene_understanding_in_f.md)
- [\[CVPR 2026\] OpenVoxel: Training-Free Grouping and Captioning Voxels for Open-Vocabulary 3D Scene Understanding](../../CVPR2026/3d_vision/openvoxel_training-free_grouping_and_captioning_voxels_for_open-vocabulary_3d_sc.md)
- [\[CVPR 2026\] EmbodiedSplat: Online Feed-Forward Semantic 3DGS for Open-Vocabulary 3D Scene Understanding](../../CVPR2026/3d_vision/embodiedsplat_online_feed-forward_semantic_3dgs_for_open-vocabulary_3d_scene_und.md)

</div>

<!-- RELATED:END -->
