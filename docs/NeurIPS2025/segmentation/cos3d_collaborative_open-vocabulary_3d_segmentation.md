---
title: >-
  [论文解读] COS3D: Collaborative Open-Vocabulary 3D Segmentation
description: >-
  [NeurIPS 2025][语义分割][开放词汇3D分割] 提出 COS3D——一种协作式 prompt-分割框架，通过构建实例场（instance field）和语言场（language field）组成的协作场，在训练阶段利用实例到语言的特征映射构建语言场，在推理阶段利用语言到实例的自适应 prompt 精炼生成精确分割，在两个主流基准上大幅超越现有方法。
tags:
  - "NeurIPS 2025"
  - "语义分割"
  - "开放词汇3D分割"
  - "3D高斯"
  - "协作式分割"
  - "实例-语言映射"
  - "提示学习"
---

# COS3D: Collaborative Open-Vocabulary 3D Segmentation

**会议**: NeurIPS 2025  
**arXiv**: [2510.20238](https://arxiv.org/abs/2510.20238)  
**代码**: [GitHub](https://github.com/Runsong123/COS3D)  
**领域**: 3D分割  
**关键词**: 开放词汇3D分割, 3D高斯, 协作式分割, 实例-语言映射, Prompt分割

## 一句话总结

提出 COS3D——一种协作式 prompt-分割框架，通过构建实例场（instance field）和语言场（language field）组成的协作场，在训练阶段利用实例到语言的特征映射构建语言场，在推理阶段利用语言到实例的自适应 prompt 精炼生成精确分割，在两个主流基准上大幅超越现有方法。

## 研究背景与动机

**开放词汇3D分割的挑战**：OV3DS 需要同时理解分割和语言两种信息。现有的基于 3D Gaussian Splatting 的方法分为两类，但各有局限：

1. **基于语言的方法**（LangSplat、LEGaussians、Dr.Splat）：直接从 2D 图像空间向 3D 语言场蒸馏 CLIP 特征。问题在于逐像素的语言蒸馏导致特征区分度不足，分割结果在边界处出现严重伪影和错误。

2. **基于分割的方法**（OpenGaussian、InstanceGaussian）：先做类无关的 3D 分割，再用 VLM 选择最匹配的 3D segment。问题在于：(a) 无语义线索时容易出现过分割/欠分割；(b) 手工设计的后匹配策略引入额外误差，导致错误累积。

**核心洞察**：两类信息互补——分割信息具有判别性且边界感知，语言信息促进对物体和场景的高层理解。实现 OV3DS 需要对两者的协同理解。

## 方法详解

### 整体框架

COS3D 基于 3D Gaussian Splatting 构建，包含三个技术组件：

1. **协作场（Collaborative Field）**：由实例场 $\Theta_I$ 和语言场 $\Theta_L$ 组成
2. **两阶段训练策略**：先学习实例场，再通过 Ins2Lang 映射构建语言场
3. **自适应推理精炼**：利用语言场生成 3D 相关性图作为 prompt，在实例场中精炼分割

### 关键设计

**1. 协作场的定义**

在每个 3D Gaussian $g_i = (p_i, s_i, q_i, o_i, c_i)$ 上扩展两个特征：
- **实例特征** $I \in \mathbb{R}^{d_I}$（$d_I=16$）：携带分割感知信息
- **语言特征** $L \in \mathbb{R}^{d_L}$（$d_L=512$，CLIP 维度）：携带语义信息

两个场在训练和推理中持续交互——训练阶段实例场辅助语言场构建，推理阶段语言场引导实例场分割。

**2. 实例到语言（Ins2Lang）映射**

构建从实例特征到语言特征的映射函数 $\Phi: L = \Phi(I)$。论文提供两种实现：

- **浅层 MLP**：学习映射函数 $\Phi_{\text{network}}$，损失为 $\mathcal{L}_{\text{mapping}} = |L^m - \Phi_{\text{network}}(I^m)|$，训练仅需 <3 分钟
- **核回归**：使用 Nadaraya-Watson 估计器，无需训练，$\sigma=0.1$

论文发现因为实例特征已具有判别性，映射任务本质上是简单的回归问题，核回归方法反而效果最优。

**3. 自适应语言到实例（Lang2Ins）Prompt 精炼**

推理时的分割流程：
1. 给定文本查询 $q_{\text{text}}$，用 CLIP 编码得到 $L_{\text{text}}$
2. 计算每个 3D Gaussian 的相关性得分 $R$，筛选高相关性点集 $\mathcal{S}$
3. **关键步骤**：以 $\mathcal{S}$ 为 prompt，在实例场中扩展邻域——找出实例特征余弦相似度超过阈值 $\mathcal{T}$ 的邻近点
4. 对扩展区域进行自适应过滤：计算区域级相关性（opacity加权平均），保留高于阈值 $\tau$ 的区域
5. 按相关性得分降序处理，逐步聚合得到最终分割 $\mathcal{S}_t$

### 损失函数 / 训练策略

**Stage 1 实例场训练**：使用 InfoNCE 对比学习损失

$$\mathcal{L}_{\text{ins}} = -\frac{1}{|\Omega|} \sum_{\Omega_j \in \Omega} \sum_{u \in \Omega_j} \log \frac{\exp(\text{sim}(I_u, \bar{I}_j))}{\sum_{\Omega_l \in \Omega} \exp(\text{sim}(I_u, \bar{I}_l))}$$

其中 $\Omega_j$ 是根据 SAM 2D 分割结果确定的同一实例的像素集合，$\bar{I}_j$ 是该实例的均值特征。

**Stage 2 语言场构建**：使用实例-CLIP 特征对进行映射学习。训练对在 SAM mask 级别构建（非像素级），减少冗余。

**两阶段策略的优势**：
- 相比一阶段联合学习：避免映射损失干扰实例特征空间，且训练时间减少 60%+
- 相比并行学习：能融合两个场的信息，性能显著更好

## 实验关键数据

### 主实验 (含表格)

**LeRF 数据集上的 3D 高斯分割**：

| 方法 | 类型 | mIoU | mAcc |
|-----|-----|:----:|:----:|
| LangSplat (CVPR'24) | 语言 | 9.66 | 12.41 |
| LEGaussians (CVPR'24) | 语言 | 16.21 | 23.82 |
| Dr.Splat (CVPR'25) | 语言 | 43.58 | 63.87 |
| OpenGaussian (NeurIPS'24) | 分割 | 38.36 | 51.43 |
| InstanceGaussian (CVPR'25) | 分割 | 45.30 | 58.44 |
| **COS3D (shallow MLPs)** | **协作 prompt** | **49.75** | **70.60** |
| **COS3D (kernel regression)** | **协作 prompt** | **50.76** | **72.08** |

**ScanNetv2 数据集**（19 类）：

| 方法 | mIoU | mAcc |
|-----|:----:|:----:|
| LangSplat | 3.78 | 9.11 |
| LEGaussians | 3.84 | 10.87 |
| OpenGaussian | 24.73 | 41.54 |
| **COS3D (kernel regression)** | **32.47** | **49.05** |

### 消融实验 (含表格)

**训练策略对比**（LeRF 数据集）：

| 学习方案 | mIoU | mAcc | 训练时间 |
|---------|:----:|:----:|:------:|
| 一阶段联合学习 | 49.15 | 69.19 | 165 min |
| 并行学习 | 43.84 | 59.81 | 95 min |
| 本文 (shallow MLPs) | 49.75 | 70.60 | 53 min |
| **本文 (kernel regression)** | **50.76** | **72.08** | **50 min** |

**推理策略对比**：

| 推理方案 | mIoU | mAcc | 查询时间 |
|---------|:----:|:----:|:------:|
| 仅实例分支 | 44.07 | 59.83 | 0.12s |
| 仅语言分支 | 48.99 | 71.31 | 0.13s |
| **协作 prompt (本文)** | **50.76** | **72.08** | 0.22s |

**训练效率**：仅用 8 分钟（3K 实例场迭代）即达到 mIoU 50.16，已超越所有基线。

### 关键发现

1. **协作策略显著优于单一策略**：无论是语言还是分割单点，都不如两者协作
2. **核回归优于 MLP**：因为判别性实例特征使映射变成简单回归任务
3. **训练效率极高**：50 分钟达到 SOTA，8 分钟已超越所有基线（LangSplat 需 240 分钟）
4. **兼容多种 2D VLM**：使用 SigLIP 替代 CLIP 或 SAM2 替代 SAM 可进一步提升性能
5. **扩展应用丰富**：支持图像查询的 3D 分割、层次化分割、机器人抓取

## 亮点与洞察

1. **协作场概念新颖**：将实例和语言信息的互补性形式化为协作场，并在训练和推理两个阶段都实现协作
2. **设计简洁高效**：核回归映射不需要训练，两阶段策略在训练时间上比联合学习快 3x+
3. **Prompt 精炼策略巧妙**：将语言场的粗略相关性图作为 prompt 引导实例场做精细分割，类似 SAM 的 prompt 范式
4. **边界质量显著提升**：定性结果显示分割物体边界更完整、伪影更少

## 局限与展望

1. **缺乏推理能力**：文本对齐的语言场无法处理关系型query（如"桌子上面的杯子"）或多物体查询
2. **离线设置**：当前框架需要完整的多视角图像集，未支持在线/增量场景
3. **依赖 SAM 质量**：实例场的训练监督来自 SAM 的 2D 分割，SAM 的错误会传播到 3D
4. **查询速度增加**：协作推理的查询时间（0.22s）比单分支（0.12-0.13s）慢约 70%
5. **大规模场景扩展**：在大规模室外场景中的表现未验证

## 相关工作与启发

- **LangSplat / LEGaussians / Dr.Splat**：基于语言特征蒸馏的方法，COS3D 的语言场构建更高效且效果更好
- **OpenGaussian / InstanceGaussian**：基于分割的方法，COS3D 通过协作避免了错误累积
- **SAM / Click-Gaussian**：prompt 分割范式，COS3D 将语言查询转化为实例场中的 prompt
- **3D Gaussian Splatting**：显式 3D 场景表示，COS3D 在此基础上扩展协作场
- 对 3D 理解的启示：**语言和分割的协作优于任一单独使用**，多模态信息的融合时机（训练还是推理）和方式是核心设计问题

## 评分

⭐⭐⭐⭐⭐ (5/5)

理由：论文的技术设计紧凑优雅——协作场的概念清晰、两阶段训练策略动机明确且高效、推理阶段的 prompt 精炼也是自然的设计延伸。实验结果在两个基准上大幅领先（LeRF 上 mIoU 50.76 vs 45.30），训练效率远超基线（50 分钟 vs 240 分钟），消融实验全面验证了每个组件的必要性。框架还展示了丰富的应用拓展（图像查询、层次化分割、机器人）。代码开源，复现友好。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] LangHOPS: Language Grounded Hierarchical Open-Vocabulary Part Segmentation](langhops_language_grounded_hierarchical_open-vocabulary_part_segmentation.md)
- [\[CVPR 2026\] GeoGuide: Hierarchical Geometric Guidance for Open-Vocabulary 3D Semantic Segmentation](../../CVPR2026/segmentation/geoguide_hierarchical_geometric_guidance_for_open-vocabulary_3d_semantic_segment.md)
- [\[NeurIPS 2025\] HumanCrafter: Synergizing Generalizable Human Reconstruction and Semantic 3D Segmentation](humancrafter_synergizing_generalizable_human_reconstruction_and_semantic_3d_segm.md)
- [\[CVPR 2025\] Exploring Simple Open-Vocabulary Semantic Segmentation](../../CVPR2025/segmentation/exploring_simple_open-vocabulary_semantic_segmentation.md)
- [\[CVPR 2025\] Effective SAM Combination for Open-Vocabulary Semantic Segmentation](../../CVPR2025/segmentation/effective_sam_combination_for_open-vocabulary_semantic_segmentation.md)

</div>

<!-- RELATED:END -->
