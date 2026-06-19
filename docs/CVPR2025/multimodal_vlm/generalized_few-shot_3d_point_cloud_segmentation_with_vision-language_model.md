---
title: >-
  [论文解读] Generalized Few-Shot 3D Point Cloud Segmentation with Vision-Language Model
description: >-
  [CVPR 2025][多模态VLM][点云分割] GFS-VL 提出一种广义小样本 3D 点云分割框架，通过将 3D 视觉语言模型（3D VLM）生成的稠密但有噪声的伪标签与精确但稀疏的小样本标注协同融合——经由原型引导的伪标签筛选、自适应填充和 novel-base 混合增强——在现有和新设的高难度 benchmark 上取得了 SOTA 性能。
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "点云分割"
  - "小样本学习"
  - "视觉语言模型"
  - "伪标签"
  - "3D语义理解"
---

# Generalized Few-Shot 3D Point Cloud Segmentation with Vision-Language Model

**会议**: CVPR 2025  
**arXiv**: [2503.16282](https://arxiv.org/abs/2503.16282)  
**代码**: [https://github.com/ZhaochongAn/GFS-VL](https://github.com/ZhaochongAn/GFS-VL)  
**领域**: 多模态VLM  
**关键词**: 点云分割, 小样本学习, 视觉语言模型, 伪标签, 3D语义理解

## 一句话总结

GFS-VL 提出一种广义小样本 3D 点云分割框架，通过将 3D 视觉语言模型（3D VLM）生成的稠密但有噪声的伪标签与精确但稀疏的小样本标注协同融合——经由原型引导的伪标签筛选、自适应填充和 novel-base 混合增强——在现有和新设的高难度 benchmark 上取得了 SOTA 性能。

## 研究背景与动机

**领域现状**：广义小样本 3D 点云分割（GFS-PCS）要求模型在少量新类样本（如 1-shot/5-shot）的条件下，同时分割基础类和新类。现有方法主要采用原型学习范式——每个类用一个原型表示，通过原型与查询点的关系进行分割。CAPL 利用共现先验增强原型，GW 编码几何结构作为辅助。

**现有痛点**：所有这些方法的核心瓶颈是——小样本数据太稀疏，无法提供足够的新类知识。仅靠 1-5 个支持样本来学习新类的原型和决策边界，信息量严重不足，导致新类分割精度远低于全监督上界。

**核心矛盾**：一方面，3D VLM（如 OpenScene、RegionPLC）通过对齐 3D 和语言特征，具备开放世界的新类识别能力，能为新类生成稠密的伪标签——但这些伪标签噪声很大，直接使用会引入错误累积。另一方面，小样本标注虽然精确，但覆盖面极窄。如何结合两者的优势——用稠密伪标签弥补稀疏标注的不足，同时用精确标注校准噪声伪标签——是核心挑战。

**本文目标** (1) 如何有效利用 3D VLM 的伪标签同时控制噪声？(2) 被过滤掉的噪声区域如何重新标注？(3) 如何充分利用宝贵的小样本支持样本？

**切入角度**：不把 3D VLM 当作独立分类器使用，而是把它当作"带噪声的标注器"，用小样本的精确原型来"校准"伪标签——筛掉与原型不一致的区域、填充缺失区域、增强训练场景。

**核心 idea**：用小样本支持原型引导筛选和修复 3D VLM 的伪标签，并通过 novel-base 混合增强在保留上下文的前提下将支持样本嵌入训练场景。

## 方法详解

### 整体框架

框架分三步：① 用 3D VLM 对训练场景生成包含所有类别的原始预测（伪标签）；② 通过"原型引导的伪标签筛选 → 自适应填充"流水线清洗伪标签；③ 通过 novel-base 混合增强将支持样本嵌入训练场景。清洗后的伪标签与原始基础类标注合并，用于训练一个简单的 backbone + 线性分类头的分割器。

### 关键设计

1. **原型引导的伪标签筛选（Prototype-guided Pseudo-label Selection）**:

    - 功能：过滤 3D VLM 预测中的低质量新类伪标签
    - 核心思路：先用 3D VLM 的视觉编码器从小样本支持样本中提取每个新类的支持原型 $\mathbf{p}^c$（masked average pooling）。然后对当前训练场景，用 3D VLM 生成原始预测 $\hat{\mathbf{Y}}$，对每个被预测为新类 $c$ 的区域同样计算预测原型 $\mathbf{u}^c$。若预测原型与支持原型的余弦相似度低于阈值 $\tau$，则该类的伪标签被过滤（标为 -1）；基础类预测直接过滤（因为有真实标注）
    - 设计动机：3D VLM 可能把背景误识别为新类、把一种新类混淆为另一种——但准确预测的区域其特征应与支持样本高度一致。这种基于原型一致性的过滤简单有效

2. **自适应填充（Adaptive Infilling）**:

    - 功能：为被过滤掉的未标注区域重新分配伪标签
    - 核心思路：被过滤的区域可能包含"完全错误的预测"（需要发现正确的新类）和"部分正确的预测"（需要补全不完整的 mask）。做法是：构建自适应原型集 $\{\mathbf{m}^c\}$——如果某新类在当前场景的筛后标签中存在，就用场景内的原型 $\mathbf{v}^c$；否则用支持原型 $\mathbf{p}^c$。然后对每个未标注点，计算其与所有新类原型的余弦相似度，若最大相似度超过阈值 $\delta$ 则分配对应类标签
    - 设计动机：用场景内原型（而非外部支持原型）来补全已有类更准确（因为来自同一场景的特征更相似），同时保留用支持原型发现完全缺失的新类的能力

3. **Novel-Base 混合增强（Novel-Base Mix）**:

    - 功能：将小样本支持样本直接嵌入训练场景，增加新类训练信号
    - 核心思路：随机采样一个支持样本，裁剪其新类物体的局部区域（保留周围上下文），然后将裁剪区域通过角点对齐"贴合"到训练场景的边缘。具体是分别提取训练场景和裁剪区域的 XY 平面四个角点，随机选一对对角点做对齐平移
    - 设计动机：与传统 3D 数据增强（如 Mix3D）不同，本方法强调**保留上下文**。直接把物体从原场景中扣出来放到任意位置会丢失空间上下文关系——但 GFS-PCS 中新类往往是难以检测的物体，上下文线索（如"水池旁边有马桶""桌子上有书"）对识别这些物体至关重要

### 训练策略

分两阶段：先在基础类数据上预训练 backbone 和基础类分类头（800 epochs），再添加新类分类头，用清洗后的伪标签和嵌入的支持样本微调 20 epochs。使用 Point Transformer V3 (PTv3) 作为 backbone。

## 实验关键数据

### 主实验（ScanNet200，5-shot/1-shot）

| 方法 | 5-shot mIoU-N | 5-shot HM | 1-shot mIoU-N | 1-shot HM |
|------|-------------|-----------|-------------|-----------|
| attMPTI | 4.99 | 8.79 | 3.28 | 6.17 |
| COSeg | 5.21 | 9.54 | 4.03 | 7.42 |
| GW | 8.30 | 14.55 | 6.47 | 11.56 |
| **GFS-VL** | **17.21** | **25.38** | **13.60** | **20.49** |
| 全监督上界 | 39.32 | 50.02 | 39.32 | 50.02 |

### 消融实验（ScanNet200, 1-shot）

| 配置 | mIoU-N | HM | 说明 |
|------|--------|-----|------|
| 无伪标签（基线） | 8.29 | 14.19 | 仅用支持样本微调 |
| + 原始伪标签 | 11.28 | 17.14 | 未筛选，有噪声 |
| + 伪标签筛选 | 12.04 | 18.61 | 去除低质量区域 |
| + 自适应填充 | 12.79 | 19.37 | 补全缺失区域 |
| + Novel-Base Mix | **13.60** | **20.49** | 完整模型 |

### 关键发现
- GFS-VL 在 ScanNet200（40 个新类）上新类 mIoU 达到 17.21（5-shot），是此前 SOTA（GW 8.30）的 2 倍以上
- 新引入的 ScanNet200 和 ScanNet++ benchmark 明显更具挑战性——40/18 个新类远多于现有 benchmark 的 6 个新类，且类别多样性更大
- 伪标签筛选的提升（11.28→12.04）相对温和，但自适应填充和 Mix 增强的叠加效果显著（12.04→13.60），说明填充缺失区域和增加训练信号同样重要
- 原始伪标签已经能提供显著增益（8.29→11.28），说明 3D VLM 的知识虽然有噪声但价值很大

## 亮点与洞察
- **"稠密噪声 + 稀疏精确"的协同范式**是本文最核心的贡献：不是选择 3D VLM 或小样本，而是让两者互补——用少量精确标注校准大量粗糙知识。这个思路可以迁移到任何"弱标注 + 强先验"的场景
- **保留上下文的数据增强**是一个被严重忽视的设计细节：大多数 3D 数据增强随机混合物体，但在小样本场景下上下文比物体本身更重要
- **新引入的 benchmark** 是实质性贡献：现有评估体系（6 个新类）过于简单，40/18 个新类的 benchmark 更接近真实世界的复杂性

## 局限与展望
- 依赖特定的 3D VLM（RegionPLC）质量——如果 3D VLM 对某些类别完全没有知识，伪标签就无法提供任何信号
- 阈值 $\tau$ 和 $\delta$ 需要为不同数据集手动调节
- 方法增加了训练复杂度：需要先运行 3D VLM 生成伪标签，再进行多步清洗，工程上不够简洁
- 在 1-shot 新类 mIoU 仅 13.60，与全监督的 39.32 仍有巨大差距，说明仍有很大改进空间
- 可以考虑在微调过程中动态更新伪标签（self-training 式的迭代），而非只做一次性的预处理

## 相关工作与启发
- **vs GW**: GW 通过共享几何结构来增强原型，但知识来源仍限于少量支持样本。GFS-VL 引入 3D VLM 作为外部知识源，信息量大幅提升
- **vs CAPL**: CAPL 利用共现先验和查询上下文，GFS-VL 用伪标签直接扩展训练集，思路更直接
- **vs OpenScene/RegionPLC**: 这些 3D VLM 做零样本分割但精度有限，GFS-VL 将它们作为"弱标注器"与少量精确标注结合，获得比任一者更好的效果

## 评分
- 新颖性: ⭐⭐⭐⭐ 将 3D VLM 伪标签与小样本协同的思路新颖且合理，但各子模块相对独立缺乏统一优化
- 实验充分度: ⭐⭐⭐⭐⭐ 4 个 benchmark（含 2 个新设）、多个基线对比、详细消融、可视化
- 写作质量: ⭐⭐⭐⭐ 框架图清晰，动机阐述充分，但公式符号稍多
- 价值: ⭐⭐⭐⭐ 新 benchmark 有长期影响，方法在 3D 小样本领域有引领性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] UNEM: UNrolled Generalized EM for Transductive Few-Shot Learning](unem_unrolled_generalized_em_for_transductive_few-shot_learning.md)
- [\[CVPR 2025\] Rethinking Few-Shot Adaptation of Vision-Language Models in Two Stages](rethinking_few-shot_adaptation_of_vision-language_models_in_two_stages.md)
- [\[ICCV 2025\] Exploiting Vision Language Model for Training-Free 3D Point Cloud OOD Detection](../../ICCV2025/multimodal_vlm/exploiting_vision_language_model_for_training-free_3d_point_cloud_ood_detection_.md)
- [\[CVPR 2025\] Visual and Semantic Prompt Collaboration for Generalized Zero-Shot Learning](visual_and_semantic_prompt_collaboration_for_generalized_zero-shot_learning.md)
- [\[CVPR 2025\] Your Large Vision-Language Model Only Needs a Few Attention Heads for Visual Grounding](your_large_vision-language_model_only_needs_a_few_attention_heads_for_visual_gro.md)

</div>

<!-- RELATED:END -->
