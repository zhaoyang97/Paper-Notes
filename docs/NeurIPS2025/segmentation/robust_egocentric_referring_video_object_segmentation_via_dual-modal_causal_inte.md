---
title: >-
  [论文解读] Robust Egocentric Referring Video Object Segmentation via Dual-Modal Causal Intervention
description: >-
  [NeurIPS 2025][语义分割][第一人称视频分割] 提出 CERES 框架，通过双模态因果干预（语言后门调整消除数据集统计偏差 + 视觉前门调整利用深度信息构建因果中介变量）来解决第一人称视频指代分割中的语言偏差和视觉混淆问题，在 VISOR/VOST/VSCOS 上取得 SOTA。 领域现状：第一人称指代视频目标…
tags:
  - "NeurIPS 2025"
  - "语义分割"
  - "第一人称视频分割"
  - "因果推断"
  - "后门调整"
  - "前门调整"
  - "深度引导"
---

# Robust Egocentric Referring Video Object Segmentation via Dual-Modal Causal Intervention

**会议**: NeurIPS 2025  
**arXiv**: [2512.24323](https://arxiv.org/abs/2512.24323)  
**代码**: 无  
**领域**: 视频理解 / 分割  
**关键词**: 第一人称视频分割, 因果推断, 后门调整, 前门调整, 深度引导

## 一句话总结

提出 CERES 框架，通过双模态因果干预（语言后门调整消除数据集统计偏差 + 视觉前门调整利用深度信息构建因果中介变量）来解决第一人称视频指代分割中的语言偏差和视觉混淆问题，在 VISOR/VOST/VSCOS 上取得 SOTA。

## 研究背景与动机

**领域现状**：第一人称指代视频目标分割（Ego-RVOS）要求根据自然语言描述（如"knife used to cut carrot"）在第一人称视频中分割正在参与特定动作的物体。现有方法如 ActionVOS 在预训练的 RVOS 模型上微调，加入动作描述来区分正样本和负样本物体。

**现有痛点**：现有方法容易学到虚假关联而非因果关系，导致鲁棒性差。具体表现为两个层面：(1) **语言偏差**——数据集中"knife-cut"这类物体-动作对频繁共现，模型依赖统计捷径而非真正理解指令；(2) **视觉混淆**——第一人称视角固有的快速运动、频繁遮挡和透视畸变使模型容易被误导，特别是从第三人称预训练数据迁移时存在域偏移。

**核心矛盾**：语言偏差来自可观测的数据集统计偏差（可用后门调整），视觉混淆来自不可观测的固有因素（需要前门调整），但现有因果学习框架要么只处理后门、要么只处理前门，缺乏统一方案。

**本文目标** 在一个统一框架中同时处理两种不同性质的混淆因素——可观测的语言偏差和不可观测的视觉偏差。

**切入角度**：作者从结构因果模型（SCM）出发，将 Ego-RVOS 建模为包含两类混淆因素的因果图，分别对文本路径和视觉路径施加不同的因果干预。关键洞察是利用深度信息构建对第一人称视觉畸变更鲁棒的中介变量。

**核心 idea**：用后门调整消除语言统计偏差、用基于深度引导的前门调整绕过不可观测的视觉混淆因素，双管齐下实现鲁棒的第一人称视频分割。

## 方法详解

### 整体框架

CERES 是一个即插即用的因果框架，建立在预训练的 RVOS 模型（如 ReferFormer）之上。输入是第一人称视频帧序列和自然语言查询，输出是逐帧的分割掩码。框架包含两个核心模块：(1) 语言后门去混淆器（LBD）处理文本表示；(2) 视觉前门去混淆器（VFD）处理视觉特征。两者分别对文本和视觉通路进行因果干预后，将去偏的多模态特征送入 RVOS 模型的后续组件（分类头、正样本判别头、掩码解码器）生成最终分割结果。

### 关键设计

1. **语言后门去混淆器（LBD）**:

    - 功能：消除由数据集统计偏差导致的语言-分割虚假关联
    - 核心思路：基于 Pearl 的后门调整，估计干预分布 $P(\mathcal{Y}|\text{do}(\mathcal{T}))$。首先从训练集中统计所有唯一的"动词-名词"对（如 cut-knife）作为混淆因子字典 $\{z_i\}_{i=1}^K$，用文本编码器获取各混淆因子的嵌入 $\mathbf{f}_\mathcal{Z}(z_i)$，按经验频率 $P(z_i)$ 加权平均得到固定向量 $\bar{\mathbf{f}}_\mathcal{Z}$。推理时将原始文本特征加上这个常量偏置得到去偏表示：$\mathbf{f}'_\mathcal{T}(t) = \mathbf{f}_\mathcal{T}(t) + \bar{\mathbf{f}}_\mathcal{Z}$。利用 NWGM 近似将 Softmax 期望简化为期望的 Softmax，并假设得分可加性分解
    - 设计动机：混淆因子 $\mathcal{Z}$ 是可观测的（来自训练数据统计），后门调整能直接切断 $\mathcal{T} \leftarrow \mathcal{Z} \rightarrow \mathcal{Y}$ 的虚假路径。实现上非常轻量——只需一次预计算加一次加法

2. **视觉前门去混淆器（VFD）— 深度引导注意力（DAttn）**:

    - 功能：构建不受第一人称视觉畸变影响的因果中介变量
    - 核心思路：从视觉输入 $\mathcal{X}$ 提取两种特征——语义视觉特征 $\mathcal{M}_v$（来自 RGB 编码器）和几何深度特征 $\mathcal{M}_d$（来自冻结的 Depth Anything V2 编码器）。核心是用深度特征作为 Query、视觉特征作为 Key/Value 进行跨模态注意力：$\hat{\mathbf{M}}(x) = \text{Attn}(Q=\hat{\mathbf{M}}_d, K=V=\mathbf{M}_v)$，这实际上是在 Attention-Linear-Family 下的 MMSE 估计。深度信息对运动模糊和遮挡本身更鲁棒，用它来引导语义特征的聚合可以减少混淆因素 $\mathcal{U}$ 的影响
    - 设计动机：视觉混淆因素 $\mathcal{U}$ 不可观测（快速运动、遮挡等固有属性），无法用后门调整。前门调整需要一个好的中介变量，纯视觉中介容易继承 $\mathcal{U}$ 的影响，而几何深度信息对这些畸变更稳定，作为"引导者"帮助筛选可靠的语义特征

3. **视觉前门去混淆器（VFD）— 时序记忆注意力（MAttn）**:

    - 功能：估计视觉输入的一般性上下文分布 $\mathbb{E}_{\mathcal{X}'}[\mathbf{X}']$，完成前门调整公式中的第二个期望
    - 核心思路：维护一个滑动窗口记忆库 $\mathcal{B}_t = \{x_{t-\tau}\}_{\tau=1}^W$（$W=5$），假设短时间窗口内帧分布近似平稳（短程平稳性假设），用当前帧与历史帧的注意力加权来近似期望。这比全局字典方案更适合动态的第一人称视频流，且在大数法则下收敛到真实期望
    - 设计动机：标准前门调整实现中用静态全局字典预计算视觉上下文的期望，但这对长时间动态变化的第一人称视频不切实际。滑动窗口方案既保证理论一致性又适应时序动态

### 损失函数 / 训练策略

总损失采用标准分割损失。训练时对 RGB 编码器最后 3 层的特征分别计算辅助分割损失，以提供更丰富的训练信号。推理时只使用最后一层的去偏视觉特征进行预测。视觉和中介特征通过门控残差连接融合：$\mathbf{f}'_\mathcal{X}(x_t) = \sigma \cdot \text{MLP}([\hat{\mathbf{M}}; \hat{\mathbf{X}}_t]) + (1-\sigma) \cdot \mathbf{X}_t$。

## 实验关键数据

### 主实验

| 数据集 | 方法 | mIoU⊕↑ | cIoU⊕↑ | gIoU↑ | Acc↑ |
|--------|------|---------|---------|-------|------|
| VISOR (R101) | ActionVOS | 59.9 | 67.2 | 69.9 | 73.4 |
| VISOR (R101) | **CERES** | **64.0** | **72.8** | **72.4** | **76.3** |
| VISOR (SwinL) | ActionVOS | 66.3 | 71.9 | 68.7 | 73.4 |
| VISOR (SwinL) | **CERES** | **67.0** | **73.6** | **71.8** | **75.2** |
| VISOR-Novel | ActionVOS | 55.3 | 62.8 | 65.8 | 69.4 |
| VISOR-Novel | **CERES** | **60.0** | **69.9** | **67.9** | **72.2** |
| VSCOS | ActionVOS | 52.5 | 57.7 | — | — |
| VSCOS | **CERES** | **55.3** | **62.5** | — | — |
| VOST | ActionVOS | 30.2 | 17.6 | — | — |
| VOST | **CERES** | **32.0** | **21.7** | — | — |

### 消融实验

| 配置 | mIoU⊕↑ | mIoU⊖↓ | gIoU↑ | Acc↑ |
|------|---------|---------|-------|------|
| Baseline (ActionVOS) | 59.9 | 16.3 | 69.9 | 73.4 |
| + LBD only | 61.2 | 16.0 | 71.4 | 74.8 |
| + DAttn (MLP depth) | 62.1 | 17.5 | 70.5 | 73.6 |
| + DAttn (cross-attn) | 63.3 | 15.8 | 71.8 | 75.3 |
| + DAttn + MAttn | 63.1 | **14.9** | 72.1 | 76.1 |
| Full CERES | **64.0** | 15.3 | **72.4** | **76.3** |

### 关键发现

- DAttn（跨注意力深度融合）贡献最大，单独引入即提升 mIoU⊕ 3.4%，且优于 MLP 深度融合方式，验证了因果中介设计的优越性
- MAttn 的加入使 mIoU⊖ 降到最低（14.9%），说明时序上下文有助于更好地区分正/负样本
- 在罕见概念子集上 CERES 比 ActionVOS 提升 3.9% mIoU⊕（62.3% vs 58.4%），证实 LBD 对统计偏差的有效消除
- 时序窗口 $W=5$ 是最优平衡点，再增大收益递减

## 亮点与洞察

- **双模态因果框架的统一性**：同一框架中同时处理可观测（语言偏差→后门）和不可观测（视觉混淆→前门）两类完全不同性质的混淆因素，比之前只处理单一类型的因果方法更全面
- **深度信息作为因果中介的巧妙运用**：不是简单地拼接 RGB+Depth，而是从因果理论出发论证深度特征对视觉混淆更鲁棒，用它引导语义特征聚合。这个思路可以迁移到任何需要处理域偏移的视觉任务
- **即插即用设计**：CERES 作为插件模块兼容多种 RVOS backbone（R101/VSwinB/SwinL），工程实用性强

## 局限与展望

- LBD 的混淆因子字典依赖训练集的"动词-名词"对统计，在真正开放词汇场景下可能不够灵活
- 前门调整中的短程平稳性假设在场景快速切换时可能不成立
- 仅在厨房场景数据集上验证，泛化到其他第一人称场景（如户外活动、工业操作）的效果未知
- 深度信息依赖预训练的单目深度估计模型（Depth Anything V2），在极端场景下深度估计本身可能不可靠

## 相关工作与启发

- **vs ActionVOS**：ActionVOS 引入动作描述但仍然学习虚假关联，CERES 通过因果干预从根本上消除偏差来源，在所有指标上一致超越
- **vs GOAT**：GOAT 用因果方法处理视觉/语言/动作历史的混淆，但不是针对 RVOS 设计的。CERES 首次将双模态因果干预引入 RVOS 并加入深度中介
- **vs 单一因果调整方法**：现有工作要么只用后门（如视觉字幕生成）、要么只用前门（如 VQA），CERES 首次在同一框架中统一两者

## 评分

- 新颖性: ⭐⭐⭐⭐ 双模态因果干预框架是新的，但后门/前门调整各自在其他任务中已有应用
- 实验充分度: ⭐⭐⭐⭐ 三个数据集+三个backbone+详细消融，但缺少更多样的场景验证
- 写作质量: ⭐⭐⭐⭐ 因果建模部分推导清晰，理论和实现对应关系明确
- 价值: ⭐⭐⭐⭐ 对第一人称视频理解领域有实际价值，因果框架设计可迁移

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] CoLA: Conditional Dropout and Language-Driven Robust Dual-Modal Salient Object Detection](../../ECCV2024/segmentation/cola_conditional_dropout_and_language-driven_robust_dual-modal_salient_object_de.md)
- [\[NeurIPS 2025\] UniPixel: Unified Object Referring and Segmentation for Pixel-Level Visual Reasoning](unipixel_unified_object_referring_and_segmentation_for_pixel-level_visual_reason.md)
- [\[ICCV 2025\] ReferDINO: Referring Video Object Segmentation with Visual Grounding Foundations](../../ICCV2025/segmentation/referdino_referring_video_object_segmentation_with_visual_grounding_foundations.md)
- [\[NeurIPS 2025\] OmniSegmentor: A Flexible Multi-Modal Learning Framework for Semantic Segmentation](omnisegmentor_a_flexible_multi-modal_learning_framework_for_semantic_segmentatio.md)
- [\[NeurIPS 2025\] Towards Robust Pseudo-Label Learning in Semantic Segmentation: An Encoding Perspective](towards_robust_pseudo-label_learning_in_semantic_segmentation_an_encoding_perspe.md)

</div>

<!-- RELATED:END -->
