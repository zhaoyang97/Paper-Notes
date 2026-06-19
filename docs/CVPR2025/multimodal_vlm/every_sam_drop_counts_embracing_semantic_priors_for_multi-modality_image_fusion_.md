---
title: >-
  [论文解读] Every SAM Drop Counts: Embracing Semantic Priors for Multi-Modality Image Fusion and Beyond
description: >-
  [CVPR 2025][多模态VLM][红外-可见光融合] 利用 SAM 的语义先验通过持久注意力模块增强红外-可见光图像融合，再通过双层优化知识蒸馏将语义知识转移到仅 0.136M 参数的轻量子网络，实现无需 SAM 的 10.47ms 推理，同时在分割任务上超越所有专用融合方法 3+ mIoU。
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "红外-可见光融合"
  - "SAM语义先验"
  - "知识蒸馏"
  - "双层优化"
  - "轻量模型"
---

# Every SAM Drop Counts: Embracing Semantic Priors for Multi-Modality Image Fusion and Beyond

**会议**: CVPR 2025  
**arXiv**: [2503.01210](https://arxiv.org/abs/2503.01210)  
**代码**: [https://github.com/RollingPlain/SAGE_IVIF](https://github.com/RollingPlain/SAGE_IVIF)  
**领域**: 多模态VLM  
**关键词**: 红外-可见光融合、SAM语义先验、知识蒸馏、双层优化、轻量模型

## 一句话总结
利用 SAM 的语义先验通过持久注意力模块增强红外-可见光图像融合，再通过双层优化知识蒸馏将语义知识转移到仅 0.136M 参数的轻量子网络，实现无需 SAM 的 10.47ms 推理，同时在分割任务上超越所有专用融合方法 3+ mIoU。

## 研究背景与动机

**领域现状**：红外-可见光图像融合（IVIF）将两种模态的互补信息合成一张图像，服务于下游的目标检测和语义分割。现有融合方法主要关注视觉质量（对比度、清晰度），对下游任务的适配不够。

**现有痛点**：(1) 纯视觉质量优化的融合图像在分割任务上不一定最优——某些视觉好看的融合结果反而会干扰分割器。(2) 引入大型预训练模型（如 SAM）作为语义先验可以提升任务性能，但 SAM 在推理时的巨大计算开销不可接受。

**核心矛盾**：需要语义级指导来提升融合图像的下游任务友好性，但语义模型（SAM, 632M 参数）在推理时不可承受。

**本文目标** 利用 SAM 的语义先验提升融合质量和下游任务性能，同时通过知识蒸馏消除推理时对 SAM 的依赖。

**切入角度**：SAM 只在训练时使用——生成语义 patch 指导主网络（教师），主网络的知识再蒸馏到轻量子网络（学生），推理时只用学生网络。

**核心 idea**：用 SAM 做"训练时的语义教练"，通过双层优化蒸馏三元组损失将其知识压缩到 0.136M 的超轻量融合网络。

## 方法详解

### 整体框架
训练时：SAM 从红外/可见光图像生成语义 patch → 主网络（教师）通过 SPA 模块融合源图像 + 语义 patch → 生成参考融合图 $I_{ref}$。同时，轻量子网络（学生）仅从源图像生成融合图 $I_f$。双层优化交替更新教师和学生，三元组损失对齐两者。推理时：仅用学生网络。

### 关键设计

1. **语义持久注意力模块（SPA）**:

    - 功能：将 SAM 的语义 patch 与源图像的完整信息融合
    - 核心思路：SPA 包含一个持久存储库（PR），存储源图像特征的潜在表示 $Z$ 和键值对 $(K_{src}, V_{src})$。SAM 语义 patch 编码后作为 query，在 PR 的键值对上做 cross-attention，将不完整的语义 patch 补充为包含全场景上下文的融合特征
    - 设计动机：SAM 的语义 patch 只覆盖部分场景（单个分割区域），PR 提供了完整场景的锚点，防止局部语义信息丢失全局上下文

2. **三元组蒸馏损失**:

    - 功能：从三个维度对齐教师和学生的输出
    - 核心思路：$\mathcal{L}_{fea}$（特征对齐）：教师和学生对应尺度的特征余弦相似度；$\mathcal{L}_{context}$（上下文）：Sobel 梯度 L1 + MSE，保持结构和亮度一致；$\mathcal{L}_{cs}$（对比语义）：用 SAM 编码器的特征和 binary mask 构建正负样本对，拉近融合图和参考图的语义表征
    - 设计动机：消融显示 $\mathcal{L}_{fea}$ 去掉后 FMB mIoU 掉 9.6 个点（最关键），三种损失各攻其面缺一不可

3. **双层优化蒸馏（DARTS 式）**:

    - 功能：教师和学生交替更新，实现双向适应
    - 核心思路：不是先训完教师再蒸馏到学生（offline），而是 DARTS 式交替——教师网络有额外的分割损失 $\mathcal{L}_{seg}$，学生网络通过三元组损失跟随教师。双向梯度使教师也能根据学生的反馈调整
    - 设计动机：离线蒸馏 FMB mIoU 仅 50.4，双层优化达 61.2（+10.8），证明互适应至关重要

### 损失函数 / 训练策略
教师：$\mathcal{L}_{seg}$（X-Decoder 的 CE 分割损失）。蒸馏：$\mathcal{L} = \mathcal{L}_{fea} + \mathcal{L}_{context} + \mathcal{L}_{cs}$。DARTS 式交替优化。

## 实验关键数据

### 主实验

| 方法 | FMB mIoU | MFNet mIoU | 推理时间 | 参数量 |
|------|---------|----------|---------|--------|
| DDFM | 58.2 | - | 280K ms | 552.7M |
| SegMiF | 57.3 | 52.4 | - | - |
| EMMA | 55.8 | - | 25.73ms | 1.516M |
| **SAGE** | **61.2** | **54.0** | **10.47ms** | **0.136M** |

### 消融实验

| 配置 | FMB mIoU | 说明 |
|------|---------|------|
| 无 SAM | 56.5 | -4.7 |
| 无 $\mathcal{L}_{fea}$ | 51.6 | -9.6，最关键 |
| 无 $\mathcal{L}_{cs}$ | 54.3 | -6.9 |
| 离线蒸馏 | 50.4 | -10.8，双层优化至关重要 |
| **完整 SAGE** | **61.2** | 所有组件协同 |

### 关键发现
- **SAM 语义先验价值 +4.7 mIoU**，证明高级语义指导对融合质量和下游任务的重要性
- **极致效率**：0.136M 参数、10.47ms，比 DDFM 快 26700×、小 4000×，比 EMMA 快 2.5×、小 11×
- **双层优化 >> 离线蒸馏**（+10.8 mIoU），交替训练使教师和学生协同进化

## 亮点与洞察
- **"训练用大模型、推理用小模型"的范式**非常实用——SAM 的语义知识被完全压缩到 0.136M 的网络中，这种思路可以推广到任何需要大模型能力但受限于推理预算的场景
- **持久存储库（PR）的设计**解决了局部语义 patch 缺失全局上下文的问题，本质上是一种跨注意力的记忆增强机制

## 局限与展望
- SAM 生成的语义 patch 质量直接影响蒸馏效果，低质量分割可能引入噪声
- 仅在红外-可见光融合上验证，对其他多模态融合（如 CT-MRI、多光谱）的泛化未确认
- 分割损失依赖 X-Decoder，换其他开放词汇分割模型的效果未知

## 相关工作与启发
- **vs SegMiF**：同时优化融合和分割，但 SegMiF 将分割器直接融入推理管线增加开销。SAGE 通过蒸馏在训练时结合分割指导、推理时完全不用
- **vs EMMA**：EMMA 用更大模型（1.516M）但效果更差（55.8 vs 61.2 mIoU），说明模型大小不是关键，语义先验的引入方式更重要

## 评分
- 新颖性: ⭐⭐⭐⭐ SAM 先验 + 双层蒸馏的组合新颖且实用，SPA 模块设计精心
- 实验充分度: ⭐⭐⭐⭐⭐ 多数据集评估、详尽的逐组件消融（SAM/SPA/蒸馏损失/蒸馏方式）
- 写作质量: ⭐⭐⭐⭐ 方法流程清晰，效率数据有说服力
- 价值: ⭐⭐⭐⭐ 对多模态融合领域有直接工程价值，蒸馏思路可推广

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] ASAP: Advancing Semantic Alignment Promotes Multi-Modal Manipulation Detecting and Grounding](asap_advancing_semantic_alignment_promotes_multi-modal_manipulation_de.md)
- [\[CVPR 2025\] Post-pre-training for Modality Alignment in Vision-Language Foundation Models](post-pre-training_for_modality_alignment_in_vision-language_foundation_models.md)
- [\[CVPR 2025\] Multi-Layer Visual Feature Fusion in Multimodal LLMs: Methods, Analysis, and Best Practices](multi-layer_visual_feature_fusion_in_multimodal_llms_methods_analysis_and_best_p.md)
- [\[AAAI 2026\] Harnessing Textual Semantic Priors for Knowledge Transfer and Refinement in CLIP-Driven Continual Learning](../../AAAI2026/multimodal_vlm/harnessing_textual_semantic_priors_for_knowledge_transfer_and_refinement_in_clip.md)
- [\[CVPR 2026\] Multi-Modal Image Fusion via Intervention-Stable Feature Learning](../../CVPR2026/multimodal_vlm/multi-modal_image_fusion_via_intervention-stable_feature_learning.md)

</div>

<!-- RELATED:END -->
