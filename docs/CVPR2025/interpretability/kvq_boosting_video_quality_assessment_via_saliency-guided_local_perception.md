---
title: >-
  [论文解读] KVQ: Boosting Video Quality Assessment via Saliency-Guided Local Perception
description: >-
  [CVPR 2025][可解释性][视频质量评估] KVQ 受人类视觉系统启发，将视频全局质量显式解耦为视觉显著性和局部纹理两个因素，通过 Fusion-Window Attention 提取跨区域显著性、Local Perception Constraint 增强独立区域的纹理感知，在五个 VQA benchmark 上显著超越 SOTA。
tags:
  - "CVPR 2025"
  - "可解释性"
  - "视频质量评估"
  - "视觉显著性"
  - "局部纹理感知"
  - "融合窗口注意力"
  - "人类视觉系统"
---

# KVQ: Boosting Video Quality Assessment via Saliency-Guided Local Perception

**会议**: CVPR 2025  
**arXiv**: [2503.10259](https://arxiv.org/abs/2503.10259)  
**代码**: [https://github.com/qyp2000/KVQ](https://github.com/qyp2000/KVQ)  
**领域**: 可解释性  
**关键词**: 视频质量评估, 视觉显著性, 局部纹理感知, 融合窗口注意力, 人类视觉系统

## 一句话总结

KVQ 受人类视觉系统启发，将视频全局质量显式解耦为视觉显著性和局部纹理两个因素，通过 Fusion-Window Attention 提取跨区域显著性、Local Perception Constraint 增强独立区域的纹理感知，在五个 VQA benchmark 上显著超越 SOTA。

## 研究背景与动机

**领域现状**：视频质量评估（VQA）旨在预测视频的感知质量，是短视频平台优化用户体验的关键技术。当前无参考 VQA 方法大多直接预测全局质量分数，少数方法（如 Fast-VQA）尝试通过采样局部 patch 来感知区域质量。

**现有痛点**：(1) 视频中不同时空区域的质量差异显著（运动模糊、压缩失真、纹理复杂度不同），但现有方法缺乏对区域级质量的可靠建模；(2) 由于标注区域质量的成本极高（约 $\mathcal{O}(N^3)$ 倍于全局标注），缺少区域级标注数据作为约束；(3) 现有方法的注意力机制（如窗口注意力）限制在局部邻域，无法有效提取全局视觉显著性。

**核心矛盾**：全局质量由局部纹理和视觉显著性共同决定，但两者是不同层次的概念——显著性涉及语义和区域间关联（高层），纹理只涉及区域内部的底层特征。现有方法将两者混为一谈，导致(a) 局部质量预测被显著性影响，(b) 显著性提取被限制在局部窗口。

**本文目标** (1) 如何在无局部标注的情况下可靠预测区域级质量；(2) 如何有效提取全局视觉显著性。

**切入角度**：从 HVS 出发提出两个假设——Assumption 1: 全局质量 = 显著性加权的局部纹理；Assumption 2: 局部纹理仅由区域内部特征决定，不受其他区域影响。据此设计显著性提取和局部感知约束。

**核心 idea**：将质量显式分解为显著性 × 纹理，用跨窗口注意力提取显著性 + 一致性约束确保纹理独立于邻域。

## 方法详解

### 整体框架

以 Video Swin-T 为 backbone 提取时空特征，顶部双分支分别预测显著性图 $\mathcal{S} \in \mathbb{R}^{T \times H \times W}$ 和局部纹理图 $\mathcal{Q} \in \mathbb{R}^{T \times H \times W}$。全局质量 $q = \frac{1}{THW} \sum_{i,j,k} \mathcal{S}_{i,j,k} \cdot \mathcal{Q}_{i,j,k}$。在 backbone 中嵌入 Fusion-Window Attention 替换标准窗口注意力，训练时加入 Local Perception Constraint 约束纹理分支。

### 关键设计

1. **Fusion-Window Attention (FWA)**:

    - 功能：实现跨区域的全局注意力分配，有效提取视觉显著性
    - 核心思路：分三步——(a) **Correlated-Window Selection (CWS)**：先计算全局 patch 相关性图 $\mathbf{I}_p = Softmax(Flatten(Q) \cdot Flatten(K)^T)$，平均池化到窗口级 $\mathbf{I}_w$，选取 top-k 最相关窗口的索引 $\mathbf{Idx}$；(b) **Intra-Window Attention (IWA)**：标准窗口内自注意力保留邻域信息；(c) **Cross-Window Attention (CWA)**：每个窗口的 Query 与其 top-k 相关窗口的 Key/Value 做 cross-attention。最终 $FWA = IWA + CWA$
    - 设计动机：标准窗口注意力（如 Swin Transformer）只在局部窗口内做注意力，无法模拟人类视觉在全局范围分配注意力的行为。FWA 通过自适应选择相关窗口实现全局长程连接，更好捕获显著性

2. **Multi-scale Ensemble Saliency Map**:

    - 功能：融合多尺度关联图生成最终显著性图
    - 核心思路：FWA 中的 patch 相关性图 $\mathbf{I}_p$ 本身反映了注意力分配。将其转置求和得到每个 patch 的被关注程度 $\mathbf{I}_p'$，reshape 后池化到统一分辨率得到各层的显著性估计 $\mathbf{I}_p^{(l)}$。与显著性分支输出 $\tilde{S}$ 加权融合：$\mathcal{S} = Softmax(w^0 \tilde{S} + \sum_{l} w^l \mathbf{I}_p^{(l)})$
    - 设计动机：参考 HVS 视觉信息流经皮层层级的过程，多尺度融合能捕获从细粒度到粗粒度的显著性信息

3. **Local Perception Constraint (LPC)**:

    - 功能：确保纹理图仅反映区域内部特征，不受其他区域影响
    - 核心思路：将完整视频输入模型得到纹理图 $\mathcal{Q}$；同时将视频切成独立 patch 分别输入同一模型，重新拼装得到 $\hat{\mathcal{Q}}$。约束两者一致：$\mathcal{L}_{lpc} = 1 - \frac{\sum \mathcal{Q}_{i,j,k} \cdot \hat{\mathcal{Q}}_{i,j,k}}{||\mathcal{Q}|| \cdot ||\hat{\mathcal{Q}}||}$（余弦相似度损失）
    - 设计动机：如果纹理预测依赖于邻域上下文，那么切成独立 patch 后预测结果会不同。通过约束一致性来强制模型的纹理分支只关注区域内部的低级特征（畸变、清晰度、纹理模式），不混入语义信息

### 损失函数 / 训练策略

总损失 $\mathcal{L} = \mathcal{L}_{plcc} + \lambda_r \mathcal{L}_{rank} + \lambda_p \mathcal{L}_{lpc}$。$\mathcal{L}_{plcc}$ 是 PLCC 损失（回归质量分数），$\mathcal{L}_{rank}$ 是排序损失（学习相对质量关系），$\mathcal{L}_{lpc}$ 是局部感知约束。使用 Video Swin-T Tiny（Kinetics-400 预训练）为 backbone，窗口大小 [8,7,7]，每个视频采样 32 帧，resize 到 448×448。

## 实验关键数据

### 主实验

| 方法 | LSVQtest SRCC↑ | LSVQ1080p SRCC↑ | KoNViD-1k SRCC↑ | LIVE-VQC SRCC↑ |
|------|----------------|-----------------|-----------------|----------------|
| Fast-VQA | 0.876 | 0.779 | 0.859 | 0.823 |
| Faster-VQA | 0.873 | 0.772 | 0.863 | 0.813 |
| **KVQ** | **0.896** | **0.814** | **0.890** | **0.820** |
| 提升 vs Fast-VQA | +2.3% | +4.5% | +3.6% | -0.4% |

Transfer learning 场景下 KVQ 优势更明显：KoNViD-1k 上 SRCC 0.909（vs Fast-VQA 0.891），YouTube-UGC 上 SRCC 0.903（vs 0.855，+5.6%）。

### 消融实验（来自论文框架描述）

| 组件 | 作用 | 影响 |
|------|------|------|
| FWA (Fusion-Window Attention) | 全局显著性提取 | 跨窗口注意力显著提升长程依赖建模 |
| LPC (Local Perception Constraint) | 局部纹理约束 | 确保纹理图不被显著性污染 |
| Multi-scale Ensemble | 多尺度显著性融合 | 层级信息互补提升显著性估计 |
| CWS (Correlated-Window Selection) | 自适应窗口选择 | 避免全局注意力的 $O(N^2)$ 计算量 |

### 关键发现

- KVQ 在 LSVQ1080p（高分辨率视频）上提升最大（+4.5% SRCC），说明 FWA 的跨区域注意力在高分辨率场景下特别有效
- Transfer learning 提升显著（YouTube-UGC +5.6%），说明显著性-纹理解耦使模型泛化性更强
- 新建的 LPVQ 数据集（50 张图、14 位专家、34,300 条标注）验证了 KVQ 确实能感知局部质量差异
- 计算量（353 GFlops）比 Li et al.（112,537 GFlops）低两个数量级，与 Fast-VQA（279 GFlops）在同一量级

## 亮点与洞察

- **从 HVS 出发的显式解耦非常优雅**：全局质量 = 显著性 × 纹理的公式化表达清晰且可解释。显著性告诉你"看哪里"，纹理告诉你"看到的质量如何"。这种解耦使得模型可以独立输出显著性图和质量图，增强可解释性
- **LPC 是一个无需标注的自监督约束**：巧妙利用了"局部特征应不受上下文影响"这一先验，不需要任何局部质量标注就能训练纹理分支，解决了标注成本高的核心难题
- **FWA 的设计思路可迁移到其他视频理解任务**：基于相关性的跨窗口注意力不局限于质量评估，任何需要长程依赖但受限于窗口注意力的 Video Transformer 都可以采用

## 局限与展望

- LPVQ 数据集仅有 50 张图像作为静态视频标注，规模较小；扩展到真正的视频时空区域标注会更有说服力
- FWA 的 top-k 窗口选择引入了额外的全局相关性计算，在极高分辨率视频上的计算效率需要关注
- 假设 1（全局质量 = 加权求和）是线性模型，实际上人类对质量的感知可能是非线性的（如"最差区域主导"效应）
- 仅在特定窗口大小 [8,7,7] 下验证，对不同分辨率和帧率的视频可能需要自适应窗口设计

## 相关工作与启发

- **vs Fast-VQA**: Fast-VQA 均匀采样局部 patch 但缺乏局部质量约束，将纹理和显著性混在一起；KVQ 显式解耦后整体提升 2-5%
- **vs SGDNet/TranSLA**: 这些方法虽然引入了显著性预测子任务，但训练复杂且依赖 SOTA 显著性模型的伪标签；KVQ 直接从注意力机制中提取显著性，不需要额外标签
- **vs PVQ**: PVQ 尝试通过众包标注局部 patch 质量，但成本高且纹理与显著性混合；KVQ 通过 LPC 无监督地实现了更可靠的解耦

## 评分

- 新颖性: ⭐⭐⭐⭐ HVS 启发的显著性-纹理解耦思路清晰，FWA 和 LPC 设计巧妙但各组件单独看并非全新
- 实验充分度: ⭐⭐⭐⭐⭐ 5 个 benchmark、3 种评估设置（intra/cross/transfer）、新建 LPVQ 数据集
- 写作质量: ⭐⭐⭐⭐ 假设驱动的方法设计逻辑清晰，数学公式化表达规范
- 价值: ⭐⭐⭐⭐ 对 VQA 领域的局部感知建模提供了新思路，LPVQ 数据集也有独立价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] DR.Experts: Differential Refinement of Distortion-Aware Experts for Blind Image Quality Assessment](../../AAAI2026/interpretability/drexperts_differential_refinement_of_distortion-aware_experts_for_blind_image_qu.md)
- [\[ICML 2026\] IQA-Spider: Unifying Multi-Granularity Image Quality Assessment with Reasoning, Grounding and Referring](../../ICML2026/interpretability/iqa-spider_unifying_multi-granularity_image_quality_assessment_with_reasoning_gr.md)
- [\[CVPR 2025\] Geometry-Guided Camera Motion Understanding in VideoLLMs](geometry-guided_camera_motion_understanding_in_videollms.md)
- [\[ICML 2025\] Towards Flexible Perception with Visual Memory](../../ICML2025/interpretability/towards_flexible_perception_with_visual_memory.md)
- [\[CVPR 2025\] Language Guided Concept Bottleneck Models for Interpretable Continual Learning](language_guided_concept_bottleneck_models_for_interpretable_continual_learning.md)

</div>

<!-- RELATED:END -->
