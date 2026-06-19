---
title: >-
  [论文解读] SLAck: Semantic, Location, and Appearance Aware Open-Vocabulary Tracking
description: >-
  [ECCV 2024][视频理解][开放词汇跟踪] SLAck 提出在多目标跟踪的关联阶段早期统一融合语义、位置和外观三种线索，通过轻量级时空目标图（STOG）学习隐式运动先验和跨线索协同，无需后处理启发式规则，在开放词汇 MOT 和 TAO TETA 基准上显著提升新类别跟踪性能。 多目标跟踪（MOT）传统上局限于行人、车…
tags:
  - "ECCV 2024"
  - "视频理解"
  - "开放词汇跟踪"
  - "多目标跟踪"
  - "语义感知关联"
  - "时空目标图"
  - "特征融合"
---

# SLAck: Semantic, Location, and Appearance Aware Open-Vocabulary Tracking

**会议**: ECCV 2024  
**arXiv**: [2409.11235](https://arxiv.org/abs/2409.11235)  
**代码**: [https://github.com/siyuanliii/SLAck](https://github.com/siyuanliii/SLAck)  
**领域**: 视频理解  
**关键词**: 开放词汇跟踪, 多目标跟踪, 语义感知关联, 时空目标图, 特征融合

## 一句话总结

SLAck 提出在多目标跟踪的关联阶段早期统一融合语义、位置和外观三种线索，通过轻量级时空目标图（STOG）学习隐式运动先验和跨线索协同，无需后处理启发式规则，在开放词汇 MOT 和 TAO TETA 基准上显著提升新类别跟踪性能。

## 研究背景与动机

多目标跟踪（MOT）传统上局限于行人、车辆等少数类别。开放词汇跟踪将目标扩展到数百个类别，但也带来了巨大挑战——不同类别的外观、行为和运动模式差异极大。

**现有方法的三类痛点**：

**运动线索（Kalman Filter）**：依赖线性运动假设，在行人/车辆场景有效，但开放词汇场景中物体运动高度非线性（动物奔跑、物体翻滚等），KF 失效严重

**外观线索（纯外观匹配）**：当前最佳方法（如 OVTrack、MASA）主要依赖外观相似度，但存在遮挡敏感、容易混淆外观相似目标、对基类过拟合等问题

**语义线索**：现有方法要么完全忽略语义，要么仅在最后阶段以硬分组（同类关联）或软分组的启发式方式使用，在开放词汇场景中分类不稳定时效果差

**核心矛盾**：不同线索各有优劣，但现有混合方法都是在关联的最后阶段通过启发式规则（如 IoU 矩阵 + 外观矩阵加权平均）来融合，这种后期融合无法学到线索之间的协同关系。

**关键观察**：运动模式与语义类别高度相关——如果模型在训练时学到了马的运动模式，就能通过语义相似性将这种知识迁移到从未见过的斑马上。这意味着语义和运动的联合建模对新类别泛化至关重要。

**核心 idea**：将语义、位置和外观线索在关联的早期阶段统一融合，通过可学习的时空目标图替代启发式后处理，端到端优化产出单一关联矩阵。

## 方法详解

### 整体框架

SLAck 构建在预训练的开放词汇检测器之上。Pipeline 分三步：(1) 从冻结的检测器中提取语义、位置和外观三种嵌入；(2) 通过特征求和融合为统一表示后送入时空目标图（STOG）；(3) STOG 通过帧内自注意力和帧间交叉注意力建模目标动态，最终输出关联矩阵，使用可微 Sinkhorn 算法端到端训练。

### 关键设计

1. **三线索提取头（Semantic / Location / Appearance Head）**:

    - 功能：从冻结检测器中提取三种互补的目标描述符
    - 核心思路：
        - **语义头**：使用 CLIP 对齐的 RCNN 分类头的输出嵌入，经 5 层 MLP 投影得到语义嵌入 $E_{\text{sem}}$。这样无需重训即可配置新类别
        - **位置头**：将检测框坐标归一化——以图像中心为原点、70%最大维度为缩放因子：$\left(\frac{x_{\min} - W/2}{s}, \frac{y_{\min} - H/2}{s}, \frac{w}{s}, \frac{h}{s}\right)$，经 MLP 投影为位置嵌入 $E_{\text{loc}}$
        - **外观头**：4 层卷积 + MLP 处理 RoI 特征，输出外观嵌入 $E_{\text{app}}$
    - 设计动机：冻结检测器保持原始检测能力不退化；归一化坐标确保尺度不变性；三种嵌入捕捉目标的不同方面

2. **时空目标图 (Spatial-Temporal Object Graph, STOG)**:

    - 功能：建模帧内目标间的空间关系和帧间目标的时序对应
    - 核心思路：先将三种嵌入通过加法融合 $E_{\text{fused}}^i = E_{\text{app}}^i + E_{\text{loc}}^i + E_{\text{sem}}^i$，然后交替进行：
        - **帧内自注意力**（Spatial Object Graph）：$\text{SA}_K(Q_K, K_K, V_K) = \sigma\left(\frac{Q_K K_K^T}{\sqrt{d}}\right)V_K$，分别处理 key 帧和 reference 帧内的目标关系，让模型感知帧内目标的相对位置和相互关系
        - **帧间交叉注意力**（Temporal Object Graph）：$\text{CA}_{K \to R}(Q_K, K_R, V_R)$，对齐和更新不同帧之间的目标特征，捕捉时序运动模式
    - 设计动机：替代显式 Kalman Filter 的线性运动假设，通过注意力机制从数据中学习隐式运动先验，能捕捉线性和非线性运动。帧内自注意力让模型理解场景级目标布局，帧间交叉注意力实现跨帧特征对齐

3. **检测感知训练 (Detection Aware Training, DAT)**:

    - 功能：解决 TAO 数据集标注不完整的问题
    - 核心思路：冻结检测器权重，用检测器的预测框（而非仅稀疏 GT）作为训练输入，仅在预测框与 GT 匹配时计算关联损失
    - 设计动机：直接用稀疏 GT 训练会导致训练-测试分布不一致。DAT 通过模拟测试条件，使训练时看到与推理时一致的检测框分布，AssocA 提升 +13.7

### 损失函数 / 训练策略

- 使用可微 Sinkhorn 算法求解最优传输问题：$\mathcal{L}_{\text{Sinkhorn}} = -\sum_{i,j} T_{ij}' \log(S_{ij}')$
- 目标匹配矩阵 $\mathbf{T}$ 由 GT 对应关系构建，增加 dustbin 类处理出现/消失目标
- 端到端训练，无需额外启发式规则
- 训练帧对在 3 秒内的相邻帧中采样

## 实验关键数据

### 主实验

| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| OV-MOT val (Novel) | TETA | 31.1 | 30.0 (MASA-R50) | +1.1 |
| OV-MOT val (Novel) | AssocA | 37.8 | 34.6 (MASA-R50) | +3.2 |
| OV-MOT test (Novel) | TETA | 27.1 | 24.1 (OVTrack) | +3.0 |
| TAO TETA (Swin-L) | AssocA | 41.8 | 40.9 (GLEE-Plus) | +0.9 |
| TAO TETA (Swin-T) | AssocA | 38.9 | 36.7 (TETer-T) | +2.2 |

### 消融实验

| 配置 | AssocA | 说明 |
|------|--------|------|
| Lck (仅位置) | 28.3 | 隐式运动，已优于 KF 的 OC-SORT (20.4) |
| SLck (语义+位置) | 35.4 (+7.1) | 语义大幅提升运动跟踪 |
| Ack (仅外观) | 32.7 | 纯外观基线 |
| SAck (语义+外观) | 35.1 (+2.4) | 语义也提升外观跟踪 |
| LAck (位置+外观) | 36.4 | 混合但无语义 |
| SLAck (全模型) | 37.8 (+1.4) | 三线索协同最优 |
| 无 DAT | 24.1 | DAT 带来 +13.7 |
| 硬分组 vs SLAck-SAck | 30.6 vs 38.0 | 早期融合远优于硬分组 |

### 关键发现

- **语义线索对新类别跟踪提升最大**：仅加语义就让位置跟踪 AssocA 从 28.3 升到 35.4（+7.1），甚至超过了纯外观 SOTA（OVTrack 33.6）
- **时序图（TOG）对语义和位置线索更重要**（+2.4 和 +4.1），空间图（SOG）对外观更重要（+0.9）
- **DAT 策略影响巨大**：+13.7 AssocA，解决训练-测试分布差异是关键
- 语义线索单独使用不足以替代外观（-4.4），但作为补充效果显著

## 亮点与洞察

- **早期融合 vs 后期启发式融合**的对比实验非常有说服力——硬分组降性能 -4.6，而 SLAck 的早期语义融合提升 +2.8
- **语义-运动协同**的洞察很巧妙：在基类上学到的运动模式可通过语义相似性迁移到新类别（马→斑马）
- **隐式运动建模**替代显式 Kalman Filter，对开放词汇场景的非线性运动更鲁棒
- DAT 训练策略简单有效，可迁移到所有使用不完整标注的 MOT 方法

## 局限与展望

- 仅在 TAO 一个大词汇数据集上训练和评估，泛化性有待更多数据集验证
- 当前使用 ResNet-50 作为 backbone，与使用更强 backbone 的方法（如 GroundingDINO）相比定位精度偏低
- STOG 的注意力机制计算量随目标数增长，密集场景可能有效率问题
- 语义头的 CLIP 对齐分类能力对长尾类别仍可能不稳定

## 相关工作与启发

- **vs OVTrack**: 纯外观匹配 + Stable Diffusion 增强，忽略了语义和位置；SLAck 在 Novel AssocA 上超 +4.2
- **vs MASA**: 学习通用外观模型，不使用语义；SLAck 在 TETA 上超 +1.1
- **vs TETer**: 使用 CEM 编码做晚期软分组；SLAck 的早期融合超其 +2.2 AssocA（同 backbone）
- **vs GLEE**: 千万级图像训练的基础模型；SLAck 仅用 TAO 训练集就超其 +0.9 AssocA

## 评分

- 新颖性: ⭐⭐⭐⭐ 早期融合替代后期启发式的思路清晰，语义-运动协同的 insight 有价值
- 实验充分度: ⭐⭐⭐⭐⭐ 消融极其详尽，逐一分析每种线索和每个模块的贡献
- 写作质量: ⭐⭐⭐⭐ 动机论述充分，图表清晰，但方法部分符号较多需仔细阅读
- 价值: ⭐⭐⭐⭐ 为开放词汇跟踪提供了清晰的统一框架，语义的重要性发现对社区有指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Attention to Trajectory: Trajectory-Aware Open-Vocabulary Tracking](../../ICCV2025/video_understanding/attention_to_trajectory_trajectory-aware_open-vocabulary_tracking.md)
- [\[ECCV 2024\] SemTrack: A Large-Scale Dataset for Semantic Tracking in the Wild](semtrack_a_large-scale_dataset_for_semantic_tracking_in_the_wild.md)
- [\[ECCV 2024\] Spatio-Temporal Proximity-Aware Dual-Path Model for Panoramic Activity Recognition](spatio-temporal_proximity-aware_dual-path_model_for_panoramic_activity_recogniti.md)
- [\[ICCV 2025\] Learning to Generalize Without Bias for Open-Vocabulary Action Recognition](../../ICCV2025/video_understanding/learning_to_generalize_without_bias_for_open-vocabulary_action_recognition.md)
- [\[ECCV 2024\] SPAMming Labels: Efficient Annotations for the Trackers of Tomorrow](spamming_labels_efficient_annotations_for_the_trackers_of_tomorrow.md)

</div>

<!-- RELATED:END -->
