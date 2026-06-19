---
title: >-
  [论文解读] OneTrack: Demystifying the Conflict Between Detection and Tracking in End-to-End 3D Trackers
description: >-
  [ECCV 2024][视频理解][3D多目标跟踪] 本文深入分析了端到端3D跟踪器中检测与跟踪任务之间性能冲突的根本原因——二者在正样本分配上的微妙差异导致了分类梯度的矛盾，并提出OneTrack通过梯度协调、查询分组和注意力掩码等策略，首次实现了检测和跟踪在统一特征表示下的无冲突联合优化，在nuScenes上取得了SOTA性能。
tags:
  - "ECCV 2024"
  - "视频理解"
  - "3D多目标跟踪"
  - "检测-跟踪冲突"
  - "梯度协调"
  - "端到端跟踪"
  - "nuScenes"
---

# OneTrack: Demystifying the Conflict Between Detection and Tracking in End-to-End 3D Trackers

**会议**: ECCV 2024  
**代码**: 无  
**领域**: 视频理解 / 3D目标跟踪  
**关键词**: 3D多目标跟踪, 检测-跟踪冲突, 梯度协调, 端到端跟踪, nuScenes

## 一句话总结
本文深入分析了端到端3D跟踪器中检测与跟踪任务之间性能冲突的根本原因——二者在正样本分配上的微妙差异导致了分类梯度的矛盾，并提出OneTrack通过梯度协调、查询分组和注意力掩码等策略，首次实现了检测和跟踪在统一特征表示下的无冲突联合优化，在nuScenes上取得了SOTA性能。

## 研究背景与动机

**领域现状**：基于视觉的3D多目标跟踪(MOT)是自动驾驶感知的关键任务。近年来，端到端(end-to-end)范式逐渐兴起，以DETR为代表的基于查询(query-based)的检测框架被扩展到跟踪任务——通过将上一帧的目标查询传播到当前帧来实现跟踪。代表性方法如MUTR3D、PF-Track、StreamPETR等采用类似架构，在多视角相机输入下同时进行3D检测和跟踪。

**现有痛点**：现有端到端跟踪器普遍面临一个关键问题——**检测与跟踪任务之间存在性能冲突**。具体表现为：当联合优化检测和跟踪时，两个任务的性能都无法达到各自单独优化时的水平，尤其是跟踪性能显著下降。之前的研究将此归因于"不同任务需要不同的目标特征"，但这个解释过于笼统，未能指出具体的技术原因和解决方案。

**核心矛盾**：检测和跟踪虽然都需要对目标进行定位和分类，但它们在"哪些查询应该被认为是正样本"的定义上存在微妙但关键的差异。检测任务将最匹配ground truth的新查询作为正样本；跟踪任务将与之前帧目标ID关联的传播查询作为正样本。**同一个查询可能在检测任务中是正样本（因为它空间上匹配了某个GT），但在跟踪任务中是负样本（因为它的ID不匹配）**，反之亦然。这导致分类头接收到矛盾的梯度信号，无法同时满足两个任务。

**本文目标** (1) 精确诊断检测-跟踪冲突的根本原因；(2) 提出具体的梯度级别解决方案；(3) 实现首个无冲突的单阶段端到端联合检测跟踪模型。

**切入角度**：作者通过仔细分析正样本分配策略和梯度流向，发现冲突的本质是分类梯度的极性矛盾(polarity conflict)——某些查询在两个任务中收到方向相反的分类梯度（一个要求推高分类得分，另一个要求压低）。一旦识别出这一点，解决方案就变得清晰：需要根据查询在两个任务中的极性进行分组，并阻断冲突查询之间的信息交互。

**核心 idea**：识别并协调检测与跟踪任务间正样本分配差异导致的分类梯度矛盾，通过查询分组和注意力掩码实现无冲突的联合优化。

## 方法详解

### 整体框架
OneTrack基于标准的查询式3D检测+跟踪框架。输入是多视角相机图像序列，通过图像backbone提取特征后，使用两组查询进行目标表示：新检测查询(Detection Queries)用于检测新出现的目标，传播查询(Track Queries)从上一帧传播而来用于跟踪已有目标。两组查询共享解码器进行特征更新，输出3D边界框和分类得分。关键创新在于对两组查询的梯度进行协调和查询间注意力的选择性掩码。

### 关键设计

1. **梯度极性分析与协调 (Gradient Polarity Coordination)**:

    - 功能：消除检测与跟踪任务对同一查询的矛盾分类梯度
    - 核心思路：首先分析每个查询在检测和跟踪两个任务中的正负样本归属，将所有查询动态划分为四组：(a) 两个任务中都是正样本(Pos-Pos)——无冲突；(b) 检测正/跟踪负(Pos-Neg)——冲突组；(c) 检测负/跟踪正(Neg-Pos)——冲突组；(d) 两个任务中都是负样本(Neg-Neg)——无冲突。对于冲突组(b)和(c)中的查询，需要特殊处理其分类损失的梯度。具体做法是：对冲突查询的分类目标进行修正，避免两个任务给出完全相反的优化信号。例如，对于Pos-Neg查询，保留检测任务的正梯度但修改跟踪任务中的目标使其不产生冲突负梯度
    - 设计动机：这是对冲突根源的精准手术——不是简单地解耦两个任务（那样会牺牲端到端的优势），而是仅针对冲突梯度进行调和

2. **基于极性的查询分组与注意力掩码 (Polarity-based Query Grouping & Attention Masking)**:

    - 功能：防止冲突查询之间的特征污染
    - 核心思路：在Transformer解码器的自注意力层中，根据上述四组分类对查询间的注意力进行选择性掩码。具体规则是：**正样本分配冲突的查询组之间的注意力被掩码掉**。即Pos-Neg组的查询不能attend到Neg-Pos组的查询，反之亦然。这防止了特征层面的冲突传播——如果一个检测正样本query attend到一个跟踪正样本query（但它们的detection/tracking极性相反），特征更新可能向矛盾的方向偏移。注意力掩码的具体模式根据当前帧每个查询的极性动态生成
    - 设计动机：梯度协调解决了损失层面的冲突，但如果冲突查询在特征层面仍然耦合，冲突信号仍会通过注意力机制传播。注意力掩码从特征交互层面进一步隔离冲突

3. **跟踪分类损失修正 (Tracking Classification Loss Modification)**:

    - 功能：抑制跟踪任务中的不准确预测，提升跟踪质量
    - 核心思路：标准的分类损失对跟踪查询可能过度自信——一个从上帧传播的查询可能因为位置预测不准确而实际上并不真正匹配目标，但分类损失仍将其作为正样本。作者对跟踪分类损失引入了基于定位质量的加权：$\mathcal{L}_{cls}^{track} = w_i \cdot \text{FocalLoss}(p_i, y_i)$，其中权重 $w_i$ 由跟踪查询的预测框与GT的IoU决定。IoU低的跟踪查询（位置不准确）其分类损失被降权，避免不准确的高置信度预测干扰跟踪关联
    - 设计动机：跟踪质量不仅取决于ID关联的正确性，还取决于定位精度。将定位质量融入分类目标，使分类得分真正反映跟踪的可靠性

### 损失函数 / 训练策略
总损失包含：(1) 检测分类损失 $\mathcal{L}_{cls}^{det}$（Focal Loss）；(2) 检测回归损失 $\mathcal{L}_{reg}^{det}$（L1 + GIoU）；(3) 修正后的跟踪分类损失 $\mathcal{L}_{cls}^{track}$；(4) 跟踪回归损失 $\mathcal{L}_{reg}^{track}$。使用匈牙利匹配进行正样本分配。训练采用两阶段策略：先在短序列（2帧）上预训练检测能力，再在长序列（多帧）上微调跟踪能力。

## 实验关键数据

### 主实验

| 方法 | nuScenes Val AMOTA↑ | nuScenes Val AMOTP↓ | nuScenes Test AMOTA↑ | 备注 |
|------|---------------------|---------------------|----------------------|------|
| OneTrack (本文) | 55.8 | 1.21 | 51.2 | 单阶段联合模型 |
| MUTR3D | 45.1 | 1.45 | 42.8 | 两阶段 |
| PF-Track | 48.9 | 1.32 | 48.1 | 两阶段 |
| StreamPETR | 50.4 | 1.28 | 49.5 | 有检测-跟踪冲突 |
| DQTrack | 52.3 | 1.25 | 49.8 | 解耦特征 |

### 消融实验

| 配置 | AMOTA↑ | AMOTP↓ | mAP (Det)↑ | 说明 |
|------|--------|--------|-----------|------|
| OneTrack Full | 55.8 | 1.21 | 44.2 | 完整模型 |
| w/o 梯度协调 | 51.2 | 1.30 | 42.5 | 去掉梯度极性协调 |
| w/o 注意力掩码 | 53.1 | 1.26 | 43.3 | 去掉查询分组掩码 |
| w/o 分类损失修正 | 54.3 | 1.23 | 44.0 | 去掉跟踪分类加权 |
| w/o 所有改进(baseline) | 48.9 | 1.35 | 41.8 | 无冲突缓解的baseline |
| 完全解耦(两个独立头) | 52.0 | 1.27 | 43.8 | 分离检测跟踪特征 |

### 关键发现
- 梯度协调是最关键的组件，去掉后AMOTA下降4.6%，说明梯度极性冲突确实是性能瓶颈的核心原因
- 注意力掩码和分类损失修正分别贡献2.7%和1.5%的AMOTA提升，三个组件互补
- 与"完全解耦"策略相比，OneTrack的统一特征+冲突缓解方案更优（55.8 vs 52.0 AMOTA），说明检测和跟踪共享特征是有益的，只要正确处理冲突
- 在nuScenes测试集上，OneTrack超过此前所有方法+3.1% AMOTA，验证了方法的实际效果和泛化性
- 检测性能（mAP）也因冲突缓解而提升，说明冲突是双向的——跟踪也在拖累检测

## 亮点与洞察
- **问题诊断的深度**：不满足于"检测和跟踪有冲突"的模糊认知，一路追溯到正样本分配差异→分类梯度极性矛盾这个根因。这种从现象到机制的深入分析是顶会论文的典范
- **精准的解决方案**：针对根因设计了三个互补的组件（梯度协调、注意力掩码、损失修正），每个都有清晰的动机和作用，没有多余的设计
- **统一 vs 解耦的新解**：长期以来，处理多任务冲突的默认方案是"解耦特征"，本文证明了"统一特征+冲突感知优化"是更优的选择。这个洞察可以推广到其他多任务学习场景
- **查询分组的动态性**：查询的极性分组是帧级动态的（同一个查询在不同帧可能属于不同的极性组），方法设计充分考虑了这种动态性

## 局限与展望
- **仅限相机输入**：当前只验证了相机输入的3D跟踪，LiDAR-based或融合(fusion)场景下的检测-跟踪冲突是否存在以及解决方案是否通用，未做验证
- **计算开销**：查询分组和注意力掩码的动态生成增加了一定的计算开销，对于实时性要求高的部署场景可能需要优化
- **遮挡和重新出现**：论文未深入讨论目标被长时间遮挡后重新出现的re-identification能力，这在实际场景中很重要
- **正样本分配策略的选择敏感性**：冲突的严重程度取决于检测和跟踪各自的正样本分配策略（如不同的IoU阈值），不同配置下冲突模式可能不同
- 改进方向：将梯度协调思路推广到LiDAR 3D跟踪；探索更优的动态正样本分配策略从根源减少冲突

## 相关工作与启发
- **vs MUTR3D**: MUTR3D是早期的端到端3D跟踪器，直接将检测查询扩展为跟踪查询但未处理任务冲突，性能受限严重；OneTrack从冲突源头解决了这个问题
- **vs StreamPETR**: StreamPETR通过流式传播查询实现在线跟踪，性能较好但仍受冲突影响；OneTrack在StreamPETR的基础上进一步提出了冲突缓解方案
- **vs 多任务学习的梯度协调方法(如GradNorm, PCGrad)**: 这些通用多任务方法在任务级别协调梯度，但未考虑样本级别的极性冲突；OneTrack的贡献是识别了样本级梯度冲突这一更细粒度的问题

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 对检测-跟踪冲突的根因分析深刻原创，解决方案优雅
- 实验充分度: ⭐⭐⭐⭐⭐ 大规模nuScenes验证+详细消融+组件贡献清晰
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义精准，分析逻辑严密，实验设计完整
- 价值: ⭐⭐⭐⭐ 对端到端跟踪领域有重要推动，冲突协调思路可推广

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] AutoCut: End-to-end Advertisement Video Editing Based on Multimodal Discretization and Controllable Generation](../../CVPR2026/video_understanding/autocut_end-to-end_advertisement_video_editing_based_on_multimodal_discretizatio.md)
- [\[ECCV 2024\] Boosting 3D Single Object Tracking with 2D Matching Distillation and 3D Pre-training](boosting_3d_single_object_tracking_with_2d_matching_distillation_and_3d_pre-trai.md)
- [\[CVPR 2026\] TimeBridge: Self-Supervised Video Representation Learning via Start-End Joint Embedding and In-Between Frame Prediction](../../CVPR2026/video_understanding/timebridge_self-supervised_video_representation_learning_via_start-end_joint_emb.md)
- [\[ECCV 2024\] SPAMming Labels: Efficient Annotations for the Trackers of Tomorrow](spamming_labels_efficient_annotations_for_the_trackers_of_tomorrow.md)
- [\[ECCV 2024\] CrossGLG: LLM Guides One-Shot Skeleton-Based 3D Action Recognition in a Cross-Level Manner](crossglg_llm_guides_one-shot_skeleton-based_3d_action_recognition_in_a_cross-lev.md)

</div>

<!-- RELATED:END -->
