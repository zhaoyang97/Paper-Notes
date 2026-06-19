---
title: >-
  [论文解读] ADMap: Anti-disturbance Framework for Vectorized HD Map Construction
description: >-
  [ECCV 2024][高精地图构建] 本文提出 ADMap 框架，通过多尺度感知颈部(MPN)、实例交互注意力(IIA)和矢量方向差异损失(VDDL)三个模块，从实例间和实例内两个层面级联式监控点序列预测过程，有效缓解了矢量化高精地图构建中的点序列抖动/锯齿问题，在 nuScenes 和 Argoverse2 上取得了 SOTA 性能。
tags:
  - "ECCV 2024"
  - "高精地图构建"
  - "矢量化地图"
  - "点序列抖动"
  - "多尺度感知"
  - "方向差异损失"
---

# ADMap: Anti-disturbance Framework for Vectorized HD Map Construction

**会议**: ECCV 2024  
**arXiv**: [2401.13172](https://arxiv.org/abs/2401.13172)  
**代码**: [https://github.com/hht1996ok/ADMap](https://github.com/hht1996ok/ADMap)  
**领域**: 其他  
**关键词**: 高精地图构建, 矢量化地图, 点序列抖动, 多尺度感知, 方向差异损失

## 一句话总结

本文提出 ADMap 框架，通过多尺度感知颈部(MPN)、实例交互注意力(IIA)和矢量方向差异损失(VDDL)三个模块，从实例间和实例内两个层面级联式监控点序列预测过程，有效缓解了矢量化高精地图构建中的点序列抖动/锯齿问题，在 nuScenes 和 Argoverse2 上取得了 SOTA 性能。

## 研究背景与动机

**领域现状**：在自动驾驶领域，在线高精(HD)地图构建对规划任务至关重要。近年来，以 MapTR 为代表的矢量化地图构建方法取得了显著进展——这些方法将地图元素（车道线、人行横道、路缘等）表示为有序点序列（即矢量），端到端地从车载传感器数据中预测。与传统的栅格化表示相比，矢量化表示更紧凑、更精确、更便于下游规划模块使用。

**现有痛点**：尽管现有矢量化地图构建模型在整体性能指标上表现良好，但生成的点序列存在明显的抖动(jittery)或锯齿(jagged)现象。具体表现为：(1) 本应平滑的车道线呈现出不规则的锯齿边缘；(2) 相邻点之间的方向突变，导致矢量不够流畅；(3) 不同实例之间的点序列缺乏协调性。这种抖动虽然可能不会严重影响 mAP 等评测指标，但对下游规划任务的影响很大——规划模块依赖精确平滑的地图信息来生成安全的行驶轨迹。

**核心矛盾**：现有方法主要通过端点坐标的回归损失来监督点序列预测，这种逐点独立的监督方式忽略了点与点之间的相对关系——既不考虑同一实例内相邻点的方向一致性，也不考虑不同实例间的空间协调性，从而导致预测偏差累积为可见的抖动和锯齿。

**本文目标** (1) 如何在实例内部确保相邻点之间方向一致、序列平滑？(2) 如何在实例之间保持空间关系的协调性？(3) 如何在不牺牲检测速度的前提下提升地图元素的几何质量？

**切入角度**：作者从点序列的"顺序关系"角度切入，认为应该在实例间和实例内两个层面级联式地探索点的顺序关系，通过关注相邻点之间的方向变化（而非仅关注绝对位置）来直接约束预测的几何平滑性。

**核心 idea**：通过级联式地在实例间和实例内探索点序列的顺序关系，用方向差异损失直接约束几何平滑性，从根本上缓解矢量化地图的抖动问题。

## 方法详解

### 整体框架

ADMap 建立在 MapTR 框架之上。输入为车载多视角相机图像（以及可选的 LiDAR 数据），经过 backbone（如 ResNet-50）和 SECOND（可选的 LiDAR backbone）提取多模态特征，然后通过 BEV 变换得到鸟瞰图特征。创新部分包含三个模块：(1) 多尺度感知颈部(MPN)对 BEV 特征进行多尺度增强；(2) 实例交互注意力(IIA)在 decoder 中建模不同地图实例之间的空间关系；(3) 矢量方向差异损失(VDDL)在训练时约束点序列的方向平滑性。三个模块形成"特征增强→实例间关系→实例内约束"的级联式流程。

### 关键设计

1. **多尺度感知颈部(Multi-Scale Perception Neck, MPN)**:

    - 功能：对 BEV 特征进行多尺度增强，使模型能够同时感知不同尺度的地图元素（如短的人行横道和长的车道线）
    - 核心思路：设计了一个多尺度特征融合模块，对 backbone 输出的 BEV 特征进行多个尺度的池化和上采样，然后通过特征金字塔风格的模块将不同尺度的信息融合。具体地，使用不同大小的卷积核（或不同膨胀率的空洞卷积）提取多尺度上下文，再通过通道注意力机制自适应地选择最有用的尺度信息
    - 设计动机：不同类型的地图元素具有截然不同的空间尺度——人行横道通常较短且定位明确，而车道线可能跨越整个感知范围。单一尺度的特征难以同时精确捕获这两类元素的几何信息

2. **实例交互注意力(Instance Interactive Attention, IIA)**:

    - 功能：在 Transformer decoder 中建模不同地图实例之间的空间关系，确保实例间的协调性
    - 核心思路：在标准 DETR-style decoder 的自注意力层中，引入实例级别的交互注意力。每个实例 query 不仅 attend 到自身的点序列信息，还通过交叉注意力与其他实例的 query 交互。这样，相邻的车道线可以"感知"彼此的位置，避免产生不合理的交叉或间距不一致。设计了级联式的注意力结构：先在实例间进行交互，再在实例内部精化点序列
    - 设计动机：道路场景中的地图元素之间存在强烈的空间约束关系——平行车道线应保持近似等距、相邻元素不应重叠等。如果完全独立地预测每个实例，就会忽略这些全局一致性约束

3. **矢量方向差异损失(Vector Direction Difference Loss, VDDL)**:

    - 功能：直接约束点序列中相邻点之间的方向一致性，从损失函数层面消除抖动
    - 核心思路：对于一个点序列 $\{p_1, p_2, ..., p_n\}$，计算相邻点构成的方向向量 $v_i = p_{i+1} - p_i$，然后约束相邻方向向量之间的角度变化尽可能小。具体损失定义为预测序列的方向差异与 ground truth 序列的方向差异之间的 L1 距离：$L_{VDDL} = \sum_i \| (v_{i+1}^{pred} - v_i^{pred}) - (v_{i+1}^{gt} - v_i^{gt}) \|_1$。这个损失直接惩罚了方向的突变——如果预测的点序列出现锯齿，相邻方向向量间的差异会很大，从而产生较大的损失
    - 设计动机：传统的坐标回归损失（如 L1/L2）只约束每个点的绝对位置，但对"点与点之间的相对关系"没有直接约束。即使每个点的坐标误差很小，累积的方向不一致也会导致可见的抖动。VDDL 通过显式建模方向变化来直接优化几何平滑性

### 损失函数 / 训练策略

总损失为分类损失、点回归损失和 VDDL 的加权和：$L = L_{cls} + \lambda_1 L_{pts} + \lambda_2 L_{VDDL}$，其中 $L_{pts}$ 为标准的逐点 L1 回归损失。训练沿用 MapTR 的策略，使用匈牙利匹配进行 GT 分配，端到端训练。

## 实验关键数据

### 主实验

| 数据集 | Backbone | 指标(mAP) | 本文(ADMap) | MapTR | 提升 |
|--------|----------|-----------|------------|-------|------|
| nuScenes val | R50 | mAP | 54.5 | 50.3 | +4.2 |
| nuScenes val | R50+SECOND | mAP | 68.0 | 62.5 | +5.5 |
| Argoverse2 val | R50 | mAP | 64.7 | 61.3 | +3.4 |
| Argoverse2 val | R50+SECOND | mAP | 68.7 | 67.4(MapTRv2) | +1.3 |

详细子类性能（nuScenes val, R50）：

| 方法 | 人行横道 | 车道分割线 | 路缘 | mAP |
|------|---------|-----------|------|-----|
| MapTR | 51.5 | 46.3 | 53.1 | 50.3 |
| ADMap | 56.2 | 49.4 | 57.9 | 54.5 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| MapTR baseline | 50.3 mAP | 基线方法 |
| + MPN | mAP提升 | 多尺度感知增强BEV特征 |
| + IIA | mAP进一步提升 | 实例间交互提升全局一致性 |
| + VDDL | mAP再次提升 | 方向约束消除抖动 |
| Full (MPN+IIA+VDDL) | 54.5 mAP | 三个模块协同最优 |

### 关键发现

- ADMap 在 nuScenes 上比 MapTR 平均提升约 4.2 mAP，在所有三种地图元素类别上都有一致提升
- 使用 R50+SECOND（融合 LiDAR）配置时，ADMap 达到 68.0 mAP，超越 MapTRv2 的 69.0 mAP（后者需要更复杂的设计）
- VDDL 在定性可视化中效果最为明显——预测的车道线明显更加平滑，锯齿现象大幅减少
- FPS 几乎不受影响（14.8 vs 15.1），说明增加的计算开销极小
- 在 Argoverse2 这个更大更具挑战性的数据集上也取得了一致的提升

## 亮点与洞察

- 精准抓住了矢量化地图构建中被忽视但对下游任务至关重要的"点序列平滑性"问题
- VDDL 是一个简洁而有效的损失设计——通过方向差异来约束几何平滑性，计算简单但效果显著
- "实例间→实例内"的级联式关系建模思路很有启发性，体现了从全局到局部的设计哲学
- 方法作为 MapTR 的增强模块，向后兼容性好，且 FPS 几乎不降

## 局限与展望

- 方法建立在 MapTR 框架之上，对其他地图构建框架（如 VectorMapNet、PivotNet）的适配性需要进一步验证
- VDDL 假设地图元素应该是平滑的，但在急弯或交叉路口区域，方向的急剧变化是合理的，损失设计需要对这种情况做特殊处理
- 仅使用了 nuScenes 和 Argoverse2 两个数据集，未在更多场景（如城市密集道路、高速公路）中测试
- 实时性虽然没有显著下降，但在嵌入式计算平台上的部署效率需要进一步评估
- 代码公开但作者提到是"额外复制版"，可能与论文中的完整实现有差异

## 相关工作与启发

- MapTR 和 MapTRv2 是矢量化地图构建的代表性工作，ADMap 在其基础上聚焦几何质量的提升
- 方向差异损失的思想可以推广到其他需要几何平滑性的任务，如轨迹预测、曲线检测等
- 实例间交互的设计与 DETR 中的 object query 交互有类似思路，但更强调空间关系的约束

## 评分
- 新颖性: ⭐⭐⭐ 问题聚焦精准，VDDL设计简洁有效，但整体框架创新性一般
- 实验充分度: ⭐⭐⭐⭐ 在两个主流数据集上全面验证，消融实验清晰，可视化直观
- 写作质量: ⭐⭐⭐ 问题定义清晰，方法描述清楚
- 价值: ⭐⭐⭐⭐ 解决了实际部署中的重要问题，代码开源且结果可复现

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] MapQaTor: An Extensible Framework for Efficient Annotation of Map-Based QA Datasets](../../ACL2025/others/mapqator_an_extensible_framework_for_efficient_annotation_of_map-based_qa_datase.md)
- [\[ECCV 2024\] An Incremental Unified Framework for Small Defect Inspection](an_incremental_unified_framework_for_small_defect_inspection.md)
- [\[ECCV 2024\] A Framework for Efficient Model Evaluation through Stratification, Sampling, and Estimation](a_framework_for_efficient_model_evaluation_through_stratific.md)
- [\[ECCV 2024\] HiEI: A Universal Framework for Generating High-quality Emerging Images from Natural Images](hiei_a_universal_framework_for_generating_high-quality_emerging_images_from_natu.md)
- [\[ACL 2025\] Autalic: A Dataset for Anti-Autistic Ableist Language In Context](../../ACL2025/others/autalic_a_dataset_for_anti-autistic_ableist_language_in_context.md)

</div>

<!-- RELATED:END -->
