---
title: >-
  [论文解读] DAMap: Distance-aware MapNet for High Quality HD Map Construction
description: >-
  [ICCV 2025][自动驾驶][HD Map] 揭示当前HD地图构建方法在高质量预测上的两大固有缺陷——不恰当的分类标签与次优的任务特征，提出DAMap（含DAFL、HLS、TMDA三个组件）系统性地解决任务错位问题，在NuScenes和Argoverse2上多个基线方法上一致提升2-3 mAP。
tags:
  - "ICCV 2025"
  - "自动驾驶"
  - "HD Map"
  - "高质量预测"
  - "任务对齐"
  - "注意力机制"
  - "Focal Loss"
---

# DAMap: Distance-aware MapNet for High Quality HD Map Construction

**会议**: ICCV 2025  
**arXiv**: [2510.22675](https://arxiv.org/abs/2510.22675)  
**代码**: [github.com/jpdong-xjtu/DAMap](https://github.com/jpdong-xjtu/DAMap)  
**领域**: Autonomous Driving / 在线高精地图构建  
**关键词**: HD Map, 高质量预测, 任务对齐, Deformable Attention, Focal Loss

## 一句话总结

揭示当前HD地图构建方法在高质量预测上的两大固有缺陷——不恰当的分类标签与次优的任务特征，提出DAMap（含DAFL、HLS、TMDA三个组件）系统性地解决任务错位问题，在NuScenes和Argoverse2上多个基线方法上一致提升2-3 mAP。

## 研究背景与动机

在线向量化HD地图构建对自动驾驶安全至关重要。以MapTRv2为代表的DETR式方法虽取得了显著进展，但在**高质量预测**（同时具有高分类得分和高定位精度）上表现很差：
- 在0.5m阈值下即使分类阈值降到0.6，recall仅35%
- 导致危险场景（如行人过街标志漏检）

作者分析了两大根本原因：

**分类标签不恰当**：one-to-many匹配中，每个GT实例对应多个候选样本（MapTRv2中7个），但它们共享相同的分类标签"1"。定位质量差的候选也获得标签"1"→模型学会输出"高分类+低定位"的预测

**任务特征次优**：分类和定位共用交叉注意力提取的实例特征。分类需要实例内部显著区域的语义信息，定位需要实例边界的精确位置→共用特征导致两个任务都达不到最优

## 方法详解

### 整体框架

标准BEV感知pipeline：多视角图像 → 共享backbone → BEV特征 → Transformer decoder（自注意力+交叉注意力+FFN）→ 分类和定位头。DAMap在此基础上引入三个即插即用组件。

### 关键设计

1. **Distance-aware Focal Loss (DAFL)**：

    - 核心思想：用定位质量作为分类标签，而非二值标签
    - 定位损失 $\mathcal{L}_{dist}$ 通过最大似然估计转为概率：$P_{dist}^i = e^{-\lambda \mathcal{L}_{dist}^i}$
        - 当 $\mathcal{L}_{dist} = 0$ 时，$P_{dist} = 1$（完美定位→高标签）
        - 当 $\mathcal{L}_{dist} \to \infty$ 时，$P_{dist} \to 0$（差定位→低标签）
    - 用连续标签替代二值标签得到DAFL：
    $\text{DAFL}(p, y) = -(y-p)^\gamma (y\log(p) + (1-y)\log(1-p))$
    - $y \in [0,1]$ 为定位置信度，使分类得分能反映定位质量
    - 负样本 $(y=0)$ 时与标准Focal Loss完全一致

2. **Hybrid Loss Scheme (HLS)**：

    - 问题：decoder早期query随机初始化导致定位质量差，影响DAFL的训练信号
    - 解决：前 $L_1$ 层decoder用标准Focal Loss，后 $L_2$ 层用DAFL
    - 利用decoder级联特性——后面层的预测定位精度更高，更适合DAFL
    - 总分类损失：$\mathcal{L}_{cls} = \sum_{l=1}^{L_1} \text{FL}(p,y) + \sum_{l=1}^{L_2} \text{DAFL}(p,y)$
    - HLS+DAFL是纯损失函数改进，**推理时零额外参数和计算量**

3. **Task Modulated Deformable Attention (TMDA)**：

    - 将query通道翻倍，一半负责分类、一半负责定位
    - 经自注意力后split为 $\mathbf{Q}_{cls}$ 和 $\mathbf{Q}_{loc}$
    - 关键设计选择：
        - 任务特定的注意力权重：$\mathbf{A}_{cls} = W_a \mathbf{Q}_{cls}$, $\mathbf{A}_{loc} = W_a' \mathbf{Q}_{loc}$
        - 任务共享的采样偏移：$\Delta r = W_p \text{Cat}(\mathbf{Q}_{cls}, \mathbf{Q}_{loc})$
    - 设计理由：(1) 同时学习特定权重和偏移的变量太多优化困难；(2) 偏移量优化目标无界，本身更难
    - 输出：$\hat{\mathbf{Q}}_{cls} = \text{Softmax}(\mathbf{A}_{cls})\mathbf{V}$, $\hat{\mathbf{Q}}_{loc} = \text{Softmax}(\mathbf{A}_{loc})\mathbf{V}$
    - 各自经独立FFN增强后送入对应任务头

### 损失函数 / 训练策略

- 定位损失：L1距离衡量预测点与GT点的偏差
- 分类损失：HLS混合Focal Loss和DAFL
- 匹配：沿用MapTRv2的one-to-many实例匹配+点级匹配
- $\lambda$ 超参数调节定位损失到概率的转换灵敏度

## 实验关键数据

### 主实验

| 方法 | Epoch | mAP(hard) | mAP(easy) | 提升(hard) | 提升(easy) |
|------|-------|----------|----------|----------|----------|
| **NuScenes ResNet-50** | | | | | |
| MapTRv2† | 24 | 36.6 | 60.4 | - | - |
| MapTRv2+**Ours** | 24 | 39.0 | 62.8 | +2.4 | +2.4 |
| MapQR† | 24 | 43.3 | 66.4 | - | - |
| MapQR+**Ours** | 24 | **46.0** | **68.8** | +2.6 | +2.4 |
| Mask2Map | 24 | - | 71.6 | - | - |
| Mask2Map+**Ours** | 24 | - | **72.6** | - | +1.0 |
| MapTRv2† | 110 | 44.9 | 68.3 | - | - |
| MapTRv2+**Ours** | 110 | **47.4** | **70.4** | +2.5 | +2.1 |
| **Argoverse2 ResNet-50** | | | | | |
| MapTRv2† | 6 | 38.1 | 63.6 | - | - |
| MapTRv2+**Ours** | 6 | 40.9 | 66.2 | +2.8 | +2.6 |
| MapQR† | 6 | 41.1 | 65.4 | - | - |
| MapQR+**Ours** | 6 | **43.6** | **67.4** | +2.5 | +2.0 |

### 消融实验

| 组件 | AP_ped | AP_div | AP_bou | mAP(easy) | 增益 |
|------|--------|--------|--------|----------|------|
| Baseline | 58.1 | 60.8 | 62.3 | 60.4 | - |
| +DAFL | 58.6 | 61.1 | 63.1 | 60.9 | +0.5 |
| +DAFL+HLS | 58.5 | 62.9 | 63.4 | 61.6 | +1.2 |
| +TMDA(单独) | 58.5 | 63.1 | 63.2 | 61.6 | +1.2 |
| +DAFL+HLS+TMDA | **58.5** | **64.7** | **65.1** | **62.8** | **+2.4** |

| TMDA设计变体 | mAP | 参数量 |
|-------------|------|-------|
| Baseline | 60.4 | 40M |
| Setting 1: 全任务特定(权重+偏移) | 60.7 | 52M |
| Setting 2: 特定偏移+共享权重 | 60.9 | 52M |
| **Setting 4(Ours): 共享偏移+特定权重** | **61.6** | 52M |

### 关键发现

- 三个组件互补，解决不同问题：DAFL解决标签错位，TMDA解决特征冲突，HLS释放DAFL潜力
- 仅HLS+DAFL（无推理额外开销）就能带来1.1-2.1 mAP提升
- TMDA的"共享偏移+特定权重"设计优于直觉上的"全特定"设计——减少优化变量更有效
- 在不同数据集(NuScenes/Argoverse2)、基线(MapTRv2/MapQR/Mask2Map)、backbone(ResNet50/SwinB)、训练schedule(6/24/110)上一致有效
- 加入Centerline类别后提升更大（mAP提升4-5），说明对多类别场景更有价值

## 亮点与洞察

- **问题分析精准**：recall@0.5m的分析清晰揭示了现有方法的高质量预测困境
- **DAFL的设计极其巧妙**：通过似然估计将无界的定位损失转为[0,1]概率作为soft标签，与QFL思路相关但做法更优雅
- **HLS的cascade洞察**：利用decoder级联的性质——早期用FL收敛、后期用DAFL细化——体现了对训练动态的深入理解
- 所有组件即插即用，可与任何DETR式HD map方法组合

## 局限与展望

- DAFL中$\lambda$超参数需要调节定位损失到概率的敏感度
- TMDA引入额外参数（40M→52M），可能在资源受限场景下需权衡
- 未探索Online/streaming场景下长序列时间建模的兼容性
- 高质量预测的recall虽提升但绝对值仍有较大提升空间
- 仅关注分类-定位错位，未考虑不同map element类型间的关系建模

## 相关工作与启发

- **GFL/VFNet**：2D检测中的质量-分类对齐方法，本文将思想引入HD map构建
- **MapTRv2/MapQR**：DETR式HD map基线方法，DAMap与它们完全互补
- **Double-Head R-CNN/TSD**：任务解耦的先驱，TMDA是在Deformable Attention层面的解耦
- 高质量预测的对齐思路可推广到其他BEV感知任务（3D检测、运动规划等）

## 评分

- 新颖性: ⭐⭐⭐⭐ 三个组件各有设计亮点，但核心思想借鉴自2D检测领域
- 实验充分度: ⭐⭐⭐⭐⭐ 多数据集/多基线/多backbone/多schedule，消融非常详细
- 写作质量: ⭐⭐⭐⭐ 问题动机分析出色，方法描述清晰
- 价值: ⭐⭐⭐⭐ 即插即用设计实用性强，尤其HLS+DAFL零推理开销的特性很有工程价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] SDTagNet: Leveraging Text-Annotated Navigation Maps for Online HD Map Construction](../../NeurIPS2025/autonomous_driving/sdtagnet_leveraging_text-annotated_navigation_maps_for_online_hd_map_constructio.md)
- [\[ICML 2025\] SafeMap: Robust HD Map Construction from Incomplete Observations](../../ICML2025/autonomous_driving/safemap_robust_hd_map_construction_from_incomplete_observations.md)
- [\[ICCV 2025\] EMD: Explicit Motion Modeling for High-Quality Street Gaussian Splatting](emd_explicit_motion_modeling_for_high-quality_street_gaussian_splatting.md)
- [\[CVPR 2025\] Uncertainty-Instructed Structure Injection for Generalizable HD Map Construction](../../CVPR2025/autonomous_driving/uncertainty-instructed_structure_injection_for_generalizable_hd_map_construction.md)
- [\[CVPR 2026\] AMap: Distilling Future Priors for Ahead-Aware Online HD Map Construction](../../CVPR2026/autonomous_driving/amap_distilling_future_priors_for_ahead-aware_online_hd_map_construction.md)

</div>

<!-- RELATED:END -->
