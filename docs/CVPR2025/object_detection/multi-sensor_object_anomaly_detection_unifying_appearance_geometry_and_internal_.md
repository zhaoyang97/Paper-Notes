---
title: >-
  [论文解读] Multi-Sensor Object Anomaly Detection: Unifying Appearance, Geometry, and Internal Properties
description: >-
  [目标检测] 提出 MulSen-AD，首个融合 RGB 相机、激光扫描仪和红外热成像三种传感器的工业物体异常检测数据集（15 类产品、14 种异常），并设计 MulSen-TripleAD 决策级融合基线方法，实现 96.1% AUROC，证明多传感器融合显著优于单传感器方法。 工业异常检测面临的核心矛盾： 1. 单一传感…
tags:
  - "目标检测"
---

# Multi-Sensor Object Anomaly Detection: Unifying Appearance, Geometry, and Internal Properties

## 一句话总结

提出 MulSen-AD，首个融合 RGB 相机、激光扫描仪和红外热成像三种传感器的工业物体异常检测数据集（15 类产品、14 种异常），并设计 MulSen-TripleAD 决策级融合基线方法，实现 96.1% AUROC，证明多传感器融合显著优于单传感器方法。

## 研究背景与动机

工业异常检测面临的核心矛盾：

1. **单一传感器无法覆盖所有异常类型**：
    - RGB 摄像头擅长检测表面缺陷（划痕、颜色异常），但无法发现内部缺陷
    - 激光扫描仪能捕捉几何变形（弯曲、褶皱），但对纹理和颜色不敏感
    - 红外热成像能揭示内部缺陷（内部裂痕、零件脱落），但牺牲了颜色和纹理信息

2. **现有数据集的局限**：
    - MVTec-AD、VisA 等仅使用 RGB 图像
    - MVTec3D-AD 使用 RGB-D，但深度图质量有限
    - Real3D-AD 仅使用点云
    - 没有任何数据集同时包含三种模态的高质量标注

3. **异常检测中的"盲区"问题**：单传感器方法在实际工厂场景中经常漏检——例如胶囊内部断裂在 RGB 下完全不可见，弹簧垫变形在红外下难以察觉。

## 方法详解

### 整体框架

MulSen-AD 框架包含两部分：
1. **MulSen-AD 数据集**：三种传感器的数据采集、标注和组织
2. **MulSen-TripleAD 基线模型**：基于 PatchCore 的多传感器决策级融合方法

### 关键设计

#### 1. 多传感器数据采集系统

- **RGB 相机** (Daheng MER2-230-168U3C)：1920×1200 分辨率，搭载 UR5 机械臂，配侧光源确保均匀照明
- **红外热成像** (Noverlteq TWILIS-180 + FLIR A600)：640×480 分辨率，7.5-14μm 波长范围，通过周期性热刺激（30-180秒）检测温度异常
- **3D 激光扫描仪** (Creaform MetraSCAN 750)：0.03mm 精度、0.05mm 分辨率，手持式 360° 扫描，翻转双面扫描+ICP 精对齐

15 类工业产品涵盖金属、塑料、纤维、橡胶、半导体、复合材料，14 种异常类型包括裂纹、孔洞、断裂、褶皱、划痕、异物、标签错误、弯曲、颜色缺陷和脱落等。

#### 2. 模态特异性标注策略

标注遵循"可见即标注"原则：
- 如果某异常仅在红外图中可见（如胶囊内部断裂），则只标注红外图
- 使用 LabelMe 标注 RGB 和红外的像素级掩码
- 使用 Geomagic Design X 手动选择点云的异常区域

#### 3. MulSen-TripleAD 决策级融合

三个组件：
- **多模态特征提取**：DINO 提取 RGB 和红外特征，PointMAE 提取点云特征
- **多模态记忆库**：为 RGB/红外/点云分别建立基于正常样本的 PatchCore 记忆库 $\mathcal{M}_{rgb}$、$\mathcal{M}_{ir}$、$\mathcal{M}_{pc}$
- **决策门控单元**：受 M3DM 启发的可学习 OCSVM，融合三个模态的异常分数：

$$S = \mathcal{G}_a(\phi(\mathcal{M}_{rgb}, f_{rgb}), \phi(\mathcal{M}_{pt}, f_{pt}), \phi(\mathcal{M}_{ir}, f_{ir}))$$

### 损失函数

MulSen-TripleAD 基于 PatchCore 的距离度量评分，使用 OCSVM 进行决策融合。每个模态的异常分数为测试特征到记忆库最近邻的 L2 距离。

## 实验关键数据

### 主实验表

**物体级异常检测 AUROC（多传感器 vs 单传感器）**：

| 方法 | 模态 | 平均 AUROC↑ |
|------|------|-------------|
| PatchCore | RGB | 0.837 |
| InvAD | RGB | 0.892 |
| PatchCore | IR | 0.843 |
| InvAD | IR | 0.832 |
| M3DM | RGB+PC | 0.830 |
| **MulSen-TripleAD** | **RGB+IR+PC** | **0.961** |

**各类别详细表现（部分）**：

| 类别 | RGB最佳 | IR最佳 | PC最佳 | MulSen-TripleAD |
|------|---------|--------|--------|-----------------|
| Capsule | 0.940 | 0.960 | 0.923 | 高融合 |
| Screen | 0.884 | 0.572 | 0.788 | — |
| Solar Panel | 0.947 | 0.867 | 0.378 | — |
| Spring Pad | 0.980 | 0.913 | 0.512 | — |

### 关键发现

1. **多传感器融合显著优于单传感器**：96.1% vs 单传感器最佳 89.2%（RGB上的InvAD），提升近 7 个百分点
2. **异常分布的互补性**：Venn 图显示 43.7% 的异常可被三种传感器同时检测，但 9.4%（RGB独有）、9.2%（红外独有）、4.3%（点云独有）的异常只能被单一传感器捕获
3. **不同类别需要不同传感器**：Screen 的异常主要由 RGB 检测（0.884 vs IR 的 0.572），Solar Panel 的异常红外更擅长
4. **现有 RGB-only 方法在红外和点云上表现差**：如 CFA 在红外上平均仅 0.584 AUROC

### 数据集统计

- 总样本：2035（训练 1391 + 测试 644）
- 15 个类别，平均每类 4.8 种异常类型
- 异常像素占比：RGB 平均 0.372%，红外平均 0.451%，点云平均 4.98%

## 亮点与洞察

1. **填补了多传感器异常检测的数据集空白**：此前没有任何数据集同时包含 RGB+红外+高精度点云用于异常检测，MulSen-AD 的出现使这一领域的研究成为可能
2. **模态特异性标注策略合理**："可见即标注"避免了强制对齐三种模态标注的不合理性——一个表面划痕在红外中可能完全不可见
3. **决策级融合的简洁有效**：MulSen-TripleAD 仅在最终分数层面融合，各模态独立处理，便于扩展新传感器
4. **真实异常而非合成**：14 种手工引入的异常类型覆盖面广，且按材料属性选择了 15 类代表性工业产品

## 局限性

1. **仅支持物体级检测**：当前数据集和方法主要面向物体级异常检测，像素级定位能力有限
2. **标注成本高**：三种模态的数据采集和标注需要专业设备和大量人工
3. **决策级融合过于简单**：MulSen-TripleAD 未利用模态间的跨模态关系，特征级或数据级融合可能更有效
4. **数据规模有限**：15 类、2035 样本相比工业现实需求仍然较小
5. **传感器对齐问题**：三种传感器的数据在空间上并非像素级对齐，限制了像素级融合方法的应用

## 相关工作与启发

- **PatchCore** [Roth et al., 2022]：基于记忆库的异常检测经典方法，MulSen-TripleAD 直接扩展其到多模态
- **M3DM** [Wang et al., 2023]：首个 RGB+点云的 3D 异常检测方法，使用可学习 OCSVM 融合
- **MVTec-AD** [Bergmann et al., 2019]：最经典的 RGB 异常检测数据集，MulSen-AD 的定位是其多传感器版本
- **Real3D-AD** [Liu et al., 2023]：纯点云异常检测数据集，但缺乏外观和内部信息
- 启发：工业检测中"没有万能传感器"，多源信息融合是从实验室走向工厂的必经之路

## 评分

⭐⭐⭐⭐ (8/10)

- 创新性：⭐⭐⭐⭐ — 首个三模态工业异常检测数据集，问题定义清晰且重要
- 实用性：⭐⭐⭐⭐⭐ — 直接面向工业质检需求，三种传感器在工厂中均有广泛应用
- 实验充分度：⭐⭐⭐⭐ — 在数据集本身的 benchmark 上测试全面，但缺乏在其他数据集上的泛化验证
- 写作清晰度：⭐⭐⭐⭐ — 数据集构建部分详细，方法部分相对简洁（作为 dataset paper 合理）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Towards Anomaly-Aware Pre-Training and Fine-Tuning for Graph Anomaly Detection](../../ICLR2026/object_detection/towards_anomaly-aware_pre-training_and_fine-tuning_for_graph_anomaly_detection.md)
- [\[CVPR 2025\] MulSen-AD: Multi-Sensor Object Anomaly Detection](mulsen_ad_multi_sensor_anomaly_detection.md)
- [\[CVPR 2025\] MI-DETR: An Object Detection Model with Multi-time Inquiries Mechanism](mi-detr_an_object_detection_model_with_multi-time_inquiries_mechanism.md)
- [\[CVPR 2025\] Mr. DETR++: Instructive Multi-Route Training for Detection Transformers with MoE](mr_detr_instructive_multi-route_training_for_detection_transformers.md)
- [\[CVPR 2025\] AA-CLIP: Enhancing Zero-Shot Anomaly Detection via Anomaly-Aware CLIP](aa-clip_enhancing_zero-shot_anomaly_detection_via_anomaly-aware_clip.md)

</div>

<!-- RELATED:END -->
