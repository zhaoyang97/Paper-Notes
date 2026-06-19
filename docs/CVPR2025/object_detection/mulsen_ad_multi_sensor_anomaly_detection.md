---
title: >-
  [论文解读] MulSen-AD: Multi-Sensor Object Anomaly Detection
description: >-
  [CVPR 2025][目标检测][多传感器融合] 提出首个多传感器异常检测数据集 MulSen-AD，整合 RGB 相机、红外热成像和激光扫描三种模态，以及基线方法 MulSen-TripleAD，通过决策级融合实现 96.1% AUROC 的物体级异常检测。 领域现状：工业异常检测主要依赖单传感器方法…
tags:
  - "CVPR 2025"
  - "目标检测"
  - "多传感器融合"
  - "异常检测"
  - "RGB-红外-点云"
  - "工业质检"
  - "数据集"
---

# MulSen-AD: Multi-Sensor Object Anomaly Detection

**会议**: CVPR 2025  
**arXiv**: [2412.14592](https://arxiv.org/abs/2412.14592)  
**代码**: [https://github.com/ZZZBBBZZZ/MulSen-AD](https://github.com/ZZZBBBZZZ/MulSen-AD)  
**领域**: 目标检测 / 异常检测  
**关键词**: 多传感器融合, 异常检测, RGB-红外-点云, 工业质检, 数据集

## 一句话总结

提出首个多传感器异常检测数据集 MulSen-AD，整合 RGB 相机、红外热成像和激光扫描三种模态，以及基线方法 MulSen-TripleAD，通过决策级融合实现 96.1% AUROC 的物体级异常检测。

## 研究背景与动机

**领域现状**：工业异常检测主要依赖单传感器方法，RGB 专注表面缺陷但无法检测内部缺陷，3D 扫描捕捉几何异常但忽视热异常，红外检测内部缺陷但缺少颜色纹理信息。

**现有痛点**：现有数据集（MVTec-AD、Real3D-AD 等）都只包含单一模态，无法全面覆盖实际工厂中的各类异常。

**核心矛盾**：不同类型的异常需要不同传感器检测，单一传感器无法胜任所有场景。

**本文目标**：构建首个多传感器异常检测数据集和基准，并验证多传感器融合的必要性。

**核心 idea**：整合 RGB + 红外 + 高精度点云三种模态，覆盖外观、内部和几何三类异常。

## 方法详解

### 整体框架

MulSen-AD 数据集包含 15 类工业产品，使用工业 RGB 相机（1920×1200）、红外锁相热成像（640×480）和 Creaform MetraSCAN 激光扫描仪（0.03mm 精度）采集。MulSen-TripleAD 基线方法对三种模态独立处理后进行决策级门控融合。

### 关键设计

1. **多传感器数据采集流水线**:

    - 功能：构建高质量多模态异常检测数据
    - 核心思路：红外锁相热成像通过周期性热激发检测内部异常（如胶囊内部破损、太阳能板损伤）；RGB 相机安装在 UR5 机械臂上确保均匀光照；手持激光扫描仪进行 360° 扫描消除盲区
    - 设计动机：三种传感器互补——RGB 检测表面、红外检测内部、点云检测几何

2. **模态特异性标注**:

    - 功能：精确标注各模态可见的异常
    - 核心思路：仅当异常在某模态中可见时才为该模态标注。例如胶囊内部异常仅标注红外图像，不标注 RGB 和点云
    - 设计动机：不同异常在不同模态中可见性不同，避免强制标注不可见的异常

3. **MulSen-TripleAD 决策级融合**:

    - 功能：融合三种模态的异常检测结果
    - 核心思路：为每种模态建立独立的记忆库进行无监督异常检测，通过决策门控单元在最终阶段融合三个模态的异常得分
    - 设计动机：决策级融合最小化传感器间的数据差异影响，比特征级融合更鲁棒

### 损失函数 / 训练策略

无监督设置，仅使用正常样本训练。每类约 33 个异常样本用于测试。数据集共 2035 个样本，涵盖 15 类产品、14 种异常类型。

## 实验关键数据

### 主实验

| 方法 | 模态 | 物体级 AUROC |
|------|------|-------------|
| 最佳单模态 | RGB/IR/PC | ~90% |
| MulSen-TripleAD | RGB+IR+PC | **96.1%** |

### 关键发现

- 多传感器融合比任何单传感器方法提升显著
- 不同产品对不同传感器的依赖程度不同
- 红外模态对内部缺陷检测不可替代

## 亮点与洞察

- 首个包含 RGB + 红外 + 点云的工业异常检测数据集
- 手动制造了 14 种真实异常类型（裂缝、孔洞、弯曲、异物等）
- 模态特异性标注策略尊重了各传感器的检测能力边界

## 局限与展望

- 数据集规模相对较小（2035 样本）
- 目前仅用简单的决策级融合，更复杂的融合策略有待探索
- 三种传感器的采集需要不同设备和流程，实际部署成本较高

## 评分

- 新颖性：8/10 — 首个多传感器 AD 数据集
- 技术深度：6/10 — 基线方法较简单
- 实验充分度：7/10 — 数据集构建充分但模型对比有限
- 写作质量：7/10 — 清晰详实

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Multi-Sensor Object Anomaly Detection: Unifying Appearance, Geometry, and Internal Properties](multi-sensor_object_anomaly_detection_unifying_appearance_geometry_and_internal_.md)
- [\[CVPR 2025\] MI-DETR: An Object Detection Model with Multi-time Inquiries Mechanism](mi-detr_an_object_detection_model_with_multi-time_inquiries_mechanism.md)
- [\[CVPR 2025\] Towards RAW Object Detection in Diverse Conditions](towards_raw_object_detection_in_diverse_conditions.md)
- [\[CVPR 2025\] One-for-More: Continual Diffusion Model for Anomaly Detection](one-for-more_continual_diffusion_model_for_anomaly_detection.md)
- [\[CVPR 2025\] Distribution Prototype Diffusion Learning for Open-set Supervised Anomaly Detection](distribution_prototype_diffusion_learning_for_open-set_supervised_anomaly_detect.md)

</div>

<!-- RELATED:END -->
