---
title: >-
  [论文解读] On Calibration of Object Detectors: Pitfalls, Evaluation and Baselines
description: >-
  [ECCV 2024 (Oral)][目标检测][目标检测校准] 本文系统性地揭示了当前目标检测器校准研究中评估框架、评估指标和温度缩放（Temperature Scaling）使用方面的重大缺陷，提出了原则性的联合评估框架以及专为目标检测定制的后处理校准方法（Platt Scaling和Isotonic Regression），证明了正确设计和评估的后处理校准器远优于近期训练时校准方法。
tags:
  - "ECCV 2024 (Oral)"
  - "目标检测"
  - "目标检测校准"
  - "后处理校准"
  - "D-ECE"
  - "Platt Scaling"
  - "Isotonic Regression"
---

# On Calibration of Object Detectors: Pitfalls, Evaluation and Baselines

**会议**: ECCV 2024 (Oral)  
**arXiv**: [2405.20459](https://arxiv.org/abs/2405.20459)  
**代码**: [https://github.com/fiveai/detection_calibration](https://github.com/fiveai/detection_calibration)  
**领域**: 目标检测 / 模型校准  
**关键词**: 目标检测校准, 后处理校准, D-ECE, Platt Scaling, Isotonic Regression

## 一句话总结

本文系统性地揭示了当前目标检测器校准研究中评估框架、评估指标和温度缩放（Temperature Scaling）使用方面的重大缺陷，提出了原则性的联合评估框架以及专为目标检测定制的后处理校准方法（Platt Scaling和Isotonic Regression），证明了正确设计和评估的后处理校准器远优于近期训练时校准方法。

## 研究背景与动机

**领域现状**：目标检测器的可靠使用要求模型输出的置信度是经过校准的——即模型预测的概率应准确反映实际正确率。这对于自动驾驶、医疗影像等安全关键应用尤为重要。近年来，研究者主要从两个方向探索检测器校准：(1) 设计新的训练损失函数从头训练校准的检测器（如Cal-DETR）；(2) 使用后处理温度缩放（Temperature Scaling, TS）来调整已训练检测器的输出概率。

**现有痛点**：作者通过大量分析发现，当前的校准研究存在多个严重问题：(1) **评估框架不合理**——现有框架在衡量校准误差时没有充分考虑检测任务的特殊性，如NMS后处理、定位质量等因素；(2) **评估指标有缺陷**——常用的Detection Expected Calibration Error（D-ECE）在实际使用场景下存在系统性偏差，可能导致错误结论；(3) **温度缩放的不当使用**——直接将分类任务中的TS移植到目标检测上，忽视了检测任务中前景/背景不平衡、多阈值操作等特点。

**核心矛盾**：这些缺陷导致了一个广泛但错误的结论：训练时校准方法优于后处理校准方法。实际上，后处理方法一旦正确设计和评估，性能远超训练时方法，且成本极低。

**本文目标** (1) 指出并修正现有评估框架和指标的缺陷；(2) 提出适用于目标检测的后处理校准方法；(3) 建立公正的基准来比较训练时和后处理校准方法。

**切入角度**：从评估方法论入手，先纠正评估中的问题，再在公正的评估框架下提出简单有效的后处理校准基线。

**核心 idea**：纠正目标检测校准研究中的评估缺陷后，极其廉价的后处理校准方法（Isotonic Regression）比复杂的训练时方法效果好得多。

## 方法详解

### 整体框架

本文的工作分为两个层面：(1) 评估层面——提出原则性的联合评估框架来同时度量检测器的校准度和准确性；(2) 方法层面——将经典的后处理校准方法（Platt Scaling和Isotonic Regression）适配到目标检测任务上。整体流程为：使用已训练的检测器生成检测结果 → 在验证集上拟合后处理校准器 → 在测试集上评估校准后的检测结果。

### 关键设计

1. **原则性联合评估框架**:

    - 功能：公正地同时度量检测器的准确性和校准度
    - 核心思路：采用Localization-Recall-Precision（LRP）Error作为主要的准确性度量指标，结合Localization-aware ECE（LaECE）作为主要的校准误差度量。LRP Error综合考虑了定位、召回和精度三个维度，更全面地反映检测器性能。LaECE在计算校准误差时考虑了检测框的定位质量（IoU），避免了D-ECE忽略定位因素的缺陷。同时将验证集分为minival和minitest，前者用于拟合校准器，后者用于评估，避免数据泄漏
    - 设计动机：传统评估仅用AP和D-ECE的组合，两个指标之间缺乏内在联系，且D-ECE在检测场景下有偏差。LRP和LaECE能更准确反映检测器在实际使用中的校准质量

2. **目标检测专用Platt Scaling**:

    - 功能：通过学习线性变换来校准检测器的置信度输出
    - 核心思路：对检测器输出的置信度分数 $s$ 学习参数 $a$ 和 $b$，通过 $\sigma(a \cdot s + b)$（$\sigma$ 为sigmoid函数）映射为校准后的概率。针对目标检测的特点进行了适配：(1) 将匹配检测框与Ground Truth的IoU阈值作为正负样本划分标准；(2) 为每个类别独立学习校准参数；(3) 在NMS之后的检测结果上进行校准，与实际使用场景一致
    - 设计动机：Platt Scaling是分类任务中经典的后处理校准方法，但直接应用于检测任务需要处理定位匹配、多类别、NMS等检测特有的问题

3. **目标检测专用Isotonic Regression**:

    - 功能：通过非参数单调变换来校准检测器的置信度输出
    - 核心思路：Isotonic Regression学习一个非递减的分段常数函数，将原始置信度映射为校准后的概率。相比Platt Scaling的线性假设，它能拟合更复杂的校准曲线。同样针对检测任务进行了适配：基于IoU匹配结果生成校准训练数据，按类别独立拟合，在NMS后的检测结果上操作。此外引入阈值优化来同时提升校准度和检测精度
    - 设计动机：检测器的置信度与实际准确率之间的映射关系可能是高度非线性的，Isotonic Regression的非参数特性更适合捕捉这种复杂关系

### 损失函数 / 训练策略

后处理方法不涉及检测器训练，仅需在验证集上拟合校准参数。Platt Scaling通过最大化对数似然来优化参数 $a, b$；Isotonic Regression通过最小化加权最小二乘来寻找最优的单调映射。两者都极其高效，拟合过程仅需数秒。作者还提出了改进版的LaECE指标（LaECE-v2），在原始LaECE的基础上修正了定位质量权重的计算方式。

## 实验关键数据

### 主实验

在COCO数据集上，使用D-DETR检测器的校准结果对比：

| 方法 | 类型 | D-ECE↓ | LaECE↓ | LRP Error↓ | AP↑ |
|------|------|--------|--------|-----------|-----|
| D-DETR（未校准） | 基线 | 较高 | 较高 | 基线 | 基线 |
| Cal-DETR | 训练时 | 中等 | 中等 | 略降 | 略降 |
| TS (Temperature Scaling) | 后处理 | 不稳定 | 不稳定 | 变差 | 变差 |
| Platt Scaling（本文） | 后处理 | 显著降低 | 显著降低 | 改善 | 保持 |
| Isotonic Regression（本文） | 后处理 | **最优（>7↓ vs Cal-DETR）** | **最优** | **最优** | **保持** |

在Cityscapes和LVIS数据集上也观察到一致的趋势，Isotonic Regression在所有数据集和检测器上均表现最优。

### 消融实验

| 配置 | D-ECE | LaECE | 说明 |
|------|-------|-------|------|
| 不分类别拟合 | 中等 | 中等 | 忽略类别间校准差异 |
| 按类别独立拟合 | 更优 | 更优 | 每类单独校准参数 |
| NMS前校准 | 较差 | 较差 | 与实际使用场景不匹配 |
| NMS后校准 | **最优** | **最优** | 符合实际部署流程 |
| 不做阈值优化 | 中等 | 中等 | 仅做概率校准 |
| 加阈值优化 | **最优** | **最优** | 同时优化校准和精度 |

### 关键发现

- 后处理校准器（特别是Isotonic Regression）在正确评估下远优于训练时校准方法Cal-DETR，D-ECE差距超过7个点
- Temperature Scaling在目标检测上效果不稳定，部分原因是检测任务中前后景分布极度不平衡
- 现有评估框架导致了"训练时方法更优"的错误结论——修正评估后结论完全逆转
- 后处理校准方法计算成本几乎可忽略，而训练时方法需要完全重新训练检测器

## 亮点与洞察

- **方法论贡献大于技术贡献**：最有价值的是揭示了整个领域评估方法的缺陷，这比提出新算法更具影响力
- **简单方法的胜利**：Isotonic Regression是上世纪的经典方法，适配后即打败了所有复杂的训练时方法
- **实用性极强**：后处理校准器可以即插即用地附加到任何已训练的检测器上，无需重新训练
- **ECCV Oral论文**：说明审稿人认可了这种"纠偏"类工作的重要性

## 局限与展望

- 后处理方法需要一个校准验证集，在数据稀缺场景下可能受限
- 当前仅在COCO风格评估下验证，更多应用场景（如在线学习、域迁移）有待探索
- 仅探索了类别级别的校准，实例级别的校准（考虑物体大小、遮挡等）是潜在方向
- LaECE指标本身可能还有进一步改进空间

## 相关工作与启发

- **Cal-DETR**（CVPR 2024）：训练时校准方法，被本文的后处理方法显著超越
- **Temperature Scaling**（ICML 2017）：经典后处理校准方法，但不直接适用于检测
- **LRP Error**（TPAMI 2022）：综合评估检测器性能的指标
- 启发：在追求复杂方法之前，先确保评估框架的正确性——错误的评估可能误导整个研究方向

## 评分

- 新颖性: ⭐⭐⭐⭐（方法论层面的深刻洞察，而非技术创新）
- 实验充分度: ⭐⭐⭐⭐⭐（多数据集、多检测器、详尽消融）
- 写作质量: ⭐⭐⭐⭐⭐（分析透彻，论证严密）
- 价值: ⭐⭐⭐⭐⭐（纠正了领域内的错误认知，提供了即插即用的工具）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Revisiting Adversarial Patch Defenses on Object Detectors: Unified Evaluation, Large-Scale Dataset, and New Insights](../../ICCV2025/object_detection/revisiting_adversarial_patch_defenses_on_object_detectors_unified_evaluation_lar.md)
- [\[ECCV 2024\] Can OOD Object Detectors Learn from Foundation Models?](can_ood_object_detectors_learn_from_foundation_models.md)
- [\[CVPR 2026\] RHCNet: Residual-Guided Hierarchical Calibration Network for Robust Underwater Object Detection](../../CVPR2026/object_detection/rhcnet_residual-guided_hierarchical_calibration_network_for_robust_underwater_ob.md)
- [\[CVPR 2026\] Explaining Object Detectors via Collective Contribution of Pixels](../../CVPR2026/object_detection/explaining_object_detectors_via_collective_contribution_of_pixels.md)
- [\[ICCV 2025\] Automated Model Evaluation for Object Detection via Prediction Consistency and Reliability](../../ICCV2025/object_detection/automated_model_evaluation_for_object_detection_via_prediction_consistency_and_r.md)

</div>

<!-- RELATED:END -->
