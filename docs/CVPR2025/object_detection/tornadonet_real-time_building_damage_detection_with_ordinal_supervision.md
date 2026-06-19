---
title: >-
  [论文解读] TornadoNet: Real-Time Building Damage Detection with Ordinal Supervision
description: >-
  [CVPR 2025][目标检测][建筑损坏检测] TornadoNet 构建了首个针对龙卷风灾后街景建筑损坏评估的系统性 benchmark，通过对比 YOLO 系列（CNN）和 RT-DETR（Transformer）在五级损坏检测任务上的表现，并提出序数感知（ordinal-aware）监督策略，使 RT-DETR 的 mAP@0.5 提升 4.8 个百分点，证明了将损坏严重度的有序性质纳入损失函数设计的有效性。
tags:
  - "CVPR 2025"
  - "目标检测"
  - "建筑损坏检测"
  - "有序分类"
  - "龙卷风灾害"
  - "实时推理"
---

# TornadoNet: Real-Time Building Damage Detection with Ordinal Supervision

**会议**: CVPR 2025  
**arXiv**: [2603.11557](https://arxiv.org/abs/2603.11557)  
**代码**: [https://github.com/crumeike/TornadoNet](https://github.com/crumeike/TornadoNet)  
**领域**: 目标检测 / 灾害评估  
**关键词**: 建筑损坏检测, 有序分类, 目标检测, 龙卷风灾害, 实时推理

## 一句话总结
TornadoNet 构建了首个针对龙卷风灾后街景建筑损坏评估的系统性 benchmark，通过对比 YOLO 系列（CNN）和 RT-DETR（Transformer）在五级损坏检测任务上的表现，并提出序数感知（ordinal-aware）监督策略，使 RT-DETR 的 mAP@0.5 提升 4.8 个百分点，证明了将损坏严重度的有序性质纳入损失函数设计的有效性。

## 研究背景与动机

**领域现状**：灾后建筑损坏评估是应急响应中的关键环节。目前主要依赖航拍图像或人工现场巡查，街景级别的自动化损坏检测方法相对缺乏。深度学习目标检测模型（YOLO、DETR 等）已在通用场景中取得优异表现，但在灾害场景中的系统评测和适配研究不足。

**现有痛点**：(1) 缺乏标准化数据集——现有灾害数据集多为航拍视角，缺少高分辨率街景图像和多级损坏标注；(2) 损坏级别被当作无序类别处理——标准目标检测将 DS0-DS4 五个损坏等级视为相互独立的类别，忽略了它们之间的有序关系（DS3 和 DS4 的误分类比 DS0 和 DS4 的误分类更"可接受"）；(3) 不同检测架构在灾害场景中的优劣缺乏控制变量对比。

**核心矛盾**：损坏评估本质上是一个有序分类（ordinal classification）问题，但现有目标检测框架均采用标准交叉熵损失进行多类别分类，完全忽略了类别间的序数关系。这导致模型可能在 mAP 指标上表现不错，但在实际应用中产生严重的等级误判（如将完全倒塌误判为无损坏）。

**本文目标**：(1) 建立标准化的街景建筑损坏检测 benchmark；(2) 系统对比 CNN 和 Transformer 检测器的架构优劣；(3) 设计并验证序数感知监督策略以提升损坏等级预测的一致性。

**切入角度**：作者利用 2021 年美国中西部龙卷风灾害中的 3,333 张高分辨率地理标记街景图像（8,890 个标注建筑实例），基于 IN-CORE 损坏状态框架进行五级标注，并引入序数指标（Ordinal Top-1 Accuracy、MAOE）来衡量模型的等级一致性。

**核心 idea**：将软序数分类目标（soft ordinal classification targets）和显式序数距离惩罚引入目标检测的分类头，使模型在训练时"知道"预测错一个等级比错两个等级的惩罚更小。

## 方法详解

### 整体框架
TornadoNet 不是提出全新的模型架构，而是一个系统性 benchmark 框架。输入为街景图像，输出为建筑物的检测框和五级损坏等级（DS0-DS4：无损坏、轻微、中度、严重、完全倒塌）。框架包含三个层面：(1) 数据集构建与标注流程；(2) 8 个基线模型的标准化训练与评测；(3) 序数感知监督策略的设计与集成。

### 关键设计

1. **五级损坏分类体系与数据集**:

    - 功能：提供标准化的多级损坏标注和高质量街景数据集
    - 核心思路：基于 IN-CORE 损坏状态框架定义 DS0（无损坏）到 DS4（完全倒塌）五个等级。3,333 张图像来自 2021 年美国中西部龙卷风灾后实地采集，经专家交叉标注验证。数据划分为 6,184 训练 / 1,342 验证 / 1,364 测试实例（75%/15%/15%）。每张图像包含高分辨率街景视角和地理坐标标记。
    - 设计动机：已有灾害数据集要么是航拍视角（如 xBD），要么标注粒度不够（只有二分类"损坏/未损坏"），街景级别的多级标注对于实际灾后快速评估（如车载巡查）至关重要。

2. **软序数分类目标（Soft Ordinal Classification Targets）**:

    - 功能：在分类头的目标标签中编码损坏等级的有序信息
    - 核心思路：传统 one-hot 标签把 DS2 表示为 $[0,0,1,0,0]$，软序数标签则改为以真实等级为中心的高斯分布 $y_k = \frac{1}{Z}\exp(-\frac{(k-c)^2}{2\psi^2})$，其中 $c$ 是真实等级，$\psi$ 控制分布宽度。这样相邻等级获得非零概率，模型学到"DS1 和 DS3 比 DS0 和 DS4 更接近 DS2"的信息。参数 $\psi=0.5$、$K=1$（截断距离）效果最佳。
    - 设计动机：标准交叉熵损失对所有误分类等同惩罚，软序数标签使模型在训练时隐式学习到等级的有序结构，且不需要修改模型架构，只需替换训练标签。

3. **显式序数距离惩罚**:

    - 功能：在损失函数中显式加入序数距离的惩罚项
    - 核心思路：在标准分类损失基础上增加一项 $L_{ord} = \lambda \sum_k |k - c| \cdot p_k$，其中 $p_k$ 是模型对等级 $k$ 的预测概率，$c$ 是真实等级。当模型预测一个远离真实等级的概率较高时，这个损失会施加更大的惩罚。$\lambda=0.05$ 为最优权重。
    - 设计动机：软标签是从标签端编码序数信息，序数惩罚则从损失端显式强制约束，两者可以互补使用，但实验显示软标签策略效果更好。

### 损失函数 / 训练策略
所有模型在统一协议下训练：图像尺寸 896×896，250 epochs，标准数据增强。检测损失为 YOLO/RT-DETR 各自的标准训练损失（box 回归 + 分类），序数监督仅修改分类分支的目标标签或损失函数。每个配置使用 3 个随机种子重复实验，报告均值±方差。

## 实验关键数据

### 主实验：基线模型对比

| 模型 | 架构 | mAP@0.5 | Ordinal Top-1 ↑ | MAOE ↓ | FPS (A100) | 参数量 |
|------|------|---------|-----------------|--------|------------|--------|
| YOLOv8-n | CNN | 40.98% | 84.01% | 0.78 | 276 | 3.0M |
| YOLOv8-l | CNN | 42.09% | 84.19% | 0.78 | 91 | 43.6M |
| YOLO11-x | CNN | **46.05%** | 85.20% | 0.76 | 66 | 56.8M |
| RT-DETR-L | Transformer | 39.87% | **88.13%** | **0.65** | 78 | 32.0M |
| RT-DETR-X | Transformer | 35.75% | 87.74% | 0.67 | 79 | 65.5M |

### 序数监督消融实验

| 模型 | 监督策略 | mAP@0.5 | Δ mAP | Ordinal Top-1 ↑ | MAOE ↓ |
|------|----------|---------|-------|-----------------|--------|
| RT-DETR-L | Baseline（标准交叉熵） | 39.87% | - | 88.13% | 0.65 |
| RT-DETR-L | 软序数 (ψ=0.5, K=1) | **44.70%** | **+4.8pp** | **91.15%** | **0.56** |
| RT-DETR-L | 序数惩罚 (λ=0.05) | 43.36% | +3.5pp | 89.54% | 0.61 |

### 关键发现
- **架构互补性明显**：YOLO 系列在检测精度（mAP）和吞吐量上领先，但 RT-DETR 在序数一致性指标（Ordinal Top-1、MAOE）上显著更优。这说明 Transformer 架构的全局注意力对捕捉损坏等级间的有序关系有天然优势
- **序数监督与架构需要匹配**：软序数标签对 RT-DETR 带来 +4.8pp mAP 的大幅提升，而对 YOLO 的提升较小，表明序数监督在全局注意力架构下效果更佳
- **MAOE 比 mAP 更实用**：在实际灾害应急中，把"严重损坏"误判为"中度损坏"（差 1 级）比误判为"无损坏"（差 3 级）危害小得多，MAOE 能更好衡量这种实用价值
- RT-DETR-X 反而不如 RT-DETR-L，可能是因为数据规模有限导致更大模型过拟合

## 亮点与洞察
- **序数感知监督策略**非常简洁高效：只需修改训练标签（软序数）或添加一项损失（序数惩罚），不改变模型结构，即插即用。这个技巧可推广到所有具有有序类别的检测/分类任务（如医学图像分级、产品质量分级）
- **Benchmark 设计规范**：统一训练协议、多种子重复、同时报告检测指标和序数指标，这种实验设计标准值得学习
- **发现了 CNN vs Transformer 在有序分类任务上的互补性**，这是一个有价值的经验性发现

## 局限与展望
- 数据规模有限（仅 3,333 张图像），可能不足以充分训练大模型
- 仅评测了龙卷风灾害，其他灾害类型（地震、洪水）的泛化性未验证
- 街景视角容易受遮挡影响（树木、其他建筑），论文未充分讨论遮挡问题
- 序数监督仅修改分类头，未探索将有序信息编码到特征提取层的可能

## 相关工作与启发
- **vs xBD 数据集**: xBD 使用航拍图像的灾前-灾后对比进行四级损坏评估，TornadoNet 聚焦街景视角且标注为五级，对实际车载巡查场景更实用
- **vs 标准目标检测 benchmark (COCO, etc.)**: TornadoNet 强调检测精度之外的序数一致性，引入了 Ordinal Top-1 和 MAOE 等新评测维度
- **vs 序数回归文献**: 传统序数回归方法（如 CORAL）主要面向全图分类，TornadoNet 首次将序数概念引入目标检测框架的分类头

## 评分
- 新颖性: ⭐⭐⭐ 技术创新有限（软标签+序数惩罚较简单），但 benchmark 构建有价值
- 实验充分度: ⭐⭐⭐⭐ 8个模型对比、多种子重复、多维度指标，实验设计规范
- 写作质量: ⭐⭐⭐⭐ 结构清晰，数据集描述详细，实验设置透明
- 价值: ⭐⭐⭐⭐ 对灾害AI应用有直接实用价值，序数监督策略可推广

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] YOLOE: Real-Time Seeing Anything](../../ICCV2025/object_detection/yoloe_realtime_seeing_anything.md)
- [\[AAAI 2026\] YOLO-IOD: Towards Real Time Incremental Object Detection](../../AAAI2026/object_detection/yolo-iod_towards_real_time_incremental_object_detection.md)
- [\[CVPR 2025\] Test-Time Backdoor Detection for Object Detection Models](test-time_backdoor_detection_for_object_detection_models.md)
- [\[ICML 2025\] When Every Millisecond Counts: Real-Time Anomaly Detection via the Multimodal Asynchronous Hybrid Network](../../ICML2025/object_detection/when_every_millisecond_counts_real-time_anomaly_detection_via_the_multimodal_asy.md)
- [\[CVPR 2025\] MI-DETR: An Object Detection Model with Multi-time Inquiries Mechanism](mi-detr_an_object_detection_model_with_multi-time_inquiries_mechanism.md)

</div>

<!-- RELATED:END -->
