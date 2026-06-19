---
title: >-
  [论文解读] An Overall Real-Time Mechanism for Classification and Quality Evaluation of Rice
description: >-
  [AAAI2026][目标检测][rice quality evaluation] 提出一个实时大米品质评估整体机制，整合改进的 YOLO-v5（品种检测）、改进的 ConvNeXt-Tiny（完整度分级）和 K-means（垩白区域量化）三个模块，在自建的六品种两万张图像数据集上实现了 99.14% mAP 和 97.89% 检测准确率。
tags:
  - "AAAI2026"
  - "目标检测"
  - "rice quality evaluation"
  - "YOLO-v5"
  - "ConvNeXt-Tiny"
  - "K-means"
  - "SimAM"
  - "ECA"
---

# An Overall Real-Time Mechanism for Classification and Quality Evaluation of Rice

**会议**: AAAI2026  
**arXiv**: [2502.13764](https://arxiv.org/abs/2502.13764)  
**代码**: [数据集](https://huggingface.co/datasets/xwk25/RiceCC)  
**领域**: 目标检测  
**关键词**: rice quality evaluation, object detection, YOLO-v5, ConvNeXt-Tiny, K-means, SimAM, ECA

## 一句话总结

提出一个实时大米品质评估整体机制，整合改进的 YOLO-v5（品种检测）、改进的 ConvNeXt-Tiny（完整度分级）和 K-means（垩白区域量化）三个模块，在自建的六品种两万张图像数据集上实现了 99.14% mAP 和 97.89% 检测准确率。

## 背景与动机

大米是全球最重要的粮食作物之一，其品质直接关系到膳食健康和市场价值。传统的大米品质评估主要依赖人工感官检验，即专家通过肉眼观察来判断大米品种、完整度和垩白度。这种方式存在明显缺陷：

- **效率低下**：人工检验速度慢，无法满足大规模快速评估需求
- **主观性强**：结果易受光照条件、人眼疲劳、情绪等因素影响
- **缺乏量化标准**：已有的基于计算机视觉的研究大多只给出定性分级，未能按国家标准（如 GB/T 1354-2018）进行定量评估

此前的相关工作虽然在单一任务上取得了不错的效果（如 EfficientNet-B0 达到 98.37% 分类准确率、RiceNet 实现 94% 五品种分类），但缺乏一个能够同时完成品种识别、完整度分级和垩白度量化评估的综合机制。

## 核心问题

如何构建一个端到端的实时大米品质评估系统，能够同时解决三个关键任务：

1. **品种识别**：在混合样本中准确检测和分类不同大米品种
2. **完整度分级**：对同一品种内的整粒、大碎粒、小碎粒进行分级
3. **垩白度量化**：精确测量大米胚乳中的垩白区域面积比例

且所有评估需符合中国国家标准 GB/T 1354-2018 的规范要求。

## 方法详解

整个框架分为三个核心模块：

### 模块一：改进的 YOLO-v5（品种检测）

在标准 YOLO-v5 的 backbone 中，将第三个 C3 层替换为 SimAM（Simple Attention Module）注意力机制。SimAM 的核心优势在于：

- **无参数增加**：通过优化能量函数来评估单个神经元的重要性，不引入额外参数
- **自诱导方式**：直接推导特征加权的解析解，避免了常规注意力机制的复杂计算
- 具体实现：计算空间维度上的均值和方差，通过能量函数的逆来生成注意力权重，最终与输入特征逐元素相乘

### 模块二：改进的 ConvNeXt-Tiny（完整度评估）

在标准 ConvNeXt-Tiny 的最后一个 ConvNeXt Block 与全局平均池化层之间插入 ECA（Efficient Channel Attention）模块：

- ECA 独立计算每个通道的注意力权重，不考虑像素间的空间关系
- 能高效捕获关键的通道信息，避免代价高昂的成对交互计算
- 使用 ImageNet-1K 预训练权重初始化，降低训练成本
- 完整度分为三级：整粒（whole）、大碎粒（large broken）、小碎粒（tiny broken），数据比例约 10:1（完整:破碎）

### 模块三：K-means 聚类（垩白度评估）

利用 K-means 对大米灰度图像进行像素聚类，分割垩白与非垩白区域：

- 设定 K=1，因为垩白区域通常聚集在米粒中部，呈单一连续不透明区域
- 聚类后提取垩白像素的总数，结合图像分辨率校准参数转换为实际面积
- 使用几何多边形拟合法估算分割面积
- 在5种不同光照强度下采集图像，经农业专家对比，选择亮度等级1作为标准光照

### 数据集

- 手工采集约 20,000 张中国大米图像，涵盖6个主要栽培品种
- 6个品种：广东丝苗米（GD，籼米）、东北糯米（NM，糯米）、五常大米（WC，籼米）、盘锦蟹田米（PJX，粳米）、万年贡米（WN，籼米）、延边大米（YB，粳米）
- 使用工业相机（Sony CMOS, 1920×1080P）在单色背景下拍摄
- 垩白实验额外采集300张图像（每品种10粒 × 5种光照）

## 实验关键数据

### 目标检测（品种识别）

| 模型 | 测试准确率 |
|------|-----------|
| Faster-RCNN（二阶段） | 87.33% |
| Tridentnet（二阶段） | 93.69% |
| YOLO-v5（一阶段） | 95.05% |
| **改进 YOLO-v5 + SimAM** | **97.89%** |

- 验证集 mAP 达到 99.14%（原版 98.76%，提升 0.38%）
- GD、NM、YB 三个品种在验证集上 precision 达到 1.0

### 完整度分类

| 模型 | 平均准确率 |
|------|-----------|
| Decision Tree | 93.72% |
| Random Forest | 94.27% |
| AlexNet | 94.90% |
| ConvNeXt-Tiny（原版） | 95.58% |
| **ConvNeXt-Tiny + ECA** | **97.61%** |

- 相比原版 ConvNeXt-Tiny 平均准确率提升约 2%
- GD 品种提升最显著：88.65% → 98.80%（+10.15%）
- YB 品种最高：99.68%

### 垩白度识别

- 亮度等级1下的自动测量结果与农业专家目视评估高度一致
- 相比 GB/T 1354-2018 规定的人工目测法，自动化方法更精确且显著更快

## 亮点

- **三步骤综合机制**：首次将品种检测、完整度分级、垩白度量化整合为一个完整的实时评估流程，是该领域较少见的全面系统
- **符合国家标准**：以 GB/T 1354-2018 为评估基准，使结果具有实际工业应用价值
- **高质量自建数据集**：20,000张图像覆盖6个中国主要栽培品种，并公开在 HuggingFace 上
- **注意力机制选择合理**：SimAM 无额外参数、ECA 聚焦通道注意力，两种轻量级注意力均适配实时检测场景
- **K-means 设 K=1 的巧妙设计**：利用垩白区域的空间聚集特性简化聚类问题

## 局限与展望

- **品种覆盖有限**：仅涉及6种中国大米品种，未验证对国际品种（如泰国香米、印度 Basmati）的泛化能力
- **检测模型较旧**：基于 YOLO-v5，未与 YOLO-v7/v8/v9 等更新版本对比
- **垩白实验规模小**：仅用300张图像（每品种50张），统计说服力有限
- **K=1 假设过强**：对于垩白分布不均匀或存在多个垩白区域的异常米粒可能失效
- **缺乏端到端集成**：三个模块各自独立，未验证级联使用时的整体效率和误差累积
- **未考虑实际部署**：缺少嵌入式设备或产线上的推理速度和资源消耗分析

## 与相关工作的对比

| 方面 | 本文 | 已有工作 |
|------|------|---------|
| 任务完整性 | 品种+完整度+垩白三合一 | 通常只关注单一任务 |
| 评估标准 | 遵循 GB/T 1354-2018 | 多数缺乏量化国标对照 |
| 检测模型 | 改进 YOLO-v5 (97.89%) | Moses et al. EfficientNet-B0 (98.37%，仅分类) |
| 分类模型 | ConvNeXt-Tiny + ECA (97.61%) | RiceNet (94%)、Lin et al. CNN (95.5%) |
| 数据规模 | 20,000张/6品种 | RiceNet 4,700张/5品种 |

相比单任务方法，本文的优势在于系统性和标准化，但在单项任务的深度（如模型架构创新）上不如专门研究。

## 启发与关联

- **农业质检自动化的范式**：三阶段流水线（检测→分级→量化）的思路可推广到其他农产品质检（如小麦、玉米、水果）
- **轻量级注意力的工程价值**：SimAM 和 ECA 这类无/少参数注意力在资源受限的工业部署中有实用价值
- **国标对接的重要性**：研究如果能对标国家/国际标准，落地价值会显著提升
- **垩白度量化思路**：K-means + 几何拟合的简单组合在低复杂度的图像分割任务中仍然有效，不必所有问题都用深度学习

## 评分

- 新颖性: ⭐⭐⭐（方法改进较为增量，SimAM/ECA 均为已有模块的直接嵌入）
- 实验充分度: ⭐⭐⭐⭐（检测和分类实验较充分，但垩白实验规模偏小）
- 写作质量: ⭐⭐⭐⭐（结构清晰、国标引用规范，但部分实验分析较浅）
- 价值: ⭐⭐⭐⭐（工程应用价值较高，学术贡献有限）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] YOLO-IOD: Towards Real Time Incremental Object Detection](yolo-iod_towards_real_time_incremental_object_detection.md)
- [\[ICCV 2025\] YOLOE: Real-Time Seeing Anything](../../ICCV2025/object_detection/yoloe_realtime_seeing_anything.md)
- [\[CVPR 2025\] MI-DETR: An Object Detection Model with Multi-time Inquiries Mechanism](../../CVPR2025/object_detection/mi-detr_an_object_detection_model_with_multi-time_inquiries_mechanism.md)
- [\[NeurIPS 2025\] ScatterAD: Temporal-Topological Scattering Mechanism for Time Series Anomaly Detection](../../NeurIPS2025/object_detection/scatterad_temporal-topological_scattering_mechanism_for_time_series_anomaly_dete.md)
- [\[CVPR 2026\] YOLO-ULM: Ultra-Lightweight Models for Real-Time Object Detection](../../CVPR2026/object_detection/yolo-ulm_ultra-lightweight_models_for_real-time_object_detection.md)

</div>

<!-- RELATED:END -->
