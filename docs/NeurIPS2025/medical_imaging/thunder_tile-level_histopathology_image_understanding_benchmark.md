---
title: >-
  [论文解读] THUNDER: Tile-level Histopathology image UNDERstanding benchmark
description: >-
  [NeurIPS 2025 Datasets and Benchmarks Track (Spotlight)][医学图像][数字病理学] 提出 THUNDER，一个面向数字病理学基础模型的 tile 级别综合基准，支持 23 个基础模型在 16 个数据集上的高效比较，覆盖下游任务性能、特征空间分析、鲁棒性和不确定性评估。
tags:
  - "NeurIPS 2025 Datasets and Benchmarks Track (Spotlight)"
  - "医学图像"
  - "数字病理学"
  - "benchmark"
  - "基础模型"
  - "鲁棒性"
  - "不确定性"
  - "tile级分析"
---

# THUNDER: Tile-level Histopathology image UNDERstanding benchmark

**会议**: NeurIPS 2025 Datasets and Benchmarks Track (Spotlight)  
**arXiv**: [2507.07860](https://arxiv.org/abs/2507.07860)  
**代码**: 有 ([https://github.com/MICS-Lab/thunder](https://github.com/MICS-Lab/thunder))  
**领域**: 医学影像 / 数字病理学  
**关键词**: 数字病理学, benchmark, 基础模型, 鲁棒性, 不确定性, tile级分析

## 一句话总结

提出 THUNDER，一个面向数字病理学基础模型的 tile 级别综合基准，支持 23 个基础模型在 16 个数据集上的高效比较，覆盖下游任务性能、特征空间分析、鲁棒性和不确定性评估。

## 研究背景与动机

### 数字病理学基础模型的爆发
近年来，大量数字病理学基础模型相继发布（如 UNI、Virchow、CONCH、Phikon、CTransPath 等），它们作为 tile 级图像的特征提取器，服务于各种下游的 tile 级和 slide 级任务。然而，这些模型的发布速度远超社区对其性能和差异的理解。

### 现有基准的不足
- **仅关注下游性能**：忽略了模型间的本质差异（如特征空间结构）
- **缺乏鲁棒性评估**：在医疗等关键领域，仅看准确率是不够的
- **缺乏不确定性分析**：模型需要在不确定时能"说不知道"
- **难以复现和扩展**：很多模型的评估使用不同的协议和数据划分

### THUNDER 的目标
构建一个**快速、易用、动态**的基准，不仅评估性能，还深入分析特征空间、鲁棒性和不确定性，为社区提供全面的模型理解。

## 方法详解

### 整体框架

THUNDER 基准包含四大评估维度：

```
输入: 基础模型 (提取 tile 嵌入)
  ├── 维度1: 下游任务性能 (分类、检索)
  ├── 维度2: 特征空间分析 (结构、可分离性)  
  ├── 维度3: 鲁棒性评估 (分布偏移、对抗扰动)
  └── 维度4: 不确定性估计 (校准、OOD检测)
```

### 关键设计

#### 1. 数据集构成
THUNDER 包含 **16 个数据集**，覆盖多种组织类型和任务：
- **癌症分类**：乳腺癌、肺癌、结直肠癌、胃癌等
- **组织分类**：正常组织类型识别
- **亚型分类**：肿瘤亚型精细分类
- **跨站点评估**：同一任务不同医院来源数据

#### 2. 评估协议
- **线性探测**：冻结基础模型，仅训练线性分类头
- **KNN 分类**：直接在嵌入空间中使用 K 最近邻
- **检索任务**：通过嵌入相似度检索相似 tile
- **统一数据划分**：所有模型使用完全相同的训练/验证/测试集

#### 3. 特征空间分析
- **t-SNE 可视化**：观察不同类别在嵌入空间中的聚类质量
- **类间/类内距离比**：量化特征空间的可分离性
- **特征维度利用率**：分析有多少嵌入维度被有效使用

#### 4. 鲁棒性与不确定性
- **分布偏移**：在不同站点/染色协议的数据上评估性能下降
- **不确定性校准**：ECE (Expected Calibration Error) 评估预测置信度的可靠性
- **OOD 检测**：区分域内和域外样本的能力

### 损失函数 / 训练策略

THUNDER 本身不训练模型，评估使用：
- 线性探测：交叉熵损失 + SGD 优化器
- KNN：无需训练
- 所有基础模型均冻结参数

## 实验关键数据

### 主实验

23 个基础模型在 16 个数据集上的 tile 级线性探测平均表现：

| 模型 | 预训练数据规模 | 架构 | 嵌入维度 | 平均 Acc ↑ | 平均 AUC ↑ |
|------|-------------|------|---------|-----------|-----------|
| UNI (v1) | 100k slides | ViT-L | 1024 | 82.4 | 0.924 |
| Virchow | 1.5M slides | ViT-H | 1280 | 83.1 | 0.931 |
| CONCH | 1.17M slides | ViT-B | 512 | 80.7 | 0.912 |
| Phikon | 40k slides | ViT-B | 768 | 78.3 | 0.896 |
| CTransPath | 15k slides | Swin-T | 768 | 76.8 | 0.882 |
| Lunit-DINO | 33k slides | ViT-S | 384 | 77.5 | 0.889 |
| Prov-GigaPath | 171k slides | ViT-G | 1536 | **84.2** | **0.938** |
| UNI v2 | 350k slides | ViT-L | 1024 | 83.8 | 0.935 |
| ResNet-50 (ImageNet) | 1.2M imgs | ResNet-50 | 2048 | 68.2 | 0.812 |

鲁棒性评估（跨站点性能下降）：

| 模型 | 原站点 Acc | 新站点 Acc | 性能下降幅度 ↓ | 校准误差 ECE ↓ |
|------|----------|----------|-------------|-------------|
| UNI | 82.4 | 76.8 | -5.6 | 0.082 |
| Virchow | 83.1 | 78.2 | -4.9 | 0.071 |
| CONCH | 80.7 | 73.4 | -7.3 | 0.095 |
| Prov-GigaPath | 84.2 | **79.5** | **-4.7** | **0.068** |
| CTransPath | 76.8 | 68.1 | -8.7 | 0.112 |
| ResNet-50 | 68.2 | 58.4 | -9.8 | 0.145 |

### 消融实验

评估协议对比：

| 评估方式 | 准确率范围 | 计算耗时 | 与全面微调相关性 |
|---------|----------|---------|-------------|
| 线性探测 | 68-84% | 快 (分钟级) | r=0.92 |
| KNN (k=5) | 65-82% | 极快 (秒级) | r=0.88 |
| KNN (k=20) | 66-81% | 极快 (秒级) | r=0.86 |
| 少样本 (10-shot) | 55-72% | 快 | r=0.84 |

### 关键发现

1. **预训练数据规模仍然是第一要素**：数据量最大的模型（Prov-GigaPath, Virchow）总体表现最好
2. **大模型不等于鲁棒模型**：某些大模型在跨站点场景下性能下降更大
3. **不确定性估计普遍不足**：多数模型的校准误差偏高，在临床部署中需要额外的校准步骤
4. **特征空间结构差异显著**：不同模型的嵌入空间在聚类质量和维度利用率上差异巨大
5. **ImageNet 预训练远不够**：通用视觉预训练模型在病理任务上显著落后于领域专用模型

## 亮点与洞察

1. **全面性**：首个同时覆盖性能、特征空间、鲁棒性和不确定性的病理基准
2. **规模化**：23 个模型 × 16 个数据集 = 368 个评估组合
3. **实用性**：快速运行、支持用户自定义模型、动态扩展
4. **Spotlight 接收**：被认为对社区有重要参考价值
5. **开源代码**：完全开源，便于社区复现和扩展

## 局限与展望

1. **仅限 tile 级**：未涵盖 slide 级任务（如 MIL 聚合后的 WSI 分类）
2. **评估协议有限**：主要使用线性探测和 KNN，未包含 prompt tuning 等方法
3. **数据集偏向**：主要覆盖 H&E 染色的常见癌症类型
4. **缺乏多模态评估**：未评估 vision-language 模型（如 CONCH）的文本能力
5. **时效性挑战**：基础模型更新迅速，基准需要持续维护

## 相关工作与启发

- **病理基础模型**：UNI, Virchow, CONCH, Phikon 等
- **通用视觉基准**：ImageNet, VTAB 等提供了基准设计的参考
- **启发方向**：开发 slide 级综合基准、加入更多罕见疾病数据、结合临床指标评估

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首个全面的 tile 级病理基准
- **理论深度**: ⭐⭐⭐ — 主要是实验驱动的基准工作
- **实验充分性**: ⭐⭐⭐⭐⭐ — 23模型×16数据集，极其全面
- **实际影响**: ⭐⭐⭐⭐⭐ — 对病理AI社区有重要参考价值
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，便于查阅

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] MedXpertQA: Benchmarking Expert-Level Medical Reasoning and Understanding](../../ICML2025/medical_imaging/medxpertqa_benchmarking_expert-level_medical_reasoning_and_understanding.md)
- [\[ECCV 2024\] OphNet: A Large-Scale Video Benchmark for Ophthalmic Surgical Workflow Understanding](../../ECCV2024/medical_imaging/ophnet_a_large-scale_video_benchmark_for_ophthalmic_surgical_workflow_understand.md)
- [\[NeurIPS 2025\] CXReasonBench: A Benchmark for Evaluating Structured Diagnostic Reasoning in Chest X-rays](cxreasonbench_a_benchmark_for_evaluating_structured_diagnostic_reasoning_in_ches.md)
- [\[CVPR 2025\] Interactive Medical Image Segmentation: A Benchmark Dataset and Baseline](../../CVPR2025/medical_imaging/interactive_medical_image_segmentation_a_benchmark_dataset_and_baseline.md)
- [\[CVPR 2025\] TopoCellGen: Generating Histopathology Cell Topology with a Diffusion Model](../../CVPR2025/medical_imaging/topocellgen_generating_histopathology_cell_topology_with_a_diffusion_model.md)

</div>

<!-- RELATED:END -->
