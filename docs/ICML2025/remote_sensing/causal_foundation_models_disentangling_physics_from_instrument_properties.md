---
title: >-
  [论文解读] Causal Foundation Models: Disentangling Physics from Instrument Properties
description: >-
  [ICML 2025][遥感][因果基础模型] 提出因果驱动的基础模型，通过双编码器架构和结构化对比学习从天文时间序列中解耦物理信号和仪器效应，利用自然存在的观测三元组（同一目标不同仪器/同一仪器不同目标），在低数据场景下显著优于单一潜空间方法。 领域现状 领域现状：天文时间序列基础模型快速发展，但观测数据混合了真实物理信号…
tags:
  - "ICML 2025"
  - "遥感"
  - "因果基础模型"
  - "物理-仪器解耦"
  - "对比学习"
  - "时间序列"
  - "天文观测"
---

# Causal Foundation Models: Disentangling Physics from Instrument Properties

**会议**: ICML 2025  
**arXiv**: [2507.05333](https://arxiv.org/abs/2507.05333)  
**代码**: 无  
**领域**: 遥感  
**关键词**: 因果基础模型, 物理-仪器解耦, 对比学习, 时间序列, 天文观测

## 一句话总结
提出因果驱动的基础模型，通过双编码器架构和结构化对比学习从天文时间序列中解耦物理信号和仪器效应，利用自然存在的观测三元组（同一目标不同仪器/同一仪器不同目标），在低数据场景下显著优于单一潜空间方法。

## 研究背景与动机

### 领域现状

**领域现状**：天文时间序列基础模型快速发展，但观测数据混合了真实物理信号（恒星变化）和仪器系统效应（传感器漂移、校准偏差）。

**现有痛点**：现有基础模型将所有变化来源编码到单一潜空间，导致跨仪器泛化能力差。

**核心矛盾**：物理信号和仪器效应纠缠在一起，但基础模型需要对两者分别进行推理。

**本文目标**：学习分离的物理和仪器潜空间表示。

**切入角度**：利用天文观测的自然结构——同一恒星被不同仪器观测（共享物理，不同仪器），不同恒星被同一仪器观测（不同物理，共享仪器）——形成对比学习的正负样本对。

**核心 idea**：双编码器 + 结构化对比学习，一个编码器捕获物理信息，另一个捕获仪器信息。

## 方法详解

### 整体框架
1. 双编码器：$E_{\text{phys}}$ 提取物理表示，$E_{\text{inst}}$ 提取仪器表示
2. 结构化对比学习：利用三元组关系——同星不同仪器的观测物理表示应接近，同仪器不同星的观测仪器表示应接近
3. 下游任务从任一表示训练

### 关键设计

1. **双编码器架构**:

    - 功能：分别学习物理和仪器的独立潜空间
    - 核心思路：两个 Transformer 编码器，分别将时间序列映射到 $z_{\text{phys}}$ 和 $z_{\text{inst}}$
    - 设计动机：因果结构要求物理和仪器变量独立

2. **结构化对比学习**:

    - 功能：用观测三元组构建正负样本关系
    - 核心思路：对于恒星 $s$，在仪器 $m_1$ 和 $m_2$ 下的观测应有相似的 $z_{\text{phys}}$；对于仪器 $m$，不同恒星的观测应有相似的 $z_{\text{inst}}$
    - 设计动机：不需要显式标注，利用观测的自然配对结构

### 损失函数 / 训练策略
- InfoNCE 对比损失，分别对物理和仪器表示
- 正则化确保两个潜空间的独立性

## 实验关键数据

### 主实验
模拟 TESS 天文时间序列的下游预测：

| 方法 | 恒星参数 R² ↑ | 仪器参数 R² ↑ | 跨仪器泛化 R² ↑ |
|------|-------------|-------------|---------------|
| 单一潜空间基线 | 0.72 | 0.68 | 0.41 |
| 对比学习基线 | 0.78 | 0.74 | 0.52 |
| **因果模型 (物理)** | **0.91** | 0.12 | **0.83** |
| **因果模型 (仪器)** | 0.08 | **0.89** | 0.15 |

### Few-Shot 性能（仅 10% 标注数据）

| 方法 | 10-shot R² | 50-shot R² | 100-shot R² |
|------|-----------|-----------|------------|
| 微调基线 | 0.35 | 0.52 | 0.61 |
| **因果模型** | **0.62** | **0.78** | **0.85** |

### 消融实验

| 配置 | 恒星参数 R² | 解耦度（MI↓） |
|------|-----------|-------------|
| Full Model | 0.91 | 0.03 |
| w/o 对比损失 | 0.79 | 0.21 |
| 单编码器 | 0.72 | 0.45 |

### 关键发现
- 物理编码器几乎不包含仪器信息（R²=0.12），仪器编码器几乎不包含物理信息（R²=0.08）——解耦成功
- 在低数据场景下（few-shot），因果模型优势更明显
- 支持跨仪器迁移

## 亮点与洞察
- **因果结构 + 对比学习**的组合自然且有效
- 利用天文观测的"自然实验"结构是精妙的数据利用方式
- 方法可推广到任何存在类似因果结构的观测数据（气象、医疗传感器等）

## 局限与展望
- 仅在模拟数据上验证，真实天文数据待测试
- 假设物理和仪器效应完全可分，现实中可能有交互
- 需要"同目标多仪器"的配对数据

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 因果基础模型在科学数据上的优雅应用
- 实验充分度: ⭐⭐⭐ 仅模拟数据
- 写作质量: ⭐⭐⭐⭐ 因果动机清晰
- 价值: ⭐⭐⭐⭐ 对科学基础模型有重要方法论启示


## 相关工作与启发
- **vs 同领域代表性方法**：本文在方法设计上有独特贡献，与现有方法形成互补
- **vs 传统方法**：相比传统方案，本文方法在关键指标上取得了显著提升
- **启发**：本文的技术路线对后续相关工作有重要参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] MapEval: A Map-Based Evaluation of Geo-Spatial Reasoning in Foundation Models](mapeval_a_map-based_evaluation_of_geo-spatial_reasoning_in_foundation_models.md)
- [\[NeurIPS 2025\] Cloud4D: Estimating Cloud Properties at a High Spatial and Temporal Resolution](../../NeurIPS2025/remote_sensing/cloud4d_estimating_cloud_properties_at_a_high_spatial_and_temporal_resolution.md)
- [\[NeurIPS 2025\] Mass Conservation on Rails – Rethinking Physics-Informed Learning of Ice Flow Vector Fields](../../NeurIPS2025/remote_sensing/mass_conservation_on_rails_--_rethinking_physics-informed_learning_of_ice_flow_v.md)
- [\[ICCV 2025\] Towards a Unified Copernicus Foundation Model for Earth Vision](../../ICCV2025/remote_sensing/towards_a_unified_copernicus_foundation_model_for_earth_vision.md)
- [\[NeurIPS 2025\] Connecting the Dots: A Machine Learning Ready Dataset for Ionospheric Forecasting Models](../../NeurIPS2025/remote_sensing/connecting_the_dots_a_machine_learning_ready_dataset_for_ionospheric_forecasting.md)

</div>

<!-- RELATED:END -->
