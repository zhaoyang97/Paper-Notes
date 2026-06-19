---
title: >-
  [论文解读] OmniSat: Self-Supervised Modality Fusion for Earth Observation
description: >-
  [ECCV 2024][时间序列][多模态] 提出OmniSat统一框架，通过模态特异编码器+跨模态对比自监督预训练，将多光谱时序（S2）、SAR时序（S1）、高分辨率单时相（SPOT/Aerial）等异构遥感数据融合为统一表示，在语义分割和作物分类上超越所有单模态和多模态基线。 领域现状：地球观测有丰富多源数据——S2(多…
tags:
  - "ECCV 2024"
  - "时间序列"
  - "多模态"
  - "earth observation"
  - "自监督学习"
  - "Sentinel"
  - "PASTIS"
---

# OmniSat: Self-Supervised Modality Fusion for Earth Observation

**会议**: ECCV 2024  
**arXiv**: [2404.08351](https://arxiv.org/abs/2404.08351)  
**代码**: [GitHub](https://github.com/gastruc/OmniSat)  
**领域**: 自监督  
**关键词**: multi-modal fusion, earth observation, self-supervised, Sentinel, PASTIS

## 一句话总结
提出OmniSat统一框架，通过模态特异编码器+跨模态对比自监督预训练，将多光谱时序（S2）、SAR时序（S1）、高分辨率单时相（SPOT/Aerial）等异构遥感数据融合为统一表示，在语义分割和作物分类上超越所有单模态和多模态基线。

## 研究背景与动机

**领域现状**：地球观测有丰富多源数据——S2(多光谱时序)、S1(SAR时序)、高分辨率单时相(SPOT/航空)，分辨率从10m到0.2m。现有方法大多只利用单一模态。

**现有痛点**：不同模态的时空分辨率、波段数、采集频率完全不同，直接拼接或简单融合无法有效利用互补信息。

**核心矛盾**：高分辨率数据空间细节好但缺时间信息；时序数据时间丰富但空间分辨率低。

**本文目标**：设计一个可以灵活接收任意模态子集的统一多模态遥感架构。

**切入角度**：每种模态用专门编码器，通过跨模态对比学习对齐到共享语义空间。

**核心 idea**：模态特异编码器+跨模态CLIP式对齐+灵活dropout实现任意模态组合的鲁棒融合。

## 方法详解

### 整体框架
每种模态有专门编码器：时序数据用时空注意力Transformer（U-TAE变体），高分辨率用CNN/ViT。各编码器输出经projection head映射到共享空间。推理时用cross-attention融合多模态token。

### 关键设计

1. **模态特异编码器**

    - 功能：分别处理异构数据（时序多光谱、SAR、单时相高分）
    - 核心思路：S2用U-TAE（时空注意力），S1类似但适配2通道SAR，高分用ViT/ResNet
    - 设计动机：输入格式差异太大，不适合共享编码器

2. **跨模态对比对齐**

    - 功能：自监督预训练将同一地块的不同模态映射到相近嵌入
    - 核心思路：CLIP式对比学习——同一地块的S1和S2嵌入拉近，不同地块拉远
    - 设计动机：标注稀缺，利用空间共现关系免费获取对齐信号

3. **模态Dropout + 灵活推理**

    - 功能：训练时随机丢弃部分模态，推理时接受任意子集
    - 核心思路：每step随机mask部分模态输入，强制模型从任意子集提取信息
    - 设计动机：实际部署中不是所有地块都有全部模态覆盖

### 损失函数 / 训练策略
- 预训练：跨模态InfoNCE；微调：交叉熵+Dice
- 数据集：PASTIS（法国S1+S2+SPOT）、FLAIR（航空+S2）

## 实验关键数据

### 主实验（PASTIS语义分割mIoU%）

| 模态组合 | 方法 | mIoU |
|---------|------|------|
| S2 only | U-TAE | 63.1 |
| S1+S2 | 双流 | 65.2 |
| S1+S2+SPOT | **OmniSat** | **67.5** |

### 消融实验

| 配置 | mIoU | 说明 |
|------|------|------|
| 完整 | 67.5 | 三模态 |
| w/o 预训练 | 64.1 | 自监督+3.4 |
| w/o dropout | 65.8 | 鲁棒性+1.7 |
| w/o S1 | 66.2 | SAR全天候 |
| w/o SPOT | 65.9 | 高分细节 |

### 关键发现
- 三模态融合比最好单模态提升4.4 mIoU
- 自监督预训练贡献3.4 mIoU，是最大性能来源
- 模态dropout使缺失任一模态退化控制在1-2 mIoU

## 亮点与洞察
- **首个统一处理光学时序+SAR时序+高分单时相的框架**
- **地理共现自监督**：利用多源数据天然空间对应关系做对齐
- **灵活推理**：一个模型接受任意模态子集，实用性极强

## 局限与展望
- 仅在法国农业区验证，全球泛化性未知
- 对比学习对地理相近但语义不同区域可能产生混淆
- 高分辨率模态空间覆盖有限

## 相关工作与启发
- **vs U-TAE**: 单模态S2基线，OmniSat扩展为多模态
- **vs SatCLIP**: 做地理编码对齐但不处理时序
- **vs SkySense**: 大规模预训练但不处理模态缺失

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个处理三类异构遥感数据的统一框架
- 实验充分度: ⭐⭐⭐⭐ 多数据集+模态消融+预训练消融
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰
- 价值: ⭐⭐⭐⭐⭐ 模态dropout对实际部署极有价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Towards Self-Supervised Foundation Models for Critical Care Time Series](../../NeurIPS2025/time_series/towards_self-supervised_foundation_models_for_critical_care_time_series.md)
- [\[ICML 2025\] TimePoint: Accelerated Time Series Alignment via Self-Supervised Keypoint and Descriptor Learning](../../ICML2025/time_series/timepoint_accelerated_time_series_alignment_via_self-supervised_keypoint_and_des.md)
- [\[NeurIPS 2025\] Universal Spectral Tokenization via Self-Supervised Panchromatic Representation Learning](../../NeurIPS2025/time_series/universal_spectral_tokenization_via_self-supervised_panchromatic_representation_.md)
- [\[ICML 2026\] HEPA: A Self-Supervised Horizon-Conditioned Event Predictive Architecture for Time Series](../../ICML2026/time_series/hepa_a_self-supervised_horizon-conditioned_event_predictive_architecture_for_tim.md)
- [\[ICML 2026\] Self-Supervised Dynamical System Representations for Physiological Time-Series](../../ICML2026/time_series/self-supervised_dynamical_system_representations_for_physiological_time-series.md)

</div>

<!-- RELATED:END -->
