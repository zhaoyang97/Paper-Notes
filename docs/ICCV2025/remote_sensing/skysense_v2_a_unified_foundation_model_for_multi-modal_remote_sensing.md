---
title: >-
  [论文解读] SkySense V2: A Unified Foundation Model for Multi-Modal Remote Sensing
description: >-
  [ICCV 2025][遥感][遥感基础模型] 本文提出SkySense V2，使用单一统一Transformer骨干网络处理高分辨率光学/多光谱/SAR三种遥感模态数据，通过自适应Patch合并、模态特异性Prompt Token和基于Query的语义聚合对比学习（QSACL）进行预训练，仅用665M参数（相比前作SkySense的1.26B）在16个数据集7种任务上平均提升1.8分。
tags:
  - "ICCV 2025"
  - "遥感"
  - "遥感基础模型"
  - "多模态学习"
  - "Transformer"
  - "自监督学习"
  - "Mixture of Experts"
---

# SkySense V2: A Unified Foundation Model for Multi-Modal Remote Sensing

**会议**: ICCV 2025  
**arXiv**: [2507.13812](https://arxiv.org/abs/2507.13812)  
**代码**: 无  
**领域**: 3D视觉 / 遥感  
**关键词**: 遥感基础模型, 多模态学习, 统一Transformer, 自监督学习, Mixture of Experts

## 一句话总结
本文提出SkySense V2，使用单一统一Transformer骨干网络处理高分辨率光学/多光谱/SAR三种遥感模态数据，通过自适应Patch合并、模态特异性Prompt Token和基于Query的语义聚合对比学习（QSACL）进行预训练，仅用665M参数（相比前作SkySense的1.26B）在16个数据集7种任务上平均提升1.8分。

## 研究背景与动机

多模态遥感基础模型（MM-RSFM）在城市规划、环境监测、自然灾害管理等地球观测任务中发挥着关键作用。前作SkySense是迄今最大的MM-RSFM，展示了强大的泛化能力，但存在两个核心问题：

**参数冗余**：SkySense为不同模态使用独立骨干网络——Swin-H处理高分辨率光学图像、ViT-L处理多光谱数据、ViT-L处理SAR数据，总参数量1.26B，参数利用效率低

**SSL方法不适配遥感**：SkySense主要采用DINOv2进行自监督预训练，但遥感图像与自然图像有本质不同——自然图像通常聚焦单一主体（如猫/狗），而遥感图像在不同区域包含多种语义对象（建筑、森林、池塘、土地等）。传统SSL在不同裁切视图间直接做对比学习，对遥感图像可能导致**语义不准确**（两个视图可能捕获完全不同的主体）

核心矛盾：**如何用一个统一骨干处理不同分辨率的多模态遥感数据，同时设计适配遥感数据分布特性的SSL策略？**

本文的解决思路：(1) 统一Transformer骨干 + 自适应分辨率处理；(2) 基于Query的语义聚合对比学习解决遥感图像多语义问题。

## 方法详解

### 整体框架

SkySense V2采用教师-学生架构进行预训练：学生网络通过骨干提取多模态特征，教师网络参数通过EMA更新。输入为地理对齐的HR光学图像（2048×2048）、Sentinel-2多光谱序列（64×64）和Sentinel-1 SAR序列（64×64）。训练使用~2100万组多模态遥感数据。

### 关键设计

1. **统一Transformer骨干**：

    - 功能：用单一四阶段层级编码器处理三种模态的数据
    - 核心思路：前两阶段使用Swin Transformer V2 Block（窗口大小8），利用局部性和平移不变性的视觉先验，同时降低计算复杂度；后两阶段使用标准Transformer Block的全局自注意力，学习全局特征表示。三种模态使用相同的参数，但各自有独立的tokenizer
    - 设计动机：全参数共享显著提升参数效率（665M vs 1.26B），局部+全局注意力组合兼顾效率和表达力

2. **自适应Patch合并（APM）**：

    - 功能：根据不同模态数据的地面采样距离（GSD），在每个阶段后选择性地降低特征分辨率
    - 核心思路：对高分辨率光学图像，APM在每个阶段将2×2相邻patch特征合并（分辨率降4倍，维度翻倍）；对中分辨率的MS/SAR数据，APM保持分辨率不变（仅做线性投影并平均权重）
    - 设计动机：地理对齐的多模态数据具有不同空间分辨率，需要自适应处理以保持特征空间对齐

3. **模态特异性Prompt Token**：

    - 功能：在后两阶段为每种模态引入少量可学习的prompt token（每阶段每模态4个token）
    - 核心思路：将模态prompt token与输入token拼接后送入Transformer block，在每阶段最后一个block中丢弃prompt token的输出：
    $[E_{drop}, E_i^4] = \mathcal{F}_3([P_i^3, E_i^3])$
    - 设计动机：全参数共享可能降低特征多样性，少量模态特异性参数可以捕获各模态的独特特征，同时保持参数效率

4. **基于Query的语义聚合对比学习（QSACL）**：

    - 功能：使用多个可学习query通过cross-attention聚合来自不同视图的相似语义特征，然后对聚合后的特征做对比学习
    - 核心思路：给定全局和局部视图的特征，$m$ 个可学习query分别与各视图特征做cross-attention，生成语义聚合特征 $z_i^g$ 和 $z_i^l$。对比损失在同一query的聚合特征对上计算：
    $\mathcal{L}_{QSACL} = \frac{1}{2m}\sum_{i=1}^m (\mathcal{L}_{CL}(z_i^g, z_i^{l'}) + \mathcal{L}_{CL}(z_i^l, z_i^{g'}))$
    - 设计动机：遥感图像不同裁切视图可能包含不同语义对象，直接对比学习会产生语义错配。通过query聚合相似语义，确保对比学习的准确性

5. **Mixture of Experts (MoE) 扩展**：

    - 功能：在最后L=6个Transformer block中，将FFN替换为MoE模块（M=8个expert，top-k=1）
    - 核心思路：$MOE(x) = \sum_{i \in \mathcal{T}} \mathcal{G}_i(x) \cdot \mathcal{E}_i(x)$，其中 $\mathcal{G}$ 是线性门控 + softmax
    - 设计动机：统一骨干设计节省的参数预算可以用于MoE扩展，利用稀疏前向层实现更大模型容量而不成比例增加计算量

### 损失函数 / 训练策略

总训练损失为三部分加权和：
$$\mathcal{L} = \lambda_1 \mathcal{L}_{MGCL} + \lambda_2 \mathcal{L}_{ITA} + \lambda_3 \mathcal{L}_{QSACL}$$
- $\mathcal{L}_{MGCL}$: 多粒度对比学习（像素、对象、图像级别）
- $\mathcal{L}_{ITA}$: 基于OpenStreetMap标签的密集图文对齐
- $\mathcal{L}_{QSACL}$: 基于Query的语义聚合对比学习

训练设置：batch size 1024，128张H20 GPU，600K迭代，AdamW优化器，初始学习率 $2 \times 10^{-4}$，cosine衰减至 $1 \times 10^{-6}$，token维度C=352。

## 实验关键数据

### 主实验（场景分类）

| 模型 | AID (20%/50%) OA | RESISC-45 (10%/20%) OA | BEN-S2 (10%/100%) mAP | fMoW-S2 Top-1/5 |
|------|------------------|----------------------|---------------------|----------------|
| SatMAE | 95.02/96.94 | 91.72/94.10 | 86.18/89.50 | 63.84/- |
| Scale-MAE | 96.44/97.58 | 92.63/95.04 | - | - |
| SkySense | 97.68/98.60 | 94.85/96.32 | 88.67/92.09 | 64.38/87.27 |
| **SkySense V2** | **98.34/99.05** | **96.42/97.24** | **89.13/93.78** | **66.65/89.32** |

### 消融实验（语义分割）

| 模型 | Dyna.-Pla. (5%/10%) mIoU | iSAID mIoU | Potsdam mF1 |
|------|--------------------------|-----------|-------------|
| SkySense | 39.7/46.5 | 70.91 | 93.99 |
| **SkySense V2** | **41.2/47.6** | **71.87** | **95.86** |

### 关键发现

- SkySense V2在所有16个数据集7种任务上均取得SOTA或接近SOTA性能
- 参数量从1.26B降至665M（减少47%），但性能平均提升1.8分
- QSACL使不同query能够聚合一致的语义特征（如建筑、植被等），有效解决遥感图像多语义对比学习的问题
- 在低训练比例（low TR）设置下表现尤为突出，展示了更强的特征表示能力
- MoE的引入以较小的额外计算代价进一步提升了性能

## 亮点与洞察

1. **统一骨干的效率优势**：用一个665M参数的骨干替代三个独立骨干（1.26B），不仅没有牺牲性能反而提升，证明了多模态参数共享的可行性
2. **APM的优雅设计**：通过简单的条件分支（合并或保持）解决了多模态不同分辨率的难题，无需复杂的对齐模块
3. **QSACL准确把握遥感数据特性**：遥感图像的多语义分布是与自然图像最本质的区别之一，通过query聚合的方式正面解决了这一问题
4. **工程规模令人印象深刻**：2100万组训练数据 × 128张H20 GPU × 600K迭代，体现了工业级别的预训练能力

## 局限与展望

- 仅支持光学RGB/多光谱/SAR三种模态，未涵盖高光谱、LiDAR等其他重要遥感数据源
- 预训练数据规模巨大（2100万组），限制了学术界的复现能力
- MoE仅在最后6个block中使用，更早阶段的引入是否有益未探索
- 统一骨干对轻量化部署可能不利（665M参数仍然很大）
- QSACL的query数量（$m$）的选择对性能影响的分析不够充分

## 相关工作与启发

- SkySense [Guo et al., CVPR 2024] 是直接前作，本文解决了其参数冗余和SSL不适配的问题
- Meta-Transformer [Zhang et al.] 探索了单一Transformer处理多种模态的思路，本文将其落地到遥感领域并做了关键适配
- AnySat [ECCV 2024] 也使用统一架构处理多模态遥感数据，但其骨干设计和SSL策略与SkySense V2有本质不同
- 启示：在遥感基础模型中，**适配数据特性的SSL策略**（如QSACL）比简单借用自然图像的SSL方法更重要

## 评分
- 新颖性: ⭐⭐⭐⭐ APM和QSACL是针对遥感特性的恰当创新，但统一骨干本身不算新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 16个数据集7种任务的大规模评测，覆盖面极广
- 写作质量: ⭐⭐⭐⭐ 结构清晰，但部分技术细节放在附录中影响完整性
- 价值: ⭐⭐⭐⭐⭐ 作为遥感领域的大统一基础模型，在实际地球观测任务中有直接应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] RS-vHeat: Heat Conduction Guided Efficient Remote Sensing Foundation Model](rs-vheat_heat_conduction_guided_efficient_remote_sensing_foundation_model.md)
- [\[ICCV 2025\] Towards a Unified Copernicus Foundation Model for Earth Vision](towards_a_unified_copernicus_foundation_model_for_earth_vision.md)
- [\[ICCV 2025\] SMARTIES: Spectrum-Aware Multi-Sensor Auto-Encoder for Remote Sensing Images](smarties_spectrum-aware_multi-sensor_auto-encoder_for_remote_sensing_images.md)
- [\[CVPR 2026\] SkySense-VITA: Towards Universal In-context Segmentation of Multi-modal Remote Sensing Imagery](../../CVPR2026/remote_sensing/skysense-vita_towards_universal_in-context_segmentation_of_multi-modal_remote_se.md)
- [\[NeurIPS 2025\] GeoLink: Empowering Remote Sensing Foundation Model with OpenStreetMap Data](../../NeurIPS2025/remote_sensing/geolink_empowering_remote_sensing_foundation_model_with_openstreetmap_data.md)

</div>

<!-- RELATED:END -->
