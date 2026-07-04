---
title: >-
  [论文解读] Resampling Augmentation for Time Series Contrastive Learning: Application to Remote Sensing
description: >-
  [ICML2025][遥感][对比学习] 论文提出一种面向时间序列对比学习的重采样增强（resampling augmentation），通过“上采样 + 不相交子序列抽取 + 对齐回原时间轴”构造正样本对，在多项 SITS 农业分类任务上优于常见增强策略，并在 S2-Agri100 上取得领先结果。
tags:
  - "ICML2025"
  - "遥感"
  - "对比学习"
  - "时间序列增强"
  - "遥感时序"
  - "Sentinel-2"
  - "农作物分类"
---

# Resampling Augmentation for Time Series Contrastive Learning: Application to Remote Sensing

**会议**: ICML2025  
**arXiv**: [2506.18587](https://arxiv.org/abs/2506.18587)  
**代码**: [GitHub - ts_ssl](https://github.com/antoinesaget/ts_ssl)  
**领域**: 遥感  
**关键词**: 对比学习, 时间序列增强, 遥感时序, Sentinel-2, 农作物分类

## 一句话总结
论文提出一种面向时间序列对比学习的重采样增强（resampling augmentation），通过“上采样 + 不相交子序列抽取 + 对齐回原时间轴”构造正样本对，在多项 SITS 农业分类任务上优于常见增强策略，并在 S2-Agri100 上取得领先结果。

## 研究背景与动机

### 遥感场景的核心矛盾
Sentinel-2 每 5 天覆盖全球一次，产生了海量多光谱时序数据。
但高质量标注稀缺且昂贵，导致大量无标注数据未被有效利用。

### 为什么选择自监督对比学习
在“无标注多、标注少”的设置下，对比学习天然适合先学表征再迁移下游任务。
问题在于：
- 图像领域成熟增强（裁剪、旋转、颜色抖动）不一定适合时序
- 时序增强如果破坏关键时间结构，会让正样本对构造失真

### 现有路线的不足
论文对比了掩码重建（masked modeling）和对比学习路线：
- 掩码方法在时空数据上计算开销大
- token 定义对数据集依赖强，迁移性不稳定
- 先前研究也提示“掩码 + 对比”通常优于单独掩码

因此作者聚焦于一个更基础的问题：
如何设计一种简单、稳定、可泛化的时序增强，让对比学习真正吃到 SITS 的时间结构信息。

## 方法详解

### 整体思路
给定输入时序 $S=\{s_1,...,s_T\}\in\mathbb{R}^{T\times C}$，构造两条“同源但不同视图”的时序作为正样本对。

方法由三步组成：
1. 上采样到更密的时间网格
2. 抽取两段互不重叠且覆盖全局时间范围的子序列
3. 插值回原始时间轴长度与对齐方式

### Step 1: Upsampling
先通过线性插值把原序列从 $T$ 扩展到 $T_{up}$（文中常用 $T_{up}=2T$）：

$$
S_{up}=f_{linear}(S)
$$

这样做可以在不丢失整体趋势的前提下，提供更细粒度的采样候选点。

### Step 2: 不相交子序列采样
从 $S_{up}$ 里采样两段子序列，满足两个关键约束：
- 时间点不重叠
- 在完整时间范围的每个季度都至少采到一定数量点（保证 temporal coverage）

这个约束很关键。
它避免了“只在局部时间段抖动”的增强，让正样本对都保留全程时序语义。

### Step 3: 重新对齐到原时间轴
将采样到的两个子序列重新映射并插值回原时间戳集合，
最终得到与原输入同长度、同时间对齐的两条视图序列。

### 与常见增强的差异
- 不像 jittering 只加噪声
- 不像 masking 直接挖空局部观测
- 不像简单 resizing 只做全局拉伸压缩

它本质上是在“保持全局时间覆盖”的前提下制造可控局部差异，更适合构造对比正样本。

### 网络与训练配置（论文设置）
- 编码器：时序版 ResNet（首层 256 filters，输出 512-d embedding）
- 投影头：2 层 MLP（512 hidden -> 128 output）
- 对比框架对比：SimCLR、MoCo、BYOL、VICReg
- 多时间序列样本的组聚合：训练时随机取 $G=4$ 条序列做共享编码后聚合

该设计强调“方法轻量 + 可插拔”，重点在增强策略而非复杂主干。

## 实验关键数据

### 预训练与下游数据规模（论文表格整理）

| 数据集 | 每样本时序数 | 时间步 | 通道数 | 样本规模 | 类别数 |
|--------|-------------|--------|--------|----------|--------|
| FranceCrops | 100 | 60 | 12 | 约 5.8M | 20 |
| FranceCrops CVdL | 100 | 60 | 12 | - | 20 |
| PASTIS | 100 | 60 | 10 | 约 85k | 18 |
| SITS-Former | 25 | 24 | 10 | 约 1.6M | - |
| S2-Agri100 | 25 | 24 | 10 | 约 120k | 15 |

### 主要结论（论文报告）

| 对比项 | 结论 |
|--------|------|
| 与 jittering / resizing / masking 对比 | Resampling 增强表现更好 |
| S2-Agri100 下游分类 | 在不使用空间信息和时间位置编码的条件下达到领先水平 |
| 与掩码重建类复杂 SSL 框架对比 | 简单对比学习 + resampling 依然可超越更复杂方案 |
| 预训练数据分布影响 | 目标域无标注预训练优于跨域预训练，甚至可让简单分类头取得更强表现 |

### 关键发现
1. 增强策略本身就是性能瓶颈，合理时序增强比堆复杂模型更重要。
2. 目标域无标注数据价值很高，很多时候比少量新标注更“划算”。
3. 线性评估与全量微调差距较小，说明学到的表征质量较高。
4. 该方法不依赖空间分支也有效，说明时间结构信息已足够强。

## 亮点与洞察

1. 增强设计非常“数据结构友好”。
	它不是盲目扰动，而是显式约束时间覆盖，减少语义破坏。

2. 方法简洁，复现与迁移成本低。
	很适合作为时序对比学习的强基线。

3. 对遥感实践有现实意义。
	在标注稀缺场景中，可以优先投资“目标域无标注采集 + 对比预训练”。

4. 给 Foundation Model 路线一个反思。
	并非所有场景都需要超大模型，正确的数据增强和训练范式同样关键。

5. 从论文结果看，复杂掩码建模并非唯一正道。
	在一些任务上，轻量对比框架可取得更高性价比。

## 局限与展望

1. 当前实验主要是农业分类场景，任务类型还可扩展到变化检测、灾害监测等。

2. 重采样策略虽简单，但超参数（上采样倍数、子序列长度、覆盖约束）仍需调优。

3. 论文主打像素/时间序列层面，尚未融合更强空间上下文。

4. 对极端不规则采样或高缺测率序列，插值稳定性有待进一步验证。

5. 在超长时序、多卫星多模态融合场景下，还需要更系统的大规模验证。

## 相关工作与启发

- 与 SeCo、SSL4EO-S12、SkySense 等遥感 SSL 工作相比，
  本文更强调“时序增强质量”而不是单纯扩大模型规模。

- 与 SatMAE/Prithvi/Presto 等掩码建模路线互补：
  本文证明了在某些设置里，对比学习仍有很强竞争力。

- 对后续研究的启发：
  1. 可把 resampling 作为基础增强，叠加少量语义约束增强。
  2. 可研究“任务自适应增强策略选择”，按数据统计特性自动选增强。
  3. 可把目标域无标注预训练和主动标注结合，优化标注预算分配。

## 评分
- 新颖性: ⭐⭐⭐⭐☆（4.0/5）
- 实验充分度: ⭐⭐⭐⭐☆（4.5/5）
- 写作质量: ⭐⭐⭐⭐☆（4.0/5）
- 价值: ⭐⭐⭐⭐⭐（5.0/5）

综合评价：这是一篇“方法不复杂但非常实用”的好论文。它抓住了时间序列对比学习最关键却常被低估的环节（augmentation design），并在遥感场景中给出了有说服力的实证收益。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] TAMMs: Change Understanding and Forecasting in Satellite Image Time Series with Temporal-Aware Multimodal Models](../../ICLR2026/remote_sensing/tamms_change_understanding_and_forecasting_in_satellite_image_time_series_with_t.md)
- [\[CVPR 2025\] Learning Occlusion-Robust Vision Transformers for Real-Time UAV Tracking](../../CVPR2025/remote_sensing/learning_occlusion-robust_vision_transformers_for_real-time_uav_tracking.md)
- [\[CVPR 2026\] QuCNet: Quantum Deep Learning Driven Multi-Circuit Network for Remote Sensing Image Classification](../../CVPR2026/remote_sensing/qucnet_quantum_deep_learning_driven_multi-circuit_network_for_remote_sensing_ima.md)
- [\[NeurIPS 2025\] GeoLink: Empowering Remote Sensing Foundation Model with OpenStreetMap Data](../../NeurIPS2025/remote_sensing/geolink_empowering_remote_sensing_foundation_model_with_openstreetmap_data.md)
- [\[ICCV 2025\] SkySense V2: A Unified Foundation Model for Multi-Modal Remote Sensing](../../ICCV2025/remote_sensing/skysense_v2_a_unified_foundation_model_for_multi-modal_remote_sensing.md)

</div>

<!-- RELATED:END -->
