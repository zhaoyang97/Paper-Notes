---
title: >-
  [论文解读] Leveraging Depth and Language for Open-Vocabulary Domain-Generalized Semantic Segmentation
description: >-
  [NeurIPS 2025][自动驾驶][开放词汇分割] 提出Vireo框架，首次将开放词汇语义分割（OVSS）和域泛化语义分割（DGSS）统一到单阶段框架中，通过GeoText Query融合深度几何特征与语言线索，在极端环境和未见类别上均实现SOTA表现。 领域现状：开放词汇语义分割（OVSS）能识别任意文本描述的类别…
tags:
  - "NeurIPS 2025"
  - "自动驾驶"
  - "开放词汇分割"
  - "域泛化"
  - "深度估计"
  - "视觉基础模型"
  - "语义分割"
---

# Leveraging Depth and Language for Open-Vocabulary Domain-Generalized Semantic Segmentation

**会议**: NeurIPS 2025  
**arXiv**: [2506.09881](https://arxiv.org/abs/2506.09881)  
**代码**: [GitHub](https://github.com/SY-Ch/Vireo)  
**领域**: Autonomous Driving / Semantic Segmentation  
**关键词**: 开放词汇分割, 域泛化, 深度估计, 视觉基础模型, 语义分割

## 一句话总结

提出Vireo框架，首次将开放词汇语义分割（OVSS）和域泛化语义分割（DGSS）统一到单阶段框架中，通过GeoText Query融合深度几何特征与语言线索，在极端环境和未见类别上均实现SOTA表现。

## 研究背景与动机

**领域现状**：开放词汇语义分割（OVSS）能识别任意文本描述的类别，域泛化语义分割（DGSS）能在未见域上保持鲁棒性，两者各有所长但互补。

**现有痛点**：OVSS的文本-视觉对齐模块在域外（如夜晚、雨天）性能大幅下降；DGSS的域不变策略可能抑制细粒度语义线索，影响对文本查询的精确响应。

**核心矛盾**：如何同时实现跨域鲁棒性和开放词汇识别能力？DGSS侧重编码器端特征泛化，OVSS侧重解码器端开放识别——两者天然互补。

**本文目标**：构建统一的OV-DGSS（开放词汇域泛化语义分割）框架，在域偏移下对未见类别也能鲁棒分割。

**切入角度**：利用深度图的域不变性（深度和几何线索对光照、纹理变化不敏感），结合冻结VFM的泛化能力。

**核心 idea**：用GeoText Query在冻结VFM层间注入深度几何和文本语义先验，辅以CMPE增强梯度流和DOV-VEH融合多模态特征。

## 方法详解

### 整体框架

Vireo包含三个核心模块：
- **Tunable Vireo + GeoText Query**：在冻结VFM编码器层间注入并对齐几何和文本信息
- **Coarse Mask Prior Embedding (CMPE)**：生成粗糙先验掩码，增强梯度回传
- **Domain-Open-Vocabulary Vector Embedding Head (DOV-VEH)**：融合视觉、几何、语义特征生成最终预测

输入图像同时送入冻结的VFM编码器和冻结的DepthAnything V2编码器（深度估计），文本类别标签通过冻结CLIP文本编码器获得语义嵌入。

### 关键设计

1. **GeoText Query**：

    - **功能**：在冻结VFM的各层间注入结构-语义先验，逐层精炼特征
    - **为什么**：深度特征提供域不变的空间约束，缓解RGB特征的域偏移；文本嵌入提供开放词汇的语义对齐
    - **怎么做**：每层通过交叉注意力机制让可学习query $P_l$ 与视觉特征 $f_l^V$、深度特征 $f_l^D$、文本嵌入 $t_k$ 交互：
    $\mathcal{A}_l = \text{CrossAttn}(P_l, f_l^V, f_l^D, \{t_k\})$
      注意力输出通过加权求和、MLP投影和残差连接精炼视觉表示 $\hat{f}_l^V$
    - **区别**：不同于REIN仅做prompt tuning，GeoText Query同时融合了深度和文本两种跨模态先验

2. **Coarse Mask Prior Embedding (CMPE)**：

    - **功能**：生成粗糙语义概率图，反向增强梯度流通过冻结编码器
    - **为什么**：冻结编码器导致梯度稀疏、收敛缓慢，CMPE注入更密集的梯度信号
    - **怎么做**：选取VFM第8/12/16/24层的精炼特征，上采样到统一分辨率，经自适应注意力门控（AAG）融合后，与文本嵌入通过爱因斯坦求和计算粗糙掩码 $\mathcal{M}(x,y,k) = \langle f^M(x,y), t_k \rangle$。进一步通过softmax归一化生成query先验：
    $q_j^{prior} = \sum_k \text{Softmax}(\langle q_j, e_k^{class} \rangle) \cdot e_k^{class}$

3. **DOV-VEH (Domain-Open-Vocabulary Vector Embedding Head)**：

    - **功能**：融合多尺度精炼特征，生成像素级分割掩码
    - **怎么做**：多尺度特征经Pixel Decoder增强空间表示，再经Transformer Decoder让GeoText Query与解码特征和文本嵌入交互，生成掩码嵌入 $\mathcal{E}_{mask}$ 和分类嵌入 $\mathcal{E}_{cls}$，最终预测：
    $\hat{\mathcal{M}}(x,y,k) = \sum_d \mathcal{E}_{mask}(x,y,d) \cdot \mathcal{E}_{cls}(k,d)$

### 损失函数 / 训练策略

- 使用AdamW优化器，初始学习率1e-4，权重衰减0.05
- 多项式学习率衰减，总迭代40K步
- 数据增强：多尺度resize、随机裁剪、随机水平翻转、光度扰动
- 单卡RTX A6000训练，batch size 8，约14小时

## 实验关键数据

### 主实验

**域泛化能力（Cityscapes → ACDC + BDD + Mapillary, mIoU %）**：

| 方法 | Night-ACDC | Fog-ACDC | Rain-ACDC | Snow-ACDC | BDD100k | Mapillary |
|------|-----------|---------|----------|----------|---------|----------|
| FC-CLIP (OVSS) | 40.8 | 64.4 | 63.2 | 61.5 | 55.9 | 66.1 |
| REIN (DGSS) | 55.9 | 79.5 | 72.5 | 70.6 | 63.5 | 74.0 |
| FADA (DGSS) | 57.4 | 80.2 | 75.0 | 73.5 | 65.1 | 75.9 |
| **Vireo** | **60.6** | **82.3** | **76.3** | **76.2** | **66.7** | **76.0** |

**开放词汇能力（Cityscapes → DELIVER + ADE, mIoU %）**：

| 方法 | Sun | Night | Cloud | Rain | Fog | ADE150 | ADE847 |
|------|-----|-------|-------|------|-----|--------|--------|
| CAT-Seg | 28.2 | 20.6 | 26.2 | 26.5 | 24.8 | 20.2 | 7.0 |
| **Vireo** | **35.7** | **27.5** | **32.3** | **31.8** | **32.7** | **21.4** | **7.3** |

### 消融实验

**组件消融（Cityscapes → ACDC + BDD + Map, mIoU %）**：

| 配置 | Snow | Night | Fog | Rain | BDD | Map |
|-----|------|-------|-----|------|-----|-----|
| REIN (baseline) | 70.6 | 55.9 | 79.5 | 72.5 | 63.5 | 74.0 |
| + DepthAnything V2 | 71.5 | 56.7 | 80.5 | 73.3 | 64.4 | 74.5 |
| + GeoText Query | 74.0 | 58.4 | 81.1 | 74.8 | 65.3 | 75.3 |
| **Vireo（完整）** | **76.2** | **60.6** | **82.3** | **76.3** | **66.7** | **76.0** |

**多backbone泛化（GTA5 → Citys+BDD+Map, mIoU）**：

| Backbone | REIN | FADA | Vireo | 可训练参数 |
|----------|------|------|-------|----------|
| EVA02-L | 63.6 | 64.9 | **66.0** | 3.78M |
| DINOv2-L | 64.3 | 66.1 | **67.7** | 3.78M |

### 关键发现

- GeoText Query是最关键组件，在夜间场景提升约2.5% mIoU
- 深度几何特征在极端天气（特别是夜间和雪天）帮助最大
- Vireo在Seen和Unseen类别上均优于现有OVSS方法，Unseen类别优势更明显（+7%以上）
- 仅需3.78M可训练参数，远少于FADA（11.65M），但性能更优

## 亮点与洞察

- **问题定义有意义**：首次提出OV-DGSS问题，将开放词汇和域泛化统一，更贴近自动驾驶实际需求
- **深度线索的巧妙利用**：深度图天然具有域不变性，用冻结的DepthAnything提取几何特征是轻量且有效的策略
- **DGSS和OVSS的互补性洞察**：DGSS优化编码器端泛化，OVSS优化解码器端识别——Vireo在两端同时发力
- **参数效率高**：3.78M可训练参数即实现SOTA，适合实际部署

## 局限与展望

- 深度估计依赖DepthAnything V2的质量，在极端场景（如强雾、夜间）深度估计可能不准确
- CMPE生成的粗糙掩码质量有限，可能引入噪声先验
- ADE847上的绝对性能仍较低（7.3% mIoU），847类超细粒度分割仍是挑战
- 训练内存需求较高（~45GB GPU显存），部署效率有待优化
- 仅验证了自动驾驶相关场景，其他OV-DGSS应用（如医疗影像）未探索

## 相关工作与启发

- **REIN / FADA**：VFM-based DGSS方法，通过在冻结VFM中插入可学习模块提升域泛化能力
- **FC-CLIP / CAT-Seg**：OVSS方法，利用CLIP对齐视觉和文本实现开放词汇识别
- **DepthForge**：证明深度query注入冻结VFM能提升域泛化，启发了本文使用DepthAnything
- **启发**：深度/几何作为域不变锚点的思路可推广到视频分割、3D场景理解等任务

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次定义OV-DGSS问题并提出统一框架，GeoText Query设计新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 8个数据集、5种评测设置、多backbone验证、详细消融
- 写作质量: ⭐⭐⭐⭐ 框架图清晰，实验组织有条理，方法描述完整
- 价值: ⭐⭐⭐⭐⭐ 统一OV+DG对自动驾驶场景非常实用，参数效率高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Open-Vocabulary Domain Generalization in Urban-Scene Segmentation](../../CVPR2026/autonomous_driving/open-vocabulary_domain_generalization_in_urban-scene_segmentation.md)
- [\[NeurIPS 2025\] SPIRAL: Semantic-Aware Progressive LiDAR Scene Generation and Understanding](spiral_semantic-aware_progressive_lidar_scene_generation_and_understanding.md)
- [\[CVPR 2025\] O3N: Omnidirectional Open-Vocabulary Occupancy Prediction](../../CVPR2025/autonomous_driving/o3n_omnidirectional_open-vocabulary_occupancy_prediction.md)
- [\[CVPR 2025\] A Dataset for Semantic Segmentation in the Presence of Unknowns](../../CVPR2025/autonomous_driving/a_dataset_for_semantic_segmentation_in_the_presence_of_unknowns.md)
- [\[NeurIPS 2025\] SDTagNet: Leveraging Text-Annotated Navigation Maps for Online HD Map Construction](sdtagnet_leveraging_text-annotated_navigation_maps_for_online_hd_map_constructio.md)

</div>

<!-- RELATED:END -->
