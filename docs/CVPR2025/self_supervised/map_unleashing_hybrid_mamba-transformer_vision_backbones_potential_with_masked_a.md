---
title: >-
  [论文解读] MAP: Unleashing Hybrid Mamba-Transformer Vision Backbone's Potential with Masked Autoregressive Pretraining
description: >-
  [CVPR 2025][自监督学习][Transformer] 提出 Masked Autoregressive Pretraining（MAP），通过局部 MAE 建模 + 行级自回归解码的层次化预训练目标，首次有效预训练混合 Mamba-Transformer 视觉骨干，显著超越 MAE 和 AR 单一策略。
tags:
  - "CVPR 2025"
  - "自监督学习"
  - "Transformer"
  - "hybrid backbone"
  - "masked autoregressive pretraining"
  - "vision backbone"
---

# MAP: Unleashing Hybrid Mamba-Transformer Vision Backbone's Potential with Masked Autoregressive Pretraining

**会议**: CVPR 2025  
**arXiv**: [2410.00871](https://arxiv.org/abs/2410.00871)  
**代码**: [yunzeliu/MAP](https://github.com/yunzeliu/MAP)  
**领域**: 自监督  
**关键词**: Mamba-Transformer, hybrid backbone, masked autoregressive pretraining, self-supervised learning, vision backbone

## 一句话总结

提出 Masked Autoregressive Pretraining（MAP），通过局部 MAE 建模 + 行级自回归解码的层次化预训练目标，首次有效预训练混合 Mamba-Transformer 视觉骨干，显著超越 MAE 和 AR 单一策略。

## 研究背景与动机

**领域现状**: 混合 Mamba-Transformer 网络近来受到广泛关注，能结合 Transformer 的可扩展性和 Mamba 在长序列建模中的效率优势。

**现有痛点**:
1. **MAE 不适合 Mamba**: MAE 预训练对 ViT 有显著提升（+1.4），但对 Vim 几乎无效（+0.2）。
2. **AR 不适合 Transformer**: AR 预训练对 Vim 有效（+1.4），但对 ViT 提升有限（+0.2）。
3. **混合架构需要兼容两种计算范式的预训练策略**: 现有 MAE 或 AR 策略各自只能充分发挥一种模块的潜力。

**核心矛盾**: Transformer 需要双向上下文建模（MAE 擅长），Mamba 需要序列连续性建模（AR 擅长），两者的最优预训练策略截然不同。

**本文切入角度**: 设计分层预训练目标——局部 MAE 让 Transformer 学习局部双向特征，全局自回归让 Mamba 学习跨区域上下文关系。

## 方法详解

### 整体框架

给定图像，先进行随机遮罩，然后以行为单位进行自回归重建：
1. 图像被划分为 $M$ 行，每行 $N$ 个 token
2. 每行内随机遮罩 50% 的 token
3. HybridNet encoder 处理未遮罩 token
4. Transformer Decoder 按行自回归解码：第 $i$ 行的重建依赖前 $i-1$ 行的全部 token + 本行的未遮罩 token

### 关键设计

**1. 混合网络架构 HybridNet (MMMTMMMT)**
- **功能**: 以 3 个 Mamba 层 + 1 个 Transformer 层为一个单元，重复 8 次，共 32 层。
- **核心思路**: 在多种混合排列中（MMMMMMTT、TTMMMMMM、TMMMTMMM、MMMTMMMT）进行从头训练比较，MMMTMMMT 表现最佳（83.12%）。
- **设计动机**: 开头的 Mamba 层负责序列特征提取，穿插的 Transformer 层增强局部特征建模和长程依赖，平衡了局部特征提取和上下文建模增强。

**2. Masked Autoregressive 解码策略**
- **功能**: 对随机遮罩后的图像，用 Transformer Decoder 按行级自回归方式重建，每步同时预测一行内所有被遮罩 token。
- **核心思路**: 损失函数 $\mathcal{L} = -\sum_{i=1}^{M}\sum_{j \in \mathbf{M}_i} \log p(\mathbf{x}_{ij} | \mathbf{x}_{i,j \notin \mathbf{M}_i}, \mathbf{r}_{<i})$。行内预测是 MAE 风格（双向），行间预测是 AR 风格（因果）。
- **设计动机**: 选择行作为子区域是因为大多数 Mamba 实现的默认扫描顺序是行优先（Row-first），AR 顺序必须与 Mamba 扫描顺序一致才能获得最大收益（实验验证：一致时 +2.9，不一致时仅 +0.2）。

**3. 先导实验的关键发现**
- **AR 与扫描顺序的关系**: 使用与 Vim 扫描顺序一致的 AR 预训练可带来 +2.9 提升，不一致仅 +0.2。这是首次通过实验系统验证此结论。
- **遮罩比例**: AR 预训练的最优遮罩比例为 20%（Mamba）；MAE 的最优为 75%（Transformer）；MAP 的平衡点为 50%。
- **重建目标**: MSE 损失重建归一化原始像素效果最好，扩散损失无显著提升。

### 损失函数 / 训练策略

- **预训练**: AdamW 优化器，1600 epochs，仅使用随机裁剪作为数据增强，遮罩比例 50%
- **微调**: 直接微调 400 epochs
- **重建目标**: 归一化原始像素的 MSE 损失

## 实验关键数据

### 主实验（ImageNet-1K 分类）

| 模型 | 预训练 | 参数量 | Top-1 Acc |
|---|---|---|---|
| HybridNet-B | 无 | 128M | 83.1 |
| HybridNet-B | MAE | 128M | 83.9 |
| HybridNet-B | AR | 128M | 83.8 |
| HybridNet-B | CL | 128M | 83.1 |
| **HybridNet-B** | **MAP** | **128M** | **84.9** |
| HybridNet-B (384) | MAP | 128M | 85.5 |
| HybridNet-L (384) | MAP | 443M | 86.2 |
| MambaR-B | AR | 99M | 83.7 |
| **MambaR-B** | **MAP** | **99M** | **84.0** |
| ViT-B | MAE | 86M | 83.6 |
| **ViT-B** | **MAP** | **86M** | **83.6** |
| ViT-L | MAE | 307M | 85.9 |
| **ViT-L** | **MAP** | **307M** | **86.1** |
| MambaVision-B | 无 | 97M | 84.2 |
| **MambaVision-B** | **MAP** | **97M** | **84.9** |
| MambaVision-L | 无 | 241M | 85.3 |
| **MambaVision-L** | **MAP** | **241M** | **86.4** |

### 消融实验

**遮罩策略**:

| 策略 | 准确率 |
|---|---|
| 从头训练 | 83.1 |
| 随机遮罩 | **84.9** |
| 顺序遮罩 | 84.0 |
| 对角遮罩 | 83.8 |

**遮罩比例**:

| 比例 | 准确率 |
|---|---|
| 0% | 83.3 |
| 25% | 84.5 |
| **50%** | **84.9** |
| 75% | 84.2 |

**解码器策略**:

| 策略 | 准确率 |
|---|---|
| AR decoder | 83.7 |
| MAE decoder | 84.1 |
| local MAE | 84.2 |
| **MAP (ours)** | **84.9** |

### 下游任务

| 任务 | 骨干 | 指标 |
|---|---|---|
| ADE20K 语义分割 | HybridNet-S + MAP | mIoU **46.9** (vs 45.6 无预训练) |
| COCO 检测 | HybridNet-Ti + MAP | AP_box **47.3** (vs 45.9 无预训练) |

### 关键发现

1. **MAP 对混合架构提升最大**: HybridNet-B 上 MAP (+1.8) >> MAE (+0.8) ≈ AR (+0.7) >> CL (0)。
2. **MAP 对纯 Mamba 也有效**: MambaR-B 上 MAP (84.0) > AR (83.7) > MAE (83.1)，MAP 的局部 MAE 机制增强了 Mamba 的局部特征建模。
3. **MAP 在大模型上超越 MAE**: ViT-L 上 MAP (86.1) > MAE (85.9)，自回归建模在更大规模模型上的优势显现，与 LLM 中观察到的 scaling law 现象一致。
4. **384 分辨率进一步提升**: HybridNet-B 在 384 分辨率下比 224 提升 0.6，证明 Mamba 的长序列建模能力确实带来增益。

## 亮点与洞察

- **深入的先导研究**: 系统性分析了 MAE/AR/CL 对 Transformer vs Mamba 的不同效果，首次验证 AR 顺序必须与 Mamba 扫描顺序一致
- **优雅的统一框架**: 将 MAE（局部双向）和 AR（全局因果）有机结合为行级解码范式
- **广泛的适用性**: MAP 不仅适用于自定义 HybridNet，还能提升 MambaVision 等现有混合框架
- **50% 最优遮罩比例**: 在 MAE 的 75% 和 AR 的 20% 之间取得平衡，是由混合架构需求自然推导出的

## 局限与展望

- 混合架构在相同设置下仍未超越纯 Transformer + MAE（MAP 的重点在于释放混合架构的潜力）
- 当前行级分区较简单，更精细的聚类策略理论上可获得更好结果
- 未探索视频和点云等其他模态（作者留待未来工作）
- 预训练需要 1600 epochs，计算开销较大

## 相关工作与启发

- **MAE**: 对 Transformer 预训练效果显著，75% 高遮罩比例 + 非对称 encoder-decoder 是关键
- **ARM**: 使用基于聚类的 AR 预训练 cross-scanned Mamba，本质是行优先+列优先的混合
- **VAR**: 提出 next-scale 预测范式，保留了空间局部性
- **MAR**: 将 AR 输出作为扩散模型的条件信号用于生成，启发了本文探索扩散损失

## 评分 ⭐⭐⭐⭐

**创新性**: ⭐⭐⭐⭐ 首次系统研究混合 Mamba-Transformer 预训练，MAP 范式有洞察力  
**实验充分度**: ⭐⭐⭐⭐⭐ 先导研究 + 主实验 + 多个下游任务 + 详尽消融  
**写作质量**: ⭐⭐⭐⭐ 结构清晰，先导研究引出方法设计，逻辑连贯  
**实用价值**: ⭐⭐⭐⭐ 提供了混合架构预训练的通用方法论和最佳实践

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] PDE-Transformer: Efficient and Versatile Transformers for Physics Simulations](../../ICML2025/self_supervised/pde-transformer_efficient_and_versatile_transformers_for_physics_simulations.md)
- [\[CVPR 2025\] SMILE: Infusing Spatial and Motion Semantics in Masked Video Learning](smile_infusing_spatial_and_motion_semantics_in_masked_video_learning.md)
- [\[CVPR 2025\] From Prototypes to General Distributions: An Efficient Curriculum for Masked Image Modeling](from_prototypes_to_general_distributions_an_efficient_curriculum_for_masked_imag.md)
- [\[CVPR 2025\] SATA: Spatial Autocorrelation Token Analysis for Enhancing the Robustness of Vision Transformers](sata_spatial_autocorrelation_token_analysis_for_enhancing_the_robustness_of_visi.md)
- [\[ICML 2025\] Update Your Transformer to the Latest Release: Re-Basin of Task Vectors](../../ICML2025/self_supervised/update_your_transformer_to_the_latest_release_re-basin_of_task_vectors.md)

</div>

<!-- RELATED:END -->
