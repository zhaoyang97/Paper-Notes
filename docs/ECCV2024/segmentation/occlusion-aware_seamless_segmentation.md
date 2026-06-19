---
title: >-
  [论文解读] Occlusion-Aware Seamless Segmentation
description: >-
  [ECCV 2024][语义分割][全景分割] 提出 Occlusion-Aware Seamless Segmentation (OASS) 新任务与 UnmaskFormer 框架，同时解决全景图像窄视场解锁、遮挡物体完整分割和针孔-全景跨域适应三大挑战，在自建 BlendPASS 数据集上达到 SOTA。
tags:
  - "ECCV 2024"
  - "语义分割"
  - "全景分割"
  - "遮挡感知"
  - "无缝分割"
  - "无监督域适应"
  - "图像分割"
---

# Occlusion-Aware Seamless Segmentation

**会议**: ECCV 2024  
**arXiv**: [2407.02182](https://arxiv.org/abs/2407.02182)  
**代码**: [https://github.com/yihong-97/OASS](https://github.com/yihong-97/OASS)  
**领域**: 图像分割  
**关键词**: 全景分割, 遮挡感知, 无缝分割, 无监督域适应, Amodal Segmentation

## 一句话总结

提出 Occlusion-Aware Seamless Segmentation (OASS) 新任务与 UnmaskFormer 框架，同时解决全景图像窄视场解锁、遮挡物体完整分割和针孔-全景跨域适应三大挑战，在自建 BlendPASS 数据集上达到 SOTA。

## 研究背景与动机

**领域现状**: 全景图像理解（panoramic scene understanding）和遮挡感知的 Amodal 分割各自均有进展，但两个方向长期独立发展。全景分割方法（Trans4PASS、DATR）能处理 360° 图像畸变但无法推理被遮挡物体；Amodal 方法（ORCNN、SLN）能预测完整遮挡轮廓但无法泛化到全景图像。

**现有痛点**:
   - 全景图像存在严重的畸变（distortion），直接使用传统分割模型性能大幅下降；
   - 现有语义/实例分割只能预测可见区域，无法推理被遮挡部分的完整形状；
   - 全景图像标注极其昂贵（每张约 210 分钟），导致标注稀缺，需要无监督域适应（UDA）从标签丰富的针孔域迁移。

**核心矛盾**: 视场遮挡（FoV occlusion）、场景内物体遮挡（in-field occlusion）和跨域差距（domain gap）三者交织，现有方法只能分别解决其中一个，无法实现"无缝"（seamless）的全面理解。

**本文目标** 统一解决三大"遮罩"问题：(1) 解锁窄视场 $\rightarrow$ 全景 360°；(2) 解锁物体遮挡 $\rightarrow$ amodal 完整分割；(3) 解锁域差距 $\rightarrow$ 从针孔到全景的 UDA。

**切入角度**: 定义全新任务 OASS，构建专用数据集 BlendPASS，设计统一框架 UnmaskFormer 一次性解决畸变处理、遮挡推理和域适应。

**核心 idea**: 通过 Unmasking Attention 处理畸变与遮挡、Amodal-oriented Mix 增强跨域适应和遮挡重建能力，在一个 transformer 框架中无缝完成五类分割任务。

## 方法详解

### 整体框架

UnmaskFormer 由三大部分组成：
- **UA-based Backbone**: 基于 Transformer 的四阶段特征提取器，包含 Deformable Patch Embedding (DPE) 和 Unmasking Attention (UA)，同时处理全景畸变和物体遮挡；
- **三分支解码器**: 语义分支（per-pixel 语义分类）、实例分支（基于 Mask R-CNN 的 top-down 实例分割）、Amodal 实例分支（预测完整遮挡区域）；
- **OAFusion 模块**: 融合三分支输出，一次生成语义分割、实例分割、Amodal 实例分割、全景分割和 Amodal 全景分割五种结果。

### 关键设计

1. **Unmasking Attention (UA)**:

    - **功能**: 在 self-attention 之后引入增强池化层，生成遮挡感知特征
    - **核心思路**: 特征 $\boldsymbol{X'}$ 经过 self-attention 后，用全局平均池化 $\boldsymbol{q} = GAP(\boldsymbol{X'})$ 得到池化查询，再通过交叉注意力得到 $\boldsymbol{q'} \in \mathbb{R}^{1 \times 1 \times C}$，应用 sigmoid 函数 $\phi(\boldsymbol{q'})$ 生成遮挡感知掩码，与原特征做逐元素乘法得到遮挡感知特征 $\boldsymbol{X''} = \phi(\boldsymbol{q'}) \odot \boldsymbol{X'}$
    - **设计动机**: 全局池化捕获全图上下文，sigmoid 掩码有选择地增强/抑制特征通道，使网络学会关注被遮挡区域的信息

2. **交错 DPE 排列 (Interleaved DPE)**:

    - **功能**: 将 Deformable Patch Embedding 从仅用于初始阶段改为交错放置在 Stage 2 和 Stage 4
    - **核心思路**: DPE 通过可学习的自适应偏移 $\boldsymbol{\Delta}^{DPE}(i,j)$ 捕获局部几何变化，在深层阶段使用 DPE 比浅层效果更好
    - **设计动机**: 全景图像在不同层级都存在畸变，仅在初始阶段处理不够。实验证明在深层阶段 (Stage 2, 4) 使用 DPE 比浅层 (Stage 1, 3) 效果更佳

3. **Amodal-oriented Mix (AoMix)**:

    - **功能**: 跨域数据增强策略，利用 amodal 标注生成遮挡训练样本并融合源域-目标域图像
    - **核心思路**:
        - 随机采样 amodal 实例掩码 $\{M_r^{(i)}\}_{i=1}^z$，经随机缩放 $RS(\cdot)$ 和随机填充 $RP(\cdot)$ 生成新掩码 $M_r = H(\sum_i RP(RS(M_r^{(i)})))$
        - 用 $M_r$ 遮盖源图像 Thing 区域：$\hat{x}_s = (1 - M_r \cap M_s) \odot x_s$
        - 随机采样一半语义类别从遮盖源图 $\hat{x}_s$ 粘贴到目标图 $x_t$，生成混合图 $\hat{x}_m$
    - **设计动机**: 用真实物体形状（而非随机 patch）做遮挡，更贴近实际场景；同时混合源域-目标域弥补域差距

4. **OAFusion (Occlusion-Aware Fusion)**:

    - **功能**: 融合三分支输出产生五类分割结果
    - **核心思路**: 语义直接输出；实例/amodal 实例的类别由语义分支多数投票决定；关键改进是在 amodal 场景下只考虑不与其他物体重叠的区域进行投票
    - **设计动机**: 传统融合在大面积遮挡时会误分类（例如行人被车大面积遮挡→误判为车），OAFusion 忽略重叠区域避免此问题

### 损失函数 / 训练策略

- **源域监督损失** $\mathcal{L}_S$：语义分支用交叉熵损失，实例/amodal 分支用 Mask R-CNN 标准损失（bbox + mask）
- **目标域自训练损失** $\mathcal{L}_T = -\omega \sum_{h,w,c} p_t^{(h,w,c)} \log \hat{y}_t^{(h,w,c)}$，其中伪标签由 Mean-Teacher 的 EMA 教师模型生成，权重 $\omega$ 基于置信度阈值 $\tau = 0.968$ 估计
- **总损失** $\mathcal{L}_{total} = \mathcal{L}_S + \mathcal{L}_T$
- 训练配置：AdamW 优化器，lr = $6 \times 10^{-5}$，weight decay 0.01，batch size 4，crop size 376×376，训练 40k 迭代

## 实验关键数据

### 主实验（KITTI360-APS → BlendPASS）

| 方法 | mIoU (SS) | mAPQ (APS) | mAP (Instance) | mAAP (Amodal) |
|------|-----------|------------|-----------------|---------------|
| DATR | 34.91% | 20.26% | 8.66% | 8.68% |
| Trans4PASS | 40.66% | 22.94% | 10.01% | 9.85% |
| UniDAPS | 38.46% | — | 3.43% | n.a. |
| EDAPS | 40.17% | 23.14% | 10.28% | 10.68% |
| Source-Only | 38.65% | 22.13% | 10.54% | 10.22% |
| **UnmaskFormer** | **43.66%** | **26.58%** | **11.10%** | 10.50% |

### 全景语义分割（其他数据集）

| 数据集 | 指标 | UnmaskFormer | 之前SOTA | 提升 |
|--------|------|-------------|----------|------|
| SynPASS | mIoU | **45.34%** | 44.80% (Trans4PASS) | +0.54% |
| DensePASS | mIoU | **48.08%** | 45.89% (Trans4PASS) | +2.19% |

### 消融实验

| 组件配置 | mIoU | mAPQ | 说明 |
|----------|------|------|------|
| PE (baseline) | 41.07% | 22.00% | 原始 Patch Embedding |
| DPE (早期阶段) | 40.70% | 22.90% | 仅初始阶段 DPE |
| AvgPool | 42.10% | 23.90% | 普通平均池化 |
| SimPool | 43.04% | 24.74% | SimPool 替代 |
| **UA (ours)** | **43.06%** | **25.04%** | Unmasking Attention |
| UA + AoMix | 42.39% | 25.17% | 加入 AoMix |
| UA + AoMix + OAFusion | **43.66%** | **26.58%** | 完整模型 |

### AoMix 策略消融

| 策略 | mIoU | mAPQ | 说明 |
|------|------|------|------|
| T for S (仅源图遮盖) | 42.53% | 23.98% | 只在源图使用遮盖 |
| T for M (仅混合图遮盖) | 43.18% | 24.78% | 只在混合图使用遮盖 |
| P for S&M (patch 遮盖) | 42.52% | 21.87% | 用随机 patch 遮盖 |
| W for S&M (全图遮盖) | 41.64% | 24.12% | 遮盖所有区域 |
| **AoMix (ours)** | **42.39%** | **25.17%** | 仅遮盖 Thing 类区域 |

### 关键发现

- UA 相比基础 PE 在 mAPQ 上提升 +3.04%，证明遮挡感知池化注意力的有效性
- AoMix 使用真实物体形状遮盖比随机 patch 遮盖效果显著更好（mAPQ +3.3%），验证了 amodal 导向增强的重要性
- OAFusion 解决了传统融合在大面积遮挡场景下的误分类问题
- DPE 在深层阶段（Stage 2, 4）比浅层阶段（Stage 1, 3）效果更好
- UnmaskFormer 参数量仅 13.96M，与 Trans4PASS (13.93M) 相当但性能明显更优

## 亮点与洞察

- **任务定义的创新性**: OASS 首次将全景分割、Amodal 分割和 UDA 统一为一个任务，填补了领域空白
- **BlendPASS 数据集**: 2000 张全景图训练 + 100 张精标注测试，包含 2960 个 Thing 实例，43% 存在遮挡，标注质量经三人交叉验证保证
- **UA 设计简洁有效**: 只在 self-attention 后加一层池化注意力，就能显著提升遮挡感知能力
- **OAFusion 的实用价值**: 解决了 amodal 场景下语义投票的根本缺陷——被大面积遮挡的物体不再被误分为遮挡物类别
- 一个模型同时输出五种分割结果（语义/实例/amodal 实例/全景/amodal 全景），设计统一且高效

## 局限与展望

- 测试集仅 100 张图像，规模较小，评估结果可能存在波动
- 全景图标注成本极高（210分钟/张/人），数据集扩展困难
- 某些小类别（van、traffic-light 等）性能仍然很低，跨域适应对长尾类不友好
- 仅在驾驶场景验证，室内全景场景未涉及
- AoMix 依赖源域 amodal 标注，限制了方法的泛化性
- Amodal 实例分割的 mAAP 仅 10.50%，绝对性能仍有很大提升空间

## 相关工作与启发

- **Trans4PASS** [CVPR 2022]: 提出 DPE 处理全景畸变，是 UnmaskFormer backbone 的基础
- **EDAPS** [CVPR 2023]: UDA 全景分割 SOTA，UnmaskFormer 在此基础上增加 amodal 能力
- **DAFormer/DACS**: 提供了 self-training 和 class-mix 的 UDA 策略框架
- **ORCNN**: 通过可见掩码推断遮挡掩码的 amodal 分割方法
- **启发**: 将 amodal 信息融入数据增强（AoMix）是一种巧妙的思路，比传统 CutMix/ClassMix 更适合遮挡场景

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首次定义 OASS 任务，将三个独立挑战统一，任务定义具有开创性
- **实验充分度**: ⭐⭐⭐⭐ 三个数据集验证、详尽消融、可视化丰富，但测试集规模偏小
- **写作质量**: ⭐⭐⭐⭐ 三重 "unmask" 叙事贯穿全文，逻辑清晰，图表美观
- **价值**: ⭐⭐⭐⭐ 数据集和基准有长期价值，但绝对性能仍偏低，实际应用仍有距离

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] EAFormer: Scene Text Segmentation with Edge-Aware Transformers](eaformer_scene_text_segmentation_with_edge-aware_transformers.md)
- [\[ICLR 2026\] AMLRIS: Alignment-aware Masked Learning for Referring Image Segmentation](../../ICLR2026/segmentation/amlris_alignment-aware_masked_learning_for_referring_image_segmentation.md)
- [\[ICCV 2025\] CAVIS: Context-Aware Video Instance Segmentation](../../ICCV2025/segmentation/cavis_context-aware_video_instance_segmentation.md)
- [\[CVPR 2026\] Frequency-Aware Affinity for Weakly Supervised Semantic Segmentation](../../CVPR2026/segmentation/frequency-aware_affinity_for_weakly_supervised_semantic_segmentation.md)
- [\[CVPR 2026\] From 2D Alignment to 3D Plausibility: Unifying Heterogeneous 2D Priors and Penetration-Free Diffusion for Occlusion-Robust Two-Hand Reconstruction](../../CVPR2026/segmentation/from_2d_alignment_to_3d_plausibility_unifying_heterogeneous_2d_priors_and_penetr.md)

</div>

<!-- RELATED:END -->
