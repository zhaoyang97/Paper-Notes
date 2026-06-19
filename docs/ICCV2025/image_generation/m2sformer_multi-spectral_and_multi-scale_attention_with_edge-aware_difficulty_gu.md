---
title: >-
  [论文解读] M2SFormer: Multi-Spectral and Multi-Scale Attention with Edge-Aware Difficulty Guidance for Image Forgery Localization
description: >-
  [ICCV 2025][图像生成][图像篡改定位] 提出 M2SFormer，在编码器-解码器的 skip connection 中统一多光谱（2D DCT 频域）和多尺度（SIFT 风格空间金字塔）注意力机制，并设计基于边缘感知曲率的难度引导注意力解码器，在图像篡改定位任务中实现跨域泛化性能 SOTA（CASIAv2 训练方案下 unseen 域平均 DSC 43.0%，mIoU 34.3%）。
tags:
  - "ICCV 2025"
  - "图像生成"
  - "图像篡改定位"
  - "多光谱注意力"
  - "多尺度注意力"
  - "难度引导"
  - "Transformer"
---

# M2SFormer: Multi-Spectral and Multi-Scale Attention with Edge-Aware Difficulty Guidance for Image Forgery Localization

**会议**: ICCV 2025  
**arXiv**: [2506.20922](https://arxiv.org/abs/2506.20922)  
**代码**: 无  
**领域**: 图像篡改检测/图像生成  
**关键词**: 图像篡改定位, 多光谱注意力, 多尺度注意力, 难度引导, Transformer

## 一句话总结

提出 M2SFormer，在编码器-解码器的 skip connection 中统一多光谱（2D DCT 频域）和多尺度（SIFT 风格空间金字塔）注意力机制，并设计基于边缘感知曲率的难度引导注意力解码器，在图像篡改定位任务中实现跨域泛化性能 SOTA（CASIAv2 训练方案下 unseen 域平均 DSC 43.0%，mIoU 34.3%）。

## 研究背景与动机

### 问题定义

图像篡改定位（Image Forgery Localization）要求模型在像素级别准确分割出被篡改（拼接或复制-移动）的区域。核心挑战在于：（1）篡改痕迹通常极其微妙；（2）模型需要对未见过的篡改类型和数据域具有泛化能力。

### 已有方法的不足

**CNN 方法（MantraNet, SPAN, RRUNet 等）**：计算开销大、表征能力有限，难以捕捉全局依赖关系，泛化能力不足

**Transformer 方法（TransForensic）**：利用自注意力建模全局依赖但未充分利用频域信息

**频域方法（FBINet, ObjectFormer）**：利用 2D DCT 揭示隐藏的篡改痕迹，但通常需要额外的双编码器或多模态训练，计算昂贵

**频域与空间域分离**：现有方法通常分别处理空间和频域特征，缺少统一的注意力机制来联合利用两者

### 核心动机

**关键问题**：如何在高效整合空间和频域注意力的同时，有效捕捉微妙的篡改特征？

人类视觉系统利用多频率带检测微妙伪影（多光谱），同时在多空间尺度上捕捉不同大小的篡改模式（多尺度）。将这两者在 skip connection 中统一融合，配合难度感知的解码器，可以同时提升准确性和跨域泛化能力。

## 方法详解

### 整体框架

M2SFormer 采用 Transformer 编码器-解码器架构（PVT-v2 骨干），包含两大核心模块：（1）skip connection 中的 M2S 注意力模块融合多光谱和多尺度信息；（2）Edge-Aware DGA 解码器根据样本难度自适应调整注意力。

### 关键设计

#### 1. **多光谱注意力（Multi-Spectral Attention）**

- **功能**：利用 2D DCT 基图像对跨尺度融合特征进行频域通道注意力重标定，捕捉不同频率分量中的篡改痕迹
- **核心思路**：首先将各层编码器特征统一到目标分辨率并拼接得到 $\mathbf{f}_c \in \mathbb{R}^{C \times H_t \times W_t}$。然后利用 2D DCT 基图像计算频域特征分量：
  $$\mathbf{f}_c^k = \sum_{h=0}^{H_t-1} \sum_{w=0}^{W_t-1} (\mathbf{f}_c)_{:,h,w} \mathbf{D}_{h,w}^{u_k,v_k}$$
  通过 top-K 选择策略选取最相关的频率分量，再经 GAP/GMP + 统计聚合块生成通道注意力图：
  $$\mathbf{M}^{\text{spectral}} = \sigma\left(\sum_{d \in \{\text{avg}, \text{max}\}} \sum_{k=1}^K \text{C2D}_{1\times1}(\delta(\text{C2D}_{1\times1}(\mathbf{f}_c^k)))\right)$$
- **设计动机**：与 FBINet 等直接在输入图像上应用 2D DCT 不同，本方法在 skip connection 的融合特征上操作，避免了双编码器的计算冗余，同时保留了粗到细的多层次语义信息。

#### 2. **多尺度注意力（Multi-Scale Attention）**

- **功能**：受 SIFT 特征金字塔启发，在多个空间尺度上构建注意力金字塔，捕捉不同尺寸的篡改模式
- **核心思路**：对经过光谱注意力重标定的特征 $\bar{\mathbf{f}}_c$ 进行多尺度下采样，每个尺度使用膨胀卷积和 $1\times1$ 卷积压缩通道：
  $$\bar{\mathbf{f}}_c^l = \text{C2D}_{1\times1}(\text{DC2D}_{3\times3}^{2l+1}(\text{Down}_l(\bar{\mathbf{f}}_c)))$$
  在每个金字塔层级引入可学习参数 $\alpha_i^l, \beta_i^l$ 控制前景/背景注意力的信息流：
  $$\hat{\mathbf{f}}_i^l = \text{C2D}_{3\times3}(\alpha_i^l(\bar{\mathbf{f}}_i^l \times \mathbf{F}_i^l) + \beta_i^l(\bar{\mathbf{f}}_i^l \times \mathbf{B}_i^l))$$
  其中 $\mathbf{F}_i^l = \sigma(\text{C2D}_{1\times1}(\bar{\mathbf{f}}_i^l))$ 是前景图，$\mathbf{B}_i^l = 1 - \mathbf{F}_i^l$。
- **设计动机**：通过在多个尺度上分解前景/背景并独立加权，模型可以灵活适应不同大小的篡改区域，从微小的拼接到大面积的复制-移动。

#### 3. **边缘感知难度引导注意力（Edge-Aware DGA）**

- **功能**：自动评估每个样本的篡改定位难度（简单/困难），生成文本描述并通过通道注意力引导解码器
- **核心思路**：从最深层特征生成全局先验图 $\mathbf{G}$，使用 Sobel 滤波器计算一阶和二阶导数，得到曲率图：
  $$\kappa = \frac{G_x^2 G_{yy} - 2G_x G_y + G_y^2 G_{xx}}{(G_x^2 + G_y^2)^{1.5}}$$
  仅在边缘区域计算平均曲率：$s = \sigma(\sum(\kappa \otimes E) / \sum E)$，若 $s \geq 0.5$ 则标记为 "hard"，否则为 "easy"。将文本标签通过 BPE 编码为向量 $\mathcal{T}$，经线性嵌入后执行通道注意力引导解码。
- **设计动机**：低曲率区域容易处理，高曲率区域需要更多感知资源。通过边缘感知的曲率度量来量化难度比直接使用全图平均更有代表性（避免大面积零曲率区域稀释信号），而将难度转换为文本驱动的注意力机制使模型能自适应分配注意力资源。

### 损失函数 / 训练策略

- 总损失：$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}}(\mathbf{R}_t, \mathbf{R}_p) + \mathcal{L}_{\text{BCE}}(\mathbf{R}_t, \text{Up}_{32}(\mathbf{G}))$
- 端到端训练，100 epochs，batch size 32，Adam 优化器，cosine annealing 学习率调度
- PVT-v2-B2 作为编码器/解码器骨干，用 ImageNet-1K 预训练权重初始化

## 实验关键数据

### 主实验

**CASIAv2 训练方案，unseen 域泛化性能（DSC/mIoU %）**：

| 方法 | CASIAv1 | Columbia | IMD2020 | CoMoFoD | In the Wild | MISD |
|------|---------|----------|---------|---------|-------------|------|
| TransForensic | 44.2/35.0 | 35.9/25.0 | 27.2/19.1 | 21.7/14.3 | 31.9/22.4 | 60.0/46.5 |
| PIMNet | 49.7/42.2 | 32.5/23.1 | 29.6/22.2 | 24.7/16.8 | 31.2/22.9 | 61.1/48.2 |
| EITLNet | 52.9/46.5 | 28.0/20.9 | 25.3/19.7 | 18.1/12.4 | 24.3/19.0 | 58.8/45.9 |
| **M2SFormer** | **58.4/50.1** | **42.4/32.4** | **32.6/24.9** | **24.9/16.8** | **35.0/27.4** | **69.1/56.9** |

### 消融实验

**M2S 注意力模块消融（CASIAv2 方案）**：

| 配置 | 光谱 | 尺度 | Seen DSC | Unseen DSC | 参数 | FLOPs |
|------|------|------|----------|------------|------|-------|
| S0 | 单 | 单 | 56.3 | 27.1 | 26.2M | 13.8G |
| S1 | 单 | 多 | 55.5 | 33.6 | 27.4M | 14.2G |
| S2 | 多 | 单 | 55.9 | 36.1 | 26.2M | 13.8G |
| S3 (Full) | 多 | 多 | **58.8** | **43.0** | 27.4M | 14.2G |

**DGA 解码器消融**：

| 配置 | Seen DSC/mIoU | Unseen DSC/mIoU |
|------|--------------|-----------------|
| No DGA | 55.5/49.5 | 32.3/26.1 |
| Simple DC + DGA | — | 提升有限 |
| Edge-Aware DC + DGA | **58.8/50.8** | **43.0/34.3** |

### 关键发现

1. **多光谱+多尺度的协同效应**：两者单独贡献 unseen DSC +6.5/+9.0，组合后达到 +15.9（S3 vs S0），说明频域和空间尺度信息高度互补
2. **边缘感知难度引导至关重要**：EADC+DGA 将 unseen DSC 从 32.3 提升至 43.0（+33%），单纯的难度引导（Simple DC）效果有限
3. **M2SFormer 在跨域泛化上大幅领先**：在 MISD 上 DSC 达到 69.1%，比次优 PIMNet 高 8 个百分点
4. **计算效率优**：仅 27.4M 参数、14.2G FLOPs，比双编码器的 FBINet 更轻量

## 亮点与洞察

1. **Skip connection 中的统一注意力**：将频域和空间多尺度注意力放在 skip connection 而非编码器/解码器内，既保留了原始特征的信息流又注入了丰富的伪影信号
2. **文本驱动的自适应难度引导**：将曲率度量转化为"hard"/"easy"文本标签再通过 BPE 编码驱动注意力，避免了对外部元数据的依赖
3. **不依赖额外训练/微调**：仅用单一训练集即可实现多个 unseen 域上的优异泛化，无需像其他方法那样在外部数据上微调

## 局限与展望

1. **二值难度划分过于粗糙**：仅分"hard"和"easy"两类，更细粒度的难度量化可能带来更好的自适应性
2. **BPE 文本编码的必要性存疑**：直接使用曲率标量做条件注意力是否足够，文本中间表示是否引入了不必要的复杂性
3. **未在 AI 生成图像检测上评估**：随着 AI 换脸/生成图像的流行，该方法对 deepfake 的泛化能力有待验证
4. **阈值 $\tau=0.5$ 是固定的**：未提供自适应阈值方案
5. **5折交叉验证的标准差较大**：部分数据集上 DSC 标准差超过 10%，稳定性有待提升

## 相关工作与启发

- 与 FBINet 的区别：FBINet 在输入图像上直接应用 2D DCT，需要独立的频域编码器；M2SFormer 在跨尺度融合特征上操作，更高效
- 与 EITLNet 的区别：EITLNet 使用双编码器+MLP解码器，参数量和 FLOPs 更高；M2SFormer 用单编码器+Transformer 解码器+DGA 实现更好的精度
- 启发：将难度感知机制引入其他像素级任务（语义分割、深度估计）可能同样有效

## 评分

- **新颖性**: ⭐⭐⭐⭐ — M2S 注意力模块和边缘感知难度引导的设计有创意，但各子模块的灵感来源明确
- **实验充分度**: ⭐⭐⭐⭐ — 多训练方案、多 unseen 域、5折交叉验证系统全面，消融充分
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，图示详尽，但频域公式部分较密集
- **价值**: ⭐⭐⭐⭐ — 为图像篡改定位提供了强泛化能力的轻量级解决方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] MMAIF: Multi-task and Multi-degradation All-in-One for Image Fusion with Language Guidance](mmaif_multi-task_and_multi-degradation_all-in-one_for_image_fusion_with_language.md)
- [\[ICCV 2025\] DiTFastAttnV2: Head-wise Attention Compression for Multi-Modality Diffusion Transformers](ditfastattnv2_head-wise_attention_compression_for_multi-modality_diffusion_trans.md)
- [\[ICCV 2025\] Semantic Discrepancy-aware Detector for Image Forgery Identification](semantic_discrepancy-aware_detector_for_image_forgery_identification.md)
- [\[CVPR 2025\] Multi-party Collaborative Attention Control for Image Customization](../../CVPR2025/image_generation/multi-party_collaborative_attention_control_for_image_customization.md)
- [\[ICCV 2025\] ForgeLens: Data-Efficient Forgery Focus for Generalizable Forgery Image Detection](forgelens_data-efficient_forgery_focus_for_generalizable_forgery_image_detection.md)

</div>

<!-- RELATED:END -->
