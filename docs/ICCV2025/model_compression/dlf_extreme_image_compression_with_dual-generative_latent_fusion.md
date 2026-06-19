---
title: >-
  [论文解读] DLF: Extreme Image Compression with Dual-generative Latent Fusion
description: >-
  [ICCV 2025][模型压缩][极低码率图像压缩] 提出双分支生成式隐空间融合（DLF）框架，将图像隐空间分解为语义和细节两个分支分别压缩，通过跨分支交互设计消除冗余，在极低码率（<0.01 bpp）下实现了超越 MS-ILLM 高达 67.82% BD-Rate 节省的 SOTA 重建质量，同时解码速度远快于扩散模型方案。
tags:
  - "ICCV 2025"
  - "模型压缩"
  - "极低码率图像压缩"
  - "生成式编解码"
  - "双分支编码"
  - "向量量化"
  - "语义-细节分解"
---

# DLF: Extreme Image Compression with Dual-generative Latent Fusion

**会议**: ICCV 2025  
**arXiv**: [2503.01428](https://arxiv.org/abs/2503.01428)  
**代码**: [dlfcodec.github.io](https://dlfcodec.github.io/)  
**领域**: 模型压缩/图像压缩  
**关键词**: 极低码率图像压缩, 生成式编解码, 双分支编码, 向量量化, 语义-细节分解

## 一句话总结

提出双分支生成式隐空间融合（DLF）框架，将图像隐空间分解为语义和细节两个分支分别压缩，通过跨分支交互设计消除冗余，在极低码率（<0.01 bpp）下实现了超越 MS-ILLM 高达 67.82% BD-Rate 节省的 SOTA 重建质量，同时解码速度远快于扩散模型方案。

## 研究背景与动机

### 领域现状

极低码率图像压缩（如 0.01 bpp 以下）是当前图像压缩领域的核心挑战。传统编解码器（VVC）和基于 MSE 优化的神经网络编解码器在低码率下会产生严重模糊。近年来，基于生成式 tokenizer（如 VQGAN）的方法通过将图像压缩为少量 token 索引来实现极高压缩比，但这种方法面临根本性矛盾。

### 现有痛点

**单一 codebook 的容量瓶颈**：VQGAN 等生成式 tokenizer 通过在整个数据集上聚类语义来学习紧凑 codebook，这种方式优先编码数据集级别的共性内容，但在 codebook 尺寸缩小时，个体对象的独特细节（如精确的几何特征）会严重失真

**扩散模型方案的局限**：PerCo、DiffEIC 等扩散模型方案虽然提升了生成真实感，但在忠实度上不够理想，且解码速度极慢（4秒以上）

**双分支方案的冗余问题**：已有的双分支方案 HybridFlow 使用独立编码器，两条码流之间存在大量信息冗余

### 核心切入

能否将隐空间灵活地分解为"语义"和"细节"两部分，分别用最适合的压缩策略处理，同时通过交互设计消除冗余？

## 方法详解

### 整体框架

DLF 采用双分支编码架构：输入图像 $X \in \mathbb{R}^{3 \times H \times W}$ 先通过 patch embedding 得到 $Emb(X) \in \mathbb{R}^{C \times h \times w}$（$h=H/16, w=W/16$），然后分别送入语义分支和细节分支。语义分支用 1-D tokenizer 聚类高层语义为紧凑 token，细节分支用标量量化（SQ）编码感知关键细节。两个分支通过多层 Interactive Transform（IT）模块进行跨分支交互。解码后，语义和细节特征通过 latent adaptor 融合，最终由预训练的 VQGAN decoder 生成重建图像。

### 关键设计

#### 1. **语义分支（1-D Tokenizer）**
- **功能**：将图像的高层语义压缩为极少量的 1-D token
- **核心思路**：将 $Emb(X)$ 划分为 $16 \times 16$ 的窗口，每个窗口内将 256 个 2-D 图像 token 和 32 个额外的 1-D token 一起送入 ViT。通过级联注意力，1-D token 高效聚合图像 token 中的关键语义信息。最终得到 $y_s \in \mathbb{R}^{N \times C \times 32}$，其中 $N = \frac{h \times w}{16 \times 16}$
- **量化方式**：向量量化（VQ），使用固定长度编码传输 codebook 索引
- **设计动机**：相比手动预定义 mask 的 token 减少策略，通过大规模学习自动压缩语义，处理不同图像时更灵活

#### 2. **细节分支（SQ + 自适应比特分配）**
- **功能**：捕获 VQ codebook 无法表示的个体级细节信息
- **核心思路**：使用 shifted window attention + ConvNeXT 提取局部和全局细节，下采样得到 $y_d \in \mathbb{R}^{C \times h/2 \times w/2}$。使用标量量化（SQ）+ 四叉树划分熵模型进行算术编码
- **量化公式**：$\hat{y}_s = VQ(y_s), \quad \hat{y}_d = Q(y_d)$
- **设计动机**：SQ 提供远大于 VQ 的量化空间，可以对不同空间区域使用不同的可学习量化步长，实现自适应比特分配——对独特的物体轮廓分配更多比特，对可被语义分支良好表示的常见内容分配更少比特

#### 3. **跨分支交互设计（Interactive Transform, IT）**
- **功能**：消除语义和细节两条码流之间的信息冗余
- **核心思路**：将细节特征 $f_d$ 按照与语义分支相同的窗口策略重排为 $\tilde{f_d} \in \mathbb{R}^{N \times C \times 256}$，然后与语义特征联合送入多头自注意力层。处理后的细节特征再恢复原始形状
- **设计动机**：（1）自注意力层动态重新分配语义和细节信息到各分支，减少冗余，同时让细节信息动态修正语义分支的生成误差；（2）为语义分支提供跨窗口感知能力——细节特征贡献全局信息，扩展语义分支的感受野

### 损失函数 / 训练策略

采用两阶段渐进式训练：

1. **阶段一（隐空间对齐）**：在隐空间施加 rate-distortion loss，用预训练 VQGAN encoder 生成的特征 $\tilde{h}$ 监督融合特征 $\hat{h}$ 的重建。256×256 patches，batch size 8，固定 $\lambda = 24.0$
2. **阶段二（端到端微调）**：在像素空间使用生成式损失微调整个模型。512×512 patches，batch size 4，$\lambda \in \{5.8, 8.5, 16.0, 28.0\}$ 实现不同码率

## 实验关键数据

### 主实验

| 方法 | 数据集 | LPIPS BD-Rate | DISTS BD-Rate | 说明 |
|------|--------|-------------|-------------|------|
| DLF | Kodak | **-43.05%** | **-67.82%** | 本文方法 |
| GLC | Kodak | -17.24% | -33.41% | Tokenizer-based |
| DiffEIC | Kodak | +66.05% | +14.67% | 扩散模型方案 |
| PerCo | Kodak | +101.74% | -4.02% | 扩散模型方案 |
| HybridFlow | Kodak | +65.30% | — | 双分支方案 |
| MS-ILLM | Kodak | 0.00% (anchor) | 0.00% (anchor) | 基准 |
| DLF | CLIC2020 | **-27.93%** | **-53.55%** | 本文方法 |

### 消融实验

| 配置 | Kodak LPIPS | Kodak DISTS | CLIC LPIPS | CLIC DISTS | 说明 |
|------|------------|------------|-----------|-----------|------|
| w/ SQ detail (DLF) | 0.0% | 0.0% | 0.0% | 0.0% | 完整方案 (anchor) |
| w/o detail | +17.5% | +20.2% | +47.9% | +47.6% | 移除细节分支 |
| w/o interactive | +64.1% | +73.6% | +68.8% | +61.8% | 移除IT模块 |
| w/ VQ detail | +18.3% | +40.7% | +27.3% | +58.1% | 细节分支用VQ代替SQ |

### 复杂度分析

| 模型 | 编码时间 | 解码时间 | DISTS BD-Rate |
|------|---------|---------|-------------|
| MS-ILLM | 0.064s | 0.070s | 0.00% |
| PerCo | 0.461s | 2.443s | -4.02% |
| DiffEIC | 0.152s | **4.093s** | +14.67% |
| DLF | 0.178s | **0.252s** | **-67.82%** |

### 关键发现

- 移除跨分支交互（IT模块）导致最严重的性能下降（>60% BD-Rate loss），证明独立双分支存在巨大冗余
- 细节分支使用 SQ 显著优于 VQ，证实了大量化空间对表示多样化细节的重要性
- DLF 的解码速度比扩散方案快 10-16 倍，同时在忠实度上显著更优
- 在 CLIC2020 768×768 上，DLF 的 FID 大幅超越 PerCo，证明高质量数据集上的优势

## 亮点与洞察

1. **语义-细节分解思想**：将"数据集级共性"和"个体级多样性"解耦为两条码流，各自用最适合的量化策略，是一个简洁而有效的设计思路
2. **跨分支交互的关键性**：消融实验清晰表明，不做交互的独立双分支反而不如单分支，交互设计是双分支方案成功的核心
3. **SQ vs VQ 的深层洞察**：VQ 的有限 codebook 天然适合编码聚类的共性语义，SQ 的大量化空间适合编码多样化的细节，混合量化策略是最优解
4. **对扩散方案的有力竞争**：在保持可比的生成真实感的同时，大幅提升忠实度和解码速度

## 局限与展望

1. **编解码速度仍未达实时**：编码 0.178s、解码 0.252s，离实时应用有差距
2. **训练成本高**：需要两阶段训练，且依赖预训练的 VQGAN tokenizer
3. **码率控制粒度**：通过调整 $\lambda$ 控制码率，灵活性有限
4. **仅评估了感知指标**：缺少下游任务（如检测、分割）的评估

## 相关工作与启发

- HybridFlow 的独立双分支设计验证了冗余消除的必要性，DLF 通过 IT 模块实现 2.6× 更高压缩比
- 1-D tokenizer 提供了紧凑语义压缩的基础，DLF 在此基础上增加了细节编码
- 与扩散模型方案的对比表明，tokenizer 路线在速度和忠实度上具有优势

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 语义-细节分解 + 跨分支交互是优雅的设计，但双分支思想本身并非全新
- **实验充分度**: ⭐⭐⭐⭐ — 多数据集评估 + 完整消融 + 复杂度分析，较为充分
- **写作质量**: ⭐⭐⭐⭐⭐ — 动机清晰，从 VQ 局限性出发推导出方案，逻辑流畅
- **价值**: ⭐⭐⭐⭐ — 在极低码率压缩领域取得显著进展，对 tokenizer-based 压缩方向有重要参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Generative Video Compression with One-Dimensional Latent Representation](../../CVPR2026/model_compression/generative_video_compression_with_one-dimensional_latent_representation.md)
- [\[ICCV 2025\] Learned Image Compression with Hierarchical Progressive Context Modeling](learned_image_compression_with_hierarchical_progressive_context_modeling.md)
- [\[CVPR 2026\] CADC: Content Adaptive Diffusion-Based Generative Image Compression](../../CVPR2026/model_compression/cadc_content_adaptive_diffusion-based_generative_image_compression.md)
- [\[ICCV 2025\] Fuse Before Transfer: Knowledge Fusion for Heterogeneous Distillation](fuse_before_transfer_knowledge_fusion_for_heterogeneous_distillation.md)
- [\[ICML 2025\] Strategic Fusion Optimizes Transformer Compression](../../ICML2025/model_compression/strategic_fusion_optimizes_transformer_compression.md)

</div>

<!-- RELATED:END -->
