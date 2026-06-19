---
title: >-
  [论文解读] Learned Image Compression with Hierarchical Progressive Context Modeling
description: >-
  [ICCV 2025][模型压缩][学习图像压缩] 提出分层渐进上下文模型 (HPCM)，通过将 latent 划分为多尺度子表征并从小到大依次编码，结合跨编码步的渐进上下文融合机制（基于交叉注意力），实现更高效的远程依赖建模和更准确的熵参数估计，在图像压缩性能和计算复杂度之间取得更好的平衡。 - 学习图像压缩的核心在于熵模…
tags:
  - "ICCV 2025"
  - "模型压缩"
  - "学习图像压缩"
  - "上下文建模"
  - "熵编码"
  - "层次编码"
  - "渐进融合"
---

# Learned Image Compression with Hierarchical Progressive Context Modeling

**会议**: ICCV 2025  
**arXiv**: [2507.19125](https://arxiv.org/abs/2507.19125)  
**代码**: [github.com/lyq133/LIC-HPCM](https://github.com/lyq133/LIC-HPCM)  
**领域**: 图像压缩  
**关键词**: 学习图像压缩, 上下文建模, 熵编码, 层次编码, 渐进融合

## 一句话总结

提出分层渐进上下文模型 (HPCM)，通过将 latent 划分为多尺度子表征并从小到大依次编码，结合跨编码步的渐进上下文融合机制（基于交叉注意力），实现更高效的远程依赖建模和更准确的熵参数估计，在图像压缩性能和计算复杂度之间取得更好的平衡。

## 研究背景与动机

- 学习图像压缩的核心在于熵模型：更准确的概率分布估计 → 更少的比特 → 更好的压缩性能
- 条件熵模型联合使用 hyperprior 和 context model（将 latent 分组顺序编码）来捕获上下文信息
- 现有方法的两个关键问题：
    - **远程依赖建模效率低**：Transformer 架构可以捕获远程依赖但引入高复杂度
    - **跨编码步的多样化上下文利用不足**：每一步只使用 hyperprior 和已编码 latent，未充分利用之前步的上下文信息
- 本文目标：通过分层编码调度和渐进上下文融合，实现更高效的上下文信息获取

## 方法详解

### 整体框架

遵循标准的学习图像压缩框架：分析变换 → 量化 → 熵编码 → 解码 → 合成变换。核心改进在于熵模型中的上下文建模部分——提出 HPCM 替代传统的单尺度上下文模型。

### 关键设计

1. **分层编码调度（Hierarchical Coding Schedule）**: 将量化后的 latent $\hat{y}$ 通过特殊采样划分为三个尺度的子表征 $\hat{y}^{S1}$（4× 降采样）、$\hat{y}^{S2}$（2× 降采样）和 $\hat{y}^{S3}$（原始尺度）。从最小尺度开始顺序编码，逐步从远程到近程建模依赖关系。关键优势：在较小尺度上用轻量级操作即可获得大感受野（ERF 可视化证实 $g_{ep}^{S1}$ 感受野远大于 $g_{ep}^{S3}$），避免了在原始分辨率上做全局注意力的高复杂度。编码完 $\hat{y}^{S1}$ 后将其填回 $\hat{y}^{S2}$ 的对应位置，依此类推。不同通道组使用不同的空间划分策略实现空间-通道交互。各尺度编码步数分配为 (2, 3, 6)，共 11 步。

2. **渐进上下文融合（Progressive Context Fusion, PCF）**: 核心创新——将前一编码步的上下文信息融入当前步。在第 i 步，上下文 $C_i^{S2}$ 通过融合前一步的 $C_{i-1}^{S2}$ 和熵参数状态 $\psi_{i-1}$ 得到。融合通过交叉注意力实现：$Q = \text{Linear}(\psi_i)$，$K,V = \text{Linear}(C_i^{S2})$，$C_{i+1}^{S2} = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V + \text{Linear}(\psi_i)$。跨尺度传播：将小尺度的上下文 $C_6^{S2}$ 填入大尺度 $C_6^{S3}$ 的对应位置，其余位置用 hyperprior 信息填充。

3. **参数高效设计与其他改进**: 使用先进网络结构（深度残差连接、非局部注意力）构建变换和上下文模型网络。不同编码步共享网络参数，显著减少模型参数。优化了 Python-C 之间的数据交换实现更高效的算术编码。

### 损失函数 / 训练策略

- 率失真损失：$L = \mathcal{R}(\hat{y}) + \mathcal{R}(\hat{z}) + \lambda \cdot \mathcal{D}(x, \hat{x})$
- 概率分布模型：广义高斯模型 $\mathcal{N}_\beta(\mu, \alpha)$，$\beta=1.5$
- MSE 优化：$\lambda \in \{0.0018, 0.0035, 0.0067, 0.0130, 0.0250, 0.0483\}$
- MS-SSIM 优化：$\lambda \in \{2.40, 4.58, 8.73, 16.64, 31.73, 60.50\}$
- 训练数据：Flicker2W，256×256 patch，batch=32
- 200 万步训练，Adam 优化器，学习率从 1e-4 分阶段衰减到 1e-6

## 实验关键数据

### 主实验

**BD-Rate 性能（相对 VTM-22.0，PSNR，负值表示节省比特）**:

| 方法 | 参数(M) | kMACs/pixel | Kodak | CLIC Pro | Tecnick |
|------|---------|-------------|-------|----------|---------|
| ELIC (CVPR'22) | 36.93 | 573.88 | -3.22% | -3.89% | -4.57% |
| TCM (CVPR'23) | 76.57 | 1823.58 | -10.70% | -8.32% | -11.84% |
| MLIC++ (ICML'23) | 116.72 | 1282.81 | -15.15% | -14.05% | -17.90% |
| FLIC (ICLR'24) | 70.96 | 1096.04 | -13.20% | -9.88% | -15.27% |
| **HPCM-Base** | 68.50 | 918.57 | **-15.31%** | -14.23% | -18.16% |
| **HPCM-Large** | 89.71 | 1261.29 | **-19.19%** | **-18.37%** | **-22.20%** |

**与相同变换下不同熵模型对比**:

| 熵模型 | kMACs/pixel | Kodak BD-Rate |
|--------|-------------|---------------|
| CHARM | 495.75 | +0.86% |
| DCVC-DC intra | 542.14 | -9.18% |
| **HPCM-Base** | 918.57 | **-15.31%** |

### 消融实验

**分层编码调度消融**:

| 配置 | kMACs/pixel | BD-Rate |
|------|-------------|---------|
| HPCM-Base (2,3,6) 默认 | 918.57 | 0.00% |
| 去除分层提取（在原始尺度建模） | 1107.48 | +1.07% |
| 编码步 (2,3,3) | 663.90 | +2.39% |
| 编码步 (2,3,12) | 1427.91 | -2.55% |
| 编码步 (4,3,6) | 925.59 | +0.35% |

**渐进上下文融合消融**:

| 配置 | kMACs/pixel | BD-Rate |
|------|-------------|---------|
| HPCM-Base（含 PCF） | 918.57 | 0.00% |
| 去除渐进融合 | 872.80 | +4.71% |
| 用 ψ_i 作为渐进上下文 | 872.80 | +1.17% |

### 关键发现

- **渐进上下文融合是最关键的组件**：去除后性能下降 4.71%，说明跨编码步的上下文积累对熵估计至关重要
- 分层编码调度有效实现远程依赖建模：在小尺度上感受野自然更大（ERF 可视化验证）
- 交叉注意力融合优于简单使用熵参数状态（4.71% vs 1.17%），精确的上下文集成机制很重要
- HPCM-Large 在 Kodak/Tecnick 上比 VTM-22.0 节省约 19-22% 比特率，达到学术 SOTA
- 编码步数对最大尺度 S3 的影响最显著（更多 latent → 更大收益），但 (2,3,12) 虽多省 2.55% 但计算量暴增
- PCF 模块的注意力图集中在高比特率区域（复杂纹理），说明模型学会了在困难区域投入更多上下文建模

## 亮点与洞察

- **分层调度的优雅设计**：不需要 Transformer 的全局注意力就能建模远程依赖——在小尺度上用轻量级操作自然获得大感受野
- **渐进融合的思路**与 RNN 的隐状态传递有异曲同工之妙，但通过交叉注意力实现更灵活的信息选择
- 跨尺度上下文传播设计精巧：小尺度的积累上下文直接填入大尺度的对应位置，hyperprior 补充其余位置
- 参数共享的编码步复用设计在减少参数的同时保持了上下文感知的适应能力
- 优化算术编码的 Python-C 接口也是实用的工程贡献

## 局限与展望

- 编码/解码时间 (~80-90ms) 虽优于部分方法但仍可优化，特别是算术编码的串行瓶颈
- (2,3,6) 编码步数是手动选择的 trade-off，可以探索学习端自动分配
- 目前主要在 MSE/MS-SSIM 上优化，感知质量和主观评价的优化空间尚未充分探索
- 可以与可变比特率（variable-rate）技术结合，实现单模型多码率

## 相关工作与启发

- checkboard 模型 (He et al.) 和通道上下文模型 (Minnen et al.) 是上下文建模的基础
- MLIC++ 的多参考上下文模型启发了多样化上下文的思路，但 HPCM 通过分层+渐进更高效
- 分层编码思想与传统编码中的多分辨率方法（如 JPEG2000 的小波分解）有理念相通之处
- 渐进融合的交叉注意力机制可以迁移到其他需要多步骤上下文积累的任务

## 评分

- **新颖性**: ⭐⭐⭐⭐ 分层编码+渐进融合的组合设计新颖，交叉注意力融合有洞察
- **实验充分度**: ⭐⭐⭐⭐⭐ 3个数据集，与VTM/多种SOTA对比，消融设计完善
- **写作质量**: ⭐⭐⭐⭐ 图示清晰（特别是Fig.2的编码调度图），方法描述条理分明
- **价值**: ⭐⭐⭐⭐⭐ SOTA压缩性能+良好的性能-复杂度trade-off，学术和工业均有价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] LALIC: Linear Attention Modeling for Learned Image Compression](../../CVPR2025/model_compression/linear_attention_modeling_for_learned_image_compression.md)
- [\[ICCV 2025\] Context Guided Transformer Entropy Modeling for Video Compression](context_guided_transformer_entropy_modeling_for_video_compression.md)
- [\[NeurIPS 2025\] 4DGCPro: Efficient Hierarchical 4D Gaussian Compression for Progressive Volumetric Video Streaming](../../NeurIPS2025/model_compression/4dgcpro_efficient_hierarchical_4d_gaussian_compression_for_p.md)
- [\[CVPR 2025\] Learned Image Compression with Dictionary-based Entropy Model](../../CVPR2025/model_compression/learned_image_compression_with_dictionary-based_entropy_model.md)
- [\[CVPR 2026\] What Matters in Practical Learned Image Compression](../../CVPR2026/model_compression/what_matters_in_practical_learned_image_compression.md)

</div>

<!-- RELATED:END -->
