---
title: >-
  [论文解读] Context Guided Transformer Entropy Modeling for Video Compression
description: >-
  [ICCV 2025][模型压缩][视频压缩] 提出Context Guided Transformer (CGT) 条件熵模型，通过时间上下文重采样器降低计算开销、依赖加权空间上下文分配器显式建模空间依赖关系，在视频压缩中将熵建模时间减少约65%，同时实现11% BD-Rate改进。 - 领域现状：深度神经网络推动的视频压…
tags:
  - "ICCV 2025"
  - "模型压缩"
  - "视频压缩"
  - "熵模型"
  - "Transformer"
  - "时空上下文"
  - "条件编码"
---

# Context Guided Transformer Entropy Modeling for Video Compression

**会议**: ICCV 2025  
**arXiv**: [2508.01852](https://arxiv.org/abs/2508.01852)  
**代码**: [https://github.com/EIT-NLP/CGT](https://github.com/EIT-NLP/CGT)  
**领域**: 模型压缩 / 视频压缩  
**关键词**: 视频压缩, 熵模型, Transformer, 时空上下文, 条件编码

## 一句话总结
提出Context Guided Transformer (CGT) 条件熵模型，通过时间上下文重采样器降低计算开销、依赖加权空间上下文分配器显式建模空间依赖关系，在视频压缩中将熵建模时间减少约65%，同时实现11% BD-Rate改进。

## 研究背景与动机
- **领域现状**：深度神经网络推动的视频压缩方法（特别是条件熵模型）已成为新兴范式，通过利用时空上下文来估计视频帧的概率质量函数(PMF)
- **现有痛点1**：时间维度方面，利用额外时间上下文不可避免地增加计算开销和推理延迟，如VCT需要处理两帧时间上下文的self-attention
- **现有痛点2**：空间维度方面，已有方法（自回归、棋盘式、最小熵解码等）采用预定义的固定顺序解码策略，缺乏对空间位置依赖关系的显式建模
- **核心矛盾**：如何在不显著增加计算开销的前提下同时有效利用时空上下文信息
- **切入角度**：以可学习查询进行时间上下文压缩重采样 + 教师-学生网络显式建模空间依赖权重
- **核心idea**：用紧凑的可学习查询重采样时间上下文以减少后续处理开销，用教师-学生Swin Transformer网络平衡token重要性与确定性来选择最优空间解码顺序

## 方法详解

### 整体框架
CGT建立在contextual-based视频编解码器之上。编码器将RGB图映射到潜在空间特征，CGT熵模型利用时间上下文（来自潜在缓冲区的历史帧信息）和空间上下文（当前帧已解码token）来估计当前帽表示的PMF，用于熵编码。整个框架包含帧编解码器和CGT熵模型两部分。

### 关键设计
1. **Temporal Context Resampler (TCR)**:

    - 功能：从多种类型和尺度的时间上下文中提取有效特征，生成固定长度的紧凑token序列
    - 核心思路：预定义一组小的可学习窗口查询(window queries)，通过Swin Transformer的窗口交叉注意力(window cross-attention)与时间上下文进行交互。小查询与大时间上下文在每个窗口内进行局部信息压缩
    - 设计动机：并非所有时间上下文信息同等重要，且信息量增加会显著影响解码速度。通过紧凑查询重采样，既能捕获关键时间依赖，又能大幅降低后续处理的计算成本

2. **Dependency-Weighted Spatial Context Assigner (DWSCA)**:

    - 功能：显式建模空间上下文的位置依赖关系，确定最具信息量的上下文用于未解码token
    - 核心思路：采用共享参数的教师-学生Swin Transformer解码器。教师网络从随机掩码输入生成注意力图(表示token重要性)和熵图(反映预测确定性)，通过加权组合计算依赖分数：$Score = \alpha H + (1-\alpha) A$，其中 $A$ 是归一化注意力图，$H$ 是归一化熵图。使用soft top-k选择最高依赖分数的位置进行解码，为学生网络提供上下文
    - 设计动机：之前方法（自回归/棋盘/最小熵）未显式建模空间依赖，难以为未解码token提供最相关的上下文信息。教师-学生结构保证训练-推理一致性

3. **随机掩码代理任务**:

    - 功能：解决教师网络中"当前帧已解码内容"在训练时无法预定义的问题
    - 核心思路：对输入潜在表示施加随机掩码 $y_t + M$，未掩码区域模拟已解码内容。教师网络基于掩码后的表示生成注意力图和熵图，指导学生网络解码
    - 设计动机：借鉴掩码图像建模思想，通过随机掩码模拟渐进解码过程，确保训练-推理一致性

### 损失函数 / 训练策略
率失真损失：$\mathcal{L}_{RD} = R(\hat{y}_t) + R(\hat{z}_t) + R(\hat{v}_t) + \lambda \cdot d(x_t - \hat{x}_t)$

其中 $R$ 为码率项，$d$ 为失真项，$\lambda \in \{256, 512, 1024, 2048\}$ 控制率失真权衡。训练集为Vimeo-90k，随机裁剪至265×256并随机翻转增强。解码采用8步正弦调度策略。

## 实验关键数据

### 主实验 (BD-Rate, PSNR, anchor=VTM)

| 数据集 | MCL-JCV | UVG | HEVC-B | 平均 |
|--------|---------|-----|--------|------|
| VTM | 0 | 0 | 0 | 0 |
| DMC | -24.5 | -26.1 | -49.4 | -33.3 |
| MIMT | -33.0 | -34.9 | -57.1 | -41.7 |
| **CGT** | **-43.8** | **-45.5** | **-62.5** | **-50.6** |

BD-Rate (MS-SSIM)：CGT平均-74.7%，大幅超越MIMT的-65.3%和DMC的-55.4%。

### 消融实验
**时间上下文重采样器消融**:

| 模型 | BD-Rate变化 | 熵建模时间 | 编码时间 | 解码时间 |
|------|-----------|-----------|---------|---------|
| CGT-w/o TCR | anchor | 1305ms | 1682ms | 1576ms |
| CGT-w/ TCR | +1.8% | 488ms (↓63%) | 1073ms (↓35%) | 984ms (↓38%) |

**空间上下文分配器消融** (anchor=最小熵解码):

| 模型 | MCL-JCV | UVG | HEVC-B | 平均 |
|------|---------|-----|--------|------|
| CGT-DWSCA (本文) | -11.3 | -7.8 | -14.6 | -11.2 |
| CGT-min-entropy | 0 | 0 | 0 | 0 |
| CGT-checkerboard | +17.7 | +15.1 | +19.2 | +17.3 |
| CGT-autoregressive | +19.3 | +16.6 | +22.8 | +19.5 |

**权重系数α分析** (λ=256, MCL-JCV):
- α=0 (仅注意力，重要性)：PSNR 35.88, Bpp 0.019
- α=1 (仅熵，确定性)：PSNR 35.3, Bpp 0.017
- α=0.5 (均衡)：PSNR 35.82, Bpp 0.018

### 关键发现
- TCR仅带来1.8% BD-Rate上升，却减少63%熵建模时间、35%编码时间、38%解码时间
- 显式依赖建模比代理任务(随机掩码)建模效果更好，因为减少了训练-推理不匹配
- CGT在更换帧编解码器(DCVC→DCVC-DC)后仍保持优异性能，展示了良好的泛化能力
- 相比VTM anchor，CGT在PSNR指标上平均降低50.6% BD-Rate

## 亮点与洞察
- 时间上下文重采样的思路非常高效——用少量可学习查询配合交叉注意力实现信息压缩，大幅降低后续计算，同时保持编码效果
- 教师-学生+soft top-k的空间解码方案比固定顺序（自回归/棋盘）和启发式顺序（最小熵）都更优，验证了显式依赖建模的必要性
- α=0(仅重要性)降低失真，α=1(仅确定性)降低码率，两者的互补性被很好地利用

## 局限与展望
- 固定α=0.5可能不是所有场景下的最优选择，自适应α可能带来进一步提升
- 8步解码调度固定为正弦函数，更灵活的调度策略可能改善性能
- 训练集Vimeo-90k分辨率有限(448×256)，对高分辨率视频的泛化能力有待验证
- 未与基于隐式表示的最新方法(NVRC, MVC)进行速度-性能综合比较

## 相关工作与启发
- MIMT的最小熵原则是重要的baseline，本文在其基础上进一步引入显式依赖建模
- 可学习查询+交叉注意力的信息压缩思路广泛适用于需要减少计算开销的场景
- 教师-学生网络的训练-推理一致性设计对其他需要渐进解码的任务有参考价值

## 评分
- 新颖性: ⭐⭐⭐⭐ 时间重采样+空间依赖显式建模的组合设计有新意，特别是教师-学生soft top-k方案
- 实验充分度: ⭐⭐⭐⭐ 消融实验全面，涵盖TCR/DWSCA/α/泛化/显式建模等多维度分析
- 写作质量: ⭐⭐⭐⭐ 问题动机清晰，方法描述详细，但部分符号使用可更统一
- 价值: ⭐⭐⭐⭐ 在压缩效率和计算成本之间取得了良好平衡，65%熵建模加速+11% BD-Rate提升的实际意义显著

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Learned Image Compression with Hierarchical Progressive Context Modeling](learned_image_compression_with_hierarchical_progressive_context_modeling.md)
- [\[ACL 2026\] Latent-Condensed Transformer for Efficient Long Context Modeling](../../ACL2026/model_compression/latent-condensed_transformer_for_efficient_long_context_modeling.md)
- [\[ICCV 2025\] EA-ViT: Efficient Adaptation for Elastic Vision Transformer](ea-vit_efficient_adaptation_for_elastic_vision_transformer.md)
- [\[ICML 2025\] Strategic Fusion Optimizes Transformer Compression](../../ICML2025/model_compression/strategic_fusion_optimizes_transformer_compression.md)
- [\[ICCV 2025\] Efficient Adaptation of Pre-Trained Vision Transformer Underpinned by Approximation Theory](efficient_adaptation_of_pre-trained_vision_transformer_underpinned_by_approximat.md)

</div>

<!-- RELATED:END -->
