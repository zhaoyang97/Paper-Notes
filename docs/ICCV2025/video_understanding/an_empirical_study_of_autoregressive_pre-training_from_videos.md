---
title: >-
  [论文解读] An Empirical Study of Autoregressive Pre-training from Videos
description: >-
  [ICCV 2025][视频理解][自回归预训练] 系统性地研究了从视频进行自回归预训练的方法（称为Toto），在超过1万亿视觉token上训练因果Transformer，发现尽管归纳偏置极少，自回归预训练在图像识别、视频分类、目标跟踪和机器人操控等多个下游任务上均具有竞争力，且展现出类似语言模型的缩放规律（但速率较慢）。
tags:
  - "ICCV 2025"
  - "视频理解"
  - "自回归预训练"
  - "视频模型"
  - "视觉Token"
  - "缩放规律"
  - "表征学习"
---

# An Empirical Study of Autoregressive Pre-training from Videos

**会议**: ICCV 2025  
**arXiv**: [2501.05453](https://arxiv.org/abs/2501.05453)  
**代码**: 无  
**领域**: 视觉表征学习 / 视频理解  
**关键词**: 自回归预训练, 视频模型, 视觉Token, 缩放规律, 表征学习

## 一句话总结
系统性地研究了从视频进行自回归预训练的方法（称为Toto），在超过1万亿视觉token上训练因果Transformer，发现尽管归纳偏置极少，自回归预训练在图像识别、视频分类、目标跟踪和机器人操控等多个下游任务上均具有竞争力，且展现出类似语言模型的缩放规律（但速率较慢）。

## 研究背景与动机
自回归预训练在NLP领域已取得巨大成功（GPT系列），其核心思想是通过"预测下一个token"来建模数据分布。然而在视觉领域，尤其是视频领域，这种方法尚未被充分探索。

现有视觉表征学习主要分两大范式：

**判别式方法**（如SimCLR、DINO）：通过实例判别或对比学习获得强识别特征，但不直接建模数据分布

**生成式方法**（如MAE、BEiT）：通过掩码自编码预训练，但并非自回归建模

视频是互联网上最大的Big Data来源，其时序结构天然适合自回归建模。然而此前的视觉自回归工作（如iGPT）主要在像素级别操作，计算开销巨大且难以扩展。

Toto的核心切入角度是：将视频帧通过tokenizer离散化为视觉token序列，然后像训练语言模型一样进行因果的下一个token预测。这使得图像和视频可以在统一格式下联合训练，并能利用语言模型领域成熟的训练技术和缩放经验。

## 方法详解

### 整体框架
Toto的流程简洁：视频/图像帧通过dVAE tokenizer转化为离散token序列→以光栅扫描顺序组成1D序列→用因果Transformer进行下一个token预测→取中间层特征进行下游任务评估。预训练完成后，通过注意力池化（attention pooling）从模型中间层提取表征用于下游迁移。

### 关键设计

1. **Tokenizer选择与评估**:

    - 功能：将图像/视频帧转化为离散token序列
    - 核心思路：默认使用dVAE（词汇量8k），每帧生成256个token（16×16网格）。视频取16帧，上下文长度为4096 token
    - 对比实验：dVAE、VQGAN和连续patch归一化token在ImageNet线性探测上表现相近（约61%），说明tokenizer选择对表征质量影响有限。但VQGAN因感知损失间接引入了ImageNet标签信息（通过VGG-net），存在数据污染问题
    - dVAE的优势：1-gram分布几乎完全覆盖所有token，而VQGAN不到50%覆盖率
    - 设计动机：使用图像级tokenizer可同时处理图像和视频，且避免监督信号泄漏

2. **架构设计（LLaMA风格）**:

    - 功能：提供高质量的因果序列建模能力
    - 核心思路：采用LLaMA架构——因果注意力Transformer，包含RMSNorm（预归一化）、SwiGLU激活和RoPE位置编码
    - 模型规模：Base(120M, 12层), Large(280M, 16层), 1B(1.1B, 22层)
    - 与GPT-2和Mamba对比：LLaMA在ImageNet线性探测上分别高出4.7%和12.5%
    - 训练配置：批量大小1M token, AdamW优化器, 最大学习率$3e-4$, $\beta_1=0.9, \beta_2=0.95$
    - 设计动机：利用语言模型领域最新的架构改进

3. **分辨率策略与RoPE适配**:

    - 功能：降低预训练成本同时提升性能
    - 核心方案：先在128×128（16×16 token）低分辨率预训练，再微调到256×256（32×32 token）。仅需1 epoch微调，竟然超过了全程256×256预训练的模型（64.4% vs 61.2%）
    - RoPE基值调整：微调时将RoPE基值从10,000提升到50,000，进一步改善高分辨率适配效果
    - 设计动机：高分辨率dVAE token数量是低分辨率的4倍，预训练计算量差异巨大

4. **下游特征提取策略**:

    - 功能：从decoder-only模型中提取高质量的视觉表征
    - 注意力池化 vs 平均池化：注意力池化在ImageNet上比平均池化高7.9%（61.1% vs 53.2%），因果注意力导致后部token看到更多上下文，简单平均会被偏斜结构稀释
    - 最优探测层：所有模型和任务中，最佳表征都出现在约50%深度处（中间层），与iGPT的发现一致。这暗示decoder-only模型前半部分像编码器压缩信息，后半部分将压缩语义投射回输入空间
    - 设计动机：decoder-only模型的表征提取方式与编码器-解码器模型不同，需要专门研究

### 数据集构建
训练数据总计约2.5万亿视觉token、10万+小时视频：
- ImageNet（13.9M图像），采样比例20%
- Kinetics-600（53万视频, 1496小时），采样比例10%
- Ego4D（5.2万视频, 3750小时），采样比例10%
- HowTo100m（117.2万视频, 9.2万小时），采样比例60%

实际训练使用约1万亿token。

### 训练策略
- 对视频采样16帧（每4帧取1帧），上下文4096 token
- 对图像随机采样16张组成序列，模拟视频格式
- 起止token：视频用[1]开头，图像用[3]开头，均用[2]结尾
- 损失函数：标准负对数似然 $\mathcal{L}_{\text{pre-train}} = \mathbb{E}_{x^j \sim X} -\log p(x^j)$

## 实验关键数据

### 主实验（多任务评估）

| 任务 | 数据集 | Toto-base | Toto-large | Toto-1b | 同类SOTA |
|------|--------|-----------|------------|---------|----------|
| 图像识别 | ImageNet | 64.7% | 71.1% | 75.3% | iGPT-XL: 72.0% (6.8B) |
| 动作识别 | K400 | 59.3% | 65.3% | 74.4% | VideoMAE: 79.8% |
| 动作预测 | Ego4D Overall | - | 2.70 | - | MAE-ST: 2.60 |
| 视频跟踪 | DAVIS J&F | 42.0 | 44.8/62.4(512) | 46.1 | DINO-B/8: 54.3 |
| 物体永续 | CATER | - | 62.8/72.9 | - | TFC-V3D: 54.6/70.2 |
| 机器人操控 | Real Franka | 63% | - | - | MVP: 75% |

### 消融实验

| 设计选择 | 最佳配置 | 关键指标 | 说明 |
|---------|---------|---------|------|
| Tokenizer | dVAE 32×32 | 61.2% Top1 | dVAE和VQGAN相当，但dVAE无数据污染 |
| 池化方式 | 注意力池化 | 61.1% vs 53.2% | 注意力池化远优于平均池化 |
| 分辨率策略 | 16→32 + RoPE 50k | 64.4% Top1 | 低分辨率预训练+高分辨率微调更优且更省算力 |
| 架构 | LLaMA | 53.2% Top1 | 优于GPT-2 (48.5%) 和 Mamba (40.7%) |
| 探测层 | ~50%深度 | - | 所有模型和任务一致 |

### 关键发现
- 在自回归生成模型中，Toto-1b用1.1B参数达到了iGPT-XL用6.8B参数才达到的水平
- 缩放规律呈幂律关系$L(C) = 7.32 \cdot C^{-0.0378}$，但比语言模型慢（GPT-3为$C^{-0.048}$）
- 视频帧的冗余性可能是缩放效率较低的原因之一
- 首次在自回归预训练中展示了竞争力的K400动作识别结果
- 中间层最佳表征的现象在所有模型尺寸和任务中一致存在

## 亮点与洞察
- 统一了图像和视频的训练格式，一个模型覆盖多种下游任务
- RoPE的分辨率适配能力使得低分辨率预训练+高分辨率微调成为可能，大幅降低训练成本
- 首次系统研究了视觉自回归模型的计算最优缩放行为
- 物体永续性（CATER）任务上超越专门设计的方法，说明自回归预训练隐式学到了长期时序推理
- 机器人操控实验展示了生成式预训练在具身智能中的潜力

## 局限与展望
- 判别式方法在多数识别任务上仍大幅领先（如DINO 80.1% vs Toto 75.3%）
- tokenizer质量成为性能瓶颈，dVAE重建质量有限
- 视频帧间冗余降低了训练数据的信息密度，可能需要更智能的帧采样策略
- 仅在ImageNet分类上验证设计选择，可能不适用于密集预测等其他任务
- 缩放效率比语言模型慢，需要更大的计算投入才能获得同等改进

## 相关工作与启发
- **vs iGPT**: Toto用token代替像素，实现了更好的缩放性（1.1B vs 6.8B达到相似性能）
- **vs MAE**: MAE是掩码自编码不是自回归，编码器-解码器结构使最优层在编码器顶部；Toto在decoder-only模型中中间层最佳
- **vs AIM**: AIM使用CLIP筛选数据，间接引入了监督信号；Toto完全无监督
- **vs DINO**: DINO在识别上更强，但Toto作为生成模型具备更广的通用性

## 评分
- 新颖性: ⭐⭐⭐ 方法框架相对直接（视频token+因果Transformer），核心贡献在于系统性的实证研究
- 实验充分度: ⭐⭐⭐⭐⭐ 涵盖图像识别、视频分类、动作预测、跟踪、物体永续性、机器人操控6大任务，设计选择消融详尽
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，实验设计严谨，但方法新颖性叙述有限
- 价值: ⭐⭐⭐⭐ 为视觉自回归模型提供了重要的实证参考，缩放规律分析对社区有指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] An Empirical Study on How Video-LLMs Answer Video Questions](../../CVPR2026/video_understanding/an_empirical_study_on_how_video-llms_answer_video_questions.md)
- [\[CVPR 2025\] VideoGEM: Training-Free Action Grounding in Videos](../../CVPR2025/video_understanding/videogem_training-free_action_grounding_in_videos.md)
- [\[ICCV 2025\] Beyond the Frame: Generating 360° Panoramic Videos from Perspective Videos](beyond_the_frame_generating_360deg_panoramic_videos_from_perspective_videos.md)
- [\[ICCV 2025\] XTrack: Multimodal Training Boosts RGB-X Video Object Trackers](xtrack_multimodal_training_boosts_rgb-x_video_object_trackers.md)
- [\[ICCV 2025\] Fine-grained Spatiotemporal Grounding on Egocentric Videos](fine-grained_spatiotemporal_grounding_on_egocentric_videos.md)

</div>

<!-- RELATED:END -->
