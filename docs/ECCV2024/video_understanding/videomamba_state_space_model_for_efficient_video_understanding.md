---
title: >-
  [论文解读] VideoMamba: State Space Model for Efficient Video Understanding
description: >-
  [ECCV 2024][视频理解][State Space Model] 将 Mamba 的选择性状态空间模型创新性地适配到视频领域，提出纯 SSM 架构的 VideoMamba，以线性复杂度实现高效的时空上下文建模，在短视频和长视频理解任务上均展现出优越性能。 领域现状：视频理解的核心在于掌握时空表征…
tags:
  - "ECCV 2024"
  - "视频理解"
  - "State Space Model"
  - "Mamba"
  - "线性复杂度"
  - "长视频建模"
---

# VideoMamba: State Space Model for Efficient Video Understanding

**会议**: ECCV 2024  
**arXiv**: [2403.06977](https://arxiv.org/abs/2403.06977)  
**代码**: [https://github.com/OpenGVLab/VideoMamba](https://github.com/OpenGVLab/VideoMamba)  
**领域**: 视频理解  
**关键词**: State Space Model, Mamba, 视频理解, 线性复杂度, 长视频建模

## 一句话总结

将 Mamba 的选择性状态空间模型创新性地适配到视频领域，提出纯 SSM 架构的 VideoMamba，以线性复杂度实现高效的时空上下文建模，在短视频和长视频理解任务上均展现出优越性能。

## 研究背景与动机

**领域现状**：视频理解的核心在于掌握时空表征，面临两大挑战——短视频片段中的大量时空冗余，以及长上下文中的复杂时空依赖关系。

**现有痛点**：3D CNN（如 SlowFast）擅长局部建模但无法捕捉长程依赖；Video Transformer（如 TimeSformer、ViViT）能建模远程依赖但自注意力的二次复杂度导致计算成本极高；UniFormer 试图结合两者优势，但在长视频上仍力不从心。

**核心矛盾**：高效率与强建模能力难以兼得——处理 64 帧视频时，TimeSformer 的吞吐量和显存消耗远不可接受。

**本文目标**：设计一种兼具线性复杂度和强时空动态建模能力的视频理解架构。

**切入角度**：利用 NLP 领域新兴的 Mamba（选择性 SSM），将其双向扩展到 3D 视频序列。

**核心 idea**：以 vanilla ViT 的简洁架构为基础，用双向 Mamba block 替代自注意力层，构建纯 SSM 视频模型。

## 方法详解

### 整体框架

VideoMamba 严格遵循 vanilla ViT 的架构设计。输入视频 $\mathbf{X}^v \in \mathbb{R}^{3 \times T \times H \times W}$ 首先通过 3D 卷积（kernel $1 \times 16 \times 16$）投射为 $L = t \times h \times w$ 个不重叠的时空 patch token $\mathbf{X}^p \in \mathbb{R}^{L \times C}$。随后加上可学习的空间位置编码 $\mathbf{p}_s$ 和时间位置编码 $\mathbf{p}_t$，并在序列前端添加 [CLS] token。token 序列依次通过 $L$ 层堆叠的 B-Mamba（双向 Mamba）block，最终 [CLS] token 经归一化和线性层完成分类。

### 关键设计

1. **选择性状态空间模型（S6）**：核心算子基于 Mamba 的选择性扫描机制。与传统线性时不变 SSM 不同，S6 的参数 $\mathbf{B}$、$\mathbf{C}$、$\boldsymbol{\Delta}$ 均由输入数据动态生成，具备上下文感知能力。连续系统通过零阶保持（ZOH）离散化：

    $\bar{\mathbf{A}} = \exp(\boldsymbol{\Delta} \mathbf{A}), \quad \bar{\mathbf{B}} = (\boldsymbol{\Delta} \mathbf{A})^{-1}(\exp(\boldsymbol{\Delta} \mathbf{A}) - \mathbf{I}) \cdot \boldsymbol{\Delta} \mathbf{B}$

    $h_t = \bar{\mathbf{A}} h_{t-1} + \bar{\mathbf{B}} x_t, \quad y_t = \mathbf{C} h_t$

   这种数据依赖的参数化使模型能够自适应地调节权重，实现内容感知的上下文建模，同时保持线性复杂度 $\mathcal{O}(n_h \cdot n_w \cdot n_t)$。

2. **双向 Mamba Block（B-Mamba）**：原始 Mamba 为单向 1D 序列设计，缺乏空间感知能力。借鉴 Vision Mamba，VideoMamba 采用双向 SSM 同时处理前向和后向序列，增强空间感知。每个 B-Mamba block 包含：线性投影（$384 \to 768$）→ 1D 卷积 → 双向 ST-SSM → 线性投影（$768 \to 384$）。

3. **时空扫描策略**：将 2D 双向扫描扩展到 3D 视频，探索了四种策略：

    - **Spatial-First（空间优先）**：按空间位置组织 token，逐帧堆叠——最简洁且最有效
    - **Temporal-First（时间优先）**：按帧排列时间 token，沿空间维度堆叠
    - **Spatiotemporal（时空混合）**：两种的混合，v1 各执行一半，v2 全部执行（2× 计算量）
   
   消融实验表明 **Spatial-First 双向扫描** 效果最优，因其能无缝利用 2D 预训练知识。

4. **自蒸馏策略（Self-Distillation）**：大规模 Mamba 模型（如 VideoMamba-B）在训练时容易过拟合。解决方案是使用训练好的小模型（如 VideoMamba-S）作为 teacher，通过 L2 loss 对齐最终特征图来指导大模型（student）训练。此策略以极小的额外计算开销实现了更好的收敛和可扩展性。

5. **掩码建模（Masked Modeling）**：为增强时间敏感性，借鉴 UMT 的掩码对齐方法。针对 B-Mamba 中 1D 卷积偏好连续 token 的特性，设计了 **Row Masking** 策略（clip-row 和 frame-row），并引入 **Attention Masking** 保留相邻 token 的有意义邻接关系。由于 SSM 与 Transformer 架构差异，仅对齐最终输出层效果最佳。

### 损失函数 / 训练策略

- **监督训练**：使用 ImageNet-1K 预训练权重初始化，采用 AdamW 优化器 + cosine 学习率调度；K400 上训练 50 个 epoch，学习率线性缩放 $2e^{-4} \cdot \frac{batchsize}{256}$
- **自蒸馏**：使用 L2 loss 对齐 teacher 和 student 最终特征图
- **自监督（UMT 风格）**：采用 CLIP-ViT-B 蒸馏 VideoMamba-M，800 epoch；最优掩码比例 80%，配合更强的正则化（droppath=0.4）
- **多模态预训练**：使用 WebVid-2M + CC3M，四个目标：视觉-文本对比学习、匹配、MLM 和无掩码 token 对齐

## 实验关键数据

### 主实验

| 数据集 | 指标 | 本文 (VideoMamba-M) | SOTA | 提升/对比 |
|--------|------|------|------|------|
| ImageNet-1K | Top-1 Acc | 84.0% (576²) | DeiT-B 83.1% | +0.9% (更少参数) |
| Kinetics-400 (监督) | Top-1 Acc | 83.3% (64f, 384²) | ViViT-L 81.3% | +2.0% |
| SthSthV2 (监督) | Top-1 Acc | 68.4% (16f, 288²) | ViViT-L 65.4% | +3.0% |
| K400 (自监督) | Top-1 Acc | 85.0% (64f) | UMT-B 85.7% | 接近 (参数更少) |
| Breakfast (长视频) | Top-1 Acc | 97.9% | ViS4mer 88.2% | +9.7% |
| COIN (长视频) | Top-1 Acc | 90.4% | Distant Sup. 90.0% | +0.4% |

### 消融实验

| 配置 | SSV2 Top-1 | 说明 |
|------|------|------|
| Spatial-First 扫描 | 65.1% | 最优扫描策略 |
| Temporal-First 扫描 | 62.4% | 最差，丢失空间信息 |
| ST-Bidirectional v2 | 64.2% | 时空混合，2× 计算 |
| Attention Masking | 68.5% | 最优掩码类型 |
| Random Masking | 67.4% | 基线掩码 |
| 掩码比例 80% | 68.5% | 最优比例 |
| Droppath 0.4 | 68.5% | 自监督最优正则强度 |

### 效率对比

| 模型 | 帧数 | 速度 (相对) | 显存 (相对) |
|------|------|------|------|
| VideoMamba | 64f | 6× 快于 TimeSformer | 40× 少于 TimeSformer |
| VideoMamba-Ti | 16f, 224² | 17 GFLOPs | 7M 参数 |
| TimeSformer-L | 96f, 224² | 2380 GFLOPs | 121M 参数 |

### 关键发现

- Spatial-First 扫描最有效，因为能无缝利用 2D 预训练知识
- 自蒸馏能有效解决大规模 Mamba 的过拟合问题，且额外计算开销极小
- 仅对齐最终输出层的蒸馏效果最佳（因 SSM 和 Transformer 架构差异）
- 掩码建模同样适用于 Mamba，且 Row Masking 配合 1D 卷积效果最好
- 长视频端到端训练显著优于基于预提取特征的方法

## 亮点与洞察

1. **架构极简**：完全保持 ViT 的 isotropic 设计，无下采样层、无额外 depthwise conv，证明纯 SSM 架构在视频领域的可行性
2. **四维度全面验证**：可扩展性、短视频敏感性、长视频优越性、多模态兼容性——评估角度非常全面
3. **效率优势惊人**：64 帧视频上比 TimeSformer 快 6×、省显存 40×，使长视频端到端训练成为可能
4. **自蒸馏解决过拟合**：优雅地解决了 SSM 模型 scale up 时的过拟合问题，无需大规模数据集预训练

## 局限与展望

1. 自监督模式下 VideoMamba 与 UMT 的对齐因架构不一致仍有差距（82.0% vs 85.7%），跨架构蒸馏值得深入研究
2. 更大模型（VideoMamba-B）即使有自蒸馏仍被排除，可扩展性的上限尚不明确
3. 多模态预训练实验规模较小（WebVid-2M + CC3M），在更大规模数据上的表现未知
4. 纯 SSM 架构在需要显式局部交互的任务上可能不如 UniFormer 等混合架构

## 相关工作与启发

- **Vision Mamba / VMamba**：VideoMamba 基于 Vim 构建，但去除了 middle CLS token 和 RoPE，在 ImageNet 上提升 +0.8%
- **UniFormer**：CNN+Attention 混合架构的代表，VideoMamba 证明纯 SSM 也能达到类似效果
- **ViS4mer**：早期将 S4 用于长视频的尝试（feature-based），VideoMamba 通过端到端训练大幅超越

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个将 Mamba 全面适配到视频理解的工作，自蒸馏策略巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖 ImageNet、K400、SSV2、Breakfast、COIN、LVU 及多模态检索，消融极其详尽
- 写作质量: ⭐⭐⭐⭐ 四大能力的组织方式清晰，但表格密度较高
- 价值: ⭐⭐⭐⭐⭐ 开创性工作，为视频 SSM 奠定基础，代码完全开源

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] VideoMamba: Spatio-Temporal Selective State Space Model](videomamba_spatio-temporal_selective_state_space_model.md)
- [\[NeurIPS 2025\] PASS: Path-Selective State Space Model for Event-Based Recognition](../../NeurIPS2025/video_understanding/pass_path-selective_state_space_model_for_event-based_recognition.md)
- [\[CVPR 2025\] MambaVLT: Time-Evolving Multimodal State Space Model for Vision-Language Tracking](../../CVPR2025/video_understanding/mambavlt_time-evolving_multimodal_state_space_model_for_vision-language_tracking.md)
- [\[ECCV 2024\] PiTe: Pixel-Temporal Alignment for Large Video-Language Model](pite_pixel-temporal_alignment_for_large_video-language_model.md)
- [\[AAAI 2026\] MambaMia: State-Space Hierarchical Compression for Hour-Long Video Understanding in Large Multimodal Models](../../AAAI2026/video_understanding/state-space_hierarchical_compression_with_gated_attention_an.md)

</div>

<!-- RELATED:END -->
