---
title: >-
  [论文解读] Linear Differential Vision Transformer: Learning Visual Contrasts via Pairwise Differentials
description: >-
  [NeurIPS 2025][图像生成][Transformer] 提出 Visual-Contrast Attention (VCA)，通过空间池化生成紧凑的正负视觉对比 token 并进行差分交互，将自注意力复杂度从 $O(N^2C)$ 降至 $O(NnC)$（$n \ll N$），同时在图像分类和生成任务上均获得显著提升。
tags:
  - "NeurIPS 2025"
  - "图像生成"
  - "Transformer"
  - "线性注意力"
  - "差分注意力"
  - "图像分类"
---

# Linear Differential Vision Transformer: Learning Visual Contrasts via Pairwise Differentials

**会议**: NeurIPS 2025  
**arXiv**: [2511.00833](https://arxiv.org/abs/2511.00833)  
**代码**: [https://github.com/LeapLabTHU/LinearDiff](https://github.com/LeapLabTHU/LinearDiff)  
**领域**: 视觉Transformer / 高效注意力  
**关键词**: Vision Transformer, 线性注意力, 差分注意力, 图像分类, 图像生成

## 一句话总结

提出 Visual-Contrast Attention (VCA)，通过空间池化生成紧凑的正负视觉对比 token 并进行差分交互，将自注意力复杂度从 $O(N^2C)$ 降至 $O(NnC)$（$n \ll N$），同时在图像分类和生成任务上均获得显著提升。

## 研究背景与动机

Vision Transformer 中的多头自注意力(MHSA)对每对 token 计算二次查询-键交互，大量计算花费在视觉上弱或冗余的相关性上。现有优化方案主要有两条路线：

**限制感受野**: 滑动窗口(Swin)、膨胀注意力(DiNAT)等，但牺牲了长程依赖

**低秩近似**: Linformer、Performer 等保持全局视野，但将所有相关性视为同等重要

语言模型中的差分注意力(Differential Attention)通过两个注意力图相减来突出区分性信号，但仍是二次复杂度且忽略了图像特有的冗余结构。

本文的核心前提是：**先压缩密集查询场，再执行昂贵的比较**。自然图像的空间平滑性意味着相邻块通常携带几乎相同的信息，可以先将查询集缩减为少量原型再进行匹配。

## 方法详解

### 整体框架

VCA 是 MHSA 的即插即用替换模块，分为两个阶段：Stage I 进行全局对比(视觉对比 token 与所有 key/value 交互)，Stage II 进行逐 patch 差分注意力(原始查询与对比图交互)。

### 关键设计

1. **视觉对比 Token 生成**: 将查询矩阵 $\mathbf{q}^{(m)} \in \mathbb{R}^{N \times d}$ 重塑为 2D 空间布局 $H \times W \times d$，通过平均池化降采样到 $h \times w$ 的粗粒度网格，得到 $n = h \cdot w$ 个视觉对比 token（如 $8 \times 8 = 64$）。然后添加两组独立的可学习位置嵌入 $\mathbf{e}^+$ 和 $\mathbf{e}^-$，分裂为正流和负流。这种设计精妙之处在于正负流共享相同的池化内容但通过不同位置嵌入解耦了互补的相关性。

2. **Stage I — 全局对比**: 正负视觉对比 token 分别与所有 key 和 value 进行注意力计算，得到 $\hat{\mathbf{v}}_+^{(m)}$ 和 $\hat{\mathbf{v}}_-^{(m)}$（均为 $n \times d$），然后执行差分操作 + RMSNorm：$\hat{\mathbf{v}}^{(m)} = (1-\lambda_{init}^{(1)}) \text{RMSNorm}(\hat{\mathbf{v}}_+^{(m)} - \lambda^{(1)} \hat{\mathbf{v}}_-^{(m)})$。这一步将全场景压缩为突出差异的对比图，复杂度为 $O(Nnd)$。

3. **Stage II — 逐 Patch 差分注意力**: 原始 $N$ 个查询与正负视觉对比 token 分别计算注意力图（$N \times n$），取差分后加权对比图中的值。因为对比图仅有 $n$ 个 token，三个矩阵乘法的复杂度都与 $nN$ 成正比。最终输出经 RMSNorm 和 $(1-\lambda_{init}^{(2)})$ 缩放后拼接各头。

### 损失函数 / 训练策略

- **图像分类**: 与 baseline 完全相同的训练设置，AdamW 优化器训练 300 epoch，余弦学习率衰减，配合 RandAugment、Mixup、CutMix、Random Erasing 等数据增强
- **图像生成**: 遵循 DiT/SiT 原始配方，256 batch size 训练 400K iterations，恒定学习率 $10^{-4}$，EMA 衰减 0.9999，仅使用随机水平翻转
- VCA 仅增加不到 0.3M 参数（DeiT-Tiny 上），不增加额外 FLOPs

## 实验关键数据

### 图像分类 (ImageNet-1K)

| 骨干网络 | 参数量 | FLOPs | Baseline Top-1 | +VCA Top-1 | 提升 |
|---------|--------|-------|----------------|------------|------|
| DeiT-Tiny | 5.7M→6.0M | 1.2G | 72.2% | **75.6%** | +3.4 |
| DeiT-Small | 22.1M→22.6M | 4.6G | 79.8% | **80.7%** | +0.9 |
| PVT-Tiny | 13.2M→11.6M | 1.9G→2.0G | 75.1% | **78.2%** | +3.1 |
| Swin-Tiny | 28.9M→28.5M | 4.5G→4.6G | 81.3% | **82.3%** | +1.0 |
| CSwin-Tiny | 20.5M→20.4M | 4.3G | 82.7% | **83.3%** | +0.6 |

### 图像生成 (ImageNet-1K 256×256, FID-50K↓)

| 模型 | Baseline FID | +VCA FID | 改善 |
|------|-------------|----------|------|
| DiT-S/2 | 67.2 | **62.3** | ↓4.9 |
| DiT-S/4 | 97.9 | **92.7** | ↓5.2 |
| DiT-B/2 | 42.9 | **38.9** | ↓4.0 |
| SiT-S/2 | 57.3 | **53.0** | ↓4.3 |
| SiT-B/2 | 35.3 | **32.7** | ↓2.6 |

### 消融实验

| 配置 | DeiT-Tiny Top-1 | DiT-S/2 FID | 说明 |
|------|-----------------|-------------|------|
| 仅 Stage I | 75.4 | 64.6 | 全局对比有贡献 |
| 仅 Stage II | 75.5 | 64.3 | 逐 patch 差分有贡献 |
| 原始 Diff Attention (两阶段) | 75.1 | 63.9 | 未利用视觉特性 |
| **VCA (两阶段)** | **75.6** | **62.3** | 两阶段协同最佳 |
| 正负流均用 Emb. | 75.1 | 63.7 | 缺少图像信息 |
| 正负流均用 Pool+Emb. | **75.6** | **62.3** | 池化+嵌入最优 |

### 关键发现

- 两个阶段的贡献近乎线性叠加，说明全局对比和逐 patch 差分捕获了互补信息
- VCA 优于直接应用语言版差分注意力(+0.5% 和 -1.8 FID)，说明视觉特定设计的必要性
- 空间池化提供低方差全局线索，双位置嵌入对解耦互补相关性不可或缺
- 对 Small 模型增益最大(DeiT-T +3.4%)，对 Base 模型仍有 +0.4~1.0% 收益
- 同时适用于 diffusion 和 flow-based 两种生成范式

## 亮点与洞察

- **核心洞察**: 不应仅将注意力视为相似性度量，还应作为显式对比的舞台——关注"什么使一个区域与另一个区域不同"比关注"它们有多相似"更有判别力
- 将线性复杂度和差分注意力巧妙结合：先池化压缩再差分，一石二鸟
- 架构无关性是巨大优势，在 plain ViT、层级 ViT(PVT/Swin/CSwin)以及生成模型(DiT/SiT)上均一致有效
- 参数开销极小(< 0.3M)且不增加 FLOPs，工程部署友好
- 复杂度降低比例 $N/n$，对 $256^2$ 图像 patch 16 来说约 256 倍

## 局限与展望

- 任务无关的平均池化可能遗漏边缘丰富的细节
- 对小图像，微注意力的额外开销可能抵消速度增益
- 尚未探索视频、3D 或语言任务的扩展
- 池化策略固定(平均池化)，自适应/可学习池化可能进一步提升
- 未报告实际推理速度(仅有 FLOPs 分析)

## 相关工作与启发

- Differential Transformer (Ye et al., 2024) 在语言建模中引入差分注意力的思想是直接灵感来源
- Agent Attention (Han et al., 2024) 等基于中介 token 的线性化方法构成了技术基础
- 与 Efficient DiT (Pu et al., 2024) 中的注意力中介思想一脉相承
- 对未来高效视觉架构设计有重要参考价值：应更多考虑"对比"而非"相似"

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] LaVin-DiT: Large Vision Diffusion Transformer](../../CVPR2025/image_generation/lavin-dit_large_vision_diffusion_transformer.md)
- [\[ICCV 2025\] LiT: Delving into a Simple Linear Diffusion Transformer for Image Generation](../../ICCV2025/image_generation/lit_delving_into_a_simple_linear_diffusion_transformer_for_image_generation.md)
- [\[NeurIPS 2025\] SparseDiT: Token Sparsification for Efficient Diffusion Transformer](sparsedit_token_sparsification_for_efficient_diffusion_transformer.md)
- [\[NeurIPS 2025\] On the Emergence of Linear Analogies in Word Embeddings](on_the_emergence_of_linear_analogies_in_word_embeddings.md)
- [\[NeurIPS 2025\] Rare Text Semantics Were Always There in Your Diffusion Transformer](rare_text_semantics_were_always_there_in_your_diffusion_transformer.md)

</div>

<!-- RELATED:END -->
