---
title: >-
  [论文解读] HMAR: Efficient Hierarchical Masked Auto-Regressive Image Generation
description: >-
  [CVPR 2025][图像生成][自回归图像生成] HMAR 将 VAR 的 next-scale 预测重构为 Markov 过程（仅依赖前一尺度的累积重建而非所有前序尺度），并在每个尺度内引入多步掩码生成来消除条件独立假设，配合自定义 IO-aware 块稀疏注意力核，在 ImageNet 上匹配或超越 VAR/DiT 质量的同时实现训练 2.5× 加速和推理 3× 内存缩减。
tags:
  - "CVPR 2025"
  - "图像生成"
  - "自回归图像生成"
  - "Next-Scale Prediction"
  - "Masked Prediction"
  - "Markov 过程"
  - "高效注意力"
  - "损失重加权"
---

# HMAR: Efficient Hierarchical Masked Auto-Regressive Image Generation

**会议**: CVPR 2025  
**arXiv**: [2506.04421](https://arxiv.org/abs/2506.04421)  
**代码**: 无  
**领域**: 图像生成 / 自回归生成  
**关键词**: 自回归图像生成, Next-Scale Prediction, Masked Prediction, Markov 过程, 高效注意力, 损失重加权

## 一句话总结

HMAR 将 VAR 的 next-scale 预测重构为 Markov 过程（仅依赖前一尺度的累积重建而非所有前序尺度），并在每个尺度内引入多步掩码生成来消除条件独立假设，配合自定义 IO-aware 块稀疏注意力核，在 ImageNet 上匹配或超越 VAR/DiT 质量的同时实现训练 2.5× 加速和推理 3× 内存缩减。

## 研究背景与动机

**领域现状**：Visual Auto-Regressive modeling (VAR) 通过 next-scale prediction 范式弥合了自回归模型与扩散模型在速度和质量上的差距。VAR 将图像分解为 K 个分辨率尺度，每步生成一个更高分辨率的尺度，条件是所有前序尺度的 token。

**现有痛点**：(1) **质量瓶颈**：VAR 在每个尺度内一步并行采样所有 token，隐式假设同一尺度内 token 条件独立，导致"过平滑"和跨尺度误差累积；(2) **效率问题**：条件依赖所有前序尺度使序列长度超线性增长（256×256 时比 next-token 长 5.84×），且 FlashAttention 不支持 VAR 的 block-causal 注意力模式；(3) **灵活性不足**：推理步数在训练时固定，增加步数需重新训练。

**核心矛盾**：VAR 的质量-效率-灵活性三方面都有明显改进空间——条件独立假设损害质量，长序列拖累效率，固定步数限制灵活性。

**本文切入角度**：注意到 VAR 编码中的累积重建 $\tilde{x}_{1:k}$ 已包含所有前 $k$ 个尺度的信息（类似拉普拉斯金字塔），因此可将 next-scale 预测重构为 Markov 过程 $p(r_k | \tilde{x}_{1:k-1})$。由此得到 block-diagonal 注意力模式（比 block-causal 稀疏 5×），并在尺度内引入类 MaskGIT 的多步掩码生成来建模 token 间依赖。

## 方法详解

### 整体框架

HMAR 由两个子模块组成：(1) **Markovian Next-Scale Prediction 模块**：将 VAR 的全历史条件改为仅依赖前一尺度的累积重建，使用 IO-aware 块对角注意力核加速训练；(2) **Intra-Scale Masked Refinement 模块**：在每个尺度内用多步掩码生成消除条件独立假设，可控平衡质量与速度。两模块分步训练。

### 关键设计

1. **Markovian Next-Scale Prediction**:

    - 功能：将 next-scale 预测的序列长度从超线性降为线性，实现块对角稀疏注意力
    - 核心思路：利用 VQ-VAE 残差编码的性质——累积重建 $\tilde{x}_{1:k} = \sum_{j=1}^{k} \tilde{x}_j$ 包含前 $k$ 个尺度的全部信息。因此 $p(r_k | r_1,...,r_{k-1}) = p(r_k | \tilde{x}_{1:k-1})$，将生成重构为 Markov 过程。实践中用插值函数将 $\tilde{x}_{1:k-1}$ 缩放到 $H_{k-1} \times W_{k-1}$ 作为条件，注意力模式从 block-causal 变为 block-diagonal（稀疏度提升 5×）
    - 设计动机：通过注意力分析（Fig. 9），VAR 中大部分注意力确实集中在前一尺度上，验证了 Markov 假设的合理性。推理时无需 KV cache，直接降低 3× 内存

2. **层次化多步掩码生成 (Hierarchical Multi-Step Masked Generation)**:

    - 功能：在每个尺度内建模 token 间依赖，消除 VAR 的条件独立假设
    - 核心思路：在每个尺度 $k$，初始 next-scale 预测得到 $r_k^0$（VAR 的单步结果），然后用 $M_k$ 步掩码生成迭代精炼——每步随机遮蔽一部分 token 并基于未遮蔽 token + 前一尺度的累积重建重新预测。$M_k=0$ 退化为 VAR，$M_k=H_k \times W_k$ 退化为 next-token prediction。训练时对微调阶段均匀采样掩码率 $\gamma \sim \mathcal{U}(0,1)$，推理时在粗尺度多步提升 FID、细尺度多步提升感知质量
    - 设计动机：VAR 的并行生成假设同一尺度 token 条件独立，这在实践中导致过平滑和错误累积（Fig. 17）。掩码生成是质量和速度之间的可控权衡

3. **多尺度损失重加权 (Multi-Scale Loss Reweighting)**:

    - 功能：平衡不同分辨率尺度的训练贡献
    - 核心思路：VAR 的均匀平均损失导致最细尺度贡献 256 倍于最粗尺度。HMAR 引入尺度权重 $w(k)$，$\sum w(k) = 1$。实验发现每尺度的学习难度近似服从对数正态分布（Fig. 12），因此采用对数正态加权函数作为 $w(k)$，使模型容量分配与学习难度分布匹配
    - 设计动机：早期粗尺度错误会累积传播到所有后续尺度（Fig. 17），且不同尺度贡献的 token 数差异极大，均匀权重不合理

### 损失函数 / 训练策略

- **阶段 1（Next-Scale）**：带 IO-aware 窗口注意力的 cross-entropy 损失 + 对数正态损失重加权
- **阶段 2（Masked Refinement）**：添加掩码预测头，用 $\mathcal{L}_{mask} = \sum_k \mathcal{L}(\gamma r_k | \bar{\gamma} r_k)$ 微调
- 使用 VAR 预训练的多尺度 VQ-VAE tokenizer
- K=10 尺度（1×1 到 16×16），与 VAR 一致
- 推理用 top-k top-p 采样，默认 14 步（10 步 next-scale + 每尺度少量 mask 步）

## 实验关键数据

### 主实验

ImageNet 256×256（cfg=无明确说明）：

| 方法 | 类型 | FID↓ | IS↑ | Params | Steps |
|------|------|------|------|--------|-------|
| DiT-XL/2 | Diffusion | 2.27 | 278.2 | 675M | 250 |
| VAR-d16 | AR | 3.36 | 277.8 | 310M | 10 |
| VAR-d24 | AR | 2.15 | 312.4 | 1.0B | 10 |
| VAR-d30 | AR | 1.95 | 303.6 | 2.0B | 10 |
| **HMAR-d16** | **Hybrid** | **3.01** | **288.6** | 465M | 14 |
| **HMAR-d24** | **Hybrid** | **2.10** | **324.3** | 1.3B | 14 |
| **HMAR-d30** | **Hybrid** | **1.95** | **334.5** | 2.4B | 14 |

ImageNet 512×512：

| 方法 | FID↓ | IS↑ | Params |
|------|------|------|--------|
| DiT-XL/2 | 3.04 | 240.8 | 675M |
| VAR-d36 | 2.63 | 303.2 | - |
| **HMAR-d24** | 匹配或超越 | 更高 IS | - |

### 效率对比

| 指标 | HMAR vs VAR |
|------|-------------|
| 训练速度 | **2.5× 更快** |
| 推理速度 | **1.75× 更快** |
| 推理内存 | **3× 更低** |
| 注意力计算 | 10× 更快（IO-aware kernel） |

### 消融实验

损失重加权策略（d16, 256×256）：

| 加权策略 | FID↓ | IS↑ |
|---------|------|------|
| 均匀 (VAR) | 3.36 | 277.8 |
| 线性 | ~3.2 | ~280 |
| **对数正态** | **3.01** | **288.6** |

### 关键发现

- HMAR-d30 在 ImageNet 256×256 上 FID 1.95 匹配 VAR-d30，但 IS 从 303.6 提升至 334.5（+31 点），图像质量有显著感知提升
- Markov 重构使训练序列稀疏度提升 5×（256×256 时），IO-aware 核使注意力计算加速 10×
- 推理无需 KV cache → 3× 内存缩减，使大规模模型和高分辨率推理可行
- 掩码步数可在推理时灵活调整：粗尺度多步改善全局结构（FID↓），细尺度多步改善细节（IS↑）
- 对数正态损失加权比均匀加权 FID 降低约 0.35，IS 提升约 11
- HMAR 可零样本应用于 inpainting、outpainting 和类别条件编辑，VAR 无此能力

## 亮点与洞察

1. **Markov 等价的精彩推导**：利用残差量化的数学性质证明 $p(r_k|r_{<k}) = p(r_k|\tilde{x}_{1:k-1})$，类比拉普拉斯/高斯金字塔，理论简洁且实践高效
2. **质量-效率-灵活性三方面同时改进**：通常三者互为 trade-off，HMAR 通过 Markov 改写和掩码生成同时在三方面胜出，是罕见的 Pareto 改进
3. **可定制的采样调度**：掩码步数在不同尺度可独立调整且无需重训练，为质量 vs 速度的权衡提供极大灵活性
4. **自定义 IO-aware GPU 核**：工程贡献同样重要——Triton 实现的块稀疏注意力核使论文的理论稀疏优势在实践中兑现

## 局限与展望

- 参数量比对应 VAR 模型大约 30-50%（HMAR-d16 465M vs VAR-d16 310M），因为增加了掩码预测头
- 目前仅在 ImageNet 类别条件生成上验证，缺少文本条件生成（text-to-image）实验
- Markov 假设虽然在注意力分析中得到验证，但在极端情况下（如前序尺度包含关键全局结构）可能丢失信息
- 两阶段训练（next-scale + mask finetune）增加了训练管线复杂度

## 相关工作与启发

- HMAR 是 VAR 和 MaskGIT 的层次化融合——VAR 提供跨尺度因果性，MaskGIT 提供尺度内非因果精炼
- 与 HART（另一 VAR 改进）的对比：HART 用连续值扩散做尺度内精炼，HMAR 用离散掩码生成，后者更高效
- 对视频 VAR 的启发：Markov 技术可应用于视频帧的时序尺度，降低长视频生成的内存开销
- 损失重加权策略可推广到任何多尺度/多阶段生成模型

## 评分

⭐⭐⭐⭐⭐ (5/5)

- **创新性** ⭐⭐⭐⭐⭐：Markov 等价推导 + 掩码精炼 + IO-aware 核，三个贡献环环相扣
- **实验充分性** ⭐⭐⭐⭐⭐：质量/效率/灵活性三维度评估，消融充分，与 VAR/DiT/MaskGIT 多类基线对比
- **清晰度** ⭐⭐⭐⭐⭐：理论推导简洁，实验展示清晰，整体结构优秀
- **实用价值** ⭐⭐⭐⭐⭐：比 VAR 更快更省内存更灵活且质量不降，是直接可替换的升级方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Collaborative Decoding Makes Visual Auto-Regressive Modeling Efficient](collaborative_decoding_makes_visual_auto-regressive_modeling_efficient.md)
- [\[CVPR 2025\] MMAR: Towards Lossless Multi-Modal Auto-Regressive Probabilistic Modeling](mmar_towards_lossless_multi-modal_auto-regressive_probabilistic_modeling.md)
- [\[CVPR 2025\] Hierarchical Flow Diffusion for Efficient Frame Interpolation](hierarchical_flow_diffusion_for_efficient_frame_interpolation.md)
- [\[NeurIPS 2025\] Distilled Decoding 2: One-step Sampling of Image Auto-regressive Models with Conditional Score Distillation](../../NeurIPS2025/image_generation/distilled_decoding_2_onestep_sampling_of_image_autoregressiv.md)
- [\[ICCV 2025\] DC-AR: Efficient Masked Autoregressive Image Generation with Deep Compression Hybrid Tokenizer](../../ICCV2025/image_generation/dc-ar_efficient_masked_autoregressive_image_generation_with_deep_compression_hyb.md)

</div>

<!-- RELATED:END -->
