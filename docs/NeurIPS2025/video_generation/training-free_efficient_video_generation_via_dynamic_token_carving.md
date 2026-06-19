---
title: >-
  [论文解读] Training-Free Efficient Video Generation via Dynamic Token Carving
description: >-
  [NeurIPS 2025][视频生成][免训练加速] 本文提出 Jenga，一种免训练的视频 DiT 推理加速方案，通过动态块注意力裁剪（基于 3D 空间填充曲线重排 token 后进行稀疏 KV block 选择）和渐进分辨率策略（从低分辨率逐步提升）正交结合，在 HunyuanVideo 上实现 8.83 倍加速且 VBench 仅下降 0.01%。
tags:
  - "NeurIPS 2025"
  - "视频生成"
  - "免训练加速"
  - "稀疏注意力"
  - "3D空间填充曲线"
  - "渐进分辨率"
  - "Transformer"
---

# Training-Free Efficient Video Generation via Dynamic Token Carving

**会议**: NeurIPS 2025  
**arXiv**: [2505.16864](https://arxiv.org/abs/2505.16864)  
**代码**: [https://github.com/dvlab-research/Jenga](https://github.com/dvlab-research/Jenga)  
**领域**: 扩散模型 / 视频生成加速  
**关键词**: 免训练加速, 稀疏注意力, 3D空间填充曲线, 渐进分辨率, 视频扩散Transformer

## 一句话总结

本文提出 Jenga，一种免训练的视频 DiT 推理加速方案，通过动态块注意力裁剪（基于 3D 空间填充曲线重排 token 后进行稀疏 KV block 选择）和渐进分辨率策略（从低分辨率逐步提升）正交结合，在 HunyuanVideo 上实现 8.83 倍加速且 VBench 仅下降 0.01%。

## 研究背景与动机

**领域现状**：视频扩散 Transformer（DiT）如 HunyuanVideo、Wan2.1 等已能生成高质量视频，但推理极其缓慢——HunyuanVideo 在单张 H800 上生成 5 秒 720P 视频需约 27 分钟，严重制约实际应用。

**现有痛点**：推理瓶颈源自两个正交因素：(1) 自注意力的 $O(N^2)$ 复杂度——720P视频约 115K token，注意力占总计算 77.8%；(2) 扩散多步采样——50步去噪引入 50 倍计算开销。现有加速方案要么只解决其一（如 STA/CLEAR 做稀疏注意力但仅 1.5-2 倍加速，TeaCache 做步跳过但不减单步计算），要么需要额外训练（步蒸馏损失质量且训练昂贵）。

**核心矛盾**：现有稀疏注意力方法使用固定的空间-时间局部模式，忽略了不同输入、不同层、不同 head 的注意力分布差异，导致加速不够激进。同时，直接减少 token 数量（降分辨率）和减少 KV 交互（稀疏注意力）是两个独立的加速维度，应该联合利用。

**本文目标**：设计一个免训练、即插即用的推理 pipeline，同时大幅减少每步的 token 交互数量和总步数，在保持生成质量的前提下实现 5-10 倍加速。

**切入角度**：两个关键洞察：(1) 扩散去噪从低频到高频——早期步骤不需要高分辨率 latent；(2) 后期步骤不需要密集全注意力——视频 latent 存在大量冗余，极端稀疏（仅 1% KV blocks）也能保留细节。

**核心 idea**：像真实的积木游戏（Jenga）一样，在保持结构稳定的前提下最大化地移除冗余块——ProRes 减少 token 总量，AttenCarve 减少 token 交互，二者正交组合实现倍增加速。

## 方法详解

### 整体框架

Jenga 将原始 $T$ 步去噪过程分为 $S$ 个阶段。第一阶段从低分辨率开始生成内容结构，后续阶段逐步提升分辨率细化细节。在每个阶段中，使用 3D 空间填充曲线将视频 latent 重排为局部相关的 blocks，然后通过动态 top-K 选择仅计算最重要的 KV block 对，跳过冗余注意力计算。整个流程无需训练，可直接应用于任何视频 DiT。

### 关键设计

1. **块注意力裁剪 (AttenCarve)**:

    - 功能：将全注意力的 $O(N^2)$ 降低为 $O(N'N)$，$N'$ 为平均选中 token 数
    - 核心思路：首先用泛化 Hilbert 曲线（3D SFC）将视频 latent token 从 $z_{thw}$ 重排为 $z_{blk} = \mathcal{G}(z_{thw})$，使 1D 相邻 token 在 3D 空间也相邻。将重排后 token 均分为 $M$ 个 block（每个 $m=128$ token）。构建三种 block-wise 掩码的并集：(a) Importance Mask $\mathbf{B}_{top}$——用 block 均值计算注意力概率图 $\mathbf{R} = \text{softmax}(\hat{Q}\hat{K}^T/\sqrt{d_k})$，每个 query block 保留 top-$k$ 个 KV block，加上概率截断阈值 $p$ 保证全局信息不丢失；(b) Condition Mask——所有文本条件相关注意力全部保留；(c) Adjacency Mask——3D 空间 26-邻域的相邻 block 保留，消除块边界伪影。
    - 设计动机：与固定局部窗口（CLEAR/SVG）不同，动态 top-K 选择可以自适应不同 head 的注意力模式：浅层偏局部、深层偏语义、部分 head 全局聚合。概率截断约束专门保护这些全局 head。SFC 重排比线性分割更好保持局部性，减少所需 block 数。

2. **渐进分辨率 (ProRes)**:

    - 功能：减少早期去噪步骤的 token 总数，压缩 pipeline 级别计算
    - 核心思路：将 $T$ 步去噪分为 $S$ 阶段，从低分辨率 $R_1$ 渐进到目标分辨率 $R_S$。每阶段结束时，预测 clean latent $\hat{x}_0^s$，用 3D area interpolation 上采样到下一阶段分辨率，再加噪继续去噪：$x_{t-1} = (1-\sigma_t) \times \mathcal{U}(\hat{x}_0^s) + \sigma_t \tilde{\epsilon}$。引入 text-attention amplifier：在低分辨率阶段对视觉-文本注意力加偏置 $\beta = -\rho \log(\text{numel}(R_s)/\text{numel}(R_S))$，增强文本条件权重以防止低分辨率过度聚焦局部导致视野缩小（FOV degradation）。另外使用固定 23 步时间步跳过（与 TeaCache-fast 效果相当但无额外计算开销）。
    - 设计动机：扩散去噪的 coarse-to-fine 特性——早期建立内容结构、后期精细化细节——使得低分辨率起步完全合理。Text-attention amplifier 优雅地解决了低分辨率 → 窄视野的问题，通过增强全局文本条件让模型"假装"在高分辨率下生成。

### 损失函数 / 训练策略

完全免训练（training-free），所有组件即插即用。AttenCarve 使用 Triton 实现自定义稀疏注意力 kernel。支持多 GPU 并行（基于 xDiT），8 GPU 可进一步加速 6.28 倍。

## 实验关键数据

### 主实验

| 方法 | NFE | VBench↑ | VBench-Q↑ | VBench-S↑ | DiT时间 | 加速比 |
|---|---|---|---|---|---|---|
| HunyuanVideo 基线 | 50 | 82.74% | 85.21% | 72.84% | 1625s | 1.00× |
| CLEAR (r=32) | 50 | 82.68% | 86.06% | 69.17% | 1848s | 0.89× |
| MInference | 50 | 83.36% | 85.41% | 75.16% | 815s | 1.99× |
| SVG | 50 | 83.11% | 85.87% | 72.07% | 988s | 1.64× |
| AttenCarve (仅注意力) | 50 | 83.42% | 85.31% | 75.85% | 748s | 2.17× |
| Jenga-Base (1阶段) | 23 | **83.34%** | 85.19% | 75.92% | 347s | 4.68× |
| Jenga-Turbo (2阶段) | 24 | 83.07% | 84.47% | **77.48%** | 225s | 7.22× |
| Jenga-Flash (2阶段高稀疏) | 24 | 82.73% | 84.01% | 77.58% | 184s | **8.83×** |

| 模型/设置 | VBench | 延迟 | 加速比 |
|---|---|---|---|
| HunyuanVideo-I2V 基线 | 87.49% | 1499s | 1.00× |
| + Jenga | 87.75% | 338s | 4.43× |
| Wan2.1-1.3B 基线 | 83.28% | 115s | 1.00× |
| + Jenga | 82.68% | 24s | 4.79× |
| AccVideo (蒸馏模型) | 83.82% | 161s | 1.00× |
| + Jenga | 83.39% | 76s | 2.12× |
| HunyuanVideo 8GPU | 82.74% | 225s | 1.00× |
| + Jenga-Flash 8GPU | 82.73% | 39s | **5.77×** |

### 消融实验

| 配置 | VBench | 延迟 | 说明 |
|---|---|---|---|
| 线性 hwt 分割 | 82.82% | 229s | 有移位伪影，需更多 block |
| SFC 分割 | 83.07% | 225s | 更好的局部性，更少 block |
| 无邻接掩码 | 81.82% | 221s | 块边界出现网格效应 |
| 无条件掩码 | 82.42% | 222s | 文本语义下降 |
| 2 阶段 ProRes | 83.07% | 225s | 质量与速度最佳平衡 |
| 3 阶段 ProRes | 80.53% | 157s | 10.35× 加速但质量有所下降 |
| Text amplifier $\rho$=0.0 | 82.40% | - | 低分辨率 FOV 退化 |
| Text amplifier $\rho$=0.5 | 83.07% | - | 最佳视野保持 |

### 关键发现

- Jenga-Base（仅注意力裁剪+步跳过）甚至**超过基线** VBench 分数（83.34% vs 82.74%），主要是语义分数大幅提升（75.92% vs 72.84%）——稀疏注意力强迫模型聚焦关键信息
- 动态块选择（AttenCarve）比固定模式方法（CLEAR/SVG）快 1.3-2.4 倍且质量更优
- Text-attention amplifier 有效解决了低分辨率生成的视野退化问题
- 在蒸馏模型（AccVideo，仅 5 步）上仍能获得 2.12 倍加速，证明方法与步蒸馏正交
- 用户研究显示 Jenga 的感知质量与基线不可区分
- Block selection 仅引入 2.8% 额外计算开销，内存增加 3.7%（71.84→74.49 GiB）

## 亮点与洞察

- 极其优雅的框架设计：将注意力加速和 pipeline 加速解耦为两个独立正交维度，可灵活组合。AttenCarve 加速单步，ProRes 减少步数和 token，二者相乘获得超线性加速
- SFC 重排 + 动态 top-K 是对视频注意力稀疏性的深刻理解：不同层/head 有不同模式（局部、位置、语义、全局），固定模式无法兼顾，但动态选择额外开销极小
- 免训练是巨大优势——直接应用于 HunyuanVideo、Wan2.1、AccVideo 等多种模型，无需任何微调

## 局限与展望

- ProRes 的 latent 空间 resize 偶尔产生边界伪影，尤其在静态场景或清晰边缘处。使用详细 prompt 可缓解，但根本解决需要像素域 resize（额外 ~50s 开销）
- 当前 SFC 分割是静态的，未利用语义信息选择 token 重要性。未来可探索可学习的 attention carving
- 3 阶段 ProRes 质量下降明显（80.53%），latent 对齐是难点
- 该方法专注于推理加速，与训练端优化（步蒸馏、架构改进）正交但未联合探索

## 相关工作与启发

- **vs STA/CLEAR/SVG**: 这些方法使用固定的局部窗口或空间-时间稀疏模式，CLEAR 甚至比基线更慢（0.89×）。Jenga 的动态选择在 2.17 倍加速下质量更优
- **vs TeaCache**: TeaCache 通过缓存特征跳步实现 2.31 倍加速，Jenga 的 ProRes 与之正交且更高效——在步级别减少 token 数。结合使用效果更好
- **vs Bottleneck Sampling**: BottleneckSampling 也用变分辨率策略，但保留首阶段原始分辨率。ProRes 更激进地从低分辨率开始，配合 text-attention amplifier 维持 FOV

## 评分

- 新颖性: ⭐⭐⭐⭐ 动态块注意力裁剪 + 渐进分辨率的正交组合设计精巧，text-attention amplifier 的 FOV 修正巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ 4种模型适配（T2V、I2V、蒸馏模型、Wan2.1）、详细消融、用户研究、多GPU部署、16维VBench细分
- 写作质量: ⭐⭐⭐⭐ 结构清晰，方法直观，附录极其详尽（算法伪代码、参数表、实现细节）
- 价值: ⭐⭐⭐⭐⭐ 免训练 8.83 倍加速几乎无质量损失，即插即用特性使其有极高的实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] S²Q-VDiT: Accurate Quantized Video Diffusion Transformer with Salient Data and Sparse Token Distillation](s2q-vdit_accurate_quantized_video_diffusion_transformer_with_salient_data_and_sp.md)
- [\[ICML 2026\] DFSAttn: Dynamic Fine-Grained Sparse Attention for Efficient Video Generation](../../ICML2026/video_generation/dfsattn_dynamic_fine-grained_sparse_attention_for_efficient_video_generation.md)
- [\[CVPR 2025\] LongDiff: Training-Free Long Video Generation in One Go](../../CVPR2025/video_generation/longdiff_training-free_long_video_generation_in_one_go.md)
- [\[NeurIPS 2025\] VORTA: Efficient Video Diffusion via Routing Sparse Attention](vorta_efficient_video_diffusion_via_routing_sparse_attention.md)
- [\[ICML 2026\] WorldCache: Accelerating World Models for Free via Heterogeneous Token Caching](../../ICML2026/video_generation/worldcache_accelerating_world_models_for_free_via_heterogeneous_token_caching.md)

</div>

<!-- RELATED:END -->
