---
title: >-
  [论文解读] Semantic Discrepancy-aware Detector for Image Forgery Identification
description: >-
  [ICCV 2025][图像生成][图像伪造检测] 提出语义差异感知检测器（SDD），通过语义 token 采样、概念级伪造差异学习和低层伪造特征增强三个模块，利用重建学习将 CLIP 的视觉语义概念空间与伪造空间进行细粒度对齐，在 UnivFD 和 SynRIS 基准上取得 SOTA 性能（$ap_m$ 98.51%，AUROC 95.1%）。
tags:
  - "ICCV 2025"
  - "图像生成"
  - "图像伪造检测"
  - "语义概念空间"
  - "重建学习"
  - "CLIP"
  - "视觉-语言模型"
---

# Semantic Discrepancy-aware Detector for Image Forgery Identification

**会议**: ICCV 2025  
**arXiv**: [2508.12341](https://arxiv.org/abs/2508.12341)  
**代码**: [GitHub](https://github.com/wzy1111111/SSD)  
**领域**: 图像生成  
**关键词**: 图像伪造检测, 语义概念空间, 重建学习, CLIP, 视觉-语言模型

## 一句话总结

提出语义差异感知检测器（SDD），通过语义 token 采样、概念级伪造差异学习和低层伪造特征增强三个模块，利用重建学习将 CLIP 的视觉语义概念空间与伪造空间进行细粒度对齐，在 UnivFD 和 SynRIS 基准上取得 SOTA 性能（$ap_m$ 98.51%，AUROC 95.1%）。

## 研究背景与动机

随着 GAN 和扩散模型的快速发展，生成的图像越来越逼真，检测伪造图像变得至关重要。现有方法存在以下问题：

**单一伪造特征方法泛化差**：早期方法（如 CNNSpot）挖掘噪声模式、纹理统计或频率信号等单一特征，但这些特征对特定生成模型过拟合，在未见模型上性能急剧下降。

**语义概念空间与伪造空间的错位**：
   - **冻结预训练模型的局限**（如 UnivFD）：直接使用 CLIP 冻结特征配合线性探测，保留了语义先验但忽略了细粒度伪造细节。由于语义概念空间被冻结，共享相似语义的真假样本容易被误分类。
   - **Prompt 调优的局限**（如 FatFormer）：虽然通过伪造感知适配器构建伪造自适应空间效果好，但基于简单 [CLASS] 嵌入的 soft prompt 在语义描述粒度上有内在限制——概念覆盖面窄可能导致检测偏向错误的语义维度。

**关键洞察**：作者通过统计分析发现，不同语义概念可能引导模型发现不同的伪造痕迹。即语义概念和伪造特征之间存在细腻的关联关系——纯粹冻结或纯粹微调语义空间都不是最优解。

## 方法详解

### 整体框架

SDD 包含三个关键模块，构成一个基于重建学习的伪造检测框架：

1. **语义 Token 采样（STS）**：从真实图像中采样代表性的语义 patch token
2. **概念级伪造差异学习（CFDL）**：利用这些 token 通过重建学习捕捉伪造差异
3. **低层伪造特征增强器（Feature Enhancer）**：将重建差异图融入低层特征，提取高泛化伪造特征

最终将 LoRA-CLIP 的 CLS token 与增强后的低层特征拼接，送入线性分类器进行真/假二分类。

### 关键设计

1. **Semantic Tokens Sampling（STS）**:

    - 功能：从真实图像中均匀采样一组代表性 patch token $f_s \in \mathbb{R}^{M \times D}$，作为视觉线索平滑连接语义概念空间和伪造空间
    - 核心思路：使用 Jensen-Shannon（JS）散度衡量 token 间的分布差异。给定初始 token $\tilde{r}$，将 JS 散度范围 $[0,1]$ 等分为 $M$ 段，从每段内选一个 token：$f_s = \mathcal{S}(\mathbb{R}^{N \times D}, \delta)$，其中 $\delta$ 是采样率，$M = 1/\delta$
    - 设计动机：直接将所有真实 patch token 融入重建模块计算量过大且包含冗余；基于 JS 散度的均匀采样确保 token 在统一 CLIP 空间中均匀分布，代表真实图像的语义分布，同时避免偏向任何特定的非伪造相关分布。关键优势是绕过了文本 prompt 引入的语义偏差

2. **Concept-level Forgery Discrepancy Learning（CFDL）**:

    - 功能：通过 Transformer 编码器-解码器进行重建学习，捕获伪造图像在语义概念层面的差异
    - 核心思路：
        - 使用 LoRA 微调 CLIP-ViT 获取高层视觉特征 $V_H = \mathcal{F}_{LoRA}(\mathcal{I})$
        - **编码器**：采样的语义 token $f_s$ 作为 query，$V_H$ 作为 key/value：$R_1 = \text{LN}(\text{MHA}(f_s, V_H, V_H))$，$R_2 = \text{LN}(\text{MHA}(R_1, V_H, V_H))$
        - **解码器**：$R_3 = \text{LN}(\text{MHA}(R_2, R_2, R_2))$，$R_e = \text{LN}(\text{MHA}(R_1, R_3, R_3))$
        - 重建损失仅对真实样本计算：$\mathcal{L}_r = \frac{1}{B}\sum_{i=0}^B \text{MSE}(R_e, V_H)$
        - 重建差异图：$\mathcal{D}_s = |R_f - f_r|$
    - 设计动机：仅缩小真实样本的重建差距，使伪造样本的重建差异被自然放大。与 FatFormer 使用文本 prompt 不同，CFDL 纯粹基于视觉信息——采样的语义 token 提供比粗粒度文本 prompt 更丰富的细节，能揭示更多隐藏在图像中的伪造痕迹

3. **Low-level Forgery Feature Enhancer（低层伪造特征增强器）**:

    - 功能：利用重建差异图引导提取多尺度低层伪造特征，弥补高层语义特征对弱语义关联伪造线索的忽视
    - 核心思路：
        - 使用多阶段卷积网络提取 $F(n)$（$n=1,2,3$）
        - 将差异图 $\mathcal{D}_s$ 反卷积后与 $F(n)$ 逐像素相乘：$F'(n) = F(n) \otimes \text{deconv}(\mathcal{D}_s)$
        - 自适应权重系数：$\frac{1}{e_n} = \frac{1}{e^{|F'(n) - F(n)|}}$，当差异大时权重小（快速区间：关注语义强相关特征），差异小时权重趋近 1（慢速区间：保留与语义弱相关但与伪造强相关的特征）
        - 最终：$F_{low}(n) = F(n) + \frac{F(n)}{e_n}$
    - 设计动机：单纯依赖高层语义概念会遗漏像素级的伪造痕迹。自适应权重的指数逆函数设计巧妙地平衡了两类特征：语义概念强相关的伪造特征（差异大时低权重避免过度依赖语义）和语义弱相关但伪造高相关的特征（差异小时高权重保留）

### 损失函数 / 训练策略

总损失：$\mathcal{L} = \mathcal{L}_{bce} + \lambda_1 \mathcal{L}_{tri} + \lambda_2 \mathcal{L}_r$

- $\mathcal{L}_{bce}$：二元交叉熵，用于真/假分类
- $\mathcal{L}_{tri}$：三元组损失，拉近同类特征、推远异类特征：$\mathcal{L}_{tri} = \max(0, d(f_p, f_a) - d(f_n, f_a) + \alpha)$
- $\mathcal{L}_r$：重建损失（MSE），仅在真实样本上计算

训练设置：学习率 $1 \times 10^{-5}$，batch size 32，LoRA 参数 $r=6, \alpha=6$，dropout=0.8。使用 ProGAN 图像作为训练数据。

## 实验关键数据

### 主实验

| 数据集 | 指标 | SDD (本文) | FatFormer | UnivFD | NPR | 提升 |
|--------|------|-----------|-----------|--------|-----|------|
| UnivFD | $ap_m$ | **98.51** | 98.18 | 93.38 | 92.19 | +0.33 vs FatFormer |
| UnivFD | $acc_m$ | **93.61** | 90.86 | 81.38 | 86.20 | +2.75 vs FatFormer |
| SynRIS | AUROC 平均 | **95.1** | 78.0 | 62.3 | 90.3 | +4.8 vs NPR |

SynRIS 上的显著优势：在高保真文本到图像模型生成的图像上，SDD (95.1%) 远超 FatFormer (78.0%) 和 UnivFD (62.3%)。

### 消融实验

| # | STS | CFDL | 增强器 | $ap_m$ | $acc_m$ | 说明 |
|---|:---:|:---:|:---:|--------|---------|------|
| 1 | - | ✓ | - | 97.37 | 81.64 | 仅 CFDL |
| 2 | - | ✓ | ✓ | 97.41 | 90.17 | 增强器贡献显著 (+8.53 acc) |
| 3 | ✓ | ✓ | - | 97.39 | 89.98 | STS 有帮助 |
| 4 | ✓ | ✓ | ✓ | **98.52** | **93.61** | 全组件最优 |

自适应权重函数对比：

| 函数 | $ap_m$ | $acc_m$ | 说明 |
|------|--------|---------|------|
| $f(x) = |x|$ | 97.77 | 92.12 | 线性权重 |
| $f(x) = x^2$ | 97.93 | 92.34 | 二次权重 |
| $1/e^x$（本文）| **98.52** | **93.61** | 指数逆最优 |

### 关键发现

- **视觉信息优于文本 prompt**：SDD 完全不使用文本 prompt，仅依靠视觉语义 token 和重建学习，在 UnivFD 上超越使用 prompt 的 FatFormer，证明细粒度视觉信息比粗粒度文本描述更适合捕捉伪造痕迹
- **SynRIS 数据集上的优势最大**（AUROC 95.1% vs NPR 90.3%），因为高保真生成模型能把握全局语义但无法精细化局部像素，SDD 同时捕获语义概念和低层伪造特征
- **注意力图可视化**显示 SDD 能对不同伪造图像关注不同区域（背景、局部目标、边缘细节），而真实图像几乎不显示伪造差异区域——验证了重建损失的有效性
- t-SNE 可视化中 ProGAN 的真假分界比其他模型更模糊——因为在语义概念监督下，决策边界更复杂和细腻

## 亮点与洞察

- **首个完全基于视觉信息的预训练 VLM 伪造检测范式**：不使用任何文本 prompt，绕过了文本描述粗粒度导致的语义偏差问题
- **JS 散度均匀采样**是简洁但有效的 token 选择策略，确保采样 token 在语义空间中均匀分布
- **重建学习的巧妙应用**：仅对真实样本计算重建损失，伪造样本的重建差异自然放大——这比显式构造对比样本更优雅
- **指数逆自适应权重** $1/e^x$ 的双区间特性：快速区间捕获语义强相关特征，慢速区间保留伪造强相关但语义弱相关的特征，实现两类特征的自动平衡

## 局限与展望

- 训练数据仅使用 ProGAN（或 Stable Diffusion v1），对新兴生成模型的泛化依赖于方法本身的鲁棒性而非数据覆盖
- STS 模块的采样率 $\delta$ 是用户定义的超参数，最优值可能随数据集变化
- 重建模块增加了推理延迟，实际部署需考虑效率
- 在 DALL-E 3 上的表现相对较弱（AUROC 85.9%），说明对部分高端商业模型的泛化仍有提升空间
- LoRA 参数设置（$r=6, \alpha=6$）的敏感性未充分讨论

## 相关工作与启发

- UnivFD（CVPR 2023）证明了 VLM 特征空间对伪造检测的价值，本文在此基础上解决了冻结特征的局限
- FatFormer（CVPR 2024）通过 prompt 调优构建伪造自适应空间，但本文指出 prompt 的语义粒度不足
- 重建学习在无监督表示学习中广泛应用（如 MAE），本文将其创新性地引入伪造检测，利用真实图像的可重建性与伪造图像的不可重建性之间的差异
- NPR 关注像素级邻居关系，与 SDD 的语义概念+低层特征的组合是互补的

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 纯视觉范式、语义概念空间与伪造空间对齐的思路、指数逆自适应权重都有显著创新
- 实验充分度: ⭐⭐⭐⭐ 两个主要基准全面评估，消融深入，但缺少视频伪造检测等扩展评估
- 写作质量: ⭐⭐⭐⭐ 动机论证有力，方法描述清晰，但部分符号略不一致
- 价值: ⭐⭐⭐⭐⭐ 随着 AI 生成图像的泛滥，通用伪造检测方法具有重大实际价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] ForgeLens: Data-Efficient Forgery Focus for Generalizable Forgery Image Detection](forgelens_data-efficient_forgery_focus_for_generalizable_forgery_image_detection.md)
- [\[ICCV 2025\] M2SFormer: Multi-Spectral and Multi-Scale Attention with Edge-Aware Difficulty Guidance for Image Forgery Localization](m2sformer_multi-spectral_and_multi-scale_attention_with_edge-aware_difficulty_gu.md)
- [\[ICCV 2025\] DeepShield: Fortifying Deepfake Video Detection with Local and Global Forgery Analysis](deepshield_fortifying_deepfake_video_detection_with_local_and_global_forgery_ana.md)
- [\[ECCV 2024\] SAIR: Learning Semantic-aware Implicit Representation](../../ECCV2024/image_generation/sair_learning_semantic-aware_implicit_representation.md)
- [\[ICCV 2025\] Describe, Don't Dictate: Semantic Image Editing with Natural Language Intent](describe_dont_dictate_semantic_image_editing_with_natural_language_intent.md)

</div>

<!-- RELATED:END -->
