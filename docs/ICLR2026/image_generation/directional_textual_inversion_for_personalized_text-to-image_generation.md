---
title: >-
  [论文解读] Directional Textual Inversion for Personalized Text-to-Image Generation
description: >-
  [ICLR 2026][图像生成][Textual Inversion] 本文发现 Textual Inversion (TI) 学到的 token embedding 存在范数膨胀（norm inflation）问题，导致复杂 prompt 的文本对齐下降；提出 Directional Textual Inversion (DTI)，将 embedding 范数固定在分布内尺度、仅在单位超球面上用 Riemannian SGD 优化方向，结合 von Mises-Fisher 先验，显著提升 prompt 忠实度。
tags:
  - "ICLR 2026"
  - "图像生成"
  - "Textual Inversion"
  - "方向优化"
  - "超球面"
  - "von Mises-Fisher"
  - "个性化文本到图像"
---

# Directional Textual Inversion for Personalized Text-to-Image Generation

**会议**: ICLR 2026  
**arXiv**: [2512.13672](https://arxiv.org/abs/2512.13672)  
**代码**: [https://github.com/kunheek/dti](https://github.com/kunheek/dti)  
**领域**: 扩散模型 / 个性化生成  
**关键词**: Textual Inversion, 方向优化, 超球面, von Mises-Fisher, 个性化文本到图像  

## 一句话总结

本文发现 Textual Inversion (TI) 学到的 token embedding 存在范数膨胀（norm inflation）问题，导致复杂 prompt 的文本对齐下降；提出 Directional Textual Inversion (DTI)，将 embedding 范数固定在分布内尺度、仅在单位超球面上用 Riemannian SGD 优化方向，结合 von Mises-Fisher 先验，显著提升 prompt 忠实度。

## 研究背景与动机

**领域现状**：个性化文本到图像生成有两大范式——参数微调（如 DreamBooth）和嵌入优化（如 Textual Inversion）。TI 因为只优化 token embedding，具备存储小、易集成的优势，是很多后续方法的基础组件。

**现有痛点**：TI 在复杂 prompt 下表现很差——例如 "A painting of \<dog\> wearing a santa hat"，模型可能生成了狗但忽略了帽子和背景细节。根本原因是 TI 优化过程中 embedding 的范数会膨胀到极端值（>20，而正常词汇约 0.4）。

**核心矛盾**：语义信息主要编码在 embedding 的**方向**中（余弦相似度语义一致，欧几里得距离则不行），但 TI 不约束范数，导致：(a) 大范数在 pre-norm Transformer 中压制位置编码信息（$\mathcal{O}(1/m)$）；(b) 残差更新停滞，后续层无法有效修改 hidden state 方向。

**本文目标** 在保持 TI 轻量级优势的同时，解决 norm inflation 导致的文本对齐失败问题。

**切入角度**：作者从 CLIP token embedding 空间的几何结构出发，通过实验和理论两条线证明了"方向编码语义、范数膨胀有害"。这是一个可解释性驱动的分析视角。

**核心 idea**：固定 embedding 范数为分布内尺度，仅在单位超球面上优化方向，用 vMF 先验正则化。

## 方法详解

### 整体框架

DTI 想解决的是 TI 那个 norm inflation 顽疾：既然语义编码在 embedding 的方向里、范数一膨胀就拖垮文本对齐，那干脆别让范数参与优化。具体做法是把 token embedding $\bm{e} \in \mathbb{R}^d$ 拆成"范数 × 方向"两部分 $\bm{e} = m^\star \bm{v}$，其中方向 $\bm{v} \in \mathbb{S}^{d-1}$ 是单位超球面上的点。整个训练过程范数 $m^\star$ 被钉死在预训练词汇表 embedding 的均值范数上，只让方向 $\bm{v}$ 在球面上滑动；优化器换成尊重球面几何的 Riemannian SGD，再挂一个 von Mises-Fisher 方向先验把 $\bm{v}$ 拉向类别词附近。相比标准 TI，它只动了"怎么优化"，没加任何网络、也不增加存储。

### 关键设计

**1. 超球面方向优化（Riemannian SGD）：把优化约束在球面上，从源头堵死范数膨胀**

TI 的根问题是用欧几里得 AdamW 自由优化整个 embedding，参数会一路漂离合理流形、范数膨胀到 20 以上。DTI 改为只在单位球面 $\mathbb{S}^{d-1}$ 上走：每步先把欧几里得梯度投影到当前点的切空间，去掉沿径向的分量 $\bm{g} = \bm{g}_{\text{euc}} - (\bm{v}_k^\top \bm{g}_{\text{euc}})\bm{v}_k$，再对切向梯度做归一化 $\bm{g}' = \bm{g}/\|\bm{g}\|$，最后用 retraction 把更新拉回球面：

$$\bm{v}_{k+1} = \frac{\bm{v}_k - \eta \bm{g}}{\|\bm{v}_k - \eta \bm{g}\|}$$

这样范数永远是 $m^\star$、方向始终在流形上。消融里直接对比了"AdamW + 投影"和 RSGD，前者文本对齐明显更差——说明真正起作用的不只是固定范数，还要用尊重流形几何的优化器才不会把更新方向带偏。

**2. von Mises-Fisher (vMF) 方向先验：给方向一个"锚"，防止它在球面上越优化越偏离语义**

只约束在球面上还不够，方向仍可能漂到离类别词很远的地方导致语义漂移。DTI 把方向优化看成 MAP 估计，引入 vMF 分布作先验 $p(\bm{v}\mid\bm{\mu}, \kappa) \propto \exp(\kappa \bm{\mu}^\top \bm{v})$，其中均值方向 $\bm{\mu}$ 取对应类别词（如 'dog'）的归一化 embedding，集中度 $\kappa$ 控制锚得多紧。它的妙处在于实现极简：负对数先验对 $\bm{v}$ 的梯度是个常量 $-\kappa\bm{\mu}$，训练时直接加到数据梯度上即可，思路类似解耦权重衰减但适配到了球面。$\kappa$ 固定为 1e-4，几乎零额外开销。

**3. 范数尺度选择：把固定下来的那个范数取在"分布内"的均值，而不是随便选**

既然范数要被钉死，取多大就成了关键。DTI 把 $m^\star$ 设为预训练词汇表所有 embedding 的均值范数（约 0.4 这个量级），让被反演的概念落在和正常词汇同一尺度上。消融验证了这个选择的必要性：取词汇表里的最小范数会让主体相似度直接崩塌（Image Sim 掉到 0.030），取一个 OOD 的大范数（如 5.0）又把文本对齐拉低，唯有均值范数能同时兼顾两端。

### 损失函数 / 训练策略

数据损失就是标准扩散去噪 MSE $\mathcal{L}_{\text{data}}(m^\star \bm{v}) = \mathbb{E}[\|\bm{\epsilon} - \bm{\epsilon}_\theta(\bm{z}_t, t, c(m^\star \bm{v}))\|^2]$，先验损失 $\mathcal{L}_{\text{prior}} = -\kappa \bm{\mu}^\top \bm{v}$，总损失是两者之和。每个概念约训 7 分钟（SDXL，单卡 A6000），和原始 TI 的训练成本基本持平。

## 实验关键数据

### 主实验

| 模型 | 方法 | Image Sim (DINOv2) | Text Sim (SigLIP) |
|------|------|------|------|
| SDXL | TI | 0.561 | 0.292 |
| SDXL | TI-rescaled | 0.243 | 0.466 |
| SDXL | CrossInit | 0.545 | 0.464 |
| SDXL | **DTI (ours)** | **0.450** | **0.522** |
| SANA 1.5-1.6B | TI | 0.480 | 0.621 |
| SANA 1.5-1.6B | **DTI (ours)** | **0.479** | **0.744** |
| SANA 1.5-4.8B | TI | 0.446 | 0.646 |
| SANA 1.5-4.8B | **DTI (ours)** | **0.452** | **0.757** |

DTI 在所有模型上大幅提升文本对齐（SDXL上 0.292→0.522），同时保持合理的主体相似度。随模型增大优势更显著。

### 消融实验

| 优化器 | $m^\star$ | $\kappa \times 10^{-3}$ | Image | Text |
|--------|---------|---------|-------|------|
| AdamW | mean | 0.1 | 0.335 | 0.463 |
| RSGD | min | 0.1 | 0.030 | 0.074 |
| RSGD | 5.0 (OOD) | 0.1 | 0.383 | 0.373 |
| RSGD | mean | 0.0 | 0.507 | 0.436 |
| RSGD | mean | 0.5 | 0.278 | 0.688 |
| **RSGD** | **mean** | **0.1** | **0.450** | **0.522** |

### 关键发现

- RSGD 显著优于 AdamW+投影，说明尊重流形几何很重要
- 范数设为最小值或 OOD 值效果极差，均值最优
- vMF 先验不可或缺（$\kappa=0$ 时文本对齐明显下降），但 $\kappa$ 过大也会损害图像相似度
- 用户研究（100 人 AMT）中 DTI 在主体忠实度（43.45%）和文本对齐（66.77%）均排第一

## 亮点与洞察

- **理论分析扎实**：从 pre-norm Transformer 的数学结构出发，证明了 norm inflation → 位置信息衰减 + 残差更新停滞的因果链（Proposition 1, Corollary 1），这是对 TI 失败模式的首个系统性理论解释
- **球面插值 (SLERP) 能力**：DTI 的超球面参数化天然支持学习到的概念之间的平滑语义插值（如狗↔茶壶、猫↔狗），这是标准 TI 做不到的。这一能力开拓了概念混合的创意应用
- **极简且高效**：整个方法相比 TI 只改了优化过程——固定范数 + RSGD + 常量先验梯度，无额外网络、无额外存储，训练时间不增加

## 局限与展望

- DTI 主要改善文本忠实度，并不直接优化主体相似度；高主体保真度需要搭配 LoRA 等方法
- 理论分析聚焦 pre-norm 架构（CLIP, Gemma），对 post-norm 或其他归一化方案是否适用未知
- vMF 先验的 $\kappa$ 需要手动设定，虽然论文说 1e-4 通用，但不同概念复杂度下可能需要调整
- 仍然需要每个概念单独训练（SDXL ~7min），无法做到 zero-shot 个性化

## 相关工作与启发

- **vs TI**: TI 不约束范数导致 embedding OOD，DTI 通过固定范数+方向优化根本解决了这个问题
- **vs CrossInit**: CrossInit 在 SDXL 上文本对齐不错但在 SANA（LLM-based encoder）上失效，DTI 跨架构泛化更好
- **vs P+/NeTI**: 这些方法通过更丰富的 embedding 空间改善 TI，但引入大量计算开销，DTI 保持了 TI 的轻量优势
- 方向优化 + vMF 先验的思路可以迁移到 VLM prompt tuning 或 LLM soft prompt 优化中

## 评分

- 新颖性: ⭐⭐⭐⭐ 从几何视角解释 TI 的失败并给出简洁方案，洞察深刻
- 实验充分度: ⭐⭐⭐⭐ 多模型（SDXL/SANA）、消融完整、用户研究、插值实验
- 写作质量: ⭐⭐⭐⭐⭐ 理论-实验-方法逻辑链非常清晰，图表精美
- 价值: ⭐⭐⭐⭐ 实用价值高，即插即用，对 TI 生态有广泛影响

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Preserve and Personalize: Personalized Text-to-Image Diffusion Models without Distributional Drift](preserve_and_personalize_personalized_text-to-image_diffusion_models_without_dis.md)
- [\[AAAI 2026\] DOS: Directional Object Separation in Text Embeddings for Multi-Object Image Generation](../../AAAI2026/image_generation/dos_directional_object_separation_in_text_embeddings_for_mul.md)
- [\[ICLR 2026\] MOSAIC: Multi-Subject Personalized Generation via Correspondence-Aware Alignment and Disentanglement](mosaic_multi-subject_personalized_generation_via_correspondence-aware_alignment_.md)
- [\[ECCV 2024\] Textual-Visual Logic Challenge: Understanding and Reasoning in Text-to-Image Generation](../../ECCV2024/image_generation/textual-visual_logic_challenge_understanding_and_reasoning_in_text-to-image_gene.md)
- [\[CVPR 2026\] Premier: Personalized Preference Modulation with Learnable User Embedding in Text-to-Image Generation](../../CVPR2026/image_generation/premier_personalized_preference_modulation_with_learnable_user_embedding_in_text.md)

</div>

<!-- RELATED:END -->
