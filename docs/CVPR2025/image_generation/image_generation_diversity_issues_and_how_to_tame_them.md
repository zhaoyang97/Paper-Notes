---
title: >-
  [论文解读] Image Generation Diversity Issues and How to Tame Them
description: >-
  [CVPR 2025][图像生成][生成多样性] 本文揭示了当前扩散模型存在严重的多样性不足问题（最先进模型仅覆盖训练数据 77% 的多样性），提出了基于图像检索的 Image Retrieval Score (IRS) 作为可解释的多样性度量指标，并引入 Diversity-Aware Diffusion Models (DiADM) 在不损失生成质量的前提下提升多样性。
tags:
  - "CVPR 2025"
  - "图像生成"
  - "生成多样性"
  - "图像检索评分"
  - "扩散模型评估"
  - "多样性感知生成"
  - "特征提取器评估"
---

# Image Generation Diversity Issues and How to Tame Them

**会议**: CVPR 2025  
**arXiv**: [2411.16171](https://arxiv.org/abs/2411.16171)  
**代码**: [https://github.com/MischaD/beyondfid](https://github.com/MischaD/beyondfid)  
**领域**: 扩散模型 / 图像生成评估  
**关键词**: 生成多样性, 图像检索评分, 扩散模型评估, 多样性感知生成, 特征提取器评估

## 一句话总结

本文揭示了当前扩散模型存在严重的多样性不足问题（最先进模型仅覆盖训练数据 77% 的多样性），提出了基于图像检索的 Image Retrieval Score (IRS) 作为可解释的多样性度量指标，并引入 Diversity-Aware Diffusion Models (DiADM) 在不损失生成质量的前提下提升多样性。

## 研究背景与动机

**领域现状**：生成模型已能产出与真实数据几乎无法区分的高质量图像，但多样性问题长期被忽视。与质量问题不同，多样性不足很难从视觉上发现，需要专门的度量指标。现有指标如 FID 主要衡量质量和分布距离，Precision/Recall 虽涉及多样性但依赖超参数且不可解释。

**现有痛点**：(1) 用于计算现有指标的特征提取器（Inception v3、DINOv2 等）在特征空间中存在"坍缩"，即使对真实数据也无法正确度量多样性（存在 measurement gap）；(2) 现有多样性指标不可解释——FID 的一个点差距意味着什么？Recall 饱和在 0.8 是否意味着多样性足够？(3) 提升多样性的方法通常以牺牲生成质量为代价（噪声扰动、降低保真度）。

**核心矛盾**：多样性与保真度长期被认为是 trade-off 关系，现有方法无法解耦这两个属性。同时，缺乏一个可靠且可解释的度量指标来量化多样性。

**本文目标** (1) 设计可解释的多样性度量指标；(2) 量化现有模型的多样性缺口；(3) 在不损失质量的前提下提升多样性。

**切入角度**：将多样性评估框架为图像检索问题——用合成图像检索训练图像，已检索到的训练图像比例即为多样性。结合 Coupon Collector 问题的概率模型给出统计估计和置信区间。

**核心 idea**：用图像检索方式度量生成模型的多样性，并通过伪无条件特征作为条件输入实现多样性与保真度的解耦。

## 方法详解

### 整体框架

方法包含两部分：(1) IRS 度量指标——对每个合成图像，在特征空间中找到最近的训练图像，统计被"检索到"的训练图像去重后的比例；基于 Coupon Collector 问题的概率模型，用少量样本推断模型在无限采样下的多样性上限 $\text{IRS}_\infty$ 及其置信区间。(2) DiADM 模型——用预训练特征提取器为训练图像计算伪标签，作为条件输入替代占位符标签，实现多样性与保真度的解耦。

### 关键设计

1. **Image Retrieval Score (IRS)**:

    - 功能：提供可解释、无超参数的生成多样性度量
    - 核心思路：定义"已学习"图像集 $\mathcal{X}_{learned} = \{x_t \in \mathcal{X} \mid \exists x_t' \in \mathcal{X}': x_t = \arg\min_{x_t} \mathcal{P}(x_t, x_t')\}$，IRS 即为 $N_{learned}/N_{train}$。关键创新在于利用 Stirling 数和 Coupon Collector 问题的概率分布 $P(k,n,s) = \frac{\text{Stir}(n,k) \cdot s!}{(s-k)! \cdot s^n}$，从少量样本（远小于训练集）推断最大似然估计 $\text{IRS}_\infty$ 及上下界。为消除特征提取器坍缩导致的系统偏差，引入调整步骤 $\text{IRS}_{\infty,a} = \text{IRS}_{\infty,snth}/\text{IRS}_{\infty,real}$
    - 设计动机：现有 Recall 指标在所有类别都存在时就饱和在 0.8，无法区分 80% 和 100% 多样性。Coverage 依赖超参数且高估低多样性场景。IRS 在 ImageNet 受控实验中与真实类别比例呈线性相关，直觉清晰

2. **特征提取器评估与选择**:

    - 功能：找到最适合度量多样性的特征空间
    - 核心思路：在 9 种主流特征提取器（BYOL、CLIP、ConvNeXt、DINOv2、Inception、MAE 等）上评估真实数据的图像检索性能。通过集成投票（5 个以上模型对同一检索结果达成共识）建立伪标准答案。衡量每个提取器与集成共识的一致率，最终选择 SwAV 作为默认特征提取器
    - 设计动机：所有特征提取器对真实数据都存在度量误差（measurement gap），说明用它们计算的 FID、Precision/Recall 等指标天然不可靠。通过调整步骤消除系统偏差，同时选择与集成共识最一致的提取器，最大化可解释性

3. **Diversity-Aware Diffusion Models (DiADM)**:

    - 功能：在不损失 FID 的前提下，提升无条件扩散模型的生成多样性
    - 核心思路：用预训练 Inception v3 为每张训练图像提取特征作为伪标签，替代无条件生成中通常使用的占位符标签。架构上基于 EDM-2-XS，修改标签维度匹配特征维度。采样时直接用训练图像的特征作为条件查询，确保模型覆盖整个训练分布。本质上让每个训练样本成为自己的一个"类"，实现了多样性与保真度的解耦
    - 设计动机：无条件生成缺乏引导信号，模型倾向于收敛到分布的高密度区域。通过伪标签提供实例级条件，指导模型覆盖长尾样本

### 损失函数 / 训练策略

DiADM 使用标准扩散损失训练，仅修改条件输入。训练预算为 574 A40 GPU 小时（仅为 EDM 完整训练的 1/10）。采样时使用训练数据特征作为条件。

## 实验关键数据

### 主实验

| 模型 | FID ↓ | IRS∞,a ↑(条件) | IRS∞,a ↑(无条件) |
|------|-------|----------------|------------------|
| ADM-256 | 6.01 | 0.44 | 0.20 |
| DiT-XL/2-256 | 22.15 | 0.23 | 0.33 |
| MAR-H-256 | 3.11 | 0.64 | 0.38 |
| EDM-2-XL-512 | 2.92 | **0.77** | 0.03 |
| EDM-2-XXL-512 | 2.87 | 0.75 | 0.05 |
| LDM-256 | 26.09 | 0.16 | 0.16 |

即使最好的条件模型 EDM-2-XL 也仅达到训练数据 77% 的多样性。无条件生成的多样性更是急剧下降。

### 消融实验（DiADM 效果）

| 数据集 | EDM FID | DiADM FID | EDM IRS∞,a | DiADM IRS∞,a |
|--------|---------|-----------|------------|--------------|
| ImageNet-512 | 51.59 | 22.28 | 0.09 | 0.15 |
| FFHQ | 40.92 | 6.24 | 0.23 | 1.51 |
| ChestX-ray14 | 24.29 | 6.76 | 0.19 | 1.08 |
| CelebV-HQ | 68.41 | 13.64 | 0.18 | 0.69 |

DiADM 在所有数据集上同时改善了 FID 和 IRS，证明多样性与质量可以解耦。在 FFHQ 上 IRS 超过 1.0，说明伪条件能引导模型覆盖训练分布之外的区域。

### 关键发现

- 所有现有特征提取器对真实数据的多样性度量都严重不足（measurement gap），这意味着 FID 等基于这些提取器的指标天然存在偏差
- 模型规模与多样性正相关：EDM 从 XS 到 XL，IRS 从 46% 增长到 77%，而 FID 差距仅 0.9 分
- 条件生成比无条件生成多样性好得多（IRS 0.77 vs 0.03），说明引导信号对多样性至关重要
- 文本到图像模型也存在严重的多样性偏差，例如 Deepfloyd 在不指定性别时生成的多样性仅约 50%

## 亮点与洞察

- **将多样性评估框架为图像检索问题极其优雅**：Coupon Collector 概率模型提供了少量样本即可估计全局多样性的数学基础，且输出值具有直观的物理含义（"模型学了百分之多少的数据"）
- **measurement gap 的发现具有警示意义**：揭示了社区广泛使用的特征提取器在多样性评估上的系统性缺陷，质疑了大量基于 FID、Recall 的结论的可靠性
- **DiADM 的伪标签思路简单有效**：用预训练特征作为自监督的"类标签"，将无条件生成转化为实例级条件生成，在概念上将多样性与保真度完全解耦。这个思路可以推广到视频生成、3D 生成等领域

## 局限与展望

- 少量样本下 IRS 的随机波动较大，微小的 IRS 差异不具统计意义
- DiADM 实验仅在受限计算预算（1/10 完整训练）下进行，完整训练的效果未知
- 伪标签使用的 Inception v3 本身存在特征坍缩问题，更好的特征提取器可能进一步提升效果
- 仅在无条件生成上验证 DiADM，扩展到文本条件生成更有实际价值但需解决条件冲突问题
- IRS 依赖 nearest neighbor 检索，大规模数据集上的计算效率需要优化

## 相关工作与启发

- **vs FID/Precision/Recall**: 这些传统指标要么不度量多样性（FID 混合了质量和多样性），要么依赖超参数且不可解释（Recall）。IRS 提供了唯一一个可直接解释为"数据覆盖率"的指标
- **vs Vendi Score**: Vendi Score 完全不用参考集，无法衡量与训练数据的关系。IRS 以训练集为参考，更适合评估生成模型是否学到了完整分布
- **vs 多样性增强方法（SDEdit noise、DDIM diversity）**: 这些方法以质量换多样性。DiADM 通过解耦实现了二者兼得

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 问题定义新颖（量化多样性缺口）、方法新颖（IRS + DiADM）、发现重要（measurement gap）
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖 9 种特征提取器、5 个数据集、十余种生成模型，受控验证设计精巧
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰，但部分符号定义略显繁琐
- 价值: ⭐⭐⭐⭐⭐ 指出了社区广泛忽视的多样性问题，提供了可靠的度量工具和改进方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Trade-offs in Image Generation: How Do Different Dimensions Interact?](../../ICCV2025/image_generation/trade-offs_in_image_generation_how_do_different_dimensions_interact.md)
- [\[CVPR 2025\] IDEA-Bench: How Far are Generative Models from Professional Designing?](idea-bench_how_far_are_generative_models_from_professional_designing.md)
- [\[CVPR 2026\] One Algorithm to Align Them All](../../CVPR2026/image_generation/one_algorithm_to_align_them_all.md)
- [\[CVPR 2026\] DiverseGRPO: Mitigating Mode Collapse in Image Generation via Diversity-Aware GRPO](../../CVPR2026/image_generation/diversegrpo_mitigating_mode_collapse_in_image_generation_via_diversity-aware_grp.md)
- [\[CVPR 2025\] OmniGen: Unified Image Generation](omnigen_unified_image_generation.md)

</div>

<!-- RELATED:END -->
