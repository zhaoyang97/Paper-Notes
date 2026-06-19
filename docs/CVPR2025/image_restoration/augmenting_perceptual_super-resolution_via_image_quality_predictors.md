---
title: >-
  [论文解读] Augmenting Perceptual Super-Resolution via Image Quality Predictors
description: >-
  [CVPR 2025][图像恢复][超分辨率] 利用无参考图像质量评估（NR-IQA）模型代替人工标注，通过加权采样和直接优化两种方式提升感知超分辨率的图像质量，在无需人工数据的条件下超越依赖人工反馈的 SOTA 方法。 单图超分辨率（SISR）是一个经典的病态逆问题：一张低分辨率输入对应多个合理的高分辨率解…
tags:
  - "CVPR 2025"
  - "图像恢复"
  - "超分辨率"
  - "无参考图像质量评估"
  - "感知质量"
  - "NR-IQA"
  - "感知-失真权衡"
---

# Augmenting Perceptual Super-Resolution via Image Quality Predictors

**会议**: CVPR 2025  
**arXiv**: [2504.18524](https://arxiv.org/abs/2504.18524)  
**代码**: 无  
**领域**: 图像修复 / 超分辨率  
**关键词**: 超分辨率, 无参考图像质量评估, 感知质量, NR-IQA, 感知-失真权衡

## 一句话总结

利用无参考图像质量评估（NR-IQA）模型代替人工标注，通过加权采样和直接优化两种方式提升感知超分辨率的图像质量，在无需人工数据的条件下超越依赖人工反馈的 SOTA 方法。

## 研究背景与动机

单图超分辨率（SISR）是一个经典的病态逆问题：一张低分辨率输入对应多个合理的高分辨率解。传统像素级损失（L1/L2）训练出来的模型倾向于输出分布的期望——即模糊图像，PSNR 高但感知质量差。为此，社区引入感知损失和 GAN 来提升感知质量，但这些方法容易引入高频伪影。

此前的 HGGT 工作提出生成多个增强 GT，由人类标注者打分筛选"正面"GT 用于训练，取得了显著的感知质量提升。然而人工标注：(1) 粒度粗（只能标正/负/相似三档），(2) 不可微，无法直接用于梯度优化，(3) 昂贵且难以扩展。

**核心动机**：能否用已有的 NR-IQA 模型替代人类标注者？NR-IQA 模型具备三个人工标注不具备的优点：细粒度连续评分、可微分性、以及在线动态评估能力。

## 方法详解

### 整体框架

方法包含两个互补模块：(1) 基于 NR-IQA 的加权采样——改变训练时多 GT 的选择策略；(2) 基于 NR-IQA 的直接优化——将质量分数作为可微损失项。两者结合即 AMO+FT 方案。

### 关键设计

#### 1. NR-IQA 指标分析与选择

作者在 SBS180K 和 HGGT 两个人类偏好数据集上系统评估了 20+ NR-IQA 指标。通过两阶段筛选：

- **Phase I**: 在 1212 个图像对上评估 42 个指标变体，选出 Top-7（PaQ-2-PiQ、NIMA、MUSIQ、LIQE、ARNIQA、Q-Align、TOPIQ-NR）
- **Phase II**: 在完整 SBS180K 上验证，MUSIQ 以 82.73% 测试准确率脱颖而出
- **互补性分析**: 在 MUSIQ 失败的样本上，NIMA 和 Q-Align 表现最好，因此用这两个指标作为补充评估

最终选择 MUSIQ 作为采样和优化的核心 IQA 模型。

#### 2. 加权采样策略（Reweighted Sampling）

基础公式：$I \sim \mathcal{P}[S_I \mid \text{SoftMax}_\tau(Q(S_I))]$

三种变体：
- **SMA (Softmax-All)**: 在所有 GT（含原始+增强）上按 IQA 加权采样，无需人工标签
- **SMP (Softmax-Positives)**: 只在人工标注的正面 GT 上加权采样，利用了人工数据
- **AMO (Argmax-Online)**: 先从每个 GT 采一个 patch，然后对 patch 级别运行 IQA 选最佳——实现更细粒度的在线判断

AMO 的关键创新：将质量评估从"图像级"下沉到"patch 级"，能发现人工标注无法区分的质量差异（如图 2 所示，两张人工均标为"positive"的 GT，MUSIQ 分别给出 36.13 vs 54.19）。

#### 3. 直接优化（Direct Optimization）

将 NR-IQA 模型 Q 加入目标函数：$\widetilde{\mathcal{L}}(\phi|\hat{I},I) = \mathcal{L}(\phi|\hat{I},I) - \lambda_Q Q(\hat{I})$

**关键问题**：直接优化神经网络 IQA 模型会导致"对抗攻击"效应——梯度下降欺骗 Q 给出高分但实际引入结构性伪影。

**解决方案**：借助 LoRA 低秩适应进行正则化。冻结主网络参数 θ，只训练 LoRA 参数 ϕ，限制了模型的修改幅度。这一做法受到文本到图像生成中人类反馈引导的启发。

### 损失函数 / 训练策略

基础损失（与 HGGT 相同）：$\mathcal{L}(\theta|\hat{I},I) = \lambda_{\ell_1}\|I-\hat{I}\|_1 + \lambda_P d_P(\hat{I},I) + \lambda_A D(\hat{I})$

微调阶段额外加入 NR-IQA 项，并默认关闭 GAN 损失（因为 IQA 本身承担了类似角色）。通过调节 $\lambda_P$ 和 $\lambda_Q$ 的比例，可以控制中层感知指标 vs 高层 NR 质量之间的权衡。

## 实验关键数据

### 主实验

| 模型 | 无人工标注 | PSNR↑ | LPIPS-ST↓ | MUSIQ↑ | NIMA↑ | Q-Align↑ | TOPIQ↑ |
|------|-----------|-------|-----------|--------|-------|----------|--------|
| SwinIR-UPos (HGGT SOTA) | ✗ | 22.30 | 0.129 | 66.39 | 5.16 | 3.56 | 0.62 |
| SwinIR-AMO | ✓ | 22.08 | 0.124 | 68.08 | 5.21 | 3.67 | 0.66 |
| SwinIR-AMO+FT | ✓ | 21.77 | 0.121 | **70.81** | **5.29** | **3.75** | **0.70** |
| Gold Standard | - | - | - | 69.64 | 5.28 | 3.78 | 0.69 |
| RESRGAN-UPos | ✗ | 21.54 | 0.192 | 65.93 | 5.25 | 3.47 | 0.63 |
| RESRGAN-AMO+FT | ✓ | 21.02 | 0.169 | **71.67** | **5.35** | **3.68** | **0.71** |

AMO+FT 在所有 NR-IQA 指标上超越 UPos（HGGT SOTA），且无需人工标注。SwinIR 上甚至超越了 Gold Standard 的 NR 上界。

### 消融实验

| 实验 | 结论 |
|------|------|
| SMA vs SMP vs AMO | AMO 一致性最优，patch 级在线选择优于图像级采样 |
| FT vs FT_HP vs FT_IG | 增大感知损失权重（FT_HP）可回调中层指标但牺牲 NR 质量；包含 GAN（FT_IG）无明显收益 |
| MUSIQ vs PaQ-2-PiQ 作为优化目标 | PaQ-2-PiQ 导致所有 NR 指标下降，验证 MUSIQ 是当前最佳选择 |
| UPos+FT (有人工数据) vs AMO+FT (无人工数据) | AMO+FT 在 SwinIR 上更优，在 RealESRGAN 上接近 |

### 关键发现

1. **NR-IQA 采样可超越人工标注**：AMO 不使用任何人工数据，但在感知指标上优于 UPos
2. **存在三级感知-失真权衡**：像素级（PSNR）→ 中层感知（LPIPS）→ 高层 NR-IQA，FT 可沿此链条灵活调控
3. **LPIPS-ST 比 LPIPS 更"感知"**：LPIPS-ST 的行为更接近 NR-IQA 指标，表明移不变性对于感知评估很重要
4. **判别器不是好的 IQA**：简单上调 GAN 损失不能替代 NR-IQA 优化

## 亮点与洞察

- **用自动化替代人工标注的完整路径**：从指标选择→采样策略→直接优化，形成系统方案
- **AMO 的 patch 级在线评估**是精妙设计：利用了 NR-IQA 可在线运行的独特优势
- **LoRA 正则化解决对抗攻击问题**：简洁而有效地解决了直接优化 IQA 的核心难题
- 发现了 LPIPS-ST 这一更具感知意义的中层指标

## 局限与展望

- 当前仅基于 MUSIQ 一个 IQA 模型，组合多个互补 IQA（如作者分析的 NIMA、Q-Align）可能进一步提升
- IQA 模型本身的偏差可能限制提升上限——若为 SR 任务微调 IQA 模型效果可能更好
- 未探索扩散模型作为 SR 骨干的适用性
- PSNR/SSIM 的下降需谨慎评估，对于某些下游应用像素保真度仍然重要

## 相关工作与启发

- **HGGT** 提供了多 GT 框架和人工标注基准线
- **RLHF 在文本到图像生成中的成功**启发了将 NR-IQA 用于 SR 微调
- **LoRA** 原本用于 LLM 适配，这里巧妙地作为优化正则化手段

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 将 NR-IQA 从评估工具升级为训练信号，AMO patch 级在线评估和 LoRA 正则化均有新意
- **实验充分度**: ⭐⭐⭐⭐⭐ — 系统的 IQA 指标分析（42 个变体）+ 两个架构 + 完整消融 + 用户研究
- **写作质量**: ⭐⭐⭐⭐ — 逻辑清晰，从分析到方法到实验层层递进
- **价值**: ⭐⭐⭐⭐ — 为感知超分提供了可扩展、无需人工标注的训练范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] PIDSR: Complementary Polarized Image Demosaicing and Super-Resolution](pidsr_complementary_polarized_image_demosaicing_and_super-resolution.md)
- [\[CVPR 2025\] Progressive Focused Transformer for Single Image Super-Resolution](progressive_focused_transformer_for_single_image_super-resolution.md)
- [\[NeurIPS 2025\] DP²O-SR: Direct Perceptual Preference Optimization for Real-World Image Super-Resolution](../../NeurIPS2025/image_restoration/dp2o-sr_direct_perceptual_preference_optimization_for_real-world_image_super-res.md)
- [\[AAAI 2026\] Temporal Inconsistency Guidance for Super-resolution Video Quality Assessment](../../AAAI2026/image_restoration/temporal_inconsistency_guidance_for_super-resolution_video_quality_assessment.md)
- [\[CVPR 2026\] Rethinking Knowledge Transfer in Image Quality Assessment: A Perceptual Preference Structure Alignment Perspective](../../CVPR2026/image_restoration/rethinking_knowledge_transfer_in_image_quality_assessment_a_perceptual_preferenc.md)

</div>

<!-- RELATED:END -->
