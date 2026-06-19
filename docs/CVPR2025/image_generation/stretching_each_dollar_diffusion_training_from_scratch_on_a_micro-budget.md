---
title: >-
  [论文解读] Stretching Each Dollar: Diffusion Training from Scratch on a Micro-Budget
description: >-
  [CVPR 2025][图像生成][低成本训练] MicroDiT 通过提出延迟遮蔽（deferred masking）策略——先用轻量级 patch-mixer 预处理所有 patch 再遮蔽 75%——配合层级宽度缩放、混合专家（MoE）及合成数据，仅用 $1,890 成本在 2.6 天内从零训练出 11.6 亿参数的稀疏 Transformer，在 COCO 上达到 12.7 FID，成本仅为 Stable Diffusion 的 1/118。
tags:
  - "CVPR 2025"
  - "图像生成"
  - "低成本训练"
  - "patch masking"
  - "延迟遮蔽"
  - "混合专家"
  - "合成数据"
---

# Stretching Each Dollar: Diffusion Training from Scratch on a Micro-Budget

**会议**: CVPR 2025  
**arXiv**: [2407.15811](https://arxiv.org/abs/2407.15811)  
**代码**: 无（作者承诺开源训练流程）  
**领域**: 扩散模型 / 图像生成  
**关键词**: 低成本训练, patch masking, 延迟遮蔽, 混合专家, 合成数据

## 一句话总结
MicroDiT 通过提出延迟遮蔽（deferred masking）策略——先用轻量级 patch-mixer 预处理所有 patch 再遮蔽 75%——配合层级宽度缩放、混合专家（MoE）及合成数据，仅用 $1,890 成本在 2.6 天内从零训练出 11.6 亿参数的稀疏 Transformer，在 COCO 上达到 12.7 FID，成本仅为 Stable Diffusion 的 1/118。

## 研究背景与动机

**领域现状**：大规模 T2I 扩散模型（如 Stable Diffusion、DALL-E 3）已能生成极高质量的图像，每年生成超过 10 亿张图片。但这些模型的训练成本极其高昂——SD 2.1 需要 200,000 A100 GPU hours，即使是最先进的低成本方法 PixArt-α 也需要 85 天训练时间（约 $28,400）。

**现有痛点**：(1) 训练成本高达数万美元，将模型开发集中在少数大型机构手中；(2) 现有低成本方法（如 MaskDiT）在高遮蔽率（>50%）下性能急剧下降；(3) 许多方法依赖十亿级规模甚至私有数据集。

**核心矛盾**：Transformer 的计算成本与输入序列长度（即 patch 数量）成正比。遮蔽可以减少 patch 数量，但朴素遮蔽在高比例下会严重丢失图像信息，因为每个 patch embedding 只包含自身的局部信息，被遮蔽的 patch 对 Transformer 完全不可见。

**本文目标**：实现比当前 SOTA 低一个数量级的训练成本，同时不依赖海量或私有数据集。

**切入角度**：遮蔽不一定要在输入层进行。如果先让所有 patch 通过一个轻量级网络交换信息，再执行遮蔽，那么被保留的 patch 仍然携带了全图的语义信息。

**核心 idea**：在主干 Transformer 前添加一个轻量级 patch-mixer（仅占主干参数的 ~13%），先处理所有 patch 再遮蔽 75%，使未被遮蔽的 patch 能"代言"整幅图像。

## 方法详解

### 整体框架
输入图像 → VAE 编码为 latent → 分割为 patch 序列 → **patch-mixer（轻量 Transformer，4-6 层）处理全部 patch** → 随机遮蔽 75% → 主干扩散 Transformer 处理剩余 25% patch → 仅在未遮蔽 patch 上计算扩散损失。推理时不使用遮蔽。

### 关键设计

1. **延迟遮蔽策略 (Deferred Masking)**:

    - 功能：在高遮蔽率下保持图像全局语义信息
    - 核心思路：在 DiT 主干前插入一个 4-6 层的 Transformer 作为 patch-mixer，其参数仅占主干的约 13%。所有 patch 先通过 mixer 进行自注意力交互，使每个 patch embedding 融合了全图信息后，再执行随机遮蔽。训练损失仅在未遮蔽 patch 上计算：$\mathcal{L} = \mathbb{E}\|F_\theta(M_\phi(\mathbf{x}+\epsilon) \odot (1-m)) - \mathbf{x} \odot (1-m)\|^2$。与 MaskDiT 相比，不需要额外的 MAE 损失和解码器
    - 设计动机：朴素遮蔽在 75% 比例下 FID 从 3.79 恶化到 16.5，而延迟遮蔽仅恶化到 5.03。这是因为未遮蔽 patch 在 mixer 中已获得全图语义，不再是"盲"的

2. **层级宽度缩放 (Layer-wise Scaling) + MoE**:

    - 功能：以相同计算量获得更强的模型表达能力
    - 核心思路：层级宽度缩放线性增加深层 Transformer block 的隐层维度（注意力头数乘以 $m_a$，FFN 宽度乘以 $m_f$），让深层分配更多参数来学习复杂特征。MoE 则在交替的 Transformer block 中使用 8-expert 的混合专家层，采用 Expert-Choice 路由（每个专家选择自己处理的 token），无需额外的负载均衡损失。两者组合得到 11.6 亿参数的稀疏模型
    - 设计动机：层级缩放基于"深层网络学习更复杂特征"的观察（在朴素遮蔽实验中 FID 从 7.85 降到 7.11）。MoE 增加参数但不成比例增加计算量，在长训练中提升 FID 从 13.7 到 12.7

3. **合成数据混合训练**:

    - 功能：在有限训练数据和预算下大幅提升生成质量
    - 核心思路：在 22M 真实图片（CC12M + SA1B + TextCaps）基础上加入 15M 合成图片（JourneyDB + DiffusionDB）。虽然 FID 和 CLIP-score 指标提升不明显（FID 12.72 → 12.66），但 GPT-4o 人类偏好评估显示显著差异：在 PartiPrompts 上 63% vs 21% 偏好合成数据训练的模型
    - 设计动机：微预算下数据多样性至关重要。合成数据提供了真实数据集缺乏的概念覆盖和风格多样性，且 37M 图片远少于大多数大规模模型使用的十亿级数据

### 损失函数 / 训练策略
两阶段训练：Phase-1 在 256×256 分辨率上训练（250K 步遮蔽 + 30K 步无遮蔽微调），Phase-2 在 512×512 上微调（50K 步遮蔽 + 5K 步无遮蔽）。使用 AdamW 优化器，余弦学习率衰减，SwiGLU 激活函数。关键发现：高 $\beta_2$（0.999 而非 LLM 常用的 0.95）更适合扩散模型；高权重衰减有益。

## 实验关键数据

### 主实验

| 模型 | 参数量 | 开源 | 训练图片 | 8×A100天数 | FID-30K↓ |
|------|--------|------|---------|-----------|---------|
| Stable-Diffusion-1.5 | 0.9B | ✓ | 2000M | ~2200 | 22.0 |
| Stable-Diffusion-2.1 | 0.9B | ✓ | 2000M | ~2200 | 20.5 |
| PixArt-α | 0.6B | ✓ | 25M* | 753 | 7.3 |
| Würstchen | 1.0B | ✓ | 1500M | 126 | 22.4 |
| **MicroDiT (Ours)** | **1.16B** | ✓ | **37M** | **6.6** | **12.7** |

*PixArt-α 包含 10M 私有高质量图片

| 模型 | Overall | Single | Two | Counting | Colors | Position | Color Attr. |
|------|---------|--------|-----|----------|--------|----------|-------------|
| SD-1.5 | 0.43 | 0.97 | 0.38 | 0.35 | 0.76 | 0.04 | 0.06 |
| PixArt-α | 0.48 | 0.98 | 0.50 | 0.44 | 0.80 | 0.08 | 0.07 |
| SD-XL | 0.55 | 0.98 | 0.74 | 0.39 | 0.85 | 0.15 | 0.23 |
| SD-3 | 0.68 | 0.98 | 0.84 | 0.66 | 0.74 | 0.40 | 0.43 |
| **MicroDiT** | **0.46** | 0.97 | 0.47 | 0.33 | 0.78 | 0.09 | **0.20** |

### 消融实验

| 配置 | FID↓ | CLIP-FID↓ | CLIP-Score↑ | 说明 |
|------|------|----------|-------------|------|
| 朴素遮蔽 75% | 16.5 | - | - | 性能崩塌 |
| MaskDiT 75% | ~15.0 | - | - | MAE损失仅微弱改善 |
| **延迟遮蔽 75%** | **5.03** | - | - | 大幅恢复性能 |
| 无遮蔽 baseline | 3.79 | - | - | 上界 |
| w/o MoE (大规模) | 13.7 | - | - | MoE贡献1.0 FID |
| w/ MoE (大规模) | **12.7** | - | - | 完整模型 |
| 仅真实数据 | 12.72 | - | 26.67 | FID相近但偏好差 |
| 真实+合成数据 | **12.66** | - | **28.14** | GPT-4o 63%偏好 |

### 关键发现
- 延迟遮蔽在 75% 遮蔽率下将 FID 从朴素遮蔽的 16.5 恢复到 5.03，接近无遮蔽的 3.79，是最核心的贡献
- 在 IsoFLOPs 对比中，延迟遮蔽（计算等价于缩小模型）在 75% 以下的遮蔽率上全面优于模型缩小策略
- 合成数据的价值被 FID 等指标严重低估——GPT-4o 偏好评估揭示了真正的质量差异
- 16 通道 VAE 在微预算训练中反而不如 4 通道 VAE，因为更高维的 latent 需要更多训练步数才能收敛
- SwiGLU 替代 GELU、高权重衰减、高 $\beta_2$ 等 LLM trick 在扩散 Transformer 上同样有效

## 亮点与洞察
- **延迟遮蔽的精妙之处**：仅在遮蔽前多加一步预处理（patch-mixer），就将不可行的 75% 遮蔽变为可行。这个设计简单到令人惊讶——为什么之前没有人想到？原因在于大家默认遮蔽必须在输入层发生，而作者打破了这个隐含假设
- **微预算训练的完整哲学**：不仅是一个技术改进，而是一套完整的低成本训练方法论——从遮蔽策略、架构改进到数据选择，每个决策都围绕"如何在有限预算内最大化性能"这一核心目标
- **合成数据的差异化价值**：在传统指标（FID）上看不出差异，但在人类偏好上差异巨大。这提醒我们评估指标本身可能是瓶颈——GPT-4o 偏好评估是更可靠的质量度量

## 局限与展望
- GenEval 组合生成能力（0.46）明显低于 SD-XL（0.55）和 SD-3（0.68），特别是在 counting 和 position 上较弱
- 文本渲染能力不足——这是所有开源模型的共同痛点，即使训练集中有 OCR 数据也未改善
- 使用 CLIP 而非 T5 文本编码器限制了复杂文本理解能力，但这是成本-性能 trade-off 的有意选择
- 仅在 512×512 分辨率上训练和评估，未验证更高分辨率的效果
- 可以探索渐进式遮蔽率策略：训练初期用高遮蔽率快速学习粗结构，后期逐步降低遮蔽率精化细节
- 将延迟遮蔽扩展到视频扩散 Transformer 训练有很大潜力

## 相关工作与启发
- **vs MaskDiT (Zheng et al. 2024)**: MaskDiT 在遮蔽后添加解码器和 MAE 损失来恢复被遮蔽 patch 的信息，但效果有限且增加了设计复杂度。MicroDiT 的延迟遮蔽在遮蔽前解决问题，更简洁且效果远好——75% 遮蔽下 FID 从约 15 降到 5
- **vs PixArt-α**: PixArt-α 是之前的低成本 SOTA，但仍需 85 天训练且使用了 10M 私有高质量图片。MicroDiT 仅用 2.6 天训练且全部使用公开数据集，成本低 14 倍
- **vs Würstchen**: Würstchen 通过 42× 极端图像压缩降低成本，但 FID（22.4）远低于 MicroDiT（12.7），说明过度压缩会严重损害生成质量

## 评分
- 新颖性: ⭐⭐⭐⭐ 延迟遮蔽概念简洁优雅，打破了"遮蔽必须在输入层"的隐含假设
- 实验充分度: ⭐⭐⭐⭐⭐ 消融极为详尽（41页论文、28图、5表），每个设计选择都有充分实验支撑
- 写作质量: ⭐⭐⭐⭐⭐ 写作清晰流畅，$1,890 的标题非常吸引眼球，动机-方法-实验逻辑链完整
- 价值: ⭐⭐⭐⭐⭐ 对整个扩散模型社区有重大影响——让更多研究者参与大规模扩散模型的训练和探索

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] AutoPresent: Designing Structured Visuals from Scratch](autopresent_designing_structured_visuals_from_scratch.md)
- [\[CVPR 2025\] Training Data Provenance Verification: Did Your Model Use Synthetic Data from My Generative Model for Training?](training_data_provenance_verification_did_your_model_use_synthetic_data_from_my_.md)
- [\[CVPR 2025\] Decoupling Training-Free Guided Diffusion by ADMM](decoupling_training-free_guided_diffusion_by_admm.md)
- [\[CVPR 2025\] Diff2Flow: Training Flow Matching Models via Diffusion Model Alignment](diff2flow_training_flow_matching_models_via_diffusion_model_alignment.md)
- [\[ICML 2026\] Budget-Constrained Step-Level Diffusion Caching](../../ICML2026/image_generation/budget-constrained_step-level_diffusion_caching.md)

</div>

<!-- RELATED:END -->
