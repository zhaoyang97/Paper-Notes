---
title: >-
  [论文解读] HierarchicalPrune: Position-Aware Compression for Large-Scale Diffusion Models
description: >-
  [AAAI 2026][图像生成][模型压缩] 提出 HierarchicalPrune，利用 MMDiT 扩散模型中块的层级功能差异（早期块建立语义结构、后期块处理纹理细节），通过层级位置剪枝（HPP）、位置权重保护（PWP）和敏感度引导蒸馏（SGDistill）三种技术协同，结合 INT4 量化，将 SD3.5 Large Turbo（8B）从 15.8GB 压缩至 3.24GB（79.5% 内存缩减），仅损失 4.8% 图像质量。
tags:
  - "AAAI 2026"
  - "图像生成"
  - "模型压缩"
  - "剪枝"
  - "知识蒸馏"
  - "MMDiT"
  - "Quantisation"
---

# HierarchicalPrune: Position-Aware Compression for Large-Scale Diffusion Models

**会议**: AAAI 2026  
**arXiv**: [2508.04663](https://arxiv.org/abs/2508.04663)  
**代码**: 无  
**领域**: 图像生成/模型压缩  
**关键词**: Diffusion Model Compression, Pruning, Knowledge Distillation, MMDiT, Quantisation

## 一句话总结
提出 HierarchicalPrune，利用 MMDiT 扩散模型中块的层级功能差异（早期块建立语义结构、后期块处理纹理细节），通过层级位置剪枝（HPP）、位置权重保护（PWP）和敏感度引导蒸馏（SGDistill）三种技术协同，结合 INT4 量化，将 SD3.5 Large Turbo（8B）从 15.8GB 压缩至 3.24GB（79.5% 内存缩减），仅损失 4.8% 图像质量。

## 研究背景与动机
当前最先进的文本到图像扩散模型（如 SD3.5、FLUX）已达到 8-11B 参数规模，虽然图像质量远超之前的 SDXL 和 SD1.5，但巨大的模型规模带来了严重的部署挑战：

**核心矛盾**：尽管量化指标（如 GenEval）显示 2B 参数的小模型（如 SANA-Sprint-1.6B）性能不错，但 Artificial Analysis 排行榜上的用户评估表明，小模型与大模型之间存在**量化指标无法捕捉的感知质量差距**。因此在资源受限场景部署大模型仍然是刚需。

**现有压缩方法的局限**：
1. **采样步骤减少和高效算子**：仅提速不减内存
2. **深度剪枝方法**（KOALA、BK-SDM）：在小模型（≤2.6B U-Net）上有效，但在 SD3.5（8B）和 FLUX（11B）等大模型上，仅 20-30% 内存缩减就会导致**38-45% 的严重质量退化**
3. 现有方法**全块移除**策略忽略了块内子组件的差异化重要性

**本文的关键发现**：基于 MMDiT 的扩散模型存在**双层级结构**：
- **块间层级**：不同位置的块对图像不同方面负责——早期块建立语义结构，后期块处理纹理细化
- **块内层级**：每个 MMDiT 块的子组件（Norm、Attention、MLP 等）重要性随位置和类型而异

## 方法详解

### 整体框架
HierarchicalPrune 是一个三阶段压缩框架：
1. **阶段一（HPP）**：基于层级位置感知进行块级和子组件级剪枝
2. **阶段二（PWP + SGDistill）**：冻结早期关键块 + 敏感度引导知识蒸馏恢复质量
3. **阶段三（PTQ）**：INT4 权重量化进一步压缩

### 关键设计

**设计一：层级位置剪枝（Hierarchical Position Pruning, HPP）**

HPP 基于核心洞察：后期 MMDiT 块对核心视觉结构贡献较小。首先，通过对校准集移除单个块/子组件，计算每个块的性能影响 $\Delta P(i,c)$。然后引入位置权重函数衡量块的可剪枝程度：

$$Score(i,c) = -|\Delta P(i,c)| \times W_{pos}(i)$$

$$W_{pos}(i) = e^{(i - |\mathcal{B}|) / |\mathcal{B}|}$$

其中 $i$ 为块索引，$|\mathcal{B}|$ 为总块数。$W_{pos}$ 是指数衰减函数，对后期块（$i$ 大）赋予更高权重，使其更容易被剪枝。与 KOALA 的余弦相似度或 BK-SDM 的 CLIP 分数排序不同，HPP 直接利用性能下降与位置的联合信息，在子组件粒度上进行精细剪枝。

**设计二：位置权重保护（Positional Weight Preservation, PWP）**

在蒸馏阶段，PWP 冻结未被剪枝的早期块的权重，仅允许后期、不太关键的块更新。这一简单策略确保对图像形成至关重要的结构性块保持完整：
- 中等压缩（20% 参数减少）：HPP alone 导致 79.4% 质量退化，加入 PWP 后仅 2.5%
- 早期块的权重保护是质量保持的关键

**设计三：敏感度引导蒸馏（Sensitivity-Guided Distillation, SGDistill）**

针对激进压缩（≥30%），即使有 PWP 仍会出现不可接受的质量下降（31.9%）。SGDistill 基于反直觉但有效的原则：**重要性越高的块对变化越敏感，强行更新反而有害**。

蒸馏损失由特征蒸馏损失和知识蒸馏损失组成：

$$\mathcal{L} = \mathcal{L}_{feat} + \mathcal{L}_{KD}$$

$$\mathcal{L}_{KD} = \mathbb{E}\left[\|v_{\boldsymbol{\theta'}}(x_t, t) - v_{\boldsymbol{\theta}}(x_t, t)\|^2\right]$$

$$\mathcal{L}_{feat} = \mathbb{E}\left[\sum_{i \in [0,|\mathcal{B}^*|-1]} \|f_{\boldsymbol{\theta'}}^{i'}(x_t, t) - f_{\boldsymbol{\theta}}^i(x_t, t)\|^2\right]$$

SGDistill 对每个块的参数更新按其敏感度的倒数 $\frac{1}{\Delta P(i,c)}$ 缩放——对最重要的块施加最小甚至零更新权重，将更新集中在不太敏感的组件上。这将质量退化从 31.9% 降至 10.1%。

### 损失函数 / 训练策略
- 蒸馏数据集：YE-POP（500K 图像）
- 4-bit 权重量化采用 bitsandbytes（W4A16）
- 目标模型：SD3.5 Large Turbo（8B参数）和 FLUX.1-Schnell（12B参数）
- 训练开销：仅需 615-1,287 A100 GPU 小时（对比从头训练 SD1.4/SD2.1 需 140k-200k A100 GPU 小时）
- 阈值 $r_{thres} = 0.25$，超过此压缩比启用 SGDistill

## 实验关键数据

### 主实验

**图像质量与内存压缩（SD3.5 Large Turbo）：**

| 方法 | 内存(GB) | 缩减比 | GenEval↑ | HPSv2↑ | 质量退化↓ |
|------|---------|--------|---------|--------|----------|
| Original | 15.8 (100%) | - | 0.71 | 30.29 | - |
| KOALA | 12.6 (79.4%) | 20.6% | 0.37 | 19.99 | 41.2% |
| KOALA+Quant | 3.56 (22.5%) | 77.5% | 0.33 | 18.44 | 46.4% |
| BK-SDM | 12.6 (79.4%) | 20.6% | 0.38 | 21.21 | 38.2% |
| BK-SDM+Quant | 3.56 (22.5%) | 77.5% | 0.34 | 19.83 | 43.3% |
| **Ours (HPP+PWP+Q)** | **3.56 (22.5%)** | **77.5%** | **0.69** | **28.15** | **4.8%** |
| **Ours (All)** | **3.24 (20.5%)** | **79.5%** | **0.62** | **26.29** | **13.3%** |
| SANA-Sprint-1.6B | 3.14 (100%) | - | 0.77 | 29.61 | - |

**FLUX.1-Schnell 结果**：Ours (All) 在 4.44GB（19.6%）下取得 GenEval 0.64、HPSv2 28.69，质量退化仅 3.2%，而 KOALA 在 15.9GB（70.5%）下退化 28.7%。

**用户研究（85人）**：HierarchicalPrune 文本对齐仅下降 4.8%，图像质量仅下降 5.3%；SANA-Sprint 分别下降 14.2% 和 11.1%；BK-SDM/KOALA 下降 44.0-52.2%。

**延迟对比（A6000 GPU）：**

| 模型 | 方法 | 延迟 | 缩减 |
|------|------|------|------|
| SD3.5 Large Turbo | Original | 823ms | - |
| SD3.5 Large Turbo | Ours (HPP+PWP+Q) | 593ms | 27.9% |
| FLUX.1-Schnell | Original | 756ms | - |
| FLUX.1-Schnell | Ours (All) | 469ms | 38.0% |

### 消融实验

**各组件贡献（SD3.5 Large Turbo）：**

| 压缩程度 | 方法 | GenEval↑ | HPSv2↑ | 质量退化↓ |
|---------|------|---------|--------|----------|
| 无(0%) | Original | 0.71 | 30.29 | - |
| 中等(20%) | HPP only | 0.03 | 11.08 | 79.4% |
| 中等(20%) | +PWP | 0.71 | 28.97 | 2.5% |
| 中等(20%) | +Quant | 0.69 | 28.15 | 4.8% |
| 激进(30%) | HPP only | 0.0 | 7.00 | 88.4% |
| 激进(30%) | +PWP | 0.46 | 21.74 | 31.9% |
| 激进(30%) | +SGDistill | 0.64 | 27.29 | 10.1% |
| 激进(30%) | +Quant | 0.62 | 26.29 | 13.3% |

关键观察：HPP alone 在中等压缩下就崩溃（GenEval 0.03），PWP 是质量恢复的核心。SGDistill 在激进压缩下将退化从 31.9% 降至 10.1%。量化额外引入 2.4-3.5% 质量损失。

### 关键发现
- 早期 MMDiT 块移除导致图像结构剧变，后期块移除仅影响精细风格细节
- 压缩后模型（3.24GB）内存与 SANA-Sprint（3.14GB）相当，但图像质量显著更优
- HierarchicalPrune 保留了原模型的文字绘制能力，而 SANA-Sprint 和其它压缩方法无法做到

## 亮点与洞察
- **双层级洞察**是本文的核心贡献：将 MMDiT 的块间功能差异和块内子组件差异系统化，为大规模扩散模型压缩提供了新范式
- **反直觉的蒸馏策略**：对最重要的块施加最小更新，与传统蒸馏中重点更新关键层的思路相反，但在激进压缩下效果显著
- 首次将深度剪枝 + 知识蒸馏 + INT4 量化统一到一个框架中用于 8B+ 级别扩散模型

## 局限与展望
- 仅验证了 SD3.5 和 FLUX 两个 MMDiT 架构，未测试 U-Net 或其他生成模型架构
- 激进压缩（30%）下的 13.3% 质量退化在某些应用场景可能仍不可接受
- 蒸馏成本（615-1,287 A100 GPU 小时）对学术团队而言仍较高
- 用户研究仅 85 人，样本量相对有限

## 相关工作与启发
- **vs KOALA**: 同为深度剪枝方法，但 KOALA 使用块级余弦相似度排序、全块移除，在 SD3.5 上退化 41.2%；HierarchicalPrune 通过位置感知 + 子组件级精细剪枝仅退化 4.8%
- **vs BK-SDM**: BK-SDM 基于 CLIP 分数的块剪枝退化 38.2%，缺乏对 MMDiT 层级结构的理解
- **vs SANA-Sprint-1.6B**: 从头训练的紧凑模型用户感知质量下降 11.1-14.2%，而压缩大模型仅下降 4.8-5.3%，证明「压缩优于训小模型」的路线可行性

## 评分
- 新颖性: ⭐⭐⭐⭐ 双层级结构洞察和反直觉蒸馏策略具有很强的原创性
- 实验充分度: ⭐⭐⭐⭐⭐ 定量+定性+85人用户研究+多GPU延迟测试+完整消融，非常充分
- 写作质量: ⭐⭐⭐⭐ 结构清晰，动机论证扎实，实验叙述详尽
- 价值: ⭐⭐⭐⭐⭐ 首次实现 8B+ 扩散模型的有效压缩，从 15.8GB 到 3.24GB，对实际部署意义重大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Compositional amortized inference for large-scale hierarchical Bayesian models](../../ICLR2026/image_generation/compositional_amortized_inference_for_large-scale_hierarchical_bayesian_models.md)
- [\[AAAI 2026\] HACK: Head-Aware KV Cache Compression for Efficient Visual Autoregressive Modeling](head-aware_kv_cache_compression_for_efficient_visual_autoreg.md)
- [\[CVPR 2026\] CG-Floor: Centroid-Guided Diffusion for Large-Scale Floorplan Generation](../../CVPR2026/image_generation/cg-floor_centroid-guided_diffusion_for_large-scale_floorplan_generation.md)
- [\[ICLR 2026\] Large Scale Diffusion Distillation via Score-Regularized Continuous-Time Consistency](../../ICLR2026/image_generation/large_scale_diffusion_distillation_via_score-regularized_continuous-time_consist.md)
- [\[AAAI 2026\] Margin-aware Preference Optimization for Aligning Diffusion Models without Reference](margin-aware_preference_optimization_for_aligning_diffusion_models_without_refer.md)

</div>

<!-- RELATED:END -->
