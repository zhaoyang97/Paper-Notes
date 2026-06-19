---
title: >-
  [论文解读] BAMM: Bidirectional Autoregressive Motion Model
description: >-
  [ECCV 2024][图像恢复][文本到动作生成] 提出 BAMM（双向自回归运动模型），通过统一生成掩码建模和自回归建模的混合注意力掩码策略，在一个框架中同时实现高质量运动生成、自适应长度预测和零样本运动编辑，在 HumanML3D 和 KIT-ML 上全面超越 SOTA。 文本到3D人体运动生成是连接自然语言与人体运动…
tags:
  - "ECCV 2024"
  - "图像恢复"
  - "文本到动作生成"
  - "双向自回归"
  - "掩码自注意力"
  - "运动编辑"
  - "VQ-VAE"
---

# BAMM: Bidirectional Autoregressive Motion Model

**会议**: ECCV 2024  
**arXiv**: [2403.19435](https://arxiv.org/abs/2403.19435)  
**代码**: [https://exitudio.github.io/BAMM-page](https://exitudio.github.io/BAMM-page)  
**领域**: 图像修复 / 人体动作生成  
**关键词**: 文本到动作生成, 双向自回归, 掩码自注意力, 运动编辑, VQ-VAE

## 一句话总结

提出 BAMM（双向自回归运动模型），通过统一生成掩码建模和自回归建模的混合注意力掩码策略，在一个框架中同时实现高质量运动生成、自适应长度预测和零样本运动编辑，在 HumanML3D 和 KIT-ML 上全面超越 SOTA。

## 研究背景与动机

文本到3D人体运动生成是连接自然语言与人体运动的重要任务，在动画、游戏、VR/AR 领域有广泛应用。当前主流方法分为两大流派，但各有本质缺陷：

**条件去噪运动模型**（扩散/生成掩码类，如 MDM、MoMask、MMM）：通过双向上下文捕获 token 间的丰富依赖关系，生成质量高且天然支持运动编辑。但**致命缺陷是需要预先知道运动长度**——实际场景中运动长度未知，使用不准确的长度估计会导致生成质量显著下降。

**条件自回归运动模型**（GPT 类，如 T2M-GPT、AttT2M）：通过逐 token 生成直到预测出 [END] token 来自适应确定运动长度，可用性强。但**单向因果注意力无法捕获全局双向依赖**，生成质量受限，且无法进行时序运动编辑（编辑区域需要双向上下文）。

**核心矛盾**：可用性（长度预测）、生成质量（双向依赖）、可编辑性三者之间存在根本性权衡。

**核心 idea**：通过混合注意力掩码将单向自回归和双向生成掩码建模统一到一个 Transformer 框架中——单向掩码负责长度预测+粗粒度生成，双向掩码负责质量提升+运动编辑，一举打破三者的权衡。

## 方法详解

### 整体框架

BAMM 采用两阶段训练：(1) 训练 VQ-VAE 运动分词器将原始3D运动序列编码为离散 token；(2) 训练条件掩码自注意力 Transformer，通过混合注意力掩码策略自回归预测被掩码的 token。推理时采用级联运动解码：先用单向自回归生成粗粒度序列并确定长度，再用双向自回归精炼低置信度 token。

### 关键设计

1. **运动分词器（Motion Tokenizer）**：基于 VQ-VAE，将运动序列 $\mathcal{M}=[m_1,...,m_\tau]$（$m \in \mathbb{R}^D$）通过编码器映射到潜空间嵌入 $z \in \mathbb{R}^{t \times d}$，然后通过最近邻查找量化到码本 $\mathcal{C} = \{\gamma_k\}_{k=1}^K$：
    $\hat{z_i} = \arg\min_j \|z - \mathcal{C}_j\|_2^2$
   训练损失为标准 VQ 损失：$L_{VQ} = \|\text{sg}(z) - e\|_2^2 + \beta\|z - \text{sg}(e)\|_2^2$。使用指数移动平均更新码本和码本重置策略。

2. **条件掩码自注意力 Transformer**：本文核心创新。输入为 CLIP 文本嵌入 $x_0$、运动 token $x_{1:t}$ 和 [END] token $x_{t+1}$。不同于传统掩码模型用 [MASK] token 替换输入，BAMM 直接修改注意力分数矩阵，设计了两种因果掩码：

    - **单向因果掩码 $M^{uc}$**：仅文本 token 可被双向注意，所有运动 token 只能看到左侧和自身，等价于标准自回归
    - **双向因果掩码 $M^{bc}$**：文本和 [END] token 为未掩码 token，可被双向注意；被掩码的运动 token 可看到左侧所有 token 和右侧未掩码 token

   掩码矩阵定义：$M_{ij} = 0$ 当 $(i \geq j \land i \notin U) \vee (j \in U)$，否则 $-\infty$。其中 $U$ 为未掩码 token 索引集。

    - 设计动机：这种设计保持了自回归的因果性（可预测 [END]），同时允许模型利用未掩码的"未来"token，实现了真正的双向条件生成。

3. **级联运动解码（Cascaded Motion Decoding）**：推理时的两阶段解码策略：

    - **第一轮**（单向自回归）：使用 $M^{uc}$ 逐 token 采样直到预测 [END]，确定运动长度并生成粗粒度序列
    - **第二轮**（双向自回归精炼）：使用 $M^{bc}$ 掩码掉一部分 token（如隔一个掩一个），利用双向上下文重新预测低置信度 token
    - 两轮使用不同的 CFG（Classifier-free Guidance）尺度：第一轮 $s=4$，第二轮 $s=3$

4. **残差运动精炼（Residual Motion Refinement）**：通过残差向量量化（RVQ）进一步减少量化损失。第一个量化器的 token 由主 Transformer 生成，剩余量化器的 token 由另一个精炼 Transformer 预测，最终合并解码。

### 损失函数 / 训练策略

**混合注意力掩码训练**：以概率 $\lambda=0.5$ 随机选择单向或双向掩码，最小化负对数似然：

$$\mathcal{L}_{\text{hybrid}} = -\mathbb{E}_{\mathbf{X} \in p(\mathbf{X})}\left[\lambda \sum_{\forall i \in [1,t]} \log p_\theta(x_i | M^{uc}) + (1-\lambda) \sum_{\forall i \in [1,t]} \log p_\theta(x_i | M^{bc})\right]$$

选择双向掩码时，随机将 50%-100% 的运动 token 放入掩码区域。训练时随机丢弃文本 token 以支持无分类器引导。

推理时使用混合 CFG：$\ell_g = (1+s) \cdot \ell_c - s \cdot \ell_u$。

## 实验关键数据

### 主实验

**HumanML3D 数据集对比**（20次评估取平均，95%置信区间）：

| 方法 | 长度预测 | R-Precision Top-1↑ | FID↓ | MM-Dist↓ | 可编辑 |
|------|---------|-------------------|------|----------|--------|
| T2M-GPT | ✓ | 0.491 | 0.116 | 3.118 | ✗ |
| AttT2M | ✓ | 0.499 | 0.112 | 3.038 | ✗ |
| MMM§ | ✗ | 0.504 | 0.080 | 2.998 | ✓ |
| MoMask§ | ✗ | 0.521 | 0.045 | 2.958 | ✓ |
| **BAMM** | **✓** | **0.525** | **0.055** | **2.919** | **✓** |

§ 表示使用真实运动长度。BAMM 无需真实长度即超越使用真实长度的方法。

**KIT-ML 数据集对比**：

| 方法 | R-Precision Top-1↑ | FID↓ | MM-Dist↓ |
|------|-------------------|------|----------|
| T2M-GPT | 0.402 | 0.717 | 3.053 |
| MoMask§ | 0.433 | 0.204 | 2.779 |
| **BAMM** | **0.438** | **0.183** | **2.723** |

### 消融实验

**级联解码中 CFG 尺度与掩码策略消融**（HumanML3D）：

| 消融项 | 配置 | R-Prec Top-1↑ | FID↓ | MM-Dist↓ |
|--------|------|---------------|------|----------|
| 第一轮CFG | s=2 / s=3 | 0.517 | 0.105 | 2.956 |
| 第一轮CFG | s=4 / s=3 (默认) | **0.525** | **0.055** | **2.919** |
| 第一轮CFG | s=5 / s=3 | 0.522 | 0.052 | 2.927 |
| 掩码策略 | 50%低置信度 | 0.525 | 0.065 | 2.921 |
| 掩码策略 | 置信度<0.5 | 0.525 | 0.064 | 2.923 |
| 掩码策略 | suffix(前50%) | 0.519 | 0.052 | 2.943 |
| 掩码策略 | **隔一个掩一个(默认)** | **0.525** | **0.055** | **2.919** |
| 迭代次数 | 1轮（仅自回归） | 0.524 | 0.064 | 2.926 |
| 迭代次数 | **2轮（默认）** | **0.525** | **0.055** | **2.919** |
| 迭代次数 | 3轮 | 0.525 | 0.055 | 2.917 |

**预测长度 vs 真实长度**（HumanML3D）：

| 长度来源 | 方法 | FID↓ | R-Prec Top-1↑ |
|---------|------|------|---------------|
| 真实长度 | MoMask | 0.045 | 0.521 |
| 预测长度 | MoMask | 0.090 (+100%) | 0.522 |
| 真实长度 | BAMM | 0.055 | 0.522 |
| 自适应长度 | **BAMM** | **0.055** | **0.525** |

**运动编辑任务**（HumanML3D，与 MDM 和 MoMask 对比）：

| 任务 | 方法 | R-Prec Top-1↑ | FID↓ |
|------|------|---------------|------|
| Inpainting | MDM | 0.391 | 2.362 |
| Inpainting | MoMask | 0.534 | 0.040 |
| Inpainting | **BAMM** | **0.535** | 0.056 |
| Outpainting | MoMask | 0.531 | 0.057 |
| Outpainting | **BAMM** | **0.535** | **0.056** |

### 关键发现

- 预测长度对去噪模型有显著影响：MoMask 的 FID 从 0.045 劣化到 0.090（翻倍），而 BAMM 不受此影响
- 双向精炼（第二轮迭代）使 FID 从 0.064 降至 0.055，但三轮迭代无明显提升，两轮即可
- 简单的"隔一个掩一个"掩码策略效果最好，比基于置信度的策略更稳定
- BAMM 在零样本运动编辑任务中表现与专门训练的 MoMask 持平，显著优于 MDM

## 亮点与洞察

- **统一框架解决三重权衡**：首次在一个模型中同时实现长度自适应(可用性)、高质量生成和零样本编辑，打破了去噪模型和自回归模型各自的局限
- **掩码设计精巧**：不替换为 [MASK] token 而是直接修改注意力矩阵的做法，避免了训练-推理不一致问题，是一个优雅的设计选择
- **级联解码策略**：粗生成→精炼的两阶段思路，类似于 coarse-to-fine 的思想在离散运动生成中的体现
- **完备的能力矩阵**：见 Table 1，BAMM 是唯一同时支持长度预测、长度输入和编辑的方法

## 局限与展望

- 级联解码需要两轮推理，虽然自回归生成已经较快，但仍比单次前向推理慢
- 残差向量量化增加了模型复杂度（需要额外的精炼 Transformer）
- VQ-VAE 分词器的码本大小和下采样率对最终生成质量有影响，但论文未详细讨论选择策略
- 仅在 HumanML3D 和 KIT-ML 两个英文描述数据集上验证，对更多样的运动类型和语言的泛化性未知
- 长序列运动生成虽然展示了可行性，但缺少定量评估

## 相关工作与启发

- **T2M-GPT (CVPR 2023)**：GPT 风格的自回归运动生成先驱，BAMM 在此基础上引入双向能力
- **MoMask (2023)**：BERT 风格的生成掩码运动模型，与 BAMM 的双向精炼阶段思路相近
- **MMM (2023)**：掩码运动建模，面临与 MoMask 相同的长度依赖问题
- **GLM/CM3Leon**：大语言模型中的掩码+自回归统一训练启发了 BAMM 的混合注意力设计
- **SoundStorm**：音频生成中的 RVQ 技术被迁移到运动精炼模块

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 统一掩码和自回归建模的混合掩码策略设计非常巧妙，解决了领域内公认的三重权衡
- 实验充分度: ⭐⭐⭐⭐ 两个标准数据集、全面的SOTA对比、丰富的消融分析和编辑任务评估
- 写作质量: ⭐⭐⭐⭐ 动机阐述清晰，方法图示直观，问题建模形式化严谨
- 价值: ⭐⭐⭐⭐ 统一框架的思路具有方法论意义，对运动生成领域有明确推进

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] OMoBlur: An Object Motion Blur Dataset and Benchmark for Real-World Local Motion Deblurring](../../CVPR2026/image_restoration/omoblur_an_object_motion_blur_dataset_and_benchmark_for_real-world_local_motion_.md)
- [\[ECCV 2024\] MambaIR: A Simple Baseline for Image Restoration with State-Space Model](mambair_a_simple_baseline_for_image_restoration_with_state-space_model.md)
- [\[ECCV 2024\] Spatially-Variant Degradation Model for Dataset-free Super-resolution](spatially-variant_degradation_model_for_dataset-free_super-resolution.md)
- [\[CVPR 2026\] Bi-Bridge: Bidirectional Diffusion Bridges for Low-Light Image Enhancement](../../CVPR2026/image_restoration/bi-bridge_bidirectional_diffusion_bridges_for_low-light_image_enhancement.md)
- [\[ECCV 2024\] Joint RGB-Spectral Decomposition Model Guided Image Enhancement in Mobile Photography](joint_rgb-spectral_decomposition_model_guided_image_enhancement_in_mobile_photog.md)

</div>

<!-- RELATED:END -->
