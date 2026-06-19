---
title: >-
  [论文解读] Flowing Backwards: Improving Normalizing Flows via Reverse Representation Alignment
description: >-
  [AAAI 2026][图像生成][Normalizing Flows] 提出 R-REPA（Reverse Representation Alignment），创造性地利用 Normalizing Flows 的可逆性，在生成（反向）路径上将中间特征与视觉基础模型对齐，同时提出免训练分类算法，在 ImageNet 64×64 和 256×256 上实现 NF 新 SOTA，训练加速 3.3 倍。
tags:
  - "AAAI 2026"
  - "图像生成"
  - "Normalizing Flows"
  - "TARFlow"
  - "表征对齐"
  - "反向对齐"
  - "ImageNet"
---

# Flowing Backwards: Improving Normalizing Flows via Reverse Representation Alignment

**会议**: AAAI 2026  
**arXiv**: [2511.22345](https://arxiv.org/abs/2511.22345)  
**代码**: 无  
**领域**: 生成模型 / 正则化流 / 表征对齐  
**关键词**: Normalizing Flows, TARFlow, 表征对齐, 反向对齐, 图像生成, ImageNet

## 一句话总结

提出 R-REPA（Reverse Representation Alignment），创造性地利用 Normalizing Flows 的可逆性，在生成（反向）路径上将中间特征与视觉基础模型对齐，同时提出免训练分类算法，在 ImageNet 64×64 和 256×256 上实现 NF 新 SOTA，训练加速 3.3 倍。

## 研究背景与动机

Normalizing Flows（NFs）是一类具有**精确数学可逆性**的生成模型：前向路径将数据映射到潜空间用于密度估计，反向路径从潜空间生成新样本。这种双向结构本质上在表征学习与数据生成之间形成了天然协同——两者是同一枚硬币的两面。

然而，标准 NF 仅优化前向路径的最大似然估计（MLE），学到的中间特征缺乏语义意义，限制了生成质量。近期 REPA 方法在扩散模型中展示了"表征优先"策略的威力——将去噪网络的内部特征与预训练视觉编码器对齐，大幅提升训练效率和生成质量。

本文核心问题：**能否利用 NF 独有的可逆结构，设计更优的表征对齐策略？** 不同于扩散模型只有前向路径可操作，NF 的反向生成路径提供了全新的对齐可能性。作者发现，在生成路径（z→x）上对齐特征比在编码路径（x→z）上对齐更有效，实现了生成质量和判别能力的双重提升。

## 方法详解

### 整体框架

方法构建在 TARFlow（Transformer AutoRegressive Flow）之上，包含三个核心贡献：

1. **免训练分类算法**：利用条件 NF 的密度估计能力，通过单步梯度实现测试时分类
2. **反向表征对齐（R-REPA）**：在生成路径上将 NF 中间特征与视觉基础模型对齐
3. **潜空间扩展**：将 TARFlow 迁移至 VAE 潜空间实现高分辨率生成

### 关键设计一：免训练分类算法

传统评估 NF 判别能力需为每层单独训练线性分类器（linear probing），开销大且不直接。本文提出直接利用模型的密度估计进行分类：

1. 定义分类 logits $\boldsymbol{\lambda} \in \mathbb{R}^K$，初始化为零
2. 计算加权类别嵌入 $\mathbf{e}_{\text{eff}} = \text{softmax}(\boldsymbol{\lambda})^T \mathbf{E}$
3. 计算条件对数似然 $\mathcal{L}(\boldsymbol{\lambda}) = \log p(\mathbf{x} | \mathbf{e}_{\text{eff}}; \theta)$
4. 对 logits 求梯度，取梯度最大分量对应的类别作为预测

整个过程仅需**一次前向+反向传播**，无需训练任何额外参数。实验验证该方法的分类精度与标准 linear probing 的最佳层结果一致，是更高效、更本质的语义评估指标。

### 关键设计二：三种对齐策略的系统探索

给定预训练冻结的视觉编码器 $\Phi(\cdot)$，对齐损失为：

$$\mathcal{L}_{\text{align}}^{(t,l)}(\theta, \phi) = -\frac{1}{P} \sum_{p=1}^{P} \text{sim}\left(\mathbf{v}^{[p]}, [\text{Proj}_\phi(\mathbf{h}^{(t,l)})]^{[p]}\right)$$

其中 $\mathbf{h}^{(t,l)}$ 为 TARFlow 第 $t$ 个 block 第 $l$ 层的特征，$\text{Proj}_\phi$ 为可学习 MLP 投影头。作者系统比较了三种梯度回传策略：

| 策略 | 梯度路径 | 更新范围 | 核心思想 |
|------|---------|---------|---------|
| Forward（F-REPA） | 前向计算图 | 对齐层之前所有 block | 直接类比 REPA |
| Detach（D-REPA） | 截断输入梯度 | 仅当前 block | 类比扩散模型的 timestep 隔离 |
| **Reverse（R-REPA）** | **反向生成计算图** | **对齐层之后所有 block** | **NF 独有：在 $f_\theta^{-1}$ 上回传** |

R-REPA 的实现：先通过前向路径得到 $\mathbf{z} = f_\theta(\mathbf{x})$，将 $\mathbf{z}$ detach 后执行反向生成 $f_\theta^{-1}$，在反向路径的中间特征上计算对齐损失。梯度通过生成路径回传，仅更新生成路径中对齐层之后的参数，不干扰前向密度建模。

### 关键设计三：加速伪反向实现

朴素的反向路径是自回归的（每个 token 依赖前面所有 token），无法并行。本文设计了加速实现：

1. 前向传播时缓存每个 block 的输入 $\hat{\mathbf{x}}^{t-1} = \text{stop\_gradient}(\mathbf{x}^{t-1})$
2. 伪反向时用缓存的 $\hat{\mathbf{x}}^{t-1}$ 提供自回归上下文，将反向计算并行化
3. 由于可逆性，伪反向输出与缓存数值相同，但构建了有效的反向计算图

加速效果：相比朴素反向实现加速约 **50 倍**，显存降低约 **50%**。

### 训练策略

总损失为标准 NF 损失与对齐损失的加权和：

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{NF}} + \lambda_{\text{align}} \cdot \frac{1}{|\mathcal{A}|} \sum_{(t,l) \in \mathcal{A}} \mathcal{L}_{\text{align}}^{(t,l)}$$

最优配置：在 Block 7 & 8 的第 6 层对齐（即生成路径最先处理潜变量 $\mathbf{z}$ 的 block），引导生成初始阶段建立正确的高层图像结构。

对于 256×256 分辨率，使用预训练 VAE 编码器将图像压缩至潜空间，NF 在潜空间上建模，训练时对潜向量添加噪声（$\sigma=0.20$），生成时先采样再用 score-based 去噪。

## 实验

### 核心消融实验：三种对齐策略对比（ImageNet 64×64, 400K iter）

| 策略 | 对齐位置 | FID ↓ | sFID ↓ | IS ↑ | Acc.(%) ↑ |
|------|---------|-------|--------|------|-----------|
| TARFlow baseline | — | 12.91 | 33.79 | 36.62 | 37.43 |
| Forward | All blocks | 12.25 | 37.97 | 40.85 | 46.97 |
| Detach | All blocks | 12.19 | 34.31 | 41.98 | 49.06 |
| Reverse | All blocks | 12.21 | 33.80 | 42.08 | 49.91 |
| Forward | Block 1&2 | 12.67 | 39.99 | 41.11 | 61.16 |
| Detach | Block 7&8 | 12.12 | 34.00 | 41.18 | 55.14 |
| **Reverse** | **Block 7&8** | **11.93** | **33.78** | 40.90 | 55.21 |
| **Reverse** | **Block 7&8, L6** | **11.71** | 33.68 | 44.31 | 57.35 |

关键发现：
- Forward 策略系统性恶化 sFID（33.79→37.97），因为无约束梯度在早期 block 造成 MLE 与对齐损失的冲突
- Reverse 策略在 FID 上一致优于 Detach（11.93 vs 12.12），因为仅更新生成路径不干扰密度建模
- 对齐 Block 7&8（生成路径最先处理 $\mathbf{z}$ 的位置）获得最佳 FID，而 Block 1&2 获得最佳精度——生成质量与判别能力存在权衡

### 与 baseline 的训练效率对比（ImageNet 64×64 & 256×256）

| 模型 | 分辨率 | 训练迭代 | FID ↓ | Acc.(%) ↑ |
|------|--------|---------|-------|-----------|
| TARFlow | 64 | 1M | 11.76 | 39.97 |
| +R-REPA | 64 | 400K | 11.71 | 57.76 |
| +R-REPA | 64 | 1M | **11.25** | 57.02 |
| Latent-TARFlow | 256 | 1M | 13.05 | 40.22 |
| +R-REPA | 256 | 1M | **12.79** | **56.24** |

R-REPA 在仅 400K 迭代时已超越 baseline 的 1M 迭代结果（FID 11.71 vs 11.76），实现 **3.3× 训练加速**。分类精度从 39.97% 跃升至 57.76%（+17.8 pp）。

### SOTA 对比：ImageNet 64×64

| 模型类别 | 模型 | FID ↓ | sFID ↓ |
|---------|------|-------|--------|
| 扩散模型 | EDM | 1.36 | — |
| 扩散模型 | ADM | 2.09 | 4.29 |
| GAN | BigGAN | 4.06 | 3.96 |
| 一致性模型 | iCT-deep | 3.25 | — |
| NF | TARFlow | 4.21 | 5.34 |
| **NF** | **+R-REPA (Ours)** | **3.69** | **4.34** |

R-REPA 将 NF 的 FID 从 4.21 降至 3.69（-12.4%），首次超越 BigGAN（4.06），且仅需**两步采样**。

### SOTA 对比：ImageNet 256×256

| 模型 | FID ↓ | sFID ↓ | IS ↑ |
|------|-------|--------|------|
| ADM | 4.59 | 5.25 | 186.70 |
| DiT | 2.27 | 4.60 | 278.24 |
| SiT | 2.06 | 4.50 | 270.30 |
| VAR | 1.73 | — | 350.2 |
| Latent-TARFlow | 5.15 | 6.78 | 243.49 |
| **+R-REPA + Patch1** | **4.18** | **4.96** | 240.8 |

在 256×256 上 FID 从 5.15 降至 4.18（-18.8%），与 ADM（4.59）处于可比水平，但推理效率远高于扩散模型。

## 亮点与洞察

1. **精准利用架构特性的方法设计**：R-REPA 不是简单地将 REPA 套用到 NF 上，而是深度挖掘 NF 可逆性这一独有属性，在反向生成路径上做对齐。这种"结构决定策略"的设计思路值得借鉴——不同生成模型应设计不同的增强方案。

2. **加速伪反向的工程智慧**：自回归反向路径的朴素实现无法用于训练，但通过缓存前向特征并 detach，将 $O(D)$ 的串行计算变为 $O(1)$ 的并行计算，同时保持数值一致性和正确的梯度流。这一技巧50× 加速使 R-REPA 在实际中可行。

3. **免训练分类作为语义探针**：比 linear probing 更高效也更本质——直接利用生成模型的密度估计能力判别，验证了"生成即理解"的假设，也为 NF 的判别能力评估提供了标准化工具。

4. **生成路径 vs 编码路径的精确分析**：Forward REPA 恶化 sFID 的原因分析（低层 block 负责低级空间统计，强制对齐高级语义会破坏）体现了对模型内部分工的深入理解。

## 局限性

1. **与扩散模型差距仍大**：在 ImageNet 256×256 上 FID 4.18 vs SiT 2.06 / VAR 1.73，NF 在绝对生成质量上仍有明显差距
2. **依赖预训练视觉编码器**：对齐需要 frozen 视觉基础模型提供监督，引入额外的模型依赖和计算开销
3. **生成质量与判别能力的权衡**：Block 7&8 对齐最优 FID，Block 1&2 最优精度，无法同时最优化两者。当前配置偏向生成质量
4. **仅在 TARFlow 上验证**：未在 RealNVP、Glow 等其他 NF 架构上测试，R-REPA 的通用性有待验证
5. **分辨率受限**：仅在 64×64 和 256×256 上实验，未探索更高分辨率（512+）的可扩展性

## 相关工作

| 方向 | 代表工作 | 与本文关系 |
|------|---------|-----------|
| NF 架构 | TARFlow, JetFormer, FARMER | 本文基于 TARFlow 构建 |
| 表征对齐加速生成 | REPA (扩散模型) | 直接启发，但本文提出 NF 特有的反向对齐 |
| REPA 扩展 | REPA-E, LightningDiT, U-REPA | 均在扩散模型前向路径上工作，未涉及可逆架构 |
| NF 扩展 | STARFlow | 并发工作，扩展 NF 规模和任务复杂度 |

## 评分

- **新颖性**: ⭐⭐⭐⭐ (反向路径对齐是 NF 独有的创新点，非直接套用 REPA)
- **技术贡献**: ⭐⭐⭐⭐ (加速伪反向实现 + 免训练分类 + 系统性策略对比)
- **实验充分度**: ⭐⭐⭐⭐ (多分辨率、完整消融、SOTA 对比、训练效率分析)
- **实用性**: ⭐⭐⭐ (NF 本身在实际应用中不如扩散模型主流，但方法思路有迁移价值)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] The Coupling Within: Flow Matching via Distilled Normalizing Flows](../../ICML2026/image_generation/the_coupling_within_flow_matching_via_distilled_normalizing_flows.md)
- [\[ICML 2025\] Normalizing Flows are Capable Generative Models](../../ICML2025/image_generation/normalizing_flows_are_capable_generative_models.md)
- [\[AAAI 2026\] Unleashing the Potential of Large Language Models for Text-to-Image Generation through Autoregressive Representation Alignment](unleashing_the_potential_of_large_language_models_for_text-to-image_generation_t.md)
- [\[CVPR 2026\] SRA 2: Variational Autoencoder Self-Representation Alignment for Efficient Diffusion Training](../../CVPR2026/image_generation/sra_2_variational_autoencoder_self-representation_alignment_for_efficient_diffus.md)
- [\[CVPR 2026\] Bidirectional Normalizing Flow: From Data to Noise and Back](../../CVPR2026/image_generation/bidirectional_normalizing_flow_from_data_to_noise_and_back.md)

</div>

<!-- RELATED:END -->
