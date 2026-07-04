---
title: >-
  [论文解读] RollingQ: Reviving the Cooperation Dynamics in Multimodal Transformer
description: >-
  [ICML2025][多模态VLM][多模态融合] 揭示多模态 Transformer 中自注意力机制因"自增强循环"导致动态适应性失效（偏向单一模态），并提出 RollingQ 算法通过旋转 query 向量打破这一循环，恢复跨模态协作动态。 多模态学习的核心挑战是如何有效融合来自不同模态的信息。现有融合范式分为： - 静…
tags:
  - "ICML2025"
  - "多模态VLM"
  - "多模态融合"
  - "注意力机制"
  - "模态偏置"
  - "动态融合"
  - "Transformer"
---

# RollingQ: Reviving the Cooperation Dynamics in Multimodal Transformer

**会议**: ICML2025  
**arXiv**: [2506.11465](https://arxiv.org/abs/2506.11465)  
**代码**: [GeWu-Lab/RollingQ_ICML2025](https://github.com/GeWu-Lab/RollingQ_ICML2025)  
**领域**: 多模态VLM  
**关键词**: 多模态融合, 注意力机制, 模态偏置, 动态融合, Transformer

## 一句话总结

揭示多模态 Transformer 中自注意力机制因"自增强循环"导致动态适应性失效（偏向单一模态），并提出 RollingQ 算法通过旋转 query 向量打破这一循环，恢复跨模态协作动态。

## 研究背景与动机

多模态学习的核心挑战是如何有效融合来自不同模态的信息。现有融合范式分为：

- **静态融合**：推理时对各模态使用固定权重，假设模态贡献一致
- **动态融合**：基于输入数据特性自适应调整各模态权重（如 Transformer 注意力机制）

作者在 Kinetic-Sound 数据集上发现一个反直觉现象：**基于自注意力的动态融合（67.0%）竟然比简单的静态融合拼接（68.0%）还差**。

深入分析注意力分数后发现，模型对音频模态分配了不成比例的高注意力，**即使将音频输入替换为高斯噪声，模型仍然过度关注音频**。这说明注意力机制完全失去了动态适应性。

**根因分析**：注意力 key 分布在不同模态间存在显著差距（distribution gap）。偏置模态的 key 与 class token 的 query 余弦相似度始终更高，导致一个"自增强循环"：

1. 前向传播：偏置模态获得更高注意力分数
2. 反向传播：更高的注意力分数为偏置模态编码器提供更大梯度
3. 偏置模态特征质量进一步提升 → 回到步骤1

这个循环持续加剧模态间 key 分布差距，最终使注意力机制丧失动态调整能力。

## 方法详解

### 核心框架

给定双模态输入 $x_i = (x_i^a, x_i^v)$，经各自编码器得到特征 $z_i^m = \Phi^m(x_i^m; \theta^m)$。在注意力层中，class token 的 query $q = z_{cls} W^Q$ 与各模态的 key 做点积得到注意力分数：

$$A_i = \frac{q K_i^T}{\sqrt{d}}, \quad h_i = \text{softmax}(A_i) V_i$$

其中 $K_i = [K_i^a, K_i^v]$。对模态 $m$ 的注意力分数可以分解为：

$$\sum_{j=1}^{L^m} \frac{q k_{(i,j)}^m}{\sqrt{d}} = \frac{L^m}{\sqrt{d}} \|q\|_2 \|\hat{k}_i^m\|_2 \cos\theta_i^m$$

其中 $\hat{k}_i^m$ 是模态 $m$ 的平均 key，$\cos\theta_i^m$ 是 query 与平均 key 的余弦相似度。

### 自增强循环的理论分析

- **训练初期**：$\mathbb{E}[Q] = 0$，$Q$ 与 $\hat{K}^m$ 独立，因此两个模态的注意力分数期望相同
- **训练过程中**：由于多模态学习的贪婪性，模型偏向特征质量更高的模态，造成 key 分布差距
- **梯度分析**：偏置模态获得更大的 $s(\cdot) \frac{\partial V_i^m}{\partial z_i^m}$ 项，进一步强化其编码器优化

### RollingQ 算法

**步骤一：检测自增强循环** — 定义 Attention Imbalance Rate (AIR)：

$$AIR = \mathbb{E}[\cos\theta^a - \cos\theta^v] \in [-2, 2]$$

当 $|AIR| \geq \beta$（阈值超参数）时，认为分布差距显著。

**步骤二：构造平衡锚点** — 计算能给弱势模态分配更多注意力的目标 query 位置：

$$q_b = \left(\alpha \frac{\mathbb{E}[\hat{K}^a]}{\|\mathbb{E}[\hat{K}^a]\|_2} + (1-\alpha) \frac{\mathbb{E}[\hat{K}^v]}{\|\mathbb{E}[\hat{K}^v]\|_2}\right) \|\mathbb{E}[Q]\|_2$$

其中权重 $\alpha$ 由 AIR 决定：

$$\alpha = \frac{1}{2}[1 + \tanh(-\rho \cdot AIR)]$$

当 $AIR > 0$（模态 $a$ 偏置）时，$\alpha < 0.5$，使锚点更偏向弱势模态 $v$。

**步骤三：旋转 query** — 通过 SVD 分解计算旋转矩阵：

$$R_b = SVD([\mathbb{E}[Q], q_b]), \quad q_r = q \cdot R_b$$

旋转后的 query 在新的平衡区域学习，鼓励模态间 key 分布差距缩小。

### 实现细节

- 超参数：$\rho > 0$ 控制旋转强度，$\beta$ 为 AIR 阈值
- 旋转次数限制：CREMA-D/MOSEI 限制 1 次，Kinetic-Sound 限制 3 次
- 多层扩展：采用渐进式训练，依次对每层应用 RollingQ
- 额外参数仅约 **1%** 增加，GFLOPs 增加约 **0.1%**

## 实验关键数据

### 主实验（Table 1）

| 方法 | 类型 | CREMA-D Acc | Kinetic-Sound Acc | CMU-MOSEI Acc |
|------|------|-------------|-------------------|---------------|
| Audio (单模态) | - | 47.6 | 53.9 | - |
| Visual (单模态) | - | 36.3 | 57.0 | 47.1 |
| Concat (静态) | 静态 | 49.3 | 68.0 | 62.8 |
| OGM | 静态 | 51.2 | 68.2 | 62.7 |
| PMR | 静态 | 50.1 | 68.2 | 63.0 |
| Vanilla MT | 动态 | 48.8 | 67.0 | 62.7 |
| MBT | 动态 | 51.5 | 72.2 | 63.0 |
| MMML | 动态 | 52.0 | 69.8 | 62.8 |
| **Vanilla MT + RollingQ** | 动态 | **51.9** | **69.3** | **63.2** |
| **MMML + RollingQ** | 动态 | **52.7** | **70.7** | **63.2** |

RollingQ 为 Vanilla MT 带来一致性提升：CREMA-D +3.1%，Kinetic-Sound +2.3%，MOSEI +0.5%。

### 噪声鲁棒性测试（Table 3, Kinetic-Sound）

| 噪声水平 | Vanilla MT | Vanilla MT + RollingQ |
|----------|-----------|----------------------|
| 0.00 | 67.0 | 69.3 |
| 0.25 | 62.7 (-4.3) | 67.2 (-1.9) |
| 0.50 | 52.9 (-14.1) | 58.2 (-11.1) |
| 1.00 | 34.7 (-32.3) | 40.6 (-28.7) |

RollingQ 在所有噪声水平下均表现更优，且精度下降更少。

### 计算效率（Table 4, CREMA-D）

| 方法 | Acc | 参数量 | GFLOPs |
|------|-----|--------|--------|
| MBT | 51.5 | 114.21M | 2746.90 |
| MMML | 52.0 | 77.88M | 1828.29 |
| JMT | 50.7 | 62.23M | 1494.87 |
| **Vanilla MT + RollingQ** | **51.9** | **60.46M** | **1489.20** |

RollingQ 以最少的参数和计算量达到与复杂架构相当的性能。

### Pearson 相关性分析（Table 2）

| 方法 | CREMA-D coef | KS coef |
|------|-------------|---------|
| Vanilla MT | 0.52 | 0.44 |
| Vanilla MT + RollingQ | **0.76** | **0.78** |

RollingQ 显著提升了注意力分数与输入质量之间的相关性（p < 0.01），证明动态适应性得到恢复。

## 亮点与洞察

1. **问题发现有深度**：揭示了多模态 Transformer 中自注意力"名为动态实为静态"的现象，并通过理论分析和噪声实验清晰展示了自增强循环机制
2. **方法极简有效**：仅需一个旋转矩阵，参数增加 ~1%，却能在多个数据集上一致提升性能
3. **分析充分**：从 key 分布可视化、梯度分析、Pearson 相关性、噪声鲁棒性、OOD 检测等多个角度验证了 RollingQ 确实恢复了动态协作能力
4. **通用性好**：对 Vanilla MT、MulT、MMML 等多种架构均有效，也适用于 ResNet 骨干

## 局限与展望

1. **理论分析局限于单层注意力**：虽然提供了多层渐进训练的扩展方案，但对多层 Transformer 中更复杂的动态交互缺乏深入建模
2. **不直接增强单模态编码器**：RollingQ 仅调整 query 的分配策略，未像 OGM/PMR 等方法直接优化单模态特征质量；结合两类方法可能效果更好
3. **数据集规模偏小**：三个主要数据集（CREMA-D、Kinetic-Sound、CMU-MOSEI）都不算大规模，缺乏在大规模多模态预训练（如 VLM）上的验证
4. **旋转次数需手动设定**：不同数据集的最大旋转次数不同（1 或 3），自适应确定旋转次数的策略有待探索
5. **仅验证双模态场景**：对三模态及以上的扩展性未充分讨论

## 相关工作与启发

- **模态不平衡学习**：OGM（梯度调制）、PMR（原型模态再平衡）等方法主要针对静态融合，本文是首次系统分析动态融合中的不平衡问题
- **多模态 Transformer**：MBT（注意力瓶颈）、MulT（跨模态注意力）等通过结构设计提升融合能力，但未关注注意力机制本身的失效问题
- **贪婪多模态学习**：Wu et al. (2022) 提出多模态学习的贪婪性问题，本文将其延伸到注意力机制的动态失效分析

**启发**：该思路可推广到大规模 VLM（如 LLaVA、Qwen-VL）中检测视觉/语言模态的注意力偏置，也可启发 MoE 架构中的专家平衡策略。

## 评分
- 新颖性: ⭐⭐⭐⭐ — 问题发现新颖且重要，"注意力动态性失效"这一观察有价值
- 实验充分度: ⭐⭐⭐⭐ — 多数据集、多架构、多角度验证，含噪声/OOD/消融实验
- 写作质量: ⭐⭐⭐⭐ — 理论分析清晰，图示直观，论述逻辑性强
- 价值: ⭐⭐⭐⭐ — 对理解多模态 Transformer 有启发，但受限于数据集规模

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] In-Context Compositional Learning via Sparse Coding Transformer](../../NeurIPS2025/multimodal_vlm/in-context_compositional_learning_via_sparse_coding_transformer.md)
- [\[ICML 2026\] Dimension-Free Multimodal Sampling via Preconditioned Annealed Langevin Dynamics](../../ICML2026/multimodal_vlm/dimension-free_multimodal_sampling_via_preconditioned_annealed_langevin_dynamics.md)
- [\[ICCV 2025\] TAB: Transformer Attention Bottlenecks enable User Intervention and Debugging in Vision-Language Models](../../ICCV2025/multimodal_vlm/tab_transformer_attention_bottlenecks_enable_user_intervention_and_debugging_in_.md)
- [\[ACL 2026\] Towards Visually Grounded Multimodal Summarization via Cross-Modal Transformer and Gated Attention](../../ACL2026/multimodal_vlm/towards_visually_grounded_multimodal_summarization_via_cross-modal_transformer_a.md)
- [\[ICCV 2025\] Dita: Scaling Diffusion Transformer for Generalist Vision-Language-Action Policy](../../ICCV2025/multimodal_vlm/dita_scaling_diffusion_transformer_for_generalist_visionlang.md)

</div>

<!-- RELATED:END -->
