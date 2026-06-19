---
title: >-
  [论文解读] The Sharpness Disparity Principle in Transformers for Accelerating Language Model Pre-Training
description: >-
  [ICML 2025][3D视觉][Transformer] 揭示了 Transformer 中不同类型模块（Emb、QK、FFN、VO、Norm）存在显著且持久的**锐度差异**（sharpness disparity），并据此提出 Blockwise LR 策略，为低锐度模块分配更大学习率，在不损失稳定性的前提下实现 LLM 预训练近 **2× 加速**。
tags:
  - "ICML 2025"
  - "3D视觉"
  - "Transformer"
  - "学习率调度"
  - "Hessian分析"
  - "预训练加速"
  - "块级锐度"
---

# The Sharpness Disparity Principle in Transformers for Accelerating Language Model Pre-Training

**会议**: ICML 2025  
**arXiv**: [2502.19002](https://arxiv.org/abs/2502.19002)  
**代码**: [GitHub](https://github.com/Wongboo/BlockwiseLearningRate)  
**领域**: LLM预训练  
**关键词**: Transformer优化, 学习率调度, Hessian分析, 预训练加速, 块级锐度

## 一句话总结

揭示了 Transformer 中不同类型模块（Emb、QK、FFN、VO、Norm）存在显著且持久的**锐度差异**（sharpness disparity），并据此提出 Blockwise LR 策略，为低锐度模块分配更大学习率，在不损失稳定性的前提下实现 LLM 预训练近 **2× 加速**。

## 研究背景与动机

Transformer 具有"合金"（alloy-like）特性：由 Embedding 层、Normalization 层、Self-Attention（QK/VO）和 FFN 等异质模块组合而成。传统的 layerwise 学习率调整策略在 MLP/CNN 上有效，但无法迁移到深层 Transformer 上。作者认为**根本原因在于 Transformer 的异质性并非在层级别呈现一致规律，而是在模块类型级别展现出清晰的锐度差异**。

先前工作（Zhang et al., 2024; Ormaniec et al.）观察到了 Hessian 的块级异质性，但未建立跨所有模块类型的统一原则。本文在此基础上，系统地比较了五类模块的锐度，首次建立了完整的锐度序关系。

此外，训练过程处于 Edge of Stability (EoS) 阶段：高锐度方向上动力学剧烈振荡（fast dynamics，决定训练稳定性），低锐度方向上缓慢推进（slow dynamics，主要贡献 loss 下降）。如果能加速低锐度方向而不扰动高锐度方向，就能实现整体加速。

## 方法详解

### 整体框架

方法分为两个核心部分：

1. **Sharpness Disparity Principle（锐度差异原则）**：通过 Fisher 信息矩阵对角近似，在模块类型粒度上量化锐度差异，建立统一的序关系。
2. **Blockwise LR（块级学习率策略）**：根据锐度差异原则，为不同类型的模块分配不同的学习率倍率。

### 关键设计

#### 1. 锐度差异原则（Principle 1）

对五类模块的平均锐度进行估计，发现如下序关系在训练全程持续成立：

$$\mathcal{S}(\text{Emb}) \ll \mathcal{S}(\text{QK}) < \mathcal{S}(\text{FFN}) < \mathcal{S}(\text{VO}) \ll \mathcal{S}(\text{Norm})$$

- **Norm 层锐度最高**：参数量极少（每层仅 D 个标量），梯度范数相对参数范数极大
- **Emb 层锐度最低**：词表维度 d 极大（如 GPT tokenizer 的 50304），参数量庞大，摊平后锐度最小
- **QK < FFN < VO**：FFN 和 VO 介于两者之间，QK 因 softmax 的特殊性较平

#### 2. 锐度的高效估计

使用 Fisher 信息矩阵的对角近似来高效估计 Hessian。对每类模块 $\bullet$，平均锐度定义为：

$$\mathcal{S}(\bullet) = \frac{B \|\nabla_{\theta[\bullet]} \hat{\mathcal{L}}_B(\theta)\|_F^2}{\#(\theta[\bullet])}$$

其中 $B$ 为 batch size，$\#(\theta[\bullet])$ 为该模块类型的参数量。关键优势在于只需计算 mini-batch 梯度，无需逐样本梯度，计算开销极低。

#### 3. 理论解释

通过三组定理（Theorem 4.1–4.3）从解析梯度表达式出发，证明了锐度差异的来源：

- **FFN vs. Norm（Theorem 4.1）**：$\mathcal{S}(W_\bullet) = \mathcal{O}(\Psi^2 / (D^2 \|W_\bullet\|_F^2))$，$\mathcal{S}(\gamma) = \mathcal{O}(\Psi^2 / (D \|\gamma\|_F^2))$。由于 $D^2 \|W_\bullet\|_F^2 \gg D \|\gamma\|_F^2$（FFN 参数量大且范数大，Norm 参数 $\gamma$ 在训练中逐渐减小），故 FFN 锐度远低于 Norm。
- **SA vs. Norm（Theorem 4.2）**：QK 和 VO 的锐度同样远低于 Norm，理由类似。
- **Emb vs. Norm（Theorem 4.3）**：$\mathcal{S}(W_E) = \mathcal{O}(\Psi^2 / (Dd \min_i \|\tilde{w}_{E_i}\|_2^2))$，词表大小 $d$（~50k）使分母极大，Emb 锐度最低。

核心直觉：Transformer 的乘法复合结构使得 $\|\nabla_\bullet \mathcal{Q}\| \propto 1/\|\theta[\bullet]\|$，参数越多/越大的模块，锐度越低。

#### 4. Blockwise LR 策略

- **Norm 块**：保持基础学习率 $\eta_{\text{Norm}} = \eta_{\text{base}}$（维持稳定性）
- **其他块**：学习率乘以调整倍率 $r(\bullet)$，按锐度差异定性趋势手动调优

最终调优得到的默认倍率（仅在 LLaMA 0.25B + MiniPile 上调一次）：

| 模块类型 | 调整倍率 $r(\bullet)$ | 锐度排序 |
|---------|---------------------|---------|
| Emb | 10× | 最低（最平坦） |
| QK | 8× | 较低 |
| FFN | 6× | 中等 |
| VO | 4× | 较高 |
| Norm | 1×（不变） | 最高（最尖锐） |

这组倍率的关键优势：**只需调一次即可跨模型、跨数据集迁移**，鲁棒性极强。

### 损失函数 / 训练策略

- **基础优化器**：AdamW，$\beta_1 = 0.9, \beta_2 = 0.95$，weight decay $\lambda = 0.1$
- **梯度裁剪**：clip norm = 1.0
- **学习率调度**：线性 warm-up + cosine decay（也验证了 WSD 调度器）
- **Blockwise LR 的集成**：在 AdamW 的参数组中，对不同类型的模块设置不同的 lr 倍率，实现极其简单（可直接嵌入 Megatron 等框架）

## 实验关键数据

### 主实验

在 GPT-2 和 LLaMA 的多种规模上验证：

| 模型 | 参数量 | 数据集 | AdamW loss | + Blockwise LR loss | 加速比 |
|------|--------|--------|-----------|---------------------|--------|
| LLaMA | 0.25B | OpenWebText | 2.834 | 2.784 | ~1.8× |
| LLaMA | 0.5B | OpenWebText | — | — | ~1.9× |
| LLaMA | 1.1B | OpenWebText | — | — | **1.92×** |
| LLaMA | 0.25B | MiniPile | — | — | ~1.9× |
| GPT-2 | 0.12B | OpenWebText | — | — | ~1.8× |
| LLaMA | 2B | C4 | — | — | ~2× |

**下游任务评估**（LLaMA 1.1B, OpenWebText, 50K steps, 0-shot）：

| 任务 | AdamW | Blockwise LR | 提升 |
|------|-------|-------------|------|
| ARC_E | 52.69 | **54.29** | +1.60 |
| ARC_C | 22.87 | **25.34** | +2.47 |
| PIQA | 68.71 | **69.53** | +0.82 |
| HellaSwag | 36.13 | **38.00** | +1.87 |
| OBQA | 19.40 | **22.60** | +3.20 |
| WinoGrande | 55.17 | **59.83** | +4.66 |
| SCIQ | 77.60 | **81.60** | +4.00 |

### 消融实验

LLaMA 0.25B on OpenWebText, 50K steps：

| 配置 | Terminal Loss | 变化 | 说明 |
|------|-------------|------|------|
| AdamW baseline | 2.834 | — | 基线 |
| + Emb only | 2.818 | -0.016 ✓ | Emb 增大 LR 有效 |
| + Emb & FFN | 2.791 | -0.043 ✓ | FFN 贡献最大（-0.027） |
| + Emb & FFN & QK & VO | 2.784 | -0.050 ✓ | 全部低锐度模块均有贡献 |
| + Norm only（翻倍） | 2.837 | +0.003 ✗ | 增大 Norm LR 反而有害 |

### 关键发现

1. **FFN 贡献最大**：占参数量最多，LR 增大后 loss 下降幅度最显著（单独贡献 0.027）
2. **Norm 增大 LR 有害**：验证了 Norm 所在的高锐度方向不应被扰动
3. **Scaling law 有利**：随模型增大，Blockwise LR 的优势在 MiniPile/OpenWebText 上**扩大**，C4 上保持稳定
4. **跨优化器兼容**：
    - Adam-mini + Blockwise LR = **2× 加速 + 2× 内存节省**
    - Lion + Blockwise LR = **2× 加速**
    - WSD scheduler + Blockwise LR = **2× 加速**

## 亮点与洞察

1. **从现象到原则**：不是简单的 layerwise 分析，而是在**模块类型**粒度上建立了统一的锐度序关系，揭示了 Transformer "合金"特性的核心
2. **理论支撑充分**：三组定理从梯度解析表达式出发，用参数量和参数范数解释锐度差异，直觉清楚（乘法复合 → 梯度反比于参数范数）
3. **极简实现**：只需 4 个额外超参（r 倍率），调一次跨模型/数据集迁移，可直接集成到 PyTorch 参数组中
4. **EoS 视角统一**：将 Blockwise LR 解释为"加速 river/stable direction 上的 slow dynamics"，与 EoS 理论优雅统一
5. **锐度差异在训练极早期（~2% 步数）就已建立**：暗示可以一开始就使用 Blockwise LR，无需等到训练稳定

## 局限与展望

1. **仅验证了语言模型预训练**：未在 Vision Transformer、多模态等领域验证
2. **倍率手动调优**：虽然鲁棒，但理论上可以从锐度比值自动推导，实现 adaptive blockwise LR
3. **与 Muon 等更新优化器的兼容性未验证**
4. **未探索 blockwise weight decay 和 blockwise gradient clipping** 等自然延伸
5. **QK < VO 关系的理论证明依赖于 Ormaniec et al.** 的结果，本文仅给出经验验证
6. **Blockwise LR 在训练中后期才显著超越 baseline**，早期阶段优势不明显且原因不明

## 相关工作与启发

- **Zhang et al. (2024) Adam-mini**：利用 Hessian 块级异质性减半 Adam 内存，本文在此基础上做加速
- **Ormaniec et al.**：单层 SA 的 QK vs. VO 锐度分析，启发了本文的全模块类型对比
- **Wang et al. (2024) SOAP**：按单个参数锐度调 LR，但计算开销大且估计不稳——本文在模块类型粒度上做调整，简洁高效
- **EoS 系列 (Cohen et al., 2021; Wen et al., 2024)**：fast-slow dynamics 图景是 Blockwise LR 的理论动机
- **WSD scheduler (Wen et al., 2024)**：与 Blockwise LR 互补，可叠加

**核心启发**：深度网络的优化不应"一视同仁"，而应根据不同组件的几何特性定制策略。模块类型粒度的锐度分析可能为优化器设计打开新范式。

## 评分

| 维度 | 分数（1-5） | 说明 |
|------|-----------|------|
| 新颖性 | 4 | 块类型级锐度原则是新发现，Blockwise LR 简单但有效 |
| 理论深度 | 4 | 三组定理提供严谨的理论支撑 |
| 实验充分性 | 5 | 多模型（GPT-2/LLaMA）、多规模（0.12B-2B）、多数据集、多优化器 |
| 实用价值 | 5 | 实现极简、迁移性强、加速显著、可叠加内存优化 |
| 写作质量 | 4 | 结构清晰，图表丰富，逻辑流畅 |
| **综合** | **4.5** | 高质量工作，兼具理论深度和实用价值 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Slamming: Training a Speech Language Model on One GPU in a Day](../../ACL2025/3d_vision/slamming_training_a_speech_language_model_on_one_gpu_in_a_day.md)
- [\[ICCV 2025\] 4D Visual Pre-training for Robot Learning](../../ICCV2025/3d_vision/4d_visual_pretraining_for_robot_learning.md)
- [\[ECCV 2024\] Formula-Supervised Visual-Geometric Pre-training (FSVGP)](../../ECCV2024/3d_vision/formula-supervised_visual-geometric_pre-training.md)
- [\[ICML 2025\] GAPrompt: Geometry-Aware Point Cloud Prompt for 3D Vision Model](gaprompt_geometry-aware_point_cloud_prompt_for_3d_vision_model.md)
- [\[CVPR 2025\] UniPre3D: Unified Pre-training of 3D Point Cloud Models with Cross-Modal Gaussian Splatting](../../CVPR2025/3d_vision/unipre3d_unified_pre-training_of_3d_point_cloud_models_with_cross-modal_gaussian.md)

</div>

<!-- RELATED:END -->
