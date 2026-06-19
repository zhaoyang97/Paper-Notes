---
title: >-
  [论文解读] Limited Preference Data? Learning Better Reward Model with Latent Space Synthesis
description: >-
  [NeurIPS 2025][LLM对齐][reward modeling] 提出 LENS 框架，通过在 LLM 嵌入的潜在空间中利用 VAE 合成偏好数据对，绕过昂贵的文本生成过程，以极低计算成本（模型缩小 16000 倍、生成速度提升 18 倍）显著提升 reward model 性能。 Reward modeling…
tags:
  - "NeurIPS 2025"
  - "LLM对齐"
  - "reward modeling"
  - "latent space synthesis"
  - "VAE"
  - "数据增强"
  - "RLHF"
---

# Limited Preference Data? Learning Better Reward Model with Latent Space Synthesis

**会议**: NeurIPS 2025  
**arXiv**: [2509.26074](https://arxiv.org/abs/2509.26074)  
**代码**: [https://github.com/deeplearning-wisc/lens](https://github.com/deeplearning-wisc/lens)  
**领域**: LLM对齐  
**关键词**: reward modeling, latent space synthesis, VAE, preference data augmentation, RLHF

## 一句话总结
提出 LENS 框架，通过在 LLM 嵌入的潜在空间中利用 VAE 合成偏好数据对，绕过昂贵的文本生成过程，以极低计算成本（模型缩小 16000 倍、生成速度提升 18 倍）显著提升 reward model 性能。

## 研究背景与动机

Reward modeling 是将 LLM 与人类偏好对齐的核心环节，但面临严重的**数据瓶颈**问题：

**人工标注成本高**：偏好数据需要人类进行成对比较标注，耗时且昂贵，难以大规模收集

**现有文本合成方法计算开销大**：传统方法需要两阶段流程——先用 LLM 生成多个响应，再用辅助 LLM 标注偏好对，计算复杂度高且随响应数量二次增长

**资源受限场景需求**：小型实验室和创业公司难以承担数十亿参数模型的推理成本

核心问题：**在偏好数据有限的情况下，能否以高效方式扩展数据集来改善 reward modeling？**

作者观察到 LLM 的嵌入空间已经捕获了丰富的语义信息，因此提出一个关键 insight：**直接在嵌入空间中合成数据，可以绕过文本生成的计算瓶颈**。

## 方法详解

### 整体框架

LENS（Latent EmbeddiNg for Synthesis）框架包含三个阶段：

1. **嵌入提取 + VAE 训练**：在响应嵌入上训练带散度损失的变分自编码器
2. **潜在空间合成**：在学到的潜在空间中通过受控扰动生成合成偏好对
3. **增强训练**：在原始 + 合成数据上训练 reward model

### 关键设计

**Step 1: 嵌入提取**

给定偏好数据集 $\mathcal{D} = \{(x_i, y_i^+, y_i^-)\}_{i=1}^N$，提取 LLM 最后一层隐状态作为嵌入：

$$\mathbf{e}_i^{\pm} = \text{LLM}_{\text{embed}}(x_i, y_i^{\pm})$$

**Step 2: 带散度学习的 VAE**

VAE 编码器将 $d$ 维 LLM 嵌入映射到高斯后验参数：

$$q_\phi(\mathbf{z}|\mathbf{e}^+) = \mathcal{N}(\mathbf{z}; \boldsymbol{\mu}_\phi(\mathbf{e}^+), \boldsymbol{\sigma}_\phi(\mathbf{e}^+)^2 \cdot \mathbf{I})$$

标准 VAE 损失：

$$\mathcal{L}_{\text{VAE}}(\mathbf{e}) = \mathcal{L}_{\text{recon}}(\mathbf{e}, \hat{\mathbf{e}}) + \beta \cdot D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{e}) \| p(\mathbf{z}))$$

**核心创新——散度损失**：引入 Wasserstein 距离最大化正负样本的潜在分布分离度：

$$\mathcal{L}_{\text{divergence}} = -\frac{1}{N}\sum_{i=1}^{N} W_2(q_\phi(\mathbf{z}^+|\mathbf{e}_i^+), q_\phi(\mathbf{z}^-|\mathbf{e}_i^-))$$

总损失：

$$\mathcal{L}_{\text{total}} = \frac{1}{N}\sum_{i=1}^{N}[\mathcal{L}_{\text{VAE}}(\mathbf{e}_i^+) + \mathcal{L}_{\text{VAE}}(\mathbf{e}_i^-)] + \gamma \cdot \mathcal{L}_{\text{divergence}}$$

其中 $\gamma$ 控制散度项的权重（最优值约 0.1）。

**Step 3: 潜在空间采样**

通过向潜在向量添加高斯噪声生成合成嵌入：

$$\hat{\mathbf{e}}_{i,j}^{\pm} = g_\theta(\mathbf{z}_i^{\pm} + \boldsymbol{\eta}_{i,j}^{\pm}), \quad \boldsymbol{\eta}_{i,j}^{\pm} \sim \mathcal{N}(0, \sigma_{\text{noise}}^2\mathbf{I})$$

通过 top-k 筛选保留似然度最高的合成样本，然后进行**组合配对**扩展数据集：

$$\mathcal{E}_{\text{aug}} = \{(\tilde{\mathbf{e}}^+, \tilde{\mathbf{e}}^-) | \tilde{\mathbf{e}}^+ \in \mathcal{E}^+ \cup \mathcal{E}_{\text{synth}}^+, \tilde{\mathbf{e}}^- \in \mathcal{E}^- \cup \mathcal{E}_{\text{synth}}^-\}$$

### 损失函数 / 训练策略

Reward model 使用轻量 MLP 在嵌入空间上训练，目标函数基于 Bradley-Terry 模型：

$$\mathcal{L}_{RM}^{\mathcal{E}_{\text{aug}}} = -\mathbb{E}_{(\tilde{\mathbf{e}}^+, \tilde{\mathbf{e}}^-) \in \mathcal{E}_{\text{aug}}}[\log\sigma(r_o(\tilde{\mathbf{e}}^+) - r_o(\tilde{\mathbf{e}}^-))]$$

**理论保证**：
- **Theorem 1**：合成偏好对在最优 reward 函数下保持原始偏好排序，误差受噪声水平和 VAE 重构质量约束
- **Theorem 2**：增强数据训练的 reward model 在满足正则条件时，估计误差上界更小

## 实验关键数据

### 主实验

基础模型 Llama-3.1-8B-Instruct，种子样本 1000 个，评估指标为 Best-of-N (N=16) 采样的 gold reward 分数。

| 方法 | HH-RLHF (Orig) | HH-RLHF (4×) | HH-RLHF (8×) | TL;DR (Orig) | TL;DR (4×) | TL;DR (8×) |
|------|------|------|------|------|------|------|
| Fully fine-tune (文本) | 1.49 | 1.78 | 1.93 | 0.69 | 0.97 | 1.23 |
| LoRA (文本) | 1.28 | 1.52 | 1.61 | 0.57 | 0.92 | 1.15 |
| Embedding MLP (文本) | 1.43 | 1.62 | 1.73 | 0.78 | 1.02 | 1.11 |
| Self-rewarding | 1.49 | 1.59 | 1.77 | 0.69 | 0.92 | 0.95 |
| Direct perturbation | 1.43 | 1.32 | 1.46 | 0.78 | 0.84 | 0.79 |
| Gaussian sampling | 1.43 | 1.12 | 0.94 | 0.78 | 0.53 | 0.43 |
| **LENS (Ours)** | 1.43 | **1.94** | **2.20** | 0.78 | **1.44** | **1.48** |

LENS 在所有增强比例和数据集上均显著领先，8× 增强在 HH-RLHF 上比最强文本基线高 0.27 分。

**计算效率对比（8× 增强，HH-RLHF）**：

| 指标 | 文本合成 | 潜在空间合成 | 缩减倍数 |
|------|---------|------------|---------|
| 生成时间 | 3.6h | 0.2h | 18× |
| 模型参数量 | 8B | 0.5M | 16,000× |
| 总运行时间 | 5.2h | 0.4h | 13× |

### 消融实验

1. **散度损失权重 $\gamma$**：$\gamma=0.1$ 最优；$\gamma=0$ 无散度项效果差；$\gamma \geq 0.5$ 过度分离导致退化
2. **合成噪声 $\sigma^2$**：$\sigma^2=0.01$ 最优（reward=1.96）；过小（0.001→1.63）探索不足；过大（1.0→1.51）破坏偏好关系
3. **原始数据规模**：从 0.1k 到 50k 样本，LENS 4× 增强始终优于原始数据（0.1k: 0.93 vs 0.68）
4. **跨模型泛化**：在 Gemma-2B、Llama-3.2-3B、Mistral-7B、Qwen-2.5-7B 等模型上均有效

### 关键发现

- 即使无增强，基于嵌入的 MLP reward model 也能接近全量微调的性能，说明 LLM 嵌入已蕴含丰富偏好信息
- 简单潜在空间基线（直接扰动、高斯采样）效果差甚至退化，说明 VAE 的结构化潜在表示至关重要
- 通过 rejection sampling 进行下游 SFT，LENS 训练的 reward model 在 GPT-4 评估中获得 61% 胜率（vs 文本合成的 39%）

## 亮点与洞察

1. **效率提升惊人**：16000 倍模型缩小和 18 倍速度提升，实际解决了资源瓶颈问题
2. **理论+实践双重验证**：不仅有理论保证偏好保持，还有大量实验证实
3. **潜在空间是偏好数据的"正确"操作空间**：绕过文本生成的思路非常优雅，值得其他数据增强任务借鉴
4. **散度损失设计**：用 Wasserstein 距离显式分离正负样本的潜在分布，比标准 VAE 更适合偏好学习

## 局限与展望

1. **仅验证了 MLP reward head**：未与全量微调的 reward model + 潜在空间增强结合测试
2. **种子数据质量依赖**：如果初始 1000 样本不够代表性，合成质量可能受限
3. **VAE 潜在空间维度固定为 16**：未充分探索不同维度对不同规模模型的影响
4. **仅在英文偏好数据上验证**：多语言场景的泛化性未知
5. **缺少与 DPO 等直接偏好优化方法的对比**：LENS 增强后的数据是否也能提升 DPO 性能

## 相关工作与启发

- **与 Self-Rewarding 的对比**：Self-Rewarding 依赖模型自身生成和判断，仍需文本生成开销；LENS 完全在嵌入空间操作
- **与 VOS（Du et al.）的联系**：同一实验室先前在 OOD 检测中使用 VAE 合成特征的思想，迁移到了偏好学习
- **对 Active Learning 方法的补充**：LENS 是数据增强路线，可与主动学习（选择性标注）结合使用
- **启发**：该框架可扩展到其他需要成对比较数据的场景（如对比学习、排序学习）

## 评分
- 新颖性: ⭐⭐⭐⭐ 在潜在空间合成偏好数据的想法新颖，散度 VAE 设计合理
- 实验充分度: ⭐⭐⭐⭐⭐ 多数据集、多模型、多消融，加上下游 SFT 验证和理论分析
- 写作质量: ⭐⭐⭐⭐ 结构清晰，方法描述详尽，图表丰富
- 价值: ⭐⭐⭐⭐ 实际解决了计算成本问题，对资源受限场景有重要价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] ResponseRank: Data-Efficient Reward Modeling through Preference Strength Learning](responserank_data-efficient_reward_modeling_through_preference_strength_learning.md)
- [\[NeurIPS 2025\] Inference-time Alignment in Continuous Space](inference-time_alignment_in_continuous_space.md)
- [\[NeurIPS 2025\] What Makes a Reward Model a Good Teacher? An Optimization Perspective](what_makes_a_reward_model_a_good_teacher_an_optimization_perspective.md)
- [\[NeurIPS 2025\] Ask a Strong LLM Judge when Your Reward Model is Uncertain](ask_a_strong_llm_judge_when_your_reward_model_is_uncertain.md)
- [\[NeurIPS 2025\] Preference Optimization by Estimating the Ratio of the Data Distribution](preference_optimization_by_estimating_the_ratio_of_the_data_distribution.md)

</div>

<!-- RELATED:END -->
