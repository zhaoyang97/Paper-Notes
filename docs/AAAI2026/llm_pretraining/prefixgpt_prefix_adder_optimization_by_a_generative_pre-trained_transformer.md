---
title: >-
  [论文解读] PrefixGPT: Prefix Adder Optimization by a Generative Pre-trained Transformer
description: >-
  [AAAI 2026][预训练][前缀加法器优化] 提出PrefixGPT，将前缀加法器优化建模为序列生成问题，通过定制的GPT模型预训练学习设计规则后用RL微调生成优化设计，在面积-延迟乘积(ADP)上取得SOTA且对初始化不敏感。 前缀加法器(Prefix Adder)因其高速特性广泛用于信号处理和AI等计算密集型应用…
tags:
  - "AAAI 2026"
  - "预训练"
  - "前缀加法器优化"
  - "GPT生成模型"
  - "硬件设计自动化"
  - "强化学习微调"
  - "合法性掩码"
---

# PrefixGPT: Prefix Adder Optimization by a Generative Pre-trained Transformer

**会议**: AAAI 2026  
**arXiv**: [2511.19472](https://arxiv.org/abs/2511.19472)  
**代码**: [https://github.com/Mightlaus/PrefixGPT-AAAI26](https://github.com/Mightlaus/PrefixGPT-AAAI26)  
**领域**: LLM预训练  
**关键词**: 前缀加法器优化, GPT生成模型, 硬件设计自动化, 强化学习微调, 合法性掩码

## 一句话总结
提出PrefixGPT，将前缀加法器优化建模为序列生成问题，通过定制的GPT模型预训练学习设计规则后用RL微调生成优化设计，在面积-延迟乘积(ADP)上取得SOTA且对初始化不敏感。

## 研究背景与动机

前缀加法器(Prefix Adder)因其高速特性广泛用于信号处理和AI等计算密集型应用。其核心在于前缀图(prefix graph)的设计，需同时优化电路面积(对应图的size)和延迟(对应图的depth)，而设计空间呈指数级增长。

**现有方法的核心痛点**：现有AI方法（PrefixRL用深度Q-learning、ArithTree用MCTS、PrefixLLM用大语言模型）都采用"结构修改"范式——从一个现有设计出发，逐步添加/删除节点或编辑局部连接。这带来两个根本性问题：

**频繁违反设计规则**：迭代修改经常破坏前缀加法器的三条严格设计规则（输入规则、输出规则、合并规则），必须额外引入修复步骤，修复本身又会引入额外硬件开销

**强烈的初始化偏差**：方法性能高度依赖初始设计选择（如Sklansky、Kogge-Stone、Brent-Kung），不同初始化下性能差异巨大甚至完全失败

**核心洞察**：前缀加法器的设计规则比自然语言更加结构化。GPT模型已展示出掌握复杂语言"语法"并生成满足结构和逻辑约束的高质量文本的能力。那么GPT能否也学习前缀加法器的"语法"，直接从零生成合法且高质量的设计？

**切入角度**：将前缀加法器设计建模为二维坐标序列生成问题，训练GPT从零开始直接生成优化设计，而非在现有设计上修改。通过合法性掩码保证每次生成都满足设计规则，完全避免修复步骤。

## 方法详解

### 整体框架

PrefixGPT采用"预训练+微调"的两阶段Pipeline：
1. 将前缀图转换为二维坐标序列表示
2. 设计合法性掩码确保每步生成都合法
3. 在随机合成的100万个合法前缀加法器上预训练，学习设计规则
4. 使用GRPO强化学习微调，以ADP为奖励信号优化设计质量

### 关键设计

#### 1. **序列表示与合法性掩码**
- **功能**：将前缀图→前缀矩阵→坐标序列，使问题可用GPT处理
- **核心思路**：n位前缀加法器对应一个n×n的二进制下三角矩阵，扫描矩阵中的'1'生成坐标序列 $L_{1:N} = (L_1, \ldots, L_N)$，每个 $L_p = (L_p^r, L_p^c)$ 表示行列坐标
- **动态合法性掩码**：基于设计规则推导出两种情况的合法下一步选择：
    - Case 1：当前坐标在第0列时，下一步只能是对角线上的下一行 $L_{k+1} = (L_k^r+1, L_k^r+1)$
    - Case 2：当前坐标不在第0列时，下一个坐标必须在同一行，且列坐标受限于行 $(L_k^c - 1)$ 上已有节点的列集合
- **设计动机**：掩码在GPU上并行实现，将非法选择的概率置零，保证"合法即构造"(valid by construction)，从根本上消除修复步骤

#### 2. **空间坐标嵌入与双头Transformer骨干**
- **功能**：定制的GPT架构，处理二维坐标数据
- **空间坐标嵌入**：对行列坐标分别学习嵌入向量 $\mathbf{L}_p^r, \mathbf{L}_p^c \in \mathbb{R}^d$，通过RoPE编码相对位置信息，拼接得到融合token $\mathbf{T}_p = \hat{\mathbf{L}}_p^r \| \hat{\mathbf{L}}_p^c \in \mathbb{R}^{2d}$
- **双头Transformer**：共享解码器(4层) → 行头(1层，预测行坐标) + 列头(2层，融合行信息后预测列坐标)。列头依赖行头输出，因为合法列坐标取决于行坐标
- **设计动机**：标准GPT只处理一维token序列，无法直接处理二维坐标。双头设计自然地建模了行列之间的依赖关系

#### 3. **预训练与GRPO微调**
- **预训练**：在100万个随机合成的合法坐标序列上自监督学习，损失为行列交叉熵之和 $\mathcal{L}_{pre} = -\frac{1}{k}\sum_{p=1}^{k}(\log P(L_p^r|L_{1:p-1}) + \log P(L_p^c|L_{1:p-1}))$
- **微调**：复制预训练模型为策略模型 $\pi_\theta$ 和冻结的参考模型 $\pi_{ref}$，采用GRPO优化。每次迭代采样G=512个合法加法器，奖励为负ADP $r_i = -\text{area}(\alpha_i) \times \text{delay}(\alpha_i)$
- **最佳设计检索机制**：维护全局数据库记录所有生成过的设计，每次迭代从中检索最优设计（≤10%批次大小）合并到当前批次，强化高质量模式学习
- **目标函数**：$\mathcal{J}(\theta) = \frac{1}{G}\sum_{i=1}^{G}\frac{1}{N_i}\sum_{p=1}^{N_i}\{\gamma^p s_\theta(L_p)\hat{A}_i - \beta \mathbb{D}_{KL}(L_p)\}$

### 损失函数 / 训练策略

- 预训练：行列交叉熵损失，5个epoch，学习率 $10^{-4}$
- 微调：GRPO目标函数，200次RL迭代，每次512个样本，温度0.8
- KL散度正则化（$\beta=0.001$）防止策略崩溃
- 预训练在n=48位宽上完成后，可即时迁移到任意m≤n位宽

## 实验关键数据

### 主实验（前缀图Size比较）

| 位宽 | 深度限制 | ArithTree (Sk/Ko/Br/Ra) | PrefixRL (Sk/Ko) | PrefixGPT (Sk/Ko/Br/Ra) |
|------|---------|------------------------|-------------------|------------------------|
| 16-bit | 5 | 31/45/-/- | -/- | **31/31/31/32** |
| 32-bit | 7 | 65/108/-/- | 77/128 | **57/61/57/60** |
| 48-bit | 7 | 119/217/-/- | 224/- | **102/94/98/102** |
| 48-bit | 9 | 94/155/-/- | 120/207 | **86/86/86/86** |

PrefixGPT在全部12个(位宽×深度限制)组合中取得最优size，最大缩减59.1%。

### ADP综合比较

| 位宽 | 方法 | 最优ADP (μm²·ns) | 平均ADP | 标准差 |
|------|------|-----------------|---------|-------|
| 32-bit | ArithTree | - | 324.64 | - |
| 32-bit | PrefixGPT | - | **91.14** (↓71.9%) | 极低 |
| 48-bit | ArithTree | 131.4 | - | 422.2 |
| 48-bit | PrefixRL | 142.8 | - | - |
| 48-bit | PrefixGPT | **121.3** (↓7.7%) | **185.15** (↓79.1%) | **25.2** (↓94%) |

### 消融实验

| 消融配置 | 最终平均奖励变化 | 标准差变化 |
|---------|---------------|----------|
| 移除预训练 | -31.6% | 显著增加 |
| 移除RoPE | -16.6% | 增加 |
| 移除KL散度+最佳设计检索 | -1.3% | +110% |
| 完整PrefixGPT | 基准 | 基准 |

### 关键发现

1. **初始化鲁棒性**：PrefixGPT即使在随机初始化下，性能也超过其他方法使用最佳手动设计的初始化
2. **生效率**：预训练后移除合法性掩码，模型生成合法率仍接近100%，说明模型真正内化了设计规则
3. **注意力机制**：列头的注意力分数与合并规则高度一致，模型在预训练阶段超越了模式记忆，发展出了对设计空间的基础理解
4. **高效推理**：生成一个32位设计仅需约7ms

## 亮点与洞察

1. **范式创新**：从"修改现有设计"转变为"从零生成设计"，完全绕过了修复步骤和初始化偏差两大根本问题
2. **合法性掩码的优雅设计**：将复杂的设计规则重新表述为可在GPU上并行计算的动态掩码，与生成过程无缝集成
3. **预训练的深层价值**：不仅学会了生成合法设计，还内化了设计空间的结构特征，使得微调阶段的搜索更高效
4. **跨领域启示**：证明GPT范式可推广到具有严格约束的组合优化问题，不限于自然语言

## 局限与展望

1. MSP约束的限制：当前约束Lk+1的MSP必须是Lk对应的节点，简化了生成但牺牲了部分最优性
2. GPU内存限制：预训练最多到48位，更高位宽需要更多内存
3. 仅优化ADP：未考虑其他电路指标如功耗
4. 可探索更强的RL算法替代GRPO

## 相关工作与启发

- PrefixRL (Roy et al., 2021)：深度Q-learning做前缀加法器优化，开创性工作但受限于初始化偏差
- ArithTree (Lai et al., 2024)：MCTS方法，可扩展但同样有初始化问题
- PrefixLLM (Xiao et al., 2024)：用LLM编辑文本表示，但生成时间过长（>200s/设计 vs PrefixGPT的7ms）
- GRPO (Shao et al., 2024)：轻量级RL方法，避免了昂贵的价值模型

## 评分
- 新颖性: ⭐⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] No-Regret Strategy Solving in Imperfect-Information Games via Pre-Trained Embedding](no-regret_strategy_solving_in_imperfect-information_games_via_pre-trained_embedd.md)
- [\[AAAI 2026\] Rectified Noise: A Generative Model Using Positive-incentive Noise](rectified_noise_a_generative_model_using_positive-incentive_noise.md)
- [\[ICML 2026\] Different Layers, Different Manifolds: Module-Wise Weight-Space Geometry in Transformer Optimization](../../ICML2026/llm_pretraining/different_layers_different_manifolds_module-wise_weight-space_geometry_in_transf.md)
- [\[CVPR 2026\] Unlocking Pre-trained Weights: Parameter Inheritance for Zero-Shot Initialization](../../CVPR2026/llm_pretraining/unlocking_pre-trained_weights_parameter_inheritance_for_zero-shot_initialization.md)
- [\[ICCV 2025\] Dataset Ownership Verification for Pre-trained Masked Models](../../ICCV2025/llm_pretraining/dataset_ownership_verification_for_pre-trained_masked_models.md)

</div>

<!-- RELATED:END -->
