---
title: >-
  [论文解读] Learning to Factorize and Adapt: A Versatile Approach Toward Universal Spatio-Temporal Foundation Models
description: >-
  [NeurIPS 2025][模型压缩][时空基础模型] 提出 FactoST-v2，一个因式分解的时空基础模型框架，将通用时间预训练与领域特定空间适配解耦，以线性复杂度实现跨领域零样本/少样本/全样本时空预测。 领域现状：时空（ST）基础模型旨在从多种领域数据中学习通用表示，实现跨数据集泛化预测。现有方法如 UniST、O…
tags:
  - "NeurIPS 2025"
  - "模型压缩"
  - "时空基础模型"
  - "时间序列预测"
  - "因式分解范式"
  - "时空适配"
  - "零样本泛化"
---

# Learning to Factorize and Adapt: A Versatile Approach Toward Universal Spatio-Temporal Foundation Models

**会议**: NeurIPS 2025  
**arXiv**: [2601.12083](https://arxiv.org/abs/2601.12083)  
**代码**: [GitHub](https://github.com/CityMind-Lab/FactoST)  
**领域**: 模型压缩  
**关键词**: 时空基础模型, 时间序列预测, 因式分解范式, 时空适配, 零样本泛化

## 一句话总结

提出 FactoST-v2，一个因式分解的时空基础模型框架，将通用时间预训练与领域特定空间适配解耦，以线性复杂度实现跨领域零样本/少样本/全样本时空预测。

## 研究背景与动机

**领域现状**：时空（ST）基础模型旨在从多种领域数据中学习通用表示，实现跨数据集泛化预测。现有方法如 UniST、OpenCity 采用联合时空预训练范式。

**现有痛点**：联合预训练面临"模式不匹配"挑战——时间动态具有跨域不变性（周期性、趋势等），但空间关联高度依赖特定拓扑（路网 vs 电网），强行联合建模导致二次复杂度 $\mathcal{O}(N^2T^2)$ 和负迁移。

**核心矛盾**：空间和时间模式的性质根本不同——时间模式领域不变，空间模式领域特定——但现有方法迫使二者联合学习。

**本文目标**：设计一个兼顾泛化性和效率的时空基础模型框架。

**切入角度**：提出"模式因式分解假说"——有效的时空泛化需要解耦领域不变的时间动态和领域特定的空间上下文。

**核心 idea**：先预训练通用时间 backbone，再用轻量适配器注入空间感知，实现线性复杂度的跨域迁移。

## 方法详解

### 整体框架

FactoST-v2 分两个阶段：**阶段一（UTP）**在大规模多源时间序列上进行通用时间预训练，学习领域不变的时间表示；**阶段二（STA）**通过轻量适配器向预训练 backbone 注入时空感知，实现下游适配。

### 关键设计

1. **Encoder-Only Backbone + 随机序列掩码**：抛弃 v1 的 Encoder-Decoder 架构，采用极简 Encoder-Only 设计。通过可学习 [REG] token 分隔上下文和预测区间，配合随机序列掩码 $l_{mask} \sim \mathcal{U}(0, L_{max} - L_{min})$ 使模型学习任意长度映射。这消除了固定输入输出长度的限制，实现 100% 参数迁移。

2. **语义感知位置编码 (p-RoPE) + 门控注意力**：将嵌入空间分解为高频和低频子空间，旋转矩阵仅应用于高频部分保持顺序感知，低频部分不变以保留语义幅值：$\text{p-RoPE}(\mathbf{x}, m) = [\mathbf{x}_{high} \otimes \mathbf{R}_{\Theta,m} \| \mathbf{x}_{low}]$。门控注意力通过 $\mathbf{O} = \text{Attention}(\mathbf{Q}', \mathbf{K}', \mathbf{V}) \odot \sigma(\mathbf{G})$ 过滤噪声。

3. **概率分位数预测**：从 v1 的确定性点估计升级为多分位数预测，使用 Pinball Loss 优化：$\mathcal{L}_{UTP} = \frac{1}{|\mathcal{Q}|}\sum_{q \in \mathcal{Q}}\max((q-1)(\mathbf{y}-\hat{\mathbf{y}}_q), q(\mathbf{y}-\hat{\mathbf{y}}_q))$。支持不确定性量化。

4. **时空元数据融合 (STMF)**：注入空间节点嵌入 $\mathbf{E}_n \in \mathbb{R}^{N \times d}$ 和多尺度日历时间嵌入，通过线性投影生成时空上下文 $\mathbf{I}_{st}$。

5. **时空过滤 (STF)**：通过计算空间亲和度 $\mathbf{S}_s$、时间亲和度 $\mathbf{S}_t$ 和时间滞后亲和度 $\mathbf{S}_d$ 三个标量，用 Softmax 融合并 Sigmoid 门控动态调整时空标识符的权重。

6. **领域特定 Prompt 对齐 (DSPA)**：引入低秩可学习 prompt tokens $\mathbf{P} = \mathbf{U}\mathbf{V}^\top$（$r \ll d$），前缀拼接到融合特征中对齐预训练和下游分布。

### 损失函数 / 训练策略

- **UTP 阶段**：Pinball Loss（分位数损失）
- **STA 阶段**：L1 Loss（确定性基准对比时取中位数分位数）
- **持续记忆回放 (CMR)**：将数据集分为记忆缓冲区和当前流，每个 mini-batch 混合两者样本以缓解灾难性遗忘

## 实验关键数据

### 主实验

**少样本预测（短期 12→12）MAE/RMSE：**

| 模型 | PEMS-03 | PEMS-04 | PEMS-07 | PEMS-08 |
|------|---------|---------|---------|---------|
| UniST (STFM) | 40.39/53.44 | 42.76/59.07 | 40.77/54.86 | 35.70/46.74 |
| OpenCity (STFM) | 17.90/28.80 | 24.78/40.41 | 44.43/65.47 | 32.16/48.47 |
| TimesFM (TSFM) | 21.99/35.31 | 27.84/43.15 | 32.61/50.20 | 22.06/33.87 |
| GWNet (STEM) | 17.25/27.79 | 23.27/35.62 | 26.51/41.08 | 18.47/29.04 |
| FactoST (v1) | 17.54/28.10 | 23.93/37.44 | 26.48/41.92 | 18.94/29.59 |
| **FactoST-v2** | **16.75/27.20** | **22.61/35.95** | **24.70/40.79** | **17.65/28.44** |

**全样本预测（短期 12→12）MAE/RMSE：**

| 模型 | PEMS-03 | PEMS-04 | PEMS-07 | PEMS-08 |
|------|---------|---------|---------|---------|
| D2STGNN (STEM) | **14.91/25.82** | **18.75/30.12** | **20.19/33.25** | **14.63/23.73** |
| GWNet (STEM) | 15.93/28.11 | 20.93/32.96 | 23.86/37.83 | 16.48/26.19 |
| **FactoST-v2** | 15.65/24.90 | 20.61/32.81 | 21.95/35.39 | 15.80/25.46 |

### 消融实验

从 v1 到 v2 的关键升级贡献（根据论文描述）：

| 升级项 | 效果 |
|--------|------|
| Encoder-Only (替换 Enc-Dec) | 100% 参数迁移，支持任意长度 |
| 随机序列掩码 | 灵活的长度泛化 |
| 概率分位数预测 | 不确定性量化 |
| DSPA Prompt (替换 HDA) | 更简洁的适配，纯预测目标 |

### 关键发现

- FactoST-v2 在少样本和零样本场景中显著优于联合预训练的 STFMs（UniST、OpenCity）
- 在全样本设置中与领域专家模型（D2STGNN、GWNet）相当甚至更好
- 联合 ST 模型在长期预测时出现 OOM（GWNet、D2STGNN），而 FactoST-v2 保持线性复杂度
- 模型规模从 Minuscule 到 Base 展现出清晰的 scaling law

## 亮点与洞察

- **因式分解范式的核心洞察极其清晰**：时间模式跨域不变，空间模式领域特定，解耦二者是正确的归纳偏置
- 从理论到实践完整论证：包含复杂度分析（线性 vs 二次）和泛化界分析
- Encoder-Only + 随机掩码的设计极其简洁，实现了任意长度泛化和 100% 参数迁移
- 预训练语料覆盖 11B 时间点，涵盖 8 个多样化领域，构建了大规模的时间基础模型

## 局限与展望

- 空间适配仍依赖轻量模块，对复杂拓扑结构（如动态图）的建模能力有限
- 预训练阶段完全忽略空间信息，可能丢失某些有价值的时空耦合模式
- 论文是会议版的期刊扩展，内容非常丰富但也较为复杂
- 在全样本充足数据条件下，与专家模型的差距说明因式分解仍有提升空间

## 相关工作与启发

- 与 TimesFM、Chronos 等纯时间序列基础模型的区别在于加入了空间适配
- 与 UniST、OpenCity 等联合 ST 基础模型的区别在于解耦了时间和空间
- 因式分解的思想可以推广到其他多模态/多领域预训练场景

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 因式分解范式是对 ST 基础模型领域的重要概念突破
- 实验充分度: ⭐⭐⭐⭐⭐ 9 个数据集、16 个基线、五大研究问题全覆盖
- 写作质量: ⭐⭐⭐⭐ 结构完整逻辑清晰，但作为期刊扩展内容过于密集
- 价值: ⭐⭐⭐⭐⭐ 为 ST 基础模型提供了一条高效且可扩展的实用路径

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Revisiting Semi-Supervised Learning in the Era of Foundation Models](revisiting_semi-supervised_learning_in_the_era_of_foundation_models.md)
- [\[ICCV 2025\] B-VLLM: A Vision Large Language Model with Balanced Spatio-Temporal Tokens](../../ICCV2025/model_compression/b-vllm_a_vision_large_language_model_with_balanced_spatio-temporal_tokens.md)
- [\[NeurIPS 2025\] A Partition Cover Approach for Tokenization](a_partition_cover_approach_to_tokenization.md)
- [\[CVPR 2026\] TimeRipples: Accelerating vDiTs by Understanding the Spatio-Temporal Correlations in Latent Space](../../CVPR2026/model_compression/timeripples_accelerating_vdits_by_understanding_the_spatio-temporal_correlations.md)
- [\[NeurIPS 2025\] Universal Cross-Tokenizer Distillation via Approximate Likelihood Matching](universal_cross-tokenizer_distillation_via_approximate_likelihood_matching.md)

</div>

<!-- RELATED:END -->
