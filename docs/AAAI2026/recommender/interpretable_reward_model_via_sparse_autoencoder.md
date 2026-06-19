---
title: >-
  [论文解读] Interpretable Reward Model via Sparse Autoencoder
description: >-
  [AAAI 2026 Oral][推荐系统][奖励模型] 提出 SARM（Sparse Autoencoder-enhanced Reward Model），将预训练的稀疏自编码器集成到奖励模型中，将隐层激活映射到可解释的稀疏单义特征空间，实现特征级的奖励归因和动态偏好操控，同时在 RewardBench 2 上取得了所有模型中的最高分。
tags:
  - "AAAI 2026 Oral"
  - "推荐系统"
  - "奖励模型"
  - "稀疏自编码器"
  - "可解释性"
  - "RLHF"
  - "偏好操控"
---

# Interpretable Reward Model via Sparse Autoencoder

**会议**: AAAI 2026 Oral  
**arXiv**: [2508.08746](https://arxiv.org/abs/2508.08746)  
**代码**: [https://github.com/schrieffer-z/sarm](https://github.com/schrieffer-z/sarm)  
**领域**: 推荐系统 / LLM对齐  
**关键词**: 奖励模型, 稀疏自编码器, 可解释性, RLHF, 偏好操控

## 一句话总结

提出 SARM（Sparse Autoencoder-enhanced Reward Model），将预训练的稀疏自编码器集成到奖励模型中，将隐层激活映射到可解释的稀疏单义特征空间，实现特征级的奖励归因和动态偏好操控，同时在 RewardBench 2 上取得了所有模型中的最高分。

## 研究背景与动机

### 问题背景

RLHF 是当前主流的 LLM 对齐范式，其中奖励模型（RM）作为人类偏好的代理来指导策略优化。RM 通常是一个 LLM 加上标量值头，对输入-响应对输出一个标量奖励分数。然而，RM 的准确性、可靠性和**可解释性**直接影响下游模型的对齐效果。

### 传统 RM 的两大缺陷

**缺乏可解释性**：标量奖励信号本身是不透明的，无法解释为什么某个回复得分高或低。这使得难以确认模型是否真正对齐了人类价值观，还是仅仅利用了训练数据中的虚假相关

**偏好操控不灵活**：训练完成后，传统 RM 是静态的，无法动态适应用户偏好的变化。这种刚性加上不透明性严重限制了实际应用

### 多维 RM 的尝试与不足

已有工作（如 ArmoRM、HelpSteer2）探索多维奖励建模来提升可解释性：训练回归层在标注的多维数据（如有用性、冗长度评分）上生成多维分数，再加权汇总。但存在两个关键限制：

**缺乏特征级可解释性**：各维度本身仍是不透明的，无法将决策归因到可解释的特征

**标注成本大幅增加**：需要在多个维度上进行人工评分标注，可扩展性差且存在主观性

### 核心洞察

稀疏自编码器（SAE）已被证明能将 LLM 的隐层激活分解为**单义（monosemantic）** 的可解释特征。如果将 SAE 集成到 RM 中，就可以将奖励分数直接归因到这些可解释特征，同时通过修改值头权重实现精细化的偏好操控。

## 方法详解

### 整体框架

SARM 采用两阶段训练流程：
1. **Stage 1：序列级 SAE 预训练** — 在通用语料上训练 SAE 提取可解释特征
2. **Stage 2：奖励建模** — 将 SAE 编码器集成到 RM，训练值头

### 关键设计

#### 1. **序列级 SAE 预训练**

**与传统 token 级 SAE 的区别**：

传统研究在 token 级别的激活上训练 SAE，提取的是 token 级特征。但奖励建模关注的是**整体响应质量**，需要更抽象的序列级特征。

研究发现（来自 Anthropic），句子最后一个 token 的激活具有特殊的激活模式。因此 SARM **仅在每个句子最后一个 token 的激活上**训练 SAE。

**具体流程**：

给定输入序列 $\mathbf{T}$，获取 RM 第 $l$ 层的隐状态：

$$\mathbf{X} = [\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_{\text{last}}] = \text{RM}_\theta^l(\mathbf{T})$$

提取最后一个 token 的激活 $\mathbf{x}_{\text{last}}$，通过 TopK SAE 编码：

$$\mathbf{z} = \text{TopK}(\mathbf{W}_{\text{enc}}(\mathbf{x}_{\text{last}} - \mathbf{b}_{\text{pre}}))$$

$$\hat{\mathbf{x}}_{\text{last}} = \mathbf{W}_{\text{dec}} \mathbf{z} + \mathbf{b}_{\text{pre}}$$

其中 $M = 16 \times d$（特征维度为隐层维度的 16 倍），稀疏度 $k = \frac{3}{64} d$。

训练最小化重建误差：$\mathcal{L} = \|\mathbf{x} - \hat{\mathbf{x}}\|_2^2$

**训练细节**：使用 50M 序列（约 1B tokens）来自 OpenWebText2，从模型 $\frac{1}{2}$ 深度的层提取激活（平衡表示质量和计算效率）。

#### 2. **奖励建模**

将预训练的 SAE 编码器嵌入 RM 的第 $l$ 层，**丢弃 $l$ 层之后的所有层**，直接在稀疏特征向量上应用可学习的线性值头：

$$r_{(x,y)} = h(\mathbf{z}) = \sum_{i=1}^{M} z_i \cdot w_i$$

- $z_i$：特征 $i$ 的激活强度（由冻结的 SAE 编码器产生）
- $w_i$：值头的可学习权重

**训练目标**：标准 Bradley-Terry 损失：

$$\mathcal{L}(\theta) = -\mathbb{E}_{(x, y_c, y_r) \sim \mathcal{D}} [\log \sigma(r_\theta(x, y_c) - r_\theta(x, y_r))]$$

仅需偏好数据（chosen vs rejected），**无需多维标注**。

训练时冻结 SAE 编码器参数，仅训练前 $l$ 层的 backbone 和最终的线性值头。

#### 3. **可解释性与偏好操控**

**特征归因**：
由于 TopK 稀疏性，每次推理只有少量特征被激活（$z_i > 0$），奖励分数可以直接分解为各可解释特征的贡献。

**正面特征示例**：
- Feature 58353：捕捉结构化的分析内容（计算、编程、数学推理）
- Feature 60427：捕捉伦理考量（隐私、尊重、负责任的交流）
- 这些特征的值头权重 $w_i$ 为正

**负面特征示例**：
- Feature 13950：捕捉贬损/冒犯性语气
- Feature 17289：在涉及黑客、信用卡盗窃等不道德建议的上下文中激活
- 这些特征的值头权重 $w_i$ 为负

**动态偏好操控**：
由于 SAE 特征近似正交且单义，修改值头权重 $w_i$ 可以精细地控制 RM 偏好：
- 增大 $w_i$：增强特征 $i$ 对奖励的贡献
- 减小 $w_i$：抑制特征 $i$ 的影响
- 由于 $w_i$ 不影响激活 $z_i$，对未激活该特征的样本无影响

### 损失函数 / 训练策略

- SAE 预训练：仅重建损失，Adam 优化器，lr = 5e-4
- RM 训练：Bradley-Terry 损失，仅在 Skywork-Reward-Preference-80K-v0.2 上训练 3 个 epoch，batch size 512，lr = 4e-6
- 每 10 步对解码器列做单位范数正则化

## 实验关键数据

### 主实验

RewardBench 2 上的性能对比（Overall 分数越高越好）：

| 模型 | 参数量 | Overall | Factuality | Precise IF | Math | Safety | Focus | Ties |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ArmoRM-8B | 7.5B | 66.5 | 65.7 | 41.9 | 66.1 | 82.2 | 76.6 | 66.3 |
| Skywork-8B | 7.5B | 71.8 | 69.7 | 40.6 | 60.1 | 94.2 | 94.1 | 71.7 |
| Tulu-70B | 70B | 72.2 | 80.8 | 36.9 | 67.8 | 86.9 | 77.8 | 83.1 |
| GPT-4o | — | 64.9 | 56.8 | 33.1 | 62.3 | 86.2 | 72.9 | 78.2 |
| GPT-4.1 | — | 72.3 | 82.9 | 39.7 | 65.2 | 87.3 | 73.4 | 85.4 |
| Claude Sonnet 4 | — | 71.2 | 76.1 | 35.9 | 70.5 | 89.1 | 76.0 | 79.4 |
| SARM-2B | 2.0B | 62.5 | 55.6 | 35.6 | 60.7 | 84.9 | 82.4 | 56.0 |
| SARM-3B | 2.7B | 64.2 | 58.6 | 34.4 | 62.8 | 87.3 | 86.3 | 55.6 |
| **SARM-4B** | **4.3B** | **73.6** | 68.5 | **42.5** | 63.9 | 91.3 | **96.0** | 79.6 |

**SARM-4B 以 4.3B 参数取得所有模型（包括 70B 开源和闭源模型）中的最高分 73.6**，在 Focus 维度上更是达到 96.0。

### 消融实验

| 配置 | 参数量 | Overall | Safety | Focus | Ties | 说明 |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| SAE 随机初始化 | (4+0.3)B | 68.4 | 88.9 | 88.2 | 64.9 | 无预训练 SAE |
| Token 级 SAE 预训练 | (4+0.3)B | 71.5 | 92.9 | 92.5 | 72.5 | Token级特征 |
| **SARM-4B** | **(4+0.3)B** | **73.6** | 91.3 | **96.0** | **79.6** | 序列级特征 |

1. SAE 随机初始化 → 68.4：证明 SARM 不是靠参数量，而是靠 SAE 提取的结构化特征
2. Token 级 SAE → 71.5：序列级预训练比 token 级提升 2.1 分，证明序列级特征更适合奖励建模
3. 两个组件缺一不可：预训练 SAE + 序列级策略的组合才能实现最佳性能

### 偏好操控实验

操控安全相关特征的权重后：
- **目标集 T**（安全查询+chosen响应）：奖励分布明显右移，表明 RM 正确地给安全响应更高奖励
- **补集 C**（其余样本）：奖励分布几乎不变，表明操控是精准的、不会影响无关属性

### 关键发现

1. **可解释性不损害性能**：SARM 不仅可解释，还取得了最佳性能，颠覆了"可解释性vs性能"的权衡
2. **参数效率惊人**：4.3B 的 SARM 超越了 70B 的 Tulu-3 和闭源的 GPT-4.1
3. **特征具有语义意义**：正面特征（数学推理、伦理考量）的权重为正，负面特征（冒犯性、非法建议）的权重为负，语义和权重极性高度一致
4. **偏好操控精准可控**：修改单个特征权重只影响激活该特征的样本，不影响其他样本

## 亮点与洞察

1. **优雅的架构设计**：通过"预训练SAE + 冻结编码器 + 可学习值头"的简洁流程，同时实现了可解释性和高性能
2. **序列级 SAE 的洞察**：利用句末 token 的特殊激活模式进行序列级特征提取，比 token 级更适合全局质量评估
3. **丢弃后半层的大胆设计**：直接在中间层接入 SAE 并丢弃后续层，说明 RM 的有用信息集中在中间层
4. **因果可控性**：不仅是事后解释，还能通过修改权重进行因果干预，这是多维 RM 无法做到的

## 局限与展望

1. **dead latents 问题**：部分 SAE 特征很少被激活，导致可解释的特征数量小于 $M$
2. **依赖 GPT-4o 做特征解释**：自动解释的质量受限于 GPT-4o 的能力，且存在成本
3. **中间层选择的先验**：固定在 $\frac{1}{2}$ 深度的层，虽然有消融但最优层位置可能因模型而异
4. **仅在 Llama-3 上验证**：未在其他架构（如 Mistral、Qwen）上验证泛化性
5. **特征间的交互未建模**：当前值头是线性加权，未考虑特征之间的非线性交互
6. **偏好操控仅在安全维度验证**：需要在更多维度上验证操控的精准性

## 相关工作与启发

- **Anthropic 的 SAE 工作**（Claude Scaling, Towards Monosemanticity）：SARM 将 SAE 的可解释性能力从"理解 LLM"扩展到"控制 RM"
- **Llama Scope / Gemma Scope**：层级 SAE 训练的基础设施
- **ArmoRM / HelpSteer2**：多维 RM 的先驱，但需要昂贵的多维标注
- **TopK SAE**：通过显式控制激活数量来平衡稀疏性和重建质量
- 启发：SAE 不仅是 LLM 可解释性工具，更可以成为模型可控性的接口

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 将 SAE 集成到 RM 是全新思路，序列级预训练有原创性
- **实验充分度**: ⭐⭐⭐⭐ — RewardBench 2 结果强力，消融完整，但偏好操控仅在安全维度验证
- **写作质量**: ⭐⭐⭐⭐⭐ — 论述清晰，动机-方法-实验一脉贯通
- **价值**: ⭐⭐⭐⭐⭐ — 在 RM 可解释性和可控性上的突破性工作，对 RLHF 安全研究有重大意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] PARM: Multi-Objective Test-Time Alignment via Preference-Aware Autoregressive Reward Model](../../ICML2025/recommender/parm_multi-objective_test-time_alignment_via_preference-aware_autoregressive_rew.md)
- [\[NeurIPS 2025\] Inference-Time Reward Hacking in Large Language Models](../../NeurIPS2025/recommender/inference-time_reward_hacking_in_large_language_models.md)
- [\[ICLR 2026\] Adaptive Regularization for Large-Scale Sparse Feature Embedding Models](../../ICLR2026/recommender/adaptive_regularization_for_large-scale_sparse_feature_embedding_models.md)
- [\[ICLR 2026\] GoalRank: Group-Relative Optimization for a Large Ranking Model](../../ICLR2026/recommender/goalrank_group-relative_optimization_for_a_large_ranking_model.md)
- [\[ICML 2025\] Recommendations with Sparse Comparison Data: Provably Fast Convergence for Nonconvex Matrix Factorization](../../ICML2025/recommender/recommendations_with_sparse_comparison_data_provably_fast_convergence_for_noncon.md)

</div>

<!-- RELATED:END -->
