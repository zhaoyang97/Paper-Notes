---
title: >-
  [论文解读] SparseRM: A Lightweight Preference Modeling with Sparse Autoencoder
description: >-
  [AAAI 2026][可解释性][稀疏自编码器] SparseRM 利用稀疏自编码器（SAE）从LLM中间表示中提取偏好相关方向，通过投影向量构建轻量级奖励模型，仅需不到1%的可训练参数即可超越大多数主流奖励模型，并在在线迭代对齐框架中表现出更强的泛化能力。 领域现状 奖励模型（RM）是LLM后训练的核心组件…
tags:
  - "AAAI 2026"
  - "可解释性"
  - "稀疏自编码器"
  - "奖励模型"
  - "偏好建模"
  - "LLM对齐"
---

# SparseRM: A Lightweight Preference Modeling with Sparse Autoencoder

**会议**: AAAI 2026  
**arXiv**: [2511.07896](https://arxiv.org/abs/2511.07896)  
**代码**: [github.com/ldc111521/SparseRM](https://github.com/ldc111521/SparseRM)  
**领域**: 模型压缩  
**关键词**: 稀疏自编码器, 奖励模型, 偏好建模, LLM对齐, 可解释性

## 一句话总结
SparseRM 利用稀疏自编码器（SAE）从LLM中间表示中提取偏好相关方向，通过投影向量构建轻量级奖励模型，仅需不到1%的可训练参数即可超越大多数主流奖励模型，并在在线迭代对齐框架中表现出更强的泛化能力。

## 研究背景与动机

### 领域现状
奖励模型（RM）是LLM后训练的核心组件，作为人类偏好评估的代理，指导模型对齐。无论是传统RLHF还是新兴的在线迭代对齐框架，RM都扮演着不可或缺的角色——评估回答质量、构建偏好对、引导策略优化。

### 现有痛点

**训练成本高**：传统RM需要对完整LLM进行微调（即使用LoRA也需要大量参数），计算和内存开销大

**数据依赖强**：需要大规模人类标注的偏好数据，在资源受限场景下难以获取

**分布漂移问题**：RM训练在有监督偏好数据上，但在线对齐时面对策略模型生成的数据，分布差异导致泛化不佳

### 核心矛盾
LLM的中间表示中已编码了丰富的偏好相关特征（如真实性、安全性），这些特征通常与表示空间中的少数线性方向相关。能否直接利用这些已有的表示，而非通过昂贵的微调来学习偏好？

### 本文切入角度
利用SAE将LLM表示分解为可解释的稀疏方向，筛选出偏好相关方向，通过投影计算偏好得分，仅训练一个极轻量的奖励头。

## 方法详解

### 整体框架
SparseRM包含三个步骤：
1. **识别偏好相关方向**：用SAE分解LLM表示，通过激活频率差异筛选偏好相关的潜在变量
2. **计算投影向量**：将表示投影到筛选出的方向上，得到偏好感知向量
3. **偏好建模**：用单层MLP奖励头对投影向量进行打分

### 关键设计

#### 1. **偏好相关方向识别**
- **输入**：给定偏好数据集 $\{x_i, y_w^i, y_l^i\}$，将正面和负面回答分别通过模型 $\mathcal{M}$ 提取目标层 $L$ 的隐藏状态 $\mathbf{z}_w$、$\mathbf{z}_l$
- **SAE分解**：将隐藏状态输入SAE的编码器，得到稀疏潜在表示 $\mathbf{f}_w$、$\mathbf{f}_l$
- **激活频率计算**：
  $$\mu_w^j = \frac{1}{|\mathcal{D}_w|} \sum_{\mathbf{z}_w} \mathbb{I}(f_j(\mathbf{z}_w) > 0), \quad \mu_l^j = \frac{1}{|\mathcal{D}_l|} \sum_{\mathbf{z}_l} \mathbb{I}(f_j(\mathbf{z}_l) > 0)$$
- **分离度评分**：
  $$\nabla_j = \mu_w^j - \mu_l^j, \quad \Delta_j = \mu_l^j - \mu_w^j$$
- **选择Top-K潜在变量**：从正面集和负面集各选K个分离度最高的，对应的解码器方向构成正/负偏好子空间 $\mathbf{F}_w$、$\mathbf{F}_l$
- **设计动机**：在正面和负面样本中激活频率差异大的潜在变量，对应的方向最能区分偏好

#### 2. **投影向量计算**
- 对输入表示 $\mathbf{z}$，计算与每个偏好方向的内积：
  $$\mathbf{p}_w = [\langle \mathbf{z}, \mathbf{d}_{j_w^1}\rangle, \ldots, \langle \mathbf{z}, \mathbf{d}_{j_w^k}\rangle]$$
  $$\mathbf{p}_l = [\langle \mathbf{z}, \mathbf{d}_{j_l^1}\rangle, \ldots, \langle \mathbf{z}, \mathbf{d}_{j_l^k}\rangle]$$
- 拼接得到偏好感知投影向量：$\mathbf{v}_p = [\mathbf{p}_w \mid \mathbf{p}_l]$
- **关键洞察**：直接使用SAE的稀疏激活值效果差（稀疏向量表征能力有限），投影向量能更好保留偏好信息

#### 3. **偏好建模与损失函数**
- 奖励头：单层MLP（隐藏维度512）
- **边际损失**：$\mathcal{L}_{\text{margin}} = \max(0, \gamma - (s_w - s_l))$
- 为什么用边际损失而非BCE：人类更擅长相对比较而非绝对打分，边际损失直接优化偏好对之间的得分差异
- 为什么不用BT损失：边际损失在三个数据集上一致优于BCE和Bradley-Terry损失

#### 4. **集成到在线迭代对齐框架**
- 每轮迭代中，策略模型生成候选回答对
- SparseRM评估每对回答的偏好得分 $(s_w, s_l)$
- 过滤掉 $s_w < s_l$ 的不一致样本
- 保留的高质量偏好对用于DPO训练

### 训练策略
- **极低参数量**：仅训练奖励头（256维输入，512维隐藏），不到LLM参数的1%
- 使用现有的开源SAE（GemmaScope、LlamaScope），无需自行训练SAE
- 对齐训练：DPO，LoRA微调，3个epoch/轮，5轮迭代

## 实验关键数据

### 主实验（RM准确率 + 对齐性能）

| Backbone | 方法 | SafeRLHF | Red-Teaming | TQA MC1 | TQA MC2 | 可训练参数 |
|----------|------|----------|-------------|---------|---------|----------|
| Gemma-2-2B-it | WO RM | 73.4 | 61.8 | 56.1 | 69.8 | — |
| Gemma-2-2B-it | StandardRM | 77.9 | 65.2 | 56.7 | 70.5 | 100% |
| Gemma-2-2B-it | GRAM | 79.0 | 65.8 | 60.0 | 73.9 | 100% |
| Gemma-2-2B-it | **SparseRM** | **79.5** | 67.0 | 59.3 | 73.1 | **<1%** |
| Gemma-2-9B-it | StandardRM | 78.7 | 59.3 | 62.5 | 77.7 | 100% |
| Gemma-2-9B-it | GRAM | 79.3 | 60.7 | 64.7 | 77.9 | 100% |
| Gemma-2-9B-it | **SparseRM** | **79.9** | 60.4 | **65.2** | **78.5** | **<1%** |

### 消融实验

| RM输入 | SafeRLHF | Red-Teaming | TruthfulQA | 说明 |
|--------|----------|-------------|------------|------|
| SAE激活值 | 92.4 | 88.4 | 91.4 | 稀疏激活表征能力有限 |
| 随机方向 | 93.0 | 88.0 | 90.7 | 随机方向下界 |
| **Top-K方向（本文）** | **94.4** | **90.2** | **93.6** | 投影向量最优 |

| 损失函数 | SafeRLHF | Red-Teaming | TruthfulQA | 说明 |
|---------|----------|-------------|------------|------|
| BT Loss | 94.0 | 88.7 | 91.4 | 标准RM损失 |
| BCE Loss | 85.7 | 83.1 | 86.3 | 绝对标签效果差 |
| **Margin Loss** | **94.4** | **90.2** | **93.6** | 一致最优 |

### SparseRM vs DenseRM

| 方法 | RM准确率(SafeRLHF) | 对齐性能(SafeRLHF) | 说明 |
|------|-------------------|-------------------|------|
| DenseRM | **94.7** | 78.7 | 分布内略好，但分布外泛化差 |
| SparseRM | 94.4 | **79.9** | RM准确率接近，对齐性能更优 |

### 关键发现
1. SparseRM仅用不到1%参数就超越或匹配大多数主流RM
2. **投影向量优于SAE激活值**：直接用稀疏激活值的表征能力有限，内积投影能更好保留偏好信息
3. **Margin Loss一致优于BT和BCE**：与人类偏好的相对比较本质一致
4. **SparseRM泛化能力优于DenseRM**：虽然DenseRM在分布内准确率略高，但对齐任务中面对分布漂移时SparseRM更鲁棒
5. 稀疏空间中正负样本分离度更高（t-SNE可视化），能更好过滤噪声样本
6. $K=128$为最优潜在变量数，过少覆盖不足，过多引入噪声
7. 可解释性分析显示Top潜在变量确实对应"判断正确性"等偏好语义

## 亮点与洞察
1. **极致参数效率**：<1%参数达到全参微调水平，打破"RM需要大量参数"的惯例
2. **分布鲁棒性优势**：SAE提取的偏好子空间对分布漂移更鲁棒，这对在线对齐至关重要
3. **可解释性**：通过Neuronpedia可解读每个偏好方向的语义，如latent 4128对应"WRONG, untrue remarks"
4. **与既有SAE生态系统的无缝衔接**：直接使用GemmaScope/LlamaScope，降低使用门槛

## 局限与展望
1. 依赖预训练SAE的质量和覆盖范围，对没有开源SAE的模型不适用
2. 层选择需要实验搜索（如Gemma-2-9B-it仅3个层有SAE）
3. 仅测试了安全性/真实性维度，对更复杂偏好（如有用性、创造性）效果未验证
4. 单层MLP奖励头的表达能力可能在更难任务上受限
5. K值选择对不同任务可能不同，缺乏自动化选择机制

## 相关工作与启发
- SAE作为LLM可解释性工具的能力已被广泛验证，本文将其推广到偏好建模
- 线性表示假说（linear representation hypothesis）是SparseRM的理论基础
- DPO的简洁性使其适合与轻量RM结合，但PPO等更复杂RL方法的兼容性待验证
- 对比DenseRM的分析揭示了一个重要原则：RM的分布外泛化能力比分布内准确率更重要

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ — 首次将SAE用于偏好建模，思路新颖且有效
- 实验充分度: ⭐⭐⭐⭐ — 三个基准+三个backbone+详细消融，但任务维度有限
- 写作质量: ⭐⭐⭐⭐⭐ — 逻辑清晰，DenseRM对比分析深入
- 价值: ⭐⭐⭐⭐⭐ — <1%参数的实用价值极高，推动高效对齐研究

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Data Whitening Improves Sparse Autoencoder Learning](data_whitening_improves_sparse_autoencoder_learning.md)
- [\[CVPR 2026\] Improving Sparse Autoencoder with Dynamic Attention](../../CVPR2026/interpretability/improving_sparse_autoencoder_with_dynamic_attention.md)
- [\[ICML 2026\] PolySAE: Modeling Feature Interactions in Sparse Autoencoders via Polynomial Decoding](../../ICML2026/interpretability/polysae_modeling_feature_interactions_in_sparse_autoencoders_via_polynomial_deco.md)
- [\[ICML 2026\] Rational Sparse Autoencoder](../../ICML2026/interpretability/rational_sparse_autoencoder.md)
- [\[ICML 2026\] CorrSteer: Generation-Time LLM Steering via Correlated Sparse Autoencoder Features](../../ICML2026/interpretability/corrsteer_generation-time_llm_steering_via_correlated_sparse_autoencoder_feature.md)

</div>

<!-- RELATED:END -->
