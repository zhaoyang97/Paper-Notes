---
title: >-
  [论文解读] AMPO: Active Multi-Preference Optimization for Self-play Preference Selection
description: >-
  [ICML 2025][LLM对齐][多偏好优化] 提出 AMPO 框架，将在线策略生成、多偏好组对比损失和主动子集选择相结合，通过从大规模候选响应池中智能挑选少量但信息丰富的子集进行偏好优化，在 AlpacaEval 上达到 SOTA。 传统偏好优化（如 DPO）依赖成对比较，无法充分捕捉人类判断的微妙之处…
tags:
  - "ICML 2025"
  - "LLM对齐"
  - "多偏好优化"
  - "主动学习"
  - "子集选择"
  - "对比学习"
  - "自博弈对齐"
  - "k-medoids"
---

# AMPO: Active Multi-Preference Optimization for Self-play Preference Selection

**会议**: ICML 2025

**arXiv**: [2502.18293](https://arxiv.org/abs/2502.18293)

**代码**: [HuggingFace Datasets](https://huggingface.co/Multi-preference-Optimization)

**领域**: LLM对齐

**关键词**: 多偏好优化, 主动学习, 子集选择, 对比学习, 自博弈对齐, k-medoids

## 一句话总结

提出 AMPO 框架，将在线策略生成、多偏好组对比损失和主动子集选择相结合，通过从大规模候选响应池中智能挑选少量但信息丰富的子集进行偏好优化，在 AlpacaEval 上达到 SOTA。

## 研究背景与动机

传统偏好优化（如 DPO）依赖成对比较，无法充分捕捉人类判断的微妙之处。多偏好方法通过同时考虑整组响应来提供更丰富的对齐信号，但面临关键瓶颈：现代 LLM 每个查询可轻松生成数十个候选响应，将所有响应纳入训练目标在计算上不可行。

具体问题包括：

**冗余问题**：大量采样响应高度相似或接近重复，对梯度更新提供有限增量信息

**计算瓶颈**：处理所有生成响应导致内存爆炸和训练收益递减

**覆盖不足**：仅关注最好和最差答案可能忽略关键的中间模式——那些存在微妙失败模式的"岛屿"

论文用"岛屿隐喻"阐述：每个 prompt 的回答空间可看作一组语义岛屿，理想的子集选择策略应覆盖所有岛屿，而非仅关注最高峰或最低谷。

## 方法详解

### 整体框架

AMPO 统一了三个核心组件：

1. **在线策略数据生成**：模型从自身当前策略分布采样响应
2. **组对比偏好学习**：使用 reference-free 的 SWEPO/REFA 目标函数
3. **主动子集选择**：从大候选池中选择小而高效的训练子集

对于每个 prompt $x$，从策略 $P_\theta(\cdot|x)$ 采样 $N$ 个响应（温度 0.8），用奖励模型打分后，选择 $K < N$ 个子集进行训练。

### 关键设计：主动选择策略

#### AMPO-BottomK（基线）

最简单方法：直接选择 $k$ 个最低奖励响应作为负样本：

$$S^- = \text{argtopk}_i(-r_i, k)$$

缺点：可能遗漏奖励略高于 bottom-k 但对学习至关重要的问题模式。

#### AMPO-Coreset（聚类选择）

在嵌入空间中将 $N$ 个候选响应聚类为 $k$ 簇，从每簇中选最低奖励响应：

$$i_j^- = \arg\min_{i \in C_j} r_i, \quad j = 1, \ldots, k$$

保证每个语义"模式"至少贡献一个负样本，实现对不同语义区域的广泛覆盖。

#### AMPO-OptSelect（理论最优选择）

基于 Lipschitz 连续性假设最大化期望奖励。定义权重 $w_i = \exp(\bar{r} - r_i)$（低奖励权重更大），覆盖代价：

$$\text{Cost}(S) = \sum_{i=1}^n w_i \min_{j \in S} A_{i,j}$$

其中 $A_{i,j} = \|\mathbf{e}_i - \mathbf{e}_j\|_2$。最小化此代价等价于加权 k-medoids 问题，可通过混合整数规划（MIP）或局部搜索近似求解。

### 损失函数

采用 reference-free 的组对比目标（SWEPO/REFA）：

$$L_{\text{swepo}}(\theta) = -\log\left(\frac{\sum_{i \in S^+} \exp[s'_\theta(y_i|x)]}{\sum_{i \in (S^+ \cup S^-)} \exp[s'_\theta(y_i|x)]}\right)$$

其中 $s'_\theta(y_i|x) = \log P_\theta(y_i|x) + \alpha(r_i - \bar{r})$，$\alpha$ 为超参数。该损失鼓励模型提高正样本的对数概率，同时降低负样本的概率。

## 实验关键数据

### 主实验结果

| 方法 | AlpacaEval LC (%) | AlpacaEval WR (%) | Arena-Hard WR (%) | MT-Bench |
|------|-------------------|--------------------|--------------------|----------|
| GPT-4 Base | 28.4 | 28.4 | 26.9 | 7.93 |
| Best-vs-worst (SimPO) | 47.6 | 44.7 | 34.6 | 7.51 |
| AMPO-BottomK | 50.8 | 50.5 | 35.3 | 8.11 |
| **AMPO-Coreset** | **52.4** | **52.1** | **39.4** | **8.12** |
| AMPO-OptSelect | 51.6 | 51.2 | 37.9 | 7.96 |

基础模型为 Llama-3-Instruct 8B，AMPO 的各变体在所有指标上均超越 SimPO 等强基线。

### 消融实验：关键超参数

| 分析维度 | 关键发现 |
|----------|----------|
| 采样温度 | 随温度增加性能普遍下降；Coreset 和 OptSelect 对温度变化更鲁棒 |
| Gamma 参数 | gamma 从 1 到 3 时，LC-WR 和 WR 分数持续提升 |
| β 参数 | 5.0-10.0 范围内一致产生强性能 |
| 嵌入空间多样性 | t-SNE 可视化显示 Coreset/OptSelect 选择的响应分布更分散、覆盖更广 |

### 关键发现

1. **BottomK 的局限性**：选择的负样本高度集中在嵌入空间的紧密区域，导致反馈冗余
2. **Coreset 的优势**：覆盖更多不同语义区域，在 Arena-Hard 上提升最为显著（+4.8% vs SimPO）
3. **理论与实践的一致性**：OptSelect 的 Lipschitz 理论保证与实验结果一致

## 亮点与洞察

1. **理论基础扎实**：证明了覆盖代价最小化等价于 Lipschitz 约束下的期望奖励最大化（Theorem 6.1），局部搜索可在多项式时间内达到 5 倍近似保证（Theorem 6.2）
2. **岛屿隐喻直观**：将子集选择问题类比为覆盖语义岛屿，帮助理解为何多样性选择优于仅基于评分的选择
3. **实用性强**：不需要穷举所有响应，少量精选子集即可实现更优对齐
4. **开源贡献**：公开了 AMPO-Coreset-Selection 和 AMPO-Opt-Selection 数据集

## 局限性

1. 实验主要在 Llama-3 8B 上验证，缺少更大模型规模的测试
2. Lipschitz 常数 $L$ 在实际中难以精确估计
3. 奖励模型的质量直接影响子集选择的有效性
4. 单正样本假设（仅选择最高奖励作为正样本）可能不够灵活

## 相关工作与启发

- **偏好优化（DPO/SimPO/ORPO）**：AMPO 在多偏好框架下统一了这些方法，用组对比替代成对比较
- **主动学习**：将响应选择视为主动学习问题，从信息论角度挑选最有价值样本
- **Coreset 构建**：借鉴计算几何中的核心集理论，实现高效子集覆盖
- **启发**：可将主动选择思想推广到其他对齐场景（如 RLHF 中的查询选择）

## 评分

- 新颖性：⭐⭐⭐⭐ — 组对比+主动选择的统一框架有新意，理论分析与实践结合好
- 实验完整性：⭐⭐⭐⭐ — AlpacaEval/Arena-Hard/MT-Bench 全面评测，消融充分
- 实用价值：⭐⭐⭐⭐ — 直接可用于改进 LLM 对齐训练流程
- 推荐指数：⭐⭐⭐⭐ — 强烈推荐关注 LLM 对齐和偏好优化的研究者阅读

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] ActiveDPO: Active Direct Preference Optimization for Sample-Efficient Alignment](../../ICLR2026/llm_alignment/activedpo_active_direct_preference_optimization_for_sample-efficient_alignment.md)
- [\[ACL 2025\] AutoMixAlign: Adaptive Data Mixing for Multi-Task Preference Optimization in LLMs](../../ACL2025/llm_alignment/automixalign_adaptive_data_mixing.md)
- [\[ACL 2025\] Expectation Confirmation Preference Optimization for Multi-Turn Conversational Recommendation Agent](../../ACL2025/llm_alignment/expectation_confirmation_preference_optimization_for_multi-turn_conversational_r.md)
- [\[ICML 2025\] TGDPO: Harnessing Token-Level Reward Guidance for Enhancing Direct Preference Optimization](tgdpo_harnessing_token-level_reward_guidance_for_enhancing_direct_preference_opt.md)
- [\[ACL 2025\] AMoPO: Adaptive Multi-objective Preference Optimization without Reward Models and Reference Models](../../ACL2025/llm_alignment/amopo_adaptive_multi-objective_preference_optimization_without_reward_models_and.md)

</div>

<!-- RELATED:END -->
