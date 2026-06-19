---
title: >-
  [论文解读] Brain-Like Processing Pathways Form in Models With Heterogeneous Experts
description: >-
  [NeurIPS 2025][Mixture-of-Experts] 在异构 Mixture-of-Experts 模型中，异构专家并不会自动形成处理通路；本文提出三个受大脑启发的归纳偏置（路由代价、任务表现缩放、专家 Dropout），使模型形成类似大脑"皮层-皮层下"动态通路的 Mixture-of-Pathways 架构。
tags:
  - "NeurIPS 2025"
  - "Mixture-of-Experts"
  - "异构专家"
  - "处理通路"
  - "归纳偏置"
  - "类脑计算"
  - "认知任务"
---

# Brain-Like Processing Pathways Form in Models With Heterogeneous Experts

**会议**: NeurIPS 2025  
**arXiv**: [2506.02813](https://arxiv.org/abs/2506.02813)  
**代码**: [jackcook/mixture-of-pathways](https://github.com/jackcook/mixture-of-pathways)  
**领域**: others (计算神经科学 × MoE 架构)  
**关键词**: Mixture-of-Experts, 异构专家, 处理通路, 归纳偏置, 类脑计算, 认知任务  

## 一句话总结

在异构 Mixture-of-Experts 模型中，异构专家并不会自动形成处理通路；本文提出三个受大脑启发的归纳偏置（路由代价、任务表现缩放、专家 Dropout），使模型形成类似大脑"皮层-皮层下"动态通路的 Mixture-of-Pathways 架构。

## 研究背景与动机

**大脑的异构通路组织**：大脑由大量异构脑区组成，这些区域根据任务需求动态组织成处理通路（如视觉通路、认知控制网络），但通路形成的机制尚不清楚。

**MoE 模型的专家专化不足**：现有 Mixture-of-Experts 模型（如 DeepSeek-MoE、Mixtral）理论上应形成任务相关的专家通路，但实践中专家专化程度有限，难以形成稳定的功能通路。

**多区域交互建模的局限**：已有的多脑区模型通常预定义特定的连接结构，无法研究区域如何动态交互形成通路；允许动态交互的模型又无法完成标准认知任务。

**代谢优化假说**：大脑的代谢成本最小化是理解脑结构与功能的核心理论，但尚未被系统性地用于驱动多专家模型中的通路形成。

**异构 MoE 的新可能**：Heterogeneous MoE（HMoE）允许不同大小/类型的专家共存，为研究异构区域如何自组织为通路提供了天然平台。

**核心问题**：异构区域是否会自动形成功能通路？还是需要额外的先验约束？形成的通路是否类似大脑中观察到的动态适应性通路？

## 方法详解

### 整体框架：Mixture-of-Pathways (MoP)

模型由三层 HMoE 层串联组成，每层包含三个异构专家（16 神经元 GRU、32 神经元 GRU、Skip Connection）和一个 64 神经元 GRU 路由器。路由器根据输入动态决定各专家的贡献权重，信息逐层前向传递。在基线 HMoE 基础上，引入三个归纳偏置使其形成 MoP 模型。模型在 Mod-Cog 数据集的 82 个认知任务上训练，任务涵盖从简单刺激-响应到复杂工作记忆等不同难度。

### 关键设计一：路由复杂度代价 (Routing Cost)

- **功能**：在损失函数中引入 Learned Pathway Complexity (LPC) 惩罚项，使模型倾向使用更简单的专家
- **核心思路**：定义 $LPC_i = \frac{1}{T_i}\sum_t^{T_i}\sum_j^{E} w_{i,j,t} s_j^2$，其中 $w$ 是路由权重，$s_j^2$ 是专家大小的平方（对应存储代价 $O(s_j^2)$）。将 LPC 加入损失函数，惩罚使用大专家
- **设计动机**：灵感来自大脑的代谢优化理论——大脑倾向于最小化能量消耗。此约束迫使模型只在"必要时"才动用更复杂的专家，从而形成与任务难度匹配的差异化通路

### 关键设计二：任务表现缩放 (Task Performance Scaling)

- **功能**：将路由代价除以当前任务的响应损失 $L_{response,i}$，得到最终路由损失 $\frac{\alpha \cdot LPC_i}{L_{response,i} + \epsilon}$
- **核心思路**：当模型在某任务上表现差（损失大）时，路由代价被缩小，允许使用复杂专家来学习；当任务已学好（损失小）时，路由代价被放大，驱动模型将任务迁移到更简单的通路
- **设计动机**：避免模型为了最小化路由代价而陷入"只用 Skip Connection 不解题"的局部最优。类比大脑中认知努力的动态调控——学习新技能时投入更多资源，熟练后转移到更节省的通路

### 关键设计三：随机专家 Dropout

- **功能**：在训练时以概率 $p_j$ 随机去激活低权重专家。当 $w_j < \gamma=0.1$ 时，$p_j = \beta - \frac{\beta}{\gamma}w_j$；否则 $p_j = 0$（$\beta = 0.8$）
- **核心思路**：权重越低的专家被去激活的概率越高（最高 80%），贡献超过 10% 的专家永不被去激活
- **设计动机**：灵感来自大脑信号处理的随机性。Dropout 使模型不能依赖所有专家的"微小贡献"来解题，迫使通路变得自给自足——即移除通路外的专家后，性能不会大幅下降

### 损失函数与训练策略

完整损失函数：

$$L = L_{fix} + \sum_{i}^{\mathcal{T}} \left( L_{response,i} + \frac{\alpha \cdot LPC_i}{L_{response,i} + \epsilon} \right)$$

- $L_{fix}$：注视期均方误差，要求输出为零
- $L_{response,i}$：任务响应期交叉熵损失
- $\alpha = 10^{-5}$：路由代价权重超参数
- 优化器：Schedule-Free AdamW（lr=0.01, betas=(0.9, 0.999)）
- 训练：10 个 epoch × 1000 步，每步 128 batch × 350 timesteps × 115 特征
- 单 NVIDIA T4 GPU 约 1 小时完成训练

## 实验关键数据

### 表1：通路形成的三个判据验证

| 判据 | 指标 | 基线模型 | MoP 模型 |
|------|------|----------|----------|
| 一致性 (Consistency) | 20 次训练的路由模式相关性 | 0.03 | **0.51** (p<0.0001) |
| 自给自足 (Self-sufficiency) | 去除低权重专家后准确率 | 98.2%→16.4% | 85.8%→**74.4%** |
| 区分性 (Distinctness) | 不同任务集群使用不同专家组合 | 均匀分布 | **幂律分布** (p<0.0001) |

### 表2：消融实验——设计选择对类脑通路的影响

| 模型变体 | 准确率 | 难度-复杂度相关 (Fig.5) | 学习动态相关 (Fig.6) |
|----------|--------|--------------------------|----------------------|
| 基线 HMoE | 91.1% ± 8.9% | -0.01 | -0.49*** |
| **MoP 完整模型** | **83.0% ± 15.5%** | **0.54***** | **0.31**** |
| 无 Dropout | 90.1% ± 8.9% | 0.55*** | 0.03 |
| α=1e-4（过强） | 69.0% ± 20.1% | -0.57*** | -0.37*** |
| α=1e-6（过弱） | 89.7% ± 9.0% | 0.62*** | 0.18 |
| 无任务嵌入 | 83.0% ± 14.2% | 0.58*** | 0.58*** |

**关键发现**：

- MoP 模型在难度-通路复杂度上呈**正相关** (r=0.54)，即难任务自动使用更复杂的专家——类比大脑的多需求 (Multiple-Demand) 系统
- 学习动态上，困难任务先被路由到复杂通路、后逐渐迁移至简单通路 (r=0.31)——复现大脑皮层→皮层下的技能转移现象
- Dropout 对学习动态至关重要：无 Dropout 时 Fig.6 相关性从 0.31 降至 0.03
- 三个归纳偏置缺一不可，体现了**交互效应**

## 亮点与洞察

1. **优雅的研究范式**：用 MoE 架构类比大脑多区域交互，用路由权重分析类比神经通路研究，架起了 AI 架构与计算神经科学之间的桥梁
2. **三个判据体系**：提出"一致性、自给自足性、区分性"三个量化判据来评估通路是否形成，为研究 MoE 内部结构提供了新工具
3. **类脑学习动态的涌现**：模型自发地展现出"学习时先用复杂通路、熟练后迁移到简单通路"的行为，与大脑皮层-皮层下交互的实验观察高度吻合
4. **对 MoE 研究的启示**：复杂度感知的路由损失可以作为一种任务驱动的负载均衡策略，可能对大规模 LLM 的 MoE 设计有参考价值

## 局限性

1. **模型规模小**：仅有三层、每层三个专家的小模型，路由代价策略能否扩展到大规模 LLM（如 DeepSeek-MoE 级别）未经验证
2. **仅支持前向传播**：不包含循环/反馈连接，无法建模大脑中大量的环路结构
3. **路由器缺乏生物对应**：未明确将路由器映射到大脑的特定结构（丘脑核团是候选）
4. **任务复杂度有限**：82 个时间序列认知任务仍较为简单，与真实认知挑战差距较大
5. **通路识别依赖多测试**：目前用三个独立测试判断通路是否形成，缺乏统一的量化指标
6. **准确率有所下降**：MoP 模型（83.0%）比基线（91.1%）低约 8 个百分点，通路形成以性能为代价

## 相关工作与启发

- **脑启发模块化**：扩展了 Achterberg et al. 的空间代谢约束工作，从模块形成扩展到通路形成
- **异构 MoE**：在 Wang et al. (HMoE) 和 Raposo et al. (Mixture of Depths) 基础上，首次研究异构专家间如何形成动态通路
- **Multi-region RNN**：相比 Kozachkov et al. 的多区域 RNN，本文的关键创新是基于任务上下文的动态路由能力
- **对 MoE 负载均衡的新视角**：路由复杂度损失可视为一种任务驱动的负载均衡，区别于 DeepSeek-MoE 中基于频率的均衡策略
- **启发**：将代谢约束引入 MoE 不仅是神经科学的建模工具，也可能成为提升 MoE 模型可解释性和效率的实用策略

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首次系统研究异构 MoE 中通路的形成条件，三个归纳偏置的组合设计有独创性
- **实验充分度**: ⭐⭐⭐⭐ — 三个判据体系完整，消融实验详尽，与大脑数据的对比有说服力
- **写作质量**: ⭐⭐⭐⭐⭐ — 叙事逻辑清晰，从"基线不形成通路"→"逐步加偏置"→"与大脑对比"的递进结构非常优雅
- **价值**: ⭐⭐⭐⭐ — 对计算神经科学有直接贡献，对大规模 MoE 设计有启发意义，但需进一步验证可扩展性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Out-of-distribution Generalisation is Hard: Evidence from ARC-like Tasks](out-of-distribution_generalisation_is_hard_evidence_from_arc-like_tasks.md)
- [\[NeurIPS 2025\] Contextual Dynamic Pricing with Heterogeneous Buyers](contextual_dynamic_pricing_with_heterogeneous_buyers.md)
- [\[ICML 2025\] Online Sparsification of Bipartite-Like Clusters in Graphs](../../ICML2025/others/online_sparsification_of_bipartite-like_clusters_in_graphs.md)
- [\[ACL 2025\] Beyond Position: the emergence of wavelet-like properties in Transformers](../../ACL2025/others/beyond_position_the_emergence_of_wavelet-like_properties_in_transformers.md)
- [\[ICML 2025\] Addressing Imbalanced Domain-Incremental Learning through Dual-Balance Collaborative Experts (DCE)](../../ICML2025/others/addressing_imbalanced_domain-incremental_learning_through_dual-balance_collabora.md)

</div>

<!-- RELATED:END -->
