---
title: >-
  [论文解读] Connectivity-Guided Sparsification of 2-FWL GNNs Preserving Full Expressivity
description: >-
  [AAAI 2026][LLM效率][图神经网络] Co-Sparsify 提出一种基于连通性感知的稀疏化框架，通过将 3-节点交互限制在双连通分量内、2-节点交互限制在连通分量内，消除可证明冗余的计算，在保持完整 2-FWL 表达力的同时显著提升效率，在合成子结构计数任务和 ZINC、QM9 等基准上取得 SOTA。
tags:
  - "AAAI 2026"
  - "LLM效率"
  - "图神经网络"
  - "2-FWL"
  - "图稀疏化"
  - "双连通分量"
  - "表达力保持"
---

# Connectivity-Guided Sparsification of 2-FWL GNNs Preserving Full Expressivity

**会议**: AAAI 2026  
**arXiv**: [2511.12838](https://arxiv.org/abs/2511.12838)  
**代码**: [有](https://github.com/RongqinChen/HOGNN-Sparsify)  
**领域**: LLM效率  
**关键词**: 高阶GNN, 2-FWL, 图稀疏化, 双连通分量, 表达力保持

## 一句话总结

Co-Sparsify 提出一种基于连通性感知的稀疏化框架，通过将 3-节点交互限制在双连通分量内、2-节点交互限制在连通分量内，消除可证明冗余的计算，在保持完整 2-FWL 表达力的同时显著提升效率，在合成子结构计数任务和 ZINC、QM9 等基准上取得 SOTA。

## 研究背景与动机

**领域现状**：标准消息传递 GNN（GCN、GIN、GAT 等）的表达力受限于 1-WL 测试。高阶 GNN（HOGNN）基于 k-WL/k-FWL 层级通过 k 元组消息传递提升表达力，但计算复杂度为 O(n^k)。2-FWL GNN 在实用性和表达力之间取得平衡，匹配 3-WL 表达力，但仍需 O(n^3) 计算和 O(n^2) 内存。

**核心痛点**：现有效率优化方法（子图采样 ESAN、集合化约 KCSetGNN、局部聚合 1-2-3-GNN）都以牺牲表达力为代价换取效率。一个自然的问题是：能否仅消除对表达力冗余的计算来提升效率？

**关键洞察**：3-节点交互 (u,t,v) 在表达力上仅在双连通分量内是必需的。由 Menger 定理，双连通分量内任意两节点之间至少存在两条内部不相交路径，需要 3-节点交互来检测路径多样性。在双连通分量之外，3-节点交互是冗余的——割点分隔的情况下 2-节点交互即可完整表达，断连节点对的结构信息由 readout 捕获。

## 方法详解

### 整体框架

Co-Sparsify 是一个连通性感知的稀疏化方案，分三步：

1. **预处理**：通过 BFS 识别连通分量，通过 block-cut tree 分解识别双连通分量，时间复杂度 O(n+m)
2. **构建稀疏化邻域**：对每个节点对 (u,v) 定义 Co-Sparsified 邻域集合
3. **稀疏化消息传递**：仅在必要的交互上执行 2-FWL 更新

### 关键设计

**1. 连通性引导的稀疏化原理**

三条核心引理支撑了稀疏化的正确性：

- **引理 1（双连通分量内 3-节点交互的必要性）**：若 u 和 v 在同一双连通分量中，由 Menger 定理它们之间至少有两条内部不相交路径，检测这种路径多样性需要 3-节点交互，2-节点交互不足以捕获。
- **引理 2（跨割点 3-节点交互的冗余性）**：若 u 到 v 经 t 的所有路径都经过一个将 t 与 {u,v} 分隔的割点 c，则 (u,t,v) 的 3-节点交互可被 2-节点交互 (u,t) 和 (t,v) 完全表达，移除不影响 2-FWL 表达力。
- **引理 3（断连分量间交互的无关性）**：不同连通分量中的节点对无共享路径，其结构独立性由 readout 函数隐式编码，无需显式建模。

**2. Co-Sparsified 邻域构建**

对节点对 (u,v) 的稀疏化邻域集合定义：

- 若 u,v 在不同连通分量：邻域为空集
- 若 u,v 在同一连通分量 C 中：
    - **3-节点交互**：仅当 u,t,v 均在同一双连通块 B 中时才包含 ((u,t),(t,v))
    - **2-节点交互**：包含 ((u,u),(u,v)) 和 ((u,v),(v,v)) 用于节点到对的信息传播
    - 对 u!=v，包含 ((v,u),(u,v)) 在自对 (v,v) 的邻域中以传播对到节点的信息
    - 自对 (u,u) 始终包含 ((u,u),(u,u)) 确保非空邻域

**3. Co-Sparsified 消息传递**

初始表示与标准 2-FWL 相同，结合节点特征、边特征和结构编码（RRWP）。在第 l 层，对同一连通分量中的 (u,v)，聚合仅遍历稀疏化邻域中的中间节点 t，而非全部 V。对断连对不做更新。

**4. 多级 Readout**

- **节点级**：在 v 所在连通分量 C 内聚合所有入射对表示
- **图级**：先对每个连通分量 C 分别聚合自对和非对角对，再聚合所有分量得到图表示

**5. 表达力等价性证明（定理 4）**

Co-Sparsified 2-FWL GNN 在注入式聚合和一致初始化条件下，检测子结构的能力与标准 2-FWL GNN 相同。证明关键：当子图 Q 在双连通分量内时所有 3-节点交互被保留；当 Q 跨越块时其结构分解为独立的 2-节点交互；断连查询由 readout 处理。

**6. 计算效率分析**

- 标准 2-FWL：O(n^2) 个对，O(n^3) 个三元组交互
- Co-Sparsify：缩减为按连通分量求和 O(sum n_i^2) 和按双连通分量求和 O(sum n_bj^3)
- 在分子图等高阶连通性稀疏的图上，可从立方降至亚立方复杂度

### 损失函数 / 训练策略

Co-Sparsify 应用于 PPGN 架构，采用逐元素乘法作为对交互函数。在 QM9 上用 L1 损失回归分子性质，ZINC 上用 MAE 损失做图级回归。预处理（block-cut tree 分解）约 1ms/图。

## 实验关键数据

### QM9 分子性质预测

| 模型 | mu | alpha | e_HOMO | e_LUMO | delta_e | ZPVE |
|------|-----|-------|--------|--------|---------|------|
| PPGN+RRWP | 0.332 | 0.200 | 0.00236 | 0.00231 | 0.00327 | 0.00012 |
| **CoSp-PPGN+RRWP** | **0.321** | **0.183** | **0.00214** | **0.00210** | **0.00299** | **0.00012** |
| N2-GNN | 0.333 | 0.193 | 0.00217 | 0.00210 | 0.00304 | 0.00013 |

CoSp-PPGN+RRWP 在 12 个目标中 10 个取得 top-2 性能。

### ZINC 图回归

| 模型 | 参数量 | ZINC-Subset MAE | ZINC-Full MAE |
|------|-------|-----------------|---------------|
| GRIT | ~500k | 0.059+/-0.002 | 0.023+/-0.001 |
| N2-GNN | 316k/414k | 0.059+/-0.002 | 0.022+/-0.002 |
| PPGN+RRWP | 478K | 0.055+/-0.002 | 0.020+/-0.002 |
| **CoSp-PPGN+RRWP** | **478K** | **0.050+/-0.001** | **0.018+/-0.002** |

### 消融实验（子结构计数 normalized MAE）

| 子结构 | PPGN | CoSp-PPGN |
|--------|------|-----------|
| Tailed Triangle | 0.0025 | **0.0016** |
| 4-Cycles | 0.0024 | **0.0007** |
| 5-Cycles | 0.0039 | **0.0032** |
| 6-Cycles | 0.0075 | **0.0056** |

CoSp-PPGN 在所有子结构计数任务上匹配或超过 PPGN。

### 关键发现

- 运行效率：时间减少 13-60%（ZINC-subset 9.3s->7.9s/epoch，QM9 97.1s->60.7s）
- 内存减少 12-52%（ZINC-subset 3.7->3.0GB，QM9 6.4->4.2GB）
- 预处理开销可忽略（约 1ms/图）
- TUD 分类基准上同样 SOTA（FRANKENSTEIN 77.65%，NCI1 82.87%）

## 亮点与洞察

- 首个为 HOGNN 提供表达力保持保证的稀疏化方法——不靠近似、采样或启发式剪枝
- 洞察精妙：将图论连通性理论（Menger 定理、block-cut tree）与 GNN 表达力分析结合
- 实现优雅：仅需 O(n+m) 预处理，然后消息传递遵循标准 2-FWL 格式
- 表达力和效率不是对立的——通过结构洞察可以两者兼得

## 局限与展望

- 在需要长程依赖的任务（如 Peptides-struct）上，HOGNN 的 over-squashing 问题导致性能不佳
- 仅聚焦 2-FWL 层级，未扩展到更高阶（k>2）
- 对高度双连通的图（如完全图），稀疏化收益有限
- 局部化变体（限制最短路径距离<=4）在某些任务上效果更好，但缺乏系统化的自适应裁剪机制

## 相关工作与启发

- **PPGN**：2-FWL GNN 的批处理实现，Co-Sparsify 的基础架构
- **ESAN / KCSetGNN / 1-2-3-GNN**：先前效率方法均牺牲表达力
- **N2-GNN**：重新设计 Folklore WL 的通道，Co-Sparsify 在保持更强理论保证的同时取得更好性能
- **Block-cut tree**：经典图论工具（Hopcroft-Tarjan 算法），Co-Sparsify 将其创造性地应用于 GNN 稀疏化

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 创新性 | 5 |
| 理论深度 | 5 |
| 实验充分度 | 4 |
| 写作质量 | 5 |
| 实用性 | 4 |
| 总评 | 4.6 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] StructKV: Preserving the Structural Skeleton for Scalable Long-Context Inference](../../ACL2026/llm_efficiency/structkv_preserving_the_structural_skeleton_for_scalable_long-context_inference.md)
- [\[CVPR 2026\] QuietPrune: Query-Guided Early Token Pruning for Vision-Language Models](../../CVPR2026/llm_efficiency/quietprune_query-guided_early_token_pruning_for_vision-language_models.md)
- [\[ICML 2026\] TuneAhead: Predicting Fine-tuning Performance Before Full Training Begins](../../ICML2026/llm_efficiency/tuneahead_predicting_fine-tuning_performance_before_full_training_begins.md)
- [\[ACL 2025\] Consistency-Preserving Contrastive Decoding for Faithful Document-Grounded Dialogue](../../ACL2025/llm_efficiency/consistency-preserving_contrastive_decoding_for_faithful_document-grounded_dial.md)
- [\[ACL 2025\] Entailment-Preserving First-order Logic Representations in Natural Language Entailment](../../ACL2025/llm_efficiency/entailment-preserving_first-order_logic_representations_in_natural_language_enta.md)

</div>

<!-- RELATED:END -->
