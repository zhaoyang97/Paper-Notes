---
title: >-
  [论文解读] Incremental Maintenance of DatalogMTL Materialisations
description: >-
  [AAAI 2026 Oral][DatalogMTL] 提出 DRed$_{\text{MTL}}$ 算法，将经典 Delete/Rederive 增量维护技术扩展到 DatalogMTL（带度量时序逻辑的 Datalog），通过在周期化物化表示上设计新的 seminaïve 评估算子和周期识别算法，实现高效增量更新，性能可达重新物化的数量级提升。
tags:
  - "AAAI 2026 Oral"
  - "DatalogMTL"
  - "incremental reasoning"
  - "materialisation"
  - "Delete/Rederive"
  - "temporal logic"
---

# Incremental Maintenance of DatalogMTL Materialisations

**会议**: AAAI 2026 Oral  
**arXiv**: [2511.12169](https://arxiv.org/abs/2511.12169)  
**代码**: [GitHub](https://github.com/Horizon12275/DREDmtl-for-DatalogMTL)  
**领域**: 音频语音  
**关键词**: DatalogMTL, incremental reasoning, materialisation, Delete/Rederive, temporal logic  

## 一句话总结

提出 DRed$_{\text{MTL}}$ 算法，将经典 Delete/Rederive 增量维护技术扩展到 DatalogMTL（带度量时序逻辑的 Datalog），通过在周期化物化表示上设计新的 seminaïve 评估算子和周期识别算法，实现高效增量更新，性能可达重新物化的数量级提升。

## 背景与动机

- DatalogMTL 是 Datalog 的时序扩展，支持度量时序逻辑 (MTL) 操作符，应用于本体查询应答、流推理、金融时序推理等领域
- 实际场景（如电力变压器异常检测）中数据频繁更新，但现有 DatalogMTL 推理系统（MeTeoR、vadalog）在数据变更时只能从头重算物化
- 对于经典 Datalog，DRed 等增量维护算法已很成熟，但将其扩展到 DatalogMTL 面临根本性挑战

## 核心问题

如何在 DatalogMTL 的无限时间域上实现增量物化维护，避免每次数据更新都从头重算？

### 关键挑战

1. DatalogMTL 支持时间递归，物化可能覆盖无界时间区间，需要**周期化有限表示**
2. DRed 依赖 seminaïve 评估策略，但其与 DatalogMTL 的周期化结构的交互未被研究
3. 终止检测（周期识别）本身开销大，需要对比 $O(n)$ 个 facts，也需"增量化"

## 方法详解

### 周期化物化表示

给定有界程序-数据集对 $(\Pi, E)$，其规范模型 $\mathfrak{C}_{\Pi,E}$ 可通过**饱和解释**有限表示：找到左右周期 $\varrho_{\text{left}}, \varrho_{\text{right}}$，使得这些区间外的内容是周期重复的。周期化物化 $\mathds{I} = \langle I, \varrho_L, \varrho_R \rangle$ 通过展开 (unfolding) 可恢复完整规范模型。

### DRed$_{\text{MTL}}$ 三阶段算法

1. **Overdeletion（过度删除）**：从删除集 $E^-$ 出发，迭代应用规则找到所有依赖于被删除事实的推导事实，使用新 seminaïve 算子 $\Pi\langle I \vdots \Delta \rangle$ 高效定位受影响事实
2. **Rederivation（重新推导）**：识别被过度删除但仍应成立的事实，逐步扩展感兴趣区间 $[t_L, t_R]$ 以覆盖可能的恢复范围
3. **Insertion（插入）**：处理新增数据集 $E^+$ 的传播，与周期识别并行进行

### 关键技术创新

- **新 seminaïve 评估算子**：从 $\Delta$（增量）而非全部事实实例化查询，可对 $\mathds{I}$ 懒展开
- **增量周期识别 (Pds)**：仅在更新事实及其后果中搜索周期结构，而非整个物化（将 $O(n)$ 降为 $O(1)$ 级别在更新量小时）
- **周期化算子**：设计 Periodic Minus 和 Periodic Union 操作，通过 Ext（扩展到 LCM 长度）和 Aln（对齐端点）实现

### 正确性保证

定理证明：执行 DRed$_{\text{MTL}}(\Pi, E, \mathds{I}, E^-, E^+)$ 后，$\mathsf{unfold}(\mathds{I}) = \mathfrak{C}_{\Pi, E'}$，其中 $E' = (E \setminus E^-) \cup E^+$

## 实验关键数据

三个公开数据集上的对比（DRed$_{\text{MTL}}$ vs 重新物化）：

| 数据集 | \|E\| | 删除量 | DRed 删除耗时(s) | 重新物化耗时(s) | DRed 插入(s) | Remat 插入(s) |
|--------|-------|--------|-------------|------------|-----------|-------------|
| LUBM$_t$ | 630.5k | 100 | 0.7k | 48.6k | 0.4k | 48.5k |
| iTemporal | 46.4k | 100 | 8.1k | 52.7k | 8.7k | 52.8k |
| Meteorological | 62.01M | 100 | 10 | 0.7k | 11 | 0.7k |

- Meteorological 数据集上 DRed$_{\text{MTL}}$ 比重新物化**快约 70 倍**
- LUBM$_t$ 大删除量 (63.1k) 时仍有明显加速：3.4k vs 45.0k 秒

## 亮点

- 首个 DatalogMTL 增量维护算法，填补该方向空白
- 理论贡献扎实：完整正确性证明（含完整技术报告）
- 增量周期识别的思路优雅——仅在"差异"中寻找周期结构
- 基于 MeTeoR 实现，具有实用性

## 局限与展望

- 仅支持有界区间的 DatalogMTL，无界情况未涉及
- DRed 本身的过度删除问题在 MTL 设置下可能被放大
- 更新量接近全量时增量优势减弱
- 可探索 B/F 或 FBF 等更先进的 Datalog 增量算法的 MTL 扩展

## 相关工作对比

| 维度 | MeTeoR | vadalog | DRed$_{\text{MTL}}$ |
|------|--------|---------|-------------------|
| 推理方式 | 混合（底-上+顶-下） | 底-上 | 增量底-上 |
| 终止保证 | 有（有界程序） | 无（递归可不终止） | 有 |
| 增量更新 | ✗ | ✗ | ✓ |

## 启发

- 周期化表示+增量维护的思路可推广到其他时序推理框架
- 在流式时序数据场景下，增量推理能大幅减少计算成本

## 评分

⭐⭐⭐⭐ — 理论扎实、算法新颖、实验验证充分，是时序推理领域的重要进展

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] An Incremental Unified Framework for Small Defect Inspection](../../ECCV2024/others/an_incremental_unified_framework_for_small_defect_inspection.md)
- [\[ICML 2025\] Addressing Imbalanced Domain-Incremental Learning through Dual-Balance Collaborative Experts (DCE)](../../ICML2025/others/addressing_imbalanced_domain-incremental_learning_through_dual-balance_collabora.md)
- [\[AAAI 2026\] TDSNNs: Competitive Topographic Deep Spiking Neural Networks for Visual Cortex Modeling](tdsnns_competitive_topographic_deep_spiking_neural_networks_for_visual_cortex_mo.md)
- [\[AAAI 2026\] Bandit Learning in Housing Markets](bandit_learning_in_housing_markets.md)
- [\[AAAI 2026\] HybriDLA: Hybrid Generation for Document Layout Analysis](hybridla_hybrid_generation_for_document_layout_analysis.md)

</div>

<!-- RELATED:END -->
