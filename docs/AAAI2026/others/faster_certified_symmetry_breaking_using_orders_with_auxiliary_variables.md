---
title: >-
  [论文解读] Faster Certified Symmetry Breaking Using Orders With Auxiliary Variables
description: >-
  [AAAI 2026][symmetry breaking] 通过引入辅助变量编码字典序来替代大整数编码，对 VeriPB 证明系统进行本质重设计，使 SAT 对称性破坏的证明生成和验证在理论和实践上均获得数量级加速。 - 对称性破坏是组合求解中的关键技术，可避免重复探索等价搜索空间 - SAT 社区强调可证明正确性——竞赛…
tags:
  - "AAAI 2026"
  - "symmetry breaking"
  - "proof logging"
  - "SAT solving"
  - "pseudo-Boolean"
  - "VeriPB"
  - "certified solving"
---

# Faster Certified Symmetry Breaking Using Orders With Auxiliary Variables

**会议**: AAAI 2026  
**arXiv**: [2511.16637](https://arxiv.org/abs/2511.16637)  
**代码**: [satsuma](https://doi.org/10.5281/zenodo.17607863), [VeriPB](https://doi.org/10.5281/zenodo.17608873), [CakePB](https://doi.org/10.5281/zenodo.17609070)  
**领域**: 其他  
**关键词**: symmetry breaking, proof logging, SAT solving, pseudo-Boolean, VeriPB, certified solving  

## 一句话总结

通过引入辅助变量编码字典序来替代大整数编码，对 VeriPB 证明系统进行本质重设计，使 SAT 对称性破坏的证明生成和验证在理论和实践上均获得数量级加速。

## 背景与动机

- 对称性破坏是组合求解中的关键技术，可避免重复探索等价搜索空间
- SAT 社区强调可证明正确性——竞赛中求解器需生成机器可验证的证明
- 对称性破坏的证明长期是公开难题：DRAT 格式无法处理多个对称性
- Bogaerts et al. (2023) 提出基于 pseudo-Boolean 不等式的完全通用方法，但使用大整数编码字典序 $\sum_{i=1}^n 2^{n-i}x_i \leq \sum_{i=1}^n 2^{n-i}y_i$，系数需 $n^2$ bit 表示，导致证明生成线性开销、验证需昂贵的任意精度算术

## 核心问题

如何在保持完全通用性的同时，消除对称性破坏证明中大整数编码的性能瓶颈？

## 方法详解

### 核心思路

用辅助变量 $s_1, \ldots, s_n$ 编码字典序（类似子句编码 $s_i$ 表示前 $i$ 个变量相等），替代单个大系数 pseudo-Boolean 不等式。

### 技术挑战

辅助变量不出现在前提中，破坏了原始证明系统的"蕴含性"不变量。需要对证明系统做本质重设计。

### 关键创新：Specification + Order 分离

将序编码拆分为两部分：

1. **Specification $\mathcal{S}_\preceq(\vec{x}, \vec{y}, \vec{a})$**：定义辅助变量 $\vec{a}$ 如何从主变量计算而来（类似电路定义），可放入前提中
2. **Order $\mathcal{O}_\preceq(\vec{x}, \vec{y}, \vec{a})$**：基于辅助变量的序约束

定义：$\alpha \preceq \beta$ 当且仅当存在赋值 $\varrho$ 使得 $\mathcal{S}_\preceq$ 和 $\mathcal{O}_\preceq$ 同时满足。

### 修改后的证明规则

- **Dominance rule**：证明义务变为 $\mathcal{C} \cup \mathcal{D} \cup \{\neg C\} \cup \mathcal{S}_\preceq(\vec{z}|_\omega, \vec{z}, \vec{a}) \vdash \mathcal{C}|_\omega \cup \mathcal{O}_\preceq(\vec{z}|_\omega, \vec{z}, \vec{a})$
- **Redundance rule**：类似地加入 $\mathcal{S}_\preceq$ 作为额外前提
- 自反性和传递性证明义务相应调整

### 渐近复杂度改进

| 操作 | 旧方法 | 新方法 |
|------|--------|--------|
| 定义字典序 ($n$ 变量) | $O(n^2)$ | $O(n)$ |
| 打破对称性 (support $k$) logging | $O(nk)$ | $O(k)$ |
| 打破对称性 checking | $O(n^2 + nk^2)$ | $O(n)$ |

## 实验关键数据

### 构造性 benchmark（5 个系列）

- 新方法的证明生成时间与不做证明的求解时间比例接近常数
- 旧方法在 PHP/RPHP 系列上证明生成时间渐近劣化明显
- 证明验证：新方法在所有 5 个系列上均展现更好的 scaling

### SAT 竞赛实例（2020-2024，982 个有对称性的实例）

| 指标 | 新方法 | 旧方法 |
|------|--------|--------|
| 成功生成证明 | 982/982 | 930/982 |
| VeriPB 验证通过 | 893 | 806 |
| CakePB 验证通过 | 799 | 732 |

新方法从不比旧方法差超过常数因子，常常快数量级。

## 亮点

- 解决了认证对称性破坏的长期性能瓶颈
- 理论改进（至少线性因子加速）在实验中得到充分验证
- 完整的端到端形式化验证链：satsuma → VeriPB → CakePB（形式化正确性保证）
- 实现在 SOTA 工具中，具有直接实用价值

## 局限与展望

- 证明验证仍可能比对称性破坏本身渐近更慢（每个对称性需推理所有序变量）
- 启用证明生成仍有常数因子开销（主要是磁盘 I/O）
- 仅处理静态对称性破坏，未涉及搜索过程中的动态对称性利用
- CakePB 的形式验证算术库效率仍有提升空间

## 相关工作对比

| 维度 | Bogaerts et al. (2023) | 本文 |
|------|------|------|
| 序编码 | 大整数 PB 不等式 | 辅助变量子句 |
| 系数位数 | $O(n^2)$ | $O(n)$ |
| 通用性 | 完全通用 | 完全通用 |
| 实际可行性 | 大对称性不可行 | 所有 982 实例均可行 |

## 启发

- "用辅助变量替代大系数"是 PB 推理中的通用优化思路
- Specification/Order 分离的证明系统设计模式可能在其他认证求解领域也有用

## 评分

⭐⭐⭐⭐ — 理论与实践结合紧密，解决长期公开难题，工具链完整且形式化验证

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Certified but Fooled! Breaking Certified Defences with Ghost Certificates](certified_but_fooled_breaking_certified_defences_with_ghost_certificates.md)
- [\[AAAI 2026\] Certified Branch-and-Bound MaxSAT Solving (Extended Version)](certified_branch-and-bound_maxsat_solving_extended_version.md)
- [\[AAAI 2026\] SynWeather: Weather Observation Data Synthesis across Multiple Regions and Variables via a General Diffusion Transformer](synweather_weather_observation_data_synthesis_across_multiple_regions_and_variab.md)
- [\[ICLR 2026\] ANO: Faster is Better in Noisy Landscapes](../../ICLR2026/others/ano_faster_is_better_in_noisy_landscape.md)
- [\[ICLR 2026\] Breaking Gradient Temporal Collinearity for Robust Spiking Neural Networks](../../ICLR2026/others/breaking_gradient_temporal_collinearity_for_robust_spiking_neural_networks.md)

</div>

<!-- RELATED:END -->
