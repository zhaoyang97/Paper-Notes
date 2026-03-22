# Empirical Stability Analysis of Kolmogorov-Arnold Networks in Hard-Constrained Recurrent Physics-Informed Discovery

**会议**: ICLR2026  
**arXiv**: [2602.09988](https://arxiv.org/abs/2602.09988)  
**代码**: 待确认  
**领域**: others  
**关键词**: KAN, physics-informed, oscillator, HRPINN, neural ODE

## 一句话总结
系统实证分析将 KAN（Kolmogorov-Arnold Networks）集成到硬约束递归物理信息架构（HRPINN）中的表现——发现小型 KAN 在单变量多项式残差（Duffing）上具有竞争力，但在乘法项（Van der Pol）上严重失败且超参数极度脆弱，标准 MLP 稳定性远优。

## 研究背景与动机

1. **领域现状**：HRPINN 将已知物理嵌入递归积分器，神经网络只学残差动力学。KAN 基于 Kolmogorov-Arnold 表示定理，用可学 B-样条替代固定激活函数，在科学 ML 中展现潜力。
2. **核心假设**：KAN 的加法归纳偏差应能高效分离和发现物理残差项。
3. **核心发现**：假设仅在单变量项（Duffing 的 $-0.3x^3$）上成立；对乘法耦合项（Van der Pol 的 $(1-x^2)v$）KAN 系统性失败。
4. **核心矛盾**：KAN 的 $\Phi(\mathbf{x}) = \sum_q \phi_q(\sum_p \psi_{q,p}(x_p))$ 天然适合加法可分函数，但乘法需要更深层组合（如 $xy = \frac{1}{4}((x+y)^2-(x-y)^2)$），在递归设置中导致不稳定。

## 方法详解

### 实验设计
- 3 项互补研究，每项 100 个随机种子
- 配置消融（grid size 和 sparsity）、参数规模消融（teacher forcing）、参数规模消融（BPTT）
- 统一候选拟合评估（非 KAN 特有的符号剪枝）

### 关键发现

1. **配置消融**: 多数 KAN 配置在 Van der Pol 上产生负 $R^2$（发散），MLP（337参数）稳定达到 $R^2=0.768$
2. **参数规模**: 极小 KAN（~120参数）在 Duffing 上匹敌同规模 MLP；但更宽/更深 KAN 坍缩
3. **BPTT 训练**: 部分缓解 Van der Pol 问题（最佳 $R^2 \approx 0.74$）但 MLP 仍占优

## 实验关键数据

### 配置消融（100 seeds）

| 配置 | Duffing $R^2$ | Van der Pol $R^2$ |
|------|------|------|
| KAN Config A | 0.835±0.030 | 0.667±0.037 |
| KAN Config C (Sparse-Low) | 0.595±0.033 | **-5.229±5.091** |
| MLP (337 params) | **0.957±0.009** | **0.768±0.015** |

### 参数规模消融

| 架构 | 参数 | Duffing (TF) | VdP (TF) | Duffing (BPTT) | VdP (BPTT) |
|------|------|------|------|------|------|
| KAN Very Small | 120 | 0.836 | 0.464 | 0.914 | 0.743 |
| MLP | 相当 | 更好 | 更好 | 更好 | 更好 |

### 关键发现
- KAN 在 Duffing 上可发现立方结构（$-0.234x^3$，真值 $-0.3x^3$）但系数低估
- KAN 在 Van der Pol 上常坍缩为粗略线性形式而非预期的抛物线调制
- BPTT 的长时域监督帮助稳定学习，但 MLP 仍更鲁棒
- KAN 超参数敏感度远高于 MLP——实践中不实用

## 亮点与洞察
- 诚实的"负面结果"——清楚揭示了当前 KAN 的实际限制
- 加法归纳偏差 vs 乘法耦合的分析直击 KAN 设计的核心假设
- 大规模种子统计（100 seeds）使结论可靠

## 局限性 / 可改进方向
- 仅测试了 vanilla KAN，改进版本（如 SKANODE）可能更好
- 仅两个振荡器系统，更复杂系统待测
- 未探讨混合 KAN-MLP 架构

## 相关工作与启发
- **vs KAN-ODEs (Koenig et al.)**: 在无约束设置中表现好，但本文揭示递归物理约束下的脆弱性
- **vs SKANODE (Liu et al.)**: 结构化 KAN 可能缓解问题

## 评分
- 新颖性: ⭐⭐⭐ 系统的负面结果有价值但非新方法
- 实验充分度: ⭐⭐⭐⭐⭐ 3 项研究 × 100 seeds，极其充分
- 写作质量: ⭐⭐⭐⭐ 分析诚实透彻
- 价值: ⭐⭐⭐⭐ 为 KAN 社区提供重要的实践警示
