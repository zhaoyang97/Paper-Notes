---
title: >-
  [论文解读] Variational Regularized Unbalanced Optimal Transport: Single Network, Least Action
description: >-
  [NeurIPS 2025][计算生物][正则化非平衡最优传输] 提出 Var-RUOT，通过将正则化非平衡最优传输（RUOT）问题的最优性必要条件融入参数化和损失设计，仅需学习单个标量场即可求解 RUOT，获得更低作用量的解并提升训练稳定性；同时分析了增长惩罚函数对生物先验的影响。 从有限快照数据恢复高维系统的连续动力学是…
tags:
  - "NeurIPS 2025"
  - "计算生物"
  - "正则化非平衡最优传输"
  - "变分方法"
  - "最小作用量"
  - "单标量场"
  - "单细胞轨迹推断"
---

# Variational Regularized Unbalanced Optimal Transport: Single Network, Least Action

**会议**: NeurIPS 2025  
**arXiv**: [2505.11823](https://arxiv.org/abs/2505.11823)  
**代码**: [GitHub](https://github.com/ZerooVector/VarRUOT)  
**领域**: 计算生物  
**关键词**: 正则化非平衡最优传输, 变分方法, 最小作用量, 单标量场, 单细胞轨迹推断

## 一句话总结

提出 Var-RUOT，通过将正则化非平衡最优传输（RUOT）问题的最优性必要条件融入参数化和损失设计，仅需学习单个标量场即可求解 RUOT，获得更低作用量的解并提升训练稳定性；同时分析了增长惩罚函数对生物先验的影响。

## 研究背景与动机

从有限快照数据恢复高维系统的连续动力学是统计物理和计算生物学中的核心挑战。在单细胞 RNA 测序中，仅有少量时间点的快照测量，需要重建连续的细胞轨迹。

多种框架被提出来解决该问题：动态最优传输（Benamou-Brenier）、Schrödinger Bridge、非平衡动态 OT（Wasserstein-Fisher-Rao）等。RUOT 框架将随机性和粒子生灭过程统一起来。然而现有深度学习求解器面临两大挑战：

**未显式约束最优性条件**：现有方法通常用独立的神经网络参数化速度场 u 和增长率 g，不利用它们之间的最优性关系，导致解偏离最小作用量原则且收敛不可靠。

**增长惩罚函数选择缺乏指导**：标准 WFR 度量使用 ψ(g) = g²/2，但不同 ψ 会隐含不同的生物学先验，这一点未被充分研究。

## 方法详解

### 整体框架

Var-RUOT 的核心思想：通过变分方法推导 RUOT 的最优性必要条件，发现速度场 u 和增长率 g 都可由单个标量场 λ(x,t) 完全确定。因此只需一个神经网络参数化 λ，大幅简化了问题。

### 关键设计

1. **最优性必要条件推导 (Theorem 4.1)**: 对各向同性时不变扩散的 RUOT 问题，通过变分法推导出三个必要条件：

    - u = ∇_x λ（速度场是标量场的梯度）
    - α·ψ'(g) = λ（增长率由标量场隐式确定）
    - HJB 方程：∂λ/∂t + ½‖∇λ‖² + ½σ²∇²λ + λg - αψ(g) = 0
   
   关键洞察：一旦 λ 确定，u 和 g 自动确定，Fokker-Planck 方程的演化也完全确定。这将原本需要分别学习 u 和 g 的问题简化为只学一个标量场。

2. **增长惩罚函数与生物先验 (Theorem 4.2)**: 证明 ψ''(g) 的符号决定了 g 沿速度场方向的单调性：

    - ψ''(g) > 0（如标准 WFR 的 g²/2）：g 沿 u 方向递增——即下游细胞增殖更快
    - ψ''(g) < 0（如本文提出的 g^{2/15}）：g 沿 u 方向递减——即上游干细胞增殖最快
   
   后者更符合生物学先验：干细胞位于轨迹上游，具有最高增殖和分化能力，沿分化方向 g 应递减。因此本文提出使用 ψ₂(g) = g^{2/15} 作为更合理的替代。

3. **加权粒子方法求解 (Theorem 5.1)**: 用 N 个加权粒子近似 Fokker-Planck 方程的解。每个粒子位置服从SDE，权重服从 ODE：

    - dX_i = u(X_i, t)dt + σdW_t
    - dw_i = g(X_i, t)w_i dt
   
   经验测度 μ^N 在 N→∞ 时收敛到真实密度 ρ。

4. **三部分损失函数**:

    - **重建损失 L_Recon**: 包含质量匹配损失（总质量 M̂(T_k) ≈ M(T_k)）和 Wasserstein-2 分布距离
    - **HJB 损失 L_HJB**: 沿粒子轨迹积分 HJB 方程的违反程度，强制 λ 满足最优性条件
    - **作用量损失 L_Action**: 直接最小化传输作用量（因为必要条件不充分，仍需显式优化）

### 训练策略

联合最小化 L = L_Recon + γ_HJB · L_HJB + γ_Action · L_Action。使用 Euler-Maruyama 方法离散化 SDE，自动微分计算 ∇λ 和 ∇²λ。训练中权重归一化用于 HJB 损失的加权。

## 实验关键数据

### 主实验

| 数据集 | 指标 | Var-RUOT | DeepRUOT | 其他基线 |
|--------|------|----------|----------|----------|
| 三基因模拟 (t=1) | W₁ | **0.0452** | 0.0569 | TIGON: 0.0519 |
| 三基因模拟 (t=2) | W₁ | **0.0385** | 0.0811 | OTCFM: 0.2078 |
| 三基因模拟 (t=4) | W₁ | **0.0572** | 0.1538 | UOTCFM: 0.4129 |
| 三基因模拟 | 路径作用量 | **1.1105** | 1.4058 | TIGON: 1.2442 |
| EMT 数据集 (10维) | 轨迹形态 | 近直线 | 弯曲 | - |

### 消融实验

| 配置 | 指标 | 说明 |
|------|------|------|
| 标准 WFR ψ₁(g)=g²/2 | g 沿 u 递增 | 不符合干细胞在上游的生物先验 |
| 修改 ψ₂(g)=g^{2/15} | g 沿 u 递减 | 符合生物先验 |
| 无 HJB 损失 | 作用量更高 | HJB 约束对找最小作用量路径至关重要 |
| 无作用量损失 | 可能收敛到鞍点 | 仅必要条件不够，需显式优化 |
| 收敛速度对比 | 训练 epoch | Var-RUOT 收敛更快、更稳定 |

### 关键发现

- Var-RUOT 在三基因模拟数据上比 DeepRUOT 作用量低 21%（1.11 vs 1.41），同时分布匹配精度更高。
- 在 EMT 数据集上，Var-RUOT 学到的轨迹接近直线（对应最小作用量），而 DeepRUOT 学到弯曲轨迹。
- 单标量场参数化显著简化了优化景观，使训练更稳定、收敛更快。
- ψ 的选择确实影响学到的生物动力学，这在之前的工作被完全忽视。

## 亮点与洞察

- **优雅的理论简化**：通过变分法将需要学 u、g 两个场的问题简化为只学 λ 一个标量场，大幅降低优化难度。
- **将物理约束嵌入网络设计**：不是作为外部惩罚而是直接改变参数化空间——u = ∇λ 天然保证速度场无旋。
- **增长惩罚函数的生物学意义**：首次揭示 ψ''(g) 符号与细胞发育方向的联系，为计算生物学提供了有指导性的建模建议。
- **推广了 Action Matching**：将 Neklyudov et al. (2023, 2024) 的方法推广到同时处理非平衡和随机动力学。

## 局限与展望

- 仅考虑了各向同性时不变扩散（σ²I），未处理各向异性或时变扩散矩阵。
- ψ₂(g) = g^{2/15} 的选择虽满足理论要求但比较 ad-hoc，更系统的选择方法有待研究。
- 自动微分计算 ∇²λ 在高维问题中可能成为计算瓶颈（Hessian trace 估计）。
- 实验仅在中低维度（3D、10D）验证，未验证在全维度单细胞数据（数千基因）上的可扩展性。
- 加权粒子方法的粒子退化问题（部分粒子权重趋近零）未被讨论。

## 相关工作与启发

- Action Matching (Neklyudov et al., 2023) 和 WLF (Neklyudov et al., 2024) 是最直接的前驱工作，本文将其推广到 RUOT 的完整框架。
- DeepRUOT (Zhang et al., 2025a) 用独立网络参数化 u 和 g，是主要对比基线。
- TIGON (Sha et al., 2024) 在计算生物学应用中表现强，但同样未利用最优性条件。
- 对最优控制和变分法在机器学习中的应用提供了新的范例。

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 将变分最优性条件融入神经网络参数化的思路非常优雅
- **实验充分度**: ⭐⭐⭐ 验证了核心主张但实验规模有限（低维、少数据集）
- **写作质量**: ⭐⭐⭐⭐ 数学推导严格，结构清晰
- **实用价值**: ⭐⭐⭐⭐ 对轨迹推断和计算生物学有直接价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] CellStream: Dynamical Optimal Transport Informed Embeddings for Reconstructing Cellular Trajectories from Snapshots Data](../../AAAI2026/computational_biology/cellstream_dynamical_optimal_transport_informed_embeddings_for_reconstructing_ce.md)
- [\[NeurIPS 2025\] Flow Density Control: Generative Optimization Beyond Entropy-Regularized Fine-Tuning](flow_density_control_generative_optimization_beyond_entropy-regularized_fine-tun.md)
- [\[NeurIPS 2025\] PRESCRIBE: Predicting Single-Cell Responses with Bayesian Estimation](prescribe_predicting_single-cell_responses_with_bayesian_estimation.md)
- [\[NeurIPS 2025\] scPilot: Large Language Model Reasoning Toward Automated Single-Cell Analysis and Discovery](scpilot_large_language_model_reasoning_toward_automated_single-cell_analysis_and.md)
- [\[ICML 2026\] Towards Universal Gene Regulatory Network Inference: Unlocking Generalizable Regulatory Knowledge in Single-cell Foundation Models](../../ICML2026/computational_biology/towards_universal_gene_regulatory_network_inference_unlocking_generalizable_regu.md)

</div>

<!-- RELATED:END -->
