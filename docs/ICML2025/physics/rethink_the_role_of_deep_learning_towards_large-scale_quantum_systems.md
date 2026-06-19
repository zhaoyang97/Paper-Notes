---
title: >-
  [论文解读] Rethink the Role of Deep Learning towards Large-scale Quantum Systems
description: >-
  [ICML2025][物理/科学计算][量子系统学习] 在统一量子资源约束下系统性地对比 ML 与 DL 在量子系统学习 (QSL) 任务中的表现，发现传统 ML（Lasso/Ridge/核方法）往往匹配甚至超越 DL，挑战了"大规模量子系统必须用深度学习"的直觉。 - 核心问题： 量子系统基态性质（关联函数、纠缠熵）估计和…
tags:
  - "ICML2025"
  - "物理/科学计算"
  - "量子系统学习"
  - "基态性质估计"
  - "量子相分类"
  - "深度学习 vs 机器学习"
  - "classical shadow"
  - "大规模量子系统"
---

# Rethink the Role of Deep Learning towards Large-scale Quantum Systems

**会议**: ICML2025  
**arXiv**: [2505.13852](https://arxiv.org/abs/2505.13852)  
**代码**: [GitHub](https://github.com/) (部分数据集生成代码已开源)  
**领域**: 物理 / 量子系统学习  
**关键词**: 量子系统学习, 基态性质估计, 量子相分类, 深度学习 vs 机器学习, classical shadow, 大规模量子系统

## 一句话总结

在统一量子资源约束下系统性地对比 ML 与 DL 在量子系统学习 (QSL) 任务中的表现，发现传统 ML（Lasso/Ridge/核方法）往往匹配甚至超越 DL，挑战了"大规模量子系统必须用深度学习"的直觉。

## 研究背景与动机

- **核心问题**: 量子系统基态性质（关联函数、纠缠熵）估计和量子相分类是量子物理的基础问题，经典精确模拟受维度诅咒限制
- **AI for Quantum**: 近年涌现大量 ML/DL 方法用于量子系统学习 (QSL)，包括线性回归、核方法、MLP、CNN、自监督大模型 (LLM4QPE) 等
- **公平性缺失**: 已有研究在对比 DL 与 ML 时，DL 常使用远多于 ML 的量子测量资源构造训练数据（如无限量测量生成标签），导致比较不公平
- **关键疑问**: 在量子资源稀缺的现实约束下，我们真的需要深度学习来完成 QSL 任务吗？

## 方法详解

### 统一资源框架

论文的核心设计是 **统一量子资源预算**：所有模型的训练数据集 $\mathcal{D}$ 满足总查询量约束 $n \times M$ 相同，其中 $n$ 为训练样本数，$M$ 为每个样本的测量快照 (snapshot) 数。

对于自监督模型 (SSL)，需满足：

$$n_{\text{pre}} \times M_{\text{pre}} + n_{\text{sft}} \times M_{\text{sft}} = n \times M$$

### 探索的哈密顿量族

1. **Heisenberg 模型 (HB)**: $\mathsf{H}_{\text{HB}}(\mathbf{x}) = \sum_{i<j} J_{ij}(X_iX_j + Y_iY_j + Z_iZ_j)$，其中 $J_{ij} = 369 / |i-j|^a$，$a \in (1,2)$
2. **横场 Ising 模型 (TFIM)**: $\mathsf{H}_{\text{TFIM}}(\mathbf{x}) = -\sum_{i=1}^{N-1} J_i Z_i Z_{i+1} - \sum_{i=1}^{N} h_i X_i$
3. **Rydberg 原子模型**: $\mathsf{H}_{\text{Ryd}}(\mathbf{x}) = \sum_{i<j} \frac{\Omega R_b^6}{a^6|i-j|^6} N_i N_j + \sum_{i=1}^{N} \frac{\Omega}{2}X_i - \Delta_i N_i$

### 基态性质目标

- **关联函数** $C_{ij}$: $C_{ij} = \frac{\text{tr}(X_iX_j\rho) + \text{tr}(Y_iY_j\rho) + \text{tr}(Z_iZ_j\rho)}{3}$
- **2阶 Rényi 纠缠熵**: $\mathcal{S}_2(\rho_A) = -\log[\text{tr}(\rho_A^2)]$
- **量子相分类**: 对 Rydberg 模型分 $Z_2$-有序相、$Z_3$-有序相和无序相三类

### Benchmark 模型

| 类别 | 模型 | 特点 |
|------|------|------|
| ML — 线性回归 | Lasso, Ridge | 随机傅里叶特征 + $L_1/L_2$ 正则化 |
| ML — 核方法 | DK, RBFK, NTK | Dirichlet 核 / 径向基核 / 神经正切核 |
| ML — 树模型 | RF, GBT, LGBM, XGBoost | 用于分类任务 |
| DL — 监督学习 | MLP, CNN (及 MLP-A, CNN-A) | 带/不带测量辅助信息 |
| DL — 自监督 | SG, LLM4QPE-F, LLM4QPE-T | Shadow Generator; LLM4QPE (± 预训练) |

### 随机化测试

为检验测量结果作为输入特征的作用，将真实测量值 $\mathbf{v}$ 替换为从 $[0,5]$ 均匀采样的随机整数 $\mathbf{v}'$，观察模型性能变化。

## 实验关键数据

### 关联函数预测 (HB, RMSE $\epsilon(\bar{C})$, $M=64$)

| 模型 | $N=48, n=100$ | $N=127, n=100$ |
|------|---------------|----------------|
| Classical Shadow | 0.2114 | 0.2145 |
| MLP-4层 | 0.0352 | 0.0861 |
| CNN-4层 | 0.0346 | 0.0522 |
| LLM4QPE-T | 0.0320 | 0.0263 |
| **Lasso** | **0.0249** | **0.0208** |
| **Ridge** | **0.0248** | **0.0216** |

**关键结论**: Lasso/Ridge 在几乎所有设置下以显著优势超越 DL 模型。

### Scaling 行为 (127-qubit HB, $n=100, M=512$)

- Lasso: $\epsilon(\bar{C}) = 0.011$
- Ridge: $\epsilon(\bar{C}) = 0.012$
- LLM4QPE-F (~18.1M 参数): $\epsilon(\bar{C}) = 0.017$

### 模型规模实验

- 带强正则化 ($\lambda$ 大) 的 MLP 仅需 LLM4QPE-F **1/36 的参数**即可匹配其性能
- 无正则化时模型越大反而过拟合更严重

### 量子相分类 (31-qubit Rydberg, 准确率 %)

| 模型 | $M=64, n=100$ | $M=256, n=100$ |
|------|---------------|----------------|
| MLP | 92.79 | 94.50 |
| CNN | 92.50 | 92.79 |
| LLM4QPE-T | — | — |

### 随机化测试结果

- **GSPE 任务**: 用随机值替换真实测量结果后，LLM4QPE-T 性能基本不变 → 测量结果对属性估计**冗余**
- **QPC 任务**: 替换后性能显著下降 → 测量结果对相分类**重要**

## 亮点与洞察

1. **公平对比框架**: 首次在统一量子资源预算 ($n \times M$ 相同) 下系统对比 ML 与 DL，填补了领域内公平 benchmark 的空白
2. **反直觉发现**: 简单的 Lasso/Ridge 在 GSPE 任务上持续优于复杂 DL（MLP/CNN/LLM4QPE），挑战"深度学习万能"假设
3. **测量冗余性**: 随机化测试揭示测量结果作为输入在 GSPE 中冗余、在 QPC 中关键——这一对偶性为未来模型设计提供明确指导
4. **规模不是万能**: 更大的 DL 模型不一定更好，正则化策略远比盲目增加参数重要
5. **大尺度验证**: 实验扩展到 127 量子比特，覆盖三大主流哈密顿量族，结论稳健

## 局限与展望

1. **任务覆盖有限**: 仅考虑关联函数、纠缠熵和相分类三类任务，未涉及量子态层析、保真度估计等更广泛任务
2. **哈密顿量范围**: 三族哈密顿量虽经典但未覆盖化学分子、拓扑序等更复杂系统
3. **真实量子硬件**: 所有实验基于模拟数据；真实量子设备的噪声可能改变 ML/DL 的优劣关系
4. **DL 架构探索不足**: 未测试 Transformer、Graph Neural Network 等更现代架构在 QSL 中的潜力
5. **理论分析缺乏**: 未从理论角度解释为何线性模型在这些任务中足够——仅有经验观察

## 相关工作与启发

- Huang et al., 2022: 证明利用量子数据的经典算法可实现高效 GSPE，为 ML 方法奠基
- Lewis et al., 2024; Wanner et al., 2024: 线性回归+几何特征图的可证明高效 ML 方法
- Wang et al., 2022 (Shadow Generator): 自回归生成 classical shadow
- Tang et al., 2024 (LLM4QPE): 将 LLM 预训练范式引入量子属性估计

## 评分

- 新颖性: ⭐⭐⭐⭐ — 视角新颖，首次系统公平对比 ML vs DL 在 QSL 中的角色
- 实验充分度: ⭐⭐⭐⭐ — 多模型、多任务、多尺度（最大 127 qubit），含随机化测试
- 写作质量: ⭐⭐⭐⭐ — 结构清晰、问题驱动，但部分记号较密集
- 价值: ⭐⭐⭐⭐ — 对量子 ML 社区有重要警示：不要盲目追求 DL 复杂度

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Erwin: A Tree-based Hierarchical Transformer for Large-scale Physical Systems](erwin_a_tree-based_hierarchical_transformer_for_large-scale_physical_systems.md)
- [\[ICML 2026\] Rethink the Role of Neural Decoders in Quantum Error Correction](../../ICML2026/physics/rethink_the_role_of_neural_decoders_in_quantum_error_correction.md)
- [\[ICML 2025\] Liger: Linearizing Large Language Models to Gated Recurrent Structures](liger_linearizing_large_language_models_to_gated_recurrent_structures.md)
- [\[NeurIPS 2025\] TITAN: A Trajectory-Informed Technique for Adaptive Parameter Freezing in Large-Scale VQE](../../NeurIPS2025/physics/titan_a_trajectory-informed_technique_for_adaptive_parameter_freezing_in_large-s.md)
- [\[NeurIPS 2025\] Toward Complete Merger Identification at Cosmic Noon with Deep Learning](../../NeurIPS2025/physics/toward_complete_merger_identification_at_cosmic_noon_with_deep_learning.md)

</div>

<!-- RELATED:END -->
