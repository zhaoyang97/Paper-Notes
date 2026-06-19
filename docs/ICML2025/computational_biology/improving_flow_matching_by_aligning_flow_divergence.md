---
title: >-
  [论文解读] Improving Flow Matching by Aligning Flow Divergence
description: >-
  [ICML 2025][计算生物][Flow Matching] 从 PDE 视角分析了 Flow Matching 中学习概率路径与真实概率路径之间的误差，证明该误差受到向量场散度(divergence)差距的控制，并提出联合匹配流和散度的 FDM 训练目标，在密度估计、DNA 序列生成和视频预测等任务上显著提升了 FM 的表现。
tags:
  - "ICML 2025"
  - "计算生物"
  - "Flow Matching"
  - "散度匹配"
  - "概率路径误差"
  - "Total Variation"
  - "条件散度损失"
---

# Improving Flow Matching by Aligning Flow Divergence

**会议**: ICML 2025  
**arXiv**: [2602.00869](https://arxiv.org/abs/2602.00869)  
**代码**: [https://github.com/Utah-Math-Data-Science](https://github.com/Utah-Math-Data-Science)  
**领域**: 计算生物  
**关键词**: Flow Matching, 散度匹配, 概率路径误差, Total Variation, 条件散度损失

## 一句话总结
从 PDE 视角分析了 Flow Matching 中学习概率路径与真实概率路径之间的误差，证明该误差受到向量场散度(divergence)差距的控制，并提出联合匹配流和散度的 FDM 训练目标，在密度估计、DNA 序列生成和视频预测等任务上显著提升了 FM 的表现。

## 研究背景与动机
**领域现状**：条件流匹配（CFM）是训练基于流的生成模型的高效方法，通过回归条件向量场来学习从噪声到数据的映射，无需模拟。

**现有痛点**：CFM 只确保学习的向量场 $\boldsymbol{v}_t$ 接近真实向量场 $\boldsymbol{u}_t$，但两者的散度（divergence）差距 $|\nabla \cdot \boldsymbol{v}_t - \nabla \cdot \boldsymbol{u}_t|$ 可能很大，导致学习到的概率路径与真实概率路径存在显著偏差。

**核心矛盾**：CFM loss 是 FM loss 加常数，最小化它能学好向量场本身，但无法保证概率路径（密度函数）的准确性——向量场的散度决定了密度的变化。

**本文目标** 如何在 FM 训练中同时控制向量场及其散度的精度，以获得更准确的概率路径？

**切入角度**：从连续性方程出发，推导精确与学习概率路径之间误差满足的 PDE ，用 Duhamel 原理求解得到误差的 TV 距离上界。

**核心 idea**：FM 的概率路径误差由向量场差异和散度差异共同决定，提出 FDM = CFM loss + 条件散度 loss 来同时优化两者。

## 方法详解

### 整体框架
FDM 在标准的 CFM 损失基础上增加一个条件散度匹配损失 $\mathcal{L}_{\text{CDM}}$，组成加权总损失：
$$\mathcal{L}_{\text{FDM}} = \lambda_1 \mathcal{L}_{\text{CFM}} + \lambda_2 \mathcal{L}_{\text{CDM}}$$

### 关键设计

1. **概率路径误差 PDE (Proposition 3.1)**:

    - 功能：刻画真实路径 $p_t$ 和学习路径 $\hat{p}_t$ 之间的误差 $\epsilon_t = p_t - \hat{p}_t$
    - 核心思路：误差满足 $\partial_t \epsilon_t + \nabla \cdot (\epsilon_t \boldsymbol{v}_t) = L_t$，其中强迫项 $L_t = -p_t[\nabla \cdot (\boldsymbol{u}_t - \boldsymbol{v}_t) + (\boldsymbol{u}_t - \boldsymbol{v}_t) \cdot \nabla \log p_t]$
    - 设计动机：强迫项同时包含向量场差异和散度差异，说明仅匹配向量场不够

2. **TV 距离上界 (Theorem 3.3)**:

    - 功能：将概率路径误差量化为可优化的目标
    - 核心思路：$\text{TV}(p_t, \hat{p}_t) \leq \frac{1}{2}\mathcal{L}_{\text{DM}}$，其中 $\mathcal{L}_{\text{DM}} = \mathbb{E}_{t, p_t}[|\nabla \cdot (\boldsymbol{u}_t - \boldsymbol{v}_t) + (\boldsymbol{u}_t - \boldsymbol{v}_t) \cdot \nabla \log p_t|]$
    - 设计动机：建立了可优化损失与分布精度之间的理论桥梁

3. **条件散度匹配 (Theorem 4.1 → FDM)**:

    - 功能：因 $\mathcal{L}_{\text{DM}}$ 不可直接计算（依赖 marginal 向量场），推导其条件版本 $\mathcal{L}_{\text{CDM}}$ 作为上界
    - 核心思路：利用与 CFM 类似的条件化技巧，将 unconditional 散度差替换为 conditional 散度差，得到可高效计算的 $\mathcal{L}_{\text{CDM}}$，并用 Hutchinson 迹估计器提高效率
    - 设计动机：单独最小化 $\mathcal{L}_{\text{CDM}}$ 因正负项抵消无法保证好结果，需与 $\mathcal{L}_{\text{CFM}}$ 联合优化

### 损失函数 / 训练策略
- $\mathcal{L}_{\text{FDM}} = \lambda_1 \mathcal{L}_{\text{CFM}} + \lambda_2 \mathcal{L}_{\text{CDM}}$
- 高效版本 $\mathcal{L}_{\text{CDM-2}}^{\text{eff}}$ 使用 stop-gradient + Hutchinson 迹估计，仅需额外一次反向传播
- 超参数 $\lambda_1, \lambda_2$ 通过搜索确定

## 实验关键数据

### 主实验

| 任务 | 模型 | FM 指标 | FDM 指标 | 提升 |
|------|------|---------|----------|------|
| Checkerboard 密度估计 (OT) | Likelihood ↑ | 2.38×10⁻² | **2.53×10⁻²** | +6.3% |
| CIFAR-10 (OT) | NLL ↓ | 2.99 | **2.85** | -4.7% |
| CIFAR-10 (OT) | FID ↓ | 6.35 | **5.62** | -11.5% |
| KTH 视频预测 | FVD ↓ | 180 | **155.5** | -13.6% |
| BAIR 视频预测 | FVD ↓ | 146 | **123** | -15.8% |

### 消融实验

| 数据集 | 指标 | FM (OT) | FDM (OT) | FM (VP) | FDM (VP) |
|--------|------|---------|----------|---------|----------|
| Lorenz 轨迹 p(x₁) | TV ↓ | 0.0348 | **0.0306** | - | - |
| FitzHugh p(x₁) | TV ↓ | 0.0314 | **0.0266** | - | - |
| DNA 序列 | MSE ↓ | 2.82E-2 | **2.78E-2** | - | - |
| DNA Dirichlet | MSE ↓ | 2.68E-2 | **2.59E-2** | - | - |

### 关键发现
- FDM 在所有路径类型（OT、VP、VE、Dirichlet）上均优于 FM
- 散度差距的影响在精确似然估计任务中最为显著（如 NLL 提升明显）
- 额外计算开销仅约 50%（一次额外反向传播），性价比高

## 亮点与洞察
- **理论驱动的方法设计**：从 PDE 误差分析出发推导损失函数，不是启发式设计
- **优雅的条件化技巧**：将不可计算的 marginal 散度差通过条件化+Jensen 不等式转化为可训练的 loss
- **广泛适用性**：适用于 OT/VP/VE/Dirichlet 等多种概率路径，不局限于图像生成

## 局限与展望
- TV 距离有界不等价于 KL 散度有界，作者承认 KL 散度的控制仍是开放问题
- 超参数 $\lambda_1, \lambda_2$ 的选择缺乏原则性方法，需要搜索
- 大规模图像生成（如 ImageNet 256）实验缺失，仅在 CIFAR-10 和小数据集上验证

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次从 PDE 角度建立 FM 概率路径误差的理论框架
- 实验充分度: ⭐⭐⭐⭐ 合成+真实任务覆盖广，但缺少大规模视觉生成实验
- 写作质量: ⭐⭐⭐⭐⭐ 理论严谨，行文流畅
- 价值: ⭐⭐⭐⭐ 对 FM 基础理论有重要推进

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Flexibility-conditioned Protein Structure Design with Flow Matching](flexibility-conditioned_protein_structure_design_with_flow_matching.md)
- [\[ICML 2025\] Efficient Molecular Conformer Generation with SO(3)-Averaged Flow Matching and Reflow](efficient_molecular_conformer_generation_with_so3-averaged_flow_matching_and_ref.md)
- [\[ICML 2025\] Scalable Generation of Spatial Transcriptomics from Histology Images via Whole-Slide Flow Matching](scalable_generation_of_spatial_transcriptomics_from_histology_images_via_whole-s.md)
- [\[NeurIPS 2025\] Curly Flow Matching for Learning Non-gradient Field Dynamics](../../NeurIPS2025/computational_biology/curly_flow_matching_for_learning_non-gradient_field_dynamics.md)
- [\[NeurIPS 2025\] Energy Matching: Unifying Flow Matching and Energy-Based Models for Generative Modeling](../../NeurIPS2025/computational_biology/energy_matching_unifying_flow_matching_and_energy-based_models_for_generative_mo.md)

</div>

<!-- RELATED:END -->
