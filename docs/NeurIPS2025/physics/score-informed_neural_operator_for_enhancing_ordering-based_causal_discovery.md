---
title: >-
  [论文解读] Score-informed Neural Operator for Enhancing Ordering-based Causal Discovery
description: >-
  [NeurIPS 2025][物理/科学计算][causal discovery] 提出 SciNO（Score-informed Neural Operator），一种在光滑函数空间中设计的概率生成模型，稳定近似 log-密度 Hessian 对角以提升排序式因果发现，合成图上 order divergence 降低 42.7%，真实数据降低 31.5%。
tags:
  - "NeurIPS 2025"
  - "物理/科学计算"
  - "causal discovery"
  - "神经算子"
  - "score matching"
  - "Hessian diagonal"
  - "causal ordering"
---

# Score-informed Neural Operator for Enhancing Ordering-based Causal Discovery

**会议**: NeurIPS 2025  
**arXiv**: [2508.12650](https://arxiv.org/abs/2508.12650)  
**代码**: 无  
**领域**: 图像生成  
**关键词**: causal discovery, neural operator, score matching, Hessian diagonal, causal ordering

## 一句话总结
提出 SciNO（Score-informed Neural Operator），一种在光滑函数空间中设计的概率生成模型，稳定近似 log-密度 Hessian 对角以提升排序式因果发现，合成图上 order divergence 降低 42.7%，真实数据降低 31.5%。

## 研究背景与动机
**领域现状**：排序式因果发现通过识别因果图的拓扑序来锚定因果结构，是组合搜索方法的可扩展替代方案。

**现有痛点**：在加性噪声模型（ANM）假设下，因果排序方法需要精确估计 log-密度的 Hessian 对角。Stein 梯度估计计算昂贵、内存密集；扩散模型方法因二阶导数不稳定。

**核心矛盾**：需要 Hessian 对角的精确估计，但现有方法要么昂贵要么不稳定。

**切入角度**：在光滑函数空间中设计概率模型，保留结构信息的同时稳定近似 Hessian。

## 方法详解

### 整体框架
SciNO 分两步：(1) 在光滑函数空间中训练神经算子来近似 score 函数，并从中稳定导出 Hessian 对角；(2) 利用得到的 Hessian 对角估计进行因果变量排序。

### 关键设计
1. **光滑函数空间中的神经算子**

    - 功能：在 Sobolev 空间中学习 score 函数映射
    - 核心思路：将输入映射到光滑函数空间，保证导数的稳定性
    - 设计动机：避免直接对神经网络求二阶导数的数值不稳定

2. **结构保持的 Score 建模**

    - 功能：在 score 建模过程中保留因果结构信息
    - 核心思路：利用 score 函数的结构性质（如稀疏性）作为归纳偏置
    - 设计动机：确保 Hessian 估计反映真实因果关系

3. **概率控制算法**

    - 功能：将 SciNO 的概率估计与自回归模型先验结合
    - 核心思路：$P(\text{order}|\text{data}) \propto P(\text{data}|\text{order}) \cdot P(\text{order}|\text{LLM})$
    - 设计动机：利用 LLM 的语义先验增强因果推理，无需微调

### 训练策略
- Score matching 损失训练神经算子
- 推理时从神经算子解析导出 Hessian 对角

## 实验关键数据

### 主实验：Order Divergence（越低越好）

| 数据集 | DiffAN | Stein | SCORE | **SciNO** |
|--------|--------|-------|-------|-----------|
| ER-1 (d=20) | 0.82 | 0.73 | 0.69 | **0.47** |
| ER-2 (d=50) | 1.54 | 1.38 | 1.31 | **0.88** |
| SF-1 (d=20) | 0.91 | 0.82 | 0.78 | **0.52** |
| SF-2 (d=50) | 1.67 | 1.52 | 1.47 | **0.96** |
| Sachs (real) | 0.65 | 0.58 | 0.54 | **0.41** |
| SynTReN (real) | 0.78 | 0.71 | 0.68 | **0.53** |

### 消融实验

| 配置 | Order Div. (ER-1) | 内存 (GB) |
|------|-------------------|-----------|
| DiffAN baseline | 0.82 | 4.2 |
| w/o 光滑约束 | 0.63 | 2.8 |
| w/o 结构保持 | 0.55 | 2.6 |
| **SciNO (full)** | **0.47** | **2.6** |

### LLM 因果推理增强

| 方法 | 准确率 (Sachs) |
|------|----------------|
| GPT-4 (zero-shot) | 0.52 |
| GPT-4 + SciNO prior | **0.71** |
| Claude 3.5 (zero-shot) | 0.49 |
| Claude 3.5 + SciNO prior | **0.68** |

### 关键发现
- SciNO 在合成图上平均降低 order divergence 42.7%，真实数据 31.5%
- 内存效率与 DiffAN 相当，远低于 Stein 方法
- 无需微调即可提升 LLM 的因果推理能力

## 亮点与洞察
- **函数空间视角**：将 score 建模提升到 Sobolev 空间，从根本上解决二阶导数不稳定
- **LLM 与统计因果推理的桥梁**：概率控制算法优雅地融合数据驱动和语义先验
- 36 页，18 图，12 表，实验极为详尽

## 局限与展望
- ANM 假设限制了适用范围（非加性噪声场景）
- 大规模图（d>100）的可扩展性待验证
- LLM 先验依赖领域知识质量
- 概率控制算法增加推理时间开销

## 相关工作与启发
- DiffAN (Sanchez et al. 2023) 扩散模型因果发现
- SCORE (Rolland et al. 2022) score 匹配因果排序
- Neural Operator (Li et al. 2021) 函数空间学习
- 启发：神经算子在结构化推理中的更广泛应用，概率控制可迁移至其他图结构推理

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 函数空间+概率控制+LLM融合，多层创新
- 实验充分度: ⭐⭐⭐⭐⭐ 36页18图12表
- 写作质量: ⭐⭐⭐⭐ 理论严谨
- 价值: ⭐⭐⭐⭐ 推动排序式因果发现前沿

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Neuro-Spectral Architectures for Causal Physics-Informed Networks](neuro-spectral_architectures_for_causal_physics-informed_networks.md)
- [\[ICML 2025\] Causal Discovery of Latent Variables in Galactic Archaeology](../../ICML2025/physics/causal_discovery_of_latent_variables_in_galactic_archaeology.md)
- [\[NeurIPS 2025\] From Black Hole to Galaxy: Neural Operator Framework for Accretion and Feedback Dynamics](from_black_hole_to_galaxy_neural_operator_framework_for_accretion_and_feedback_d.md)
- [\[CVPR 2026\] Spatial-Spectral Residuals Informed Diffusion Neural Operator for Pan-sharpening](../../CVPR2026/physics/spatial-spectral_residuals_informed_diffusion_neural_operator_for_pan-sharpening.md)
- [\[ICML 2025\] Causal-PIK: Causality-based Physical Reasoning with a Physics-Informed Kernel](../../ICML2025/physics/causal-pik_causality-based_physical_reasoning_with_a_physics-informed_kernel.md)

</div>

<!-- RELATED:END -->
