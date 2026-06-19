---
title: >-
  [论文解读] Prior-Guided Flow Matching for Target-Aware Molecule Design with Learnable Atom Number
description: >-
  [NeurIPS 2025][计算生物][基于结构的药物设计] 提出 PAFlow，基于流匹配框架的 3D 分子生成模型，通过蛋白-配体交互预测器引导向量场和可学习原子数预测器，在 CrossDocked2020 上实现 -8.31 Avg. Vina Score 的新 SOTA，大幅超越已有方法。
tags:
  - "NeurIPS 2025"
  - "计算生物"
  - "基于结构的药物设计"
  - "流匹配"
  - "蛋白-配体交互引导"
  - "原子数预测"
  - "3D分子生成"
---

# Prior-Guided Flow Matching for Target-Aware Molecule Design with Learnable Atom Number

**会议**: NeurIPS 2025  
**arXiv**: [2509.01486](https://arxiv.org/abs/2509.01486)  
**代码**: [GitHub](https://github.com/CMACH508/PAFlow)  
**领域**: 计算生物  
**关键词**: 基于结构的药物设计, 流匹配, 蛋白-配体交互引导, 原子数预测, 3D分子生成

## 一句话总结

提出 PAFlow，基于流匹配框架的 3D 分子生成模型，通过蛋白-配体交互预测器引导向量场和可学习原子数预测器，在 CrossDocked2020 上实现 -8.31 Avg. Vina Score 的新 SOTA，大幅超越已有方法。

## 研究背景与动机

基于结构的药物设计（SBDD）旨在生成对靶点蛋白具有高结合亲和力的 3D 分子。现有方法面临三大问题：(1) 自回归模型的不自然生成顺序导致片段不合理和误差累积；(2) 扩散模型的去噪轨迹高度随机，分子质量不稳定；(3) 所有非自回归方法通过预定义分布采样原子数量，依赖参考配体先验，容易导致分子大小与口袋几何不匹配。流匹配（FM）框架通过 ODE 求解器实现快速稳定的生成，有望解决前两个问题。

## 方法详解

### 整体框架

PAFlow 采用 FM 框架建模分子生成过程。原子坐标使用 Variance Preserving (VP) 路径建模，原子类型使用新推导的离散条件流匹配（CFM）建模。生成过程通过 Euler ODE 求解器从初始噪声迭代更新到目标分子，同时引入交互预测器引导和原子数预测器。

### 关键设计

1. **双路径流匹配 (Dual-Path Flow Matching)**: 连续原子坐标 $\mathbf{x}$ 使用 VP 路径的条件概率 $p_t(\mathbf{x}|\mathbf{x}_1) = \mathcal{N}(\mathbf{x}|\sqrt{\bar{\alpha}_{1-t}}\mathbf{x}_1, (1-\bar{\alpha}_{1-t})\mathbf{I})$；离散原子类型 $\mathbf{a}$ 使用分类分布路径 $\mathbf{c}(\mathbf{a}, \mathbf{a}_1) = \bar{\alpha}_{1-t}\mathbf{a}_1 + (1-\bar{\alpha}_{1-t})/K$。作者推导了离散原子类型的条件向量场 $u_t^a = \bar{\alpha}_{1-t}'(\mathbf{a}_1 - \mathbf{a}_0)$，将两者统一到 FM 框架下。整体使用 SE(3)-等变 GNN $\phi_\theta$ 参数化，保证生成过程对蛋白-配体复合物的平移旋转不变性。

2. **蛋白-配体交互引导 (Prior-Guided Generation)**: 在 SE(3)-EGNN 中集成一个结合亲和力预测器，通过聚合最终隐藏嵌入预测归一化的结合亲和力 $\hat{y}$。采样时，以梯度形式引导坐标向量场：$\mathbf{x}_{t+\Delta t} = \mathbf{x}_t + (v_\theta^x + \gamma \frac{\bar{\alpha}'_{1-t}}{2\bar{\alpha}_{1-t}} \nabla \log p_\theta(y=1|\mathbf{m}_t))\Delta t$，其中 $\gamma$ 控制引导强度。虽然无法直接引导离散原子类型，但优化后的坐标可通过 GNN 间接影响原子类型预测。

3. **可学习原子数预测器 (Learnable Atom Number Predictor)**: 仅利用蛋白口袋信息（原子数 $N_P$、体积 $V$、表面积 $A$、空间大小 $S$）预测配体原子数，完全不依赖参考配体。使用归一化标签训练，预测时输出添加高斯噪声 $\tau \sim \mathcal{N}(0, \delta^2)$ 作为正则化，增加多样性并避免过拟合。

### 损失函数 / 训练策略

总损失包含三部分：(1) 坐标 CFM 损失 $\mathcal{L}_{CFM}^x$：回归目标坐标向量场；(2) 原子类型 CFM 损失 $\mathcal{L}_{CFM}^a$：回归目标类型向量场；(3) 交互预测损失 $\mathcal{L}_{inter}$：MSE 损失预测结合亲和力。原子数预测器独立训练，使用归一化标签和去归一化输出。采样时使用 Euler ODE 求解器，默认 $T=100$ 步，也支持 $T=20$ 的快速模式。坐标初始化为口袋内标准高斯分布，原子类型初始化为均匀分布。

## 实验关键数据

### 主实验

| 方法 | Avg. Vina Score↓ | Avg. Vina Dock↓ | High Affinity↑ | QED↑ | SA↑ |
|------|-------------------|------------------|----------------|------|-----|
| PAFlow | **-8.31** | **-9.46** | **80.8%** | 0.49 | 0.57 |
| ALiDiff | -7.07 | -8.90 | 73.4% | 0.50 | 0.57 |
| TAGMol | -7.02 | -8.59 | 69.8% | 0.55 | 0.56 |
| MolCRAFT (BFN) | -6.59 | -7.92 | 59.1% | 0.50 | 0.69 |
| FlowSBDD (FM) | -3.62 | -8.50 | 63.4% | 0.47 | 0.51 |

### 消融实验

| 配置 | Avg. Vina Score | 说明 |
|------|-----------------|------|
| 完整 PAFlow | -8.31 | 完整模型 |
| 去掉交互引导 (w/o P) | -5.18 | 引导贡献 60.4% 提升 |
| FM vs 扩散 (w/o PA vs TargetDiff) | -5.13 vs -5.47 | FM采样策略本身更优 |
| 原子数预测器 vs 预定义分布 | MAE 3.35 vs ~5+ | 预测结果更贴合口袋 |

### 关键发现

- PAFlow 在 77% 的测试靶标上达到最高结合亲和力
- 采样速度比 TargetDiff 快 5.5 倍（717s vs 3968s），步数减少到 20 步时比 MolCRAFT 更快
- 交互引导是最大贡献因素：Avg. Vina Score 从 -5.18 提升到 -8.31
- 使用线性路径的 FlowSBDD 表现不佳，验证了 VP 路径对 SBDD 复杂任务的必要性
- 原子数预测器的 MAE 为 3.35，显著优于预定义分布采样
- Median Vina Score 更优（-8.92 vs ALiDiff -7.95），表明生成质量更稳定
- $T=20$ 步快速模式下仍优于 MolCRAFT 的结合亲和力

## 亮点与洞察

- 首次为离散原子类型推导了 FM 框架下的条件向量场，使坐标和类型在统一框架下生成
- 交互预测引导策略效果极为显著，60%+ 的亲和力提升仅来自引导
- 原子数预测器的设计理念——仅用口袋信息而非参考配体——更符合实际药物发现场景
- 噪声注入的数学合理性证明是一个有趣的细节

## 局限与展望

- QED/SA 等分子性质未在生成中显式优化，有进一步提升空间
- 交互预测器的引导强度 $\gamma$ 需要手动调节
- 仅在 CrossDocked2020 数据集上评估，泛化性有待验证
- 可将预测器扩展到分子性质，实现多目标引导生成

## 相关工作与启发

- TargetDiff 使用相同概率路径但扩散去噪采样，PAFlow 的 ODE 采样更稳定
- FlowSBDD 使用线性路径的 FM，但不适合 SBDD 的复杂性
- 引导策略受 TAGMol/ALiDiff 启发，但在 FM 框架下推导了新的引导公式
- 原子数预测器思路可推广到其他需要确定生成对象大小的任务
- AR 和 Pocket2Mol 的自回归模式虽然灵活但误差累积严重
- DecompDiff 的分解策略和 IPDiff 的交互感知扩散虽有效但仍受限于扩散框架的随机性
- MolCRAFT 的 BFN 框架在连续参数空间生成，但结合亲和力不如 PAFlow
- SE(3)-等变 GNN 的选择保证了物理对称性，是分子生成的标准设计
- 结构分析显示 PAFlow 生成的分子在几何形状上更贴合蛋白口袋

## 评分

- 新颖性: ⭐⭐⭐⭐ 离散类型 CFM 推导和交互引导 FM 组合新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 全面的基线对比、消融、采样效率、可视化分析
- 写作质量: ⭐⭐⭐⭐ 方法推导清晰，实验展示直观
- 价值: ⭐⭐⭐⭐⭐ SBDD 领域的显著进步，-8.31 Vina Score 大幅刷新 SOTA

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Unified All-Atom Molecule Generation with Neural Fields](unified_all-atom_molecule_generation_with_neural_fields.md)
- [\[NeurIPS 2025\] Energy Matching: Unifying Flow Matching and Energy-Based Models for Generative Modeling](energy_matching_unifying_flow_matching_and_energy-based_models_for_generative_mo.md)
- [\[ICML 2025\] Flexibility-conditioned Protein Structure Design with Flow Matching](../../ICML2025/computational_biology/flexibility-conditioned_protein_structure_design_with_flow_matching.md)
- [\[NeurIPS 2025\] Uncertainty-Aware Multi-Objective Reinforcement Learning-Guided Diffusion Models for 3D De Novo Molecular Design](uncertainty-aware_multi-objective_reinforcement_learning-guided_diffusion_models.md)
- [\[NeurIPS 2025\] Curly Flow Matching for Learning Non-gradient Field Dynamics](curly_flow_matching_for_learning_non-gradient_field_dynamics.md)

</div>

<!-- RELATED:END -->
