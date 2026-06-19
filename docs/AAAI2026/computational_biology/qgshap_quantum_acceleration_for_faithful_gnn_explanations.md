---
title: >-
  [论文解读] QGShap: Quantum Acceleration for Faithful GNN Explanations
description: >-
  [AAAI 2026 (QC+AI Workshop)][计算生物][图神经网络] 提出 QGShap，一种利用量子振幅放大技术加速精确 Shapley 值计算的图神经网络可解释性框架，在保持精确计算（非近似）的同时实现了相对经典 Monte Carlo 方法的二次加速。 图神经网络（GNN）在药物发现、社交网络分析和推荐系…
tags:
  - "AAAI 2026 (QC+AI Workshop)"
  - "计算生物"
  - "图神经网络"
  - "Shapley值"
  - "量子振幅估计"
  - "精确归因"
  - "图可解释性"
---

# QGShap: Quantum Acceleration for Faithful GNN Explanations

**会议**: AAAI 2026 (QC+AI Workshop)  
**arXiv**: [2512.03099](https://arxiv.org/abs/2512.03099)  
**代码**: [GitHub](https://github.com/smlab-niser/qgshap)  
**领域**: 可解释性  
**关键词**: GNN解释, Shapley值, 量子振幅估计, 精确归因, 图可解释性

## 一句话总结

提出 QGShap，一种利用量子振幅放大技术加速精确 Shapley 值计算的图神经网络可解释性框架，在保持精确计算（非近似）的同时实现了相对经典 Monte Carlo 方法的二次加速。

## 研究背景与动机

图神经网络（GNN）在药物发现、社交网络分析和推荐系统等关键领域取得了巨大成功，但其"黑盒"特性阻碍了在需要透明性和可问责性的场景中的部署。

**Shapley 值**作为合作博弈论中唯一满足效率性、对称性、虚拟玩家和可加性四大公理的归因方案，被认为是量化各组件对模型预测贡献的数学最严谨方法。然而，精确计算 Shapley 值需要遍历所有 $2^n$ 个联盟（coalition），对于一个有 $n$ 个节点的图来说计算量呈指数级增长，这是经典的 #P-complete 问题。

现有近似策略的局限：

**SubgraphX**：使用 Monte Carlo Tree Search (MCTS) 和采样来近似 Shapley 值，对大规模稠密图不可行。

**GraphSVX**：构建代理模型在扰动数据集上采样联盟，对中等规模联盟欠采样，降低解释保真度。

**GNNShap**：利用 GPU 并行和批处理加速估计，但本质仍是依赖采样的近似技术。

**GraphSHAP-IQ**：利用消息传递 GNN 的结构特性计算精确的任意阶 Shapley 交互，但对深层架构、稠密图或非线性 readout 函数的 GNN 不适用。

核心矛盾在于：**保真度（fidelity）和效率不可兼得**——精确方法太慢，快速方法牺牲精度。量子计算的振幅放大技术提供了一条新路径：达到 $\mathcal{O}(1/\epsilon)$ 的查询复杂度，相比经典 MC 的 $\mathcal{O}(1/\epsilon^2)$ 实现二次加速，同时保持精确计算。

## 方法详解

### 整体框架

QGShap 的流程如下：

1. **穷举联盟生成**：对输入图 $G=(V,E)$，枚举所有非空节点子集 $\mathcal{C} = \{S \subseteq V : S \neq \emptyset\}$。
2. **联盟打分**：通过零填充编码（zero-fill encoding）为每个子集构造遮蔽图 $G_S$，被排除的节点用零向量替代，然后由训练好的 GNN $f_\theta$ 评估得到合作博弈值 $v(S) = f_\theta(G_S)$。
3. **量子 Shapley 值估计**：将合作博弈值归一化后编码到量子态中，利用量子振幅估计（QAE）计算每个节点的精确 Shapley 值。
4. **归一化与排序**：对所有节点的 Shapley 值进行归一化，生成全局重要性排名。

### 关键设计

1. **三寄存器量子电路**：

    - **分区寄存器 $Q_{pt}$**（$\ell$ 个量子比特）：通过 beta 函数旋转编码 Shapley 权重系数 $w_{|S|,|V|}$ 的振幅分布，其中 $\ell = \mathcal{O}(\log \frac{(U_{max} - U_{min})n}{\varepsilon})$。
    - **播放器寄存器 $Q_{pl}$**（$|V|$ 个量子比特）：存储所有不含节点 $p_j$ 的联盟 $S \subseteq V \setminus \{p_j\}$ 的叠加态。
    - **效用寄存器 $Q_{ut}$**（1 个量子比特）：存储每个联盟的归一化合作博弈值。
   设计动机：通过受控旋转使每个基态 $|S\rangle$ 的振幅正比于 $\sqrt{w_{|S|,|V|}}$，精确编码 Shapley 权重。

2. **归一化合作博弈值**：将原始值映射到 $[0,1]$：$\hat{v}(S) = \frac{v(S) - \min_{S'} v(S')}{\max_{S'} v(S') - \min_{S'} v(S')}$。这是量子态编码的必要步骤，确保效用值可以通过量子比特的振幅旋转来表示。

3. **量子振幅估计加速**：两个量子预言机 $U_{val}^{(+)}$ 和 $U_{val}^{(-)}$ 分别实现包含和不包含节点 $p_j$ 时的归一化效用函数。通过 QAE 提取加权期望贡献 $\phi^{(+)}(p_j)$ 和 $\phi^{(-)}(p_j)$，最终 Shapley 值为：$\phi(p_j) = (\max_S v(S) - \min_S v(S)) \cdot (\phi^{(+)}(p_j) - \phi^{(-)}(p_j))$。总查询复杂度为 $\mathcal{O}(\frac{U_{max} - U_{min}}{\varepsilon})$，相比经典 MC 的 $\mathcal{O}(1/\varepsilon^2)$ 实现近二次加速。

4. **节点中心的层次化方法**：QGShap 在每个联盟内计算各节点的 Shapley 贡献，将节点视为"博弈玩家"。这与 SubgraphX（将采样的子图和剩余节点同时作为玩家）和 GraphSHAP-IQ（在感受野内计算精确交互但不显式遍历联盟）有本质区别。

### 损失函数 / 训练策略

QGShap 是一个**事后解释（post hoc）框架**，不涉及额外训练。底层 GNN 使用标准方式训练：采用 GIN（Graph Isomorphism Network）架构，隐藏维度 128，3 层 GIN 层，用 Adam 优化器训练 100 epoch，学习率 $10^{-3}$，二元交叉熵损失。QGShap 只在已训练好的 GNN 之上运行，计算解释归因。

## 实验关键数据

### 主实验

在 Bridge 和 BA2-Motif 两个合成基准数据集上与经典解释方法比较：

| 数据集 | 方法 | Fid+ | Fid- | Sparsity | GEA | Top-2 Acc |
|--------|------|:---:|:---:|:---:|:---:|:---:|
| Bridge | GNNExplainer | 0.60 | 1.00 | 0.75 | 0.07 | 0.10 |
| Bridge | PGExplainer | 0.99 | 1.00 | 0.75 | 0.00 | 0.00 |
| Bridge | SubgraphX | **1.00** | 1.00 | 0.75 | **1.00** | **1.00** |
| Bridge | **QGShap** | **1.00** | 1.00 | 0.75 | **1.00** | **1.00** |
| BA2-Motif | GNNExplainer | **1.00** | 1.00 | 0.75 | 0.32 | 0.50 |
| BA2-Motif | PGExplainer | 0.01 | 1.00 | 0.75 | 0.00 | 0.00 |
| BA2-Motif | SubgraphX | **1.00** | 1.00 | 0.75 | **0.40** | 0.79 |
| BA2-Motif | **QGShap** | **1.00** | 1.00 | 0.75 | **0.40** | **1.00** |

### 消融实验

| 配置 | 关键结果 | 说明 |
|------|---------|------|
| 经典穷举 vs 量子加速 | 查询复杂度 $\mathcal{O}(1/\varepsilon)$ vs $\mathcal{O}(1/\varepsilon^2)$ | 量子方法实现近二次加速 |
| 节点数 ≤ 8 | 可行 | 受限于当前量子仿真资源 |
| Bridge 运行时间 | ~31 小时 | 48 核 AMD CPU + A100 GPU |
| BA2-Motif 运行时间 | ~42 小时 | 同上配置的经典模拟 |

### 关键发现

1. **精确 Shapley 值 = 完美解释**：在 Bridge 数据集上，QGShap 和 SubgraphX 均取得完美的 GEA 和 Top-2 准确率（1.00），而 GNNExplainer 和 PGExplainer 表现显著较差。这验证了 Shapley 值方法在捕获关键图结构方面的优越性。
2. **Top-2 准确率的关键区别**：在 BA2-Motif 上，QGShap 的 Top-2 准确率为 1.00，而 SubgraphX 仅为 0.79。这意味着 QGShap 在所有情况下都能正确找到最重要的两个节点，而 SubgraphX 因采样近似有时会错过。
3. **PGExplainer 的失败**：PGExplainer 在两个数据集上的 GEA 和 Top-2 Acc 均为 0.00，说明其参数化概率模型在这些结构化任务上完全失效。

## 亮点与洞察

- **精确性与效率的统一**：QGShap 是首个利用量子计算在 GNN 可解释性中实现精确（非近似）Shapley 值计算的框架，通过振幅放大获得二次加速。这打破了"精确但慢 vs 快但近似"的传统困境。
- **公理化保证**：由于计算的是精确 Shapley 值，QGShap 的解释自动满足效率性、对称性、虚拟玩家和可加性四大公理，这是近似方法无法保证的。
- **可作为评估基准**：精确 Shapley 值可以作为评估其他近似解释方法的 ground truth 基准。
- **跨领域桥梁**：本文连接了量子计算和 GNN 可解释性两个领域，为量子-经典混合方法在 AI 可解释性中的应用开辟了方向。

## 局限与展望

1. **图规模受限**：当前仅支持 ≤8 个节点的图。联盟数 $2^n$ 随节点数指数增长，量子比特需求和电路深度快速增加。
2. **经典模拟开销巨大**：在 A100 GPU 上运行量子仿真需要 31-42 小时，实际量子硬件有限。
3. **量子噪声影响**：真实量子硬件的噪声和退相干会降低振幅估计的精度，论文仅在模拟器上验证。
4. **仅在合成数据集上验证**：Bridge 和 BA2-Motif 都是合成数据集，未在真实世界的大规模图数据上测试。
5. **未与 GraphSHAP-IQ 直接比较**：GraphSHAP-IQ 也能在特定条件下计算精确 Shapley 交互，缺少与之的系统比较。

## 相关工作与启发

本文系统梳理了 GNN 可解释性的发展：从梯度/扰动方法（GNNExplainer, PGExplainer）→ 代理模型（GraphLIME）→ Shapley 值方法（SubgraphX, GNNShap, GraphSVX, GraphSHAP-IQ）→ 量子加速（QGShap）。关键启发是量子计算不仅可以用于加速模型训练/推理，也可以用于加速模型**解释**——一个此前未被充分探索的方向。随着量子硬件的进步，这类方法可能从理论验证走向实际应用。此外，论文对现有 Shapley 近似方法在保真度 vs 效率上的权衡总结非常到位，有助于理解各方法的适用场景。

## 评分

- 新颖性：⭐⭐⭐⭐⭐ — 首次将量子计算用于 GNN 精确 Shapley 值解释
- 实验充分性：⭐⭐⭐ — 仅两个小规模合成数据集，缺乏真实场景验证
- 实用性：⭐⭐ — 受限于量子硬件和图规模，当前实用价值有限
- 写作质量：⭐⭐⭐⭐ — 理论阐述清晰，量子算法细节完整

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] EPO: Diverse and Realistic Protein Ensemble Generation via Energy Preference Optimization](epo_diverse_and_realistic_protein_ensemble_generation_via_energy_preference_opti.md)
- [\[AAAI 2026\] MergeDNA: Context-aware Genome Modeling with Dynamic Tokenization through Token Merging](mergedna_context-aware_genome_modeling_with_dynamic_tokenization_through_token_m.md)
- [\[AAAI 2026\] On the Information Processing of One-Dimensional Wasserstein Distances with Finite Samples](on_the_information_processing_of_one-dimensional_wasserstein_distances_with_fini.md)
- [\[AAAI 2026\] Apo2Mol: 3D Molecule Generation via Dynamic Pocket-Aware Diffusion Models](apo2mol_3d_molecule_generation_via_dynamic_pocket-aware_diff.md)
- [\[AAAI 2026\] Distributional Priors Guided Diffusion for Generating 3D Molecules in Low Data Regimes](distributional_priors_guided_diffusion_for_generating_3d_molecules_in_low_data_r.md)

</div>

<!-- RELATED:END -->
