---
title: >-
  [论文解读] Pharmacophore-Guided Generative Design of Novel Drug-Like Molecules
description: >-
  [NeurIPS 2025][计算生物][药效团引导] 提出一种药效团引导的分子生成框架，在强化学习模型（FREED++）的奖励函数中同时最大化药效团相似度和最小化结构相似度，生成既保留生物活性特征又具有高结构新颖性的候选药物分子。 AI驱动的早期药物发现正在变革制药范式，但现有方法存在明显痛点： 分子对接的局限性：传统方法…
tags:
  - "NeurIPS 2025"
  - "计算生物"
  - "药效团引导"
  - "分子生成"
  - "强化学习"
  - "药物设计"
  - "结构多样性"
---

# Pharmacophore-Guided Generative Design of Novel Drug-Like Molecules

**会议**: NeurIPS 2025  
**arXiv**: [2510.01480](https://arxiv.org/abs/2510.01480)  
**代码**: 暂无（camera-ready版本后公开）  
**领域**: 计算生物  
**关键词**: 药效团引导, 分子生成, 强化学习, 药物设计, 结构多样性

## 一句话总结

提出一种药效团引导的分子生成框架，在强化学习模型（FREED++）的奖励函数中同时最大化药效团相似度和最小化结构相似度，生成既保留生物活性特征又具有高结构新颖性的候选药物分子。

## 研究背景与动机

AI驱动的早期药物发现正在变革制药范式，但现有方法存在明显痛点：

**分子对接的局限性**：传统方法依赖分子对接（docking）评估结合亲和力，但对接计算开销大，且评分函数基于线性能量组合，与实验结合亲和力的相关性差

**现有生成方法的不足**：已有框架如DrugMetric、NP-VAE等虽能生成高新颖性分子，但常牺牲对接保真度或药效团一致性。大多数方法仅优化对接分数或依赖特定结合位点

**药效团方法的独特优势**：药效团关注关键相互作用特征（如氢键供体/受体、芳香基团、疏水基团）的空间排列，提供了更可解释、更鲁棒的生物活性代理指标，可跨越不同化学骨架

本文的关键洞察是：好的候选药物应该**药效团相似（保留活性特征）但结构不同（确保新颖性和可专利性）**。这是一个双目标优化问题。

## 方法详解

### 整体框架

框架核心是在FREED++强化学习模型的奖励函数中引入双重目标。每个RL循环中，生成的分子通过两种不同的分子表示编码，并与用户提供的参考集（如FDA批准药物）进行比较：

- **药效团相似度**：使用CATS（Chemically Advanced Template Search）描述符捕获药效团模式
- **结构相似度**：使用MACCS（Molecular ACCess System）键或MAP4指纹表示子结构特征

### 关键设计

1. **双表示编码**：CATS描述符是连续值向量，编码药效团特征的空间排列（如氢键供体-受体对之间的拓扑距离分布），适合捕捉功能层面的相似性。MACCS键是二值指纹，直接编码子结构片段的存在与否，适合衡量骨架层面的相似性。MAP4则结合原子对关系和圆形子结构，提供更丰富的表达。

2. **对偶相似度度量**：针对不同表示选用最适合的度量方法：

    - 药效团相似度（CATS）：余弦相似度（衡量向量方向一致性）和欧氏距离（同时捕获幅度和方向）
    - 结构相似度（MACCS/MAP4）：Tanimoto系数（经典二值指纹度量）和MAP4评分

3. **四种奖励函数配置**：系统测试了不同度量组合：

    - Setup 1: QED + Tanimoto + 欧氏距离
    - Setup 2: QED + Tanimoto + 余弦相似度
    - Setup 3: QED + MAP4 + 欧氏距离
    - Setup 4: QED + MAP4 + 余弦相似度

   奖励函数显式设计为**最大化药效团相似度、最小化结构相似度**的双目标优化。

### 损失函数 / 训练策略

框架基于FREED++强化学习模型，奖励函数由三部分组成：
- QED（定量药物类似度估计）：确保生成分子具备药物性质
- 药效团相似度：驱动功能特征保留
- 结构不相似度（取反）：鼓励骨架创新

案例研究：靶向alpha雌激素受体（PDB ID: 8AWG），用于乳腺癌。参考集为已知的雌激素受体调节剂和拮抗剂。对接使用QVina。

## 实验关键数据

### 主实验

| 配置 | Tanimoto(↓) | 余弦相似度(↑) | 欧氏距离(↓) | QED(↑) | 对接分数(↓) | SA分数(↓) | 新颖性(↑) |
|---|---|---|---|---|---|---|---|
| 基线 | 0.34±0.05 | 0.58±0.27 | 70.3±13.0 | 0.30±0.08 | **-8.64**±1.03 | 6.28±0.64 | 100% |
| Setup 1 | 0.34±0.05 | **0.94**±0.06 | **34.8**±7.84 | 0.33±0.13 | -6.49±1.17 | **4.64**±0.51 | 100% |
| Setup 2 | 0.36±0.05 | 0.83±0.05 | 54.9±8.60 | **0.59**±0.16 | -6.71±0.55 | 4.72±0.49 | 99.6% |
| Setup 3 | 0.35±0.05 | 0.94±0.06 | 50.5±10.2 | 0.44±0.16 | -7.09±0.66 | 4.67±0.45 | 84.5% |
| Setup 4 | 0.35±0.05 | 0.87±0.07 | 38.9±9.37 | 0.34±0.15 | -6.47±1.02 | 4.61±0.50 | 100% |

*已知活性分子的平均对接分数为-6.64*

### 消融实验

| 对比项 | 结论 | 说明 |
|---|---|---|
| 欧氏 vs 余弦（固定Tanimoto） | 欧氏距离药效团相似度更高 | Setup 1余弦0.94 vs Setup 2余弦0.83 |
| Tanimoto vs MAP4（固定余弦） | MAP4配置QED更高、对接更好 | Setup 4 QED 0.34 vs Setup 2 QED 0.59不成立；Setup 3对接-7.09最佳 |
| 基线 vs 所有药效团配置 | 药效团引导显著改善SA和QED | 基线SA 6.28→所有配置<4.72 |
| 药效团引导 vs 基线对接 | 对接分数略有降低但接近已知活性物 | -6.47~-7.09 vs 已知活性物-6.64 |

### 关键发现

- 药效团引导大幅提升了药效团保真度（余弦从0.58→0.83-0.94），同时保持低结构相似度
- QED和SA显著改善：QED从0.30提升至0.33-0.59，SA从6.28降至4.61-4.72
- 对接分数虽不如基线（基线过度优化对接），但与已知活性分子(-6.64)相当
- 生成的TOP分子保留了关键药效团模式（三芳香/杂芳香基团、保守的氢键向量、疏水间隔）同时重塑骨架
- MAP4+余弦配置在药效团相似度和对接之间取得最佳平衡

## 亮点与洞察

- **靶标无关且对接无关**：不需要蛋白质结构或结合位点信息即可生成候选分子，适合早期探索
- **双目标平衡**的设计理念实用：药效团相似确保功能相关性，结构不相似确保可专利性
- 将对接作为后筛工具而非生成奖励的思路减少了计算开销和对接评分函数的不可靠影响

## 局限与展望

- 仅在一个靶点（alpha雌激素受体）上验证，泛化性未知
- 生成分子仅做了计算验证，缺少合成和生物测定的实验验证
- 药效团描述符集合较有限（仅CATS+MACCS/MAP4），可能限制骨架多样性
- 对接分数仅作为中等代理指标，MAP4配置的新颖性仅84.5%
- 未与PGMG、PharmaDiff等同类药效团引导方法进行直接对比实验

## 相关工作与启发

- **PGMG**：基于图神经网络的药效团引导分子生成
- **PharmaDiff**：基于扩散模型的药效团条件化分子设计
- **FREED++**：本文使用的RL基础模型
- 启发：药效团作为生物活性的代理比对接更稳定，适合作为生成模型的奖励信号

## 评分

- 新颖性: ⭐⭐⭐⭐ 双目标优化思路合理但并非突破性创新
- 实验充分度: ⭐⭐⭐⭐ 单靶点、缺少与同类方法的对比、无实验验证
- 写作质量: ⭐⭐⭐⭐ 结构清晰但部分细节不够深入
- 价值: ⭐⭐⭐⭐ 为药效团引导的分子生成提供了实用框架，但验证有限

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] UniMoMo: Unified Generative Modeling of 3D Molecules for De Novo Binder Design](../../ICML2025/computational_biology/unimomo_unified_generative_modeling_of_3d_molecules_for_de_novo_binder_design.md)
- [\[NeurIPS 2025\] Manipulating 3D Molecules in a Fixed-Dimensional E(3)-Equivariant Latent Space](manipulating_3d_molecules_in_a_fixed-dimensional_e3-equivariant_latent_space.md)
- [\[NeurIPS 2025\] Uncertainty-Aware Multi-Objective Reinforcement Learning-Guided Diffusion Models for 3D De Novo Molecular Design](uncertainty-aware_multi-objective_reinforcement_learning-guided_diffusion_models.md)
- [\[NeurIPS 2025\] GFlowNets for Learning Better Drug-Drug Interaction Representations](gflownets_for_learning_better_drug-drug_interaction_representations.md)
- [\[NeurIPS 2025\] Prior-Guided Flow Matching for Target-Aware Molecule Design with Learnable Atom Number](prior-guided_flow_matching_for_target-aware_molecule_design_with_learnable_atom_.md)

</div>

<!-- RELATED:END -->
