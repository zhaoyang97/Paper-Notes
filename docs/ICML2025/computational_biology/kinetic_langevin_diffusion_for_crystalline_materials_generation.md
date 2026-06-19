---
title: >-
  [论文解读] Kinetic Langevin Diffusion for Crystalline Materials Generation
description: >-
  [ICML 2025][计算生物][晶体材料生成] KLDM 提出用 Kinetic Langevin Diffusion 处理晶体材料生成中原子分数坐标位于超环面的问题，通过引入辅助速度变量将扩散偏移到平坦欧几里得空间，同时保持周期平移对称性，在晶体结构预测和从头生成任务上达到竞争力性能。 领域现状：使用扩散模型进行晶体材…
tags:
  - "ICML 2025"
  - "计算生物"
  - "晶体材料生成"
  - "Langevin 扩散"
  - "超环面"
  - "等变性"
  - "结构预测"
---

# Kinetic Langevin Diffusion for Crystalline Materials Generation

**会议**: ICML 2025  
**arXiv**: [2507.03602](https://arxiv.org/abs/2507.03602)  
**代码**: 无  
**领域**: 计算生物  
**关键词**: 晶体材料生成, Langevin 扩散, 超环面, 等变性, 结构预测

## 一句话总结
KLDM 提出用 Kinetic Langevin Diffusion 处理晶体材料生成中原子分数坐标位于超环面的问题，通过引入辅助速度变量将扩散偏移到平坦欧几里得空间，同时保持周期平移对称性，在晶体结构预测和从头生成任务上达到竞争力性能。

## 研究背景与动机

**领域现状**：使用扩散模型进行晶体材料生成已成为材料科学中的重要方向。晶体结构由晶格参数、原子类型和原子在单元格内的分数坐标定义。

**现有痛点**：晶体数据的分布具有内在对称性，且涉及多模态（离散的原子类型、连续的坐标和晶格参数）。关键挑战在于分数坐标位于超环面 $\mathbb{T}^d$（具有周期边界条件），需要特殊处理。

**核心矛盾**：直接在超环面上做黎曼扩散虽然数学上正确，但训练目标复杂且难以处理晶体的周期平移对称性。而简单地忽略周期性在欧几里得空间做扩散会导致生成无效结构。

**本文目标**：设计一个既尊重超环面几何又能在平坦空间高效训练的扩散模型。

**切入角度**：借鉴物理学中的 Kinetic Langevin Dynamics（引入速度作为辅助变量），将环面上的扩散"提升"到切空间（欧几里得空间）。

**核心 idea**：用 Kinetic Langevin 方法将超环面上的扩散过程耦合到平坦的速度空间，使训练目标自然考虑周期对称性。

## 方法详解

### 整体框架

- **输入**：材料组成信息（原子类型）和可选的目标属性
- **扩散空间**：坐标在超环面+辅助速度在欧几里得空间的联合空间
- **逆过程**：学习联合逆扩散去噪，生成坐标和晶格参数
- **输出**：完整的晶体结构（晶格 + 原子位置 + 原子类型）

### 关键设计

1. **Trivialized Diffusion Model (TDM) 的推广**:

    - TDM 原本在流形上用切空间做扩散,  KLDM 将其推广以处理晶体特有的周期平移对称性
    - 分数坐标 $\mathbf{x} \in \mathbb{T}^{3N}$ 通过指数映射与切空间中的速度变量 $\mathbf{v} \in \mathbb{R}^{3N}$ 耦合
    - **设计动机**：超环面没有全局坐标系，TDM 提供了在切空间（平坦空间）做扩散的优雅方式

2. **周期平移对称性的处理**:

    - 晶体的真实数据分布满足周期平移不变性：将所有原子同时平移一个晶格向量不改变物理结构
    - KLDM 的训练目标自然地考虑了这种对称性
    - 通过速度变量的耦合，梯度信息正确地反映周期边界条件
    - **设计动机**：忽略此对称性会导致模型将物理上等价的结构视为不同，降低数据利用效率

3. **多模态联合生成**:

    - 同时处理连续变量（坐标、晶格参数）和离散变量（原子类型）
    - 坐标用 Kinetic Langevin 扩散，晶格参数用标准欧几里得扩散，原子类型用离散扩散
    - **设计动机**：晶体生成本质上是多模态生成问题

### 损失函数 / 训练策略

- Kinetic Langevin 版本的去噪得分匹配损失
- 位置和速度的联合去噪目标
- 晶格参数使用标准扩散损失
- 原子类型使用交叉熵损失

## 实验关键数据

### 主实验

| 任务 | 数据集 | KLDM | 之前 SOTA | 说明 |
|------|--------|------|-----------|------|
| 晶体结构预测 (CSP) | Perov-5 | 竞争力 | DiffCSP/FlowMM | Match Rate 指标 |
| CSP | MP-20 | 竞争力 | DiffCSP++ | Match Rate |
| 从头生成 (DNG) | Perov-5 | 竞争力 | CDVAE/DiffCSP | 有效性+多样性 |
| DNG | MP-20 | 竞争力 | 现有方法 | 稳定性 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 标准欧几里得扩散 | 下降 | 忽略周期性导致无效结构 |
| 直接黎曼扩散 | 可比但更复杂 | 训练目标更难优化 |
| 无速度耦合 | 下降 | 速度变量提供重要梯度信息 |
| 无对称性处理 | 下降 | 周期对称性对数据效率关键 |

### 关键发现

- Kinetic Langevin 方法有效地将环面扩散转化为平坦空间操作
- 周期对称性的正确处理对生成有效晶体结构至关重要
- KLDM 在 CSP 和 DNG 任务上均达到与当前最佳方法竞争的性能
- 方法具有良好的理论基础，不是纯经验性的

## 亮点与洞察

1. **物理启发**：从 Kinetic Langevin Dynamics 获得灵感，用速度变量解决几何扩散问题
2. **对称性感知**：训练目标自然编码了晶体的物理对称性
3. **理论严谨**：不是 heuristic 的解决方案，有明确的数学推导
4. **TDM 的有意义推广**：将 Trivialized Diffusion 扩展到处理群对称性

## 局限与展望

1. 引入速度变量增加了扩散空间的维度，可能影响采样效率
2. 目前仅在相对小的数据集（Perov-5, MP-20）上验证
3. 缺乏与最新 Flow Matching 方法的直接比较
4. 条件生成（给定目标属性）的能力未充分展示

## 相关工作与启发

- DiffCSP 系列是晶体扩散生成的主要基线
- FlowMM 用 Flow Matching 处理晶体生成
- 启发：Kinetic Langevin 方法可能在蛋白质等其他具有周期/对称结构的生成任务中也有价值

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ Kinetic Langevin 处理超环面扩散是新颖且优雅的想法
- 实验充分度: ⭐⭐⭐⭐ 覆盖两个标准任务，但数据集规模有限
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰
- 价值: ⭐⭐⭐⭐ 为流形上的扩散提供了新工具

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] CFP-Gen: Combinatorial Functional Protein Generation via Diffusion Language Models](cfp-gen_combinatorial_functional_protein_generation_via_diffusion_language_model.md)
- [\[ICML 2025\] PepTune: De Novo Generation of Therapeutic Peptides with Multi-Objective-Guided Discrete Diffusion](peptune_de_novo_generation_of_therapeutic_peptides_with_multi-objective-guided_d.md)
- [\[ICLR 2026\] Zatom-1: A Multimodal Flow Foundation Model for 3D Molecules and Materials](../../ICLR2026/computational_biology/zatom-1_a_multimodal_flow_foundation_model_for_3d_molecules_and_materials.md)
- [\[CVPR 2025\] DiffVsgg: Diffusion-Driven Online Video Scene Graph Generation](../../CVPR2025/computational_biology/diffvsgg_diffusion-driven_online_video_scene_graph_generation.md)
- [\[CVPR 2025\] Multimodal Protein Language Models for Enzyme Kinetic Parameters: From Substrate Recognition to Conformational Adaptation](../../CVPR2025/computational_biology/multimodal_protein_language_models_for_enzyme_kinetic_parameters_from_substrate_.md)

</div>

<!-- RELATED:END -->
