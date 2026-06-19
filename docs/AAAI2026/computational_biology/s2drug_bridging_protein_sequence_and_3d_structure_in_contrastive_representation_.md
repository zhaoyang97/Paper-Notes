---
title: >-
  [论文解读] S2Drug: Bridging Protein Sequence and 3D Structure in Contrastive Representation Learning for Virtual Screening
description: >-
  [AAAI 2026][计算生物][虚拟筛选] 提出 S2Drug，一个两阶段对比学习框架，第一阶段在 ChemBL 大规模数据上用蛋白质序列-配体对比预训练（含双边数据采样策略降噪去冗），第二阶段在 PDBBind 上通过残基级门控模块融合序列与 3D 结构信息并引入结合位点预测辅助任务，在 DUD-E 和 LIT-PCBA 虚拟筛选基准上大幅超越现有方法。
tags:
  - "AAAI 2026"
  - "计算生物"
  - "虚拟筛选"
  - "蛋白质-配体交互"
  - "对比学习"
  - "蛋白质序列"
  - "3D结构"
  - "结合位点预测"
  - "药物发现"
---

# S2Drug: Bridging Protein Sequence and 3D Structure in Contrastive Representation Learning for Virtual Screening

**会议**: AAAI 2026  
**arXiv**: [2511.07006](https://arxiv.org/abs/2511.07006)  
**代码**: 有（附录中提供）  
**领域**: 医学图像 / 药物发现 / 虚拟筛选  
**关键词**: 虚拟筛选, 蛋白质-配体交互, 对比学习, 蛋白质序列, 3D结构, 结合位点预测, 药物发现

## 一句话总结
提出 S2Drug，一个两阶段对比学习框架，第一阶段在 ChemBL 大规模数据上用蛋白质序列-配体对比预训练（含双边数据采样策略降噪去冗），第二阶段在 PDBBind 上通过残基级门控模块融合序列与 3D 结构信息并引入结合位点预测辅助任务，在 DUD-E 和 LIT-PCBA 虚拟筛选基准上大幅超越现有方法。

## 研究背景与动机

**领域现状**：虚拟筛选（VS）是药物发现的核心步骤，目标是从海量化合物库中找到与靶蛋白口袋结合的小分子。现有方法分为分子对接（如 AutoDock Vina，精确但慢）和深度学习方法（如 DrugCLIP/DrugHash，用对比学习对齐蛋白-配体表示）。

**现有方法的关键盲区 — 蛋白质序列被忽视**：
   - 几乎所有主流方法都仅依赖 3D 结构信息
   - 单构象原子级结构模型对输入扰动敏感，且难以应对口袋构象柔性
   - 获取蛋白质 3D 结构（X-ray、Cryo-EM）成本高耗时，限制了大规模训练数据的扩展
   - 而蛋白质序列数据广泛可得，且"序列决定结构，结构决定功能"是蛋白质研究的基本原则

**直接使用序列的挑战**：
   - 大规模蛋白质-配体数据集（如 ChemBL，745K 条目）存在严重的**冗余和噪声**
   - 蛋白质侧：同源冗余、功能异构体重复
   - 配体侧：亲和力测量变异性、频繁命中的非特异性化合物
   - 仅用序列而丢弃结构上下文会失去关键的空间交互信息

**切入角度**：两阶段学习——序列预训练解决数据规模与泛化问题，结构融合微调解决空间精度问题。

## 方法详解

### 整体框架：两阶段对比学习

**Stage 1: 序列预训练**（ChemBL，大规模）→ **Stage 2: 序列-结构融合微调**（PDBBind，小规模高质量）

### Stage 1: 序列模型预训练

#### 双边数据采样策略（Bilateral Data Sampling）

从 ChemBL 的 745K 条目中清洗出高质量子集：

**蛋白质侧冗余削减**：
1. 同源感知降权：MMseqs2 在 40% 序列一致性阈值聚类，大族中蛋白质的采样概率降低：$\Pr(P_n) = |C_m^{hom}|^{-\alpha}$，$\alpha=0.5$
2. 功能去重：基于 UniProt/GO 注释，每个功能组仅保留配体多样性最大的代表蛋白

**配体侧噪声缓解**：
1. 亲和力变异过滤：同一蛋白-配体对在不同实验条件下的亲和力值标准差 $\sigma_n > 1.0$ 的去除
2. 频繁命中去除：与超过 20 个蛋白结合的配体视为非特异性化合物移除，PAINS 反应性亚结构也过滤

#### 表示学习

- **序列编码器**：ESM2-650M，输入氨基酸序列，Mean Pooling 得到蛋白质嵌入
- **配体结构编码器**：Uni-Mol，输入 3D 构象（原子坐标+类型），Mean Pooling 得到配体嵌入
- 两个 MLP 投影到共享空间，用对称 InfoNCE 对比损失训练

### Stage 2: 序列-结构融合微调

#### 残基级门控融合模块

对每个口袋残基 $r_i$，从序列编码器和结构编码器分别获得表示 $x_{n,i}^s$ 和 $x_{n,i}^g$，通过自适应门控融合：

$$\beta_{n,i} = \sigma(W_\beta^\top [W_s x_{n,i}^s; W_g x_{n,i}^g] + b_\beta)$$
$$x_{n,i}^f = \beta_{n,i} \cdot W_s x_{n,i}^s + (1-\beta_{n,i}) \cdot W_g x_{n,i}^g$$

门控权重 $\beta$ 是学习得到的，允许模型对每个残基动态选择更有信息量的模态。融合后经两层 Transformer 再 Mean Pooling 得到最终口袋表示。

#### 结合位点预测辅助任务

核心思想：口袋是散布在一级序列上、在 3D 空间中聚集形成结合腔的残基集合。预测结合位点帮助模型理解蛋白质 3D 折叠（尤其是口袋区域）。

- 采样 $K$ 个配体探针，计算其与每个残基的注意力相关性
- 仅使用序列表示 $x_{n,i}^s$（避免信息泄漏），用 BCE 损失训练

#### 总损失

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{fc}} + \lambda \cdot \mathcal{L}_{\text{bsp}}$$

其中 $\mathcal{L}_{\text{fc}}$ 为融合表示的对比损失，$\mathcal{L}_{\text{bsp}}$ 为结合位点预测损失。

## 实验

### 主实验：虚拟筛选性能

**DUD-E 数据集**（大规模，含 decoy）：

| 方法 | AUROC | BEDROC | EF 0.5% | EF 1% | EF 5% |
|------|-------|--------|---------|-------|-------|
| Glide-SP | 76.70 | 40.70 | 19.39 | 16.18 | 7.23 |
| DrugCLIP | 79.45 | 47.82 | 37.86 | 30.76 | 10.10 |
| DrugHash | 83.73 | 57.16 | 43.03 | 37.18 | 12.07 |
| **S2Drug** | **92.46** | **79.25** | **58.37** | **43.06** | **18.82** |

S2Drug 在 AUROC 上超过 DrugHash 8.73 个点、超过 DrugCLIP 13.01 个点。

**LIT-PCBA 数据集**（更真实的筛选场景）：

| 方法 | AUROC | BEDROC | EF 0.5% | EF 1% |
|------|-------|--------|---------|-------|
| DrugCLIP | 56.36 | 6.78 | 7.77 | 5.66 |
| DrugHash | 54.58 | 7.14 | 9.65 | 6.14 |
| **S2Drug** | **58.23** | **8.69** | **11.44** | **7.38** |

### 同源排除实验（泛化性评估）

在不同序列一致性阈值（90%/60%/30%/HMM）下排除训练-测试重叠：
- S2Drug 在所有阈值下均大幅优于 DrugCLIP
- 即使在 90% 和 60% 阈值下，S2Drug 仍优于 DrugHash/DrugCLIP 在无排除设置下的性能
- 证明双边数据采样有效减少了过拟合和冗余依赖

### 消融实验

| 变体 | DUD-E AUROC | LIT-PCBA AUROC |
|------|-------------|----------------|
| - BDS（去双边采样）| 88.73 | 56.12 |
| - SSF（去序列-结构融合）| 87.92 | 55.03 |
| - BSP（去结合位点预测）| 89.58 | 56.47 |
| **S2Drug** | **92.46** | **58.23** |

- 移除序列-结构融合 (SSF) 影响最大：AUROC 下降 4.54/3.20
- 双边数据采样 (BDS) 的贡献也很显著：3.73/2.11
- 结合位点预测辅助任务 (BSP) 提供约 2.88/1.76 的提升

### 结合位点预测

在 HOLO4K、COACH420、ASD 三个基准上，S2Drug 也展现了有竞争力的结合位点预测性能，验证了辅助任务的有效性。

## 亮点与洞察

1. **"序列决定结构，结构决定功能"原则的实践化**：首次系统性将蛋白质序列引入虚拟筛选的对比表示学习
2. **双边数据采样策略精妙**：同源降权+功能去重+亲和力过滤+频繁命中去除，四管齐下解决 ChemBL 的数据质量问题
3. **残基级门控融合**：允许每个残基动态选择更依赖序列还是结构信息，比简单拼接或加权求和更灵活
4. **辅助任务设计有生物学洞察**：结合位点是序列上散布但空间聚集的残基，预测它帮助模型理解 3D 折叠
5. **在严格同源排除下仍显著领先**，表明模型真正学到了蛋白质-配体交互的底层规律而非记忆同源模式

## 局限性

1. 两阶段训练较复杂，预训练阶段在 ChemBL 上需 8×A6000 GPU
2. 第二阶段微调数据集 PDBBind 规模有限（~19K 条目），可能限制了融合模块的学习
3. 结合位点预测辅助任务需要真实结合位点标注，对无标注蛋白不适用
4. LIT-PCBA 上的 AUROC 绝对值仍偏低（58.23），说明更真实的筛选场景仍极具挑战
5. 未讨论计算成本（ESM2-650M + Uni-Mol 的推理延迟对大规模筛选的影响）

## 相关工作

- **虚拟筛选**：分子对接（Glide-SP/Vina）、回归（DeepDTA/Planet）、分类（OnionNet-2）、检索（DrugCLIP/DrugHash）
- **蛋白质表示学习**：ESM2（序列）、UniMol（结构），近期趋势是序列+结构协同（SaProt, ESMFold）
- **对比学习在药物发现**：DrugCLIP 首创检索范式，DrugHash 引入哈希加速

## 评分 ⭐⭐⭐⭐

- **创新性**：⭐⭐⭐⭐ — 两阶段学习 + 双边数据采样 + 残基门控融合的组合新颖
- **实验**：⭐⭐⭐⭐⭐ — DUD-E + LIT-PCBA + 同源排除 + 结合位点预测，全面且严格
- **写作**：⭐⭐⭐⭐ — 方法描述清楚，数学符号规范
- **实用性**：⭐⭐⭐⭐ — 对药物发现有直接应用价值，代码开源

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] AANet: Virtual Screening under Structural Uncertainty via Alignment and Aggregation](../../NeurIPS2025/computational_biology/aanet_virtual_screening_under_structural_uncertainty_via_alignment_and_aggregati.md)
- [\[ICML 2026\] Learning Protein Structure-Function Relationships through Knowledge-guided Representation Decomposition](../../ICML2026/computational_biology/learning_protein_structure-function_relationships_through_knowledge-guided_repre.md)
- [\[ICML 2026\] SIGMA: Structure-Invariant Generative Molecular Alignment for Chemical Language Models via Autoregressive Contrastive Learning](../../ICML2026/computational_biology/sigma_structure-invariant_generative_molecular_alignment_for_chemical_language_m.md)
- [\[ICML 2026\] Advancing Ligand-based Virtual Screening and Molecular Generation with Pretrained Molecular Embedding Distance](../../ICML2026/computational_biology/advancing_ligand-based_virtual_screening_and_molecular_generation_with_pretraine.md)
- [\[AAAI 2026\] Dual-Path Knowledge-Augmented Contrastive Alignment Network for Spatially Resolved Transcriptomics](dual-path_knowledge-augmented_contrastive_alignment_network_for_spatially_resolv.md)

</div>

<!-- RELATED:END -->
