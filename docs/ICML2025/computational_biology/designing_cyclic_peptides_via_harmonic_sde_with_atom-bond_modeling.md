---
title: >-
  [论文解读] Designing Cyclic Peptides via Harmonic SDE with Atom-Bond Modeling
description: >-
  [ICML2025][计算生物][环肽设计] 提出 CpSDE 框架，通过谐波 SDE 生成模型 (AtomSDE) 和残基类型预测器 (ResRouter) 的交替采样，首次实现基于 3D 受体结构的全类型环肽设计，在稳定性和亲和力上超越现有线性肽设计方法。 - 线性肽的局限：传统线性肽药物半衰期短、稳定性差、易被水解酶降…
tags:
  - "ICML2025"
  - "计算生物"
  - "环肽设计"
  - "谐波SDE"
  - "全原子建模"
  - "化学键建模"
  - "扩散模型"
  - "药物发现"
---

# Designing Cyclic Peptides via Harmonic SDE with Atom-Bond Modeling

**会议**: ICML2025  
**arXiv**: [2505.21452](https://arxiv.org/abs/2505.21452)  
**代码**: 待确认  
**领域**: 分子设计  
**关键词**: 环肽设计, 谐波SDE, 全原子建模, 化学键建模, 扩散模型, 药物发现

## 一句话总结

提出 CpSDE 框架，通过谐波 SDE 生成模型 (AtomSDE) 和残基类型预测器 (ResRouter) 的交替采样，首次实现基于 3D 受体结构的全类型环肽设计，在稳定性和亲和力上超越现有线性肽设计方法。

## 研究背景与动机

- **线性肽的局限**：传统线性肽药物半衰期短、稳定性差、易被水解酶降解，限制了其治疗潜力
- **环肽的优势**：环肽通过残基首尾或侧链间形成闭环，增强酶解稳定性并以更稳定构象高亲和力结合蛋白表面
- **环肽类型多样**：根据成环原子分为四类——头尾环化 (head-to-tail)、侧链-尾 (side-to-tail)、头-侧链 (head-to-side)、侧链-侧链 (side-to-side)
- **现有方法局限**：已有方法仅支持单一环肽类型（如二硫键环肽或头尾环肽），无法统一处理不同环化约束；蛋白-环肽复合物的 3D 结构数据极度匮乏
- **核心挑战**：需在数据稀缺条件下，同时解决环化几何约束、非标准氨基酸建模以及序列-结构联合生成问题

## 方法详解

### 整体框架

CpSDE 由两个核心模型和一个采样算法组成：

1. **AtomSDE**：基于谐波 SDE 的生成式结构预测模型，学习配体原子坐标的条件分布
2. **ResRouter**：残基类型预测器，基于去噪后的结构预测氨基酸类型
3. **Routed Sampling**：交替调用上述两个模型，迭代更新序列和结构

### AtomSDE——谐波 SDE 结构生成

选用 Variance Preserving (VP) SDE（而非 VE SDE，后者会使含噪配体远离受体丢失有效交互）。引入与化学图相关的谐波 SDE：

$$
\mathrm{d}\mathbf{x}^L = -\frac{1}{2}\beta(t)\tilde{\mathbf{x}}^L \mathrm{d}t + \sqrt{\beta(t)} \mathbf{\Lambda}^{1/2} \mathbf{P}^\top \mathrm{d}\mathbf{w}
$$

其中 $\mathbf{H} = \mathbf{L} + \sigma_P^{-2}\mathbf{I}$ 为结合化学图拉普拉斯矩阵 $\mathbf{L}$ 与受体相关标量的正定矩阵，$\mathbf{H} = \mathbf{P}\mathbf{\Lambda}\mathbf{P}^\top$ 为特征分解。该各向异性扰动过程利用化学图连接信息，使成键原子初始位置靠近并以相关噪声扰动。

扰动核具有解析形式：

$$
p_{0t}(\mathbf{x}^L | \mathbf{x}_0^L) = \mathcal{N}\left(\mathbf{x}_t^L;\; \mathbf{x}_0^L e^{-\frac{1}{2}\int_0^t \beta(s)\mathrm{d}s},\; \mathbf{H} - \mathbf{H}e^{-\int_0^t \beta(s)\mathrm{d}s}\right)
$$

**训练损失**——简单重构损失（近似等价于 score matching）：

$$
\mathcal{L}_{\text{AtomSDE}} = \mathbb{E}_{t, p_0(\mathbf{x}_0^L), p_{0t}(\mathbf{x}_t^L|\mathbf{x}_0^L)} \left[\| \mathbf{D}_\theta(\mathbf{x}_t^L, t) - \mathbf{x}_0^L \|^2 \right]
$$

模型基于 SE(3)-等变神经网络，同时编码蛋白-配体 KNN 图和配体化学图。

### ResRouter——残基类型预测

解决序列-结构"鸡与蛋"问题：给定含噪结构预测残基类型。输入时移除标准氨基酸侧链（防止模型走捷径），聚合骨架原子 (N-Cα-C-O) 隐状态后用 MLP 预测氨基酸类型：

$$
\mathcal{L}_{\text{ResRouter}} = \sum_{i}^{N} -\log p_\phi(a_i | \mathbf{D}_\theta(\mathbf{x}_t^L, t), \mathcal{G}_C, \mathcal{T}, t)
$$

AtomSDE 预训练后固定参数，再训练 ResRouter。

### Routed Sampling 采样策略

- 原子分为两类：**环化约束原子**（骨架原子+环化相关原子，化学图已知）和**自由残基原子**（非环化标准残基侧链，化学图未知）
- 自由残基部分维护 Atom73 状态（"叠加态"），所有可能氨基酸共享骨架和 Cβ、各自拥有独立侧链原子
- 每步 reverse SDE：AtomSDE 去噪坐标 → ResRouter 预测残基类型 → 根据预测类型折叠到具体原子状态 → 更新化学图 → 下一步
- 对齐所有原子到相同噪声水平，避免因侧链原子间歇采样导致更新不足

## 实验关键数据

### 数据集

- 小分子数据集：来自 PDBBind，14,348 个蛋白-配体复合物
- 肽数据集：来自 RCSB PDB / Propedia / PepBDB，20,033 个复合物（配体 < 30 残基）
- 按受体序列同源性 0.3 聚类划分训练/验证集

### 主实验结果

| 方法 | 共设计 | 肽类型 | 稳定性 Avg↓ | 稳定性 Med↓ | 亲和力 Avg↓ | 亲和力 Med↓ | 多样性↑ |
|------|--------|--------|------------|------------|------------|------------|---------|
| Reference | N/A | 线性 | -672.53 | -634.71 | -85.03 | -78.70 | N/A |
| RFDiffusion | ✗ | 线性 | **-633.51** | **-607.82** | **-70.30** | **-61.35** | 0.55 |
| ProteinGenerator | ✓ | 线性 | -576.39 | -554.70 | -46.98 | -40.39 | 0.58 |
| PepFlow | ✓ | 线性 | -576.16 | -498.31 | -47.88 | -42.40 | 0.70 |
| PepGLAD | ✓ | 线性 | -359.44 | -310.33 | -45.06 | -38.56 | 0.79 |
| **CpSDE (Mix)** | ✓ | 混合环肽 | **-580.67** | **-527.80** | **-55.71** | **-48.42** | **0.79** |

- 在所有共设计方法中，CpSDE 在稳定性和亲和力上均最优，多样性也最高
- Head-to-tail 和 head-to-side 环肽优于其他类型，可能因训练数据中 C-N 键多于 S-S/C-S 键
- RFDiffusion 能量最优但多样性低（倾向生成 α-螺旋）

### 案例研究——分子动力学验证

**SMYD2 抑制剂（head-to-tail 环化）**：

- 生成 8 个环肽，H2T-6 Rosetta 亲和力最优 (-33.9 kcal/mol)
- 100 ns MD 模拟：H2T-6 RMSD = 3.05 Å（PepFlow 线性肽 4.59 Å）
- MM-PBSA 结合自由能：H2T-6 = **-24.02** kcal/mol，真实线性肽 -19.00，PepFlow -7.26

**SET8 抑制剂（side-to-side 环化）**：

- S2S-4 RMSD = 2.54 Å（真实 4.06 Å，PepFlow 5.23 Å）
- 结合自由能：S2S-4 = **-12.48** kcal/mol，真实 -6.39，PepFlow -9.26

## 亮点与洞察

1. **首个全类型环肽生成框架**：统一处理四种环化类型，无需针对特定类型修改模型
2. **全原子+化学键建模**：避免残基级表征的局限，天然支持非标准氨基酸和环化键约束
3. **谐波 SDE 巧妙利用化学图**：各向异性噪声扰动保持成键原子空间相关性，采样质量更高
4. **数据效率高**：小分子+线性肽数据联合训练 AtomSDE，极大缓解环肽 3D 数据稀缺问题
5. **MD 验证充分**：两个实际药物靶点的案例研究中，设计的环肽在稳定性和亲和力上均超越真实线性肽

## 局限与展望

- 当前仅用 Rosetta 能量和短时 MD 模拟评估，缺乏湿实验验证
- 训练数据中某些环化类型（如 S-S、C-S 键）样本少，对应环肽设计效果相对较弱
- 不支持含完全非天然构件（如 D-氨基酸、β-氨基酸）的环化设计
- Atom73 表征引入的"叠加态"增加内存和计算开销
- 未与 AlphaFold3 等最新结构预测方法对比

## 相关工作与启发

- **Protpardelle** (Chu et al., 2024)：Atom73 叠加态表征的灵感来源
- **RFDiffusion / ProteinGenerator**：蛋白质骨架设计的扩散模型
- **PepFlow / PepGLAD**：线性肽全原子设计基准
- **EigenFold** (Jing et al., 2023)：谐波先验在蛋白质构象采样中的应用

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 首个统一全类型环肽生成框架，谐波 SDE + Routed Sampling 思路新颖
- 实验充分度: ⭐⭐⭐⭐ — 与多个基准对比，并有 MD 案例验证；但缺乏湿实验
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，环肽分类图示直观，公式推导完整
- 价值: ⭐⭐⭐⭐⭐ — 对肽类药物发现具有显著推动作用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Generative Modeling of Full-Atom Protein Conformations using Latent Diffusion on Graph Embeddings](../../NeurIPS2025/computational_biology/generative_modeling_of_full-atom_protein_conformations_using_latent_diffusion_on.md)
- [\[ICML 2025\] Geometric Generative Modeling with Noise-Conditioned Graph Networks](geometric_generative_modeling_with_noise-conditioned_graph_networks.md)
- [\[ICML 2025\] PepTune: De Novo Generation of Therapeutic Peptides with Multi-Objective-Guided Discrete Diffusion](peptune_de_novo_generation_of_therapeutic_peptides_with_multi-objective-guided_d.md)
- [\[NeurIPS 2025\] Unified All-Atom Molecule Generation with Neural Fields](../../NeurIPS2025/computational_biology/unified_all-atom_molecule_generation_with_neural_fields.md)
- [\[ICML 2026\] SwitchCraft: A Programmatic Framework for Designing State-Switching Proteins](../../ICML2026/computational_biology/switchcraft_a_programmatic_framework_for_designing_state-switching_proteins.md)

</div>

<!-- RELATED:END -->
