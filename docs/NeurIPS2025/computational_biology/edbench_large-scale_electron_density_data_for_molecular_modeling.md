---
title: >-
  [论文解读] EDBench: Large-Scale Electron Density Data for Molecular Modeling
description: >-
  [NeurIPS 2025][计算生物][电子密度] 构建了目前最大规模的电子密度（ED）数据集 EDBench（330 万分子，基于 B3LYP/6-31G DFT 计算），并设计了涵盖预测、检索、生成三类任务的 ED 基准评估体系，首次系统评估了深度学习模型对电子密度的理解和利用能力。 ：领域现状：机器学习力场（MLFF…
tags:
  - "NeurIPS 2025"
  - "计算生物"
  - "电子密度"
  - "分子力场"
  - "密度泛函理论"
  - "基准数据集"
  - "几何深度学习"
---

# EDBench: Large-Scale Electron Density Data for Molecular Modeling

**会议**: NeurIPS 2025  
**arXiv**: [2505.09262](https://arxiv.org/abs/2505.09262)  
**代码**: 有（见论文主页）  
**领域**: 计算生物  
**关键词**: 电子密度, 分子力场, 密度泛函理论, 基准数据集, 几何深度学习

## 一句话总结
构建了目前最大规模的电子密度（ED）数据集 EDBench（330 万分子，基于 B3LYP/6-31G** DFT 计算），并设计了涵盖预测、检索、生成三类任务的 ED 基准评估体系，首次系统评估了深度学习模型对电子密度的理解和利用能力。

## 研究背景与动机

**领域现状**：机器学习力场（MLFFs）已成为分子动力学模拟的重要工具，但主流方法聚焦于原子层级的多体交互建模（原子类型、坐标、距离、角度、扭转角等），对微观层面的电子分布关注不足。

**现有痛点**：根据 Hohenberg-Kohn 定理，电子密度 $\rho(\mathbf{r})$ 唯一确定多粒子系统的所有基态性质（能量、分子结构等），提供了比原子级表征更细粒度、更物理真实的分子描述。但 ED 的计算依赖耗时的 DFT，导致缺乏大规模 ED 数据集。

**核心矛盾**：现有 QC 数据集（QM7、QM9、MD17 等）主要提供能量和力数据，提供 ED 的数据集极少（MP 约 122K PBE 精度，ECD 约 140K），且多集中在材料领域。对药物类分子而言，缺乏大规模 ED 数据和配套基准。

**本文目标** (1) 构建大规模高质量的分子 ED 数据集；(2) 设计 ED 中心的基准任务体系，系统评估模型对电子信息的理解和利用能力。

**切入角度**：基于 PCQM4Mv2 数据集的 330 万药物类分子，使用 B3LYP 混合泛函（Jacob's ladder 高阶）+ Psi4 计算引擎，耗费 20.5 万核时（约 23.4 年单核计算），生成高质量 CUBE 文件格式的 ED 数据。

**核心 idea**：构建首个百万级分子电子密度数据集 + 设计预测/检索/生成三类基准任务，推动 MLFFs 从原子级向电子级建模演进。

## 方法详解

### 整体框架
EDBench 包含两部分：(1) 数据集——330 万分子的 ED 分布 + 量子化学性质（能量分量、轨道能量、多极矩等）；(2) 基准任务——6 个任务分三类：预测（4 个）、检索（1 个）、生成（1 个）。每个任务从全集中条件采样约 50K 分子，使用 scaffold split 进行 80/10/10 划分。

### 关键设计

1. **数据集构建**:

    - 功能：为 PCQM4Mv2 的 3,359,472 个分子生成高精度 ED
    - 核心思路：使用 Psi4 1.7 计算引擎，B3LYP 混合泛函，闭壳层系统用 RHF 参考，开壳层用 UHF 参考。基组根据元素组成选择：不含硫用 6-31G**，含硫用 6-31+G**（弥散函数更适合极化性强的重原子）。SCF 收敛后生成 CUBE 文件，网格间距 0.4 Bohr，padding 4.0 Bohr，密度分数阈值 0.85
    - 计算规模：8 × Intel Xeon Platinum 8270 (26 核×2 线程 = 416 逻辑核)，总计 20.5 万核时
    - ED 定义：$\rho(\mathbf{r}) = \rho_\alpha(\mathbf{r}) + \rho_\beta(\mathbf{r})$，由 Kohn-Sham 方程 $[-\frac{1}{2}\nabla^2 + V_{\text{eff}}(r)]\psi_i(r) = \epsilon_i \psi_i(r)$ 通过 SCF 迭代收敛得到

2. **预测任务（ED5-EC/OE/MM/OCS）**:

    - 功能：输入 ED 数据，预测各类量子化学性质
    - 核心思路：ED 编码器 $\text{Enc}_\mathcal{P}$ 提取 ED 特征，接任务特定预测头 $\text{Enc}_t$：$\hat{y}^\bullet = \text{Enc}_t^\bullet(\text{Enc}_\mathcal{P}(\mathcal{P}))$
    - 包括 4 个子任务：6 个能量分量（EC）、7 个轨道能量（OE）、4 个多极矩（MM）、开/闭壳层分类（OCS）
    - 采样策略：结构聚类 $C^s$（ECFP4+USR 指纹, k=100）× 标签聚类，均匀采样确保多样性

3. **检索任务（ED5-MER）**:

    - 功能：分子结构 ↔ 电子密度的双向跨模态检索
    - 核心思路：分别用分子编码器 $\text{Enc}_\mathcal{G}$ 和 ED 编码器 $\text{Enc}_\mathcal{P}$ 提取潜在表征 $h_\mathcal{G}, h_\mathcal{P}$，用 InfoNCE 损失训练对齐：
    $\mathcal{L}_{\text{ret}} = -\log \frac{\exp(\text{sim}(h_{\mathcal{G}_i}, h_{\mathcal{P}_i})/\tau)}{\sum_j \exp(\text{sim}(h_{\mathcal{G}_i}, h_{\mathcal{P}_j})/\tau)}$
    - 每个锚点配 10 个负样本（半数同簇易负、半数跨簇难负）

4. **生成任务（ED5-EDP）**:

    - 功能：从分子结构预测电子密度分布
    - 核心思路：构建异质图 $\mathcal{HG}$，包含原子节点和电子节点，用 k-NN (k=9) 建立原子-原子、原子-电子、电子-电子三类边。将 EGNN 扩展为异质图版 HGEGNN，掩码 ED 值后预测：
    $h^{\mathcal{HG}} = \text{HGEGNN}(\hat{\mathcal{HG}}), \quad \hat{\mathcal{D}} = \text{Enc}_t^{\text{EDP}}(h_\mathcal{P}^{\mathcal{HG}})$
    - 训练损失：$\mathcal{L}_{\text{gen}} = \|\hat{\mathcal{D}} - \mathcal{D}\|_2$

### 损失函数 / 训练策略
- 预测：回归任务用 L2 损失，分类任务用交叉熵
- 检索：InfoNCE，温度 $\tau = 0.07$
- 生成：L2 损失
- ED 阈值 $\rho_\tau$：过滤低密度区域（通常设 0.05-0.2），平衡精度与计算效率
- scaffold split 确保 OOD 评估

## 实验关键数据

### 预测任务（ED5-OE，轨道能量 MAE×100）

| 模型 | HOMO-2 | HOMO-1 | HOMO-0 | LUMO+0 | LUMO+1 | LUMO+2 |
|------|--------|--------|--------|--------|--------|--------|
| PointVector | 1.73 | 1.68 | 1.92 | 3.08 | 2.86 | 3.05 |
| X-3D | 1.75 | 1.72 | 1.98 | 3.21 | 3.02 | 3.25 |

### 生成任务（ED5-EDP，HGEGNN）

| 阈值 $\rho_\tau$ | MAE | Pearson (%) | Spearman (%) | 时间 (s/mol) | DFT 时间 |
|---------|------|------------|-------------|-------------|---------|
| 0.10 | 0.3362 | 81.0 | 56.4 | 0.024 | 245.8 |
| 0.15 | 0.0463 | 98.0 | 87.0 | 0.015 | 245.8 |
| 0.20 | 0.0448 | **99.2** | 91.0 | **0.013** | 245.8 |

### 关键发现
- X-3D 在能量分量、多极矩、开闭壳分类三个预测任务上优于 PointVector
- 轨道能量预测（OE）比能量分量预测（EC）容易得多——轨道能量具有更强的局部性，与局部 ED 模式直接关联
- **HGEGNN 生成 ED 比 DFT 快约 10,000 倍**（0.013 vs 245.8 秒/分子），且 Pearson 相关达 99.2%
- 惊人发现：用 HGEGNN 生成的 ED 在下游能量预测任务上甚至**优于** DFT 计算的 ED，可能因为模型生成的 ED 更平滑、更符合下游模型的归纳偏置
- 检索任务中 EquiformerV2 作为分子编码器的组合（E+P, E+X）显著优于 GeoFormer

## 亮点与洞察
- **规模化贡献**：330 万分子的 ED 数据集是同类最大，耗时 23.4 年单核计算，为社区提供了关键基础设施
- **从原子到电子的范式推进**：首次系统性地论证了 ED 作为 MLFFs 建模对象的可行性和价值，为"电子级力场"开辟了新方向
- **异质图构建的巧妙设计**：将原子和电子作为两种节点类型，用 k-NN 建立跨类型边，自然地将分子结构和电子分布耦合到统一框架
- **生成的 ED 反超 DFT**：这一非直觉发现暗示学习到的 ED 虽然物理精度可能不如 DFT，但其更平滑的模式更适合下游模型消费——对"是否一定要追求物理精确"提出了有趣的思考
- **全面的基准设计**：预测/检索/生成三类任务覆盖了 ED 理解的不同维度，scaffold split 确保 OOD 泛化评估

## 局限与展望
- 仅使用 B3LYP 泛函，精度仍有提升空间（更高阶泛函如 ωB97X-D）
- ED 的表征方式仅探索了点云，未尝试体素或图像表征
- 未考虑周期性系统（材料科学场景），当前仅限药物类分子
- 检索任务的负样本可能过于简单，高级对比学习策略（如 MoCo、hard negatives mining）可能进一步提升
- 数据集构建成本极高（20.5 万核时），限制了扩展到更高精度泛函

## 相关工作与启发
- **vs QM9/QM7**：经典 QC 数据集仅有 ~134K/7K 分子且不提供 ED，EDBench 在规模（330 万）和信息丰富度（ED + 多种量子性质）上显著超越
- **vs QMugs/∇²DFT**：提供密度矩阵而非直接 ED，EDBench 提供 CUBE 格式的空间 ED 分布，可直接用于几何深度学习
- **vs DeepDFT (Jorgensen)**：使用 VASP 生成的 ED 数据集规模很小，EDBench 覆盖百万级分子
- 对药物发现中的虚拟筛选、分子逆设计、量子感知建模有直接推动作用

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个百万级分子 ED 数据集 + 系统基准，"原子→电子"的方向推进有开创性
- 实验充分度: ⭐⭐⭐⭐⭐ 6 个基准任务、多种模型评估、消融分析（阈值/采样点/温度）、质量验证全面
- 写作质量: ⭐⭐⭐⭐ 背景知识铺垫充分（DFT/Kohn-Sham 入门友好），数据集对比表格清晰
- 价值: ⭐⭐⭐⭐⭐ 作为基础设施型工作，为 ED 驱动的分子建模奠定了数据和评估基础，长期影响大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Mol-LLaMA: Towards General Understanding of Molecules in Large Molecular Language Models](mol-llama_towards_general_understanding_of_molecules_in_large_molecular_language.md)
- [\[ICML 2026\] From Holo Pockets to Electron Density: GPT-style Drug Design with Density](../../ICML2026/computational_biology/from_holo_pockets_to_electron_density_gpt-style_drug_design_with_density.md)
- [\[NeurIPS 2025\] FGBench: A Dataset and Benchmark for Molecular Property Reasoning at Functional Group-Level in Large Language Models](fgbench_a_dataset_and_benchmark_for_molecular_property_reasoning_at_functional_g.md)
- [\[NeurIPS 2025\] Towards Unified and Lossless Latent Space for 3D Molecular Latent Diffusion Modeling](towards_unified_and_lossless_latent_space_for_3d_molecular_latent_diffusion_mode.md)
- [\[NeurIPS 2025\] Flow Density Control: Generative Optimization Beyond Entropy-Regularized Fine-Tuning](flow_density_control_generative_optimization_beyond_entropy-regularized_fine-tun.md)

</div>

<!-- RELATED:END -->
