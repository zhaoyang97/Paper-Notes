---
title: >-
  [论文解读] Unified All-Atom Molecule Generation with Neural Fields
description: >-
  [NeurIPS 2025][计算生物][神经场] 提出 FuncBind 框架，利用神经场（Neural Fields）将分子表示为连续原子密度函数，构建统一的条件生成模型，能够同时处理小分子、大环肽和抗体 CDR 环三种药物模态的靶标条件生成。 基于结构的药物设计（SBDD）是药物发现的核心任务，目标是根据靶标蛋白的 3…
tags:
  - "NeurIPS 2025"
  - "计算生物"
  - "神经场"
  - "分子生成"
  - "全原子表示"
  - "基于结构的药物设计"
  - "扩散模型"
---

# Unified All-Atom Molecule Generation with Neural Fields

**会议**: NeurIPS 2025  
**arXiv**: [2511.15906](https://arxiv.org/abs/2511.15906)  
**代码**: [GitHub](https://github.com/prescient-design/funcbind)  
**领域**: 医学图像 / 药物设计  
**关键词**: 神经场, 分子生成, 全原子表示, 基于结构的药物设计, 扩散模型

## 一句话总结

提出 FuncBind 框架，利用神经场（Neural Fields）将分子表示为连续原子密度函数，构建统一的条件生成模型，能够同时处理小分子、大环肽和抗体 CDR 环三种药物模态的靶标条件生成。

## 研究背景与动机

基于结构的药物设计（SBDD）是药物发现的核心任务，目标是根据靶标蛋白的 3D 结构生成具有高亲和力的候选分子。当前的生成模型通常专注于单一分子模态：小分子模型使用点云或体素表示，蛋白质模型利用残基级别的点云并依赖序列数据库。这种模态特定的设计限制了模型的泛化能力——不同模态之间无法迁移，而实际药物发现中常涉及多模态的分子界面设计。

作者认为，模态无关的表示更适合跨模态学习，能够从更多样的训练数据中学习物理性质。已有工作如 AlphaFold3、RoseTTAFold 等在结构预测中的成功证明了跨模态统一建模的可行性。因此，本文提出用神经场作为分子的统一表示——将分子建模为从3D坐标到原子密度的连续函数——从而在一个模型中同时训练三种药物模态。

## 方法详解

### 整体框架

FuncBind 是一个两阶段的潜在空间条件生成模型：
1. **阶段一**：训练神经场自编码器（VAE），学习分子的潜在表示
2. **阶段二**：在潜在空间训练条件去噪器，用于生成新分子

生成时，输入靶标蛋白结构，采样噪声并通过去噪器迭代生成潜在向量，再通过解码器恢复原子密度场，最后经后处理得到分子结构。

### 关键设计

1. **空间特征图神经场表示**：不同于之前 FuncMol 使用全局嵌入，FuncBind 将潜在变量 $z$ 组织为空间特征图（$C \times L^3$），每个空间位置的特征捕获局部信息。编码器 $E_\psi$ 是 3D CNN，将分子的低分辨率体素映射到特征图空间。解码器 $D_\phi$ 基于 Gabor 滤波器的乘法滤波网络，利用最近邻插值从特征图中提取位置相关嵌入 $z_x$，解码原子密度场：$D_\phi(x, z_x) \to \mathbb{R}^n$。这种设计既能捕获局部原子细节，又兼容 U-Net 等表达力强的去噪架构。VAE 训练损失为重建损失加 KL 正则化：

$$\mathcal{L}_{AE} = \sum_{v \in \mathcal{D}} \mathbb{E}_{z \sim q_\psi(z|v)} \left[\int \|D_\phi(x,z) - v(x)\|_2^2 dx\right] + \beta \text{KL}(q_\psi(z|v) \| \mathcal{N}(0,I_d))$$

2. **条件去噪器**：去噪器接收噪声潜在向量 $y = z + \sigma\varepsilon$，以靶标结构 $z^{tar}$、分子模态 $c$ 和噪声水平 $\sigma$ 为条件。靶标编码器 $E_{\psi'}$ 与分子编码器架构相似但参数独立。核心网络 $U_\theta$ 采用 3D U-Net，借鉴 Karras 的预处理方案：

$$\hat{z}_\theta(y|z^{tar}, \sigma, c) = \frac{1}{\sigma^2+1}y + \frac{\sigma}{\sqrt{\sigma^2+1}} U_\theta\left(\frac{y}{\sqrt{\sigma^2+1}}, z^{tar}, \frac{1}{4}\log\sigma, c\right)$$

关键决策是不使用 SE(3) 等变性约束，而是用数据增强（旋转和平移）替代。

3. **采样策略**：支持两种基于 score 的采样方法——扩散模型（通过反向 SDE 积分）和 Walk-Jump Sampling（单噪声水平采样，训练更简单、混合更快）。两者都利用 Tweedie-Miyasawa 公式将去噪器与条件 score 函数关联。

### 损失函数 / 训练策略

- **自编码器**：重建损失 + KL 散度，训练时对原子中心附近的坐标上采样集中训练
- **去噪器**：在多噪声水平上 MSE 去噪损失，采用 Karras 的自适应重加权方案
- **后处理**：从生成的潜在码渲染出 0.25Å 分辨率体素，峰值检测+梯度上升定位原子坐标，OpenBabel 推断化学键和残基身份
- 模型规模达 **5B 参数**，在三种模态上联合训练

## 实验关键数据

### 主实验：小分子生成（CrossDocked2020）

| 指标 | FuncBind | VoxBind（SOTA）| MolCraft | TargetDiff |
|------|----------|----------------|----------|------------|
| VinaScore ↓ | -5.71 | **-6.94** | -6.59 | -5.47 |
| VinaDock ↓ | -7.26 | **-8.30** | -7.92 | -7.80 |
| QED ↑ | 0.50 | **0.57** | 0.50 | 0.48 |
| SA ↑ | 0.65 | 0.70 | **0.69** | 0.58 |
| Diversity ↑ | **0.70** | 0.73 | 0.72 | 0.72 |
| Strain Energy ↓ | 217 | **162** | **195** | 1243 |
| # Atoms | 19.0 | 23.4 | 22.7 | 24.2 |

### 主实验：抗体 CDR 环重设计（SAbDab）

| 方法 | H3-AAR ↑ | H3-RMSD ↓ | H1-AAR ↑ | H1-RMSD ↓ |
|------|----------|-----------|----------|-----------|
| FuncBind | **47.5%** | **2.04 Å** | **86.9%** | **0.41 Å** |
| AbDiffuser | 34.1% | 3.35 Å | 76.3% | 1.58 Å |
| DiffAb† | 26.8% | 3.60 Å | 65.8% | 1.19 Å |
| RAbD† | 22.1% | 2.90 Å | 22.9% | 2.26 Å |

### 主实验：大环肽生成

| 方法 | TS ↑ | L-RMSD ↓ | I-RMSD ↓ | TM-Score ↑ | Vina dock ↑ |
|------|------|----------|----------|------------|-------------|
| FuncBind | 0.33 | **2.6 Å** | **1.8 Å** | **0.36** | **41%** |
| AfCycDesign | **0.34** | 7.6 Å | 3.7 Å | 0.33 | 29% |
| RFPeptide | 0.31 | 12 Å | 3.3 Å | 0.33 | 8.8% |

### 消融实验

| 配置 | 说明 |
|------|------|
| 统一模型 vs 单模态模型 | 性能相当，但统一模型的**唯一性（uniqueness）显著更高** |
| 扩散 vs Walk-Jump Sampling | 扩散在多数指标上表现更优；WJS 训练更简单、混合更快 |
| 全局嵌入 vs 空间特征图 | 空间特征图能更好扩展到大分子，兼容 U-Net 架构 |

### 关键发现

- FuncBind 在 CDR 环重设计上的 AAR 和 RMSD 指标超越所有基线 1.5-3 倍
- 湿实验验证：H3 环重设计在刚性表位上达到 **45% 结合率**
- 能生成新颖且化学合理的非标准氨基酸（不到 1% 为不合理结构）
- 大环肽基准中生成 41% 的样本 Vina 对接得分优于种子分子

## 亮点与洞察

- **统一表示的威力**：用神经场统一三种差异巨大的分子模态是一个优雅的设计，空间特征图的引入使其既能捕获局部精细结构，又兼容成熟的视觉网络架构
- **用数据增强替代等变性**：放弃 SE(3) 等变约束，转而使用旋转/平移增强，简化了架构设计且不影响性能
- **湿实验闭环**：在 CDR 设计上完成了从生成到实验验证的完整闭环，45% 的结合率具有实际转化价值
- **新基准贡献**：引入 ~190K 合成大环肽/蛋白复合物数据集

## 局限与展望

- 小分子指标上仍略弱于 VoxBind/MolCraft 等专用模型
- 依赖准确的分子界面 3D 结构，而此类结构在实际药物发现中获取成本高
- 未处理可合成性（小分子）和可开发性（抗体）等实际应用约束
- 5B 参数的训练和推理成本极高，部署门槛高
- 跨模态迁移学习尚未深入探索

## 相关工作与启发

- **FuncMol**: 本文直接扩展，引入空间特征图和条件生成
- **VoxBind**: 体素方法的 SOTA，FuncBind 可视为其连续版本
- **AlphaFold3/RoseTTAFold**: 跨模态结构预测的成功先例
- 启发：神经场表示在分子科学中的应用潜力巨大，未来可扩展到更多生物体系

## 评分

- **新颖性**: ⭐⭐⭐⭐ 统一三种分子模态的神经场方法新颖，但核心技术（VAE+扩散）较成熟
- **实验充分度**: ⭐⭐⭐⭐⭐ 三模态全面评估 + 湿实验验证 + 新数据集贡献
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，技术描述完整
- **价值**: ⭐⭐⭐⭐⭐ 统一分子生成框架具有重要实际意义，湿实验结果增强说服力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Prior-Guided Flow Matching for Target-Aware Molecule Design with Learnable Atom Number](prior-guided_flow_matching_for_target-aware_molecule_design_with_learnable_atom_.md)
- [\[ICML 2025\] Geometric Representation Condition Improves Equivariant Molecule Generation](../../ICML2025/computational_biology/geometric_representation_condition_improves_equivariant_molecule_generation.md)
- [\[ICML 2025\] Scalable Non-Equivariant 3D Molecule Generation via Rotational Alignment](../../ICML2025/computational_biology/scalable_non-equivariant_3d_molecule_generation_via_rotational_alignment.md)
- [\[NeurIPS 2025\] Towards Unified and Lossless Latent Space for 3D Molecular Latent Diffusion Modeling](towards_unified_and_lossless_latent_space_for_3d_molecular_latent_diffusion_mode.md)
- [\[ICLR 2026\] BioMD: All-atom Generative Model for Biomolecular Dynamics Simulation](../../ICLR2026/computational_biology/biomd_all-atom_generative_model_for_biomolecular_dynamics_simulation.md)

</div>

<!-- RELATED:END -->
