---
title: >-
  [论文解读] Uncertainty-Aware Multi-Objective Reinforcement Learning-Guided Diffusion Models for 3D De Novo Molecular Design
description: >-
  [NeurIPS 2025][计算生物][扩散模型] 提出不确定性感知的多目标强化学习框架，引导 3D 分子扩散模型（EDM）同时优化药物相关性（QED）、合成可及性（SAS）和结合亲和力（binding affinity），通过代理模型的预测不确定性动态塑造奖励函数，在三个基准数据集上一致超越基线，并通过分子动力学模拟和 ADMET 验证候选分子的药物潜力。
tags:
  - "NeurIPS 2025"
  - "计算生物"
  - "扩散模型"
  - "强化学习"
  - "多目标优化"
  - "不确定性量化"
  - "3D 分子生成"
  - "药物发现"
---

# Uncertainty-Aware Multi-Objective Reinforcement Learning-Guided Diffusion Models for 3D De Novo Molecular Design

**会议**: NeurIPS 2025  
**arXiv**: [2510.21153](https://arxiv.org/abs/2510.21153)  
**代码**: [Kyle4490/RL-Diffusion](https://github.com/Kyle4490/RL-Diffusion)  
**领域**: 计算生物  
**关键词**: 扩散模型, 强化学习, 多目标优化, 不确定性量化, 3D 分子生成, 药物发现

## 一句话总结
提出不确定性感知的多目标强化学习框架，引导 3D 分子扩散模型（EDM）同时优化药物相关性（QED）、合成可及性（SAS）和结合亲和力（binding affinity），通过代理模型的预测不确定性动态塑造奖励函数，在三个基准数据集上一致超越基线，并通过分子动力学模拟和 ADMET 验证候选分子的药物潜力。

## 研究背景与动机
设计具有理想性质的全新 3D 分子是药物发现的核心挑战。扩散模型在 3D 分子生成上展现出色能力，但现有方法大多仅满足基本化学有效性约束，缺乏对多个药物相关属性的显式控制。

**现有方法的局限**:
- **Flow matching / Energy-guided 方法**：需要显式可微的奖励函数，无法处理 QED、SAS、binding affinity 等黑盒目标
- **RL 引导的生成模型**：已应用于 RNN、VAE、Transformer，但主要处理 1D SMILES 或 2D 分子图，RL 引导 3D 分子扩散模型尚未充分探索
- **图像领域的 RL-扩散模型**（SFT-PG、DDPO、DPOK）：仅针对单目标优化，直接迁移到多目标分子设计效果不佳
- **传统多目标优化**（加权求和、约束法、梯度法）：需要精细的权重调参，且无法建模全局 Pareto 前沿

**核心动机**: 3D 分子几何信息对分子对接和分子动力学等下游任务至关重要——1D/2D 表示无法胜任。需要一个端到端框架，将 RL、扩散模型和不确定性量化统一起来，实现多目标 3D 分子生成。

## 方法详解

### 整体框架
框架由三部分组成：条件 EDM 骨干网络 → 代理模型不确定性量化 → RL 引导优化。

### 1. 条件 EDM 骨干网络
采用 E(3)-等变扩散模型（EDM）作为骨干：
- **前向过程**: 对原子坐标 $\mathbf{r} \in \mathbb{R}^{M \times 3}$ 和特征 $\mathbf{h} \in \mathbb{R}^{M \times d}$ 逐步加噪：$q(\mathbf{z}_t | \mathbf{x}) = \mathcal{N}(\mathbf{z}_t; \alpha_t \mathbf{x}, \sigma_t^2 \mathbf{I})$
- **反向过程**: 参数化去噪分布 $p_\theta(\mathbf{z}_{t-1} | \mathbf{z}_t, c) = \mathcal{N}(\mathbf{z}_{t-1}; \mu_\theta(\mathbf{z}_t, t, c), \sigma_t^2 \mathbf{I})$，其中 $c$ 是目标属性条件向量
- 噪声预测器采用 E(n)-等变 GNN (EGNN)

### 2. 代理模型与多目标不确定性量化
使用 Chemprop 的有向消息传递神经网络 (D-MPNN) 作为代理预测器，为每个属性训练独立模型。

**单属性不确定性奖励**: 估计分子 $m$ 的属性满足阈值 $\delta$ 的概率：
$$U_{\text{single}}(m; \delta) = \eta \int_\delta^\infty \frac{1}{\sigma(m)\sqrt{2\pi}} \exp\left(-\frac{1}{2}\left(\frac{x - \mu(m)}{\sigma(m)}\right)^2\right) dx$$
其中 $\eta = +1$ 表示越高越好（如 QED），$\eta = -1$ 表示越低越好（如 SAS、binding affinity）。

**多目标奖励聚合**: 假设属性条件独立，联合满足概率为各单属性概率之积：
$$U_{\text{multi}}(m; \delta_1, \ldots, \delta_k) = \prod_{i=1}^k U_{\text{single}}^i(m; \delta_i)$$

### 3. RL 引导优化

**轨迹采样**: 每轮采样 $n$ 个分子，记录完整去噪轨迹 $\{\mathbf{z}_T, \ldots, \mathbf{z}_0\}$，将反向去噪重写为概率密度形式以支持梯度估计。

**奖励设计**: 总奖励包含三个辅助组件：
$$R_{\text{total}}(m) = U_{\text{multi}}(m) \cdot R_{\text{bonus}}(m) - \lambda(t_{\text{episode}}) \cdot D(m)$$
- **Reward boosting** $R_{\text{bonus}}$：根据分子的有效性、唯一性、新颖性给予递增奖励
- **Diversity penalty** $D(m)$：批内 Tanimoto 相似度惩罚，防止模式坍塌
- **Dynamic cutoff**: 属性阈值 $\delta_i$ 基于历史生成分子的移动平均动态更新
- 惩罚权重随训练衰减 $\lambda(t) = \lambda_0 e^{-\alpha t}$，前期探索后期利用

**策略更新**: 采用 PPO 风格的裁剪策略梯度损失：
$$\mathcal{L}_{\text{PPO}} = -\mathbb{E}_{m \sim p_\theta}\left[\min\left(r(m) \cdot R_{\text{total}}(m), \text{clip}(r(m), 1-\epsilon, 1+\epsilon) \cdot R_{\text{total}}(m)\right)\right]$$

## 实验关键数据

### 实验设置
- **数据集**: QM9（小有机分子）、ZINC15（类药分子）、PubChem（大型复杂分子）
- **目标属性**: QED > 0.4, SAS < 8, binding affinity < -4.5（EGFR 靶点）
- **基线**: 无 RL 的 vanilla EDM、SFT-PG、DDPO-SF、DDPO-IS、DPOK
- **评估**: 每次生成 2000 个分子，3 次独立运行取平均

### Table 1: 主实验——各方法在三个数据集上的表现

| 数据集 | 方法 | Val (%) | Uni (%) | VUN (%) | MSta (%) | Top (%) |
|--------|------|---------|---------|---------|----------|---------|
| QM9 | W/O RL | 88.55 | 97.57 | 86.19 | 95.90 | 25.17 |
| QM9 | SFT-PG | 88.57 | 96.80 | 85.57 | 95.62 | 25.58 |
| QM9 | DDPO-IS | 88.82 | 96.59 | 85.27 | 86.10 | 25.77 |
| QM9 | **Ours** | **98.17** | 90.90 | **88.90** | **99.17** | **28.33** |
| ZINC15 | W/O RL | 30.05 | 100.00 | 30.05 | 12.00 | 8.02 |
| ZINC15 | SFT-PG | 41.25 | 100.00 | 41.25 | 25.55 | 10.43 |
| ZINC15 | **Ours** | **99.02** | 99.75 | **98.77** | **98.08** | **33.40** |
| PubChem | W/O RL | 7.18 | 99.67 | 7.17 | 38.18 | 2.23 |
| PubChem | DDPO-IS | 10.50 | 99.90 | 10.48 | 45.37 | 2.52 |
| PubChem | **Ours** | **16.23** | 100.00 | **16.23** | **88.65** | **2.97** |

关键发现：
- ZINC15 上改进最为显著：validity 从 30.05% 提升至 **99.02%**，Top 从 8.02% 提升至 **33.40%**
- QM9 上 validity 较所有基线提升超过 **9%**
- PubChem 上分子稳定性 MSta 从 38.18% 提升至 **88.65%**

### Table 2: 消融实验——多目标策略对比（QM9 数据集）

| 类别 | 方法 | Val (%) | VUN (%) | Top (%) |
|------|------|---------|---------|---------|
| Scalarization | WS | 91.78 | 87.86 | 27.02 |
| Scalarization | POO | 89.13 | 77.88 | 24.60 |
| Constraint | NMD | 93.30 | 77.67 | 25.75 |
| Constraint | PFM | 91.98 | 88.75 | 24.68 |
| Gradient | GradVac | 88.50 | 84.83 | 24.43 |
| Uncertainty | UCB | 86.10 | 82.28 | 13.40 |
| Uncertainty | BORE | 89.33 | 86.57 | 23.73 |
| Ours W/O Reward Boost | — | 90.00 | 86.95 | 25.92 |
| Ours W/O Diversity Penalty | — | 83.55 | 65.77 | 25.43 |
| Ours W/O Dynamic Cutoff | Static | 95.73 | 90.65 | 24.88 |
| **Ours (完整)** | — | **98.17** | **88.90** | **28.33** |

关键发现：
- 移除 diversity penalty 后 VUN 从 88.90% 骤降至 **65.77%**，证明其对防止模式坍塌至关重要
- 完整方法在 Top 指标上稳定领先所有替代策略
- 4 类 16 种替代多目标策略均不如本文的不确定性联合概率方法

### MD 与 ADMET 验证
- 生成的候选分子在分子动力学模拟中 RMSD 稳定在 0.20-0.30 nm 内，与已知 EGFR 抑制剂相当
- ADMET 分析确认良好的吸收性、低 CYP 抑制和低毒性
- 框架还扩展到 GeoLDM 和 GFMDiff 架构，验证通用性

## 亮点
- **首个端到端 RL+扩散+不确定性量化框架**：将三者统一用于 3D 多目标分子生成，方法论贡献明确
- **不确定性驱动的联合概率奖励**：将代理模型的预测不确定性转化为平滑可解释的 $[0,1]$ 奖励信号，天然处理黑盒目标
- **完整的奖励工程**：reward boosting + diversity penalty + dynamic cutoff 三组件缺一不可，消融实验证据充分
- **真实药物发现验证**：不止于生成指标，通过 MD 模拟和 ADMET 分析验证候选分子的实际药物潜力，与已知 EGFR 抑制剂对标

## 局限与展望
- **PubChem 大分子表现有限**：validity 仅 16.23%，主要受骨干扩散模型对复杂大分子的建模能力制约，非 RL 框架本身的问题
- **代理模型依赖**: 奖励质量取决于代理模型的预测精度和不确定性校准，binding affinity 的 R² 仅 0.86-0.88
- **属性独立性假设**: 多目标奖励假设各属性条件独立，但药物属性间往往存在相关性（如 QED 与 SAS 负相关）
- **计算开销**: RL 训练阶段需反复生成分子+评估属性+策略更新，训练成本较高
- **评估阈值宽松**: Top 指标使用宽松阈值（QED>0.4, SAS<8），实际药物开发可能需要更严格的标准

## 相关工作
- **3D 分子生成**: G-SchNet（自回归）→ E-NF（等变流）→ EDM（等变扩散）→ GeoLDM（潜空间扩散）→ GFMDiff（物理约束）；本文基于 EDM 并扩展到 GeoLDM/GFMDiff 验证通用性
- **RL 引导扩散**: SFT-PG（减少分布不匹配）、DDPO-IS/SF（去噪作为多步决策）、DPOK（KL 正则化）→ 均为图像领域单目标优化，本文首次迁移到 3D 分子多目标场景
- **多目标优化**: 标量化（WS/POO/MMM）、约束法（NMD/CP）、梯度法（PCGrad/CAGrad）、不确定性法（UCB/EI/BORE）→ 本文提出的联合概率方法在所有 16 种替代策略中表现最优

## 评分
- 新颖性: ⭐⭐⭐⭐ — 首个 RL+扩散+不确定性量化的 3D 分子生成框架，将代理不确定性转化为联合概率奖励的思路简洁有效
- 实验充分度: ⭐⭐⭐⭐⭐ — 3 数据集、5 基线、16 种替代多目标策略消融、3 组件消融、3 扩散架构对比、MD+ADMET 验证
- 写作质量: ⭐⭐⭐⭐ — 结构完整、公式推导清晰，但内容量大导致主文较为密集
- 价值: ⭐⭐⭐⭐ — 对 RL 引导分子生成和多目标药物设计均有实用价值，MD/ADMET 验证增强了实际可信度

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] PepTune: De Novo Generation of Therapeutic Peptides with Multi-Objective-Guided Discrete Diffusion](../../ICML2025/computational_biology/peptune_de_novo_generation_of_therapeutic_peptides_with_multi-objective-guided_d.md)
- [\[NeurIPS 2025\] Towards Unified and Lossless Latent Space for 3D Molecular Latent Diffusion Modeling](towards_unified_and_lossless_latent_space_for_3d_molecular_latent_diffusion_mode.md)
- [\[NeurIPS 2025\] Consistent Sampling and Simulation: Molecular Dynamics with Energy-Based Diffusion Models](consistent_sampling_and_simulation_molecular_dynamics_with_energy-based_diffusio.md)
- [\[NeurIPS 2025\] De novo generation of functional terpene synthases using TpsGPT](de_novo_generation_of_functional_terpene_synthases_using_tpsgpt.md)
- [\[NeurIPS 2025\] Pharmacophore-Guided Generative Design of Novel Drug-Like Molecules](pharmacophore-guided_generative_design_of_novel_drug-like_molecules.md)

</div>

<!-- RELATED:END -->
