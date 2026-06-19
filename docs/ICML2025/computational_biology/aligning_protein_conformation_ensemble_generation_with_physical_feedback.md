---
title: >-
  [论文解读] Aligning Protein Conformation Ensemble Generation with Physical Feedback
description: >-
  [ICML 2025][计算生物][蛋白质构象生成] 提出 Energy-based Alignment (EBA)，将物理力场的能量反馈融入扩散生成模型的微调过程，通过 Boltzmann 因子加权的分类目标函数对齐生成分布与物理能量景观，在 ATLAS MD 基准上实现蛋白质构象集合生成的 SOTA 性能。
tags:
  - "ICML 2025"
  - "计算生物"
  - "蛋白质构象生成"
  - "扩散模型"
  - "物理对齐"
  - "Boltzmann分布"
  - "偏好优化"
---

# Aligning Protein Conformation Ensemble Generation with Physical Feedback

**会议**: ICML 2025  
**arXiv**: [2505.24203](https://arxiv.org/abs/2505.24203)  
**代码**: 无  
**领域**: 计算生物  
**关键词**: 蛋白质构象生成, 扩散模型, 物理对齐, Boltzmann分布, 偏好优化

## 一句话总结

提出 Energy-based Alignment (EBA)，将物理力场的能量反馈融入扩散生成模型的微调过程，通过 Boltzmann 因子加权的分类目标函数对齐生成分布与物理能量景观，在 ATLAS MD 基准上实现蛋白质构象集合生成的 SOTA 性能。

## 研究背景与动机

蛋白质动力学是理解蛋白质功能和调控的关键，蛋白质结构在不同空间和时间尺度上在多个构象状态之间转换。传统的分子动力学（MD）模拟虽然能捕获这些动态行为，但计算代价极高——捕获折叠/解折叠等生物学相关转变通常需要微秒到毫秒的时间尺度模拟，往往需要数百到数千个 GPU 天。

近年来，去噪扩散模型被用于蛋白质构象生成，将其重构为条件生成任务。然而，这些数据驱动方法存在两个核心问题：

**缺乏热力学建模**：纯数据驱动的方法虽然能生成结构上合理的候选构象，但不显式建模热力学性质，无法保证生成样本服从 Boltzmann 分布

**配分函数不可解**：更合理的形式化——从 Boltzmann 分布采样均衡构象集——由于配分函数 $Z = \sum_{\mathbf{x}} e^{-\beta E(\mathbf{x};\mathbf{c})}$ 需要对高维空间所有可能状态求和，直接优化不可行

**已有方法局限**：现有的分摊采样方法（如 GFlowNet）难以扩展到包含数千原子的蛋白质结构

## 方法详解

### 整体框架

EBA 的核心思想是：不去近似不可解的配分函数 $Z$，而是利用 **Boltzmann 因子**——两个状态概率比与能量差的关系：

$$\frac{p_B(\mathbf{x}^i|\mathbf{c})}{p_B(\mathbf{x}^j|\mathbf{c})} = e^{-\beta \Delta E_{ij}}$$

其中 $\Delta E_{ij} = E(\mathbf{x}^i;\mathbf{c}) - E(\mathbf{x}^j;\mathbf{c})$。这种依赖能量差的形式对绝对能量值的平移不变，特别适合生成模型训练，因为能量尺度随蛋白质原子数变化显著。

整个训练流程分为两个阶段：

- **Stage 1 — 监督微调**：在 ATLAS MD 轨迹数据上对预训练的 AlphaFold3 扩散模块进行微调，使模型粗略适应构象空间的数据分布
- **Stage 2 — 物理对齐**：使用 EBA 目标函数，利用力场能量反馈对扩散模型进行对齐训练

### 关键设计

#### 1. EBA 目标函数推导

假设可学习概率模型 $p_\theta(\mathbf{x}|\mathbf{c}) = e^{-\alpha E_\theta(\mathbf{x};\mathbf{c})}/Z$，通过最小化与目标 Boltzmann 分布的 KL 散度（即交叉熵），可得：

$$\mathbb{D}_{\text{KL}}(p_B \| p_\theta) = -\sum_i p_B(\mathbf{x}^i|\mathbf{c}) \log p_\theta(\mathbf{x}^i|\mathbf{c}) + \text{Const}$$

由于对所有可能构象状态求和不可解，EBA 用**随机有限子集近似**：从某提议分布 $p^*$ 中采样 $K$ 个代表性状态 $\{\mathbf{x}^i\}_{i=1}^K$，得到 EBA 目标：

$$\mathcal{L}_{\text{EBA}}(\theta) = -\mathbb{E} \left[ \sum_{i=1}^K \frac{e^{-\beta E(\mathbf{x}^i;\mathbf{c})}}{\sum_{j=1}^K e^{-\beta E(\mathbf{x}^j;\mathbf{c})}} \log \frac{e^{-\alpha E_\theta(\mathbf{x}^i;\mathbf{c})}}{\sum_{j=1}^K e^{-\alpha E_\theta(\mathbf{x}^j;\mathbf{c})}} \right]$$

这是一个能量加权的分类式目标，保证了 mini-batch 内的 Boltzmann 因子不变性。

#### 2. EBA 适配扩散模型

将能量函数定义为扩散链的 KL 散度之和，并利用 Jensen 不等式（LSE 函数的凸性）推出上界，最终以去噪误差替代 KL 散度，得到扩散版 EBA 目标：

$$\mathcal{L}_{\text{EBA-Diff}} = -\mathbb{E} \sum_{i=1}^K \frac{e^{-\beta E(\mathbf{x}^i;\mathbf{c})}}{\sum_j e^{-\beta E(\mathbf{x}^j;\mathbf{c})}} \log \frac{e^{-\alpha T \|\epsilon^i - \epsilon_\theta(\mathbf{x}_t^i,t,\mathbf{c})\|_2^2}}{\sum_j e^{-\alpha T \|\epsilon^j - \epsilon_\theta(\mathbf{x}_t^j,t,\mathbf{c})\|_2^2}}$$

#### 3. DPO 是 EBA 的特例

当 $K=2$ 且温度趋于零（$\beta \to \infty$）时，EBA 退化为标准 DPO 目标。这建立了 EBA 与 RLHF/DPO 文献的理论联系，同时表明 EBA 是更一般的形式——它支持多于两个状态的比较，且保留了精细的能量差信息而非仅二元偏好。

#### 4. SE(3) 不变损失设计

标准 MSE 对蛋白质构象生成并非最优，因为输入条件（氨基酸序列）是 SE(3) 不变的。论文设计了两个 SE(3) 不变损失：

- **刚体对齐 MSE**：先用 Kabsch 算法将预测坐标对齐到真值，再计算对齐后的 MSE
- **Smooth LDDT**：基于成对距离矩阵的辅助损失，捕获原子间几何关系，对 15Å 内的原子对加权评估

#### 5. 能量归一化

蛋白质大小差异巨大导致能量值方差极大，作者引入样本特定的归一化因子 $L^{0.5}$（$L$ 为残基数），对 $\beta$ 进行缩放：$\beta \leftarrow \beta / L^{0.5}$，灵感来自折叠时间与残基数呈 0.5 次幂关系的经验发现。

### 损失函数 / 训练策略

最终的去噪训练损失为：

$$L_{\text{total}} = \lambda_{\text{mse}} L_{\text{Aligned MSE}} + \lambda_{\text{lddt}} L_{\text{Smooth LDDT}}$$

在 EBA 框架中，这个 $L_{\text{total}}$ 作为每个候选样本的"能量"输入 softmax 归一化：

$$\mathcal{L}_{\text{EBA-Diffusion}} = -\sum_{i=1}^K w(\mathbf{x}_0^i) \log \frac{e^{-L_{\text{total}}^i}}{\sum_{j=1}^K e^{-L_{\text{total}}^j}}$$

其中 $w(\mathbf{x}_0^i)$ 是由物理能量计算的 Boltzmann 权重。训练使用 Protenix（AlphaFold3 开源实现），冻结 MSA Module 和 PairFormer，仅微调扩散模块。能量标注通过离线局部最小化预计算。

## 实验关键数据

### 主实验

在 ATLAS MD 基准测试集（N=250 个蛋白质靶点）上评估，报告中位数结果：

| 指标类别 | 指标 | AlphaFlow-MD | MSA-sub(256) | MDGen | Pre-train | EBA-DPO | **EBA** |
|---------|------|-------------|-------------|-------|-----------|---------|---------|
| 灵活性 | Pairwise RMSD r↑ | 0.48 | 0.15 | 0.48 | 0.43 | 0.59 | **0.62** |
| 灵活性 | Global RMSF r↑ | 0.60 | 0.26 | 0.50 | 0.50 | 0.69 | **0.71** |
| 灵活性 | Per-target RMSF r↑ | 0.85 | 0.55 | 0.71 | 0.72 | 0.90 | **0.90** |
| 分布精度 | Root mean W₂↓ | 2.61 | 3.62 | 2.69 | 3.22 | 2.43 | **2.43** |
| 分布精度 | MD PCA W₂↓ | 1.52 | 1.88 | 1.89 | 1.78 | 1.20 | **1.19** |
| 集合观测 | Weak contacts J↑ | 0.62 | 0.30 | 0.51 | 0.23 | 0.63 | **0.65** |
| 集合观测 | Exposed residue J↑ | 0.50 | 0.33 | 0.29 | 0.29 | 0.68 | **0.70** |
| 集合观测 | Exposed MI ρ↑ | 0.25 | 0.06 | - | 0.01 | 0.35 | **0.36** |

EBA 在所有 14 个指标上均达到最优或次优，运行效率为 0.9 GPU秒/样本，远快于 AlphaFlow-MD 的 70 秒（约 78 倍加速）。

### 消融实验

不同 mini-batch 大小 $K$ 对 EBA 性能的影响：

| 配置 | K=2 | K=3 | K=5 | 说明 |
|------|-----|-----|-----|------|
| Pairwise RMSD r↑ | 0.62 | 0.61 | 0.62 | 性能稳定 |
| Global RMSF r↑ | 0.71 | 0.71 | 0.72 | 略有提升 |
| Root mean W₂↓ | 2.43 | 2.42 | 2.40 | 略有改善 |
| MD PCA W₂↓ | 1.19 | 1.18 | **1.16** | K=5最优 |
| Exposed MI ρ↑ | 0.36 | **0.37** | 0.34 | K=3最优 |
| 迭代时间(s) | 4.3 | 5.4 | 7.8 | 准线性增长 |
| GPU内存(GB) | 12.0 | 13.9 | 16.3 | 开销温和 |

### 关键发现

1. **EBA 显著优于 DPO 变体**：在 Exposed residue J（0.70 vs 0.68）和 Exposed MI ρ（0.36 vs 0.35）上的提升表明，保留精细能量差信息（而非仅二元偏好）对捕获长程动力学至关重要
2. **物理对齐的有效性**：Pre-train → EBA 的提升巨大（如 Pairwise RMSD r 从 0.43 到 0.62），证明物理反馈能有效校正纯数据驱动模型的偏差
3. **K 值鲁棒性**：K=2,3,5 的性能差异很小，说明 mini-batch 近似是有效的，且 K=2 已足够获得良好性能
4. **效率优势**：0.9 秒/样本，比 AlphaFlow-MD（70秒）快约 78 倍，比 MDGen（0.2秒）略慢但精度大幅领先

## 亮点与洞察

1. **理论优雅**：统一了 RLHF/DPO 与物理 Boltzmann 分布对齐的理论框架，证明 DPO 是 EBA 的特例（K=2, β→∞），为两个原本独立的研究方向建立了深层连接
2. **避免配分函数计算**：通过 Boltzmann 因子（状态间的相对权重）而非绝对概率建模，巧妙规避了不可解的配分函数
3. **全原子建模**：基于 AlphaFold3 的全原子扩散模型，相比依赖粗粒化或内坐标表示的方法，能更直接捕获精细构象变化
4. **能量归一化技巧**：$L^{0.5}$ 归一化解决了不同大小蛋白质能量尺度差异巨大的实际问题，显示了对物理直觉的深刻理解

## 局限与展望

1. **长时间尺度动力学受限**：AlphaFold3 原设计用于折叠预测，微调后可能不适合建模微秒-毫秒级的长时间尺度动力学
2. **力场精度不足**：使用的能量函数精度低于量子级别的单点能量计算，可能限制生成构象的物理准确性
3. **仅限单链蛋白**：当前研究局限于单链蛋白质集合生成，未扩展到多链复合物
4. **生成模型框架单一**：仅在扩散框架中实现和评估了 EBA，未探索 Flow Matching、VAE 等替代方案
5. **数据依赖 MD 模拟**：训练仍需 ATLAS 提供的 MD 轨迹数据作为参考分布，未完全摆脱对昂贵模拟的依赖

## 相关工作与启发

- **AlphaFlow**（Jing et al., 2024a）：将 AlphaFold2 改造为去噪网络，是本文的主要对比方法。EBA 在其基础上进一步引入物理反馈
- **Diffusion-DPO**（Wallace et al., 2024）：将 DPO 扩展到扩散模型用于文本生图，本文在其基础上推广到多状态比较和物理能量加权
- **ConfDiff**（Wang et al., 2024）：在逆扩散过程中引入能量和力引导，但属于推理时干预而非训练时对齐
- **Boltzmann Generator**（Noé et al., 2019）：使用 Normalizing Flow 近似 Boltzmann 分布，但难以扩展到大蛋白

**启发**：EBA 的"物理反馈对齐"范式可推广到其他物理系统（晶体、分子），凡是存在目标能量景观的生成任务都可借鉴。将 RLHF 思想迁移到科学计算领域是一个值得深入探索的方向。

## 评分

- **新颖性**: ⭐⭐⭐⭐ — EBA 框架将 RLHF/DPO 与 Boltzmann 分布统一，理论贡献扎实；但核心想法（用能量加权 softmax 分类目标）并非全新
- **实验充分度**: ⭐⭐⭐⭐⭐ — 在标准 ATLAS 基准上全面评估，14 个指标均领先；消融实验覆盖了关键设计选择
- **写作质量**: ⭐⭐⭐⭐⭐ — 数学推导清晰完整，DPO 特例的推导优雅，动机-方法-实验逻辑连贯
- **价值**: ⭐⭐⭐⭐ — 为蛋白质动力学建模提供了新范式，但应用场景受限于特定领域

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] PolyConf: Unlocking Polymer Conformation Generation through Hierarchical Generative Models](polyconf_unlocking_polymer_conformation_generation_through_hierarchical_generati.md)
- [\[AAAI 2026\] EPO: Diverse and Realistic Protein Ensemble Generation via Energy Preference Optimization](../../AAAI2026/computational_biology/epo_diverse_and_realistic_protein_ensemble_generation_via_energy_preference_opti.md)
- [\[NeurIPS 2025\] ConfRover: Simultaneous Modeling of Protein Conformation and Dynamics via Autoregression](../../NeurIPS2025/computational_biology/confrover_simultaneous_modeling_of_protein_conformation_and_dynamics_via_autoreg.md)
- [\[ICML 2025\] Improving Flow Matching by Aligning Flow Divergence](improving_flow_matching_by_aligning_flow_divergence.md)
- [\[NeurIPS 2025\] Energy Loss Functions for Physical Systems](../../NeurIPS2025/computational_biology/energy_loss_functions_for_physical_systems.md)

</div>

<!-- RELATED:END -->
