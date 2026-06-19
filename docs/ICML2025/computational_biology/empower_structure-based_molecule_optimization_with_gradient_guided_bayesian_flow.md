---
title: >-
  [论文解读] Empower Structure-Based Molecule Optimization with Gradient Guided Bayesian Flow Networks
description: >-
  [ICML2025][计算生物][Bayesian Flow Network] 提出 MolJO 框架，利用贝叶斯流网络（BFN）的连续可微参数空间 $\boldsymbol{\theta}$，实现对分子坐标（连续）和原子类型（离散）的联合梯度引导优化，并设计滑动窗口后向校正策略平衡探索与利用，在 CrossDocked2020 上以 51.3% Success Rate 大幅领先现有方法。
tags:
  - "ICML2025"
  - "计算生物"
  - "Bayesian Flow Network"
  - "结构感知分子优化"
  - "梯度引导"
  - "连续-离散联合优化"
  - "SE(3)-等变"
---

# Empower Structure-Based Molecule Optimization with Gradient Guided Bayesian Flow Networks

**会议**: ICML2025  
**arXiv**: [2411.13280](https://arxiv.org/abs/2411.13280)  
**代码**: [AlgoMole/MolCRAFT](https://github.com/AlgoMole/MolCRAFT)  
**领域**: 计算生物  
**关键词**: Bayesian Flow Network, 结构感知分子优化, 梯度引导, 连续-离散联合优化, SE(3)-等变

## 一句话总结

提出 MolJO 框架，利用贝叶斯流网络（BFN）的连续可微参数空间 $\boldsymbol{\theta}$，实现对分子坐标（连续）和原子类型（离散）的联合梯度引导优化，并设计滑动窗口后向校正策略平衡探索与利用，在 CrossDocked2020 上以 51.3% Success Rate 大幅领先现有方法。

## 研究背景与动机

**问题定义**：结构感知分子优化（SBMO）旨在给定蛋白质靶标口袋的条件下，同时优化配体分子的3D坐标 $\mathbf{x} \in \mathbb{R}^{N \times 3}$ 和离散原子类型 $\mathbf{v} \in \{1,\dots,K\}^N$，使其满足结合亲和力、可合成性等多种药物性质指标。

**现有方法的不足**：

**生成模型**（如 TargetDiff、DecompDiff、MolCRAFT）主要最大化似然拟合训练数据，缺乏对分子性质的针对性优化能力

**基于Oracle的方法**（如 DecompOpt）需要反复调用对接模拟进行 top-of-N 筛选，计算代价大

**梯度引导方法**面临连续-离散挑战：
   - 离散原子类型无法直接做梯度回传，已有近似（加高斯噪声、假设分类器服从高斯）不准确
   - TAGMol 仅引导连续坐标而忽略离散类型，导致跨模态不一致——结合亲和力改善但可合成性下降

**核心动机**：BFN 的参数空间 $\boldsymbol{\theta}$ 是通过贝叶斯推断对含噪样本的聚合，天然连续可微，且同时涵盖连续和离散模态，为联合梯度引导提供了理想载体。

## 方法详解

### 整体框架：MolJO

MolJO（Molecule Joint Optimization）在 BFN 的参数空间 $\boldsymbol{\theta} = [\boldsymbol{\theta}^x, \boldsymbol{\theta}^v]$ 上施加梯度引导。与扩散模型在含噪潜变量 $\mathbf{y}$ 上引导不同，MolJO 在低方差的贝叶斯后验 $\boldsymbol{\theta}$ 上操作，保证了梯度流的平滑性。

### 联合梯度引导

将引导分布定义为产品专家（Product of Experts）形式：

$$\pi(\boldsymbol{\theta}_i | \boldsymbol{\theta}_{i-1}) \propto p_\phi(\boldsymbol{\theta}_i | \boldsymbol{\theta}_{i-1}) \cdot p_E(\boldsymbol{\theta}_i)$$

其中 $p_E(\boldsymbol{\theta}_i) = \exp[-E(\boldsymbol{\theta}_i, t_i)]$ 为能量函数对应的 Boltzmann 分布。通过一阶Taylor展开近似引导后的转移核：

**连续坐标引导**：

$$\boldsymbol{\theta}_i^x \sim \mathcal{N}\left(\boldsymbol{\theta}_\phi^x + \sigma^x \mathbf{g}_{\boldsymbol{\theta}^x},\; \sigma^x \mathbb{I}\right)$$

其中 $\mathbf{g}_{\boldsymbol{\theta}^x} = -\nabla_{\boldsymbol{\theta}^x} E(\boldsymbol{\theta}, t_i)|_{\boldsymbol{\theta}=\boldsymbol{\theta}_{i-1}}$。这等价于用不确定性调整的梯度 $(\rho_i / \alpha_i)^2 \mathbf{g}_{\mathbf{y}^x}$ 引导含噪潜变量 $\mathbf{y}^x$。

**离散类型引导**：

$$\mathbf{y}_i^v \sim \mathcal{N}\left(\mathbf{y}_\phi^v + \sigma^v \mathbf{g}_{\mathbf{y}^v},\; \sigma^v \mathbb{I}\right)$$

引导通过高斯分布的潜变量 $\mathbf{y}^v$ 作用于离散数据，最终效果是对分类分布 $\boldsymbol{\theta}^v$ 做类别概率的重加权——梯度指向的类别概率增大，其他类别概率相应减小。关键优势：无需假设分类器服从高斯分布，且保证离散变量始终在概率单纯形上。

**SE(3)-等变性**：当网络 $\boldsymbol{\Phi}$ 和能量函数 $E$ 均满足 SE(3)-等变性，且蛋白质质心归零时，引导后的采样过程仍保持 SE(3)-等变性。

### 后向校正采样策略（Backward Correction）

标准 BFN 采样中，步骤 $i$ 的更新仅依赖前一步 $\boldsymbol{\theta}_{i-1}$。后向校正维护一个大小为 $k$ 的滑动窗口，将当前步的优化预测 $\hat{\mathbf{x}}_i$ 回溯替换过去 $k$ 步的历史，重新聚合 $\boldsymbol{\theta}$：

$$p_\phi(\boldsymbol{\theta}_n | \boldsymbol{\theta}_{n-1}, \boldsymbol{\theta}_{n-k}) = \mathbb{E}_{p_O(\hat{\mathbf{x}}_n | \boldsymbol{\Phi}(\boldsymbol{\theta}_{n-1}, t_n))} \; p_U\!\left(\boldsymbol{\theta}_n \,|\, \boldsymbol{\theta}_{n-k}, \hat{\mathbf{x}}_n;\; \sum_{i=n-k+1}^{n} \alpha_i\right)$$

具体更新形式——连续部分：

$$\boldsymbol{\theta}_n^x \sim \mathcal{N}\!\left(\frac{\Delta\beta \hat{\mathbf{x}}_n + \boldsymbol{\theta}_{n-k}^x \rho_{n-k}}{\rho_n},\; \frac{\Delta\beta}{\rho_n^2}\mathbb{I}\right)$$

**探索-利用权衡**：$k=1$ 退化为标准单步更新（最大探索），$k=n$ 使用全部历史（最大利用）。中等 $k$ 值在优化前期允许快速探索分子空间，后期利用更一致的梯度信号精细优化。实验中梯度余弦相似度可视化验证了该效果。

### 损失函数

BFN 训练目标为最小化 sender 与 receiver 分布的 KL 散度：

$$L^n(\mathbf{x}) = \mathbb{E}_{\prod_{i=1}^n p_U(\boldsymbol{\theta}_i | \boldsymbol{\theta}_{i-1}, \mathbf{x}; \alpha_i)} \sum_{i=1}^n D_{\text{KL}}\!\left(p_S(\mathbf{y}_i | \mathbf{x}; \alpha_i) \,\|\, p_R(\mathbf{y}_i | \boldsymbol{\theta}_{i-1}, t_i, \alpha_i)\right)$$

推理时引入梯度缩放因子 $s$ 作为温度参数，等价于 $p_E^s(\boldsymbol{\theta},t) \propto \exp[-sE(\boldsymbol{\theta},t)]$。

## 实验关键数据

### 数据集与设置

- **数据集**：CrossDocked2020，过滤 RMSD > 1Å，按 30% 序列同源性聚类，100K 训练 pose + 100 测试蛋白
- **评价指标**：Vina Score/Min/Dock（结合能 ↓）、QED（类药性 ↑）、SA（可合成性 ↑）、Success Rate（Vina Dock < -8.18 且 QED > 0.25 且 SA > 0.59）
- 每个蛋白生成 100 个分子

### 无约束优化主结果（Table 1）

| 方法 | 类别 | Vina Dock ↓ | QED ↑ | SA ↑ | Success Rate ↑ |
|------|------|-------------|-------|------|----------------|
| Reference | — | -7.45 | 0.48 | 0.73 | 25.0% |
| TargetDiff | Gen | -7.80 | 0.48 | 0.58 | 10.5% |
| MolCRAFT | Gen | -7.67 | 0.50 | 0.67 | 26.8% |
| DecompOpt | Oracle | -7.63 | 0.56 | 0.73 | 39.4% |
| TAGMol | Grad | -8.59 | 0.55 | 0.56 | 11.1% |
| **MolJO** | **Grad** | **-9.05** | **0.56** | **0.78** | **51.3%** |
| MolJO† (N=10) | G+O | -10.50 | 0.67 | 0.79 | 70.3% |

**关键观察**：

- MolJO 在 Vina Dock、SA 和 Success Rate 上均达到 SOTA
- 相比唯一的梯度引导基线 TAGMol，Success Rate 从 11.1% → 51.3%（约 **4.6× 提升**）
- TAGMol 的 SA 仅 0.56（最低之一），印证了仅引导连续坐标导致可合成性下降的论断
- MolJO 的 "Me-Better" 比率（改善分子占比）是其他 3D 基线的 **2×**

### 约束优化（R-group 优化 & 骨架跳跃）

MolJO 可灵活扩展至 R-group 重设计（固定母核替换取代基）和骨架跳跃（scaffold hopping）等实用药物设计场景，进一步展示了方法的通用性。

## 亮点与洞察

1. **首个对连续-离散数据的原理性联合梯度引导框架**：利用 BFN 参数空间的连续可微性，绕过了扩散模型引导离散数据的根本困难
2. **后向校正策略 novel 且实用**：滑动窗口大小 $k$ 提供了探索-利用的灵活旋钮，梯度余弦相似度的可视化分析直观有效
3. **跨模态一致性**：联合引导同时改善结合亲和力和可合成性，解决了 TAGMol 等方法中离散与连续模态脱节的问题
4. **即插即用**：MolJO 作为引导方法可以与不同的预训练生成模型组合，且支持多目标优化

## 局限与展望

1. **能量函数的局限**：当前依赖可微的代理能量函数进行梯度计算，其与真实对接评分之间存在差距（即 guidance 信号的质量取决于代理的准确度）
2. **计算开销**：后向校正中滑动窗口增大了每步的计算量和存储需求，$k$ 较大时代价更高
3. **评估局限**：仅在 CrossDocked2020 benchmark 上做了主要评估，尚未在真实的药物发现流程中进行验证
4. **原子类型离散引导的近似**：虽然比 TAGMol 更合理，但仍依赖一阶 Taylor 展开近似，在分布偏离高斯时可能不够准确
5. **分子有效性**：论文未重点讨论化学有效性检查和后处理，生成分子的实际可用性有待检验

## 相关工作与启发

- **MolCRAFT**（Qu et al., 2024）：本文的 BFN 基座模型，MolJO 在其上加入梯度引导
- **TAGMol**（Dorna et al., 2024）：仅引导连续坐标的梯度方法，本文的直接竞争对手
- **Classifier Guidance**（Dhariwal & Nichol, 2021）：扩散模型的分类器引导，MolJO 将其推广到 BFN + 离散数据
- **BFN**（Graves et al., 2023）：贝叶斯流网络原始论文，提供了 $\boldsymbol{\theta}$ 空间的理论基础

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — BFN参数空间做联合引导的思路新颖，后向校正策略有数学推导支撑
- 实验充分度: ⭐⭐⭐⭐ — 无约束/约束/多目标/R-group/scaffold hopping 覆盖全面，缺少真实药物验证
- 写作质量: ⭐⭐⭐⭐ — 数学推导清晰，动机阐述到位，图示设计精良
- 价值: ⭐⭐⭐⭐⭐ — 解决了SBMO中连续-离散梯度引导的核心痛点，Success Rate提升显著

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Flexibility-conditioned Protein Structure Design with Flow Matching](flexibility-conditioned_protein_structure_design_with_flow_matching.md)
- [\[NeurIPS 2025\] Prior-Guided Flow Matching for Target-Aware Molecule Design with Learnable Atom Number](../../NeurIPS2025/computational_biology/prior-guided_flow_matching_for_target-aware_molecule_design_with_learnable_atom_.md)
- [\[NeurIPS 2025\] Is Sequence Information All You Need for Bayesian Optimization of Antibodies?](../../NeurIPS2025/computational_biology/is_sequence_information_all_you_need_for_bayesian_optimization_of_antibodies.md)
- [\[NeurIPS 2025\] Curly Flow Matching for Learning Non-gradient Field Dynamics](../../NeurIPS2025/computational_biology/curly_flow_matching_for_learning_non-gradient_field_dynamics.md)
- [\[ICML 2025\] Geometric Generative Modeling with Noise-Conditioned Graph Networks](geometric_generative_modeling_with_noise-conditioned_graph_networks.md)

</div>

<!-- RELATED:END -->
