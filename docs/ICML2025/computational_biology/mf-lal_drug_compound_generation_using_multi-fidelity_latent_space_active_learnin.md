---
title: >-
  [论文解读] MF-LAL: Drug Compound Generation Using Multi-Fidelity Latent Space Active Learning
description: >-
  [ICML 2025][计算生物][多保真度建模] 提出 MF-LAL 框架，将多保真度代理模型与分子生成模型统一到层次化潜空间中，通过主动学习高效整合分子对接（低保真）和结合自由能计算（高保真）两类预言机，生成具有显著更优结合自由能的候选药物分子（平均 ABFE 得分提升约 50%）。 当前药物发现中的分子生成模型主要依赖…
tags:
  - "ICML 2025"
  - "计算生物"
  - "多保真度建模"
  - "主动学习"
  - "药物生成"
  - "潜空间优化"
  - "结合自由能"
---

# MF-LAL: Drug Compound Generation Using Multi-Fidelity Latent Space Active Learning

**会议**: ICML 2025  
**arXiv**: [2410.11226](https://arxiv.org/abs/2410.11226)  
**代码**: [Rose-STL-Lab/MF-LAL](https://github.com/Rose-STL-Lab/MF-LAL)  
**领域**: 医学图像 / 药物发现  
**关键词**: 多保真度建模, 主动学习, 药物生成, 潜空间优化, 结合自由能

## 一句话总结

提出 MF-LAL 框架，将多保真度代理模型与分子生成模型统一到层次化潜空间中，通过主动学习高效整合分子对接（低保真）和结合自由能计算（高保真）两类预言机，生成具有显著更优结合自由能的候选药物分子（平均 ABFE 得分提升约 50%）。

## 研究背景与动机

当前药物发现中的分子生成模型主要依赖**分子对接（molecular docking）**作为预言机来评估生成化合物的活性。然而，对接得分较高的化合物在实际实验中并不总能展现真实的生物活性，这使得生成模型在实际应用中价值有限。

更准确的活性预测方法——**基于分子动力学模拟的结合自由能（binding free energy）计算**——虽然被认为是预测亲和力的金标准，但单次计算需要数小时到数天，计算成本过于昂贵，无法直接用于生成模型的迭代循环中。

**多保真度代理建模**可以融合多个精度/成本级别的预言机。然而，现有方法（如 MF-AL-GFN）将生成模型和代理模型**分离训练**，导致生成模型无法感知不同保真度级别下最优查询化合物分布的差异，查询效率受限。

MF-LAL 的核心洞察是：**不同保真度级别下的最优分子分布可能不同**，因此需要为每个保真度级别设计独立的潜空间和解码器，同时通过层次化网络共享跨保真度信息。

## 方法详解

### 整体框架

MF-LAL 由三个核心模块组成：

1. **层次化潜空间编码**（左）：将分子编码到一系列按保真度级别排列的潜空间中
2. **代理建模与反向优化**（中）：在每个潜空间中学习代理模型预测预言机输出，并执行梯度优化生成高分子
3. **主动学习循环**（右）：将生成的分子送入相应保真度的预言机，用结果重新训练潜空间表示和代理模型

本文使用 4 个保真度级别的预言机：
- **$f_1$**：线性回归（~0.1s，ROC-AUC 0.59/0.68）
- **$f_2$**：AutoDock4（~4s，ROC-AUC 0.73/0.72）
- **$f_3$**：集成 AutoDock4（~44-68s，ROC-AUC 0.80/0.80）
- **$f_4$**：绝对结合自由能 ABFE（~9.3h，ROC-AUC 0.92/0.89）

### 关键设计

#### 1. 多保真度潜空间层次结构

使用单一概率编码器 $q_\phi$ 将分子 $x$（以 SELFIES 字符串表示）编码到最低保真度潜空间 $\mathbf{z}_1 \sim \mathcal{N}(\mu_1, \sigma_1)$。

通过一组概率映射网络 $h_{\xi_1}, \ldots, h_{\xi_{K-1}}$ 在相邻保真度潜空间间传递信息：

$$\mathbf{z}_{k+1} \sim \mathcal{N}(\mu_{k+1}, \sigma_{k+1}), \quad (\mu_{k+1}, \sigma_{k+1}) = h_{\xi_k}(\mathbf{z}_k)$$

每个保真度级别有独立的解码器 $p_{\theta_k}(\cdot | \mathbf{z}_k)$ 用于重构分子。这比共享单一潜空间的方法在两方面更优：
- **生成质量更高**：针对每个保真度有专门的解码器
- **代理建模更准确**：每个潜空间可以独立组织以优化该级别的预测

#### 2. 代理建模：SVGP

每个保真度级别使用**随机变分高斯过程（SVGP）**作为代理模型 $\hat{f}_k$，从对应潜向量 $\mathbf{z}_k$ 预测预言机输出。使用 4 层深度核（Deep Kernel）编码输入，Matern 核作为协方差函数。SVGP 选择的原因：训练速度快，支持 mini-batch 训练，且能产生不确定性估计。

#### 3. 新颖的似然约束生成

在高保真度潜空间中生成分子时，引入**似然项**约束：确保在保真度 $k$ 生成的分子在保真度 $k-1$ 处也曾获得高分。具体做法：
- 首先在保真度 $k-1$ 生成 $M$ 个高分分子
- 通过 $h_{\xi_{k-1}}$ 映射到保真度 $k$ 的潜空间，形成高斯混合分布
- 最大化待生成点在该混合分布中的似然

$$P(\mathbf{z}_k^{(i)} | \{(\mu_k^{(j)}, \sigma_k^{(j)})\}_{j=1}^M) = \sum_{j=1}^M \frac{1}{\sqrt{2\pi (\sigma_k^{(j)})^2}} \exp\left(-\frac{(\mathbf{z}_k^{(i)} - \mu_k^{(j)})^2}{2(\sigma_k^{(j)})^2}\right)$$

这一设计极大缩小了高保真度预言机需要探索的化学空间，使昂贵的 ABFE 计算变得可行。

#### 4. 阶梯式主动学习策略

从最低保真度 $k=1$ 开始查询，当 GP 后验方差 $\Sigma_{\lambda_k}(\mathbf{z}_k) < \gamma_k$ 时，永久递增 $k$。使用上置信界（UCB）作为采集函数：

$$a(\mathbf{z}_k^{(i)}, k) = m_{\lambda_k}(\mathbf{z}_k^{(i)}) + \beta \cdot \Sigma_{\lambda_k}(\mathbf{z}_k^{(i)}) - \|\mathbf{z}_k^{(i)}\|_2^2$$

主动学习时 $\beta=1$（探索+利用），推理时 $\beta=0$（纯利用）。L2 正则化项保证生成的化合物接近训练集的药物样分子分布。

### 损失函数 / 训练策略

联合最小化 ELBO 和 GP 的边际对数似然（MLL）：

$$L(\phi, \xi_{k-1}, \theta_k, \lambda_k; k, x, y) = \underbrace{\mathbb{E}_{\mathbf{z}_k \sim g(\cdot|x)} \log \frac{p_{\theta_k}(x, \mathbf{z}_k)}{g(\mathbf{z}_k|x)}}_{\text{ELBO}} + \underbrace{\int p(y|\hat{f}_k(\mathbf{z}_k)) p(\hat{f}_k(\mathbf{z}_k)|\mathbf{z}_k) d\hat{f}_k}_{\text{MLL}}$$

关键训练细节：
- 损失在保真度 $k$ 评估，但**反向传播到所有更低保真度**
- 每个主动学习步重新从头训练到收敛（Adam, lr=0.0001）
- 分子生成时用 Adam（lr=0.1, 100 epochs）做梯度优化
- 编码器/解码器/映射网络：3 层全连接，ReLU，512 维隐层，64 维潜空间
- 生成时加入余弦相似度多样性惩罚

## 实验关键数据

### 主实验

在两个与癌症相关的人类蛋白（BRD4(2) 和 c-MET）上进行评估，固定 7 天计算预算：

| 方法 | BRD4(2) Mean ABFE | BRD4(2) Top-3 | c-MET Mean ABFE | c-MET Top-3 | 类型 |
|------|-------------------|---------------|-----------------|-------------|------|
| **MF-LAL (ours)** | **-6.2 ± 3.9** | **-12.0 / -10.2 / -9.8** | **-6.7 ± 3.1** | **-12.9 / -7.9 / -7.7** | 多保真度 |
| Pocket2Mol | -4.3 ± 3.8 | -9.8 / -8.7 / -8.0 | -2.2 ± 4.2 | -4.5 / -3.9 / -3.2 | 3D结构 |
| MF-AL-PPO | -2.8 ± 2.5 | -9.2 / -6.5 / -5.2 | -4.2 ± 2.8 | -6.6 / -5.8 / -5.5 | 多保真度 |
| REINVENT (ABFE) | -3.9 ± 3.4 | -8.7 / -8.3 / -8.2 | -2.9 ± 3.7 | -6.5 / -5.8 / -5.1 | 单保真度 |
| DecompDiff | -2.7 ± 4.0 | -8.9 / -8.1 / -7.5 | -1.9 ± 6.4 | -8.0 / -5.1 / -2.7 | 3D结构 |
| MF-AL-GFN | -2.5 ± 2.2 | -6.5 / -5.8 / -5.1 | -3.1 ± 1.8 | -5.5 / -4.5 / -4.1 | 多保真度 |

MF-LAL 在 40 个生成化合物的扩展评估中：BRD4(2) 平均 ABFE **-6.3**（p<0.05），c-MET 平均 ABFE **-7.1**（p<0.05），活性骨架数分别为 **8** 和 **6**，显著超越基线。

### 消融实验

| 配置 | BRD4(2) Mean | BRD4(2) Top-3 | c-MET Mean | c-MET Top-3 | 说明 |
|------|-------------|---------------|------------|-------------|------|
| MF-LAL (完整) | -6.2 | -12.0/-10.2/-9.8 | -6.7 | -12.9/-7.9/-7.7 | 基准 |
| 去掉保真度 1 | -6.1 | -7.7/-7.6/-7.4 | -6.0 | -8.8/-7.0/-6.0 | Top 严重下降 |
| 去掉保真度 2 | -5.1 | -8.5/-6.5/-6.0 | -5.2 | -8.0/-7.3/-6.1 | 明显退化 |
| 去掉保真度 3 | -4.2 | -9.2/-5.9/-5.7 | -4.2 | -9.8/-7.1/-6.1 | 变差 |
| 去掉保真度 4 | -2.4 | -8.6/-4.3/-3.4 | -3.1 | -7.6/-6.7/-5.1 | 退化最严重 |
| **去掉似然项** | **-3.4** | -11.9/-9.7/-9.0 | **-3.8** | -10.9/-7.7/-6.3 | **关键组件** |
| Transformer 编解码 | -6.1 | -11.5/-9.9/-9.0 | -6.5 | -11.6/-7.6/-6.5 | 差异不大 |
| GCN 编解码 | -5.9 | -10.9/-10.1/-9.0 | -6.1 | -11.1/-7.5/-6.5 | 略差 |

### 关键发现

1. **所有保真度级别都有贡献**：去掉任何一个保真度都会导致性能下降，去掉最高保真度（ABFE）影响最大
2. **似然约束是核心**：去掉似然项后平均 ABFE 从 -6.2 骤降至 -3.4（BRD4(2)），说明限制高保真度搜索空间至关重要
3. **简单编码器足矣**：全连接网络与 Transformer/GCN 编解码器效果相当，无需复杂架构
4. **其他多保真度方法效果不佳**：MF-AL-GFN 和 MF-AL-PPO 与单保真度方法表现相近，说明成功利用多保真度需要像 MF-LAL 这样为多保真度生成量身定制的架构
5. 生成化合物的药物样性质良好：平均 QED 0.59-0.63，SAscore 3.5-3.6，多样性（1 - 平均 Tanimoto 相似度）0.81-0.83

## 亮点与洞察

- **核心创新**：将生成模型和多保真度代理模型从"分离"变为"一体化"，每个保真度级别拥有独立的潜空间和解码器，同时通过层次化映射网络共享信息
- **似然约束的思想发人深省**：只在低保真度已证明有前途的区域内探索高保真度预言机，这是一种优雅的计算预算分配策略
- **实用性强**：首次将 ABFE（金标准亲和力预测）成功整合进分子生成流程，在 7 天固定预算下即可获得显著优于基线的结果
- **查询合成 > 候选集选择**：MF-GP + ZINC250k 基线表现不佳，说明生成新查询比从固定集合中选择更有效

## 局限与展望

1. **预言机集合有限**：仅使用 4 个预言机，增加更多中间保真度级别可能进一步提升效果
2. **可合成性保证不足**：SAscore 已知是不完美的合成可行性评估指标
3. **SVGP 后验方差偏估**：远离诱导点时可能高估后验方差，导致偏向分布外分子
4. **低保真度"关卡"效应**：设计上可能遗漏低保真度得分差但高保真度表现好的骨架
5. **超参数敏感**：KLD/重构比例和多样性系数需要仔细调参
6. **评估方差**：7 天固定运行时间的实验设置可能导致结果方差较高，模型未训练到收敛

## 相关工作与启发

- **LIMO** (Eckmann et al., 2022)：同一课题组此前的工作，使用单保真度 VAE + 对接优化，是 MF-LAL 的直接前身
- **MF-AL-GFN** (Hernandez-Garcia et al., 2023)：GFlowNet + 分离多保真度 GP 代理，代表了"分离"范式的 SOTA
- **REINVENT** (Olivecrona et al., 2017)：RL 分子生成的经典方法
- **Pocket2Mol / DecompDiff**：3D 结构驱动的药物设计方法，不需要预言机但也无法避免对接评估的不准确性
- 启发：层次化潜空间 + 似然约束的思想可推广到其他多保真度科学计算场景（如材料设计、蛋白质工程）

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 新颖性 | 4 | 统一多保真度代理与生成模型的框架设计新颖；似然约束生成有独创性 |
| 技术深度 | 4 | 涉及 VAE、GP、主动学习、分子动力学多个领域的深度整合 |
| 实验充分性 | 4 | 两个真实蛋白靶标、13 个基线、充分消融、统计检验 |
| 实用价值 | 4 | 首次在生成模型中成功利用金标准 ABFE，有实际药物发现价值 |
| 写作质量 | 4 | 结构清晰，方法描述详尽，图示直观 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Amortized Active Generation of Pareto Sets](../../NeurIPS2025/computational_biology/amortized_active_generation_of_pareto_sets.md)
- [\[NeurIPS 2025\] Towards Unified and Lossless Latent Space for 3D Molecular Latent Diffusion Modeling](../../NeurIPS2025/computational_biology/towards_unified_and_lossless_latent_space_for_3d_molecular_latent_diffusion_mode.md)
- [\[NeurIPS 2025\] Manipulating 3D Molecules in a Fixed-Dimensional E(3)-Equivariant Latent Space](../../NeurIPS2025/computational_biology/manipulating_3d_molecules_in_a_fixed-dimensional_e3-equivariant_latent_space.md)
- [\[NeurIPS 2025\] PROSPERO: Active Learning for Robust Protein Design Beyond Wild-Type Neighborhood](../../NeurIPS2025/computational_biology/prospero_active_learning_for_robust_protein_design_beyond_wild-type_neighborhood.md)
- [\[NeurIPS 2025\] GFlowNets for Learning Better Drug-Drug Interaction Representations](../../NeurIPS2025/computational_biology/gflownets_for_learning_better_drug-drug_interaction_representations.md)

</div>

<!-- RELATED:END -->
