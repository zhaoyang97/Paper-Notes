---
title: >-
  [论文解读] Investigating Data Pruning for Pretraining Biological Foundation Models at Scale
description: >-
  [AAAI 2026][计算生物][数据剪枝] 提出一个基于影响函数的后验数据剪枝框架，通过子集自影响估计（Subset-Based Self-Influence）和两种选择策略（Top-k Influence 和 Coverage-Centric Influence），在超过 99% 的极端剪枝率下，用仅 0.2M 序列预训练的 RNA-FM 在多项下游任务上媲美甚至超越用 23M 序列训练的完整模型，揭示了生物序列数据集的巨大冗余性。
tags:
  - "AAAI 2026"
  - "计算生物"
  - "数据剪枝"
  - "生物基础模型"
  - "影响函数"
  - "核心集选择"
  - "RNA-FM"
  - "ESM"
  - "蛋白质语言模型"
---

# Investigating Data Pruning for Pretraining Biological Foundation Models at Scale

**会议**: AAAI 2026  
**arXiv**: [2512.12932](https://arxiv.org/abs/2512.12932)  
**代码**: [github.com/victor-yifanwu/bio-coreset](https://github.com/victor-yifanwu/bio-coreset)  
**领域**: 医学图像 / 生物信息 / 基础模型  
**关键词**: 数据剪枝, 生物基础模型, 影响函数, 核心集选择, RNA-FM, ESM, 蛋白质语言模型

## 一句话总结
提出一个基于影响函数的后验数据剪枝框架，通过子集自影响估计（Subset-Based Self-Influence）和两种选择策略（Top-k Influence 和 Coverage-Centric Influence），在超过 99% 的极端剪枝率下，用仅 0.2M 序列预训练的 RNA-FM 在多项下游任务上媲美甚至超越用 23M 序列训练的完整模型，揭示了生物序列数据集的巨大冗余性。

## 研究背景与动机

**领域现状**：生物基础模型（BioFMs）如 RNA-FM（23M RNA序列）和 ESM（28亿蛋白质序列）在结构预测、功能注释等任务上表现出色，但训练成本极高（RNA-FM：8×A100 训练 30天），严重限制了学术实验室的可复现性和可及性。

**数据剪枝在生物领域的空白**：
   - CV/NLP 领域已有大量数据剪枝工作，但 BioFMs 预训练几乎未被探索
   - 基于训练动力学的方法（EL2N、AUM）：需要完整训练过程，对 BioFMs 不可行
   - 基于局部密度的方法：需要两两相似度计算，数百万级序列不可扩展
   - 基于影响函数的方法：需要计算全训练集 Hessian 逆，参数量达数亿时不可行

**核心问题**：能否在不访问完整训练过程的前提下，通过后验（post-hoc）方法找到信息量最大的训练子集？

**切入角度**：利用影响函数理论，在小子集上近似全训练集的曲率信息，高效估计每个样本的重要性。

## 方法详解

### 整体框架

三步走：
1. **影响分数估计**：基于子集的自影响函数
2. **核心集选择**：Top I 或 CCI 策略
3. **从头预训练**：在选出的核心集上重新训练 BioFMs

### 子集自影响函数（Subset-Based Self-Influence）

**经典影响函数回顾**：训练样本 $z_{tr}$ 对验证样本 $z_{val}$ 的影响为：
$$\mathcal{I}(z_{tr}; z_{val}) = g_{z_{val}}^\top H_{\theta^*}^{-1} g_{z_{tr}}$$

其中 $g$ 为梯度，$H$ 为 Hessian。但计算 $H^{-1}$ 在大模型上不可行。

**关键创新 — 子集近似**：

**Assumption 1**：在随机采样的子集 $D_{sub}$ 上训练得到 $\tilde{\theta}$，模型在子集上近似最优。

基于此假设，用子集 Hessian $\tilde{H}_{sub}$ 替代全训练集 Hessian $H_{tr}$：

$$\mathcal{I}(z_{tr}, D_{sub}) = \tilde{g}_{z_{tr}}^\top \tilde{H}_{sub}^{-1} \tilde{g}_{z_{tr}}$$

**理论支撑（Proposition 1）**：在大模型损失景观平坦的条件下（已被最近研究证实），子集曲率可以充分近似全训练集曲率。

**进一步加速 — 对角经验 Fisher 矩阵近似**：

$$\tilde{H}_{sub}^{-1} \approx \text{diag}(\tilde{F}_{sub})^{-1}$$

其中 $\text{diag}(\tilde{F}_{sub}) = \frac{1}{M}\sum_{m=1}^M \tilde{g}_{z_m} \odot \tilde{g}_{z_m}$

计算复杂度从 $O(M \cdot d^2 + d^3)$ 降至 $O(M \cdot d)$，对数十亿参数模型可行。

**实践要点**：在子集上做一个 epoch 的轻量微调即可满足 Assumption 1，成本可忽略。

### 两种核心集选择策略

1. **Top-k Influence (Top I)**：直接选影响分数最高的 $k$ 个样本
    - 优先保留对模型参数影响最大的样本
    - 理论上对应最有信息量的数据点

2. **Coverage-Centric Influence (CCI)**：在影响分数分布上分层采样
    - 保持"简单"和"困难"样本的均衡分布
    - 受 Sorscher et al. 2022 启发：极端剪枝下，仅保留最难样本会导致过拟合
    - 分层采样确保数据分布的覆盖性

### 实验设置

- **极端剪枝率**：仅保留 0.2M 序列（RNA: ~1% of 23M；蛋白质: ~4.4% of 4.5M）
- 在核心集上从头训练 10 个 epoch
- 评估多种下游任务

## 实验

### RNA-FM 实验

#### 功能与工程预测任务

| 方法 | 数据量 | TypeCls ACC(%) | TypeCls F1(%) | Modif AUC(%) | CRI-On SC(%) | CRI-On MSE↓ |
|------|--------|----------------|---------------|--------------|--------------|-------------|
| RNA-FM | 23M | 91.93 | 91.87 | 94.98 | 31.87 | .0118 |
| Random | 2M | 82.21 | 82.01 | 92.82 | 26.72 | .0158 |
| Random | 0.2M | 82.15 | 81.97 | 91.86 | 26.67 | .0161 |
| **Top I** | 0.2M | 82.51 | 82.53 | 93.20 | 27.08 | .0149 |
| **CCI** | 0.2M | **82.88** | **83.12** | **93.86** | **32.90** | **.0135** |

- CCI 在 CRISPR On-Target 上**超越了完整 RNA-FM**（23M 序列）！
- 0.2M 的核心集竟然优于 2M 的随机选择

#### 结构与交互预测任务

| 方法 | 二级结构 F1(%) | 距离图 SC(%) | 接触图 Top-1.0L(%) | RBP交互 ACC(%) |
|------|---------------|-------------|-------------------|---------------|
| RNA-FM | 62.20 | 89.21 | 93.93 | 72.47 |
| Random 0.2M | 55.60 | 84.90 | 94.18 | 69.65 |
| **Top I** | **57.05** | **86.47** | **94.36** | **71.25** |
| CCI | 56.36 | 85.59 | 94.20 | 69.46 |

- 结构相关任务中 **Top I 优于 CCI**：高影响样本编码了更丰富的结构信息
- 接触图预测中 Top I 甚至**超越了完整 RNA-FM**

### ESM-C 蛋白质实验（泛化性验证）

| 方法 | 数据量 | 定位 ACC(%) | 二级结构 ACC(%) | PPI MAE↓ | PPI RMSE↓ |
|------|--------|------------|----------------|----------|-----------|
| ESM-C | 2.78B | 91.63 | 86.10 | 1.92 | 2.44 |
| Random | 2M | 75.76 | 67.20 | 2.39 | 2.87 |
| Random | 0.2M | 73.64 | 66.18 | 2.51 | 3.01 |
| **Top I** | 0.2M | 77.13 | 69.34 | **2.06** | **2.64** |
| **CCI** | 0.2M | **79.25** | **71.48** | 2.14 | 2.69 |

- Top I 和 CCI 在 0.2M 下均超过 2M Random，再次证明蛋白质数据大量冗余
- CCI 在蛋白质场景表现更好

### 消融实验：适应（fine-tuning）的必要性

| 变体 | Modif AUC(%) | 距离图 SC(%) |
|------|--------------|-------------|
| Top I (w/o ft) | 92.94 | 84.13 |
| CCI (w/o ft) | 93.31 | 84.95 |
| **Top I** | **93.20** | **86.47** |
| **CCI** | **93.86** | **85.59** |

在子集上做轻量微调后再计算影响分数，结果一致更好，验证了 Assumption 1 的重要性。

## 亮点与洞察

1. **揭示了生物训练数据的巨大冗余性**：不到 1% 的数据就能达到接近甚至超越完整模型的性能
2. **后验框架无需训练过程**：只需预训练模型权重和一小批子集微调，对已发布但未公开训练细节的模型也适用
3. **Top I vs CCI 的互补性**：
    - CCI 擅长功能/工程预测（需要覆盖多样性）
    - Top I 擅长结构/交互预测（需要信息密度）
4. **理论推导完整**：从经典影响函数到子集近似到 Fisher 对角化，每步都有理论支撑
5. **极高的实际价值**：对学术实验室来说，能以极低成本复现 BioFMs 的训练

## 局限性

1. RNA 实验仅在 RNA-FM 上验证，未测试更大的 RNA 模型（如 Evo 2）
2. 蛋白质实验受限于资源仅用 4.5M（远小于 ESM-C 的 2.78B 训练集），核心集效果未在全尺度数据上验证
3. Assumption 1（子集上近似最优）的误差上界未定量分析
4. 对角 Fisher 近似在高度非对角 Hessian 结构时可能不准确
5. 仅在自监督预训练（MLM）场景验证，有监督微调场景的数据剪枝效果未讨论
6. 未与基于密度/多样性的方法（如 Facility Location）做直接对比

## 相关工作

- **数据剪枝**：EL2N（训练动力学）、Sorscher et al. 2022（数据密度）、D2 Pruning（difficulty-diversity 平衡）
- **影响函数**：Koh & Liang 2017（经典IF）、DataInf、TRAK（高效近似）
- **生物基础模型**：RNA-FM（Chen 2022，23M序列）、ESM-C/ESM3（Hayes 2025，2.78B序列）、Evo 2

## 评分 ⭐⭐⭐⭐

- **创新性**：⭐⭐⭐⭐ — 子集影响函数近似方法有理论贡献，首次将数据剪枝系统性地引入 BioFMs
- **实验**：⭐⭐⭐⭐ — RNA+蛋白质双模态验证，多种下游任务全面评估
- **写作**：⭐⭐⭐⭐ — 理论推导清晰，实验呈现规范
- **实用性**：⭐⭐⭐⭐⭐ — 直接降低 BioFMs 的预训练成本，对计算资源受限的研究组有巨大价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Protein Fold Classification at Scale: Benchmarking and Pretraining](../../ICML2026/computational_biology/protein_fold_classification_at_scale_benchmarking_and_pretraining.md)
- [\[ICML 2025\] SToFM: a Multi-scale Foundation Model for Spatial Transcriptomics](../../ICML2025/computational_biology/stofm_a_multi-scale_foundation_model_for_spatial_transcriptomics.md)
- [\[NeurIPS 2025\] EDBench: Large-Scale Electron Density Data for Molecular Modeling](../../NeurIPS2025/computational_biology/edbench_large-scale_electron_density_data_for_molecular_modeling.md)
- [\[AAAI 2026\] ProtSAE: Disentangling and Interpreting Protein Language Models via Semantically-Guided Sparse Autoencoders](protsae_disentangling_and_interpreting_protein_language_models_via_semantically-.md)
- [\[AAAI 2026\] Distributional Priors Guided Diffusion for Generating 3D Molecules in Low Data Regimes](distributional_priors_guided_diffusion_for_generating_3d_molecules_in_low_data_r.md)

</div>

<!-- RELATED:END -->
