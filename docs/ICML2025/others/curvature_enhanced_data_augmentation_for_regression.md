---
title: >-
  [论文解读] Curvature Enhanced Data Augmentation for Regression
description: >-
  [ICML 2025][数据增强] 提出 CEMS（Curvature-Enhanced Manifold Sampling），利用数据流形的二阶近似（曲率信息）生成合成样本，用于回归任务的数据增强，在分布内和分布外场景均取得 SOTA 或接近 SOTA 的性能。 数据增强在分类任务中已被广泛研究并取得巨大成功…
tags:
  - "ICML 2025"
  - "数据增强"
  - "回归任务"
  - "流形学习"
  - "曲率"
  - "二阶近似"
---

# Curvature Enhanced Data Augmentation for Regression

**会议**: ICML 2025  
**arXiv**: [2506.06853](https://arxiv.org/abs/2506.06853)  
**代码**: [azencot-group/CEMS](https://github.com/azencot-group/CEMS)  
**领域**: 其他  
**关键词**: 数据增强, 回归任务, 流形学习, 曲率, 二阶近似

## 一句话总结

提出 CEMS（Curvature-Enhanced Manifold Sampling），利用数据流形的二阶近似（曲率信息）生成合成样本，用于回归任务的数据增强，在分布内和分布外场景均取得 SOTA 或接近 SOTA 的性能。

## 研究背景与动机

数据增强在分类任务中已被广泛研究并取得巨大成功，但在**回归任务**中的应用相对欠缺。分类任务的标签是离散的，定义保标签变换相对容易；而回归任务的输出是连续值，如何在变换后保持输入-输出对的有效性是一个核心挑战。

现有的回归数据增强方法主要基于 mixup 系列（如 C-mixup、RegMix），通过对样本进行凸组合来生成新数据。然而 mixup 在回归任务中的效果不稳定。最近的 FOMA 方法从流形学习的角度出发，利用数据流形的**一阶近似（切空间）**来采样新点，但一阶方法在高曲率区域会偏离真实流形，生成质量较差的样本。

本文的核心动机是：**一阶近似不足以捕获复杂、弯曲的真实数据结构，而二阶近似（包含曲率/Hessian信息）能在有效性和计算成本之间取得更好的平衡。**

## 方法详解

### 整体框架

CEMS 将数据增强视为一个流形近似与采样问题。给定回归训练集 $\mathcal{D} = \{(x^i, y^i)\}_{i=1}^{N}$，将输入和标签拼接为 $z^i = [x^i, y^i] \in \mathbb{R}^D$，假设这些点位于一个内在维度 $d \ll D$ 的低维流形 $\mathcal{M}$ 上。

CEMS 的四个核心步骤：

1. **邻域提取**：对每个点 $z$，在联合输入-输出空间中找到其 $k$ 近邻 $N_z$
2. **基构造与投影**：通过 SVD 构造切空间 $\mathcal{T}_u\mathcal{M}$ 和法空间 $\mathcal{N}_u\mathcal{M}$ 的正交基 $B_u = [B_{\mathcal{T}_u}, B_{\mathcal{N}_u}]$，将邻域投影到局部坐标
3. **求解线性方程组**：利用二阶 Taylor 展开构造并求解线性系统，获得梯度 $\nabla g(u)$ 和 Hessian $H(u)$ 的估计
4. **采样与反投影**：在切空间中采样新点 $\eta$，通过二阶近似计算其法空间分量 $g(\eta)$，然后反投影回原始空间

### 关键设计

**二阶流形表示**：CEMS 的核心创新在于使用嵌入映射 $g: \mathcal{T}_u\mathcal{M} \rightarrow \mathcal{N}_u\mathcal{M}$ 的二阶 Taylor 展开：

$$g(\eta) = \eta^T \nabla g(u) + \frac{1}{2} \eta^T H(u) \eta$$

其中 $\nabla g(u)$ 是梯度（一阶项），$H(u)$ 是 Hessian（二阶项）。与 FOMA 只使用切空间投影并缩放法空间分量不同，CEMS 通过求解最小二乘问题精确估计这两个量。

**与 FOMA 的本质区别**：FOMA 对法空间分量做简单的缩放 $\tilde{g}_j = \lambda g_j$（$\lambda \in (0,1)$），不使用嵌入映射 $g$；而 CEMS 显式估计梯度和 Hessian，通过 Taylor 展开在新采样点处计算法空间分量。这使得 CEMS 在高曲率区域能更好地拟合流形结构。

**批次适配**：为提高计算效率，CEMS 对同一批次中的所有点共享邻域和基构造。具体地，对批次中的锚点 $z$ 及其邻域 $N_z$，所有 $z_j \in N_z$ 共用相同的正交基 $B_u$，仅线性求解为每个点单独计算。

**内在维度估计**：虽然 $d$ 可视为超参数，但实践中使用鲁棒估计器（Facco et al., 2017）自动确定。

**全可微**：所有步骤（SVD、最小二乘求解、采样与反投影）均可微，允许端到端训练。

### 损失函数 / 训练策略

CEMS 是一种**在线数据增强方法**：在每个训练 mini-batch 中，对每个样本 $z$ 生成对应的增强样本 $\tilde{z}$，然后使用标准的回归损失（如 MSE）在原始样本和增强样本上进行训练。

采样过程使用简单的高斯采样器：$\eta \sim \mathcal{N}(0, \sigma I_d)$，其中 $\sigma$ 是唯一的采样超参数，控制新样本离原始点的距离。

反投影公式为：

$$z_\eta = f(\eta) = B_u \cdot [\eta, g(\eta)] + z$$

**计算复杂度**：整体时间复杂度为 $\mathcal{O}(b^2 D)$，其中 $b$ 为 batch size，$D$ 为数据维度。由于 $d \ll D$，二阶方法的额外开销相比一阶方法仅为最小增量。

## 实验关键数据

### 主实验

**分布内泛化**（4 个数据集，RMSE / MAPE，↓ 更好）：

| 方法 | Airfoil RMSE | Airfoil MAPE | NO2 RMSE | NO2 MAPE | Exchange RMSE | Electricity RMSE |
|------|-------------|-------------|---------|---------|--------------|-----------------|
| ERM | 2.901 | 1.753 | 0.537 | 13.615 | 0.024 | 0.058 |
| Mixup | 3.730 | 2.327 | 0.528 | 13.534 | 0.024 | 0.058 |
| C-Mixup | 2.717 | 1.610 | 0.509 | 12.998 | 0.020 | 0.057 |
| ADA | 2.360 | 1.373 | 0.515 | 13.128 | 0.021 | 0.059 |
| FOMA | 1.471 | 0.816 | 0.512 | 12.894 | **0.013** | 0.058 |
| **CEMS** | **1.455** | **0.809** | **0.507** | **12.807** | 0.014 | **0.058** |

**分布外泛化**（5 个数据集）：

| 方法 | RCF Avg↓ | Crimes Avg↓ | SkillCraft Avg↓ | DTI Avg R↑ | Poverty Avg R↑ |
|------|---------|------------|----------------|-----------|---------------|
| ERM | 0.164 | 0.136 | 6.147 | 0.483 | 0.80 |
| C-Mixup | 0.146 | 0.123 | 5.201 | 0.498 | 0.81 |
| FOMA | 0.159 | 0.128 | - | 0.503 | 0.78 |
| **CEMS** | **0.146** | 0.128 | **5.142** | **0.511** | **0.81** |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| CEMS（一阶） | 在高曲率区域采样偏离流形 | 退化为类似 FOMA 的行为 |
| CEMS（二阶） | 高曲率区域也能准确采样 | 完整方法，Hessian 项捕获曲率 |
| 预计算 vs 在线计算 | 预计算更快但内存更高 | 在线计算牺牲速度换灵活性 |
| 点级(CEMSp) vs 批级(CEMS) | 批级共享基构造更高效 | 精度略有损失但实用性更强 |

### 关键发现

- **一阶 vs 二阶**：在 sine wave 玩具实验中直观展示了一阶方法（FOMA、CEMS 一阶模式）在高曲率处采样偏离流形，而 CEMS 的二阶近似能准确跟踪曲率
- **分布内**：CEMS 在 Airfoil 和 NO2 上取得最优，Exchange-Rate 和 Electricity 上接近最优
- **分布外**：CEMS 在 9 个测试中 6 个取得最佳结果，特别在 SkillCraft 上 Avg 改进 1%、Worst 改进 8%
- **计算开销极小**：由于内在维度 $d \ll D$，二阶计算的额外成本可忽略

## 亮点与洞察

1. **将 DA for regression 统一到流形学习框架下**：不是简单的 mixup 变体，而是从流形假设出发给出理论基础，将一阶（FOMA）和二阶（CEMS）方法纳入统一视角
2. **巧妙利用 $d \ll D$**：二阶方法通常因计算成本高而不可行，但本文利用流形的低内在维度使 Hessian 的计算规模从 $D^2$ 降到 $d^2$
3. **领域无关性**：CEMS 不依赖特定数据域（图像、时序、表格均适用），这是比 mixup 类方法更强的通用性
4. **全可微设计**：SVD 和最小二乘均可微，可无缝集成到端到端训练流程中

## 局限与展望

1. **大内在维度 $d$ 的扩展性**：线性系统可能欠定，需要 $\mathcal{O}(d^2)$ 个邻居，对大 $d$ 数据集不实用（可通过 ridge regression 缓解）
2. **SVD 内存需求**：当 $d$ 较大时可能需要完整 SVD，内存为 $\mathcal{O}(bD^2)$
3. **邻域共享假设**：批级 CEMS 假设邻域内所有点共享相同的基，这在非均匀曲率区域可能引入误差
4. **采样器过于简单**：目前使用各向同性高斯采样，未考虑流形的局部几何结构（如沿不同主曲率方向的采样幅度应不同）
5. **缺少自适应阶数选择**：不同局部区域可能适合不同阶数的近似

## 相关工作与启发

- **FOMA**（Kaufman & Azencot, 2024）：本文的直接前身，一阶流形采样，CEMS 可视为其自然推广
- **Mixup 系列**：Zhang et al. 2018 的 Mixup 及其变体（C-Mixup、ADA）在分类中有效但回归中不稳定
- **Hessian Eigenmaps**（Donoho & Grimes, 2003）：在降维中使用二阶信息，本文将类似思路用于数据增强
- **VRM 理论**：数据增强的理论基础是 Vicinal Risk Minimization，CEMS 提供了一种几何感知的 vicinal distribution

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 新颖性 | 4 | 二阶流形采样用于 DA 是新颖的，但核心数学工具已存在 |
| 理论深度 | 4 | 有完整的流形学习理论框架和采样误差分析 |
| 实验充分性 | 4 | 9 个数据集覆盖 ID/OOD，但缺少大规模深度学习场景 |
| 实用性 | 3.5 | 领域无关且额外开销小，但对大内在维度数据有限制 |
| 写作质量 | 4 | 结构清晰，理论与实验并重 |
| **总分** | **4/5** | 流形学习视角下的回归 DA 统一框架，理论扎实实验充分 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Is Linguistically-Motivated Data Augmentation Worth It?](../../ACL2025/others/is_linguistically-motivated_data_augmentation_worth_it.md)
- [\[ICML 2025\] Improved Exploration in GFlowNets via Enhanced Epistemic Neural Networks](improved_exploration_in_gflownets_via_enhanced_epistemic_neural_networks.md)
- [\[ECCV 2024\] FreeAugment: Data Augmentation Search Across All Degrees of Freedom](../../ECCV2024/others/freeaugment_data_augmentation_search_across_all_degrees_of_freedom.md)
- [\[ICCV 2025\] Adversarial Data Augmentation for Single Domain Generalization via Lyapunov Exponents](../../ICCV2025/others/adversarial_data_augmentation_for_single_domain_generalization_via_lyapunov_expo.md)
- [\[ICML 2025\] Regression for the Mean: Auto-Evaluation and Inference with Few Labels through Post-hoc Regression](regression_for_the_mean_auto-evaluation_and_inference_with_few_labels_through_po.md)

</div>

<!-- RELATED:END -->
