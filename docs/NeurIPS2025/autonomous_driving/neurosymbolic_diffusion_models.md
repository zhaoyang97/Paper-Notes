---
title: >-
  [论文解读] Neurosymbolic Diffusion Models
description: >-
  [NeurIPS 2025][自动驾驶][neurosymbolic] 本文提出神经符号扩散模型（NeSyDM），通过将离散掩码扩散模型与符号程序结合，突破了传统神经符号预测器中概念条件独立假设的限制，在保持可扩展性的同时建模概念间依赖关系和不确定性，在视觉推理和自动驾驶任务上取得了 SOTA 准确率和校准性能。
tags:
  - "NeurIPS 2025"
  - "自动驾驶"
  - "neurosymbolic"
  - "扩散模型"
  - "视觉推理"
---

# Neurosymbolic Diffusion Models

**会议**: NeurIPS 2025  
**arXiv**: [2505.13138](https://arxiv.org/abs/2505.13138)  
**代码**: 无  
**领域**: 自动驾驶  
**关键词**: neurosymbolic, 扩散模型, discrete diffusion, 视觉推理, 自动驾驶

## 一句话总结

本文提出神经符号扩散模型（NeSyDM），通过将离散掩码扩散模型与符号程序结合，突破了传统神经符号预测器中概念条件独立假设的限制，在保持可扩展性的同时建模概念间依赖关系和不确定性，在视觉推理和自动驾驶任务上取得了 SOTA 准确率和校准性能。

## 背景与动机

1. **神经符号预测范式的核心问题**：现有神经符号（NeSy）预测器通过神经网络提取符号概念，再利用符号程序推理输出标签。但训练只有输入-输出对，概念作为隐变量无标注，存在概念与真实语义不一致的"推理捷径"（Reasoning Shortcuts, RS）风险。

2. **条件独立假设的根本缺陷**：绝大多数 NeSy 预测器假设概念在给定输入下条件独立，即 $p_\theta(\mathbf{c}|\mathbf{x}) = \prod_i p_\theta(c_i|\mathbf{x})$。这一假设虽然使加权模型计数（WMC）高效，但理论上已被证明无法同时表达正确的不确定性并最大化似然。

3. **推理捷径导致OOD泛化失败**：当数据和程序允许多种概念赋值时，独立模型只能确定性地选择一种映射，无法对所有一致的概念方案分配概率，导致过度自信和分布外泛化能力差。

4. **现有替代方案扩展性不足**：混合模型和概率电路需要知识编译（最坏情况指数时间），难以扩展到高维问题；自回归模型的边际化不与条件分解交换，计算复杂度高。

5. **掩码扩散模型的契合性**：掩码扩散模型（MDM）在每个去噪步骤使用局部条件独立假设，但全局可以建模依赖关系。这与 NeSy 预测器的独立假设高度兼容，可直接复用现有 NeSy 的高效推理机制。

6. **自动驾驶等真实场景需求**：在 BDD-OIA 等自动驾驶任务中，模型需要从行车记录仪图像中提取高层概念（如行人、交通灯状态），并通过规则推理允许的驾驶动作，概念校准和不确定性量化对安全至关重要。

## 方法详解

### 核心框架：NeSyDM

NeSyDM 将掩码扩散模型（MDM）整合到 NeSy 预测器中，具体扩展包含三个方面：

1. **条件化输入**：去掩模型 $p_\theta(\tilde{\mathbf{c}}^0 | \mathbf{c}^t, \mathbf{x})$ 以输入 $\mathbf{x}$ 为条件
2. **联合建模概念与输出**：同时在概念 $\mathbf{c}$ 和输出 $\mathbf{y}$ 上定义扩散过程，概念为潜变量
3. **符号程序反馈**：通过程序 $\varphi$ 将概念映射到输出，提供可微分梯度信号

### 前向过程

采用连续时间掩码扩散，前向过程逐步将数据 $\mathbf{c}^0$ 掩码为 $\mathbf{c}^t$：

$$q(\mathbf{c}^t | \mathbf{c}^s) = \prod_{i=1}^C \frac{\alpha_t}{\alpha_s} \mathbb{1}[c_i^t = c_i^s] + (1 - \frac{\alpha_t}{\alpha_s})\mathbb{1}[c_i^t = \text{m}]$$

每个维度以概率 $1 - \alpha_t/\alpha_s$ 被掩码，一旦掩码则保持掩码状态。

### 损失函数（NELBO）

NeSyDM 的损失由三部分组成：

- **概念去掩损失 $\mathcal{L}_\mathbf{c}$**：与标准 MDM 类似，从变分分布采样 $\mathbf{c}^0$，部分掩码后要求模型重构
- **输出去掩损失 $\mathcal{L}_\mathbf{y}$**：每个输出维度独立计算 WMC，利用概念去掩模型的条件独立性实现高效计算
- **变分熵 $\mathcal{L}_{H[q]}$**：最大化变分分布的熵，鼓励覆盖所有与输入-输出一致的概念

### 变分后验与梯度估计

由于直接从约束后验采样是 NP-hard 的，采用松弛约束 $r_\beta$ 近似采样：从 $p_\theta$ 采样 $K$ 次，选择违反最少约束的样本。梯度优化使用 REINFORCE Leave-One-Out（RLOO）估计器，将 WMC 问题分解为 $Y$ 个独立子问题。

### 推理

使用多数投票策略：采样 $L$ 个概念 $\mathbf{c}_l^0$，通过程序 $\varphi$ 计算输出，取最频繁的输出作为预测。

## 实验结果

### 实验一：可扩展性（视觉路径规划）

| 方法 | 12×12 准确率 | 30×30 准确率 |
|------|------------|------------|
| I-MLE（连续代价） | 97.20±0.5 | 93.70±0.6 |
| EXAL | 94.19±1.74 | 80.85±3.83 |
| A-NeSI | 94.57±2.27 | 17.13±16.32 |
| A-NeSI+RL | 98.96±1.33 | 67.57±36.76 |
| **NeSyDM（本文）** | **99.41±0.06** | **97.40±1.23** |

在 30×30 网格（$5^{900}$ 组合空间）上，NeSyDM 以 97.40% 准确率大幅超越所有基线。A-NeSI 因独立假设在高维问题上崩溃（仅 17.13%），而 NeSyDM 的方差极低（0.06 vs 1.33），展示了更高的可靠性。

### 实验二：推理捷径感知（RSBench）

| 方法 | MNIST Half Acc_c(ID)↑ | ECE_c(ID)↓ | MNIST E-O ECE_c(ID)↓ |
|------|---------------------|------------|---------------------|
| PNP（独立） | 42.76±0.14 | 69.40±0.35 | 81.04±1.15 |
| SL（独立） | 42.88±0.09 | 70.61±0.18 | 82.18±1.57 |
| BEARS（集成） | 43.26±0.75 | 36.81±0.17 | 28.82±2.19 |
| NeSyDM（条件熵） | **71.16±1.77** | **4.18±2.56** | **2.70±1.21** |

NeSyDM 在概念准确率和校准误差（ECE）上全面领先。条件熵变体的 ECE 仅为 4.18%，远低于独立模型的 69-70% 和 BEARS 的 36.81%，表明 NeSyDM 能有效感知推理捷径。在 BDD-OIA 自动驾驶任务上，NeSyDM 同时提升了输出预测性能和概念校准。

## 亮点

- **理论贡献扎实**：证明了 MDM 的连续时间 NELBO 可扩展到非因子化分布，为 NeSy 之外的 MDM 架构也提供了理论基础
- **极强的可扩展性**：在 30×30 路径规划（900 维离散空间）上实现 97.4% 准确率，比最佳基线高 27 个百分点
- **不确定性量化出色**：ECE 校准误差从 70% 级别降至 4%，使模型可在主动学习等场景中可靠部署
- **架构优雅**：复用现有 NeSy 预测器的神经网络仅需额外条件化当前掩码状态，实现成本低

## 局限性

- **变分熵估计为有偏近似**：使用无条件或 1-step 近似而非精确变分熵，理论保证有限
- **推理速度较慢**：多步扩散采样 + 多数投票策略需要多次网络前向传播，比独立模型推理慢数个数量级
- **损失权重超参数敏感**：三个损失组件的权重 $\gamma_\mathbf{c}, \gamma_\mathbf{y}, \gamma_H$ 对性能影响关键，消融实验表明不当设置会显著降低效果
- **RLOO 梯度估计在高维问题中仍有方差**：当概率空间极大时，采样到一致概念的概率低，梯度信号可能不足

## 相关工作对比

### vs. BEARS（Marconato et al., 2024）

BEARS 通过独立分布的集成来实现 RS 感知，需要知识编译构建逻辑电路且每个混合组件需预测额外参数。NeSyDM 通过扩散过程自然建模依赖关系，无需显式电路编译，在高维问题上扩展性远优于 BEARS（BEARS 在 30×30 路径规划上不可行）。但 BEARS 的单步推理更快。

### vs. A-NeSI（van Krieken et al., 2023）

A-NeSI 是高效的近似 NeSy 方法但保留独立假设，在低维问题上与 NeSyDM 性能相当（MNIST 加法：92.56 vs 92.49），但在高维路径规划上崩溃（17.13% vs 97.40%）。NeSyDM 的优势主要体现在需要概念间依赖建模的复杂任务上。

### vs. I-MLE（Niepert et al., 2021）

I-MLE 使用连续代价预测而非离散 NeSy 框架，在 30×30 路径规划上达到 93.70%。NeSyDM 虽使用离散概念仍超越 I-MLE（97.40%），且提供可解释的概念提取和不确定性量化，而 I-MLE 不具备这些能力。

## 评分

| 维度 | 评分 | 说明 |
|------|------|------|
| 新颖性 | ⭐⭐⭐⭐ | 首次将离散扩散模型引入 NeSy 预测器，理论与方法均有创新 |
| 技术深度 | ⭐⭐⭐⭐⭐ | 严谨的连续时间 NELBO 推导、非因子化扩展定理、可扩展梯度估计 |
| 实验充分度 | ⭐⭐⭐⭐ | 涵盖合成与真实任务、多基线、10 seeds 统计检验，但缺少推理效率对比 |
| 实用价值 | ⭐⭐⭐ | 自动驾驶等安全关键场景有实际意义，但推理开销和超参敏感性限制了落地 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Optimizing Diffusion Models for Joint Trajectory Prediction and Controllable Generation](../../ECCV2024/autonomous_driving/optimizing_diffusion_models_for_joint_trajectory_prediction_and_controllable_gen.md)
- [\[ICCV 2025\] Distilling Diffusion Models to Efficient 3D LiDAR Scene Completion](../../ICCV2025/autonomous_driving/distilling_diffusion_models_to_efficient_3d_lidar_scene_completion.md)
- [\[NeurIPS 2025\] CymbaDiff: Structured Spatial Diffusion for Sketch-based 3D Semantic Urban Scene Generation](cymbadiff_structured_spatial_diffusion_for_sketch-based_3d_semantic_urban_scene_.md)
- [\[NeurIPS 2025\] Towards Foundational LiDAR World Models with Efficient Latent Flow Matching](towards_foundational_lidar_world_models_with_efficient_latent_flow_matching.md)
- [\[NeurIPS 2025\] RAW2Drive: Reinforcement Learning with Aligned World Models for End-to-End Autonomous Driving](raw2drive_reinforcement_learning_with_aligned_world_models_for_end-to-end_autono.md)

</div>

<!-- RELATED:END -->
