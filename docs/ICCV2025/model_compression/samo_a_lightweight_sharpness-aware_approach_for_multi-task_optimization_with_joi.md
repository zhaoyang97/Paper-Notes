---
title: >-
  [论文解读] SAMO: A Lightweight Sharpness-Aware Approach for Multi-Task Optimization with Joint Global-Local Perturbation
description: >-
  [ICCV 2025][模型压缩][多任务学习] 提出 SAMO，一种轻量级锐度感知多任务优化方法，通过全局-局部联合扰动缓解任务梯度冲突，并利用零阶梯度近似和层级归一化大幅降低计算开销。 多任务学习（MTL）旨在让一个模型同时学习多个任务，共享知识以提升数据效率和泛化能力。然而，MTL 面临的核心挑战是任务冲突：不同任务的…
tags:
  - "ICCV 2025"
  - "模型压缩"
  - "多任务学习"
  - "锐度感知最小化"
  - "梯度冲突"
  - "零阶梯度估计"
  - "层级归一化"
---

# SAMO: A Lightweight Sharpness-Aware Approach for Multi-Task Optimization with Joint Global-Local Perturbation

**会议**: ICCV 2025  
**arXiv**: [2507.07883](https://arxiv.org/abs/2507.07883)  
**代码**: [GitHub](https://github.com/OptMN-Lab/SAMO)  
**领域**: 模型压缩  
**关键词**: 多任务学习, 锐度感知最小化, 梯度冲突, 零阶梯度估计, 层级归一化

## 一句话总结

提出 SAMO，一种轻量级锐度感知多任务优化方法，通过全局-局部联合扰动缓解任务梯度冲突，并利用零阶梯度近似和层级归一化大幅降低计算开销。

## 研究背景与动机

多任务学习（MTL）旨在让一个模型同时学习多个任务，共享知识以提升数据效率和泛化能力。然而，MTL 面临的核心挑战是**任务冲突**：不同任务的梯度在方向或幅度上可能存在矛盾，导致简单的梯度平均会被某个任务主导，影响整体性能。

现有方法主要分两类应对：(1) **梯度操纵方法**（如 MGDA、CAGrad、FairGrad），通过调整梯度方向/权重来寻找折中更新方向；(2) **架构设计方法**（如 MoE、软参数共享），通过模型结构减少冲突。但这些方法大多忽略了损失景观形状的作用。

**锐度感知最小化（SAM）** 在单任务中广泛应用，通过同时最小化损失值和损失景观的"锐度"来提升泛化。作者通过实验发现了一个关键洞察：**SAM 能有效缓解 MTL 中的任务冲突**。具体表现为：

- SAM 使模型收敛到更平坦的区域，在该区域中一个任务目标的变化不会显著影响其他任务
- SAM 显著提高了任务梯度之间的余弦相似度（从负值提升为正值）
- 损失景观的锐度指标 $\lambda_{\max}$ 和 $\lambda_{\max}/\lambda_5$ 均大幅下降

然而，将 SAM 引入 MTL 面临两大挑战：(1) **全局信息（平均梯度）和局部信息（单任务梯度）对 SAM 都有帮助**，但如何有效组合它们尚不清楚——在不同方法/数据集上，G-SAM 和 L-SAM 各有优劣；(2) 直接计算每个任务的梯度用于局部扰动需要 $K$ 次额外反向传播，**计算开销巨大**。唯一的前期工作 F-MTL 将 SAM 应用于每个单独任务，但带来了 $K$ 倍的反向传播成本和双倍的内存需求。

## 方法详解

### 整体框架

SAMO 在标准 MTL 梯度操纵方法的基础上增加了一个轻量级的锐度感知模块。核心思想是：先计算全局-局部联合扰动，在扰动后的参数点上计算梯度，再将扰动后的梯度送入任意现有的梯度操纵方法（如 FairGrad）。整体流程：计算平均损失梯度 → 近似各任务局部梯度 → 组合全局-局部扰动 → 在扰动参数处计算新梯度 → 送入 MTL 方法得到更新方向。

### 关键设计

1. **全局-局部联合扰动**：SAMO 将每个任务 $i$ 的扰动定义为全局梯度和局部梯度的加权平均：

$$\hat{\epsilon}_i(\theta) = \rho \frac{\alpha \nabla_\theta l_0(\theta) + (1-\alpha) \nabla_\theta l_i(\theta)}{\|\alpha \nabla_\theta l_0(\theta) + (1-\alpha) \nabla_\theta l_i(\theta)\|}$$

其中 $\alpha \in [0,1]$ 平衡全局与局部信息。全局扰动捕获任务间的正迁移，局部扰动保留任务特异性。这样每个任务在扰动方向上既考虑了共享模式，又关注了自身的损失景观。设计动机来源于实验发现：G-SAM 和 L-SAM 各有优势，联合使用能取得更好的平衡。

2. **零阶梯度近似（SPSA）**：为了避免计算 $K$ 个任务的反向传播，SAMO 使用 **仅需前向传播** 的随机扰动同时近似（SPSA）估计局部梯度：

$$\hat{\nabla}_\theta l_i(\theta) \approx \frac{l_i(\theta + \mu z_i) - l_i(\theta - \mu z_i)}{2\mu} z_i$$

其中 $z_i$ 是标准高斯采样的随机向量，$\mu$ 是微小扰动因子。这种近似只需要 $2K$ 次前向传播（替代 $K$ 次反向传播），而前向传播成本 $C_f$ 远小于反向传播成本 $C_b$。这一设计灵感来自参数高效微调（PEFT）：将全局扰动视为"基座"，局部扰动作为"适配器"。

3. **层级归一化策略**：零阶梯度估计的方差可能很大，导致训练不稳定。SAMO 提出了逐层归一化——将估计的局部梯度在每一层上调整为与全局梯度相同的幅度：

$$\hat{\nabla}_\theta l_i(\theta^d) \leftarrow \hat{\nabla}_\theta l_i(\theta^d) \frac{\|\nabla_\theta l_0(\theta^d)\|}{\|\hat{\nabla}_\theta l_i(\theta^d)\|}$$

其中 $\theta^d$ 表示网络第 $d$ 层的参数。这保证了零阶估计的方向信息被保留，同时幅度与精确梯度一致，避免了因方差波动导致的优化不稳定。

### 损失函数 / 训练策略

SAMO 不引入额外的损失函数，而是作为一个**即插即用模块**集成到现有 MTL 方法中。训练策略完全复用原方法的设置。与 F-MTL 相比，SAMO 的额外计算开销仅为 $C_b + 2KC_f$（1 次反向传播 + $2K$ 次前向传播），而 F-MTL 需要 $KC_b + C_{gm}$（$K$ 次反向传播 + 梯度操纵的开销）。由于 $C_f \ll C_b$，SAMO 的时间开销与仅使用全局 SAM（G-SAM）相当。

超参数方面：扰动步长 $\rho$ 在 $\{0.01, 0.05, 0.1, 0.5\}$ 中搜索；全局-局部权重 $\alpha$ 在 $\{0.3, 0.5, 0.7\}$ 中选择；零阶扰动因子 $\mu$ 设为 $0.01$。

## 实验关键数据

### 主实验

**Cityscapes（2任务：语义分割 + 深度估计）**

| 方法 | mIoU ↑ | Pix Acc ↑ | Abs Err ↓ | Rel Err ↓ | Δm% ↓ |
|------|--------|-----------|-----------|-----------|--------|
| STL | 74.01 | 93.16 | 0.0125 | 27.77 | — |
| FairGrad | 74.10 | 93.03 | 0.0135 | 29.92 | 3.90 |
| F-MTL (最佳) | 73.77 | 93.12 | 0.0129 | 27.44 | 0.67 |
| **SAMO-FairGrad** | **74.37** | **93.14** | **0.0129** | **26.30** | **-0.62** |

**NYU-v2（3任务：分割 + 深度 + 法线）**

| 方法 | mIoU ↑ | Abs Err ↓ | Angle Dist ↓ | Δm% ↓ |
|------|--------|-----------|--------------|--------|
| FairGrad | 38.80 | 0.5572 | 24.55 | -4.96 |
| F-MTL (最佳) | 40.42 | 0.5389 | 25.03 | -4.77 |
| **SAMO-FairGrad** | **39.05** | **0.5359** | **24.43** | **-6.55** |

### 消融实验

| 配置 | Cityscapes Δm% | NYU-v2 Δm% | CelebA Δm% | 说明 |
|------|----------------|-------------|-------------|------|
| G-SAM-MGDA | 7.51 | -0.23 | 11.78 | 仅全局信息 |
| L-SAM-MGDA | 11.94 | 0.01 | 8.47 | 仅局部信息 |
| SAMO-MGDA | 4.30 | -2.19 | 9.59 | 联合全局-局部（本文） |
| G-SAM-FairGrad | 0.93 | -5.70 | 0.41 | 仅全局信息 |
| L-SAM-FairGrad | 1.01 | -5.42 | -0.42 | 仅局部信息 |
| SAMO-FairGrad | **-0.62** | **-6.55** | **-0.74** | 联合全局-局部（本文） |

### 关键发现

- SAMO 在所有数据集上均一致性地提升了 LS、MGDA、FairGrad 三种基线方法
- 联合全局-局部扰动始终优于单独使用全局或局部扰动
- 在 CelebA（40任务）和 QM9（11任务）上仍保持一致改进，证明方法的可扩展性
- Office-Home 多输入场景下 SAMO 同样有效，表明方法不局限于共享输入设置
- 运行时间对比：SAMO 在 Cityscapes/NYU-v2 上仅比基线多 2-6% 时间，远低于 F-MTL 的 80%+ 额外开销

## 亮点与洞察

- **SAM 缓解任务冲突的机理发现**是重要贡献：通过 Hessian 频谱分析和余弦相似度可视化，定量验证了 SAM 引导模型到平坦区域等价于缓解任务冲突
- **零阶梯度近似 + 层级归一化**的组合很巧妙——保留了方向信息同时控制了方差
- 方法的**即插即用特性**使其能与任意现有梯度操纵方法组合，实用价值高
- 二维合成问题的可视化分析非常直观地展示了 SAM 如何改变优化轨迹

## 局限与展望

- 零阶梯度估计本质上是一个近似，在高维空间中方向精度有限，单次采样可能引入较大噪声
- $\alpha$ 作为固定超参数需要手动调节，可以探索自适应调整策略
- 实验未涉及大规模预训练模型（如 ViT-Large），在参数量更大时零阶估计的效果需进一步验证
- 与 SAM 的高级变体（如 ASAM、LookSAM）的组合未探索

## 相关工作与启发

- **F-MTL**（Phan et al.）是最相关的前期工作，直接将 SAM 应用于每个任务，但计算成本过高
- **SPSA 梯度估计**来自随机优化文献，在 MTL 场景下首次被用于近似局部扰动
- 启发：零阶优化在非 MTL 领域（如 prompt tuning）也有应用，SAMO 的方法论可能适用于其他需要多目标优化的场景

## 评分

- **新颖性**: ⭐⭐⭐⭐ 将 SAM 引入 MTL 的全局-局部联合范式是新的，但零阶近似本身是已有技术
- **实验充分度**: ⭐⭐⭐⭐⭐ 5 个数据集覆盖分割/分类/回归/多输入场景，消融完整
- **写作质量**: ⭐⭐⭐⭐ 逻辑清晰，分析到位，图表丰富
- **价值**: ⭐⭐⭐⭐ 即插即用的实用性强，轻量级设计有工程价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] JamMa: Ultra-lightweight Local Feature Matching with Joint Mamba](../../CVPR2025/model_compression/jamma_ultra-lightweight_local_feature_matching_with_joint_mamba.md)
- [\[ICML 2025\] Joker: Joint Optimization Framework for Lightweight Kernel Machines](../../ICML2025/model_compression/joker_joint_optimization_framework_for_lightweight_kernel_machines.md)
- [\[ACL 2025\] DAC: A Dynamic Attention-aware Approach for Task-Agnostic Prompt Compression](../../ACL2025/model_compression/dac_prompt_compression.md)
- [\[ICCV 2025\] Task Vector Quantization for Memory-Efficient Model Merging](task_vector_quantization_for_memory-efficient_model_merging.md)
- [\[CVPR 2026\] Efficiency Follows Global-Local Decoupling](../../CVPR2026/model_compression/efficiency_follows_global-local_decoupling.md)

</div>

<!-- RELATED:END -->
