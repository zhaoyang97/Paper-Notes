---
title: >-
  [论文解读] Learning to Integrate Diffusion ODEs by Averaging the Derivatives
description: >-
  [NeurIPS 2025][图像生成][扩散模型加速] 提出"割线损失"(Secant Losses)家族，通过蒙特卡洛积分和Picard迭代学习扩散ODE的积分，将扩散模型的切线逐步延展为割线，在训练稳定性和少步推理之间取得优异平衡。 扩散模型虽然生成质量优秀，但推理时通常需要数百到数千次函数评估(NFE)来生成一张图…
tags:
  - "NeurIPS 2025"
  - "图像生成"
  - "扩散模型加速"
  - "割线损失"
  - "ODE积分"
  - "蒙特卡洛积分"
  - "Picard迭代"
---

# Learning to Integrate Diffusion ODEs by Averaging the Derivatives

**会议**: NeurIPS 2025  
**arXiv**: [2505.14502](https://arxiv.org/abs/2505.14502)  
**代码**: [GitHub](https://github.com/poppuppy/secant-expectation)  
**领域**: 扩散模型 / 图像生成  
**关键词**: 扩散模型加速, 割线损失, ODE积分, 蒙特卡洛积分, Picard迭代

## 一句话总结

提出"割线损失"(Secant Losses)家族，通过蒙特卡洛积分和Picard迭代学习扩散ODE的积分，将扩散模型的切线逐步延展为割线，在训练稳定性和少步推理之间取得优异平衡。

## 研究背景与动机

扩散模型虽然生成质量优秀，但推理时通常需要数百到数千次函数评估(NFE)来生成一张图，严重限制了实际应用。现有加速方案主要分两类：

**快速采样器**（DPM-Solver、UniPC等）：在NFE<10时性能急剧下降，因为数值求解器在极少步下精度不够。

**蒸馏方法**（一致性模型、对抗蒸馏等）：虽然能实现少步生成，但往往引入复杂训练流程、训练不稳定、模型坍缩或过度平滑等问题。

**核心矛盾**：快速采样器步数太少不行，蒸馏方法又太复杂不稳定。两者之间缺少一个简单、稳定且有效的中间方案。

**本文切入角度**：从几何视角出发，扩散模型学习的是PF-ODE的**切线**（瞬时变化率），而真正需要的是两个时间点之间的**割线**（平均变化率）。割线恰好是区间内所有切线的平均值，这个关系可以用蒙特卡洛积分来近似，用Picard迭代来解决训练中只能采样一个点的问题。

## 方法详解

### 整体框架

给定PF-ODE $\frac{d\boldsymbol{x}_t}{dt} = \boldsymbol{v}(\boldsymbol{x}_t, t)$，要从 $\boldsymbol{x}_t$ 跳到 $\boldsymbol{x}_s$，传统方法需要逐步数值积分。本文改为用神经网络直接建模**割线函数** $\boldsymbol{f}_\theta(\boldsymbol{x}_t, t, s)$，使得：

$$\boldsymbol{x}_s = \boldsymbol{x}_t + (s-t) \boldsymbol{f}_\theta(\boldsymbol{x}_t, t, s)$$

割线函数定义为区间内所有切线的期望：$\boldsymbol{f}(\boldsymbol{x}_t, t, s) = \mathbb{E}_{r \sim U(t,s)} \boldsymbol{v}(\boldsymbol{x}_r, r)$。

### 关键设计

1. **割线期望损失 (Secant Expectation Loss)**：
   核心观察是割线等于切线的均匀采样期望。因此可以构造损失：$\mathcal{L} = \|\boldsymbol{f}_\theta(\boldsymbol{x}_t, t, s) - \boldsymbol{v}(\boldsymbol{x}_r, r)\|^2$，其中 $r \sim U(t,s)$。但训练时只能访问 $\boldsymbol{x}_t$ 或 $\boldsymbol{x}_r$ 之一，不能同时获得两者。

2. **Picard迭代估计**：
   受Picard迭代启发，用模型自身估计缺失的那个点。有两种方式：
    - **估计内点 (EI)**：采样 $\boldsymbol{x}_t$，用 $\hat{\boldsymbol{x}}_r = \boldsymbol{x}_t + (r-t)\boldsymbol{f}_{\theta^-}(\boldsymbol{x}_t, t, r)$ 估计内部点，然后用教师模型评估 $\boldsymbol{v}(\hat{\boldsymbol{x}}_r, r)$ 作为目标。
    - **估计端点 (EE)**：采样 $\boldsymbol{x}_r$，用模型反推 $\hat{\boldsymbol{x}}_t$，直接用真实的 $\alpha_r'\boldsymbol{x}_0 + \sigma_r'\boldsymbol{z}$ 作为目标。

3. **四种变体**：

    - **SDEI**（蒸馏+估计内点）：需要教师模型，3次前向+1次反向
    - **STEI**（训练+估计内点）：不需教师，加入扩散损失正则，4次前向+2次反向
    - **SDEE**（蒸馏+估计端点）：需要教师，3次前向+1次反向
    - **STEE**（训练+估计端点）：不需教师，最轻量，仅2次前向+1次反向

4. **目标稳定性优势**：
   与一致性模型相比，割线损失的目标要么与扩散损失相同（$\alpha_t'\boldsymbol{x}_0 + \sigma_t'\boldsymbol{z}$），要么就是扩散模型本身 $\boldsymbol{v}(\boldsymbol{x}_t, t)$，不涉及模型依赖的导数项 $\frac{d}{dt}\boldsymbol{f}_{\theta^-}$，因此训练稳定性远优于一致性模型。

### 损失函数 / 训练策略

- **扩散模型初始化**：加载预训练权重，使 $\boldsymbol{f}_\theta(\boldsymbol{x}_t, t, t) = \boldsymbol{v}(\boldsymbol{x}_t, t)$，大幅加速收敛
- **STEI中的平衡因子**：$\lambda=1$ 最优，平衡扩散损失和割线损失
- **CFG嵌入**：蒸馏变体直接将CFG嵌入到损失中的 $\boldsymbol{v}$ 里；STEE则像训练扩散模型一样用随机丢弃标签+推理时CFG
- **均匀步长采样**：推理时使用均匀步长 $(t,s) = (i/N, (i-1)/N)$
- **训练仅需原始训练的1%**：50K-100K迭代 vs. SiT的7M迭代

## 实验关键数据

### 主实验 — CIFAR-10 无条件生成

| 方法 | FID↓ | 步数 | 类别 |
|------|------|------|------|
| EDM (Teacher) | 1.97 | 35 | 扩散模型 |
| DPM-Solver-v3 | 2.51 | 10 | 快速采样器 |
| LD3 | 2.38 | 10 | 快速采样器 |
| sCD | 2.52 | 2 | 一致性蒸馏 |
| ECT | 2.11 | 2 | 一致性训练 |
| IMM | 1.98 | 2 | 训练/微调 |
| **SDEI (本文)** | **2.14** | **10** | 微调 |

### 主实验 — ImageNet 256×256 类条件生成

| 方法 | FID↓ | IS↑ | 步数 |
|------|------|-----|------|
| SiT-XL/2 (Teacher) | 2.15 | 258.09 | 250 |
| IMM (4步) | 2.51 | - | 4 |
| IMM (8步) | 1.99 | - | 8 |
| **STEI (4步)** | **2.78** | **269.87** | **4** |
| **STEI+guid.int. (4步)** | **2.27** | **273.76** | **4** |
| **STEE (8步)** | **2.33** | **274.47** | **8** |
| **STEE+guid.int. (8步)** | **1.96** | **275.81** | **8** |
| **STEI (1步)** | **7.12** | **241.75** | **1** |

### 消融实验

| 配置 | FID↓ | 说明 |
|------|------|------|
| $\lambda=0.1$ | 3.96 | 扩散损失权重过小 |
| $\lambda=0.5$ | 3.15 | 次优 |
| $\lambda=1.0$ | **2.84** | 最佳平衡 |
| $\lambda=2.0$ | 3.96 | 扩散损失开始恶化 |
| 离散t采样，仅生成 | **3.23** | 最佳固定步数 |
| 连续t采样，仅生成 | 4.29 | 灵活步数但性能下降 |
| 连续t采样，生成+反演 | 5.47 | 模型容量被分摊 |

### 关键发现

- 一致性模型在ImageNet-256上训练发散，而割线损失始终稳定快速收敛
- 估计内点变体普遍优于估计端点，因为输入更干净、误差路径更短
- 从头训练3000K迭代可达8步FID 2.68，验证了可扩展性
- 1步生成时STEI (FID 7.12)甚至超过IMM和Shortcut Models的4步

## 亮点与洞察

- **理论优雅**：从蒙特卡洛积分和Picard迭代自然推导出损失函数，几何直觉清晰（切线→割线）
- **训练极度稳定**：因为目标与扩散模型相同，避免了一致性模型的导数项不稳定问题
- **实现简单**：与训练扩散模型高度并行，不需额外判别器、分数蒸馏等复杂组件
- **训练效率高**：仅需教师模型1%的训练量

## 局限与展望

- 1步和8步之间仍有显著性能差距，大步跨越仍然困难
- ImageNet性能依赖CFG，CFG与割线损失的理论关系未充分探索
- 需要训练数据，数据有限场景下可能受限
- 割线精度保证是局部的，全局扩展依赖bootstrapping

## 相关工作与启发

- 与一致性模型互为"微分vs积分"的对偶关系，用积分避免了导数带来的不稳定
- 与Rectified Flow的多时间训练策略有联系，但割线损失更强调局部精度
- 可以看作快速采样器和蒸馏之间的有效折中方案

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 从积分视角重新定义少步生成问题，与一致性模型形成优美对偶
- **实验充分度**: ⭐⭐⭐⭐ — CIFAR-10和ImageNet充分，但缺少文本到图像的验证
- **写作质量**: ⭐⭐⭐⭐⭐ — 几何直觉、理论推导和实验验证层层递进，非常清晰
- **价值**: ⭐⭐⭐⭐ — 提供了一个稳定、简洁的少步扩散方案，训练友好

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Information Theoretic Learning for Diffusion Models with Warm Start](information_theoretic_learning_for_diffusion_models_with_warm_start.md)
- [\[ICML 2026\] Mitigating the Contractivity Trap in Diffusion ODEs via Stein Stabilization](../../ICML2026/image_generation/mitigating_the_contractivity_trap_in_diffusion_odes_via_stein_stabilization.md)
- [\[NeurIPS 2025\] Towards Robust Zero-Shot Reinforcement Learning](towards_robust_zero-shot_reinforcement_learning.md)
- [\[ICCV 2025\] Joint Diffusion Models in Continual Learning](../../ICCV2025/image_generation/joint_diffusion_models_in_continual_learning.md)
- [\[NeurIPS 2025\] Co-Reinforcement Learning for Unified Multimodal Understanding and Generation](coreinforcement_learning_for_unified_multimodal_understandin.md)

</div>

<!-- RELATED:END -->
