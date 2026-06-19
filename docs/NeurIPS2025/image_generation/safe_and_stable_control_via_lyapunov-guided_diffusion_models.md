---
title: >-
  [论文解读] Safe and Stable Control via Lyapunov-Guided Diffusion Models
description: >-
  [NeurIPS 2025][图像生成][扩散模型] 提出 S²Diff，一个基于模型的扩散规划框架，利用控制 Lyapunov 屏障函数（CLBF）引导扩散采样生成轨迹级控制策略，无需控制仿射假设与二次规划，在多种非线性动力系统上同时保证安全性和稳定性，平均安全率达 98.75%。 现实世界中的控制问题（机器人、航空航天等…
tags:
  - "NeurIPS 2025"
  - "图像生成"
  - "扩散模型"
  - "Lyapunov稳定性"
  - "安全控制"
  - "证书函数"
  - "Almost Lyapunov理论"
---

# Safe and Stable Control via Lyapunov-Guided Diffusion Models

**会议**: NeurIPS 2025  
**arXiv**: [2509.25375](https://arxiv.org/abs/2509.25375)  
**代码**: 无  
**领域**: Image Generation / Control  
**关键词**: 扩散模型, Lyapunov稳定性, 安全控制, 证书函数, Almost Lyapunov理论

## 一句话总结

提出 S²Diff，一个基于模型的扩散规划框架，利用控制 Lyapunov 屏障函数（CLBF）引导扩散采样生成轨迹级控制策略，无需控制仿射假设与二次规划，在多种非线性动力系统上同时保证安全性和稳定性，平均安全率达 98.75%。

## 研究背景与动机

现实世界中的控制问题（机器人、航空航天等）不仅需要最小化成本，更需要同时保证**安全性**（轨迹不进入危险区域）和**稳定性**（系统收敛到目标状态）。然而同时满足两者在控制理论中仍是开放问题。

现有方法的核心痛点：

**模型预测控制（MPC）**：通过凸优化在滚动时域内最小化累积代价。但策略通常次优，计算复杂度随时域增长急剧上升，高维非线性问题中容易变得不可行。

**基于证书函数的方法（CLBF-QP）**：通过学习 CLBF 并用二次规划（QP）求解控制策略。存在三个根本性问题：(a) QP 构建需要控制仿射（control-affine）动力学假设；(b) 逐步贪心优化导致全局不一致行为，需引入松弛变量改变优化目标；(c) CLBF 与策略联合学习导致 QP 可行域不稳定，训练崩溃。

**现有扩散规划方法**（如 Diffuser、SafeDiffuser）：在长时决策中表现良好，但仅关注代价最小化，安全性和稳定性保证仍为空白。SafeDiffuser 需手工设计 CBF 先验，实用性受限。

**核心矛盾**：梯度方法受限于控制仿射假设、松弛变量引入、联合训练不稳定；采样方法缺乏安全稳定性保证。

**核心 idea**：将 CLBF 作为扩散模型的引导函数，用扩散采样替代 QP 求解，并揭示扩散采样与 Almost Lyapunov 理论的内在联系——即使 Lie 导数条件在小区域被违反，全局系统仍可保持近似指数衰减。

## 方法详解

### 整体框架

S²Diff 采用迭代的两阶段框架：(1) 用当前 CLBF 作为引导函数，通过扩散采样生成满足安全稳定约束的轨迹级控制策略；(2) 用采样的轨迹更新 CLBF 参数。两个阶段交替进行直到收敛。

### 关键设计

1. **概率化建模与 CLBF 引导采样**:

    - 功能：将带约束的控制优化问题转化为概率采样问题
    - 核心思路：定义目标轨迹分布 $p(U) \propto p_{\text{safe}}(U) \cdot p_{\text{stable}}(U) \cdot p_{\text{cost}}(U)$，其中安全项为 $V(x_t) \leq c$ 的指示函数乘积，稳定项采用**软约束**形式 $p_{\text{stable}} \propto \exp(-\frac{1}{\gamma_2}\sum_t \|[\mathcal{L}_f V(x_t) + \lambda V(x_t)]^+\|^2)$，代价项为累积代价的指数形式
    - 设计动机：软约束对应 Almost Lyapunov 理论——允许 Lie 导数条件在小概率区域被违反。当温度 $\gamma_2$ 足够小时，采样策略可在轨迹级保证安全稳定。相比 QP 方法引入松弛变量改变优化目标，软约束保持优化问题不变，避免高拒绝率

2. **基于蒙特卡洛的扩散采样**:

    - 功能：通过 DDPM 式的前向-逆向扩散过程生成控制轨迹
    - 核心思路：前向过程对轨迹加高斯噪声，逆向过程利用序贯蒙特卡洛（SMC）估计得分函数 $\nabla \log p(U^i)$，通过后验期望 $\mathbb{E}[U^0 | U^i]$ 的无偏估计进行去噪
    - 设计动机：采样方法不需要控制仿射结构，避免引入松弛变量，可直接处理一般可微非线性动力学。生成的干净轨迹 $U^0$ 直接给出控制策略 $u_{1:T}$

3. **CLBF 迭代更新**:

    - 功能：用扩散采样得到的轨迹数据更新神经网络参数化的 CLBF
    - 核心思路：损失函数包含 6 个项，分别约束均衡点值为零、正定性、安全集子水平集、不安全集超水平集、连续时间 Lie 导数递减（自动微分）和离散时间 Lyapunov 递减
    - 设计动机：使用一般的多层神经网络 $\hat{V} = W_N \sigma_{N-1}(\cdots \sigma_1(W_1 x))$ 参数化 CLBF，而非传统的二次形式，可以处理非凸安全集。同时结合连续和离散 Lie 导数约束互补提升学习质量

### 理论保证

**定理 3.1（Almost Lyapunov 安全稳定性保证）**：若 CLBF $V$ 在紧状态空间 $\mathcal{X}$ 上除了一个体积小于 $\epsilon$ 的连通集 $\Omega$ 外均满足 $\min_u \mathcal{L}_f V(x) < -\lambda V(x)$，则系统在扩散采样策略下满足 $V(x_t) \leq \exp(-\lambda_1 t) V(x_0) + M \epsilon^{1/n}$。即违反区域仅引入 $O(\epsilon^{1/n})$ 量级的加性缓冲项，整体仍保持近似指数衰减。

## 实验关键数据

### 主实验

在 8 个非线性动力系统上（含控制仿射和非控制仿射）与 rCLBF-QP、MPC、MBD 对比：

| 任务 | 方法 | 安全率 | ‖x−x⋆‖ | 推理时间(ms) |
|------|------|--------|---------|-------------|
| Segway | rCLBF-QP | 90% | 0.11 | 5.2 |
| Segway | S²Diff | **100%** | 0.23 | 21.8 |
| Neural Lander | rCLBF-QP | 55% | 0.13 | 12.7 |
| Neural Lander | S²Diff | **100%** | **0.06** | 35.4 |
| 2D Quad | rCLBF-QP | 70% | 0.19 | 18.6 |
| 2D Quad | S²Diff | **95%** | **0.11** | 82.4 |
| F-16 (非仿射) | MBD | 100% | 68.34 | 611.3 |
| F-16 (非仿射) | S²Diff | **100%** | **47.61** | **257.2** |
| **平均** | rCLBF-QP | 78.75% | 0.384 | 10.06 |
| **平均** | S²Diff | **98.75%** | **0.226** | 45.64 |

### 消融实验

| 配置 | 安全率 | ‖x−x⋆‖ | 说明 |
|------|--------|---------|------|
| γ₂=0.5 | 35% | 0.18 | 温度过高，稳定性约束过松 |
| γ₂=0.1 | **100%** | **0.06** | 最佳平衡点 |
| γ₂=0.01 | 100% | 0.12 | 温度过低，约束过严导致次优 |
| 仅离散Lie导数 (α₁=0) | 85% | 0.21 | 缺少自动微分约束降低安全率 |
| 仅连续Lie导数 (α₂=0) | 100% | 0.15 | 缺少离散约束精度下降 |
| 双Lie导数 (α₁=α₂=1) | **100%** | **0.06** | 互补约束最优 |

### 关键发现

- 扩散采样相比 QP 方法学到的 CLBF 具有更大的收缩区域，稳定性保证更强
- CLBF 引导使扩散采样效率大幅提升：无引导 MBD 安全率仅 73.75%，有 CLBF 引导升至 98.75%
- 违反率实测极低：Segway 0.5%、Neural Lander 1.1%、F-16 2.4%，验证 Almost Lyapunov 理论
- S²Diff 是唯一能处理非控制仿射 F-16 系统（16维状态空间）的证书方法

## 亮点与洞察

- 首次建立扩散采样与 Almost Lyapunov 理论的理论联系，将"采样时局部违反 Lie 导数条件"这一实践现象赋予坚实的理论基础
- 方法框架优雅：CLBF 引导扩散采样，采样轨迹反过来改进 CLBF，形成自增强迭代
- 消除了对控制仿射假设、松弛变量和 QP 求解器的依赖，大幅扩展了适用范围

## 局限与展望

- 推理速度（平均 45.6ms）虽优于 MPC（249.5ms），但慢于 QP（10ms），可通过策略蒸馏加速
- 目前仅在已知动力学模型的 model-based 设置下验证，model-free 扩展是未来方向
- CLBF 的神经网络参数化可能在极高维系统中面临可扩展性挑战

## 相关工作与启发

- 扩散规划（Diffuser、SafeDiffuser）+ 证书函数（CLBF、CBF）的交叉融合方向值得关注
- Almost Lyapunov 理论为"允许少量违反、保证全局性能"提供了理论工具，可迁移到其他约束优化场景
- 对 model-based RL 社区的启示：学习的 Lyapunov 函数可作为通用的安全引导信号

## 评分
- 新颖性: ⭐⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Training-Free Safe Text Embedding Guidance for Text-to-Image Diffusion Models](training-free_safe_text_embedding_guidance_for_text-to-image_diffusion_models.md)
- [\[NeurIPS 2025\] Training-Free Constrained Generation with Stable Diffusion Models](training-free_constrained_generation_with_stable_diffusion_models.md)
- [\[CVPR 2025\] Enhancing Creative Generation on Stable Diffusion-based Models](../../CVPR2025/image_generation/enhancing_creative_generation_on_stable_diffusion-based_models.md)
- [\[CVPR 2026\] Stable Mean Flow: Lyapunov-Inspired One-Step Flow Matching](../../CVPR2026/image_generation/stable_mean_flow_lyapunov-inspired_one-step_flow_matching.md)
- [\[NeurIPS 2025\] One Stone with Two Birds: A Null-Text-Null Frequency-Aware Diffusion Models for Text-Guided Image Inpainting](one_stone_with_two_birds_a_null-text-null_frequency-aware_diffusion_models_for_t.md)

</div>

<!-- RELATED:END -->
