---
title: >-
  [论文解读] Ψ-Sampler: Initial Particle Sampling for SMC-Based Inference-Time Reward Alignment in Score Models
description: >-
  [NeurIPS 2025 Spotlight][图像生成][推理时对齐] 提出Ψ-Sampler框架，在SMC（序贯蒙特卡洛）推理时奖励对齐中引入基于pCNL（预条件Crank-Nicolson Langevin）算法的初始粒子采样，从奖励感知的后验分布初始化粒子，显著提升布局生成、数量感知生成和美学偏好生成的对齐效果。
tags:
  - "NeurIPS 2025 Spotlight"
  - "图像生成"
  - "推理时对齐"
  - "序贯蒙特卡洛"
  - "奖励对齐"
  - "MCMC"
  - "preconditioned Crank-Nicolson"
---

# Ψ-Sampler: Initial Particle Sampling for SMC-Based Inference-Time Reward Alignment in Score Models

**会议**: NeurIPS 2025 Spotlight  
**arXiv**: [2506.01320](https://arxiv.org/abs/2506.01320)  
**代码**: [项目页面](https://psi-sampler.github.io/)  
**领域**: 扩散模型 / 图像生成  
**关键词**: 推理时对齐, 序贯蒙特卡洛, 奖励对齐, MCMC, preconditioned Crank-Nicolson

## 一句话总结

提出Ψ-Sampler框架，在SMC（序贯蒙特卡洛）推理时奖励对齐中引入基于pCNL（预条件Crank-Nicolson Langevin）算法的初始粒子采样，从奖励感知的后验分布初始化粒子，显著提升布局生成、数量感知生成和美学偏好生成的对齐效果。

## 研究背景与动机

**推理时对齐的范式转变**：类似于LLM从预训练到后训练的范式转移（如GPT-o3、DeepSeek的"Aha moment"），扩散模型也越来越重视推理阶段的对齐优化。SMC方法将去噪过程视为序贯采样，通过维护多个粒子并根据奖励函数重采样来引导生成。

**现有SMC方法的核心缺陷**：所有现有SMC方法（TDS、DAS、FPS等）都从标准高斯先验初始化粒子，完全忽略奖励信息。这有两个严重问题：

**扩散系数衰减问题**：在去噪后期 $g(t)^2 \to 0$，奖励梯度的影响被 $g^2(t)$ 缩放导致逐渐消失，后期难以引导粒子到高奖励区域

**多模态奖励函数问题**：当奖励函数高度非凸时，后期中间分布已经高度集中在特定模式附近，模式间的连通性降低，粒子难以逃逸局部最优

**核心idea**：与其依赖后期阶段的粒子探索，不如在初始阶段就从**奖励感知的后验分布**采样高质量粒子。最优初始分布有解析形式：$\tilde{p}_1^*(\mathbf{x}_1) = \frac{1}{Z_1} p_1(\mathbf{x}_1) \exp\left(\frac{r(\mathbf{x}_{0|1})}{\alpha}\right)$。

**高维MCMC的挑战**：FLUX模型的潜空间维度达65536，传统MALA的接受率随维度增加急剧衰减（步长需按 $O(d^{-1/3})$ 缩小），导致极慢的混合速度。

## 方法详解

### 整体框架

Ψ-Sampler = pCNL初始粒子采样 + 标准SMC去噪过程。先用pCNL从后验分布采样一组高质量初始粒子，再将它们输入标准SMC流程。总NFE预算一半分配给初始粒子采样，一半给SMC。

### 关键设计

1. **奖励感知后验分布**：
   最优初始分布为 $\tilde{p}_1^*(\mathbf{x}_1) \propto p_1(\mathbf{x}_1) \exp(r(\mathbf{x}_{0|1})/\alpha)$，其中 $\mathbf{x}_{0|1}$ 是通过Tweedie公式从 $\mathbf{x}_1$ 估计的干净图像。这是已知形式的非归一化密度，适合MCMC采样。
    - 近期蒸馏技术（如rectified flow）使轨迹更直、Tweedie估计更早准确，为从全噪声状态就能有效评估奖励创造了条件
    - 正则化参数 $\alpha$ 控制奖励最大化和先验保持的平衡

2. **预条件Crank-Nicolson Langevin (pCNL) 算法**：
   pCN算法专为无限维/函数空间设计，核心是半隐式Euler离散化：
    $\mathbf{x}' = \rho \mathbf{x} + \sqrt{1-\rho^2}\left(\mathbf{z} + \frac{\sqrt{\epsilon}}{2} \nabla \frac{r(\mathbf{x}_{0|1})}{\alpha}\right), \quad \rho = \frac{1-\epsilon/4}{1+\epsilon/4}$
   
   **与MALA的关键区别**：
    - pCNL的提议分布保持了高斯先验的不变性（prior-preserving），因此接受率不随维度退化
    - MALA在65536维空间中步长0.05以上接受率就趋近于零，而pCNL在步长2.0时仍保持合理接受率
    - pCNL步长更大意味着更快的混合速度和更高效的探索
    - pCNL同样使用MH修正保证收敛到正确分布

3. **初始粒子采样流程**：

    - 从先验采样初始状态
    - 运行pCNL链，丢弃burn-in部分
    - 以固定间隔(thinning)采样得到K个粒子
    - 使用固定步长，简单有效
    - 采样的粒子直接作为SMC的初始粒子

### 损失函数 / 训练策略

本方法是纯推理时方法，无需训练。核心计算开销在于：
- pCNL每步需要计算 $\nabla r(\mathbf{x}_{0|1})$，即奖励对Tweedie估计的梯度
- SMC每步需要计算加权的去噪步骤
- 总NFE预算在pCNL和SMC之间分配

## 实验关键数据

### 主实验 — 三个任务的定量比较 (FLUX模型)

| 任务 | 指标 | DPS | FreeDoM | TDS | DAS | Top-K | ULA | MALA | **Ψ-Sampler** |
|------|------|-----|---------|-----|-----|-------|-----|------|-----------|
| 布局生成 | GroundingDINO↑ | 0.166 | 0.177 | 0.417 | 0.363 | 0.425 | 0.370 | 0.401 | **0.467** |
| 布局生成 | mIoU (held-out)↑ | 0.215 | 0.229 | 0.402 | 0.342 | 0.427 | 0.374 | 0.401 | **0.471** |
| 数量感知 | T2I-Count↓ | 14.19 | 15.21 | 1.804 | 1.151 | 1.077 | 3.035 | 1.601 | **0.850** |
| 数量感知 | MAE (held-out)↓ | 15.7 | 15.68 | 5.3 | 4.18 | 3.68 | 4.83 | 3.58 | **2.93** |
| 美学偏好 | Aesthetic↑ | 6.139 | 6.310 | 6.853 | 6.935 | 6.879 | 6.869 | 6.909 | **7.012** |

### pCNL vs MALA 步长对比实验

| 步长 | MALA接受率 | pCNL接受率 | MALA mIoU | pCNL mIoU |
|------|-----------|-----------|-----------|-----------|
| 0.05 | ~25% | ~80% | ~0.40 | ~0.43 |
| 0.5 | ~0% | ~50% | 退化 | **~0.47** |
| 2.0 | ~0% | ~25% | 退化 | ~0.45 |

### 关键发现

- **后验初始化 vs 先验初始化**：所有后验初始化方法都优于先验初始化的SMC，证实了初始粒子质量的重要性
- **pCNL vs ULA/MALA**：ULA无MH修正导致偏差、MALA在高维空间接受率骤降；pCNL两者兼顾
- **单粒子方法严重不足**：DPS和FreeDoM在所有任务上都远逊于SMC方法
- **held-out奖励上的泛化**：Ψ-Sampler不仅在训练奖励上最优，在held-out奖励上也表现最好，说明改进来自真正更高质量的样本
- **2D toy实验直观验证**：在6模高斯混合分布上，MALA+SMC仍有模式遗漏，Ψ-Sampler完全覆盖目标分布

## 亮点与洞察

- **首次将pCN算法用于生成模型**：将Bayesian逆问题中的高维MCMC技术迁移到生成模型的推理时对齐
- **诊断准确**：清晰指出扩散系数衰减导致后期奖励引导失效的问题，因此将计算预算前置到初始化阶段
- **与蒸馏趋势的协同**：当前模型趋向更直的轨迹和更好的早期Tweedie估计，这恰好使初始后验采样更有效
- **一致的理论-实验对应**：从SOC框架推导的最优初始分布在实验中得到验证

## 局限与展望

- 假设奖励模型可微分——不适用于不可微奖励
- 依赖Tweedie近似的准确性——早期步的Tweedie估计对非蒸馏模型可能不够准确
- pCNL要求先验为高斯分布（即只能在 $t=1$ 处应用）
- 固定步长策略，自适应步长可能进一步提升效率
- 可能被滥用于生成误导性或有害的超写实假图——需要负责任的部署

## 相关工作与启发

- 与TDS (Twisted Diffusion Sampler)、DAS等SMC方法互补：它们改进SMC中间步骤，Ψ-Sampler改进初始化
- pCN算法来自PDE约束的贝叶斯逆问题领域，跨领域迁移带来了显著增益
- 与LLM推理时scaling的思路一致：通过搜索而非微调来提升对齐效果

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 首次在生成建模中引入pCN算法，初始后验采样的思路清晰有力
- **实验充分度**: ⭐⭐⭐⭐ — 三个差异显著的任务+多维度指标+步长分析+toy实验
- **写作质量**: ⭐⭐⭐⭐ — 问题定义清晰，理论推导完整，但MCMC背景知识门槛较高
- **价值**: ⭐⭐⭐⭐⭐ — 为推理时scaling提供了新维度（初始化质量），方法通用且与现有SMC方法正交

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Progressive Inference-Time Annealing of Diffusion Models for Sampling from Boltzmann Densities](progressive_inference-time_annealing_of_diffusion_models_for_sampling_from_boltz.md)
- [\[ICLR 2026\] Diffusion Blend: Inference-Time Multi-Preference Alignment for Diffusion Models](../../ICLR2026/image_generation/diffusion_blend_inference-time_multi-preference_alignment_for_diffusion_models.md)
- [\[ICLR 2026\] GLASS Flows: Efficient Inference for Reward Alignment of Flow and Diffusion Models](../../ICLR2026/image_generation/glass_flows_reward_alignment_diffusion.md)
- [\[NeurIPS 2025\] Learnable Sampler Distillation for Discrete Diffusion Models](learnable_sampler_distillation_for_discrete_diffusion_models.md)
- [\[NeurIPS 2025\] Inference-Time Scaling for Flow Models via Stochastic Generation and Rollover Budget Forcing](inference-time_scaling_for_flow_models_via_stochastic_generation_and_rollover_bu.md)

</div>

<!-- RELATED:END -->
