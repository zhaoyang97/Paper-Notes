---
title: >-
  [论文解读] FALCON: Few-step Accurate Likelihoods for Continuous Flows
description: >-
  [NEURIPS2025][图像生成][Continuous Normalizing Flows] 提出 FALCON，通过混合训练目标（flow matching + 平均速度损失 + 可逆性正则化）使连续归一化流在少步采样下仍能提供足够精确的似然估计，从而实现比传统 CNF 快两个数量级的 Boltzmann 采样。
tags:
  - "NEURIPS2025"
  - "图像生成"
  - "Continuous Normalizing Flows"
  - "Boltzmann Generators"
  - "Flow Matching"
  - "Importance Sampling"
  - "Molecular Sampling"
---

# FALCON: Few-step Accurate Likelihoods for Continuous Flows

**会议**: NEURIPS2025  
**arXiv**: [2512.09914](https://arxiv.org/abs/2512.09914)  
**代码**: 待确认  
**领域**: 图像生成  
**关键词**: Continuous Normalizing Flows, Boltzmann Generators, Flow Matching, Importance Sampling, Molecular Sampling  

## 一句话总结

提出 FALCON，通过混合训练目标（flow matching + 平均速度损失 + 可逆性正则化）使连续归一化流在少步采样下仍能提供足够精确的似然估计，从而实现比传统 CNF 快两个数量级的 Boltzmann 采样。

## 背景与动机

从 Boltzmann 分布 $p(x) \propto \exp(-\mathcal{E}(x))$ 中采样分子构型是统计物理中的核心挑战。传统方法（分子动力学、MCMC）容易陷入局部极小值，产生大量相关样本。Boltzmann Generators (BGs) 通过训练生成模型学习目标分布的近似 $p_\theta(x)$，再用自归一化重要性采样（SNIS）校正至真实分布，实现高效且统计一致的采样。

当前最先进的 BGs 主要基于连续归一化流（CNF），通过 flow matching 训练。CNF 表达能力强、训练稳定，但**似然计算极其昂贵**：需要沿流的每一步计算完整 Jacobian 的迹，且需要数千步离散化来控制误差。这严重限制了 CNF 在实际大规模分子采样中的应用。

近年来少步生成模型（consistency models、MeanFlow 等）虽然能大幅加速采样，但它们无法原生地提供高精度似然估计——因为学到的 flow map 在训练未完全收敛时不保证可逆，标准换元公式不可用。这使得它们不适合需要精确似然的重要性采样应用。

## 核心问题

如何设计一个既能**少步高效采样**，又能提供**足够精确似然估计**以支持重要性采样的连续流模型？具体需要同时满足四个条件：可逆性、回归损失训练、少步生成、自由形式架构。现有方法均无法同时满足这四点。

## 方法详解

### 1. FALCON 的核心思想

FALCON 基于 flow map 模型 $u_\theta(x_s, s, t)$（学习从时间 $s$ 到 $t$ 的平均速度），通过引入**可逆性正则化**确保 flow map 在训练过程中保持可逆，从而可以使用换元公式高效计算似然。

### 2. 关键理论洞察

- **Proposition 1**：在最优条件下（$u_\theta^*$ 完美最小化平均速度目标），flow map $X_u(\cdot, s, t)$ 处处可逆，换元公式成立
- **Proposition 2**：更弱的条件——只要可逆性损失 $\mathcal{L}_{\text{inv}}$ 被最小化，flow map 就是可逆的，换元公式就成立。这不要求 flow map 与连续时间流完全匹配

这意味着我们无需完美拟合连续流，只需要保证映射可逆即可实现有效的 Boltzmann 生成。

### 3. 混合训练目标

$$\mathcal{L}(\theta) = \mathcal{L}_{\text{cfm}}(\theta) + \lambda_{\text{avg}} \mathcal{L}_{\text{avg}}(\theta) + \lambda_r \mathcal{L}_{\text{inv}}(\theta)$$

三个组成部分：

- **$\mathcal{L}_{\text{cfm}}$**：标准 flow matching 损失，学习瞬时速度场
- **$\mathcal{L}_{\text{avg}}$**：平均速度损失（等价于 MeanFlow），训练模型在大步长下仍能准确预测目标位置。使用 JVP（Jacobian-vector product）高效实现，只需一次前向自动微分调用
- **$\mathcal{L}_{\text{inv}}$**：可逆性正则化，即 cycle-consistency 损失

$$\mathcal{L}_{\text{inv}}(\theta) = \mathbb{E}_{s,t,x_s} \|x_s - X_u(X_u(x_s, s, t), t, s)\|^2$$

要求正向映射后再反向映射能回到原点，鼓励映射可逆。

### 4. 参数化技巧

由于 FALCON 需要前向和反向两个方向的 flow map，在 $s = t$ 处存在不连续性。解决方案是将模型参数化为 $u_\theta(x_s, s, t) = \text{sign}(t - s) \cdot h_\theta(x_s, s, t)$，消除方向切换时的不连续。

### 5. 可扩展架构

得益于少步采样的低推理成本，FALCON 首次在 Boltzmann Generator 场景中使用 **Diffusion Transformer (DiT)** 架构（带额外时间嵌入头），突破了此前仅能使用 230 万参数小型等变网络的限制。通过数据增强实现软 SO(3) 等变性，减去均值实现平移不变性。

### 6. 似然计算

对于少步（如 4 步）的可逆 flow map，似然计算通过换元公式：

$$\log p_t(x_t) = \log p_s(x_s) - \log |\det \mathbf{J}_{X_u}(x_s)|$$

每步只需 $d$ 次函数评估计算 Jacobian，行列式计算代价相对可忽略。由于步数极少，总计算量远小于传统 CNF 的数千步积分。

## 实验关键数据

在四个分子系统上评估：alanine dipeptide (ALDP)、tri-alanine (AL3)、alanine tetrapeptide (AL4)、hexa-alanine (AL6)。

### 对比连续流（ECNF++）

| 系统 | 方法 | ESS ↑ | $\mathcal{E}$-$\mathcal{W}_2$ ↓ | $\mathbb{T}$-$\mathcal{W}_2$ ↓ |
|------|------|-------|-----|-----|
| AL3 | ECNF++ | 0.003 | 2.206 | 0.962 |
| AL3 | **FALCON** | **0.077** | **0.544** | **0.452** |
| AL4 | ECNF++ | 0.016 | 5.638 | 1.002 |
| AL4 | **FALCON** | **0.055** | **0.686** | **0.858** |
| AL6 | ECNF++ | 0.006 | 10.668 | 1.902 |
| AL6 | **FALCON** | **0.060** | **0.892** | **1.256** |

### 对比离散流（SBG）

- 在所有大分子系统上，FALCON 4 步采样全面优于 SBG（当前最优离散 NF）
- 即使 SBG 使用 $5 \times 10^6$ 个样本（FALCON 的 250 倍），其能量 Wasserstein 距离仍明显差于 FALCON

### 计算效率

- FALCON 推理速度比等性能 CNF **快两个数量级**
- 在 hexa-alanine 上，传统 CNF 甚至无法在合理时间内生成 $10^4$ 个样本
- 训练+推理总时间：FALCON 在所有系统上均为最快（AL6 上 25.76h vs CNF 的 82.10h vs SBG 的 57.50h）

## 亮点

1. **理论严谨**：提出并证明了可逆性正则化的充分性（Proposition 2），为方法提供了理论保障
2. **两个数量级加速**：相比传统 CNF 推理速度提升 100 倍以上，使大规模分子采样变得可行
3. **架构解放**：少步推理使得首次在 Boltzmann Generator 中使用 DiT 等大规模 Transformer 成为可能
4. **高效实现**：利用 JVP 一次调用同时计算前向传播和梯度，$\mathcal{L}_{\text{cfm}}$ 和 $\mathcal{L}_{\text{avg}}$ 可合并为单一损失
5. **统计一致性**：配合 SNIS 重加权，生成样本在理论上收敛到真实 Boltzmann 分布

## 局限与展望

1. **可逆性非完美**：实践中 $\mathcal{L}_{\text{inv}}$ 只能被近似最小化，cycle-consistency 误差仍会影响似然估计精度
2. **分子系统规模有限**：最大测试系统为 hexa-alanine（6 个残基），距离蛋白质等真实大规模系统仍有较大差距
3. **训练数据依赖**：依赖有偏 MD 模拟数据，数据质量和偏差程度会影响最终性能
4. **等变性为软约束**：通过数据增强实现的软 SO(3) 等变性不如精确等变架构严格
5. **ESS 指标表现参差**：在 ALDP 上 ESS 不如 ECNF++，说明在小分子上可逆性正则化可能带来一定精度损失

## 与相关工作的对比

| 方法 | 可逆 | 回归损失 | 少步 | 自由架构 | 似然精度 |
|------|------|---------|------|---------|---------|
| ECNF/ECNF++ | ✓（连续） | ✓ | ✗（数千步） | ✓ | 高但慢 |
| SBG (TARFlow) | ✓ | ✗（MLE） | ✓（1步） | ✗ | 精确但表达力受限 |
| RegFlow | ✓ | ✓ | ✓ | ✗ | 中等 |
| MeanFlow/ConsistencyFM | ✗ | ✓ | ✓ | ✓ | 无法计算 |
| **FALCON** | **✓** | **✓** | **✓** | **✓** | **快且准** |

FALCON 是首个同时满足四项条件的方法，核心区别在于引入可逆性正则化弥补了少步流模型缺乏似然估计能力的短板。

## 启发与关联

1. **通用性潜力**：cycle-consistency 正则化驱动可逆性的思路可推广到任何需要精确似然的流模型应用（图像生成、异常检测等）
2. **架构解耦**：将"好的生成"和"可逆性"解耦为不同损失项的思路值得借鉴——不需要显式设计可逆架构也能获得可逆性
3. **推理时间缩放**：SNIS 框架天然支持推理时间缩放（更多样本 → 更精确估计），这与当前 AI 推理缩放趋势吻合
4. **科学计算中的 Transformer**：本文证明了 DiT 在科学计算领域（分子采样）的可行性，可能启发更多将 vision/language Transformer 迁移到科学任务的工作

## 评分
- 新颖性: 8/10 — 可逆性正则化与少步流的结合思路新颖，理论支撑充分
- 实验充分度: 8/10 — 四个分子系统、多个基线、消融实验完备，但系统规模仍偏小
- 写作质量: 9/10 — 结构清晰，动机充分，理论和实验衔接紧密
- 价值: 8/10 — 两个数量级加速对科学计算实用性提升显著，但离大规模应用仍有距离

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] SANA-Sprint: One-Step Diffusion with Continuous-Time Consistency Distillation](../../ICCV2025/image_generation/sana-sprint_one-step_diffusion_with_continuous-time_consistency_distillation.md)
- [\[NeurIPS 2025\] Continuous Diffusion Model for Language Modeling](continuous_diffusion_model_for_language_modeling.md)
- [\[NeurIPS 2025\] BoltzNCE: Learning Likelihoods for Boltzmann Generation with Stochastic Interpolants](boltznce_learning_likelihoods_for_boltzmann_generation_with_stochastic_interpola.md)
- [\[NeurIPS 2025\] On the Relation between Rectified Flows and Optimal Transport](on_the_relation_between_rectified_flows_and_optimal_transport.md)
- [\[ICCV 2025\] Learning Few-Step Diffusion Models by Trajectory Distribution Matching](../../ICCV2025/image_generation/learning_few-step_diffusion_models_by_trajectory_distribution_matching.md)

</div>

<!-- RELATED:END -->
