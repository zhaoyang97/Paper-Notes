---
title: >-
  [论文解读] Discrete Guidance Matching: Exact Guidance for Discrete Flow Matching
description: >-
  [ICLR 2026][图像生成][离散流匹配] 给定一个预训练的离散流匹配/扩散模型和目标-源分布的密度比，本文推导出精确：的转移率引导公式，把每步采样从需要多次前向降到单次前向：，并把能量引导、分类器引导、RLHF 偏好对齐统一为同一框架的特例。 - 领域现状：离散扩散模型与离散流匹配（DFM）已成为生成离散数据（文本、…
tags:
  - "ICLR 2026"
  - "图像生成"
  - "离散流匹配"
  - "引导采样"
  - "连续时间马尔可夫链"
  - "后验引导"
  - "偏好对齐"
  - "文生图"
---

# Discrete Guidance Matching: Exact Guidance for Discrete Flow Matching

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=N1RYhOg6ib](https://openreview.net/forum?id=N1RYhOg6ib)  
**代码**: [https://github.com/WanZhengyan/Discrete-Guidance-Matching](https://github.com/WanZhengyan/Discrete-Guidance-Matching)  
**领域**: image_generation（离散流匹配 / 离散扩散引导、后验采样、偏好对齐）  
**关键词**: 离散流匹配, 引导采样, 连续时间马尔可夫链, 后验引导, 偏好对齐, 文生图  

## 一句话总结
给定一个预训练的离散流匹配/扩散模型和目标-源分布的密度比，本文推导出**精确**的转移率引导公式，把每步采样从需要多次前向降到**单次前向**，并把能量引导、分类器引导、RLHF 偏好对齐统一为同一框架的特例。

## 研究背景与动机
- **领域现状**：离散扩散模型与离散流匹配（DFM）已成为生成离散数据（文本、token 化图像）的有力替代 AR 模型的方案。在这类模型上做"引导"（guidance），即把预训练模型引向某个目标分布（条件生成、能量加权、偏好对齐），是控制生成的关键手段。
- **现有痛点**：离散引导本质上要修正一个**转移率/转移概率**矩阵，而转移率涉及当前状态可跳转到的所有目标位置，朴素计算需要对每个候选位置都做一次前向，开销巨大。为提速，现有方法（Vignac 2023、Schiff 2025、Nisonoff 2025）把离散模型当连续函数，用**一阶泰勒近似**估计 log 密度比来减少前向次数。
- **核心矛盾**：一阶近似在**离散状态空间里根本不成立**——近似的好坏取决于 $z$ 与 $x$ 在欧氏空间中的相对位置，而离散 token 之间并没有有意义的欧氏距离，近似误差可能很大（实验中引导强度 $\gamma=10,20$ 时分布明显偏离真值）。同时已有方法只覆盖类条件或能量加权等特定情形，缺乏统一性。
- **本文目标**：构造一个**既精确（无近似误差）又高效（单次前向）**、且足够通用（能涵盖能量引导/分类器引导/偏好对齐）的离散引导框架。
- **核心 idea**：**【精确转移率重写】** 在 CTMC 框架下，若源分布与目标分布共用同一条条件概率路径，则目标分布的后验只需用密度比 $r(x)=q_1(x)/p_1(x)$ 对源后验**重加权**即可精确得到，无需任何泰勒展开；这个重加权项（密度比的条件期望）可以用一个网络通过 **Bregman 散度**离线学好，采样时一次前向搞定。

## 方法详解

### 整体框架
方法建立在连续时间马尔可夫链（CTMC）上：预训练模型给出从源分布 $p_1$ 采样的后验 $p_{1|t}$，再给定一个已知的密度比 $r(x)=q_1(x)/p_1(x)$，框架直接算出生成目标分布 $q_1$ 所需的目标速度场/转移率。整条管线是"**学密度比的条件期望 → 重加权源后验 → always-valid 采样**"，训练侧用 Bregman 散度学引导网络 $h_t$，采样侧把 $h_t$ 与预训练后验逐元素相乘得到目标后验。

```mermaid
flowchart LR
    A[预训练离散流模型<br/>源后验 p_1|t] --> D[目标后验 q_1|t]
    B[密度比 r=q_1/p_1<br/>来自能量/分类器/奖励] --> C[引导网络 h_t<br/>Bregman 散度训练]
    C --> D
    D --> E[always-valid 采样<br/>单次前向/步]
    E --> F[目标分布样本 q_1]
```

### 关键设计

**1. 后验引导（Posterior-Based Guidance）：用密度比重加权源后验，精确且只需单次前向。** 这是全文的理论基石（定理 1）。在 Assumption 1（目标分布对源分布绝对连续，保证密度比良定义）下，只要源与目标共用同一条**条件**概率路径 $p_{t|1}=q_{t|1}$，目标后验就有闭式
$$q_{1|t}(z^d|x) = \frac{\mathbb{E}_{x_1^{\setminus d}\sim p(x_1^{\setminus d}|x_1^d=z^d,x_t=x)}[r(x_1)]}{\mathbb{E}_{x_1\sim p_{1|t}(x_1|x)}[r(x_1)]}\, p_{1|t}(z^d|x).$$
关键在于这是**精确等式而非近似**：分子分母都是密度比关于后验的条件期望，没有任何泰勒展开。当 $q_1(x)=p_1(x|y)$（类条件）时，密度比退化为分类器比值 $p(y|x_1^d=z^d,x_t=x)/p(y|x_t=x)$，自然恢复经典分类器引导。由于只用到当前状态 $x$ 的一次后验前向，每步采样只需 1 次函数评估。

**2. 与 rate-based 引导的统一与对比：把已有方法收编为更强假设下的特例。** 定理 2 给出 rate-based 形式：若进一步要求源与目标的**前向加噪转移率相同**（比定理 1 更强的条件），则反向转移率为 $u_t^q(z,x)=\frac{\mathbb{E}_{x_1\sim p_{1|t}(x_1|z)}[r(x_1)]}{\mathbb{E}_{x_1\sim p_{1|t}(x_1|x)}[r(x_1)]}u_t^p(z,x)$，恰好恢复 Nisonoff et al. (2025) 的 predictor guidance。论文据此把三类引导排成一张谱：posterior-based（本文，精确，1 次前向）、rate-based（精确但需 $D+1$ 次前向）、first-order approximated（有近似误差，2 次前向）。一阶近似式 $u_t^q(z,x)=\exp\langle z-x,\nabla_x\log\mathbb{E}[r(x_1)]\rangle u_t^p(z,x)$ 的问题被点破：右端值主要由 $z,x$ 的欧氏位置决定，对离散数据不合理。

**3. Bregman 散度训练 + 目标分布正则项：学密度比的条件期望，且能利用目标样本。** 要落地定理 1 需估计 $h_t^d(z^d,x)=\mathbb{E}[r(x_1)\mid \cdot]$。由于密度比天然为正，直接用 $\ell_2$ 损失（即 $F(x)=\|x\|^2/2$ 的 Bregman 散度）效果差；本文改取 $F(x)=\langle x,\log x\rangle$，得到训练目标
$$\mathcal{L}_{h,p}(\theta)=\mathbb{E}\Big[\sum_{d=1}^{D} h_t^{d,\theta}(x_1^d,x_t)-r(x_1)\log h_t^{d,\theta}(x_1^d,x_t)\Big],$$
它**只需源分布数据**。当还能拿到目标分布样本时，再加一个正则项 $\mathcal{L}_{h,q}$（其极小点同样是精确引导 $h_t$），最终目标 $\mathcal{L}_h=\mathcal{L}_{h,p}+\lambda\mathcal{L}_{h,q}$，$\lambda$ 控制利用目标样本的强度。

**4. 统一三类任务（能量引导 / 分类器引导 / RLHF 偏好对齐）。** 框架的通用性来自密度比可由不同来源给出：能量引导取 $p_1^{(\gamma)}(x)\propto p_1(x)e^{-\gamma E(x)}$ 故 $r\propto p^\gamma(y=1|x)$；分类器引导取分类器比值；偏好对齐则借 RLHF 的闭式最优策略 $\pi^*(o_1|c)\propto \pi_{\text{ref}}(o_1|c)\exp(R(c,o_1)/\tau)$，令 $p_1=\pi_{\text{ref}}$、$q_1=\pi^*$，于是引导网络去逼近 $\exp(R(c,o_1)/\tau)$。同一套精确后验重加权即可服务所有这些场景，还能无缝套到 masked diffusion（其后验时间无关，引导也变成时间无关）。

## 实验关键数据

### 主实验表格（GenEval 文生图，基于 FUDOKI）

| 方法 | Single | Two | Counting | Colors | Position | Color Attri. | Overall ↑ |
|---|---|---|---|---|---|---|---|
| FUDOKI（无引导基线） | 0.96 | 0.85 | 0.56 | 0.88 | 0.68 | 0.67 | 0.77 |
| **Ours（精确引导）** | 0.94 | 0.86 | 0.53 | 0.89 | 0.70 | **0.77** | **0.78** |

六个子任务中四项超过基线，Color Attribution 从 0.67 提升到 0.77 最为显著。

### 消融实验表格（多模态理解，1.5B 参数）

| 模型 | POPE ↑ | MME-P ↑ | MMB ↑ | GQA ↑ | MMMU ↑ | MM-Vet ↑ |
|---|---|---|---|---|---|---|
| FUDOKI（基线） | 86.1 | 1485.4 | 73.9 | 57.6 | 34.3 | 38.0 |
| **Ours** | **86.8** | **1492.7** | **74.2** | **58.2** | **35.4** | **38.6** |

引导在全部六个理解基准上一致提升。能量引导的 2-D 模拟实验中，$\gamma=10,20$ 时一阶近似的 predictor guidance 明显偏离真值分布，而本文的 posterior/rate-based 引导都接近 ground truth。

### 关键发现
- **采样效率**：posterior-based 引导比 rate-based 快约 **1.6×**（rate-based 每步需 $D+1$ 次前向，posterior-based 仅 1 次）。
- **精度**：高引导强度下一阶近似失真严重，本文无近似方法稳定逼近目标分布。
- **统一性**：同一框架在能量引导、文生图 RLHF、多模态理解三类任务上都有效，验证了密度比视角的通用性。

## 亮点与洞察
- **诊断到位**：把"一阶近似在离散空间不成立"讲得很透——近似值依赖 token 的欧氏位置，而离散 token 没有有意义的几何距离，这是已有方法的根本缺陷。
- **精确 vs 高效不再二选一**：定理 1 用"共用条件路径"这个温和假设，把引导降成对源后验的一次重加权，既消除近似误差又只需单次前向。
- **理论收编**：定理 2 把 Nisonoff 等的 rate-based predictor guidance 解释为更强假设下的特例，给出清晰的方法谱系（精确度 × 前向次数）。
- **Bregman 散度选型**：针对密度比为正的特性，用 $F=\langle x,\log x\rangle$ 而非 $\ell_2$，是一个有依据的工程选择。

## 局限与展望
- **依赖密度比可得**：框架假设密度比 $r(x)=q_1/p_1$ 已知或可学，对那些密度比难以良定义/估计的目标分布适用性受限（Assumption 1 的绝对连续性也是前提）。
- **提升幅度温和**：在 GenEval（0.77→0.78）和多模态理解上的绝对增益较小，更多是稳健的小幅改进而非数量级飞跃。
- **同路径假设**：定理 1 要求源/目标共用条件概率路径、定理 2 还要求前向加噪率相同，限制了某些跨模型迁移场景。
- **展望**：把精确引导扩展到更复杂的 reward/能量结构、更大规模多模态模型，以及探索 $\lambda$ 正则项在目标样本稀缺时的自适应策略。

## 相关工作与启发
- **连续空间引导**：classifier guidance（Dhariwal & Nichol 2021）、能量引导（Lu 2023、Ouyang 2024）是思想源头，本文是其离散对应物的精确化。
- **离散引导**：Vignac 2023、Schiff 2025、Nisonoff 2025 用一阶近似，本文正是要超越它们；定理 2 显式恢复 Nisonoff 的 predictor guidance。
- **离散流匹配**：Campbell 2024、Gat 2024、Shaul 2025、Qin 2025 提供了 marginalization trick 与条件转移率的基础设施；FUDOKI（Wang 2025）作为多模态 backbone。
- **偏好对齐**：借用 DPO/RLHF（Rafailov 2023、Ouyang 2022）的闭式最优策略，把奖励对齐纳入同一引导框架。
- **启发**：在离散生成里做控制时，与其在转移率上做几何近似，不如回到 CTMC 后验的概率重加权——精确性和效率往往能同时获得。

## 评分
- **新颖性**: ⭐⭐⭐⭐ — 首个一般形式的精确离散引导，统一了能量/分类器/偏好对齐三类任务，并把已有近似方法收编为特例，理论贡献清晰。
- **实验充分度**: ⭐⭐⭐⭐ — 覆盖 2-D 能量引导模拟、文生图（GenEval）、多模态理解（6 个基准）三层验证，含采样效率对比；但下游绝对增益偏温和。
- **写作质量**: ⭐⭐⭐⭐ — 定理-框架-实验脉络清楚，Table 1 的方法谱对比一目了然，对一阶近似失效的剖析很有说服力。
- **价值**: ⭐⭐⭐⭐ — 单次前向的精确引导对离散扩散/流匹配的可控生成有直接实用价值，框架通用且能套到 masked diffusion 与多模态大模型。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Discrete Adjoint Matching](discrete_adjoint_matching.md)
- [\[ICLR 2026\] What Exactly Does Guidance Do in Masked Discrete Diffusion Models](what_exactly_does_guidance_do_in_masked_discrete_diffusion_models.md)
- [\[ICLR 2026\] Delay Flow Matching](delay_flow_matching.md)
- [\[ICLR 2026\] Flow Matching with Semidiscrete Couplings](flow_matching_with_semidiscrete_couplings.md)
- [\[ICLR 2026\] Carré du champ Flow Matching: 用几何感知噪声改善生成模型的质量-泛化权衡](carré_du_champ_flow_matching_better_quality-generalisation_tradeoff_in_generativ.md)

</div>

<!-- RELATED:END -->
