---
title: >-
  [论文解读] Incorporating Expert Priors into Bayesian Optimization via Dynamic Mean Decay
description: >-
  [ICLR 2026][优化/理论][Bayesian Optimization] 把专家先验（关于最优点位置的分布 $\pi(x)$）直接塞进高斯过程的**均值函数**里，再用一个随迭代衰减的权重让它早期发力、后期淡出，从而得到一个与任意采集函数兼容、几乎零额外开销、且对坏先验鲁棒的 prior-informed 贝叶斯优化框架 DynMeanBO。
tags:
  - "ICLR 2026"
  - "优化/理论"
  - "Bayesian Optimization"
  - "Expert Prior"
  - "Gaussian Process"
  - "Mean Function"
  - "Dynamic Decay"
  - "Robustness"
---

# Incorporating Expert Priors into Bayesian Optimization via Dynamic Mean Decay

**会议**: ICLR 2026  
**代码**: [https://github.com/quchongqi/DynMeanBO](https://github.com/quchongqi/DynMeanBO)  
**领域**: 贝叶斯优化 / 黑盒优化 / 概率方法  
**关键词**: Bayesian Optimization, Expert Prior, Gaussian Process, Mean Function, Dynamic Decay, Robustness  

## 一句话总结
把专家先验（关于最优点位置的分布 $\pi(x)$）直接塞进高斯过程的**均值函数**里，再用一个随迭代衰减的权重让它早期发力、后期淡出，从而得到一个与任意采集函数兼容、几乎零额外开销、且对坏先验鲁棒的 prior-informed 贝叶斯优化框架 DynMeanBO。

## 研究背景与动机
**领域现状**：贝叶斯优化（BO）是黑盒优化的主力工具——用高斯过程（GP）拟合昂贵目标函数，再靠采集函数（EI/UCB 等）权衡探索与利用，逐点采样。现实中实验代价极高（HPO、材料发现、机器人控制），评估预算往往只有几十次，因此「样本效率」是命门。

**现有痛点**：很多场景里领域专家其实知道「最优解大概在哪片区域」，这种先验若能用上能大幅加速早期收敛。但已有的 prior-informed BO 方法都各有硬伤：BOPro 把先验绑死在 TPE 结构上、Ramachandran 等把先验嵌进核函数（一旦先验不准会污染整个核、性能崩塌）、πBO 用加权机制改采集函数（简单但本质是启发式，没在代理模型里显式建模先验）、ColaBO 灵活但只能配蒙特卡洛类采集函数且计算开销大。

**核心矛盾**：要么方法太复杂难落地，要么绑死特定采集函数/采样策略而不通用，要么对先验质量过度敏感、专家一旦判断错就把优化带歪。三者很难同时解决。

**本文目标**：做一个**简单、通用、鲁棒**的框架——先验好时加速、先验坏时不被坑、对采集函数无要求、计算开销可忽略，并且有收敛理论保证。

**核心 idea**：把专家先验注入 GP 的**均值函数**而非核函数或采集函数（`模型层注入`），并给它配一个随观测增多而指数衰减的权重（`动态衰减`），让 BO 早期吃专家红利、后期自动退回标准 BO 行为（`鲁棒兜底`）。

## 方法详解

### 整体框架
DynMeanBO 的全部改动只发生在 GP 的均值函数上：标准 BO 通常取零均值 $m(x)=0$，本文改成一个由专家先验构造、且随迭代逐渐淡出的动态均值 $m_n(x)$。流程是先用专家先验做混合初始化采样，再在每轮迭代中更新这个动态均值、重算 GP 后验、用任意采集函数选下一个点。

```mermaid
flowchart LR
    A[专家先验 π(x)<br/>最优点位置分布] --> B[先验均值 m_prior=A·π+B]
    A --> C[混合初始化<br/>ρ 比例从 π 采样 + Sobol]
    B --> D[动态衰减均值<br/>m_n=γ_n·m_prior+1-γ_n·μ_0]
    C --> E[GP 后验 p_f|D_n]
    D --> E
    E --> F[任意采集函数 α_x<br/>EI/UCB/MES/...]
    F --> G[选下一点 x_n 评估]
    G --> H{更新数据集 D_n}
    H -->|γ_n 衰减| D
```

### 关键设计

**1. 把专家先验写进 GP 均值函数：在模型层而非采集层注入知识。** 专家通常不知道目标函数 $f$ 的解析形式，但能给出最优点位置的先验分布 $\pi(x)$（单峰高斯代表一个看好的区域，混合高斯可表达多个区域，甚至偏好/成对比较也能转成这种分布）。本文把它直接当作 GP 的先验均值：

$$m_{\text{prior}}(x) = A \cdot \pi(x) + B$$

其中 $A>0$ 控制 $\pi(x)$ 的缩放幅度、$B$ 是加性平移。在没有任何观测时，$m_{\text{prior}}(x)$ 就勾勒出一幅与专家信念一致的粗略地形——先验/后验采样都会在专家看好的区域出现明显峰值。这与 πBO（改采集函数权重）、Ramachandran（改核函数）的路线不同：均值函数是 GP 里最「软」的部件，注入这里既能让先验真正进入代理模型，又不会像污染核函数那样牵一发动全身。

**2. 动态均值衰减：让先验早期主导、后期自动退场。** 随着评估点增多，数据本身就能更准地刻画 $f$，此时应当降低对专家先验的依赖——既体现对观测的信心增长，也是对坏先验的鲁棒兜底。受 πBO 衰减机制启发，本文把迭代 $n$ 时的均值定义为先验均值和标准基线均值 $\mu_0(x)$（通常是零均值）的凸组合：

$$m_n(x) = \gamma_n \cdot m_{\text{prior}}(x) + (1-\gamma_n)\cdot \mu_0(x), \qquad \gamma_n = \exp\big(-\lambda(n-N_0)\big)$$

其中 $\lambda>0$ 控制衰减速率、$N_0$ 是初始评估数。$\gamma_n$ 从 1 指数衰减到 0：前期 $m_n\approx m_{\text{prior}}$ 吃专家红利，后期 $m_n\approx \mu_0$ 退回标准 BO。这正是「先验坏也不会被长期带偏」的关键——哪怕专家判断离真最优很远，衰减也会把影响洗掉。

**3. 混合初始化：用先验补一部分起点，但不放弃空间覆盖。** 初始化阶段按比例 $\rho$ 从专家先验 $\pi(x)$ 采 $\lfloor\rho N_0\rfloor$ 个点（让代理模型一开始就在看好的区域有数据），剩余 $N_0-\lfloor\rho N_0\rfloor$ 个点用 Sobol 序列采（保证搜索空间均匀覆盖、避免被先验完全锁死）。这种混合策略既能更快建出准确的代理模型，又保留了对全局的探索性。

**4. 对采集函数无侵入 + 双采集函数收敛保证。** 因为改动只在均值函数、不碰采集函数，DynMeanBO 可与任意采集函数（PI/EI/LogEI/TS/UCB/KG/MES）即插即用，无需任何算法修改，额外开销可忽略。理论上作者给出两条保证：在 **EI** 下（沿用 Bull 2011 的 RKHS 假设），DynMeanBO-EI 与标准 BO-EI 同阶渐近收敛率 $L_n=O\big(n^{-(\nu\wedge1)/d}(\log n)^\beta\big)$；在 **UCB** 下（沿用 Srinivas 2010），累积遗憾满足 $R_N\le C_1\sqrt{N\beta_N G_N}+C_2\sum_{n=1}^N\gamma_n$，只要 $\sum_n\gamma_n<\infty$（衰减足够快），就与 BO-UCB 同阶 $R_N=O(\sqrt{N\beta_N G_N})$。直观上，衰减项 $\sum\gamma_n$ 有限保证了先验带来的偏置不会破坏长期最优性。

## 实验关键数据

### 主实验表格
任务：合成函数 Hartmann(4D)、Levy(5D)、Hartmann(6D)、Rosenbrock(6D)、Stybtang(7D) + PD1 HPO 基准（WMT/CIFAR/LM1B，4D）；对比 πBO、ColaBO 及标准 BO。

| 设置 | DynMeanBO 表现 | 说明 |
|------|----------------|------|
| 好先验 + 7 种采集函数 | 全面快于标准 BO | PI/EI/LogEI/TS/UCB/KG/MES 都加速，早期收敛优势尤其明显 |
| 好先验 vs πBO/ColaBO | 性能持平、收敛同样快 | 与两个 SOTA prior-informed 方法 on par |
| 坏先验（偏离最优 70%） | 鲁棒、迅速逼近标准 BO | 鲁棒性明显优于 ColaBO，对 πBO 优势尤其大 |
| 计算开销 | 比 πBO/ColaBO 快 1.4×–6.4× | 相对标准 BO 几乎零额外开销 |

### 消融实验表格

| 消融维度 | 结论（出处） |
|----------|--------------|
| 衰减速率 $\lambda$ | 附录 J 做了敏感性分析，衰减节奏可调 |
| 初始化比例 $\rho$ | 附录 K 分析从 $\pi(x)$ 采样的比例选择 |
| 先验缩放/平移 $A,B$ | 附录 B 给出解释与敏感性分析 |
| 先验质量（强/弱/错） | 附录 L 在不同先验方差下分析三种先验质量 |
| 高维扩展 | 附录 I 在 Levy(20D)、Rosenbrock(20D) 上验证 |

### 关键发现
- **好先验加速、坏先验不崩**：DynMeanBO 在好先验下与 SOTA 持平、坏先验下鲁棒性最强，验证了「动态衰减」同时拿下加速与鲁棒两个目标。
- **专家红利集中在早期**：BO 评估预算稀缺，DynMeanBO 的增益主要体现在前几十次迭代，正是现实昂贵任务最在意的阶段。
- **「坏先验」也可能有用**：在 PD1(LM1B) 上，即便先验偏离全局最优，所有 prior-informed 方法仍比 vanilla BO 快——因为坏先验虽然不在全局最优，却指向了一个高质量的次优区域。
- **更便宜的 SOTA**：在性能持平的前提下，DynMeanBO 比 πBO/ColaBO 单步评估时间快数倍，性价比更高。

## 亮点与洞察
- **找对了注入点**：均值函数是 GP 里最适合放「位置先验」的部件——它直接塑造预测地形的高低，又不像核函数那样一错全错。这个选择让方法既简单又可控。
- **衰减把「加速」和「鲁棒」统一**：同一个 $\gamma_n$ 既负责早期吃先验、又负责后期洗掉偏置，一个机制解决两个看似矛盾的诉求，且能写进收敛证明里。
- **通用性是真通用**：不碰采集函数意味着任何现成的 BO 工具链（BoTorch 里 7 种采集函数）都能直接套上，落地成本极低。

## 局限与展望
- **先验形式仍需人给**：方法假设专家能给出 $\pi(x)$ 这种位置分布，如何从模糊的人类偏好稳健地构造高质量 $\pi(x)$ 本身是个开放问题。
- **超参需调**：$\lambda$（衰减率）、$\rho$（初始化比例）、$A,B$（缩放平移）都需设定，虽有敏感性分析但缺自适应机制。
- **维度与规模**：主实验集中在 4D–7D，虽有 20D 附录结果，但大规模 HPO、复杂搜索空间仍待验证。
- **展望**：作者计划结合多保真度优化与并行评估，进一步提速并扩展到更大规模任务。

## 相关工作与启发
- **πBO (Hvarfner 2022)**：把先验作为权重乘进采集函数、并随迭代衰减——本文的衰减机制正是受其启发，但 DynMeanBO 把先验放进代理模型而非采集层，更原理化且通用。
- **ColaBO (Hvarfner 2024)**：把先验作为代理模型上的额外先验、与核超参先验正交，灵活但只配蒙特卡洛采集函数且开销大——DynMeanBO 在性能持平下显著更快、且兼容非 MC 采集函数。
- **核函数注入 (Ramachandran 2020)**：把先验累积分布嵌入核函数，但对坏先验极敏感（污染整个核）——反衬出本文「注入均值 + 动态衰减」在鲁棒性上的优势。
- **启发**：当你想给一个概率模型注入外部知识时，选择「注入哪个部件」往往比「注入多强」更重要——选一个软、可逆、影响局部的部件（如均值），再配一个能随证据增长自动退场的衰减权重，就能在「利用知识」与「不被错误知识坑」之间拿到一个干净的折中。

## 评分
- 新颖性: ⭐⭐⭐⭐ 「先验注入均值 + 动态衰减」的组合简洁而切中要害，相比改核函数/改采集函数的前作是更优雅的注入点；但衰减机制本身借鉴自 πBO，属于巧妙的重新定位而非全新范式。
- 实验充分度: ⭐⭐⭐⭐ 覆盖 7 种采集函数、合成 + HPO 双类任务、好/坏先验两种场景、与两个 SOTA 横向对比，外加丰富的附录消融（$\lambda$/$\rho$/$A,B$/先验质量/20D），相当扎实；略欠真实大规模 HPO 与更高维主表。
- 写作质量: ⭐⭐⭐⭐ 动机清晰、方法推导干净、理论与实验呼应紧密，图表信息量大；部分关键结果（坏先验下的具体数值）依赖图而非表格。
- 价值: ⭐⭐⭐⭐ 即插即用、零额外开销、有收敛保证、比 SOTA 快数倍，对实际昂贵优化（HPO、材料、控制）有直接落地价值，是 prior-informed BO 里实用性很高的一篇。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Local Entropy Search over Descent Sequences for Bayesian Optimization](local_entropy_search_over_descent_sequences_for_bayesian_optimization.md)
- [\[CVPR 2026\] DABO: Difficulty-Aware Bayesian Optimization with Diffusion-Learned Priors](../../CVPR2026/optimization/dabo_difficulty-aware_bayesian_optimization_with_diffusion-learned_priors.md)
- [\[ICLR 2026\] From Sorting Algorithms to Scalable Kernels: Bayesian Optimization in High-Dimensional Permutation Spaces](from_sorting_algorithms_to_scalable_kernels_bayesian_optimization_in_high-dimens.md)
- [\[ICLR 2026\] Generative Bayesian Optimization: Generative Models as Acquisition Functions](generative_bayesian_optimization_generative_models_as_acquisition_functions.md)
- [\[ICLR 2026\] Symmetry-Aware Bayesian Optimization via Max Kernels](symmetry-aware_bayesian_optimization_via_max_kernels.md)

</div>

<!-- RELATED:END -->
