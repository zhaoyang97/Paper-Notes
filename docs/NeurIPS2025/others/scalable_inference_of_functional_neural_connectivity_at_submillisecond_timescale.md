---
title: >-
  [论文解读] Scalable Inference of Functional Neural Connectivity at Submillisecond Timescales
description: >-
  [NeurIPS 2025][功能连接] 将传统离散时间Poisson GLM推广到连续时间Poisson点过程，通过蒙特卡洛采样和二阶多项式近似两种方法绕过不可解的积分项，配合正交的广义Laguerre基函数，在数百神经元、数千秒记录的数据上实现分钟级训练和亚毫秒级突触连接识别。 领域现状：Poisson广义线性模型（GL…
tags:
  - "NeurIPS 2025"
  - "功能连接"
  - "Poisson点过程"
  - "GLM"
  - "蒙特卡洛估计"
  - "突触耦合滤波器"
---

# Scalable Inference of Functional Neural Connectivity at Submillisecond Timescales

**会议**: NeurIPS 2025  
**arXiv**: [2510.20966](https://arxiv.org/abs/2510.20966)  
**代码**: [有](https://github.com/macari216/poisson-process-glm.git)  
**领域**: 计算神经科学  
**关键词**: 功能连接, Poisson点过程, GLM, 蒙特卡洛估计, 突触耦合滤波器

## 一句话总结
将传统离散时间Poisson GLM推广到连续时间Poisson点过程，通过蒙特卡洛采样和二阶多项式近似两种方法绕过不可解的积分项，配合正交的广义Laguerre基函数，在数百神经元、数千秒记录的数据上实现分钟级训练和亚毫秒级突触连接识别。

## 研究背景与动机

**领域现状**：Poisson广义线性模型（GLM）是分析神经元脉冲列数据、推断功能连接的基础工具。研究者通过估计神经元之间的耦合滤波器来识别突触连接。

**现有痛点**：传统GLM需要将连续脉冲时间离散化为时间bin的计数数据，构建巨大的设计矩阵 $\mathbf{X}$。突触动力学的时间尺度在亚毫秒级（1-5 ms内快速上升和下降），但常用的1-10 ms bin分辨率远不够精细。如果缩小bin到0.1 ms，设计矩阵会膨胀到 $10^{10}$-$10^{12}$ bits，内存完全无法容纳。即使采用分批（batched）策略，由于神经脉冲的稀疏性，不同batch间梯度方差极大，导致优化不稳定、拟合质量差。

**核心矛盾**：时间分辨率与计算可行性之间存在根本性冲突——要精确识别单突触连接需要亚毫秒分辨率，但离散化方法在这种分辨率下计算不可行。

**本文目标** (1) 如何在不离散化的情况下拟合Poisson GLM？ (2) 如何高效近似连续时间似然函数中的不可解积分？ (3) 如何在大规模神经记录上实现可扩展推断？

**切入角度**：将bin大小取极限趋于零，模型变为连续时间Poisson点过程。输入从巨大的设计矩阵变为紧凑的脉冲时间序列，从根本上解决内存问题。

**核心 idea**：用连续时间Poisson点过程替代离散时间GLM，通过MC采样或多项式近似处理似然函数中的不可解积分，实现亚毫秒精度的大规模神经连接推断。

## 方法详解

### 整体框架
输入是多个神经元的脉冲时间序列 $\mathbf{X} = \{(n_s, t_s)\}$（神经元编号+脉冲时间），输出是神经元之间的耦合滤波器 $\mathbf{w}$。模型基于连续时间Poisson点过程，对数似然由两项组成：第一项是在观测脉冲时刻评估的发放率对数之和，第二项是发放率在整个记录时长上的积分。第一项可精确计算，第二项（累积强度函数CIF）不可解，需要近似。

### 关键设计

1. **蒙特卡洛采样（MC）估计CIF**:

    - 功能：用分层采样近似CIF积分 $\int_0^T \lambda(t) dt$
    - 核心思路：将时间区间 $[0,T]$ 等分为 $M$ 个子区间，每个子区间内均匀抽取一个采样点 $\tau_m$，用 $\frac{T}{M}\sum_{m=1}^M \lambda(\tau_m)$ 近似积分。每次梯度更新重新采样，保持无偏估计
    - 设计动机：相比离散batched方法，MC方法的第一项（脉冲项）在所有脉冲上精确计算而非子集，分层采样保证对整个记录均匀覆盖，梯度方差远低于离散batch，即使只用 $M \ll T/\Delta t$ 个样本也能高质量拟合

2. **多项式近似（PA）连续GLM**:

    - 功能：将CIF中的非线性函数用二阶Chebyshev多项式近似，使似然函数变为模型参数的二次型
    - 核心思路：将 $\exp(x)\Delta$ 近似为 $a_2 x^2 + a_1 x + a_0$，则CIF变为 $a_2 \mathbf{w}^\top \mathbf{M} \mathbf{w} + a_1 \mathbf{m}^\top \mathbf{w} + Ta_0$，其中 $\mathbf{M}$ 和 $\mathbf{m}$ 是可以预计算的充分统计量。整个对数似然变为二次型，可直接求闭式解
    - 设计动机：闭式解意味着无需迭代优化，速度极快。但多项式近似本身引入误差，尤其在发放率波动较大时。PA可作为MC的高质量初始化（warm start），hybrid模式结合两者优势

3. **广义Laguerre多项式基函数**:

    - 功能：替代传统的raised cosine基函数来参数化耦合滤波器
    - 核心思路：使用含参数 $\alpha$ 和缩放系数 $c$ 的Laguerre多项式 $L_n^{(\alpha)}(ct) \cdot e^{-ct/2} \cdot (ct)^{\alpha/2}$，这些函数在权重 $t^\alpha e^{-t}$ 下正交，且具有类gamma函数的包络形状，天然匹配突触电导的快速上升和缓慢衰减特性
    - 设计动机：(1) 正交性使得更少的基函数即可高效表示滤波器变异（3个GL基 > 4个RC基）；(2) 单基和成对基函数积分有解析闭式解（通过下不完全gamma函数），避免数值积分；(3) 定义在完整历史窗口上，无需像RC基那样确定每个cosine bump的支持边界

### 损失函数 / 训练策略
MC模型使用带自适应步长（backtracking line search）的梯度下降，通过梯度步长范数 $u_t = \|\eta_t \cdot \nabla \mathcal{L}(\theta_t)\|$ 判断收敛。Hybrid PA-MC模型先用PA的闭式解初始化，再用MC微调。所有模型使用ridge正则化（$\beta=1000$）鼓励连接稀疏性。

## 实验关键数据

### 主实验

| 方法 | 计算时间（1000s记录） | 滤波器MSE（8神经元） | 计算时间（350神经元） | 滤波器MSE（350神经元） |
|------|---------------------|---------------------|---------------------|---------------------|
| 离散Batched (DB) | ~5小时 | 较高 | 最慢 | 最高 |
| 离散PA (PA-d) | 中等 | 中等 | 随规模差 | 中等 |
| 连续PA (PA-c) | 最快 | 中等偏高 | 最快 | 中等 |
| 连续MC | 快 | 最低 | 快 | 最低 |
| Hybrid PA-MC | 较快 | 最低 | 适中 | 最低 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| GL基 3个 vs RC基 4个 | GL MSE更低 | 正交基函数效率更高 |
| PA-c vs PA-d | PA-c略优 | 连续版避免了离散化误差 |
| MC vs 高斯-Lobatto求积 | MC MSE=2.54 vs 求积=5.37 | MC快400倍且更准确 |
| 交叉验证GL vs RC | GL所有J下log-likelihood更高 | GL在真实数据上也优于RC |

### 关键发现
- **离散batched方法在大规模数据上彻底失败**：即使用SVRG优化器，batch间梯度方差仍然过大，运行时间比连续方法高若干数量级仍无法收敛到最优
- **Hybrid模式是最佳权衡**：PA初始化 + MC微调结合了速度和精度优势
- **海马体实验验证**：在106个神经元、2.7小时记录上，hybrid模型恢复的耦合滤波器与交叉相关图（CCG）高度一致。识别出的兴奋性连接密度与已知海马解剖一致：CA3→CA3约4%（最高，符合dense recurrent EE连接），跨区域连接时延大于区域内（符合轴突传导时间）

## 亮点与洞察
- **从离散到连续的思路切换**极为巧妙：不是试图让离散方法更快（那条路已走到头），而是直接切换到点过程框架，从根本上绕过设计矩阵膨胀问题。这种"换问题定义"的思路可迁移到很多领域
- **hybrid warm start策略**是一个高度可复用的trick：用快速但不精确的方法（PA闭式解）初始化，再用精确但较慢的方法（MC）微调，比两种方法单独使用都好
- **Laguerre基函数的选择**体现了domain knowledge的价值：其gamma包络天然匹配突触时间特性，不是盲目选择"通用"基函数

## 局限与展望
- Poisson分布假设本身可能不够好——神经脉冲的方差特性可能更适合负二项分布等替代分布
- 无法明确区分单突触连接和共输入导致的相关发放，这是所有GLM方法的通病
- PA方法的近似范围需要手动设置（围绕平均发放率的3-7 Hz窗口），过宽增加近似误差，过窄限制滤波器幅度
- 未引入潜变量建模慢时间尺度的种群动态（如GPFA），可能有助于分离快耦合动态与慢协调活动

## 相关工作与启发
- **vs 离散GLM (Pillow et al., Zoltowski & Pillow)**: 本文在他们的PA方法基础上推到连续时间，消除bin误差并解决大规模可扩展性
- **vs Hawkes过程**: Hawkes过程只能建模兴奋性连接，本文的Poisson过程GLM可同时捕捉兴奋性和抑制性连接
- **vs 高斯-Lobatto求积 (Cai et al.)**: 求积方法需要在每对脉冲间插入节点，计算量与脉冲数线性增长，对大规模数据不可行

## 评分
- 新颖性: ⭐⭐⭐⭐ 连续时间GLM并非全新概念，但MC/PA估计器和Laguerre基的组合有明确技术贡献
- 实验充分度: ⭐⭐⭐⭐⭐ 合成数据（多配置）+ 真实海马体数据，时间-精度-规模全面对比
- 写作质量: ⭐⭐⭐⭐⭐ 从离散GLM到连续的过渡逻辑清晰，数学推导严谨但不晦涩
- 价值: ⭐⭐⭐⭐ 对计算神经科学领域有直接实用价值，开源代码可推动快速采用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Learning to Condition: A Neural Heuristic for Scalable MPE Inference](learning_to_condition_a_neural_heuristic_for_scalable_mpe_inference.md)
- [\[NeurIPS 2025\] Statistical Inference Under Performativity](statistical_inference_under_performativity.md)
- [\[ICML 2026\] Functional Equivalence in Attention: A Comprehensive Study with Applications to Linear Mode Connectivity](../../ICML2026/others/functional_equivalence_in_attention_a_comprehensive_study_with_applications_to_l.md)
- [\[NeurIPS 2025\] Statistical Inference for Gradient Boosting Regression](statistical_inference_for_gradient_boosting_regression.md)
- [\[NeurIPS 2025\] Robust Sampling for Active Statistical Inference](robust_sampling_for_active_statistical_inference.md)

</div>

<!-- RELATED:END -->
