---
title: >-
  [论文解读] Scaling Laws of SignSGD in Linear Regression: When Does It Outperform SGD?
description: >-
  [ICLR 2026][优化/理论][SignSGD] 在幂律随机特征（Power-Law Random Features）模型下，系统分析了 SignSGD 的缩放定律，揭示了 SignSGD 相对于 SGD 的两个独特效应——漂移归一化和噪声重塑，并证明在噪声主导的情形下 SignSGD 的计算最优斜率可以超过 SGD。
tags:
  - "ICLR 2026"
  - "优化/理论"
  - "SignSGD"
  - "缩放定律"
  - "线性回归"
  - "随机特征"
  - "学习率调度"
---

# Scaling Laws of SignSGD in Linear Regression: When Does It Outperform SGD?

**会议**: ICLR 2026  
**arXiv**: [2603.02069](https://arxiv.org/abs/2603.02069)  
**代码**: 无  
**领域**: 优化理论  
**关键词**: SignSGD, 缩放定律, 线性回归, 随机特征, 学习率调度

## 一句话总结

在幂律随机特征（Power-Law Random Features）模型下，系统分析了 SignSGD 的缩放定律，揭示了 SignSGD 相对于 SGD 的两个独特效应——漂移归一化和噪声重塑，并证明在噪声主导的情形下 SignSGD 的计算最优斜率可以超过 SGD。

## 研究背景与动机

SignSGD 是 Adam 等自适应优化器的核心组件之一——它使用梯度的符号而非梯度本身来更新参数。在大规模语言模型训练中，Adam/AdamW 几乎是事实标准，但对 SignSGD 为什么以及何时优于 SGD 的理论理解仍然有限。

**现有理论的不足**：
1. 传统的 SignSGD 分析大多关注凸优化或简单设定，未考虑现代深度学习中观察到的缩放定律现象
2. Paquette et al. (2024) 分析了 SGD 在幂律随机特征模型下的缩放定律，但未涉及 SignSGD
3. 缺乏对 SignSGD 在什么条件下优于 SGD 的精确刻画

**核心问题**：
- SignSGD 的 population risk 如何随模型大小、训练步数、学习率缩放？
- 计算最优配置下，SignSGD 和 SGD 的缩放行为何时不同？
- WSD（warmup-stable-decay）学习率调度对 SignSGD 有何独特影响？

作者选择幂律随机特征模型作为分析框架，这是因为它能同时捕获特征衰减和目标衰减两个关键维度，而这两个维度正是理解缩放定律的核心。

## 方法详解

### 整体框架

全文搭建在幂律随机特征（Power-Law Random Features, PLRF）这一可解析的玩具模型上。具体地，用一个作用在随机草图特征 $S\mathbf{x}$ 上的线性模型拟合目标：数据特征 $\mathbf{x}$ 的协方差特征值按 $i^{-2\alpha}$ 衰减（$\alpha$ 为特征衰减指数），最优解沿各特征方向的系数按 $i^{-\beta}$ 衰减（$\beta$ 为目标衰减指数），再用单遍（one-pass，每个样本只见一次）signSGD 优化，以 population risk $R(M,N,\gamma_0)$ 衡量泛化——其中 $M$ 是模型大小、$N$ 是训练步数、$\gamma_0$ 是学习率。这套设定的好处是：只用 $\alpha$、$\beta$ 两个标量，就能同时刻画现代缩放定律里"特征有多容易学"和"目标有多复杂"两个维度，从而把 signSGD 的整套缩放行为压缩成 $(M,N,\gamma_0,\alpha,\beta)$ 的解析函数，与 Paquette et al. (2024) 对 SGD 的已知结果逐项对照。整条分析链条是：先把符号更新的非线性递推化成可解的 ODE、得到 risk 的四项闭合公式；再逐项与 SGD 公式对比、分离出 signSGD 独有的两个效应；进而求计算最优配置、回答"何时赢 SGD"；最后把 WSD 学习率调度代入、看它如何进一步锐化缩放斜率。（本文为纯理论分析，无 pipeline 结构，故不画框架图。）

### 关键设计

**1. 四项闭合的 risk 缩放公式：把符号更新变成可解析的对象**

signSGD 的更新 $\boldsymbol{\theta}_{k+1} = \boldsymbol{\theta}_k - \gamma_k\,\mathrm{sign}(\mathbf{g}_k)$ 因为取符号而高度非线性，直接分析无从下手。作者对二次目标做二阶 Taylor 展开、配合 sign–Gaussian 恒等式，把这条递推在每个模态上写成"漂移（drift）+ 二次噪声（quadratic noise）"的一步更新，再转成连续时间 ODE、用确定性等价（deterministic equivalent）求解，最终给出 population risk 的四项渐近公式：

$$R(M,N,\gamma_0) \asymp \underbrace{A(M)}_{\text{近似误差}} + \underbrace{D^{\text{sign}}_{\text{al}}}_{\text{对齐特征损失}} + \underbrace{D^{\text{sign}}_{\text{dis}}}_{\text{扭曲特征损失}} + \underbrace{N^{\text{sign}}}_{\text{噪声项}}.$$

四项分别是：模型容量决定的近似误差 $A(M)\asymp M^{-2\alpha+\max(0,1-2\beta)}$、由漂移指数衰减出的对齐/扭曲特征损失 $D^{\text{sign}}_{\text{al}}, D^{\text{sign}}_{\text{dis}}$、以及来自一步 Taylor 展开的二次噪声 $N^{\text{sign}}=\gamma_0^2 M^{2-\min(1,2\alpha)}$。这个四项结构与 SGD 的缩放公式一一对应，正是后续所有结论能和 SGD 摆在同一坐标系里逐项比较的基础。

**2. 两个独特效应：漂移归一化与噪声重塑**

把上面的公式逐项与 SGD 公式（Paquette et al., 2024）对比，符号操作只在两处改了结构，这两处正是 signSGD 区别于 SGD 的全部来源。其一是**漂移归一化（drift-normalization）**：signSGD 的漂移项是 $\frac{4\gamma_k}{\pi\sqrt{L(k)}}\lambda_i$，比 SGD 多了 $1/\sqrt{L(k)}$ 的自归一化与对角预条件带来的 $M^{\min(\alpha,1/2)}$ 因子——前者把有效"流动时间"从 $N\gamma_0$ 换成 $\gamma_0\int_0^N L(u)^{-1/2}\,\mathrm{d}u$，在 $L\lesssim 1$ 时加速收敛，使对齐/扭曲特征项的 $N$ 指数严格变大（绝对值从 $x$ 增到 $\tfrac{2x}{2-x}$），即漂移项在步数 $N$ 上比 SGD 衰减得更快。其二是**噪声重塑（noise-reshaping）**：signSGD 的二次噪声项里没有 SGD 那个乘性的 $L(k)$ 因子，于是噪声项不再随 $N$ 衰减、也去掉了 SGD 噪声里的 $(N\gamma_0)^{-(4\alpha-1)/(2\alpha)}$ 结构，同时因为是在 $\overline{K}$ 特征基（而非 $K$ 特征基）下工作而获得额外的 $M$ 依赖。两个效应在解析式里各占独立的项、可分开追踪谁在哪个区域占上风——其中噪声重塑正是 signSGD 能在 SGD"噪声瓶颈"区反超的真正机制。

**3. 计算最优缩放定律：给定算力该怎么配 $M$ 和 $N$、何时赢 SGD**

固定总算力 $f = MN$，把学习率写成 $\gamma_0 = M^{-e}$、模型大小写成 $M=f^x$，对 $(e,x)$ 求解最小化 risk，得到 signSGD 的计算最优（compute-optimal）斜率 $\eta(\alpha,\beta)$（即 $R\asymp f^{-\eta}$）。这与 Chinchilla 定律思路一致，但首次落到 signSGD 上。比较两条计算最优曲线就能回答全文的核心问题：在 SGD 的噪声瓶颈相（Paquette 的 Phase III–IV 中被称作 Area III-IV$_{\text{sub}}$ 的子区域），signSGD 的计算最优斜率比 SGD 更陡、最优模型也更大；其余相位两者持平。还有一条实用结论：signSGD 的最优学习率指数 $e^*$ 始终大于 SGD，即 signSGD 总是用更小的学习率——若取 $\gamma_0\asymp 1$，噪声项 $N^{\text{sign}}$ 会大到把斜率压成 0，必须靠调小 $\gamma_0$ 在"压噪声"与"保漂移"之间找平衡。

**4. WSD 调度分析：为什么 warmup-stable-decay 配 signSGD 特别有效**

WSD（warmup-stable-decay，先线性升温、再恒定、最后多项式降温）是当下 LLM 训练的标准学习率调度，却缺乏理论支撑。作者把这三段学习率代入解析式累加各阶段贡献，发现 stable 段维持漂移速度、decay 段则进一步压低后期噪声，把噪声上界改善到优于恒定学习率。优化各超参后，当特征衰减快（$\alpha>0.5$）而目标衰减慢（具体落在 $0.5-\alpha<\beta<\tfrac{2\alpha-1}{2(4\alpha-1)}$ 的 Area Aa$^*$）时，WSD 的计算最优斜率严格大于恒定学习率、也大于 SGD（SGD 在对应相位用调度并无改善）。这给了 WSD 在 signSGD 下有效性的首个解释：它本质上是借末段降学习率来收割噪声重塑带来的红利。

全程为纯理论推导，核心工具是随机矩阵理论、确定性等价与幂律渐近展开，所有结论附严格证明（正文 89 页、25 个图覆盖参数空间各区域），并用数值实验验证理论指数（Figure 1）。

## 实验关键数据

### 主实验

**SignSGD vs SGD 的计算最优斜率比较**

| 参数区域 | SignSGD 斜率 | SGD 斜率 | 胜者 | 说明 |
|---------|------------|---------|------|------|
| 噪声主导区（高 $N$ 相对于 $M$，对应 SGD 的 Phase III–IV 噪声瓶颈）| 更陡 | 较缓 | SignSGD | 噪声重塑效应使 SignSGD 受益 |
| 偏差主导区（低 $N$ 相对于 $M$）| 相近或较缓 | 相近 | SGD 或持平 | 漂移归一化可能不利 |
| 平衡区 | 过渡行为 | 过渡行为 | 取决于具体 $\alpha, \beta$ | 两个效应竞争 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 固定 $\alpha$，变化 $\beta$ | 斜率变化 | 目标衰减越慢，SignSGD 优势越大 |
| 固定 $\beta$，变化 $\alpha$ | 斜率变化 | 特征衰减对两者影响类似 |
| 有/无 WSD 调度 | 噪声项减少 | WSD 在特征衰减快 + 目标衰减慢时效果显著 |
| 不同学习率 | 最优配置 | SignSGD 的最优学习率指数 $e^*$ 始终大于 SGD（用更小学习率） |

### 关键发现

1. **SignSGD 优于 SGD 的条件明确**：当训练处于噪声主导区域（高步数/低模型大小比）时，SignSGD 的计算最优缩放更优。这与实践中观察到的"Adam 比 SGD 好"的现象一致——大规模训练通常处于这个区域
2. **两个效应可以分离**：漂移归一化和噪声重塑效应有明确的数学表达，可以独立分析其对缩放律的贡献
3. **WSD 调度的理论支撑**：首次为 WSD 调度在 SignSGD 上的有效性提供了理论解释——它通过降低噪声项来锐化缩放斜率
4. **幂律参数决定一切**：$\alpha$（特征衰减）和 $\beta$（目标衰减）两个参数完全决定了 SignSGD 和 SGD 的相对表现

## 亮点与洞察

1. **填补重要理论空白**：首次为 SignSGD 建立了完整的缩放定律理论，与 SGD 的已有分析形成对比
2. **两个独特效应的发现**：漂移归一化和噪声重塑是理解 Adam 类优化器为何有效的关键概念，有望推广到更一般的设定
3. **实践指导意义**：明确了 SignSGD/Adam 超越 SGD 的条件——噪声主导区域，这与大规模 LLM 训练的实际场景吻合
4. **WSD 调度的理论基础**：为目前广泛使用但缺乏理论支撑的 WSD 调度提供了解释
5. **分析的彻底性**：89 页的论文、25 个图，覆盖了参数空间的各种区域，是一项极其细致的理论工作

## 局限与展望

1. **模型假设的简化性**：线性回归 + 随机特征模型与实际的深度网络训练有较大差距。虽然幂律特征模型能捕获一些核心现象，但非线性效应、参数间耦合等被忽略
2. **单遍训练假设**：one-pass SGD/SignSGD 意味着每个样本只见一次，而实际训练通常是多 epoch 的
3. **未考虑动量**：实际的 Adam 包含一阶和二阶动量，而 SignSGD 只是其简化。动量的加入可能改变缩放行为
4. **Gaussian-sketched 特征**：特征的 Gaussian 假设可能无法捕获真实数据中的结构
5. **向非线性扩展**：将分析扩展到两层或多层网络是重要但困难的方向
6. **实验验证**：理论预测与实际深度学习训练中的缩放行为是否一致，需要实证验证

## 相关工作与启发

- **Paquette et al. (2024)**：SGD 在幂律随机特征模型下的缩放定律分析，是本文最直接的比较对象
- **Chinchilla 缩放定律**（Hoffmann et al., 2022）：计算最优配置的原始工作，本文在 SignSGD 下得到类似的分析
- **Neural Scaling Laws**（Kaplan et al., 2020）：关于模型大小和数据量的缩放关系，本文从优化器角度补充
- 启发：理解优化器对缩放定律的影响与理解模型架构和数据同样重要；签名操作这样简单的非线性能产生复杂而有益的效应

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次系统分析 SignSGD 缩放定律，漂移归一化和噪声重塑是新概念
- 实验充分度: ⭐⭐⭐⭐ — 极其彻底的理论分析+数值验证（89页），但缺少深度学习实验
- 写作质量: ⭐⭐⭐⭐ — 理论严谨，但篇幅极长可能影响可读性
- 价值: ⭐⭐⭐⭐ — 为理解自适应优化器的理论提供了重要洞察

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Convex Dominance in Deep Learning I: A Scaling Law of Loss and Learning Rate](convex_dominance_in_deep_learning_i_a_scaling_law_of_loss_and_learning_rate.md)
- [\[NeurIPS 2025\] Functional Scaling Laws in Kernel Regression: Loss Dynamics and Learning Rate Schedules](../../NeurIPS2025/optimization/functional_scaling_laws_in_kernel_regression_loss_dynamics_and_learning_rate_sch.md)
- [\[ICLR 2026\] Distributionally Robust Linear Regression with Block Lewis Weights](distributionally_robust_linear_regression_with_block_lewis_weights.md)
- [\[NeurIPS 2025\] Emergence and Scaling Laws in SGD Learning of Shallow Neural Networks](../../NeurIPS2025/optimization/emergence_and_scaling_laws_in_sgd_learning_of_shallow_neural_networks.md)
- [\[NeurIPS 2025\] Learning Quadratic Neural Networks in High Dimensions: SGD Dynamics and Scaling Laws](../../NeurIPS2025/optimization/learning_quadratic_neural_networks_in_high_dimensions_sgd_dynamics_and_scaling_l.md)

</div>

<!-- RELATED:END -->
