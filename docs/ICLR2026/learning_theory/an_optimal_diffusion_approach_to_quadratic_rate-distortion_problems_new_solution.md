---
title: >-
  [论文解读] An Optimal Diffusion Approach to Quadratic Rate-Distortion Problems: New Solution and Approximation Methods
description: >-
  [ICLR2026][学习理论][率失真函数] 本文把连续信源在 MSE 失真下的率失真（RD）函数计算重写成一个"终端熵正则随机控制"问题，证明率-失真的权衡等价于控制能量-终端熵的权衡，并指出在正则条件下最优控制恰是倒向热方程解的 Stein score；由此给出高斯混合等信源的全新闭式解，以及一个不受码率上界限制的扩散神经估计器 R2D2。
tags:
  - "ICLR2026"
  - "学习理论"
  - "信息论"
  - "率失真"
  - "随机控制"
  - "扩散过程"
  - "率失真函数"
  - "熵正则最优传输"
  - "Schrödinger 桥"
  - "倒向热方程"
  - "扩散估计"
---

# An Optimal Diffusion Approach to Quadratic Rate-Distortion Problems: New Solution and Approximation Methods

**会议**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=upReXsENIl](https://openreview.net/forum?id=upReXsENIl)  
**代码**: https://github.com/ML-group-il/r2d2  
**领域**: 学习理论 / 信息论 / 率失真 / 随机控制 / 扩散过程  
**关键词**: 率失真函数、熵正则最优传输、Schrödinger 桥、倒向热方程、扩散估计

## 一句话总结
本文把连续信源在 MSE 失真下的率失真（RD）函数计算重写成一个"终端熵正则随机控制"问题，证明率-失真的权衡等价于控制能量-终端熵的权衡，并指出在正则条件下最优控制恰是倒向热方程解的 Stein score；由此给出高斯混合等信源的全新闭式解，以及一个不受码率上界限制的扩散神经估计器 R2D2。

## 研究背景与动机
**领域现状**：率失真理论刻画的是"在容许平均失真 $D$ 的前提下，编码一个连续信源所需的最小码率" $R(D)=\min_{p_{\hat X|X}:\,D(\hat X,X)\le D} I(X;\hat X)$。对离散小字母表信源，经典的 Blahut–Arimoto（BA）算法能直接迭代求解；但现实中的有损压缩对象几乎都是**连续信源**，此时高效计算 $R(D)$ 一直是个开放难题。

**现有痛点**：连续源的闭式 RD 解只在极少数标准例子（高斯源 + MSE、二元源 + Hamming）下已知。近年 Lei、Yang 等人发现 BA 与**熵正则最优传输（EOT）**之间的联系，可借此逼近 RD，但这类方法（NERD、WGD）把重建分布建成离散原子分布或有限粒子集，**码率被批大小/支撑大小所限**：$R<\log M$。于是它们在**低失真（高码率）区**——也正是现代高带宽通信最关心的区间——很快撞到天花板，实测被卡在约 13 nats。

**核心矛盾**：要在连续空间里既算准 RD、又拿到重建分布本身，离散逼近这条路天然有码率上限，而直接优化测试信道又缺少可计算的解析结构。

**切入角度**：作者注意到 EOT 与一个经典随机控制问题——**Schrödinger 桥（SB）**——等价：SB 用一个最小能量的漂移控制 $u$，把初始分布 $P_0$ 在带噪 SDE $dX_t=u\,dt+\sqrt{\epsilon}\,dW_t$ 下驱动到目标分布 $P_1$。既然现代扩散生成模型本质就是这种有限能量扩散过程，那把信源"扩散"到失真最优的重建分布，是不是也能用同一套语言写出来？

**核心 idea**：把 RD 问题改写成**目标分布自由**的随机控制——不再固定终端分布，而是对"终端状态的不确定性（微分熵）"加惩罚。作者称之为 **Terminal-Entropy Control（TEC）**，并证明它与 MSE 失真下的 RD 问题严格等价，从而把"率 vs 失真"翻译成"控制能量 vs 终端熵"这一对可解析、可数值优化的权衡。

## 方法详解

### 整体框架
全文围绕一条等价链展开：**RD 拉格朗日量 → 熵正则 OT → Schrödinger 桥式随机控制 → 终端熵控制（TEC）**。

出发点是 RD 拉格朗日量的熵分解形式（互信息按 $I=H(P_1)+H(P_0)-H(\pi)$ 展开），取二次代价 $d(\hat x,x)=\tfrac12\|\hat x-x\|^2$、平均失真 $D=\tfrac12\mathbb{E}\|X-\hat X\|^2$，得到

$$L_{RD}(P_0,\epsilon)=\min_{P_1}\min_{\pi\in\Pi(P_0,P_1)}\Big(\tfrac{1}{2\epsilon}\!\int\!\|\hat x-x\|^2 d\pi-H(\pi)+H(P_1)\Big)+H(P_0).$$

接着作者把内层耦合 $\pi$ 用一条**有限能量扩散轨迹** $dX_t=u(X_t,t)dt+\sqrt\epsilon dW_t$ 来实现：基于 Léonard / Pavon–Wakolbinger 的经典结果，最优轨迹的 KL 代价正好等于控制能量 $\tfrac{1}{2\epsilon}\mathbb{E}\!\int_0^1\|u\|^2dt$。把这一步代回，RD 就被替换成一个对漂移 $u$ 和终端分布 $P_1$ 的代理目标。最后再把"终端分布固定"这个约束松掉、改成对终端熵的惩罚，就得到 TEC——一个**目标自由、带终端熵正则**的随机控制问题。整条链路的价值在于：RD 这个组合优化问题，被转成了一个有 PDE 结构（Fokker–Planck / 倒向热方程）的控制问题，既能解析求解特殊情形，也能用单个扩散网络数值逼近一般情形。

### 关键设计

**1. 终端熵控制（TEC）：把 RD 重写成能量-熵权衡的随机控制**

针对"连续源 RD 缺乏可计算解析结构"的痛点，作者提出 TEC：

$$\inf_{u\in\mathcal U}\ \tfrac12\mathbb{E}\!\int_0^1\|u(X_t,t)\|^2dt+\epsilon H(X_1)\quad\text{s.t. } X_0\sim P_0,\ P_{X_1}\text{ 自由},\ dX_t=u\,dt+\sqrt\epsilon dW_t.$$

直觉上，$u$ 是一个**降低终端状态不确定性、同时尽量省能量**的控制器：能量项对应码率，终端熵项对应重建分布的"摊开程度"，$\epsilon$ 是调节两者的乘子。定理 3.1 证明在假设 A1（最优重建分布绝对连续、有有限二阶矩与微分熵）下 $L_{RD}=\tilde L_{RD}$，且 TEC 的最优控制 $u^\*$ 导出的终端分布 $P_{X_1^\*}$ 恰是 RD 问题的最优重建分布、对应耦合 $\pi^\*$ 即最优传输计划，反方向也成立。这把"率-失真权衡"严格翻译成"控制能量-终端熵权衡"——这是全文的支点，区别于以往把 RD 当作离散测试信道优化的做法。注意这里 $H(X_1)$ 是**终端熵**，与强化学习/随机控制里常见的"最大化策略熵"方向相反，作者是去**惩罚终端不确定性**，得到一种新的能量-熵权衡。

**2. 最优控制 = 倒向热方程解的 Stein score**

有了 TEC 的变分形式（var-TEC，配合密度演化的 Fokker–Planck 方程 $\partial_t p_t=-\nabla\!\cdot(p_t u)+\tfrac{1}{2\epsilon}\Delta p_t$），作者进一步刻画它的最优解。定理 3.2 给出一个出奇简洁的结论：若 $p_t^\*$ 满足**倒向热方程（BHE）**

$$\partial_t p_t^\*(x)=-\tfrac{1}{2\epsilon}\Delta_{xx}p_t^\*(x),\qquad p_0^\*\sim P_0,$$

并满足相应正则/衰减条件，则最优控制就是 $u^\*=\epsilon\nabla\log p_t^\*(x)$，即（缩放后的）**Stein score 函数**，对应最优 SDE $dX_t^\*=\epsilon\nabla\log p_t^\*(X_t)dt+\sqrt\epsilon dW_t$。这一刻画把控制问题降维成"解一个 PDE + 取对数梯度"，并和扩散生成模型里"学 score 再代回反向 SDE"的范式形成漂亮的对应。需要提醒的是：BHE 是**倒向**热传导，一般是病态/不稳定的，所以它只在满足条件的特殊源上给出精确解，一般情形交给后面的神经方法处理。

**3. 用 Fourier 分析求特殊源的全新闭式解**

定理 3.2 的真正威力在于它能算出**以前没人会算**的闭式重建分布。由于 BHE 是线性方程，对**高斯混合源** $p_0=\sum_i \tfrac{p_i}{\sqrt{2\pi\sigma_i^2}}e^{-(x-\mu_i)^2/2\sigma_i^2}$，解可由叠加直接写出：

$$p_t(x)=\sum_{i=1}^N \frac{p_i}{\sqrt{2\pi(\sigma_i^2-\epsilon t)}}\,e^{-\frac{(x-\mu_i)^2}{2(\sigma_i^2-\epsilon t)}},\qquad \epsilon\in\big(0,\min_i\sigma_i^2\big),$$

最优控制 $u=\epsilon\nabla\log p_t$ 随之确定——这是高斯混合 RD 此前未知的闭式结果。对更一般的**非高斯混合**（如 $\mathrm{sinc}^4$ 混合，带限信号），作者用频域分析：$p_t(x)=\tfrac{1}{2\pi}\int e^{i\omega x+\frac12\epsilon\omega^2 t}\hat p(\omega)d\omega$，只要积分对所有 $t\in[0,1]$ 收敛就成立，从而把 BHE 的求解化为对特征函数的数值积分。这类带限源此前无任何方法可处理，凸显了该路线的理论贡献。

**4. R2D2：单网络、无码率上界的扩散神经估计器**

定理 3.2 的假设（$p_t$ 处处非零、二阶可微、源分布显式已知）在实际中往往不满足——现实里只有样本。为此作者提出 **R2D2（Revealing RD functions with Diffusion）**：用一个 DNN $u_\theta(x,t,\epsilon)$ 建模控制器，把 $\epsilon$ 也喂进网络，于是**单个模型即可覆盖整条 RD 曲线上的多个工作点**。训练时从 $[\epsilon_{\min},\epsilon_{\max}]$ 均匀采 $\epsilon$、用 Euler–Maruyama 模拟轨迹，按 TEC 的代理损失 $L_\theta^\epsilon=\tfrac{1}{2M}\sum_m\sum_{t_i}\|u_\theta\|^2\Delta t+\epsilon\hat H(X_1)$ 优化（终端熵 $\hat H(X_1)$ 用负熵近似或核方法估计）。评估时由经验失真 $\hat D=\tfrac{1}{2M}\sum_m\|X_1^m-X_0^m\|^2$ 和

$$\hat R(\epsilon)=\frac{L_\theta^\epsilon-\hat D(\epsilon)}{\epsilon}-\frac d2\log(2\pi\epsilon)$$

读出 $(\hat R,\hat D)$。关键优势（Remark 4.1）：NERD/WGD 因离散/原子建模有 $R<\log M$ 的硬上界，而 R2D2 直接优化连续扩散漂移，**理论上码率不设上限**，特别适合低失真高码率区。

### 一个完整示例：从高斯源验证理论
拿标量高斯源 $P_0=\mathcal N(0,\sigma_0^2)$ 走一遍：对 $\epsilon<\sigma_0^2$，BHE 的解是 $p_t(x)=\tfrac{1}{\sqrt{2\pi(\sigma_0^2-\epsilon t)}}e^{-x^2/2(\sigma_0^2-\epsilon t)}$，于是最优控制 $u(x,t)=\epsilon\nabla\log p_t(x)=-\tfrac{\epsilon}{\sigma_0^2-\epsilon t}x$。在此控制下 $X_0,X_1$ 联合高斯，失真 $D=\tfrac12\mathbb{E}[(X_0-X_1)^2]=\tfrac12\epsilon$、码率 $R=-\tfrac12\log(\epsilon/\sigma_0^2)$，恰好复原经典闭式结果

$$R_{\mathrm{Gauss}}(D)=\tfrac12\log\frac{\sigma_0^2}{2D},\quad 0<2D<\sigma_0^2$$

（因子 2 来自把失真定义成半个 MSE）。这个例子既验证了整套等价链与定理 3.2 的自洽，也说明 $\epsilon$ 就是沿 RD 曲线滑动的旋钮。

### 损失函数 / 训练策略
核心训练目标即 TEC 代理损失：能量项 $\tfrac{1}{2M}\sum_m\sum_{t_i}\|u_\theta(X_{t_i}^m,t_i,\epsilon)\|^2\Delta t$ 加上终端熵 $\epsilon\hat H(X_1)$。每步随机抽 $\epsilon$，用 Euler–Maruyama 离散 SDE 采轨迹，再反传更新 $\theta$。终端熵的估计（近似负熵 / 核密度）是数值稳定性的关键环节，详见原文附录 B。

## 实验关键数据

### 主实验
在合成与真实信源上对比 R2D2 与两类基于 EOT 的最新估计器 NERD（Lei 等 2022）、WGD（Yang 等 2024）。

| 信源 | 维度 | 关注区间 | R2D2 表现 | 基线表现 |
|--------|------|----------|-----------|----------|
| 1-D 高斯 | 1 | 高码率 + 低码率 | 估计误差最低（64 seeds 中位绝对误差 + 四分位区间） | NERD / WGD 误差更大 |
| 高斯混合（$N{=}3$） | 1 | $\epsilon\in[4\times10^{-4},1.64\times10^{-2}]$ | 经验重建分布与闭式 Eq.(22) 吻合，曲线贴近 Shannon 下界 SLB | — |
| CIFAR10 灰度 $4\times4$ patch | 16 | 全区间 | 给出完整 RD 曲线并能采样重建 patch | NERD / WGD 同图对比 |
| Free Spoken Digit 语音 | 33 | 低失真高码率 | 可逼近 $>20$ nats 的（理论无上界）码率 | 实际被卡在约 13 nats |

最具说服力的是语音那组：在 33 维白化特征上，NERD（latent 1024、$M=10^6$）和 WGD（粒子数 $n=2\times10^5$）受 $R<\log M$ 约束，码率实际封顶在 ~13 nats；而 R2D2 顺利估计到 20+ nats，直接印证了 Remark 4.1 关于"无码率上界"的论断。

### 关键发现
- **低失真区是 R2D2 的主场**：离散/原子型基线的码率上界恰好压在高码率区，而现代高带宽通信关心的正是这一区间，连续扩散建模在此结构性占优。
- **理论与经验自洽**：高斯混合源上，Alg.1 采出的经验重建分布与定理 3.2 给出的闭式 $p_1(x)$ 重合，说明神经估计器确实在逼近 TEC 的真解而非凑曲线。
- **单模型多工作点**：把 $\epsilon$ 作为网络输入，一个控制器覆盖整条 RD 曲线，避免逐点重训。

## 亮点与洞察
- **把信息论量算成 PDE**：用倒向热方程刻画最优控制、用 Stein score 给出漂移，等于把"算 RD"变成"解一个热方程再取对数梯度"，与扩散模型"学 score"的范式严丝合缝——这是最漂亮的"啊哈"点。
- **能量-熵权衡是可迁移的视角**：以往随机控制/RL 多是最大化策略熵以鼓励探索，本文反其道惩罚**终端熵**，得到一种新权衡，作者也指出这对控制理论本身是一项贡献。
- **闭式解填补空白**：高斯混合、$\mathrm{sinc}^4$ 带限混合这些此前无解的 RD 例子被首次给出闭式/半解析结果，叠加原理 + 频域分析的组合可复用到其它线性 PDE 可解的信源族。
- **去掉码率上界**：用连续扩散漂移替代离散原子建模，直接拆掉 $R<\log M$ 这道结构性天花板。

## 局限与展望
- **只覆盖 MSE / 二次代价 + 连续源**：整套等价链依赖 MSE 失真才能落到二次 OT 与 SB；非 MSE 失真、离散信源尚需后续工作（作者在结论里也把这列为未来方向）。
- **假设 A1 在低码率失效**：当 $\epsilon$ 很大（码率很低）时最优重建分布会奇异、不再绝对连续，定理前提被破坏，方法主要面向低失真区。
- **倒向热方程病态**：BHE 一般不稳定，闭式路线只在满足非零、二阶可微等条件的特殊源上可用，限制了解析结果的适用面。
- **终端熵估计是软肋**：一般情形下 $\hat H(X_1)$ 靠负熵近似或核方法，高维下的偏差与方差会直接影响码率读数，原文也将其放在附录细讲。

## 相关工作与启发
- **vs NERD / WGD（Yang et al. 2024、Lei et al. 2022）**：同样借 BA↔EOT 的联系估计 RD，但它们把重建分布建成离散原子/有限粒子，码率被 $\log M$ 限死；本文直接优化连续扩散控制器，无码率上界、且能输出连续重建分布，高码率区更准。
- **vs Schrödinger 桥求解器（Gushchin et al. 2022 等）**：最接近的工作用 SB↔EOT 等价 + 博弈式公式求固定目标分布的桥；本文把**目标分布放开**、改成终端熵惩罚（TEC），并证明这等价于 RD，且直接优化漂移 $u$ 而非采样势函数（potentials）。
- **vs 扩散有损压缩（Theis et al. 2022、Ohayon et al. 2025 等）**：那类工作用（反向）扩散做实际压缩/重建；本文用**前向**扩散过程同时算出码率与失真，目标是估计 RD 极限本身而非设计编码器。
- **启发**：TEC 的能量-熵权衡形式或可推广到非 MSE 失真、感知失真（perception-distortion）以及更一般的随机控制/统计问题，作者明确把它视为连接信息论与随机控制两大学科的桥梁。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 把连续源 RD 重写成终端熵控制、并用倒向热方程的 Stein score 刻画最优解，是一条全新且自洽的路线。
- 实验充分度: ⭐⭐⭐⭐ 合成（高斯/混合）+ 真实（CIFAR10/语音）兼顾，语音上的无上界码率最有说服力；规模偏小、维度有限。
- 写作质量: ⭐⭐⭐⭐⭐ 从 RD→EOT→SB→TEC 的等价链推导清晰，定理与示例相互印证。
- 价值: ⭐⭐⭐⭐⭐ 既补上多类信源的闭式 RD 空白，又给出可扩展、无码率上界的神经估计器，为非 MSE/离散设定的后续研究铺路。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] A New Approach to Controlling Linear Dynamical Systems](a_new_approach_to_controlling_linear_dynamical_systems.md)
- [\[ICLR 2026\] Bi-Criteria Metric Distortion](bi-criteria_metric_distortion.md)
- [\[ICLR 2026\] Diffusion Language Models are Provably Optimal Parallel Samplers](diffusion_language_models_are_provably_optimal_parallel_samplers.md)
- [\[ICLR 2026\] Sharp Asymptotic Theory for Q-Learning with LD2Z Learning Rate and Its Generalization](sharp_asymptotic_theory_for_q-learning_with_textttld2z_learning_rate_and_its_gen.md)
- [\[ICLR 2026\] Diffusion and Flow-based Copulas: Forgetting and Remembering Dependencies](diffusion_and_flow-based_copulas_forgetting_and_remembering_dependencies.md)

</div>

<!-- RELATED:END -->
